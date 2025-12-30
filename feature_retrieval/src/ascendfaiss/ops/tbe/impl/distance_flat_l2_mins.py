#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
-------------------------------------------------------------------------
This file is part of the MindStudio project.
Copyright (c) 2025 Huawei Technologies Co.,Ltd.

MindStudio is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:

         http://license.coscl.org.cn/MulanPSL2

THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details.
-------------------------------------------------------------------------
"""
from te import tik
from mxIndex_impl.common import set_soc_info


class DistanceFlatL2Mins:
    def __init__(self,
                 input_queries,
                 input_code,
                 input_precomputed,
                 output_distances,
                 output_mins,
                 output_flag,
                 kernel_name="distance_flat_l2_mins"):
        self.shape_queries = input_queries.get("shape")
        self.dtype_queries = input_queries.get("dtype")
        self.shape_code = input_code.get("shape")
        self.dtype_code = input_code.get("dtype")
        self.shape_precomputed = input_precomputed.get("shape")
        self.dtype_precomputed = input_precomputed.get("dtype")
        self.shape_distances = output_distances.get("shape")
        self.dtype_distances = output_distances.get("dtype")
        self.shape_mins = output_mins.get("shape")
        self.dtype_mins = output_mins.get("dtype")
        self.shape_flag = output_flag.get("shape")
        self.dtype_flag = output_flag.get("dtype")
        self.kernel_name = kernel_name

        # compute parameter
        self.queries_num, self.dim = self.shape_queries
        self.code_num, = self.shape_precomputed
        self.min_num = self.shape_mins[1]
        self.queries_align = (self.queries_num + 15) // 16 * 16
        self.min_batch = 32
        self.max_repeats = 128

        # check parameter
        self.check_parameter()

        # set vector fp32 mask and fp16 mask
        self.fp32_mask = 64

        # set tik instance
        self.set_tik_instance()

    def check_parameter(self):
        if self.dim % 16 != 0:
            raise RuntimeError("feature dim must be a multiple of 16")
        if self.code_num % 16 != 0:
            raise RuntimeError("code num must be a multiple of 16")

    def set_tik_instance(self):
        """
        set tik_instance
        """
        set_soc_info()
        self.tik_instance = tik.Tik()

        self.aicore_available = self.shape_flag[0]
        self.aicore_use = min(self.aicore_available, (self.code_num + 255) // 256)
        self.code_each_core = (self.code_num // self.aicore_use + 15) // 16 * 16
        self.code_last_core = self.code_num - (self.aicore_use - 1) * self.code_each_core

        self.queries_each_loop = min(self.queries_num, 32)
        self.code_each_loop = min(min(
            min(228 * 1024 // 2 // (4 * self.queries_align + 6 * self.queries_each_loop) // 256 * 256, 512),    # UB
            (self.code_each_core + 255) // 256 * 256),      # L1
            (1024 * 1024 - self.dim * 32) // 4 // self.dim // 256 * 256)    # core_num

        self.set_src_dst_tensor()

    def set_src_dst_tensor(self):
        """
        set input and output tensor
        """
        self.coeff = self.tik_instance.Scalar("float32", name="coeff", init_value=-2)

        # creat input tensor: input_queries_gm, input_code_gm
        # and input_precomputed_gm
        # and output tensor: output_distances_gm, output_flag_gm in global buffer
        self.input_queries_gm = self.tik_instance.Tensor(
            self.dtype_queries,
            self.shape_queries,
            name="input_queries_gm",
            scope=tik.scope_gm)
        self.input_code_gm = self.tik_instance.Tensor(
            self.dtype_code,
            self.shape_code,
            name="input_code_gm",
            scope=tik.scope_gm)
        self.input_precomputed_gm = self.tik_instance.Tensor(
            self.dtype_precomputed,
            self.shape_precomputed,
            name="input_precomputed_gm",
            scope=tik.scope_gm)
        self.output_distances_gm = self.tik_instance.Tensor(
            self.dtype_distances,
            self.shape_distances,
            name="output_distances_gm",
            scope=tik.scope_gm)
        self.output_mins_gm = self.tik_instance.Tensor(
            self.dtype_mins,
            self.shape_mins,
            name="output_mins_gm",
            scope=tik.scope_gm
        )
        self.output_flag_gm = self.tik_instance.Tensor(
            self.dtype_flag,
            self.shape_flag,
            name="output_flag_gm",
            scope=tik.scope_gm)

    def distance_compute_each_loop(self, aicore_offset, aicore_code_num, move_offset, move_num):
        queries_align = (move_num + 15) // 16 * 16
        add_scalar_list = [self.tik_instance.Scalar(dtype="float32") for i in range(move_num)]
        queries_l1 = self.tik_instance.Tensor("float16",
                                              (self.dim // 16, queries_align, 16),
                                              name="queries_l1", scope=tik.scope_cbuf)

        with self.tik_instance.new_stmt_scope():
            for i in range(move_num):
                self.tik_instance.data_move(queries_l1[0, i, 0],
                                            self.input_queries_gm[move_offset + i, 0],
                                            0, self.dim // 16, 1, 0, queries_align - 1)

            queries_square_l0c = self.tik_instance.Tensor("float32",
                                                          (queries_align // 16, queries_align, 16),
                                                          name="queries_square_l0c", scope=tik.scope_cbuf_out)
            self.tik_instance.matmul(queries_square_l0c,
                                     queries_l1, queries_l1,
                                     queries_align, self.dim, queries_align)
            queries_square_ub = self.tik_instance.Tensor("float32",
                                                         (queries_align // 16, queries_align, 16),
                                                         name="queries_square_ub", scope=tik.scope_ubuf)
            self.tik_instance.data_move(queries_square_ub,
                                        queries_square_l0c,
                                        0, 1, queries_align * queries_align // 256, 0, 0)

            queries_l2_ub = self.tik_instance.Tensor("float32",
                                                     (queries_align,),
                                                     name="queries_l2_ub", scope=tik.scope_ubuf)
            for i in range(move_num):
                mask = 2 ** (i % 16)
                self.tik_instance.vcadd([0, mask],
                                        queries_l2_ub[i],
                                        queries_square_ub[i // 16, i, 0],
                                        1, 1, 1, 8)
            for i in range(move_num):
                add_scalar_list[i].set_as(queries_l2_ub[i])

        # compute xy using cube
        code_loop_times = aicore_code_num // self.code_each_loop
        if code_loop_times > 0:
            thread_num_need = 2 if code_loop_times > 1 else 1
            with self.tik_instance.for_range(0, code_loop_times, thread_num=thread_num_need) as loop_code:
                self.cube_compute_each_loop(queries_l1, add_scalar_list,
                                            aicore_offset, loop_code * self.code_each_loop,
                                            self.code_each_loop, move_offset, move_num)

        code_last_num = aicore_code_num % self.code_each_loop
        if code_last_num > 0:
            self.cube_compute_each_loop(queries_l1, add_scalar_list,
                                        aicore_offset,
                                        code_loop_times * self.code_each_loop,
                                        code_last_num, move_offset, move_num)

    def cube_compute_each_loop(self, queries_l1, add_scalar_list,
                               aicore_offset, code_offset, code_num, queries_offset, queries_num):
        queries_align = (queries_num + 15) // 16 * 16
        code_l1 = self.tik_instance.Tensor("float16",
                                           (self.dim // 16, self.code_each_loop, 16),
                                           name="code_l1", scope=tik.scope_cbuf)
        for i in range(self.dim // 16):
            self.tik_instance.data_move(code_l1[i, 0, 0],
                                        self.input_code_gm[(aicore_offset + code_offset) // 16,
                                                           i, 0, 0],
                                        0, code_num // 16, 16, self.dim - 16, 0)

        inner_product_l0c = self.tik_instance.Tensor("float32",
                                                     (self.code_each_loop // 16, queries_align, 16),
                                                     name="inner_product_l0c", scope=tik.scope_cbuf_out)
        self.tik_instance.matmul(inner_product_l0c,
                                 queries_l1, code_l1,
                                 queries_align, self.dim, self.code_each_loop)

        # mov code l2 from out to UB
        code_l2_ub_fp16 = self.tik_instance.Tensor("float16",
                                                   (self.code_each_loop,),
                                                   name="code_l2_ub_fp16", scope=tik.scope_ubuf)
        self.tik_instance.data_move(code_l2_ub_fp16,
                                    self.input_precomputed_gm[aicore_offset + code_offset],
                                    0, 1, code_num // 16, 0, 0)

        # code_l2 do conv from fp16 to fp32
        code_l2_ub = self.tik_instance.Tensor("float32",
                                              (self.code_each_loop,),
                                              name="code_l2_ub", scope=tik.scope_ubuf)
        vconv_repeat_times = self.code_each_loop // self.fp32_mask
        vconv_offset = 0
        if vconv_repeat_times > 0:
            self.tik_instance.vconv(self.fp32_mask, "none",
                                    code_l2_ub[vconv_offset],
                                    code_l2_ub_fp16[vconv_offset],
                                    vconv_repeat_times, 1, 1, 8, 4)
            vconv_offset += vconv_repeat_times * self.fp32_mask

        vconv_last_num = self.code_each_loop % self.fp32_mask
        if vconv_last_num > 0:
            self.tik_instance.vconv(vconv_last_num, "none",
                                    code_l2_ub[vconv_offset],
                                    code_l2_ub_fp16[vconv_offset],
                                    1, 1, 1, 8, 4)

        add_ub = self.tik_instance.Tensor("float32",
                                          (queries_num, self.code_each_loop),
                                          name="add_ub", scope=tik.scope_ubuf)
        for loop_queries in range(0, queries_num):
            repeat = self.code_each_loop // self.fp32_mask
            offset = 0
            if repeat > 0:
                self.tik_instance.vadds(self.fp32_mask,
                                        add_ub[loop_queries, offset],
                                        code_l2_ub[offset], add_scalar_list[loop_queries],
                                        repeat, 1, 1, 8, 8)
                offset += repeat * self.fp32_mask

            remain = self.code_each_loop % self.fp32_mask
            if remain > 0:
                self.tik_instance.vadds(remain,
                                        add_ub[loop_queries, offset],
                                        code_l2_ub[offset], add_scalar_list[loop_queries],
                                        1, 1, 1, 8, 8)

        # mov xy from L0-C to UB
        inner_product_ub = self.tik_instance.Tensor("float32",
                                                    (self.code_each_loop // 16, queries_align, 16),
                                                    name="inner_product_ub", scope=tik.scope_ubuf)
        self.tik_instance.data_move(inner_product_ub,
                                    inner_product_l0c,
                                    0, 1, self.code_each_loop * queries_align // 256, 0, 0)

        with self.tik_instance.for_range(0, queries_num) as loop_queries:
            self.tik_instance.vaxpy(16,
                                    add_ub[loop_queries, 0],
                                    inner_product_ub[0, loop_queries, 0], self.coeff,
                                    code_num // 16, 1, 1, 2, queries_align * 2)

        dst_ub = self.tik_instance.Tensor("float16",
                                          (queries_num, self.code_each_loop),
                                          name="dst_ub", scope=tik.scope_ubuf)
        vconv_repeat_times = queries_num * self.code_each_loop // self.fp32_mask
        vconv_offset = 0
        if vconv_repeat_times > 0:
            self.tik_instance.vconv(self.fp32_mask, "none",
                                    dst_ub[vconv_offset],
                                    add_ub[vconv_offset],
                                    vconv_repeat_times, 1, 1, 4, 8)
            vconv_offset += vconv_repeat_times * self.fp32_mask

        vconv_last_num = queries_num * self.code_each_loop % self.fp32_mask
        if vconv_last_num > 0:
            self.tik_instance.vconv(vconv_last_num, "none",
                                    dst_ub[vconv_offset],
                                    add_ub[vconv_offset],
                                    1, 1, 1, 4, 8)

        dst_min_ub = self.tik_instance.Tensor("float16",
                                              (queries_num, self.code_each_loop // self.min_batch * 2),
                                              name="dst_ub", scope=tik.scope_ubuf)
        if code_num == self.code_each_loop:
            vcmin_repeats = queries_num * self.code_each_loop // self.min_batch
            repeat_loops = vcmin_repeats // self.max_repeats
            for i in range(repeat_loops):
                self.tik_instance.vcmin(self.min_batch,
                                        dst_min_ub[(i * self.max_repeats * 2):],
                                        dst_ub[(i * self.max_repeats * self.min_batch):],
                                        self.max_repeats, 1, 1, self.min_batch // 16)
            last_repeats = vcmin_repeats % self.max_repeats
            if last_repeats > 0:
                self.tik_instance.vcmin(self.min_batch,
                                        dst_min_ub[(repeat_loops * self.max_repeats * 2):],
                                        dst_ub[(repeat_loops * self.max_repeats * self.min_batch):],
                                        last_repeats, 1, 1, self.min_batch // 16)
        else:
            vcmin_repeats = code_num // self.min_batch
            if vcmin_repeats > 0:
                with self.tik_instance.for_range(0, queries_num) as loop_queries:
                    self.tik_instance.vcmin(self.min_batch,
                                            dst_min_ub[loop_queries, 0],
                                            dst_ub[loop_queries, 0],
                                            vcmin_repeats, 1, 1, self.min_batch // 16)

            last_code_num = code_num % self.min_batch
            if last_code_num > 0:
                with self.tik_instance.for_range(0, queries_num) as loop_queries:
                    self.tik_instance.vcmin(last_code_num,
                                            dst_min_ub[loop_queries, self.min_batch * vcmin_repeats],
                                            dst_ub[loop_queries, 2 * vcmin_repeats],
                                            1, 1, 1, self.min_batch // 16)

        self.tik_instance.data_move(self.output_distances_gm[queries_offset, aicore_offset + code_offset],
                                    dst_ub,
                                    0, queries_num, code_num // 16,
                                    (self.code_each_loop - code_num) // 16,
                                    (self.code_num - code_num) // 16)
        self.tik_instance.data_move(self.output_mins_gm[queries_offset,
                                                        (aicore_offset + code_offset) // self.min_batch * 2],
                                    dst_min_ub,
                                    0, queries_num, self.code_each_loop // self.min_batch * 2 // 16,
                                    0, (self.min_num - self.code_each_loop // self.min_batch * 2) // 16)

    def distance_compute_flat(self):
        """
        the compute process
        """
        with self.tik_instance.for_range(0, self.aicore_available, block_num=self.aicore_available) as block_index:
            # compute coarse code num and move offset every core
            with self.tik_instance.if_scope(block_index < self.aicore_use - 1):
                aicore_code_num = self.code_each_core
            with self.tik_instance.else_scope():
                aicore_code_num = self.code_last_core

            with self.tik_instance.if_scope(block_index < self.aicore_use):
                queries_loops = self.queries_num // self.queries_each_loop
                if queries_loops > 0:
                    with self.tik_instance.for_range(0, queries_loops) as loop_queries:
                        self.distance_compute_each_loop(block_index * self.code_each_core,
                                                        aicore_code_num,
                                                        loop_queries * self.queries_each_loop,
                                                        self.queries_each_loop)

                queries_last_num = self.queries_num % self.queries_each_loop
                if queries_last_num > 0:
                    self.distance_compute_each_loop(block_index * self.code_each_core,
                                                    aicore_code_num,
                                                    queries_loops * self.queries_each_loop,
                                                    queries_last_num)

            one = self.tik_instance.Scalar(dtype="uint16", name="one", init_value=1)
            flag_ub = self.tik_instance.Tensor("uint16",
                                               (16,),
                                               name="flag_ub", scope=tik.scope_ubuf)
            flag_ub[0].set_as(one)
            self.tik_instance.pipe_barrier("PIPE_MTE3")
            self.tik_instance.data_move(self.output_flag_gm[block_index, 0],
                                        flag_ub, 0, 1, 1, 0, 0)

    def get_tik_instance(self):
        """
        obtain tik instance
        """
        self.distance_compute_flat()

        self.tik_instance.BuildCCE(kernel_name=self.kernel_name,
                                   inputs=[
                                       self.input_queries_gm,
                                       self.input_code_gm,
                                       self.input_precomputed_gm
                                   ],
                                   outputs=[self.output_distances_gm, self.output_mins_gm, self.output_flag_gm])

        return self.tik_instance


def distance_flat_l2_mins(input_queries,
                          input_code,
                          input_precomputed,
                          output_distances,
                          output_mins,
                          output_flag,
                          kernel_name="distance_flat_l2_mins"):
    """
    calculating distance

    Parameters
    ----------
    input_queries : dict
        shape and dtype of query vector
    input_code : dict
        shape and dtype of coarse code
    input_precomputed : dict
        shape and dtype of precomputed L2 distance of coarse code
    output_distances : dict
        shape and dtype of distances, should be same dtype as input_queries
    output_mins : dict
        shape and dtype of min distances
    output_flag : dict
        shape and dtype of flag, only the 1th and 17th is valid
    kernel_name : str
        kernel name, default value is "distance_flat_l2_mins"

    Returns
    -------
    None
    """
    distance_flat = DistanceFlatL2Mins(input_queries, input_code, input_precomputed,
                                       output_distances, output_mins, output_flag, kernel_name)
    tik_instance = distance_flat.get_tik_instance()
    return tik_instance
