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


class DistanceIVFInt8L2L1:
    def __init__(self,
                 input_queries,
                 input_code,
                 input_precomputed,
                 output_distances,
                 output_flag,
                 kernel_name="distance_ivf_int8_l2"):
        self.shape_queries = input_queries.get("shape")
        self.dtype_queries = input_queries.get("dtype")
        self.shape_code = input_code.get("shape")
        self.dtype_code = input_code.get("dtype")
        self.shape_precomputed = input_precomputed.get("shape")
        self.dtype_precomputed = input_precomputed.get("dtype")
        self.shape_distances = output_distances.get("shape")
        self.dtype_distances = output_distances.get("dtype")
        self.shape_flag = output_flag.get("shape")
        self.dtype_flag = output_flag.get("dtype")
        self.kernel_name = kernel_name

        # compute parameter
        self.queries_num, self.dim = self.shape_queries
        self.code_num, = self.shape_precomputed
        self.queries_align = (self.queries_num + 15) // 16 * 16

        # check parameter
        self.check_parameter()

        # set vector fp32 mask and fp16 mask
        self.int32_mask = 64
        # scale changed with dim
        self.scale = 0.01 / min(self.dim // 64, max(self.dim // 128 + 1, 4))

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

        self.aicore_use = 2
        self.queries_num_each_loop = min(48, self.queries_num)
        if self.dim < 1024:
            self.code_num_each_loop = min(min((48 // self.queries_num_each_loop) * 512, 1024), self.code_num)
        else:
            self.code_num_each_loop = min(min((48 // self.queries_num_each_loop) * 512, 512), self.code_num)
        self.code_num_each_core = self.code_num // self.aicore_use // 16 * 16
        self.code_num_last_core = self.code_num - (self.aicore_use - 1) * self.code_num_each_core

        self.set_src_dst_tensor()

    def set_src_dst_tensor(self):
        """
        set input and output tensor
        """
        self.coeff = self.tik_instance.Scalar("int32", name="coeff", init_value=-2)
        self.query_l2 = self.tik_instance.Scalar("int32", name="query_l2", init_value=0)

        # create input tensor: input_queries_gm, input_code_gm
        # and input_precomputed_gm
        # and output tensor: output_distances_gm, output_flag_gm in global buffer
        self.input_queries_gm = self.tik_instance.Tensor(self.dtype_queries,
                                                         self.shape_queries,
                                                         name="input_queries_gm",
                                                         scope=tik.scope_gm)
        self.input_code_gm = self.tik_instance.Tensor(self.dtype_code,
                                                      self.shape_code,
                                                      name="input_code_gm",
                                                      scope=tik.scope_gm)
        self.input_precomputed_gm = self.tik_instance.Tensor(self.dtype_precomputed,
                                                             self.shape_precomputed,
                                                             name="input_precomputed_gm",
                                                             scope=tik.scope_gm)
        self.output_distances_gm = self.tik_instance.Tensor(self.dtype_distances,
                                                            self.shape_distances,
                                                            name="output_distances_gm",
                                                            scope=tik.scope_gm)
        self.output_flag_gm = self.tik_instance.Tensor(self.dtype_flag,
                                                       self.shape_flag,
                                                       name="output_flag_gm",
                                                       scope=tik.scope_gm)

    def distance_compute_each_loop(self, aicore_move_offset, aicore_code_num, move_offset, move_num):
        queries_align = (move_num + 15) // 16 * 16
        queries_l1 = self.tik_instance.Tensor("int8",
                                              (self.dim // 32, queries_align, 32),
                                              name="queries_l1", scope=tik.scope_cbuf)
        queries_l2_ub = self.tik_instance.Tensor("int32",
                                                 (move_num,),
                                                 name="queries_l2_ub", scope=tik.scope_ubuf)

        with self.tik_instance.new_stmt_scope():
            for i in range(move_num):
                self.tik_instance.data_move(queries_l1[0, i, 0], self.input_queries_gm[move_offset + i, 0],
                                            0, self.dim // 32, 1, 0, queries_align - 1)

            queries_square_l0c = self.tik_instance.Tensor("int32",
                                                          (queries_align // 16, queries_align, 16),
                                                          name="queries_square_l0c", scope=tik.scope_cbuf_out)
            self.tik_instance.matmul(queries_square_l0c,
                                     queries_l1, queries_l1,
                                     queries_align, self.dim, queries_align)
            queries_square_ub = self.tik_instance.Tensor("int32",
                                                         (queries_align // 16, queries_align, 16),
                                                         name="queries_square_ub", scope=tik.scope_ubuf)
            self.tik_instance.data_move(queries_square_ub,
                                        queries_square_l0c,
                                        0, 1, queries_align * queries_align // 256, 0, 0)

            repeat = move_num // self.int32_mask
            offset = 0
            if repeat > 0:
                self.tik_instance.vec_dup(self.int32_mask, queries_l2_ub[offset], 0, repeat, 8)
                offset += repeat * self.int32_mask

            remain = move_num % self.int32_mask
            if remain > 0:
                self.tik_instance.vec_dup(remain, queries_l2_ub[offset], 0, 1, 8)

            for i in range(move_num):
                mask = 2 ** (i % 16)
                self.tik_instance.vadd([0, mask],
                                       queries_l2_ub[i // 16 * 16],
                                       queries_l2_ub[i // 16 * 16], queries_square_ub[i // 16, i, 0],
                                       1, 1, 1, 1, 8, 8, 8)

        # compute xy using cube
        code_loop_time = aicore_code_num // self.code_num_each_loop
        with self.tik_instance.if_scope(code_loop_time > 0):
            with self.tik_instance.for_range(0, code_loop_time) as loop_code:
                self.cube_compute_each_loop(queries_l1, queries_l2_ub, aicore_move_offset,
                                            loop_code * self.code_num_each_loop,
                                            self.code_num_each_loop, move_offset, move_num)

        code_last_num = aicore_code_num % self.code_num_each_loop
        with self.tik_instance.if_scope(code_last_num > 0):
            self.cube_compute_each_loop(queries_l1, queries_l2_ub, aicore_move_offset,
                                        code_loop_time * self.code_num_each_loop,
                                        code_last_num, move_offset, move_num)

    def cube_compute_each_loop(self, queries_l1, queries_l2_ub, aicore_move_offset, code_move_offset,
                               code_move_num, queries_move_offset, queries_move_num):
        queries_align = (queries_move_num + 15) // 16 * 16
        code_align_16 = (code_move_num + 15) // 16 * 16
        code_l1 = self.tik_instance.Tensor("int8", (self.dim // 32, self.code_num_each_loop, 32),
                                           name="code_l1", scope=tik.scope_cbuf)
        for i in range(self.dim // 32):
            self.tik_instance.data_move(code_l1[i, 0, 0],
                                        self.input_code_gm[(aicore_move_offset + code_move_offset) // 16, i, 0, 0],
                                        0, code_align_16 // 16, 16, self.dim // 2 - 16, 0)

        inner_product_l0c = self.tik_instance.Tensor("int32",
                                                     (self.code_num_each_loop // 16, queries_align, 16),
                                                     name="inner_product_l0c", scope=tik.scope_cbuf_out)
        self.tik_instance.matmul(inner_product_l0c,
                                 queries_l1, code_l1,
                                 queries_align, self.dim, self.code_num_each_loop)

        # mov xy from L0-C to UB
        inner_product_ub = self.tik_instance.Tensor("int32",
                                                    (self.code_num_each_loop // 16, queries_align, 16),
                                                    name="inner_product_ub", scope=tik.scope_ubuf)
        self.tik_instance.data_move(inner_product_ub,
                                    inner_product_l0c,
                                    0, 1, code_align_16 * queries_align // 256, 0, 0)

        # mov code l2 from out to UB
        code_l2_ub = self.tik_instance.Tensor("int32",
                                              (self.code_num_each_loop,),
                                              name="code_l2_ub", scope=tik.scope_ubuf)
        self.tik_instance.data_move(code_l2_ub,
                                    self.input_precomputed_gm[aicore_move_offset + code_move_offset],
                                    0, 1, code_align_16 // 8, 0, 0)

        # x^2 + y^2 - 2xy
        add_ub = self.tik_instance.Tensor("int32",
                                          (queries_move_num, self.code_num_each_loop),
                                          name="add_ub", scope=tik.scope_ubuf)
        coeff_ub = self.tik_instance.Tensor("int32", (16,), tik.scope_ubuf, "coeff_ub")
        self.tik_instance.vec_dup(16, coeff_ub, self.coeff, 1, 8)

        query_l2_ub = self.tik_instance.Tensor("int32",
                                               (self.int32_mask,),
                                               name="query_l2_ub", scope=tik.scope_ubuf)

        for loop_index in range(0, queries_move_num):
            self.query_l2.set_as(queries_l2_ub[loop_index])
            self.tik_instance.vec_dup(self.int32_mask,
                                      query_l2_ub, self.query_l2,
                                      1, self.int32_mask // 8)

            # -2xy
            self.tik_instance.vmul(16,
                                   add_ub[loop_index, 0],
                                   inner_product_ub[0, loop_index, 0], coeff_ub,
                                   code_align_16 // 16, 1, 1, 1, 2, queries_align * 2, 0)

            # x^2
            self.tik_instance.vadd(16,
                                   add_ub[loop_index, 0],
                                   add_ub[loop_index, 0], query_l2_ub,
                                   code_align_16 // 16, 1, 1, 1, 16 // 8, 16 // 8, 0)

            # y^2
            self.tik_instance.vadd(16,
                                   add_ub[loop_index, 0],
                                   add_ub[loop_index, 0], code_l2_ub,
                                   code_align_16 // 16, 1, 1, 1, 16 // 8, 16 // 8, 2)

        res_ub = self.tik_instance.Tensor("float16",
                                          (queries_move_num, self.code_num_each_loop),
                                          name="add_ub", scope=tik.scope_ubuf)

        if self.code_num_each_loop > self.int32_mask:
            for i in range(0, queries_move_num):
                self.tik_instance.vconv(self.int32_mask, 'none',
                                        res_ub[i, 0], add_ub[i, 0],
                                        self.code_num_each_loop // self.int32_mask, 1, 1, 4, 8, deqscale=self.scale)
        else:
            for i in range(0, queries_move_num):
                self.tik_instance.vconv(self.code_num_each_loop, 'none',
                                        res_ub[i, 0], add_ub[i, 0],
                                        1, 1, 1, 4, 8, deqscale=self.scale)

        self.tik_instance.data_move(
            self.output_distances_gm[queries_move_offset, aicore_move_offset + code_move_offset],
            res_ub,
            0, queries_move_num, code_align_16 // 16,
            0, (self.code_num - code_align_16) // 16)

    def distance_int8_l2(self):
        """
        the compute process
        """
        with self.tik_instance.for_range(0, self.aicore_use, block_num=self.aicore_use) as block_index:
            aicore_code_num = self.tik_instance.Scalar(dtype="uint32", name="aicore_code_num", init_value=0)
            # compute  code num and move offset every core
            with self.tik_instance.if_scope(block_index != self.aicore_use - 1):
                aicore_code_num.set_as(self.code_num_each_core)
            with self.tik_instance.else_scope():
                aicore_code_num.set_as(self.code_num_last_core)

            queries_loop_time = self.queries_num // self.queries_num_each_loop
            if queries_loop_time > 0:
                with self.tik_instance.for_range(0, queries_loop_time) as loop_queries:
                    self.distance_compute_each_loop(block_index * self.code_num_each_core, aicore_code_num,
                                                    loop_queries * self.queries_num_each_loop,
                                                    self.queries_num_each_loop)

            queries_last_num = self.queries_num % self.queries_num_each_loop
            if queries_last_num > 0:
                self.distance_compute_each_loop(block_index * self.code_num_each_core, aicore_code_num,
                                                queries_loop_time * self.queries_num_each_loop,
                                                queries_last_num)

            one = self.tik_instance.Scalar(dtype="uint16", name="one", init_value=1)
            flag_ub = self.tik_instance.Tensor("uint16", (16,), name="flag_ub", scope=tik.scope_ubuf)

            flag_ub[0].set_as(one)
            self.tik_instance.pipe_barrier("PIPE_MTE3")
            self.tik_instance.data_move(self.output_flag_gm[block_index * 16], flag_ub, 0, 1, 1, 0, 0)

    def get_tik_instance(self):
        """
        obtain tik instance
        """
        self.distance_int8_l2()

        self.tik_instance.BuildCCE(kernel_name=self.kernel_name,
                                   inputs=[
                                       self.input_queries_gm,
                                       self.input_code_gm,
                                       self.input_precomputed_gm,
                                   ],
                                   outputs=[self.output_distances_gm,
                                            self.output_flag_gm])

        return self.tik_instance


def distance_ivf_int8_l2_l1(input_queries,
                            input_code,
                            input_precomputed,
                            output_distances,
                            output_flag,
                            kernel_name="distance_ivf_int8_l2_l1"):
    """
    calculating distance

    Parameters
    ----------
    input_queries : dict
        shape and dtype of query vector
    input_code : dict
        shape and dtype of  code
    input_precomputed : dict
        shape and dtype of precomputed L2 distance of  code
    output_distances : dict
        shape and dtype of distances, should be same dtype as input_queries
    output_flag : dict
        shape and dtype of flag, only the 1th and 17th is valid
    kernel_name : str
        kernel name, default value is "distance_int8_l2"

    Returns
    -------
    None
    """
    distance_int8_l2_ = DistanceIVFInt8L2L1(input_queries, input_code, input_precomputed,
                                            output_distances, output_flag, kernel_name)
    tik_instance_ = distance_int8_l2_.get_tik_instance()
    return tik_instance_
