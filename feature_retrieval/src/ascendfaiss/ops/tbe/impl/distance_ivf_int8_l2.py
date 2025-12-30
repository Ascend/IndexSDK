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


class DistanceIVFInt8L2:
    def __init__(self,
                 input_queries,
                 input_code,
                 input_precomputed,
                 input_actual_num,
                 output_distances,
                 output_min_dist,
                 output_flag,
                 kernel_name="distance_ivf_int8_l2"):
        self.shape_queries = input_queries.get("shape")
        self.dtype_queries = input_queries.get("dtype")
        self.shape_code = input_code.get("shape")
        self.dtype_code = input_code.get("dtype")
        self.shape_precomputed = input_precomputed.get("shape")
        self.dtype_precomputed = input_precomputed.get("dtype")
        self.shape_actual_num = input_actual_num.get("shape")
        self.dtype_actual_num = input_actual_num.get("dtype")
        self.shape_distances = output_distances.get("shape")
        self.dtype_distances = output_distances.get("dtype")
        self.shape_min_dist = output_min_dist.get("shape")
        self.dtype_min_dist = output_min_dist.get("dtype")
        self.shape_flag = output_flag.get("shape")
        self.dtype_flag = output_flag.get("dtype")
        self.kernel_name = kernel_name

        # compute parameter
        self.queries_num, self.dim = self.shape_queries
        self.code_num, = self.shape_precomputed

        # check parameter
        self.check_parameter()

        # set vector fp32 mask and fp16 mask
        self.int32_mask = 64
        self.fp16_mask = 128
        self.min_mask = 32
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

        self.aicore_use = self.shape_actual_num[0]
        self.queries_align = 16

        self.code_loops = 2
        self.code_num_sub_loop = 512
        while self.code_num_sub_loop * self.dim > 256 * 1024:
            self.code_num_sub_loop = self.code_num_sub_loop // 2
        self.code_num_each_loop = self.code_num_sub_loop * self.code_loops

        self.set_src_dst_tensor()

    def set_src_dst_tensor(self):
        """
        set input and output tensor
        """
        self.coeff = self.tik_instance.Scalar("int32", name="coeff", init_value=-2)

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
        self.input_actual_num_gm = self.tik_instance.Tensor(self.dtype_actual_num,
                                                            self.shape_actual_num,
                                                            name="input_actual_num_gm",
                                                            scope=tik.scope_gm)
        self.output_distances_gm = self.tik_instance.Tensor(self.dtype_distances,
                                                            self.shape_distances,
                                                            name="output_distances_gm",
                                                            scope=tik.scope_gm)
        self.output_min_dist_gm = self.tik_instance.Tensor(self.dtype_min_dist,
                                                           self.shape_min_dist,
                                                           name="output_min_dist_gm",
                                                           scope=tik.scope_gm)
        self.output_flag_gm = self.tik_instance.Tensor(self.dtype_flag,
                                                       self.shape_flag,
                                                       name="output_flag_gm",
                                                       scope=tik.scope_gm)

    def cal_num_each_core(self):
        """
        calculate actual code num of each core
        """
        # move actual code num from out to UB
        actual_num_ub = self.tik_instance.Tensor("uint32",
                                                 (8,),
                                                 name="actual_code_num_ub", scope=tik.scope_ubuf)
        self.tik_instance.data_move(actual_num_ub,
                                    self.input_actual_num_gm,
                                    0, 1, 1, 0, 0)
        actual_num = self.tik_instance.Scalar(dtype="uint32", name="actual_code_num", init_value=0)
        actual_num.set_as(actual_num_ub[0])

        if self.aicore_use == 2:
            self.code_num_each_core = (actual_num // self.aicore_use + self.code_num_each_loop // 2) // \
                self.code_num_each_loop * self.code_num_each_loop
        else:
            self.code_num_each_core = actual_num // self.aicore_use // self.code_num_each_loop * self.code_num_each_loop
        self.code_num_last_core = actual_num - (self.aicore_use - 1) * self.code_num_each_core

    def distance_compute_each_loop(self, aicore_code_offset, aicore_code_num):
        queries_l1 = self.tik_instance.Tensor("int8",
                                              (self.dim // 32, self.queries_align, 32),
                                              name="queries_l1", scope=tik.scope_cbuf)

        self.tik_instance.data_move(queries_l1[0, 0, 0],
                                    self.input_queries_gm,
                                    0, self.dim // 32, 1, 0, self.queries_align - 1)

        queries_square_l0c = self.tik_instance.Tensor("int32",
                                                      (self.queries_align // 16, self.queries_align, 16),
                                                      name="queries_square_l0c", scope=tik.scope_cbuf_out)
        self.tik_instance.matmul(queries_square_l0c,
                                 queries_l1, queries_l1,
                                 self.queries_align, self.dim, self.queries_align)

        queries_square_ub = self.tik_instance.Tensor("int32",
                                                     (self.queries_align // 16, self.queries_align, 16),
                                                     name="queries_square_ub", scope=tik.scope_ubuf)
        self.tik_instance.data_move(queries_square_ub,
                                    queries_square_l0c, 0, 1,
                                    self.queries_align * self.queries_align // 256, 0, 0)

        query_square = self.tik_instance.Scalar("int32", name="query_square", init_value=0)
        query_square.set_as(queries_square_ub[0])

        # compute xy using cube
        code_loop_times = aicore_code_num // self.code_num_each_loop
        with self.tik_instance.if_scope(code_loop_times > 0):
            with self.tik_instance.for_range(0, code_loop_times) as loop_code:
                self.cube_compute_each_loop(queries_l1, query_square, aicore_code_offset,
                                            loop_code * self.code_num_each_loop)

        code_last_num = aicore_code_num % self.code_num_each_loop
        with self.tik_instance.if_scope(code_last_num > 0):
            self.cube_compute_last_loop(queries_l1, query_square, aicore_code_offset,
                                        code_loop_times * self.code_num_each_loop, code_last_num)

    def cube_compute_each_loop(self, queries_l1, query_square, aicore_code_offset, code_offset):

        with self.tik_instance.for_range(0, self.code_loops, thread_num=2) as loop_ub:
            code_l1 = self.tik_instance.Tensor("int8",
                                               (self.dim // 32, self.code_num_sub_loop, 32),
                                               name="code_l1", scope=tik.scope_cbuf)
            for i in range(self.dim // 32):
                self.tik_instance.data_move(code_l1[i, 0, 0],
                                            self.input_code_gm[(aicore_code_offset + code_offset
                                                                + loop_ub * self.code_num_sub_loop) // 16,
                                                               i, 0, 0],
                                            0, self.code_num_sub_loop // 16, 16, self.dim // 2 - 16, 0)

            inner_product_l0c = self.tik_instance.Tensor("int32",
                                                         (self.code_num_sub_loop // 16, self.queries_align, 16),
                                                         name="inner_product_l0c", scope=tik.scope_cbuf_out)
            self.tik_instance.matmul(inner_product_l0c,
                                     queries_l1, code_l1,
                                     self.queries_align, self.dim, self.code_num_sub_loop)

            # mov code l2 from out to UB
            code_l2_ub = self.tik_instance.Tensor("int32",
                                                  (self.code_num_sub_loop,),
                                                  name="code_l2_ub", scope=tik.scope_ubuf)
            self.tik_instance.data_move(code_l2_ub,
                                        self.input_precomputed_gm[aicore_code_offset + code_offset
                                                                  + loop_ub * self.code_num_sub_loop],
                                        0, 1, self.code_num_sub_loop // 8, 0, 0)

            query_l2_ub = self.tik_instance.Tensor("int32",
                                                   (self.int32_mask,),
                                                   name="query_l2_ub", scope=tik.scope_ubuf)
            self.tik_instance.vec_dup(self.int32_mask, query_l2_ub, query_square, 1, self.int32_mask // 8)

            coeff_ub = self.tik_instance.Tensor("int32",
                                                (16,),
                                                tik.scope_ubuf, "coeff_ub")
            self.tik_instance.vec_dup(16, coeff_ub, self.coeff, 1, 8)

            # mov xy from L0-C to UB
            inner_product_ub = self.tik_instance.Tensor("int32",
                                                        (self.code_num_sub_loop // 16, self.queries_align, 16),
                                                        name="inner_product_ub", scope=tik.scope_ubuf)
            self.tik_instance.tensor_mov(inner_product_ub,
                                         inner_product_l0c,
                                         'm',
                                         1, self.code_num_sub_loop * self.queries_align // 256,
                                         0, 0)

            res_ub = self.tik_instance.Tensor("int32",
                                              (self.code_num_sub_loop, ),
                                              name="res_ub", scope=tik.scope_ubuf)
            # -2xy
            self.tik_instance.vmul(16,
                                   res_ub,
                                   inner_product_ub, coeff_ub,
                                   self.code_num_sub_loop // 16,
                                   1, 1, 1, 2, self.queries_align * 2, 0)

            # x^2
            self.tik_instance.vadd(16, res_ub,
                                   res_ub, query_l2_ub,
                                   self.code_num_sub_loop // 16,
                                   1, 1, 1, 16 // 8, 16 // 8, 0)

            # y^2
            self.tik_instance.vadd(16, res_ub,
                                   res_ub, code_l2_ub,
                                   self.code_num_sub_loop // 16,
                                   1, 1, 1, 16 // 8, 16 // 8, 2)

            res_ub_fp16 = self.tik_instance.Tensor("float16",
                                                   (self.code_num_sub_loop,),
                                                   name="add_ub", scope=tik.scope_ubuf)
            self.tik_instance.vconv(self.int32_mask, 'none',
                                    res_ub_fp16, res_ub,
                                    self.code_num_sub_loop // self.int32_mask,
                                    1, 1, self.int32_mask // 16, self.int32_mask // 8, deqscale=self.scale)

            self.tik_instance.data_move(self.output_distances_gm[0, aicore_code_offset + code_offset
                                                                 + loop_ub * self.code_num_sub_loop],
                                        res_ub_fp16,
                                        0, 1, self.code_num_sub_loop // 16, 0, 0)

            min_ub = self.tik_instance.Tensor("float16",
                                              (self.code_num_sub_loop // 16, ),
                                              name="min_ub", scope=tik.scope_ubuf)
            repeat_times = self.code_num_sub_loop // self.min_mask
            self.tik_instance.vcmin(self.min_mask, min_ub, res_ub_fp16,
                                    repeat_times, 1, 1, self.min_mask // 16)
            self.tik_instance.data_move(
                self.output_min_dist_gm[0, (aicore_code_offset + code_offset
                                            + loop_ub * self.code_num_sub_loop) // self.min_mask * 2],
                min_ub,
                0, 1, self.code_num_sub_loop // self.min_mask * 2 // 16, 0, 0)

    def cube_compute_last_loop(self, queries_l1, query_square, aicore_code_offset, code_offset, code_move_num):
        code_num_align16 = (code_move_num + 15) // 16 * 16
        code_l1 = self.tik_instance.Tensor("int8",
                                           (self.dim // 32, self.code_num_each_loop, 32),
                                           name="code_l1", scope=tik.scope_cbuf)
        for i in range(self.dim // 32):
            self.tik_instance.data_move(code_l1[i, 0, 0],
                                        self.input_code_gm[(aicore_code_offset + code_offset) // 16, i, 0, 0],
                                        0, code_num_align16 // 16, 16, self.dim // 2 - 16, 0)

        inner_product_l0c = self.tik_instance.Tensor("int32",
                                                     (self.code_num_each_loop // 16, self.queries_align, 16),
                                                     name="inner_product_l0c", scope=tik.scope_cbuf_out)
        self.tik_instance.matmul(inner_product_l0c,
                                 queries_l1, code_l1,
                                 self.queries_align, self.dim, self.code_num_each_loop)

        query_l2_ub = self.tik_instance.Tensor("int32",
                                               (self.int32_mask,),
                                               name="query_l2_ub", scope=tik.scope_ubuf)
        self.tik_instance.vec_dup(self.int32_mask, query_l2_ub, query_square, 1, self.int32_mask // 8)

        coeff_ub = self.tik_instance.Tensor("int32",
                                            (16,),
                                            tik.scope_ubuf, "coeff_ub")
        self.tik_instance.vec_dup(16, coeff_ub, self.coeff, 1, 8)

        # mov xy from L0-C to UB
        inner_product_ub = self.tik_instance.Tensor("int32",
                                                    (self.code_num_each_loop // 16, self.queries_align, 16),
                                                    name="inner_product_ub", scope=tik.scope_ubuf)
        self.tik_instance.tensor_mov(inner_product_ub,
                                     inner_product_l0c,
                                     'm',
                                     1, code_num_align16 * self.queries_align // 256,
                                     0, 0)

        # mov code l2 from out to UB
        code_l2_ub = self.tik_instance.Tensor("int32",
                                              (self.code_num_each_loop,),
                                              name="code_l2_ub", scope=tik.scope_ubuf)
        self.tik_instance.data_move(code_l2_ub,
                                    self.input_precomputed_gm[aicore_code_offset + code_offset],
                                    0, 1, (code_move_num + 7) // 8, 0, 0)

        res_ub = self.tik_instance.Tensor("int32",
                                          (self.code_num_each_loop, ),
                                          name="res_ub", scope=tik.scope_ubuf)
        # -2xy
        self.tik_instance.vmul(16,
                               res_ub,
                               inner_product_ub, coeff_ub,
                               self.code_num_each_loop // 16,
                               1, 1, 1, 2, self.queries_align * 2, 0)

        # x^2
        self.tik_instance.vadd(16,
                               res_ub,
                               res_ub, query_l2_ub,
                               self.code_num_each_loop // 16,
                               1, 1, 1, 16 // 8, 16 // 8, 0)

        # y^2
        self.tik_instance.vadd(16,
                               res_ub,
                               res_ub, code_l2_ub,
                               self.code_num_each_loop // 16,
                               1, 1, 1, 16 // 8, 16 // 8, 2)

        res_ub_fp16 = self.tik_instance.Tensor("float16",
                                               (self.code_num_each_loop,),
                                               name="add_ub", scope=tik.scope_ubuf)
        self.tik_instance.vconv(self.int32_mask, 'none',
                                res_ub_fp16, res_ub,
                                self.code_num_each_loop // self.int32_mask,
                                1, 1, self.int32_mask // 16, self.int32_mask // 8, deqscale=self.scale)

        self.tik_instance.data_move(self.output_distances_gm[0, aicore_code_offset + code_offset],
                                    res_ub_fp16,
                                    0, 1, code_num_align16 // 16, 0, 0)
        min_ub = self.tik_instance.Tensor("float16",
                                          (self.code_num_each_loop // 16, ),
                                          name="min_ub", scope=tik.scope_ubuf)
        repeat_times = code_move_num // self.min_mask
        with self.tik_instance.if_scope(repeat_times > 0):
            self.tik_instance.vcmin(self.min_mask, min_ub, res_ub_fp16,
                                    repeat_times, 1, 1, self.min_mask // 16)
        offset = repeat_times * self.min_mask
        last_mask = code_move_num % self.min_mask
        with self.tik_instance.if_scope(last_mask > 0):
            self.tik_instance.vcmin(last_mask, min_ub[repeat_times * 2], res_ub_fp16[offset],
                                    1, 1, 1, self.min_mask // 16)
        self.tik_instance.data_move(
            self.output_min_dist_gm[0, (aicore_code_offset + code_offset) // self.min_mask * 2],
            min_ub,
            0, 1, ((code_move_num + self.min_mask - 1) // self.min_mask * 2 + 15) // 16, 0, 0)

    def distance_ivf_int8_l2(self):
        """
        the compute process
        """
        self.cal_num_each_core()

        with self.tik_instance.for_range(0, self.aicore_use, block_num=self.aicore_use) as block_index:
            aicore_code_num = self.tik_instance.Scalar(dtype="uint32", name="aicore_code_num", init_value=0)
            # compute code num and move offset every core
            with self.tik_instance.if_scope(block_index != self.aicore_use - 1):
                aicore_code_num.set_as(self.code_num_each_core)
            with self.tik_instance.else_scope():
                aicore_code_num.set_as(self.code_num_last_core)

            with self.tik_instance.if_scope(aicore_code_num > 0):
                self.distance_compute_each_loop(block_index * self.code_num_each_core, aicore_code_num)

            one = self.tik_instance.Scalar(dtype="uint16", name="one", init_value=1)
            flag_ub = self.tik_instance.Tensor("uint16",
                                               (16,),
                                               name="flag_ub", scope=tik.scope_ubuf)

            flag_ub[0].set_as(one)
            self.tik_instance.pipe_barrier("PIPE_MTE3")
            self.tik_instance.data_move(self.output_flag_gm[block_index, 0],
                                        flag_ub,
                                        0, 1, 1, 0, 0)

    def get_tik_instance(self):
        """
        obtain tik instance
        """
        self.distance_ivf_int8_l2()

        self.tik_instance.BuildCCE(kernel_name=self.kernel_name,
                                   inputs=[
                                       self.input_queries_gm,
                                       self.input_code_gm,
                                       self.input_precomputed_gm,
                                       self.input_actual_num_gm
                                   ],
                                   outputs=[self.output_distances_gm, self.output_min_dist_gm, self.output_flag_gm])

        return self.tik_instance


def distance_ivf_int8_l2(input_queries,
                         input_code,
                         input_precomputed,
                         input_actual_num,
                         output_distances,
                         output_min_dist,
                         output_flag,
                         kernel_name="distance_ivf_int8_l2"):
    """
    calculating distance

    Parameters
    ----------
    input_queries : dict
        shape and dtype of query vector
    input_code : dict
        shape and dtype of  code
    input_precomputed : dict
        shape and dtype of precomputed L2 distance of code
    input_actual_num: dict
        shape and dtype of actual code num,
        shape must be (8,) and dtype must be uint32
    output_distances : dict
        shape and dtype of distances, should be same dtype as input_queries
    output_min_dist : dict
        shape and dtype of min distances, should be same dtype as input_queries
    output_flag : dict
        shape and dtype of flag, only the 1th and 17th is valid
    kernel_name : str
        kernel name, default value is "distance_ivf_int8_l2"

    Returns
    -------
    None
    """
    distance_ivf_int8_ = DistanceIVFInt8L2(input_queries,
                                           input_code,
                                           input_precomputed,
                                           input_actual_num,
                                           output_distances,
                                           output_min_dist,
                                           output_flag,
                                           kernel_name)
    tik_instance_ = distance_ivf_int8_.get_tik_instance()
    return tik_instance_
