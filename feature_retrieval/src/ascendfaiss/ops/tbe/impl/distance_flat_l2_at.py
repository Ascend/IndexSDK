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


class DistanceFlatL2At:
    def __init__(self,
                 input_queries,
                 input_actual_queries_num,
                 input_code,
                 input_code_norm,
                 input_dm,
                 output_sub_centroids_cores,
                 output_hassign_cores,
                 output_hassign_queries,
                 output_flag,
                 kernel_name="distance_flat_l2_at"):
        self.shape_queries = input_queries.get("shape")
        self.dtype_queries = input_queries.get("dtype")
        self.shape_actual_queries_num = input_actual_queries_num.get("shape")
        self.dtype_actual_queries_num = input_actual_queries_num.get("dtype")
        self.shape_code = input_code.get("shape")
        self.dtype_code = input_code.get("dtype")
        self.shape_code_norm = input_code_norm.get("shape")
        self.dtype_code_norm = input_code_norm.get("dtype")
        self.shape_dm = input_dm.get("shape")
        self.dtype_dm = input_dm.get("dtype")
        self.shape_sub_centroids_cores = output_sub_centroids_cores.get("shape")
        self.dtype_sub_centroids_cores = output_sub_centroids_cores.get("dtype")
        self.shape_hassign_cores = output_hassign_cores.get("shape")
        self.dtype_hassign_cores = output_hassign_cores.get("dtype")
        self.shape_hassign_queries = output_hassign_queries.get("shape")
        self.dtype_hassign_queries = output_hassign_queries.get("dtype")

        self.shape_flag = output_flag.get("shape")
        self.dtype_flag = output_flag.get("dtype")
        self.kernel_name = kernel_name

        self.code_num = 64

        self.dim = self.shape_code[1] * 16
        self.cube_align = 16
        self.min_batch = 64
        self.max_repeats = 128
        self.transfer_mask = 64
        self.vcadd_mask = 16

        # set vector fp32 mask and fp16 mask
        self.fp16_mask = 128
        self.fp32_mask = 64

        self.query_each_loop = 32
        self.sub_loops = 2
        self.query_sub_loop = self.query_each_loop // self.sub_loops

        # set tik instance
        self.set_tik_instance()

        # check parameter
        self.check_parameter()

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
       
        self.set_src_dst_tensor()

    def set_src_dst_tensor(self):
        """
        set input and output tensor
        """
        self.coeff = self.tik_instance.Scalar("float32", name="coeff", init_value=-2)

        # creat input tensor: input_queries_gm, input_code_gm
        # and input_queries_norm_gm
        # and output tensor: output_distances_gm, output_flag_gm in global buffer
        self.input_queries_gm = self.tik_instance.Tensor(
            self.dtype_queries,
            self.shape_queries,
            name="input_queries_gm",
            scope=tik.scope_gm)
        self.input_actual_queries_num_gm = self.tik_instance.Tensor(
            self.dtype_actual_queries_num, 
            self.shape_actual_queries_num,
            name="input_actual_queries_num_gm",
            scope=tik.scope_gm)
        self.input_code_gm = self.tik_instance.Tensor(
            self.dtype_code,
            self.shape_code,
            name="input_code_gm",
            scope=tik.scope_gm)
        self.input_code_norm_gm = self.tik_instance.Tensor(
            self.dtype_code_norm,
            self.shape_code_norm,
            name="input_code_norm_gm",
            scope=tik.scope_gm)
        self.input_dm_gm = self.tik_instance.Tensor(
            self.dtype_dm,
            self.shape_dm,
            name="input_dm_gm",
            scope=tik.scope_gm)
        self.output_sub_centroids_cores_gm = self.tik_instance.Tensor(
            self.dtype_sub_centroids_cores, 
            self.shape_sub_centroids_cores,
            name="output_sub_centroids_cores_gm",
            scope=tik.scope_gm)
        self.output_hassign_cores_gm = self.tik_instance.Tensor(
            self.dtype_hassign_cores, 
            self.shape_hassign_cores, 
            name="output_hassign_cores_gm",
            scope=tik.scope_gm)
        self.output_hassign_queries_gm = self.tik_instance.Tensor(
            self.dtype_hassign_queries,
            self.shape_hassign_queries,
            name="output_hassign_queries_gm",
            scope=tik.scope_gm)
        self.output_flag_gm = self.tik_instance.Tensor(
            self.dtype_flag,
            self.shape_flag,
            name="output_flag_gm",
            scope=tik.scope_gm)

    def cal_num_each_core(self):
        self.num = self.tik_instance.Scalar("uint32")
        actual_queries_num_ub = self.tik_instance.Tensor("uint32",
                                                 (8,),
                                                 name="actual_queries_num_ub", scope=tik.scope_ubuf)
        self.tik_instance.data_move(actual_queries_num_ub, self.input_actual_queries_num_gm, 0, 1, 1, 0, 0)
        self.num.set_as(actual_queries_num_ub[0])
        self.aicore_use = self.tik_instance.Scalar("uint32")
        self.aicore_use.set_as(actual_queries_num_ub[1])

        self.query_each_core = self.num // self.aicore_use // self.query_each_loop * self.query_each_loop
        self.query_loops = self.query_each_core // self.query_each_loop
        self.query_last_core = self.num - self.query_each_core * (self.aicore_use - 1)
        self.query_last_loops = (self.query_last_core + self.query_each_loop - 1) // self.query_each_loop
        self.query_last_num = self.query_last_core - (self.query_last_loops - 1) * self.query_each_loop 

    def distance_compute_each_loop(self, aicore_id, aicore_offset):
        min_ub = self.tik_instance.Tensor("float16",
                                          (self.dim,),
                                          name="min_ub", scope=tik.scope_ubuf)
        diff_ub = self.tik_instance.Tensor("float16",
                                           (self.dim,),
                                           name="diff_ub", scope=tik.scope_ubuf)

        # move diff adn min from out to UB
        self.tik_instance.data_move(min_ub, self.input_dm_gm,
                                    0, 1, self.dim // 16, 0, 0)
        self.tik_instance.data_move(diff_ub, self.input_dm_gm[self.dim],
                                    0, 1, self.dim // 16, 0, 0)

        code_norm_l1 = self.tik_instance.Tensor(
                                "float32",
                                (1, self.query_sub_loop // 16, self.code_num, 16),
                                name="code_norm_l1", scope=tik.scope_cbuf)
        self.tik_instance.data_move(code_norm_l1,
                                    self.input_code_norm_gm[0, 0, 0, 0],
                                    0, 1, self.query_sub_loop * self.code_num // 8, 0, 0)
        code_l1 = self.tik_instance.Tensor(
                                "float16",
                                (1, self.dim // 16, self.code_num, 16),
                                name="code_l1", scope=tik.scope_cbuf)
        sub_centroids_ub = self.tik_instance.Tensor("float16", (self.code_num, self.dim),
                                                    name="sub_centroids_ub", scope=tik.scope_ubuf)
        self.tik_instance.vec_dup(self.fp16_mask, sub_centroids_ub, 0, self.code_num * self.dim // self.fp16_mask, 8)
        hassign_ub = self.tik_instance.Tensor("int16", (self.code_num, ), name="hassign", scope=tik.scope_ubuf)
        self.tik_instance.vec_dup(self.code_num, hassign_ub, 0, 1, 8)

        self.tik_instance.data_move(code_l1,
                                    self.input_code_gm[0, 0, 0, 0],
                                    0, 1, self.code_num * self.dim // 16, 0, 0)

        # compute xy using cube
        with self.tik_instance.for_range(0, self.query_loops) as loop:
            self.cube_compute_each_loop(code_l1, code_norm_l1, sub_centroids_ub, hassign_ub, diff_ub, min_ub,
                                        aicore_offset + loop * self.query_each_loop)

        self.tik_instance.data_move(self.output_sub_centroids_cores_gm[aicore_id, 0, 0],
                                    sub_centroids_ub, 0, 1, self.code_num * self.dim // 16, 0, 0)
        self.tik_instance.data_move(self.output_hassign_cores_gm[aicore_id, 0],
                                    hassign_ub, 0, 1, self.code_num // 16, 0, 0)

    def distance_compute_last_loop(self, aicore_id, aicore_offset):
        min_ub = self.tik_instance.Tensor("float16",
                                          (self.dim,),
                                          name="min_ub", scope=tik.scope_ubuf)
        diff_ub = self.tik_instance.Tensor("float16",
                                           (self.dim,),
                                           name="diff_ub", scope=tik.scope_ubuf)
        # move diff adn min from out to UB
        self.tik_instance.data_move(min_ub, self.input_dm_gm,
                                    0, 1, self.dim // 16, 0, 0)
        self.tik_instance.data_move(diff_ub, self.input_dm_gm[self.dim],
                                    0, 1, self.dim // 16, 0, 0)

        code_norm_l1 = self.tik_instance.Tensor(
                                "float32",
                                (1, self.query_sub_loop // 16, self.code_num, 16),
                                name="code_norm_l1", scope=tik.scope_cbuf)
        self.tik_instance.data_move(code_norm_l1,
                                    self.input_code_norm_gm[0, 0, 0, 0],
                                    0, 1, self.query_sub_loop * self.code_num // 8, 0, 0)
        code_l1 = self.tik_instance.Tensor(
                                "float16",
                                (1, self.dim // 16, self.code_num, 16),
                                name="code_l1", scope=tik.scope_cbuf)
        sub_centroids_ub = self.tik_instance.Tensor("float16", (self.code_num, self.dim),
                                                    name="sub_centroids_ub", scope=tik.scope_ubuf)
        self.tik_instance.vec_dup(self.fp16_mask, sub_centroids_ub, 0, self.code_num * self.dim // self.fp16_mask, 8)
        hassign_ub = self.tik_instance.Tensor("int16", (self.code_num, ), name="hassign", scope=tik.scope_ubuf)
        self.tik_instance.vec_dup(self.code_num, hassign_ub, 0, 1, 8)

        self.tik_instance.data_move(code_l1,
                                    self.input_code_gm[0, 0, 0, 0],
                                    0, 1, self.code_num * self.dim // 16, 0, 0)
        # compute xy using cube
        with self.tik_instance.for_range(0, self.query_last_loops - 1) as loop:
            self.cube_compute_each_loop(code_l1, code_norm_l1, sub_centroids_ub, hassign_ub, diff_ub, min_ub,
                                        aicore_offset + loop * self.query_each_loop)

        self.cube_compute_last_loop(code_l1, code_norm_l1, sub_centroids_ub, hassign_ub, diff_ub, min_ub,
                                        aicore_offset + (self.query_last_loops - 1) * self.query_each_loop)

        self.tik_instance.data_move(self.output_sub_centroids_cores_gm[aicore_id, 0, 0],
                                    sub_centroids_ub, 0, 1, self.code_num * self.dim // 16, 0, 0)
        self.tik_instance.data_move(self.output_hassign_cores_gm[aicore_id, 0],
                                    hassign_ub, 0, 1, self.code_num // 16, 0, 0)

    def cube_compute_each_loop(self, code_l1, code_norm_l1, sub_centroids_ub, hassign_ub, diff_ub, min_ub,
                               query_offset):
        dst_ub = self.tik_instance.Tensor("float16",
                                          (self.query_each_loop, self.code_num),
                                          name="dst_ub", scope=tik.scope_ubuf)
        queries_ub_fp16 = self.tik_instance.Tensor("float16",
                                                   (self.query_each_loop, self.dim),
                                                   name="queries_ub_fp16", scope=tik.scope_ubuf)

        with self.tik_instance.for_range(0, self.sub_loops, thread_num=2) as loop_ub:
            queries_ub = self.tik_instance.Tensor("uint8", (self.query_sub_loop, self.dim),
                                                  name="queries_ub", scope=tik.scope_ubuf)
            self.tik_instance.data_move(queries_ub,
                                        self.input_queries_gm[query_offset + loop_ub * self.query_sub_loop, 0],
                                        0, 1, self.query_sub_loop * self.dim // 32, 0, 0)
            self.tik_instance.vconv(self.fp16_mask, "none", queries_ub_fp16[loop_ub * self.query_sub_loop, 0],
                                    queries_ub, self.query_sub_loop * self.dim // self.fp16_mask, 1, 1, 8, 4)
            self.tik_instance.vadds(self.fp16_mask, queries_ub_fp16[loop_ub * self.query_sub_loop, 0],
                                    queries_ub_fp16[loop_ub * self.query_sub_loop, 0], 0.5,
                                    self.query_sub_loop * self.dim // self.fp16_mask, 1, 1, 8, 8)
            self.tik_instance.vmuls(self.fp16_mask, queries_ub_fp16[loop_ub * self.query_sub_loop, 0],
                                    queries_ub_fp16[loop_ub * self.query_sub_loop, 0], 1 / 255,
                                    self.query_sub_loop * self.dim // self.fp16_mask, 1, 1, 8, 8)
            if self.dim <= 128:
                self.tik_instance.vmul(self.dim, queries_ub_fp16[loop_ub * self.query_sub_loop, 0],
                                       queries_ub_fp16[loop_ub * self.query_sub_loop, 0],
                                       diff_ub, self.query_sub_loop, 1, 1, 1,
                                       self.dim // 16, self.dim // 16, 0)
                self.tik_instance.vadd(self.dim, queries_ub_fp16[loop_ub * self.query_sub_loop, 0],
                                       queries_ub_fp16[loop_ub * self.query_sub_loop, 0],
                                       min_ub, self.query_sub_loop, 1, 1, 1,
                                       self.dim // 16, self.dim // 16, 0)
            else:
                with self.tik_instance.for_range(0, self.query_sub_loop) as i:
                    repeat = self.dim // self.fp16_mask
                    offset = 0
                    if repeat > 0:
                        self.tik_instance.vmul(self.fp16_mask,
                                               queries_ub_fp16[i + loop_ub * self.query_sub_loop, offset],
                                               queries_ub_fp16[i + loop_ub * self.query_sub_loop, offset],
                                               diff_ub[offset], repeat, 1, 1,
                                               1, 8, 8, 8)
                        self.tik_instance.vadd(self.fp16_mask,
                                               queries_ub_fp16[i + loop_ub * self.query_sub_loop, offset],
                                               queries_ub_fp16[i + loop_ub * self.query_sub_loop, offset],
                                               min_ub[offset], repeat, 1, 1,
                                               1, 8, 8, 8)
                        offset += repeat * self.fp16_mask

                    remain = self.dim % self.fp16_mask
                    if remain > 0:
                        self.tik_instance.vmul(remain, queries_ub_fp16[i + loop_ub * self.query_sub_loop, offset],
                                               queries_ub_fp16[i + loop_ub * self.query_sub_loop, offset],
                                               diff_ub[offset], 1, 1, 1, 1, 8,
                                               8, 8)
                        self.tik_instance.vadd(remain, queries_ub_fp16[i + loop_ub * self.query_sub_loop, offset],
                                               queries_ub_fp16[i + loop_ub * self.query_sub_loop, offset],
                                               min_ub[offset], 1, 1, 1, 1, 8,
                                               8, 8)

            queries_l1 = self.tik_instance.Tensor("float16",
                                                  (self.dim // 16, self.query_sub_loop, 16),
                                                  name="queries_l1", scope=tik.scope_cbuf)

            for i in range(self.query_sub_loop):
                self.tik_instance.data_move(queries_l1[0, i, 0],
                                            queries_ub_fp16[i + self.query_sub_loop * loop_ub, 0],
                                            0, self.dim // 16, 1, 0, self.query_sub_loop - 1)

            inner_product_l0c = self.tik_instance.Tensor("float32",
                                                         (self.query_sub_loop // 16, self.code_num, 16),
                                                         name="inner_product_l0c", scope=tik.scope_cbuf_out)
            self.tik_instance.matmul(inner_product_l0c,
                                     code_l1, queries_l1,
                                     self.code_num, self.dim, self.query_sub_loop)

            add_ub = self.tik_instance.Tensor("float32",
                                              (self.query_sub_loop // 16, self.code_num, 16),
                                              name="add_ub", scope=tik.scope_ubuf)
            self.tik_instance.data_move(add_ub,
                                        code_norm_l1[0, 0, 0, 0],
                                        0, 1, self.code_num * self.query_sub_loop // 8, 0, 0)

            # mov xy from L0-C to UB
            inner_product_ub = self.tik_instance.Tensor("float32",
                                                        (self.query_sub_loop // 16, self.code_num, 16),
                                                        name="inner_product_ub", scope=tik.scope_ubuf)
            self.tik_instance.data_move(inner_product_ub,
                                        inner_product_l0c,
                                        0, 1, self.query_sub_loop * self.code_num // 256, 0, 0)

            vapxy_repeats = self.query_sub_loop * self.code_num // self.fp32_mask
            self.tik_instance.vaxpy(self.fp32_mask,
                                    add_ub, inner_product_ub, self.coeff,
                                    vapxy_repeats, 1, 1, 8, 8)

            dst_ub_ = self.tik_instance.Tensor("float16",
                                               (self.query_sub_loop // 16, self.code_num // 16, 16, 16),
                                               name="dst_ub", scope=tik.scope_ubuf)

            vconv_repeats = self.query_sub_loop * self.code_num // self.fp32_mask
            self.tik_instance.vconv(self.fp32_mask, "none",
                                    dst_ub_, add_ub,
                                    vconv_repeats, 1, 1, 4, 8)

            trans_repeats = self.query_sub_loop * self.code_num // 256
            self.tik_instance.vec_trans(dst_ub_, dst_ub_, trans_repeats, 1, 1)

            for i in range(self.query_sub_loop):
                self.tik_instance.data_move(dst_ub[loop_ub * self.query_sub_loop + i, 0],
                                            dst_ub_[i // 16, 0, i % 16, 0],
                                            0, self.code_num // 16, 1, 15, 0)

        dst_min_ub = self.tik_instance.Tensor("float16",
                                    (self.query_each_loop * 2, ),
                                    name="dst_min_ub", scope=tik.scope_ubuf)

        self.tik_instance.vcmin(self.min_batch, dst_min_ub, dst_ub,
                                self.query_each_loop, 1, 1, self.code_num // 16)

        dst_min_index_ub = self.tik_instance.Tensor("float16",
                                          (self.query_each_loop, ),
                                          name="dst_min_index_ub", scope=tik.scope_ubuf)

        self.tik_instance.vreduce(self.query_each_loop * 2, dst_min_index_ub,
                                  dst_min_ub, 2, 1, 1, 0, 0, 0, None, "counter")
        dst_min_index_ub_uint16 = self.tik_instance.Tensor("uint16",
                                                           (self.query_each_loop, ),
                                                           name="dst_min_index_ub_uint16",
                                                           scope=tik.scope_ubuf)
        dst_min_index_ub_uint16 = dst_min_index_ub.reinterpret_cast_to("uint16")
        self.tik_instance.data_move(self.output_hassign_queries_gm[query_offset],
                                    dst_min_index_ub_uint16, 0, 1, self.query_each_loop // 16, 0, 0)

        with self.tik_instance.for_range(0, self.query_each_loop, thread_num=2) as loop_i:
            idx = self.tik_instance.Scalar("uint16", name="idx", init_value=dst_min_index_ub_uint16[loop_i])
            self.tik_instance.vadd(self.fp16_mask, sub_centroids_ub[idx, 0], 
                                   sub_centroids_ub[idx, 0], queries_ub_fp16[loop_i, 0],
                                   self.dim // self.fp16_mask, 1, 1, 1, 8, 8, 8)
            self.tik_instance.vadd(self.code_num, hassign_ub, hassign_ub,
                                   self.diag_matrix_ub[idx, 0], 1, 1, 1, 1, 8, 8, 8)

    def cube_compute_last_loop(self, code_l1, code_norm_l1, sub_centroids_ub, hassign_ub, diff_ub, min_ub,
                               query_offset):
        dst_ub = self.tik_instance.Tensor("float16",
                                          (self.query_each_loop, self.code_num),
                                          name="dst_ub", scope=tik.scope_ubuf)
        queries_ub_fp16 = self.tik_instance.Tensor("float16",
                                                   (self.query_each_loop, self.dim),
                                                   name="queries_ub_fp16", scope=tik.scope_ubuf)

        with self.tik_instance.for_range(0, self.sub_loops, thread_num=2) as loop_ub:
            queries_ub = self.tik_instance.Tensor("uint8", (self.query_sub_loop, self.dim),
                                                  name="queries_ub", scope=tik.scope_ubuf)
            self.tik_instance.data_move(queries_ub,
                                        self.input_queries_gm[query_offset + loop_ub * self.query_sub_loop, 0],
                                        0, 1, self.query_sub_loop * self.dim // 32, 0, 0)

            self.tik_instance.vconv(self.fp16_mask, "none", queries_ub_fp16[loop_ub * self.query_sub_loop, 0],
                                    queries_ub, self.query_sub_loop * self.dim // self.fp16_mask, 1, 1, 8, 4)
            self.tik_instance.vadds(self.fp16_mask, queries_ub_fp16[loop_ub * self.query_sub_loop, 0],
                                    queries_ub_fp16[loop_ub * self.query_sub_loop, 0], 0.5,
                                    self.query_sub_loop * self.dim // self.fp16_mask, 1, 1, 8, 8)
            self.tik_instance.vmuls(self.fp16_mask, queries_ub_fp16[loop_ub * self.query_sub_loop, 0],
                                    queries_ub_fp16[loop_ub * self.query_sub_loop, 0], 1 / 255,
                                    self.query_sub_loop * self.dim // self.fp16_mask, 1, 1, 8, 8)
            if self.dim <= 128:
                self.tik_instance.vmul(self.dim, queries_ub_fp16[loop_ub * self.query_sub_loop, 0],
                                       queries_ub_fp16[loop_ub * self.query_sub_loop, 0],
                                       diff_ub, self.query_sub_loop, 1, 1, 1,
                                       self.dim // 16, self.dim // 16, 0)
                self.tik_instance.vadd(self.dim, queries_ub_fp16[loop_ub * self.query_sub_loop, 0],
                                       queries_ub_fp16[loop_ub * self.query_sub_loop, 0],
                                       min_ub, self.query_sub_loop, 1, 1, 1,
                                       self.dim // 16, self.dim // 16, 0)
            else:
                with self.tik_instance.for_range(0, self.query_sub_loop) as i:
                    repeat = self.dim // self.fp16_mask
                    offset = 0
                    if repeat > 0:
                        self.tik_instance.vmul(self.fp16_mask,
                                               queries_ub_fp16[i + loop_ub * self.query_sub_loop, offset],
                                               queries_ub_fp16[i + loop_ub * self.query_sub_loop, offset],
                                               diff_ub[offset], repeat, 1, 1,
                                               1, 8, 8, 8)
                        self.tik_instance.vadd(self.fp16_mask,
                                               queries_ub_fp16[i + loop_ub * self.query_sub_loop, offset],
                                               queries_ub_fp16[i + loop_ub * self.query_sub_loop, offset],
                                               min_ub[offset], repeat, 1, 1,
                                               1, 8, 8, 8)
                        offset += repeat * self.fp16_mask

                    remain = self.dim % self.fp16_mask
                    if remain > 0:
                        self.tik_instance.vmul(remain, queries_ub_fp16[i + loop_ub * self.query_sub_loop, offset],
                                               queries_ub_fp16[i + loop_ub * self.query_sub_loop, offset],
                                               diff_ub[offset], 1, 1, 1, 1, 8,
                                               8, 8)
                        self.tik_instance.vadd(remain, queries_ub_fp16[i + loop_ub * self.query_sub_loop, offset],
                                               queries_ub_fp16[i + loop_ub * self.query_sub_loop, offset],
                                               min_ub[offset], 1, 1, 1, 1, 8,
                                               8, 8)

            queries_l1 = self.tik_instance.Tensor("float16",
                                                  (self.dim // 16, self.query_sub_loop, 16),
                                                  name="queries_l1", scope=tik.scope_cbuf)

            for i in range(self.query_sub_loop):
                self.tik_instance.data_move(queries_l1[0, i, 0],
                                            queries_ub_fp16[i + self.query_sub_loop * loop_ub, 0],
                                            0, self.dim // 16, 1, 0, self.query_sub_loop - 1)

            inner_product_l0c = self.tik_instance.Tensor("float32",
                                                         (self.query_sub_loop // 16, self.code_num, 16),
                                                         name="inner_product_l0c", scope=tik.scope_cbuf_out)
            self.tik_instance.matmul(inner_product_l0c,
                                     code_l1, queries_l1,
                                     self.code_num, self.dim, self.query_sub_loop)

            add_ub = self.tik_instance.Tensor("float32",
                                              (self.query_sub_loop // 16, self.code_num, 16),
                                              name="add_ub", scope=tik.scope_ubuf)
            self.tik_instance.data_move(add_ub,
                                        code_norm_l1[0, 0, 0, 0],
                                        0, 1, self.code_num * self.query_sub_loop // 8, 0, 0)

            # mov xy from L0-C to UB
            inner_product_ub = self.tik_instance.Tensor("float32",
                                                        (self.query_sub_loop // 16, self.code_num, 16),
                                                        name="inner_product_ub", scope=tik.scope_ubuf)
            self.tik_instance.data_move(inner_product_ub,
                                        inner_product_l0c,
                                        0, 1, self.query_sub_loop * self.code_num // 256, 0, 0)

            vapxy_repeats = self.query_sub_loop * self.code_num // self.fp32_mask
            self.tik_instance.vaxpy(self.fp32_mask,
                                    add_ub, inner_product_ub, self.coeff,
                                    vapxy_repeats, 1, 1, 8, 8)

            dst_ub_ = self.tik_instance.Tensor("float16",
                                               (self.query_sub_loop // 16, self.code_num // 16, 16, 16),
                                               name="dst_ub", scope=tik.scope_ubuf)

            vconv_repeats = self.query_sub_loop * self.code_num // self.fp32_mask
            self.tik_instance.vconv(self.fp32_mask, "none",
                                    dst_ub_, add_ub,
                                    vconv_repeats, 1, 1, 4, 8)

            trans_repeats = self.query_sub_loop * self.code_num // 256
            self.tik_instance.vec_trans(dst_ub_, dst_ub_, trans_repeats, 1, 1)

            for i in range(self.query_sub_loop):
                self.tik_instance.data_move(dst_ub[loop_ub * self.query_sub_loop + i, 0],
                                            dst_ub_[i // 16, 0, i % 16, 0],
                                            0, self.code_num // 16, 1, 15, 0)
        dst_min_ub = self.tik_instance.Tensor("float16",
                                    (self.query_each_loop * 2, ),
                                    name="dst_min_ub", scope=tik.scope_ubuf)

        self.tik_instance.vcmin(self.min_batch, dst_min_ub, dst_ub,
                                self.query_each_loop, 1, 1, self.code_num // 16)

        dst_min_index_ub = self.tik_instance.Tensor("float16",
                                          (self.query_each_loop, ),
                                          name="dst_min_index_ub", scope=tik.scope_ubuf)

        self.tik_instance.vreduce(self.query_each_loop * 2, dst_min_index_ub, dst_min_ub,
                                  2, 1, 1, 0, 0, 0, None, "counter")
        dst_min_index_ub_uint16 = self.tik_instance.Tensor("uint16",
                                    (self.query_each_loop, ),
                                    name="dst_min_index_ub_uint16", scope=tik.scope_ubuf)
        dst_min_index_ub_uint16 = dst_min_index_ub.reinterpret_cast_to("uint16")

        self.tik_instance.data_move(self.output_hassign_queries_gm[query_offset], dst_min_index_ub_uint16, 0, 1,
                                    self.query_each_loop // 16, 0, 0)

        with self.tik_instance.for_range(0, self.query_each_loop, thread_num=2) as loop_i:
            with self.tik_instance.if_scope(query_offset + loop_i < self.num):
                idx = self.tik_instance.Scalar("uint16", name="idx", init_value=dst_min_index_ub_uint16[loop_i])
                self.tik_instance.vadd(self.fp16_mask, sub_centroids_ub[idx, 0],
                                       sub_centroids_ub[idx, 0], queries_ub_fp16[loop_i, 0],
                                       self.dim // self.fp16_mask, 1, 1, 1, 8, 8, 8)
                self.tik_instance.vadd(self.code_num, hassign_ub, hassign_ub,
                                       self.diag_matrix_ub[idx, 0], 1, 1, 1, 1, 8, 8, 8)

    def dump_zero_each_loop(self, aicore_id, aicore_offset):

        sub_centroids_ub = self.tik_instance.Tensor("float16", (self.code_num, self.dim),
                                                    name="sub_centroids_ub", scope=tik.scope_ubuf)
        self.tik_instance.vec_dup(self.fp16_mask, sub_centroids_ub, 0, self.code_num * self.dim // self.fp16_mask, 8)
        hassign_ub = self.tik_instance.Tensor("int16", (self.code_num, ), name="hassign", scope=tik.scope_ubuf)
        self.tik_instance.vec_dup(self.code_num, hassign_ub, 0, 1, 8)



        self.tik_instance.data_move(self.output_sub_centroids_cores_gm[aicore_id, 0, 0],
                                    sub_centroids_ub, 0, 1, self.code_num * self.dim // 16, 0, 0)
        self.tik_instance.data_move(self.output_hassign_cores_gm[aicore_id, 0],
                                    hassign_ub, 0, 1, self.code_num // 16, 0, 0)

    def dump_zero_last_loop(self, aicore_id, aicore_offset):

        sub_centroids_ub = self.tik_instance.Tensor("float16", (self.code_num, self.dim),
                                                    name="sub_centroids_ub", scope=tik.scope_ubuf)
        self.tik_instance.vec_dup(self.fp16_mask, sub_centroids_ub, 0, self.code_num * self.dim // self.fp16_mask, 8)
        hassign_ub = self.tik_instance.Tensor("int16", (self.code_num, ), name="hassign", scope=tik.scope_ubuf)
        self.tik_instance.vec_dup(self.code_num, hassign_ub, 0, 1, 8)


        self.tik_instance.data_move(self.output_sub_centroids_cores_gm[aicore_id, 0, 0],
                                    sub_centroids_ub, 0, 1, self.code_num * self.dim // 16, 0, 0)
        self.tik_instance.data_move(self.output_hassign_cores_gm[aicore_id, 0],
                                    hassign_ub, 0, 1, self.code_num // 16, 0, 0)

    def distance_compute_flat(self):
        """
        the compute process
        """
        self.cal_num_each_core()

        self.diag_matrix_ub = self.tik_instance.Tensor("int16", (self.code_num, self.code_num),
                                                       name="diag_matrix_ub", scope=tik.scope_ubuf)
        self.tik_instance.vec_dup(self.fp16_mask, self.diag_matrix_ub, 0,
                                  self.code_num * self.code_num // self.fp16_mask, 8)
        with self.tik_instance.for_range(0, self.code_num) as i:
            self.diag_matrix_ub[i, i].set_as(1)
        with self.tik_instance.for_range(0, 8, block_num=8) as block_index:
            # compute coarse code num and move offset every core\
            self.dump_zero_each_loop(block_index, block_index * self.query_each_core)
            self.dump_zero_last_loop(block_index, block_index * self.query_each_core)
            with self.tik_instance.if_scope(block_index < self.aicore_use - 1):
                self.distance_compute_each_loop(block_index, block_index * self.query_each_core)
            with self.tik_instance.if_scope(block_index == self.aicore_use - 1):
                self.distance_compute_last_loop(block_index, block_index * self.query_each_core)

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
        self.distance_compute_flat()

        self.tik_instance.BuildCCE(kernel_name=self.kernel_name,
                                   inputs=[
                                       self.input_queries_gm,
                                       self.input_actual_queries_num_gm,
                                       self.input_code_gm,
                                       self.input_code_norm_gm,
                                       self.input_dm_gm,
                                   ],
                                   outputs=[self.output_sub_centroids_cores_gm,
                                            self.output_hassign_cores_gm,
                                            self.output_hassign_queries_gm,
                                            self.output_flag_gm])

        return self.tik_instance


def distance_flat_l2_at(input_queries,
                        input_actual_queries_num,
                        input_code,
                        input_code_norm,
                        input_dm,
                        output_sub_centroids_cores,
                        output_hassign_cores,
                        output_hassign_queries,
                        output_flag,
                        kernel_name="distance_flat_l2_at"):
    """
    calculating distance

    Parameters
    ----------
    input_queries : dict
        shape and dtype of query vector
    input_code : dict
        shape and dtype of coarse code
    input_queries_norm : dict
        shape and dtype of L2 distance of query vector
    input_code_norm : dict
        shape and dtype of precomputed L2 distance of coarse code
    output_distances : dict
        shape and dtype of distances, should be same dtype as input_queries
    output_flag : dict
        shape and dtype of flag, only the 1th and 17th is valid
    kernel_name : str
        kernel name, default value is "distance_flat_at"

    Returns
    -------
    None
    """
    distance_flat = DistanceFlatL2At(input_queries, input_actual_queries_num, input_code,
                                     input_code_norm, input_dm, output_sub_centroids_cores,
                                     output_hassign_cores, output_hassign_queries, output_flag, kernel_name)
    tik_instance = distance_flat.get_tik_instance()
    return tik_instance