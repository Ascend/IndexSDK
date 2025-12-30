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

class DistanceComputeQC():
    def __init__(self,
                 input_queries,
                 input_coarse_centroids,
                 input_precomputed,
                 output_distances,
                 output_flag,
                 kernel_name="distance_compute_qc"):
        self.shape_queries = input_queries.get("shape")
        self.dtype_queries = input_queries.get("dtype")
        self.shape_coarse_centroids = input_coarse_centroids.get("shape")
        self.dtype_coarse_centroids = input_coarse_centroids.get("dtype")
        self.shape_precomputed = input_precomputed.get("shape")
        self.dtype_precomputed = input_precomputed.get("dtype")
        self.shape_distances = output_distances.get("shape")
        self.dtype_distances = output_distances.get("dtype")
        self.shape_flag = output_flag.get("shape")
        self.dtype_flag = output_flag.get("dtype")
        self.kernel_name = kernel_name

        # compute parameter
        self.queries_num, self.dim = self.shape_queries
        self.coarse_centroids_num, = self.shape_precomputed
        self.dim2 = self.shape_coarse_centroids[0] * 16 // self.coarse_centroids_num

        # check parameter
        self.check_parameter()

        # set vector fp32 mask and fp16 mask
        self.fp32_vector_mask_max = 64
        self.fp16_vector_mask_max = 128

        # set tik instance
        self.set_tik_instance()

    def check_parameter(self):
        if self.dim % 16 != 0:
            raise RuntimeError("feature dim must be a multiple of 16")
        if self.coarse_centroids_num % 16 != 0:
            raise RuntimeError("coarse_centroids num must be a multiple of 16")

    def set_tik_instance(self):
        """
        set tik_instance
        """
        tik_dprofile = tik.Dprofile("v200", "aic")
        self.tik_instance = tik.Tik(tik_dprofile)

        self.aicore_use = 8
        self.coarse_centroids_num_each_core = self.coarse_centroids_num // self.aicore_use // 16 * 16
        self.coarse_centroids_num_last_core = self.coarse_centroids_num - (self.aicore_use - 1) * \
                                              self.coarse_centroids_num_each_core

        self.coarse_centroids_num_each_loop = min((65536 // self.dim) // 16 * 16, 16)
        self.queries_num_each_loop = min((122880 - 6 * self.coarse_centroids_num_each_loop) // \
                                     (10 * self.coarse_centroids_num_each_loop) // 16 * 16, 64)
        self.set_src_dst_tensor()

    def set_src_dst_tensor(self):
        """
        set input and output tensor
        """
        self.coeff = self.tik_instance.Scalar("float32",
                                              name="coeff",
                                              init_value=-2)

        # creat input tensor: input_queries_gm, input_coarse_centroids_gm
        # and input_precomputed_gm
        # and output tensor: output_distances_gm, output_flag_gm in global buffer
        self.input_queries_gm = self.tik_instance.Tensor(
            self.dtype_queries,
            self.shape_queries,
            name="input_queries_gm",
            scope=tik.scope_gm)
        self.input_coarse_centroids_gm = self.tik_instance.Tensor(
            self.dtype_coarse_centroids,
            self.shape_coarse_centroids,
            name="input_coarse_centroids_gm",
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
        self.output_flag_gm = self.tik_instance.Tensor(
            self.dtype_flag,
            self.shape_flag,
            name="output_flag_gm",
            scope=tik.scope_gm)

    def distance_compute_flat_each_loop_v2(self, aicore_move_offset,
                                           aicore_coarse_centroids_num, move_offset,
                                           move_num):
        queries_align = (move_num + 15) // 16 * 16
        queries_l1 = self.tik_instance.Tensor(
            "float16", (self.dim // 16, queries_align, 16),
            name="queries_l1",
            scope=tik.scope_cbuf)
        for i in range(move_num):
            self.tik_instance.data_move(
                queries_l1[0, i, 0], self.input_queries_gm[move_offset + i,
                                                           0], 0,
                self.dim // 16, 1, 0, queries_align - 1)

        # compute xy using cube
        coarse_centroids_loop_time = aicore_coarse_centroids_num // self.coarse_centroids_num_each_loop

        if coarse_centroids_loop_time > 0:
            thread_num_need = 1
            with self.tik_instance.for_range(
                    0, coarse_centroids_loop_time,
                    thread_num=thread_num_need) as loop_coarse_centroids:
                self.cube_compute_each_loop(
                    queries_l1, aicore_move_offset,
                    loop_coarse_centroids *
                    self.coarse_centroids_num_each_loop,
                    self.coarse_centroids_num_each_loop, move_offset, move_num)

        coarse_centroids_last_num = aicore_coarse_centroids_num % self.coarse_centroids_num_each_loop
        if coarse_centroids_last_num > 0:
            self.cube_compute_each_loop(
                queries_l1, aicore_move_offset,
                coarse_centroids_loop_time *
                self.coarse_centroids_num_each_loop, coarse_centroids_last_num,
                move_offset, move_num)

    def cube_compute_each_loop(self, queries_l1,
                                  aicore_move_offset,
                                  coarse_centroids_move_offset,
                                  coarse_centroids_move_num, queries_move_offset,
                                  queries_move_num):
        queries_align = (queries_move_num + 15) // 16 * 16

        qc_l2_ub = self.tik_instance.Tensor("float16",
                                            (queries_align, coarse_centroids_move_num),
                                            name="qc_l2_ub", scope=tik.scope_ubuf)

        with self.tik_instance.new_stmt_scope():
            loop_times = self.dim2 // 16
            dim2_each_loop = self.dim2 // loop_times

            inner_product_f16_ub = self.tik_instance.Tensor(
                "float16",
                (dim2_each_loop * coarse_centroids_move_num // 16, queries_align, 16),
                name="inner_product_reshape_fp16_ub",
                scope=tik.scope_ubuf)
            inner_product_l0c = self.tik_instance.Tensor(
                "float32",
                (coarse_centroids_move_num * dim2_each_loop // 16, queries_align, 16),
                name="inner_product_l0c",
                scope=tik.scope_cbuf_out)
            coarse_centroids_l1 = self.tik_instance.Tensor(
                "float16",
                (self.dim // 16, coarse_centroids_move_num * dim2_each_loop, 16),
                name="coarse_centroids_l1",
                scope=tik.scope_cbuf)

            inner_product_ub = self.tik_instance.Tensor("float32",
                                                        (coarse_centroids_move_num * dim2_each_loop // 16,
                                                         queries_align, 16),
                                                        name="inner_product_ub", scope=tik.scope_ubuf)

            with self.tik_instance.for_range(0, loop_times, thread_num=1) as loop_id:
                qc_l2_temp_ub = self.tik_instance.Tensor("float16",
                                                         (1, coarse_centroids_move_num),
                                                         name="qc_l2_temp_ub", scope=tik.scope_ubuf)

                for i in range(coarse_centroids_move_num):
                    self.tik_instance.data_move(
                        coarse_centroids_l1[0, i * dim2_each_loop, 0], self.input_coarse_centroids_gm[
                            (aicore_move_offset + coarse_centroids_move_offset + i) 
                                * self.dim2 // 16 + loop_id * dim2_each_loop // 16,
                            0, 0, 0], 0, self.dim // 16, 16, 0,
                        (coarse_centroids_move_num * dim2_each_loop - dim2_each_loop))
                self.tik_instance.matmul(
                    inner_product_l0c, queries_l1,
                    coarse_centroids_l1, queries_align, self.dim,
                    coarse_centroids_move_num * dim2_each_loop)

                self.tik_instance.data_move(inner_product_ub,
                                            inner_product_l0c,
                                            0, 1, coarse_centroids_move_num * dim2_each_loop * queries_align // 256, 0,
                                            0)
                vconv_loop = queries_align * coarse_centroids_move_num * dim2_each_loop // (
                        self.fp32_vector_mask_max * 255)
                vconv_offset = 0
                if vconv_loop > 0:
                    for conv_index in range(vconv_loop):
                        vconv_offset = conv_index * self.fp32_vector_mask_max * 255
                        self.tik_instance.vconv(self.fp32_vector_mask_max, "none",
                                                inner_product_f16_ub[vconv_offset],
                                                inner_product_ub[vconv_offset],
                                                255, 1, 1, 4, 8)
                    vconv_offset += self.fp32_vector_mask_max * 255
                vconv_repeat_time = (queries_align * coarse_centroids_move_num * dim2_each_loop) % (
                        self.fp32_vector_mask_max * 255) // self.fp32_vector_mask_max
                if vconv_repeat_time > 0:
                    self.tik_instance.vconv(self.fp32_vector_mask_max, "none",
                                            inner_product_f16_ub[vconv_offset],
                                            inner_product_ub[vconv_offset],
                                            vconv_repeat_time, 1, 1, 4, 8)

                inner_product_reshape_f16_ub = self.tik_instance.Tensor(
                    "float16",
                    (1, coarse_centroids_move_num * dim2_each_loop),
                    name="inner_product_reshape_ub",
                    scope=tik.scope_ubuf)

                with self.tik_instance.for_range(0, queries_move_num, thread_num=1) as loop_queries:

                    self.tik_instance.data_move(inner_product_reshape_f16_ub[0, 0],
                                                inner_product_f16_ub[0, loop_queries, 0],
                                                0, coarse_centroids_move_num * dim2_each_loop // 16, 1,
                                                queries_align * 1 - 1, 0)

                    repeat = coarse_centroids_move_num * dim2_each_loop // self.fp16_vector_mask_max
                    offset = 0
                    if repeat > 0:
                        self.tik_instance.vmul(self.fp16_vector_mask_max,
                                               inner_product_reshape_f16_ub[0, offset],
                                               inner_product_reshape_f16_ub[0, offset],
                                               inner_product_reshape_f16_ub[0, offset], repeat, 1, 1,
                                               1, 8, 8, 8)
                        offset += repeat * self.fp16_vector_mask_max

                    remain = (coarse_centroids_move_num * dim2_each_loop) % self.fp16_vector_mask_max
                    if remain > 0:
                        self.tik_instance.vmul(remain, inner_product_reshape_f16_ub[0, offset],
                                               inner_product_reshape_f16_ub[0, offset],
                                               inner_product_reshape_f16_ub[0, offset], 1, 1, 1, 1, 8,
                                               8, 8)

                    with self.tik_instance.if_scope(loop_id == 0):
                        self.tik_instance.vcadd(dim2_each_loop, qc_l2_ub[loop_queries, 0],
                                                inner_product_reshape_f16_ub[0, 0],
                                                coarse_centroids_move_num, 1, 1, dim2_each_loop // 16)
                    with self.tik_instance.else_scope():

                        self.tik_instance.vcadd(dim2_each_loop, qc_l2_temp_ub[0, 0],
                                                inner_product_reshape_f16_ub[0, 0],
                                                coarse_centroids_move_num, 1, 1, dim2_each_loop // 16)
                        repeat = coarse_centroids_move_num // self.fp16_vector_mask_max
                        offset = 0
                        if repeat > 0:
                            self.tik_instance.vadd(self.fp16_vector_mask_max, qc_l2_ub[loop_queries, offset],
                                                   qc_l2_ub[loop_queries, offset],
                                                   qc_l2_temp_ub[0, offset], repeat, 1, 1, 1, 8, 8, 8)
                            offset += repeat * self.fp16_vector_mask_max

                        remain = coarse_centroids_move_num % self.fp16_vector_mask_max
                        if remain > 0:
                            self.tik_instance.vadd(remain, qc_l2_ub[loop_queries, offset],
                                                   qc_l2_ub[loop_queries, offset],
                                                   qc_l2_temp_ub[0, offset], 1, 1, 1, 1, 8, 8, 8)

        self.tik_instance.data_move(
            self.output_distances_gm[queries_move_offset, aicore_move_offset + coarse_centroids_move_offset],
            qc_l2_ub, 0, queries_move_num, coarse_centroids_move_num // 16,
            0, (self.coarse_centroids_num - coarse_centroids_move_num) // 16)

    def distance_compute_flat_v2(self):
        """
        the compute process
        """
        with self.tik_instance.for_range(
                0, self.aicore_use, block_num=self.aicore_use) as block_index:
            # compute coarse centroids num and move offest every core
            with self.tik_instance.if_scope(
                    block_index != self.aicore_use - 1):
                aicore_coarse_centroids_num = \
                    self.coarse_centroids_num_each_core
            with self.tik_instance.else_scope():
                aicore_coarse_centroids_num = \
                    self.coarse_centroids_num_last_core

            queries_loop_time = self.queries_num // self.queries_num_each_loop
            if queries_loop_time > 0:
                with self.tik_instance.for_range(
                        0, queries_loop_time) as loop_queries:
                    self.distance_compute_flat_each_loop_v2(
                        block_index * self.coarse_centroids_num_each_core,
                        aicore_coarse_centroids_num,
                        loop_queries * self.queries_num_each_loop,
                        self.queries_num_each_loop)

            queries_last_num = self.queries_num % self.queries_num_each_loop
            if queries_last_num > 0:
                self.distance_compute_flat_each_loop_v2(
                    block_index * self.coarse_centroids_num_each_core,
                    aicore_coarse_centroids_num,
                    queries_loop_time * self.queries_num_each_loop,
                    queries_last_num)

            one = self.tik_instance.Scalar(dtype="uint16",
                                           name="one",
                                           init_value=1)
            flag_ub = self.tik_instance.Tensor("uint16",
                                               (16,),
                                               name="flag_ub",
                                               scope=tik.scope_ubuf)

            flag_ub[0].set_as(one)
            self.tik_instance.data_move(self.output_flag_gm[block_index * 16],
                                        flag_ub, 0, 1, 1, 0, 0)

    def get_tik_instance(self):
        """
        obtain tik instance
        """
        self.distance_compute_flat_v2()

        self.tik_instance.BuildCCE(kernel_name=self.kernel_name,
                                   inputs=[
                                       self.input_queries_gm,
                                       self.input_coarse_centroids_gm,
                                       self.input_precomputed_gm
                                   ],
                                   outputs=[self.output_distances_gm, self.output_flag_gm])

        return self.tik_instance
    
    def _add(self, dst, src0, src1, compute_num, mask):
        # process 256B data per repeat for vadd
        repeat = compute_num // mask
        offset = 0
        if repeat > 0:
            self.tik_instance.vadd(mask, dst[offset], src0[offset],
                                   src1[offset], repeat, 1, 1, 1, 8, 8, 8)
            offset += repeat * mask

        remain = compute_num % mask
        if remain > 0:
            self.tik_instance.vadd(remain, dst[offset], src0[offset],
                                   src1[offset], 1, 1, 1, 1, 8, 8, 8)


def distance_compute_qc(input_queries,
                        input_coarse_centroids,
                        input_precomputed,
                        output_distances,
                        output_flag,
                        kernel_name="distance_compute_qc"):
    """
    calculating distance

    Parameters
    ----------
    input_queries : dict
        shape and dtype of query vector
    input_coarse_centroids : dict
        shape and dtype of coarse centroids
    input_precomputed : dict
        shape and dtype of precomputed L2 distance of coarse centroids
    output_distances : dict
        shape and dtype of distances, should be same dtype as input_queries
    output_flag : dict
        shape and dtype of flag, only the 1th and 17th is valid
    kernel_name : str
        kernel name, default value is "distance_compute_flat"

    Returns
    -------
    None
    """
    dist_cmp_flat = DistanceComputeQC(input_queries, input_coarse_centroids,
                                      input_precomputed, output_distances,
                                      output_flag, kernel_name)
    tik_instance = dist_cmp_flat.get_tik_instance()
    return tik_instance