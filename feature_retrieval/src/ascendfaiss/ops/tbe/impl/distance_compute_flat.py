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


class DistanceComputeFlat():
    def __init__(self,
                 input_queries,
                 input_coarse_centroids,
                 input_precomputed,
                 output_distances,
                 output_flag,
                 kernel_name="distance_compute_flat"):
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
        set_soc_info()
        self.tik_instance = tik.Tik()

        self.aicore_use = 2
        self.coarse_centroids_num_each_core = self.coarse_centroids_num // self.aicore_use // 16 * 16
        self.coarse_centroids_num_last_core = self.coarse_centroids_num - (self.aicore_use - 1) * \
            self.coarse_centroids_num_each_core

        self.coarse_centroids_num_each_loop = min((65536 // self.dim) // 16 * 16, 736)
        self.queries_num_each_loop = (122880 - 6 * self.coarse_centroids_num_each_loop) // \
                                     (10 * self.coarse_centroids_num_each_loop) // 16 * 16

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

    def distance_compute_flat_each_loop(self, aicore_move_offset,
                                        aicore_coarse_centroids_num, move_offset,
                                        move_num):
        queries_align = (move_num + 15) // 16 * 16
        add_scalar_list = [self.tik_instance.Scalar(dtype="float32") for i in range(move_num)]
        queries_l1 = self.tik_instance.Tensor(
            "float16", (self.dim // 16, queries_align, 16),
            name="queries_l1",
            scope=tik.scope_cbuf)

        with self.tik_instance.new_stmt_scope():
            for i in range(move_num):
                self.tik_instance.data_move(
                    queries_l1[0, i, 0], self.input_queries_gm[move_offset + i,
                                                               0], 0,
                    self.dim // 16, 1, 0, queries_align - 1)

            queries_square_l0c = self.tik_instance.Tensor(
                "float32", (queries_align // 16, queries_align, 16),
                name="queries_square_l0c",
                scope=tik.scope_cbuf_out)
            self.tik_instance.matmul(queries_square_l0c, queries_l1,
                                     queries_l1, queries_align, self.dim,
                                     queries_align)
            queries_square_ub = self.tik_instance.Tensor(
                "float32", (queries_align // 16, queries_align, 16),
                name="queries_square_ub",
                scope=tik.scope_ubuf)
            self.tik_instance.data_move(queries_square_ub, queries_square_l0c,
                                        0, 1,
                                        queries_align * queries_align // 256,
                                        0, 0)

            queries_l2_ub = self.tik_instance.Tensor("float32",
                                                     (queries_align,),
                                                     name="queries_l2_ub",
                                                     scope=tik.scope_ubuf)
            for i in range(move_num):
                mask = 2 ** (i % 16)
                self.tik_instance.vcadd([0, mask], queries_l2_ub[i],
                                        queries_square_ub[i // 16, i,
                                                          0], 1, 1, 1, 8)
            for i in range(move_num):
                add_scalar_list[i].set_as(queries_l2_ub[i])

        # compute xy using cube
        coarse_centroids_loop_time = aicore_coarse_centroids_num // self.coarse_centroids_num_each_loop
        if coarse_centroids_loop_time > 0:
            thread_num_need = 2 if coarse_centroids_loop_time > 1 else 1
            with self.tik_instance.for_range(
                    0, coarse_centroids_loop_time,
                    thread_num=thread_num_need) as loop_coarse_centroids:
                self.cube_compute_each_loop(
                    queries_l1, add_scalar_list, aicore_move_offset,
                    loop_coarse_centroids *
                    self.coarse_centroids_num_each_loop,
                    self.coarse_centroids_num_each_loop, move_offset, move_num)

        coarse_centroids_last_num = aicore_coarse_centroids_num % self.coarse_centroids_num_each_loop
        if coarse_centroids_last_num > 0:
            self.cube_compute_each_loop(
                queries_l1, add_scalar_list, aicore_move_offset,
                coarse_centroids_loop_time *
                self.coarse_centroids_num_each_loop, coarse_centroids_last_num,
                move_offset, move_num)

    def cube_compute_each_loop(self, queries_l1, add_scalar_list,
                               aicore_move_offset,
                               coarse_centroids_move_offset,
                               coarse_centroids_move_num, queries_move_offset,
                               queries_move_num):
        queries_align = (queries_move_num + 15) // 16 * 16
        coarse_centroids_l1 = self.tik_instance.Tensor(
            "float16",
            (self.dim // 16, coarse_centroids_move_num, 16),
            name="coarse_centroids_l1",
            scope=tik.scope_cbuf)
        for i in range(self.dim // 16):
            self.tik_instance.data_move(
                coarse_centroids_l1[i, 0, 0], self.input_coarse_centroids_gm[
                    (aicore_move_offset + coarse_centroids_move_offset) // 16,
                    i, 0, 0], 0, coarse_centroids_move_num // 16, 16, self.dim - 16, 0)

        inner_product_l0c = self.tik_instance.Tensor(
            "float32",
            (coarse_centroids_move_num // 16, queries_align, 16),
            name="inner_product_l0c",
            scope=tik.scope_cbuf_out)
        self.tik_instance.matmul(inner_product_l0c, queries_l1,
                                 coarse_centroids_l1, queries_align, self.dim,
                                 coarse_centroids_move_num)

        # mov coarse_centroids l2 from out to UB
        coarse_centroids_l2_ub_fp16 = self.tik_instance.Tensor(
            "float16", (coarse_centroids_move_num,),
            name="coarse_centroids_l2_ub_fp16",
            scope=tik.scope_ubuf)
        self.tik_instance.data_move(
            coarse_centroids_l2_ub_fp16,
            self.input_precomputed_gm[aicore_move_offset +
                                      coarse_centroids_move_offset], 0, 1,
            coarse_centroids_move_num // 16, 0, 0)

        # coarse_centroids_l2 do conv from fp16 to fp32
        coarse_centroids_l2_ub = self.tik_instance.Tensor(
            "float32", (coarse_centroids_move_num,),
            name="coarse_centroids_l2_ub",
            scope=tik.scope_ubuf)
        vconv_repeat_time = coarse_centroids_move_num // self.fp32_vector_mask_max
        vconv_offset = 0
        if vconv_repeat_time > 0:
            self.tik_instance.vconv(self.fp32_vector_mask_max, "none",
                                    coarse_centroids_l2_ub[vconv_offset],
                                    coarse_centroids_l2_ub_fp16[vconv_offset],
                                    vconv_repeat_time, 1, 1, 8, 4)
            vconv_offset += vconv_repeat_time * self.fp32_vector_mask_max

        vconv_last_num = coarse_centroids_move_num % self.fp32_vector_mask_max
        if vconv_last_num > 0:
            self.tik_instance.vconv(vconv_last_num, "none",
                                    coarse_centroids_l2_ub[vconv_offset],
                                    coarse_centroids_l2_ub_fp16[vconv_offset],
                                    1, 1, 1, 8, 4)

        add_ub = self.tik_instance.Tensor(
            "float32", (queries_move_num, coarse_centroids_move_num),
            name="add_ub",
            scope=tik.scope_ubuf)
        for loop_queries in range(0, queries_move_num):
            repeat = coarse_centroids_move_num // self.fp32_vector_mask_max
            offest = 0
            if repeat > 0:
                self.tik_instance.vadds(self.fp32_vector_mask_max,
                                        add_ub[loop_queries, offest],
                                        coarse_centroids_l2_ub[offest],
                                        add_scalar_list[loop_queries], repeat,
                                        1, 1, 8, 8)
                offest += repeat * self.fp32_vector_mask_max

            remain = coarse_centroids_move_num % self.fp32_vector_mask_max
            if remain > 0:
                self.tik_instance.vadds(remain, add_ub[loop_queries, offest],
                                        coarse_centroids_l2_ub[offest],
                                        add_scalar_list[loop_queries], 1, 1, 1,
                                        8, 8)

        # mov xy from L0-C to UB
        inner_product_ub = self.tik_instance.Tensor(
            "float32",
            (coarse_centroids_move_num // 16, queries_align, 16),
            name="inner_product_ub",
            scope=tik.scope_ubuf)
        self.tik_instance.data_move(
            inner_product_ub, inner_product_l0c, 0, 1,
            coarse_centroids_move_num * queries_align // 256, 0, 0)

        with self.tik_instance.for_range(0, queries_move_num) as loop_queries:
            self.tik_instance.vaxpy(16,
                                    add_ub[loop_queries, 0],
                                    inner_product_ub[0, loop_queries, 0],
                                    self.coeff,
                                    coarse_centroids_move_num // 16,
                                    1,
                                    1,
                                    2,
                                    queries_align * 2)

        dst_ub = self.tik_instance.Tensor(
            "float16", (queries_move_num, coarse_centroids_move_num),
            name="dst_ub",
            scope=tik.scope_ubuf)
        vconv_repeat_time = queries_move_num * coarse_centroids_move_num // self.fp32_vector_mask_max
        vconv_offset = 0
        if vconv_repeat_time > 0:
            self.tik_instance.vconv(self.fp32_vector_mask_max, "none",
                                    dst_ub[vconv_offset],
                                    add_ub[vconv_offset],
                                    vconv_repeat_time, 1, 1, 4, 8)
            vconv_offset += vconv_repeat_time * self.fp32_vector_mask_max

        vconv_last_num = queries_move_num * coarse_centroids_move_num % self.fp32_vector_mask_max
        if vconv_last_num > 0:
            self.tik_instance.vconv(vconv_last_num, "none",
                                    dst_ub[vconv_offset],
                                    add_ub[vconv_offset], 1, 1, 1, 4,
                                    8)

        self.tik_instance.data_move(
            self.output_distances_gm[queries_move_offset, aicore_move_offset + coarse_centroids_move_offset],
            dst_ub, 0, queries_move_num, coarse_centroids_move_num // 16,
            0, (self.coarse_centroids_num - coarse_centroids_move_num) // 16)

    def distance_compute_flat(self):
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
                    self.distance_compute_flat_each_loop(
                        block_index * self.coarse_centroids_num_each_core,
                        aicore_coarse_centroids_num,
                        loop_queries * self.queries_num_each_loop,
                        self.queries_num_each_loop)

            queries_last_num = self.queries_num % self.queries_num_each_loop
            if queries_last_num > 0:
                self.distance_compute_flat_each_loop(
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
        self.distance_compute_flat()

        self.tik_instance.BuildCCE(kernel_name=self.kernel_name,
                                   inputs=[
                                       self.input_queries_gm,
                                       self.input_coarse_centroids_gm,
                                       self.input_precomputed_gm
                                   ],
                                   outputs=[self.output_distances_gm, self.output_flag_gm])

        return self.tik_instance


def distance_compute_flat(input_queries,
                          input_coarse_centroids,
                          input_precomputed,
                          output_distances,
                          output_flag,
                          kernel_name="distance_compute_flat"):
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
    dist_cmp_flat = DistanceComputeFlat(input_queries, input_coarse_centroids,
                                                input_precomputed, output_distances,
                                                output_flag, kernel_name)
    tik_instance = dist_cmp_flat.get_tik_instance()
    return tik_instance
