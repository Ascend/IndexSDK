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


class DistanceTableBuild:
    def __init__(self,
                 input_queries,
                 input_pq_centroids,
                 input_sequence_number,
                 input_coarse_centroids,
                 output_distance_table,
                 output_flag,
                 kernel_name="distance_table_build"):
        self.shape_queries = input_queries.get("shape")
        self.dtype_queries = input_queries.get("dtype")
        self.shape_pq_centroids = input_pq_centroids.get("shape")
        self.dtype_pq_centroids = input_pq_centroids.get("dtype")
        self.shape_sequence_number = input_sequence_number.get("shape")
        self.dtype_sequence_number = input_sequence_number.get("dtype")
        self.shape_coarse_centroids = input_coarse_centroids.get("shape")
        self.dtype_coarse_centroids = input_coarse_centroids.get("dtype")
        self.shape_distance_table = output_distance_table.get("shape")
        self.dtype_distance_table = output_distance_table.get("dtype")
        self.shape_flag = output_flag.get("shape")
        self.dtype_flag = output_flag.get("dtype").lower()
        self.kernel_name = kernel_name

        # compute parameter
        self.queries_num, self.dim = self.shape_queries
        self.sub_quantizers_num, pq_dim = self.shape_pq_centroids
        self.sub_dim = self.dim // self.sub_quantizers_num
        self.pq_centroids_num = pq_dim // self.sub_dim
        _, self.closest_num = self.shape_sequence_number

        # check parameter
        if self.closest_num % 8 != 0:
            raise RuntimeError("the closest num of IVF centroids to the query vector must be a multiple of 8")
        if self.dim % 16 != 0 or self.pq_centroids_num % 16 != 0:
            raise RuntimeError("feature dim and pq centroids num must be a multiple of 16")
        if self.sub_dim > 128 and self.sub_dim % 128 != 0:
            raise RuntimeError("if sub_dim  greater than 128, it must be a multiple of 128")
        if self.sub_dim != 4 and self.sub_dim != 8 and self.sub_dim % 16 != 0:
            raise RuntimeError("sub_dim must be 4 or 8 or a multiple of 16")
        if self.shape_flag[0] != 16:
            raise RuntimeError("output flag num must be 16")

        # set max vector mask
        self.fp16_mask = 128

        self.aicore_use = 2

        # The target machine is defined by the Dprofile function,
        # and the TIK DSL container is constructed by the Tik function.
        set_soc_info()
        self.tik_instance = tik.Tik()

        self.zero = self.tik_instance.Scalar('float16')
        self.zero.set_as(0)

        # creat input tensor: input_queries_gm, input_pq_centroids_gm,
        # input_sequence_number_gm and input_coarse_centroids_gm
        # output tensor: output_distance_table_gm in global buffer
        self.input_queries_gm = self.tik_instance.Tensor(self.dtype_queries,
                                                         self.shape_queries,
                                                         name="input_queries_gm", scope=tik.scope_gm)
        self.input_pq_centroids_gm = self.tik_instance.Tensor(self.dtype_pq_centroids,
                                                              self.shape_pq_centroids,
                                                              name="input_pq_centroids_gm", scope=tik.scope_gm)
        self.input_sequence_number_gm = self.tik_instance.Tensor(self.dtype_sequence_number,
                                                                 self.shape_sequence_number,
                                                                 name="input_sequence_number_gm", scope=tik.scope_gm)
        self.input_coarse_centroids_gm = self.tik_instance.Tensor(self.dtype_coarse_centroids,
                                                                  self.shape_coarse_centroids,
                                                                  name="input_coarse_centroids_gm", scope=tik.scope_gm)
        self.output_distance_table_gm = self.tik_instance.Tensor(self.dtype_distance_table,
                                                                 self.shape_distance_table,
                                                                 name="output_distance_table_gm", scope=tik.scope_gm)
        self.output_flag_gm = self.tik_instance.Tensor(self.dtype_flag,
                                                       self.shape_flag,
                                                       name="output_flag_gm", scope=tik.scope_gm)

    # compute vadds mask of vector count
    @staticmethod
    def compute_mask(num, mode, tag):
        if num >= 64:
            num -= 64

        proj = {
            0: {0: 15, 1: 240, 2: 3840, 3: 61440},
            1: {0: 255, 1: 65280}
        }

        base = proj.get(mode, {}).get(tag)
        if base is None:
            raise Exception(f"No defined combination of mode: {mode} and tag: {tag}")

        mask = 0
        for i in range(num // 16):
            mask += base << (i * 16)
        return mask

    # compute distance table build which sub dim is 4
    def distance_table_build_compute_subdim_4(self, loop_queries, sequence_ub, aicore_offset, aicore_sequence_num):
        queries_pad_ub = self.tik_instance.Tensor("float16",
                                                  (self.dim * 4,),
                                                  name="queries_pad_ub", scope=tik.scope_ubuf)
        dup_repeats = (self.dim * 4) // self.fp16_mask
        dup_offset = 0
        if dup_repeats > 0:
            self.tik_instance.vector_dup(self.fp16_mask,
                                         queries_pad_ub[0], self.zero,
                                         dup_repeats, 1, 8)
            dup_offset += dup_repeats * self.fp16_mask

        dup_last_num = (self.dim * 4) % self.fp16_mask
        if dup_last_num > 0:
            self.tik_instance.vector_dup(dup_last_num,
                                         queries_pad_ub[dup_offset], self.zero,
                                         1, 1, 8)

        # move queries from out to UB
        queries_ub = self.tik_instance.Tensor("float16",
                                              (self.dim,),
                                              name="queries_ub", scope=tik.scope_ubuf)
        self.tik_instance.data_move(queries_ub[0],
                                    self.input_queries_gm[loop_queries, 0], 0,
                                    1, self.dim // 16, 0, 0)

        # compute distance per list
        thread_num_need = 1
        with self.tik_instance.if_scope(aicore_sequence_num > 1):
            thread_num_need = 2
        with self.tik_instance.for_range(0, aicore_sequence_num, thread_num=thread_num_need) as loop_sequence:
            coarse_centroids_ub = self.tik_instance.Tensor("float16",
                                                           (self.dim,),
                                                           name="coarse_centroids_ub", scope=tik.scope_ubuf)
            residual_ub = self.tik_instance.Tensor("float16",
                                                   (self.dim,),
                                                   name="residual_ub", scope=tik.scope_ubuf)
            sequence_id = self.tik_instance.Scalar('int32')
            sequence_id.set_as(sequence_ub[loop_queries, aicore_offset + loop_sequence])

            # x-yc
            self.tik_instance.data_move(coarse_centroids_ub[0],
                                        self.input_coarse_centroids_gm[sequence_id, 0],
                                        0, 1, self.dim // 16, 0, 0)

            vsub_repeats = self.dim // self.fp16_mask
            vsub_offset = 0
            if vsub_repeats > 0:
                self.tik_instance.vsub(self.fp16_mask, residual_ub[0],
                                       queries_ub[0], coarse_centroids_ub[0],
                                       vsub_repeats, 1, 1, 1, 8, 8, 8)
                vsub_offset += vsub_repeats * self.fp16_mask

            vsub_last_num = self.dim % self.fp16_mask
            if vsub_last_num > 0:
                self.tik_instance.vsub(vsub_last_num, residual_ub[vsub_offset],
                                       queries_ub[vsub_offset], coarse_centroids_ub[vsub_offset],
                                       1, 1, 1, 1, 8, 8, 8)

            # expand x-yc per sub num
            vadds_repeats = self.dim // self.fp16_mask
            vadds_offset = 0
            if vadds_repeats > 0:
                self.tik_instance.vadds([4222189076152335, 4222189076152335],
                                        queries_pad_ub[0], residual_ub[0],
                                        0, vadds_repeats, 4, 1, 32, 8)
                self.tik_instance.vadds([67555025218437360, 67555025218437360],
                                        queries_pad_ub[16], residual_ub[0],
                                        0, vadds_repeats, 4, 1, 32, 8)
                self.tik_instance.vadds([1080880403494997760, 1080880403494997760],
                                        queries_pad_ub[32], residual_ub[0],
                                        0, vadds_repeats, 4, 1, 32, 8)
                self.tik_instance.vadds([17294086455919964160, 17294086455919964160],
                                        queries_pad_ub[48], residual_ub[0],
                                        0, vadds_repeats, 4, 1, 32, 8)
                vadds_offset += vadds_repeats * self.fp16_mask

            vadds_last_num = self.dim % self.fp16_mask
            if vadds_last_num > 0:
                mask0 = self.compute_mask(vadds_last_num, 0, 0)
                mask1 = self.compute_mask(vadds_last_num, 0, 1)
                mask2 = self.compute_mask(vadds_last_num, 0, 2)
                mask3 = self.compute_mask(vadds_last_num, 0, 3)
                if vadds_last_num < 64:
                    vadds_mask0 = [0, mask0]
                    vadds_mask1 = [0, mask1]
                    vadds_mask2 = [0, mask2]
                    vadds_mask3 = [0, mask3]
                else:
                    vadds_mask0 = [mask0, 4222189076152335]
                    vadds_mask1 = [mask1, 67555025218437360]
                    vadds_mask2 = [mask2, 1080880403494997760]
                    vadds_mask3 = [mask3, 17294086455919964160]

                self.tik_instance.vadds(vadds_mask0,
                                        queries_pad_ub[vadds_offset * 4],
                                        residual_ub[vadds_offset], 0, 1, 4, 1, 32, 8)
                self.tik_instance.vadds(vadds_mask1,
                                        queries_pad_ub[vadds_offset * 4 + 16],
                                        residual_ub[vadds_offset], 0, 1, 4, 1, 32, 8)
                self.tik_instance.vadds(vadds_mask2,
                                        queries_pad_ub[vadds_offset * 4 + 32],
                                        residual_ub[vadds_offset], 0, 1, 4, 1, 32, 8)
                self.tik_instance.vadds(vadds_mask3,
                                        queries_pad_ub[vadds_offset * 4 + 48],
                                        residual_ub[vadds_offset], 0, 1, 4, 1, 32, 8)

            mul_pad_ub = self.tik_instance.Tensor("float16",
                                                  (self.pq_centroids_num, 16),
                                                  name="mul_pad_ub", scope=tik.scope_ubuf)
            dup_repeats = (self.pq_centroids_num * 16) // self.fp16_mask
            if dup_repeats > 0:
                self.tik_instance.vector_dup(self.fp16_mask,
                                             mul_pad_ub[0], self.zero,
                                             dup_repeats, 1, 8)

            # compute distance per quantizers
            with self.tik_instance.for_range(0, self.sub_quantizers_num, thread_num=2) as loop_quantizers:
                data1 = self.tik_instance.Scalar(dtype="float16")
                data2 = self.tik_instance.Scalar(dtype="float16")
                data3 = self.tik_instance.Scalar(dtype="float16")
                data4 = self.tik_instance.Scalar(dtype="float16")

                pq_centroids_ub = self.tik_instance.Tensor("float16",
                                                           (self.pq_centroids_num, self.sub_dim),
                                                           name="pq_centroids_ub", scope=tik.scope_ubuf)
                self.tik_instance.data_move(pq_centroids_ub[0],
                                            self.input_pq_centroids_gm[loop_quantizers, 0],
                                            0, 1, self.pq_centroids_num * self.sub_dim // 16, 0, 0)

                sub_residual_ub = self.tik_instance.Tensor("float16",
                                                           (16,),
                                                           name="sub_residual_ub", scope=tik.scope_ubuf)
                self.tik_instance.data_move(sub_residual_ub[0],
                                            queries_pad_ub[loop_quantizers * 16],
                                            0, 1, 1, 0, 0)

                # copy sub residual per sub num
                with self.tik_instance.if_scope(loop_quantizers % 4 == 0):
                    data1.set_as(sub_residual_ub[0])
                    data2.set_as(sub_residual_ub[1])
                    data3.set_as(sub_residual_ub[2])
                    data4.set_as(sub_residual_ub[3])
                    sub_residual_ub[4].set_as(data1)
                    sub_residual_ub[5].set_as(data2)
                    sub_residual_ub[6].set_as(data3)
                    sub_residual_ub[7].set_as(data4)
                    sub_residual_ub[8].set_as(data1)
                    sub_residual_ub[9].set_as(data2)
                    sub_residual_ub[10].set_as(data3)
                    sub_residual_ub[11].set_as(data4)
                    sub_residual_ub[12].set_as(data1)
                    sub_residual_ub[13].set_as(data2)
                    sub_residual_ub[14].set_as(data3)
                    sub_residual_ub[15].set_as(data4)
                with self.tik_instance.if_scope(loop_quantizers % 4 == 1):
                    data1.set_as(sub_residual_ub[4])
                    data2.set_as(sub_residual_ub[5])
                    data3.set_as(sub_residual_ub[6])
                    data4.set_as(sub_residual_ub[7])
                    sub_residual_ub[0].set_as(data1)
                    sub_residual_ub[1].set_as(data2)
                    sub_residual_ub[2].set_as(data3)
                    sub_residual_ub[3].set_as(data4)
                    sub_residual_ub[8].set_as(data1)
                    sub_residual_ub[9].set_as(data2)
                    sub_residual_ub[10].set_as(data3)
                    sub_residual_ub[11].set_as(data4)
                    sub_residual_ub[12].set_as(data1)
                    sub_residual_ub[13].set_as(data2)
                    sub_residual_ub[14].set_as(data3)
                    sub_residual_ub[15].set_as(data4)
                with self.tik_instance.if_scope(loop_quantizers % 4 == 2):
                    data1.set_as(sub_residual_ub[8])
                    data2.set_as(sub_residual_ub[9])
                    data3.set_as(sub_residual_ub[10])
                    data4.set_as(sub_residual_ub[11])
                    sub_residual_ub[0].set_as(data1)
                    sub_residual_ub[1].set_as(data2)
                    sub_residual_ub[2].set_as(data3)
                    sub_residual_ub[3].set_as(data4)
                    sub_residual_ub[4].set_as(data1)
                    sub_residual_ub[5].set_as(data2)
                    sub_residual_ub[6].set_as(data3)
                    sub_residual_ub[7].set_as(data4)
                    sub_residual_ub[12].set_as(data1)
                    sub_residual_ub[13].set_as(data2)
                    sub_residual_ub[14].set_as(data3)
                    sub_residual_ub[15].set_as(data4)
                with self.tik_instance.else_scope():
                    data1.set_as(sub_residual_ub[12])
                    data2.set_as(sub_residual_ub[13])
                    data3.set_as(sub_residual_ub[14])
                    data4.set_as(sub_residual_ub[15])
                    sub_residual_ub[0].set_as(data1)
                    sub_residual_ub[1].set_as(data2)
                    sub_residual_ub[2].set_as(data3)
                    sub_residual_ub[3].set_as(data4)
                    sub_residual_ub[4].set_as(data1)
                    sub_residual_ub[5].set_as(data2)
                    sub_residual_ub[6].set_as(data3)
                    sub_residual_ub[7].set_as(data4)
                    sub_residual_ub[8].set_as(data1)
                    sub_residual_ub[9].set_as(data2)
                    sub_residual_ub[10].set_as(data3)
                    sub_residual_ub[11].set_as(data4)

                mul_ub = self.tik_instance.Tensor("float16",
                                                  (self.pq_centroids_num, self.sub_dim),
                                                  name="mul_ub", scope=tik.scope_ubuf)
                dst_ub = self.tik_instance.Tensor("float16",
                                                  (self.pq_centroids_num,),
                                                  name="dst_ub", scope=tik.scope_ubuf)

                # sub residual do sub with pq centroids
                vsub_loop = (self.pq_centroids_num * self.sub_dim) // (16 * 255)
                vsub_offset = self.tik_instance.Scalar(dtype="int32")
                vsub_offset.set_as(0)
                if vsub_loop > 0:
                    with self.tik_instance.for_range(0, vsub_loop):
                        self.tik_instance.vsub(16, mul_ub[vsub_offset // self.sub_dim, vsub_offset % self.sub_dim],
                                               sub_residual_ub[0],
                                               pq_centroids_ub[vsub_offset // self.sub_dim, vsub_offset % self.sub_dim],
                                               255, 1, 1, 1, 1, 0, 1)
                        vsub_offset.set_as(vsub_offset + 255 * 16)

                vsub_repeats = (self.pq_centroids_num * self.sub_dim) % (16 * 255) // 16
                if vsub_repeats > 0:
                    self.tik_instance.vsub(16, mul_ub[vsub_offset // self.sub_dim, vsub_offset % self.sub_dim],
                                           sub_residual_ub[0],
                                           pq_centroids_ub[vsub_offset // self.sub_dim, vsub_offset % self.sub_dim],
                                           vsub_repeats, 1, 1, 1, 1, 0, 1)

                # do mul
                vmul_repeats = (self.pq_centroids_num * self.sub_dim) // self.fp16_mask
                if vmul_repeats > 0:
                    self.tik_instance.vmul(self.fp16_mask, mul_ub[0],
                                           mul_ub[0], mul_ub[0],
                                           vmul_repeats, 1, 1, 1, 8, 8, 8)

                vadds_repeats = (self.pq_centroids_num * self.sub_dim) // self.fp16_mask
                if vadds_repeats > 0:
                    self.tik_instance.vadds([4222189076152335, 4222189076152335],
                                            mul_pad_ub[0, 0], mul_ub[0],
                                            0, vadds_repeats, 4, 1, 32, 8)
                    self.tik_instance.vadds([67555025218437360, 67555025218437360],
                                            mul_pad_ub[1, 0], mul_ub[0],
                                            0, vadds_repeats, 4, 1, 32, 8)
                    self.tik_instance.vadds([1080880403494997760, 1080880403494997760],
                                            mul_pad_ub[2, 0], mul_ub[0],
                                            0, vadds_repeats, 4, 1, 32, 8)
                    self.tik_instance.vadds([17294086455919964160, 17294086455919964160],
                                            mul_pad_ub[3, 0], mul_ub[0],
                                            0, vadds_repeats, 4, 1, 32, 8)

                # do vcgadd
                vcgadd_repeats = (self.pq_centroids_num * 16) // self.fp16_mask
                if vcgadd_repeats > 0:
                    self.tik_instance.vcgadd(self.fp16_mask,
                                             dst_ub[0], mul_pad_ub[0],
                                             vcgadd_repeats, 1, 1, 8)

                self.tik_instance.data_move(self.output_distance_table_gm[loop_queries,
                                                                          aicore_offset + loop_sequence,
                                                                          loop_quantizers, 0],
                                            dst_ub[0],
                                            0, 1, self.pq_centroids_num // 16, 0, 0)

    # compute distance table build which sub dim is 8
    def distance_table_build_compute_subdim_8(self, loop_queries, sequence_ub, aicore_offset, aicore_sequence_num):
        queries_pad_ub = self.tik_instance.Tensor("float16",
                                                  (self.dim * 2,),
                                                  name="queries_pad_ub", scope=tik.scope_ubuf)
        dup_repeats = (self.dim * 2) // self.fp16_mask
        dup_offset = 0
        if dup_repeats > 0:
            self.tik_instance.vector_dup(self.fp16_mask,
                                         queries_pad_ub[0], self.zero,
                                         dup_repeats, 1, 8)
            dup_offset += dup_repeats * self.fp16_mask

        dup_last_num = (self.dim * 2) % self.fp16_mask
        if dup_last_num > 0:
            self.tik_instance.vector_dup(dup_last_num,
                                         queries_pad_ub[dup_offset], self.zero,
                                         1, 1, 8)

        # move queries from out to UB
        queries_ub = self.tik_instance.Tensor("float16",
                                              (self.dim,),
                                              name="queries_ub", scope=tik.scope_ubuf)
        self.tik_instance.data_move(queries_ub[0],
                                    self.input_queries_gm[loop_queries, 0],
                                    0, 1, self.dim // 16, 0, 0)

        # compute distance per list

        with self.tik_instance.for_range(0, aicore_sequence_num, thread_num=2) as loop_sequence:
            coarse_centroids_ub = self.tik_instance.Tensor("float16",
                                                           (self.dim,),
                                                           name="coarse_centroids_ub", scope=tik.scope_ubuf)
            residual_ub = self.tik_instance.Tensor("float16",
                                                   (self.dim,),
                                                   name="residual_ub", scope=tik.scope_ubuf)
            sequence_id = self.tik_instance.Scalar('int32')
            sequence_id.set_as(sequence_ub[loop_queries, aicore_offset + loop_sequence])

            # x-yc
            self.tik_instance.data_move(coarse_centroids_ub[0],
                                        self.input_coarse_centroids_gm[sequence_id, 0],
                                        0, 1, self.dim // 16, 0, 0)

            vsub_repeats = self.dim // self.fp16_mask
            vsub_offset = 0
            if vsub_repeats > 0:
                self.tik_instance.vsub(self.fp16_mask, residual_ub[0],
                                       queries_ub[0], coarse_centroids_ub[0],
                                       vsub_repeats, 1, 1, 1, 8, 8, 8)
                vsub_offset += vsub_repeats * self.fp16_mask

            vsub_last_num = self.dim % self.fp16_mask
            if vsub_last_num > 0:
                self.tik_instance.vsub(vsub_last_num, residual_ub[vsub_offset],
                                       queries_ub[vsub_offset], coarse_centroids_ub[vsub_offset],
                                       1, 1, 1, 1, 8, 8, 8)

            # expand x-yc per sub num
            vadds_repeats = self.dim // self.fp16_mask
            vadds_offset = 0
            if vadds_repeats > 0:
                self.tik_instance.vadds([71777214294589695, 71777214294589695],
                                        queries_pad_ub[0], residual_ub[0],
                                        0, vadds_repeats, 2, 1, 16, 8)
                self.tik_instance.vadds([18374966859414961920, 18374966859414961920],
                                        queries_pad_ub[16], residual_ub[0],
                                        0, vadds_repeats, 2, 1, 16, 8)
                vadds_offset += vadds_repeats * self.fp16_mask

            vadds_last_num = self.dim % self.fp16_mask
            if vadds_last_num > 0:
                mask0 = self.compute_mask(vadds_last_num, 1, 0)
                mask1 = self.compute_mask(vadds_last_num, 1, 1)
                if vadds_last_num < 64:
                    vadds_mask0 = [0, mask0]
                    vadds_mask1 = [0, mask1]
                else:
                    vadds_mask0 = [mask0, 71777214294589695]
                    vadds_mask1 = [mask1, 18374966859414961920]

                self.tik_instance.vadds(vadds_mask0, queries_pad_ub[vadds_offset * 2],
                                        residual_ub[vadds_offset], 0,
                                        1, 2, 1, 16, 8)
                self.tik_instance.vadds(vadds_mask1, queries_pad_ub[vadds_offset * 2 + 16],
                                        residual_ub[vadds_offset], 0,
                                        1, 2, 1, 16, 8)

            mul_pad_ub = self.tik_instance.Tensor("float16",
                                                  (self.pq_centroids_num, 16),
                                                  name="mul_pad_ub", scope=tik.scope_ubuf)
            dup_repeats = (self.pq_centroids_num * 16) // self.fp16_mask
            if dup_repeats > 0:
                self.tik_instance.vector_dup(self.fp16_mask,
                                             mul_pad_ub[0], self.zero,
                                             dup_repeats, 1, 8)

            # compute distance per quantizers
            with self.tik_instance.for_range(0, self.sub_quantizers_num, thread_num=2) as loop_quantizers:
                data1 = self.tik_instance.Scalar(dtype="float16")
                data2 = self.tik_instance.Scalar(dtype="float16")
                data3 = self.tik_instance.Scalar(dtype="float16")
                data4 = self.tik_instance.Scalar(dtype="float16")
                data5 = self.tik_instance.Scalar(dtype="float16")
                data6 = self.tik_instance.Scalar(dtype="float16")
                data7 = self.tik_instance.Scalar(dtype="float16")
                data8 = self.tik_instance.Scalar(dtype="float16")

                pq_centroids_ub = self.tik_instance.Tensor("float16",
                                                           (self.pq_centroids_num, self.sub_dim),
                                                           name="pq_centroids_ub", scope=tik.scope_ubuf)
                self.tik_instance.data_move(pq_centroids_ub[0],
                                            self.input_pq_centroids_gm[loop_quantizers, 0],
                                            0, 1, self.pq_centroids_num * self.sub_dim // 16, 0, 0)

                sub_residual_ub = self.tik_instance.Tensor("float16",
                                                           (16,),
                                                           name="sub_residual_ub", scope=tik.scope_ubuf)
                self.tik_instance.data_move(sub_residual_ub[0],
                                            queries_pad_ub[loop_quantizers * 16],
                                            0, 1, 1, 0, 0)

                # copy sub residual per sub num
                with self.tik_instance.if_scope(loop_quantizers % 2 == 0):
                    data1.set_as(sub_residual_ub[0])
                    data2.set_as(sub_residual_ub[1])
                    data3.set_as(sub_residual_ub[2])
                    data4.set_as(sub_residual_ub[3])
                    data5.set_as(sub_residual_ub[4])
                    data6.set_as(sub_residual_ub[5])
                    data7.set_as(sub_residual_ub[6])
                    data8.set_as(sub_residual_ub[7])
                    sub_residual_ub[8].set_as(data1)
                    sub_residual_ub[9].set_as(data2)
                    sub_residual_ub[10].set_as(data3)
                    sub_residual_ub[11].set_as(data4)
                    sub_residual_ub[12].set_as(data5)
                    sub_residual_ub[13].set_as(data6)
                    sub_residual_ub[14].set_as(data7)
                    sub_residual_ub[15].set_as(data8)
                with self.tik_instance.else_scope():
                    data1.set_as(sub_residual_ub[8])
                    data2.set_as(sub_residual_ub[9])
                    data3.set_as(sub_residual_ub[10])
                    data4.set_as(sub_residual_ub[11])
                    data5.set_as(sub_residual_ub[12])
                    data6.set_as(sub_residual_ub[13])
                    data7.set_as(sub_residual_ub[14])
                    data8.set_as(sub_residual_ub[15])
                    sub_residual_ub[0].set_as(data1)
                    sub_residual_ub[1].set_as(data2)
                    sub_residual_ub[2].set_as(data3)
                    sub_residual_ub[3].set_as(data4)
                    sub_residual_ub[4].set_as(data5)
                    sub_residual_ub[5].set_as(data6)
                    sub_residual_ub[6].set_as(data7)
                    sub_residual_ub[7].set_as(data8)

                mul_ub = self.tik_instance.Tensor("float16",
                                                  (self.pq_centroids_num, self.sub_dim),
                                                  name="mul_ub", scope=tik.scope_ubuf)
                dst_ub = self.tik_instance.Tensor("float16",
                                                  (self.pq_centroids_num,),
                                                  name="dst_ub", scope=tik.scope_ubuf)

                # sub residual do sub with pq centroids
                vsub_loop = (self.pq_centroids_num * self.sub_dim) // (16 * 255)
                vsub_offset = self.tik_instance.Scalar(dtype="int32")
                vsub_offset.set_as(0)
                if vsub_loop > 0:
                    with self.tik_instance.for_range(0, vsub_loop):
                        self.tik_instance.vsub(16, mul_ub[vsub_offset // self.sub_dim, vsub_offset % self.sub_dim],
                                               sub_residual_ub[0],
                                               pq_centroids_ub[vsub_offset // self.sub_dim, vsub_offset % self.sub_dim],
                                               255, 1, 1, 1, 1, 0, 1)
                        vsub_offset.set_as(vsub_offset + 255 * 16)

                vsub_repeats = (self.pq_centroids_num * self.sub_dim) % (16 * 255) // 16
                if vsub_repeats > 0:
                    self.tik_instance.vsub(16, mul_ub[vsub_offset // self.sub_dim, vsub_offset % self.sub_dim],
                                           sub_residual_ub[0],
                                           pq_centroids_ub[vsub_offset // self.sub_dim, vsub_offset % self.sub_dim],
                                           vsub_repeats, 1, 1, 1, 1, 0, 1)

                # do mul
                vmul_repeats = (self.pq_centroids_num * self.sub_dim) // self.fp16_mask
                if vmul_repeats > 0:
                    self.tik_instance.vmul(self.fp16_mask, mul_ub[0],
                                           mul_ub[0], mul_ub[0],
                                           vmul_repeats, 1, 1, 1, 8, 8, 8)

                vadds_repeats = (self.pq_centroids_num * self.sub_dim) // self.fp16_mask
                if vadds_repeats > 0:
                    self.tik_instance.vadds([71777214294589695, 71777214294589695],
                                            mul_pad_ub[0, 0], mul_ub[0],
                                            0, vadds_repeats, 2, 1, 16, 8)
                    self.tik_instance.vadds([18374966859414961920, 18374966859414961920],
                                            mul_pad_ub[1, 0], mul_ub[0],
                                            0, vadds_repeats, 2, 1, 16, 8)

                # do vcgadd
                vcgadd_repeats = (self.pq_centroids_num * 16) // self.fp16_mask
                if vcgadd_repeats > 0:
                    self.tik_instance.vcgadd(self.fp16_mask,
                                             dst_ub[0], mul_pad_ub[0],
                                             vcgadd_repeats, 1, 1, 8)

                self.tik_instance.data_move(self.output_distance_table_gm[loop_queries,
                                                                          aicore_offset + loop_sequence,
                                                                          loop_quantizers, 0],
                                            dst_ub[0],
                                            0, 1, self.pq_centroids_num // 16, 0, 0)

    # compute distance table build with other sub dim
    def distance_table_build_compute(self, loop_queries, sequence_ub, aicore_offset, aicore_sequence_num):
        # move queries from out to UB
        queries_ub = self.tik_instance.Tensor("float16",
                                              (self.dim,),
                                              name="queries_ub", scope=tik.scope_ubuf)
        self.tik_instance.data_move(queries_ub[0],
                                    self.input_queries_gm[loop_queries, 0],
                                    0, 1, self.dim // 16, 0, 0)

        # compute distance per list
        with self.tik_instance.for_range(0, aicore_sequence_num, thread_num=2) as loop_sequence:
            coarse_centroids_ub = self.tik_instance.Tensor("float16",
                                                           (self.dim,),
                                                           name="coarse_centroids_ub", scope=tik.scope_ubuf)
            residual_ub = self.tik_instance.Tensor("float16",
                                                   (self.dim,),
                                                   name="residual_ub", scope=tik.scope_ubuf)
            sequence_id = self.tik_instance.Scalar('int32')
            sequence_id.set_as(sequence_ub[loop_queries, aicore_offset + loop_sequence])

            self.tik_instance.data_move(coarse_centroids_ub[0],
                                        self.input_coarse_centroids_gm[sequence_id, 0],
                                        0, 1, self.dim // 16, 0, 0)

            # x-yc
            vsub_repeats = self.dim // self.fp16_mask
            vsub_offset = 0
            if vsub_repeats > 0:
                self.tik_instance.vsub(self.fp16_mask, residual_ub[0],
                                       queries_ub[0], coarse_centroids_ub[0],
                                       vsub_repeats, 1, 1, 1, 8, 8, 8)
                vsub_offset += vsub_repeats * self.fp16_mask

            vsub_last_num = self.dim % self.fp16_mask
            if vsub_last_num > 0:
                self.tik_instance.vsub(vsub_last_num, residual_ub[vsub_offset],
                                       queries_ub[vsub_offset], coarse_centroids_ub[vsub_offset],
                                       1, 1, 1, 1, 8, 8, 8)

            sub_dim_each_loop = min(256 * 128 // self.pq_centroids_num, self.sub_dim)
            while sub_dim_each_loop > self.fp16_mask:
                sub_dim_each_loop /= 2
            sub_dim_loops = self.sub_dim // sub_dim_each_loop

            # compute distance per quantizers
            with self.tik_instance.for_range(0, self.sub_quantizers_num, thread_num=2) as loop_quantizers:
                dst_ub = self.tik_instance.Tensor("float16",
                                                  (sub_dim_loops, self.pq_centroids_num),
                                                  name="dst_ub", scope=tik.scope_ubuf)

                pq_centroids_ub = self.tik_instance.Tensor("float16",
                                                           (self.pq_centroids_num, sub_dim_each_loop),
                                                           name="pq_centroids_ub", scope=tik.scope_ubuf)

                sub_residual_ub = self.tik_instance.Tensor("float16",
                                                           (sub_dim_each_loop,),
                                                           name="sub_residual_ub", scope=tik.scope_ubuf)

                for loop_sub_dim in range(sub_dim_loops):
                    self.tik_instance.data_move(
                        pq_centroids_ub,
                        self.input_pq_centroids_gm[loop_quantizers, loop_sub_dim * sub_dim_each_loop],
                        0, self.pq_centroids_num, sub_dim_each_loop // 16,
                        (self.sub_dim - sub_dim_each_loop) // 16, 0)

                    self.tik_instance.data_move(
                        sub_residual_ub,
                        residual_ub[loop_quantizers * self.sub_dim + loop_sub_dim * sub_dim_each_loop],
                        0, 1, sub_dim_each_loop // 16, 0, 0)

                    # do vsub
                    vsub_loops = self.pq_centroids_num // 255
                    if vsub_loops > 0:
                        for loop_vsub in range(vsub_loops):
                            self.tik_instance.vsub(sub_dim_each_loop, pq_centroids_ub[loop_vsub * 255, 0],
                                                   sub_residual_ub,
                                                   pq_centroids_ub[loop_vsub * 255, 0],
                                                   255, 1, 1, 1, sub_dim_each_loop // 16, 0, sub_dim_each_loop // 16)

                    vsub_repeats = self.pq_centroids_num % 255
                    if vsub_repeats > 0:
                        self.tik_instance.vsub(sub_dim_each_loop, pq_centroids_ub[vsub_loops * 255, 0],
                                               sub_residual_ub, pq_centroids_ub[vsub_loops * 255, 0],
                                               vsub_repeats,
                                               1, 1, 1, sub_dim_each_loop // 16, 0, sub_dim_each_loop // 16)

                    # do vmul
                    vmul_loops = (self.pq_centroids_num * sub_dim_each_loop) // (self.fp16_mask * 255)
                    if vmul_loops > 0:
                        for mul_index in range(vmul_loops):
                            self.tik_instance.vmul(
                                self.fp16_mask,
                                pq_centroids_ub[mul_index * self.fp16_mask * 255 // sub_dim_each_loop,
                                                mul_index * self.fp16_mask * 255 % sub_dim_each_loop],
                                pq_centroids_ub[mul_index * self.fp16_mask * 255 // sub_dim_each_loop,
                                                mul_index * self.fp16_mask * 255 % sub_dim_each_loop],
                                pq_centroids_ub[mul_index * self.fp16_mask * 255 // sub_dim_each_loop,
                                                mul_index * self.fp16_mask * 255 % sub_dim_each_loop],
                                255, 1, 1, 1, 8, 8, 8)

                    vmul_repeats = (self.pq_centroids_num * sub_dim_each_loop) % (
                                self.fp16_mask * 255) // self.fp16_mask
                    if vmul_repeats > 0:
                        self.tik_instance.vmul(self.fp16_mask,
                                               pq_centroids_ub[vmul_loops * self.fp16_mask * 255 // sub_dim_each_loop,
                                                               vmul_loops * self.fp16_mask * 255 % sub_dim_each_loop],
                                               pq_centroids_ub[vmul_loops * self.fp16_mask * 255 // sub_dim_each_loop,
                                                               vmul_loops * self.fp16_mask * 255 % sub_dim_each_loop],
                                               pq_centroids_ub[vmul_loops * self.fp16_mask * 255 // sub_dim_each_loop,
                                                               vmul_loops * self.fp16_mask * 255 % sub_dim_each_loop],
                                               vmul_repeats, 1, 1, 1, 8, 8, 8)

                    # do vcadd
                    vcadd_loops = self.pq_centroids_num // 255
                    if vcadd_loops > 0:
                        for loop_vcadd in range(vcadd_loops):
                            self.tik_instance.vcadd(sub_dim_each_loop, dst_ub[loop_sub_dim, loop_vcadd * 255],
                                                    pq_centroids_ub[loop_vcadd * 255, 0],
                                                    255, 1, 1, sub_dim_each_loop // 16)

                    vcadd_repeats = self.pq_centroids_num % 255
                    if vcadd_repeats > 0:
                        self.tik_instance.vcadd(sub_dim_each_loop, dst_ub[loop_sub_dim, vcadd_loops * 255],
                                                pq_centroids_ub[vcadd_loops * 255, 0],
                                                vcadd_repeats, 1, 1, sub_dim_each_loop // 16)

                vadd_repeats = self.pq_centroids_num // self.fp16_mask
                vadd_last_mask = self.pq_centroids_num % self.fp16_mask
                for loop in range(1, sub_dim_loops):
                    if vadd_repeats > 0:
                        self.tik_instance.vadd(self.fp16_mask, dst_ub[0, 0],
                                               dst_ub[0, 0], dst_ub[loop, 0],
                                               vadd_repeats, 1, 1, 1,
                                               self.fp16_mask // 16, self.fp16_mask // 16, self.fp16_mask // 16)
                    if vadd_last_mask > 0:
                        self.tik_instance.vadd(vadd_last_mask, dst_ub[0, vadd_repeats * self.fp16_mask],
                                               dst_ub[0, vadd_repeats * self.fp16_mask],
                                               dst_ub[loop, vadd_repeats * self.fp16_mask],
                                               1, 1, 1, 1, 0, 0, 0)

                self.tik_instance.data_move(self.output_distance_table_gm[loop_queries,
                                                                          aicore_offset + loop_sequence,
                                                                          loop_quantizers, 0],
                                            dst_ub,
                                            0, 1, self.pq_centroids_num // 16, 0, 0)

    def forward(self):
        aicore_offset = self.tik_instance.Scalar(dtype="int32")
        aicore_offset.set_as(0)

        with self.tik_instance.for_range(0, self.aicore_use, block_num=self.aicore_use) as block_index:
            # move sequence number from out to UB
            sequence_ub = self.tik_instance.Tensor("int32",
                                                   self.shape_sequence_number,
                                                   name="sequence_ub", scope=tik.scope_ubuf)
            self.tik_instance.data_move(sequence_ub[0],
                                        self.input_sequence_number_gm[0],
                                        0, 1, self.queries_num * self.closest_num // 8, 0, 0)

            # compute distance per queries
            with self.tik_instance.for_range(0, self.queries_num) as loop_queries:
                # compute number of closest IVF centroids to the query vector
                num = self.tik_instance.Scalar('int32')
                num.set_as(0)
                with self.tik_instance.for_range(0, self.closest_num) as loop_i:
                    with self.tik_instance.if_scope(sequence_ub[loop_queries, loop_i] > -1):
                        num.set_as(num + 1)

                sequence_num_each_core = num // self.aicore_use
                sequence_num_last_core = num - (self.aicore_use - 1) * sequence_num_each_core
                aicore_offset.set_as(block_index * sequence_num_each_core)

                with self.tik_instance.if_scope(block_index != self.aicore_use - 1):
                    aicore_sequence_num = sequence_num_each_core
                with self.tik_instance.else_scope():
                    aicore_sequence_num = sequence_num_last_core

                if self.sub_dim == 4:
                    self.distance_table_build_compute_subdim_4(loop_queries, sequence_ub,
                                                               aicore_offset, aicore_sequence_num)
                elif self.sub_dim == 8:
                    self.distance_table_build_compute_subdim_8(loop_queries, sequence_ub,
                                                               aicore_offset, aicore_sequence_num)
                else:
                    self.distance_table_build_compute(loop_queries, sequence_ub,
                                                      aicore_offset, aicore_sequence_num)

        one = self.tik_instance.Scalar("uint16", "one", 1)
        flag_ub = self.tik_instance.Tensor("uint16",
                                           self.shape_flag,
                                           name="flag_ub", scope=tik.scope_ubuf)
        flag_ub[0].set_as(one)
        self.tik_instance.pipe_barrier("PIPE_MTE3")
        self.tik_instance.data_move(self.output_flag_gm, flag_ub, 0, 1, 1, 0, 0)

        self.tik_instance.BuildCCE(
            kernel_name=self.kernel_name,
            inputs=[
                self.input_queries_gm, self.input_pq_centroids_gm,
                self.input_sequence_number_gm, self.input_coarse_centroids_gm
            ],
            outputs=[self.output_distance_table_gm, self.output_flag_gm])

        return self.tik_instance


def distance_table_build(input_queries,
                         input_pq_centroids,
                         input_sequence_number,
                         input_coarse_centroids,
                         output_distance_table,
                         output_flag,
                         kernel_name="distance_table_build"):
    """
    calculating distance

    Parameters
    ----------
    input_queries : dict
        shape and dtype of query vector
    input_pq_centroids : dict
        shape and dtype of pq coarse centroids
    input_sequence_number : dict
        shape and dtype of sequence number of the np closest
        IVF centroids to the query vector
    input_coarse_centroids : dict
        shape and dtype of coarse centroids
    output_distance_table : dict
        shape and dtype of distance table, should be same
        dtype as input_queries
    output_flag : dict
        shape and dtype of flag
    kernel_name : str
        kernel name, default value is "distance_table_build"

    Returns
    -------
    None
    """
    distance_table_build_instance = DistanceTableBuild(
        input_queries, input_pq_centroids, input_sequence_number,
        input_coarse_centroids, output_distance_table, output_flag,
        kernel_name)
    tik_instance = distance_table_build_instance.forward()
    return tik_instance
