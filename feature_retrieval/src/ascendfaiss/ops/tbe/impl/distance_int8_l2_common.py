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
from enum import Enum
from te import tik
from mxIndex_impl.common import set_soc_info


class DistanceInt8L2Mins:
    def __init__(self, input_queries, input_mask, input_centroids, input_precomputed, input_actual_num,
                 output_dist, output_min_dist, output_flag, use_fp16, kernel_name):
        self.shape_queries = input_queries.get("shape")
        self.dtype_queries = input_queries.get("dtype")
        self.shape_mask = input_mask.get("shape")
        self.dtype_mask = input_mask.get("dtype")
        self.shape_centroids = input_centroids.get("shape")
        self.dtype_centroids = input_centroids.get("dtype")
        self.shape_precomputed = input_precomputed.get("shape")
        self.dtype_precomputed = input_precomputed.get("dtype")
        self.shape_actual_num = input_actual_num.get("shape")
        self.dtype_actual_num = input_actual_num.get("dtype")
        self.shape_dist = output_dist.get("shape")
        self.dtype_dist = output_dist.get("dtype")
        self.shape_min_dist = output_min_dist.get("shape")
        self.dtype_min_dist = output_min_dist.get("dtype")
        self.shape_flag = output_flag.get("shape")
        self.dtype_flag = output_flag.get("dtype")
        self.kernel_name = kernel_name

        self.use_fp16 = use_fp16

        # compute parameter
        self.queries_num, self.dim = self.shape_queries
        self.centroids_num, = self.shape_precomputed

        # check parameter
        self.check_parameter()

        self.block_offset = None
        self.mask_len = None
        self.use_mask = None

        # set vector fp32 mask and fp16 mask
        self.int32_mask = 64
        self.min_mask = 64
        self.fp16_mask = 128
        self.max_repeat = 192
        # scale changed with dim
        self.scale = 0.01 / min(self.dim // 64, max(self.dim // 128 + 1, 4))

        # set tik instance
        set_soc_info()
        self.tik_instance = tik.Tik()
        self.available_core_num = self.shape_actual_num[0]
        self.query_l2 = self.tik_instance.Scalar("int32", name="query_l2")
        self.version = tik.Dprofile().get_product_name()

        self.min_num_each_loop = 128 if self.use_fp16 else 64
        self.queries_num_each_loop = min(self.min_num_each_loop, self.queries_num)
        self.centroids_num_each_loop = min((self.min_num_each_loop // self.queries_num_each_loop) * 512, 1024)

        self.centroids_num_each_loop = 512 if self.dim > 512 else self.centroids_num_each_loop
        self.centroids_num_each_loop_half = self.centroids_num_each_loop // 2

        self.set_src_dst_tensor()

    def check_parameter(self):
        if self.dim % 32 != 0:
            raise RuntimeError("feature dim must be a multiple of 32")
        if self.centroids_num % 16 != 0:
            raise RuntimeError("centroids num must be a multiple of 16")
        if self.dim < 64:
            raise RuntimeError("feature dim must be greater than or equal to 64")

    def set_src_dst_tensor(self):
        self.input_queries_gm = self.tik_instance.Tensor(self.dtype_queries, self.shape_queries,
                                                         name="input_queries_gm", scope=tik.scope_gm)
        self.input_mask_gm = self.tik_instance.Tensor(self.dtype_mask, self.shape_mask,
                                                      name="input_mask_gm", scope=tik.scope_gm)
        self.input_centroids_gm = self.tik_instance.Tensor(self.dtype_centroids, self.shape_centroids,
                                                           name="input_centroids_gm", scope=tik.scope_gm)
        self.input_precomputed_gm = self.tik_instance.Tensor(self.dtype_precomputed, self.shape_precomputed,
                                                             name="input_precomputed_gm", scope=tik.scope_gm)
        self.input_actual_num_gm = self.tik_instance.Tensor(self.dtype_actual_num, self.shape_actual_num,
                                                            name="input_actual_num_gm", scope=tik.scope_gm)
        self.output_dist_gm = self.tik_instance.Tensor(self.dtype_dist, self.shape_dist,
                                                       name="output_dist_gm", scope=tik.scope_gm)
        self.output_min_dist_gm = self.tik_instance.Tensor(self.dtype_min_dist, self.shape_min_dist,
                                                           name="output_min_dist_gm", scope=tik.scope_gm)
        self.output_flag_gm = self.tik_instance.Tensor(self.dtype_flag, self.shape_flag,
                                                       name="output_flag_gm", scope=tik.scope_gm)

    def cal_num_each_core(self):
        """
        calculate actual code num of each core
        """
        # move actual code num from out to UB
        actual_num_ub = self.tik_instance.Tensor("uint32", (8,), name="actual_code_num_ub", scope=tik.scope_ubuf)
        self.tik_instance.data_move(actual_num_ub, self.input_actual_num_gm, 0, 1, 1, 0, 0)

        self.block_offset = self.tik_instance.Scalar(dtype="uint32", name="block_offset")
        self.block_offset.set_as(actual_num_ub[1])

        self.mask_len = self.tik_instance.Scalar(dtype="uint32", name="mask_len")
        self.mask_len.set_as(actual_num_ub[2])

        self.use_mask = self.tik_instance.Scalar(dtype="uint32", name="use_mask")
        self.use_mask.set_as(actual_num_ub[3])

        self.actual_num = self.tik_instance.Scalar(dtype="uint32", name="int8_l2_actual_num")
        self.actual_num.set_as(actual_num_ub[0])

    def cube_compute_each_loop(self, queries_l1, queries_square_ub,
                               centroids_offset, centroids_offset_in_loop, queries_offset, queries_num):
        """
        compute centroids of each loop
        """
        queries_align = (queries_num + 15) // 16 * 16
        loop_times = queries_num * self.centroids_num_each_loop // self.min_mask // self.max_repeat
        remain = queries_num * self.centroids_num_each_loop // self.min_mask % self.max_repeat

        centroids_l1 = self.tik_instance.Tensor("int8",
                                                (self.dim // 32, self.centroids_num_each_loop, 32),
                                                name="centroids_l1", scope=tik.scope_cbuf)
        for i in range(self.dim // 32):
            self.tik_instance.data_move(centroids_l1[i, 0, 0],
                                        self.input_centroids_gm[(centroids_offset + centroids_offset_in_loop) // 16,
                                                                i, 0, 0],
                                        0, self.centroids_num_each_loop // 16, 16, self.dim // 2 - 16, 0)

        # for x^2 + y^2 - 2xy
        add_ub = self.tik_instance.Tensor("int32", (queries_num, self.centroids_num_each_loop),
                                          name="add_ub", scope=tik.scope_ubuf)

        add_ub = self.compute_part(queries_align, queries_l1, queries_num, centroids_l1, add_ub)
        add_ub = self.compute_part_norm(add_ub, queries_square_ub, queries_num,
                                        centroids_offset, centroids_offset_in_loop)

        dst_ub = self.tik_instance.Tensor("float16", (queries_num, self.centroids_num_each_loop),
                                          name="add_ub", scope=tik.scope_ubuf)
        offset = 0
        for _ in range(loop_times):
            self.tik_instance.vconv(self.int32_mask, 'none', dst_ub[offset, 0], add_ub[offset, 0], self.max_repeat,
                                    1, 1, 4, 8, deqscale=self.scale)
            offset += (self.max_repeat * self.min_mask // self.centroids_num_each_loop)
        if remain > 0:
            self.tik_instance.vconv(self.int32_mask, 'none', dst_ub[offset, 0], add_ub[offset, 0], remain,
                                    1, 1, 4, 8, deqscale=self.scale)
        dst_ub = self.compute_part_mask(queries_offset, centroids_offset, queries_num, centroids_offset_in_loop, dst_ub)

        self.tik_instance.data_move(self.output_dist_gm[queries_offset,
                                                        centroids_offset + centroids_offset_in_loop],
                                    dst_ub, 0, queries_num, self.centroids_num_each_loop // 16, 0,
                                    (self.centroids_num - self.centroids_num_each_loop) // 16)

        min_ub = self.tik_instance.Tensor("float16", (queries_num, self.centroids_num_each_loop // 32),
                                          name="min_ub", scope=tik.scope_ubuf)

        offset = 0
        for _ in range(loop_times):
            self.tik_instance.vcmin(self.min_mask, min_ub[offset, 0], dst_ub[offset, 0], self.max_repeat, 1, 1,
                                    self.min_mask // 16)
            offset += (self.max_repeat * self.min_mask // self.centroids_num_each_loop)

        if remain > 0:
            self.tik_instance.vcmin(self.min_mask, min_ub[offset, 0], dst_ub[offset, 0], remain, 1, 1,
                                    self.min_mask // 16)

        self.tik_instance.data_move(
            self.output_min_dist_gm[queries_offset, (centroids_offset + centroids_offset_in_loop) // self.min_mask * 2],
            min_ub, 0, queries_num, self.centroids_num_each_loop // self.min_mask // 8, 0,
                                    (self.centroids_num - self.centroids_num_each_loop) // self.min_mask // 8)

    def cube_compute_each_loop_without_query_norm(self, queries_l1,
                                                  centroids_offset, centroids_offset_in_loop, queries_offset,
                                                  queries_num):
        """
        compute centroids of each loop
        """
        queries_align = (queries_num + 15) // 16 * 16
        loop_times = queries_num * self.centroids_num_each_loop // self.min_mask // self.max_repeat
        remain = queries_num * self.centroids_num_each_loop // self.min_mask % self.max_repeat

        centroids_l1 = self.tik_instance.Tensor("int8",
                                                (self.dim // 32, self.centroids_num_each_loop, 32),
                                                name="centroids_l1", scope=tik.scope_cbuf)
        for i in range(self.dim // 32):
            self.tik_instance.data_move(centroids_l1[i, 0, 0],
                                        self.input_centroids_gm[(centroids_offset + centroids_offset_in_loop) // 16,
                                                                i, 0, 0],
                                        0, self.centroids_num_each_loop // 16, 16, self.dim // 2 - 16, 0)

        # for y^2 - 2xy
        add_ub = self.tik_instance.Tensor("float16", (queries_num, self.centroids_num_each_loop),
                                          name="add_ub", scope=tik.scope_ubuf)

        add_ub = self.compute_part_fp16(queries_align, queries_l1, queries_num, centroids_l1, add_ub)
        add_ub = self.compute_part_norm_fp16(add_ub, queries_num, centroids_offset, centroids_offset_in_loop)

        add_ub = self.compute_part_mask(queries_offset, centroids_offset, queries_num, centroids_offset_in_loop, add_ub)

        self.tik_instance.data_move(self.output_dist_gm[queries_offset,
                                                        centroids_offset + centroids_offset_in_loop],
                                    add_ub, 0, queries_num, self.centroids_num_each_loop // 16, 0,
                                    (self.centroids_num - self.centroids_num_each_loop) // 16)
        min_ub = self.tik_instance.Tensor("float16", (queries_num, self.centroids_num_each_loop // 32),
                                          name="min_ub", scope=tik.scope_ubuf)

        offset = 0
        for _ in range(loop_times):
            self.tik_instance.vcmin(self.min_mask, min_ub[offset, 0], add_ub[offset, 0], self.max_repeat, 1, 1,
                                    self.min_mask // 16)
            offset += (self.max_repeat * self.min_mask // self.centroids_num_each_loop)

        if remain > 0:
            self.tik_instance.vcmin(self.min_mask, min_ub[offset, 0], add_ub[offset, 0], remain, 1, 1,
                                    self.min_mask // 16)

        self.tik_instance.data_move(
            self.output_min_dist_gm[queries_offset, (centroids_offset + centroids_offset_in_loop) // self.min_mask * 2],
            min_ub, 0, queries_num, self.centroids_num_each_loop // self.min_mask // 8, 0,
                                    (self.centroids_num - self.centroids_num_each_loop) // self.min_mask // 8)

    def compute_part(self, queries_align, queries_l1, queries_num, centroids_l1, add_ub):
        """
        compute -2xy of each loop
        """
        with self.tik_instance.new_stmt_scope():
            # -2xy
            inner_product_l0c = self.tik_instance.Tensor("int32",
                                                         (self.centroids_num_each_loop // 16, queries_align, 16),
                                                         name="inner_product_l0c", scope=tik.scope_cbuf_out)
            self.tik_instance.matmul(inner_product_l0c, queries_l1, centroids_l1,
                                     queries_align, self.dim, self.centroids_num_each_loop)

            # mov xy from L0-C to UB
            inner_product_ub = self.tik_instance.Tensor("int32",
                                                        (self.centroids_num_each_loop_half // 16, queries_align, 16),
                                                        name="inner_product_ub", scope=tik.scope_ubuf)

            self.tik_instance.tensor_mov(inner_product_ub, inner_product_l0c, 'm', 1,
                                         self.centroids_num_each_loop_half * queries_align // 256, 0, 0)
            if self.version == "mini":
                coeff_ub = self.tik_instance.Tensor("int32", (16,), tik.scope_ubuf, "coeff_ub")
                self.tik_instance.vec_dup(16, coeff_ub, -2, 1, 8)
                for loop_index in range(0, queries_num):
                    self.tik_instance.vmul(16, add_ub[loop_index, 0], inner_product_ub[0, loop_index, 0], coeff_ub,
                                           self.centroids_num_each_loop_half // 16, 1, 1, 1, 2, queries_align * 2, 0)
            else:
                for loop_index in range(0, queries_num):
                    self.tik_instance.vmuls(16, add_ub[loop_index, 0], inner_product_ub[0, loop_index, 0], -2,
                                            self.centroids_num_each_loop_half // 16, 1, 1, 2, queries_align * 2)

            self.tik_instance.tensor_mov(inner_product_ub,
                                         inner_product_l0c[self.centroids_num_each_loop_half // 16, 0, 0], 'm', 1,
                                         self.centroids_num_each_loop_half * queries_align // 256, 0, 0)
            if self.version == "mini":
                for loop_index in range(0, queries_num):
                    self.tik_instance.vmul(16, add_ub[loop_index, self.centroids_num_each_loop_half],
                                           inner_product_ub[0, loop_index, 0], coeff_ub,
                                           self.centroids_num_each_loop_half // 16, 1, 1, 1, 2, queries_align * 2, 0)
            else:
                for loop_index in range(0, queries_num):
                    self.tik_instance.vmuls(16, add_ub[loop_index, self.centroids_num_each_loop_half],
                                            inner_product_ub[0, loop_index, 0], -2,
                                            self.centroids_num_each_loop_half // 16, 1, 1, 2, queries_align * 2)

        return add_ub

    def compute_part_fp16(self, queries_align, queries_l1, queries_num, centroids_l1, add_ub):
        """
        compute -2xy of each loop
        """
        with self.tik_instance.new_stmt_scope():
            # -2xy
            inner_product_l0c = self.tik_instance.Tensor("int32",
                                                         (self.centroids_num_each_loop // 16, queries_align, 16),
                                                         name="inner_product_l0c", scope=tik.scope_cbuf_out)
            self.tik_instance.matmul(inner_product_l0c, queries_l1, centroids_l1,
                                     queries_align, self.dim, self.centroids_num_each_loop)

            # mov xy from L0-C to UB
            inner_product_ub = self.tik_instance.Tensor("float16",
                                                        (self.centroids_num_each_loop_half // 16, queries_align, 16),
                                                        name="inner_product_ub", scope=tik.scope_ubuf)

            self.tik_instance.tensor_mov(inner_product_ub, inner_product_l0c, 'm', 1,
                                         self.centroids_num_each_loop_half * queries_align // 256, 0, 0,
                                         deqscale=self.scale * (-2))
            for loop_index in range(0, queries_num):
                self.tik_instance.data_move(add_ub[loop_index, 0], inner_product_ub[0, loop_index, 0], 0,
                                            self.centroids_num_each_loop_half // 16, 1, queries_align - 1, 0)

            self.tik_instance.tensor_mov(inner_product_ub,
                                         inner_product_l0c[self.centroids_num_each_loop_half // 16, 0, 0], 'm', 1,
                                         self.centroids_num_each_loop_half * queries_align // 256, 0, 0,
                                         deqscale=self.scale * (-2))

            for loop_index in range(0, queries_num):
                self.tik_instance.data_move(add_ub[loop_index, self.centroids_num_each_loop_half],
                                            inner_product_ub[0, loop_index, 0], 0,
                                            self.centroids_num_each_loop_half // 16, 1, queries_align - 1, 0)

        return add_ub

    def compute_part_mask(self, queries_offset, centroids_offset, queries_num, centroids_offset_in_loop, dst_ub):
        """
        Filter dst_ub of each loop
        """
        with self.tik_instance.if_scope(self.use_mask > 0):
            min_val_ub = self.tik_instance.Tensor("float16", (128,), name="min_val_ub", scope=tik.scope_ubuf)
            self.tik_instance.vec_dup(self.fp16_mask, min_val_ub, 65500, 1, 8)

            # malloc memory on chip
            sel_ub = self.tik_instance.Tensor("uint8", (queries_num, (self.centroids_num_each_loop + 7) // 8),
                                              name="sel_ub", scope=tik.scope_ubuf)
            with self.tik_instance.for_range(0, queries_num) as j:
                # move data from input_mask_gm to sel_ub
                self.tik_instance.data_move(sel_ub[j, 0],
                                            self.input_mask_gm[(j + queries_offset) * self.mask_len +
                                                               (self.block_offset + centroids_offset +
                                                                centroids_offset_in_loop) // 8],
                                            0, 1, (self.centroids_num_each_loop + 255) // 256, 8, 8)

                # cal the loop need to execute the selection process
                vsel_loop = self.centroids_num_each_loop // self.fp16_mask
                if vsel_loop > 0:
                    for vloop in range(vsel_loop):
                        # sel_ub can not use repeat times > 1, use for + offset
                        voffset = vloop * self.fp16_mask
                        # select value in dst_ub according to sel_ub
                        self.tik_instance.vec_sel(self.fp16_mask, 0, dst_ub[j, voffset],
                                                  sel_ub[j, voffset // 8], dst_ub[j, voffset],
                                                  min_val_ub, 1, 8, 8, 0)

                # handle tail in case for self.centroids_num_each_loop % self.fp16_mask != 0
                vsel_last = self.centroids_num_each_loop % self.fp16_mask
                if vsel_last > 0:
                    vsel_offset = vsel_loop * self.fp16_mask
                    self.tik_instance.vec_sel(vsel_last, 0, dst_ub[j, vsel_offset], sel_ub[j, vsel_offset // 8],
                                              dst_ub[j, vsel_offset], min_val_ub, 1, 8, 8, 0)
        return dst_ub

    def compute_part_norm(self, add_ub, queries_square_ub, queries_num, centroids_offset, centroids_offset_in_loop):
        """
        compute x^2 + y^2 of each loop
        """
        with self.tik_instance.new_stmt_scope():
            # x^2
            query_l2_ub = self.tik_instance.Tensor("int32", (self.int32_mask,), name="query_l2_ub",
                                                   scope=tik.scope_ubuf)
            for loop_index in range(0, queries_num):
                self.query_l2.set_as(queries_square_ub[loop_index // 16, loop_index, loop_index % 16])
                if self.version == "mini":
                    self.tik_instance.vec_dup(self.int32_mask, query_l2_ub, self.query_l2, 1, self.int32_mask // 8)
                    self.tik_instance.vadd(self.int32_mask, add_ub[loop_index, 0], add_ub[loop_index, 0], query_l2_ub,
                                           self.centroids_num_each_loop // self.int32_mask, 1, 1, 1, 8, 8, 0)
                else:
                    self.tik_instance.vadds(self.int32_mask, add_ub[loop_index, 0], add_ub[loop_index, 0],
                                            self.query_l2, self.centroids_num_each_loop // self.int32_mask, 1, 1, 8, 8)

            # y^2
            centroids_l2_ub = self.tik_instance.Tensor("int32", (self.centroids_num_each_loop,),
                                                       name="centroids_l2_ub", scope=tik.scope_ubuf)
            self.tik_instance.data_move(centroids_l2_ub,
                                        self.input_precomputed_gm[centroids_offset + centroids_offset_in_loop],
                                        0, 1, self.centroids_num_each_loop // 8, 0, 0)

            for loop_index in range(0, self.centroids_num_each_loop // self.int32_mask):
                self.tik_instance.vadd(self.int32_mask, add_ub[0, loop_index * self.int32_mask],
                                       add_ub[0, loop_index * self.int32_mask],
                                       centroids_l2_ub[loop_index * self.int32_mask],
                                       queries_num, 1, 1, 1, self.centroids_num_each_loop // 8,
                                       self.centroids_num_each_loop // 8, 0)
        return add_ub

    def compute_part_norm_fp16(self, add_ub, queries_num, centroids_offset, centroids_offset_in_loop):
        """
        compute y^2 of each loop
        """
        with self.tik_instance.new_stmt_scope():
            # y^2
            centroids_l2_ub = self.tik_instance.Tensor("float16", (self.centroids_num_each_loop,),
                                                       name="centroids_l2_ub", scope=tik.scope_ubuf)
            self.tik_instance.data_move(centroids_l2_ub,
                                        self.input_precomputed_gm[centroids_offset + centroids_offset_in_loop],
                                        0, 1, self.centroids_num_each_loop // 16, 0, 0)

            for loop_index in range(0, self.centroids_num_each_loop // self.fp16_mask):
                self.tik_instance.vadd(self.fp16_mask, add_ub[0, loop_index * self.fp16_mask],
                                       add_ub[0, loop_index * self.fp16_mask],
                                       centroids_l2_ub[loop_index * self.fp16_mask],
                                       queries_num, 1, 1, 1, self.centroids_num_each_loop // 16,
                                       self.centroids_num_each_loop // 16, 0)
        return add_ub

    def distance_int8_l2_each_loop(self, centroids_offset, centroids_num, queries_offset, queries_num):
        """
        compute queries of each core
        """
        queries_align = (queries_num + 15) // 16 * 16

        queries_l1 = self.tik_instance.Tensor("int8", (self.dim // 32, queries_align, 32),
                                              name="queries_l1", scope=tik.scope_cbuf)
        for i in range(queries_num):
            self.tik_instance.data_move(queries_l1[0, i, 0], self.input_queries_gm[queries_offset + i, 0],
                                        0, self.dim // 32, 1, 0, queries_align - 1)

        if self.use_fp16 is True:
            with self.tik_instance.for_range(0, centroids_num // self.centroids_num_each_loop) as loop_centroids:
                self.cube_compute_each_loop_without_query_norm(queries_l1, centroids_offset,
                                                               loop_centroids * self.centroids_num_each_loop,
                                                               queries_offset, queries_num)
        else:
            queries_square_ub = self.tik_instance.Tensor("int32", (queries_align // 16, queries_align, 16),
                                                         name="queries_square_ub", scope=tik.scope_ubuf)
            with self.tik_instance.new_stmt_scope():
                queries_square_l0c = self.tik_instance.Tensor("int32", (queries_align // 16, queries_align, 16),
                                                              name="queries_square_l0c",
                                                              scope=tik.scope_cbuf_out)
                self.tik_instance.matmul(queries_square_l0c, queries_l1, queries_l1,
                                         queries_align, self.dim, queries_align)

                self.tik_instance.tensor_mov(queries_square_ub, queries_square_l0c, 'm', 1,
                                             queries_align * queries_align // 256, 0, 0)

            with self.tik_instance.for_range(0, centroids_num // self.centroids_num_each_loop) as loop_centroids:
                self.cube_compute_each_loop(queries_l1, queries_square_ub, centroids_offset,
                                            loop_centroids * self.centroids_num_each_loop,
                                            queries_offset, queries_num)

    def distance_compute(self):
        """
        the compute process
        """
        self.cal_num_each_core()
        queries_loop_time = self.queries_num // self.queries_num_each_loop
        queries_last_num = self.queries_num % self.queries_num_each_loop

        with self.tik_instance.new_stmt_scope():
            self.task_num = (self.actual_num + self.centroids_num_each_loop - 1) // self.centroids_num_each_loop
            self.aicore_use = self.tik_instance.Scalar(dtype="uint32", name="aicore_use",
                                                       init_value=self.available_core_num)
            self.batch_num_per_core_process = self.task_num // self.aicore_use
            self.batch_tail = self.task_num % self.aicore_use
            self.centroids_num_each_core = self.batch_num_per_core_process * self.centroids_num_each_loop
            self.centroids_num_each_core_head = self.centroids_num_each_core + self.centroids_num_each_loop

            with self.tik_instance.for_range(0, self.available_core_num,
                                             block_num=self.available_core_num) as block_index:
                with self.tik_instance.if_scope(block_index < self.batch_tail):
                    if queries_loop_time > 0:
                        with self.tik_instance.for_range(0, queries_loop_time) as loop_queries:
                            self.distance_int8_l2_each_loop(
                                block_index * self.centroids_num_each_core_head,
                                self.centroids_num_each_core_head,
                                loop_queries * self.queries_num_each_loop,
                                self.queries_num_each_loop)
                    if queries_last_num > 0:
                        self.distance_int8_l2_each_loop(
                            block_index * self.centroids_num_each_core_head, self.centroids_num_each_core_head,
                            queries_loop_time * self.queries_num_each_loop, queries_last_num)
                with self.tik_instance.else_scope():
                    if queries_loop_time > 0:
                        with self.tik_instance.for_range(0, queries_loop_time) as loop_queries:
                            self.distance_int8_l2_each_loop(
                                self.batch_tail * self.centroids_num_each_core_head + (
                                        block_index - self.batch_tail) * self.centroids_num_each_core,
                                self.centroids_num_each_core,
                                loop_queries * self.queries_num_each_loop,
                                self.queries_num_each_loop)
                    if queries_last_num > 0:
                        self.distance_int8_l2_each_loop(
                            self.batch_tail * self.centroids_num_each_core_head + (
                                    block_index - self.batch_tail) * self.centroids_num_each_core,
                            self.centroids_num_each_core,
                            queries_loop_time * self.queries_num_each_loop, queries_last_num)
                one = self.tik_instance.Scalar(dtype="uint16", name="one", init_value=1)
                flag_ub = self.tik_instance.Tensor("uint16", (16,), name="flag_ub", scope=tik.scope_ubuf)
                flag_ub[0].set_as(one)
                self.tik_instance.pipe_barrier("PIPE_MTE3")
                self.tik_instance.data_move(self.output_flag_gm[block_index, 0],
                                            flag_ub, 0, 1, 1, 0, 0)

    def get_tik_instance(self):
        """
        obtain tik instance
        """
        self.distance_compute()

        self.tik_instance.BuildCCE(kernel_name=self.kernel_name,
                                   inputs=[
                                       self.input_queries_gm,
                                       self.input_mask_gm,
                                       self.input_centroids_gm,
                                       self.input_precomputed_gm,
                                       self.input_actual_num_gm
                                   ],
                                   outputs=[self.output_dist_gm,
                                            self.output_min_dist_gm,
                                            self.output_flag_gm])
        return self.tik_instance