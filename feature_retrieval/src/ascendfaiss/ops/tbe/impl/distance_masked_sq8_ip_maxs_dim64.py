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


class DistanceSQ8IPMaxs:
    def __init__(self,
                 input_queries,
                 input_mask,
                 input_code,
                 input_diff,
                 input_min,
                 input_actual_num,
                 output_distances,
                 output_maxs,
                 output_flag,
                 kernel_name="distance_sq8_ip"):
        self.shape_queries = input_queries.get("shape")
        self.dtype_queries = input_queries.get("dtype")
        self.shape_mask = input_mask.get("shape")
        self.dtype_mask = input_mask.get("dtype")
        self.shape_code = input_code.get("shape")
        self.dtype_code = input_code.get("dtype")
        self.shape_diff = input_diff.get("shape")
        self.dtype_diff = input_diff.get("dtype")
        self.shape_min = input_min.get("shape")
        self.dtype_min = input_min.get("dtype")
        self.shape_actual_num = input_actual_num.get("shape")
        self.dtype_actual_num = input_actual_num.get("dtype")
        self.shape_distances = output_distances.get("shape")
        self.dtype_distances = output_distances.get("dtype")
        self.shape_maxs = output_maxs.get("shape")
        self.dtype_maxs = output_maxs.get("dtype")
        self.shape_flag = output_flag.get("shape")
        self.dtype_flag = output_flag.get("dtype")
        self.kernel_name = kernel_name

        # compute parameter
        self.queries_num, self.dim = self.shape_queries
        self.code_num = self.shape_code[0] * 16
        self.repeat_times = 128

        # check parameter
        self.check_parameter()

        # set vector fp32 mask and fp16 mask
        self.fp32_mask = 64
        self.fp16_mask = 128
        self.max_batch = 64

        # set tik instance
        self.set_tik_instance()

    def check_parameter(self):
        if self.dim != 64:
            raise RuntimeError("feature dim must be 64")
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
        self.query_each_loop = min(16, self.queries_num)
        self.query_loops = self.queries_num // self.query_each_loop

        self.code_each_loop = 1024
        self.code_loops = 4
        self.code_sub_loop = self.code_each_loop // self.code_loops
        self.maxs_each_loops = self.code_each_loop // self.max_batch * 2

        self.set_src_dst_tensor()

    def set_src_dst_tensor(self):
        """
        set input and output tensor
        """
        self.input_queries_gm = self.tik_instance.Tensor(
            self.dtype_queries,
            self.shape_queries,
            name="input_queries_gm",
            scope=tik.scope_gm)
        self.input_mask_gm = self.tik_instance.Tensor(
            self.dtype_mask,
            self.shape_mask,
            name="input_mask_gm",
            scope=tik.scope_gm)
        self.input_code_gm = self.tik_instance.Tensor(
            self.dtype_code,
            self.shape_code,
            name="input_code_gm",
            scope=tik.scope_gm)
        self.input_diff_gm = self.tik_instance.Tensor(
            self.dtype_diff,
            self.shape_diff,
            name="input_diff_gm",
            scope=tik.scope_gm)
        self.input_min_gm = self.tik_instance.Tensor(
            self.dtype_min,
            self.shape_min,
            name="input_min_gm",
            scope=tik.scope_gm)
        self.input_actual_num_gm = self.tik_instance.Tensor(
            self.dtype_actual_num,
            self.shape_actual_num,
            name="input_actual_num_gm",
            scope=tik.scope_gm)
        self.output_distances_gm = self.tik_instance.Tensor(
            self.dtype_distances,
            self.shape_distances,
            name="output_distances_gm",
            scope=tik.scope_gm)
        self.output_maxs_gm = self.tik_instance.Tensor(
            self.dtype_maxs,
            self.shape_maxs,
            name="output_maxs_gm",
            scope=tik.scope_gm)
        self.output_flag_gm = self.tik_instance.Tensor(
            self.dtype_flag,
            self.shape_flag,
            name="output_flag_gm",
            scope=tik.scope_gm)

    def cal_num_each_core(self):
        """
        calculate actual code num of each core
        """
        # move actual code num from out to UB
        actual_code_num_ub = self.tik_instance.Tensor("uint32", (8,), name="actual_code_num_ub", scope=tik.scope_ubuf)
        self.tik_instance.data_move(actual_code_num_ub, self.input_actual_num_gm, 0, 1, 1, 0, 0)

        actual_code_num = self.tik_instance.Scalar(dtype="uint32", name="actual_code_num")
        actual_code_num.set_as(actual_code_num_ub[0])
        self.mask_offset = self.tik_instance.Scalar(dtype="uint32", name="mask_offset")
        self.mask_offset.set_as(actual_code_num_ub[1])
        self.mask_len = self.tik_instance.Scalar(dtype="uint32", name="mask_len")
        self.mask_len.set_as(actual_code_num_ub[2])
 
        code_size = self.tik_instance.Scalar(dtype="uint32", name="code_size")
        code_size.set_as(actual_code_num_ub[3])
 
        if self.aicore_use == 2:
            self.code_num_each_core = ((code_size + 2048) // self.aicore_use + self.max_batch * 8) \
                // self.max_batch // 16 * self.max_batch * 16
        else:
            self.code_num_each_core = (code_size + 2048) // self.aicore_use // self.max_batch \
                // 16 * self.max_batch * 16
 
        self.code_num_last_core = code_size - (self.aicore_use - 1) * self.code_num_each_core

    def distance_compute_each_loop(self, aicore_offset, aicore_code_num, query_offset):
        add_scalar_list = [self.tik_instance.Scalar(dtype="float32") for i in range(self.query_each_loop)]
        data_xd_l1 = self.tik_instance.Tensor("float16", (self.dim // 16, self.queries_align, 16),
                                              name="data_xd_l1", scope=tik.scope_cbuf)

        with self.tik_instance.new_stmt_scope():
            queries_ub = self.tik_instance.Tensor("float16", (self.query_each_loop, self.dim),
                                                  name="queries_ub", scope=tik.scope_ubuf)
            diff_ub = self.tik_instance.Tensor("float16", (self.dim,), name="diff_ub", scope=tik.scope_ubuf)
            min_ub = self.tik_instance.Tensor("float16", (self.dim,), name="min_ub", scope=tik.scope_ubuf)

            # move diff adn min from out to UB
            self.tik_instance.data_move(diff_ub, self.input_diff_gm, 0, 1, self.dim // 16, 0, 0)
            self.tik_instance.data_move(min_ub, self.input_min_gm, 0, 1, self.dim // 16, 0, 0)
            diff2_ub = self.tik_instance.Tensor("float16", (self.dim,), name="diff2_ub", scope=tik.scope_ubuf)
            # cal 0.5 * d
            self._muls(diff2_ub, diff_ub, 0.5, self.dim, self.fp16_mask)
            # cal 255 * m
            self._muls(min_ub, min_ub, 255.0, self.dim, self.fp16_mask)
            # cal 0.5 * d + 255 * m
            self._add(diff2_ub, diff2_ub, min_ub, self.dim, self.fp16_mask)
            dm_transfer_ub = self.tik_instance.Tensor("float16", (self.dim // 16, 16, 16),
                                                      name="dm_transfer_ub", scope=tik.scope_ubuf)
            # use data_move function to make 0.5 * d + 255 * m align 16
            self.tik_instance.data_move(dm_transfer_ub, diff2_ub, 0, self.dim // 16, 1, 0, 15)
            # move d+m from UB to L1
            data_dm_l1 = self.tik_instance.Tensor("float16", (self.dim // 16, 16, 16),
                                                  name="data_dm_l1", scope=tik.scope_cbuf)
            self.tik_instance.data_move(data_dm_l1, dm_transfer_ub, 0, 1, self.dim, 0, 0)

            # move x from out to UB
            self.tik_instance.data_move(queries_ub, self.input_queries_gm[query_offset, 0],
                                        0, 1, self.query_each_loop * self.dim // 16, 0, 0)

            self.tik_instance.vmuls(self.dim, queries_ub, queries_ub,
                                    1 / 255, self.query_each_loop * self.dim // self.dim,
                                    1, 1, self.dim // 16, self.dim // 16)
            queries_transfer_ub = self.tik_instance.Tensor("float16", (self.dim // 16, self.queries_align, 16),
                                                           name="queries_transfer_ub", scope=tik.scope_ubuf)
            # use data_move function to make x align 16
            for i in range(self.query_each_loop):
                self.tik_instance.data_move(queries_transfer_ub[0, i, 0],
                                            queries_ub[i, 0],
                                            0, self.dim // 16, 1, 0, self.queries_align - 1)

            self.tik_instance.vmul(self.dim, queries_ub, queries_ub,
                                   diff_ub, self.query_each_loop, 1, 1, 1,
                                   self.dim // 16, self.dim // 16, 0)

            # use data_move function to make x*d align 16
            xd_transfer_ub = self.tik_instance.Tensor("float16", (self.dim // 16, self.queries_align, 16),
                                                      name="xd_transfer_ub", scope=tik.scope_ubuf)
            with self.tik_instance.for_range(0, self.query_each_loop) as i:
                self.tik_instance.data_move(xd_transfer_ub[0, i, 0],
                                            queries_ub[i, 0], 0, self.dim // 16,
                                            1, 0, self.queries_align - 1)

            # move x from ub to L1
            queries_l1 = self.tik_instance.Tensor("float16", (self.dim // 16, self.queries_align, 16),
                                                  name="queries_l1", scope=tik.scope_cbuf)
            self.tik_instance.data_move(queries_l1, queries_transfer_ub, 0, 1,
                                        self.queries_align * self.dim // 16, 0, 0)

            # move x*d from UB to L1
            self.tik_instance.data_move(data_xd_l1, xd_transfer_ub, 0, 1,
                                        self.queries_align * self.dim // 16, 0, 0)

            data_xdm_l0c = self.tik_instance.Tensor("float32",
                                                    (self.queries_align // 16, 16, 16),
                                                    name="data_xdm_l0c", scope=tik.scope_cbuf_out)
            # cal x * (0.5 * d + 255 * m)
            self.tik_instance.matmul(data_xdm_l0c, data_dm_l1, queries_l1, 16,
                                     self.dim, self.queries_align)

            data_xdm_ub = self.tik_instance.Tensor("float32",
                                                   (self.queries_align // 16, 16, 16),
                                                   name="data_xdm_ub", scope=tik.scope_ubuf)
            # move data from l0 to ub
            self.tik_instance.data_move(data_xdm_ub, data_xdm_l0c, 0, 1, self.queries_align // 16, 0, 0)
            data_xdm_align_ub = self.tik_instance.Tensor("float32", (self.queries_align,),
                                                         name="data_xdm_align_ub", scope=tik.scope_ubuf)
            # 1 / 255 * x * (0.5 * d + 255 * m)
            self.tik_instance.data_move(data_xdm_align_ub, data_xdm_ub, 0, self.queries_align // 16, 2, 30, 0)

            # set as scalar for next cal
            for i in range(self.query_each_loop):
                add_scalar_list[i].set_as(data_xdm_align_ub[i])

        code_loop_times = (aicore_code_num + self.code_each_loop - 1) // self.code_each_loop

        inner_product_l0c = self.tik_instance.Tensor("float32",
                                                     (self.code_each_loop // 16, self.queries_align, 16),
                                                     name="inner_product_l0c", scope=tik.scope_cbuf_out)
        # start for mask filter
        min_val_ub = self.tik_instance.Tensor("float16", (128,), name="min_val_ub", scope=tik.scope_ubuf)
        self.tik_instance.vec_dup(self.fp16_mask, min_val_ub, -65504, 1, 8)

        # compute xy using cube
        with self.tik_instance.new_stmt_scope():
            self.cube_compute_first(inner_product_l0c, data_xd_l1, aicore_offset)

        with self.tik_instance.for_range(1, code_loop_times) as loop_code:
            self.cube_compute_each_loop(inner_product_l0c, data_xd_l1, add_scalar_list,
                                        query_offset, aicore_offset, loop_code * self.code_each_loop, min_val_ub)

        with self.tik_instance.new_stmt_scope():
            code_last = aicore_code_num % self.code_each_loop
            self.cube_compute_last(inner_product_l0c, add_scalar_list, query_offset, aicore_offset,
                                   (code_loop_times - 1) * self.code_each_loop, code_last, min_val_ub)

    def cube_compute_first(self, inner_product_l0c, data_xd_l1, aicore_offset):
        with self.tik_instance.for_range(0, self.code_loops, thread_num=2) as loop_ub:
            code_ub_fp16 = self.tik_instance.Tensor("float16",
                                                    (self.dim // 16, self.code_sub_loop // 16, 16, 16),
                                                    name="code_ub_fp16", scope=tik.scope_ubuf)
            # move y from out to UB
            code_ub = self.tik_instance.Tensor("uint8",
                                               (self.dim // 16, self.code_sub_loop // 16, 16, 16),
                                               name="code_ub", scope=tik.scope_ubuf)
            # y do conv from uint8 to fp16
            vconv_repeats = (self.code_sub_loop * 16) // self.fp16_mask
            for i in range(self.dim // 16):
                self.tik_instance.data_move(code_ub[i, 0, 0, 0],
                                            self.input_code_gm[(aicore_offset
                                                                + loop_ub * self.code_sub_loop) // 16,
                                                               i, 0, 0],
                                            0, self.code_sub_loop // 16, 8, self.dim // 2 - 8, 0)
                self.tik_instance.vconv(self.fp16_mask, "none",
                                        code_ub_fp16[i, 0, 0, 0], code_ub[i, 0, 0, 0],
                                        vconv_repeats, 1, 1, 8, 4)

            # move y from ub to L1
            code_l1 = self.tik_instance.Tensor("float16", (self.dim // 16, self.code_sub_loop, 16),
                                               name="code_l1", scope=tik.scope_cbuf)

            self.tik_instance.data_move(code_l1, code_ub_fp16, 0, 1, self.dim * self.code_sub_loop // 16, 0, 0)

            # cal (x*d)*y
            self.tik_instance.matmul(inner_product_l0c[(loop_ub * self.code_sub_loop * self.queries_align):],
                                     data_xd_l1, code_l1, self.queries_align, self.dim, self.code_sub_loop)

    def cube_compute_each_loop(self, inner_product_l0c, data_xd_l1, add_scalar_list,
                               query_offset, aicore_offset, code_offset, min_val_ub):
        add_ub = self.tik_instance.Tensor("float32",
                                          (self.query_each_loop, self.code_each_loop),
                                          name="add_ub", scope=tik.scope_ubuf)
        dst_ub = self.tik_instance.Tensor("float16",
                                          (self.query_each_loop, self.code_each_loop),
                                          name="dst_ub", scope=tik.scope_ubuf)
        dst_max_ub = self.tik_instance.Tensor("float16",
                                              (self.query_each_loop, self.maxs_each_loops),
                                              name="dst_max_ub", scope=tik.scope_ubuf)

        with self.tik_instance.for_range(0, self.code_loops, thread_num=2) as loop_ub:
            code_ub_fp16 = self.tik_instance.Tensor("float16", (self.dim // 16, self.code_sub_loop // 16, 16, 16),
                                                    name="code_ub_fp16", scope=tik.scope_ubuf)
            # move y from out to UB
            code_ub = self.tik_instance.Tensor("uint8", (self.dim // 16, self.code_sub_loop // 16, 16, 16),
                                               name="code_ub", scope=tik.scope_ubuf)

            # y do conv from uint8 to fp16
            vconv_repeats = (self.code_sub_loop * 16) // self.fp16_mask
            for i in range(self.dim // 16):
                self.tik_instance.data_move(code_ub[i, 0, 0, 0],
                                            self.input_code_gm[(aicore_offset + code_offset
                                                                + loop_ub * self.code_sub_loop) // 16,
                                                               i, 0, 0],
                                            0, self.code_sub_loop // 16, 8, self.dim // 2 - 8, 0)

                self.tik_instance.vconv(self.fp16_mask, "none",
                                        code_ub_fp16[i, 0, 0, 0], code_ub[i, 0, 0, 0],
                                        vconv_repeats, 1, 1, 8, 4)

            # move y from ub to L1
            code_l1 = self.tik_instance.Tensor("float16",
                                               (self.dim // 16, self.code_sub_loop, 16),
                                               name="code_l1", scope=tik.scope_cbuf)
            self.tik_instance.data_move(code_l1, code_ub_fp16, 0, 1, self.dim * self.code_sub_loop // 16, 0, 0)

            if self.queries_num == 1:
                inner_product_ub = self.tik_instance.Tensor("float32",
                                                            (self.code_sub_loop // 32, self.queries_align, 16),
                                                            name="inner_product_ub", scope=tik.scope_ubuf)

                # move (x*d)*y from l0 to ub
                self.tik_instance.data_move(inner_product_ub,
                                            inner_product_l0c[loop_ub * self.code_sub_loop // 16, 0, 0],
                                            0, 1, self.code_sub_loop * self.queries_align // 512, 0, 0)

                # cal x*(0.5*d+255*m)+(x*d)*y
                self.tik_instance.vadds(16,
                                        add_ub[loop_ub * self.code_sub_loop],
                                        inner_product_ub, add_scalar_list[0],
                                        self.code_sub_loop // 32,
                                        1, 1, 2, self.queries_align * 2)

                # move (x*d)*y from l0 to ub
                self.tik_instance.data_move(inner_product_ub,
                                            inner_product_l0c[
                                                loop_ub * self.code_sub_loop // 16 + self.code_sub_loop // 32, 0, 0],
                                            0, 1, self.code_sub_loop * self.queries_align // 512, 0, 0)

                # cal x*(0.5*d+255*m)+(x*d)*y
                self.tik_instance.vadds(16,
                                        add_ub[loop_ub * self.code_sub_loop + self.code_sub_loop // 2],
                                        inner_product_ub, add_scalar_list[0],
                                        self.code_sub_loop // 32,
                                        1, 1, 2, self.queries_align * 2)
            else:
                inner_product_ub = self.tik_instance.Tensor("float32",
                                                            (self.code_sub_loop // 16, self.queries_align, 16),
                                                            name="inner_product_ub", scope=tik.scope_ubuf)

                # move (x*d)*y from l0 to ub
                self.tik_instance.data_move(inner_product_ub,
                                            inner_product_l0c[loop_ub * self.code_sub_loop // 16, 0, 0],
                                            0, 1, self.code_sub_loop * self.queries_align // 256, 0, 0)

                for i in range(self.query_each_loop):
                    # cal x*(0.5*d+255*m)+(x*d)*y
                    self.tik_instance.vadds(16,
                                            add_ub[i, loop_ub * self.code_sub_loop],
                                            inner_product_ub[0, i, 0], add_scalar_list[i],
                                            self.code_sub_loop // 16,
                                            1, 1, 2, self.queries_align * 2)
            # cal (x multiply d) multiply y
            self.tik_instance.matmul(inner_product_l0c[(loop_ub * self.code_sub_loop * self.queries_align):],
                                     data_xd_l1, code_l1,
                                     self.queries_align, self.dim, self.code_sub_loop)

        vconv_repeats = self.query_each_loop * self.code_each_loop // self.fp32_mask
        vcmax_repeats = self.query_each_loop * self.code_each_loop // self.max_batch

        sel_ub = self.tik_instance.Tensor("uint8", (self.query_each_loop, self.code_each_loop // 8),
                                          name="sel_ub", scope=tik.scope_ubuf)
        with self.tik_instance.for_range(0, self.query_each_loop, dtype="int64") as loop_query:
            self.tik_instance.data_move(
                sel_ub[loop_query, 0],
                self.input_mask_gm[(loop_query + query_offset) * self.mask_len +
                                   (self.mask_offset + aicore_offset
                                   + code_offset - self.code_each_loop) // 8],
                0, 1, (self.code_each_loop + 255) // 256, 8, 8)
        vsel_loop = self.code_each_loop // self.fp16_mask
        if self.query_each_loop < 16:
            self.tik_instance.vconv(self.fp32_mask, "none",
                                    dst_ub, add_ub,
                                    vconv_repeats, 1, 1, 4, 8)
        else:
            for i in range(vconv_repeats // self.repeat_times):
                self.tik_instance.vconv(self.fp32_mask, "none",
                                        dst_ub[(i * self.repeat_times * self.fp32_mask):],
                                        add_ub[(i * self.repeat_times * self.fp32_mask):],
                                        self.repeat_times, 1, 1, 4, 8)

        with self.tik_instance.new_stmt_scope(disable_sync=True):
            with self.tik_instance.for_range(0, self.query_each_loop) as loop_query:
                for vloop in range(vsel_loop):
                    # sel_ub can not use repeat times > 1, use for + offset
                    voffset = vloop * self.fp16_mask
                    self.tik_instance.vec_sel(self.fp16_mask, 0, dst_ub[loop_query, voffset],
                                              sel_ub[loop_query, voffset // 8],
                                              dst_ub[loop_query, voffset],
                                              min_val_ub, 1, 8, 8, 0)
        if self.query_each_loop < 16:
            # end for mask filter
            self.tik_instance.vcmax(self.max_batch,
                                    dst_max_ub, dst_ub,
                                    vcmax_repeats, 1, 1, self.max_batch // 16)
        else:
            # end for mask filter
            for i in range(vcmax_repeats // self.repeat_times):
                self.tik_instance.vcmax(self.max_batch,
                                        dst_max_ub[(i * self.repeat_times * 2):],
                                        dst_ub[(i * self.repeat_times * self.fp32_mask):],
                                        self.repeat_times, 1, 1, self.max_batch // 16)

        self.tik_instance.data_move(
            self.output_distances_gm[query_offset, aicore_offset + code_offset - self.code_each_loop],
            dst_ub,
            0, self.query_each_loop, self.code_each_loop // 16,
            0, (self.code_num - self.code_each_loop) // 16)

        self.tik_instance.data_move(
            self.output_maxs_gm[query_offset,
                                (aicore_offset + code_offset - self.code_each_loop)
                                // self.max_batch * 2],
            dst_max_ub,
            0, self.query_each_loop, (self.code_each_loop // self.max_batch) // 8,
            0, (self.code_num - self.code_each_loop) // self.max_batch // 8)

    def cube_compute_last(self, inner_product_l0c, add_scalar_list, query_offset, aicore_offset, code_offset, code_num,
                          min_val_ub):
        inner_product_ub = self.tik_instance.Tensor("float32",
                                                    (self.code_each_loop // 16, self.queries_align, 16),
                                                    name="inner_product_ub", scope=tik.scope_ubuf)
        add_ub = self.tik_instance.Tensor("float32",
                                          (self.query_each_loop, self.code_each_loop),
                                          name="add_ub", scope=tik.scope_ubuf)
        dst_ub = self.tik_instance.Tensor("float16",
                                          (self.query_each_loop, self.code_each_loop),
                                          name="dst_ub", scope=tik.scope_ubuf)
        dst_max_ub = self.tik_instance.Tensor("float16",
                                              (self.query_each_loop, self.maxs_each_loops),
                                              name="dst_max_ub", scope=tik.scope_ubuf)

        self.tik_instance.data_move(inner_product_ub, inner_product_l0c,
                                    0, 1, self.code_each_loop * self.queries_align // 256, 0, 0)

        sel_ub = self.tik_instance.Tensor("uint8",
                                          (self.query_each_loop, self.code_each_loop // 8),
                                          name="sel_ub", scope=tik.scope_ubuf)
        with self.tik_instance.for_range(0, self.query_each_loop, dtype="int64") as loop_query:
                self.tik_instance.data_move(
                    sel_ub[loop_query, 0],
                    self.input_mask_gm[(loop_query + query_offset) * self.mask_len +
                                       (self.mask_offset + aicore_offset + code_offset) // 8],
                    0, 1, (self.code_each_loop + 255) // 256, 8, 8)

        vsel_loop = self.code_each_loop // self.fp16_mask
        with self.tik_instance.if_scope(code_num == 0):
            for i in range(self.query_each_loop):
                # cal x*(0.5*d+255*m)+(x*d)*y
                self.tik_instance.vadds(16,
                                        add_ub[i, 0],
                                        inner_product_ub[0, i, 0],
                                        add_scalar_list[i],
                                        self.code_each_loop // 16, 1, 1, 2, self.queries_align * 2)
            vconv_repeats = self.query_each_loop * self.code_each_loop // self.fp32_mask

            if self.query_each_loop < 16:
                self.tik_instance.vconv(self.fp32_mask, "none",
                                        dst_ub, add_ub,
                                        vconv_repeats, 1, 1, 4, 8)
            else:
                for i in range(vconv_repeats // self.repeat_times):
                    self.tik_instance.vconv(self.fp32_mask, "none",
                                            dst_ub[(i * self.fp32_mask * self.repeat_times):],
                                            add_ub[(i * self.fp32_mask * self.repeat_times):],
                                            self.repeat_times, 1, 1, 4, 8)

            with self.tik_instance.new_stmt_scope(disable_sync=True):
                with self.tik_instance.for_range(0, self.query_each_loop) as loop_query:
                    for vloop in range(vsel_loop):
                        # sel_ub can not use repeat times > 1, use for + offset
                        voffset = vloop * self.fp16_mask
                        self.tik_instance.vec_sel(self.fp16_mask, 0, dst_ub[loop_query, voffset],
                                                  sel_ub[loop_query, voffset // 8],
                                                  dst_ub[loop_query, voffset],
                                                  min_val_ub, 1, 8, 8, 0)

            # end for mask filter
            vcmax_repeats = self.query_each_loop * self.code_each_loop // self.max_batch
            if self.query_each_loop < 16:
                self.tik_instance.vcmax(self.max_batch,
                                        dst_max_ub, dst_ub,
                                        vcmax_repeats, 1, 1, self.max_batch // 16)
            else:
                for i in range(vcmax_repeats // self.repeat_times):
                    self.tik_instance.vcmax(self.max_batch,
                                            dst_max_ub[(i * self.repeat_times * 2):],
                                            dst_ub[(i * self.repeat_times * self.fp32_mask):],
                                            self.repeat_times, 1, 1, self.max_batch // 16)

            self.tik_instance.data_move(
                self.output_distances_gm[query_offset, aicore_offset + code_offset],
                dst_ub,
                0, self.query_each_loop, self.code_each_loop // 16,
                0, (self.code_num - self.code_each_loop) // 16)
        with self.tik_instance.else_scope():
            code_align = (code_num + 15) // 16 * 16
            for i in range(self.query_each_loop):
                # cal x*(0.5*d+255*m)+(x*d)*y
                self.tik_instance.vadds(16,
                                        add_ub[i, 0],
                                        inner_product_ub[0, i, 0],
                                        add_scalar_list[i],
                                        self.code_each_loop // 16, 1, 1, 2, self.queries_align * 2)
            vconv_repeats = self.query_each_loop * self.code_each_loop // self.fp32_mask
            if self.query_each_loop < 16:
                self.tik_instance.vconv(self.fp32_mask, "none",
                                        dst_ub, add_ub,
                                        vconv_repeats, 1, 1, 4, 8)
            else:
                for i in range(vconv_repeats // self.repeat_times):
                    self.tik_instance.vconv(self.fp32_mask, "none",
                                            dst_ub[(i * self.fp32_mask * self.repeat_times):],
                                            add_ub[(i * self.fp32_mask * self.repeat_times):],
                                            self.repeat_times, 1, 1, 4, 8)

            with self.tik_instance.new_stmt_scope(disable_sync=True):
                with self.tik_instance.for_range(0, self.query_each_loop) as loop_query:
                    for vloop in range(vsel_loop):
                        # sel_ub can not use repeat times > 1, use for + offset
                        voffset = vloop * self.fp16_mask
                        self.tik_instance.vec_sel(self.fp16_mask, 0, dst_ub[loop_query, voffset],
                                                  sel_ub[loop_query, voffset // 8],
                                                  dst_ub[loop_query, voffset],
                                                  min_val_ub, 1, 8, 8, 0)

            # end for mask filter
            vcmax_repeats = code_num // self.max_batch
            with self.tik_instance.if_scope(vcmax_repeats > 0):
                for i in range(self.query_each_loop):
                    self.tik_instance.vcmax(self.max_batch,
                                            dst_max_ub[i, 0], dst_ub[i, 0],
                                            vcmax_repeats, 1, 1, self.max_batch // 16)
            last_mask = code_num % self.max_batch
            with self.tik_instance.if_scope(last_mask > 0):
                for i in range(self.query_each_loop):
                    self.tik_instance.vcmax(last_mask,
                                            dst_max_ub[i, vcmax_repeats * 2],
                                            dst_ub[i, vcmax_repeats * self.max_batch],
                                            1, 1, 1, self.max_batch // 16)
            self.tik_instance.data_move(
                self.output_distances_gm[query_offset, aicore_offset + code_offset],
                dst_ub,
                0, self.query_each_loop, code_align // 16,
                                         (self.code_each_loop - code_align) // 16, (self.code_num - code_align) // 16)

        self.tik_instance.data_move(
            self.output_maxs_gm[query_offset, (aicore_offset + code_offset) // self.max_batch * 2],
            dst_max_ub,
            0, self.query_each_loop, (self.code_each_loop // self.max_batch) // 8,
            0, (self.code_num - self.code_each_loop) // self.max_batch // 8)

    def distance_compute(self):
        with self.tik_instance.for_range(0, self.aicore_use, block_num=self.aicore_use) as block_index:
            self.cal_num_each_core()
            # compute coarse centroids num and move offset every core
            aicore_code_num = self.tik_instance.Scalar(dtype="uint32", name="aicore_code_num")
            with self.tik_instance.if_scope(block_index != self.aicore_use - 1):
                aicore_code_num.set_as(self.code_num_each_core)
            with self.tik_instance.else_scope():
                aicore_code_num.set_as(self.code_num_last_core)

            with self.tik_instance.if_scope(aicore_code_num > 0):
                with self.tik_instance.for_range(0, self.query_loops) as loop_query:
                    self.distance_compute_each_loop(block_index * self.code_num_each_core, aicore_code_num,
                                                    loop_query * self.query_each_loop)

            one = self.tik_instance.Scalar(dtype="uint16", name="one", init_value=1)
            flag_ub = self.tik_instance.Tensor("uint16", (16,), name="flag_ub", scope=tik.scope_ubuf)
            flag_ub[0].set_as(one)
            self.tik_instance.pipe_barrier("PIPE_MTE3")
            self.tik_instance.data_move(self.output_flag_gm[block_index, 0], flag_ub, 0, 1, 1, 0, 0)

    def get_tik_instance(self):
        """
        obtain tik instance
        """
        self.distance_compute()

        self.tik_instance.BuildCCE(
            kernel_name=self.kernel_name,
            inputs=[
                self.input_queries_gm, self.input_mask_gm, self.input_code_gm, self.input_diff_gm,
                self.input_min_gm, self.input_actual_num_gm
            ],
            outputs=[self.output_distances_gm, self.output_maxs_gm, self.output_flag_gm],
            config={"tbe_debug_level": 1})

        return self.tik_instance

    def _muls(self, dst, src, scalar, compute_num, max_mask):
        # process 256B data per repeat for vmuls
        repeat = compute_num // max_mask
        offset = 0
        if repeat > 0:
            self.tik_instance.vmuls(max_mask, dst[0], src[0], scalar,
                                    repeat, 1, 1, 8, 8)
            offset += repeat * max_mask

        remain = compute_num % max_mask
        if remain > 0:
            self.tik_instance.vmuls(remain, dst[offset], src[offset], scalar,
                                    1, 1, 1, 8, 8)

    def _add(self, dst, src0, src1, compute_num, max_mask):
        # process 256B data per repeat for vadd
        repeat = compute_num // max_mask
        offset = 0
        if repeat > 0:
            self.tik_instance.vadd(max_mask, dst[offset], src0[offset],
                                   src1[offset], repeat, 1, 1, 1, 8, 8, 8)
            offset += repeat * max_mask

        remain = compute_num % max_mask
        if remain > 0:
            self.tik_instance.vadd(remain, dst[offset], src0[offset],
                                   src1[offset], 1, 1, 1, 1, 8, 8, 8)


def distance_masked_sq8_ip_maxs_dim64(input_queries,
                                      input_mask,
                                      input_code,
                                      input_diff,
                                      input_min,
                                      input_actual_num,
                                      output_distances,
                                      output_maxs,
                                      output_flag,
                                      kernel_name="distance_masked_sq8_ip_maxs_dim64"):
    """
    calculating distance

    Parameters
    ----------
    input_queries : dict
        shape and dtype of query vector
    input_mask: dict
        shape and dtype of query mask
    input_code : dict
        shape and dtype of coding vector
    input_diff: dict
        shape and dtype of coding coefficient
    input_min: dict
        shape and dtype of coding offset
    input_actual_num: dict
        shape and dtype of actual code num,
        shape must be (8,) and dtype must be uint32
    output_distances : dict
        shape and dtype of distances, should be same dtype as input_queries
    output_maxs : dict
        shape and dtype of maxs
    output_flag: dict
        shape and dtype of output flag,
        shape must be (32,) and dtype must be uint16
    kernel_name : str
        kernel name, default value is "distance_masked_sq8_ip_maxs_dim64"

    Returns
    -------
    None
    """
    distance_sq8 = DistanceSQ8IPMaxs(input_queries, input_mask, input_code, input_diff, input_min, input_actual_num,
                                     output_distances, output_maxs, output_flag, kernel_name)
    tik_instance = distance_sq8.get_tik_instance()
    return tik_instance
