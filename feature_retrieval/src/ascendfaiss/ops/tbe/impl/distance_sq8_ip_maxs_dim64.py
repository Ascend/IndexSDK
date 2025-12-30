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
        self.coeff = self.tik_instance.Scalar("float32",
                                              name="coeff",
                                              init_value=1 / 255)
        self.default_scalar = self.tik_instance.Scalar("float16",
                                                       name="default_scalar",
                                                       init_value=-65504)

        # creat input tensor: input_queries_gm, input_code_gm,
        # input_diff_gm, input_min_gm,
        # input_actual_num_gm
        # and creat output tensor: output_distances_gm
        # and output_flag_gm in global buffer
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
        actual_code_num_ub = self.tik_instance.Tensor(
            "uint32", (8,), name="actual_code_num_ub", scope=tik.scope_ubuf)
        self.tik_instance.data_move(actual_code_num_ub,
                                    self.input_actual_num_gm, 0, 1, 1, 0,
                                    0)
        actual_code_num = self.tik_instance.Scalar(dtype="uint32", name="actual_code_num", init_value=0)
        actual_code_num.set_as(actual_code_num_ub[0])

        code_size = self.tik_instance.Scalar(dtype="uint32", name="code_size", init_value=0)
        code_size.set_as(actual_code_num_ub[3])

        if self.aicore_use == 2:
            self.code_num_each_core = ((code_size + 2048) // self.aicore_use + self.code_each_loop // 2) \
                // self.code_each_loop * self.code_each_loop
        else:
            self.code_num_each_core = \
                (code_size + 2048) // self.aicore_use // self.code_each_loop * self.code_each_loop

        self.code_num_last_core = code_size - (self.aicore_use - 1) * self.code_num_each_core

    def distance_compute_each_loop(self, aicore_offset, aicore_code_num, query_offset):
        add_scalar_list = [self.tik_instance.Scalar(dtype="float32") for i in range(self.query_each_loop)]
        data_xd_l1 = self.tik_instance.Tensor("float16",
                                              (self.dim // 16, self.queries_align, 16),
                                              name="data_xd_l1", scope=tik.scope_cbuf)

        with self.tik_instance.new_stmt_scope():
            queries_ub = self.tik_instance.Tensor("float16",
                                                  (self.query_each_loop, self.dim),
                                                  name="queries_ub", scope=tik.scope_ubuf)
            diff_ub = self.tik_instance.Tensor("float16",
                                               (self.dim,),
                                               name="diff_ub", scope=tik.scope_ubuf)
            min_ub = self.tik_instance.Tensor("float16",
                                              (self.dim,),
                                              name="min_ub", scope=tik.scope_ubuf)

            # move diff adn min from out to UB
            self.tik_instance.data_move(diff_ub, self.input_diff_gm, 0, 1, self.dim // 16, 0, 0)
            self.tik_instance.data_move(min_ub, self.input_min_gm, 0, 1, self.dim // 16, 0, 0)
            diff2_ub = self.tik_instance.Tensor("float16",
                                                (self.dim,),
                                                name="diff2_ub", scope=tik.scope_ubuf)
            # cal 0.5 * d
            self._muls(diff2_ub, diff_ub, 0.5, self.dim, self.fp16_mask)
            # cal 255 * m
            self._muls(min_ub, min_ub, 255.0, self.dim, self.fp16_mask)
            # cal 0.5 * d + 255 * m
            self._add(diff2_ub, diff2_ub, min_ub, self.dim, self.fp16_mask)
            dm_transfer_ub = self.tik_instance.Tensor("float16",
                                                      (self.dim // 16, 16, 16),
                                                      name="dm_transfer_ub", scope=tik.scope_ubuf)
            # use data_move function to make 0.5 * d + 255 * m align 16
            self.tik_instance.data_move(dm_transfer_ub, diff2_ub, 0, self.dim // 16, 1, 0, 15)
            # move d+m from UB to L1
            data_dm_l1 = self.tik_instance.Tensor("float16",
                                                  (self.dim // 16, 16, 16),
                                                  name="data_dm_l1", scope=tik.scope_cbuf)
            self.tik_instance.data_move(data_dm_l1, dm_transfer_ub, 0, 1, self.dim, 0, 0)

            # move x from out to UB
            self.tik_instance.data_move(queries_ub,
                                        self.input_queries_gm[query_offset, 0],
                                        0, 1, self.query_each_loop * self.dim // 16, 0, 0)
            vmuls_mask = min(self.dim, self.fp16_mask)
            self.tik_instance.vmuls(vmuls_mask, queries_ub, queries_ub,
                                    1 / 255, self.query_each_loop * self.dim // vmuls_mask,
                                    1, 1, vmuls_mask // 16, vmuls_mask // 16)
            queries_transfer_ub = self.tik_instance.Tensor("float16",
                                                           (self.dim // 16, self.queries_align, 16),
                                                           name="queries_transfer_ub", scope=tik.scope_ubuf)
            # use data_move function to make x align 16
            with self.tik_instance.for_range(0, self.query_each_loop) as i:
                self.tik_instance.data_move(queries_transfer_ub[0, i, 0],
                                            queries_ub[i, 0],
                                            0, self.dim // 16, 1, 0, self.queries_align - 1)

            if self.dim <= 128:
                self.tik_instance.vmul(self.dim, queries_ub, queries_ub,
                                       diff_ub, self.query_each_loop, 1, 1, 1,
                                       self.dim // 16, self.dim // 16, 0)
            else:
                with self.tik_instance.for_range(0, self.query_each_loop) as i:
                    repeat = self.dim // self.fp16_mask
                    offset = 0
                    if repeat > 0:
                        self.tik_instance.vmul(self.fp16_mask,
                                               queries_ub[i, offset],
                                               queries_ub[i, offset],
                                               diff_ub[offset], repeat, 1, 1,
                                               1, 8, 8, 8)
                        offset += repeat * self.fp16_mask

                    remain = self.dim % self.fp16_mask
                    if remain > 0:
                        self.tik_instance.vmul(remain, queries_ub[i, offset],
                                               queries_ub[i, offset],
                                               diff_ub[offset], 1, 1, 1, 1, 8,
                                               8, 8)

            # use data_move function to make x*d align 16
            xd_transfer_ub = self.tik_instance.Tensor("float16",
                                                      (self.dim // 16, self.queries_align, 16),
                                                      name="xd_transfer_ub", scope=tik.scope_ubuf)
            with self.tik_instance.for_range(0, self.query_each_loop) as i:
                self.tik_instance.data_move(xd_transfer_ub[0, i, 0],
                                            queries_ub[i, 0], 0, self.dim // 16,
                                            1, 0, self.queries_align - 1)

            # move x from ub to L1
            queries_l1 = self.tik_instance.Tensor("float16",
                                                  (self.dim // 16, self.queries_align, 16),
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
            data_xdm_align_ub = self.tik_instance.Tensor("float32",
                                                         (self.queries_align,),
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

        # compute xy using cube
        with self.tik_instance.if_scope(code_loop_times >= 1):
            with self.tik_instance.for_range(0, code_loop_times - 1) as loop_code:
                self.cube_compute_each_loop(inner_product_l0c, data_xd_l1, add_scalar_list,
                                            query_offset, aicore_offset, loop_code * self.code_each_loop,
                                            self.code_each_loop)
            self.cube_compute_each_loop(inner_product_l0c, data_xd_l1, add_scalar_list,
                                        query_offset, aicore_offset, (code_loop_times - 1) * self.code_each_loop,
                                        self.code_each_loop)

        with self.tik_instance.if_scope(code_loop_times > 0):
            code_last = aicore_code_num % self.code_each_loop
            self.cube_compute_last(inner_product_l0c, add_scalar_list, query_offset, aicore_offset,
                                   (code_loop_times - 1) * self.code_each_loop, code_last)

    def cube_compute_each_loop(self, inner_product_l0c, data_xd_l1, add_scalar_list,
                               query_offset, aicore_offset, code_offset, code_num):
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
            code_ub_fp16 = self.tik_instance.Tensor("float16",
                                                    (self.dim // 16, self.code_sub_loop // 16, 16, 16),
                                                    name="code_ub_fp16", scope=tik.scope_ubuf)
            # move y from out to UB
            code_ub = self.tik_instance.Tensor("uint8",
                                               (self.dim // 16, self.code_sub_loop // 16, 16, 16),
                                               name="code_ub", scope=tik.scope_ubuf)

            with self.tik_instance.for_range(0, self.dim // 16) as i:
                self.tik_instance.data_move(code_ub[i, 0, 0, 0],
                                            self.input_code_gm[(aicore_offset + code_offset
                                                                + code_num - self.code_each_loop
                                                                + loop_ub * self.code_sub_loop) // 16,
                                                               i, 0, 0],
                                            0, self.code_sub_loop // 16, 8, self.dim // 2 - 8, 0)

            # y do conv from uint8 to fp16
            vconv_repeats = (self.code_sub_loop * self.dim) // self.fp16_mask
            self.tik_instance.vconv(self.fp16_mask, "none",
                                    code_ub_fp16, code_ub,
                                    vconv_repeats, 1, 1, 8, 4)

            # move y from ub to L1
            code_l1 = self.tik_instance.Tensor("float16",
                                               (self.dim // 16, self.code_sub_loop, 16),
                                               name="code_l1", scope=tik.scope_cbuf)
            self.tik_instance.data_move(code_l1, code_ub_fp16, 0, self.dim // 16, self.code_sub_loop, 0, 0)

            with self.tik_instance.if_scope(code_offset > 0):
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

                with self.tik_instance.if_scope(loop_ub == self.code_loops - 1):
                    vconv_repeats = self.query_each_loop * self.code_each_loop // self.fp32_mask
                    vcmax_repeats = self.query_each_loop * self.code_each_loop // self.max_batch
                    if self.query_each_loop < 16:
                        self.tik_instance.vconv(self.fp32_mask, "none",
                                                dst_ub, add_ub,
                                                vconv_repeats, 1, 1, 4, 8)
                        self.tik_instance.vcmax(self.max_batch,
                                                dst_max_ub, dst_ub,
                                                vcmax_repeats, 1, 1, self.max_batch // 16)
                    else:
                        for i in range(vconv_repeats // self.repeat_times):
                            self.tik_instance.vconv(self.fp32_mask, "none",
                                                    dst_ub[(i * self.repeat_times * self.fp32_mask):],
                                                    add_ub[(i * self.repeat_times * self.fp32_mask):],
                                                    self.repeat_times, 1, 1, 4, 8)

                        for i in range(vcmax_repeats // self.repeat_times):
                            self.tik_instance.vcmax(self.max_batch,
                                                    dst_max_ub[(i * self.repeat_times * 2):],
                                                    dst_ub[(i * self.repeat_times * self.fp32_mask):],
                                                    self.repeat_times, 1, 1, self.max_batch // 16)

            # # cal (x*d)*y
            self.tik_instance.matmul(inner_product_l0c[(loop_ub * self.code_sub_loop * self.queries_align):],
                                     data_xd_l1, code_l1,
                                     self.queries_align, self.dim, self.code_sub_loop)

        with self.tik_instance.if_scope(code_offset > 0):
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

    def cube_compute_last(self, inner_product_l0c, add_scalar_list, query_offset, aicore_offset, code_offset, code_num):
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

        self.tik_instance.data_move(inner_product_ub,
                                    inner_product_l0c,
                                    0, 1, self.code_each_loop * self.queries_align // 256, 0, 0)

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

            self.tik_instance.data_move(
                self.output_distances_gm[query_offset, aicore_offset + code_offset],
                dst_ub,
                0, self.query_each_loop, self.code_each_loop // 16,
                0, (self.code_num - self.code_each_loop) // 16)

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
            aicore_code_num = self.tik_instance.Scalar(dtype="uint32",
                                                       name="aicore_code_num",
                                                       init_value=0)
            with self.tik_instance.if_scope(block_index != self.aicore_use - 1):
                aicore_code_num.set_as(self.code_num_each_core)
            with self.tik_instance.else_scope():
                aicore_code_num.set_as(self.code_num_last_core)

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
                self.input_queries_gm, self.input_code_gm, self.input_diff_gm,
                self.input_min_gm, self.input_actual_num_gm
            ],
            outputs=[self.output_distances_gm, self.output_maxs_gm, self.output_flag_gm])

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


def distance_sq8_ip_maxs_dim64(input_queries,
                               input_code,
                               input_diff,
                               input_min,
                               input_actual_num,
                               output_distances,
                               output_maxs,
                               output_flag,
                               kernel_name="distance_sq8_ip_maxs_dim64"):
    """
    calculating distance

    Parameters
    ----------
    input_queries : dict
        shape and dtype of query vector
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
        kernel name, default value is "distance_compute"

    Returns
    -------
    None
    """
    distance_sq8 = DistanceSQ8IPMaxs(input_queries, input_code, input_diff,
                                     input_min, input_actual_num,
                                     output_distances, output_maxs, output_flag,
                                     kernel_name)
    tik_instance = distance_sq8.get_tik_instance()
    return tik_instance
