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


class DistanceIVFSQ8IP4:
    def __init__(self,
                 input_queries,
                 input_code0,
                 input_code1,
                 input_code2,
                 input_code3,
                 input_dm,
                 input_actual_num,
                 output_distances,
                 output_maxs,
                 output_flag,
                 kernel_name="distance_ivf_sq8_ip4"):
        self.shape_queries = input_queries.get("shape")
        self.dtype_queries = input_queries.get("dtype")
        self.shape_code0 = input_code0.get("shape")
        self.dtype_code0 = input_code0.get("dtype")
        self.shape_code1 = input_code1.get("shape")
        self.dtype_code1 = input_code1.get("dtype")
        self.shape_code2 = input_code2.get("shape")
        self.dtype_code2 = input_code2.get("dtype")
        self.shape_code3 = input_code3.get("shape")
        self.dtype_code3 = input_code3.get("dtype")
        self.shape_dm = input_dm.get("shape")
        self.dtype_dm = input_dm.get("dtype")
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
        self.code_num = self.shape_code0[0] * 16

        # check parameter
        self.check_parameter()

        # set vector fp32 mask and fp16 mask
        self.code_list_num = 4
        self.sub_list_num = self.code_list_num // 2
        self.fp32_mask = 64
        self.fp16_mask = 128
        self.max_batch = 32 // (1 + (self.dim + 255) // 512)
        self.queries_align = 16
        self.cube_align = 16
        self.repeat_times = 128

        # set tik instance
        self.set_tik_instance()

    def check_parameter(self):
        # check shape and dtype of input
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

        self.queries_num_each_loop = min(self.queries_num, 48)
        self.code_each_loops = 2
        if self.dim == 64:
            self.code_each_loop = 512
        else:
            self.code_each_loop = 512 // (1 << ((self.dim + 255) // 256))
        self.code_sub_loop = self.code_each_loop // self.code_each_loops
        self.maxs_each_loop = self.code_each_loop // self.max_batch * 2
        self.maxs_sub_loop = self.code_sub_loop // self.max_batch * 2

        self.set_src_dst_tensor()

    def set_src_dst_tensor(self):
        """
        set input and output tensor
        """
        self.coeff = self.tik_instance.Scalar("float32",
                                              name="coeff",
                                              init_value=1 / 255)

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
        self.input_code0_gm = self.tik_instance.Tensor(
            self.dtype_code0,
            self.shape_code0,
            name="input_code0_gm",
            scope=tik.scope_gm)
        self.input_code1_gm = self.tik_instance.Tensor(
            self.dtype_code1,
            self.shape_code1,
            name="input_code1_gm",
            scope=tik.scope_gm)
        self.input_code2_gm = self.tik_instance.Tensor(
            self.dtype_code2,
            self.shape_code2,
            name="input_code2_gm",
            scope=tik.scope_gm)
        self.input_code3_gm = self.tik_instance.Tensor(
            self.dtype_code3,
            self.shape_code3,
            name="input_code3_gm",
            scope=tik.scope_gm)
        self.input_dm_gm = self.tik_instance.Tensor(
            self.dtype_dm,
            self.shape_dm,
            name="input_dm_gm",
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
        self.input_code_gm = [self.input_code0_gm, self.input_code1_gm, self.input_code2_gm, self.input_code3_gm]

    def cal_num_each_core(self):
        """
        calculate actual code num of each core
        """
        # move actual code num from out to UB
        actual_num_ub = self.tik_instance.Tensor("uint32",
                                                 (16,),
                                                 name="actual_num_ub", scope=tik.scope_ubuf)
        self.tik_instance.data_move(actual_num_ub, self.input_actual_num_gm, 0, 1, 2, 0, 0)
        self.code_list_offset = 8192
        self.max_list_offset = 1024

        self.actual_num = [self.tik_instance.Scalar(dtype="uint32") for x in range(self.code_list_num)]
        for i in range(self.code_list_num):
            self.actual_num[i].set_as(actual_num_ub[i])

        self.code_loops = [(self.actual_num[0] + self.code_each_loop - 1) // self.code_each_loop,
                           (self.actual_num[1] + self.code_each_loop - 1) // self.code_each_loop,
                           (self.actual_num[2] + self.code_each_loop - 1) // self.code_each_loop,
                           (self.actual_num[3] + self.code_each_loop - 1) // self.code_each_loop]

        for i in range(1, self.code_list_num):
            self.code_loops[i] = self.code_loops[i] + self.code_loops[i - 1]
        self.total_loops = self.code_loops[self.code_list_num - 1]

        # 计算每个core上的loops
        self.each_loops = (self.total_loops + self.aicore_use - 1) // self.aicore_use  # + 1

    def distance_compute_(self, block_id, start, end):
        add_scalar = self.tik_instance.Scalar(dtype="float32", name="add_scalar")
        data_xd_l1 = self.tik_instance.Tensor("float16",
                                              (self.dim // 16, self.queries_align, 16),
                                              name="data_xd_l1", scope=tik.scope_cbuf)

        with self.tik_instance.new_stmt_scope():
            queries_ub = self.tik_instance.Tensor("float16",
                                                  (1, self.dim),
                                                  name="queries_ub", scope=tik.scope_ubuf)
            diff_ub = self.tik_instance.Tensor("float16",
                                               (self.dim,),
                                               name="diff_ub", scope=tik.scope_ubuf)
            min_ub = self.tik_instance.Tensor("float16",
                                              (self.dim,),
                                              name="min_ub", scope=tik.scope_ubuf)

            # move diff adn min from out to UB
            self.tik_instance.data_move(diff_ub, self.input_dm_gm,
                                        0, 1, self.dim // 16, 0, 0)
            self.tik_instance.data_move(min_ub, self.input_dm_gm[self.dim],
                                        0, 1, self.dim // 16, 0, 0)
            diff2_ub = self.tik_instance.Tensor("float16",
                                                (self.dim,),
                                                name="diff2_ub", scope=tik.scope_ubuf)
            self._muls(diff2_ub, diff_ub, 0.5, self.dim, self.fp16_mask)
            self._muls(min_ub, min_ub, 255.0, self.dim, self.fp16_mask)
            self._add(diff2_ub, diff2_ub, min_ub, self.dim, self.fp16_mask)
            dm_transfer_ub = self.tik_instance.Tensor("float16",
                                                      (self.dim // 16, 16, 16),
                                                      name="dm_transfer_ub", scope=tik.scope_ubuf)
            self.tik_instance.data_move(dm_transfer_ub, diff2_ub,
                                        0, self.dim // 16, 1, 0, 15)
            # move d+m from UB to L1
            data_dm_l1 = self.tik_instance.Tensor("float16",
                                                  (self.dim // 16, 16, 16),
                                                  name="data_dm_l1", scope=tik.scope_cbuf)
            self.tik_instance.data_move(data_dm_l1, dm_transfer_ub,
                                        0, 1, self.dim, 0, 0)

            # move x from out to UB
            self.tik_instance.data_move(queries_ub,
                                        self.input_queries_gm,
                                        0, 1, self.dim // 16, 0, 0)
            queries_transfer_ub = self.tik_instance.Tensor("float16",
                                                           (self.dim // 16, self.queries_align, 16),
                                                           name="queries_transfer_ub", scope=tik.scope_ubuf)
            self.tik_instance.data_move(queries_transfer_ub,
                                        queries_ub,
                                        0, self.dim // 16, 1, 0, self.queries_align - 1)

            # cal x*d
            if self.dim <= 128:
                self.tik_instance.vmul(self.dim, queries_ub, queries_ub,
                                       diff_ub, 1, 1, 1, 1,
                                       self.dim // 16, self.dim // 16, 0)
            else:
                repeat = self.dim // self.fp16_mask
                offset = 0
                if repeat > 0:
                    self.tik_instance.vmul(self.fp16_mask,
                                           queries_ub[0, offset],
                                           queries_ub[0, offset], diff_ub[offset],
                                           repeat, 1, 1, 1, 8, 8, 8)
                    offset += repeat * self.fp16_mask

                remain = self.dim % self.fp16_mask
                if remain > 0:
                    self.tik_instance.vmul(remain, queries_ub[0, offset],
                                           queries_ub[0, offset], diff_ub[offset],
                                           1, 1, 1, 1, 8, 8, 8)

            xd_transfer_ub = self.tik_instance.Tensor("float16",
                                                      (self.dim // 16, self.queries_align, 16),
                                                      name="xd_transfer_ub", scope=tik.scope_ubuf)
            self.tik_instance.data_move(xd_transfer_ub,
                                        queries_ub,
                                        0, self.dim // 16, 1, 0, self.queries_align - 1)

            # move x from ub to L1
            queries_l1 = self.tik_instance.Tensor("float16",
                                                  (self.dim // 16, self.queries_align, 16),
                                                  name="queries_l1", scope=tik.scope_cbuf)
            self.tik_instance.data_move(queries_l1, queries_transfer_ub,
                                        0, 1, self.queries_align * self.dim // 16, 0, 0)
            # move x*d from UB to L1
            self.tik_instance.data_move(data_xd_l1, xd_transfer_ub,
                                        0, 1, self.queries_align * self.dim // 16, 0, 0)

            data_xdm_l0c = self.tik_instance.Tensor("float32",
                                                    (self.queries_align // 16, 16, 16),
                                                    name="data_xdm_l0c", scope=tik.scope_cbuf_out)
            self.tik_instance.matmul(data_xdm_l0c,
                                     data_dm_l1, queries_l1,
                                     16, self.dim, self.queries_align)

            data_xdm_ub = self.tik_instance.Tensor("float32",
                                                   (self.queries_align // 16, 16, 16),
                                                   name="data_xdm_ub", scope=tik.scope_ubuf)
            self.tik_instance.data_move(data_xdm_ub, data_xdm_l0c,
                                        0, 1, self.queries_align // 16, 0, 0)
            data_xdm_align_ub = self.tik_instance.Tensor("float32",
                                                         (self.queries_align,),
                                                         name="data_xdm_align_ub", scope=tik.scope_ubuf)
            self.tik_instance.vmuls(16, data_xdm_align_ub,
                                    data_xdm_ub, self.coeff,
                                    self.queries_align // 16, 1, 1, 2, 32)

            add_scalar.set_as(data_xdm_align_ub[0])

        # compute xy using cube
        inner_product_l0c = self.tik_instance.Tensor("float32",
                                                     (self.code_each_loop // 16, self.queries_align, 16),
                                                     name="inner_product_l0c", scope=tik.scope_cbuf_out)
        code_last_l1 = self.tik_instance.Tensor("float16",
                                                (self.code_list_num, self.code_each_loop),
                                                name="scope_cbuf", scope=tik.scope_cbuf)
        # 处理list不足self.code_each_loop的情形
        with self.tik_instance.if_scope(block_id == 0):
            for i in range(0, self.sub_list_num):
                with self.tik_instance.if_scope(self.actual_num[i] < self.code_each_loop):
                    self.cube_compute_(self.input_code_gm[i], inner_product_l0c, data_xd_l1, add_scalar,
                                       self.actual_num[i], i)
        with self.tik_instance.else_scope():
            for i in range(self.sub_list_num, self.code_list_num):
                with self.tik_instance.if_scope(self.actual_num[i] < self.code_each_loop):
                    self.cube_compute_(self.input_code_gm[i], inner_product_l0c, data_xd_l1, add_scalar,
                                       self.actual_num[i], i)

        with self.tik_instance.for_range(start, end) as loop:
            self.cube_compute_each_loop(self.input_code_gm, inner_product_l0c, data_xd_l1,
                                        add_scalar, code_last_l1, start, loop, 0)
        self.cube_compute_each_loop(self.input_code_gm, inner_product_l0c, data_xd_l1,
                                    add_scalar, code_last_l1, start, end, 1)

        with self.tik_instance.if_scope(block_id == 0):
            for i in range(self.code_list_num):
                with self.tik_instance.if_scope(self.code_loops[i] <= self.each_loops):
                    self.compute_last_maxs(code_last_l1, i, self.actual_num[i])
        with self.tik_instance.else_scope():
            for i in range(self.code_list_num):
                with self.tik_instance.if_scope(self.code_loops[i] > self.each_loops):
                    self.compute_last_maxs(code_last_l1, i, self.actual_num[i])
    
    def conv_code_type(self, code_ub_fp16, code_ub):
        vconv_repeat_times = min(self.code_sub_loop * self.dim // self.fp16_mask, self.repeat_times)
        total_code = self.code_sub_loop * self.dim
        vconv_repeat_loops = ((total_code // self.fp16_mask) + vconv_repeat_times - 1) // vconv_repeat_times
        remain_code = total_code
        for i in range(0, vconv_repeat_loops):
            repeat = min(remain_code // self.fp16_mask, vconv_repeat_times)
            self.tik_instance.vconv(self.fp16_mask, "none",
                                    code_ub_fp16[i * self.fp16_mask * self.repeat_times],
                                    code_ub[i * self.fp16_mask * self.repeat_times],
                                    repeat, 1, 1, 8, 4)
            remain_code -= self.fp16_mask * repeat

    def cube_compute_each_loop(self, code_gm, inner_product_l0c, data_xd_l1, add_scalar, code_last_l1,
                               start_loop, code_loop, flag):
        dst_ub = self.tik_instance.Tensor("float16",
                                          (self.code_each_loop,),
                                          name="dst_ub", scope=tik.scope_ubuf)
        dst_max_ub = self.tik_instance.Tensor("float16",
                                              (self.maxs_each_loop,),
                                              name="dst_max_ub", scope=tik.scope_ubuf)

        with self.tik_instance.for_range(0, self.code_each_loops, thread_num=2) as loop_ub:
            # MTE2: move y from out to UB
            code_ub = self.tik_instance.Tensor("uint8",
                                               (self.dim // 16, self.code_sub_loop // 16, 16, 16),
                                               name="code_ub", scope=tik.scope_ubuf)
            if flag == 0:
                # handle the data in list0
                with self.tik_instance.if_scope(code_loop < self.code_loops[0]):
                    # handle the loop except the last one
                    with self.tik_instance.if_scope(code_loop + 1 < self.code_loops[0]):
                        with self.tik_instance.for_range(0, self.dim // 16) as i:
                            self.tik_instance.data_move(
                                code_ub[i, 0, 0, 0],
                                code_gm[0][(code_loop * self.code_each_loop
                                            + loop_ub * self.code_sub_loop) // 16, i, 0, 0],
                                0, self.code_sub_loop // 16, 8, self.dim // 2 - 8, 0)
                    # handle the last loop, because it maybe not enough self.code_each_loop
                    with self.tik_instance.else_scope():
                        # if the list data not enough self.code_each_loop
                        with self.tik_instance.if_scope(
                                self.actual_num[0] + self.cube_align - 1 >= self.code_each_loop):
                            code_align0 = (self.actual_num[0] + self.cube_align - 1) \
                                          // self.cube_align * self.cube_align
                            with self.tik_instance.for_range(0, self.dim // 16) as i:
                                self.tik_instance.data_move(
                                    code_ub[i, 0, 0, 0],
                                    code_gm[0][(code_align0 - self.code_each_loop
                                                + loop_ub * self.code_sub_loop) // 16, i, 0, 0],
                                    0, self.code_sub_loop // 16, 8, self.dim // 2 - 8, 0)
                with self.tik_instance.else_scope():
                    # handle the data in list1
                    with self.tik_instance.if_scope(code_loop < self.code_loops[1]):
                        # handle the loop except the last one
                        with self.tik_instance.if_scope(code_loop + 1 < self.code_loops[1]):
                            with self.tik_instance.for_range(0, self.dim // 16) as i:
                                self.tik_instance.data_move(
                                    code_ub[i, 0, 0, 0],
                                    code_gm[1][((code_loop - self.code_loops[0]) * self.code_each_loop
                                                + loop_ub * self.code_sub_loop) // 16, i, 0, 0],
                                    0, self.code_sub_loop // 16, 8, self.dim // 2 - 8, 0)
                        # handle the last loop, because it maybe not enough self.code_each_loop
                        with self.tik_instance.else_scope():
                            # if the list data not enough self.code_each_loop
                            with self.tik_instance.if_scope(
                                    self.actual_num[1] + self.cube_align - 1 >= self.code_each_loop):
                                code_align1 = (self.actual_num[1] + self.cube_align - 1) \
                                              // self.cube_align * self.cube_align
                                with self.tik_instance.for_range(0, self.dim // 16) as i:
                                    self.tik_instance.data_move(
                                        code_ub[i, 0, 0, 0],
                                        code_gm[1][(code_align1 - self.code_each_loop
                                                    + loop_ub * self.code_sub_loop) // 16, i, 0, 0],
                                        0, self.code_sub_loop // 16, 8, self.dim // 2 - 8, 0)
                    with self.tik_instance.else_scope():
                        # handle the data in list2
                        with self.tik_instance.if_scope(code_loop < self.code_loops[2]):
                            # handle the loop except the last one
                            with self.tik_instance.if_scope(code_loop + 1 < self.code_loops[2]):
                                with self.tik_instance.for_range(0, self.dim // 16) as i:
                                    self.tik_instance.data_move(
                                        code_ub[i, 0, 0, 0],
                                        code_gm[2][((code_loop - self.code_loops[1]) * self.code_each_loop
                                                    + loop_ub * self.code_sub_loop) // 16, i, 0, 0],
                                        0, self.code_sub_loop // 16, 8, self.dim // 2 - 8, 0)
                            # handle the last loop, because it maybe not enough self.code_each_loop
                            with self.tik_instance.else_scope():
                                # if the list data not enough self.code_each_loop
                                with self.tik_instance.if_scope(
                                        self.actual_num[2] + self.cube_align - 1 >= self.code_each_loop):
                                    code_align2 = (self.actual_num[2] + self.cube_align - 1) \
                                                  // self.cube_align * self.cube_align
                                    with self.tik_instance.for_range(0, self.dim // 16) as i:
                                        self.tik_instance.data_move(
                                            code_ub[i, 0, 0, 0],
                                            code_gm[2][(code_align2 - self.code_each_loop
                                                        + loop_ub * self.code_sub_loop) // 16, i, 0, 0],
                                            0, self.code_sub_loop // 16, 8, self.dim // 2 - 8, 0)
                        with self.tik_instance.else_scope():
                            # handle the data in list3
                            with self.tik_instance.if_scope(code_loop < self.code_loops[3]):
                                # handle the loop except the last one
                                with self.tik_instance.if_scope(code_loop + 1 < self.code_loops[3]):
                                    with self.tik_instance.for_range(0, self.dim // 16) as i:
                                        self.tik_instance.data_move(
                                            code_ub[i, 0, 0, 0],
                                            code_gm[3][((code_loop - self.code_loops[2]) * self.code_each_loop
                                                        + loop_ub * self.code_sub_loop) // 16, i, 0, 0],
                                            0, self.code_sub_loop // 16, 8, self.dim // 2 - 8, 0)
                                # handle the last loop, because it maybe not enough self.code_each_loop
                                with self.tik_instance.else_scope():
                                    # if the list data not enough self.code_each_loop
                                    with self.tik_instance.if_scope(
                                            self.actual_num[3] + self.cube_align - 1 >= self.code_each_loop):
                                        code_align3 = (self.actual_num[3] + self.cube_align - 1) \
                                                      // self.cube_align * self.cube_align
                                        with self.tik_instance.for_range(0, self.dim // 16) as i:
                                            self.tik_instance.data_move(
                                                code_ub[i, 0, 0, 0],
                                                code_gm[3][(code_align3 - self.code_each_loop
                                                            + loop_ub * self.code_sub_loop) // 16, i, 0, 0],
                                                0, self.code_sub_loop // 16, 8, self.dim // 2 - 8, 0)

            # VECTOR: y do conv from uint8 to fp16
            code_ub_fp16 = self.tik_instance.Tensor("float16",
                                                    (self.dim // 16, self.code_sub_loop // 16, 16, 16),
                                                    name="code_ub_fp16", scope=tik.scope_ubuf)
            self.conv_code_type(code_ub_fp16, code_ub)
            # VECTOR: mov xy from L0-C to UB
            inner_product_ub = self.tik_instance.Tensor("float32",
                                                        (self.code_sub_loop // 16, self.queries_align, 16),
                                                        name="inner_product_ub", scope=tik.scope_ubuf)
            self.tik_instance.data_move(inner_product_ub,
                                        inner_product_l0c[(self.code_each_loops + loop_ub - 2) % self.code_each_loops
                                                          * self.code_sub_loop // 16, 0, 0],
                                        0, 1, self.code_sub_loop * self.queries_align // 256, 0, 0)
            # VECTOR: compute distance each query
            dst_ub_fp32 = self.tik_instance.Tensor("float32",
                                                   (self.code_sub_loop,),
                                                   name="dst_ub_fp32", scope=tik.scope_ubuf)
            with self.tik_instance.if_scope(code_loop - start_loop != 0):
                self.tik_instance.vmuls(16,
                                        dst_ub_fp32,
                                        inner_product_ub, self.coeff,
                                        self.code_sub_loop // 16, 1, 1, 2, self.queries_align * 2)
                repeat_times = self.code_sub_loop // self.fp32_mask
                self.tik_instance.vadds(self.fp32_mask, dst_ub_fp32,
                                        dst_ub_fp32, add_scalar,
                                        repeat_times, 1, 1, 8, 8)
                self.tik_instance.vconv(self.fp32_mask, "none",
                                        dst_ub[(loop_ub % 2) * self.code_sub_loop:], dst_ub_fp32,
                                        repeat_times, 1, 1, 4, 8)
                # the loop not enough self.code_each_loop, wo not use
                vcmax_repeat_times = self.code_sub_loop // self.max_batch
                self.tik_instance.vcmax(self.max_batch,
                                        dst_max_ub[(loop_ub % 2) * self.maxs_sub_loop:],
                                        dst_ub[(loop_ub % 2) * self.code_sub_loop:],
                                        vcmax_repeat_times, 1, 1, self.max_batch // 16)
            # MTE3: move y from ub to L1
            code_l1 = self.tik_instance.Tensor("float16",
                                               (self.dim // 16, self.code_sub_loop, 16),
                                               name="code_l1", scope=tik.scope_cbuf)
            self.tik_instance.data_move(code_l1, code_ub_fp16,
                                        0, self.dim // 16, self.code_sub_loop, 0, 0)

            self.tik_instance.matmul(inner_product_l0c[(loop_ub * self.code_sub_loop * self.queries_align):],
                                     data_xd_l1, code_l1,
                                     self.queries_align, self.dim, self.code_sub_loop)

            with self.tik_instance.if_scope(loop_ub % 2 * (code_loop - start_loop) != 0):
                # handle for list0
                with self.tik_instance.if_scope(code_loop <= self.code_loops[0]):
                    # handle loops except the last loop
                    with self.tik_instance.if_scope(code_loop < self.code_loops[0]):
                        self.tik_instance.data_move(
                            self.output_distances_gm[
                                code_loop * self.code_each_loop
                                + (loop_ub - 3) * self.code_sub_loop],
                            dst_ub,
                            0, 1, self.code_each_loop // 16, 0, 0)
                        self.tik_instance.data_move(
                            self.output_maxs_gm[
                                code_loop * self.maxs_each_loop
                                + (loop_ub - 3) * self.maxs_sub_loop],
                            dst_max_ub,
                            0, 1, self.maxs_each_loop // 16, 0, 0)
                    # handle for the last loop
                    with self.tik_instance.else_scope():
                        code_align0 = (self.actual_num[0] + self.cube_align - 1) \
                                      // self.cube_align * self.cube_align
                        with self.tik_instance.if_scope(code_align0 >= self.code_each_loop):
                            self.tik_instance.data_move(
                                self.output_distances_gm[code_align0 - self.code_each_loop],
                                dst_ub,
                                0, 1, self.code_each_loop // 16, 0, 0)
                            with self.tik_instance.if_scope(code_align0 % self.code_each_loop != 0):
                                # move to l1, because it maybe not enough self.code_each_loop
                                self.tik_instance.data_move(
                                    code_last_l1[0, 0],
                                    dst_ub[self.code_each_loop - code_align0 % self.code_each_loop],
                                    0, 1, (code_align0 % self.code_each_loop) // 16, 0, 0)
                with self.tik_instance.else_scope():
                    # handle for list1
                    with self.tik_instance.if_scope(code_loop <= self.code_loops[1]):
                        # handle loops except the last loop
                        with self.tik_instance.if_scope(code_loop < self.code_loops[1]):
                            self.tik_instance.data_move(
                                self.output_distances_gm[self.code_list_offset
                                                         + (code_loop - self.code_loops[0]) * self.code_each_loop
                                                         + (loop_ub - 3) * self.code_sub_loop],
                                dst_ub,
                                0, 1, self.code_each_loop // 16, 0, 0)
                            self.tik_instance.data_move(
                                self.output_maxs_gm[self.max_list_offset
                                                    + (code_loop - self.code_loops[0]) * self.maxs_each_loop
                                                    + (loop_ub - 3) * self.maxs_sub_loop],
                                dst_max_ub,
                                0, 1, self.maxs_each_loop // 16, 0, 0)
                        # handle for the last loop
                        with self.tik_instance.else_scope():
                            code_align1 = (self.actual_num[1] + self.cube_align - 1) \
                                          // self.cube_align * self.cube_align
                            with self.tik_instance.if_scope(code_align1 >= self.code_each_loop):
                                self.tik_instance.data_move(
                                    self.output_distances_gm[self.code_list_offset
                                                             + code_align1 - self.code_each_loop],
                                    dst_ub,
                                    0, 1, self.code_each_loop // 16, 0, 0)
                                with self.tik_instance.if_scope(code_align1 % self.code_each_loop != 0):
                                    # move to l1, because it maybe not enough self.code_each_loop
                                    self.tik_instance.data_move(
                                        code_last_l1[1, 0],
                                        dst_ub[self.code_each_loop - code_align1 % self.code_each_loop],
                                        0, 1, (code_align1 % self.code_each_loop) // 16, 0, 0)
                    with self.tik_instance.else_scope():
                        # handle for list2
                        with self.tik_instance.if_scope(code_loop <= self.code_loops[2]):
                            # handle loops except the last loop
                            with self.tik_instance.if_scope(code_loop < self.code_loops[2]):
                                self.tik_instance.data_move(
                                    self.output_distances_gm[2 * self.code_list_offset
                                                             + (code_loop - self.code_loops[1]) * self.code_each_loop
                                                             + (loop_ub - 3) * self.code_sub_loop],
                                    dst_ub,
                                    0, 1, self.code_each_loop // 16, 0, 0)
                                self.tik_instance.data_move(
                                    self.output_maxs_gm[2 * self.max_list_offset
                                                        + (code_loop - self.code_loops[1]) * self.maxs_each_loop
                                                        + (loop_ub - 3) * self.maxs_sub_loop],
                                    dst_max_ub,
                                    0, 1, self.maxs_each_loop // 16, 0, 0)
                            # handle for the last loop
                            with self.tik_instance.else_scope():
                                code_align2 = (self.actual_num[2] + self.cube_align - 1) \
                                              // self.cube_align * self.cube_align
                                with self.tik_instance.if_scope(code_align2 >= self.code_each_loop):
                                    self.tik_instance.data_move(
                                        self.output_distances_gm[2 * self.code_list_offset
                                                                 + code_align2 - self.code_each_loop],
                                        dst_ub,
                                        0, 1, self.code_each_loop // 16, 0, 0)
                                    with self.tik_instance.if_scope(code_align2 % self.code_each_loop != 0):
                                        # move to l1, because it maybe not enough self.code_each_loop
                                        self.tik_instance.data_move(
                                            code_last_l1[2, 0],
                                            dst_ub[self.code_each_loop - code_align2 % self.code_each_loop],
                                            0, 1, (code_align2 % self.code_each_loop) // 16, 0, 0)
                        with self.tik_instance.else_scope():
                            # handle for list3
                            with self.tik_instance.if_scope(code_loop <= self.code_loops[3]):
                                # handle loops except the last loop
                                with self.tik_instance.if_scope(code_loop < self.code_loops[3]):
                                    self.tik_instance.data_move(
                                        self.output_distances_gm[
                                            3 * self.code_list_offset
                                            + (code_loop - self.code_loops[2]) * self.code_each_loop
                                            + (loop_ub - 3) * self.code_sub_loop],
                                        dst_ub,
                                        0, 1, self.code_each_loop // 16, 0, 0)
                                    self.tik_instance.data_move(
                                        self.output_maxs_gm[
                                            3 * self.max_list_offset
                                            + (code_loop - self.code_loops[2]) * self.maxs_each_loop
                                            + (loop_ub - 3) * self.maxs_sub_loop],
                                        dst_max_ub,
                                        0, 1, self.maxs_each_loop // 16, 0, 0)
                                # handle for the last loop
                                with self.tik_instance.else_scope():
                                    code_align3 = (self.actual_num[3] + self.cube_align - 1) \
                                                  // self.cube_align * self.cube_align
                                    with self.tik_instance.if_scope(code_align3 >= self.code_each_loop):
                                        self.tik_instance.data_move(
                                            self.output_distances_gm[3 * self.code_list_offset
                                                                     + code_align3 - self.code_each_loop],
                                            dst_ub,
                                            0, 1, self.code_each_loop // 16, 0, 0)
                                        with self.tik_instance.if_scope(code_align3 % self.code_each_loop != 0):
                                            # move to l1, because it maybe not enough self.code_each_loop
                                            self.tik_instance.data_move(
                                                code_last_l1[3, 0],
                                                dst_ub[self.code_each_loop - code_align3 % self.code_each_loop],
                                                0, 1, (code_align3 % self.code_each_loop) // 16, 0, 0)

    def cube_compute_(self, code_gm, inner_product_l0c, data_xd_l1, add_scalar, code_num, list_id):
        with self.tik_instance.new_stmt_scope():
            code_ub = self.tik_instance.Tensor("uint8",
                                               (self.dim // 16, self.code_each_loop // 16, 16, 16),
                                               name="code_ub", scope=tik.scope_ubuf)
            code_ub_fp16 = self.tik_instance.Tensor("float16",
                                                    (self.dim // 16, self.code_each_loop // 16, 16, 16),
                                                    name="code_ub_fp16", scope=tik.scope_ubuf)
            with self.tik_instance.if_scope(code_num > 0):
                with self.tik_instance.for_range(0, self.dim // 16) as i:
                    self.tik_instance.data_move(code_ub[i, 0, 0, 0],
                                                code_gm[0, i, 0, 0],
                                                0, (code_num + 15) // 16, 8, self.dim // 2 - 8, 0)

                # y do conv from uint8 to fp16
                vconv_loop = (self.code_each_loop * self.dim) // (self.fp16_mask * 255)
                vconv_offset = 0
                if vconv_loop > 0:
                    for conv_index in range(vconv_loop):
                        vconv_offset = conv_index * self.fp16_mask * 255
                        self.tik_instance.vconv(self.fp16_mask, "none",
                                                code_ub_fp16[vconv_offset], code_ub[vconv_offset],
                                                255, 1, 1, 8, 4)
                    vconv_offset += self.fp16_mask * 255

                vconv_repeat_time = (self.code_each_loop * self.dim) % (self.fp16_mask * 255) // self.fp16_mask
                if vconv_repeat_time > 0:
                    self.tik_instance.vconv(self.fp16_mask, "none",
                                            code_ub_fp16[vconv_offset], code_ub[vconv_offset],
                                            vconv_repeat_time, 1, 1, 8, 4)
                # move y from ub to L1
                code_l1 = self.tik_instance.Tensor("float16",
                                                   (self.dim // 16, self.code_each_loop, 16),
                                                   name="code_l1", scope=tik.scope_cbuf)
                self.tik_instance.data_move(code_l1, code_ub_fp16,
                                            0, self.dim // 16, self.code_each_loop, 0, 0)
                self.tik_instance.matmul(inner_product_l0c,
                                         data_xd_l1, code_l1,
                                         self.queries_align, self.dim, self.code_each_loop)

                # mov xy from L0-C to UB
                inner_product_ub = self.tik_instance.Tensor("float32",
                                                            (self.code_each_loop // 16, self.queries_align, 16),
                                                            name="inner_product_ub", scope=tik.scope_ubuf)
                self.tik_instance.data_move(inner_product_ub,
                                            inner_product_l0c,
                                            0, 1, self.code_each_loop * self.queries_align // 256, 0, 0)
                # compute distance each query
                dst_ub_fp32 = self.tik_instance.Tensor("float32",
                                                       (self.code_each_loop,),
                                                       name="dst_ub_fp32", scope=tik.scope_ubuf)
                self.tik_instance.vmuls(16,
                                        dst_ub_fp32,
                                        inner_product_ub, self.coeff,
                                        self.code_each_loop // 16, 1, 1, 2, self.queries_align * 2)

                self.tik_instance.vadds(self.fp32_mask, dst_ub_fp32,
                                        dst_ub_fp32, add_scalar,
                                        self.code_each_loop // self.fp32_mask, 1, 1, 8, 8)

                dst_ub = self.tik_instance.Tensor("float16",
                                                  (self.code_each_loop,),
                                                  name="dst_ub", scope=tik.scope_ubuf)
                self.tik_instance.vconv(self.fp32_mask, "none",
                                        dst_ub, dst_ub_fp32,
                                        self.code_each_loop // self.fp32_mask, 1, 1, 4, 8)

                dst_max_ub = self.tik_instance.Tensor("float16",
                                                      (self.maxs_each_loop,),
                                                      name="dst_max_ub", scope=tik.scope_ubuf)
                vcmax_repeat_times = code_num // self.max_batch
                with self.tik_instance.if_scope(vcmax_repeat_times > 0):
                    self.tik_instance.vcmax(self.max_batch,
                                            dst_max_ub, dst_ub,
                                            vcmax_repeat_times, 1, 1, self.max_batch // 16)
                code_last_retain = code_num % self.max_batch
                with self.tik_instance.if_scope(code_last_retain > 0):
                    self.tik_instance.vcmax(code_last_retain,
                                            dst_max_ub[2 * vcmax_repeat_times],
                                            dst_ub[self.max_batch * vcmax_repeat_times],
                                            1, 1, 1, 1)
                self.tik_instance.data_move(self.output_distances_gm[list_id * self.code_list_offset],
                                            dst_ub,
                                            0, 1, (code_num + 15) // 16, 0, 0)
                self.tik_instance.data_move(
                    self.output_maxs_gm[list_id * self.max_list_offset],
                    dst_max_ub,
                    0, 1, ((code_num + self.max_batch - 1) // self.max_batch * 2 + 15) // 16, 0, 0)

    def compute_last_maxs(self, code_last_l1, list_id, code_num):
            code_loops_ = code_num // self.code_each_loop
            last_num = code_num % self.code_each_loop
            with self.tik_instance.if_scope(code_loops_ * last_num > 0):
                dst_ub = self.tik_instance.Tensor("float16",
                                                  (self.code_each_loop,),
                                                  name="dst_ub", scope=tik.scope_ubuf)
                self.tik_instance.data_move(dst_ub,
                                            code_last_l1[list_id, 0],
                                            0, 1, (last_num + 15) // 16, 0, 0)
                dst_max_ub = self.tik_instance.Tensor("float16",
                                                      (self.maxs_each_loop,),
                                                      name="dst_max_ub", scope=tik.scope_ubuf)
                vcmax_repeat_times = last_num // self.max_batch
                with self.tik_instance.if_scope(vcmax_repeat_times > 0):
                    self.tik_instance.vcmax(self.max_batch,
                                            dst_max_ub, dst_ub,
                                            vcmax_repeat_times, 1, 1, self.max_batch // 16)
                code_last_retain = last_num % self.max_batch
                with self.tik_instance.if_scope(code_last_retain > 0):
                    self.tik_instance.vcmax(code_last_retain,
                                            dst_max_ub[2 * vcmax_repeat_times],
                                            dst_ub[self.max_batch * vcmax_repeat_times],
                                            1, 1, 1, 1)
                self.tik_instance.data_move(
                    self.output_maxs_gm[list_id * self.max_list_offset
                                        + code_loops_ * self.maxs_each_loop],
                    dst_max_ub,
                    0, 1, self.maxs_each_loop // 16, 0, 0)

    def distance_compute(self):
        """
        the compute process
        """
        self.cal_num_each_core()

        with self.tik_instance.for_range(0, self.aicore_use, block_num=self.aicore_use) as block_index:
            #     # compute coarse centroids num and move offset every core
            self.distance_compute_(block_index, block_index * self.each_loops,
                                   (1 - block_index) * self.each_loops + block_index * self.total_loops)

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
        self.distance_compute()

        self.tik_instance.BuildCCE(
            kernel_name=self.kernel_name,
            inputs=[
                self.input_queries_gm,
                self.input_code0_gm,
                self.input_code1_gm,
                self.input_code2_gm,
                self.input_code3_gm,
                self.input_dm_gm,
                self.input_actual_num_gm
            ],
            outputs=[self.output_distances_gm, self.output_maxs_gm, self.output_flag_gm])

        return self.tik_instance

    def _muls(self, dst, src, scalar, compute_num, max_mask):
        # process 256B data per repeat for vmuls
        repeat = compute_num // max_mask
        offset = 0
        if repeat > 0:
            self.tik_instance.vmuls(max_mask, dst[offset], src[offset], scalar,
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
            self.tik_instance.vadd(max_mask, dst[offset],
                                   src0[offset], src1[offset],
                                   repeat, 1, 1, 1, 8, 8, 8)
            offset += repeat * max_mask

        remain = compute_num % max_mask
        if remain > 0:
            self.tik_instance.vadd(remain, dst[offset],
                                   src0[offset], src1[offset],
                                   1, 1, 1, 1, 8, 8, 8)


def distance_ivf_sq8_ip4(input_queries,
                         input_code0,
                         input_code1,
                         input_code2,
                         input_code3,
                         input_dm,
                         input_actual_num,
                         output_distances,
                         output_maxs,
                         output_flag,
                         kernel_name="distance_ivf_sq8_ip4"):
    """
    calculating distance

    Parameters
    ----------
    input_queries : dict
        shape and dtype of query vector
    input_code0 : dict
        shape and dtype of coding vector 0
    input_code1 : dict
        shape and dtype of coding vector 1
    input_code2 : dict
        shape and dtype of coding vector 2
    input_code3 : dict
        shape and dtype of coding vector 3
    input_dm: dict
        shape and dtype of coding coefficient and offset
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
    distance_ivf_sq8 = DistanceIVFSQ8IP4(input_queries,
                                         input_code0,
                                         input_code1,
                                         input_code2,
                                         input_code3,
                                         input_dm,
                                         input_actual_num,
                                         output_distances,
                                         output_maxs,
                                         output_flag,
                                         kernel_name)
    tik_instance = distance_ivf_sq8.get_tik_instance()
    return tik_instance
