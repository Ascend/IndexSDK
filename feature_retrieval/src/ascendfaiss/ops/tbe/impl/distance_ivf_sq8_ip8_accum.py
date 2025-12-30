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
import collections
from te import tik
from mxIndex_impl.common import set_soc_info

DisLoopCtrl = collections.namedtuple('DisLoopCtrl',
                                      ['code_loop', 'code_num0', 'code_num1', 'code_last', 'accum_idx'])
ComputePara = collections.namedtuple('ComputePara', ['dst', 'compute_num', 'max_mask', 'list_idx', \
                                     'inner_product_l0c', 'add_scalar'])


class DistanceIVFSQ8IP8Accum():
    def __init__(self,
                 input_queries,
                 input_base_list,
                 input_offset_addrs,
                 input_dm,
                 input_actual_num_addrs,
                 input_accum_num,
                 input_query_offset,
                 output_distances_addrs,
                 output_maxs_addrs,
                 output_flag_addrs,
                 kernel_name="distance_ivf_sq8_ip8_accum"):
        self.shape_queries = input_queries.get("shape")
        self.dtype_queries = input_queries.get("dtype")
        self.shape_base_list = input_base_list.get("shape")
        self.dtype_base_list = input_base_list.get("dtype")
        self.shape_offset_addrs = input_offset_addrs.get("shape")
        self.shape_dm = input_dm.get("shape")
        self.dtype_dm = input_dm.get("dtype")
        self.shape_actual_num_addrs = input_actual_num_addrs.get("shape")
        self.shape_accum_num = input_accum_num.get("shape")
        self.dtype_accum_num = input_accum_num.get("dtype")
        self.shape_query_offet = input_query_offset.get("shape")
        self.dtype_query_offset = input_query_offset.get("dtype")
        self.shape_distances_addrs = output_distances_addrs.get("shape")
        self.shape_maxs_addrs = output_maxs_addrs.get("shape")
        self.shape_flag_addrs = output_flag_addrs.get("shape")
        self.queries_num, self.dim = self.shape_queries
        self.code_num = self.shape_base_list[0] // self.dim
        self.check_parameter()
        self.code_list_num = 8
        self.fp32_mask = 64
        self.fp16_mask = 128
        self.max_batch = 32 // (1 + (self.dim + 255) // 512)
        self.queries_align = 16
        self.cube_align = 16
        self.repeat_times = 128
        self.output_dist_num = self.shape_base_list[0] // self.dim
        self.output_max_num = self.output_dist_num // self.max_batch * 2
        self.output_flag_num = 32
        self.coeff = None
        self.input_queries_gm = None
        self.input_base_list_gm = None
        self.input_offset_addrs_gm = None
        self.input_dm_gm = None
        self.input_actual_num_addrs_gm = None
        self.input_accum_num_gm = None
        self.input_query_offset_gm = None
        self.output_distances_addrs_gm = None
        self.output_maxs_addrs_gm = None
        self.output_flag_addrs_gm = None
        self.distance_addrs_ub = None
        self.max_addrs_ub = None
        self.list_offset = None
        self.actual_num = None
        self.code_last_offset = None
        self.maxs_last_offset = None
        self.accum_num = None
        self.addrs_num = None
        self.query_offset = None
        self.kernel_name = kernel_name
        self.set_tik_instance()

    def check_parameter(self):
        # check shape and dtype of input
        if self.dim % 16 != 0:
            raise RuntimeError("feature dim is not multiple of 16")
        if self.code_num % 16 != 0:
            raise RuntimeError("code num is not multiple of 16")

    def set_tik_instance(self):
        """
        set tik_instance
        """
        set_soc_info()
        self.tik_instance = tik.Tik()

        self.aicore_use = 8

        self.queries_num_each_loop = min(self.queries_num, 48)
        self.code_each_loops = 2
        # the compute formula: (240 * 1024) // (70.25 + 3 * dim) // 256 * 256
        self.code_each_loop = 512 // (1 << ((self.dim + 255) // 256))
        self.code_sub_loop = self.code_each_loop // self.code_each_loops
        self.maxs_each_loop = self.code_each_loop // self.max_batch * 2
        self.maxs_sub_loop = self.code_sub_loop // self.max_batch * 2
        self.coeff = self.tik_instance.Scalar("float32", name="coeff", init_value=1 / 255)

        self.set_src_dst_tensor()

    def set_src_dst_tensor(self):
        """
        set input and output tensor
        """
        # creat input tensor: input_queries_gm, input_base_list_gm, input_list_offset_gm
        # input_diff_gm, input_min_gm, input_actual_num_gm
        # creat output tensor: output_distances_addrs_gm output_flag_addrs_gm output_flag_addrs_gm
        # 输出使用TensorAddrList, 用来保存离散的输出结果的地址，其中有效的地址数为 self.accum_num
        self.input_queries_gm = self.tik_instance.Tensor(self.dtype_queries, self.shape_queries,
                                                         name="input_queries_gm", scope=tik.scope_gm)
        self.input_base_list_gm = self.tik_instance.Tensor(self.dtype_base_list, self.shape_base_list,
                                                           name="input_base_list_gm", scope=tik.scope_gm)
        self.input_offset_addrs_gm = self.tik_instance.TensorAddrList(self.shape_offset_addrs[0],
                                                                      name="input_offset_addrs_gm", scope=tik.scope_gm)
        self.input_dm_gm = self.tik_instance.Tensor(self.dtype_dm, self.shape_dm,
                                                    name="input_dm_gm", scope=tik.scope_gm)
        self.input_actual_num_addrs_gm = self.tik_instance.TensorAddrList(self.shape_actual_num_addrs[0],
                                                                          name="input_actual_num_addrs_gm",
                                                                          scope=tik.scope_gm)
        self.input_accum_num_gm = self.tik_instance.Tensor(self.dtype_accum_num, self.shape_accum_num,
                                                           name="input_accum_num_gm", scope=tik.scope_gm)
        self.input_query_offset_gm = self.tik_instance.Tensor(self.dtype_query_offset, self.shape_query_offet,
                                                              name="input_query_offset_gm", scope=tik.scope_gm)
        self.output_distances_addrs_gm = self.tik_instance.TensorAddrList(self.shape_distances_addrs[0],
                                                                          name="output_distances_addrs_gm",
                                                                          scope=tik.scope_gm)
        self.output_maxs_addrs_gm = self.tik_instance.TensorAddrList(self.shape_maxs_addrs[0],
                                                                     name="output_maxs_addrs_gm", scope=tik.scope_gm)
        self.output_flag_addrs_gm = self.tik_instance.TensorAddrList(self.shape_flag_addrs[0],
                                                                     name="output_flag_addrs_gm", scope=tik.scope_gm)
        self.distance_compute()
        

    def cal_num_each_core(self):
        """
        calculate actual code num of each core
        """
        # read list_offset from gm
        self.accum_num = self.tik_instance.Scalar(dtype="uint32", name="accum_num")
        self.addrs_num = self.shape_distances_addrs[0]
        accum_num_ub = self.tik_instance.Tensor(self.dtype_accum_num, self.shape_accum_num,
                                                name="accum_num_ub", scope=tik.scope_ubuf)
        self.tik_instance.data_move(accum_num_ub, self.input_accum_num_gm, 0, 1, 1, 0, 0)
        self.accum_num.set_as(accum_num_ub[0])
        list_offset_ub = self.tik_instance.Tensor("uint64", (8,), name="list_offset_ub", scope=tik.scope_ubuf)
        self.list_offset = self.tik_instance.ScalarArray(dtype="uint64", length=self.code_list_num * self.addrs_num)
        self.query_offset = self.tik_instance.ScalarArray(dtype="uint64", length=self.shape_query_offet[0])
        offset_addrs_ub = self.tik_instance.TensorAddrList(self.shape_offset_addrs[0],
                                                          name="offset_addrs_ub", scope=tik.scope_ubuf)
        self.tik_instance.data_move(offset_addrs_ub, self.input_offset_addrs_gm,
                                    0, 1, self.shape_offset_addrs[0] // 4, 0, 0)
        
        with self.tik_instance.for_range(0, self.accum_num) as accum_idx:
            self.tik_instance.data_move(list_offset_ub, offset_addrs_ub[accum_idx].value, 0, 1, 2, 0, 0)
            with self.tik_instance.for_range(0, self.code_list_num) as i:
                self.list_offset[accum_idx * self.code_list_num + i].set_as(list_offset_ub[i])

        # read actual code num from gm
        actual_num_ub = self.tik_instance.Tensor("uint32", (8,), name="actual_num_ub", scope=tik.scope_ubuf)
        self.actual_num = self.tik_instance.ScalarArray(dtype="uint32",
                                                        length=self.code_list_num * self.addrs_num, name="actual_num")
        actual_num_addrs_ub = self.tik_instance.TensorAddrList(self.shape_actual_num_addrs[0],
                                                              name="actual_num_addrs_ub", scope=tik.scope_ubuf)
        self.tik_instance.data_move(actual_num_addrs_ub, self.input_actual_num_addrs_gm,
                                    0, 1, self.shape_actual_num_addrs[0] // 4, 0, 0)
        with self.tik_instance.for_range(0, self.accum_num) as accum_idx:
            self.tik_instance.data_move(actual_num_ub, actual_num_addrs_ub[accum_idx].value, 0, 1, 1, 0, 0)
            with self.tik_instance.for_range(0, self.code_list_num) as i:
                self.actual_num[self.code_list_num * accum_idx + i].set_as(actual_num_ub[i])

        self.code_last_offset = self.tik_instance.ScalarArray(dtype="uint32",
                                                              length=self.addrs_num * self.code_list_num,
                                                              name="code_last_offset", init_value=0)

        self.maxs_last_offset = self.tik_instance.ScalarArray(dtype="uint32",
                                                              length=self.addrs_num * self.code_list_num,
                                                              name="maxs_last_offset", init_value=0)
        self.distance_addrs_ub = self.tik_instance.TensorAddrList(self.shape_distances_addrs[0],
                                                                 name="distance_addrs_ub", scope=tik.scope_ubuf)
        self.tik_instance.data_move(self.distance_addrs_ub, self.output_distances_addrs_gm, 
                                    0, 1, self.shape_distances_addrs[0] // 4, 0, 0)

        self.max_addrs_ub = self.tik_instance.TensorAddrList(self.shape_maxs_addrs[0],
                                                                name="max_addrs_ub", scope=tik.scope_ubuf)
        self.tik_instance.data_move(self.max_addrs_ub, self.output_maxs_addrs_gm,
                                    0, 1, self.shape_maxs_addrs[0] // 4, 0, 0)
        with self.tik_instance.for_range(0, self.accum_num) as accum_idx:
            with self.tik_instance.for_range(0, self.code_list_num) as i:
                self.code_last_offset[self.code_list_num * accum_idx + i].set_as(
                    self.actual_num[accum_idx * self.code_list_num + i] % \
                        self.code_sub_loop // self.max_batch * self.max_batch)
                self.maxs_last_offset[self.code_list_num * accum_idx + i].set_as(
                    self.code_last_offset[accum_idx * self.code_list_num + i] // self.max_batch * 2)

    def calculate_add_scalar(self, add_scalar, data_xd_l1, accum_idx):
        queries_ub = self.tik_instance.Tensor("float16", (1, self.dim), name="queries_ub", scope=tik.scope_ubuf)
        diff_ub = self.tik_instance.Tensor("float16", (self.dim,), name="diff_ub", scope=tik.scope_ubuf)
        min_ub = self.tik_instance.Tensor("float16", (self.dim,), name="min_ub", scope=tik.scope_ubuf)
        dim_align = self.dim // 16

        # move diff adn min from out to UB
        self.tik_instance.data_move(diff_ub, self.input_dm_gm, 0, 1, dim_align, 0, 0)
        self.tik_instance.data_move(min_ub, self.input_dm_gm[self.dim], 0, 1, dim_align, 0, 0)
        diff2_ub = self.tik_instance.Tensor("float16", (self.dim,), name="diff2_ub", scope=tik.scope_ubuf)
        diff2_muls_para = ComputePara(diff2_ub, self.dim, self.fp16_mask, None, None, None)
        self._muls(diff2_muls_para, diff_ub, 0.5)
        min_muls_para = ComputePara(min_ub, self.dim, self.fp16_mask, None, None, None)
        self._muls(min_muls_para, min_ub, 255.0)
        _add_para = ComputePara(diff2_ub, self.dim, self.fp16_mask, None, None, None)
        self._add(_add_para, diff2_ub, min_ub)
        dm_transfer_ub = self.tik_instance.Tensor("float16", (dim_align, 16, 16),
                                                  name="dm_transfer_ub", scope=tik.scope_ubuf)
        self.tik_instance.data_move(dm_transfer_ub, diff2_ub, 0, dim_align, 1, 0, 15)
        # move d+m from UB to L1
        data_dm_l1 = self.tik_instance.Tensor("float16", (dim_align, 16, 16), name="data_dm_l1", scope=tik.scope_cbuf)
        self.tik_instance.data_move(data_dm_l1, dm_transfer_ub, 0, 1, self.dim, 0, 0)

        # move x from out to UB
        self.tik_instance.data_move(queries_ub, self.input_queries_gm[self.query_offset[accum_idx], 0],
                                    0, 1, dim_align, 0, 0)
        queries_transfer_ub = self.tik_instance.Tensor("float16", (dim_align, self.queries_align, 16),
                                                       name="queries_transfer_ub", scope=tik.scope_ubuf)
        self.tik_instance.data_move(queries_transfer_ub, queries_ub, 0, dim_align, 1, 0, self.queries_align - 1)

        # cal x*d
        if self.dim <= 128:
            self.tik_instance.vmul(self.dim, queries_ub, queries_ub, diff_ub, 1, 1, 1, 1, dim_align, dim_align, 0)
        else:
            repeat = self.dim // self.fp16_mask
            offset = 0
            if repeat > 0:
                self.tik_instance.vmul(self.fp16_mask, queries_ub[0, offset], queries_ub[0, offset],
                                       diff_ub[offset], repeat, 1, 1, 1, 8, 8, 8)
                offset += repeat * self.fp16_mask

            remain = self.dim % self.fp16_mask
            if remain > 0:
                self.tik_instance.vmul(remain, queries_ub[0, offset], queries_ub[0, offset],
                                       diff_ub[offset], 1, 1, 1, 1, 8, 8, 8)

        xd_transfer_ub = self.tik_instance.Tensor("float16", (dim_align, self.queries_align, 16),
                                                  name="xd_transfer_ub", scope=tik.scope_ubuf)
        self.tik_instance.data_move(xd_transfer_ub, queries_ub, 0, dim_align, 1, 0, self.queries_align - 1)

        # move x from ub to L1
        queries_l1 = self.tik_instance.Tensor("float16", (dim_align, self.queries_align, 16),
                                              name="queries_l1", scope=tik.scope_cbuf)
        self.tik_instance.data_move(queries_l1, queries_transfer_ub, 0, 1, self.queries_align * dim_align, 0, 0)
        # move x*d from UB to L1
        self.tik_instance.data_move(data_xd_l1, xd_transfer_ub, 0, 1, self.queries_align * dim_align, 0, 0)

        data_xdm_l0c = self.tik_instance.Tensor("float32", (self.queries_align // 16, 16, 16),
                                                name="data_xdm_l0c", scope=tik.scope_cbuf_out)
        self.tik_instance.matmul(data_xdm_l0c, data_dm_l1, queries_l1, 16, self.dim, self.queries_align)

        data_xdm_ub = self.tik_instance.Tensor("float32", (self.queries_align // 16, 16, 16),
                                               name="data_xdm_ub", scope=tik.scope_ubuf)
        self.tik_instance.data_move(data_xdm_ub, data_xdm_l0c, 0, 1, self.queries_align // 16, 0, 0)
        data_xdm_align_ub = self.tik_instance.Tensor("float32", (self.queries_align,),
                                                     name="data_xdm_align_ub", scope=tik.scope_ubuf)
        self.tik_instance.vmuls(16, data_xdm_align_ub, data_xdm_ub, self.coeff, self.queries_align // 16, 1, 1, 2, 32)
        add_scalar.set_as(data_xdm_align_ub[0])

    def compute_each_loop(self, compute_each_loop_para, accum_idx, data_xd_l1):
        dst, compute_num, max_mask, list_idx, inner_product_l0c, add_scalar = compute_each_loop_para
        code_loop_times_ = self.actual_num[accum_idx * self.code_list_num + list_idx] // self.code_each_loop
        code_loop_times = (self.actual_num[accum_idx * self.code_list_num + list_idx] + \
                                            self.code_each_loop - 1) // self.code_each_loop
        code_last = self.actual_num[accum_idx * self.code_list_num + list_idx] - \
                                                                    (code_loop_times - 1) * self.code_each_loop
        with self.tik_instance.if_scope(self.actual_num[accum_idx * self.code_list_num + list_idx] > 0):
            with self.tik_instance.if_scope(code_loop_times_ > 0):
                with self.tik_instance.for_range(0, code_loop_times_) as loop_code:
                    loop_ctrl = DisLoopCtrl(loop_code, self.code_sub_loop, self.code_sub_loop, None, accum_idx)
                    cube_compute_each_loop_para = ComputePara(None, None, None, list_idx, inner_product_l0c, add_scalar)
                    self.cube_compute_each_loop(cube_compute_each_loop_para, data_xd_l1, loop_ctrl)

            # 512 * 6 + 156 向量， 512 * 64 * 8192 = 256M
            with self.tik_instance.if_scope(code_last != self.code_each_loop):
                last_repeats = code_last // self.code_sub_loop
                last_retain = code_last % self.code_sub_loop
                code_last_num0 = last_repeats * self.code_sub_loop + (1 - last_repeats) * last_retain
                code_last_num1 = code_last - code_last_num0
                loop_ctrl = DisLoopCtrl(code_loop_times - 1, code_last_num0, code_last_num1, None, accum_idx)
                cube_compute_each_loop_para = ComputePara(None, None, None, list_idx, inner_product_l0c, add_scalar)
                self.cube_compute_each_loop(cube_compute_each_loop_para, data_xd_l1, loop_ctrl)

            loop_ctrl = DisLoopCtrl(None, None, None, code_last, accum_idx)
            cube_compute_last_para = ComputePara(None, None, None, list_idx, inner_product_l0c, add_scalar)
            self.cube_compute_last(cube_compute_last_para, code_loop_times, loop_ctrl)

    def distance_compute_(self, list_idx, accum_idx):
        add_scalar = self.tik_instance.Scalar(dtype="float32", name="add_scalar")
        data_xd_l1 = self.tik_instance.Tensor("float16",
                                              (self.dim // 16, self.queries_align, 16),
                                              name="data_xd_l1", scope=tik.scope_cbuf)
        self.calculate_add_scalar(add_scalar, data_xd_l1, accum_idx)

        # compute xy using cube
        inner_product_l0c = self.tik_instance.Tensor("float32",
                                                     (self.code_each_loop // 16, self.queries_align, 16),
                                                     name="inner_product_l0c", scope=tik.scope_cbuf_out)
        
        flag_addrs_ub = self.tik_instance.TensorAddrList(self.shape_flag_addrs[0],
                                                         name="flag_addrs_ub", scope=tik.scope_ubuf)
        self.tik_instance.data_move(flag_addrs_ub, self.output_flag_addrs_gm,
                                    0, 1, self.shape_flag_addrs[0] // 4, 0, 0)
        one = self.tik_instance.Scalar(dtype="uint16", name="one", init_value=1)
        flag_ub = self.tik_instance.Tensor("uint16", (16,), name="flag_ub", scope=tik.scope_ubuf)
        flag_ub[0].set_as(one)
        compute_each_loop_para = ComputePara(None, None, None, list_idx, inner_product_l0c, add_scalar)
        self.compute_each_loop(compute_each_loop_para, accum_idx, data_xd_l1)
        flag_addr = flag_addrs_ub[accum_idx].value + list_idx * self.output_flag_num
        self.tik_instance.pipe_barrier("PIPE_MTE3")
        self.tik_instance.data_move(flag_addr, flag_ub, 0, 1, 1, 0, 0)

    def cube_compute_each_loop(self, cube_compute_each_loop_para, data_xd_l1, loop_ctrl):
        dst, compute_num, max_mask, list_idx, inner_product_l0c, add_scalar = cube_compute_each_loop_para
        code_loop, code_num0, code_num1, code_last, accum_idx = loop_ctrl
        dst_ub = self.tik_instance.Tensor("float16", (self.code_each_loop,), name="dst_ub", scope=tik.scope_ubuf)
        dst_max_ub = self.tik_instance.Tensor("float16", (self.maxs_each_loop,),
                                              name="dst_max_ub", scope=tik.scope_ubuf)

        with self.tik_instance.for_range(0, self.code_each_loops, thread_num=2) as loop_ub:
            # MTE2: move y from out to UB
            code_ub = self.tik_instance.Tensor("uint8", (self.dim // 16, self.code_sub_loop // 16, 16, 16),
                                               name="code_ub", scope=tik.scope_ubuf)

            code_sub_num = (1 - loop_ub) * code_num0 + loop_ub * code_num1
            with self.tik_instance.if_scope(code_sub_num != 0):
                with self.tik_instance.for_range(0, self.dim // 16) as i:
                    base_offset = self.list_offset[self.code_list_num * accum_idx + list_idx] +\
                                  (code_loop * self.code_each_loop + loop_ub * self.code_sub_loop) * self.dim
                    self.tik_instance.data_move(code_ub[i, 0, 0, 0], self.input_base_list_gm[base_offset + i *\
                        self.cube_align * self.cube_align], 0, (code_sub_num + 15) // 16, 8, self.dim // 2 - 8, 0)

            # VECTOR: y do conv from uint8 to fp16
            code_ub_fp16 = self.tik_instance.Tensor("float16", (self.dim // 16, self.code_sub_loop // 16, 16, 16),
                                                    name="code_ub_fp16", scope=tik.scope_ubuf)
            vconv_repeat_times = min(self.code_sub_loop * self.dim // self.fp16_mask, self.repeat_times)
            total_code = self.code_sub_loop * self.dim
            vconv_repeat_loops = ((total_code // self.fp16_mask) + vconv_repeat_times - 1) // vconv_repeat_times
            remain_code = total_code
            for i in range(0, vconv_repeat_loops):
                repeat = min(remain_code // self.fp16_mask, vconv_repeat_times)
                self.tik_instance.vconv(self.fp16_mask, "none", code_ub_fp16[i * self.fp16_mask * self.repeat_times],
                                        code_ub[i * self.fp16_mask * self.repeat_times], repeat, 1, 1, 8, 4)
                remain_code -= self.fp16_mask * repeat

            # VECTOR: mov xy from L0-C to UB
            inner_product_ub = self.tik_instance.Tensor("float32", (self.code_sub_loop // 16, self.queries_align, 16),
                                                        name="inner_product_ub", scope=tik.scope_ubuf)
            self.tik_instance.data_move(inner_product_ub, inner_product_l0c[(self.code_each_loops + loop_ub - 2) %\
                                        self.code_each_loops * self.code_sub_loop // 16, 0, 0], 0, 1,
                                        self.code_sub_loop * self.queries_align // 256, 0, 0)

            # VECTOR: compute distance each query
            dst_ub_fp32 = self.tik_instance.Tensor("float32", (self.code_sub_loop,),
                                                   name="dst_ub_fp32", scope=tik.scope_ubuf)
            with self.tik_instance.if_scope(code_loop != 0):
                self.tik_instance.vmuls(16, dst_ub_fp32, inner_product_ub, self.coeff,
                                        self.code_sub_loop // 16, 1, 1, 2, self.queries_align * 2)

                repeat_times = self.code_sub_loop // self.fp32_mask
                self.tik_instance.vadds(self.fp32_mask, dst_ub_fp32, dst_ub_fp32, add_scalar, repeat_times, 1, 1, 8, 8)

                self.tik_instance.vconv(self.fp32_mask, "none", dst_ub[(loop_ub % 2) * self.code_sub_loop:],
                                        dst_ub_fp32, repeat_times, 1, 1, 4, 8)
                # the loop not enough self.code_each_loop, wo not use
                vcmax_repeat_times = self.code_sub_loop // self.max_batch
                self.tik_instance.vcmax(self.max_batch, dst_max_ub[(loop_ub % 2) * self.maxs_sub_loop:],
                        dst_ub[(loop_ub % 2) * self.code_sub_loop:], vcmax_repeat_times, 1, 1, self.max_batch // 16)
            # MTE3: move y from ub to L1
            code_l1 = self.tik_instance.Tensor("float16", (self.dim // 16, self.code_sub_loop, 16),
                                               name="code_l1", scope=tik.scope_cbuf)
            self.tik_instance.data_move(code_l1, code_ub_fp16, 0, self.dim // 16, self.code_sub_loop, 0, 0)

            self.tik_instance.matmul(inner_product_l0c[(loop_ub * self.code_sub_loop * self.queries_align):],
                                     data_xd_l1, code_l1, self.queries_align, self.dim, self.code_sub_loop)

            with self.tik_instance.if_scope(loop_ub % 2 * code_loop != 0):
                dist_gm_addr = self.distance_addrs_ub[accum_idx].value + list_idx * self.output_dist_num
                max_gm_addr = self.max_addrs_ub[accum_idx].value + list_idx * self.output_max_num
                self.tik_instance.data_move(dist_gm_addr + code_loop * self.code_each_loop +\
                    (loop_ub - 3) * self.code_sub_loop, dst_ub, 0, 1, self.code_each_loop // 16, 0, 0)
                self.tik_instance.data_move(max_gm_addr + code_loop * self.maxs_each_loop + (loop_ub - 3) *\
                    self.maxs_sub_loop, dst_max_ub, 0, 1, self.maxs_each_loop // 16, 0, 0)

    def cube_compute_last(self, cube_compute_last_para, code_loop_times, loop_ctrl):
        dst, compute_num, max_mask, list_idx, inner_product_l0c, add_scalar = cube_compute_last_para
        code_last = loop_ctrl.code_last
        accum_idx = loop_ctrl.accum_idx
        code_last_loops = code_last // self.code_sub_loop

        dst_max_ub = self.tik_instance.Tensor("float16", (self.maxs_each_loop,),
                                              name="dst_max_ub", scope=tik.scope_ubuf)

        with self.tik_instance.for_range(0, self.code_each_loops, thread_num=2) as loop_ub:
            inner_product_ub = self.tik_instance.Tensor("float32", (self.code_sub_loop // 16, self.queries_align, 16),
                                                        name="inner_product_ub", scope=tik.scope_ubuf)
            self.tik_instance.data_move(inner_product_ub,
                                        inner_product_l0c[loop_ub * self.code_sub_loop // 16, 0, 0],
                                        0, 1, self.code_sub_loop * self.queries_align // 256, 0, 0)

            dst_ub_fp32 = self.tik_instance.Tensor("float32", (self.code_sub_loop,),
                                                   name="dst_ub_fp32", scope=tik.scope_ubuf)
            self.tik_instance.vmuls(16, dst_ub_fp32, inner_product_ub, self.coeff,
                                    self.code_sub_loop // 16, 1, 1, 2, self.queries_align * 2)

            dst_ub = self.tik_instance.Tensor("float16", (self.code_sub_loop,), name="dst_ub", scope=tik.scope_ubuf)
            repeat_times = self.code_sub_loop // self.fp32_mask
            self.tik_instance.vadds(self.fp32_mask, dst_ub_fp32, dst_ub_fp32, add_scalar, repeat_times, 1, 1, 8, 8)
            self.tik_instance.vconv(self.fp32_mask, "none", dst_ub, dst_ub_fp32, repeat_times, 1, 1, 4, 8)

            with self.tik_instance.if_scope(code_last_loops > loop_ub):
                vcmax_repeat_times = self.code_sub_loop // self.max_batch
                self.tik_instance.vcmax(self.max_batch, dst_max_ub[loop_ub * self.maxs_sub_loop], dst_ub,
                                        vcmax_repeat_times, 1, 1, self.max_batch // 16)
            
            with self.tik_instance.else_scope():
                vcmax_repeat_times = (code_last - self.code_sub_loop * code_last_loops) // self.max_batch
                with self.tik_instance.if_scope(vcmax_repeat_times > 0):
                    self.tik_instance.vcmax(self.max_batch, dst_max_ub[loop_ub * self.maxs_sub_loop], dst_ub,
                                            vcmax_repeat_times, 1, 1, self.max_batch // 16)
                code_last_retain = code_last % self.max_batch
                with self.tik_instance.if_scope(code_last_retain > 0):
                    self.tik_instance.vcmax(code_last_retain, dst_max_ub[loop_ub * self.maxs_sub_loop +\
                                            self.maxs_last_offset[accum_idx * self.code_list_num + list_idx]],
                                            dst_ub[self.code_last_offset[accum_idx * self.code_list_num + list_idx]],
                                            1, 1, 1, 1)
            dist_gm_addr = self.distance_addrs_ub[accum_idx].value + list_idx * self.output_dist_num
            self.tik_instance.data_move(dist_gm_addr + (code_loop_times - 1) * self.code_each_loop +
                                        loop_ub * self.code_sub_loop, dst_ub, 0, 1, self.code_sub_loop // 16, 0, 0)
        max_gm_addr = self.max_addrs_ub[accum_idx].value + list_idx * self.output_max_num
        self.tik_instance.data_move(max_gm_addr + (code_loop_times - 1) * self.code_each_loop // self.max_batch * 2,
                                    dst_max_ub, 0, 1, self.maxs_each_loop // 16, 0, 0)

    def distance_compute_loop(self, accum_idx):
        with self.tik_instance.for_range(0, self.aicore_use, block_num=self.aicore_use) as block_idx:
            for list_idx in range(self.aicore_use):
                with self.tik_instance.if_scope(list_idx == block_idx):
                    # compute coarse centroids num and move offset every core
                    self.distance_compute_(list_idx, accum_idx)

    def distance_compute(self):
        """
        the compute process
        """
        self.cal_num_each_core()
        query_offset_ub = self.tik_instance.Tensor("uint64", (self.shape_query_offet[0], ),
                                                   name="query_offset_ub", scope=tik.scope_ubuf)
        self.tik_instance.data_move(query_offset_ub, self.input_query_offset_gm,
                                    0, 1, self.shape_query_offet[0] // 4, 0, 0)
        with self.tik_instance.for_range(0, self.accum_num) as accum_idx:
            self.query_offset[accum_idx].set_as(query_offset_ub[accum_idx])
            self.distance_compute_loop(accum_idx)
            

    def get_tik_instance(self):
        """
        obtain tik instance
        """
        

        self.tik_instance.BuildCCE(
            kernel_name=self.kernel_name,
            inputs=[
                self.input_queries_gm,
                self.input_base_list_gm,
                self.input_offset_addrs_gm,
                self.input_dm_gm,
                self.input_actual_num_addrs_gm,
                self.input_accum_num_gm,
                self.input_query_offset_gm
            ],
            outputs=[self.output_distances_addrs_gm, self.output_maxs_addrs_gm, self.output_flag_addrs_gm])

        return self.tik_instance

    def _muls(self, _muls_para, src, scalar):
        dst, compute_num, max_mask, list_idx, inner_product_l0c, add_scalar = _muls_para
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

    def _add(self, _add_para, src0, src1):
        dst, compute_num, max_mask, list_idx, inner_product_l0c, add_scalar = _add_para
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


def distance_ivf_sq8_ip8_accum(input_queries,
                               input_base_list,
                               input_offset_addrs,
                               input_dm,
                               input_actual_num_addrs,
                               input_accum_num,
                               input_query_offset,
                               output_distances_addrs,
                               output_maxs_addrs,
                               output_flag_addrs,
                               kernel_name="distance_ivf_sq8_ip8_accum"):
    distance_op = DistanceIVFSQ8IP8Accum(input_queries, input_base_list, input_offset_addrs, input_dm, 
                                         input_actual_num_addrs, input_accum_num, input_query_offset,
                                         output_distances_addrs, output_maxs_addrs, output_flag_addrs, kernel_name)
    tik_instance = distance_op.get_tik_instance()
    return tik_instance
