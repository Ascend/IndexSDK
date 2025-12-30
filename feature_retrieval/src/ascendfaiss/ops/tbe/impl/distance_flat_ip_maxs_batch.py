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


class DistanceFlatIPMaxs:
    def __init__(self,
                 input_queries,
                 input_mask,
                 input_code,
                 input_actual_num,
                 output_distances,
                 output_maxs,
                 output_flag,
                 kernel_name="distance_flat_ip_batch"):
        self.shape_queries = input_queries.get("shape")
        self.dtype_queries = input_queries.get("dtype")
        self.shape_mask = input_mask.get("shape")
        self.dtype_mask = input_mask.get("dtype")
        self.shape_code = input_code.get("shape")
        self.dtype_code = input_code.get("dtype")
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
        self.code_num = self.shape_code[0] * self.shape_code[2]
        self.code_batch = self.shape_code[2]

        # check parameter
        self.check_parameter()

        self.block_offset = None
        self.mask_len = None
        self.use_mask = None

        # set vector fp32 mask and fp16 mask
        self.fp32_mask = 64
        self.fp16_mask = 128
        self.max_batch = 32

        self.matmul_thread_num = 1 if self.dim > 2048 else 2
        self.code_div_num = 2
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
        self.tik_instance = tik.Tik(disable_debug=False)

        # if nq为64, dim{32, 64, 128}, code_num_each_loop为1024，else 512
        if self.dim == 4096 or self.dim == 3584:
            query_min_each_loop = 64
            code_num_base = 512
        elif self.dim == 3072:
            query_min_each_loop = 96
            code_num_base = 512
        else:
            query_min_each_loop = 128
            code_num_base = 512
        self.queries_num_each_loop = min(self.queries_num, query_min_each_loop)
        self.code_num_each_loop = min(128 // 16 // ((self.queries_num_each_loop + 15) // 16) * code_num_base,
                                      (1 + 256 // (128 + self.dim)) * code_num_base)

        self.set_src_dst_tensor()

    def set_src_dst_tensor(self):
        """
        set input and output tensor
        """

        self.default_scalar = self.tik_instance.Scalar("float16",
                                                       name="default_scalar",
                                                       init_value=-65504)
        # creat input tensor: input_queries_gm, input_code_gm, input_actual_num_gm
        # and output tensor: output_dist_gm, output_flag_gm in global buffer
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
        self.input_actual_num_gm = self.tik_instance.Tensor(
            self.dtype_actual_num,
            self.shape_actual_num,
            name="input_actual_num_gm",
            scope=tik.scope_gm)
        self.output_distances_gm = self.tik_instance.Tensor(
            self.dtype_distances,
            self.shape_distances,
            name="output_dist_gm",
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
        actual_num_ub = self.tik_instance.Tensor("uint32",
                                                 (8,),
                                                 name="actual_num_ub",
                                                 scope=tik.scope_ubuf)
        self.tik_instance.data_move(actual_num_ub,
                                    self.input_actual_num_gm,
                                    0, 1, 1, 0, 0)
        actual_num = self.tik_instance.Scalar(dtype="uint32",
                                              name="actual_code_num",
                                              init_value=0)
        actual_num.set_as(actual_num_ub[0])
        
        self.block_offset = self.tik_instance.Scalar(dtype="uint32",
                                                    name="block_offset",
                                                    init_value=0)
        self.block_offset.set_as(actual_num_ub[1])
 
        self.mask_len = self.tik_instance.Scalar(dtype="uint32",
                                                 name="mask_len",
                                                 init_value=0)
        self.mask_len.set_as(actual_num_ub[2])

        self.use_mask = self.tik_instance.Scalar(dtype="uint32",
                                                 name="use_mask",
                                                 init_value=0)
        self.use_mask.set_as(actual_num_ub[3])

        self.aicore_use = self.shape_flag[0]

        if self.aicore_use == 2:
            self.code_num_each_core = \
                (actual_num // self.aicore_use + self.max_batch * 8) // self.max_batch // 16 * self.max_batch * 16
        else:
            self.code_num_each_core = actual_num // self.aicore_use // self.max_batch // 16 * self.max_batch * 16
        self.code_num_last_core = actual_num - (self.aicore_use - 1) * self.code_num_each_core

    def distance_compute_each_loop(self,
                                   aicore_move_offset,
                                   aicore_code_num,
                                   move_offset,
                                   move_num):
        queries_align = (move_num + 15) // 16 * 16
        queries_l1 = self.tik_instance.Tensor("float16",
                                              (self.dim // 16,
                                               queries_align, 16),
                                              name="queries_l1",
                                              scope=tik.scope_cbuf)

        with self.tik_instance.new_stmt_scope():
            with self.tik_instance.for_range(0, move_num) as i:
                self.tik_instance.data_move(queries_l1[0, i, 0],
                                            self.input_queries_gm[move_offset + i, 0],
                                            0, self.dim // 16, 1, 0,
                                            queries_align - 1)
        # compute xy using cube
        code_loop_time = aicore_code_num // self.code_num_each_loop
        with self.tik_instance.if_scope(code_loop_time > 0):
            with self.tik_instance.for_range(0, code_loop_time) as loop_code:
                self.cube_compute_each_loop(queries_l1, aicore_move_offset,
                                            loop_code * self.code_num_each_loop,
                                            self.code_num_each_loop, move_offset,
                                            move_num, 0)

        code_last_num = aicore_code_num % self.code_num_each_loop
        with self.tik_instance.if_scope(code_last_num > 0):
            self.cube_compute_each_loop(queries_l1, aicore_move_offset,
                                        code_loop_time * self.code_num_each_loop,
                                        code_last_num, move_offset, move_num, 1)


    def do_mask(self, dst_ub, queries_move_num, code_out_num, aicore_move_offset, code_move_offset,
                i, queries_move_offset):
        min_val_ub = self.tik_instance.Tensor("float16", (128,), name="min_val_ub", scope=tik.scope_ubuf)
        self.tik_instance.vec_dup(self.fp16_mask, min_val_ub, self.default_scalar, 1, 8)

        # malloc memory on chip
        sel_ub = self.tik_instance.Tensor("uint8", (queries_move_num, (code_out_num + 7) // 8),
                                            name="sel_ub", scope=tik.scope_ubuf)
        mask_offset_of_code = aicore_move_offset + code_move_offset + i * code_out_num
        with self.tik_instance.for_range(0, queries_move_num) as j:
            mask_offset_of_query = (j + queries_move_offset) * self.mask_len
            # move data from input_mask_gm to sel_ub
            # 搬运的burst为(code_out_num + 255) // 256，等同于((code_out_num + 7) // 8 + 31) // 32  # 按照8对齐，再按照32对齐
            self.tik_instance.data_move(
                sel_ub[j, 0],
                self.input_mask_gm[mask_offset_of_query + (self.block_offset + mask_offset_of_code) // 8],
                0, 1, (code_out_num + 255) // 256, 8, 8)
            # cal the loop need to execute the selection process
            vsel_loop = code_out_num // self.fp16_mask
            for vloop in range(vsel_loop):
                voffset = vloop * self.fp16_mask
                self.tik_instance.vec_sel(self.fp16_mask, 0, dst_ub[j, voffset],
                                            sel_ub[j, voffset // 8], dst_ub[j, voffset],
                                            min_val_ub, 1, 8, 8, 0)


    def cube_compute_each_loop(self,
                               queries_l1,
                               aicore_move_offset,
                               code_move_offset,
                               code_move_num,
                               queries_move_offset,
                               queries_move_num,
                               flag):
        queries_align = (queries_move_num + 15) // 16 * 16
        inner_product_l0c = self.tik_instance.Tensor("float32",
                                                     (self.code_num_each_loop // 16,
                                                      queries_align, 16),
                                                     name="inner_product_l0c",
                                                     scope=tik.scope_cbuf_out)
        code_move_num_ub = (code_move_num + 15) // 16 * 16
        code_num_ub_each_loop = self.code_num_each_loop // 8
        vcmax_loops = 8
        if flag == 0:
            with self.tik_instance.for_range(0, 8, thread_num=self.matmul_thread_num) as loop_ub:
                # move y from out to UB
                code_l1 = self.tik_instance.Tensor("float16",
                                                   (self.dim // 16, code_num_ub_each_loop, 16),
                                                   name="code_l1", scope=tik.scope_cbuf)
                with self.tik_instance.for_range(0, self.dim // 16) as i:
                    self.tik_instance.data_move(
                        code_l1[i, 0, 0],
                        self.input_code_gm[(aicore_move_offset +
                                            code_move_offset +
                                            loop_ub * code_num_ub_each_loop) // self.code_batch,
                        i, 0, 0],
                        0, code_num_ub_each_loop // self.code_batch,
                        self.code_batch,
                           (self.dim - 16) * self.code_batch * 2 // 32, 0)

                self.tik_instance.matmul(
                    inner_product_l0c[(loop_ub * code_num_ub_each_loop *
                                       queries_align):],
                    queries_l1, code_l1,
                    queries_align, self.dim,
                    code_num_ub_each_loop)
        else:
            code_loop = code_move_num_ub // code_num_ub_each_loop
            with self.tik_instance.for_range(0, code_loop, thread_num=self.matmul_thread_num) as loop_ub:
                code_l1 = self.tik_instance.Tensor("float16",
                                                   (self.dim // 16, code_num_ub_each_loop, 16),
                                                   name="code_l1", scope=tik.scope_cbuf)
                with self.tik_instance.for_range(0, self.dim // 16) as i:
                    self.tik_instance.data_move(code_l1[i, 0, 0],
                                                self.input_code_gm[(aicore_move_offset + code_move_offset +
                                                loop_ub * code_num_ub_each_loop) // self.code_batch,
                                                i, 0, 0],
                                                0, code_num_ub_each_loop // self.code_batch,
                                                self.code_batch, (self.dim - 16) * self.code_batch * 2 // 32, 0)
                self.tik_instance.matmul(
                    inner_product_l0c[(loop_ub * code_num_ub_each_loop * queries_align):],
                    queries_l1, code_l1,
                    queries_align, self.dim, code_num_ub_each_loop)
            code_last = code_move_num_ub % code_num_ub_each_loop
            with self.tik_instance.if_scope(code_last > 0):
                code_l1 = self.tik_instance.Tensor("float16",
                                                   (self.dim // 16, code_num_ub_each_loop, 16),
                                                   name="code_l1", scope=tik.scope_cbuf)
                with self.tik_instance.for_range(0, self.dim // 16) as i:
                    self.tik_instance.data_move(
                        code_l1[i, 0, 0],
                        self.input_code_gm[(aicore_move_offset +
                                            code_move_offset +
                                            code_loop * code_num_ub_each_loop) // self.code_batch,
                        i, 0, 0],
                        0, code_num_ub_each_loop // self.code_batch,
                        self.code_batch,
                           (self.dim - 16) * self.code_batch * 2 // 32, 0)
                self.tik_instance.matmul(
                    inner_product_l0c[(code_loop *
                                       code_num_ub_each_loop * queries_align):],
                    queries_l1, code_l1,
                    queries_align, self.dim, code_num_ub_each_loop)

        # code_out_num为256 or 512
        code_out_num = self.code_num_each_loop // self.code_div_num
        if flag == 0:
            with self.tik_instance.for_range(0, 2, thread_num=1) as i:
                # mov xy from L0-C to UB
                inner_product_l1 = self.tik_instance.Tensor("float32",
                                                            (code_out_num // 16,
                                                             queries_align, 16),
                                                            name="inner_product_l1",
                                                            scope=tik.scope_cbuf)
                with self.tik_instance.new_stmt_scope():
                    inner_product_ub = self.tik_instance.Tensor("float32",
                                                                (code_out_num // 16,
                                                                 queries_align, 16),
                                                                name="inner_product_ub",
                                                                scope=tik.scope_ubuf)
                    self.tik_instance.data_move(inner_product_ub,
                                                inner_product_l0c[i * code_out_num // 16, 0, 0],
                                                0, 1, code_out_num * queries_align // 256, 0, 0)
                    self.tik_instance.data_move(inner_product_l1,
                                                inner_product_ub,
                                                0, 1, code_out_num * queries_align // 8, 0, 0)
                add_ub = self.tik_instance.Tensor("float32",
                                                  (queries_move_num, code_out_num),
                                                  name="add_ub", scope=tik.scope_ubuf)
                with self.tik_instance.for_range(0,
                                                 queries_move_num) as loop_queries:
                    self.tik_instance.data_move(add_ub[loop_queries, 0],
                                                inner_product_l1[0, loop_queries, 0],
                                                0, code_out_num // 16, 2,
                                                queries_align * 2 - 2, 0)

                dst_ub = self.tik_instance.Tensor("float16",
                                                  (queries_move_num, code_out_num),
                                                  name="dst_ub", scope=tik.scope_ubuf)
                vconv_loops = 4 if queries_move_num > 1 else 1
                query_num_each_loop = queries_move_num // vconv_loops
                vconv_repeat_time = query_num_each_loop * code_out_num // self.fp32_mask

                with self.tik_instance.for_range(0, vconv_loops,
                                                 thread_num=2) as vconv_loop:
                    self.tik_instance.vconv(self.fp32_mask, "none",
                                            dst_ub[query_num_each_loop * vconv_loop, 0],
                                            add_ub[query_num_each_loop * vconv_loop, 0],
                                            vconv_repeat_time, 1, 1, 4, 8)
                    
                with self.tik_instance.if_scope(self.use_mask > 0):
                    self.do_mask(dst_ub, queries_move_num, code_out_num, aicore_move_offset, code_move_offset,
                                 i, queries_move_offset)
                    

                self.tik_instance.data_move(
                    self.output_distances_gm[queries_move_offset,
                    aicore_move_offset + code_move_offset +
                    i * code_out_num],
                    dst_ub,
                    0, queries_move_num, code_out_num // 16, 0,
                                         (self.code_num - code_out_num) // 16)

                max_size = code_out_num // self.max_batch * 2
                dst_max_ub = self.tik_instance.Tensor("float16",
                                                      (queries_move_num, max_size),
                                                      name="dst_max_ub",
                                                      scope=tik.scope_ubuf)
                query_num_each_loop = queries_move_num // vcmax_loops
                vcmax_repeat_times = query_num_each_loop * code_out_num // self.max_batch
                with self.tik_instance.for_range(0, vcmax_loops,
                                                 thread_num=2) as vcmax_loop:
                    self.tik_instance.vcmax(self.max_batch,
                                            dst_max_ub[vcmax_loop * query_num_each_loop, 0],
                                            dst_ub[vcmax_loop * query_num_each_loop, 0],
                                            vcmax_repeat_times, 1, 1, self.max_batch // 16)
                self.tik_instance.data_move(
                    self.output_maxs_gm[queries_move_offset,
                    (aicore_move_offset + code_move_offset +
                     i * code_out_num) // self.max_batch * 2],
                    dst_max_ub,
                    0, queries_move_num,
                    (code_out_num // self.max_batch) // 8,
                    0, (self.code_num - code_out_num) // self.max_batch // 8)
        else:
            code_loops = (code_move_num_ub + code_out_num - 1) // code_out_num
            with self.tik_instance.for_range(0, code_loops, thread_num=1) as i:
                # mov xy from L0-C to UB
                inner_product_l1 = self.tik_instance.Tensor("float32",
                                                            (code_out_num // 16, queries_align, 16),
                                                            name="inner_product_l1",
                                                            scope=tik.scope_cbuf)
                with self.tik_instance.new_stmt_scope():
                    inner_product_ub = self.tik_instance.Tensor("float32",
                                                                (code_out_num // 16, queries_align, 16),
                                                                name="inner_product_ub",
                                                                scope=tik.scope_ubuf)
                    self.tik_instance.data_move(inner_product_ub,
                                                inner_product_l0c[i * code_out_num // 16, 0, 0],
                                                0, 1, code_out_num * queries_align // 256, 0, 0)
                    self.tik_instance.data_move(inner_product_l1,
                                                inner_product_ub,
                                                0, 1, code_out_num *
                                                queries_align // 8, 0, 0)
                add_ub = self.tik_instance.Tensor("float32",
                                                  (queries_move_num, code_out_num),
                                                  name="add_ub", scope=tik.scope_ubuf)
                with self.tik_instance.for_range(0, queries_move_num) as loop_queries:
                    self.tik_instance.data_move(add_ub[loop_queries, 0],
                                                inner_product_l1[0, loop_queries, 0],
                                                0, code_out_num // 16, 2,
                                                queries_align * 2 - 2, 0)

                dst_ub = self.tik_instance.Tensor("float16",
                                                  (queries_move_num, code_out_num),
                                                  name="dst_ub", scope=tik.scope_ubuf)
                vconv_loops = 4 if queries_move_num > 1 else 1
                query_num_each_loop = queries_move_num // vconv_loops
                vconv_repeat_time = query_num_each_loop * code_out_num // self.fp32_mask
                with self.tik_instance.for_range(0, vconv_loops,
                                                 thread_num=vconv_loops) as vconv_loop:
                    self.tik_instance.vconv(self.fp32_mask, "none",
                                            dst_ub[query_num_each_loop * vconv_loop, 0],
                                            add_ub[query_num_each_loop * vconv_loop, 0],
                                            vconv_repeat_time, 1, 1, 4, 8)

                with self.tik_instance.if_scope(self.use_mask > 0):
                    self.do_mask(dst_ub, queries_move_num, code_out_num, aicore_move_offset, code_move_offset,
                                 i, queries_move_offset)

                self.tik_instance.data_move(
                    self.output_distances_gm[queries_move_offset,
                    aicore_move_offset + code_move_offset
                    + i * code_out_num],
                    dst_ub,
                    0, queries_move_num, code_out_num // 16, 0,
                                         (self.code_num - code_out_num) // 16)

                max_size = code_out_num // self.max_batch * 2
                dst_max_ub = self.tik_instance.Tensor("float16",
                                                      (queries_move_num, max_size),
                                                      name="dst_max_ub", scope=tik.scope_ubuf)
                with self.tik_instance.if_scope(i == code_loops - 1):
                    code_last_num = code_move_num - i * code_out_num
                    thread_num = 2 if queries_move_num > 1 else 1
                    with self.tik_instance.for_range(0, queries_move_num,
                                                     thread_num=thread_num) as query:
                        vcmax_repeat_times = code_last_num // self.max_batch
                        offset = 0
                        with self.tik_instance.if_scope(vcmax_repeat_times > 0):
                            self.tik_instance.vcmax(self.max_batch,
                                                    dst_max_ub[query, 0],
                                                    dst_ub[query, 0],
                                                    vcmax_repeat_times, 1, 1,
                                                    self.max_batch // 16)
                            offset += vcmax_repeat_times * self.max_batch
                        vcmax_remain = code_last_num % self.max_batch
                        with self.tik_instance.if_scope(vcmax_remain > 0):
                            self.tik_instance.vcmax(vcmax_remain,
                                                    dst_max_ub[query, vcmax_repeat_times * 2],
                                                    dst_ub[query, offset],
                                                    1, 1, 1, 1)
                        self.tik_instance.data_move(
                            self.output_maxs_gm[queries_move_offset,
                            (aicore_move_offset + code_move_offset
                             + i * code_out_num) // self.max_batch * 2],
                            dst_max_ub,
                            0, queries_move_num, code_out_num // self.max_batch // 8,
                            0, (self.code_num - code_out_num) // self.max_batch // 8)
                with self.tik_instance.else_scope():
                    query_num_each_loop = queries_move_num // vcmax_loops
                    vcmax_repeat_times = query_num_each_loop * code_out_num // self.max_batch
                    with self.tik_instance.for_range(0, vcmax_loops,
                                                     thread_num=2) as vcmax_loop:
                        self.tik_instance.vcmax(self.max_batch,
                                                dst_max_ub[vcmax_loop * query_num_each_loop, 0],
                                                dst_ub[vcmax_loop * query_num_each_loop, 0],
                                                vcmax_repeat_times, 1, 1, self.max_batch // 16)
                    self.tik_instance.data_move(self.output_maxs_gm[queries_move_offset,
                    (aicore_move_offset + code_move_offset +
                     i * code_out_num) // self.max_batch * 2],
                                                dst_max_ub,
                                                0, queries_move_num, (code_out_num // self.max_batch) // 8,
                                                0, (self.code_num - code_out_num) // self.max_batch // 8)

    def distance_compute(self):
        """
        the compute process
        """
        self.cal_num_each_core()

        with self.tik_instance.for_range(0, self.aicore_use,
                                         block_num=self.aicore_use) as block_index:
            aicore_code_num = self.tik_instance.Scalar(dtype="uint32",
                                                       name="aicore_code_num",
                                                       init_value=0)
            # compute coarse code num and move offset every core
            with self.tik_instance.if_scope(block_index != self.aicore_use - 1):
                aicore_code_num.set_as(self.code_num_each_core)
            with self.tik_instance.else_scope():
                aicore_code_num.set_as(self.code_num_last_core)

            queries_loop_time = self.queries_num // self.queries_num_each_loop
            if queries_loop_time > 0:
                with self.tik_instance.for_range(0, queries_loop_time) as loop_queries:
                    self.distance_compute_each_loop(block_index * self.code_num_each_core,
                                                    aicore_code_num,
                                                    loop_queries * self.queries_num_each_loop,
                                                    self.queries_num_each_loop)

            queries_last_num = self.queries_num % self.queries_num_each_loop
            if queries_last_num > 0:
                self.distance_compute_each_loop(block_index * self.code_num_each_core,
                                                aicore_code_num,
                                                queries_loop_time * self.queries_num_each_loop,
                                                queries_last_num)

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
                                       self.input_code_gm,
                                       self.input_actual_num_gm
                                   ],
                                   outputs=[
                                       self.output_distances_gm,
                                       self.output_maxs_gm,
                                       self.output_flag_gm
                                   ])

        return self.tik_instance


def distance_flat_ip_maxs_batch(input_queries,
                                input_mask,
                                input_code,
                                input_actual_num,
                                output_distances,
                                output_maxs,
                                output_flag,
                                kernel_name="distance_flat_ip_batch"):
    """
    calculating distance

    Parameters
    ----------
    input_queries : dict
        shape and dtype of query vector
    input_mask : dict
        shape and dtype of mask vector
    input_code : dict
        shape and dtype of coarse code
    input_actual_num: dict
        shape and dtype of actual code num,
        shape must be (8,) and dtype must be uint32
    output_distances : dict
        shape and dtype of distances, should be same dtype as input_queries
    output_maxs : dict
        shape and dtype of maxs
    output_flag : dict
        shape and dtype of flag, only the 1th and 17th is valid
    kernel_name : str
        kernel name, default value is "distance_flat_ip"

    Returns
    -------
    None
    """
    distance_flat = DistanceFlatIPMaxs(input_queries,
                                       input_mask,
                                       input_code,
                                       input_actual_num,
                                       output_distances,
                                       output_maxs,
                                       output_flag,
                                       kernel_name)

    tik_instance = distance_flat.get_tik_instance()
    return tik_instance
