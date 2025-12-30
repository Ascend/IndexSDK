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

MIN_FP16        = -65500.0
 
 
def div_up(dividend_val, divisor_val):
    return ((dividend_val + divisor_val - 1) // divisor_val)
 
 
def filter_distance_using_mask_in_ub(tik_instance, input_ub, input_cnt, output_ub, mask, value):
    """
    根据Mask值对input_ub Tensor中的每个元素值进行更新。
    Args:
        input_ub：输入的UB Tensor，一维, dtype=float16
        input_cnt：要处理的输入数据元素数量
        output_ub：输出的UB Tensor，一维，dtype=float16
        mask：mask值是一个Tensor，一维，dtype=uint8，bit=1时，使用input_ub中元素值赋值给output_ub，bit=0时，使用value赋值给output_ub。
        value：立即数或者Scalar变量
    约束：
        元素类型须为float16，input_cnt必须小于等于input_ub Tensor的长度。
    """
    max_repeat_each_call    = 128
    total_repeat            = input_cnt // 128
    
    tmp_repeat_sc           = tik_instance.Scalar(dtype="uint32")
    call_cnt                = div_up(total_repeat, max_repeat_each_call)
    with tik_instance.for_range(0, call_cnt) as cid:
        with tik_instance.if_scope(cid != (call_cnt - 1)):
            tmp_repeat_sc.set_as(max_repeat_each_call)
        with tik_instance.else_scope():
            tmp_repeat_sc.set_as(total_repeat - (call_cnt - 1) * max_repeat_each_call)
        tik_instance.vsel(128,
                          1,
                          output_ub[cid * max_repeat_each_call * 128 :],
                          mask[cid * max_repeat_each_call * 128 // 8 :],
                          input_ub[cid * max_repeat_each_call * 128 :],
                          value,
                          tmp_repeat_sc, 
                          1, 1, 1, 8, 8, 8)
 
    last_remain = input_cnt % 128
    with tik_instance.if_scope(last_remain > 0):
        tik_instance.vsel(last_remain,
                          1,
                          output_ub[total_repeat * 128 :],
                          mask[total_repeat * 128 // 8 :],
                          input_ub[total_repeat * 128 :],
                          value,
                          1,
                          1, 1, 1, 8, 8, 8)
 
 
class DistanceFlatIPMaxs:
    def __init__(self,
                 input_queries,
                 input_code,
                 input_actual_num,
                 input_mask,
                 output_distances,
                 output_maxs,
                 output_flag,
                 kernel_name="distance_flat_ip"):
        self.shape_queries = input_queries.get("shape")
        self.dtype_queries = input_queries.get("dtype")
        self.shape_code = input_code.get("shape")
        self.dtype_code = input_code.get("dtype")
        self.shape_actual_num = input_actual_num.get("shape")
        self.dtype_actual_num = input_actual_num.get("dtype")
        self.use_extra_score = False
        self.use_distance_mask = False
        self.use_scale = False
        if input_mask is not None:
            self.shape_input_mask = input_mask.get("shape")
            self.dtype_input_mask = input_mask.get("dtype")
            self.use_distance_mask = True
            if self.shape_input_mask[0] == 1 and self.shape_input_mask[0] != self.shape_queries[0]:
                self.share_distance_mask = True
            else:
                self.share_distance_mask = False

        self.shape_distances = output_distances.get("shape")
        self.dtype_distances = output_distances.get("dtype")
        self.shape_maxs = output_maxs.get("shape")
        self.dtype_maxs = output_maxs.get("dtype")
        self.shape_flag = output_flag.get("shape")
        self.dtype_flag = output_flag.get("dtype")
        self.kernel_name = kernel_name
        self.input_extra_score_gm = None
        self.input_scale_gm = None
        self.block_offset = None
        self.extra_score_len = None
 
        # compute parameter
        self.queries_num, self.dim = self.shape_queries
        self.code_num = self.shape_code[0] * self.shape_code[2]
        self.code_batch = self.shape_code[2]
 
        # check parameter
        self.check_parameter()
 
        # set vector fp32 mask and fp16 mask
        self.fp32_mask = 64
        self.fp16_mask = 128
        self.max_batch = 64
 
        # set tik instance
        self.set_tik_instance()

    def set_scale(self, input_scale):
        self.shape_input_scale = input_scale.get("shape")
        self.dtype_input_scale = input_scale.get("dtype")
        self.use_scale = True

    def set_extra_score(self, extra_score):
        self.shape_extra_score = extra_score.get("shape")
        self.dtype_extra_score = extra_score.get("dtype")
        self.use_extra_score = True

    def check_parameter(self):
        if self.dim % 16 != 0:
            raise RuntimeError("feature dim must be a multiple of 16")
        if self.code_num % 16 != 0:
            raise RuntimeError("code num must be a multiple of 16")
        code_batch_list = [1, 2, 4, 8, 16, 32, 64, 128]
        if self.code_batch not in code_batch_list:
            raise RuntimeError("code batch must in [1, 2, 4, 8, 16, 32, 64, 128]")
 
    def set_tik_instance(self):
        """
        set tik_instance
        """
        tik_dprofile = tik.Dprofile("v200", "aic")
        self.tik_instance = tik.Tik(tik_dprofile, disable_debug=False)
 
        self.queries_num_each_loop = min(self.queries_num, 48)
        self.code_num_each_loop = min(48 // 16 // ((self.queries_num_each_loop + 15) // 16) * 1024,
                                      (1 + 256 // (128 + self.dim)) * 1024)  # 最小1024 最大2048
 
        self.set_src_dst_tensor()
 
    def set_src_dst_tensor(self):
        """
        set input and output tensor
        """
 
        # creat input tensor: input_queries_gm, input_code_gm, input_actual_num_gm
        # and output tensor: output_dist_gm, output_flag_gm in global buffer
        self.input_queries_gm = self.tik_instance.Tensor(self.dtype_queries,
                                                         self.shape_queries,
                                                         name="input_queries_gm", scope=tik.scope_gm)
        self.input_code_gm = self.tik_instance.Tensor(self.dtype_code,
                                                      self.shape_code,
                                                      name="input_code_gm", scope=tik.scope_gm)
        self.input_actual_num_gm = self.tik_instance.Tensor(self.dtype_actual_num,
                                                            self.shape_actual_num,
                                                            name="input_actual_num_gm", scope=tik.scope_gm)
        self.output_distances_gm = self.tik_instance.Tensor(self.dtype_distances,
                                                            self.shape_distances,
                                                            name="output_dist_gm", scope=tik.scope_gm)
        self.output_maxs_gm = self.tik_instance.Tensor(self.dtype_maxs,
                                                       self.shape_maxs,
                                                       name="output_maxs_gm", scope=tik.scope_gm)
        self.output_flag_gm = self.tik_instance.Tensor(self.dtype_flag,
                                                       self.shape_flag,
                                                       name="output_flag_gm", scope=tik.scope_gm)

    def cal_num_each_core(self):
        """
        calculate actual code num of each core
        """
        # move actual code num from out to UB
        actual_num_ub = self.tik_instance.Tensor("uint32",
                                                 (8,),
                                                 name="actual_num_ub", scope=tik.scope_ubuf)
        self.tik_instance.data_move(actual_num_ub,
                                    self.input_actual_num_gm,
                                    0, 1, 1, 0, 0)
        actual_num = self.tik_instance.Scalar(dtype="uint32",
                                              name="actual_code_num",
                                              init_value=0)
        actual_num.set_as(actual_num_ub[0])
 
        self.aicore_use = self.shape_flag[0]

        if self.use_extra_score is True:
            self.block_offset = self.tik_instance.Scalar(dtype="uint32",
                                                    name="block_offset",
                                                    init_value=0)
            self.block_offset.set_as(actual_num_ub[1])
            self.extra_score_len = self.tik_instance.Scalar(dtype="uint64",
                                                    name="extra_score_len",
                                                    init_value=0)
            self.extra_score_len.set_as(actual_num_ub[2])

        if self.aicore_use == 2:
            self.code_num_each_core = \
                (actual_num // self.aicore_use + self.max_batch * 8) // self.max_batch // 16 * self.max_batch * 16
        else:
            self.code_num_each_core = actual_num // self.aicore_use // self.max_batch // 16 * self.max_batch * 16
        self.code_num_last_core = actual_num - (self.aicore_use - 1) * self.code_num_each_core
 
    def distance_compute_each_loop(self, aicore_move_offset, aicore_code_num, move_offset, move_num):
        queries_align = (move_num + 15) // 16 * 16
        queries_l1 = self.tik_instance.Tensor("float16",
                                              (self.dim // 16, queries_align, 16),
                                              name="queries_l1", scope=tik.scope_cbuf)
 
        with self.tik_instance.new_stmt_scope():
            with self.tik_instance.for_range(0, move_num) as i:
                self.tik_instance.data_move(queries_l1[0, i, 0],
                                            self.input_queries_gm[move_offset + i, 0],
                                            0, self.dim // 16, 1, 0, queries_align - 1)
        # compute xy using cube
        code_loop_time = aicore_code_num // self.code_num_each_loop
        with self.tik_instance.if_scope(code_loop_time > 0):
            with self.tik_instance.for_range(0, code_loop_time) as loop_code:
                self.cube_compute_each_loop(queries_l1, aicore_move_offset,
                                            loop_code * self.code_num_each_loop,
                                            self.code_num_each_loop, move_offset, move_num, 0)
 
        code_last_num = aicore_code_num % self.code_num_each_loop
        with self.tik_instance.if_scope(code_last_num > 0):
            self.cube_compute_each_loop(queries_l1, aicore_move_offset,
                                        code_loop_time * self.code_num_each_loop,
                                        code_last_num, move_offset, move_num, 1)

    def vconv_int8_to_fp16(self, code_num_ub_each_loop, code_fp16_ub, code_int8_ub):
        vconv_loops = (code_num_ub_each_loop * self.dim) // (self.fp16_mask * 255)
        vconv_offset = 0
        if vconv_loops > 0:
            for conv_num in range(vconv_loops):
                vconv_offset = conv_num * self.fp16_mask * 255
                self.tik_instance.vconv(self.fp16_mask, "none",
                                        code_fp16_ub[vconv_offset],
                                        code_int8_ub[vconv_offset],
                                        255, 1, 1, 8, 4)
            vconv_offset += self.fp16_mask * 255

        vconv_repeat_time = (code_num_ub_each_loop * self.dim) % (self.fp16_mask * 255) // self.fp16_mask
        if vconv_repeat_time > 0:
            self.tik_instance.vconv(self.fp16_mask, "none",
                                    code_fp16_ub[vconv_offset],
                                    code_int8_ub[vconv_offset],
                                    vconv_repeat_time, 1, 1, 8, 4)       

    def vmul_code_and_scale(self, code_num_ub_each_loop, code_fp16_ub, code_scale_ub):
        with self.tik_instance.for_range(0, code_num_ub_each_loop) as i:
            repeat = self.dim // self.fp16_mask
            offset = 0
            if repeat > 0:
                self.tik_instance.vmul(self.fp16_mask,
                                        code_fp16_ub[i, offset],
                                        code_scale_ub[offset],
                                        code_fp16_ub[i, offset],
                                        repeat, 1, 1,
                                        1, 8, 8, 8)
                offset += repeat * self.fp16_mask
            remain = self.dim % self.fp16_mask
            if remain > 0:
                self.tik_instance.vmul(remain,
                                        code_fp16_ub[i, offset],
                                        code_scale_ub[offset],
                                        code_fp16_ub[i, offset],
                                        1, 1, 1, 1, 8, 8, 8)

    def cube_compute_each_loop_with_scale(self, code_l1, aicore_move_offset, code_move_offset, loop_ub,
        code_scale_ub, code_num_ub_each_loop):

        code_fp16_ub = self.tik_instance.Tensor("float16",
                                                (code_num_ub_each_loop, self.dim),
                                                name="code_fp16_ub", scope=tik.scope_ubuf)
        with self.tik_instance.new_stmt_scope():
            code_int8_ub = self.tik_instance.Tensor("int8",
                                            (code_num_ub_each_loop, self.dim),
                                            name="code_int8_ub", scope=tik.scope_ubuf)

            with self.tik_instance.for_range(0, self.dim // 32) as i:
                with self.tik_instance.for_range(0, 16) as j:
                    self.tik_instance.data_move(code_int8_ub[j, i * 32],
                                                self.input_code_gm[(aicore_move_offset + code_move_offset
                                                + loop_ub * code_num_ub_each_loop) // 16, i, j, 0],
                                                0, code_num_ub_each_loop // 16,
                                                1, (self.dim - 32) * 16 // 32 + 15,
                                                (self.dim - 32) * 16 // 32 + 15)

            self.vconv_int8_to_fp16(code_num_ub_each_loop, code_fp16_ub, code_int8_ub)

        self.vmul_code_and_scale(code_num_ub_each_loop, code_fp16_ub, code_scale_ub)

        with self.tik_instance.for_range(0, self.dim // 16) as i:
            self.tik_instance.data_move(code_l1[i, 0, 0],
                                        code_fp16_ub[0, i * 16],
                                        0, code_num_ub_each_loop,
                                        1, (self.dim - 16) * 2 // 32, 0)

    def cube_last_compute_each_loop_with_scale(self, code_l1, aicore_move_offset, code_move_offset, code_loop,
        code_scale_ub, code_last, code_num_ub_each_loop):

        code_fp16_ub = self.tik_instance.Tensor("float16",
                                                (code_num_ub_each_loop, self.dim),
                                                name="code_fp16_ub", scope=tik.scope_ubuf)
        with self.tik_instance.new_stmt_scope():
            code_int8_ub = self.tik_instance.Tensor("int8",
                                            (code_num_ub_each_loop, self.dim),
                                            name="code_int8_ub", scope=tik.scope_ubuf)
            with self.tik_instance.for_range(0, self.dim // 32) as i:
                with self.tik_instance.for_range(0, 16) as j:
                    self.tik_instance.data_move(code_int8_ub[j, i * 32],
                                                self.input_code_gm[(aicore_move_offset + code_move_offset
                                                + code_loop * code_num_ub_each_loop) // 16, i, j, 0],
                                                0, code_num_ub_each_loop // 16,
                                                1, (self.dim - 32) * 16 // 32 + 15,
                                                (self.dim - 32) * 16 // 32 + 15)

            self.vconv_int8_to_fp16(code_num_ub_each_loop, code_fp16_ub, code_int8_ub)

        self.vmul_code_and_scale(code_num_ub_each_loop, code_fp16_ub, code_scale_ub)

        with self.tik_instance.for_range(0, self.dim // 16) as i:
            self.tik_instance.data_move(code_l1[i, 0, 0],
                                        code_fp16_ub[0, i * 16],
                                        0, code_num_ub_each_loop,
                                        1, self.dim // 16 - 1, 0)


    def cube_compute_each_loop(self, queries_l1, aicore_move_offset, code_move_offset, code_move_num,
                               queries_move_offset, queries_move_num, flag):
        queries_align = (queries_move_num + 15) // 16 * 16
        inner_product_l0c = self.tik_instance.Tensor("float32",
                                                     (self.code_num_each_loop // 16, queries_align, 16),
                                                     name="inner_product_l0c", scope=tik.scope_cbuf_out)
        code_move_num_ub = (code_move_num + 15) // 16 * 16
        loop_times = 8
        thread_nums = 2

        if self.use_scale is True:
            code_scale_ub = self.tik_instance.Tensor("float16", (self.dim, ),
                                                    name="code_scale_ub", scope=tik.scope_ubuf)

            self.tik_instance.data_move(code_scale_ub, self.input_scale_gm, 0, 1, self.dim // 16, 0, 0)

            if (self.dim >= 384):
                loop_times = 16
                thread_nums = 1

        code_num_ub_each_loop = self.code_num_each_loop // loop_times

        if flag == 0:
            with self.tik_instance.for_range(0, loop_times, thread_num=thread_nums) as loop_ub:
                # move y from out to UB
                code_l1 = self.tik_instance.Tensor("float16",
                                                  (self.dim // 16, code_num_ub_each_loop, 16),
                                                  name="code_l1", scope=tik.scope_cbuf)
                if self.use_scale is True:
                    self.cube_compute_each_loop_with_scale(code_l1, aicore_move_offset, code_move_offset, loop_ub,
                                                      code_scale_ub, code_num_ub_each_loop)
                else:
                    with self.tik_instance.for_range(0, self.dim // 16) as i:
                        self.tik_instance.data_move(code_l1[i, 0, 0],
                            self.input_code_gm[(aicore_move_offset + code_move_offset
                                                + loop_ub * code_num_ub_each_loop) // self.code_batch,
                                                i, 0, 0],
                            0, code_num_ub_each_loop // self.code_batch,
                            self.code_batch, (self.dim - 16) * self.code_batch * 2 // 32, 0)

                self.tik_instance.matmul(inner_product_l0c[(loop_ub * code_num_ub_each_loop * queries_align):],
                                         queries_l1, code_l1,
                                         queries_align, self.dim, code_num_ub_each_loop)
        else:
            code_loop = code_move_num_ub // code_num_ub_each_loop
            with self.tik_instance.for_range(0, code_loop, thread_num=thread_nums) as loop_ub:
                code_l1 = self.tik_instance.Tensor("float16",
                                                  (self.dim // 16, code_num_ub_each_loop, 16),
                                                  name="code_l1", scope=tik.scope_cbuf)
                if self.use_scale is True:
                    self.cube_compute_each_loop_with_scale(code_l1, aicore_move_offset, code_move_offset, loop_ub,
                                                      code_scale_ub, code_num_ub_each_loop)
                else:
                    with self.tik_instance.for_range(0, self.dim // 16) as i:
                        self.tik_instance.data_move(code_l1[i, 0, 0],
                            self.input_code_gm[(aicore_move_offset + code_move_offset
                                                + loop_ub * code_num_ub_each_loop) // self.code_batch,
                                                i, 0, 0],
                            0, code_num_ub_each_loop // self.code_batch,
                            self.code_batch, (self.dim - 16) * self.code_batch * 2 // 32, 0)
                
                self.tik_instance.matmul(inner_product_l0c[(loop_ub * code_num_ub_each_loop * queries_align):],
                                         queries_l1, code_l1,
                                         queries_align, self.dim, code_num_ub_each_loop)

            code_last = code_move_num_ub % code_num_ub_each_loop
            with self.tik_instance.if_scope(code_last > 0):
                code_l1 = self.tik_instance.Tensor("float16",
                                                  (self.dim // 16, code_num_ub_each_loop, 16),
                                                  name="code_l1", scope=tik.scope_cbuf)
                if self.use_scale is True:
                    self.cube_last_compute_each_loop_with_scale(code_l1, aicore_move_offset, code_move_offset, code_loop,
                        code_scale_ub, code_last, code_num_ub_each_loop)
                else:
                    with self.tik_instance.for_range(0, self.dim // 16) as i:
                        self.tik_instance.data_move(code_l1[i, 0, 0],
                                self.input_code_gm[(aicore_move_offset + code_move_offset
                                                    + code_loop * code_num_ub_each_loop) // self.code_batch,
                                                i, 0, 0],
                                0, code_num_ub_each_loop // self.code_batch,
                                self.code_batch, (self.dim - 16) * self.code_batch * 2 // 32, 0)

                self.tik_instance.matmul(inner_product_l0c[(code_loop * code_num_ub_each_loop * queries_align):],
                                         queries_l1, code_l1,
                                         queries_align, self.dim, code_num_ub_each_loop)

        # 执行存储
        code_out_num = self.code_num_each_loop // 2  # 最小512 最大1024
        if flag == 0:
            with self.tik_instance.for_range(0, 2, thread_num=1) as i:
                dst_ub = self.tik_instance.Tensor("float16",
                                                  (queries_move_num, code_out_num),
                                                  name="dst_ub", scope=tik.scope_ubuf)

                with self.tik_instance.if_scope(True):  # 将vconv操作移到此作用域下，出作用域之后能够释放UB
                    # mov xy from L0-C to UB
                    inner_product_ub = self.tik_instance.Tensor("float32",
                                                                (code_out_num // 16, queries_align, 16),
                                                                name="inner_product_ub", scope=tik.scope_ubuf)
                    self.tik_instance.data_move(inner_product_ub,
                                                inner_product_l0c[i * code_out_num // 16, 0, 0],
                                                0, 1, code_out_num * queries_align // 256, 0, 0)
                    add_ub = self.tik_instance.Tensor("float32",
                                                    (queries_move_num, code_out_num),
                                                    name="add_ub", scope=tik.scope_ubuf)
                    with self.tik_instance.for_range(0, queries_move_num) as loop_queries:
                        self.tik_instance.data_move(add_ub[loop_queries, 0],
                                                    inner_product_ub[0, loop_queries, 0],
                                                    0, code_out_num // 16, 2, queries_align * 2 - 2, 0)
 
                    vconv_loops = 2 if queries_move_num > 1 else 1
                    query_num_each_loop = queries_move_num // vconv_loops
                    vconv_repeat_time = query_num_each_loop * code_out_num // self.fp32_mask
 
                    with self.tik_instance.for_range(0, vconv_loops, thread_num=vconv_loops) as vconv_loop:
                        self.tik_instance.vconv(self.fp32_mask, "none",
                                                dst_ub[query_num_each_loop * vconv_loop, 0],
                                                add_ub[query_num_each_loop * vconv_loop, 0],
                                                vconv_repeat_time, 1, 1, 4, 8)

                if self.use_extra_score is True:

                    extra_score_ub = self.tik_instance.Tensor("float16",
                                                              (queries_move_num, code_out_num),
                                                              name="extra_score_ub",
                                                              scope=tik.scope_ubuf)
                    # 跟随底库计算需要的偏移量
                    move_offset = self.block_offset + aicore_move_offset + code_move_offset + i * code_out_num
                    repeat = code_out_num // self.fp16_mask
                    with self.tik_instance.for_range(0, queries_move_num) as j:
                        # 跟随query计算需要的偏移
                        move_offset_from_query = (j + queries_move_offset) * self.extra_score_len
                        self.tik_instance.data_move(extra_score_ub[j, 0],
                                                    self.input_extra_score_gm[move_offset_from_query + move_offset],
                                                    0,
                                                    1,
                                                    (code_out_num) // 16,
                                                    8, 8)
                        self.tik_instance.vec_add(self.fp16_mask,
                                                  dst_ub[j, 0],
                                                  dst_ub[j, 0],
                                                  extra_score_ub[j, 0],
                                                  repeat,
                                                  8, 8, 8)
                        self.tik_instance.vec_muls(self.fp16_mask,
                                                   dst_ub[j, 0],
                                                   dst_ub[j, 0],
                                                   0.5,
                                                   repeat, 8, 8)

                # 按照mask，执行过滤1。
                if self.use_distance_mask is True:
                    distance_mask_ub = self.tik_instance.Tensor("uint8",
                                                                (queries_move_num, code_out_num // 8),
                                                                name="distance_mask_ub",
                                                                scope=tik.scope_ubuf)
                    if self.share_distance_mask is not True:                                            
                        self.tik_instance.data_move(distance_mask_ub[0, 0],
                                                    self.distance_mask_gm[queries_move_offset, 
                                                                        (aicore_move_offset + 
                                                                        code_move_offset + 
                                                                        i * code_out_num) // 8],
                                                    0,
                                                    queries_move_num, (code_out_num // 8) // 32,
                                                    (self.code_num - code_out_num) // 8 // 32, 0)
                    else:
                        self.tik_instance.data_move(distance_mask_ub[0, 0],
                                                    self.distance_mask_gm[0, 
                                                                        (aicore_move_offset + 
                                                                        code_move_offset + 
                                                                        i * code_out_num) // 8],
                                                    0,
                                                    1, (code_out_num // 8) // 32,
                                                    0, 0)

                        for cid in range(queries_move_num - 1):
                            self.tik_instance.data_move(distance_mask_ub[cid + 1, 0],
                                                        distance_mask_ub[0, 0],
                                                        0,
                                                        1, (code_out_num // 8) // 32,
                                                        0, 0)
                    
                    filter_distance_using_mask_in_ub(self.tik_instance, 
                                                     dst_ub[:],
                                                     queries_move_num * code_out_num,
                                                     dst_ub[:],
                                                     distance_mask_ub[:],
                                                     MIN_FP16)
                
 
                self.tik_instance.data_move(self.output_distances_gm[queries_move_offset,
                                                                     aicore_move_offset + code_move_offset
                                                                     + i * code_out_num],
                                            dst_ub,
                                            0, queries_move_num, code_out_num // 16, 0,
                                            (self.code_num - code_out_num) // 16)
 
                max_size = code_out_num // self.max_batch * 2
                dst_max_ub = self.tik_instance.Tensor("float16",
                                                      (queries_move_num, max_size),
                                                      name="dst_max_ub", scope=tik.scope_ubuf)
                vcmax_loops = 2 if queries_move_num > 1 else 1
                vcmax_repeat_times = query_num_each_loop * code_out_num // self.max_batch
                with self.tik_instance.for_range(0, vcmax_loops, thread_num=vcmax_loops) as vcmax_loop:
                    self.tik_instance.vcmax(self.max_batch,
                                            dst_max_ub[vcmax_loop * query_num_each_loop, 0],
                                            dst_ub[vcmax_loop * query_num_each_loop, 0],
                                            vcmax_repeat_times, 1, 1, self.max_batch // 16)
                self.tik_instance.data_move(self.output_maxs_gm[queries_move_offset,
                                                                (aicore_move_offset + code_move_offset
                                                                 + i * code_out_num) // self.max_batch * 2],
                                            dst_max_ub,
                                            0, queries_move_num, (code_out_num // self.max_batch) // 8,
                                            0, (self.code_num - code_out_num) // self.max_batch // 8)
        else:
            code_loops = (code_move_num_ub + code_out_num - 1) // code_out_num
            with self.tik_instance.for_range(0, code_loops, thread_num=1) as i:
                dst_ub = self.tik_instance.Tensor("float16",
                                                  (queries_move_num, code_out_num),
                                                  name="dst_ub", scope=tik.scope_ubuf)
                with self.tik_instance.if_scope(True):
                    # mov xy from L0-C to UB
                    inner_product_ub = self.tik_instance.Tensor("float32",
                                                                (code_out_num // 16, queries_align, 16),
                                                                name="inner_product_ub", scope=tik.scope_ubuf)
                    self.tik_instance.data_move(inner_product_ub,
                                                inner_product_l0c[i * code_out_num // 16, 0, 0],
                                                0, 1, code_out_num * queries_align // 256, 0, 0)
 
                    add_ub = self.tik_instance.Tensor("float32",
                                                    (queries_move_num, code_out_num),
                                                    name="add_ub", scope=tik.scope_ubuf)
                    # compute distance each query
                    with self.tik_instance.for_range(0, queries_move_num) as loop_queries:
                        self.tik_instance.data_move(add_ub[loop_queries, 0],
                                                    inner_product_ub[0, loop_queries, 0],
                                                    0, code_out_num // 16, 2, queries_align * 2 - 2, 0)
 
                    vconv_loops = 2 if queries_move_num > 1 else 1
                    query_num_each_loop = queries_move_num // vconv_loops
                    vconv_repeat_time = query_num_each_loop * code_out_num // self.fp32_mask
                    with self.tik_instance.for_range(0, vconv_loops, thread_num=vconv_loops) as vconv_loop:
                        self.tik_instance.vconv(self.fp32_mask, "none",
                                                dst_ub[query_num_each_loop * vconv_loop, 0],
                                                add_ub[query_num_each_loop * vconv_loop, 0],
                                                vconv_repeat_time, 1, 1, 4, 8)

                if self.use_extra_score is True:
                    extra_score_ub = self.tik_instance.Tensor("float16",
                                                              (queries_move_num, code_out_num),
                                                              name="extra_score_ub",
                                                              scope=tik.scope_ubuf)
                    # 跟随底库计算需要的偏移量
                    move_offset = self.block_offset + aicore_move_offset + code_move_offset + i * code_out_num
                    repeat = code_out_num // self.fp16_mask
                    with self.tik_instance.if_scope(i == code_loops - 1):
                        extra_score_move_num = code_move_num_ub - i * code_out_num
                        with self.tik_instance.for_range(0, queries_move_num) as j:
                            # 跟随query计算需要的偏移
                            move_offset_from_query = (j + queries_move_offset) * self.extra_score_len
                            self.tik_instance.data_move(extra_score_ub[j, 0],
                                                        self.input_extra_score_gm[move_offset_from_query + move_offset],
                                                        0,
                                                        1,
                                                        extra_score_move_num // 16,
                                                        8, 8)
                            
                            self.tik_instance.vec_add(self.fp16_mask,
                                                      dst_ub[j, 0],
                                                      dst_ub[j, 0],
                                                      extra_score_ub[j, 0],
                                                      repeat,
                                                      8, 8, 8)
                            self.tik_instance.vec_muls(self.fp16_mask,
                                                       dst_ub[j, 0],
                                                       dst_ub[j, 0],
                                                       0.5,
                                                       repeat, 8, 8)
                    with self.tik_instance.else_scope():
                        extra_score_move_num = code_out_num
                        with self.tik_instance.for_range(0, queries_move_num) as j:
                            # 跟随query计算需要的偏移
                            move_offset_from_query = (j + queries_move_offset) * self.extra_score_len
                            self.tik_instance.data_move(extra_score_ub[j, 0],
                                                        self.input_extra_score_gm[move_offset_from_query + move_offset],
                                                        0,
                                                        1,
                                                        extra_score_move_num // 16,
                                                        8, 8)
                            self.tik_instance.vec_add(self.fp16_mask,
                                                      dst_ub[j, 0],
                                                      dst_ub[j, 0],
                                                      extra_score_ub[j, 0],
                                                      repeat,
                                                      8, 8, 8)
                            self.tik_instance.vec_muls(self.fp16_mask,
                                                       dst_ub[j, 0],
                                                       dst_ub[j, 0],
                                                       0.5,
                                                       repeat, 8, 8)

                # 按照mask，执行过滤2。
                if self.use_distance_mask is True:
                    distance_mask_ub = self.tik_instance.Tensor("uint8",
                                                                (queries_move_num, code_out_num // 8),
                                                                name="distance_mask_ub",
                                                                scope=tik.scope_ubuf)
                    if self.share_distance_mask is not True:                                            
                        self.tik_instance.data_move(distance_mask_ub[0, 0],
                                                    self.distance_mask_gm[queries_move_offset, 
                                                                        (aicore_move_offset + 
                                                                        code_move_offset + 
                                                                        i * code_out_num) // 8],
                                                    0,
                                                    queries_move_num, (code_out_num // 8) // 32,
                                                    (self.code_num - code_out_num) // 8 // 32, 0)
                    else:
                        self.tik_instance.data_move(distance_mask_ub[0, 0],
                                                    self.distance_mask_gm[0, 
                                                                        (aicore_move_offset + 
                                                                        code_move_offset + 
                                                                        i * code_out_num) // 8],
                                                    0,
                                                    1, (code_out_num // 8) // 32,
                                                    0, 0)
                        for cid in range(queries_move_num - 1):
                            self.tik_instance.data_move(distance_mask_ub[cid + 1, 0],
                                                        distance_mask_ub[0, 0],
                                                        0,
                                                        1, (code_out_num // 8) // 32,
                                                        0, 0)
                    
                    filter_distance_using_mask_in_ub(self.tik_instance, 
                                                     dst_ub[:],
                                                     queries_move_num * code_out_num,
                                                     dst_ub[:],
                                                     distance_mask_ub[:],
                                                     MIN_FP16)            
 
                self.tik_instance.data_move(self.output_distances_gm[queries_move_offset,
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
                    with self.tik_instance.for_range(0, queries_move_num, thread_num=thread_num) as query:
                        vcmax_repeat_times = code_last_num // self.max_batch
                        offset = 0
                        with self.tik_instance.if_scope(vcmax_repeat_times > 0):
                            self.tik_instance.vcmax(self.max_batch,
                                                    dst_max_ub[query, 0],
                                                    dst_ub[query, 0],
                                                    vcmax_repeat_times, 1, 1, self.max_batch // 16)
                            offset += vcmax_repeat_times * self.max_batch
                        vcmax_remain = code_last_num % self.max_batch
                        with self.tik_instance.if_scope(vcmax_remain > 0):
                            self.tik_instance.vcmax(vcmax_remain,
                                                    dst_max_ub[query, vcmax_repeat_times * 2],
                                                    dst_ub[query, offset],
                                                    1, 1, 1, 1)
                        self.tik_instance.data_move(self.output_maxs_gm[queries_move_offset,
                                                                        (aicore_move_offset + code_move_offset
                                                                         + i * code_out_num) // self.max_batch * 2],
                                                    dst_max_ub,
                                                    0, queries_move_num, code_out_num // self.max_batch // 8, 0,
                                                    (self.code_num - code_out_num) // self.max_batch // 8)
                with self.tik_instance.else_scope():
                    vcmax_loops = 2 if queries_move_num > 1 else 1
                    vcmax_repeat_times = query_num_each_loop * code_out_num // self.max_batch
                    with self.tik_instance.for_range(0, vcmax_loops, thread_num=vcmax_loops) as vcmax_loop:
                        self.tik_instance.vcmax(self.max_batch,
                                                dst_max_ub[vcmax_loop * query_num_each_loop, 0],
                                                dst_ub[vcmax_loop * query_num_each_loop, 0],
                                                vcmax_repeat_times, 1, 1, self.max_batch // 16)
                    self.tik_instance.data_move(self.output_maxs_gm[queries_move_offset,
                                                                    (aicore_move_offset + code_move_offset
                                                                     + i * code_out_num) // self.max_batch * 2],
                                                dst_max_ub,
                                                0, queries_move_num, (code_out_num // self.max_batch) // 8, 0,
                                                (self.code_num - code_out_num) // self.max_batch // 8)
 
    def distance_compute(self):
        """
        the compute process
        """
        self.cal_num_each_core()
 
        with self.tik_instance.for_range(0, self.aicore_use, block_num=self.aicore_use) as block_index:
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
        inputs = [self.input_queries_gm,
                  self.input_code_gm,
                  self.input_actual_num_gm]
        outputs = [self.output_distances_gm,
                   self.output_maxs_gm,
                   self.output_flag_gm]

        if self.use_distance_mask is True:
            self.distance_mask_gm   = self.tik_instance.Tensor(self.dtype_input_mask,
                                                               self.shape_input_mask,
                                                               name="distance_mask_gm",
                                                               scope=tik.scope_gm)
            inputs.append(self.distance_mask_gm)

        if self.use_extra_score is True:
            self.input_extra_score_gm = self.tik_instance.Tensor(self.dtype_extra_score, 
                                                                self.shape_extra_score, 
                                                                name="input_extra_score_gm", 
                                                                scope=tik.scope_gm)
            inputs.append(self.input_extra_score_gm)

        if self.use_scale is True:
            self.input_scale_gm = self.tik_instance.Tensor(self.dtype_input_scale,
                                                           self.shape_input_scale,
                                                           name="input_scale_gm",
                                                           scope=tik.scope_gm)
            inputs.append(self.input_scale_gm)

        self.distance_compute()
 
        self.tik_instance.BuildCCE(self.kernel_name,
                                   inputs, 
                                   outputs)

        return self.tik_instance
 
 
def distance_flat_ip_maxs_with_mask(input_queries, input_code, input_actual_num, input_mask,
                                    output_distances, output_maxs, output_flag,
                                    kernel_name="distance_flat_ip_maxs_with_mask"):
    """
    算子接口函数
    Args:
        input_queries:      查询向量，二维Tensor，shape=(N, Dim)，dtype=float16，Dim为向量的维度。
        input_code:         底库向量，四维Tensor，shape=(M/H, Dim/16, H, 16)，dtype=float16，M为对齐后的底库总容量，
                            H是Z区域高度，即Z区域包含的向量个数。
        input_actual_num:   底库实际特征向量的条数，二维Tensor，shape=(CORE_NUM, 8)，dtype=uint32。
        input_mask:         距离过滤Mask，每个查询向量对应一个，二维Tensor，shape=(N1, M/8)，dtype=uint8,
                            当N1=N时，每个查询对应的Mask不相同，当N1=1且不等于N时，算子内部对于所有查询采用同样的Mask。
        output_distances:   距离结果，二维Tensor，shape=(N, M)，dtype=float16。
        output_maxs:        块topk结果，二维Tensor，shape=(N, (M + 64 - 1)/64 * 2)，每一个块输出64个(Max, Index)二元组，dtype=float16。
        output_flag：       算子执行完成标志，二维Tensor，shape=(CORE_NUM, 16)，dtype=uint16。 
    """
    distance_flat = DistanceFlatIPMaxs(input_queries, input_code, input_actual_num, input_mask,
                                       output_distances, output_maxs, output_flag, kernel_name)
    tik_instance = distance_flat.get_tik_instance()
    return tik_instance
