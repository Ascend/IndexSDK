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

AI_CORE_NUM = 8
INT_MAX = 2147483647


def div_up(dividend_val, divisor_val):
    return ((dividend_val + divisor_val - 1) // divisor_val)


class DistanceMaskGenerator:
    def __init__(self, 
                 query_time_stamp, query_token_set, db_time_stamp, db_divisor, db_remainder, extra_mask,
                 distance_mask,
                 kernel_name):
        self.shape_query_time_stamp         = query_time_stamp.get("shape")
        self.dtype_query_time_stamp         = query_time_stamp.get("dtype")

        self.shape_query_token_set          = query_token_set.get("shape")
        self.dtype_query_token_set          = query_token_set.get("dtype")
        
        self.shape_db_time_stamp            = db_time_stamp.get("shape")
        self.dtype_db_time_stamp            = db_time_stamp.get("dtype")
       
        self.shape_db_divisor               = db_divisor.get("shape")
        self.dtype_db_divisor               = db_divisor.get("dtype")
        
        self.shape_db_remainder             = db_remainder.get("shape")
        self.dtype_db_remainder             = db_remainder.get("dtype")

        self.use_extra_mask                 = False
        if extra_mask is not None:
            self.shape_extra_mask           = extra_mask.get("shape")
            self.dtype_extra_mask           = extra_mask.get("dtype")
            self.use_extra_mask             = True
        
        self.shape_distance_mask            = distance_mask.get("shape")
        self.dtype_distance_mask            = distance_mask.get("dtype")
     
        self.kernel_name                    = kernel_name

        self.total_db_num                   = self.shape_db_time_stamp[0] 
        self.max_token_cnt                  = self.shape_query_token_set[0] * 8 // 2
        self.db_stride_each_call            = 8192  # 每次处理的底库特征距离个数
    
        self.core_num                       = AI_CORE_NUM

        set_soc_info()
        self.tik_instance   = tik.Tik(disable_debug=False)
        # 输入1：查询起始时间戳+结束时间戳
        self.query_time_stamp_gm    = self.tik_instance.Tensor(self.dtype_query_time_stamp, 
                                                               self.shape_query_time_stamp, 
                                                               name="query_time_stamp_gm", 
                                                               scope=tik.scope_gm)
        # 输入2：查询token id集
        self.query_token_set_gm     = self.tik_instance.Tensor(self.dtype_query_token_set, 
                                                               self.shape_query_token_set, 
                                                               name="query_token_set_gm", 
                                                               scope=tik.scope_gm)
        # 输入3：底库特征向量对应的时间戳
        self.db_time_stamp_gm       = self.tik_instance.Tensor(self.dtype_db_time_stamp,
                                                               self.shape_db_time_stamp,
                                                               name="db_time_stamp_gm",
                                                               scope=tik.scope_gm)
        # 输入4：底库特征向量对应的token id除以8后的除数
        self.db_divisor_gm          = self.tik_instance.Tensor(self.dtype_db_divisor, 
                                                               self.shape_db_divisor, 
                                                               name="db_divisor_gm", 
                                                               scope=tik.scope_gm)
        # 输入5：底库特征向量对应的token id除以8后的余数
        self.db_remainder_gm        = self.tik_instance.Tensor(self.dtype_db_remainder, 
                                                               self.shape_db_remainder, 
                                                               name="db_remainder_gm", 
                                                               scope=tik.scope_gm)

        # 输入6：用户自定义的Mask，可选。
        if self.use_extra_mask is True:
            self.extra_mask_gm      = self.tik_instance.Tensor(self.dtype_extra_mask,
                                                               self.shape_extra_mask,
                                                               name="extra_mask_gm",
                                                               scope=tik.scope_gm)

        # 输出1：距离结果掩码
        self.distance_mask_gm       = self.tik_instance.Tensor(self.dtype_distance_mask, 
                                                               self.shape_distance_mask, 
                                                               name="distance_mask_gm", 
                                                               scope=tik.scope_gm)

        self.use_extra_val = False


    def set_val(self, extra_val_filter, extra_val_attr):
        self.shape_extra_val_filter = extra_val_filter.get("shape")
        self.dtype_extra_val_filter = extra_val_filter.get("dtype")
        self.shape_extra_val_attr = extra_val_attr.get("shape")
        self.dtype_extra_val_attr = extra_val_attr.get("dtype")
        self.use_extra_val = True

        # 输入7：用户自定义的附加属性filter，可选。
        self.extra_val_filter_gm = self.tik_instance.Tensor(self.dtype_extra_val_filter,
                                                            self.shape_extra_val_filter,
                                                            name="extra_val_filter_gm",
                                                            scope=tik.scope_gm)
        # 输入8：自定义附加属性信息
        self.extra_val_attr_gm = self.tik_instance.Tensor(self.dtype_extra_val_attr, 
                                                          self.shape_extra_val_attr, 
                                                          name="extra_val_attr_gm", 
                                                          scope=tik.scope_gm)
    
        self.val_filter_sc = self.tik_instance.Scalar(dtype="int16", 
                                                      name="val_filter_sc", 
                                                      init_value=0)
        self.val_filter_model_sc = self.tik_instance.Scalar(dtype="int16", 
                                                            name="val_filter_model_sc", 
                                                            init_value=0)


    def set_optimize_policy(self):
        self.db_stride_each_call    = 8192
        self.core_num               = AI_CORE_NUM * 2

    def check_settings(self):
        if (self.total_db_num % self.db_stride_each_call != 0):
            raise RuntimeError("total_db_num must be multiple of db_stride_each_call!")
        if (self.db_stride_each_call % 256 != 0):
            raise RuntimeError("db_stride_each_call be multiple of 256!")

    def get_tik_instance(self):
        # 设置优化策略
        self.set_optimize_policy()
        # 参数检查
        self.check_settings()

        with self.tik_instance.for_range(0, self.core_num, block_num=self.core_num) as tid:
            self.execute_task(tid)
       
        if self.use_extra_mask is not True:
            if self.use_extra_val is not True:
                inputs = [self.query_time_stamp_gm,
                          self.query_token_set_gm,
                          self.db_time_stamp_gm,
                          self.db_divisor_gm,
                          self.db_remainder_gm
                ]
                outputs = [self.distance_mask_gm]
            else:
                inputs = [self.query_time_stamp_gm,
                          self.query_token_set_gm,
                          self.db_time_stamp_gm,
                          self.db_divisor_gm,
                          self.db_remainder_gm,
                          self.extra_val_filter_gm,
                          self.extra_val_attr_gm
                ]
                outputs = [self.distance_mask_gm] 
        else:
            inputs = [self.query_time_stamp_gm,
                      self.query_token_set_gm,
                      self.db_time_stamp_gm,
                      self.db_divisor_gm,
                      self.db_remainder_gm,
                      self.extra_mask_gm]
            outputs = [self.distance_mask_gm]
           
        self.tik_instance.BuildCCE(self.kernel_name,
                                   inputs, 
                                   outputs)
        
        return self.tik_instance
       
    def execute_task(self, task_id):
        """
        执行任务函数：由于总的特征向量个数是db_stride_each_call的整数倍，
                      所以每个任务处理total_stride_cnt // self.core_num个stride。
                      简单起见，最后一个任务处理不整除部分的stride。
        """
        total_stride_cnt        = self.total_db_num // self.db_stride_each_call
        stride_cnt_each_core    = total_stride_cnt // self.core_num
        stride_cnt_last_remain  = total_stride_cnt % self.core_num

        start_time_stamp_sc = self.tik_instance.Scalar(dtype="int32", 
                                                       name="start_time_stamp_sc", 
                                                       init_value=0)
        end_time_stamp_sc   = self.tik_instance.Scalar(dtype="int32", 
                                                       name="end_time_stamp_sc", 
                                                       init_value=0)
        query_token_set_ub  = self.tik_instance.Tensor("uint8", 
                                                        (div_up(self.shape_query_token_set[0], 32) * 32,),
                                                        name="query_token_set_ub",
                                                        scope=tik.scope_ubuf)

        extra_val_filter_ub_tmp = self.tik_instance.Tensor("int16", 
                                                            (16, ),
                                                            name="extra_val_filter_ub_tmp",
                                                            scope=tik.scope_ubuf)
        
        extra_val_filter_ub = self.tik_instance.Tensor("int16", 
                                                        (self.db_stride_each_call, ),
                                                        name="extra_val_filter_ub",
                                                        scope=tik.scope_ubuf)

        with self.tik_instance.for_range(0, 1):
            # 读取查询开始时间戳和结束时间戳
            query_time_stamp_ub = self.tik_instance.Tensor("int32",
                                                           (8,),
                                                           name="query_time_stamp_ub",
                                                           scope=tik.scope_ubuf)
            self.tik_instance.data_move(query_time_stamp_ub[0],
                                       self.query_time_stamp_gm[0],
                                       0, 
                                       1, 1, 0, 0)
            start_time_stamp_sc.set_as(query_time_stamp_ub[0])
            end_time_stamp_sc.set_as(query_time_stamp_ub[1])
            # 读取查询token id集
            self.tik_instance.data_move(query_token_set_ub[0],
                                        self.query_token_set_gm[0],
                                        0, 
                                        1, query_token_set_ub.shape[0] // 32,
                                        0, 0)
        # 每个任务开始计算mask
        db_vector_offset_sc     = self.tik_instance.Scalar(dtype="uint32")
        db_vector_cnt_sc        = self.tik_instance.Scalar(dtype="uint32")
        db_vector_offset_sc.set_as(task_id * stride_cnt_each_core * self.db_stride_each_call)
        with self.tik_instance.if_scope(task_id != (self.core_num - 1)):
            db_vector_cnt_sc.set_as(stride_cnt_each_core * self.db_stride_each_call)
        with self.tik_instance.else_scope():
            db_vector_cnt_sc.set_as((stride_cnt_each_core + stride_cnt_last_remain) * self.db_stride_each_call)
        
        if self.use_extra_mask is True:
            extra_mask_gm = self.extra_mask_gm[db_vector_offset_sc // 8 :]
        else:
            extra_mask_gm = None

        if self.use_extra_val is True:
            self.tik_instance.data_move(extra_val_filter_ub_tmp[0],
                                        self.extra_val_filter_gm[0],
                                        0, 1, 1, 0, 0)

            self.val_filter_sc.set_as(extra_val_filter_ub_tmp[0])
            self.val_filter_model_sc.set_as(extra_val_filter_ub_tmp[1])

            # 一次计算8192条 extra_val_filter_ub大小为8192
            self.tik_instance.vec_dup(128, extra_val_filter_ub, self.val_filter_sc, 8192 // 128, 8)
            extra_val_attr_gm = self.extra_val_attr_gm[db_vector_offset_sc:]
        else:
            extra_val_attr_gm = None

        self.compute_distance_mask_each_task(start_time_stamp_sc,
                                             end_time_stamp_sc,
                                             query_token_set_ub,
                                             extra_val_filter_ub,
                                             self.db_time_stamp_gm[db_vector_offset_sc :],
                                             self.db_divisor_gm[db_vector_offset_sc :],
                                             self.db_remainder_gm[db_vector_offset_sc * 2 :],
                                             extra_mask_gm,
                                             self.distance_mask_gm[db_vector_offset_sc // 8 :],
                                             extra_val_attr_gm,
                                             db_vector_cnt_sc)
    
    def compute_distance_mask_each_task(self,
                                         query_start_time, 
                                         query_end_time, 
                                         query_token_set,
                                         extra_val_filter_ub,
                                         db_time_stamp_gm, 
                                         db_divisor_gm, 
                                         db_remainder_gm,
                                         extra_mask_gm,
                                         distance_mask_gm,
                                         extra_val_attr_gm,
                                         db_vector_cnt):
        """
        根据时间戳以及token id集计算距离值mask
        Args:
            query_start_time：查询起始时间戳
            query_end_time: 查询结束时间戳
            query_token_set: 查询token id集
            extra_val_filter_ub: 附加属性过滤条件
            db_time_stamp_gm: 底库特征向量对应的时间戳
            db_divisor_gm: 底库特征向量对应的token id除以8的除数
            db_remainder_gm: 底库特征向量对应的token id除以8的余数
            extra_mask_gm: 用户自定义属性过滤逻辑生成的Mask，可选。
            distance_mask_gm: 距离值对应的mask，输出结果。
            db_vector_cnt: 特征向量的条数
        """
        # 时间戳比较的结果
        time_stamp_cmp_res_ub   = self.tik_instance.Tensor("uint8",
                                                           (self.db_stride_each_call // 8,),
                                                           name="time_stamp_cmp_res_ub",
                                                           scope=tik.scope_ubuf)
        # token id比较的结果
        token_cmp_res_ub        = self.tik_instance.Tensor("uint8",
                                                           (self.db_stride_each_call // 8,),
                                                           name="token_cmp_res_ub",
                                                           scope=tik.scope_ubuf)
        # 以上两个结果做与计算后的最终结果
        dst_res_ub              = self.tik_instance.Tensor("uint8",
                                                           (self.db_stride_each_call // 8,),
                                                           name="dst_res_ub",
                                                           scope=tik.scope_ubuf)

        val_cmp_res_ub = self.tik_instance.Tensor("int16",
                                                  (self.db_stride_each_call,),
                                                  name="val_cmp_res_ub",
                                                  scope=tik.scope_ubuf)

        # 判断是否需要进行时间戳属性过滤
        enable_time_filter_sc   = self.tik_instance.Scalar(dtype="int32",
                                                           name="enable_time_filter_sc",
                                                           init_value=1)
        with self.tik_instance.if_scope(tik.all(query_start_time == 0, query_end_time == -INT_MAX)):
            enable_time_filter_sc.set_as(0)

        time_stamp_cmp_res_ub_int16 = time_stamp_cmp_res_ub.reinterpret_cast_to("int16")
        token_cmp_res_ub_int16      = token_cmp_res_ub.reinterpret_cast_to("int16")
        dst_res_ub_int16            = dst_res_ub.reinterpret_cast_to("int16")
        dst_res_ub_uint8            = dst_res_ub.reinterpret_cast_to("uint8")
       
        loop_times  = db_vector_cnt // self.db_stride_each_call
        with self.tik_instance.for_range(0, loop_times) as lid:
            with self.tik_instance.if_scope(enable_time_filter_sc == 1):
                self.compare_time_stamp(query_start_time, query_end_time, 
                                        db_time_stamp_gm[lid * self.db_stride_each_call :], 
                                        self.db_stride_each_call, 
                                        time_stamp_cmp_res_ub)
            
            with self.tik_instance.for_range(0, 1):
                self.compare_token_id(query_token_set, 
                                      db_divisor_gm[lid * self.db_stride_each_call :],
                                      db_remainder_gm[lid * self.db_stride_each_call * 2 :],
                                      self.db_stride_each_call,
                                      token_cmp_res_ub)
            
            # 对以上两个结果进行与操作
            with self.tik_instance.if_scope(enable_time_filter_sc == 1):
                repeat = dst_res_ub_int16.shape[0] // 128
                with self.tik_instance.if_scope(repeat != 0):
                    self.tik_instance.vand(128, 
                                           dst_res_ub_int16, 
                                           time_stamp_cmp_res_ub_int16,
                                           token_cmp_res_ub_int16,
                                           repeat,
                                           1, 1, 1, 8, 8, 8)
                last_remain = dst_res_ub_int16.shape[0] % 128
                with self.tik_instance.if_scope(last_remain != 0):
                    self.tik_instance.vand(last_remain,
                                           dst_res_ub_int16[repeat * 128 :],
                                           time_stamp_cmp_res_ub_int16[repeat * 128 :],
                                           token_cmp_res_ub_int16[repeat * 128 :],
                                           1,
                                           1, 1, 1, 8, 8, 8)
            with self.tik_instance.else_scope():
                self.tik_instance.data_move(dst_res_ub[0],
                                            token_cmp_res_ub[0],
                                            0,
                                            1, (self.db_stride_each_call // 8) // 32, 
                                            0, 0)
            # 与用户的extra_mask进行与操作，可选。
            if self.use_extra_mask is True:
                with self.tik_instance.for_range(0, 1):
                    self.process_with_extra_mask(dst_res_ub_int16[0 :], 
                                                 extra_mask_gm[lid * self.db_stride_each_call // 8:])

            # 与用户的extra_val进行与操作，可选。
            if self.use_extra_val is True:
                self.process_with_extra_val(extra_val_filter_ub,
                                            extra_val_attr_gm[lid * self.db_stride_each_call:],
                                            self.db_stride_each_call,
                                            val_cmp_res_ub,
                                            dst_res_ub_int16)
                
            # 保存结果
            self.tik_instance.data_move(distance_mask_gm[lid * self.db_stride_each_call // 8 :],
                                        dst_res_ub_uint8[0],
                                        0,
                                        1, (self.db_stride_each_call // 8) // 32,
                                        0, 0)

    def process_with_extra_val(self, extra_val_filter_ub, extra_val_attr_gm, db_vector_cnt, res_ub, dst_res_ub_int16):
        extra_val_attr_ub = self.tik_instance.Tensor("int16",
                                                     (db_vector_cnt, ),
                                                     name="extra_val_attr_ub",
                                                     scope=tik.scope_ubuf)

        self.tik_instance.data_move(extra_val_attr_ub,
                                    extra_val_attr_gm,
                                    0, 1, db_vector_cnt * 2 // 32, 0, 0)

        repeat = db_vector_cnt // 128
        last_remain = db_vector_cnt % 128
        with self.tik_instance.if_scope(repeat != 0):
            self.tik_instance.vand(128,
                                   res_ub,
                                   extra_val_filter_ub,
                                   extra_val_attr_ub,
                                   repeat,
                                   1, 1, 1, 8, 8, 8)
        with self.tik_instance.if_scope(last_remain != 0):
            self.tik_instance.vand(last_remain,
                                   res_ub[repeat * 128:],
                                   extra_val_filter_ub[repeat * 128:],
                                   extra_val_attr_ub[repeat * 128:],
                                   1, 1, 1, 1, 8, 8, 8)
        
        mask_val_ub = self.tik_instance.Tensor("uint8",
                                               (db_vector_cnt // 8,),
                                               name="mask_val_ub",
                                               scope=tik.scope_ubuf)

        # 附加属性模式0
        with self.tik_instance.if_scope(self.val_filter_model_sc == 0):
            self.compute_model0(extra_val_filter_ub, res_ub, mask_val_ub)

        # 附加属性模式1
        with self.tik_instance.else_scope():
            self.compute_model1(res_ub, mask_val_ub, db_vector_cnt)

        # 附加属性mask和time及其tokenid的mask进行按位与
        mask_val_ub_int16 = mask_val_ub.reinterpret_cast_to("int16")
        repeat = dst_res_ub_int16.shape[0] // 128
        with self.tik_instance.if_scope(repeat != 0):
            self.tik_instance.vand(128,
                                   dst_res_ub_int16, 
                                   mask_val_ub_int16,
                                   dst_res_ub_int16,
                                   repeat,
                                   1, 1, 1, 8, 8, 8)
        last_remain = dst_res_ub_int16.shape[0] % 128
        with self.tik_instance.if_scope(last_remain != 0):
            self.tik_instance.vand(last_remain,
                                   dst_res_ub_int16[repeat * 128:],
                                   mask_val_ub_int16[repeat * 128:],
                                   dst_res_ub_int16[repeat * 128:],
                                   1, 1, 1, 1, 8, 8, 8)

    def compute_model0(self, extra_val_filter_ub, res_ub, mask_val_ub):
        extra_val_filter = self.tik_instance.Scalar(dtype="float16", 
                                                    name="extra_val_filter", 
                                                    init_value=0)
        extra_val_filter_int16 = self.tik_instance.Scalar(dtype="int16", 
                                                          name="extra_val_filter_int16",
                                                          init_value=0)
        extra_val_filter_int16.set_as(extra_val_filter_ub[0])

        # float16的Scalar无法直接set_as int16ub的值
        # 通过int16 Scalar接一下再进行转换
        repeat = res_ub.shape[0] // 128
        extra_val_filter.set_as(extra_val_filter_int16)
        res_ub_fp16 = res_ub.reinterpret_cast_to("float16")
        self.tik_instance.vconv(128,
                                "none",
                                res_ub_fp16,
                                res_ub,
                                repeat,
                                1, 1, 8, 8)

        with self.tik_instance.if_scope(repeat != 0):
            self.tik_instance.vcmpvs_eq(mask_val_ub, 
                                        res_ub_fp16,
                                        extra_val_filter,
                                        repeat,
                                        1, 8)

    def compute_model1(self, res_ub, mask_val_ub, db_vector_cnt):
        res_ub_fp16 = res_ub.reinterpret_cast_to("float16")
        repeat = db_vector_cnt // 128
        with self.tik_instance.if_scope(repeat != 0):
            self.tik_instance.vconv(128,
                                    "none",
                                    res_ub_fp16,
                                    res_ub,
                                    repeat,
                                    1, 1, 8, 8)

            self.tik_instance.vcmpvs_gt(mask_val_ub, 
                                        res_ub_fp16,
                                        0.0,
                                        repeat,
                                        1, 8)
    
    def process_with_extra_mask(self, dst_ub_int16, extra_mask_gm):
        extra_mask_ub_uint8 = self.tik_instance.Tensor("uint8",
                                                       (dst_ub_int16.shape[0] * 2, ),
                                                       name="extra_mask_ub_uint8",
                                                       scope=tik.scope_ubuf)
        self.tik_instance.data_move(extra_mask_ub_uint8[0:],
                                    extra_mask_gm[0:],
                                    0,
                                    1, dst_ub_int16.shape[0] * 2 // 32,
                                    0, 0)
        extra_mask_ub_int16 = extra_mask_ub_uint8.reinterpret_cast_to("int16")

        repeat = dst_ub_int16.shape[0] // 128
        with self.tik_instance.if_scope(repeat != 0):
            self.tik_instance.vand(128, 
                                   dst_ub_int16,
                                   extra_mask_ub_int16,
                                   dst_ub_int16,
                                   repeat,
                                   1, 1, 1, 8, 8, 8)
        last_remain = dst_ub_int16.shape[0] % 128
        with self.tik_instance.if_scope(last_remain != 0):
            self.tik_instance.vand(last_remain,
                                   dst_ub_int16[repeat * 128 :],
                                   extra_mask_ub_int16[repeat * 128 :],
                                   dst_ub_int16[repeat * 128 :],
                                   1,
                                   1, 1, 1, 8, 8, 8)        
    
    def compare_time_stamp(self, 
                           query_start_time, 
                           query_end_time, 
                           db_time_stamp_gm, 
                           db_vector_cnt, 
                           res_ub):
        aligned_cnt = res_ub.shape[0]

        res_ub1     = self.tik_instance.Tensor("uint8",
                                               (aligned_cnt,),
                                               name="res_ub1",
                                               scope=tik.scope_ubuf)
        res_ub2     = self.tik_instance.Tensor("uint8",
                                               (aligned_cnt,),
                                               name="res_ub2",
                                               scope=tik.scope_ubuf)
        # 加载底库特征向量对应的时间戳信息
        db_time_stamp_ub    = self.tik_instance.Tensor("int32",
                                                       (aligned_cnt * 8, ),
                                                       name="db_time_stamp_ub",
                                                       scope=tik.scope_ubuf)
        self.tik_instance.data_move(db_time_stamp_ub[0],
                                    db_time_stamp_gm[0],
                                    0,
                                    1, db_vector_cnt * 4 // 32,
                                    0, 0)
        # 分别减去起始时间戳和结束时间戳
        db_time_stamp_ub1    = self.tik_instance.Tensor("int32",
                                                        (aligned_cnt * 8, ),
                                                        name="db_time_stamp_ub1",
                                                        scope=tik.scope_ubuf)
        db_time_stamp_ub2    = self.tik_instance.Tensor("int32",
                                                        (aligned_cnt * 8, ),
                                                        name="db_time_stamp_ub2",
                                                        scope=tik.scope_ubuf)
        repeat = db_vector_cnt // 64
        self.tik_instance.vadds(64, 
                                db_time_stamp_ub1, 
                                db_time_stamp_ub, 
                                query_start_time,
                                repeat, 
                                1, 1, 8, 8)
        self.tik_instance.vadds(64, 
                                db_time_stamp_ub2, 
                                db_time_stamp_ub, 
                                query_end_time,
                                repeat, 
                                1, 1, 8, 8)
        
        
        db_time_stamp_ub1_fp32 = db_time_stamp_ub1.reinterpret_cast_to("float32")
        self.tik_instance.vconv(64,
                                "none",
                                db_time_stamp_ub1_fp32,
                                db_time_stamp_ub1,
                                repeat,
                                1, 1, 8, 8)

        db_time_stamp_ub2_fp32 = db_time_stamp_ub2.reinterpret_cast_to("float32")
        self.tik_instance.vconv(64,
                                "none",
                                db_time_stamp_ub2_fp32,
                                db_time_stamp_ub2,
                                repeat,
                                1, 1, 8, 8)

        # 比较操作
        repeat         = db_vector_cnt // 64
        with self.tik_instance.if_scope(repeat != 0):
            self.tik_instance.vcmpvs_ge(res_ub1[0 :],
                                        db_time_stamp_ub1_fp32[0 :],
                                        0.0,
                                        repeat,
                                        1, 8)
            self.tik_instance.vcmpvs_le(res_ub2[0 :],
                                        db_time_stamp_ub2_fp32[0 :],
                                        0.0,
                                        repeat,
                                        1, 8)
        # 与操作：由于vand指令只支持uint16/int16类型，所以这里先做个转换。
        res_ub1_int16   = res_ub1.reinterpret_cast_to("int16")
        res_ub2_int16   = res_ub2.reinterpret_cast_to("int16")
        res_ub_int16    = res_ub.reinterpret_cast_to("int16")

        repeat      = (aligned_cnt // 2) // 128
        with self.tik_instance.if_scope(repeat != 0):
            self.tik_instance.vand(128, res_ub_int16, res_ub1_int16, res_ub2_int16, repeat,
                                   1, 1, 1, 8, 8, 8)
        last_remain = (aligned_cnt // 2) % 128
        with self.tik_instance.if_scope(last_remain != 0):
            self.tik_instance.vand(last_remain, res_ub_int16[repeat * 128 :], res_ub1_int16[repeat * 128 :],
                                   res_ub2_int16[repeat * 128 :],
                                   1, 1, 1, 1, 8, 8, 8)       
        
    def compare_token_id(self, 
                         query_token_set, 
                         db_divisor_gm, 
                         db_remainder_gm, 
                         db_vector_cnt, 
                         res_ub):
        db_vector_cnt_align = res_ub.shape[0] * 8

        db_divisor_ub_int32 = self.tik_instance.Tensor("int32",
                                                       (db_vector_cnt_align,),
                                                       name="db_divisor_ub_int32",
                                                       scope=tik.scope_ubuf)
        db_remainder_ub_int16 = self.tik_instance.Tensor("int16",
                                                         (db_vector_cnt_align,),
                                                         name="db_remainder_ub_int16",
                                                         scope=tik.scope_ubuf)
        tmp_remainder_ub_int16 = self.tik_instance.Tensor("int16",
                                                          (db_vector_cnt_align,),
                                                          name="tmp_remainder_ub_int16",
                                                          scope=tik.scope_ubuf)

        db_remainder_ub_uint8       = db_remainder_ub_int16.reinterpret_cast_to("uint8")
        query_token_set_int16       = query_token_set.reinterpret_cast_to("int16")

        self.tik_instance.data_move(db_remainder_ub_uint8[0:],
                                    db_remainder_gm[0:],
                                    0,
                                    1, db_vector_cnt * 2 // 32,
                                    0, 0)
        self.tik_instance.data_move(db_divisor_ub_int32[0 :],
                                    db_divisor_gm[0 :],
                                    0,
                                    1, db_vector_cnt * 4 // 32,
                                    0, 0)
        # 使用vgather指令生成tmp_remainder_ub_int16
        repeat = db_vector_cnt // 128
        with self.tik_instance.if_scope(repeat != 0):
            self.tik_instance.vgather(128, 
                                      tmp_remainder_ub_int16,
                                      query_token_set_int16,
                                      db_divisor_ub_int32,
                                      repeat, 8, 0, 0)
        # 使用vand指令将tmp_remainder_ub_int16与db_remainder_ub_int16相与
        res_remainder_ub_int16 = self.tik_instance.Tensor("int16",
                                                          (db_vector_cnt_align,),
                                                          name="res_remainder_ub_int16",
                                                          scope=tik.scope_ubuf)
        self.tik_instance.vand(128, 
                               res_remainder_ub_int16, 
                               tmp_remainder_ub_int16, 
                               db_remainder_ub_int16,
                               repeat,
                               1, 1, 1, 8, 8, 8)
        # 使用比较指令，判断res_remainder_ub_int16中的每一个值是否等于2.0
        res_remainder_ub_fp16 = res_remainder_ub_int16.reinterpret_cast_to("float16")
        self.tik_instance.vcmpvs_gt(res_ub, 
                                    res_remainder_ub_fp16,
                                    2.0,
                                    repeat,
                                    1, 8)


