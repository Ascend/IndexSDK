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


class CidFilter:
    def __init__(self,
                 input_cid_idx,
                 input_cid_val,
                 input_ts,
                 input_cid_mask,
                 input_ts_filter,
                 input_and,
                 input_mul,
                 output_res,
                 output_flag,
                 kernel_name="cid_filter"):
        self.shape_cid_idx = input_cid_idx.get("shape")
        self.dtype_cid_idx = input_cid_idx.get("dtype")
        self.shape_cid_val = input_cid_val.get("shape")
        self.dtype_cid_val = input_cid_val.get("dtype")
        self.shape_ts = input_ts.get("shape")
        self.dtype_ts = input_ts.get("dtype")
        self.shape_cid_mask = input_cid_mask.get("shape")
        self.dtype_cid_mask = input_cid_mask.get("dtype")
        self.shape_ts_filter = input_ts_filter.get("shape")
        self.dtype_ts_filter = input_ts_filter.get("dtype")
        self.shape_and = input_and.get("shape")
        self.dtype_and = input_and.get("dtype")
        self.shape_mul = input_mul.get("shape")
        self.dtype_mul = input_mul.get("dtype")
        self.shape_res = output_res.get("shape")
        self.dtype_res = output_res.get("dtype")
        self.shape_flag = output_flag.get("shape")
        self.dtype_flag = output_flag.get("dtype")
        self.kernel_name = kernel_name

        # compute parameter
        self.cid_num = self.shape_cid_idx[0]
        self.fp16_mask = 128
        self.fp32_mask = 64
        self.uint16_mask = 128
        self.int32_mask = 64
        self.repeat_times = 128
        self.and_size = self.uint16_mask
        self.mul_size = self.fp16_mask
        self.cid_mask_size = self.int32_mask
        self.zero_size = 128
        self.filter_size = self.int32_mask
        self.move_repeats = 4

        # check parameter
        self.check_parameter()

        # set tik instance
        self.set_tik_instance()

    def check_parameter(self):
        if self.cid_num % 128 != 0:
            raise RuntimeError("cid num must be a multiple of 128")

    def set_tik_instance(self):
        """
        set tik_instance
        """
        set_soc_info()
        self.tik_instance = tik.Tik()

        self.aicore_use = self.shape_flag[0]
        self.cid_sub_loop = 4096
        self.cid_each_loop = self.cid_sub_loop

        self.cid_num_each_core = self.cid_num // self.aicore_use // self.cid_each_loop * self.cid_each_loop
        self.cid_num_last_core = self.cid_num - (self.aicore_use - 1) * self.cid_num_each_core

        self.set_src_dst_tensor()

    def set_src_dst_tensor(self):
        """
        set input and output tensor
        """
        self.zero = self.tik_instance.Scalar("float16", name="zero", init_value=0)

        # creat input tensor: input_cid_idx_gm, input_cid_val_gm, input_cid_mask_gm, input_actual_num_gm
        # and creat output tensor: output_res_gm
        self.input_cid_idx_gm = self.tik_instance.Tensor(
            self.dtype_cid_idx,
            self.shape_cid_idx,
            name="input_cid_idx_gm",
            scope=tik.scope_gm)
        self.input_cid_val_gm = self.tik_instance.Tensor(
            self.dtype_cid_val,
            self.shape_cid_val,
            name="input_cid_val_gm",
            scope=tik.scope_gm)
        self.input_ts_gm = self.tik_instance.Tensor(
            self.dtype_ts,
            self.shape_ts,
            name="input_ts_gm",
            scope=tik.scope_gm)
        self.input_cid_mask_gm = self.tik_instance.Tensor(
            self.dtype_cid_mask,
            self.shape_cid_mask,
            name="input_cid_mask_gm",
            scope=tik.scope_gm)
        self.input_ts_filter_gm = self.tik_instance.Tensor(
            self.dtype_ts_filter,
            self.shape_ts_filter,
            name="input_ts_filter_gm",
            scope=tik.scope_gm)
        self.input_and_gm = self.tik_instance.Tensor(
            self.dtype_and,
            self.shape_and,
            name="input_and_gm",
            scope=tik.scope_gm)
        self.input_mul_gm = self.tik_instance.Tensor(
            self.dtype_mul,
            self.shape_mul,
            name="input_mul_gm",
            scope=tik.scope_gm)
        self.output_res_gm = self.tik_instance.Tensor(
            self.dtype_res,
            self.shape_res,
            name="output_res_gm",
            scope=tik.scope_gm)
        self.output_flag_gm = self.tik_instance.Tensor(
            self.dtype_flag,
            self.shape_flag,
            name="output_flag_gm",
            scope=tik.scope_gm)

    def compute_each_loop(self, aicore_offset, cid_offset, start_ts, end_ts):

        # move cid_idx(0, 0, 0, 1) to idx_int8_ub
        idx_int8_ub = self.tik_instance.Tensor("uint8",
                                               (self.cid_sub_loop,),
                                               name="idx_int8_ub", scope=tik.scope_ubuf)
        self.tik_instance.data_move(idx_int8_ub,
                                    self.input_cid_idx_gm[aicore_offset + cid_offset],
                                    0, 1, self.cid_sub_loop // 32, 0, 0)
        # move (1, 2, 4, 8) to and_ub
        and_ub = self.tik_instance.Tensor("uint16",
                                          (self.move_repeats, self.and_size),
                                          name="and_ub", scope=tik.scope_ubuf)
        self.tik_instance.data_move(and_ub,
                                    self.input_and_gm,
                                    0, 1, self.move_repeats * self.and_size // 16, 0, 0)
        # move (1, 0.5, 0.25, 0.125) to mul_ub
        mul_ub = self.tik_instance.Tensor("float16",
                                          (self.move_repeats, self.mul_size),
                                          name="mul_ub", scope=tik.scope_ubuf)
        self.tik_instance.data_move(mul_ub,
                                    self.input_mul_gm,
                                    0, 1, self.move_repeats * self.mul_size // 16, 0, 0)
        # move cid_mask to mask_ub
        mask_ub = self.tik_instance.Tensor("int32",
                                           (self.move_repeats, self.cid_mask_size),
                                           name="mask_ub", scope=tik.scope_ubuf)
        self.tik_instance.data_move(mask_ub,
                                    self.input_cid_mask_gm,
                                    0, 1, self.move_repeats * self.cid_mask_size // 8, 0, 0)
        val_ub = self.tik_instance.Tensor("uint32",
                                          (self.cid_sub_loop,),
                                          name="val_ub", scope=tik.scope_ubuf)
        self.tik_instance.data_move(val_ub,
                                    self.input_cid_val_gm[aicore_offset + cid_offset],
                                    0, 1, self.cid_sub_loop // 8, 0, 0)
        ts_ub = self.tik_instance.Tensor("int32",
                                         (self.cid_sub_loop,),
                                         name="ts_ub", scope=tik.scope_ubuf)
        self.tik_instance.data_move(ts_ub,
                                    self.input_ts_gm[aicore_offset + cid_offset],
                                    0, 1, self.cid_sub_loop // 8, 0, 0)
        # move idx_int8 to idx_uint16
        idx_uint16_ub = idx_int8_ub.reinterpret_cast_to("uint16")

        idx_res_uint16_ub = self.tik_instance.Tensor("uint16",
                                                     (self.move_repeats, self.cid_sub_loop // 2,),
                                                     name="idx_res_uint16_ub", scope=tik.scope_ubuf)

        # idx_uint16 & and_ub
        for i in range(self.move_repeats):
            self.tik_instance.vand(self.uint16_mask, idx_res_uint16_ub[i, 0],
                                   idx_uint16_ub, and_ub[i, 0],
                                   self.cid_sub_loop // 2 // self.uint16_mask,
                                   1, 1, 1, self.uint16_mask // 16, self.uint16_mask // 16, 0)

        # vconv cid_idx to fp16
        idx_res_int8_ub_ = idx_res_uint16_ub.reinterpret_cast_to("uint8")

        idx_res_fp16_ub = self.tik_instance.Tensor("float16",
                                                   (self.move_repeats, self.cid_sub_loop),
                                                   name="idx_fp16_ub", scope=tik.scope_ubuf)
        self.tik_instance.vconv(self.fp16_mask, 'none',
                                idx_res_fp16_ub, idx_res_int8_ub_,
                                self.cid_sub_loop * self.move_repeats // self.fp16_mask,
                                1, 1, self.fp16_mask // 16, self.fp16_mask // 32)

        # idx_fp16 * mul_ub
        for i in range(1, self.move_repeats):
            self.tik_instance.vmul(self.fp16_mask, idx_res_fp16_ub[i, 0],
                                   idx_res_fp16_ub[i, 0], mul_ub[i, 0],
                                   self.cid_sub_loop // self.fp16_mask,
                                   1, 1, 1, self.fp16_mask // 16, self.fp16_mask // 16, 0)

        # vconv idx_fp16 to idx_int32
        idx_res_int32_ub = self.tik_instance.Tensor("int32",
                                                    (self.move_repeats, self.cid_sub_loop),
                                                    name="idx_int32_ub", scope=tik.scope_ubuf)

        self.tik_instance.vconv(self.int32_mask, 'floor', idx_res_int32_ub, idx_res_fp16_ub,
                                self.cid_sub_loop * self.move_repeats // self.int32_mask // 2,
                                1, 1, self.int32_mask // 8, self.int32_mask // 16)
        self.tik_instance.vconv(self.int32_mask, 'floor', idx_res_int32_ub[self.move_repeats // 2, 0],
                                idx_res_fp16_ub[self.move_repeats // 2, 0],
                                self.cid_sub_loop * self.move_repeats // self.int32_mask // 2,
                                1, 1, self.int32_mask // 8, self.int32_mask // 16)

        # (0, 0, 0, 1) * (mask0, mask1, mask2, mask3)
        for i in range(self.move_repeats):
            self.tik_instance.vmul(self.int32_mask,
                                   idx_res_int32_ub[i, 0], idx_res_int32_ub[i, 0], mask_ub[i, 0],
                                   self.cid_sub_loop // self.int32_mask,
                                   1, 1, 1, self.int32_mask // 8, self.int32_mask // 8, 0)

        # (0, 0, 0, mask1), (0, mask2, 0, 0)  ==> mask1, mask2
        for i in range(1, self.move_repeats):
            self.tik_instance.vadd(self.int32_mask, idx_res_int32_ub[0, 0],
                                   idx_res_int32_ub[0, 0], idx_res_int32_ub[i, 0],
                                   self.cid_sub_loop // self.int32_mask,
                                   1, 1, 1, self.int32_mask // 8, self.int32_mask // 8, self.int32_mask // 8)

        # cid_val & cid_mask
        val_uint16_ub = val_ub.reinterpret_cast_to("uint16")
        idx_res_uint16_ub = idx_res_int32_ub.reinterpret_cast_to("uint16")
        self.tik_instance.vand(self.uint16_mask, val_uint16_ub,
                               val_uint16_ub, idx_res_uint16_ub,
                               self.cid_sub_loop * 2 // self.uint16_mask,
                               1, 1, 1, self.uint16_mask // 16, self.uint16_mask // 16, self.uint16_mask // 16)

        # mask res of uint16 reinterpret to int32
        mask_res_ub_ = val_uint16_ub.reinterpret_cast_to("int32")
        mask_res_ub = self.tik_instance.Tensor("float16",
                                               (self.cid_sub_loop,),
                                               name="mask_res_ub", scope=tik.scope_ubuf)

        # add for every pair in mask_res_ub, because type int32 can not used in vcmpv
        self.tik_instance.vconv(self.int32_mask, 'none',
                                mask_res_ub, mask_res_ub_,
                                self.cid_sub_loop // self.int32_mask, 1, 1,
                                self.int32_mask // 16, self.int32_mask // 8, deqscale=1.0)

        # mask res compare 0 to bit res
        cid_res_ub = self.tik_instance.Tensor("uint16",
                                              (self.cid_sub_loop // 16,),
                                              name="cid_res_ub", scope=tik.scope_ubuf)

        # create zero_ub for compare
        zero_ub = self.tik_instance.Tensor("float16",
                                           (self.zero_size,),
                                           name="zero_ub", scope=tik.scope_ubuf)
        self.tik_instance.vec_dup(self.zero_size, zero_ub, self.zero, 1, 1)

        # compare cid_res_ub != 0
        self.tik_instance.vcmpv_ne(cid_res_ub,
                                   mask_res_ub, zero_ub,
                                   self.cid_sub_loop // self.fp16_mask,
                                   1, 1, self.fp16_mask // 16, 0)

        # create start_ts_ub, end_ts_ub for add
        start_ts_ub = self.tik_instance.Tensor("int32",
                                               (self.filter_size,),
                                               name="start_ts_ub", scope=tik.scope_ubuf)
        end_ts_ub = self.tik_instance.Tensor("int32",
                                             (self.filter_size,),
                                             name="end_ts_ub", scope=tik.scope_ubuf)
        self.tik_instance.vec_dup(self.filter_size, start_ts_ub, start_ts, 1, self.filter_size // 8)
        self.tik_instance.vec_dup(self.filter_size, end_ts_ub, end_ts, 1, self.filter_size // 8)

        start_add_ub = self.tik_instance.Tensor("int32",
                                                (self.cid_sub_loop,),
                                                name="start_add_ub", scope=tik.scope_ubuf)
        end_add_ub = self.tik_instance.Tensor("int32",
                                              (self.cid_sub_loop,),
                                              name="end_add_ub", scope=tik.scope_ubuf)

        # ts_ub - start_ts_ub
        self.tik_instance.vadd(self.int32_mask, start_add_ub,
                               ts_ub, start_ts_ub,
                               self.cid_sub_loop // self.int32_mask,
                               1, 1, 1, self.int32_mask // 8, self.int32_mask // 8, 0)
        # ts_ub - end_ts_ub
        self.tik_instance.vadd(self.int32_mask, end_add_ub,
                               ts_ub, end_ts_ub,
                               self.cid_sub_loop // self.int32_mask,
                               1, 1, 1, self.int32_mask // 8, self.int32_mask // 8, 0)

        # vconv start_add_ub, end_add_ub from int32 to fp 16
        start_add_fp16_ub = self.tik_instance.Tensor("float16",
                                                     (self.cid_sub_loop,),
                                                     name="start_add_fp16_ub", scope=tik.scope_ubuf)
        end_add_fp16_ub = self.tik_instance.Tensor("float16",
                                                   (self.cid_sub_loop,),
                                                   name="end_add_fp16_ub", scope=tik.scope_ubuf)
        self.tik_instance.vconv(self.int32_mask, 'none',
                                start_add_fp16_ub, start_add_ub,
                                self.cid_sub_loop // self.int32_mask,
                                1, 1, self.int32_mask // 16, self.int32_mask // 8, deqscale=1.0)
        self.tik_instance.vconv(self.int32_mask, 'none',
                                end_add_fp16_ub, end_add_ub,
                                self.cid_sub_loop // self.int32_mask,
                                1, 1, self.int32_mask // 16, self.int32_mask // 8, deqscale=1.0)

        start_res_ub = self.tik_instance.Tensor("uint16",
                                                (self.cid_sub_loop // 16,),
                                                name="start_res_ub", scope=tik.scope_ubuf)
        end_res_ub = self.tik_instance.Tensor("uint16",
                                              (self.cid_sub_loop // 16,),
                                              name="end_res_ub", scope=tik.scope_ubuf)
        # start_add_fp16_ub >= 0
        self.tik_instance.vcmpv_ge(start_res_ub,
                                   start_add_fp16_ub, zero_ub,
                                   self.cid_sub_loop // self.fp16_mask,
                                   1, 1, self.fp16_mask // 16, 0)

        # end_add_fp16_ub < 0
        self.tik_instance.vcmpv_lt(end_res_ub,
                                   end_add_fp16_ub, zero_ub,
                                   self.cid_sub_loop // self.fp16_mask,
                                   1, 1, self.fp16_mask // 16, 0)

        res_ub = self.tik_instance.Tensor("uint16",
                                          (self.cid_sub_loop // 16,),
                                          name="res_ub", scope=tik.scope_ubuf)

        # cid_res_ub & start_res_ub
        self.tik_instance.vand(128, res_ub,
                               cid_res_ub, start_res_ub,
                               2, 1, 1, 1, 8, 8, 8)

        # (cid_res_ub & start_res_ub) & end_res_ub
        self.tik_instance.vand(128, res_ub,
                               res_ub, end_res_ub,
                               2, 1, 1, 1, 8, 8, 8)

        self.tik_instance.data_move(self.output_res_gm[(aicore_offset + cid_offset) // 16],
                                    res_ub,
                                    0, 1, self.cid_sub_loop // 16 // 16, 0, 0)

    def cid_filter(self):

        ts_filter_ub = self.tik_instance.Tensor("uint32",
                                                (8,),
                                                name="ts_filter_ub", scope=tik.scope_ubuf)
        self.tik_instance.data_move(ts_filter_ub, self.input_ts_filter_gm, 0, 1, 1, 0, 0)

        start_ts = self.tik_instance.Scalar(dtype="int32", name="start_ts", init_value=0)
        end_ts = self.tik_instance.Scalar(dtype="int32", name="end_ts", init_value=0)
        start_ts.set_as(ts_filter_ub[0])
        end_ts.set_as(ts_filter_ub[1])

        with self.tik_instance.for_range(0, self.aicore_use, block_num=self.aicore_use) as block_idx:
            cid_loops = self.cid_num_each_core // self.cid_each_loop
            with self.tik_instance.for_range(0, cid_loops) as loop_cid:
                self.compute_each_loop(block_idx * self.cid_num_each_core, loop_cid * self.cid_each_loop,
                                       start_ts, end_ts)

    def get_tik_instance(self):
        """
        obtain tik instance
        """
        self.cid_filter()

        self.tik_instance.BuildCCE(
            kernel_name=self.kernel_name,
            inputs=[
                self.input_cid_idx_gm, self.input_cid_val_gm, self.input_ts_gm,
                self.input_cid_mask_gm, self.input_ts_filter_gm,
                self.input_and_gm, self.input_mul_gm
            ],
            outputs=[self.output_res_gm, self.output_flag_gm])

        return self.tik_instance


def cid_filter(input_cid_idx,
               input_cid_val,
               input_ts,
               input_cid_mask,
               input_ts_filter,
               input_and,
               input_mul,
               output_res,
               output_flag,
               kernel_name="cid_filter"):
    """
    calculating distance

    Parameters
    ----------
    input_cid_idx : dict
        shape and dtype of cid idx vector
    input_cid_val : dict
        shape and dtype of cid offset vector
    input_ts : dict
        shape and dtype of timestamp vector
    input_cid_mask: dict
        shape and dtype of cid mask vector
    input_ts_filter: dict
        shape and dtype of timestamp filter vector
    input_and: dict
        shape and dtype of and vector
    input_mul: dict
        shape and dtype of mul vector
    output_res : dict
        shape and dtype of distances, should be same dtype as input_queries
    output_flag : dict
        shape and dtype of flag, only the 1th and 17th is valid
    kernel_name : str
        kernel name, default value is "distance_compute"

    Returns
    -------
    None
    """
    cf = CidFilter(input_cid_idx, input_cid_val, input_ts, input_cid_mask,
                   input_ts_filter, input_and, input_mul, output_res, output_flag, kernel_name)
    tik_instance = cf.get_tik_instance()
    return tik_instance
