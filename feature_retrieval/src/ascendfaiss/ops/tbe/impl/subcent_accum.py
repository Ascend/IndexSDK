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


class SubcentAccum:
    def __init__(self,
                input_subcents_cores,
                input_hassign_cores,
                input_actual_pages_num, 
                output_subcents,
                output_hassign,
                output_flag,
                kernel_name="subcent_accum"):
        self.shape_subcents_cores = input_subcents_cores.get("shape")
        self.dtype_subcents_cores = input_subcents_cores.get("dtype")
        self.shape_hassign_cores = input_hassign_cores.get("shape")
        self.dtype_hassign_cores = input_hassign_cores.get("dtype")
        self.shape_actual_pages_num = input_actual_pages_num.get("shape")
        self.dtype_actual_pages_num = input_actual_pages_num.get("dtype")
        self.shape_subcents = output_subcents.get("shape")
        self.dtype_subcents = output_subcents.get("dtype")
        self.shape_hassign = output_hassign.get("shape")
        self.dtype_hassign = output_hassign.get("dtype")
        self.shape_flag = output_flag.get("shape")
        self.dtype_flag = output_flag.get("dtype")
        self.kernel_name = kernel_name

        self.code_num = 64

        self.dim = self.shape_subcents_cores[3]

        # set vector fp32 mask and fp16 mask
        self.fp16_mask = 128
        self.fp32_mask = 64

        # set tik instance
        self.set_tik_instance()

    def set_tik_instance(self):
        """
        set tik_instance
        """
        set_soc_info()
        self.tik_instance = tik.Tik()
       
        self.set_src_dst_tensor()

    def set_src_dst_tensor(self):
        """
        set input and output tensor
        """
        self.input_subcents_cores_gm = self.tik_instance.Tensor(
            self.dtype_subcents_cores,
            self.shape_subcents_cores,
            name="input_subcents_cores_gm",
            scope=tik.scope_gm)
        self.input_hassign_cores_gm = self.tik_instance.Tensor(
            self.dtype_hassign_cores, 
            self.shape_hassign_cores,
            name="input_hassign_cores_gm",
            scope=tik.scope_gm)
        self.input_actual_pages_num_gm = self.tik_instance.Tensor(
            self.dtype_actual_pages_num,
            self.shape_actual_pages_num,
            name="input_actual_pages_num_gm",
            scope=tik.scope_gm)
        self.output_subcents_gm = self.tik_instance.Tensor(
            self.dtype_subcents, 
            self.shape_subcents,
            name="output_subcents_gm",
            scope=tik.scope_gm)
        self.output_hassign_gm = self.tik_instance.Tensor(
            self.dtype_hassign, 
            self.shape_hassign, 
            name="output_hassign_gm",
            scope=tik.scope_gm)
        self.output_flag_gm = self.tik_instance.Tensor(
            self.dtype_flag,
            self.shape_flag,
            name="output_flag_gm",
            scope=tik.scope_gm)

    def distance_compute_flat(self):
        actual_pages_num_ub = self.tik_instance.Tensor("uint16", (16, ),
                                                       name="actual_pages_num_ub",
                                                       scope=tik.scope_ubuf)
        self.tik_instance.data_move(actual_pages_num_ub, self.input_actual_pages_num_gm, 0, 1, 1, 0, 0)
        pages = self.tik_instance.Scalar("int16")
        pages.set_as(actual_pages_num_ub[0])
        actual_centrorids_num = self.tik_instance.Scalar("int16")
        actual_centrorids_num.set_as(actual_pages_num_ub[1])
        subcent_result_ub = self.tik_instance.Tensor("float16", (self.code_num, self.dim),
                                                     name="subcent_result_ub", scope=tik.scope_ubuf)
        self.tik_instance.vec_dup(self.fp16_mask, subcent_result_ub, 0,
                                  actual_centrorids_num * self.dim // self.fp16_mask, 8)

        hassign_result_ub = self.tik_instance.Tensor("int16", (self.code_num, ),
                                                     name="hassign_result_ub",
                                                     scope=tik.scope_ubuf)
        self.tik_instance.vec_dup(self.code_num, hassign_result_ub, 0, 1, 8)

        with self.tik_instance.for_range(0, pages) as page:
            with self.tik_instance.for_range(0, 8) as core:
                hassign_tmp_ub = self.tik_instance.Tensor("int16", (self.code_num, ),
                                                          name="hassign_tmp_ub", scope=tik.scope_ubuf)
                self.tik_instance.data_move(hassign_tmp_ub, self.input_hassign_cores_gm[page, core, 0],
                                            0, 1, self.code_num // 16, 0, 0)
                self.tik_instance.vadd(self.code_num, hassign_result_ub, hassign_result_ub,
                                       hassign_tmp_ub, 1, 1, 1, 1, 8, 8, 8)
                subcent_tmp_ub = self.tik_instance.Tensor("float16", (actual_centrorids_num, self.dim),
                                                          name="subcent_tmp_ub", scope=tik.scope_ubuf)
                with self.tik_instance.for_range(0, actual_centrorids_num) as k:
                    self.tik_instance.data_move(subcent_tmp_ub[k, 0],
                                                self.input_subcents_cores_gm[page, core, k, 0],
                                                0, 1, self.dim // 16, 0, 0)
                    self.tik_instance.vadd(self.fp16_mask, subcent_result_ub[k, 0],
                                           subcent_result_ub[k, 0], subcent_tmp_ub[k, 0],
                                           self.dim // self.fp16_mask, 1, 1, 1, 8, 8, 8)

        hassign_fp16_ub = self.tik_instance.Tensor("float16", (self.code_num, ),
                                                   name="hassign_fp16_ub", scope=tik.scope_ubuf)
        self.tik_instance.vconv(self.code_num, "none", hassign_fp16_ub, hassign_result_ub,
                                1, 1, 1, self.code_num // 16, self.code_num // 16)
        self.tik_instance.vec_rec(self.code_num, hassign_fp16_ub, hassign_fp16_ub, 1,
                                  self.code_num // 16, self.code_num // 16)
        coffe = self.tik_instance.Scalar("float16")
        with self.tik_instance.for_range(0, actual_centrorids_num) as i:
            with self.tik_instance.if_scope(hassign_result_ub[i] != 0):
                coffe.set_as(hassign_fp16_ub[i])
                self.tik_instance.vmuls(self.fp16_mask, subcent_result_ub[i, 0],
                                        subcent_result_ub[i, 0], coffe,
                                        self.dim // self.fp16_mask, 1, 1, 8, 8)

        self.tik_instance.data_move(self.output_hassign_gm, hassign_result_ub,
                                    0, 1, self.code_num // 16, 0, 0)
        self.tik_instance.data_move(self.output_subcents_gm, subcent_result_ub,
                                    0, 1, actual_centrorids_num * self.dim // 16, 0, 0)

        with self.tik_instance.if_scope(actual_centrorids_num < self.code_num): 
            subcent_extra_ub = self.tik_instance.Tensor("float16",
                                                        (self.code_num - actual_centrorids_num, self.dim),
                                                        name="subcent_extra_ub", scope=tik.scope_ubuf)
            self.tik_instance.vec_dup(self.fp16_mask, subcent_extra_ub, 65504,
                                      (self.code_num - actual_centrorids_num) * self.dim // self.fp16_mask, 8)
            self.tik_instance.data_move(self.output_subcents_gm[actual_centrorids_num, 0],
                                        subcent_extra_ub, 0, 1,
                                        (self.code_num - actual_centrorids_num) * self.dim // 16, 0, 0)

        one = self.tik_instance.Scalar(dtype="uint16", name="one", init_value=1)
        flag_ub = self.tik_instance.Tensor("uint16",
                                            (16,),
                                            name="flag_ub", scope=tik.scope_ubuf)
        flag_ub[0].set_as(one)
        self.tik_instance.data_move(self.output_flag_gm,
                                    flag_ub, 0, 1, 1, 0, 0)

    def get_tik_instance(self):
        """
        obtain tik instance
        """
        self.distance_compute_flat()

        self.tik_instance.BuildCCE(kernel_name=self.kernel_name,
                                   inputs=[
                                       self.input_subcents_cores_gm,
                                       self.input_hassign_cores_gm,
                                       self.input_actual_pages_num_gm
                                   ],
                                   outputs=[self.output_subcents_gm, self.output_hassign_gm, self.output_flag_gm])

        return self.tik_instance


def subcent_accum(input_subcents_cores,
                input_hassign_cores,
                input_actual_pages_num,
                output_subcents,
                output_hassign,
                output_flag,
                kernel_name="subcent_accum"):

    subcnt_accum = SubcentAccum(input_subcents_cores, input_hassign_cores,
                                input_actual_pages_num, output_subcents,
                                output_hassign, output_flag, kernel_name)
    tik_instance = subcnt_accum.get_tik_instance()
    return tik_instance
