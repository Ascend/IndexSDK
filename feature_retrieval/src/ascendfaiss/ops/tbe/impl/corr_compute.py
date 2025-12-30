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


class CorrCompute:
    def __init__(self,
                 input_code,
                 input_actual_num,
                 output_corr_result,
                 output_flag,
                 kernel_name="corr_compute"):
        self.shape_code = input_code.get("shape")
        self.dtype_code = input_code.get("dtype")
        self.shape_actual_num = input_actual_num.get("shape")
        self.dtype_actual_num = input_actual_num.get("dtype")
        self.shape_corr_result = output_corr_result.get("shape")
        self.dtype_corr_result = output_corr_result.get("dtype")
        self.shape_flag = output_flag.get("shape")
        self.dtype_flag = output_flag.get("dtype")
        self.kernel_name = kernel_name

        # compute parameter
        self.dim = self.shape_code[2]

        # check parameter
        self.check_parameter()

        # set vector fp32 mask and fp16 mask
        self.fp32_mask = 64
        self.fp16_mask = 128

        # set tik instance
        self.set_tik_instance()

    def check_parameter(self):
        if self.dim % 16 != 0:
            raise RuntimeError("feature dim must be a multiple of 16")

    def set_tik_instance(self):
        """
        set tik_instance
        """
        set_soc_info()
        self.tik_instance = tik.Tik()

        self.seg_each_loop = 64 // 16

        self.dim_loops = (self.dim * self.dim * 4 + 65536 - 1) // 65536
        self.dim_each_loop = self.dim // self.dim_loops

        self.set_src_dst_tensor()

    def set_src_dst_tensor(self):
        """
        set input and output tensor
        """

        # creat input tensor: input_queries_gm, input_code_gm, input_actual_num_gm
        # and output tensor: output_dist_gm, output_flag_gm in global buffer
        self.input_code_gm = self.tik_instance.Tensor(self.dtype_code,
                                                      self.shape_code,
                                                      name="input_code_gm", scope=tik.scope_gm)
        self.input_actual_num_gm = self.tik_instance.Tensor(self.dtype_actual_num,
                                                            self.shape_actual_num,
                                                            name="input_actual_num_gm", scope=tik.scope_gm)
        self.output_corr_result_gm = self.tik_instance.Tensor(self.dtype_corr_result,
                                                             self.shape_corr_result,
                                                             name="output_dist_gm", scope=tik.scope_gm)
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
        self.seg_num = actual_num // 16

    def corr_compute_each_core(self):
        result_l0c = self.tik_instance.Tensor("float32",
                                              (self.dim // 16, self.dim, 16),
                                              name="result_l0c", scope=tik.scope_cbuf_out)
        seg_loops = self.seg_num // self.seg_each_loop

        # first seg loop
        with self.tik_instance.new_stmt_scope():
            code_l1 = self.tik_instance.Tensor("float16",
                                               (self.seg_each_loop, self.dim, 16),
                                               name="code_l1", scope=tik.scope_cbuf)
            self.tik_instance.data_move(code_l1, self.input_code_gm[0, 0, 0, 0],
                                        0, 1, self.seg_each_loop * self.dim, 0, 0)
            self.tik_instance.matmul(result_l0c,
                                     code_l1, code_l1,
                                     self.dim, self.seg_each_loop * 16, self.dim, init_l1out=True)

        with self.tik_instance.for_range(1, seg_loops) as seg_loop_i:
            code_l1 = self.tik_instance.Tensor("float16",
                                               (self.seg_each_loop, self.dim, 16),
                                               name="code_l1", scope=tik.scope_cbuf)
            self.tik_instance.data_move(code_l1, self.input_code_gm[0, seg_loop_i * self.seg_each_loop, 0, 0],
                                        0, 1, self.seg_each_loop * self.dim, 0, 0)
            self.tik_instance.matmul(result_l0c,
                                     code_l1, code_l1,
                                     self.dim, self.seg_each_loop * 16, self.dim, init_l1out=False)

        if self.dim * self.dim_each_loop // self.fp32_mask <= 128:
            vconv_loops = 1
            vconv_repeat_times = self.dim * self.dim_each_loop // self.fp32_mask
        else:
            vconv_repeat_times = 128
            vconv_loops = self.dim * self.dim_each_loop // self.fp32_mask // vconv_repeat_times

        dim_loop_thread_num = 1 if self.dim_loops == 1 else 2
        with self.tik_instance.for_range(0, self.dim_loops, thread_num=dim_loop_thread_num) as dim_idx:
            result_ub = self.tik_instance.Tensor("float32",
                                                 (self.dim_each_loop // 16, self.dim, 16),
                                                 name="result_ub", scope=tik.scope_ubuf)
            self.tik_instance.data_move(result_ub, result_l0c[dim_idx * self.dim_each_loop // 16, 0, 0],
                                        0, 1, self.dim * self.dim_each_loop // 256, 0, 0)
            result_ub_fp16 = self.tik_instance.Tensor("float16",
                                                      (self.dim_each_loop // 16, self.dim, 16),
                                                      name="result_ub_fp16", scope=tik.scope_ubuf)
            with self.tik_instance.for_range(0, vconv_loops) as vconv_loop:
                self.tik_instance.vconv(self.fp32_mask, "none",
                                        result_ub_fp16[vconv_loop * vconv_repeat_times * self.fp32_mask],
                                        result_ub[vconv_loop * vconv_repeat_times * self.fp32_mask],
                                        vconv_repeat_times, 1, 1, 4, 8)
            for dim_i in range(self.dim):
                self.tik_instance.data_move(self.output_corr_result_gm[dim_i, dim_idx * self.dim_each_loop],
                                            result_ub_fp16[0, dim_i, 0],
                                            0, self.dim_each_loop // 16, 1, self.dim - 1, 0)

    def distance_compute(self):
        """
        the compute process
        """
        self.cal_num_each_core()

        self.corr_compute_each_core()

        one = self.tik_instance.Scalar(dtype="uint16", name="one", init_value=1)
        flag_ub = self.tik_instance.Tensor("uint16", (16,), name="flag_ub", scope=tik.scope_ubuf)
        flag_ub[0].set_as(one)
        self.tik_instance.data_move(self.output_flag_gm[0, 0],
                                    flag_ub, 0, 1, 1, 0, 0)

    def get_tik_instance(self):
        """
        obtain tik instance
        """
        self.distance_compute()

        self.tik_instance.BuildCCE(kernel_name=self.kernel_name,
                                   inputs=[
                                       self.input_code_gm,
                                       self.input_actual_num_gm
                                   ],
                                   outputs=[
                                       self.output_corr_result_gm,
                                       self.output_flag_gm
                                   ])

        return self.tik_instance


def corr_compute(input_code,
                 input_actual_num,
                 output_corr_result,
                 output_flag,
                 kernel_name="corr_compute"):
    """
    calculating distance

    Parameters
    ----------
    input_queries : dict
        shape and dtype of query vector
    input_code : dict
        shape and dtype of coarse code
    input_actual_num: dict
        shape and dtype of actual code num,
        shape must be (8,) and dtype must be uint32
    output_distances : dict
        shape and dtype of distances, should be same dtype as input_queries
    output_flag : dict
        shape and dtype of flag, only the 1th and 17th is valid
    kernel_name : str
        kernel name, default value is "corr_compute"

    Returns
    -------
    None
    """
    corr_compute_obj = CorrCompute(input_code, input_actual_num,
                                   output_corr_result, output_flag, kernel_name)
    tik_instance = corr_compute_obj.get_tik_instance()
    return tik_instance
