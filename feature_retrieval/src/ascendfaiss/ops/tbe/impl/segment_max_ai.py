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


class SegmentMaxAI:
    def __init__(self,
                 input_data,
                 output_max,
                 output_flag,
                 kernel_name="segment_max_ai"):
        self.shape_data = input_data.get("shape")
        self.dtype_data = input_data.get("dtype")
        self.shape_max = output_max.get("shape")
        self.dtype_max = output_max.get("dtype")
        self.shape_flag = output_flag.get("shape")
        self.dtype_flag = output_flag.get("dtype")
        self.kernel_name = kernel_name

        # compute parameter
        self.batch, self.nlist = self.shape_data
        self.repeats = 128

        # check parameter
        self.check_parameter()

        # set vector fp32 mask and fp16 mask
        self.fp32_mask = 64
        self.fp16_mask = 128
        self.max_mask = 32
        self.max_size = self.nlist // self.max_mask * 2

        # set tik instance
        self.set_tik_instance()

    def check_parameter(self):
        if self.nlist % 32 != 0:
            raise RuntimeError("nlist must be a multiple of 32")

    def set_tik_instance(self):
        """
        set tik_instance
        """
        set_soc_info()
        self.tik_instance = tik.Tik()

        self.aicore_use = 2

        self.batch_each_loop = max(min(128 * 1024 // 2 // self.nlist, self.batch // 2), 1)
        self.batch_loops = min(2, self.batch_each_loop)
        self.batch_sub_loop = self.batch_each_loop // self.batch_loops
        self.set_src_dst_tensor()

    def set_src_dst_tensor(self):
        """
        set input and output tensor
        """
        # creat input tensor: input_data_gm
        # creat output tensor: output_max_gm  and output_flag_gm in global buffer
        self.input_data_gm = self.tik_instance.Tensor(
            self.dtype_data,
            self.shape_data,
            name="input_data_gm",
            scope=tik.scope_gm)
        self.output_max_gm = self.tik_instance.Tensor(
            self.dtype_max,
            self.shape_max,
            name="output_max_gm",
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
        self.batch_each_core = (self.batch + self.aicore_use - 1) // self.aicore_use
        self.batch_last_core = self.batch - (self.aicore_use - 1) * self.batch_each_core

    def compute_each_core(self, batch_offset, batch_num):
        batch_loops = batch_num // self.batch_each_loop
        with self.tik_instance.for_range(0, batch_loops) as loop_batch:
            self.compute_each_loop(batch_offset + loop_batch * self.batch_each_loop)

    def compute_each_loop(self, batch_offset):
        with self.tik_instance.for_range(0, self.batch_loops, thread_num=self.batch_loops) as loop_ub:
            data_ub = self.tik_instance.Tensor("float16",
                                               (self.batch_sub_loop, self.nlist),
                                               name="data_ub", scope=tik.scope_ubuf)
            self.tik_instance.data_move(data_ub,
                                        self.input_data_gm[batch_offset + loop_ub * self.batch_sub_loop, 0],
                                        0, 1, self.batch_sub_loop * self.nlist // 16, 0, 0)

            max_ub = self.tik_instance.Tensor("float16",
                                              (self.batch_sub_loop, self.max_size),
                                              name="max_ub", scope=tik.scope_ubuf)

            repeat_times = min(self.repeats, self.batch_sub_loop * self.nlist // self.max_mask)
            vcmax_loops = self.batch_sub_loop * self.nlist // self.max_mask // repeat_times
            for i in range(vcmax_loops):
                self.tik_instance.vcmax(self.max_mask,
                                        max_ub[(i * repeat_times * 2):],
                                        data_ub[(i * repeat_times * self.max_mask):],
                                        repeat_times, 1, 1, self.max_mask // 16)

            self.tik_instance.data_move(self.output_max_gm[batch_offset + loop_ub * self.batch_sub_loop, 0],
                                        max_ub,
                                        0, 1, self.batch_sub_loop * self.max_size // 16, 0, 0)

    def max_compute(self):
        with self.tik_instance.for_range(0, self.aicore_use, block_num=self.aicore_use) as block_index:
            self.cal_num_each_core()
            # compute coarse centroids num and move offset every core
            aicore_code_num = self.tik_instance.Scalar(dtype="uint32",
                                                       name="aicore_code_num",
                                                       init_value=0)
            with self.tik_instance.if_scope(block_index != self.aicore_use - 1):
                aicore_code_num.set_as(self.batch_each_core)
            with self.tik_instance.else_scope():
                aicore_code_num.set_as(self.batch_last_core)

            self.compute_each_core(block_index * self.batch_each_core, aicore_code_num)

            one = self.tik_instance.Scalar(dtype="uint16", name="one", init_value=1)
            flag_ub = self.tik_instance.Tensor("uint16", (16,), name="flag_ub", scope=tik.scope_ubuf)
            flag_ub[0].set_as(one)

            self.tik_instance.data_move(self.output_flag_gm[block_index, 0], flag_ub, 0, 1, 1, 0, 0)

    def get_tik_instance(self):
        """
        obtain tik instance
        """
        self.max_compute()

        self.tik_instance.BuildCCE(
            kernel_name=self.kernel_name,
            inputs=[self.input_data_gm],
            outputs=[self.output_max_gm, self.output_flag_gm])

        return self.tik_instance


def segment_max_ai(input_data,
                   output_max,
                   output_flag,
                   kernel_name="segment_max_ai"):
    """
    calculating distance

    Parameters
    ----------
    input_data : dict
        shape and dtype of data
    output_max : dict
        shape and dtype of max
    output_flag: dict
        shape and dtype of output flag,
        shape must be (32,) and dtype must be uint16
    kernel_name : str
        kernel name, default value is "distance_compute"

    Returns
    -------
    None
    """
    segment_max = SegmentMaxAI(input_data, output_max, output_flag, kernel_name)
    tik_instance = segment_max.get_tik_instance()
    return tik_instance
