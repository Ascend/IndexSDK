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

class FpToFp16:
    def __init__(self,
                 input_queries,
                 output_queries_fp16,
                 output_flag,
                 kernel_name="fp_to_fp16"):
        self.shape_queries = input_queries.get("shape")
        self.dtype_queries = input_queries.get("dtype")
        self.shape_queries_fp16 = output_queries_fp16.get("shape")
        self.dtype_queries_fp16 = output_queries_fp16.get("dtype")
        self.shape_flag = output_flag.get("shape")
        self.dtype_flag = output_flag.get("dtype")
        self.kernel_name = kernel_name

        # compute parameter
        self.queries_num, self.dim = self.shape_queries

        # check parameter
        self.check_parameter()

        # set vector fp32 mask and fp16 mask
        self.fp32_mask = 64
        self.max_repeat = 255

        # set tik instance
        tik_dprofile = tik.Dprofile("v200", "aic")
        self.tik_instance = tik.Tik(tik_dprofile)
        self.available_core_num = 8

        self.aicore_use = self.available_core_num if self.queries_num > self.available_core_num else self.queries_num

        self.queries_num_each_core = self.queries_num // self.aicore_use
        self.batch_tail = self.queries_num % self.aicore_use
        self.queries_num_last_core = self.queries_num_each_core + self.batch_tail
        self.set_src_dst_tensor()

        # set ti_ling
        self.ti_ling()

    def ti_ling(self):
        self.block_size = 128 * 1024 # 128 KB
        self.query_num_per_core = self.block_size // (self.dim * 4)
        self.query_num_per_round = self.query_num_per_core * self.aicore_use

        # calculate rounds
        self.rounds = (self.queries_num * self.dim * 4 +
            (self.aicore_use * self.query_num_per_core * self.dim * 4 - 1)) // (self.aicore_use *
            self.query_num_per_core * self.dim * 4)

        # per round: per core handle query nums
        self.query_num_last_round = (self.queries_num -
            (self.query_num_per_round * (self.rounds - 1))) // self.aicore_use
        self.query_num_last_round_core = self.query_num_last_round +\
            (self.queries_num - (self.query_num_per_round * (self.rounds - 1))) % self.aicore_use

    def set_src_dst_tensor(self):
        self.input_queries_gm = self.tik_instance.Tensor(self.dtype_queries, self.shape_queries,
                                                         name="input_queries_gm", scope=tik.scope_gm)
        self.output_queries_fp16_gm = self.tik_instance.Tensor(self.dtype_queries_fp16, self.shape_queries_fp16,
                                                         name="output_queries_fp16_gm", scope=tik.scope_gm)
        self.output_flag_gm = self.tik_instance.Tensor(self.dtype_flag, self.shape_flag,
                                                       name="output_flag_gm", scope=tik.scope_gm)

    def check_parameter(self):
        if self.dim % 32 != 0:
            raise RuntimeError("feature dim must be a multiple of 32")

    def fp2fp16_each_core(self, aicore_offset, query_num):
        fp_ub = self.tik_instance.Tensor("float32", (query_num, self.dim), name="fp_ub", scope=tik.scope_ubuf)
        fp16_ub = self.tik_instance.Tensor("float16", (query_num, self.dim), name="fp16_ub", scope=tik.scope_ubuf)

        self.tik_instance.data_move(fp_ub, self.input_queries_gm[aicore_offset, 0], 0, 1,
                                    query_num * self.dim // 8, 0, 0)
        vconv_repeats = query_num * self.dim // self.fp32_mask
        vconv_repeats_loops = vconv_repeats // 255
        offset = 0
        for _ in range(vconv_repeats_loops):
            self.tik_instance.vconv(self.fp32_mask, 'none', fp16_ub[offset], fp_ub[offset], 255, 1, 1, 4, 8)
            offset += 255 * self.fp32_mask
        vconv_repeats_remain = ((query_num * self.dim) - offset) // self.fp32_mask
        if (vconv_repeats_remain > 0):
            self.tik_instance.vconv(self.fp32_mask, 'none', fp16_ub[offset], fp_ub[offset],
                                    vconv_repeats_remain, 1, 1, 4, 8)
            offset += vconv_repeats_remain * self.fp32_mask
        remain = query_num * self.dim % self.fp32_mask
        if (remain > 0):
            self.tik_instance.vconv(remain, 'none', fp16_ub[offset], fp_ub[offset], 1, 1, 1, 4, 8)
        self.tik_instance.data_move(self.output_queries_fp16_gm[aicore_offset, 0], fp16_ub, 0, 1,
                                    query_num * self.dim // 16, 0, 0)

    def convert(self):
        """
        the compute process
        """
        for round_ in range(self.rounds):
            if round_ != self.rounds - 1:
                with self.tik_instance.for_range(0, self.aicore_use, block_num=self.aicore_use) as block_index:
                    self.fp2fp16_each_core(self.query_num_per_round * round_ +
                        self.query_num_per_core * block_index, self.query_num_per_core)

                    one = self.tik_instance.Scalar(dtype="uint16", name="one", init_value=1)
                    flag_ub = self.tik_instance.Tensor("uint16", (16,), name="flag_ub", scope=tik.scope_ubuf)
                    flag_ub[0].set_as(one)
                    self.tik_instance.data_move(self.output_flag_gm[block_index, 0],
                                                flag_ub, 0, 1, 1, 0, 0)
            else:
                with self.tik_instance.for_range(0, self.aicore_use, block_num=self.aicore_use) as block_index:
                    with self.tik_instance.if_scope(block_index != self.aicore_use - 1):
                        self.fp2fp16_each_core(self.query_num_per_round * round_ +
                            self.query_num_last_round * block_index, self.query_num_last_round)
                    with self.tik_instance.else_scope():
                        self.fp2fp16_each_core(self.query_num_per_round * round_ +
                            self.query_num_last_round * block_index, self.query_num_last_round_core)

                    one = self.tik_instance.Scalar(dtype="uint16", name="one", init_value=1)
                    flag_ub = self.tik_instance.Tensor("uint16", (16,), name="flag_ub", scope=tik.scope_ubuf)
                    flag_ub[0].set_as(one)
                    self.tik_instance.data_move(self.output_flag_gm[block_index, 0],
                                                flag_ub, 0, 1, 1, 0, 0)

    def get_tik_instance(self):
        """
        obtain tik instance
        """
        self.convert()

        self.tik_instance.BuildCCE(kernel_name=self.kernel_name,
                                   inputs=[
                                       self.input_queries_gm,
                                   ],
                                   outputs=[self.output_queries_fp16_gm,
                                            self.output_flag_gm])
        return self.tik_instance

def fp_to_fp16(input_queries, output_queries_fp16, output_flag,
                                kernel_name="fp_to_fp16"):

    fp_to_fp16_ = FpToFp16(input_queries,
                            output_queries_fp16,
                            output_flag,
                            kernel_name)
    tik_instance_ = fp_to_fp16_.get_tik_instance()
    return tik_instance_