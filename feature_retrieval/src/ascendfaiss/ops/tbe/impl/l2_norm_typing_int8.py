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


class L2NormTypingInt8:
    def __init__(self,
                 input_queries,
                 input_transfer,
                 input_actual_num,
                 output_norms,
                 output_typing_queries,
                 kernel_name="l2_norm_typing_int8"):
        self.shape_queries = input_queries.get("shape")
        self.dtype_queries = input_queries.get("dtype")
        self.shape_transfer = input_transfer.get("shape")
        self.dtype_transfer = input_transfer.get("dtype")
        self.shape_actual_num = input_actual_num.get("shape")
        self.dtype_actual_num = input_actual_num.get("dtype")
        self.shape_norms = output_norms.get("shape")
        self.dtype_norms = output_norms.get("dtype")
        self.shape_typing_queries = output_typing_queries.get("shape")
        self.dtype_typing_queries = output_typing_queries.get("dtype")
        self.kernel_name = kernel_name

        # compute parameter
        self.queries_num, self.dim = self.shape_queries
        self.max_repeats = 128
        self.transfer_mask = 64
        self.vcadd_mask = 16

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

        self.aicore_use = 2

        self.query_each_loop = 256
        self.query_loops = 2
        self.query_sub_loop = self.query_each_loop // self.query_loops

        self.set_src_dst_tensor()

    def set_src_dst_tensor(self):
        """
        set input and output tensor
        """
        # creat input tensor: input_queries_gm, input_transfer_gm, input_actual_num_gm
        # and output tensor: output_norms_gm, output_typing_queries_gm, in global buffer
        self.input_queries_gm = self.tik_instance.Tensor(
            self.dtype_queries,
            self.shape_queries,
            name="input_queries_gm",
            scope=tik.scope_gm)
        self.input_transfer_gm = self.tik_instance.Tensor(
            self.dtype_transfer,
            self.shape_transfer,
            name="input_transfer_gm",
            scope=tik.scope_gm)
        self.input_actual_num_gm = self.tik_instance.Tensor(
            self.dtype_actual_num,
            self.shape_actual_num,
            name="input_actual_num_gm",
            scope=tik.scope_gm)
        self.output_norms_gm = self.tik_instance.Tensor(
            self.dtype_norms,
            self.shape_norms,
            name="output_distances_gm",
            scope=tik.scope_gm)
        self.output_typing_queries_gm = self.tik_instance.Tensor(
            self.dtype_typing_queries,
            self.shape_typing_queries,
            name="output_typing_queries_gm",
            scope=tik.scope_gm)

    def cal_num_each_core(self):
        """
        calculate actual query num of each core
        """
        # move actual query num from out to UB
        actual_num_ub = self.tik_instance.Tensor("uint32",
                                                 (8,),
                                                 name="actual_num_ub", scope=tik.scope_ubuf)
        self.tik_instance.data_move(actual_num_ub,
                                    self.input_actual_num_gm,
                                    0, 1, 1, 0, 0)
        actual_num = self.tik_instance.Scalar(dtype="uint32",
                                              name="actual_num",
                                              init_value=0)
        actual_num.set_as(actual_num_ub[0])
        self.query_num_each_core = (actual_num // self.aicore_use + self.query_each_loop // 2) \
            // self.query_each_loop * self.query_each_loop
        self.query_num_last_core = actual_num - (self.aicore_use - 1) * self.query_num_each_core

    def compute_each_core(self, aicore_offset, aicore_num):
        transfer_ub = self.tik_instance.Tensor("int32",
                                               (self.query_each_loop, 16),
                                               name="transfer_ub", scope=tik.scope_ubuf)
        self.tik_instance.data_move(transfer_ub,
                                    self.input_transfer_gm,
                                    0, 1, self.query_each_loop * 2, 0, 0)

        # compute xy using cube
        query_loops = aicore_num // self.query_each_loop
        with self.tik_instance.if_scope(query_loops > 0):
            with self.tik_instance.for_range(0, query_loops) as loop_query:
                self.cube_compute_each_loop(transfer_ub, aicore_offset, loop_query * self.query_each_loop)

        query_last_num = aicore_num % self.query_each_loop
        with self.tik_instance.if_scope(query_last_num > 0):
            self.cube_compute_last(transfer_ub, aicore_offset, query_loops * self.query_each_loop, query_last_num)

    def cube_compute_each_loop(self, transfer_ub, aicore_offset, query_offset):
        with self.tik_instance.for_range(0, self.query_loops, thread_num=2) as loop_ub:
            queries_l1 = self.tik_instance.Tensor("int8",
                                                  (self.dim // 32, self.query_sub_loop, 32),
                                                  name="queries_l1", scope=tik.scope_cbuf)

            with self.tik_instance.for_range(0, self.dim // 32) as i:
                self.tik_instance.data_move(queries_l1[i, 0, 0],
                                            self.input_queries_gm[aicore_offset + query_offset
                                                                  + loop_ub * self.query_sub_loop, 32 * i],
                                            0, self.query_sub_loop, 1, self.dim // 32 - 1, 0)

            queries_square_l0c = self.tik_instance.Tensor("int32",
                                                          (self.query_sub_loop // 16, self.query_sub_loop, 16),
                                                          name="queries_square_l0c", scope=tik.scope_cbuf_out)

            self.tik_instance.matmul(queries_square_l0c,
                                     queries_l1, queries_l1,
                                     self.query_sub_loop, self.dim, self.query_sub_loop)

            # move x^2 from LoC to UB
            queries_square_ub = self.tik_instance.Tensor("int32",
                                                         (self.query_sub_loop, 16),
                                                         name="queries_square_ub", scope=tik.scope_ubuf)
            self.tik_instance.tensor_mov(queries_square_ub,
                                         queries_square_l0c,
                                         'm',
                                         self.query_sub_loop // 16, 1, 0, self.query_sub_loop // 16)

            vmul_repeats = 16 * self.query_sub_loop // self.transfer_mask
            self.tik_instance.vmul(self.transfer_mask,
                                   queries_square_ub,
                                   queries_square_ub, transfer_ub[loop_ub * self.query_sub_loop, 0],
                                   vmul_repeats, 1, 1, 1, 8, 8, 8)

            # convert int32 to fp32
            queries_square_fp_ub = self.tik_instance.Tensor("float32",
                                                         (self.query_sub_loop, 16),
                                                         name="queries_square_fp_ub", scope=tik.scope_ubuf)

            repeat_times = self.query_sub_loop * 16 // self.fp32_mask
            self.tik_instance.vconv(self.fp32_mask, "none", queries_square_fp_ub, queries_square_ub,
                                    repeat_times, 1, 1, 8, 8)

            queries_l2_ub = self.tik_instance.Tensor("float32",
                                                     (self.query_sub_loop,),
                                                     name="queries_l2_ub", scope=tik.scope_ubuf)

            self.tik_instance.vcadd(self.vcadd_mask,
                                    queries_l2_ub, queries_square_fp_ub,
                                    self.query_sub_loop, 1, 1, 2)

            queries_l2_int_ub = self.tik_instance.Tensor("int32",
                                                     (self.query_sub_loop,),
                                                     name="queries_l2_int_ub", scope=tik.scope_ubuf)

            self.tik_instance.vconv(self.fp32_mask, "round", queries_l2_int_ub, queries_l2_ub,
                                    self.query_sub_loop // self.fp32_mask, 1, 1, 8, 8)

            self.tik_instance.data_move(
                self.output_norms_gm[aicore_offset + query_offset + loop_ub * self.query_sub_loop],
                queries_l2_int_ub,
                0, 1, self.query_sub_loop // 8, 0, 0)

            # re-typing
            queries_ub = self.tik_instance.Tensor("int8",
                                                  (self.query_sub_loop // 32, self.dim // 32, 32, 32),
                                                  name="queries_ub", scope=tik.scope_ubuf)
            for i in range(self.dim // 32):
                self.tik_instance.data_move(queries_ub[0, i, 0, 0],
                                            queries_l1[i, 0, 0],
                                            0, self.query_sub_loop // 32, 32, 0, self.dim - 32)

            self.tik_instance.data_move(
                self.output_typing_queries_gm[(aicore_offset + query_offset + loop_ub * self.query_sub_loop) // 32,
                                              0, 0, 0],
                queries_ub,
                0, 1, self.query_sub_loop * self.dim // 32, 0, 0)

    def cube_compute_last(self, transfer_ub, aicore_offset, query_offset, query_num):
        queries_l1 = self.tik_instance.Tensor("int8",
                                              (self.dim // 32, self.query_each_loop, 32),
                                              name="queries_l1", scope=tik.scope_cbuf)

        with self.tik_instance.for_range(0, self.dim // 32) as i:
            self.tik_instance.data_move(queries_l1[i, 0, 0],
                                        self.input_queries_gm[aicore_offset + query_offset, 32 * i],
                                        0, query_num, 1, self.dim // 32 - 1, 0)

        queries_square_l0c = self.tik_instance.Tensor("int32",
                                                      (self.query_each_loop // 16, self.query_each_loop, 16),
                                                      name="queries_square_l0c", scope=tik.scope_cbuf_out)
        self.tik_instance.matmul(queries_square_l0c,
                                 queries_l1, queries_l1,
                                 self.query_each_loop, self.dim, self.query_each_loop)

        # move x^2 from LoC to UB
        queries_square_ub = self.tik_instance.Tensor("int32",
                                                     (self.query_each_loop, 16),
                                                     name="queries_square_ub", scope=tik.scope_ubuf)
        self.tik_instance.tensor_mov(queries_square_ub,
                                     queries_square_l0c,
                                     'm',
                                     self.query_each_loop // 16, 1, 0, self.query_each_loop // 16)

        vmul_repeats = 16 * self.query_each_loop // self.transfer_mask
        self.tik_instance.vmul(self.transfer_mask,
                               queries_square_ub,
                               queries_square_ub, transfer_ub,
                               vmul_repeats, 1, 1, 1, 8, 8, 8)
        # convert int32 to fp32
        queries_square_fp_ub = self.tik_instance.Tensor("float32",
                                                        (self.query_each_loop, 16),
                                                        name="queries_square_fp_ub", scope=tik.scope_ubuf)

        repeat_times = self.query_each_loop * 16 // self.fp32_mask
        self.tik_instance.vconv(self.fp32_mask, "none", queries_square_fp_ub, queries_square_ub,
                                repeat_times, 1, 1, 8, 8)

        queries_l2_ub = self.tik_instance.Tensor("float32",
                                                 (self.query_each_loop,),
                                                 name="queries_l2_ub", scope=tik.scope_ubuf)
        self.tik_instance.vcadd(self.vcadd_mask,
                                queries_l2_ub, queries_square_fp_ub,
                                query_num, 1, 1, 2)

        queries_l2_int_ub = self.tik_instance.Tensor("int32",
                                                     (self.query_each_loop,),
                                                     name="queries_l2_int_ub", scope=tik.scope_ubuf)

        self.tik_instance.vconv(self.fp32_mask, "round", queries_l2_int_ub, queries_l2_ub,
                                self.query_each_loop // self.fp32_mask, 1, 1, 8, 8)

        self.tik_instance.data_move(self.output_norms_gm[aicore_offset + query_offset],
                                    queries_l2_int_ub,
                                    0, 1, self.query_each_loop // 8, 0, 0)

        # re-typing
        queries_ub = self.tik_instance.Tensor("int8",
                                              (self.query_each_loop // 32, self.dim // 32, 32, 32),
                                              name="queries_ub", scope=tik.scope_ubuf)
        for i in range(self.dim // 32):
            self.tik_instance.data_move(queries_ub[0, i, 0, 0],
                                        queries_l1[i, 0, 0],
                                        0, self.query_each_loop // 32, 32, 0, self.dim - 32)

        self.tik_instance.data_move(
            self.output_typing_queries_gm[(aicore_offset + query_offset) // 32,
                                          0, 0, 0],
            queries_ub,
            0, 1, self.query_each_loop * self.dim // 32, 0, 0)

    def compute_norms(self):
        """
        the compute process
        """
        self.cal_num_each_core()

        with self.tik_instance.for_range(0, self.aicore_use, block_num=self.aicore_use) as block_index:
            # compute coarse query num and move offset every core
            aicore_queries_num = self.tik_instance.Scalar(dtype="uint32", name="aicore_code_num", init_value=0)
            with self.tik_instance.if_scope(block_index != self.aicore_use - 1):
                aicore_queries_num.set_as(self.query_num_each_core)
            with self.tik_instance.else_scope():
                aicore_queries_num.set_as(self.query_num_last_core)

            self.compute_each_core(block_index * self.query_num_each_core, aicore_queries_num)

    def get_tik_instance(self):
        """
        obtain tik instance
        """
        self.compute_norms()

        self.tik_instance.BuildCCE(kernel_name=self.kernel_name,
                                   inputs=[self.input_queries_gm, self.input_transfer_gm, self.input_actual_num_gm],
                                   outputs=[self.output_norms_gm, self.output_typing_queries_gm])

        return self.tik_instance


def l2_norm_typing_int8(input_queries,
                        input_transfer,
                        input_actual_num,
                        output_norms,
                        output_typing_queries,
                        kernel_name="square"):
    """
    calculating distance

    Parameters
    ----------
    input_queries : dict
        shape and dtype of query vector
    input_transfer : dict
        shape and dtype of transfer vector
    input_actual_num: dict
        shape and dtype of actual query num,
        shape must be (8,) and dtype must be uint32
    output_norms : dict
        shape and dtype of distances, should be same dtype as input_queries
    output_typing_queries : dict
        shape and dtype of typing queries
    kernel_name : str
        kernel name, default value is "l2_norm_typing_int8"

    Returns
    -------
    None
    """
    norm = L2NormTypingInt8(input_queries, input_transfer, input_actual_num, output_norms,
                            output_typing_queries, kernel_name)
    tik_instance = norm.get_tik_instance()
    return tik_instance