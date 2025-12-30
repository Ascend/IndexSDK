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


class L2NormFlatSub:
    def __init__(self,
                 input_queries,
                 output_norms,
                 output_typing_queries,
                 kernel_name="l2_norm_flat_sub"):
        self.shape_queries = input_queries.get("shape")
        self.dtype_queries = input_queries.get("dtype")
        self.shape_norms = output_norms.get("shape")
        self.dtype_norms = output_norms.get("dtype")
        self.shape_typing_queries = output_typing_queries.get("shape")
        self.dtype_typing_queries = output_typing_queries.get("dtype")
        self.kernel_name = kernel_name

        # compute parameter
        self.queries_num, self.dim = self.shape_queries

        # check parameter
        self.check_parameter()

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

        self.aicore_use = 4

        self.set_src_dst_tensor()

    def set_src_dst_tensor(self):
        """
        set input and output tensor
        """
        # creat input tensor: input_queries_gm
        # and output tensor: output_norms_gm, in global buffer
        self.input_queries_gm = self.tik_instance.Tensor(
            self.dtype_queries,
            self.shape_queries,
            name="input_queries_gm",
            scope=tik.scope_gm)
        self.output_norms_gm = self.tik_instance.Tensor(
            self.dtype_norms,
            self.shape_norms,
            name="output_norms_gm",
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
        # 64 // 4 = 16
        self.query_num_each_core = self.queries_num // self.aicore_use

    def compute_each_core(self, aicore_offset, aicore_num):
        queries_l1 = self.tik_instance.Tensor("float16",
                                              (self.dim // 16, aicore_num, 16),  # (dim // 16, 16, 16)
                                              name="queries_l1", scope=tik.scope_cbuf)

        with self.tik_instance.for_range(0, self.dim // 16) as i:
            self.tik_instance.data_move(queries_l1[i, 0, 0],
                                        self.input_queries_gm[aicore_offset, 16 * i],
                                        0, aicore_num, 1, self.dim // 16 - 1, 0)

        queries_square_l0c = self.tik_instance.Tensor("float32",
                                                      (aicore_num // 16, aicore_num, 16),     # (1, 16, 16)
                                                      name="queries_square_l0c", scope=tik.scope_cbuf_out)
        self.tik_instance.matmul(queries_square_l0c,
                                 queries_l1, queries_l1,
                                 aicore_num, self.dim, aicore_num)
        
        queries_square_ub = self.tik_instance.Tensor("float32",
                                                (aicore_num, 16),
                                                name="queries_square_ub", scope=tik.scope_ubuf)
        self.tik_instance.tensor_mov(queries_square_ub,
                                     queries_square_l0c,
                                     'm',
                                     aicore_num // 16, 1, 0, aicore_num // 16)

        queries_l2_ub = self.tik_instance.Tensor("float32",
                                                 (aicore_num, 16),
                                                 name="queries_l2_ub", scope=tik.scope_ubuf)
        with self.tik_instance.for_range(0, aicore_num) as i:
            square = self.tik_instance.Scalar(dtype="float32", init_value=queries_square_ub[i, i % 16])
            self.tik_instance.vector_dup(16, queries_l2_ub[i * 16], square, 1, 1, 2, 0)

        self.tik_instance.data_move(self.output_norms_gm[aicore_offset * 16],
                                    queries_l2_ub,
                                    0, 1, aicore_num * 2, 0, 0)

        queries_ub = self.tik_instance.Tensor("float16",
                                              (self.dim // 16, aicore_num, 16),
                                              name="queries_ub", scope=tik.scope_ubuf)

        with self.tik_instance.for_range(0, self.dim // 16) as i:
            self.tik_instance.data_move(queries_ub[i, 0, 0],
                                        queries_l1[i, 0, 0],
                                        0, 1, aicore_num, 0, 0)

        with self.tik_instance.for_range(0, self.dim // 16) as i:
            self.tik_instance.data_move(self.output_typing_queries_gm[i * self.queries_num * 16 + aicore_offset * 16],
                                        queries_ub[i, 0, 0],
                                        0, 1, aicore_num, 0, 0)

    def compute_norms(self):
        """
        the compute process
        """
        self.cal_num_each_core()

        with self.tik_instance.for_range(0, self.aicore_use, block_num=self.aicore_use) as block_index:
            self.compute_each_core(block_index * self.query_num_each_core, self.query_num_each_core)

    def get_tik_instance(self):
        """
        obtain tik instance
        """
        self.compute_norms()

        self.tik_instance.BuildCCE(kernel_name=self.kernel_name,
                                   inputs=[self.input_queries_gm],
                                   outputs=[self.output_norms_gm, self.output_typing_queries_gm])

        return self.tik_instance


def l2_norm_flat_sub(input_queries,
                     output_norms,
                     output_typing_queries,
                     kernel_name="l2_norm_flat_sub"):
    """
    calculating distance

    Parameters
    ----------
    input_queries : dict
        shape and dtype of query vector
    output_norms : dict
        shape and dtype of l2 norm
    output_typing_queries : dict
        shape and dtype of typing queries
    kernel_name : str
        kernel name, default value is "l2_norm_flat_sub"

    Returns
    -------
    None
    """
    norm = L2NormFlatSub(input_queries, output_norms, output_typing_queries, kernel_name)
    tik_instance = norm.get_tik_instance()
    return tik_instance
