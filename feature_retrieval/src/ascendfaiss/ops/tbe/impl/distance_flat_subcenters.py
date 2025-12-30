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


class DistanceFlatSubcenters:
    def __init__(self,
                 input_queries,
                 input_subcenters,
                 input_precomputed,
                 input_offsets,
                 output_distances,
                 output_flag,
                 kernel_name="distance_flat_subcenters"):
        self.shape_queries = input_queries.get("shape")
        self.dtype_queries = input_queries.get("dtype")
        self.shape_subcenters = input_subcenters.get("shape")
        self.dtype_subcenters = input_subcenters.get("dtype")
        self.shape_precomputed = input_precomputed.get("shape")
        self.dtype_precomputed = input_precomputed.get("dtype")
        self.shape_offsets = input_offsets.get("shape")
        self.dtype_offsets = input_offsets.get("dtype")
        self.shape_distances = output_distances.get("shape")
        self.dtype_distances = output_distances.get("dtype")
        self.shape_flag = output_flag.get("shape")
        self.dtype_flag = output_flag.get("dtype")
        self.kernel_name = kernel_name

        # compute parameter
        self.queries_num, self.dim = self.shape_queries
        _, self.code_num = self.shape_precomputed
        _, self.picked_num = self.shape_offsets
        self.queries_align = 16
        self.picked_align = (self.picked_num + 15) // 16 * 16

        # check parameter
        self.check_parameter()

        # set vector fp32 mask and fp16 mask
        self.fp32_mask = 64

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
        self.tik_instance = tik.Tik()

        self.aicore_use = self.shape_flag[0]
        self.query_each_core = self.queries_num // self.aicore_use
        self.query_last_core = self.queries_num - (self.aicore_use - 1) * self.query_each_core
        self.coarse_each_loop = 4

        self.set_src_dst_tensor()

    def set_src_dst_tensor(self):
        """
        set input and output tensor
        """
        self.coeff = self.tik_instance.Scalar("float32", name="coeff", init_value=-2.0)

        # creat input tensor: input_queries_gm, input_code_gm
        # and input_precomputed_gm
        # and output tensor: output_distances_gm, output_flag_gm in global buffer
        self.input_queries_gm = self.tik_instance.Tensor(
            self.dtype_queries,
            self.shape_queries,
            name="input_queries_gm",
            scope=tik.scope_gm)
        self.input_subcenters_gm = self.tik_instance.Tensor(
            self.dtype_subcenters,
            self.shape_subcenters,
            name="input_subcenters_gm",
            scope=tik.scope_gm)
        self.input_precomputed_gm = self.tik_instance.Tensor(
            self.dtype_precomputed,
            self.shape_precomputed,
            name="input_precomputed_gm",
            scope=tik.scope_gm)
        self.input_offsets_gm = self.tik_instance.Tensor(
            self.dtype_offsets,
            self.shape_offsets,
            name="input_offsets_gm",
            scope=tik.scope_gm)
        self.output_distances_gm = self.tik_instance.Tensor(
            self.dtype_distances,
            self.shape_distances,
            name="output_distances_gm",
            scope=tik.scope_gm)
        self.output_flag_gm = self.tik_instance.Tensor(
            self.dtype_flag,
            self.shape_flag,
            name="output_flag_gm",
            scope=tik.scope_gm)

        # construct offsets tensor in ub
        self.offsets_ub = self.tik_instance.Tensor("uint16", self.shape_offsets,
                                                   name="offsets_ub", scope=tik.scope_ubuf)
        self.tik_instance.data_move(self.offsets_ub, self.input_offsets_gm,
                                    0, 1, self.queries_num * self.picked_align // 16, 0, 0)

    def distance_compute_each_picked(self, queries_l1, query_offset, query_index, picked_index):
        query_global_idx = query_offset + query_index
        subcenter_idxs = [self.tik_instance.Scalar("uint16") for i in range(self.coarse_each_loop)]
        for i in range(self.coarse_each_loop):
            subcenter_idxs[i].set_as(self.offsets_ub[query_global_idx, self.coarse_each_loop * picked_index + i])

        code_l1 = self.tik_instance.Tensor("float16",
                                           (self.coarse_each_loop, self.dim // 16, self.code_num, 16),
                                           name="code_l1", scope=tik.scope_cbuf)
        for i in range(self.coarse_each_loop):
            self.tik_instance.data_move(code_l1[i, 0, 0, 0],
                                        self.input_subcenters_gm[subcenter_idxs[i], 0, 0, 0],
                                        0, 1, self.code_num * self.dim // 16, 0, 0)

        inner_product_l0c = self.tik_instance.Tensor("float32",
                                                     (self.coarse_each_loop * self.code_num // 16,
                                                      self.queries_align, 16),
                                                     name="inner_product_l0c", scope=tik.scope_cbuf_out)
        for i in range(self.coarse_each_loop):
            self.tik_instance.matmul(inner_product_l0c[(i * self.code_num * self.queries_align):],
                                     queries_l1, code_l1[(i * self.code_num * self.dim):],
                                     self.queries_align, self.dim, self.code_num)

        precomputed_ub_fp16 = self.tik_instance.Tensor("float16",
                                                       (self.coarse_each_loop * self.code_num,),
                                                       name="precomputed_ub_fp16", scope=tik.scope_ubuf)
        for i in range(self.coarse_each_loop):
            self.tik_instance.data_move(precomputed_ub_fp16[i * self.code_num],
                                        self.input_precomputed_gm[subcenter_idxs[i], 0],
                                        0, 1, self.code_num // 16, 0, 0)
        precomputed_ub = self.tik_instance.Tensor("float32",
                                                  (self.coarse_each_loop * self.code_num,),
                                                  name="precomputed_ub", scope=tik.scope_ubuf)

        vconv_repeat_times = self.coarse_each_loop * self.code_num // self.fp32_mask
        vconv_offset = 0
        if vconv_repeat_times > 0:
            self.tik_instance.vconv(self.fp32_mask, "none",
                                    precomputed_ub[vconv_offset],
                                    precomputed_ub_fp16[vconv_offset],
                                    vconv_repeat_times, 1, 1, 8, 4)
            vconv_offset += vconv_repeat_times * self.fp32_mask
        vconv_last_num = self.coarse_each_loop * self.code_num % self.fp32_mask
        if vconv_last_num > 0:
            self.tik_instance.vconv(vconv_last_num, "none",
                                    precomputed_ub[vconv_offset],
                                    precomputed_ub_fp16[vconv_offset],
                                    1, 1, 1, 8, 4)

        inner_product_ub = self.tik_instance.Tensor("float32",
                                                    (self.coarse_each_loop * self.code_num // 16,
                                                     self.queries_align, 16),
                                                    name="inner_product_ub", scope=tik.scope_ubuf)
        self.tik_instance.data_move(inner_product_ub,
                                    inner_product_l0c,
                                    0, 1, self.coarse_each_loop * self.code_num * self.queries_align // 256, 0, 0)
        self.tik_instance.vaxpy(16,
                                precomputed_ub,
                                inner_product_ub, self.coeff,
                                self.coarse_each_loop * self.code_num // 16, 1, 1, 2, self.queries_align * 2)

        dst_ub = self.tik_instance.Tensor("float16",
                                          (self.coarse_each_loop * self.code_num,),
                                          name="dst_ub", scope=tik.scope_ubuf)
        vconv_repeat_times = self.coarse_each_loop * self.code_num // self.fp32_mask
        vconv_offset = 0
        if vconv_repeat_times > 0:
            self.tik_instance.vconv(self.fp32_mask, "none",
                                    dst_ub[vconv_offset],
                                    precomputed_ub[vconv_offset],
                                    vconv_repeat_times, 1, 1, 4, 8)
            vconv_offset += vconv_repeat_times * self.fp32_mask
        vconv_last_num = self.coarse_each_loop * self.code_num % self.fp32_mask
        if vconv_last_num > 0:
            self.tik_instance.vconv(vconv_last_num, "none",
                                    dst_ub[vconv_offset],
                                    precomputed_ub[vconv_offset],
                                    vconv_repeat_times, 1, 1, 4, 8)

        self.tik_instance.data_move(self.output_distances_gm[query_global_idx,
                                                             self.coarse_each_loop * picked_index, 0],
                                    dst_ub,
                                    0, 1, self.coarse_each_loop * self.code_num // 16, 0, 0)

    def distance_compute_each_query(self, query_offset, query_index):
        queries_l1 = self.tik_instance.Tensor("float16",
                                              (self.dim // 16, self.queries_align, 16),
                                              name="queries_l1", scope=tik.scope_cbuf)
        self.tik_instance.data_move(queries_l1,
                                    self.input_queries_gm[query_offset + query_index, 0],
                                    0, self.dim // 16, 1, 0, self.queries_align - 1)

        with self.tik_instance.for_range(0, self.picked_num // self.coarse_each_loop, thread_num=2) as picked_index:
            self.distance_compute_each_picked(queries_l1, query_offset, query_index, picked_index)

    def distance_compute_flat(self):
        """
        the compute process
        """
        with self.tik_instance.for_range(0, self.aicore_use, block_num=self.aicore_use) as block_index:
            # compute coarse code num and move offset every core
            with self.tik_instance.if_scope(block_index != self.aicore_use - 1):
                aicore_query_num = self.query_each_core
            with self.tik_instance.else_scope():
                aicore_query_num = self.query_last_core
            aicore_query_start = block_index * self.query_each_core

            with self.tik_instance.for_range(0, aicore_query_num) as query_index:
                self.distance_compute_each_query(aicore_query_start, query_index)

            one = self.tik_instance.Scalar(dtype="uint16", name="one", init_value=1)
            flag_ub = self.tik_instance.Tensor("uint16",
                                               (16,),
                                               name="flag_ub", scope=tik.scope_ubuf)
            flag_ub[0].set_as(one)
            self.tik_instance.pipe_barrier("PIPE_MTE3")
            self.tik_instance.data_move(self.output_flag_gm[block_index, 0],
                                        flag_ub, 0, 1, 1, 0, 0)

    def get_tik_instance(self):
        """
        obtain tik instance
        """
        self.distance_compute_flat()

        self.tik_instance.BuildCCE(kernel_name=self.kernel_name,
                                   inputs=[
                                       self.input_queries_gm,
                                       self.input_subcenters_gm,
                                       self.input_precomputed_gm,
                                       self.input_offsets_gm
                                   ],
                                   outputs=[self.output_distances_gm, self.output_flag_gm])

        return self.tik_instance


def distance_flat_subcenters(input_queries,
                             input_subcenters,
                             input_precomputed,
                             input_offsets,
                             output_distances,
                             output_flag,
                             kernel_name="distance_flat_subcenters"):
    """
    calculating distance

    Parameters
    ----------
    input_queries : dict
        shape and dtype of query vector
    input_code : dict
        shape and dtype of coarse code
    input_precomputed : dict
        shape and dtype of precomputed L2 distance of coarse code
    output_distances : dict
        shape and dtype of distances, should be same dtype as input_queries
    output_flag : dict
        shape and dtype of flag, only the 1th and 17th is valid
    kernel_name : str
        kernel name, default value is "distance_flat_subcenters"

    Returns
    -------
    None
    """
    distance_flat = DistanceFlatSubcenters(input_queries, input_subcenters, input_precomputed, input_offsets,
                                           output_distances, output_flag, kernel_name)
    tik_instance = distance_flat.get_tik_instance()
    return tik_instance
