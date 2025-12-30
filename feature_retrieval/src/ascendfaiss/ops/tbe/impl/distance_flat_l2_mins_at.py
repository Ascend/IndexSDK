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


class DistanceFlatL2MinsAt:
    def __init__(self,
                 input_queries,
                 input_code,
                 input_queries_norm,
                 input_code_norm,
                 output_distances,
                 output_mins,
                 output_flag,
                 kernel_name="distance_flat_l2_mins_at"):
        self.shape_queries = input_queries.get("shape")
        self.dtype_queries = input_queries.get("dtype")
        self.shape_code = input_code.get("shape")
        self.dtype_code = input_code.get("dtype")
        self.shape_queries_norm = input_queries_norm.get("shape")
        self.dtype_queries_norm = input_queries_norm.get("dtype")
        self.shape_code_norm = input_code_norm.get("shape")
        self.dtype_code_norm = input_code_norm.get("dtype")
        self.shape_distances = output_distances.get("shape")
        self.dtype_distances = output_distances.get("dtype")
        self.shape_mins = output_mins.get("shape")
        self.dtype_mins = output_mins.get("dtype")
        self.shape_flag = output_flag.get("shape")
        self.dtype_flag = output_flag.get("dtype")
        self.kernel_name = kernel_name

        self.code_each_loop = 512

        # compute parameter
        self.queries_num = self.shape_queries_norm[0]
        self.query_segment = self.shape_queries[2]
        self.dim = self.shape_queries[1] * 16
        self.code_num = self.shape_code[0] * self.code_each_loop
        self.min_num = self.shape_mins[1]
        self.cube_align = 16
        self.min_batch = 64
        self.max_repeats = 128
        self.transfer_mask = 64
        self.vcadd_mask = 16

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

        self.query_each_loop = 32
        self.query_loops = 2
        self.query_sub_loop = self.query_each_loop // self.query_loops

        self.set_src_dst_tensor()

    def set_src_dst_tensor(self):
        """
        set input and output tensor
        """
        self.coeff = self.tik_instance.Scalar("float32", name="coeff", init_value=-2)

        # creat input tensor: input_queries_gm, input_code_gm
        # and input_queries_norm_gm
        # and output tensor: output_distances_gm, output_flag_gm in global buffer
        self.input_queries_gm = self.tik_instance.Tensor(
            self.dtype_queries,
            self.shape_queries,
            name="input_queries_gm",
            scope=tik.scope_gm)
        self.input_code_gm = self.tik_instance.Tensor(
            self.dtype_code,
            self.shape_code,
            name="input_code_gm",
            scope=tik.scope_gm)
        self.input_queries_norm_gm = self.tik_instance.Tensor(
            self.dtype_queries_norm,
            self.shape_queries_norm,
            name="input_queries_norm_gm",
            scope=tik.scope_gm)
        self.input_code_norm_gm = self.tik_instance.Tensor(
            self.dtype_code_norm,
            self.shape_code_norm,
            name="input_code_norm_gm",
            scope=tik.scope_gm)
        self.output_distances_gm = self.tik_instance.Tensor(
            self.dtype_distances,
            self.shape_distances,
            name="output_distances_gm",
            scope=tik.scope_gm)
        self.output_mins_gm = self.tik_instance.Tensor(
            self.dtype_mins,
            self.shape_mins,
            name="output_mins_gm",
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
        self.aicore_use = min(self.shape_flag[0], self.code_num // self.code_each_loop)
        self.code_each_core = self.code_num // self.aicore_use // self.code_each_loop * self.code_each_loop
        self.code_page = min(256 * 1024 // self.dim, self.code_each_core)
        page_num = (self.code_each_core + self.code_page - 1) // self.code_page
        self.code_page = self.code_each_core // page_num

    def distance_compute_each_loop(self, aicore_offset, page_offset):
        query_norm_l1 = self.tik_instance.Tensor("float32",
                                                 (self.queries_num,),
                                                 name="query_norm_l1", scope=tik.scope_cbuf)
        self.tik_instance.data_move(query_norm_l1,
                                    self.input_queries_norm_gm,
                                    0, 1, self.queries_num // 8, 0, 0)
        code_norm_l1 = self.tik_instance.Tensor(
            "float32",
            (self.code_page // self.code_each_loop, self.query_sub_loop // 16, self.code_each_loop, 16),
            name="code_norm_l1", scope=tik.scope_cbuf)
        self.tik_instance.data_move(code_norm_l1,
                                    self.input_code_norm_gm[(aicore_offset + page_offset) // self.code_each_loop,
                                                            0, 0, 0],
                                    0, 1, self.query_sub_loop * self.code_page // 8, 0, 0)
        code_l1 = self.tik_instance.Tensor(
            "float16",
            (self.code_page // self.code_each_loop, self.dim // 16, self.code_each_loop, 16),
            name="code_l1", scope=tik.scope_cbuf)
        self.tik_instance.data_move(code_l1,
                                    self.input_code_gm[(aicore_offset + page_offset) // self.code_each_loop,
                                                       0, 0, 0],
                                    0, 1, self.code_page * self.dim // 16, 0, 0)

        code_loops = self.code_page // self.code_each_loop
        query_loops = self.queries_num // self.query_each_loop
        # compute xy using cube
        with self.tik_instance.for_range(0, query_loops * code_loops) as loop:
            self.cube_compute_each_loop(code_l1, query_norm_l1, code_norm_l1,
                                        aicore_offset, page_offset,
                                        loop % code_loops * self.code_each_loop,
                                        loop // code_loops * self.query_each_loop)

    def cube_compute_each_loop(self, code_l1, query_norm_l1, code_norm_l1,
                               aicore_offset, page_offset, code_offset, query_offset):
        dst_ub = self.tik_instance.Tensor("float16",
                                          (self.query_each_loop, self.code_each_loop),
                                          name="dst_ub", scope=tik.scope_ubuf)
        dst_min_ub = self.tik_instance.Tensor("float16",
                                              (self.query_each_loop, self.code_each_loop // self.min_batch * 2),
                                              name="dst_min_ub", scope=tik.scope_ubuf)

        with self.tik_instance.for_range(0, self.query_loops, thread_num=2) as loop_ub:
            queries_l1 = self.tik_instance.Tensor("float16",
                                                  (self.dim // 16, self.query_sub_loop, 16),
                                                  name="queries_l1", scope=tik.scope_cbuf)
            self.tik_instance.data_move(queries_l1,
                                        self.input_queries_gm[query_offset // self.query_segment, 0,
                                                              loop_ub * self.query_sub_loop, 0],
                                        0, self.dim // 16, self.query_sub_loop, self.query_sub_loop, 0)

            inner_product_l0c = self.tik_instance.Tensor("float32",
                                                         (self.query_sub_loop // 16, self.code_each_loop, 16),
                                                         name="inner_product_l0c", scope=tik.scope_cbuf_out)
            self.tik_instance.matmul(inner_product_l0c,
                                     code_l1[(code_offset * self.dim):], queries_l1,
                                     self.code_each_loop, self.dim, self.query_sub_loop)

            query_norm_ub = self.tik_instance.Tensor("float32",
                                                     (self.query_sub_loop // 16, 16),
                                                     name="query_norm_ub", scope=tik.scope_ubuf)
            # mov query norm to UB
            self.tik_instance.data_move(query_norm_ub,
                                        query_norm_l1[query_offset + loop_ub * self.query_sub_loop],
                                        0, 1, self.query_sub_loop // 8, 0, 0)

            add_ub = self.tik_instance.Tensor("float32",
                                              (self.query_sub_loop // 16, self.code_each_loop, 16),
                                              name="add_ub", scope=tik.scope_ubuf)
            self.tik_instance.data_move(add_ub,
                                        code_norm_l1[code_offset // self.code_each_loop, 0, 0, 0],
                                        0, 1, self.code_each_loop * self.query_sub_loop // 8, 0, 0)

            # mov xy from L0-C to UB
            inner_product_ub = self.tik_instance.Tensor("float32",
                                                        (self.query_sub_loop // 16, self.code_each_loop, 16),
                                                        name="inner_product_ub", scope=tik.scope_ubuf)
            self.tik_instance.data_move(inner_product_ub,
                                        inner_product_l0c,
                                        0, 1, self.query_sub_loop * self.code_each_loop // 256, 0, 0)

            for i in range(self.query_sub_loop // 16):
                for j in range(self.code_each_loop // self.max_repeats):
                    self.tik_instance.vadd(16, add_ub[i, j * self.max_repeats, 0],
                                           add_ub[i, j * self.max_repeats, 0], query_norm_ub[i, 0],
                                           self.max_repeats, 1, 1, 1, 2, 2, 0)

            self.tik_instance.data_move(inner_product_ub,
                                        inner_product_l0c,
                                        0, 1, self.query_sub_loop * self.code_each_loop // 256, 0, 0)

            vapxy_repeats = self.query_sub_loop * self.code_each_loop // self.fp32_mask
            self.tik_instance.vaxpy(self.fp32_mask,
                                    add_ub, inner_product_ub, self.coeff,
                                    vapxy_repeats, 1, 1, 8, 8)

            dst_ub_ = self.tik_instance.Tensor("float16",
                                               (self.query_sub_loop // 16, self.code_each_loop // 16, 16, 16),
                                               name="dst_ub", scope=tik.scope_ubuf)

            vconv_repeats = self.query_sub_loop * self.code_each_loop // self.fp32_mask
            self.tik_instance.vconv(self.fp32_mask, "none",
                                    dst_ub_, add_ub,
                                    vconv_repeats, 1, 1, 4, 8)

            trans_repeats = self.query_sub_loop * self.code_each_loop // 256
            self.tik_instance.vec_trans(dst_ub_, dst_ub_, trans_repeats, 1, 1)

            for i in range(self.query_sub_loop):
                self.tik_instance.data_move(dst_ub[loop_ub * self.query_sub_loop + i, 0],
                                            dst_ub_[i // 16, 0, i % 16, 0],
                                            0, self.code_each_loop // 16, 1, 15, 0)

            vcmin_loops = self.query_sub_loop * self.code_each_loop // self.min_batch // self.max_repeats
            for i in range(vcmin_loops):
                self.tik_instance.vcmin(
                    self.min_batch,
                    dst_min_ub[((loop_ub * vcmin_loops + i) * 2 * self.max_repeats):],
                    dst_ub[((loop_ub * vcmin_loops + i) * self.min_batch * self.max_repeats):],
                    self.max_repeats, 1, 1, self.min_batch // 16)

        self.tik_instance.data_move(
            self.output_distances_gm[query_offset,
                                     aicore_offset + page_offset + code_offset],
            dst_ub,
            0, self.query_each_loop, self.code_each_loop // 16,
            0, (self.code_num - self.code_each_loop) // 16)
        self.tik_instance.data_move(
            self.output_mins_gm[query_offset,
                                (aicore_offset + page_offset + code_offset) // self.min_batch * 2],
            dst_min_ub,
            0, self.query_each_loop, self.code_each_loop // self.min_batch * 2 // 16,
            0, (self.min_num - self.code_each_loop // self.min_batch * 2) // 16)

    def distance_compute_flat(self):
        """
        the compute process
        """
        self.cal_num_each_core()

        with self.tik_instance.for_range(0, self.aicore_use, block_num=self.aicore_use) as block_index:
            # compute coarse code num and move offset every core
            with self.tik_instance.for_range(0, self.code_each_core // self.code_page) as i:
                self.distance_compute_each_loop(block_index * self.code_each_core, i * self.code_page)

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
                                       self.input_code_gm,
                                       self.input_queries_norm_gm,
                                       self.input_code_norm_gm
                                   ],
                                   outputs=[self.output_distances_gm, self.output_mins_gm, self.output_flag_gm])

        return self.tik_instance


def distance_flat_l2_mins_at(input_queries,
                             input_code,
                             input_queries_norm,
                             input_code_norm,
                             output_distances,
                             output_mins,
                             output_flag,
                             kernel_name="distance_flat_l2_mins_at"):
    """
    calculating distance

    Parameters
    ----------
    input_queries : dict
        shape and dtype of query vector
    input_code : dict
        shape and dtype of coarse code
    input_queries_norm : dict
        shape and dtype of L2 distance of query vector
    input_code_norm : dict
        shape and dtype of precomputed L2 distance of coarse code
    output_distances : dict
        shape and dtype of distances, should be same dtype as input_queries
    output_mins : dict
        shape and dtype of min distances
    output_flag : dict
        shape and dtype of flag, only the 1th and 17th is valid
    kernel_name : str
        kernel name, default value is "distance_flat_l2_mins"

    Returns
    -------
    None
    """
    distance_flat = DistanceFlatL2MinsAt(input_queries, input_code, input_queries_norm, input_code_norm,
                                         output_distances, output_mins, output_flag, kernel_name)
    tik_instance = distance_flat.get_tik_instance()
    return tik_instance
