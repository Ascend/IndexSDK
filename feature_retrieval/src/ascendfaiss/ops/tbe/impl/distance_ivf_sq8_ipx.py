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


class DistanceIVFSQ8IPX:
    def __init__(self,
                 input_queries,
                 input_base_segment,
                 input_segment_offset,
                 input_dm,
                 output_distances,
                 output_maxs,
                 output_flag,
                 kernel_name="distance_ivf_sq8_ipx"):
        self.shape_queries = input_queries.get("shape")
        self.dtype_queries = input_queries.get("dtype")
        self.shape_base_segment = input_base_segment.get("shape")
        self.dtype_base_segment = input_base_segment.get("dtype")
        self.shape_segment_offset = input_segment_offset.get("shape")
        self.dtype_segment_offset = input_segment_offset.get("dtype")
        self.shape_dm = input_dm.get("shape")
        self.dtype_dm = input_dm.get("dtype")
        self.shape_distances = output_distances.get("shape")
        self.dtype_distances = output_distances.get("dtype")
        self.shape_maxs = output_maxs.get("shape")
        self.dtype_maxs = output_maxs.get("dtype")
        self.shape_flag = output_flag.get("shape")
        self.dtype_flag = output_flag.get("dtype")
        self.kernel_name = kernel_name

        # compute parameter
        self.queries_num, self.dim = self.shape_queries
        self.code_num = self.shape_base_segment[0] // self.dim

        # check parameter
        self.check_parameter()

        self.segment_size = 64
        self.segment_size_align32 = (self.segment_size + 31) // 32 * 32
        self.segment_num = self.shape_segment_offset[1]

        # set vector fp32 mask and fp16 mask
        self.page_size = 8192
        self.fp32_mask = 64
        self.fp16_mask = 128
        self.max_batch = 16
        self.queries_align = 16
        self.cube_align = 16

        # set tik instance
        self.set_tik_instance()

    def check_parameter(self):
        # check shape and dtype of input
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
        self.queries_num_each_core = self.queries_num // self.aicore_use
        self.queries_num_last_core = self.queries_num - (self.aicore_use - 1) * self.queries_num_each_core

        self.code_each_loops = 2
        self.code_each_segs = 12 // max(1, self.dim // 64)
        self.code_sub_segs = self.code_each_segs // self.code_each_loops
        self.each_query_loops = (self.segment_num + self.code_each_segs - 1) // self.code_each_segs

        self.maxs_each_loop = self.code_each_segs * self.segment_size // self.max_batch * 2
        self.maxs_sub_loop = self.code_sub_segs * self.segment_size // self.max_batch * 2

        self.repeat_times = self.code_sub_segs * self.segment_size // 4

        self.set_src_dst_tensor()

    def set_src_dst_tensor(self):
        """
        set input and output tensor
        """
        self.coeff = self.tik_instance.Scalar("float32", name="coeff", init_value=1 / 255)
        self.zero_scalar = self.tik_instance.Scalar("float16", name="zero_scalar", init_value=0)

        # creat input tensor: input_queries_gm, input_base_segment_gm, input_segment_offset_gm
        # input_diff_gm, input_min_gm, input_actual_num_gm
        # creat output tensor: output_distances_gm
        # and output_flag_gm in global buffer
        self.input_queries_gm = self.tik_instance.Tensor(
            self.dtype_queries,
            self.shape_queries,
            name="input_queries_gm",
            scope=tik.scope_gm)
        self.input_base_segment_gm = self.tik_instance.Tensor(
            self.dtype_base_segment,
            self.shape_base_segment,
            name="input_base_segment_gm",
            scope=tik.scope_gm)
        self.input_segment_offset_gm = self.tik_instance.Tensor(
            self.dtype_segment_offset,
            self.shape_segment_offset,
            name="input_segment_offset_gm",
            scope=tik.scope_gm)
        self.input_dm_gm = self.tik_instance.Tensor(
            self.dtype_dm,
            self.shape_dm,
            name="input_dm_gm",
            scope=tik.scope_gm)
        self.output_distances_gm = self.tik_instance.Tensor(
            self.dtype_distances,
            self.shape_distances,
            name="output_distances_gm",
            scope=tik.scope_gm)
        self.output_maxs_gm = self.tik_instance.Tensor(
            self.dtype_maxs,
            self.shape_maxs,
            name="output_maxs_gm",
            scope=tik.scope_gm)
        self.output_flag_gm = self.tik_instance.Tensor(
            self.dtype_flag,
            self.shape_flag,
            name="output_flag_gm",
            scope=tik.scope_gm)

    def distance_compute_(self, query_num, query_offset):
        data_xd_l1 = self.tik_instance.Tensor("float16",
                                              (self.dim // 16, self.queries_align, 16),
                                              name="data_xd_l1", scope=tik.scope_cbuf)
        xdm_l1 = self.tik_instance.Tensor("float32",
                                          (query_num, self.fp16_mask),
                                          name="xdm_l1", scope=tik.scope_cbuf)

        with self.tik_instance.new_stmt_scope():
            queries_ub = self.tik_instance.Tensor("float16",
                                                  (query_num, self.dim),
                                                  name="queries_ub", scope=tik.scope_ubuf)
            diff_ub = self.tik_instance.Tensor("float16",
                                               (self.dim,),
                                               name="diff_ub", scope=tik.scope_ubuf)
            min_ub = self.tik_instance.Tensor("float16",
                                              (self.dim,),
                                              name="min_ub", scope=tik.scope_ubuf)

            # move diff adn min from out to UB
            self.tik_instance.data_move(diff_ub, self.input_dm_gm,
                                        0, 1, self.dim // 16, 0, 0)
            self.tik_instance.data_move(min_ub, self.input_dm_gm[self.dim],
                                        0, 1, self.dim // 16, 0, 0)
            diff2_ub = self.tik_instance.Tensor("float16",
                                                (self.dim,),
                                                name="diff2_ub", scope=tik.scope_ubuf)
            self._muls(diff2_ub, diff_ub, 0.5, self.dim, self.fp16_mask)
            self._muls(min_ub, min_ub, 255.0, self.dim, self.fp16_mask)
            self._add(diff2_ub, diff2_ub, min_ub, self.dim, self.fp16_mask)

            diff2_l0c = self.tik_instance.Tensor("float16",
                                                 (self.dim // 16, 16, 16),
                                                 name="diff_ub", scope=tik.scope_cbuf_out)
            self.tik_instance.broadcast_ub_to_l0c(diff2_l0c, diff2_ub, 1, self.dim // 16, 0, 0)

            dm_transfer_ub = self.tik_instance.Tensor("float16",
                                                      (self.dim // 16, 16, 16),
                                                      name="dm_transfer_ub", scope=tik.scope_ubuf)
            self.tik_instance.data_move(dm_transfer_ub, diff2_l0c,
                                        0, 1, self.dim * 16 * 2 // 512, 0, 0)

            # move d+m from UB to L1
            data_dm_l1 = self.tik_instance.Tensor("float16",
                                                  (self.dim // 16, 16, 16),
                                                  name="data_dm_l1", scope=tik.scope_cbuf)
            self.tik_instance.data_move(data_dm_l1, dm_transfer_ub,
                                        0, 1, self.dim, 0, 0)

            # move x from out to UB
            self.tik_instance.data_move(queries_ub,
                                        self.input_queries_gm[query_offset, 0],
                                        0, 1, query_num * self.dim // 16, 0, 0)
            queries_transfer_ub = self.tik_instance.Tensor("float16",
                                                           (self.dim // 16, self.queries_align, 16),
                                                           name="queries_transfer_ub", scope=tik.scope_ubuf)
            for i in range(query_num):
                self.tik_instance.data_move(queries_transfer_ub[0, i, 0],
                                            queries_ub[i, 0],
                                            0, self.dim // 16, 1, 0, self.queries_align - 1)

            # cal x*d
            self.tik_instance.vmul(self.dim, queries_ub,
                                   queries_ub, diff_ub,
                                   query_num, 1, 1, 1, self.dim // 16, self.dim // 16, 0)

            xd_transfer_ub = self.tik_instance.Tensor("float16",
                                                      (self.dim // 16, self.queries_align, 16),
                                                      name="xd_transfer_ub", scope=tik.scope_ubuf)

            for i in range(query_num):
                self.tik_instance.data_move(xd_transfer_ub[0, i, 0],
                                            queries_ub[i, 0],
                                            0, self.dim // 16, 1, 0, self.queries_align - 1)

            # move x from ub to L1
            queries_l1 = self.tik_instance.Tensor("float16",
                                                  (self.dim // 16, self.queries_align, 16),
                                                  name="queries_l1", scope=tik.scope_cbuf)
            self.tik_instance.data_move(queries_l1, queries_transfer_ub,
                                        0, 1, self.queries_align * self.dim // 16, 0, 0)
            # move x*d from UB to L1
            self.tik_instance.data_move(data_xd_l1, xd_transfer_ub,
                                        0, 1, self.queries_align * self.dim // 16, 0, 0)

            data_xdm_l0c = self.tik_instance.Tensor("float32",
                                                    (1, self.queries_align, 16),
                                                    name="data_xdm_l0c", scope=tik.scope_cbuf_out)
            self.tik_instance.matmul(data_xdm_l0c,
                                     queries_l1, data_dm_l1,
                                     16, self.dim, self.queries_align)

            data_xdm_ub = self.tik_instance.Tensor("float32",
                                                   (1, self.queries_align, 16),
                                                   name="data_xdm_ub", scope=tik.scope_ubuf)
            self.tik_instance.data_move(data_xdm_ub, data_xdm_l0c,
                                        0, 1, self.queries_align // 16, 0, 0)
            data_xdm_align_ub = self.tik_instance.Tensor("float32",
                                                         (self.queries_align, 16),
                                                         name="data_xdm_align_ub", scope=tik.scope_ubuf)
            self.tik_instance.vmuls(16, data_xdm_align_ub,
                                    data_xdm_ub, self.coeff,
                                    self.queries_align, 1, 1, 2, 2)

            for i in range(self.fp16_mask // 16):
                self.tik_instance.data_move(xdm_l1[0, 16 * i], data_xdm_align_ub,
                                            0, query_num, 2, 0, self.fp16_mask // 8 - 2)

        # compute xy using cube
        inner_product_l0c = self.tik_instance.Tensor(
            "float32",
            (self.code_each_segs * self.segment_size // 16, self.queries_align, 16),
            name="inner_product_l0c", scope=tik.scope_cbuf_out)

        seg_loops = query_num * self.each_query_loops

        with self.tik_instance.for_range(0, seg_loops) as loop_seg:
            self.cube_compute_each_loop(inner_product_l0c, data_xd_l1, xdm_l1, query_offset, loop_seg, 1)
        self.cube_compute_each_loop(inner_product_l0c, data_xd_l1, xdm_l1, query_offset, seg_loops, 0)

    def cube_compute_each_loop(self, inner_product_l0c, data_xd_l1, xdm_l1, query_offset, code_loop, flag):
        dst_ub = self.tik_instance.Tensor(
            "float16",
            (self.code_each_segs * self.segment_size,),
            name="dst_ub", scope=tik.scope_ubuf)
        dst_max_ub = self.tik_instance.Tensor(
            "float16",
            (self.maxs_each_loop,),
            name="dst_max_ub", scope=tik.scope_ubuf)

        with self.tik_instance.for_range(0, self.code_each_loops, thread_num=2) as loop_ub:
            if flag:
                segment_offset_ub = self.tik_instance.Tensor("uint64",
                                                             ((self.code_sub_segs + 3) // 4 * 4,),
                                                             name="segment_offset_ub", scope=tik.scope_ubuf)
                self.tik_instance.data_move(
                    segment_offset_ub,
                    self.input_segment_offset_gm[query_offset + code_loop // self.each_query_loops,
                                                 code_loop % self.each_query_loops * self.code_each_segs
                                                 + loop_ub * self.code_sub_segs],
                    0, 1, (self.code_sub_segs + 3) // 4, 0, 0)

            xdm_ub = self.tik_instance.Tensor("float32", (self.fp16_mask,), name="xdm_ub", scope=tik.scope_ubuf)
            with self.tik_instance.if_scope(code_loop != 0):
                self.tik_instance.data_move(xdm_ub,
                                            xdm_l1[(code_loop - 1) // self.each_query_loops, 0],
                                            0, 1, self.fp16_mask // 8, 0, 0)

            # MTE2: move y from out to UB
            code_ub = self.tik_instance.Tensor(
                "uint8",
                (self.dim // 16, self.code_sub_segs * self.segment_size, 16),
                name="code_ub", scope=tik.scope_ubuf)

            if flag:
                segment_offset = [self.tik_instance.Scalar(dtype="uint64") for x in range(self.code_sub_segs)]
                for i in range(self.code_sub_segs):
                    segment_offset[i].set_as(segment_offset_ub[i])

                for li in range(self.code_sub_segs):
                    self.tik_instance.data_move(
                        code_ub[0, li * self.segment_size, 0],
                        self.input_base_segment_gm[segment_offset[li]],
                        0, self.dim // 16, self.segment_size // 2,
                        (self.page_size - self.segment_size) // 2,
                        (self.code_sub_segs - 1) * self.segment_size // 2)

            # VECTOR: y do conv from uint8 to fp16
            code_ub_fp16 = self.tik_instance.Tensor(
                "float16",
                (self.dim // 16, self.code_sub_segs * self.segment_size, 16),
                name="code_ub_fp16", scope=tik.scope_ubuf)

            vconv_repeat_times_ = self.code_sub_segs * self.segment_size * self.dim // self.fp16_mask
            vconv_repeat_times = min(vconv_repeat_times_, self.repeat_times)
            vconv_repeat_loops = vconv_repeat_times_ // vconv_repeat_times
            for i in range(0, vconv_repeat_loops):
                self.tik_instance.vconv(self.fp16_mask, "none",
                                        code_ub_fp16[(i * self.fp16_mask * self.repeat_times):],
                                        code_ub[(i * self.fp16_mask * self.repeat_times):],
                                        vconv_repeat_times, 1, 1, 8, 4)

            # VECTOR: mov xy from L0-C to UB
            inner_product_ub = self.tik_instance.Tensor(
                "float32",
                (self.code_sub_segs * self.segment_size // 16, self.queries_align, 16),
                name="inner_product_ub", scope=tik.scope_ubuf)

            self.tik_instance.data_move(
                inner_product_ub,
                inner_product_l0c[loop_ub * self.segment_size * self.code_sub_segs // 16, 0, 0],
                0, 1, self.code_sub_segs * self.segment_size * self.queries_align // 256, 0, 0)

            # VECTOR: compute distance each query
            dst_ub_fp32 = self.tik_instance.Tensor("float32",
                                                   (self.code_sub_segs * self.segment_size,),
                                                   name="dst_ub_fp32", scope=tik.scope_ubuf)
            with self.tik_instance.if_scope(code_loop != 0):
                self.tik_instance.vmuls(16,
                                        dst_ub_fp32,
                                        inner_product_ub[0, (code_loop - 1) // self.each_query_loops, 0], self.coeff,
                                        self.code_sub_segs * self.segment_size // 16, 1, 1, 2, self.queries_align * 2)

                repeat_times = self.code_sub_segs * self.segment_size // self.fp32_mask
                self.tik_instance.vadd(self.fp32_mask, dst_ub_fp32,
                                       dst_ub_fp32, xdm_ub,
                                       repeat_times, 1, 1, 1, 8, 8, 0)

                self.tik_instance.vconv(self.fp32_mask, "none",
                                        dst_ub[(loop_ub * self.code_sub_segs * self.segment_size):], dst_ub_fp32,
                                        repeat_times, 1, 1, 4, 8)

                zero_ub = self.tik_instance.Tensor("float16",
                                                   (self.fp16_mask,),
                                                   name="min_val_ub", scope=tik.scope_ubuf)
                self.tik_instance.vec_dup(self.fp16_mask, zero_ub, self.zero_scalar, 1, 8)

                # the loop not enough self.code_each_loop, wo not use
                vcmax_repeat_times = self.code_sub_segs * self.segment_size // self.max_batch
                self.tik_instance.vcmax(self.max_batch,
                                        dst_max_ub[(loop_ub * self.maxs_sub_loop):],
                                        dst_ub[(loop_ub * self.code_sub_segs * self.segment_size):],
                                        vcmax_repeat_times, 1, 1, self.max_batch // 16)

            # MTE3: move y from ub to L1
            code_l1 = self.tik_instance.Tensor("float16",
                                               (self.dim // 16, self.code_sub_segs * self.segment_size, 16),
                                               name="code_l1", scope=tik.scope_cbuf)
            self.tik_instance.data_move(code_l1, code_ub_fp16,
                                        0, self.dim // 16, self.code_sub_segs * self.segment_size, 0, 0)

            self.tik_instance.matmul(
                inner_product_l0c[(loop_ub * self.code_sub_segs * self.segment_size * self.queries_align):],
                data_xd_l1, code_l1,
                self.queries_align, self.dim, self.code_sub_segs * self.segment_size)

        with self.tik_instance.if_scope(code_loop != 0):
            self.tik_instance.data_move(
                self.output_distances_gm[
                    query_offset + (code_loop - 1) // self.each_query_loops,
                    (code_loop - 1) % self.each_query_loops * self.code_each_segs * self.segment_size],
                dst_ub,
                0, 1, self.code_each_segs * self.segment_size // 16, 0, 0)
            self.tik_instance.data_move(
                self.output_maxs_gm[query_offset + (code_loop - 1) // self.each_query_loops,
                                    (code_loop - 1) % self.each_query_loops * self.maxs_each_loop],
                dst_max_ub,
                0, 1, self.maxs_each_loop // 16, 0, 0)

    def distance_compute(self):
        """
        the compute process
        """
        with self.tik_instance.for_range(0, self.aicore_use, block_num=self.aicore_use) as block_idx:
            with self.tik_instance.if_scope(block_idx != self.aicore_use - 1):
                aicore_query_num = self.queries_num_each_core
            with self.tik_instance.else_scope():
                aicore_query_num = self.queries_num_last_core

            with self.tik_instance.if_scope(aicore_query_num != 0):
                self.distance_compute_(aicore_query_num, block_idx * self.queries_num_each_core)

            one = self.tik_instance.Scalar(dtype="uint16", name="one", init_value=1)
            flag_ub = self.tik_instance.Tensor("uint16",
                                               (16,),
                                               name="flag_ub", scope=tik.scope_ubuf)

            flag_ub[0].set_as(one)
            self.tik_instance.pipe_barrier("PIPE_MTE3")
            self.tik_instance.data_move(self.output_flag_gm[block_idx, 0],
                                        flag_ub,
                                        0, 1, 1, 0, 0)

    def get_tik_instance(self):
        """
        obtain tik instance
        """
        self.distance_compute()

        self.tik_instance.BuildCCE(
            kernel_name=self.kernel_name,
            inputs=[
                self.input_queries_gm,
                self.input_base_segment_gm,
                self.input_segment_offset_gm,
                self.input_dm_gm,
            ],
            outputs=[self.output_distances_gm, self.output_maxs_gm, self.output_flag_gm])

        return self.tik_instance

    def _muls(self, dst, src, scalar, compute_num, max_mask):
        # process 256B data per repeat for vmuls
        repeat = compute_num // max_mask
        offset = 0
        if repeat > 0:
            self.tik_instance.vmuls(max_mask, dst[offset], src[offset], scalar,
                                    repeat, 1, 1, 8, 8)
            offset += repeat * max_mask

        remain = compute_num % max_mask
        if remain > 0:
            self.tik_instance.vmuls(remain, dst[offset], src[offset], scalar,
                                    1, 1, 1, 8, 8)

    def _add(self, dst, src0, src1, compute_num, max_mask):
        # process 256B data per repeat for vadd
        repeat = compute_num // max_mask
        offset = 0
        if repeat > 0:
            self.tik_instance.vadd(max_mask, dst[offset],
                                   src0[offset], src1[offset],
                                   repeat, 1, 1, 1, 8, 8, 8)
            offset += repeat * max_mask

        remain = compute_num % max_mask
        if remain > 0:
            self.tik_instance.vadd(remain, dst[offset],
                                   src0[offset], src1[offset],
                                   1, 1, 1, 1, 8, 8, 8)


def distance_ivf_sq8_ipx(input_queries,
                         input_base_segment,
                         input_segment_offset,
                         input_dm,
                         output_distances,
                         output_maxs,
                         output_flag,
                         kernel_name="distance_ivf_sq8_ipx"):
    """
    calculating distance

    Parameters
    ----------
    input_queries : dict
        shape and dtype of query vector
    input_base_segment : dict
        shape and dtype of coding vector
    input_segment_offset : dict
        shape and dtype of offset vector
    input_dm: dict
        shape and dtype of coding coefficient and offset
    output_distances : dict
        shape and dtype of distances, should be same dtype as input_queries
    output_maxs : dict
        shape and dtype of maxs
    output_flag: dict
        shape and dtype of output flag,
        shape must be (32,) and dtype must be uint16
    kernel_name : str
        kernel name, default value is "distance_compute"

    Returns
    -------
    None
    """
    distance_ivf_sq = DistanceIVFSQ8IPX(input_queries,
                                        input_base_segment,
                                        input_segment_offset,
                                        input_dm,
                                        output_distances,
                                        output_maxs,
                                        output_flag,
                                        kernel_name)
    tik_instance = distance_ivf_sq.get_tik_instance()
    return tik_instance
