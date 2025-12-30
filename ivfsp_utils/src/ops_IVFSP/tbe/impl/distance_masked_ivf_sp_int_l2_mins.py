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

class DistanceIVFSpIntL2Mins:
    def __init__(self,
                 input_queries,
                 input_codebook,
                 input_code,
                 input_mask,
                 input_precomputed,
                 input_dm,
                 input_actual_code_num,
                 input_codebook_offset,
                 input_base_offset,
                 input_precomputed_offset,
                 output_distances,
                 output_mins,
                 output_flag,
                 kernel_name="distance_ivf_sp_int_l2_mins"):
        self.shape_queries = input_queries.get("shape")
        self.dtype_queries = input_queries.get("dtype")
        self.shape_codebook = input_codebook.get("shape")
        self.dtype_codebook = input_codebook.get("dtype")
        self.shape_code = input_code.get("shape")
        self.dtype_code = input_code.get("dtype")
        self.shape_precomputed = input_precomputed.get("shape")
        self.dtype_precomputed = input_precomputed.get("dtype")
        self.shape_dm = input_dm.get("shape")
        self.dtype_dm = input_dm.get("dtype")
        
        self.shape_actual_code_num = input_actual_code_num.get("shape")
        self.dtype_actual_code_num = input_actual_code_num.get("dtype")
        self.shape_codebook_offset = input_codebook_offset.get("shape")
        self.dtype_codebook_offset = input_codebook_offset.get("dtype")
        self.shape_base_offset = input_base_offset.get("shape")
        self.dtype_base_offset = input_base_offset.get("dtype")
        self.shape_precomputed_offset = input_precomputed_offset.get("shape")
        self.dtype_precomputed_offset = input_precomputed_offset.get("dtype")
        self.shape_mask = input_mask.get("shape")
        self.dtype_mask = input_mask.get("dtype")
        self.shape_distances = output_distances.get("shape")
        self.dtype_distances = output_distances.get("dtype")
        self.shape_output_mins = output_mins.get("shape")
        self.dtype_output_mins = output_mins.get("dtype")
        self.shape_flag = output_flag.get("shape")
        self.dtype_flag = output_flag.get("dtype")
        self.kernel_name = kernel_name

        # compute parameter
        self.queries_num, self.dim = self.shape_queries
        self.code_num, = self.shape_precomputed
        self.nprobe, = self.shape_actual_code_num
        self.dim2 = self.shape_code[0] // self.code_num
        self.dim2_repeat = self.dim2 // 16

        self.base_offset = []
        self.precomputed_offset = []
        self.actual_num = []
        self.codebook_offset = []

        # check parameter
        self.check_parameter()

        # set vector fp32 mask and fp16 mask
        self.fp32_mask = 64
        self.fp16_mask = 128
        self.min_batch = 64

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
        tik_dprofile = tik.Dprofile("v200", "aic")
        self.tik_instance = tik.Tik(tik_dprofile)

        self.aicore_use = 8
        self.queries_num_each_loop = min(self.queries_num, 48)
        self.code_num_each_loop = min(48 // 16 // ((self.queries_num_each_loop + 15) // 16) * 1024,
                                      (1 + 256 // (self.dim2 + 128)) * 1024)
        self.nprobe_each_core = self.nprobe // self.aicore_use
        self.set_src_dst_tensor()

    def set_src_dst_tensor(self):
        """
        set input and output tensor
        """
        self.coeff = self.tik_instance.Scalar("float32",
                                              name="coeff",
                                              init_value=-2 / 255)
        self.default_scalar = self.tik_instance.Scalar("float16",
                                                       name="default_scalar",
                                                       init_value=65504)
        self.input_queries_gm = self.tik_instance.Tensor(
            self.dtype_queries,
            self.shape_queries,
            name="input_queries_gm",
            scope=tik.scope_gm)
        self.input_codebook_gm = self.tik_instance.Tensor(
            self.dtype_codebook,
            self.shape_codebook,
            name="input_codebook_gm",
            scope=tik.scope_gm)
        self.input_code_gm = self.tik_instance.Tensor(
            self.dtype_code,
            self.shape_code,
            name="input_code_gm",
            scope=tik.scope_gm)
        self.input_precomputed_gm = self.tik_instance.Tensor(
            self.dtype_precomputed,
            self.shape_precomputed,
            name="input_precomputed_gm",
            scope=tik.scope_gm)
        self.input_dm_gm = self.tik_instance.Tensor(
            self.dtype_dm,
            self.shape_dm,
            name="input_dm_gm",
            scope=tik.scope_gm)
       
        self.input_actual_code_num_gm = self.tik_instance.Tensor(
            self.dtype_actual_code_num,
            self.shape_actual_code_num,
            name="input_actual_code_num_gm",
            scope=tik.scope_gm)
        self.input_codebook_offset_gm = self.tik_instance.Tensor(
            self.dtype_codebook_offset,
            self.shape_codebook_offset,
            name="input_codebook_offset_gm",
            scope=tik.scope_gm)
        self.input_base_offset_gm = self.tik_instance.Tensor(
            self.dtype_base_offset,
            self.shape_base_offset,
            name="input_base_offset_gm",
            scope=tik.scope_gm)
        self.input_precomputed_offset_gm = self.tik_instance.Tensor(
            self.dtype_precomputed_offset,
            self.shape_precomputed_offset,
            name="input_precomputed_offset_gm",
            scope=tik.scope_gm)
        self.input_mask_gm = self.tik_instance.Tensor(
            self.dtype_mask,
            self.shape_mask,
            name="input_mask_gm",
            scope=tik.scope_gm)
        self.output_distances_gm = self.tik_instance.Tensor(
            self.dtype_distances,
            self.shape_distances,
            name="output_distances_gm",
            scope=tik.scope_gm)
        self.output_mins_gm = self.tik_instance.Tensor(
            self.dtype_output_mins,
            self.shape_output_mins,
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
        self.base_offset = [self.tik_instance.Scalar(dtype="uint64") for x in range(self.nprobe)]
        self.precomputed_offset = [self.tik_instance.Scalar(dtype="uint64") for x in range(self.nprobe)]
        self.actual_num = [self.tik_instance.Scalar(dtype="uint32") for x in range(self.nprobe)]
        self.codebook_offset = [self.tik_instance.Scalar(dtype="uint64") for x in range(self.nprobe)]
        with self.tik_instance.new_stmt_scope():
            base_offset_ub = self.tik_instance.Tensor("uint64",
                                                      (self.nprobe,),
                                                      name="base_offset_ub", scope=tik.scope_ubuf)
            self.tik_instance.data_move(base_offset_ub, self.input_base_offset_gm, 0, 1, self.nprobe//4, 0, 0)

            for i in range(self.nprobe):
                self.base_offset[i].set_as(base_offset_ub[i])

            precomputed_offset_ub = self.tik_instance.Tensor("uint64",
                                                      (self.nprobe,),
                                                      name="precomputed_offset_ub", scope=tik.scope_ubuf)
            self.tik_instance.data_move(precomputed_offset_ub, self.input_precomputed_offset_gm, 0, 1, 
                                            self.nprobe//4, 0, 0)

            for i in range(self.nprobe):
                self.precomputed_offset[i].set_as(precomputed_offset_ub[i])

            # read actual code num from gm
            actual_num_ub = self.tik_instance.Tensor("uint32",
                                                     (self.nprobe,),
                                                     name="actual_num_ub", scope=tik.scope_ubuf)
            self.tik_instance.data_move(actual_num_ub, self.input_actual_code_num_gm, 0, 1, self.nprobe//8, 0, 0)


            for i in range(self.nprobe):
                self.actual_num[i].set_as(actual_num_ub[i])

            codebook_offset = self.tik_instance.Tensor("uint64", (self.nprobe,),
                                                            name="codebook_offset_ub",
                                                            scope=tik.scope_ubuf)

            self.tik_instance.data_move(codebook_offset, self.input_codebook_offset_gm, 0, 1, self.nprobe//4, 0, 0)
            for i in range(self.nprobe):
                self.codebook_offset[i].set_as(codebook_offset[i])

    def cal_xd_lowdim_v2(self, queries_ub, queries_l1, queries_l2_ub, add_scalar, data_xd_l1, move_num, queries_align):

        with self.tik_instance.new_stmt_scope():
            diff_ub = self.tik_instance.Tensor("float16",
                                               (self.dim2,),
                                               name="diff_ub", scope=tik.scope_ubuf)
            min_ub = self.tik_instance.Tensor("float16",
                                              (self.dim2,),
                                              name="min_ub", scope=tik.scope_ubuf)

            # move diff adn min from out to UB
            self.tik_instance.data_move(diff_ub, self.input_dm_gm, 0, 1, self.dim2 // 16, 0, 0)
            self.tik_instance.data_move(min_ub, self.input_dm_gm[self.dim2], 0, 1, self.dim2 // 16, 0, 0)
            diff2_ub = self.tik_instance.Tensor("float16",
                                                (self.dim2,),
                                                name="diff2_ub", scope=tik.scope_ubuf)
            # calculate 0.5 * d
            self._muls(diff2_ub, diff_ub, 0.5, self.dim2, self.fp16_mask)
            # calculate 255 * m
            self._muls(min_ub, min_ub, 255.0, self.dim2, self.fp16_mask)
            # calculate 0.5 * d + 255 * m
            self._add(diff2_ub, diff2_ub, min_ub, self.dim2, self.fp16_mask)
            dm_transfer_ub = self.tik_instance.Tensor("float16",
                                                      (self.dim2 // 16, 16, 16),
                                                      name="dm_transfer_ub", scope=tik.scope_ubuf)
            # use data_move function to make 0.5 * d + 255 * m align 16
            self.tik_instance.data_move(dm_transfer_ub, diff2_ub, 0, self.dim2 // 16, 1, 0, 15)
            # move d+m from UB to L1
            data_dm_l1 = self.tik_instance.Tensor("float16",
                                                  (self.dim2 // 16, 16, 16),
                                                  name="data_dm_l1", scope=tik.scope_cbuf)
            self.tik_instance.data_move(data_dm_l1, dm_transfer_ub, 0, 1, self.dim2, 0, 0)

            if self.dim2 <= 128:
                self.tik_instance.vmul(self.dim2, queries_ub, queries_ub,
                                       diff_ub, move_num, 1, 1, 1,
                                       self.dim2 // 16, self.dim2 // 16, 0)
            else:
                with self.tik_instance.for_range(0, move_num) as i:
                    repeat = self.dim2 // self.fp16_mask
                    offset = 0
                    if repeat > 0:
                        self.tik_instance.vmul(self.fp16_mask,
                                               queries_ub[i, offset],
                                               queries_ub[i, offset],
                                               diff_ub[offset], repeat, 1, 1,
                                               1, 8, 8, 8)
                        offset += repeat * self.fp16_mask

                    remain = self.dim2 % self.fp16_mask
                    if remain > 0:
                        self.tik_instance.vmul(remain, queries_ub[i, offset],
                                               queries_ub[i, offset],
                                               diff_ub[offset], 1, 1, 1, 1, 8,
                                               8, 8)

            # use data_move function to make x*d align 16
            xd_transfer_ub = self.tik_instance.Tensor("float16",
                                                      (self.dim2 // 16, queries_align, 16),
                                                      name="xd_transfer_ub", scope=tik.scope_ubuf)
            with self.tik_instance.for_range(0, move_num) as i:
                self.tik_instance.data_move(xd_transfer_ub[0, i, 0],
                                            queries_ub[i, 0], 0, self.dim2 // 16,
                                            1, 0, queries_align - 1)

            # move x*d from UB to L1
            self.tik_instance.data_move(data_xd_l1, xd_transfer_ub, 0, 1,
                                        queries_align * self.dim2 // 16, 0, 0)

            data_xdm_l0c = self.tik_instance.Tensor("float32",
                                                    (queries_align // 16, 16, 16),
                                                    name="data_xdm_l0c", scope=tik.scope_cbuf_out)
            # calculate x * (0.5 * d + 255 * m)
            self.tik_instance.matmul(data_xdm_l0c, data_dm_l1, queries_l1, 16,
                                     self.dim2, queries_align)

            data_xdm_ub = self.tik_instance.Tensor("float32",
                                                   (queries_align // 16, 16, 16),
                                                   name="data_xdm_ub", scope=tik.scope_ubuf)
            # move data from l0 to ub
            self.tik_instance.data_move(data_xdm_ub, data_xdm_l0c, 0, 1, queries_align // 16, 0, 0)
            data_xdm_align_ub = self.tik_instance.Tensor("float32",
                                                         (queries_align,),
                                                         name="data_xdm_align_ub", scope=tik.scope_ubuf)
            # calculate 2 / 255 * x * (0.5 * d + 255 * m)
            self.tik_instance.vmuls(16, data_xdm_align_ub, data_xdm_ub,
                                    self.coeff, queries_align // 16, 1, 1, 2, 32)
            # get sum for x*x - 2/255*(0.5*d + 255_*m)
            self._add(data_xdm_align_ub, queries_l2_ub, data_xdm_align_ub, queries_align, self.fp16_mask)

            # set as scalar for next cal
            add_scalar.set_as(data_xdm_align_ub[0])

    def cal_q_l2(self, queries_ub, queries_l1, queries_l2_ub, move_num, queries_align):
        # use data_move function to make x align 16
        with self.tik_instance.new_stmt_scope():
            with self.tik_instance.for_range(0, move_num) as i:
                self.tik_instance.data_move(queries_l1[0, i, 0],
                                            queries_ub[i, 0],
                                            0, self.dim // 16, 1, 0, queries_align - 1)

            queries_square_l0c = self.tik_instance.Tensor("float32",
                                                          (queries_align // 16, queries_align, 16),
                                                          name="queries_square_l0c", scope=tik.scope_cbuf_out)
            # calculate x*x
            self.tik_instance.matmul(queries_square_l0c, queries_l1,
                                     queries_l1, queries_align, self.dim,
                                     queries_align)
            queries_square_ub = self.tik_instance.Tensor("float32",
                                                         (queries_align // 16, queries_align, 16),
                                                         name="queries_square_ub", scope=tik.scope_ubuf)
            self.tik_instance.data_move(queries_square_ub, queries_square_l0c,
                                        0, 1,
                                        queries_align * queries_align // 256,
                                        0, 0)
            for i in range(move_num):
                mask = 2 ** (i % 16)
                # get sum
                self.tik_instance.vcadd([0, mask],
                                        queries_l2_ub[i],
                                        queries_square_ub[i // 16, i, 0],
                                        1, 1, 1, 8)

    def distance_compute_each_loop_v3(self, core_idx):
        codebook_k = self.tik_instance.Scalar(dtype="int32", name="codebook_k", init_value=0)
        queries_align = 16
        add_scalar = self.tik_instance.Scalar(dtype="float32")
        data_xd_l1 = self.tik_instance.Tensor("float16",
                                              (self.dim2 // 16, queries_align, 16),
                                              name="data_xd_l1", scope=tik.scope_cbuf)

        queries_ub = self.tik_instance.Tensor("float16",
                                              (1, self.dim),
                                              name="queries_ub", scope=tik.scope_ubuf)
        queries_l1 = self.tik_instance.Tensor("float16",
                                              (self.dim // 16, queries_align, 16),
                                              name="queries_l1", scope=tik.scope_cbuf)
        queries_l2_ub = self.tik_instance.Tensor("float32",
                                                 (queries_align,),
                                                 name="queries_l2_ub", scope=tik.scope_ubuf)

        queries_l1_lowdim = self.tik_instance.Tensor("float16",
                                                     (self.dim2 // 16, queries_align, 16),
                                                     name="queries_l1_lowdim", scope=tik.scope_cbuf)
        queries_ub_lowdim = self.tik_instance.Tensor("float16",
                                                     (1, self.dim2),
                                                     name="queries_ub_lowdim", scope=tik.scope_ubuf)
        self.tik_instance.data_move(queries_ub,
                                    self.input_queries_gm,
                                    0, 1, 1 * self.dim // 16, 0, 0)
        self.cal_q_l2(queries_ub, queries_l1, queries_l2_ub, 1, queries_align)
        codebook = self.tik_instance.Tensor("float16",
                                            (self.shape_codebook[1], self.dim2, 16),
                                            name='codebook_cbuf', scope=tik.scope_cbuf)

        for list_idx in range(core_idx*self.nprobe_each_core, (core_idx+1)*self.nprobe_each_core):
            centroids_loop_time = self.actual_num[list_idx] // self.code_num_each_loop
            centroids_last_num = self.actual_num[list_idx] % self.code_num_each_loop

            codebook_k.set_as(self.codebook_offset[list_idx])
            for i in range(self.shape_codebook[1]):
                self.tik_instance.data_move(codebook[i, 0, 0],
                                            self.input_codebook_gm[codebook_k * self.dim2_repeat, i, 0, 0],
                                            0, self.dim2 // 16, 16, self.shape_codebook[1] * 16 - 16, 0)

            self.cal_qc(queries_l1_lowdim, queries_l1, codebook, queries_align)

            self.tik_instance.data_move(queries_ub_lowdim[0, 0], queries_l1_lowdim[0, 0, 0], 0, self.dim2 // 16,
                                        1, queries_align - 1, 0)

            self.cal_xd_lowdim_v2(queries_ub_lowdim, queries_l1_lowdim, queries_l2_ub, add_scalar, data_xd_l1, 1,
                                  queries_align)

            with self.tik_instance.if_scope(centroids_loop_time > 0):
                with self.tik_instance.for_range(0, centroids_loop_time) as loop_centroids:
                    self.cube_compute_each_loop_v2(list_idx, data_xd_l1, add_scalar, self.base_offset[list_idx], 
                                                self.precomputed_offset[list_idx],
                                                self.code_num_each_loop * loop_centroids,
                                                self.code_num_each_loop, 0)

            with self.tik_instance.if_scope(centroids_last_num > 0):
                self.cube_compute_each_loop_v2(list_idx, data_xd_l1, add_scalar, self.base_offset[list_idx], 
                                            self.precomputed_offset[list_idx],
                                            centroids_loop_time * self.code_num_each_loop,
                                            centroids_last_num, 1)

    def cal_qc(self, queries_l1_low_dim, queries_l1, codebook, queries_align):
        
        with self.tik_instance.new_stmt_scope():
            queries_low_dim_l0c = self.tik_instance.Tensor("float32",
                                                           (self.dim2 // 16, queries_align, 16),
                                                           name="queries_low_dim_l0c", scope=tik.scope_cbuf_out)
            self.tik_instance.matmul(queries_low_dim_l0c,
                                     queries_l1, codebook,
                                     queries_align, self.dim, self.dim2)
            queries_low_dim_fp32_ub = self.tik_instance.Tensor("float32",
                                                               (self.dim2 // 16, queries_align, 16),
                                                               name="queries_low_dim_fp32_ub", scope=tik.scope_ubuf)
            self.tik_instance.data_move(queries_low_dim_fp32_ub,
                                        queries_low_dim_l0c,
                                        0, 1, queries_align * self.dim2 // 256, 0, 0)

            queries_low_dim_fp16_ub = self.tik_instance.Tensor("float16",
                                                               (self.dim2 // 16, queries_align, 16),
                                                               name="queries_low_dim_fp16_ub", scope=tik.scope_ubuf)
            vconv_repeat_times = self.dim2 * queries_align // self.fp32_mask
            self.tik_instance.vconv(self.fp32_mask, "none",
                                    queries_low_dim_fp16_ub,
                                    queries_low_dim_fp32_ub,
                                    vconv_repeat_times, 1, 1, 4, 8)

            self.tik_instance.data_move(queries_l1_low_dim, queries_low_dim_fp16_ub, 0, 1,
                                        queries_align * self.dim2 * 2 // 32, 0, 0)

    def cube_compute_each_loop(self, list_idx, data_xd_l1, add_scalar,
                               aicore_move_offset, precomputed_move_offset, code_move_offset,
                               code_move_num, flag):
        queries_align = 16
        queries_move_num = 1
        inner_product_l0c = self.tik_instance.Tensor("float32",
                                                     (self.code_num_each_loop // 16, queries_align, 16),
                                                     name="inner_product_l0c", scope=tik.scope_cbuf_out)
        code_l2_ub_fp16 = self.tik_instance.Tensor("float32",
                                                   (self.code_num_each_loop,),
                                                   name="code_l2_ub_fp16", scope=tik.scope_ubuf)
        code_move_num_ub = (code_move_num + 15) // 16 * 16
        
        loop_times = min((self.dim2 + 63) // 64 * 8, 64)
        loop_times = 64 if loop_times > 32 else loop_times
        code_num_ub_each_loop = self.code_num_each_loop // loop_times
        mat_thread = 1 if self.dim2 >= 64 and self.nprobe > 64 else 2
        if flag == 0:
            with self.tik_instance.for_range(0, loop_times, thread_num=mat_thread) as loop_ub:
                # move y from out to UB
                with self.tik_instance.new_stmt_scope():
                    code_ub = self.tik_instance.Tensor("uint8",
                                                       (self.dim2 // 16, code_num_ub_each_loop // 16, 16, 16),
                                                       name="code_ub", scope=tik.scope_ubuf)
                    code_ub_fp16 = self.tik_instance.Tensor("float16",
                                                            (self.dim2 // 16, code_num_ub_each_loop // 16, 16, 16),
                                                            name="code_ub_fp16", scope=tik.scope_ubuf)
                    with self.tik_instance.for_range(0, self.dim2 // 16) as i:
                        self.tik_instance.data_move(code_ub[i, 0, 0, 0],
                                                    self.input_code_gm[aicore_move_offset + 
                                                        (code_move_offset + loop_ub * code_num_ub_each_loop) * 
                                                            self.dim2 + i * 16 * 16],
                                                    0, (code_num_ub_each_loop+15) // 16, 8, self.dim2 // 2 - 8, 0)
    
                    # y do conv from uint8 to fp16
                    vconv_loop = (code_num_ub_each_loop * self.dim2) // (self.fp16_mask * 255)
                    vconv_offset = 0
                    if vconv_loop > 0:
                        for conv_index in range(vconv_loop):
                            vconv_offset = conv_index * self.fp16_mask * 255
                            self.tik_instance.vconv(self.fp16_mask, "none",
                                                    code_ub_fp16[vconv_offset],
                                                    code_ub[vconv_offset],
                                                    255, 1, 1, 8, 4)
                        vconv_offset += self.fp16_mask * 255
    
                    vconv_repeat_time = (code_num_ub_each_loop * self.dim2) % (self.fp16_mask * 255) // self.fp16_mask
                    if vconv_repeat_time > 0:
                        self.tik_instance.vconv(self.fp16_mask, "none",
                                                code_ub_fp16[vconv_offset],
                                                code_ub[vconv_offset],
                                                vconv_repeat_time, 1, 1, 8, 4)
    
                    # move y from ub to L1
                    code_l1 = self.tik_instance.Tensor("float16",
                                                       (self.dim2 // 16, code_num_ub_each_loop, 16),
                                                       name="code_l1", scope=tik.scope_cbuf)
                    self.tik_instance.data_move(code_l1, code_ub_fp16, 0, self.dim2 // 16, code_num_ub_each_loop, 0, 0)

                    self.tik_instance.matmul(inner_product_l0c[(loop_ub * code_num_ub_each_loop * queries_align):],
                                             data_xd_l1, code_l1,
                                             queries_align, self.dim2, code_num_ub_each_loop)

            # move v*v to ub
            self.tik_instance.data_move(code_l2_ub_fp16,
                                        self.input_precomputed_gm[precomputed_move_offset + code_move_offset],
                                        0, 1, self.code_num_each_loop // 8, 0, 0)
        else:
            code_loop = code_move_num_ub // code_num_ub_each_loop
            with self.tik_instance.for_range(0, code_loop, thread_num=mat_thread) as loop_ub:
                # move y from out to UB
                with self.tik_instance.new_stmt_scope():
                    code_ub = self.tik_instance.Tensor("uint8",
                                                       (self.dim2 // 16, code_num_ub_each_loop // 16, 16, 16),
                                                       name="code_ub", scope=tik.scope_ubuf)
                    code_ub_fp16 = self.tik_instance.Tensor("float16",
                                                            (self.dim2 // 16, code_num_ub_each_loop // 16, 16, 16),
                                                            name="code_ub_fp16", scope=tik.scope_ubuf)
                    with self.tik_instance.for_range(0, self.dim2 // 16) as i:
                        self.tik_instance.data_move(code_ub[i, 0, 0, 0],
                                                    self.input_code_gm[aicore_move_offset + 
                                                        (code_move_offset + loop_ub * code_num_ub_each_loop) * 
                                                                self.dim2 + i * 16 * 16],
                                                    0, (code_num_ub_each_loop+15) // 16, 8, self.dim2 // 2 - 8, 0)
    
                    # y do conv from uint8 to fp16
                    vconv_loop = (code_num_ub_each_loop * self.dim2) // (self.fp16_mask * 255)
                    vconv_offset = 0
                    if vconv_loop > 0:
                        for conv_index in range(vconv_loop):
                            vconv_offset = conv_index * self.fp16_mask * 255
                            self.tik_instance.vconv(self.fp16_mask, "none",
                                                    code_ub_fp16[vconv_offset],
                                                    code_ub[vconv_offset], 255,
                                                    1, 1, 8, 4)
                        vconv_offset += self.fp16_mask * 255
    
                    vconv_repeat_time = (code_num_ub_each_loop * self.dim2) % (self.fp16_mask * 255) // self.fp16_mask
                    if vconv_repeat_time > 0:
                        self.tik_instance.vconv(self.fp16_mask, "none",
                                                code_ub_fp16[vconv_offset],
                                                code_ub[vconv_offset],
                                                vconv_repeat_time, 1, 1, 8, 4)
    
                    # move y from ub to L1
                    code_l1 = self.tik_instance.Tensor("float16",
                                                       (self.dim2 // 16, code_num_ub_each_loop, 16),
                                                       name="code_l1", scope=tik.scope_cbuf)
                    self.tik_instance.data_move(code_l1, code_ub_fp16,
                                                0, self.dim2 // 16, code_num_ub_each_loop, 0, 0)
    
                    self.tik_instance.matmul(inner_product_l0c[(loop_ub * code_num_ub_each_loop * queries_align):],
                                             data_xd_l1, code_l1,
                                             queries_align, self.dim2, code_num_ub_each_loop)

            code_last = code_move_num_ub % code_num_ub_each_loop
            with self.tik_instance.if_scope(code_last > 0):
                # move y from out to UB
                with self.tik_instance.new_stmt_scope():
                    code_ub = self.tik_instance.Tensor("uint8",
                                                       (self.dim2 // 16, code_num_ub_each_loop // 16, 16, 16),
                                                       name="code_ub", scope=tik.scope_ubuf)
                    code_ub_fp16 = self.tik_instance.Tensor("float16",
                                                            (self.dim2 // 16, code_num_ub_each_loop // 16, 16, 16),
                                                            name="code_ub_fp16", scope=tik.scope_ubuf)
                    with self.tik_instance.for_range(0, self.dim2 // 16) as i:
                        self.tik_instance.data_move(code_ub[i, 0, 0, 0],
                                                    self.input_code_gm[aicore_move_offset + 
                                                    (code_move_offset + code_loop * code_num_ub_each_loop) * 
                                                            self.dim2 + i * 16 * 16],
                                                    0, (code_last+15) // 16, 8, self.dim2 // 2 - 8, 0)
    
                    with self.tik_instance.for_range(0, self.dim2 // 16) as i:
                        vconv_repeat_time = (code_last * 16) // self.fp16_mask
                        self.tik_instance.vconv(self.fp16_mask, "none",
                                                code_ub_fp16[i, 0, 0, 0],
                                                code_ub[i, 0, 0, 0],
                                                vconv_repeat_time, 1, 1, 8, 4)
    
                    # move y from ub to L1
                    code_l1 = self.tik_instance.Tensor("float16",
                                                       (self.dim2 // 16, code_num_ub_each_loop, 16),
                                                       name="code_l1", scope=tik.scope_cbuf)
                    self.tik_instance.data_move(code_l1, code_ub_fp16,
                                                0, self.dim2 // 16, code_last,
                                                (code_num_ub_each_loop - code_last),
                                                (code_num_ub_each_loop - code_last))
                    self.tik_instance.matmul(inner_product_l0c[(code_loop * code_num_ub_each_loop * queries_align):],
                                             data_xd_l1, code_l1,
                                             queries_align, self.dim2, code_num_ub_each_loop)

            # move code l2 from out to UB
            self.tik_instance.data_move(code_l2_ub_fp16,
                                        self.input_precomputed_gm[precomputed_move_offset + code_move_offset],
                                        0, 1, code_move_num_ub // 8, 0, 0)

        code_out_num = self.code_num_each_loop // 2
        if flag == 0:
            with self.tik_instance.for_range(0, 2, thread_num=1) as i:
                add_ub = self.tik_instance.Tensor("float32",
                                                  (queries_move_num, code_out_num),
                                                  name="add_ub", scope=tik.scope_ubuf)
                repeat = code_out_num // self.fp32_mask
                # calculate x*x - 2/255*x*(0.5*d + 255*m)+v*v
                self.tik_instance.vadds(self.fp32_mask,
                                        add_ub[0, 0],
                                        code_l2_ub_fp16[code_out_num * i], add_scalar,
                                        repeat, 1, 1, 8, 8)
                # move xy from L0-C to UB
                inner_product_ub = self.tik_instance.Tensor("float32",
                                                            (code_out_num // 16, queries_align, 16),
                                                            name="inner_product_ub", scope=tik.scope_ubuf)
                # move (x*d)*y from l0 to ub
                self.tik_instance.data_move(inner_product_ub,
                                            inner_product_l0c[i * code_out_num // 16, 0, 0],
                                            0, 1, code_out_num * queries_align // 256, 0, 0)

                # calculate x*x -2/255*x*(0.5*d+255*m)+v*v-2/255*(x*d)*y
                self.tik_instance.vaxpy(16,
                                        add_ub[0, 0],
                                        inner_product_ub[0, 0, 0],
                                        self.coeff,
                                        code_out_num // 16, 1, 1, 2, queries_align * 2)

                dst_ub = self.tik_instance.Tensor("float16",
                                                  (1, code_out_num),
                                                  name="dst_ub", scope=tik.scope_ubuf)
                query_num_each_loop = 1
                vconv_repeat_times = query_num_each_loop * code_out_num // self.fp32_mask

                self.tik_instance.vconv(self.fp32_mask, "none",
                                        dst_ub[0, 0],
                                        add_ub[0, 0],
                                        vconv_repeat_times, 1, 1, 4, 8)

                max_val_ub = self.tik_instance.Tensor("float16", (128,), name="max_val_ub", scope=tik.scope_ubuf)
                self.tik_instance.vec_dup(self.fp16_mask, max_val_ub, self.default_scalar, 1, 8)

                sel_ub = self.tik_instance.Tensor("uint8", (queries_move_num, code_out_num // 8),
                                                  name="sel_ub", scope=tik.scope_ubuf)
                with self.tik_instance.for_range(0, queries_move_num) as j:
                    self.tik_instance.data_move(sel_ub[j, 0],
                                                self.input_mask_gm[list_idx, 
                                                    (code_move_offset + i * code_out_num) // 8],
                                                0, 1, (code_out_num + 255) // 256, 8, 8)
                    vsel_loop = code_out_num // self.fp16_mask
                    if vsel_loop > 0:
                        for vloop in range(vsel_loop):
                            # sel_ub can not use repeat times > 1, use for + offset
                            voffset = vloop * self.fp16_mask
                            self.tik_instance.vec_sel(self.fp16_mask, 0, dst_ub[j, voffset],
                                                      sel_ub[j, voffset // 8], dst_ub[j, voffset],
                                                      max_val_ub, 1, 8, 8, 0)
                    vsel_last = code_out_num % self.fp16_mask
                    if vsel_last > 0:
                        vsel_offset = vsel_loop * self.fp16_mask
                        self.tik_instance.vec_sel(vsel_last, 0, dst_ub[j, vsel_offset], sel_ub[j, vsel_offset // 8],
                                                  dst_ub[j, vsel_offset], max_val_ub, 1, 8, 8, 0)

                self.tik_instance.data_move(self.output_distances_gm[list_idx,
                                                                     code_move_offset
                                                                     + i * code_out_num],
                                            dst_ub,
                                            0, 1, code_out_num // 16, 0,
                                            (self.code_num - code_out_num) // 16)

                min_size = code_out_num // self.min_batch * 2
                dst_min_ub = self.tik_instance.Tensor("float16",
                                                      (1, min_size),
                                                      name="dst_min_ub", scope=tik.scope_ubuf)

                vcmin_repeat_times = code_out_num // self.min_batch
                self.tik_instance.vcmin(self.min_batch,
                                        dst_min_ub[0, 0],
                                        dst_ub[0, 0],
                                        vcmin_repeat_times, 1, 1, self.min_batch // 16)
                self.tik_instance.data_move(self.output_mins_gm[list_idx,
                                                                (code_move_offset
                                                                 + i * code_out_num) // self.min_batch * 2],
                                            dst_min_ub,
                                            0, 1, (code_out_num // self.min_batch) // 8, 0,
                                            (self.code_num - code_out_num) // self.min_batch // 8)
        else:
            code_loops = (code_move_num_ub + code_out_num - 1) // code_out_num
            with self.tik_instance.for_range(0, code_loops, thread_num=1) as i:
                add_ub = self.tik_instance.Tensor("float32",
                                                  (queries_move_num, code_out_num),
                                                  name="add_ub", scope=tik.scope_ubuf)
                repeat = code_out_num // self.fp32_mask
                # calculate x*x - 2/255*x*(0.5*d + 255*m)+v*v
                self.tik_instance.vadds(self.fp32_mask,
                                        add_ub[0, 0],
                                        code_l2_ub_fp16[code_out_num * i], add_scalar,
                                        repeat, 1, 1, 8, 8)
                # move xy from L0-C to UB
                inner_product_ub = self.tik_instance.Tensor("float32",
                                                            (code_out_num // 16, queries_align, 16),
                                                            name="inner_product_ub", scope=tik.scope_ubuf)
                # move (x*d)*y from l0 to ub
                self.tik_instance.data_move(inner_product_ub,
                                            inner_product_l0c[i * code_out_num // 16, 0, 0],
                                            0, 1, code_out_num * queries_align // 256, 0, 0)
                # calculate 2/255*(x*d)
                self.tik_instance.vaxpy(16,
                                        add_ub[0, 0],
                                        inner_product_ub[0, 0, 0],
                                        self.coeff, code_out_num // 16, 1, 1, 2, queries_align * 2)

                dst_ub = self.tik_instance.Tensor("float16",
                                                  (1, code_out_num),
                                                  name="dst_ub", scope=tik.scope_ubuf)

                vconv_repeat_time = code_out_num // self.fp32_mask
                self.tik_instance.vconv(self.fp32_mask, "none",
                                        dst_ub[0, 0],
                                        add_ub[0, 0],
                                        vconv_repeat_time, 1, 1, 4, 8)

                max_val_ub = self.tik_instance.Tensor("float16", (128,), name="max_val_ub", scope=tik.scope_ubuf)
                self.tik_instance.vec_dup(self.fp16_mask, max_val_ub, self.default_scalar, 1, 8)

                sel_ub = self.tik_instance.Tensor("uint8", (queries_move_num, code_out_num // 8),
                                                  name="sel_ub", scope=tik.scope_ubuf)
                with self.tik_instance.for_range(0, queries_move_num) as j:
                    self.tik_instance.data_move(sel_ub[j, 0],
                                                self.input_mask_gm[list_idx, 
                                                                   (code_move_offset + i * code_out_num) // 8],
                                                0, 1, (code_out_num + 255) // 256, 8, 8)
                    vsel_loop = code_out_num // self.fp16_mask
                    if vsel_loop > 0:
                        for vloop in range(vsel_loop):
                            # sel_ub can not use repeat times > 1, use for + offset
                            voffset = vloop * self.fp16_mask
                            self.tik_instance.vec_sel(self.fp16_mask, 0, dst_ub[j, voffset],
                                                      sel_ub[j, voffset // 8], dst_ub[j, voffset],
                                                      max_val_ub, 1, 8, 8, 0)
                    vsel_last = code_out_num % self.fp16_mask
                    if vsel_last > 0:
                        vsel_offset = vsel_loop * self.fp16_mask
                        self.tik_instance.vec_sel(vsel_last, 0, dst_ub[j, vsel_offset], sel_ub[j, vsel_offset // 8],
                                                  dst_ub[j, vsel_offset], max_val_ub, 1, 8, 8, 0)

                self.tik_instance.data_move(self.output_distances_gm[list_idx,
                                                                     code_move_offset
                                                                     + i * code_out_num],
                                            dst_ub,
                                            0, 1, code_out_num // 16, 0,
                                            (self.code_num - code_out_num) // 16)

                min_size = code_out_num // self.min_batch * 2
                dst_min_ub = self.tik_instance.Tensor("float16",
                                                      (1, min_size),
                                                      name="dst_min_ub", scope=tik.scope_ubuf)
                with self.tik_instance.if_scope(i == code_loops - 1):
                    code_last_num = code_move_num - i * code_out_num
                    vcmin_repeat_times = code_last_num // self.min_batch
                    offset = 0
                    with self.tik_instance.if_scope(vcmin_repeat_times > 0):
                        self.tik_instance.vcmin(self.min_batch,
                                                dst_min_ub[0, 0],
                                                dst_ub[0, 0],
                                                vcmin_repeat_times, 1, 1, self.min_batch // 16)
                        offset += vcmin_repeat_times * self.min_batch
                    vcmin_remain = code_last_num % self.min_batch
                    with self.tik_instance.if_scope(vcmin_remain > 0):
                        self.tik_instance.vcmin(vcmin_remain,
                                                dst_min_ub[0, vcmin_repeat_times * 2],
                                                dst_ub[0, offset],
                                                1, 1, 1, 1)
                    self.tik_instance.data_move(self.output_mins_gm[list_idx,
                                                                    (code_move_offset
                                                                     + i * code_out_num) // self.min_batch * 2],
                                                dst_min_ub,
                                                0, queries_move_num, code_out_num // self.min_batch // 8, 0,
                                                (self.code_num - code_out_num) // self.min_batch // 8)
                with self.tik_instance.else_scope():
                    vcmin_repeat_times = code_out_num // self.min_batch
                    self.tik_instance.vcmin(self.min_batch,
                                            dst_min_ub[0, 0],
                                            dst_ub[0, 0],
                                            vcmin_repeat_times, 1, 1, self.min_batch // 16)

                    self.tik_instance.data_move(self.output_mins_gm[list_idx,
                                                                    (code_move_offset
                                                                     + i * code_out_num) // self.min_batch * 2],
                                                dst_min_ub,
                                                0, queries_move_num, (code_out_num // self.min_batch) // 8, 0,
                                                (self.code_num - code_out_num) // self.min_batch // 8)

    def cube_compute_each_loop_v2(self, list_idx, data_xd_l1, add_scalar,
                                  aicore_move_offset, precomputed_move_offset, code_move_offset,
                                  code_move_num, flag):
        queries_align = 16
        queries_move_num = 1
        inner_product_l0c = self.tik_instance.Tensor("float32",
                                                     (self.code_num_each_loop // 16, queries_align, 16),
                                                     name="inner_product_l0c", scope=tik.scope_cbuf_out)
        code_l2_ub_fp16 = self.tik_instance.Tensor("float32",
                                                   (self.code_num_each_loop,),
                                                   name="code_l2_ub_fp16", scope=tik.scope_ubuf)
        code_move_num_ub = (code_move_num + 15) // 16 * 16

        dim2_each_loop = 16
        loop_times = self.dim2 // dim2_each_loop
        code_num_ub_each_loop = self.code_num_each_loop
        mat_thread = 1
        if flag == 0:
            with self.tik_instance.for_range(0, loop_times, thread_num=mat_thread) as loop_ub:
                # move y from out to UB
                with self.tik_instance.new_stmt_scope():
                    code_ub = self.tik_instance.Tensor("uint8",
                                                       (dim2_each_loop // 16, code_num_ub_each_loop // 16, 16, 16),
                                                       name="code_ub", scope=tik.scope_ubuf)
                    code_ub_fp16 = self.tik_instance.Tensor("float16",
                                                            (dim2_each_loop // 16, code_num_ub_each_loop // 16, 16, 16),
                                                            name="code_ub_fp16", scope=tik.scope_ubuf)
                    with self.tik_instance.for_range(0, dim2_each_loop // 16) as i:
                        self.tik_instance.data_move(code_ub[i, 0, 0, 0],
                                                    self.input_code_gm[
                                                        aicore_move_offset + code_move_offset * 
                                                                self.dim2 + loop_ub * 256 + i * 16 * 16],
                                                    0, (code_num_ub_each_loop + 15) // 16, 8, self.dim2 // 2 - 8, 0)

                    # y do conv from uint8 to fp16
                    vconv_loop = (code_num_ub_each_loop * dim2_each_loop) // (self.fp16_mask * 255)
                    vconv_offset = 0
                    if vconv_loop > 0:
                        for conv_index in range(vconv_loop):
                            vconv_offset = conv_index * self.fp16_mask * 255
                            self.tik_instance.vconv(self.fp16_mask, "none",
                                                    code_ub_fp16[vconv_offset],
                                                    code_ub[vconv_offset],
                                                    255, 1, 1, 8, 4)
                        vconv_offset += self.fp16_mask * 255

                    vconv_repeat_time = (code_num_ub_each_loop * dim2_each_loop) % (
                            self.fp16_mask * 255) // self.fp16_mask
                    if vconv_repeat_time > 0:
                        self.tik_instance.vconv(self.fp16_mask, "none",
                                                code_ub_fp16[vconv_offset],
                                                code_ub[vconv_offset],
                                                vconv_repeat_time, 1, 1, 8, 4)

                    # move y from ub to L1
                    code_l1 = self.tik_instance.Tensor("float16",
                                                       (dim2_each_loop // 16, code_num_ub_each_loop, 16),
                                                       name="code_l1", scope=tik.scope_cbuf)
                    self.tik_instance.data_move(code_l1, code_ub_fp16, 0, dim2_each_loop // 16, code_num_ub_each_loop,
                                                0, 0)

                    with self.tik_instance.if_scope(loop_ub == 0):

                        self.tik_instance.matmul(inner_product_l0c,
                                                 data_xd_l1[loop_ub:(loop_ub + 1), :, :], code_l1,
                                                 queries_align, dim2_each_loop, code_num_ub_each_loop)
                    with self.tik_instance.else_scope():
                        self.tik_instance.matmul(inner_product_l0c,
                                                 data_xd_l1[loop_ub:(loop_ub + 1), :, :], code_l1,
                                                 queries_align, dim2_each_loop, code_num_ub_each_loop, False)

            # move v*v to ub
            self.tik_instance.data_move(code_l2_ub_fp16,
                                        self.input_precomputed_gm[precomputed_move_offset + code_move_offset],
                                        0, 1, self.code_num_each_loop // 8, 0, 0)
        else:
            code_last = code_move_num_ub
            with self.tik_instance.if_scope(code_last > 0):
                # move y from out to UB
                with self.tik_instance.for_range(0, loop_times, thread_num=mat_thread) as loop_ub:
                    with self.tik_instance.new_stmt_scope():
                        code_ub = self.tik_instance.Tensor("uint8",
                                                           (dim2_each_loop // 16, code_num_ub_each_loop // 16, 16, 16),
                                                           name="code_ub", scope=tik.scope_ubuf)
                        code_ub_fp16 = self.tik_instance.Tensor("float16",
                                                                (dim2_each_loop // 16, code_num_ub_each_loop // 16, 16,
                                                                 16),
                                                                name="code_ub_fp16", scope=tik.scope_ubuf)
                        with self.tik_instance.for_range(0, dim2_each_loop // 16) as i:
                            self.tik_instance.data_move(code_ub[i, 0, 0, 0],
                                                        self.input_code_gm[
                                                            aicore_move_offset + code_move_offset * 
                                                                    self.dim2 + loop_ub * 256 + i * 16 * 16],
                                                        0, (code_last + 15) // 16, 8, self.dim2 // 2 - 8, 0)

                        vconv_loop = (code_num_ub_each_loop * dim2_each_loop) // (self.fp16_mask * 255)
                        vconv_offset = 0
                        if vconv_loop > 0:
                            for conv_index in range(vconv_loop):
                                vconv_offset = conv_index * self.fp16_mask * 255
                                self.tik_instance.vconv(self.fp16_mask, "none",
                                                        code_ub_fp16[vconv_offset],
                                                        code_ub[vconv_offset],
                                                        255, 1, 1, 8, 4)
                            vconv_offset += self.fp16_mask * 255

                        vconv_repeat_time = (code_num_ub_each_loop * dim2_each_loop) % (
                                self.fp16_mask * 255) // self.fp16_mask
                        if vconv_repeat_time > 0:
                            self.tik_instance.vconv(self.fp16_mask, "none",
                                                    code_ub_fp16[vconv_offset],
                                                    code_ub[vconv_offset],
                                                    vconv_repeat_time, 1, 1, 8, 4)

                        # move y from ub to L1
                        code_l1 = self.tik_instance.Tensor("float16",
                                                           (dim2_each_loop // 16, code_num_ub_each_loop, 16),
                                                           name="code_l1", scope=tik.scope_cbuf)
                        self.tik_instance.data_move(code_l1, code_ub_fp16,
                                                    0, dim2_each_loop // 16, code_last,
                                                    (code_num_ub_each_loop - code_last),
                                                    (code_num_ub_each_loop - code_last))
                        with self.tik_instance.if_scope(loop_ub == 0):
                            self.tik_instance.matmul(inner_product_l0c,
                                                     data_xd_l1[loop_ub:(loop_ub + 1), :, :], code_l1,
                                                     queries_align, dim2_each_loop, code_num_ub_each_loop)
                        with self.tik_instance.else_scope():
                            self.tik_instance.matmul(
                                inner_product_l0c,
                                data_xd_l1[loop_ub:(loop_ub + 1), :, :], code_l1,
                                queries_align, dim2_each_loop, code_num_ub_each_loop, False)

            # move code l2 from out to UB
            self.tik_instance.data_move(code_l2_ub_fp16,
                                        self.input_precomputed_gm[precomputed_move_offset + code_move_offset],
                                        0, 1, code_move_num_ub // 8, 0, 0)

        code_out_num = self.code_num_each_loop // 2
        if flag == 0:
            with self.tik_instance.for_range(0, 2, thread_num=1) as i:
                add_ub = self.tik_instance.Tensor("float32",
                                                  (queries_move_num, code_out_num),
                                                  name="add_ub", scope=tik.scope_ubuf)
                repeat = code_out_num // self.fp32_mask
                # calculate x*x - 2/255*x*(0.5*d + 255*m)+v*v
                self.tik_instance.vadds(self.fp32_mask,
                                        add_ub[0, 0],
                                        code_l2_ub_fp16[code_out_num * i], add_scalar,
                                        repeat, 1, 1, 8, 8)
                # move xy from L0-C to UB
                inner_product_ub = self.tik_instance.Tensor("float32",
                                                            (code_out_num // 16, queries_align, 16),
                                                            name="inner_product_ub", scope=tik.scope_ubuf)
                # move (x*d)*y from l0 to ub
                self.tik_instance.data_move(inner_product_ub,
                                            inner_product_l0c[i * code_out_num // 16, 0, 0],
                                            0, 1, code_out_num * queries_align // 256, 0, 0)

                # calculate x*x -2/255*x*(0.5*d+255*m)+v*v-2/255*(x*d)*y
                self.tik_instance.vaxpy(16,
                                        add_ub[0, 0],
                                        inner_product_ub[0, 0, 0],
                                        self.coeff,
                                        code_out_num // 16, 1, 1, 2, queries_align * 2)

                dst_ub = self.tik_instance.Tensor("float16",
                                                  (1, code_out_num),
                                                  name="dst_ub", scope=tik.scope_ubuf)
                # vconv_times 
                query_num_each_loop = 1
                vconv_repeat_times = query_num_each_loop * code_out_num // self.fp32_mask

                self.tik_instance.vconv(self.fp32_mask, "none",
                                        dst_ub[0, 0],
                                        add_ub[0, 0],
                                        vconv_repeat_times, 1, 1, 4, 8)
                
                max_val_ub = self.tik_instance.Tensor("float16", (128,), name="max_val_ub", scope=tik.scope_ubuf)
                self.tik_instance.vec_dup(self.fp16_mask, max_val_ub, self.default_scalar, 1, 8)

                sel_ub = self.tik_instance.Tensor("uint8", (queries_move_num, code_out_num // 8),
                                                  name="sel_ub", scope=tik.scope_ubuf)
                with self.tik_instance.for_range(0, queries_move_num) as j:
                    self.tik_instance.data_move(sel_ub[j, 0],
                                                self.input_mask_gm[
                                                    list_idx, (code_move_offset + i * code_out_num) // 8],
                                                0, 1, (code_out_num + 255) // 256, 8, 8)
                    vsel_loop = code_out_num // self.fp16_mask
                    if vsel_loop > 0:
                        for vloop in range(vsel_loop):
                            # sel_ub can not use repeat times > 1, use for + offset
                            voffset = vloop * self.fp16_mask
                            self.tik_instance.vec_sel(self.fp16_mask, 0, dst_ub[j, voffset],
                                                      sel_ub[j, voffset // 8], dst_ub[j, voffset],
                                                      max_val_ub, 1, 8, 8, 0)
                    vsel_last = code_out_num % self.fp16_mask
                    if vsel_last > 0:
                        vsel_offset = vsel_loop * self.fp16_mask
                        self.tik_instance.vec_sel(vsel_last, 0, dst_ub[j, vsel_offset], sel_ub[j, vsel_offset // 8],
                                                  dst_ub[j, vsel_offset], max_val_ub, 1, 8, 8, 0)

                self.tik_instance.data_move(self.output_distances_gm[list_idx,
                                                                     code_move_offset
                                                                     + i * code_out_num],
                                            dst_ub,
                                            0, 1, code_out_num // 16, 0,
                                            (self.code_num - code_out_num) // 16)

                min_size = code_out_num // self.min_batch * 2
                dst_min_ub = self.tik_instance.Tensor("float16",
                                                      (1, min_size),
                                                      name="dst_min_ub", scope=tik.scope_ubuf)

                vcmin_repeat_times = code_out_num // self.min_batch
                self.tik_instance.vcmin(self.min_batch,
                                        dst_min_ub[0, 0],
                                        dst_ub[0, 0],
                                        vcmin_repeat_times, 1, 1, self.min_batch // 16)
                self.tik_instance.data_move(self.output_mins_gm[list_idx,
                                                                (code_move_offset
                                                                 + i * code_out_num) // self.min_batch * 2],
                                            dst_min_ub,
                                            0, 1, (code_out_num // self.min_batch) // 8, 0,
                                            (self.code_num - code_out_num) // self.min_batch // 8)
        else:
            code_loops = (code_move_num_ub + code_out_num - 1) // code_out_num
            with self.tik_instance.for_range(0, code_loops, thread_num=1) as i:
                add_ub = self.tik_instance.Tensor("float32",
                                                  (queries_move_num, code_out_num),
                                                  name="add_ub", scope=tik.scope_ubuf)
                repeat = code_out_num // self.fp32_mask
                # calculate x*x - 2/255*x*(0.5*d + 255*m)+v*v
                self.tik_instance.vadds(self.fp32_mask,
                                        add_ub[0, 0],
                                        code_l2_ub_fp16[code_out_num * i], add_scalar,
                                        repeat, 1, 1, 8, 8)
                # move xy from L0-C to UB
                inner_product_ub = self.tik_instance.Tensor("float32",
                                                            (code_out_num // 16, queries_align, 16),
                                                            name="inner_product_ub", scope=tik.scope_ubuf)
                # move (x*d)*y from l0 to ub
                self.tik_instance.data_move(inner_product_ub,
                                            inner_product_l0c[i * code_out_num // 16, 0, 0],
                                            0, 1, code_out_num * queries_align // 256, 0, 0)

                # calculate 2/255*(x*d)
                self.tik_instance.vaxpy(16,
                                        add_ub[0, 0],
                                        inner_product_ub[0, 0, 0],
                                        self.coeff, code_out_num // 16, 1, 1, 2, queries_align * 2)

                dst_ub = self.tik_instance.Tensor("float16",
                                                  (1, code_out_num),
                                                  name="dst_ub", scope=tik.scope_ubuf)

                vconv_repeat_time = code_out_num // self.fp32_mask
                self.tik_instance.vconv(self.fp32_mask, "none",
                                        dst_ub[0, 0],
                                        add_ub[0, 0],
                                        vconv_repeat_time, 1, 1, 4, 8)
                
                max_val_ub = self.tik_instance.Tensor("float16", (128,), name="max_val_ub", scope=tik.scope_ubuf)
                self.tik_instance.vec_dup(self.fp16_mask, max_val_ub, self.default_scalar, 1, 8)

                sel_ub = self.tik_instance.Tensor("uint8", (queries_move_num, code_out_num // 8),
                                                  name="sel_ub", scope=tik.scope_ubuf)
                with self.tik_instance.for_range(0, queries_move_num) as j:
                    self.tik_instance.data_move(sel_ub[j, 0],
                                                self.input_mask_gm[list_idx,
                                                                   (code_move_offset + i * code_out_num) // 8],
                                                0, 1, (code_out_num + 255) // 256, 8, 8)
                    vsel_loop = code_out_num // self.fp16_mask
                    if vsel_loop > 0:
                        for vloop in range(vsel_loop):
                            # sel_ub can not use repeat times > 1, use for + offset
                            voffset = vloop * self.fp16_mask
                            self.tik_instance.vec_sel(self.fp16_mask, 0, dst_ub[j, voffset],
                                                      sel_ub[j, voffset // 8], dst_ub[j, voffset],
                                                      max_val_ub, 1, 8, 8, 0)
                    vsel_last = code_out_num % self.fp16_mask
                    if vsel_last > 0:
                        vsel_offset = vsel_loop * self.fp16_mask
                        self.tik_instance.vec_sel(vsel_last, 0, dst_ub[j, vsel_offset], sel_ub[j, vsel_offset // 8],
                                                  dst_ub[j, vsel_offset], max_val_ub, 1, 8, 8, 0)

                self.tik_instance.data_move(self.output_distances_gm[list_idx,
                                                                     code_move_offset
                                                                     + i * code_out_num],
                                            dst_ub,
                                            0, 1, code_out_num // 16, 0,
                                            (self.code_num - code_out_num) // 16)

                min_size = code_out_num // self.min_batch * 2
                dst_min_ub = self.tik_instance.Tensor("float16",
                                                      (1, min_size),
                                                      name="dst_min_ub", scope=tik.scope_ubuf)
                with self.tik_instance.if_scope(i == code_loops - 1):
                    code_last_num = code_move_num - i * code_out_num
                    vcmin_repeat_times = code_last_num // self.min_batch
                    offset = 0
                    with self.tik_instance.if_scope(vcmin_repeat_times > 0):
                        self.tik_instance.vcmin(self.min_batch,
                                                dst_min_ub[0, 0],
                                                dst_ub[0, 0],
                                                vcmin_repeat_times, 1, 1, self.min_batch // 16)
                        offset += vcmin_repeat_times * self.min_batch
                    vcmin_remain = code_last_num % self.min_batch
                    with self.tik_instance.if_scope(vcmin_remain > 0):
                        self.tik_instance.vcmin(vcmin_remain,
                                                dst_min_ub[0, vcmin_repeat_times * 2],
                                                dst_ub[0, offset],
                                                1, 1, 1, 1)
                    self.tik_instance.data_move(self.output_mins_gm[list_idx,
                                                                    (code_move_offset
                                                                     + i * code_out_num) // self.min_batch * 2],
                                                dst_min_ub,
                                                0, queries_move_num, code_out_num // self.min_batch // 8, 0,
                                                (self.code_num - code_out_num) // self.min_batch // 8)
                with self.tik_instance.else_scope():
                    vcmin_repeat_times = code_out_num // self.min_batch
                    self.tik_instance.vcmin(self.min_batch,
                                            dst_min_ub[0, 0],
                                            dst_ub[0, 0],
                                            vcmin_repeat_times, 1, 1, self.min_batch // 16)

                    self.tik_instance.data_move(self.output_mins_gm[list_idx,
                                                                    (code_move_offset
                                                                     + i * code_out_num) // self.min_batch * 2],
                                                dst_min_ub,
                                                0, queries_move_num, (code_out_num // self.min_batch) // 8, 0,
                                                (self.code_num - code_out_num) // self.min_batch // 8)

    def compute_and_min_v3(self):
        self.cal_num_each_core()
        with self.tik_instance.for_range(0, self.aicore_use, block_num=self.aicore_use) as block_index:
            for core_idx in range(self.aicore_use):
                with self.tik_instance.if_scope(core_idx == block_index):
                    self.distance_compute_each_loop_v3(core_idx)

            one = self.tik_instance.Scalar(dtype="uint16", name="one", init_value=1)
            flag_ub = self.tik_instance.Tensor("uint16",
                                               (16,),
                                               name="flag_ub", scope=tik.scope_ubuf)

            flag_ub[0].set_as(one)
            self.tik_instance.data_move(self.output_flag_gm[block_index, 0],
                                        flag_ub,
                                        0, 1, 1, 0, 0)

    def get_tik_instance(self):
        """
        obtain tik instance
        """
        self.compute_and_min_v3()

        self.tik_instance.BuildCCE(
            kernel_name=self.kernel_name,
            inputs=[
                self.input_queries_gm, self.input_codebook_gm, self.input_code_gm, self.input_mask_gm,
                self.input_precomputed_gm, self.input_dm_gm, self.input_actual_code_num_gm,
                self.input_codebook_offset_gm, self.input_base_offset_gm, self.input_precomputed_offset_gm,
            ],
            outputs=[self.output_distances_gm, self.output_mins_gm, self.output_flag_gm])

        return self.tik_instance

    def _muls(self, dst, src, scalar, compute_num, mask):
        # process 256B data per repeat for vmuls
        repeat = compute_num // mask
        offset = 0
        if repeat > 0:
            self.tik_instance.vmuls(mask, dst[0], src[0], scalar,
                                    repeat, 1, 1, 8, 8)
            offset += repeat * mask

        remain = compute_num % mask
        if remain > 0:
            self.tik_instance.vmuls(remain, dst[offset], src[offset], scalar,
                                    1, 1, 1, 8, 8)

    def _add(self, dst, src0, src1, compute_num, mask):
        # process 256B data per repeat for vadd
        repeat = compute_num // mask
        offset = 0
        if repeat > 0:
            self.tik_instance.vadd(mask, dst[offset], src0[offset],
                                   src1[offset], repeat, 1, 1, 1, 8, 8, 8)
            offset += repeat * mask
        remain = compute_num % mask
        if remain > 0:
            self.tik_instance.vadd(remain, dst[offset], src0[offset],
                                   src1[offset], 1, 1, 1, 1, 8, 8, 8)


def distance_masked_ivf_sp_int_l2_mins(input_queries,
                                input_codebook,
                                input_code,
                                input_mask,
                                input_precomputed,
                                input_dm,
                                input_actual_code_num,
                                input_codebook_offset,
                                input_base_offset,
                                input_precomputed_offset,
                                output_distances,
                                output_mins,
                                output_flag,
                                kernel_name="distance_masked_ivf_sp_int_l2_mins"):
    """
    calculating distance

    Parameters
    ----------
    input_queries : dict
        shape and dtype of query vector
    input_code : dict
        shape and dtype of coding vector
    input_precomputed : dict
        shape and dtype of precomputed L2 distance of coding vector
    input_diff: dict
        shape and dtype of coding coefficient
    input_min: dict
        shape and dtype of coding offset
    input_actual_code_num: dict
        shape and dtype of actual code num,
        shape must be (8,) and dtype must be uint32
    output_distances : dict
        shape and dtype of distances, should be same dtype as input_queries
    output_mins : dict
        shape and dtype of mins
    output_flag: dict
        shape and dtype of output flag,
        shape must be (32,) and dtype must be uint16
    kernel_name : str
        kernel name, default value is "distance_sq8_l2_mins"

    Returns
    -------
    None
    """
    distance_sq8 = DistanceIVFSpIntL2Mins(input_queries, input_codebook, input_code, input_mask,
                                          input_precomputed, input_dm, input_actual_code_num,
                                          input_codebook_offset, input_base_offset, input_precomputed_offset,
                                          output_distances, output_mins, output_flag,
                                          kernel_name)
    tik_instance = distance_sq8.get_tik_instance()
    return tik_instance