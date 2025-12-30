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


class DistanceInt8CosMaxsWithMask:
    def __init__(self,
                 input_queries,
                 input_mask,
                 input_centroids,
                 input_queries_m,
                 input_centroids_m,
                 input_actual_num,
                 output_distances,
                 output_max_dist,
                 output_flag,
                 kernel_name="distance_int8_cos_maxs"):
        self.shape_queries = input_queries.get("shape")
        self.dtype_queries = input_queries.get("dtype")
        self.shape_mask = input_mask.get("shape")
        self.dtype_mask = input_mask.get("dtype")
        self.shape_centroids = input_centroids.get("shape")
        self.dtype_centroids = input_centroids.get("dtype")
        self.shape_queries_m = input_queries_m.get("shape")
        self.dtype_queries_m = input_queries_m.get("dtype")
        self.shape_centroids_m = input_centroids_m.get("shape")
        self.dtype_centroids_m = input_centroids_m.get("dtype")
        self.shape_actual_num = input_actual_num.get("shape")
        self.dtype_actual_num = input_actual_num.get("dtype")
        self.shape_distances = output_distances.get("shape")
        self.dtype_distances = output_distances.get("dtype")
        self.shape_max_dist = output_max_dist.get("shape")
        self.dtype_max_dist = output_max_dist.get("dtype")
        self.shape_flag = output_flag.get("shape")
        self.dtype_flag = output_flag.get("dtype")
        self.kernel_name = kernel_name

        self.queries_num, self.dim = self.shape_queries
        self.centroids_num, = self.shape_centroids_m
        self.queries_num_align = (self.queries_num + 15) // 16 * 16

        # check parameter
        self.check_parameter()

        self.block_mask_len  = self.centroids_num // 8
        if self.shape_mask[0] == 1 and self.shape_mask[0] != self.shape_queries[0]:
            self.share_distance_mask    = True
        else:
            self.share_distance_mask    = False

        # set vector fp32 mask and fp16 mask
        self.int32_mask = 64
        self.fp16_mask = 128
        self.max_mask = 64
        self.max_repeat = 192
        self.scale = 0.01
        self.use_extra_score = False

        # set tik instance
        self.set_tik_instance()

    def set_extra_score(self, input_extra_score):
        self.use_extra_score = True
        self.shape_extra_score = input_extra_score.get("shape")
        self.dtype_extra_score = input_extra_score.get("dtype")
        self.input_extra_score_gm = self.tik_instance.Tensor(self.dtype_extra_score,
                                                             self.shape_extra_score,
                                                             name="input_extra_score_gm",
                                                             scope=tik.scope_gm)

    def check_parameter(self):
        if self.dim % 16 != 0:
            raise RuntimeError("feature dim must be a multiple of 16")
        if self.centroids_num % 16 != 0:
            raise RuntimeError("centroids num must be a multiple of 16")

    def set_tik_instance(self):
        """
        set tik_instance
        """
        set_soc_info()
        self.tik_instance = tik.Tik()

        self.aicore_use = self.shape_actual_num[0]
        self.queries_num_each_loop = min(48, self.queries_num)
        if self.dim == 768:
            self.centroids_num_each_loop = min(min((48 // self.queries_num_each_loop) * 512, 1024), 512)
        else:
            self.centroids_num_each_loop = min(min((48 // self.queries_num_each_loop) * 512, 1024),
                                               512 * 1024 // self.dim)

        self.set_src_dst_tensor()

    def set_src_dst_tensor(self):
        """
        set input and output tensor
        """
        self.coeff = self.tik_instance.Scalar("int32", name="coeff", init_value=-2)

        self.query_m = self.tik_instance.Scalar("float16", name="query_l2", init_value=0)
        self.default_scalar = self.tik_instance.Scalar("float16",
                                                       name="default_scalar",
                                                       init_value=-65500)
        # creat input tensor: input_queries_gm, input_centroids_gm,
        # input_queries_m_gm, input_centroids_m_gm, input_actual_num_gm
        # and output tensor: output_distances_gm, output_max_dist_gm, output_flag_gm in global buffer
        self.input_queries_gm = self.tik_instance.Tensor(self.dtype_queries,
                                                         self.shape_queries,
                                                         name="input_queries_gm",
                                                         scope=tik.scope_gm)
        self.input_mask_gm = self.tik_instance.Tensor(self.dtype_mask,
                                                      self.shape_mask,
                                                      name="input_mask_gm",
                                                      scope=tik.scope_gm)
        self.input_centroids_gm = self.tik_instance.Tensor(self.dtype_centroids,
                                                           self.shape_centroids,
                                                           name="input_centroids_gm",
                                                           scope=tik.scope_gm)
        self.input_queries_m_gm = self.tik_instance.Tensor(self.dtype_queries_m,
                                                           self.shape_queries_m,
                                                           name="input_queries_m_gm",
                                                           scope=tik.scope_gm)
        self.input_centroids_m_gm = self.tik_instance.Tensor(self.dtype_centroids_m,
                                                             self.shape_centroids_m,
                                                             name="input_centroids_m_gm",
                                                             scope=tik.scope_gm)
        self.input_actual_num_gm = self.tik_instance.Tensor(self.dtype_actual_num,
                                                            self.shape_actual_num,
                                                            name="input_actual_num_gm",
                                                            scope=tik.scope_gm)
        self.output_distances_gm = self.tik_instance.Tensor(self.dtype_distances,
                                                            self.shape_distances,
                                                            name="output_distances_gm",
                                                            scope=tik.scope_gm)
        self.output_max_dist_gm = self.tik_instance.Tensor(self.dtype_max_dist,
                                                           self.shape_max_dist,
                                                           name="output_max_dist_gm", scope=tik.scope_gm)
        self.output_flag_gm = self.tik_instance.Tensor(self.dtype_flag,
                                                       self.shape_flag,
                                                       name="output_flag_gm",
                                                       scope=tik.scope_gm)

    def cal_num_each_core(self):
        """
        calculate actual code num of each core
        """
        # move actual code num from out to UB
        actual_num_ub = self.tik_instance.Tensor("uint32",
                                                 (8,),
                                                 name="actual_code_num_ub", scope=tik.scope_ubuf)
        self.tik_instance.data_move(actual_num_ub,
                                    self.input_actual_num_gm,
                                    0, 1, 1, 0, 0)
        actual_num = self.tik_instance.Scalar(dtype="uint32", name="actual_code_num", init_value=0)
        self.extra_score_num = self.tik_instance.Scalar(dtype="uint64", name="extra_score_num", init_value=0)
        actual_num.set_as(actual_num_ub[0])
        self.extra_score_num.set_as(actual_num_ub[1])


        if self.aicore_use == 2:
            self.centroids_num_each_core = (actual_num  // self.aicore_use + self.max_mask * 8) \
                    // self.max_mask // 16 * self.max_mask * 16
        else:
            self.centroids_num_each_core = actual_num  // self.aicore_use // self.max_mask // 16 * self.max_mask * 16

        self.centroids_num_last_core = actual_num  - (self.aicore_use - 1) * self.centroids_num_each_core


    def distance_compute_each_loop(self, aicore_move_offset, aicore_centroids_num, move_num):
        queries_align = (move_num + 15) // 16 * 16

        queries_l1 = self.tik_instance.Tensor("int8",
                                              (self.dim // 32, queries_align, 32),
                                              name="queries_l1", scope=tik.scope_cbuf)
        with self.tik_instance.new_stmt_scope():
            for i in range(move_num):
                self.tik_instance.data_move(queries_l1[0, i, 0],
                                            self.input_queries_gm[i, 0],
                                            0, self.dim // 32, 1, 0, queries_align - 1)

        queries_m_ub = self.tik_instance.Tensor("float16",
                                                (self.queries_num_align,),
                                                name="queries_m_ub", scope=tik.scope_ubuf)

        self.tik_instance.data_move(queries_m_ub,
                                    self.input_queries_m_gm,
                                    0, 1, self.queries_num_align // 16, 0, 0)
        # compute xy using cube
        centroids_loop_time = aicore_centroids_num // self.centroids_num_each_loop
        with self.tik_instance.if_scope(centroids_loop_time > 0):
            with self.tik_instance.for_range(0, centroids_loop_time) as loop_centroids:
                self.cube_compute_each_loop(queries_l1, queries_m_ub, aicore_move_offset,
                                            loop_centroids * self.centroids_num_each_loop,
                                            self.centroids_num_each_loop, move_num, 0)

        centroids_last_num = aicore_centroids_num % self.centroids_num_each_loop
        with self.tik_instance.if_scope(centroids_last_num > 0):
            self.cube_compute_each_loop(queries_l1, queries_m_ub, aicore_move_offset,
                                        centroids_loop_time * self.centroids_num_each_loop,
                                        centroids_last_num, move_num, 1)

    def cube_compute_each_loop(self, queries_l1, queries_m_ub,
                               aicore_move_offset, centroids_move_offset, centroids_move_num,
                               queries_move_num, flag):
        queries_align = (queries_move_num + 15) // 16 * 16
        centroids_align_16 = (centroids_move_num + 15) // 16 * 16
        centroids_l1 = self.tik_instance.Tensor("int8",
                                                (self.dim // 32, self.centroids_num_each_loop, 32),
                                                name="centroids_l1", scope=tik.scope_cbuf)
        for i in range(self.dim // 32):
            self.tik_instance.data_move(centroids_l1[i, 0, 0],
                                        self.input_centroids_gm[(aicore_move_offset + centroids_move_offset) // 16,
                                                                i, 0, 0],
                                        0, centroids_align_16 // 16, 16, self.dim // 2 - 16, 0)

        # mov centroids l2 from out to UB
        centroids_m_ub = self.tik_instance.Tensor("float16",
                                                  (self.centroids_num_each_loop,),
                                                  name="centroids_m_ub", scope=tik.scope_ubuf)
        self.tik_instance.data_move(centroids_m_ub,
                                    self.input_centroids_m_gm[aicore_move_offset + centroids_move_offset],
                                    0, 1, self.centroids_num_each_loop // 16, 0, 0)

        res_ub = self.tik_instance.Tensor("float16",
                                          (queries_move_num, self.centroids_num_each_loop),
                                          name="res_ub", scope=tik.scope_ubuf)

        for loop_index in range(0, queries_move_num):
            self.query_m.set_as(queries_m_ub[loop_index])

            # 1 / x_m * 1 / y_m
            self.tik_instance.vmuls(self.fp16_mask,
                                    res_ub[loop_index, 0],
                                    centroids_m_ub, self.query_m,
                                    (centroids_move_num + self.fp16_mask - 1) // self.fp16_mask,
                                    1, 1, self.fp16_mask // 16, self.fp16_mask // 16)

        inner_product_l0c = self.tik_instance.Tensor("int32",
                                                     (self.centroids_num_each_loop // 16, queries_align, 16),
                                                     name="inner_product_l0c", scope=tik.scope_cbuf_out)
        self.tik_instance.matmul(inner_product_l0c,
                                 queries_l1, centroids_l1,
                                 queries_align, self.dim, self.centroids_num_each_loop)

        # mov xy from L0-C to UB
        inner_product_ub = self.tik_instance.Tensor("float16",
                                                    (self.centroids_num_each_loop // 16, queries_align, 16),
                                                    name="inner_product_ub", scope=tik.scope_ubuf)
        self.tik_instance.tensor_mov(inner_product_ub,
                                     inner_product_l0c,
                                     'm',
                                     1, centroids_align_16 * queries_align // 256, 0, 0, deqscale=self.scale)
        extra_score_ub = None
        if self.use_extra_score:
            extra_score_ub = self.tik_instance.Tensor("float16",
                                                  (queries_move_num, self.centroids_num_each_loop),
                                                  name="extra_score_ub", scope=tik.scope_ubuf)
        for loop_index in range(0, queries_move_num):
            self.tik_instance.vmul(16,
                                   res_ub[loop_index, 0],
                                   inner_product_ub[0, loop_index, 0], res_ub[loop_index, 0],
                                   centroids_align_16 // 16, 1, 1, 1, 1, queries_align, 1)
            if self.use_extra_score:
                self.tik_instance.data_move(extra_score_ub[loop_index, 0],
                    self.input_extra_score_gm[0, loop_index * self.extra_score_num + \
                        aicore_move_offset + centroids_move_offset], 0, 1, self.centroids_num_each_loop // 16, 0, 0)
                self.tik_instance.vadd(16, res_ub[loop_index, 0],
                                    extra_score_ub[loop_index, 0], res_ub[loop_index, 0],
                                    self.centroids_num_each_loop // 16, 1, 1, 1, 1, 1, 1)
                self.tik_instance.vec_muls(16, res_ub[loop_index, 0], res_ub[loop_index, 0], 0.5,
                                        self.centroids_num_each_loop // 16, 1, 1)

        with self.tik_instance.for_range(0, 1):
            min_val_ub = self.tik_instance.Tensor("float16", (128,), name="min_val_ub", scope=tik.scope_ubuf)
            self.tik_instance.vec_dup(self.fp16_mask, min_val_ub, self.default_scalar, 1, 8)

            # malloc memory on chip
            sel_ub = self.tik_instance.Tensor("uint8", (queries_move_num, (self.centroids_num_each_loop + 7) // 8),
                                              name="sel_ub", scope=tik.scope_ubuf)
            with self.tik_instance.for_range(0, queries_move_num) as j:
                # move data from input_mask_gm to sel_ub
                if self.share_distance_mask is not True:
                    self.tik_instance.data_move(
                        sel_ub[j, 0],
                        self.input_mask_gm[j * self.block_mask_len
                                        + (aicore_move_offset + centroids_move_offset) // 8],
                        0, 1, (self.centroids_num_each_loop + 255) // 256, 8, 8)
                else:
                    self.tik_instance.data_move(
                        sel_ub[j, 0],
                        self.input_mask_gm[(aicore_move_offset + centroids_move_offset) // 8],
                        0, 1, (self.centroids_num_each_loop + 255) // 256, 8, 8)

                # cal the loop need to execute the selection process
                vsel_loop = self.centroids_num_each_loop // self.fp16_mask
                if vsel_loop > 0:
                    for vloop in range(vsel_loop):
                        # sel_ub can not use repeat times > 1, use for + offset
                        voffset = vloop * self.fp16_mask
                        # select value in res_ub according to sel_ub
                        self.tik_instance.vec_sel(self.fp16_mask, 0, res_ub[j, voffset],
                                                  sel_ub[j, voffset // 8], res_ub[j, voffset],
                                                  min_val_ub, 1, 8, 8, 0)

                # handle tail in case for self.centroids_num_each_loop % self.fp16_mask != 0
                vsel_last = self.centroids_num_each_loop % self.fp16_mask
                if vsel_last > 0:
                    vsel_offset = vsel_loop * self.fp16_mask
                    self.tik_instance.vec_sel(vsel_last, 0, res_ub[j, vsel_offset], sel_ub[j, vsel_offset // 8],
                                              res_ub[j, vsel_offset], min_val_ub, 1, 8, 8, 0)

        self.tik_instance.data_move(self.output_distances_gm[0, aicore_move_offset + centroids_move_offset],
                                    res_ub,
                                    0, queries_move_num, self.centroids_num_each_loop // 16, 0,
                                    (self.centroids_num - self.centroids_num_each_loop) // 16)
        max_ub = self.tik_instance.Tensor("float16",
                                          (queries_move_num, self.centroids_num_each_loop // 32),
                                          name="max_ub", scope=tik.scope_ubuf)
        self._max(max_ub, res_ub, queries_move_num, self.centroids_num_each_loop, self.max_mask)
        self.tik_instance.data_move(
            self.output_max_dist_gm[0, (aicore_move_offset + centroids_move_offset) // self.max_mask * 2],
            max_ub,
            0, queries_move_num, (self.centroids_num_each_loop + self.max_mask - 1) // self.max_mask // 8,
            0, (self.centroids_num - self.centroids_num_each_loop) // self.max_mask // 8)

    def distance_compute_cos_maxs(self):
        """
        the compute process
        """
        self.cal_num_each_core()

        with self.tik_instance.for_range(0, self.aicore_use, block_num=self.aicore_use) as block_index:
            aicore_centroids_num = self.tik_instance.Scalar(dtype="uint32", name="aicore_code_num", init_value=0)
            # compute centroids num and move offset every core
            with self.tik_instance.if_scope(block_index != self.aicore_use - 1):
                aicore_centroids_num.set_as(self.centroids_num_each_core)
            with self.tik_instance.else_scope():
                aicore_centroids_num.set_as(self.centroids_num_last_core)

            self.distance_compute_each_loop(block_index * self.centroids_num_each_core,
                                            aicore_centroids_num,
                                            self.queries_num)

            one = self.tik_instance.Scalar(dtype="uint16", name="one", init_value=1)
            flag_ub = self.tik_instance.Tensor("uint16",
                                               (16,),
                                               name="flag_ub", scope=tik.scope_ubuf)

            flag_ub[0].set_as(one)
            self.tik_instance.pipe_barrier("PIPE_MTE3")
            self.tik_instance.data_move(self.output_flag_gm[block_index, 0],
                                        flag_ub,
                                        0, 1, 1, 0, 0)

    def get_tik_instance(self):
        """
        obtain tik instance
        """
        self.distance_compute_cos_maxs()
        if self.use_extra_score:
            self.tik_instance.BuildCCE(kernel_name=self.kernel_name,
                                   inputs=[
                                       self.input_queries_gm,
                                       self.input_mask_gm,
                                       self.input_centroids_gm,
                                       self.input_queries_m_gm,
                                       self.input_centroids_m_gm,
                                       self.input_actual_num_gm,
                                       self.input_extra_score_gm
                                   ],
                                   outputs=[self.output_distances_gm, self.output_max_dist_gm, self.output_flag_gm])
        else:
            self.tik_instance.BuildCCE(kernel_name=self.kernel_name,
                                    inputs=[
                                        self.input_queries_gm,
                                        self.input_mask_gm,
                                        self.input_centroids_gm,
                                        self.input_queries_m_gm,
                                        self.input_centroids_m_gm,
                                        self.input_actual_num_gm
                                    ],
                                    outputs=[self.output_distances_gm, self.output_max_dist_gm, self.output_flag_gm])
        return self.tik_instance

    def _max(self, dst, src, queries_num, centroids_num, mask):
        # process 256B data per repeat for vcmax
        loop_times = queries_num * centroids_num // mask // self.max_repeat
        offset = 0
        for _ in range(loop_times):
            self.tik_instance.vcmax(mask, dst[offset, 0], src[offset, 0], self.max_repeat, 1, 1, mask // 16)
            offset += (self.max_repeat * mask // centroids_num)

        remain = queries_num * centroids_num // mask % self.max_repeat
        if remain > 0:
            self.tik_instance.vcmax(mask, dst[offset, 0], src[offset, 0], remain, 1, 1, mask // 16)


def distance_int8_cos_maxs_with_mask(input_queries,
                           input_mask,
                           input_centroids,
                           input_queries_m,
                           input_centroids_m,
                           input_actual_num,
                           output_distances,
                           output_max_dist,
                           output_flag,
                           kernel_name="distance_int8_cos_maxs_with_mask"):
    """
    calculating distance

    Parameters
    ----------
    input_queries : dict
        shape and dtype of query vector
    input_centroids : dict
        shape and dtype of  centroids
    input_queries_m: dict
        shape and dtype of input_queries
    input_centroids_m : dict
        shape and dtype of centroids_m
    input_actual_num: dict
        shape and dtype of actual code num,
        shape must be (8,) and dtype must be uint32
    output_distances : dict
        shape and dtype of distances, should be same dtype as input_queries
    output_max_dist : dict
        shape and dtype of distances, should be same dtype as input_queries
    output_flag : dict
        shape and dtype of flag, only the 1th and 17th is valid
    kernel_name : str
        kernel name, default value is "distance_int8_cos_maxs_with_mask"

    Returns
    -------
    None
    """
    distance_cos_maxs_with_mask = DistanceInt8CosMaxsWithMask(input_queries, input_mask, input_centroids,
                                            input_queries_m, input_centroids_m, input_actual_num,
                                            output_distances, output_max_dist, output_flag, kernel_name)
    tik_instance = distance_cos_maxs_with_mask.get_tik_instance()
    return tik_instance
