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
import collections
from te import tik
from mxIndex_impl.common import set_soc_info

DisMovePara = collections.namedtuple('DisMovePara', ['aicore_move_offset', 'centroids_move_offset',
    'centroids_move_num', 'queries_move_num'])


class DistanceInt8CosMaxsFilter:
    def __init__(self,
                 input_queries,
                 input_mask,
                 input_centroids,
                 input_queries_m,
                 input_centroids_m,
                 input_actual_num,
                 input_threshold,
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
        self.shape_threshold = input_threshold.get("shape")  # [NewOp] 新增
        self.dtype_threshold = input_threshold.get("dtype")  # [NewOp] 新增
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

        self.block_offset = None
        self.mask_len = None
        self.use_mask = None
        self.threshold_ub = None
        self.centroids_num_each_core = None
        self.centroids_num_last_core = None
        self.actual_num = None

        # set vector fp32 mask and fp16 mask
        self.int32_mask = 64
        self.fp16_mask = 128
        self.max_mask = 64
        self.max_repeat = 192
        self.scale = 0.01

        # set tik instance
        self.set_tik_instance()

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
        self.even_mask = ((1 << (self.centroids_num_each_loop // 32)) - 1) // 3  # [NewOp] 新增

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
        # [NewOp] 新增 入参 threshold
        self.input_threshold_gm = self.tik_instance.Tensor(self.dtype_threshold,
                                                         self.shape_threshold,
                                                         name="input_threshold_gm",
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
        self.actual_num = self.tik_instance.Scalar(dtype="uint32", name="actual_code_num", init_value=0)
        self.actual_num.set_as(actual_num_ub[0])

        self.block_offset = self.tik_instance.Scalar(dtype="uint32",
                                                    name="block_offset",
                                                    init_value=0)
        self.block_offset.set_as(actual_num_ub[1])
 
        self.mask_len = self.tik_instance.Scalar(dtype="uint32",
                                                 name="mask_len",
                                                 init_value=0)
        self.mask_len.set_as(actual_num_ub[2])

        self.use_mask = self.tik_instance.Scalar(dtype="uint32",
                                                 name="use_mask",
                                                 init_value=0)
        self.use_mask.set_as(actual_num_ub[3])

        code_size = self.tik_instance.Scalar(dtype="uint32",
                                             name="code_size",
                                             init_value=0)
        code_size.set_as(actual_num_ub[4])

        if self.aicore_use == 2:
            self.centroids_num_each_core = (self.actual_num // self.aicore_use + self.max_mask * 8) \
                    // self.max_mask // 16 * self.max_mask * 16
        else:
            self.centroids_num_each_core \
                = self.actual_num // self.aicore_use // self.max_mask // 16 * self.max_mask * 16

        self.centroids_num_last_core = self.actual_num - (self.aicore_use - 1) * self.centroids_num_each_core


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
                move_para = DisMovePara(aicore_move_offset, loop_centroids * self.centroids_num_each_loop,
                                        self.centroids_num_each_loop, move_num)
                self.cube_compute_each_loop(queries_l1, queries_m_ub, move_para, 0)

        centroids_last_num = aicore_centroids_num % self.centroids_num_each_loop
        with self.tik_instance.if_scope(centroids_last_num > 0):
            move_para = DisMovePara(aicore_move_offset, centroids_loop_time * self.centroids_num_each_loop,
                                    centroids_last_num, move_num)
            self.cube_compute_each_loop(queries_l1, queries_m_ub, move_para, 1)

    def cube_compute_each_loop(self, queries_l1, queries_m_ub, move_para, flag):
        aicore_move_offset, centroids_move_offset, centroids_move_num, queries_move_num = move_para
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
        for loop_index in range(0, queries_move_num):
            # calc x * y * (1 / x_m * 1 / y_m)
            self.tik_instance.vmul(16,
                                   res_ub[loop_index, 0],
                                   inner_product_ub[0, loop_index, 0], res_ub[loop_index, 0],
                                   centroids_align_16 // 16, 1, 1, 1, 1, queries_align, 1)

        with self.tik_instance.if_scope(self.use_mask > 0):
            min_val_ub = self.tik_instance.Tensor("float16", (128,), name="min_val_ub", scope=tik.scope_ubuf)
            self.tik_instance.vec_dup(self.fp16_mask, min_val_ub, self.default_scalar, 1, 8)

            # malloc memory on chip
            sel_ub = self.tik_instance.Tensor("uint8", (queries_move_num, (self.centroids_num_each_loop + 7) // 8),
                                              name="sel_ub", scope=tik.scope_ubuf)
            with self.tik_instance.for_range(0, queries_move_num) as j:
                # move data from input_mask_gm to sel_ub
                self.tik_instance.data_move(
                    sel_ub[j, 0],
                    self.input_mask_gm[j * self.mask_len
                                       + (self.block_offset + aicore_move_offset + centroids_move_offset) // 8],
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

        max_ub = self.tik_instance.Tensor("float16",
                                          (queries_move_num, self.centroids_num_each_loop // 32),
                                          name="max_ub", scope=tik.scope_ubuf)

        # [NewOp] 对不满的block仍然使用原来的"dist全部搬运"
        with self.tik_instance.if_scope(self.actual_num < self.centroids_num):
            self.tik_instance.data_move(self.output_distances_gm[0, aicore_move_offset + centroids_move_offset],
                                        res_ub,
                                        0, queries_move_num, self.centroids_num_each_loop // 16, 0,
                                        (self.centroids_num - self.centroids_num_each_loop) // 16)
            self._max(max_ub, res_ub, queries_move_num, self.centroids_num_each_loop, self.max_mask)

        # [NewOp] 对满的block使用新算子的"dist部分搬运"
        with self.tik_instance.else_scope():
            self._max(max_ub, res_ub, queries_move_num, self.centroids_num_each_loop, self.max_mask)
            # [NewOp] 对 res_ub 进行部分搬运，只搬运 "代表"(max_ub) 比 threshold 大的组
            # (如果"代表"比threshold(当前topK的最小值)还小，所在的组就不够大，就不搬出去了[cos值越大,距离越小])
            # 注意：比较指令 h_cmpv 没有mask参数，所以 threshold 的shape与 max_ub 相同，没有剔除奇数index的labels

            # [NewOp] 比较 max_ub 和 threshold_ub，得到需要搬运的组的fp16标签 is_valid_fp
            # 使用 h_cmpv 进行比较，比较结果存储于 mask_tensor(可能因为Scalar不支持bool而无法直接被使用，只能被h_sel识别）中
            # 使用 h_sel 提取比较结果 is_valid_fp (需要制作全为0.0和全为1.0的fp16 Tensor all_zeros和all_ones)
            # 准备Tensor
            all_zeros = self.tik_instance.Tensor(
                dtype="float16", shape=(queries_move_num, self.centroids_num_each_loop // 32),
                name="all_zeros", scope=tik.scope_ubuf)
            all_ones = self.tik_instance.Tensor(
                dtype="float16", shape=(queries_move_num, self.centroids_num_each_loop // 32),
                name="all_ones", scope=tik.scope_ubuf)
            self.tik_instance.vec_dup(mask=16, dst=all_zeros, scalar=0.0,
                                      repeat_times=queries_move_num * self.centroids_num_each_loop // 32 // 16,
                                      dst_rep_stride=1)
            self.tik_instance.vec_dup(mask=16, dst=all_ones, scalar=1.0,
                                      repeat_times=queries_move_num * self.centroids_num_each_loop // 32 // 16,
                                      dst_rep_stride=1)
            mask_tensor = self.tik_instance.Tensor(
                dtype="bool", shape=(queries_move_num, self.centroids_num_each_loop // 32),
                name="mask_tensor", scope=tik.scope_ubuf)

            is_valid_fp = self.tik_instance.Tensor(
                dtype="float16", shape=(queries_move_num, self.centroids_num_each_loop // 32),
                name="is_valid_fp", scope=tik.scope_ubuf)
            # 进行比较和提取
            self.tik_instance.h_cmpv(mask_tensor=mask_tensor, src0=max_ub, src1=self.threshold_ub, cmp_mode="GE")
            self.tik_instance.h_sel(dst=is_valid_fp, src0=all_ones, src1=all_zeros, mask_tensor=mask_tensor)

            # [NewOp] 为了性能，对于每个查询向量，只要当前loop中有一组需要搬运，就搬运整个loop
            # 所以需要对 is_valid_fp 的每行进行求和，如果结果大于等于1，就进行搬运
            # 注意：vec_reduce_add仅支持fp16/fp32，不支持int32，所以在求和前不对is_valid_fp转换数据类型
            # 准备Tensor （vec_reduce_add 本身用于求整个的和，但是副产物是各次repeat的和，所以要为两者都准备Tensor）
            is_valid_loop_fp = self.tik_instance.Tensor(
                dtype="float16", shape=(queries_move_num, ), name="is_valid_loop_fp", scope=tik.scope_ubuf)
            is_valid_loop_sum_fp = self.tik_instance.Tensor(
                dtype="float16", shape=(queries_move_num, ), name="is_valid_loop_sum_fp", scope=tik.scope_ubuf)
            # 对 is_valid_fp 进行求和 (注意：要跳过奇数index处的"比较labels")
            # is_valid_fp的shape是 (nq, 16or32), 要跳过偶数位置，mask对应 21845 or 1431655765
            # fp16时，一次最多处理128位; stride=一次处理的长度=(self.centroids_num_each_loop//32)*2B/32B
            self.tik_instance.vec_reduce_add(
                [0, self.even_mask], is_valid_loop_sum_fp, is_valid_fp, is_valid_loop_fp,
                repeat_times=queries_move_num, src_rep_stride=self.centroids_num_each_loop // 32 // 16)

            # [NewOp] 使用 h_cast 将 is_valid_loop_fp 转换为 int32型, 得到真正可用的 is_valid_loop_ub
            # h_sel提取到的比较结果为float型， vec_reduce_add也仅支持float型，但是float型无法参与>或者<比较,
            # 所以㤇将其转换为整型；注意：h_cast 在310p 上不支持转换为 int16, 只能转换为 int32
            is_valid_loop_ub = self.tik_instance.Tensor(
                dtype="int32", shape=(queries_move_num, ), name="is_valid_loop_ub", scope=tik.scope_ubuf)
            with self.tik_instance.if_scope(self.queries_num == 1):
                self.tik_instance.h_cast(dst=is_valid_loop_ub, src=is_valid_loop_sum_fp, round_mode="round")
            with self.tik_instance.else_scope():
                self.tik_instance.h_cast(dst=is_valid_loop_ub, src=is_valid_loop_fp, round_mode="round")

            # [NewOp] 进行部分搬运
            with self.tik_instance.for_range(0, queries_move_num) as q_index: # 查询向量的index
                with self.tik_instance.if_scope(is_valid_loop_ub[q_index] >= 1):
                    self.tik_instance.data_move(
                        dst=self.output_distances_gm[  # 形状为 (nq, blockSize)
                            q_index, aicore_move_offset + centroids_move_offset],
                        src=res_ub[q_index, 0],
                        sid=0, nburst=1, burst=self.centroids_num_each_loop // 16,
                        src_stride=0, dst_stride=0)

        self.tik_instance.data_move(
            self.output_max_dist_gm[0, (aicore_move_offset + centroids_move_offset) // self.max_mask * 2],
            max_ub,
            0, queries_move_num, (self.centroids_num_each_loop + self.max_mask - 1) // self.max_mask // 8,
            0, (self.centroids_num - self.centroids_num_each_loop) // self.max_mask // 8)

    def distance_compute_cos_maxs(self):
        """
        the compute process
        """
        # [NewOp] self.input_threshold_gm 搬运到 ub (src和dst的shape都与每个loop的 max_ub 相同)
        # 每个loop使用的threshold_ub都相同，所以在这里处理
        self.threshold_ub = self.tik_instance.Tensor(
            dtype="float16", shape=(self.queries_num, self.centroids_num_each_loop // 32),
            name="threshold_ub", scope=tik.scope_ubuf)
        # 要搬运 queries_move_num*centroids_num_each_loop//32*2B//32B = batch * 1or2 个burst
        self.tik_instance.data_move(
            dst=self.threshold_ub, src=self.input_threshold_gm, sid=0,
            nburst=self.queries_num, burst=self.centroids_num_each_loop // 16 // 32,
            src_stride=0, dst_stride=0)

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

        self.tik_instance.BuildCCE(kernel_name=self.kernel_name,
                                   inputs=[
                                       self.input_queries_gm,
                                       self.input_mask_gm,
                                       self.input_centroids_gm,
                                       self.input_queries_m_gm,
                                       self.input_centroids_m_gm,
                                       self.input_actual_num_gm,
                                       self.input_threshold_gm  # [NewOp] 新增入参
                                   ],
                                   outputs=[
                                       self.output_distances_gm,
                                       self.output_max_dist_gm,
                                       self.output_flag_gm])

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


def distance_int8_cos_maxs_filter(input_queries,
                                  input_mask,
                                  input_centroids,
                                  input_queries_m,
                                  input_centroids_m,
                                  input_actual_num,
                                  input_threshold,  # [NewOp] 新增入参
                                  output_distances,
                                  output_max_dist,
                                  output_flag,
                                  kernel_name="distance_int8_cos_maxs_filter"):  # [NewOp] 新算子名字
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
        kernel name, default value is "distance_int8_cos_maxs"

    Returns
    -------
    None
    """
    # [newOp] 使用新算子名字；增加入参input_threshold
    distance_cos_maxs_filter = DistanceInt8CosMaxsFilter(input_queries, input_mask, input_centroids,
                                            input_queries_m, input_centroids_m, input_actual_num,
                                            input_threshold, output_distances, output_max_dist,
                                            output_flag, kernel_name)
    tik_instance = distance_cos_maxs_filter.get_tik_instance()
    return tik_instance
