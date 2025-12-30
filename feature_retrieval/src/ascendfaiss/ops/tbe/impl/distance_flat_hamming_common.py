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

AI_CORE_NUM     = 8
MAX_FP16        = 65500.0
MIN_FP16        = -65500.0


def div_up(dividend_val, divisor_val):
    return ((dividend_val + divisor_val - 1) // divisor_val)


def scan_block_topk_in_ub(tik_instance, input_ub, input_cnt, output_ub, block_size, block_topk=1):
    """
    输入UB Tensor，输出每一个块的topk，存放在另一个UB Tensor中，当前只支持block_topk=1。
    Args:
        input_ub: 需要扫描的UB Tensor，一维，数据类型必须是float16
        input_cnt: 要处理的输入数据元素数量
        output_ub: 输出的block topk结果UB Tensor，一维，数据类型也必须是float16
        block_size: 块中包含的元素个数
        block_topk：块topk值
    约束：
        元素类型须为float16
    """
    if block_size % 16 != 0 or block_size > 128:
        raise RuntimeError("scan_block_topk_in_ub: invalid block size!")

    max_repeat_each_call    = 128                               # 一次vcmax执行最大的repeat数
    total_repeat            = input_cnt // block_size           # 总的repeat数
    
    tmp_repeat_sc           = tik_instance.Scalar(dtype="uint32")
    call_cnt                = div_up(total_repeat, max_repeat_each_call)
    with tik_instance.for_range(0, call_cnt) as cid: 
        with tik_instance.if_scope(cid != (call_cnt - 1)):
            tmp_repeat_sc.set_as(max_repeat_each_call)
        with tik_instance.else_scope():
            tmp_repeat_sc.set_as(total_repeat - (call_cnt - 1) * max_repeat_each_call)
        
        tik_instance.vcmax(block_size,
                           output_ub[cid * max_repeat_each_call * 2 :],
                           input_ub[cid * max_repeat_each_call * block_size :],
                           tmp_repeat_sc,
                           1, 1, 
                           block_size * 2 // 32)

    last_remain = input_cnt % block_size
    with tik_instance.if_scope(last_remain > 0):
        tik_instance.vcmax(last_remain,
                           output_ub[total_repeat * 2 :],
                           input_ub[total_repeat * block_size :],
                           1,
                           1, 1,
                           8)


def filter_distance_using_mask_in_ub(tik_instance, input_ub, input_cnt, output_ub, mask, value):
    """
    根据Mask值对input_ub Tensor中的每个元素值进行更新。
    Args:
        input_ub：输入的UB Tensor，一维, dtype=float16
        input_cnt：要处理的输入数据元素数量
        output_ub：输出的UB Tensor，一维，dtype=float16
        mask：mask值是一个Tensor，一维，dtype=uint8，bit=1时，使用input_ub中元素值赋值给output_ub，bit=0时，使用value赋值给output_ub。
        value：立即数或者Scalar变量
    约束：
        元素类型须为float16，input_cnt必须小于等于input_ub Tensor的长度。
    """
    max_repeat_each_call    = 128
    total_repeat            = input_cnt // 128
    
    tmp_repeat_sc           = tik_instance.Scalar(dtype="uint32")
    call_cnt                = div_up(total_repeat, max_repeat_each_call)
    with tik_instance.for_range(0, call_cnt) as cid:
        with tik_instance.if_scope(cid != (call_cnt - 1)):
            tmp_repeat_sc.set_as(max_repeat_each_call)
        with tik_instance.else_scope():
            tmp_repeat_sc.set_as(total_repeat - (call_cnt - 1) * max_repeat_each_call)
        tik_instance.vsel(128,
                          1,
                          output_ub[cid * max_repeat_each_call * 128 :],
                          mask[cid * max_repeat_each_call * 128 // 8 :],
                          input_ub[cid * max_repeat_each_call * 128 :],
                          value,
                          tmp_repeat_sc, 
                          1, 1, 1, 8, 8, 8)

    last_remain = input_cnt % 128
    with tik_instance.if_scope(last_remain > 0):
        tik_instance.vsel(last_remain,
                          1,
                          output_ub[total_repeat * 128 :],
                          mask[total_repeat * 128 // 8 :],
                          input_ub[total_repeat * 128 :],
                          value,
                          1,
                          1, 1, 1, 8, 8, 8)

    
class DistanceFlatHamming:
    def __init__(self, input_queries, input_db, db_num, distance_mask,
                 output_distance, output_block_topk, output_flag, kernel_name):
        self.shape_queries                  = input_queries.get("shape")
        self.dtype_queries                  = input_queries.get("dtype")

        self.shape_db                       = input_db.get("shape")
        self.dtype_db                       = input_db.get("dtype")
        
        self.shape_db_num                   = db_num.get("shape")
        self.dtype_db_num                   = db_num.get("dtype")
        # 距离值掩码，可选参数。
        self.use_distance_mask              = False
        if distance_mask is not None:
            self.shape_distance_mask        = distance_mask.get("shape")
            self.dtype_distance_mask        = distance_mask.get("dtype")
            self.use_distance_mask          = True
            if self.shape_distance_mask[0] == 1 and self.shape_distance_mask[0] != self.shape_queries[0]:
                self.share_distance_mask    = True
            else:
                self.share_distance_mask    = False
        # Z区域高度，即Z区域包含的向量个数。
        self.db_zregion_h                   = self.shape_db[2]
        
        self.shape_distance                 = output_distance.get("shape")
        self.dtype_distance                 = output_distance.get("dtype")

        self.shape_block_topk               = output_block_topk.get("shape")
        self.dtype_block_topk               = output_block_topk.get("dtype")

        self.shape_output_flag              = output_flag.get("shape")
        self.dtype_output_flag              = output_flag.get("dtype")
       
        self.kernel_name                    = kernel_name

        self.total_queries                  = self.shape_queries[0]
        self.dim                            = self.shape_queries[1] * 8

        self.total_db_num                   = self.shape_db[0] * self.shape_db[2] 
        # 每次执行matmul指令时右矩阵的跨度，即一次做多少个底库向量的距离计算，16的倍数。
        self.db_slice_stride                = 1024 
        # 每次从GM种加载的二值化短特征个数，受限于UB大小。
        # 为了加快从GM中加载底库向量，mini_batch设置为Z区域高度，即Z区域包含的向量个数。
        # 为了统一起见，加载查询向量时也采用这个设置。
        self.mini_batch                     = self.db_zregion_h
        # 每次计算距离对应的查询条数
        self.query_num_each_compute         = self.total_queries
        # 保存距离结果时按照一次多少条查询对应的距离值进行保存。
        self.query_num_each_save            = self.query_num_each_compute
    
        self.core_num                       = AI_CORE_NUM

        self.block_size                     = 128
        self.block_topk                     = 1

        self.blocks_each_db_slice           = self.db_slice_stride // self.block_size

        set_soc_info()
        self.tik_instance       = tik.Tik(disable_debug=False)
        self.queries_gm         = self.tik_instance.Tensor(self.dtype_queries, self.shape_queries, 
                                                           name="queries_gm", scope=tik.scope_gm)
        self.db_gm              = self.tik_instance.Tensor(self.dtype_db, self.shape_db, 
                                                           name="db_gm", scope=tik.scope_gm)
        self.db_num_gm          = self.tik_instance.Tensor(self.dtype_db_num, self.shape_db_num, 
                                                           name="db_num_gm", scope=tik.scope_gm)
        if self.use_distance_mask is True:
            self.distance_mask_gm   = self.tik_instance.Tensor(self.dtype_distance_mask,
                                                               self.shape_distance_mask,
                                                               name="distance_mask_gm",
                                                               scope=tik.scope_gm)

        self.distance_gm        = self.tik_instance.Tensor(self.dtype_distance, self.shape_distance, 
                                                           name="distance_gm", scope=tik.scope_gm)
        self.block_topk_gm      = self.tik_instance.Tensor(self.dtype_block_topk, self.shape_block_topk, 
                                                           name="block_topk_gm", scope=tik.scope_gm)
        self.output_flag_gm     = self.tik_instance.Tensor(self.dtype_output_flag, self.shape_output_flag,
                                                           name="output_flag_gm", scope=tik.scope_gm)
    
    def set_optimize_policy(self):
        if self.total_queries >= 32: 
            self.query_num_each_compute = 32
            self.query_num_each_save    = 32
            self.core_num               = AI_CORE_NUM
        if self.dim == 1024:
            self.db_slice_stride        = 512
            self.block_size             = 64

    def check_settings(self):
        if self.dim % 32 != 0:
            raise RuntimeError("dim must be multiple of 32!")
        if self.db_zregion_h % 16 != 0:
            raise RuntimeError("z region height must be multiple of 16!")
        if self.db_slice_stride % self.db_zregion_h != 0:
            raise RuntimeError("db_slice_stride must be multiple of z region height!")
        if (((self.mini_batch * self.dim)) % 128 != 0) or (((self.mini_batch * self.dim)) // 128 >= 256):
            raise RuntimeError("mini_batch is too large!")

        if self.total_db_num % self.db_slice_stride != 0:
            raise RuntimeError("total_db_num should be multiple of db_slice_stride!")
        if self.block_size > 128:
            raise RuntimeError("block size should be smaller than 128!")
        if self.db_slice_stride % self.block_size != 0:
            raise RuntimeError("db_slice_stride should be multiple of block_size!")
        if ((self.db_slice_stride // self.block_size * 4) % 32) != 0:  
            raise RuntimeError("invalid db_slice_stride!")
        

    def get_tik_instance(self):
        # 设置优化策略
        self.set_optimize_policy()
        # 参数检查
        self.check_settings()

        sync_workspace = self.tik_instance.Tensor("int64", 
                                                 (self.core_num * 32 // 8,), 
                                                 name="barrier_workspace",
                                                 scope=tik.scope_gm, 
                                                 is_workspace=True, 
                                                 is_atomic_add=True)

        with self.tik_instance.for_range(0, self.core_num, block_num=self.core_num) as tid:
            self.execute_task(tid)
            # 等待所有的task完成，然后由第一个task置完成位标志。
            self.tik_instance.block_barrier(sync_workspace)
            with self.tik_instance.if_scope(tid == 0):
                output_flag_ub = self.tik_instance.Tensor("uint16", (16,), name="output_flag_ub", scope=tik.scope_ubuf)
                output_flag_ub[0].set_as(1)
                self.tik_instance.data_move(self.output_flag_gm[0],
                                            output_flag_ub[0],
                                            0,
                                            1, 1, 0, 0)
        
        if self.use_distance_mask is not True:
            inputs  = [self.queries_gm, 
                       self.db_gm, 
                       self.db_num_gm]
            outputs = [self.distance_gm, 
                       self.block_topk_gm, 
                       self.output_flag_gm]
        else:
            inputs  = [self.queries_gm, 
                       self.db_gm, 
                       self.db_num_gm,
                       self.distance_mask_gm]
            outputs = [self.distance_gm, 
                       self.block_topk_gm, 
                       self.output_flag_gm]

        self.tik_instance.BuildCCE(self.kernel_name,
                                   inputs, 
                                   outputs)
        
        return self.tik_instance
       
    def execute_task(self, task_id):
        """
        执行任务函数：由于db_slice的个数可能不整除core_num，所以前面的task会多分配一个db_slice。
        """
        # 获取底库向量的实际条数
        db_num_sc = self.tik_instance.Scalar(dtype="uint32", name="db_num_sc", init_value=0)
        with self.tik_instance.for_range(0, 1):
            db_num_ub = self.tik_instance.Tensor(self.dtype_db_num, self.shape_db_num, 
                                                 name="db_num_ub", scope=tik.scope_ubuf)
            self.tik_instance.data_move(db_num_ub[0], 
                                        self.db_num_gm[0], 
                                        0, 
                                        1, self.shape_db_num[0] * 4 // 32, 0, 0)
            db_num_sc.set_as(db_num_ub[0])

        # 计算本任务对应的底库向量起始位置以及要处理的底库向量个数。
        db_slice_num_total      = div_up(db_num_sc, self.db_slice_stride)
        db_slice_num_each_task  = db_slice_num_total // self.core_num

        db_slice_num_this_task  = self.tik_instance.Scalar(dtype="uint32")
        db_slice_off_this_task  = self.tik_instance.Scalar(dtype="uint32")

        db_slice_last_remain    = db_slice_num_total % self.core_num
        with self.tik_instance.if_scope(db_slice_last_remain == 0):
            db_slice_num_this_task.set_as(db_slice_num_each_task)
            db_slice_off_this_task.set_as(task_id * db_slice_num_each_task)
        with self.tik_instance.else_scope():
            with self.tik_instance.if_scope(task_id < db_slice_last_remain):
                db_slice_num_this_task.set_as(db_slice_num_each_task + 1)
                db_slice_off_this_task.set_as(task_id * (db_slice_num_each_task + 1))
            with self.tik_instance.else_scope():
                db_slice_num_this_task.set_as(db_slice_num_each_task)
                tmp_slice_num = db_slice_last_remain * (db_slice_num_each_task + 1)
                tmp_slice_off = (task_id - db_slice_last_remain) * db_slice_num_each_task
                db_slice_off_this_task.set_as(tmp_slice_num + tmp_slice_off)

        db_vector_num       = db_slice_num_this_task * self.db_slice_stride 
        db_vector_offset    = db_slice_off_this_task * self.db_slice_stride

        # 开始计算
        self.compute_distance_each_task(self.queries_gm, 
                                        db_vector_offset,
                                        db_vector_num, 
                                        self.db_gm, 
                                        db_num_sc, 
                                        self.distance_gm, 
                                        self.block_topk_gm)
        
    def compute_distance_each_task(self, queries_gm, db_vector_offset, 
                                   db_vector_num, db_gm, db_num_sc, 
                                   distance_gm, block_topk_gm):
        with self.tik_instance.if_scope(db_vector_num != 0):  
            # 定义全1.0的UB Tensor
            pos_one_ub = self.tik_instance.Tensor("float16", 
                                                  (self.mini_batch * self.dim,), 
                                                  name="pos_one_ub", 
                                                  scope=tik.scope_ubuf)
            # 定义全-1.0的UB Tensor
            neg_one_ub = self.tik_instance.Tensor("float16",
                                                  (self.mini_batch * self.dim,),
                                                  name="neg_one_ub",
                                                  scope=tik.scope_ubuf)
            # 初始化以上两个Tensor
            self.tik_instance.vector_dup(128, pos_one_ub[0], 1.0, (self.mini_batch * self.dim) // 128, 1, 8, 0)
            self.tik_instance.vector_dup(128, neg_one_ub[0], -1.0, (self.mini_batch * self.dim) // 128, 1, 8, 0)

            # 加载左矩阵数据：一次性将查询向量全部加载至L1中，形成matmul的左矩阵（四维），dtype采用int8!
            query_batch_cnt             = div_up(self.total_queries, self.query_num_each_compute)
            queries_align_each_compute  = div_up(self.query_num_each_compute, 16) * 16 
            queries_matrix_l1           = self.tik_instance.Tensor("int8", 
                                                                   (query_batch_cnt, 
                                                                   self.dim // 32, 
                                                                   queries_align_each_compute, 
                                                                   32), 
                                                                   name="queries_matrix_l1", 
                                                                   scope=tik.scope_cbuf)
            tmp_cnt_sc  = self.tik_instance.Scalar("uint32")
            with self.tik_instance.for_range(0, query_batch_cnt, thread_num=1) as bid:
                tmp_index   = bid * self.query_num_each_compute
                with self.tik_instance.if_scope(bid != (query_batch_cnt - 1)):
                    tmp_cnt_sc.set_as(self.query_num_each_compute)
                with self.tik_instance.else_scope():
                    tmp_cnt_sc.set_as(self.total_queries - (query_batch_cnt - 1) * self.query_num_each_compute)

                self.load_queries_from_gm_to_l1(queries_gm[tmp_index : (tmp_index + tmp_cnt_sc), :],
                                                tmp_index,
                                                tmp_cnt_sc, 
                                                pos_one_ub,
                                                neg_one_ub,
                                                queries_matrix_l1[bid : bid + 1, :, :, :])

            # 每次处理一个db_slice，每次db_slice_stride个，最后一次可能不足db_slice_stride个。
            slice_cnt               = div_up(db_vector_num, self.db_slice_stride)
            slice_vector_cnt_sc     = self.tik_instance.Scalar(dtype='uint32')
            with self.tik_instance.for_range(0, slice_cnt, thread_num=1) as slice_index:
                with self.tik_instance.if_scope(slice_index != slice_cnt - 1):
                    slice_vector_cnt_sc.set_as(self.db_slice_stride)
                with self.tik_instance.else_scope():
                    slice_vector_cnt_sc.set_as(db_vector_num - (slice_cnt - 1) * self.db_slice_stride)

                self.compute_distance_each_slice(queries_matrix_l1,
                                                 db_gm, 
                                                 distance_gm,
                                                 block_topk_gm,
                                                 db_num_sc,
                                                 db_vector_offset, 
                                                 slice_index, 
                                                 slice_vector_cnt_sc,
                                                 pos_one_ub, neg_one_ub)

    def compute_distance_each_slice(self, 
                                    queries_matrix_l1, db_gm, distance_gm, block_topk_gm,
                                    db_num_sc, db_vector_offset, 
                                    slice_index, db_slice_vector_cnt, 
                                    pos_one_ub, neg_one_ub):
        db_slice_vector_offset = db_vector_offset + slice_index * self.db_slice_stride
        
        # 定义Scalar变量用于存储每次计算对应的查询个数以及偏移位置        
        queries_num_sc  = self.tik_instance.Scalar(dtype='uint32', name='queries_num_sc', init_value=0)
        # 加载右矩阵，即底库向量
        db_slice_matrix_l1  = self.tik_instance.Tensor("int8",
                                                       (self.dim // 32, self.db_slice_stride, 32),
                                                       name="db_slice_matrix_l1",        
                                                       scope=tik.scope_cbuf)
        with self.tik_instance.for_range(0, 1):
            self.load_db_from_gm_to_l1(db_gm,
                                       db_slice_vector_offset, 
                                       db_slice_vector_cnt, 
                                       pos_one_ub, neg_one_ub, 
                                       db_slice_matrix_l1)

        
        # 定义matmul结果矩阵
        queries_align_each_compute = div_up(self.query_num_each_compute, 16) * 16
        distance_matrix_l0c = self.tik_instance.Tensor("int32", 
                                                       (self.db_slice_stride // 16, queries_align_each_compute, 16),
                                                       name="distance_matrix_l0c",
                                                       scope=tik.scope_cbuf_out)
        
        # 对查询进行切分，每次计算query_num_each_compute个查询。 
        query_batch_cnt     = div_up(self.total_queries, self.query_num_each_compute)
        queries_num_sc      = self.tik_instance.Scalar(dtype='uint32', name='queries_num_sc', init_value=0)
        queries_off_sc      = self.tik_instance.Scalar(dtype='uint32', name='queries_off_sc', init_value=0)
        with self.tik_instance.for_range(0, query_batch_cnt, thread_num=1) as bid:
            with self.tik_instance.if_scope(bid != (query_batch_cnt - 1)):
                queries_num_sc.set_as(self.query_num_each_compute)
            with self.tik_instance.else_scope():
                queries_num_sc.set_as(self.total_queries - (query_batch_cnt - 1) * self.query_num_each_compute)
            queries_off_sc.set_as(bid * self.query_num_each_compute)
            # 执行matmul
            self.tik_instance.matmul(distance_matrix_l0c, 
                                     queries_matrix_l1[bid : bid + 1, :, :, :], 
                                     db_slice_matrix_l1,
                                     queries_align_each_compute, 
                                     self.dim, 
                                     self.db_slice_stride)
            # 保存结果
            with self.tik_instance.for_range(0, 1):
                tmp_dist_start              = bid * self.query_num_each_compute * self.total_db_num
                tmp_dist_end                = tmp_dist_start + queries_num_sc * self.total_db_num
                
                block_topk_size_each_query  = div_up(self.total_db_num, self.block_size) * self.block_topk * 2
                tmp_block_topk_start        = bid * self.query_num_each_compute * block_topk_size_each_query
                tmp_block_topk_end          = tmp_block_topk_start + queries_num_sc * block_topk_size_each_query

                self.save_distance_from_l0c_to_gm(distance_gm[tmp_dist_start : tmp_dist_end],
                                                  block_topk_gm[tmp_block_topk_start : tmp_block_topk_end],
                                                  distance_matrix_l0c, 
                                                  db_slice_vector_offset, 
                                                  db_slice_vector_cnt,
                                                  db_num_sc,
                                                  queries_num_sc,
                                                  queries_off_sc)

    def load_queries_from_gm_to_l1(self, queries_gm, query_off, query_cnt, pos_one_ub, neg_one_ub, matrix_l1):
        """
        从GM中加载二值化短特征查询向量并转化为INT8矩阵至L1，形成左矩阵。
        Args:
            queries_gm:     二值化短特征查询向量Tensor：shape=(N, Dim//8), dtype=uint8
            query_off:      查询向量偏移
            query_cnt:      查询向量个数
            pos_one_ub:     全1.0 UB Tensor，外部初始化
            neg_one_ub:     全-1.0 UB Tensor，外部初始化
            matrix_l1:      l1矩阵，这里是左矩阵。
        """
        mini_batch_uint8_bit_ub     = self.tik_instance.Tensor("uint8",
                                                               (self.mini_batch, self.dim // 8),
                                                               name="mini_batch_uint8_bit_ub",
                                                               scope=tik.scope_ubuf)
        mini_batch_fp16_ub          = self.tik_instance.Tensor("float16",
                                                               (self.mini_batch, self.dim),
                                                               name="mini_batch_fp16_ub",
                                                               scope=tik.scope_ubuf)
        mini_batch_int8_ub          = self.tik_instance.Tensor("int8",
                                                               (self.mini_batch, self.dim),
                                                               name="mini_batch_int8_ub",
                                                               scope=tik.scope_ubuf)

        # 按照一次mini_batch个向量进行分批加载
        tmp_offset_sc   = self.tik_instance.Scalar(dtype="uint32", name="tmp_offset_sc", init_value=0)
        tmp_cnt_sc      = self.tik_instance.Scalar(dtype="uint32", name="tmp_cnt_sc", init_value=0)
        repeat          = div_up(query_cnt, self.mini_batch)
        with self.tik_instance.for_range(0, repeat) as rid:
            tmp_offset_sc.set_as(rid * self.mini_batch + query_off)
            with self.tik_instance.if_scope(rid != (repeat - 1)):
                tmp_cnt_sc.set_as(self.mini_batch)
            with self.tik_instance.else_scope():
                tmp_cnt_sc.set_as(query_cnt - (repeat - 1) * self.mini_batch)

            # 从GM中加载数据至mini_batch_uint8_bit_ub
            self.tik_instance.data_move(mini_batch_uint8_bit_ub[0:], 
                                        queries_gm[(tmp_offset_sc * self.dim // 8):],
                                        0,
                                        1, (tmp_cnt_sc * self.dim // 8) // 32, 0, 0)
            # 通过vsel指令组合成mini_batch_fp16_ub
            self.tik_instance.vsel(128, 2, 
                                   mini_batch_fp16_ub[0:], 
                                   mini_batch_uint8_bit_ub[0:], 
                                   pos_one_ub[0:], neg_one_ub[0:], 
                                   (self.mini_batch * self.dim) // 128, 
                                   1, 1, 1, 8, 8, 8)
            # 将fp16转换为int8
            self.tik_instance.vconv(128,
                                    "away-zero",
                                    mini_batch_int8_ub[0:],
                                    mini_batch_fp16_ub[0:],
                                    (self.mini_batch * self.dim) // 128,
                                    1, 1, 4, 8)

            # 加载向量至L1
            for i in range(self.dim // 32):
                self.tik_instance.data_move(matrix_l1[0, i, rid * self.mini_batch, 0], 
                                            mini_batch_int8_ub[0, i * 32], 
                                            0,
                                            tmp_cnt_sc, 1, 
                                            (self.dim // 32 - 1), 0)

    def load_db_from_gm_to_l1(self, db_gm, db_vector_offset, db_vector_cnt, pos_one_ub, neg_one_ub, matrix_l1):
        """
        从GM中加载二值化短底库向量并转化为INT8矩阵至L1，形成右矩阵。
        Args:
            db_gm:              二值化底库向量Tensor：shape=(M/H, Dim/32, H, 4), dtype=uint8
            db_vector_offset:   底库向量偏移
            db_vector_cnt:      底库向量个数，这里有一个约束：db_vector_cnt <= db_slice_stride
            pos_one_ub:         全1.0 UB Tensor，外部初始化
            neg_one_ub:         全-1.0 UB Tensor，外部初始化
            matrix_l1:          l1矩阵，这里是右矩阵。
        """
        tmp_uint8_bit_l1            = self.tik_instance.Tensor("uint8",
                                                               (self.db_slice_stride // self.db_zregion_h, 
                                                                self.dim // 32, 
                                                                self.db_zregion_h, 
                                                                4),
                                                               name="tmp_uint8_bit_l1",
                                                               scope=tik.scope_cbuf)
        mini_batch_uint8_bit_ub     = self.tik_instance.Tensor("uint8",
                                                               (self.dim // 32, self.mini_batch, 4),
                                                               name="mini_batch_uint8_bit_ub",
                                                               scope=tik.scope_ubuf)
        mini_batch_fp16_ub          = self.tik_instance.Tensor("float16",
                                                               (self.dim // 32, self.mini_batch, 32),
                                                               name="mini_batch_fp16_ub",
                                                               scope=tik.scope_ubuf)
        mini_batch_int8_ub          = self.tik_instance.Tensor("int8",
                                                               (self.dim // 32, self.mini_batch, 32),
                                                               name="mini_batch_int8_ub",
                                                               scope=tik.scope_ubuf)
        
        # 先将GM中的二值化底库特征向量加载至临时的L1变量中
        with self.tik_instance.if_scope(db_vector_cnt == self.db_slice_stride):
            self.tik_instance.data_move(tmp_uint8_bit_l1[0:],
                                        db_gm[db_vector_offset * self.dim // 8:],
                                        0,
                                        1, (db_vector_cnt * self.dim // 8) // 32, 
                                        0, 0)
        with self.tik_instance.else_scope():
            self.tik_instance.data_move(tmp_uint8_bit_l1[0:],
                                        db_gm[db_vector_offset * self.dim // 8:],
                                        0,
                                        db_vector_cnt // self.db_zregion_h, (self.db_zregion_h * self.dim // 8) // 32,
                                        0, 0)
        
        # 按照一次mini_batch个向量从上面的临时L1变量进行分批加载。
        # 由于mini_batch值与Z区域高度一致，所以这里每次加载一个Z区域，db_vector_cnt是mini_batch的倍数 !!!!!
        with self.tik_instance.for_range(0, db_vector_cnt // self.mini_batch) as rid:
            # 从GM中加载数据至mini_batch_uint8_bit_ub
            self.tik_instance.pipe_barrier("PIPE_ALL")
            self.tik_instance.data_move(mini_batch_uint8_bit_ub[0:], 
                                        tmp_uint8_bit_l1[(rid * self.mini_batch * self.dim // 8):],
                                        0,
                                        1, (self.mini_batch * self.dim // 8) // 32, 
                                        0, 0)
            # 通过vsel指令组合成mini_batch_fp16_ub
            self.tik_instance.vsel(128, 2, 
                                   mini_batch_fp16_ub[0:], 
                                   mini_batch_uint8_bit_ub[0:], 
                                   pos_one_ub[0:], neg_one_ub[0:], 
                                   (self.mini_batch * self.dim) // 128, 
                                   1, 1, 1, 8, 8, 8)

            # 将fp16转换为int8
            self.tik_instance.vconv(128,
                                    "away-zero",
                                    mini_batch_int8_ub[0:],
                                    mini_batch_fp16_ub[0:],
                                    self.mini_batch * self.dim // 128,
                                    1, 1, 4, 8)
            
            # 加载向量至L1
            self.tik_instance.data_move(matrix_l1[0, rid * self.mini_batch, 0],
                                        mini_batch_int8_ub[0, 0, 0],
                                        0,
                                        self.dim // 32, self.mini_batch,
                                        0, (self.db_slice_stride // self.mini_batch - 1) * self.mini_batch)
    
    def save_distance_from_l0c_to_gm(self, 
                                     distance_gm,
                                     block_topk_gm,
                                     distance_matrix_l0c, 
                                     db_vector_offset, 
                                     db_vector_cnt, 
                                     db_num, 
                                     queries_num,
                                     queries_off):
        """
        将L0C中的距离结果矩阵保存至GM中
        Args:
            distance_gm:            距离结果在GM中的Tensor，shape=(N*M,)，dtype=float16
            block_topk_gm:          块topk结果在GM中的Tensor，shape=(N*DIV_UP(M, block_size)*block_topk*2,)，dtype=float16
            distance_matrix_l0c:    距离结果在L0C中的矩阵，shape=(db_slice_stride/16, queries_align, 16)，dtype=int32
            db_vector_offset:       底库向量起始位置
            db_vector_cnt:          底库向量个数
            db_num:                 底库向量总个数
            queries_num:            查询的实际个数
            queries_off:            查询的偏移量
        """
        queries_num_align = distance_matrix_l0c.shape[1] 
        
        # Step1: 将distance_matrix_l0c结果一次性转移至UB中，并且完成int32->fp16的转换。
        distance_result_3d_ub = self.tik_instance.Tensor("float16",
                                                         (self.db_slice_stride // 16, queries_num_align, 16),
                                                         name="distance_result_3d_ub",
                                                         scope=tik.scope_ubuf)
        self.tik_instance.tensor_mov(distance_result_3d_ub[0, 0, 0],
                                     distance_matrix_l0c[0, 0, 0],
                                     "m",
                                     1,
                                     (queries_num_align // 16) * (self.db_slice_stride // 16),
                                     0, 0,
                                     deqscale=1.0)

        # Step2: 将distance_result_3d_ub中的结果分批完成三维到二维的转换，并写入GM中。
        tmp_off_sc      = self.tik_instance.Scalar(dtype='uint32', name='tmp_off_sc', init_value=0)
        tmp_cnt_sc      = self.tik_instance.Scalar(dtype='uint32', name='tmp_cnt_sc', init_value=0)
        repeat          = div_up(queries_num, self.query_num_each_save)
        with self.tik_instance.for_range(0, repeat, thread_num=1) as rid:
            tmp_off_sc.set_as(rid * self.query_num_each_save)
            with self.tik_instance.if_scope(rid != repeat - 1):
                tmp_cnt_sc.set_as(self.query_num_each_save)
            with self.tik_instance.else_scope():
                tmp_cnt_sc.set_as(queries_num - (repeat - 1) * self.query_num_each_save)
            
            # 定义距离结果的二维Tensor, UB中分配，数据类型为FP16，不必担心数据溢出。    
            distance_result_2d_ub = self.tik_instance.Tensor("float16", 
                                                             (self.query_num_each_save, self.db_slice_stride),
                                                             name="distance_result_2d_ub",
                                                             scope=tik.scope_ubuf)
            # 三维Tensor转二维Tensor
            for k in range(self.db_slice_stride // 16):
                self.tik_instance.data_move(distance_result_2d_ub[0, k * 16], 
                                            distance_result_3d_ub[k, rid * self.query_num_each_save, 0], 
                                            0, 
                                            self.query_num_each_save, 1, 
                                            0, (self.db_slice_stride // 16 - 1) * 1)
            # 根据mask值对距离结果进行过滤（可选步骤）
            if self.use_distance_mask is True:
                distance_mask_ub = self.tik_instance.Tensor("uint8",
                                                            (self.query_num_each_save, self.db_slice_stride // 8),
                                                            name="distance_mask_ub",
                                                            scope=tik.scope_ubuf)
                if self.share_distance_mask is not True:
                    self.tik_instance.data_move(distance_mask_ub[0, 0],
                        self.distance_mask_gm[queries_off + rid * self.query_num_each_save, db_vector_offset // 8],
                        0,
                        tmp_cnt_sc, (self.db_slice_stride // 8) // 32,
                        ((self.total_db_num - self.db_slice_stride) // 8) // 32, 0)
                else:
                    self.tik_instance.data_move(distance_mask_ub[0, 0],
                                                self.distance_mask_gm[0, db_vector_offset // 8],
                                                0,
                                                1, (self.db_slice_stride // 8) // 32,
                                                0, 0)
                    for cid in range(self.query_num_each_save - 1):
                        self.tik_instance.data_move(distance_mask_ub[cid + 1, 0],
                                                    distance_mask_ub[0, 0],
                                                    0,
                                                    1, (self.db_slice_stride // 8) // 32,
                                                    0, 0)

                # 按照mask，执行过滤。
                filter_distance_using_mask_in_ub(self.tik_instance, 
                                                 distance_result_2d_ub[:],
                                                 tmp_cnt_sc * self.db_slice_stride,
                                                 distance_result_2d_ub[:],
                                                 distance_mask_ub[:],
                                                 MIN_FP16)             

            # 扫描每个块，生成每个块的topk数据。
            block_topk_ub = self.tik_instance.Tensor("float16", 
                                                     (self.query_num_each_save, 
                                                     self.blocks_each_db_slice * self.block_topk * 2), 
                                                     name="block_topk_ub", 
                                                     scope=tik.scope_ubuf)
            with self.tik_instance.for_range(0, 1):
                db_vector_real_cnt = self.tik_instance.Scalar("uint32")
                db_vector_real_cnt.set_as(self.db_slice_stride)
                with self.tik_instance.if_scope((db_num - db_vector_offset) < self.db_slice_stride):
                    db_vector_real_cnt.set_as(db_num - db_vector_offset)
                
                with self.tik_instance.if_scope(db_vector_real_cnt == self.db_slice_stride):
                    scan_block_topk_in_ub(self.tik_instance, 
                                          distance_result_2d_ub[0:], 
                                          self.query_num_each_save * self.db_slice_stride,
                                          block_topk_ub[0:],
                                          self.block_size)
                with self.tik_instance.else_scope():
                    with self.tik_instance.for_range(0, self.query_num_each_save) as qid:
                        scan_block_topk_in_ub(self.tik_instance,
                                              distance_result_2d_ub[qid * self.db_slice_stride:],
                                              db_vector_real_cnt,
                                              block_topk_ub[qid * self.blocks_each_db_slice * 2:],
                                              self.block_size)
                        
            tmp_dist_gm_offset          = tmp_off_sc * self.total_db_num + db_vector_offset
            tmp_block_topk_gm_offset    = div_up(tmp_dist_gm_offset, self.block_size) * self.block_topk * 2
            # 距离结果写入GM
            self.tik_instance.data_move(distance_gm[tmp_dist_gm_offset], 
                                        distance_result_2d_ub[0, 0], 
                                        0, 
                                        tmp_cnt_sc, self.db_slice_stride * 2 // 32, 
                                        0, (self.total_db_num - self.db_slice_stride) * 2 // 32)
            
            # block topk结果写入GM
            tmp_block_stride = div_up(self.total_db_num - self.db_slice_stride, self.block_size)
            self.tik_instance.data_move(block_topk_gm[tmp_block_topk_gm_offset],
                                        block_topk_ub[0, 0],
                                        0,
                                        tmp_cnt_sc, self.blocks_each_db_slice * 2 * 2 // 32,
                                        0, 
                                        tmp_block_stride * self.block_topk * 2 * 2 // 32)            
