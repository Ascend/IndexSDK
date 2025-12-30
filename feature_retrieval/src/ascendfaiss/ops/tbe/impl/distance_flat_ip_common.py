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
from enum import Enum
from te import tik
from mxIndex_impl.common import set_soc_info


AI_CORE_NUM     = 8
# 由于FP16的精度，在向量距离计算时，距离值可能接近1.0，转换为下标时会超过映射表长度。
# 外部程序调用算子时，实际表的长度会比传入算子时对应的Table Tensor长度多TABLE_EXTRA_LEN。
# 计算下标时会以实际表的长度减去TABLE_EXTRA_LEN作为(1+cos)/*l公式中的l值。
TABLE_EXTRA_LEN = 48


def div_up(a, b):
    return ((a+b-1)//b)


# 查表函数
def lookup_table_in_ub(tik_instance, cos_ub, table_ub, table_len, res_ub):
    num = cos_ub.shape[0]
    if not(num % 64 == 0 or (num < 64 and num % 8 == 0)):
        raise RuntimeError("lookup_table_in_ub require num of elements is multiple of 64 "
                           "or less than 64 and multiple of 8")
    mask = 64
    repeat = num//64
    if num < 64:
        mask = num
        repeat = 1

    # 每个元素+1
    cos_ub1 = tik_instance.Tensor("float32", (num,), name="cos_ub1", scope=tik.scope_ubuf)
    tik_instance.vadds(mask, cos_ub1[0], cos_ub[0], 1.0, repeat, 1, 1, 8, 8)
    # 每个元素乘以l/2
    cos_ub2 = tik_instance.Tensor("float32", (num,), name="cos_ub2", scope=tik.scope_ubuf)
    tik_instance.vmuls(mask, cos_ub2[0], cos_ub1[0], (table_len-TABLE_EXTRA_LEN)/2.0, repeat, 1, 1, 8, 8)
    
    # 精度转换：float32->int32
    cos_ub3 = tik_instance.Tensor("int32", (num,), name="cos_ub3", scope=tik.scope_ubuf)
    tik_instance.vconv(mask, "away-zero", cos_ub3[0], cos_ub2[0], repeat, 1, 1, 8, 8)
        
    # 根据下标生成地址偏移
    cos_ub4 = tik_instance.Tensor("int32", (num,), name="cos_ub4", scope=tik.scope_ubuf)
    tik_instance.vmuls(mask, cos_ub4[0], cos_ub3[0], 4, repeat, 1, 1, 8, 8)
    # 使用vgather指令进行离散读取然后顺序排列
    cos_ub5 = tik_instance.Tensor("float32", (num,), name="cos_ub5", scope=tik.scope_ubuf)
    tik_instance.vgather(mask, cos_ub5[0], table_ub[0], cos_ub4[0], repeat, 8, 0, 0)
    # 拷贝结果至res_ub中
    tik_instance.data_move(res_ub[0], cos_ub5[0], 0, 1, num*4//32, 0, 0)


def distance_filter_in_ub(tik_instance, 
                          dist_ub, real_dist_cnt, thresh_ub, idx_ub, 
                          dist_filter_ub, idx_filter_ub, cnt_filter_sc):
    """
    距离过滤函数
    Args:
        tik_instance:   tik实例
        dist_ub:        距离UB变量
        real_dist_cnt:  实际的距离个数
        thresh_ub:      距离阈值UB变量，由外部初始化。
        idx_ub:         距离值对应的序号，UB变量，由外部初始化。
        dist_filter_ub: 过滤后的距离UB变量，由外部申请变量空间。
        idx_filter_ub:  过滤后的距离值对应的序号UB变量，由外部申请变量空间。        
        cnt_filter_sc:  过滤后满足阈值的距离个数，scalar变量，由外部申请变量空间。
    """
    dist_num = dist_ub.shape[0]
    if dist_num % 64 != 0:
        raise RuntimeError("distance_filter_in_ub require num of elements is multiple of 64 ")
    
    with tik_instance.for_range(0, 1):
        # 在UB中分配存放阈值比较后的bit位结果值
        cmp_res_bit_ub = tik_instance.Tensor("uint32", 
                                             (dist_num // 32,), 
                                             name="cmp_res_bit_ub", 
                                             scope=tik.scope_ubuf)

        repeat  = div_up(real_dist_cnt, 64)
        mask    = real_dist_cnt
        # 执行比较操作
        tik_instance.vcmpv_gt(cmp_res_bit_ub, dist_ub, thresh_ub, repeat, 1, 1, 8, 8)
        # 执行reduce操作，针对距离值
        tik_instance.vreduce(mask, 
                             dist_filter_ub, dist_ub, 
                             cmp_res_bit_ub, 
                             repeat, 1, 8, 1, 0, cnt_filter_sc, "counter")
        # 执行reduce操作，针对序号下标
        tik_instance.vreduce(mask,
                             idx_filter_ub, idx_ub,
                             cmp_res_bit_ub,
                             repeat, 1, 8, 1, 0, None, "counter")


class DistanceFlatIp:
    def __init__(self, input_queries, input_db, db_num, output_distance, kernel_name):
        self.shape_queries                  = input_queries.get("shape")
        self.dtype_queries                  = input_queries.get("dtype")

        self.shape_db                       = input_db.get("shape")
        self.dtype_db                       = input_db.get("dtype")
        # Z区域高度，即Z区域包含的向量个数。
        self.db_zregion_h                   = self.shape_db[2]
        
        self.shape_db_num                   = db_num.get("shape")
        self.dtype_db_num                   = db_num.get("dtype")
        
        self.shape_distance                 = output_distance.get("shape")
        self.dtype_distance                 = output_distance.get("dtype")

        self.kernel_name                    = kernel_name
        
        self.total_queries                  = self.shape_queries[0]
        self.dim                            = self.shape_queries[1]

        # 每次执行matmul指令时右矩阵的跨度，即一次做多少个底库向量的距离计算，16的倍数。
        self.db_slice_stride                = 384
        # 每次计算距离对应的查询条数
        self.query_num_each_compute         = div_up(self.total_queries, 16) * 16
        # 保存距离结果时按照一次多少条查询对应的距离值进行保存。
        self.query_num_each_save            = 4
    
        self.core_num                       = AI_CORE_NUM 
        # 映射表的初始化参数，映射表是可选的。
        self.need_lookup                    = False
        self.table_gm                       = None
        self.shape_table                    = None
        self.dtype_table                    = None
        self.table_len                      = 0

        self.tik_instance           = tik.Tik(tik.Dprofile("v200", "aic"), disable_debug=False)
        # 输入1：查询向量
        self.queries_gm             = self.tik_instance.Tensor(self.dtype_queries, 
                                                               self.shape_queries, 
                                                               name="queries_gm", 
                                                               scope=tik.scope_gm)
        # 输入2：底库向量
        self.db_gm                  = self.tik_instance.Tensor(self.dtype_db, 
                                                               self.shape_db, 
                                                               name="db_gm", 
                                                               scope=tik.scope_gm)
        # 输入3：实际的底库向量个数
        self.db_num_gm              = self.tik_instance.Tensor(self.dtype_db_num, 
                                                               self.shape_db_num, 
                                                               name="db_num_gm", 
                                                               scope=tik.scope_gm)
        # 输出1：距离值结果
        self.distance_gm    = self.tik_instance.Tensor(self.dtype_distance, 
                                                       self.shape_distance, 
                                                       name="distance_gm", 
                                                       scope=tik.scope_gm)

    def set_lookup_table(self, lookup_table):
        self.shape_table                    = lookup_table.get("shape")
        self.dtype_table                    = lookup_table.get("dtype")
        self.table_len                      = self.shape_table[0]       # 映射表的实际长度
        self.need_lookup                    = True 

    def set_optimize_policy(self):
        if self.total_queries == 64:
            if self.need_lookup is not True:
                self.db_slice_stride        = 384
                self.query_num_each_compute = 64
                self.query_num_each_save    = 64
            else:
                self.db_slice_stride        = 384
                self.query_num_each_compute = 64
                self.query_num_each_save    = 8
        if self.total_queries == 128:
            if self.need_lookup is not True:
                self.db_slice_stride        = 384 if self.dim == 1024 else 512
                self.query_num_each_compute = 64
                self.query_num_each_save    = 64
            else:
                self.db_slice_stride        = 384
                self.query_num_each_compute = 64
                self.query_num_each_save    = 4
        if self.total_queries == 256:
            if self.need_lookup is not True:
                self.db_slice_stride        = 256 if self.dim == 1024 else 512
                self.query_num_each_compute = 64
                self.query_num_each_save    = 64
            else:
                self.db_slice_stride        = 256 if self.dim == 1024 else 384
                self.query_num_each_compute = 64
                self.query_num_each_save    = 4

        if self.need_lookup is True:
            self.core_num               = AI_CORE_NUM * 2   # 带映射表时，设置为2倍性能更优。

    def check_settings(self):
        if self.dim % 16 != 0:
            raise RuntimeError("dim must be multiple of 16!")
        if self.db_slice_stride % self.db_zregion_h != 0:
            raise RuntimeError("db_slice_stride must be multiple of z region height!")
        if (self.query_num_each_compute % 16 != 0):
            raise RuntimeError("query_num_each_compute must be multiple of 16!")
        if self.query_num_each_compute % self.query_num_each_save != 0:
            raise RuntimeError("query_num_each_compute must be multiple of query_num_each_save!")

    def get_tik_instance(self):
        # 设置优化策略
        self.set_optimize_policy()
        # 参数检查
        self.check_settings()
        
        if self.need_lookup is True:
            self.table_gm   = self.tik_instance.Tensor(self.dtype_table, 
                                                       self.shape_table, 
                                                       name="table_gm", 
                                                       scope=tik.scope_gm)

        with self.tik_instance.for_range(0, self.core_num, block_num=self.core_num) as tid:
            self.execute_task(tid)
        
        if self.need_lookup is True:
            inputs  = [self.queries_gm, self.db_gm, self.db_num_gm, self.table_gm]
            outputs = [self.distance_gm]
        else:
            inputs  = [self.queries_gm, self.db_gm, self.db_num_gm]
            outputs = [self.distance_gm]
        
        self.tik_instance.BuildCCE(self.kernel_name,
                                   inputs, 
                                   outputs)
        
        return self.tik_instance
       
    def execute_task(self, task_id):
        """
        执行任务函数：第一个任务比较特殊，它用于处理第一块底库数据以及整除剩余的底库向量。
        """
        # 获取底库向量的实际条数
        db_num_sc = self.tik_instance.Scalar(dtype="uint32", name="db_num_sc", init_value=0)
        with self.tik_instance.for_range(0, 1):
            db_num_ub = self.tik_instance.Tensor(self.dtype_db_num, 
                                                 self.shape_db_num, 
                                                 name="db_num_ub", 
                                                 scope=tik.scope_ubuf)
            self.tik_instance.data_move(db_num_ub[0], 
                                        self.db_num_gm[0], 
                                        0, 
                                        1, self.shape_db_num[0] * 4 // 32, 
                                        0, 0)
            db_num_sc.set_as(db_num_ub[0])
        # 如果需要查找映射表，则在每个任务中先将映射表加载至UB中。
        table_each_core_ub = None
        if self.need_lookup is True:
            table_len_align     = div_up(self.table_len, 8)*8
            table_each_core_ub  = self.tik_instance.Tensor(self.dtype_table, 
                                                           (table_len_align,), 
                                                           name="table_each_core_ub", 
                                                           scope=tik.scope_ubuf)
            self.tik_instance.data_move(table_each_core_ub[0], 
                                        self.table_gm[0], 
                                        0, 
                                        1, table_len_align * 4 // 32, 
                                        0, 0)

        # 计算每个任务对应的底库向量起始位置以及底库向量个数
        db_slice_num_align = db_num_sc // self.db_slice_stride
        # 简单考虑，将底库数量地板除以AI Core数量，即为每个任务需要处理的底库向量个数，剩余的底库向量放在第一个任务中。
        db_vector_num_each_task         = (db_slice_num_align // self.core_num) * self.db_slice_stride
        db_vector_offset_each_task      = task_id*db_vector_num_each_task
        db_vector_num_last              = db_num_sc-db_vector_num_each_task*self.core_num
        db_vector_offset_last           = db_vector_num_each_task*self.core_num

        with self.tik_instance.if_scope(task_id != 0):
            self.compute_distance_each_task(self.queries_gm, db_vector_offset_each_task, 
                                            db_vector_num_each_task, self.db_gm, db_num_sc, 
                                            self.distance_gm, table_each_core_ub)
        with self.tik_instance.else_scope():
            # 这里做了一个特殊处理：对于第一个任务先计算后面剩余的底库向量再处理第一块底库向量，
            # 这样就不会因为底库向量个数不整除db_slice_stride而引起的距离计算结果覆盖的问题了。
            self.compute_distance_each_task(self.queries_gm, db_vector_offset_last, 
                                            db_vector_num_last, self.db_gm, db_num_sc, 
                                            self.distance_gm, table_each_core_ub)
            self.compute_distance_each_task(self.queries_gm, db_vector_offset_each_task, 
                                            db_vector_num_each_task, self.db_gm, db_num_sc, 
                                            self.distance_gm, table_each_core_ub)
        
    def compute_distance_each_task(self, queries_gm, db_vector_offset, 
                                   db_vector_num, db_gm, db_num_sc,
                                   distance_gm, table_each_core_ub):
        with self.tik_instance.if_scope(db_vector_num != 0):  
            # 加载左矩阵数据：一次性将查询向量全部加载至L1中，形成matmul的左矩阵（四维），dtype采用float16!
            query_batch_cnt             = div_up(self.total_queries, self.query_num_each_compute)
            queries_matrix_l1           = self.tik_instance.Tensor("float16", 
                                                                   (query_batch_cnt, 
                                                                   self.dim // 16, 
                                                                   self.query_num_each_compute, 
                                                                   16), 
                                                                   name="queries_matrix_l1", 
                                                                   scope=tik.scope_cbuf)
            tmp_cnt_sc  = self.tik_instance.Scalar("uint32")
            with self.tik_instance.for_range(0, query_batch_cnt, thread_num=1) as bid:
                tmp_index   = bid * self.query_num_each_compute
                with self.tik_instance.if_scope(bid != (query_batch_cnt - 1)):
                    tmp_cnt_sc.set_as(self.query_num_each_compute)
                with self.tik_instance.else_scope():
                    tmp_cnt_sc.set_as(self.total_queries - (query_batch_cnt - 1) * self.query_num_each_compute)

                self.load_queries_from_gm_to_l1(queries_gm,
                                                tmp_index,
                                                tmp_cnt_sc, 
                                                queries_matrix_l1[bid : bid + 1, :, :, :])
            # 以下是性能最佳写法，另一种实现方式是将下面的两次调用合在一起，并使用Scalar变量记录每次处理的数据个数。
            # 每次处理一个db_slice，每次db_slice_stride个。
            with self.tik_instance.for_range(0, db_vector_num // self.db_slice_stride, thread_num=1) as slice_index:
                self.compute_distance_each_slice(queries_matrix_l1,
                                                 db_gm,
                                                 distance_gm,
                                                 db_num_sc,
                                                 db_vector_offset,
                                                 slice_index,
                                                 self.db_slice_stride,
                                                 table_each_core_ub)
            # 最后一次可能不足db_slice_stride个
            db_vector_remain = db_vector_num % self.db_slice_stride
            with self.tik_instance.if_scope(db_vector_remain > 0):
                self.compute_distance_each_slice(queries_matrix_l1,
                                                 db_gm,
                                                 distance_gm,
                                                 db_num_sc,
                                                 db_vector_offset,
                                                 db_vector_num // self.db_slice_stride,
                                                 db_vector_remain,
                                                 table_each_core_ub)

    def compute_distance_each_slice(self, 
                                    queries_matrix_l1, 
                                    db_gm, 
                                    distance_gm, 
                                    db_num_sc, 
                                    db_vector_offset, 
                                    slice_index, 
                                    db_slice_vector_cnt,
                                    table_each_core_ub):
        db_slice_vector_offset = db_vector_offset + slice_index * self.db_slice_stride
        # 定义Scalar变量用于存储每次计算对应的查询个数        
        queries_num_sc  = self.tik_instance.Scalar(dtype='uint32', name='queries_num_sc', init_value=0)
        # 定义matmul右矩阵，即底库向量
        db_slice_matrix_l1  = self.tik_instance.Tensor("float16",
                                                       (self.dim // 16, self.db_slice_stride, 16),
                                                       name="db_slice_matrix_l1",        
                                                       scope=tik.scope_cbuf)
        # 定义matmul结果矩阵
        distance_matrix_l0c = self.tik_instance.Tensor("float32", 
                                                       (self.db_slice_stride // 16, self.query_num_each_compute, 16),
                                                       name="distance_matrix_l0c",
                                                       scope=tik.scope_cbuf_out)
        # 加载右矩阵
        with self.tik_instance.for_range(0, 1):
            self.load_db_from_gm_to_l1(db_gm,
                                       db_slice_vector_offset, 
                                       db_slice_vector_cnt, 
                                       db_slice_matrix_l1)
        
        # 对查询进行切分，每次计算query_num_each_compute个查询。 
        query_batch_cnt     = div_up(self.total_queries, self.query_num_each_compute)
        queries_num_sc      = self.tik_instance.Scalar(dtype='uint32', name='queries_num_sc', init_value=0)
        with self.tik_instance.for_range(0, query_batch_cnt, thread_num=1) as bid:
            with self.tik_instance.if_scope(bid != (query_batch_cnt - 1)):
                queries_num_sc.set_as(self.query_num_each_compute)
            with self.tik_instance.else_scope():
                queries_num_sc.set_as(self.total_queries - (query_batch_cnt - 1) * self.query_num_each_compute)
            # 执行matmul
            self.tik_instance.matmul(distance_matrix_l0c, 
                                     queries_matrix_l1[bid : bid + 1, :, :, :], 
                                     db_slice_matrix_l1,
                                     self.query_num_each_compute, 
                                     self.dim, 
                                     self.db_slice_stride)
            # 保存结果
            with self.tik_instance.for_range(0, 1):
                tmp_dist_start              = bid * self.query_num_each_compute * db_num_sc
                tmp_dist_end                = tmp_dist_start + queries_num_sc * db_num_sc
                
                self.save_distance_from_l0c_to_gm(distance_gm[tmp_dist_start : tmp_dist_end],
                                                  distance_matrix_l0c, 
                                                  db_slice_vector_offset, 
                                                  db_slice_vector_cnt,
                                                  db_num_sc,
                                                  queries_num_sc,
                                                  table_each_core_ub)

    def load_queries_from_gm_to_l1(self, queries_gm, query_off, query_cnt, matrix_l1):
        """
        从GM中加载查询向量至L1，形成左矩阵。
        Args:
            queries_gm:     查询向量Tensor：shape=(N, Dim), dtype=float16
            query_off:      查询向量偏移
            query_cnt:      查询向量个数
            matrix_l1:      l1矩阵，这里是左矩阵，四维，多出的一维是batch。
        """
        # 加载向量至L1
        for d in range(self.dim // 16):
            self.tik_instance.data_move(matrix_l1[0, d, 0, 0],
                                        queries_gm[query_off, d * 16], 
                                        0,
                                        query_cnt, 1,
                                        (self.dim // 16 - 1), 0)

    def load_db_from_gm_to_l1(self, db_gm, db_vector_off, db_vector_cnt, matrix_l1, method=0):
        """
        从GM中加载底库向量至L1，形成右矩阵。
        Args:
            db_gm:              底库向量Tensor：shape=(M/H, Dim/16, H, 16), dtype=float16
            db_vector_off:      底库向量偏移
            db_vector_cnt:      底库向量个数，这里有一个约束：db_vector_cnt <= db_slice_stride
            matrix_l1:          l1矩阵，这里是右矩阵，三维。
            method:             加载的方法选择，根据实际调试结果选择。
        """
        if method == 0:
            # 按照维度分段方式进行加载 
            for d in range(self.dim // 16):
                nburst      = db_vector_cnt // self.db_zregion_h
                burst_len   = (self.db_zregion_h * 16 * 2) // 32
                with self.tik_instance.if_scope(nburst != 0):
                    self.tik_instance.data_move(matrix_l1[d, 0, 0],
                                                db_gm[db_vector_off // self.db_zregion_h, d, 0, 0],
                                                0,
                                                nburst,
                                                burst_len,
                                                (self.dim // 16 - 1) * self.db_zregion_h,
                                                0)
                remain  = db_vector_cnt % self.db_zregion_h
                with self.tik_instance.if_scope(remain != 0):
                    self.tik_instance.data_move(matrix_l1[d, nburst * self.db_zregion_h, 0],
                                                db_gm[(db_vector_off + db_vector_cnt) // self.db_zregion_h, d, 0, 0],
                                                0,
                                                1,
                                                remain,
                                                0, 0)
        elif method == 1:
            # 按照每次加载一个Z区域方式，这样读GM时是连续的。
            matrix_l1_stride    = matrix_l1.shape[1]    
            repeat              = div_up(db_vector_cnt, self.db_zregion_h)
            tmp_cnt_sc          = self.tik_instance.Scalar(dtype='uint32')
            with self.tik_instance.for_range(0, repeat) as rid:
                with self.tik_instance.if_scope(rid != (repeat - 1)):
                    tmp_cnt_sc.set_as(self.db_zregion_h)
                with self.tik_instance.else_scope():
                    tmp_cnt_sc.set_as(db_vector_cnt - (repeat - 1) * self.db_zregion_h)

                nburst      = self.dim // 16
                burst_len   = (tmp_cnt_sc * 16 * 2) // 32

                self.tik_instance.data_move(matrix_l1[0, rid * self.db_zregion_h, 0],
                                            db_gm[db_vector_off // self.db_zregion_h + rid, 0, 0, 0],
                                            0,
                                            nburst, 
                                            burst_len,
                                            0, 
                                            (matrix_l1_stride - self.db_zregion_h) + (self.db_zregion_h - tmp_cnt_sc))
        else:
            raise RuntimeError("load_db_from_gm_to_l1: unsupported load method!")
    
    def save_distance_from_l0c_to_gm(self, 
                                     distance_gm,
                                     distance_matrix_l0c, 
                                     db_vector_offset, 
                                     db_vector_cnt, 
                                     db_num, 
                                     queries_num,
                                     table_each_core_ub=None):
        """
        将L0C中的距离结果矩阵保存至GM中
        Args:
            distance_gm:                    距离结果在GM中的Tensor，shape=(N*M,)，dtype=float16
            distance_matrix_l0c:            距离结果在L0C中的矩阵，shape=(db_slice_stride/16, queries_align, 16)，dtype=float32
            db_vector_offset:               底库向量起始位置
            db_vector_cnt:                  底库向量个数
            db_num:                         底库向量实际总个数
            queries_num:                    查询的实际个数
            table_each_core_ub:             映射表，可选。
        """
        queries_num_align = distance_matrix_l0c.shape[1] 
        # Step1: 将distance_matrix_l0c结果一次性转移至UB中，并从三维转换为二维。
        distance_result_2d_ub = self.tik_instance.Tensor("float32",
                                                         (queries_num_align, self.db_slice_stride),
                                                         name="distance_result_2d_ub",
                                                         scope=tik.scope_ubuf)
        with self.tik_instance.for_range(0, 1):
            distance_result_3d_ub = self.tik_instance.Tensor("float32",
                                                             (self.db_slice_stride // 16, queries_num_align, 16),
                                                             name="distance_result_3d_ub",
                                                             scope=tik.scope_ubuf)
            # L0C(3d) -> UB(3d)
            self.tik_instance.data_move(distance_result_3d_ub[0, 0, 0],
                                        distance_matrix_l0c[0, 0, 0],
                                        0,
                                        1, queries_num_align * self.db_slice_stride * 4 // 1024,
                                        0, 0)
            # UB(3d) -> UB(2d)
            for k in range(self.db_slice_stride // 16):
                self.tik_instance.data_move(distance_result_2d_ub[0, k * 16],
                                            distance_result_3d_ub[k, 0, 0],
                                            0, 
                                            queries_num_align, 2,
                                            0, (self.db_slice_stride // 16 - 1) * 2)        

        # Step2: 将distance_result_2d_ub中的结果分批写入GM中，分为有无映射表两种情况。
        repeat      = queries_num // self.query_num_each_save
        last_remain = queries_num % self.query_num_each_save
        if self.need_lookup is not True:
            with self.tik_instance.for_range(0, repeat) as rid:
                query_start = rid * self.query_num_each_save
                with self.tik_instance.for_range(0, self.query_num_each_save) as k:
                    tmp_gm_offset = (query_start + k) * db_num + db_vector_offset
                    self.tik_instance.data_move(distance_gm[tmp_gm_offset],
                                                distance_result_2d_ub[query_start + k, 0],
                                                0,
                                                1, db_vector_cnt * 4 // 32,
                                                0, 0)
            with self.tik_instance.if_scope(last_remain > 0):
                with self.tik_instance.for_range(0, last_remain) as k:
                    tmp_gm_offset = (repeat * self.query_num_each_save + k) * db_num + db_vector_offset
                    self.tik_instance.data_move(distance_gm[tmp_gm_offset],
                                                distance_result_2d_ub[repeat * self.query_num_each_save + k, 0],
                                                0, 
                                                1, db_vector_cnt * 4 // 32,
                                                0, 0)
        else:
            lookup_result_2d_ub = self.tik_instance.Tensor(self.dtype_table,
                                                           (self.query_num_each_save, self.db_slice_stride),
                                                           name="lookup_result_2d_ub",
                                                           scope=tik.scope_ubuf)
            with self.tik_instance.for_range(0, repeat) as rid:
                query_start = rid * self.query_num_each_save
                tmp_start   = rid * self.query_num_each_save * self.db_slice_stride
                tmp_end     = tmp_start + self.query_num_each_save * self.db_slice_stride
                # 查表
                with self.tik_instance.for_range(0, 1):
                    lookup_table_in_ub(self.tik_instance,
                                       distance_result_2d_ub[tmp_start : tmp_end],
                                       table_each_core_ub, self.table_len,
                                       lookup_result_2d_ub[:])
                # 保存
                with self.tik_instance.for_range(0, self.query_num_each_save) as k:
                    tmp_gm_offset = (query_start + k) * db_num + db_vector_offset
                    self.tik_instance.data_move(distance_gm[tmp_gm_offset],
                                                lookup_result_2d_ub[k, 0],
                                                0,
                                                1, db_vector_cnt * 4 // 32,
                                                0, 0)
            with self.tik_instance.if_scope(last_remain > 0):
                tmp_start   = repeat * self.query_num_each_save * self.db_slice_stride
                tmp_end     = tmp_start + self.query_num_each_save * self.db_slice_stride
                # 查表
                with self.tik_instance.for_range(0, 1):
                    lookup_table_in_ub(self.tik_instance,
                                       distance_result_2d_ub[tmp_start : tmp_end],
                                       table_each_core_ub, self.table_len,
                                       lookup_result_2d_ub[:])
                # 保存
                with self.tik_instance.for_range(0, last_remain) as k:
                    tmp_gm_offset = (repeat * self.query_num_each_save + k) * db_num + db_vector_offset
                    self.tik_instance.data_move(distance_gm[tmp_gm_offset],
                                                lookup_result_2d_ub[k, 0],
                                                0,
                                                1, db_vector_cnt * 4 // 32,
                                                0, 0)           


class DistanceFlatIpByIdx:
    def __init__(self, input_queries, input_index, input_index_num, input_db, output_distance, kernel_name):
        self.shape_queries                  = input_queries.get("shape")
        self.dtype_queries                  = input_queries.get("dtype")

        self.shape_index                    = input_index.get("shape")
        self.dtype_index                    = input_index.get("dtype")

        self.shape_index_num                = input_index_num.get("shape")
        self.dtype_index_num                = input_index_num.get("dtype")
        
        self.shape_db                       = input_db.get("shape")
        self.dtype_db                       = input_db.get("dtype")
        
        self.shape_distance                 = output_distance.get("shape")
        self.dtype_distance                 = output_distance.get("dtype")

        self.kernel_name                    = kernel_name

        self.total_queries, self.dim        = self.shape_queries
        self.index_num_each_query           = self.shape_index[1]

        self.need_lookup                    = False
        self.shape_table                    = None
        self.dtype_table                    = None
        self.table_len                      = 0
        
        # 底库中小z大Z区域的高度，根据场景由外部调用者确定合适的值。
        self.db_zregion_h                   = self.shape_db[2]     
        self.core_num                       = AI_CORE_NUM

        self.tik_instance                   = tik.Tik(tik.Dprofile("v200", "aic"), disable_debug=False)
        
        self.queries_gm     = self.tik_instance.Tensor(self.dtype_queries, self.shape_queries, 
                                                       name="queries_gm", scope=tik.scope_gm)
        self.index_gm       = self.tik_instance.Tensor(self.dtype_index, self.shape_index, 
                                                       name="index_gm", scope=tik.scope_gm)
        self.index_num_gm   = self.tik_instance.Tensor(self.dtype_index_num, self.shape_index_num, 
                                                       name="index_num_gm", scope=tik.scope_gm)
        self.db_gm          = self.tik_instance.Tensor(self.dtype_db, self.shape_db, 
                                                       name="db_gm", scope=tik.scope_gm)
        self.distance_gm    = self.tik_instance.Tensor(self.dtype_distance, self.shape_distance, 
                                                       name="distance_gm", scope=tik.scope_gm)
        self.table_gm       = None

        if self.index_num_each_query % 16 != 0:
            raise RuntimeError("number of index for each query must be multiple of 16")

    def set_lookup_table(self, lookup_table):
        self.shape_table                    = lookup_table.get("shape")
        self.dtype_table                    = lookup_table.get("dtype")
        self.table_len                      = self.shape_table[0]       # 映射表的实际长度
        self.need_lookup                    = True
    
    def get_tik_instance(self):        
        if self.need_lookup is True:
            self.table_gm       = self.tik_instance.Tensor(self.dtype_table, self.shape_table, 
                                                           name="table_gm", scope=tik.scope_gm)
        
        with self.tik_instance.for_range(0, self.core_num, block_num=self.core_num) as tid:
            self.execute_task(tid)

        if self.need_lookup is True:
            self.tik_instance.BuildCCE(kernel_name=self.kernel_name, 
                                       inputs=[self.queries_gm, 
                                               self.index_gm, 
                                               self.index_num_gm, 
                                               self.db_gm, 
                                               self.table_gm], 
                                       outputs=[self.distance_gm])
        else:
            self.tik_instance.BuildCCE(kernel_name=self.kernel_name, 
                                       inputs=[self.queries_gm, 
                                               self.index_gm, 
                                               self.index_num_gm,
                                               self.db_gm], 
                                       outputs=[self.distance_gm])
        return self.tik_instance

    def execute_task(self, task_id):
        # 获取每个查询向量对应的实际有效索引个数
        index_num_ub = self.tik_instance.Tensor(self.dtype_index_num, self.shape_index_num,
                                             name="index_num_ub", scope=tik.scope_ubuf)
        self.tik_instance.data_move(index_num_ub[0], self.index_num_gm[0], 
                                    0, 1, self.shape_index_num[0] * 4 // 32, 0, 0)
        index_num_sc = self.tik_instance.Scalar(dtype="uint32", name="index_num_sc", init_value=0)
        index_num_sc.set_as(index_num_ub[0])

        if self.total_queries < self.core_num:
            with self.tik_instance.if_scope(task_id < self.total_queries):
                queries_each_core   = 1
                query_offset        = task_id
                self.compute_distance_each_core(self.queries_gm[query_offset:query_offset+queries_each_core, :],
                                                self.index_gm[query_offset:query_offset+queries_each_core, :],
                                                index_num_sc,
                                                self.db_gm,
                                                self.distance_gm,
                                                queries_each_core,
                                                query_offset,
                                                self.need_lookup)
        elif self.total_queries % self.core_num == 0:
            queries_each_core       = self.total_queries // self.core_num
            query_offset            = task_id * queries_each_core
            self.compute_distance_each_core(self.queries_gm[query_offset:query_offset+queries_each_core, :],
                                            self.index_gm[query_offset:query_offset+queries_each_core, :],
                                            index_num_sc,
                                            self.db_gm,
                                            self.distance_gm,
                                            queries_each_core,
                                            query_offset,
                                            self.need_lookup)
        else:
            # 前面几个核会比后面几个核多分配一个query
            with self.tik_instance.if_scope(task_id < (self.total_queries % self.core_num)):
                queries_each_core   = self.total_queries // self.core_num + 1
                query_offset        = task_id * queries_each_core

                self.compute_distance_each_core(self.queries_gm[query_offset:query_offset+queries_each_core, :],
                                                self.index_gm[query_offset:query_offset+queries_each_core, :],
                                                index_num_sc,
                                                self.db_gm,
                                                self.distance_gm,
                                                queries_each_core,
                                                query_offset,
                                                self.need_lookup)
            with self.tik_instance.else_scope():
                queries_each_core   = self.total_queries // self.core_num
                base_offset         = (self.total_queries // self.core_num + 1) * (self.total_queries % self.core_num)
                base_id             = self.total_queries % self.core_num
                query_offset        = base_offset + (task_id-base_id) * queries_each_core

                self.compute_distance_each_core(self.queries_gm[query_offset:query_offset+queries_each_core, :],
                                                self.index_gm[query_offset:query_offset+queries_each_core, :],
                                                index_num_sc,
                                                self.db_gm,
                                                self.distance_gm,
                                                queries_each_core,
                                                query_offset,
                                                self.need_lookup)             
    
    def compute_distance_each_core(self, queries_each_core_gm, idx_each_core_gm, index_num_sc, 
                                   db_gm, distance_gm, queries_each_core, 
                                   query_offset, need_lookup):
        # 定义UB中的Tensor，用于存放查询向量以及索引矩阵。
        queries_each_core_ub    = self.tik_instance.Tensor(self.dtype_queries, 
                                                           (queries_each_core, self.dim), 
                                                           name="queries_each_core_ub", 
                                                           scope=tik.scope_ubuf)
        idx_each_core_ub        = self.tik_instance.Tensor(self.dtype_index, 
                                                           (queries_each_core, self.shape_index[1]), 
                                                           name="index_each_core_ub", 
                                                           scope=tik.scope_ubuf)
        if need_lookup is True:
            table_len_align = div_up(self.table_len, 8)*8
            table_each_core_ub  = self.tik_instance.Tensor(self.dtype_table, 
                                                           (table_len_align,), 
                                                           name="table_each_core_ub", 
                                                           scope=tik.scope_ubuf)
            self.tik_instance.data_move(table_each_core_ub[0], self.table_gm[0], 0, 1, table_len_align * 4 // 32, 0, 0)
        # 从GM中加载以上两个Tensor数据
        self.tik_instance.data_move(queries_each_core_ub[0, 0], 
                                    queries_each_core_gm[0, 0], 
                                    0, 1, queries_each_core * self.dim // 16, 0, 0)
        self.tik_instance.data_move(idx_each_core_ub[0, 0], 
                                    idx_each_core_gm[0, 0], 
                                    0, 1, queries_each_core*self.index_num_each_query//8, 0, 0)

        # 按照每条查询进行matmul左右矩阵构造，然后进行计算。
        # 左矩阵是一个16行的矩阵，其中只有第一行数据有效，其余15行数据我们不需要填充，M=16, K=dim
        # 右矩阵根据索引矩阵中的索引值从底库向量中加载构造，K=dim, N=index_num_each_query
        with self.tik_instance.for_range(0, queries_each_core, thread_num=1) as query_id:
            query_matrix_l1     = self.tik_instance.Tensor(self.dtype_queries, 
                                                           (self.dim // 16, 16, 16), 
                                                           name="query_matrix_l1", 
                                                           scope=tik.scope_cbuf)
            db_matrix_l1        = self.tik_instance.Tensor(self.dtype_db, 
                                                           (self.dim // 16, self.index_num_each_query, 16), 
                                                           name="db_matrix_l1", 
                                                           scope=tik.scope_cbuf)            
            # 从UB中搬运数据至L1左矩阵，只搬运第一行数据
            self.tik_instance.data_move(query_matrix_l1[0, 0, 0], 
                                        queries_each_core_ub[query_id, 0], 
                                        0, self.dim // 16, 1, 0, 15)  
            # 根据索引值从GM中搬运L2右矩阵
            # 除了index_num_sc是<=64的，还需要加限制dim <= 512，防止所有ubuf Tensor之和超过256K
            if self.dim <= 512:
                x_db_matrix_ub = self.tik_instance.Tensor(self.dtype_db,
                                                          (index_num_sc, self.dim * self.db_zregion_h),
                                                          name="x_db_matrix_ub",
                                                          scope=tik.scope_ubuf)
                with self.tik_instance.for_range(0, index_num_sc, thread_num=2) as index_id:
                    # ui表示该查询向量对应的底库向量的index号
                    ui = self.tik_instance.Scalar(dtype="uint32", name="ui", init_value=0)
                    ui.set_as(idx_each_core_ub[query_id, index_id])
                    self.tik_instance.data_move(x_db_matrix_ub[index_id, 0],
                                                db_gm[ui // self.db_zregion_h, 0, ui % self.db_zregion_h, 0],
                                                0,
                                                1, self.dim * self.db_zregion_h // 16,
                                                0, 0)
                    self.tik_instance.data_move(db_matrix_l1[0, index_id, 0],
                                                x_db_matrix_ub[index_id, 0],
                                                0,
                                                self.dim // 16, 1,
                                                self.db_zregion_h - 1, (self.index_num_each_query - 1))
            else:
                with self.tik_instance.for_range(0, index_num_sc,
                                                 thread_num=2) as index_id:
                    ui = self.tik_instance.Scalar(dtype="uint32", name="ui", init_value=0)
                    ui.set_as(idx_each_core_ub[query_id, index_id])
                    self.tik_instance.data_move(db_matrix_l1[0, index_id, 0],
                                                db_gm[ui // self.db_zregion_h, 0, ui % self.db_zregion_h, 0],
                                                0,
                                                self.dim // 16, 1,
                                                self.db_zregion_h - 1, (self.index_num_each_query - 1))
            # 执行matmul
            distance_matrix_l0c = self.tik_instance.Tensor("float32", 
                                                           (self.index_num_each_query // 16, 16, 16), 
                                                           name="distance_matrix_l0c", 
                                                           scope=tik.scope_cbuf_out)
            self.tik_instance.matmul(distance_matrix_l0c, query_matrix_l1, db_matrix_l1, 
                                     16, self.dim, self.index_num_each_query)
            # 将距离结果先拷贝至UB中
            distance_result_ub = self.tik_instance.Tensor("float32", 
                                                          (self.index_num_each_query // 16, 16, 16), 
                                                          name="distance_result_ub", 
                                                          scope=tik.scope_ubuf)
            self.tik_instance.data_move(distance_result_ub[0, 0, 0], 
                                        distance_matrix_l0c[0, 0, 0], 
                                        0, 1, 16 * self.index_num_each_query * 4 // 1024, 0, 0)
            # 不做距离精度转换，直接将结果存到GM中，后续可以考虑先做精度转换然后再保存，这样可以减少UB到GM的写操作数据量。
            # distance_result_ub中只有第一行数据是有效的！            
            if need_lookup is False:  
                self.tik_instance.data_move(distance_gm[query_offset+query_id, 0], 
                                            distance_result_ub[0, 0, 0], 
                                            0, self.index_num_each_query // 16, 2, 15 * 2, 0)
            else:
                cos_ub = self.tik_instance.Tensor("float32", 
                                                  (self.index_num_each_query,), 
                                                  name="cos_ub", 
                                                  scope=tik.scope_ubuf)
                lookup_result_ub = self.tik_instance.Tensor(self.dtype_table, 
                                                            (self.index_num_each_query,), 
                                                            name="lookup_result_ub", 
                                                            scope=tik.scope_ubuf)
                
                self.tik_instance.data_move(cos_ub[0], distance_result_ub[0, 0, 0], 
                                            0, self.index_num_each_query // 16, 2, 15 * 2, 0)
                with self.tik_instance.for_range(0, 1):
                    lookup_table_in_ub(self.tik_instance, cos_ub, table_each_core_ub, self.table_len, lookup_result_ub)
                self.tik_instance.data_move(distance_gm[query_offset+query_id, 0], 
                                            lookup_result_ub[0], 
                                            0, 1, self.index_num_each_query * 4 // 32, 0, 0)               


class DistanceFlatIpByIdx2:
    """
        DistanceFlatIpByIdx2：查询向量根据序号指定，待比较向量是底库中的一段连续向量。
    """
    def __init__(self, input_queries_index, input_code_info, input_db, output_distance, kernel_name):
        self.shape_queries_index            = input_queries_index.get("shape")
        self.dtype_queries_index            = input_queries_index.get("dtype")

        self.shape_code_info                = input_code_info.get("shape")
        self.dtype_code_info                = input_code_info.get("dtype")

        self.shape_db                       = input_db.get("shape")
        self.dtype_db                       = input_db.get("dtype")
        # Z区域高度，即Z区域包含的向量个数。
        self.db_zregion_h                   = self.shape_db[2]
        
        self.shape_distance                 = output_distance.get("shape")
        self.dtype_distance                 = output_distance.get("dtype")

        self.kernel_name                    = kernel_name
        
        self.total_queries                  = self.shape_queries_index[0]
        self.dim                            = self.shape_db[1] * self.shape_db[3]

        # 每次执行matmul指令时右矩阵的跨度，即一次做多少个底库向量的距离计算，16的倍数。
        self.db_slice_stride                = 512
        # 每次计算距离对应的查询条数
        self.query_num_each_compute         = div_up(self.total_queries, 16) * 16
        # 保存距离结果时按照一次多少条查询对应的距离值进行保存。
        self.query_num_each_save            = self.query_num_each_compute
    
        self.core_num                       = AI_CORE_NUM 

        self.tik_instance           = tik.Tik(tik.Dprofile("v200", "aic"), disable_debug=False)
        # 输入1：查询向量的索引
        self.queries_index_gm       = self.tik_instance.Tensor(self.dtype_queries_index, 
                                                               self.shape_queries_index, 
                                                               name="queries_index_gm", 
                                                               scope=tik.scope_gm)
        # 输入2：待比较向量的信息
        self.code_info_gm           = self.tik_instance.Tensor(self.dtype_code_info, 
                                                               self.shape_code_info, 
                                                               name="code_info_gm", 
                                                               scope=tik.scope_gm)
        # 输入3：底库特征
        self.db_gm                  = self.tik_instance.Tensor(self.dtype_db, 
                                                               self.shape_db, 
                                                               name="db_gm", 
                                                               scope=tik.scope_gm)
        # 输出1：距离值结果
        self.distance_gm    = self.tik_instance.Tensor(self.dtype_distance, 
                                                       self.shape_distance, 
                                                       name="distance_gm", 
                                                       scope=tik.scope_gm)

    def set_optimize_policy(self):
        pass

    def check_settings(self):
        if self.dim % 16 != 0:
            raise RuntimeError("dim must be multiple of 16!")
        if self.db_slice_stride % self.db_zregion_h != 0:
            raise RuntimeError("db_slice_stride must be multiple of z region height!")
        if (self.query_num_each_compute % 16 != 0):
            raise RuntimeError("query_num_each_compute must be multiple of 16!")
        if self.query_num_each_compute % self.query_num_each_save != 0:
            raise RuntimeError("query_num_each_compute must be multiple of query_num_each_save!")

    def get_tik_instance(self):
        # 设置优化策略
        self.set_optimize_policy()
        # 参数检查
        self.check_settings()
        
        with self.tik_instance.for_range(0, self.core_num, block_num=self.core_num) as tid:
            self.execute_task(tid)
        
        inputs  = [self.queries_index_gm, self.code_info_gm, self.db_gm]
        outputs = [self.distance_gm]
        
        self.tik_instance.BuildCCE(self.kernel_name,
                                   inputs, 
                                   outputs)
        
        return self.tik_instance
       
    def execute_task(self, task_id):
        """
        执行任务函数：第一个任务比较特殊，它用于处理第一块底库数据以及整除剩余的底库向量。
        """
        # 获取待比较向量的起始序号以及数量
        db_start_idx_sc = self.tik_instance.Scalar(dtype="uint32", name="db_start_idx_sc", init_value=0)
        db_num_sc = self.tik_instance.Scalar(dtype="uint32", name="db_num_sc", init_value=0)
        with self.tik_instance.for_range(0, 1):
            code_info_ub = self.tik_instance.Tensor(self.dtype_code_info, 
                                                    self.shape_code_info, 
                                                    name="code_info_ub", 
                                                    scope=tik.scope_ubuf)
            self.tik_instance.data_move(code_info_ub[0], 
                                        self.code_info_gm[0], 
                                        0, 
                                        1, self.shape_code_info[0] * 4 // 32, 
                                        0, 0)
            db_start_idx_sc.set_as(code_info_ub[0])
            db_num_sc.set_as(code_info_ub[1])

        # 计算每个任务对应的底库向量起始位置以及底库向量个数
        db_slice_num_align = db_num_sc // self.db_slice_stride
        # 简单考虑，将底库数量地板除以AI Core数量，即为每个任务需要处理的底库向量个数，剩余的底库向量放在第一个任务中。
        db_vector_num_each_task         = (db_slice_num_align // self.core_num) * self.db_slice_stride
        db_vector_offset_each_task      = db_start_idx_sc + task_id * db_vector_num_each_task
        db_vector_num_last              = db_num_sc - db_vector_num_each_task * self.core_num
        db_vector_offset_last           = db_start_idx_sc + db_vector_num_each_task * self.core_num

        with self.tik_instance.if_scope(task_id != 0):
            self.compute_distance_each_task(self.queries_index_gm, db_vector_offset_each_task, 
                                            db_vector_num_each_task, self.db_gm, db_start_idx_sc, db_num_sc, 
                                            self.distance_gm)
        with self.tik_instance.else_scope():
            # 这里做了一个特殊处理：对于第一个任务先计算后面剩余的底库向量再处理第一块底库向量，
            # 这样就不会因为底库向量个数不整除db_slice_stride而引起的距离计算结果覆盖的问题了。
            self.compute_distance_each_task(self.queries_index_gm, db_vector_offset_last, 
                                            db_vector_num_last, self.db_gm, db_start_idx_sc, db_num_sc, 
                                            self.distance_gm)
            self.compute_distance_each_task(self.queries_index_gm, db_vector_offset_each_task, 
                                            db_vector_num_each_task, self.db_gm, db_start_idx_sc, db_num_sc, 
                                            self.distance_gm)
        
    def compute_distance_each_task(self, queries_index_gm, db_vector_offset, 
                                   db_vector_num, db_gm, db_start_idx_sc, db_num_sc,
                                   distance_gm):
        with self.tik_instance.if_scope(db_vector_num != 0): 
            # 加载左矩阵数据：一次性将查询向量全部加载至L1中，形成matmul的左矩阵（四维），dtype采用float16!
            query_batch_cnt             = div_up(self.total_queries, self.query_num_each_compute)
            queries_matrix_l1           = self.tik_instance.Tensor("float16", 
                                                                   (query_batch_cnt, 
                                                                   self.dim // 16, 
                                                                   self.query_num_each_compute, 
                                                                   16), 
                                                                   name="queries_matrix_l1", 
                                                                   scope=tik.scope_cbuf)
            with self.tik_instance.for_range(0, 1):
                total_queries_align = div_up(self.total_queries, 8) * 8
                queries_index_ub = self.tik_instance.Tensor("uint32", 
                                                            (total_queries_align,), 
                                                            name="queries_index_ub",
                                                            scope=tik.scope_ubuf)
                self.tik_instance.data_move(queries_index_ub[0],
                                            self.queries_index_gm[0],
                                            0,
                                            1, total_queries_align * 4 // 32,
                                            0, 0)
                tmp_cnt_sc  = self.tik_instance.Scalar("uint32")
                with self.tik_instance.for_range(0, query_batch_cnt, thread_num=1) as bid:
                    tmp_index   = bid * self.query_num_each_compute
                    with self.tik_instance.if_scope(bid != (query_batch_cnt - 1)):
                        tmp_cnt_sc.set_as(self.query_num_each_compute)
                    with self.tik_instance.else_scope():
                        tmp_cnt_sc.set_as(self.total_queries - (query_batch_cnt - 1) * self.query_num_each_compute)

                    self.load_queries_from_gm_to_l1(queries_index_ub,
                                                    tmp_index,
                                                    tmp_cnt_sc, 
                                                    queries_matrix_l1[bid : bid + 1, :, :, :])
            # 以下是性能最佳写法，另一种实现方式是将下面的两次调用合在一起，并使用Scalar变量记录每次处理的数据个数。
            # 每次处理一个db_slice，每次db_slice_stride个。
            with self.tik_instance.for_range(0, db_vector_num // self.db_slice_stride, thread_num=1) as slice_index:
                self.compute_distance_each_slice(queries_matrix_l1,
                                                 db_gm,
                                                 distance_gm,
                                                 db_start_idx_sc,
                                                 db_num_sc,
                                                 db_vector_offset,
                                                 slice_index,
                                                 self.db_slice_stride)
            # 最后一次可能不足db_slice_stride个
            db_vector_remain = db_vector_num % self.db_slice_stride
            with self.tik_instance.if_scope(db_vector_remain > 0):
                self.compute_distance_each_slice(queries_matrix_l1,
                                                 db_gm,
                                                 distance_gm,
                                                 db_start_idx_sc,
                                                 db_num_sc,
                                                 db_vector_offset,
                                                 db_vector_num // self.db_slice_stride,
                                                 db_vector_remain)

    def compute_distance_each_slice(self, 
                                    queries_matrix_l1, 
                                    db_gm, 
                                    distance_gm, 
                                    db_start_idx_sc,
                                    db_num_sc, 
                                    db_vector_offset, 
                                    slice_index, 
                                    db_slice_vector_cnt):
        db_slice_vector_offset = db_vector_offset + slice_index * self.db_slice_stride
        # 定义Scalar变量用于存储每次计算对应的查询个数        
        queries_num_sc  = self.tik_instance.Scalar(dtype='uint32', name='queries_num_sc', init_value=0)
        # 定义matmul右矩阵，即底库向量
        db_slice_matrix_l1  = self.tik_instance.Tensor("float16",
                                                       (self.dim // 16, self.db_slice_stride, 16),
                                                       name="db_slice_matrix_l1",        
                                                       scope=tik.scope_cbuf)
        # 定义matmul结果矩阵
        distance_matrix_l0c = self.tik_instance.Tensor("float32", 
                                                       (self.db_slice_stride // 16, self.query_num_each_compute, 16),
                                                       name="distance_matrix_l0c",
                                                       scope=tik.scope_cbuf_out)
        # 加载右矩阵
        with self.tik_instance.for_range(0, 1):
            self.load_db_from_gm_to_l1(db_gm,
                                       db_slice_vector_offset, 
                                       db_slice_vector_cnt, 
                                       db_slice_matrix_l1)
        
        # 对查询进行切分，每次计算query_num_each_compute个查询。 
        query_batch_cnt     = div_up(self.total_queries, self.query_num_each_compute)
        queries_num_sc      = self.tik_instance.Scalar(dtype='uint32', name='queries_num_sc', init_value=0)
        with self.tik_instance.for_range(0, query_batch_cnt, thread_num=1) as bid:
            with self.tik_instance.if_scope(bid != (query_batch_cnt - 1)):
                queries_num_sc.set_as(self.query_num_each_compute)
            with self.tik_instance.else_scope():
                queries_num_sc.set_as(self.total_queries - (query_batch_cnt - 1) * self.query_num_each_compute)
            # 执行matmul
            self.tik_instance.matmul(distance_matrix_l0c, 
                                     queries_matrix_l1[bid : bid + 1, :, :, :], 
                                     db_slice_matrix_l1,
                                     self.query_num_each_compute, 
                                     self.dim, 
                                     self.db_slice_stride)
            # 保存结果
            with self.tik_instance.for_range(0, 1):
                tmp_dist_start              = bid * self.query_num_each_compute * db_num_sc
                tmp_dist_end                = tmp_dist_start + queries_num_sc * db_num_sc
                
                self.save_distance_from_l0c_to_gm(distance_gm[tmp_dist_start : tmp_dist_end],
                                                  distance_matrix_l0c, 
                                                  db_slice_vector_offset, 
                                                  db_slice_vector_cnt,
                                                  db_start_idx_sc,
                                                  db_num_sc,
                                                  queries_num_sc)

    def load_queries_from_gm_to_l1(self, queries_index_ub, query_off, query_cnt, matrix_l1):
        """
        从GM中加载查询向量至L1，形成左矩阵。
        Args:
            queries_index_ub:   查询向量索引，shape=(N,), dtype=float16
            query_off:          查询向量偏移
            query_cnt:          查询向量个数
            matrix_l1:          l1矩阵，这里是左矩阵，四维，多出的一维是batch。
        """
        query_index_sc = self.tik_instance.Scalar(dtype='uint32', name='query_index_sc', init_value=0)
        with self.tik_instance.for_range(0, query_cnt) as qid:
            query_index_sc.set_as(queries_index_ub[query_off + qid])
            for d in range(self.dim // 16):
                self.tik_instance.data_move(matrix_l1[0, d, qid, 0],
                    self.db_gm[query_index_sc // self.db_zregion_h, d, query_index_sc % self.db_zregion_h, 0],
                    0,
                    1, 16 * 2 // 32,
                    0, 0)

    def load_db_from_gm_to_l1(self, db_gm, db_vector_off, db_vector_cnt, matrix_l1, method=0):
        """
        从GM中加载底库向量至L1，形成右矩阵。
        Args:
            db_gm:              底库向量Tensor：shape=(M/H, Dim/16, H, 16), dtype=float16
            db_vector_off:      底库向量偏移
            db_vector_cnt:      底库向量个数，这里有一个约束：db_vector_cnt <= db_slice_stride
            matrix_l1:          l1矩阵，这里是右矩阵，三维。
            method:             加载的方法选择，根据实际调试结果选择。
        """

        if method == 0:
            # 按照维度分段方式进行加载 
            for d in range(self.dim // 16):
                nburst      = db_vector_cnt // self.db_zregion_h
                burst_len   = (self.db_zregion_h * 16 * 2) // 32
                with self.tik_instance.if_scope(nburst != 0):
                    self.tik_instance.data_move(matrix_l1[d, 0, 0],
                                                db_gm[db_vector_off // self.db_zregion_h, d, 0, 0],
                                                0,
                                                nburst,
                                                burst_len,
                                                (self.dim // 16 - 1) * self.db_zregion_h,
                                                0)
                remain  = db_vector_cnt % self.db_zregion_h
                with self.tik_instance.if_scope(remain != 0):
                    self.tik_instance.data_move(matrix_l1[d, nburst * self.db_zregion_h, 0],
                                                db_gm[(db_vector_off + db_vector_cnt) // self.db_zregion_h, d, 0, 0],
                                                0,
                                                1,
                                                remain,
                                                0, 0)
        elif method == 1:
            # 按照每次加载一个Z区域方式，这样读GM时是连续的。
            matrix_l1_stride    = matrix_l1.shape[1]    
            repeat              = div_up(db_vector_cnt, self.db_zregion_h)
            tmp_cnt_sc          = self.tik_instance.Scalar(dtype='uint32')
            with self.tik_instance.for_range(0, repeat) as rid:
                with self.tik_instance.if_scope(rid != (repeat - 1)):
                    tmp_cnt_sc.set_as(self.db_zregion_h)
                with self.tik_instance.else_scope():
                    tmp_cnt_sc.set_as(db_vector_cnt - (repeat - 1) * self.db_zregion_h)

                nburst      = self.dim // 16
                burst_len   = (tmp_cnt_sc * 16 * 2) // 32

                self.tik_instance.data_move(matrix_l1[0, rid * self.db_zregion_h, 0],
                                            db_gm[db_vector_off // self.db_zregion_h + rid, 0, 0, 0],
                                            0,
                                            nburst, 
                                            burst_len,
                                            0, 
                                            (matrix_l1_stride - self.db_zregion_h) + (self.db_zregion_h - tmp_cnt_sc))
        else:
            raise RuntimeError("load_db_from_gm_to_l1: unsupported load method!")
    
    def save_distance_from_l0c_to_gm(self, 
                                     distance_gm,
                                     distance_matrix_l0c, 
                                     db_vector_offset, 
                                     db_vector_cnt,
                                     db_start_idx,
                                     db_num, 
                                     queries_num):
        """
        将L0C中的距离结果矩阵保存至GM中
        Args:
            distance_gm:                    距离结果在GM中的Tensor，shape=(N*M,)，dtype=float16
            distance_matrix_l0c:            距离结果在L0C中的矩阵，shape=(db_slice_stride/16, queries_align, 16)，dtype=float32
            db_vector_offset:               底库向量起始位置
            db_vector_cnt:                  底库向量个数
            db_num:                         底库向量实际总个数
            queries_num:                    查询的实际个数
        """
        queries_num_align = distance_matrix_l0c.shape[1] 
        # Step1: 将distance_matrix_l0c结果一次性转移至UB中，并从三维转换为二维。
        distance_result_2d_ub = self.tik_instance.Tensor("float32",
                                                         (queries_num_align, self.db_slice_stride),
                                                         name="distance_result_2d_ub",
                                                         scope=tik.scope_ubuf)
        with self.tik_instance.for_range(0, 1):
            distance_result_3d_ub = self.tik_instance.Tensor("float32",
                                                             (self.db_slice_stride // 16, queries_num_align, 16),
                                                             name="distance_result_3d_ub",
                                                             scope=tik.scope_ubuf)
            # L0C(3d) -> UB(3d)
            self.tik_instance.data_move(distance_result_3d_ub[0, 0, 0],
                                        distance_matrix_l0c[0, 0, 0],
                                        0,
                                        1, queries_num_align * self.db_slice_stride * 4 // 1024,
                                        0, 0)
            # UB(3d) -> UB(2d)
            for k in range(self.db_slice_stride // 16):
                self.tik_instance.data_move(distance_result_2d_ub[0, k * 16],
                                            distance_result_3d_ub[k, 0, 0],
                                            0, 
                                            queries_num_align, 2,
                                            0, (self.db_slice_stride // 16 - 1) * 2)        

        # Step2: 将distance_result_2d_ub中的结果分批写入GM中，分为有无映射表两种情况。
        repeat      = queries_num // self.query_num_each_save
        last_remain = queries_num % self.query_num_each_save
            
        with self.tik_instance.for_range(0, repeat) as rid:
            query_start = rid * self.query_num_each_save
            with self.tik_instance.for_range(0, self.query_num_each_save) as k:
                tmp_gm_offset = (query_start + k) * db_num + db_vector_offset - db_start_idx
                self.tik_instance.data_move(distance_gm[tmp_gm_offset],
                                            distance_result_2d_ub[query_start + k, 0],
                                            0,
                                            1, div_up(db_vector_cnt, 8) * 8 * 4 // 32,
                                            0, 0)
        with self.tik_instance.if_scope(last_remain > 0):
            with self.tik_instance.for_range(0, last_remain) as k:
                tmp_gm_offset = (repeat * self.query_num_each_save + k) * db_num + db_vector_offset - db_start_idx
                self.tik_instance.data_move(distance_gm[tmp_gm_offset],
                                            distance_result_2d_ub[repeat * self.query_num_each_save + k, 0],
                                            0, 
                                            1, div_up(db_vector_cnt, 8) * 8 * 4 // 32,
                                            0, 0)


class DistanceFilter:
    def __init__(self, dist, thresh, dist_num, base_idx, dist_filter, idx_filter, cnt_filter, kernel_name):
        self.shape_dist             = dist.get("shape")
        self.dtype_dist             = dist.get("dtype")
        
        self.shape_thresh           = thresh.get("shape")
        self.dtype_thresh           = thresh.get("dtype")

        self.shape_dist_num         = dist_num.get("shape")
        self.dtype_dist_num         = dist_num.get("dtype")

        self.shape_base_idx         = base_idx.get("shape")
        self.dtype_base_idx         = base_idx.get("dtype")

        self.shape_dist_filter      = dist_filter.get("shape")
        self.dtype_dist_filter      = dist_filter.get("dtype")

        self.shape_idx_filter       = idx_filter.get("shape")
        self.dtype_idx_filter       = idx_filter.get("dtype")

        self.shape_cnt_filter       = cnt_filter.get("shape")
        self.dtype_cnt_filter       = cnt_filter.get("dtype")

        self.kernel_name            = kernel_name

        self.total_queries, _       = self.shape_dist
        self.core_num               = AI_CORE_NUM

        self.dist_num_each_filter   = (64 * 128)     # 超参，每次执行过滤的距离个数，64的倍数，受UB大小限制。

        self.dist_num_aligned       = True # DH需求中输入的每个查询对应的距离个数是16对齐的，其它场景不需要。

        self.tik_instance           = tik.Tik(tik.Dprofile("v200", "aic"), disable_debug=False)

        self.dist_gm                = self.tik_instance.Tensor(self.dtype_dist, self.shape_dist, 
                                                        name="dist_gm", scope=tik.scope_gm)
        self.thresh_gm              = self.tik_instance.Tensor(self.dtype_thresh, self.shape_thresh,
                                                        name="thresh_gm", scope=tik.scope_gm)
        self.dist_num_gm            = self.tik_instance.Tensor(self.dtype_dist_num, self.shape_dist_num,
                                                        name="dist_num_gm", scope=tik.scope_gm)
        self.base_idx_gm            = self.tik_instance.Tensor(self.dtype_base_idx, self.shape_base_idx,
                                                        name="base_idx_gm", scope=tik.scope_gm)
        self.dist_filter_gm         = self.tik_instance.Tensor(self.dtype_dist_filter, self.shape_dist_filter,
                                                        name="dist_filter_gm", scope=tik.scope_gm)
        self.idx_filter_gm          = self.tik_instance.Tensor(self.dtype_idx_filter, self.shape_idx_filter,
                                                        name="idx_filter_gm", scope=tik.scope_gm)
        self.cnt_filter_gm          = self.tik_instance.Tensor(self.dtype_cnt_filter, self.shape_cnt_filter,
                                                        name="cnt_filter_gm", scope=tik.scope_gm)
    
    def get_tik_instance(self):
        with self.tik_instance.for_range(0, self.core_num, block_num=self.core_num) as tid:
            self.execute_task(tid)
        
        self.tik_instance.BuildCCE(kernel_name=self.kernel_name,
                                   inputs=[self.dist_gm, self.thresh_gm, self.dist_num_gm, self.base_idx_gm],
                                   outputs=[self.dist_filter_gm, self.idx_filter_gm, self.cnt_filter_gm])
        return self.tik_instance

    def execute_task(self, task_id):
        if self.total_queries < self.core_num:
            with self.tik_instance.if_scope(task_id < self.total_queries):
                queries_each_core   = 1
                query_offset        = task_id
                self.filter_distance_each_core(query_offset, queries_each_core)
        elif self.total_queries % self.core_num == 0:
            queries_each_core       = self.total_queries // self.core_num
            query_offset            = task_id * queries_each_core
            self.filter_distance_each_core(query_offset, queries_each_core)
        else:
            # 前面几个核会比后面几个核多分配一个query
            with self.tik_instance.if_scope(task_id < (self.total_queries % self.core_num)):
                queries_each_core   = self.total_queries//self.core_num + 1
                query_offset        = task_id * queries_each_core
                self.filter_distance_each_core(query_offset, queries_each_core)
            with self.tik_instance.else_scope():
                queries_each_core   = self.total_queries // self.core_num
                base_offset         = (self.total_queries // self.core_num + 1) * (self.total_queries % self.core_num)
                base_id             = self.total_queries % self.core_num
                query_offset        = base_offset+(task_id-base_id)*queries_each_core
                self.filter_distance_each_core(query_offset, queries_each_core)

    def filter_distance_each_core(self, query_offset, queries_each_core):
        # 定义每个查询对应的实际距离个数，Scalar变量
        dist_num_sc = self.tik_instance.Scalar(dtype="uint32", name="db_num_sc", init_value=0)
        with self.tik_instance.for_range(0, 1):
            tmp_dist_num_ub = self.tik_instance.Tensor("uint32", 
                                                        self.shape_dist_num, 
                                                        name="tmp_dist_num_ub",
                                                        scope=tik.scope_ubuf)
            self.tik_instance.data_move(tmp_dist_num_ub, 
                                        self.dist_num_gm, 
                                        0, 1, self.shape_dist_num[0] * 4 // 32, 0, 0)
            dist_num_sc.set_as(tmp_dist_num_ub[0])
        # 定义距离值对应的起始序号，Scalar变量
        base_idx_sc = self.tik_instance.Scalar(dtype="uint32", name="base_idx_sc", init_value=0)
        with self.tik_instance.for_range(0, 1):
            tmp_base_idx_ub = self.tik_instance.Tensor("uint32",
                                                       self.shape_base_idx,
                                                       name="tmp_base_idx_ub",
                                                       scope=tik.scope_ubuf)
            self.tik_instance.data_move(tmp_base_idx_ub,
                                        self.base_idx_gm,
                                        0, 1, self.shape_base_idx[0] * 4 // 32, 0, 0)
            base_idx_sc.set_as(tmp_base_idx_ub[0])
        # 定义过滤阈值，Scalar变量
        thresh_sc = self.tik_instance.Scalar(dtype="float32", name="thresh_sc", init_value=0)
        with self.tik_instance.for_range(0, 1):
            tmp_thresh_ub = self.tik_instance.Tensor("float32",
                                                     self.shape_thresh,
                                                     name="tmp_thresh_ub",
                                                     scope=tik.scope_ubuf)
            self.tik_instance.data_move(tmp_thresh_ub,
                                        self.thresh_gm,
                                        0, 1, self.shape_thresh[0] * 4 // 32, 0, 0)
            thresh_sc.set_as(tmp_thresh_ub[0])
        # 定义阈值Tensor，在UB中分配
        thresh_ub = self.tik_instance.Tensor("float32",
                                             (self.dist_num_each_filter, ),
                                             name="thresh_ub",
                                             scope=tik.scope_ubuf)
        # 初始化阈值Tensor
        self.tik_instance.vec_dup(64, thresh_ub, thresh_sc, self.dist_num_each_filter // 64, 8)
        # 定义输入距离Tensor，在UB中分配
        dist_ub = self.tik_instance.Tensor("float32",
                                           (self.dist_num_each_filter, ),
                                           name="dist_ub",
                                           scope=tik.scope_ubuf)
        # 定义距离值序号Tensor，在UB中分配
        idx_ub = self.tik_instance.Tensor("int32",
                                           (self.dist_num_each_filter, ),
                                           name="idx_ub",
                                           scope=tik.scope_ubuf)
        # 定义距离值序号基础Tensor，用于生成距离序号值，在UB中分配
        idx_base_ub = self.tik_instance.Tensor("int32",
                                           (self.dist_num_each_filter, ),
                                           name="idx_base_ub",
                                           scope=tik.scope_ubuf)
        # 初始化距离值序号基础Tensor
        with self.tik_instance.for_range(0, self.dist_num_each_filter) as index:
            idx_base_ub[index] = index

        # 定义过滤后的距离Tensor，在UB中分配
        dist_filter_ub = self.tik_instance.Tensor("float32",
                                                  (self.dist_num_each_filter, ),
                                                  name="dist_filter_ub",
                                                  scope=tik.scope_ubuf)
        # 定义过滤后的序号Tensor，在UB中分配
        idx_filter_ub = self.tik_instance.Tensor("int32",
                                                 (self.dist_num_each_filter, ),
                                                 name="idx_filter_ub",
                                                 scope=tik.scope_ubuf)
        # 定义过滤后满足条件的距离个数，Scalar变量
        cnt_filter_sc = self.tik_instance.Scalar(dtype="uint32", name="cnt_filter_sc", init_value=0)
        # 定义每个查询满足阈值条件的距离值个数，动态增加，Scalar变量
        cnt_filter_sum_sc = self.tik_instance.Scalar(dtype="uint32", name="cnt_filter_sum_sc", init_value=0)
        if self.dist_num_aligned is True:
            dist_num_sc_align = div_up(dist_num_sc, 16) * 16
        with self.tik_instance.for_range(0, queries_each_core) as qid:
            repeat = div_up(dist_num_sc, self.dist_num_each_filter) # 每个查询需要执行过滤的次数
            cnt_filter_sum_sc.set_as(0)                             # 重置为0
            with self.tik_instance.for_range(0, repeat) as rid:
                # 生成距离值序号
                self.tik_instance.vadds(64, idx_ub, idx_base_ub, rid * self.dist_num_each_filter + base_idx_sc,
                                        self.dist_num_each_filter // 64, 1, 1, 8, 8)
                tmp_dist_cnt = self.tik_instance.Scalar(dtype="uint32", name="tmp_dist_cnt", init_value=0)
                if self.dist_num_aligned is True:
                    tmp_dist_base = (query_offset + qid) * dist_num_sc_align + rid * self.dist_num_each_filter
                else:
                    tmp_dist_base = (query_offset + qid) * dist_num_sc + rid * self.dist_num_each_filter
                with self.tik_instance.if_scope(rid != (repeat - 1)):
                    tmp_dist_cnt.set_as(self.dist_num_each_filter)
                with self.tik_instance.else_scope():
                    tmp_dist_cnt.set_as(dist_num_sc - (repeat - 1) * self.dist_num_each_filter)

                tmp_dist_cnt_align = div_up(tmp_dist_cnt, 8) * 8
                # 加载距离值数据至dist_ub
                self.tik_instance.data_move(dist_ub[0],
                                            self.dist_gm[tmp_dist_base],
                                            0, 1, tmp_dist_cnt_align * 4 // 32, 0, 0)
                # 执行过滤函数
                distance_filter_in_ub(self.tik_instance,
                                      dist_ub, tmp_dist_cnt, thresh_ub, idx_ub,
                                      dist_filter_ub, idx_filter_ub, cnt_filter_sc)
                # 保存结果至GM
                if self.dist_num_aligned is True:
                    wr_pos = (query_offset + qid) * dist_num_sc_align + cnt_filter_sum_sc
                else:
                    wr_pos = (query_offset + qid) * dist_num_sc + cnt_filter_sum_sc
                with self.tik_instance.if_scope(cnt_filter_sc > 0):
                    # 结果1：满足阈值条件的距离值结果
                    self.tik_instance.data_move(self.dist_filter_gm[wr_pos],
                                            dist_filter_ub[0],
                                            0, 1, div_up(cnt_filter_sc, 8), 0, 0)
                    # 结果2：满足阈值条件的序号值结果
                    self.tik_instance.data_move(self.idx_filter_gm[wr_pos],
                                            idx_filter_ub[0],
                                            0, 1, div_up(cnt_filter_sc, 8), 0, 0)
                # 更新满足阈值条件的距离值个数
                cnt_filter_sum_sc.set_as(cnt_filter_sum_sc + cnt_filter_sc)
            # 结果3：每个查询对应的距离值中满足阈值条件的个数
            tmp_cnt_filter_sum_ub = self.tik_instance.Tensor("uint32",
                                                             (8,),
                                                             name="tmp_cnt_filter_sum_ub",
                                                             scope=tik.scope_ubuf)
            tmp_cnt_filter_sum_ub[0] = cnt_filter_sum_sc
            self.tik_instance.data_move(self.cnt_filter_gm[(query_offset + qid) * 8],
                                        tmp_cnt_filter_sum_ub[0],
                                        0, 1, 1, 0, 0)
