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
from mxIndex_impl.distance_flat_ip_maxs_with_mask import DistanceFlatIPMaxs


def distance_flat_ip_maxs_with_scale(input_queries,
                                     input_code,
                                     input_actual_num,
                                     input_mask,
                                     input_extra_score,
                                     input_scale,
                                     output_distances,
                                     output_maxs,
                                     output_flag,
                                     kernel_name="distance_flat_ip_maxs_with_scale"):
    """
    算子接口函数
    Args:
        input_queries: 查询向量，二维Tensor，shape=(N, Dim/8)，dtype=uint8，Dim为向量的维度，要求为32的倍数。
        input_code: 底库向量，二维Tensor，shape=(M/H, Dim/32, H, 4)，dtype=uint8，M为对齐后的底库总容量，
                  H是Z区域高度，即Z区域包含的向量个数。
        input_actual_num: 实际底库向量条数，一维Tensor，shape=(8,)，dtype=uint32，要求小于等于M。
        input_mask: 距离过滤Mask，每个查询向量对应一个，二维Tensor，shape=(N1, M/8)，dtype=uint8。
                  当N1=N时，每个查询对应的Mask不相同，当N1=1且不等于N时，算子内部对于所有查询采用同样的Mask。
        input_extra_score：附加相似度，二维Tensor，shape(N, ntotalPad)，ntotalPad为ntotal按照16对齐的长度，dtype=fp16
        input_scale：底库向量化，一维Tensor，shape(dim,)，dtype=fp16
        output_distances: 距离结果，一维Tensor，shape=(N*M, )，dtype=float16。
        output_maxs: 块topk结果，一维Tensor，shape=(N*DIV_UP(M, block_size)*block_topk*2, )，每一个块输出
                   BLOCK_TOPK个(Max, Index)二元组，dtype=float16
        output_flag: 算子执行完成标志，算子调用者可以轮巡该标志位，为了兼容已有代码，定义为uint16类型，一维Tensor，shape=(16,)。 
    约束：
        对齐后的底库容量须满足db_slice_stride的倍数，并且应用须按照M值申请输入和输出内存。
    """
    obj = DistanceFlatIPMaxs(input_queries, input_code, input_actual_num, input_mask,
                             output_distances, output_maxs, output_flag, kernel_name)
    obj.set_extra_score(input_extra_score)
    obj.set_scale(input_scale)
    tik_instance = obj.get_tik_instance()

    return tik_instance
