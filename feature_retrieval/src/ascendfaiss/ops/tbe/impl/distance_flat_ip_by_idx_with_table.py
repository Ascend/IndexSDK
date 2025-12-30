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
from mxIndex_impl.distance_flat_ip_common import DistanceFlatIpByIdx


def distance_flat_ip_by_idx_with_table(input_x0, input_x1, 
                                       input_x2, input_x3, 
                                       input_x4, output_y0, kernel_name="distance_flat_ip_by_idx_with_table"):
    """
    算子接口函数
    Args:
        input_x0: 查询向量，二维Tensor，shape=(N, Dim)，dtype=float16
        input_x1: 索引矩阵，二维Tensor，shape=(N, IndexNum)，dtype=uint32
        input_x2: 每条查询向量对应的实际索引个数，一维Tensor，shape=(8,)，dtype=uint32
        input_x3: 底库向量，四维Tensor，shape=(M/16, Dim/16, 16, 16)，或者二维Tensor，shape=(M, Dim)，dtype=float16
        input_x4: 映射表，一维Tensor，shape=(L,)，dtype=float32
        output_y0: 距离结果矩阵，二维Tensor，shape=(N, IndexNum)，dtype=float32
    """
    obj = DistanceFlatIpByIdx(input_x0, input_x1, input_x2, input_x3, output_y0, kernel_name)
    obj.set_lookup_table(input_x4)
    tik_instance = obj.get_tik_instance()

    return tik_instance




