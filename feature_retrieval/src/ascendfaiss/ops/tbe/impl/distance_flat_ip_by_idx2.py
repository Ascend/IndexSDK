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
from mxIndex_impl.distance_flat_ip_common import DistanceFlatIpByIdx2


def distance_flat_ip_by_idx2(input_x0, input_x1, input_x2, output_y0, kernel_name="distance_flat_ip_by_idx2"):
    """
    算子接口函数
    Args:
        input_x0: 查询向量的索引，一维Tensor，shape=(N,)，dtype=uint32。
        input_x1: 待比较向量的位置信息，一维Tensor，shape=(8,)，dtype=uint32，[0]表示起始序号，[1]表示向量条数。
        input_x2: 底库向量，四维Tensor，shape=(M/Z_H, Dim/16, Z_H, 16)，dtype=float16
        output_y0: 距离结果矩阵，一维Tensor，shape=(N*M,)，dtype=float32，实际的内存大小是N*input_x1[1]。
        约束：底库向量的起始序号必须是Z_H的整数倍。
    """
    obj = DistanceFlatIpByIdx2(input_x0, input_x1, input_x2, output_y0, kernel_name)
    tik_instance = obj.get_tik_instance()

    return tik_instance


