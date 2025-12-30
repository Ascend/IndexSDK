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
from mxIndex_impl.distance_flat_ip_common import DistanceFilter


def distance_filter(input_x0, input_x1, input_x2, input_x3, 
                    output_x0, output_x1, output_x2, 
                    kernel_name="distance_filter"):
    """
    算子接口函数
    Args:
        input_x0: 距离矩阵，二维Tensor，shape=(N, M)，dtype=float32，
                  N为查询个数，M为每个查询对应的距离值个数，取一个比较大的值。
        input_x1: 过滤阈值，一维Tensor，shape=(8,)，dtype=float32
        input_x2: 每个查询实际对应的距离值个数，一维Tensor，shape=(8,)，dtype=uint32
        input_x3: 距离矩阵中第一个距离值对应的序号，默认连续，一维Tensor，shape=(8,)，dtype=uint32
        output_y0: 过滤后的距离矩阵，二维Tensor，shape=(N, M)，dtype=float32，N,M含义与input_x0相同
        output_y1: 过滤后的序号矩阵，二维Tensor，shape=(N,M)，dtype=int32，N,M含义与input_x0相同
        output_y2: 每个查询满足阈值过滤条件的距离值个数，shape=(N*8,)，dtype=uint32，实际值保存在[0, 8, 16...]
    """
    obj = DistanceFilter(input_x0, input_x1, input_x2, input_x3, output_x0, output_x1, output_x2, kernel_name)
    tik_instance = obj.get_tik_instance()

    return tik_instance
