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
from mxIndex_impl.distance_int8_l2_full_common import DistanceInt8L2FullMins


def distance_int8_l2_full_mins(input_queries, input_mask, input_centroids,
                          input_precomputed, input_actual_num,
                          output_dist, output_min_dist, output_flag,
                          kernel_name="distance_int8_l2_full_mins"):
    obj = DistanceInt8L2FullMins(input_queries,
                             input_mask,
                             input_centroids,
                             input_precomputed,
                             input_actual_num,
                             output_dist,
                             output_min_dist,
                             output_flag,
                             False,
                             kernel_name)
    tik_instance = obj.get_tik_instance()
    
    return tik_instance