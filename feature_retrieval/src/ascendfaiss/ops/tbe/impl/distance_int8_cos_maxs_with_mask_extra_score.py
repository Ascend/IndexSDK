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
from mxIndex_impl.distance_int8_cos_maxs_with_mask import DistanceInt8CosMaxsWithMask


def distance_int8_cos_maxs_with_mask_extra_score(input_queries,
                           input_mask,
                           input_centroids,
                           input_queries_m,
                           input_centroids_m,
                           input_actual_num,
                           input_extra_score,
                           output_distances,
                           output_max_dist,
                           output_flag,
                           kernel_name="distance_int8_cos_maxs_with_mask_extra_score"):
   
    distance_cos_maxs_with_mask_extra_score = DistanceInt8CosMaxsWithMask(input_queries, input_mask,
        input_centroids, input_queries_m, input_centroids_m, input_actual_num, output_distances, output_max_dist,
        output_flag, kernel_name)
    distance_cos_maxs_with_mask_extra_score.set_extra_score(input_extra_score)
    tik_instance = distance_cos_maxs_with_mask_extra_score.get_tik_instance()
    return tik_instance
