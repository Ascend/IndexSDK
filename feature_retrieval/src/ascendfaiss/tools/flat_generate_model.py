#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
-------------------------------------------------------------------------
This file is part of the IndexSDK project.
Copyright (c) 2025 Huawei Technologies Co.,Ltd.

IndexSDK is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:

         http://license.coscl.org.cn/MulanPSL2

THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details.
-------------------------------------------------------------------------
"""
import os
import argparse
import traceback
from multiprocessing import Pool

import common as utils
from common import OpJsonGenerator

# number of code per time in distance_compute_flat_min64
_CODE_NUM = 16384 * 16
_HUGE_BASE = 20000000
_TABLE_LEN = 10048
_IDX_BLOCK = 64
_910B_DIM_LIST = [32, 64, 128, 256, 384, 512, 768, 1024, 1408, 1536, 2048, 3072, 3584, 4096]
_DIM_LIST = [32, 64, 128, 256, 384, 512, 1024]
_EXTEND_DIM_LIST = [1408, 1536, 2048, 3072, 3584, 4096]
_TS_DIM_LIST = [64, 128, 256, 384, 512, 768, 1024]
_Z_DEFAULT = 16
_Z_SOC = 2
_BURST_LEN = 64
_BURST_LEN_LOW = 32

FORMAT = "format"
SHAPE = "shape"
TYPE = "type"
ND = "ND"


def arg_parse():
    """
    Parse arguements to the operator model
    """

    parser = argparse.ArgumentParser(
        description="generate aicore operator model")

    utils.op_common_parse(parser, "--cores", "core_num", 2, int, "Core number")
    utils.op_common_parse(parser, "-d", "dim", 512, int, "Feature dimension")
    utils.op_common_parse(parser, "-p", "process_id", 0, int, "Number of process_id")
    utils.op_common_parse(parser, "-pool", "pool_size", 10, int, "Number of pool_size")
    utils.op_common_parse(parser, "-t", 'npu_type', "310", str,
            "NPU type, 310 / 310P / 910B1 / 910B2 / 910B3 / 910B4 / 910_{NPU Name}. 310 by default")
    return parser.parse_args()


def generate_distance_flat_l2_mins_json(core_num, search_page_sizes, dim, zregion_height, file_path):
    # write dist_compute_flat_mins json
    dist_compute_flat_mins_obj = []
    for query_num in search_page_sizes:
        burst_len = _BURST_LEN_LOW if query_num > 48 else _BURST_LEN
        generator = OpJsonGenerator("DistanceComputeFlatMin64")
        generator.add_input("ND", [query_num, dim], "float16")
        generator.add_input("ND", [query_num, (_CODE_NUM + 7) // 8], "uint8")
        generator.add_input("ND", [_CODE_NUM // zregion_height, dim // 16, zregion_height, 16], "float16")
        generator.add_input("ND", [_CODE_NUM], "float16")
        generator.add_input("ND", [core_num, 8], "uint32")
        generator.add_output("ND", [query_num, _CODE_NUM], "float16")
        generator.add_output("ND", [query_num, (_CODE_NUM + burst_len-1) // burst_len * 2], "float16")
        generator.add_output("ND", [16, 16], "uint16")
        obj = generator.generate_obj()
        dist_compute_flat_mins_obj.append(obj)

    utils.generate_op_config(dist_compute_flat_mins_obj, file_path)


def generate_distance_flat_ip_json(core_num, search_page_sizes, dim, zregion_height, file_path):
    # write dist_compute_flat_mins json
    dist_flat_ip_obj = []
    for query_num in search_page_sizes:
        generator = OpJsonGenerator("DistanceFlatIp")
        generator.add_input("ND", [query_num, dim], "float16")
        generator.add_input("ND", [_HUGE_BASE // zregion_height, dim // 16, zregion_height, 16], "float16")
        generator.add_input("ND", [8], "uint32")
        generator.add_output("ND", [query_num * _HUGE_BASE], "float32")
        obj = generator.generate_obj()
        
        dist_flat_ip_obj.append(obj)

    utils.generate_op_config(dist_flat_ip_obj, file_path)


def generate_distance_flat_ip_with_table_json(core_num, search_page_sizes, dim, zregion_height, file_path):
    # write dist_compute_flat_mins json

    dist_flat_ip_with_table_obj = []
    for queries_num in search_page_sizes:
        generator = OpJsonGenerator("DistanceFlatIpWithTable")
        generator.add_input("ND", [queries_num, dim], "float16")
        generator.add_input("ND", [_HUGE_BASE // zregion_height, dim // 16, zregion_height, 16], "float16")
        generator.add_input("ND", [8], "uint32")
        generator.add_input("ND", [_TABLE_LEN], "float32")
        generator.add_output("ND", [queries_num * _HUGE_BASE], "float32")
        obj = generator.generate_obj()
        dist_flat_ip_with_table_obj.append(obj)

    utils.generate_op_config(dist_flat_ip_with_table_obj, file_path)


def generate_distance_flat_ip_maxs_json(core_num, search_page_sizes, dim, zregion_height, file_path):
    # write dist_compute_flat_mins json
    dist_flat_ip_maxs_obj = []
    burst_len = 64
    for queries_num in search_page_sizes:
        generator = OpJsonGenerator("DistanceFlatIPMaxs")
        generator.add_input("ND", [queries_num, dim], "float16")
        generator.add_input("ND", [queries_num, (_CODE_NUM + 7) // 8], "uint8")
        generator.add_input("ND", [_CODE_NUM // zregion_height, dim // 16, zregion_height, 16], "float16")
        generator.add_input("ND", [core_num, 8], "uint32")
        generator.add_output("ND", [queries_num, _CODE_NUM], "float16")
        generator.add_output("ND", [queries_num, (_CODE_NUM + burst_len - 1) // burst_len * 2], "float16")
        generator.add_output("ND", [core_num, 16], "uint16")
        obj = generator.generate_obj()
        dist_flat_ip_maxs_obj.append(obj)

    utils.generate_op_config(dist_flat_ip_maxs_obj, file_path)


def generate_distance_flat_ip_maxs_with_mask_common_json(op_name, core_num, search_page_sizes, dim, zregion_height,
                                                         shared, with_score, file_path):
    dist_flat_ip_maxs_with_mask_obj = []
    burst_len = 64
    for queries_num in search_page_sizes:
        generator = OpJsonGenerator(op_name)
        generator.add_input("ND", [queries_num, dim], "float16")
        generator.add_input("ND", [_CODE_NUM // zregion_height, dim // 16, zregion_height, 16], "float16")
        generator.add_input("ND", [core_num, 8], "uint32")
        if shared:
            generator.add_input("ND", [1, (_CODE_NUM + 7) // 8], "uint8")
        else:
            generator.add_input("ND", [queries_num, (_CODE_NUM + 7) // 8], "uint8")
        if with_score:
            generator.add_input("ND", [queries_num, _CODE_NUM], "float16")
        generator.add_output("ND", [queries_num, _CODE_NUM], "float16")
        generator.add_output("ND", [queries_num, (_CODE_NUM + burst_len - 1) // burst_len * 2], "float16")
        generator.add_output("ND", [core_num, 16], "uint16")
        obj = generator.generate_obj()
        dist_flat_ip_maxs_with_mask_obj.append(obj)

    utils.generate_op_config(dist_flat_ip_maxs_with_mask_obj, file_path)


def generate_distance_flat_ip_maxs_with_mask_and_scale_json(op_name, core_num, search_page_sizes, dim, zregion_height,
                                                            with_score, file_path):
    dist_flat_ip_maxs_with_mask_scale_obj = []
    burst_len = 64
    for queries_num in search_page_sizes:
        generator = OpJsonGenerator(op_name)
        generator.add_input("ND", [queries_num, dim], "float16")
        generator.add_input("ND", [_CODE_NUM // zregion_height, dim // 32, zregion_height, 32], "int8")
        generator.add_input("ND", [core_num, 8], "uint32")
        generator.add_input("ND", [queries_num, (_CODE_NUM + 7) // 8], "uint8")
        if with_score:
            generator.add_input("ND", [queries_num, _CODE_NUM], "float16")
        generator.add_input("ND", [dim, ], "float16")
        generator.add_output("ND", [queries_num, _CODE_NUM], "float16")
        generator.add_output("ND", [queries_num, (_CODE_NUM + burst_len - 1) // burst_len * 2], "float16")
        generator.add_output("ND", [core_num, 16], "uint16")
        obj = generator.generate_obj()
        dist_flat_ip_maxs_with_mask_scale_obj.append(obj)

    utils.generate_op_config(dist_flat_ip_maxs_with_mask_scale_obj, file_path)


def generate_distance_flat_ip_maxs_with_share_mask_json(core_num, search_page_sizes, dim, zregion_height, file_path):
    op_name = "DistanceFlatIPMaxsWithMask"
    shared = True
    generate_distance_flat_ip_maxs_with_mask_common_json(
        op_name, core_num, search_page_sizes, dim, zregion_height, shared, False, file_path)


def generate_distance_flat_ip_maxs_with_nonshare_mask_json(core_num, search_page_sizes, dim, zregion_height, file_path):
    op_name = "DistanceFlatIPMaxsWithMask"
    shared = False
    generate_distance_flat_ip_maxs_with_mask_common_json(
        op_name, core_num, search_page_sizes, dim, zregion_height, shared, False, file_path)


def generate_distance_flat_ip_maxs_with_share_and_score_json(core_num, search_page_sizes, dim, zregion_height, file_path):
    op_name = "DistanceFlatIPMaxsWithExtraScore"
    shared = True
    generate_distance_flat_ip_maxs_with_mask_common_json(
        op_name, core_num, search_page_sizes, dim, zregion_height, shared, True, file_path)


def generate_distance_flat_ip_maxs_with_nonshare_and_score_json(core_num, search_page_sizes, dim, zregion_height, file_path):
    op_name = "DistanceFlatIPMaxsWithExtraScore"
    shared = False
    generate_distance_flat_ip_maxs_with_mask_common_json(
        op_name, core_num, search_page_sizes, dim, zregion_height, shared, True, file_path)


def generate_distance_flat_ip_maxs_with_score_and_scale_json(core_num, search_page_sizes, dim, zregion_height, file_path):
    op_name = "DistanceFlatIPMaxsWithScale"
    score = True
    generate_distance_flat_ip_maxs_with_mask_and_scale_json(
        op_name, core_num, search_page_sizes, dim, zregion_height, score, file_path)


def generate_distance_flat_ip_maxs_with_nonscore_and_scale_json(core_num, search_page_sizes, dim, zregion_height, file_path):
    op_name = "DistanceFlatIPMaxsNoScoreWithScale"
    score = False
    generate_distance_flat_ip_maxs_with_mask_and_scale_json(
        op_name, core_num, search_page_sizes, dim, zregion_height, score, file_path)


def generate_distance_flat_ip_maxs_batch_json(core_num, search_page_sizes, dim, zregion_height, file_path):
    # write dist_compute_flat_mins json
    dist_flat_ip_maxs_batch_obj = []
    for queries_num in search_page_sizes:
        generator = OpJsonGenerator("DistanceFlatIPMaxsBatch")
        generator.add_input("ND", [queries_num, dim], "float16")
        generator.add_input("ND", [queries_num, (_CODE_NUM + 7) // 8], "uint8")
        generator.add_input("ND", [_CODE_NUM // zregion_height, dim // 16, zregion_height, 16], "float16")
        generator.add_input("ND", [core_num, 8], "uint32")
        generator.add_output("ND", [queries_num, _CODE_NUM], "float16")
        generator.add_output("ND", [queries_num, (_CODE_NUM + 31) // 32 * 2], "float16")
        generator.add_output("ND", [core_num, 16], "uint16")
        obj = generator.generate_obj()
        dist_flat_ip_maxs_batch_obj.append(obj)
    utils.generate_op_config(dist_flat_ip_maxs_batch_obj, file_path)


def generate_distance_flat_ip_by_idx_json(core_num, compute_distance_by_idx_batches, dim, zregion_height, file_path):
    # write dist_compute_flat_mins json
    distance_flat_ip_by_idx_obj = []
    for queries_num in compute_distance_by_idx_batches:
        generator = OpJsonGenerator("DistanceFlatIpByIdx")
        generator.add_input("ND", [queries_num, dim], "float16")
        generator.add_input("ND", [queries_num, _IDX_BLOCK], "uint32")
        generator.add_input("ND", [8], "uint32")
        generator.add_input("ND", [_HUGE_BASE // zregion_height, dim // 16, zregion_height, 16], "float16")
        generator.add_output("ND", [queries_num, _IDX_BLOCK], "float32")
        obj = generator.generate_obj()
        distance_flat_ip_by_idx_obj.append(obj)

    utils.generate_op_config(distance_flat_ip_by_idx_obj, file_path)

    
def generate_distance_flat_ip_by_idx2_json(core_num, compute_distance_by_idx_batches, dim, zregion_height, file_path):
    # write dist_compute_flat_mins json
    distance_flat_ip_by_idx2_obj = []
    for queries_num in compute_distance_by_idx_batches:
        generator = OpJsonGenerator("DistanceFlatIpByIdx2")
        generator.add_input("ND", [queries_num], "uint32")
        generator.add_input("ND", [8], "uint32")
        generator.add_input("ND", [_HUGE_BASE // zregion_height, dim // 16, zregion_height, 16], "float16")
        generator.add_output("ND", [queries_num * _HUGE_BASE], "float32")
        obj = generator.generate_obj()
        distance_flat_ip_by_idx2_obj.append(obj)

    utils.generate_op_config(distance_flat_ip_by_idx2_obj, file_path)


def generate_distance_flat_ip_by_idx_with_table_json(core_num, compute_distance_by_idx_batches, dim, zregion_height,
                                                     file_path):
    # write dist_compute_flat_mins json
    distance_flat_ip_by_idx_with_table_obj = []
    for queries_num in compute_distance_by_idx_batches:
        generator = OpJsonGenerator("DistanceFlatIpByIdxWithTable")
        generator.add_input("ND", [queries_num, dim], "float16")
        generator.add_input("ND", [queries_num, _IDX_BLOCK], "uint32")
        generator.add_input("ND", [8], "uint32")
        generator.add_input("ND", [_HUGE_BASE // zregion_height, dim // 16, zregion_height, 16], "float16")
        generator.add_input("ND", [_TABLE_LEN], "float32")
        generator.add_output("ND", [queries_num, _IDX_BLOCK], "float32")
        obj = generator.generate_obj()

        distance_flat_ip_by_idx_with_table_obj.append(obj)

    utils.generate_op_config(distance_flat_ip_by_idx_with_table_obj, file_path)


def generate_distance_filter_json(search_page_sizes, file_path):
    # write distance_filter json
    dist_flat_distance_compute_threshold_obj = []
    for queries_num in search_page_sizes:
        generator = OpJsonGenerator("DistanceFilter")
        generator.add_input("ND", [queries_num, _HUGE_BASE], "float32")
        generator.add_input("ND", [8], "float32")
        generator.add_input("ND", [8], "uint32")
        generator.add_input("ND", [8], "uint32")
        generator.add_output("ND", [queries_num, _HUGE_BASE], "float32")
        generator.add_output("ND", [queries_num, _HUGE_BASE], "int32")
        generator.add_output("ND", [queries_num * 8], "uint32")
        obj = generator.generate_obj()
        dist_flat_distance_compute_threshold_obj.append(obj)
    utils.generate_op_config(dist_flat_distance_compute_threshold_obj, file_path)


def generate_flat_ip_extend_json(map_args, args, zregion_height, config_path, soc_version):
    core_num = utils.get_core_num_by_npu_type(args.core_num, args.npu_type)
    process_id = args.process_id
    dim = args.dim
 
    flat_ip_max_op_name = "distance_flat_ip_maxs_op_pid{}"
    search_page_sizes = (48, 36, 32, 30, 24, 18, 16, 12, 8, 6, 4, 2, 1)
    op_name_ = flat_ip_max_op_name.format(process_id)
    file_path_ = os.path.join(config_path, '{}.json'.format(op_name_))
    generate_distance_flat_ip_maxs_json(core_num, search_page_sizes, dim, zregion_height, file_path_)
    map_args.append((op_name_, soc_version))
 
    search_optimize_page_sizes = (128, 64,)
    flat_ip_max_batch_op_name = "distance_flat_ip_maxs_batch_op_pid{}"
    op_name_ = flat_ip_max_batch_op_name.format(process_id)
    file_path_ = os.path.join(config_path, '{}.json'.format(op_name_))
    generate_distance_flat_ip_maxs_batch_json(core_num, search_optimize_page_sizes, dim, zregion_height, file_path_)
    map_args.append((op_name_, soc_version))


def generate_flat_l2_extend_json(map_args, args, zregion_height, config_path, soc_version):
    core_num = utils.get_core_num_by_npu_type(args.core_num, args.npu_type)
    process_id = args.process_id
    dim = args.dim
    flat_l2_max_op_name = "distance_flat_l2_maxs_op_pid{}"
    if dim == 768:
        search_page_sizes = (96, 80, 64, 48, 36, 32, 30, 24, 18, 16, 12, 8, 6, 4, 2, 1)
    else:
        search_page_sizes = (48, 36, 32, 30, 24, 18, 20, 16, 12, 8, 6, 4, 2, 1)
    op_name_ = flat_l2_max_op_name.format(process_id)
    file_path_ = os.path.join(config_path, '{}.json'.format(op_name_))
    generate_distance_flat_l2_mins_json(core_num, search_page_sizes, dim, zregion_height, file_path_)
    map_args.append((op_name_, soc_version))


def generate_scale_json(map_args, args, soc_version, core_num, config_path, search_page_sizes, zregion_height):
    flat_ip_max_with_score_and_scale_op_name = "distance_flat_max_with_score_and_scale_op_pid{}"
    flat_ip_max_with_nonscore_and_scale_op_name = "distance_flat_max_nonscore_and_scale_op_pid{}"

    process_id = args.process_id
    dim = args.dim
    op_name_ = flat_ip_max_with_score_and_scale_op_name.format(process_id)
    file_path_ = os.path.join(config_path, '{}.json'.format(op_name_))
    generate_distance_flat_ip_maxs_with_score_and_scale_json(core_num, search_page_sizes,
        dim, zregion_height, file_path_)
    map_args.append((op_name_, soc_version))

    op_name_ = flat_ip_max_with_nonscore_and_scale_op_name.format(process_id)
    file_path_ = os.path.join(config_path, '{}.json'.format(op_name_))
    generate_distance_flat_ip_maxs_with_nonscore_and_scale_json(core_num, search_page_sizes,
        dim, zregion_height, file_path_)
    map_args.append((op_name_, soc_version)) 


def generate_normal_json(map_args, args, zregion_height, config_path, soc_version):
    process_id = args.process_id
    dim = args.dim
    core_num = utils.get_core_num_by_npu_type(args.core_num, args.npu_type)
    flat_l2_max_op_name = "distance_flat_l2_maxs_op_pid{}"
    flat_ip_max_op_name = "distance_flat_ip_maxs_op_pid{}"
    flat_ip_max_batch_op_name = "distance_flat_ip_maxs_batch_op_pid{}"
    flat_ip_max_with_share_mask_op_name = "distance_flat_ip_maxs_with_share_mask_op_pid{}"
    flat_ip_max_with_nonshare_mask_op_name = "distance_flat_ip_maxs_with_nonshare_mask_op_pid{}"
    flat_ip_max_with_share_mask_extra_score_op_name = "distance_flat_max_with_share_mask_extra_score_op_pid{}"
    flat_ip_max_with_nonshare_mask_extra_score_op_name = "distance_flat_max_with_nonshare_mask_extra_score_op_pid{}"

    search_page_sizes = (48, 36, 32, 30, 24, 18, 16, 12, 8, 6, 4, 2, 1)
    # Search related operator of IP distance
    op_name_ = flat_ip_max_op_name.format(process_id)
    file_path_ = os.path.join(config_path, '{}.json'.format(op_name_))
    generate_distance_flat_ip_maxs_json(core_num, search_page_sizes, dim, zregion_height, file_path_)
    map_args.append((op_name_, soc_version))

    if soc_version == "Ascend310P3":
        # Search related operator of IP distance with share mask
        op_name_ = flat_ip_max_with_share_mask_op_name.format(process_id)
        file_path_ = os.path.join(config_path, '{}.json'.format(op_name_))
        generate_distance_flat_ip_maxs_with_share_mask_json(core_num, search_page_sizes,
            dim, zregion_height, file_path_)
        map_args.append((op_name_, soc_version))

        # Search related operator of IP distance with nonshare mask
        op_name_ = flat_ip_max_with_nonshare_mask_op_name.format(process_id)
        file_path_ = os.path.join(config_path, '{}.json'.format(op_name_))
        generate_distance_flat_ip_maxs_with_nonshare_mask_json(core_num, search_page_sizes,
            dim, zregion_height, file_path_)
        map_args.append((op_name_, soc_version))

        op_name_ = flat_ip_max_with_share_mask_extra_score_op_name.format(process_id)
        file_path_ = os.path.join(config_path, '{}.json'.format(op_name_))
        generate_distance_flat_ip_maxs_with_share_and_score_json(core_num, search_page_sizes,
            dim, zregion_height, file_path_)
        map_args.append((op_name_, soc_version))

        op_name_ = flat_ip_max_with_nonshare_mask_extra_score_op_name.format(process_id)
        file_path_ = os.path.join(config_path, '{}.json'.format(op_name_))
        generate_distance_flat_ip_maxs_with_nonshare_and_score_json(core_num, search_page_sizes,
            dim, zregion_height, file_path_)
        map_args.append((op_name_, soc_version))

        generate_scale_json(map_args, args, soc_version, core_num, config_path, search_page_sizes, zregion_height)

    # Search related operator of IP distance for batch 128 and 64
    search_optimize_page_sizes = (128, 64,)
    op_name_ = flat_ip_max_batch_op_name.format(process_id)
    file_path_ = os.path.join(config_path, '{}.json'.format(op_name_))
    generate_distance_flat_ip_maxs_batch_json(core_num, search_optimize_page_sizes, dim, zregion_height, file_path_)
    map_args.append((op_name_, soc_version))

   # Search related operator of L2 distance for larger batch
    search_optimize_l2_page_sizes = (96, 80, 64, 48, 36, 32, 30, 24, 18, 16, 12, 8, 6, 4, 2, 1)
    op_name_ = flat_l2_max_op_name.format(process_id)
    file_path_ = os.path.join(config_path, '{}.json'.format(op_name_))
    generate_distance_flat_l2_mins_json(core_num, search_optimize_l2_page_sizes, dim, zregion_height, file_path_)
    map_args.append((op_name_, soc_version))



def generate_soc_json(map_args, args, zregion_height, config_path, soc_version):
    process_id = args.process_id
    dim = args.dim
    core_num = utils.get_core_num_by_npu_type(args.core_num, args.npu_type)
    search_page_sizes = (256, 128, 64, 48, 32, 30, 18, 16, 8, 6, 4, 2, 1)
    compute_distance_by_idx_batches = (256, 128, 64, 48, 32, 16, 8, 6, 4, 2, 1)
    flat_ip_max_op_name = "distance_flat_ip_maxs_soc_op_pid{}"
    flat_ip_by_idx_op_name = "distance_flat_ip_by_idx_op_pid{}"
    flat_ip_by_idx_with_table_op_name = "distance_flat_ip_by_idx_with_table_op_pid{}"
    flat_ip_op_name = "distance_flat_ip_op_pid{}"
    flat_ip_with_table_op_name = "distance_flat_ip_with_table_op_pid{}"
    flat_distance_filter_op_name = "distance_filter_op_pid{}"
    flat_ip_max_batch_ilflat_op_name = "distance_flat_ip_maxs_batch_ilflat_op_pid{}"

    # ComputeDistanceByIdx underlayer operator, support batch256
    op_name_ = flat_ip_by_idx_op_name.format(process_id)
    file_path_ = os.path.join(config_path, '{}.json'.format(op_name_))
    generate_distance_flat_ip_by_idx_json(
        core_num, compute_distance_by_idx_batches, dim, zregion_height, file_path_)
    map_args.append((op_name_, soc_version))

    # ComputeDistanceByIdxWithTable underlayer operator, support batch256
    op_name_ = flat_ip_by_idx_with_table_op_name.format(process_id)
    file_path_ = os.path.join(config_path, '{}.json'.format(op_name_))
    generate_distance_flat_ip_by_idx_with_table_json(
        core_num, compute_distance_by_idx_batches, dim, zregion_height, file_path_)
    map_args.append((op_name_, soc_version))

    # Search related operator of IP distance
    op_name_ = flat_ip_max_op_name.format(process_id)
    file_path_ = os.path.join(config_path, '{}.json'.format(op_name_))
    generate_distance_flat_ip_maxs_json(core_num, search_page_sizes, dim, zregion_height, file_path_)
    map_args.append((op_name_, soc_version))

    # ComputeDistance underlayer operator
    op_name_ = flat_ip_op_name.format(process_id)
    file_path_ = os.path.join(config_path, '{}.json'.format(op_name_))
    generate_distance_flat_ip_json(core_num, search_page_sizes, dim, zregion_height, file_path_)
    map_args.append((op_name_, soc_version))

    # ComputeDistanceWithTable underlayer operator
    op_name_ = flat_ip_with_table_op_name.format(process_id)
    file_path_ = os.path.join(config_path, '{}.json'.format(op_name_))
    generate_distance_flat_ip_with_table_json(core_num, search_page_sizes, dim, zregion_height, file_path_)
    map_args.append((op_name_, soc_version))

    # distancefilter operator
    op_name_ = flat_distance_filter_op_name.format(process_id)
    file_path_ = os.path.join(config_path, '{}.json'.format(op_name_))
    generate_distance_filter_json(search_page_sizes, file_path_)
    map_args.append((op_name_, soc_version))

    # For ILFlat, use the operators of IP distance with batch 128 and 64, with normal dims, with _Z_SOC.
    search_optimize_page_sizes = (128, 64,)
    op_name_ = flat_ip_max_batch_ilflat_op_name.format(process_id)
    file_path_ = os.path.join(config_path, '{}.json'.format(op_name_))
    generate_distance_flat_ip_maxs_batch_json(core_num, search_optimize_page_sizes, dim, zregion_height, file_path_)
    map_args.append((op_name_, soc_version))


def generate_soc_ilflat_json(map_args, args, zregion_height, config_path, soc_version):
    process_id = args.process_id
    dim = args.dim
    core_num = utils.get_core_num_by_npu_type(args.core_num, args.npu_type)
    compute_distance_by_idx_batches = (64, 48, 32, 30, 18, 16, 8, 6, 4, 2, 1)
    ilflat_ip_op_name = "distance_ilflat_ip_op_pid{}"
    flat_ip_by_idx2_op_name = "distance_flat_ip_by_idx2_op_pid{}"

    # ComputeDistanceByIdx underlayer operator, support batch256
    op_name_ = flat_ip_by_idx2_op_name.format(process_id)
    file_path_ = os.path.join(config_path, '{}.json'.format(op_name_))
    generate_distance_flat_ip_by_idx2_json(
        core_num, compute_distance_by_idx_batches, dim, zregion_height, file_path_)
    map_args.append((op_name_, soc_version))


def generate_910b_flat_ip_json(core_num, search_page_sizes, dim, zregion_height, file_path):
    # write dist_compute_flat_mins json
    dist_flat_ip_obj = []
    for query_num in search_page_sizes:
        burst_len = 64
        obj = {
            "op":
                "DistanceFlatIP",
            "input_desc": [{
                FORMAT: ND,
                SHAPE: [query_num, dim],
                TYPE: "float16"
            }, {
                FORMAT: ND,
                SHAPE: [query_num, (_CODE_NUM + 7) // 8],
                TYPE: "uint8"
            }, {
                FORMAT: ND,
                SHAPE: [_CODE_NUM // zregion_height, dim // 16, zregion_height, 16],
                TYPE: "float16"
            }, {
                FORMAT: ND,
                SHAPE: [core_num, 8],
                TYPE: "uint32"
            }],
            "output_desc": [{
                FORMAT: ND,
                SHAPE: [query_num, _CODE_NUM],
                TYPE: "float16"
            }, {
                FORMAT: ND,
                SHAPE: [query_num, (_CODE_NUM + burst_len - 1) // burst_len * 2],
                TYPE: "float16"
            }, {
                FORMAT: ND,
                SHAPE: [core_num, 16],
                TYPE: "uint16"
            }]
        }
        dist_flat_ip_obj.append(obj)

    utils.generate_op_config(dist_flat_ip_obj, file_path)


def generate_910b_flat_l2_json(core_num, search_page_sizes, dim, zregion_height, file_path):
    # write dist_compute_flat_mins json
    dist_flat_l2_obj = []
    for query_num in search_page_sizes:
        burst_len = 64
        obj = {
            "op":
                "DistanceFlatL2",
            "input_desc": [{
                FORMAT: ND,
                SHAPE: [query_num, dim],
                TYPE: "float16"
            }, {
                FORMAT: "ND",
                SHAPE: [query_num, (_CODE_NUM + 7) // 8],
                TYPE: "uint8"
            }, {
                FORMAT: ND,
                SHAPE: [_CODE_NUM // zregion_height, dim // 16, zregion_height, 16],
                TYPE: "float16"
            }, {
                FORMAT: ND,
                SHAPE: [_CODE_NUM],
                TYPE: "float16"
            }, {
                FORMAT: ND,
                SHAPE: [core_num, 8],
                TYPE: "uint32"
            }],
            "output_desc": [{
                FORMAT: ND,
                SHAPE: [query_num, _CODE_NUM],
                TYPE: "float16"
            }, {
                FORMAT: ND,
                SHAPE: [query_num, (_CODE_NUM + burst_len - 1) // burst_len * 2],
                TYPE: "float16"
            }, {
                FORMAT: ND,
                SHAPE: [core_num, 16],
                TYPE: "uint16"
            }]
        }
        dist_flat_l2_obj.append(obj)

    utils.generate_op_config(dist_flat_l2_obj, file_path)


def generate_910b_flat_offline_model(map_args, args, zregion_height, config_path, soc_version):
    process_id = args.process_id
    dim = args.dim
    core_num = utils.get_core_num_by_npu_type(args.core_num, args.npu_type)
    flat_ip_op_name = "distance_flat_ip_op_pid{}"

    search_page_sizes = (128, 64, 48, 36, 32, 30, 24, 18, 16, 12, 8, 6, 4, 2, 1)
    op_name_ = flat_ip_op_name.format(process_id)
    file_path_ = os.path.join(config_path, '{}.json'.format(op_name_))
    generate_910b_flat_ip_json(core_num, search_page_sizes, dim, zregion_height, file_path_)
    map_args.append((op_name_, soc_version))

    flat_l2_op_name = "distance_flat_l2_op_pid{}"
    search_page_sizes = (96, 80, 64, 48, 36, 32, 30, 24, 18, 16, 12, 8, 6, 4, 2, 1)
    op_name_ = flat_l2_op_name.format(process_id)
    file_path_ = os.path.join(config_path, '{}.json'.format(op_name_))
    generate_910b_flat_l2_json(core_num, search_page_sizes, dim, zregion_height, file_path_)
    map_args.append((op_name_, soc_version))


def generate_flat_offline_model():
    utils.set_env()
    args = arg_parse()
    dim = args.dim
    process_id = args.process_id
    soc_version = utils.get_soc_version_from_npu_type(args.npu_type)
    core_num = utils.get_core_num_by_npu_type(args.core_num, args.npu_type)
    work_dir = "."
    map_args = []
    config_path = utils.get_config_path(work_dir)
    if args.npu_type.find('910') != -1:
        valid_dim = _910B_DIM_LIST
        utils.check_param_range(dim, valid_dim, "dim")
        generate_910b_flat_offline_model(map_args, args, _Z_DEFAULT, config_path, soc_version)
    else:
        if dim in _DIM_LIST:

            # generate normal operators of zregion_height 16
            generate_normal_json(map_args, args, _Z_DEFAULT, config_path, soc_version)
            
            # generate soc related operators of zregion_height 2
            generate_soc_json(map_args, args, _Z_SOC, config_path, soc_version)

            if dim < 1024:
                # generate normal operators of zregion_height 16
                generate_soc_ilflat_json(map_args, args, _Z_DEFAULT, config_path, soc_version)
        elif dim in _EXTEND_DIM_LIST:
            # 310P flat 不带TS
            generate_flat_ip_extend_json(map_args, args, _Z_DEFAULT, config_path, soc_version)
            generate_flat_l2_extend_json(map_args, args, _Z_DEFAULT, config_path, soc_version)
        elif dim in _TS_DIM_LIST:
            # generate normal operators of zregion_height 16
            generate_normal_json(map_args, args, _Z_DEFAULT, config_path, soc_version)
        else:
            raise ValueError(f"Given dim is invalid")

    utils.run_generate_model_task(args, map_args)


if __name__ == "__main__":
    generate_flat_offline_model()
