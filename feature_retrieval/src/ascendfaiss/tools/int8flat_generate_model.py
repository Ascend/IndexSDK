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
import sys
import argparse
from multiprocessing import Pool

import common as utils
from common import OpJsonGenerator

# number of norm code per time in int8_generate_model
NORM_CODE_NUM = 16384
MASK_BIT_NUM = 8
ACTUAL_NUM_LEN = 8
FLAG_SIZE = 16
TRANSFER_ROW = 256
CUBE_ALIGN = 16
INT8_CUBU_ALIGN = 32
BURST_LEN = 64
BURST_RESULT_SIZE = 2
CODE_NUM_LIST = [16384, 32768, 65536, 131072, 262144]
JSON_FILE = "{}.json"


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
    utils.op_common_parse(parser, "-code", "code_num", 262144, int, "size of code block")
    return parser.parse_args()


def generate_distance_int8_cos_maxs_filter_json(core_num, query_num, dim, file_path, code_num_list):
    centroids_num_each_loop = min(min((48 // min(48, query_num)) * 512, 1024), 512 * 1024 // dim)
    int8_cos_maxs_obj = []
    for code_num in code_num_list:
        generator = OpJsonGenerator("DistanceInt8CosMaxsFilter")
        generator.add_input("ND", [query_num, dim], "int8")
        generator.add_input("ND", [query_num, (code_num + 7) // 8], "uint8")
        generator.add_input("ND", [code_num // 16, dim // 32, 16, 32], "int8")
        generator.add_input("ND", [(query_num + 15) // 16 * 16], "float16")
        generator.add_input("ND", [code_num], "float16")
        generator.add_input("ND", [core_num, 8], "uint32")
        generator.add_input("ND", [query_num, (centroids_num_each_loop + 63) // 64 * 2], "float16")
        generator.add_output("ND", [query_num, code_num], "float16")
        generator.add_output("ND", [query_num, (code_num + 63) // 64 * 2], "float16")
        generator.add_output("ND", [16, 16], "uint16")
        obj = generator.generate_obj()
        int8_cos_maxs_obj.append(obj)
    utils.generate_op_config(int8_cos_maxs_obj, file_path)


def generate_distance_int8_cos_maxs_json(core_num, query_num, dim, file_path, code_num_list):
    # write dist_int8_cos_maxs json
    int8_cos_maxs_obj = []
    for code_num in code_num_list:
        generator = OpJsonGenerator("DistanceInt8CosMaxs")
        generator.add_input("ND", [query_num, dim], "int8")
        generator.add_input("ND", [query_num, (code_num + 7) // 8], "uint8")
        generator.add_input("ND", [code_num // 16, dim // 32, 16, 32], "int8")
        generator.add_input("ND", [(query_num + 15) // 16 * 16], "float16")
        generator.add_input("ND", [code_num], "float16")
        generator.add_input("ND", [core_num, 8], "uint32")
        generator.add_output("ND", [query_num, code_num], "float16")
        generator.add_output("ND", [query_num, (code_num + 63) // 64 * 2], "float16")
        generator.add_output("ND", [16, 16], "uint16")
        obj = generator.generate_obj()
        int8_cos_maxs_obj.append(obj)

    utils.generate_op_config(int8_cos_maxs_obj, file_path)


def generate_distance_int8_cos_maxs_with_mask_json(core_num, query_num, dim, file_path, code_num_list):
    # write dist_int8_cos_maxs_with_mask json
    int8_cos_maxs_with_mask_obj = []
    for code_num in code_num_list:
        generator = OpJsonGenerator("DistanceInt8CosMaxsWithMask")
        generator.add_input("ND", [query_num, dim], "int8")
        generator.add_input("ND", [query_num, (code_num + 7) // 8], "uint8")
        generator.add_input("ND", [code_num // 16, dim // 32, 16, 32], "int8")
        generator.add_input("ND", [(query_num + 15) // 16 * 16], "float16")
        generator.add_input("ND", [code_num], "float16")
        generator.add_input("ND", [core_num, 8], "uint32")
        generator.add_output("ND", [query_num, code_num], "float16")
        generator.add_output("ND", [query_num, (code_num + 63) // 64 * 2], "float16")
        generator.add_output("ND", [16, 16], "uint16")
        obj = generator.generate_obj()
        int8_cos_maxs_with_mask_obj.append(obj)

    utils.generate_op_config(int8_cos_maxs_with_mask_obj, file_path)


def generate_distance_int8_cos_maxs_with_mask_extra_score_json(core_num, query_num, dim, file_path, code_num_list):
    # write dist_int8_cos_maxs_with_mask json
    int8_cos_maxs_with_mask_obj = []
    for code_num in code_num_list:
        generator = OpJsonGenerator("DistanceInt8CosMaxsWithMaskExtraScore")
        generator.add_input("ND", [query_num, dim], "int8")
        generator.add_input("ND", [query_num, (code_num + 7) // 8], "uint8")
        generator.add_input("ND", [code_num // 16, dim // 32, 16, 32], "int8")
        generator.add_input("ND", [(query_num + 15) // 16 * 16], "float16")
        generator.add_input("ND", [code_num], "float16")
        generator.add_input("ND", [core_num, 8], "uint32")
        generator.add_input("ND", [query_num, code_num], "float16")
        generator.add_output("ND", [query_num, code_num], "float16")
        generator.add_output("ND", [query_num, (code_num + 63) // 64 * 2], "float16")
        generator.add_output("ND", [16, 16], "uint16")
        obj = generator.generate_obj()
        int8_cos_maxs_with_mask_obj.append(obj)

    utils.generate_op_config(int8_cos_maxs_with_mask_obj, file_path)


def generate_distance_int8_cos_maxs_with_share_mask_json(core_num, query_num, dim, file_path, code_num_list):
    # write dist_int8_cos_maxs_with_mask json
    int8_cos_maxs_with_share_mask_obj = []
    for code_num in code_num_list:
        generator = OpJsonGenerator("DistanceInt8CosMaxsWithMask")
        generator.add_input("ND", [query_num, dim], "int8")
        generator.add_input("ND", [1, (code_num + 7) // 8], "uint8")
        generator.add_input("ND", [code_num // 16, dim // 32, 16, 32], "int8")
        generator.add_input("ND", [(query_num + 15) // 16 * 16], "float16")
        generator.add_input("ND", [code_num], "float16")
        generator.add_input("ND", [core_num, 8], "uint32")
        generator.add_output("ND", [query_num, code_num], "float16")
        generator.add_output("ND", [query_num, (code_num + 63) // 64 * 2], "float16")
        generator.add_output("ND", [16, 16], "uint16")
        obj = generator.generate_obj()
        int8_cos_maxs_with_share_mask_obj.append(obj)
    utils.generate_op_config(int8_cos_maxs_with_share_mask_obj, file_path)


def generate_distance_int8_cos_maxs_with_share_mask_extra_score_json(core_num, query_num,
                                                                     dim, file_path, code_num_list):
    # write dist_int8_cos_maxs_with_mask json
    int8_cos_maxs_with_mask_obj = []
    for code_num in code_num_list:
        generator = OpJsonGenerator("DistanceInt8CosMaxsWithMaskExtraScore")
        generator.add_input("ND", [query_num, dim], "int8")
        generator.add_input("ND", [1, (code_num + 7) // 8], "uint8")
        generator.add_input("ND", [code_num // 16, dim // 32, 16, 32], "int8")
        generator.add_input("ND", [(query_num + 15) // 16 * 16], "float16")
        generator.add_input("ND", [code_num], "float16")
        generator.add_input("ND", [core_num, 8], "uint32")
        generator.add_input("ND", [query_num, code_num], "float16")
        generator.add_output("ND", [query_num, code_num], "float16")
        generator.add_output("ND", [query_num, (code_num + 63) // 64 * 2], "float16")
        generator.add_output("ND", [16, 16], "uint16")
        obj = generator.generate_obj()
        int8_cos_maxs_with_mask_obj.append(obj)
    utils.generate_op_config(int8_cos_maxs_with_mask_obj, file_path)


def generate_distance_int8_l2_mins_json(core_num, query_num, dim, file_path, code_num_list):
    # write dist_compute_int8_l2_mins json
    int8_l2_mins_obj = []
    for code_num in code_num_list:
        generator = OpJsonGenerator("DistanceInt8L2Mins")
        generator.add_input("ND", [query_num, dim], "int8")
        generator.add_input("ND", [query_num, (code_num + 7) // 8], "uint8")
        generator.add_input("ND", [code_num // 16, dim // 32, 16, 32], "int8")
        generator.add_input("ND", [code_num], "int32")
        generator.add_input("ND", [core_num, 8], "uint32")
        generator.add_output("ND", [query_num, code_num], "float16")
        generator.add_output("ND", [query_num, (code_num + 63) // 64 * 2], "float16")
        generator.add_output("ND", [16, 16], "uint16")
        obj = generator.generate_obj()
        int8_l2_mins_obj.append(obj)

    utils.generate_op_config(int8_l2_mins_obj, file_path)


def generate_ascendc_dist_int8_flat_l2_json(core_num, search_page_sizes, dim, file_path, code_num_list):
    # write dist_compute_int8_l2_min json for 910B
    acsendc_dist_int8_flat_l2_obj = []
    for code_num in code_num_list:
        for query_num in search_page_sizes:
            generator = OpJsonGenerator("AscendcDistInt8FlatL2")
            generator.add_input("ND", [query_num, dim], "int8")
            generator.add_input("ND", [query_num, (code_num + MASK_BIT_NUM - 1) // MASK_BIT_NUM], "uint8")
            generator.add_input("ND", [code_num // CUBE_ALIGN, dim // INT8_CUBU_ALIGN, CUBE_ALIGN, INT8_CUBU_ALIGN],
                                "int8")
            generator.add_input("ND", [code_num], "int32")
            generator.add_input("ND", [core_num, ACTUAL_NUM_LEN], "uint32")
            generator.add_output("ND", [query_num, code_num], "float16")
            generator.add_output("ND", [query_num, (code_num + BURST_LEN - 1) // BURST_LEN * BURST_RESULT_SIZE],
                                 "float16")
            generator.add_output("ND", [core_num, FLAG_SIZE], "uint16")
            obj = generator.generate_obj()
            acsendc_dist_int8_flat_l2_obj.append(obj)

    utils.generate_op_config(acsendc_dist_int8_flat_l2_obj, file_path)


def generate_distance_int8_l2_mins_with_mask_json(core_num, query_num, dim, file_path, code_num_list):
    # write dist_compute_int8_l2_mins_with_mask json
    int8_l2_mins_with_mask_obj = []
    for code_num in code_num_list:
        generator = OpJsonGenerator("DistanceInt8L2MinsWithMask")
        generator.add_input("ND", [query_num, dim], "int8")
        generator.add_input("ND", [query_num, (code_num + 7) // 8], "uint8")
        generator.add_input("ND", [code_num // 16, dim // 32, 16, 32], "int8")
        generator.add_input("ND", [code_num], "int32")
        generator.add_input("ND", [core_num, 8], "uint32")
        generator.add_output("ND", [query_num, code_num], "float16")
        generator.add_output("ND", [query_num, (code_num + 63) // 64 * 2], "float16")
        generator.add_output("ND", [16, 16], "uint16")
        obj = generator.generate_obj()
        int8_l2_mins_with_mask_obj.append(obj)

    utils.generate_op_config(int8_l2_mins_with_mask_obj, file_path)


def generate_distance_int8_l2_mins_with_share_mask_json(core_num, query_num, dim, file_path, code_num_list):
    # write dist_compute_int8_l2_mins_with_share_mask json
    int8_l2_mins_with_share_mask_obj = []
    for code_num in code_num_list:
        generator = OpJsonGenerator("DistanceInt8L2MinsWithMask")
        generator.add_input("ND", [query_num, dim], "int8")
        generator.add_input("ND", [1, (code_num + 7) // 8], "uint8")
        generator.add_input("ND", [code_num // 16, dim // 32, 16, 32], "int8")
        generator.add_input("ND", [code_num], "int32")
        generator.add_input("ND", [core_num, 8], "uint32")
        generator.add_output("ND", [query_num, code_num], "float16")
        generator.add_output("ND", [query_num, (code_num + 63) // 64 * 2], "float16")
        generator.add_output("ND", [16, 16], "uint16")
        obj = generator.generate_obj()
        int8_l2_mins_with_share_mask_obj.append(obj)

    utils.generate_op_config(int8_l2_mins_with_share_mask_obj, file_path)


def generate_distance_int8_l2_full_mins_json(core_num, query_num, dim, file_path, code_num_list):
    # write dist_compute_int8_l2_mins json
    int8_l2_mins_obj = []
    for code_num in code_num_list:
        generator = OpJsonGenerator("DistanceInt8L2FullMins")
        generator.add_input("ND", [query_num, dim], "int8")
        generator.add_input("ND", [query_num, (code_num + 7) // 8], "uint8")
        generator.add_input("ND", [code_num // 16, dim // 32, 16, 32], "int8")
        generator.add_input("ND", [code_num], "int32")
        generator.add_input("ND", [core_num, 8], "uint32")
        generator.add_output("ND", [query_num, code_num], "float16")
        generator.add_output("ND", [query_num, (code_num + 63) // 64 * 2], "float16")
        generator.add_output("ND", [16, 16], "uint16")
        obj = generator.generate_obj()
        int8_l2_mins_obj.append(obj)

    utils.generate_op_config(int8_l2_mins_obj, file_path)


def generate_distance_int8_l2_full_mins_with_mask_json(core_num, query_num, dim, file_path, code_num_list):
    # write dist_compute_int8_l2_mins_with_mask json
    int8_l2_mins_with_mask_obj = []
    for code_num in code_num_list:
        generator = OpJsonGenerator("DistanceInt8L2FullMinsWithMask")
        generator.add_input("ND", [query_num, dim], "int8")
        generator.add_input("ND", [query_num, (code_num + 7) // 8], "uint8")
        generator.add_input("ND", [code_num // 16, dim // 32, 16, 32], "int8")
        generator.add_input("ND", [code_num], "int32")
        generator.add_input("ND", [core_num, 8], "uint32")
        generator.add_output("ND", [query_num, code_num], "float16")
        generator.add_output("ND", [query_num, (code_num + 63) // 64 * 2], "float16")
        generator.add_output("ND", [16, 16], "uint16")
        obj = generator.generate_obj()
        int8_l2_mins_with_mask_obj.append(obj)

    utils.generate_op_config(int8_l2_mins_with_mask_obj, file_path)


def generate_distance_int8_l2_full_mins_with_share_mask_json(core_num, query_num, dim, file_path, code_num_list):
    # write dist_compute_int8_l2_mins_with_share_mask json
    int8_l2_mins_with_share_mask_obj = []
    for code_num in code_num_list:
        generator = OpJsonGenerator("DistanceInt8L2FullMinsWithMask")
        generator.add_input("ND", [query_num, dim], "int8")
        generator.add_input("ND", [1, (code_num + 7) // 8], "uint8")
        generator.add_input("ND", [code_num // 16, dim // 32, 16, 32], "int8")
        generator.add_input("ND", [code_num], "int32")
        generator.add_input("ND", [core_num, 8], "uint32")
        generator.add_output("ND", [query_num, code_num], "float16")
        generator.add_output("ND", [query_num, (code_num + 63) // 64 * 2], "float16")
        generator.add_output("ND", [16, 16], "uint16")
        obj = generator.generate_obj()
        int8_l2_mins_with_share_mask_obj.append(obj)

    utils.generate_op_config(int8_l2_mins_with_share_mask_obj, file_path)


def generate_distance_int8_l2_mins_wo_query_norm_json(core_num, query_num, dim, file_path, code_num_list):
    # write dist_compute_int8_l2_mins json
    int8_l2_mins_wo_query_norm_obj = []
    for code_num in code_num_list:
        generator = OpJsonGenerator("DistanceInt8L2MinsWoQueryNorm")
        generator.add_input("ND", [query_num, dim], "int8")
        generator.add_input("ND", [query_num, (code_num + 7) // 8], "uint8")
        generator.add_input("ND", [code_num // 16, dim // 32, 16, 32], "int8")
        generator.add_input("ND", [code_num], "float16")
        generator.add_input("ND", [core_num, 8], "uint32")
        generator.add_output("ND", [query_num, code_num], "float16")
        generator.add_output("ND", [query_num, (code_num + 63) // 64 * 2], "float16")
        generator.add_output("ND", [16, 16], "uint16")
        obj = generator.generate_obj()
        int8_l2_mins_wo_query_norm_obj.append(obj)

    utils.generate_op_config(int8_l2_mins_wo_query_norm_obj, file_path)


def generate_ascendc_l2_norm_json(search_page_sizes, dim, file_path):
    ascendc_l2_norm_obj = []
    for queries_num in search_page_sizes:
        generator = OpJsonGenerator("AscendcL2Norm")
        # feature
        generator.add_input("ND", [queries_num, dim], "int8")
        # transfer
        generator.add_input("ND", [TRANSFER_ROW, CUBE_ALIGN], "float16")
        # actualNum
        generator.add_input("ND", [ACTUAL_NUM_LEN], "uint32")
        # normResult
        generator.add_output("ND", [queries_num], "float16")
        obj = generator.generate_obj()
        ascendc_l2_norm_obj.append(obj)

    utils.generate_op_config(ascendc_l2_norm_obj, file_path)


def generate_ascendc_dist_int8_flat_cos_json(core_num, search_page_sizes, dim, file_path, code_num_list):
    ascendc_dist_int8_flat_cos_obj = []
    for code_num in code_num_list:
        for queries_num in search_page_sizes:
            generator = OpJsonGenerator("AscendcDistInt8FlatCos")
            # query
            generator.add_input("ND", [queries_num, dim], "int8")
            # mask
            generator.add_input("ND", [queries_num, (code_num + MASK_BIT_NUM - 1) // MASK_BIT_NUM], "uint8")
            # base
            generator.add_input("ND", [code_num // CUBE_ALIGN, dim // INT8_CUBU_ALIGN, CUBE_ALIGN, INT8_CUBU_ALIGN],
                                "int8")
            # queryNorm
            generator.add_input("ND", [(queries_num + CUBE_ALIGN - 1) // CUBE_ALIGN * CUBE_ALIGN], "float16")
            # baseNorm
            generator.add_input("ND", [code_num], "float16")
            # actualNum
            generator.add_input("ND", [core_num, ACTUAL_NUM_LEN], "uint32")
            # dist
            generator.add_output("ND", [queries_num, code_num], "float16")
            # distMax
            generator.add_output("ND", [queries_num, (code_num + BURST_LEN - 1) // BURST_LEN * BURST_RESULT_SIZE],
                                 "float16")
            # flag
            generator.add_output("ND", [core_num, FLAG_SIZE], "uint16")
            obj = generator.generate_obj()
            ascendc_dist_int8_flat_cos_obj.append(obj)

    utils.generate_op_config(ascendc_dist_int8_flat_cos_obj, file_path)


def generate_910b_int8_offline_model(args, config_path, core_num, soc_version, code_num_list):
    map_args = []
    dim = args.dim
    process_id = args.process_id
    op_name_ = "ascendc_l2_norm_d{}_pid{}".format(dim, process_id)
    file_path_ = os.path.join(config_path, JSON_FILE.format(op_name_))
    generate_ascendc_l2_norm_json([NORM_CODE_NUM], dim, file_path_)
    map_args.append((op_name_, soc_version))

    search_page_sizes = (128, 112, 96, 80, 64, 48, 36, 32, 24, 18, 16, 12, 8, 6, 4, 2, 1)
    op_name_ = "ascendc_dist_int8_flat_cos_d{}_pid{}".format(dim, process_id)
    file_path_ = os.path.join(config_path, JSON_FILE.format(op_name_))
    generate_ascendc_dist_int8_flat_cos_json(core_num, search_page_sizes, dim, file_path_, code_num_list)
    map_args.append((op_name_, soc_version))

    search_page_sizes = (64, 48, 36, 32, 24, 18, 16, 12, 8, 6, 4, 2, 1)
    op_name_ = "ascendc_dist_int8_flat_l2_d{}_pid{}".format(dim, process_id)
    file_path_ = os.path.join(config_path, JSON_FILE.format(op_name_))
    generate_ascendc_dist_int8_flat_l2_json(core_num, search_page_sizes, dim, file_path_, code_num_list)
    map_args.append((op_name_, soc_version))

    utils.run_generate_model_task(args, map_args)


def append_int8_cos_maxs_extra_score_op(extra_score_op_args):
    map_args, core_num, page_size, dim, code_num_list, process_id, config_path, soc_version = extra_score_op_args
    int8_cos_maxs_with_mask_extra_score_op_name = "int8_flat_cos_maxs_with_mask_extra_score_op{}_pid{}"
    int8_cos_maxs_with_share_mask_extra_score_op_name = "int8_flat_cos_maxs_with_share_mask_extra_score_op{}_pid{}"
    int8_cos_maxs_with_mask_op_name = "int8_flat_cos_maxs_with_mask_op{}_pid{}"
    int8_cos_maxs_with_share_mask_op_name = "int8_flat_cos_maxs_with_share_mask_op{}_pid{}"
    op_name_ = int8_cos_maxs_with_mask_extra_score_op_name.format(page_size, process_id)
    file_path_ = os.path.join(config_path, JSON_FILE.format(op_name_))
    generate_distance_int8_cos_maxs_with_mask_extra_score_json(core_num, page_size, dim, file_path_, code_num_list)
    map_args.append((op_name_, soc_version))

    op_name_ = int8_cos_maxs_with_share_mask_extra_score_op_name.format(page_size, process_id)
    file_path_ = os.path.join(config_path, JSON_FILE.format(op_name_))
    generate_distance_int8_cos_maxs_with_share_mask_extra_score_json(core_num,
                                                                        page_size, dim, file_path_, code_num_list)
    map_args.append((op_name_, soc_version))

    op_name_ = int8_cos_maxs_with_mask_op_name.format(page_size, process_id)
    file_path_ = os.path.join(config_path, JSON_FILE.format(op_name_))
    generate_distance_int8_cos_maxs_with_mask_json(core_num, page_size, dim, file_path_, code_num_list)
    map_args.append((op_name_, soc_version))

    op_name_ = int8_cos_maxs_with_share_mask_op_name.format(page_size, process_id)
    file_path_ = os.path.join(config_path, JSON_FILE.format(op_name_))
    generate_distance_int8_cos_maxs_with_share_mask_json(core_num, page_size, dim, file_path_, code_num_list)
    map_args.append((op_name_, soc_version))


def generate_normal_json(map_args, args, config_path, soc_version):
    core_num = utils.get_core_num_by_npu_type(args.core_num, args.npu_type)
    dim = args.dim
    process_id = args.process_id
    int8_l2_mins_op_name = "int8_flat_l2_mins_op{}_pid{}"
    int8_l2_mins_with_mask_op_name = "int8_flat_l2_mins_with_mask_op{}_pid{}"
    int8_l2_mins_with_share_mask_op_name = "int8_flat_l2_mins_with_share_mask_op{}_pid{}"
    int8_l2_full_mins_op_name = "int8_flat_l2_full_mins_op{}_pid{}"
    int8_l2_full_mins_with_mask_op_name = "int8_flat_l2_full_mins_with_mask_op{}_pid{}"
    int8_l2_full_mins_with_share_mask_op_name = "int8_flat_l2_full_mins_with_share_mask_op{}_pid{}"
    int8_cos_maxs_op_name = "int8_flat_cos_maxs_op{}_pid{}"
    int8_cos_maxs_filter_op_name = "int8_flat_cos_maxs_filter_op{}_pid{}"
    int8_l2_mins_wo_query_norm_op_name = "int8_flat_l2_mins_wo_query_norm_op{}_pid{}"
    search_page_sizes = (64, 48, 36, 32, 24, 18, 16, 12, 8, 6, 4, 2, 1)
    search_page_sizes_int8_cos = (224, 192, 128, 112, 96, 64, 48, 36, 32, 24, 18, 16, 12, 8, 6, 4, 2, 1)
    search_page_sizes_wo_query_norm = (128, 96, 64, 48, 36, 32, 24, 18, 16, 12, 8, 6, 4, 2, 1)
    code_num_list = (16384, 32768, 65536, 131072, 262144)

    for page_size in search_page_sizes_int8_cos:
        op_name_ = int8_cos_maxs_op_name.format(page_size, process_id)
        file_path_ = os.path.join(config_path, JSON_FILE.format(op_name_))
        generate_distance_int8_cos_maxs_json(core_num, page_size, dim, file_path_, code_num_list)
        map_args.append((op_name_, soc_version))

    for page_size in search_page_sizes:
        op_name_ = int8_l2_mins_op_name.format(page_size, process_id)
        file_path_ = os.path.join(config_path, JSON_FILE.format(op_name_))
        generate_distance_int8_l2_mins_json(core_num, page_size, dim, file_path_, code_num_list)
        map_args.append((op_name_, soc_version))

        op_name_ = int8_l2_mins_with_mask_op_name.format(page_size, process_id)
        file_path_ = os.path.join(config_path, JSON_FILE.format(op_name_))
        generate_distance_int8_l2_mins_with_mask_json(core_num, page_size, dim, file_path_, code_num_list)
        map_args.append((op_name_, soc_version))

        op_name_ = int8_l2_mins_with_share_mask_op_name.format(page_size, process_id)
        file_path_ = os.path.join(config_path, JSON_FILE.format(op_name_))
        generate_distance_int8_l2_mins_with_share_mask_json(core_num, page_size, dim, file_path_, code_num_list)
        map_args.append((op_name_, soc_version))

        op_name_ = int8_l2_full_mins_op_name.format(page_size, process_id)
        file_path_ = os.path.join(config_path, JSON_FILE.format(op_name_))
        generate_distance_int8_l2_full_mins_json(core_num, page_size, dim, file_path_, code_num_list)
        map_args.append((op_name_, soc_version))

        op_name_ = int8_l2_full_mins_with_mask_op_name.format(page_size, process_id)
        file_path_ = os.path.join(config_path, JSON_FILE.format(op_name_))
        generate_distance_int8_l2_full_mins_with_mask_json(core_num, page_size, dim, file_path_, code_num_list)
        map_args.append((op_name_, soc_version))

        op_name_ = int8_l2_full_mins_with_share_mask_op_name.format(page_size, process_id)
        file_path_ = os.path.join(config_path, JSON_FILE.format(op_name_))
        generate_distance_int8_l2_full_mins_with_share_mask_json(core_num, page_size, dim, file_path_, code_num_list)
        map_args.append((op_name_, soc_version))

        op_name_ = int8_cos_maxs_filter_op_name.format(page_size, process_id)
        file_path_ = os.path.join(config_path, JSON_FILE.format(op_name_))
        generate_distance_int8_cos_maxs_filter_json(core_num, page_size, dim, file_path_, code_num_list)
        map_args.append((op_name_, soc_version))

        extra_score_op_args = map_args, core_num, page_size, dim, code_num_list, process_id, config_path, soc_version
        append_int8_cos_maxs_extra_score_op(extra_score_op_args)

    for page_size in search_page_sizes_wo_query_norm:
        op_name_ = int8_l2_mins_wo_query_norm_op_name.format(page_size, process_id)
        file_path_ = os.path.join(config_path, '%s.json' % op_name_)
        generate_distance_int8_l2_mins_wo_query_norm_json(core_num, page_size, dim, file_path_, code_num_list)
        map_args.append((op_name_, soc_version))


def generate_int8_offline_model():
    utils.set_env()
    args = arg_parse()
    dim = args.dim
    utils.check_param_range(dim, [64, 128, 256, 384, 512, 768, 1024], "dim")
    process_id = args.process_id
    soc_version = utils.get_soc_version_from_npu_type(args.npu_type)
    core_num = utils.get_core_num_by_npu_type(args.core_num, args.npu_type)
    work_dir = "."
    config_path = utils.get_config_path(work_dir)

    if args.npu_type.find('910B') != -1:
        valid_code_num_list = CODE_NUM_LIST
        if "-code" in sys.argv:
            code_num = args.code_num
            utils.check_param_range(code_num, valid_code_num_list, "code_num")
            valid_code_num_list = [code_num, ]

        generate_910b_int8_offline_model(args, config_path, core_num, soc_version, valid_code_num_list)
        return

    map_args = []
    generate_normal_json(map_args, args, config_path, soc_version)

    op_name_ = "int8_l2_norm_d{}_pid{}".format(dim, process_id)
    file_path_ = os.path.join(config_path, JSON_FILE.format(op_name_))
    utils.get_int8_l2_norm_json(NORM_CODE_NUM, dim, file_path_)
    map_args.append((op_name_, soc_version))
    utils.run_generate_model_task(args, map_args)


if __name__ == "__main__":
    generate_int8_offline_model()
