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

_IVFRABITQ_DIM_LIST = [128]
_CODE_NUM = 16384 * 2
_SCAN_BIT = 8
_LUT_NUM = 2 ** _SCAN_BIT


def arg_parse():
    """
    Parse arguements to the operator model
    """

    parser = argparse.ArgumentParser(
        description="generate distance_compute_ivf_sq8 operator model")
    utils.op_common_parse(parser, "--cores", 'core_num', 40, int, "Core number, 40 by default")
    utils.op_common_parse(parser, "-d", "dim", 128, int, "Feature dimension, 128 is only support now")
    utils.op_common_parse(parser, "-c", "coarse_centroid_num", 16384, int, "Number of coarse centroid")
    utils.op_common_parse(parser, "-p", "process_id", 0, int, "Number of process_id")
    utils.op_common_parse(parser, "-pool", "pool_size", 10, int, "Number of pool_size")
    utils.op_common_parse(parser, "-t", 'npu_type', "910B4", str, "NPU type, 910 series. 910B4 by default")
    return parser.parse_args()


def generate_distance_rabitq_l2_fp32_json(core_num, list_num, dim, file_path):
    search_batch_sizes = (64, 32, 16, 8, 4, 2, 1)
    dist_rabitq_l2_obj = []
    burst_len = 16 if dim > 256 else 64
    for query_num in search_batch_sizes:
        generator = OpJsonGenerator("DistanceIVFRabitqL2FP32")
        generator.add_input("ND", [1, dim], "float32")
        generator.add_input("ND", [query_num, dim], "float32")
        generator.add_input("ND", [list_num, dim], "float32")
        generator.add_input("ND", [core_num], "uint32")
        generator.add_input("ND", [core_num], "uint32")
        generator.add_input("ND", [core_num], "float32")
        generator.add_input("ND", [_CODE_NUM, dim // 8], "uint8")
        generator.add_input("ND", [core_num], "uint64")
        generator.add_input("ND", [core_num], "uint32")
        generator.add_input("ND", [_CODE_NUM], "float32")
        generator.add_input("ND", [_CODE_NUM], "float32")
        generator.add_input("ND", [core_num], "uint64")
        generator.add_input("ND", [core_num], "uint64")
        generator.add_output("ND", [core_num, _CODE_NUM], "float32")
        generator.add_output("ND", [core_num, (_CODE_NUM + burst_len - 1) // burst_len * 2], "float32")
        generator.add_output("ND", [core_num, 16], "uint16")
        dist_rabitq_l2_obj.append(generator.generate_obj())
    
    utils.generate_op_config(dist_rabitq_l2_obj, file_path)


def generate_distance_rabitq_l2_fp32_simt_json(core_num, list_num, dim, file_path):
    search_batch_sizes = (64, 32, 16, 8, 4, 2, 1)
    dist_rabitq_l2_simt_obj = []
    burst_len = 16 if dim > 256 else 64
    for query_num in search_batch_sizes:
        generator = OpJsonGenerator("DistanceIVFRabitqL2FP32Simt")
        generator.add_input("ND", [1, dim], "float32")
        generator.add_input("ND", [query_num * dim // _SCAN_BIT, _LUT_NUM], "float32")
        generator.add_input("ND", [list_num * dim // _SCAN_BIT, _LUT_NUM], "float32")
        generator.add_input("ND", [core_num], "uint32")
        generator.add_input("ND", [core_num], "uint32")
        generator.add_input("ND", [core_num], "float32")
        generator.add_input("ND", [_CODE_NUM, dim // 8], "uint8")
        generator.add_input("ND", [core_num], "uint64")
        generator.add_input("ND", [core_num], "uint32")
        generator.add_input("ND", [_CODE_NUM], "float32")
        generator.add_input("ND", [_CODE_NUM], "float32")
        generator.add_input("ND", [core_num], "uint64")
        generator.add_input("ND", [core_num], "uint64")
        generator.add_output("ND", [core_num, _CODE_NUM], "float32")
        generator.add_output("ND", [core_num, (_CODE_NUM + burst_len - 1) // burst_len * 2], "float32")
        generator.add_output("ND", [core_num, 16], "uint16")
        dist_rabitq_l2_simt_obj.append(generator.generate_obj())
    
    utils.generate_op_config(dist_rabitq_l2_simt_obj, file_path)


def generate_910b_index_code_and_precompute_json(dim, file_path):
    index_code_and_precompute_obj = []
    burst_len = 16 if dim > 256 else 64
    generator = OpJsonGenerator("IndexCodeAndPrecompute")
    generator.add_input("ND", [1], "int32")
    generator.add_input("ND", [_CODE_NUM, dim], "float32")
    generator.add_input("ND", [_CODE_NUM], "float32")
    generator.add_input("ND", [1, dim], "float32")
    generator.add_input("ND", [1], "float32")
    generator.add_output("ND", [_CODE_NUM, dim // 8], "uint8")
    generator.add_output("ND", [_CODE_NUM], "float32")
    generator.add_output("ND", [_CODE_NUM], "float32")
    index_code_and_precompute_obj.append(generator.generate_obj())
    utils.generate_op_config(index_code_and_precompute_obj, file_path)


def generate_910b_matmul_at_fp32_json(list_num, dim, file_path):
    search_batch_sizes = (64, 32, 16, 8, 4, 2, 1)
    matmul_at_fp32_obj = []
    for query_num in search_batch_sizes:
        generator = OpJsonGenerator("MatmulAtFP32")
        generator.add_input("ND", [query_num, dim], "float32")
        generator.add_input("ND", [dim, dim], "float32")
        generator.add_output("ND", [query_num, dim], "float32")
        matmul_at_fp32_obj.append(generator.generate_obj())

        generator = OpJsonGenerator("MatmulAtFP32")
        generator.add_input("ND", [query_num * dim // _SCAN_BIT, _SCAN_BIT], "float32")
        generator.add_input("ND", [_SCAN_BIT, _LUT_NUM], "float32")
        generator.add_output("ND", [query_num * dim // _SCAN_BIT, _LUT_NUM], "float32")
        matmul_at_fp32_obj.append(generator.generate_obj())

    generator = OpJsonGenerator("MatmulAtFP32")
    generator.add_input("ND", [list_num * dim // _SCAN_BIT, _SCAN_BIT], "float32")
    generator.add_input("ND", [_SCAN_BIT, _LUT_NUM], "float32")
    generator.add_output("ND", [list_num * dim // _SCAN_BIT, _LUT_NUM], "float32")
    matmul_at_fp32_obj.append(generator.generate_obj())

    utils.generate_op_config(matmul_at_fp32_obj, file_path)


def generate_910b_rotate_and_l2_at_fp32_json(list_num, dim, file_path):
    # write dist_compute_flat_mins json
    rotate_and_l2_at_fp32_obj = []
    generator = OpJsonGenerator("RotateAndL2AtFP32")
    generator.add_input("ND", [list_num, dim], "float32")
    generator.add_input("ND", [1], "int32")
    generator.add_input("ND", [dim, dim], "float32")
    generator.add_output("ND", [list_num, dim], "float32")
    generator.add_output("ND", [list_num], "float32")
    rotate_and_l2_at_fp32_obj.append(generator.generate_obj())

    generator = OpJsonGenerator("RotateAndL2AtFP32")
    generator.add_input("ND", [_CODE_NUM, dim], "float32")
    generator.add_input("ND", [1], "int32")
    generator.add_input("ND", [dim, dim], "float32")
    generator.add_output("ND", [_CODE_NUM, dim], "float32")
    generator.add_output("ND", [_CODE_NUM], "float32")
    rotate_and_l2_at_fp32_obj.append(generator.generate_obj())

    utils.generate_op_config(rotate_and_l2_at_fp32_obj, file_path)


def generate_rabitq_offline_model():
    utils.set_env()
    args = arg_parse()
    dim = args.dim
    process_id = args.process_id
    soc_version = utils.get_soc_version_from_npu_type(args.npu_type)
    core_num = utils.get_core_num_by_npu_type(args.core_num, args.npu_type)
    work_dir = "."
    valid_centroid_num = {1024, 2048, 4096, 8192, 16384, 32768}
    utils.check_param_range(args.coarse_centroid_num, valid_centroid_num, "coarse_centroid_num")
    config_path = utils.get_config_path(work_dir)


    valid_dim = _IVFRABITQ_DIM_LIST
    utils.check_param_range(dim, valid_dim, "dim")

    op_name_ = f"rotate_and_l2_at_fp32_op_pid{process_id}"
    file_path_ = os.path.join(config_path, f"{op_name_}.json")
    generate_910b_rotate_and_l2_at_fp32_json(args.coarse_centroid_num, dim, file_path_)
    utils.atc_model(op_name_, soc_version)

    if args.npu_type == 'Ascend950PR':
        op_name_ = f"distance_rabitq_l2_fp32_simt_op_pid{process_id}"
        file_path_ = os.path.join(config_path, f"{op_name_}.json")
        generate_distance_rabitq_l2_fp32_simt_json(core_num, args.coarse_centroid_num, dim, file_path_)
        utils.atc_model(op_name_, soc_version)
    else:
        op_name_ = f"distance_rabitq_l2_fp32_op_pid{process_id}"
        file_path_ = os.path.join(config_path, f"{op_name_}.json")
        generate_distance_rabitq_l2_fp32_json(core_num, args.coarse_centroid_num, dim, file_path_)
        utils.atc_model(op_name_, soc_version)

    op_name_ = f"index_code_and_precompute_op_pid{process_id}"
    file_path_ = os.path.join(config_path, f"{op_name_}.json")
    generate_910b_index_code_and_precompute_json(dim, file_path_)
    utils.atc_model(op_name_, soc_version)

    op_name_ = f"matmul_at_fp32_op_pid{process_id}"
    file_path_ = os.path.join(config_path, f"{op_name_}.json")
    generate_910b_matmul_at_fp32_json(args.coarse_centroid_num, dim, file_path_)
    utils.atc_model(op_name_, soc_version)



if __name__ == "__main__":
    generate_rabitq_offline_model()