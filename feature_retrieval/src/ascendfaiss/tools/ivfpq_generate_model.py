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
import common as utils
from common import OpJsonGenerator


_BlockSize = 16384 * 16
_MaxSize = 1024 * 256
_CoreNum = 56



def arg_parse():
    """
    Parse arguements to the operator model
    """

    parser = argparse.ArgumentParser(
        description='generate aicore operator model')

    utils.op_common_parse(parser, "-c", 'nlist', 1024, int, "number of centroids")
    utils.op_common_parse(parser, "-m", 'm', 4, int, "number of sub space")
    utils.op_common_parse(parser, "-d", 'dim', 128, int, "number of dim")
    utils.op_common_parse(parser, "-n", 'nbit', 8, int, "number of bits per quantized subvector component")
    utils.op_common_parse(parser, "-topK", 'topK', 320, int, "number of topk")
    utils.op_common_parse(parser, "-b", 'blockNum', 32, int, "Number of block")
    utils.op_common_parse(parser, "-p", 'process_id', 0, int, "Number of process_id")
    utils.op_common_parse(parser, "-t", 'npu_type', "Ascend950PR", str, "NPU type, Ascend950PR by default")

    return parser.parse_args()


def generate_distance_flat_l2_mins_at_fp32_json(nlist, dim, ksub, msub, file_path):
    # write dist_compute_flat_mins json
    search_batch_sizes = (64, 32, 16, 8, 4, 2, 1)
    dist_flat_mins_obj = []
    for batch_size in search_batch_sizes:
        generator = OpJsonGenerator("DistanceFlatL2MinsAtFP32")
        generator.add_input("ND", [batch_size, dim], "float32")
        generator.add_input("ND", [nlist, dim], "float32")
        generator.add_input("ND", [nlist], "float32")

        generator.add_output("ND", [batch_size, nlist], "float32")
        generator.add_output("ND", [batch_size, nlist // 64 * 2], "float32")
        generator.add_output("ND", [_CoreNum, 16], "uint16")
        dist_flat_mins_obj.append(generator.generate_obj())

    for batch_size in search_batch_sizes:
        generator = OpJsonGenerator("DistanceFlatL2MinsAtFP32")
        generator.add_input("ND", [batch_size, msub], "float32")
        generator.add_input("ND", [ksub, msub], "float32")
        generator.add_input("ND", [ksub], "float32")

        generator.add_output("ND", [batch_size, ksub], "float32")
        generator.add_output("ND", [batch_size, ksub // 64 * 2], "float32")
        generator.add_output("ND", [_CoreNum, 16], "uint16")
        dist_flat_mins_obj.append(generator.generate_obj())
    utils.generate_op_config(dist_flat_mins_obj, file_path)


def generate_subspace_distance_json(m, dim, ksub, file_path):
    # write dist_compute_flat_mins json
    subspace_distance_obj = []
    search_batch_sizes = (64, 32, 16, 8, 4, 2, 1)
    for batch_size in search_batch_sizes:
        generator = OpJsonGenerator("AscendcIvfpqSubspaceDistance")
        generator.add_input("ND", [batch_size, dim], "float32")
        generator.add_input("ND", [m, ksub, int(dim / m)], "float32")

        generator.add_output("ND", [batch_size, m, ksub], "float32")
        generator.add_output("ND", [batch_size, m, int(ksub / 64 * 2)], "float32")
        generator.add_output("ND", [batch_size, m, int(ksub / 16)], "uint16")
        subspace_distance_obj.append(generator.generate_obj())
    utils.generate_op_config(subspace_distance_obj, file_path)


def generate_search_distance_l2_json(m, ksub, blockNum, topK, file_path):
    # write dist_compute_flat_mins json
    search_distance_obj = []
    generator = OpJsonGenerator("AscendcIvfpqSearchDistanceL2")
    generator.add_input("ND", [m, ksub], "float32")
    generator.add_input("ND", [m], "uint8")
    generator.add_input("ND", [blockNum], "int64")
    generator.add_input("ND", [blockNum], "int64")
    generator.add_input("ND", [topK, _MaxSize], "int32")

    generator.add_output("ND", [blockNum, _BlockSize], "float32")
    generator.add_output("ND", [blockNum, topK], "int32")
    generator.add_output("ND", [blockNum, topK], "float32")
    generator.add_output("ND", [blockNum, 16], "uint16")
    search_distance_obj.append(generator.generate_obj())
    utils.generate_op_config(search_distance_obj, file_path)


def generate_ivfpq_offline_model():
    utils.set_env()
    args = arg_parse()
    nlist = args.nlist
    process_id = args.process_id
    m = args.m
    dim = args.dim
    nbit = args.nbit
    topk = args.topK
    blockNum = args.blockNum
    ksub = 1 << nbit

    utils.check_param_range(dim, [128], "dim")
    utils.check_param_range(nlist, [1024, 2048, 4096], "nlist")
    utils.check_param_range(m, [4], "m")
    utils.check_param_range(nbit, [8], "nbit")
    utils.check_param_range(ksub, [256], "ksub")
    soc_version = utils.get_soc_version_from_npu_type(args.npu_type)

    work_dir = '.'
    config_path = utils.get_config_path(work_dir)

    op_name_ = f"flat_l2_mins_at_fp32_op_pid{process_id}"
    file_path_ = os.path.join(config_path, f"{op_name_}.json")
    generate_distance_flat_l2_mins_at_fp32_json(nlist, dim, ksub, dim / m, file_path_)
    utils.atc_model(op_name_, soc_version)

    op_name_ = f"ascendc_ivfpq_subspace_distance_op_pid{process_id}"
    file_path_ = os.path.join(config_path, f"{op_name_}.json")
    generate_subspace_distance_json(m, dim, ksub, file_path_)
    utils.atc_model(op_name_, soc_version)

    op_name_ = f"ascendc_ivfpq_search_distance_op_pid{process_id}"
    file_path_ = os.path.join(config_path, f"{op_name_}.json")
    generate_search_distance_l2_json(m, ksub, blockNum, topk, file_path_)
    utils.atc_model(op_name_, soc_version)

if __name__ == '__main__':
    generate_ivfpq_offline_model()
