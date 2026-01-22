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
import collections
from multiprocessing import Pool

import common as utils
from common import OpJsonGenerator

shape_paras = collections.namedtuple('shape_paras', ['z_region', 'burst_len', 'dim_align', 'dim_batch'])

_BIT_OF_UINT8 = 8
_CUBE_ALIGN = 32
_CUBE_ALIGN_HAMMING = 4
_BLOCK_SIZE = 262144
_Z_REGION_HEIGHT = 32
_BURST_LENGTH = 128
_BURST_LEN = 64
_BURST_LEN_LOW = 32
_CORE_NUM = 8


def arg_parse():
    """
    Parse arguements to the operator model
    """
    parser = argparse.ArgumentParser(
        description='generate aicore operator model')

    utils.op_common_parse(parser, "-d", 'dim', 512, int, "Feature dimension, default 512")
    utils.op_common_parse(parser, "-q", 'query_type', "uint8", str, "Query type, uint8 or float. uint8 by default")
    utils.op_common_parse(parser, "-p", 'process_id', 0, int, "Number of process_id, default 0")
    utils.op_common_parse(parser, "-pool", 'pool_size', 16, int, "Number of pool_size, default 16")
    return parser.parse_args()


def generate_hamming_distance_json(query_num, dim, z_region_height, burst_length, file_path):
    # write hamming distance op json
    hamming_distance_json_obj = []
    generator = OpJsonGenerator("DistanceFlatHamming")
    generator.add_input("ND", [query_num, dim // _BIT_OF_UINT8], "uint8")
    generator.add_input("ND",
                [_BLOCK_SIZE // z_region_height, dim // _CUBE_ALIGN, z_region_height, _CUBE_ALIGN_HAMMING], "uint8")
    generator.add_input("ND", [8], "uint32")
    generator.add_output("ND", [_BLOCK_SIZE * query_num], "float16")
    generator.add_output("ND", [query_num * _BLOCK_SIZE // burst_length * 2], "float16")
    generator.add_output("ND", [16], "uint16")
    hamming_distance_json_obj.append(generator.generate_obj())
    utils.generate_op_config(hamming_distance_json_obj, file_path)


def generate_nonshare_masked_hamming_distance_json(query_num, dim, z_region_height, burst_length, file_path):
    # write nonshare mask hamming distance op json
    masked_hamming_distance_json_obj = []
    generator = OpJsonGenerator("DistanceFlatHammingWithMask")
    generator.add_input("ND", [query_num, dim // _BIT_OF_UINT8], "uint8")
    generator.add_input("ND",
                [_BLOCK_SIZE // z_region_height, dim // _CUBE_ALIGN, z_region_height, _CUBE_ALIGN_HAMMING], "uint8")
    generator.add_input("ND", [8], "uint32")
    generator.add_input("ND", [query_num, (_BLOCK_SIZE + 7) // 8], "uint8")
    generator.add_output("ND", [_BLOCK_SIZE * query_num], "float16")
    generator.add_output("ND", [query_num * _BLOCK_SIZE // burst_length * 2], "float16")
    generator.add_output("ND", [16], "uint16")
    masked_hamming_distance_json_obj.append(generator.generate_obj())
    utils.generate_op_config(masked_hamming_distance_json_obj, file_path)


def generate_share_masked_hamming_distance_json(query_num, dim, z_region_height, burst_length, file_path):
    # write share mask hamming distance op json
    masked_hamming_distance_json_obj = []
    generator = OpJsonGenerator("DistanceFlatHammingWithMask")
    generator.add_input("ND", [query_num, dim // _BIT_OF_UINT8], "uint8")
    generator.add_input("ND",
                [_BLOCK_SIZE // z_region_height, dim // _CUBE_ALIGN, z_region_height, _CUBE_ALIGN_HAMMING], "uint8")
    generator.add_input("ND", [8], "uint32")
    generator.add_input("ND", [1, (_BLOCK_SIZE + 7) // 8], "uint8")
    generator.add_output("ND", [_BLOCK_SIZE * query_num], "float16")
    generator.add_output("ND", [query_num * _BLOCK_SIZE // burst_length * 2], "float16")
    generator.add_output("ND", [16], "uint16")
    masked_hamming_distance_json_obj.append(generator.generate_obj())
    utils.generate_op_config(masked_hamming_distance_json_obj, file_path)


def generate_distance_binary_float_json(query_num, dim, shape_para, file_path):
    # write binary float distance op json
    distance_binary_float_json_obj = []
    generator = OpJsonGenerator("DistanceBinaryFloat")
    generator.add_input("ND", [query_num, dim], "float16")
    generator.add_input("ND", [_BLOCK_SIZE // shape_para.z_region, dim // shape_para.dim_align,
                               shape_para.z_region, shape_para.dim_batch], "uint8")
    generator.add_input("ND", [_CORE_NUM, 8], "uint32")
    generator.add_output("ND", [query_num, _BLOCK_SIZE], "float16")
    generator.add_output("ND", [query_num, _BLOCK_SIZE // shape_para.burst_len * 2], "float16")
    generator.add_output("ND", [_CORE_NUM, 16], "uint16")
    distance_binary_float_json_obj.append(generator.generate_obj())
    utils.generate_op_config(distance_binary_float_json_obj, file_path)


def generate_hamming_offline_model():
    utils.set_env()
    args = arg_parse()
    dim = args.dim
    qtype = args.query_type
    utils.check_param_range(dim, [256, 512, 1024], "dim")
    # hamming当前仅支持310P
    soc_version = "Ascend310P3"
    dim_align = _CUBE_ALIGN
    dim_batch = _CUBE_ALIGN_HAMMING
    if dim == 1024:
        z_region_height = 16
        burst_length = 64
        z_region = 16
    else:
        z_region_height = _Z_REGION_HEIGHT
        burst_length = _BURST_LENGTH
        z_region = _Z_REGION_HEIGHT
    if qtype == "float":
        z_region = _Z_REGION_HEIGHT
        dim_align = _CUBE_ALIGN * _BIT_OF_UINT8
        dim_batch = _CUBE_ALIGN

    process_id = args.process_id
    work_dir = '.'
    config_path = utils.get_config_path(work_dir)
    queries = (256, 128, 64, 32, 16, 8, 7, 6, 5, 4, 3, 2, 1)
    hamming_op_name = "distance_compute_op{}_pid{}"
    nonshare_masked_hamming_op_name = "distance_nonshare_masked_compute_op{}_pid{}"
    share_masked_hamming_op_name = "distance_share_masked_compute_op{}_pid{}"
    binary_float_op_name = "distance_binary_float_op_pid{}"

    map_args = []

    for query in queries:
        # generate hamming op
        hamming_op_name_ = hamming_op_name.format(query, process_id)
        file_path = os.path.join(config_path, '{}.json'.format(hamming_op_name_))
        generate_hamming_distance_json(query, dim, z_region_height, burst_length, file_path)
        map_args.append((hamming_op_name_, soc_version))

        # generate share masked_hamming op
        share_masked_hamming_op_name_ = share_masked_hamming_op_name.format(query, process_id)
        file_path = os.path.join(config_path, '{}.json'.format(share_masked_hamming_op_name_))
        generate_share_masked_hamming_distance_json(query, dim, z_region_height, burst_length, file_path)
        map_args.append((share_masked_hamming_op_name_, soc_version))

        # generate nonshare masked_hamming op
        nonshare_masked_hamming_op_name_ = nonshare_masked_hamming_op_name.format(query, process_id)
        file_path = os.path.join(config_path, '{}.json'.format(nonshare_masked_hamming_op_name_))
        generate_nonshare_masked_hamming_distance_json(query, dim, z_region_height, burst_length, file_path)
        map_args.append((nonshare_masked_hamming_op_name_, soc_version))

        burst_len = _BURST_LEN_LOW if query > 48 else _BURST_LEN
        binary_float_op_name_ = binary_float_op_name.format(query, process_id)
        file_path = os.path.join(config_path, '{}.json'.format(binary_float_op_name_))
        shape_para = shape_paras(z_region, burst_len, dim_align, dim_batch)
        generate_distance_binary_float_json(query, dim, shape_para, file_path)
        map_args.append((binary_float_op_name_, soc_version))


    utils.run_generate_model_task(args, map_args)


if __name__ == '__main__':
    generate_hamming_offline_model()
