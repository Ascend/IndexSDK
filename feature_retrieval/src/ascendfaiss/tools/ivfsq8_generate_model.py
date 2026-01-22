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
from multiprocessing import Pool

import common as utils
from common import OpJsonGenerator


def arg_parse():
    """
    Parse arguements to the operator model
    """

    parser = argparse.ArgumentParser(
        description="generate distance_compute_ivf_sq8 operator model")

    utils.op_common_parse(parser, "--cores", "core_num", 2, int, "Core number")
    utils.op_common_parse(parser, "-d", "dim", 128, int, "Feature dimension")
    utils.op_common_parse(parser, "-c", "coarse_centroid_num", 16384, int, "Number of coarse centroid")
    utils.op_common_parse(parser, "-p", "process_id", 0, int, "Number of process_id")
    utils.op_common_parse(parser, "-pool", "pool_size", 10, int, "Number of pool_size")
    utils.op_common_parse(parser, "-t", 'npu_type', "310", str, "NPU type, 310 or 310P. 310 by default")
    return parser.parse_args()


def generate_distance_sq8_ip4_json(core_num, query_num, dim, file_path):
    # write distance_sq8_ip json
    code_num = 8192
    code_list_num = 4
    dist_sq8_obj = []
    generator = OpJsonGenerator("DistanceIVFSQ8IP4")
    generator.add_input("ND", [query_num, dim], "float16")
    generator.add_input("ND", [code_num // 16, dim // 16, 16, 16], "uint8")
    generator.add_input("ND", [code_num // 16, dim // 16, 16, 16], "uint8")
    generator.add_input("ND", [code_num // 16, dim // 16, 16, 16], "uint8")
    generator.add_input("ND", [code_num // 16, dim // 16, 16, 16], "uint8")
    generator.add_input("ND", [2, dim], "float16")
    generator.add_input("ND", [8], "uint32")
    generator.add_output("ND", [(code_list_num * code_num)], "float16")
    generator.add_output("ND", [(code_list_num * code_num // 16 * 2)], "float16")
    generator.add_output("ND", [core_num, 32], "uint16")
    dist_sq8_obj.append(generator.generate_obj())
    utils.generate_op_config(dist_sq8_obj, file_path)


def generate_distance_sq8_ip8_json(core_num, query_num, dim, file_path):
    # write distance_sq8_ip8 json
    min_batch = 16 if dim > 256 else 32
    code_num = 32768
    code_list_num = 8
    dist_sq8_obj = []
    generator = OpJsonGenerator("DistanceIVFSQ8IP8")
    generator.add_input("ND", [query_num, dim], "float16")
    generator.add_input("ND", [code_num * dim], "uint8")
    generator.add_input("ND", [code_list_num], "uint64")
    generator.add_input("ND", [2, dim], "float16")
    generator.add_input("ND", [8], "uint32")
    generator.add_output("ND", [code_list_num, code_num], "float16")
    generator.add_output("ND", [code_list_num, code_num // min_batch * 2], "float16")
    generator.add_output("ND", [core_num, 32], "uint16")
    dist_sq8_obj.append(generator.generate_obj())

    utils.generate_op_config(dist_sq8_obj, file_path)


def generate_distance_sq8_ip8_batch_json(core_num, accum, dim, file_path):
    # write distance_sq8_ip8 json
    code_num = 32768
    code_list_num = 8
    addr_num_align = (accum + 3) // 4 * 4
    dist_sq8_obj = []
    generator = OpJsonGenerator("DistanceIVFSQ8IP8Accum")
    generator.add_input("ND", [1, dim], "float16")
    generator.add_input("ND", [code_num * dim], "uint8")
    generator.add_input("ND", [addr_num_align], "int64")
    generator.add_input("ND", [2, dim], "float16")
    generator.add_input("ND", [addr_num_align], "int64")
    generator.add_input("ND", [code_list_num], "uint32")
    generator.add_input("ND", [addr_num_align], "uint64")
    generator.add_output("ND", [addr_num_align], "int64")
    generator.add_output("ND", [addr_num_align], "int64")
    generator.add_output("ND", [addr_num_align], "int64")
    dist_sq8_obj.append(generator.generate_obj())
    utils.generate_op_config(dist_sq8_obj, file_path)


def generate_distance_sq8_l2_json(core_num, query_num, dim, file_path):
    # number of code per time
    centroid_num = 65536
    min_batch = 32
    if dim > 128:
        min_batch = 16

    # write distance_sq8_l2 json
    dist_sq8_obj = []
    generator = OpJsonGenerator("DistanceIVFSQ8L2")
    generator.add_input("ND", [query_num, dim], "float16")
    generator.add_input("ND", [centroid_num // 16, dim // 16, 16, 16], "uint8")
    generator.add_input("ND", [centroid_num], "float32")
    generator.add_input("ND", [dim], "float16")
    generator.add_input("ND", [dim], "float16")
    generator.add_input("ND", [core_num, 8], "uint32")
    generator.add_output("ND", [query_num, centroid_num], "float16")
    generator.add_output("ND", [query_num, (centroid_num + min_batch - 1) // min_batch * 2], "float16")
    generator.add_output("ND", [16, 16], "uint16")
    dist_sq8_obj.append(generator.generate_obj())
    utils.generate_op_config(dist_sq8_obj, file_path)


def generate_ivf_sq8_offline_model():
    utils.set_env()
    args = arg_parse()
    dim = args.dim
    valid_dim = {64, 128, 256, 384, 512}
    utils.check_param_range(dim, valid_dim, "dim")
    coarse_centroid_num = args.coarse_centroid_num
    valid_centroid_num = {1024, 2048, 4096, 8192, 16384, 32768}
    utils.check_param_range(coarse_centroid_num, valid_centroid_num, "coarse_centroid_num")
    process_id = args.process_id
    soc_version = utils.get_soc_version_from_npu_type(args.npu_type)
    core_num = utils.get_core_num_by_npu_type(args.core_num, args.npu_type)

    config_path = utils.get_config_path(work_dir=".")
    flat_l2_op_name = "distance_compute_flat_op{}_pid{}"
    flat_ip_op_name = "distance_flat_l2_mins_op{}_pid{}"
    search_page_sizes = (64, 32, 16, 8, 4, 2, 1)
    ivfsq_accums = (32, 16, 8, 4, 2, 1)

    map_args = []

    for page_size in search_page_sizes:
        flat_l2_op_name_ = flat_l2_op_name.format(page_size, process_id)
        file_l2_path_ = os.path.join(config_path, "{}.json".format(flat_l2_op_name_))
        utils.generate_dist_compute_json(page_size, dim, coarse_centroid_num, file_l2_path_)
        map_args.append((flat_l2_op_name_, soc_version))

        flat_ip_op_name_ = flat_ip_op_name.format(page_size, process_id)
        file_ip_path_ = os.path.join(config_path, "{}.json".format(flat_ip_op_name_))
        utils.generate_dist_flat_l2_mins_json(core_num, page_size, dim, coarse_centroid_num, file_ip_path_)
        map_args.append((flat_ip_op_name_, soc_version))

    if core_num == 2:
        op_name_ = "distance_ivf_sq8_ip_op_pid{}".format(process_id)
        file_path_ = os.path.join(config_path, "{}.json".format(op_name_))
        generate_distance_sq8_ip4_json(core_num, 1, dim, file_path_)
        map_args.append((op_name_, soc_version))
    else:
        op_name_ = "distance_ivf_sq8_ip8_op_pid{}".format(process_id)
        file_path_ = os.path.join(config_path, "{}.json".format(op_name_))
        generate_distance_sq8_ip8_json(core_num, 1, dim, file_path_)
        map_args.append((op_name_, soc_version))
        # ip8 batch
        for accum in ivfsq_accums:
            if dim >= 512:
                continue
            op_name_ = "distance_ivf_sq8_ip8b_op{}_pid{}".format(accum, process_id)
            file_path_ = os.path.join(config_path, "{}.json".format(op_name_))
            generate_distance_sq8_ip8_batch_json(core_num, accum, dim, file_path_)
            map_args.append((op_name_, soc_version))

    op_name_ = "distance_ivf_sq8_l2_op_pid{}".format(process_id)
    file_path_ = os.path.join(config_path, "{}.json".format(op_name_))
    generate_distance_sq8_l2_json(core_num, 1, dim, file_path_)
    map_args.append((op_name_, soc_version))

    utils.run_generate_model_task(args, map_args)


if __name__ == "__main__":
    generate_ivf_sq8_offline_model()
