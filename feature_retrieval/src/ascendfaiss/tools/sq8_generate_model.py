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
        description="generate distance_compute_sq8 operator model")

    utils.op_common_parse(parser, "--cores", "core_num", 2, int, "Core number")
    utils.op_common_parse(parser, "-d", "dim", 128, int, "Feature dimension")
    utils.op_common_parse(parser, "-p", "process_id", 0, int, "Number of process_id")
    utils.op_common_parse(parser, "-pool", "pool_size", 10, int, "Number of pool_size")
    utils.op_common_parse(parser, "-t", 'npu_type', "310", str, "NPU type, 310 or 310P. 310 by default")
    return parser.parse_args()


def generate_distance_sq8_ip_maxs_json(core_num, query_num, code_nums, dim, file_path):
    # write distance_sq8_ip_maxs json
    dist_sq8_obj = []
    for code_num in code_nums:
        generator = OpJsonGenerator("DistanceSQ8IPMaxs")
        generator.add_input("ND", [query_num, dim], "float16")
        generator.add_input("ND", [code_num // 16, dim // 16, 16, 16], "uint8")
        generator.add_input("ND", [dim], "float16")
        generator.add_input("ND", [dim], "float16")
        generator.add_input("ND", [core_num, 8], "uint32")
        generator.add_output("ND", [query_num, code_num], "float16")
        generator.add_output("ND", [query_num, (code_num + 63) // 64 * 2], "float16")
        generator.add_output("ND", [16, 16], "uint16")
        obj = generator.generate_obj()
        dist_sq8_obj.append(obj)

    utils.generate_op_config(dist_sq8_obj, file_path)


def generate_distance_masked_sq8_ip_maxs_json(core_num, query_num, code_nums, dim, file_path):
    # write distance_sq8_ip_maxs json
    dist_sq8_obj = []
    for code_num in code_nums:
        generator = OpJsonGenerator("DistanceMaskedSQ8IPMaxs")
        generator.add_input("ND", [query_num, dim], "float16")
        generator.add_input("ND", [query_num, (code_num + 7) // 8], "uint8")
        generator.add_input("ND", [code_num // 16, dim // 16, 16, 16], "uint8")
        generator.add_input("ND", [dim], "float16")
        generator.add_input("ND", [dim], "float16")
        generator.add_input("ND", [core_num, 8], "uint32")
        generator.add_output("ND", [query_num, code_num], "float16")
        generator.add_output("ND", [query_num, (code_num + 63) // 64 * 2], "float16")
        generator.add_output("ND", [16, 16], "uint16")
        obj = generator.generate_obj()
        dist_sq8_obj.append(obj)

    utils.generate_op_config(dist_sq8_obj, file_path)


def generate_distance_sq8_ip_maxs_dim64_json(core_num, query_num, code_nums, dim, file_path):
    # write distance_sq8_ip_maxs_dim64 json
    dist_sq8_obj = []
    for code_num in code_nums:
        generator = OpJsonGenerator("DistanceSQ8IPMaxsDim64")
        generator.add_input("ND", [query_num, dim], "float16")
        generator.add_input("ND", [code_num // 16, dim // 16, 16, 16], "uint8")
        generator.add_input("ND", [dim], "float16")
        generator.add_input("ND", [dim], "float16")
        generator.add_input("ND", [core_num, 8], "uint32")
        generator.add_output("ND", [query_num, code_num], "float16")
        generator.add_output("ND", [query_num, (code_num + 63) // 64 * 2], "float16")
        generator.add_output("ND", [16, 16], "uint16")
        obj = generator.generate_obj()
        dist_sq8_obj.append(obj)
    utils.generate_op_config(dist_sq8_obj, file_path)


def generate_distance_masked_sq8_ip_maxs_dim64_json(core_num, query_num, code_nums, dim, file_path):
    # write distance_sq8_ip_maxs_dim64 json
    dist_sq8_obj = []
    for code_num in code_nums:
        generator = OpJsonGenerator("DistanceMaskedSQ8IPMaxsDim64")
        generator.add_input("ND", [query_num, dim], "float16")
        generator.add_input("ND", [query_num, (code_num + 7) // 8], "uint8")
        generator.add_input("ND", [code_num // 16, dim // 16, 16, 16], "uint8")
        generator.add_input("ND", [dim], "float16")
        generator.add_input("ND", [dim], "float16")
        generator.add_input("ND", [core_num, 8], "uint32")
        generator.add_output("ND", [query_num, code_num], "float16")
        generator.add_output("ND", [query_num, (code_num + 63) // 64 * 2], "float16")
        generator.add_output("ND", [16, 16], "uint16")
        obj = generator.generate_obj()
        dist_sq8_obj.append(obj)

    utils.generate_op_config(dist_sq8_obj, file_path)


def generate_distance_sq8_l2_mins_json(core_num, query_num, code_nums, dim, file_path):
    # write distance_compute_sq8 json
    distance_compute_sq8_mins_obj = []
    for code_num in code_nums:
        generator = OpJsonGenerator("DistanceSQ8L2Mins")
        generator.add_input("ND", [query_num, dim], "float16")
        generator.add_input("ND", [code_num // 16, dim // 16, 16, 16], "uint8")
        generator.add_input("ND", [code_num], "float32")
        generator.add_input("ND", [dim], "float16")
        generator.add_input("ND", [dim], "float16")
        generator.add_input("ND", [core_num, 8], "uint32")
        generator.add_output("ND", [query_num, code_num], "float16")
        generator.add_output("ND", [query_num, (code_num + 63) // 64 * 2], "float16")
        generator.add_output("ND", [16, 16], "uint16")
        obj = generator.generate_obj()
        distance_compute_sq8_mins_obj.append(obj)

    utils.generate_op_config(distance_compute_sq8_mins_obj, file_path)


def generate_distance_masked_sq8_l2_mins_json(core_num, query_num, code_nums, dim, file_path):
    # write distance_compute_sq8 json
    distance_compute_sq8_mins_obj = []
    for code_num in code_nums:
        generator = OpJsonGenerator("DistanceMaskedSQ8L2Mins")
        generator.add_input("ND", [query_num, dim], "float16")
        generator.add_input("ND", [query_num, (code_num + 7) // 8], "uint8")
        generator.add_input("ND", [code_num // 16, dim // 16, 16, 16], "uint8")
        generator.add_input("ND", [code_num], "float32")
        generator.add_input("ND", [dim], "float16")
        generator.add_input("ND", [dim], "float16")
        generator.add_input("ND", [core_num, 8], "uint32")
        generator.add_output("ND", [query_num, code_num], "float16")
        generator.add_output("ND", [query_num, (code_num + 63) // 64 * 2], "float16")
        generator.add_output("ND", [16, 16], "uint16")
        obj = generator.generate_obj()
        distance_compute_sq8_mins_obj.append(obj)

    utils.generate_op_config(distance_compute_sq8_mins_obj, file_path)


def generate_cid_filter_json(core_num, cid_nums, file_path):
    # write cid_filter json
    cid_filter_obj = []
    for cid_num in cid_nums:
        generator = OpJsonGenerator("CidFilter")
        generator.add_input("ND", [cid_num], "uint8")
        generator.add_input("ND", [cid_num], "int32")
        generator.add_input("ND", [cid_num], "int32")
        generator.add_input("ND", [4, 64], "int32")
        generator.add_input("ND", [8], "int32")
        generator.add_input("ND", [4, 128], "uint16")
        generator.add_input("ND", [4, 128], "float16")
        generator.add_output("ND", [cid_num // 16], "uint16")
        generator.add_output("ND", [core_num, 16], "uint16")
        obj = generator.generate_obj()
        cid_filter_obj.append(obj)

    utils.generate_op_config(cid_filter_obj, file_path)


def generate_sq8_offline_model():
    utils.set_env()
    args = arg_parse()
    dim, process_id, work_dir = args.dim, args.process_id, "."
    utils.check_param_range(dim, [64, 128, 256, 384, 512, 768], "dim")
    soc_version = utils.get_soc_version_from_npu_type(args.npu_type)
    core_num = utils.get_core_num_by_npu_type(args.core_num, args.npu_type)
    config_path = utils.get_config_path(work_dir)

    cid_filter_op_name = "cid_filter_op_pid{}"
    sq8_l2_op_name = "distance_compute_sq8_l2_mins_op{}_pid{}"
    sq8_ip_op_name = "distance_compute_sq8_ip_maxs_op{}_pid{}"
    sq8_ip_op_name_dim64 = "distance_compute_sq8_ip_maxs_op{}_pid{}_dim64"
    masked_sq8_l2_op_name = "distance_masked_compute_sq8_l2_mins_op{}_pid{}"
    masked_sq8_ip_op_name = "distance_masked_compute_sq8_ip_maxs_op{}_pid{}"
    # dim 768场景受限算子内存，最大仅支持batch32
    if dim == 768:
        search_page_sizes = (32, 16, 8, 4, 2, 1)
    else:
        search_page_sizes = (48, 32, 16, 8, 4, 2, 1)
    # number of code per time in distance_compute_sq8
    code_nums = (32768, 32768 * 2, 32768 * 4, 32768 * 8, 32768 * 16, 32768 * 32)
    map_args = []
    # generate om for code_num=262144 for code_num in code_nums :
    cid_filter_op_name_ = cid_filter_op_name.format(process_id)
    file_path_ = os.path.join(config_path, f"{cid_filter_op_name_}.json")
    generate_cid_filter_json(core_num, code_nums, file_path_)
    map_args.append((cid_filter_op_name_, soc_version))

    for page_size in search_page_sizes:
        # generate sq_ip op
        if dim == 64:
            sq8_ip_op_name_ = sq8_ip_op_name_dim64.format(page_size, process_id)
            file_path_ = os.path.join(config_path, f"{sq8_ip_op_name_}.json")
            generate_distance_sq8_ip_maxs_dim64_json(core_num, page_size, code_nums, dim, file_path_)
            map_args.append((sq8_ip_op_name_, soc_version))

        sq8_ip_op_name_ = sq8_ip_op_name.format(page_size, process_id)
        file_path_ = os.path.join(config_path, f"{sq8_ip_op_name_}.json")
        generate_distance_sq8_ip_maxs_json(core_num, page_size, code_nums, dim, file_path_)
        map_args.append((sq8_ip_op_name_, soc_version))

        # generate masked sq_ip op
        masked_sq8_ip_op_name_ = masked_sq8_ip_op_name.format(page_size, process_id)
        file_path_ = os.path.join(config_path, f"{masked_sq8_ip_op_name_}.json")
        if dim == 64:
            generate_distance_masked_sq8_ip_maxs_dim64_json(core_num, page_size, code_nums, dim, file_path_)
        else:
            generate_distance_masked_sq8_ip_maxs_json(core_num, page_size, code_nums, dim, file_path_)
        map_args.append((masked_sq8_ip_op_name_, soc_version))

        # generate sq_l2 op
        sq8_l2_op_name_ = sq8_l2_op_name.format(page_size, process_id)
        file_path_ = os.path.join(config_path, f"{sq8_l2_op_name_}.json")
        generate_distance_sq8_l2_mins_json(core_num, page_size, code_nums, dim, file_path_)
        map_args.append((sq8_l2_op_name_, soc_version))

        # generate masked sq_l2 op
        masked_sq8_l2_op_name_ = masked_sq8_l2_op_name.format(page_size, process_id)
        file_path_ = os.path.join(config_path, f"{masked_sq8_l2_op_name_}.json")
        generate_distance_masked_sq8_l2_mins_json(core_num, page_size, code_nums, dim, file_path_)
        map_args.append((masked_sq8_l2_op_name_, soc_version))

    utils.run_generate_model_task(args, map_args)


if __name__ == "__main__":
    generate_sq8_offline_model()
