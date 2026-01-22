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
from ivfsq8_generate_model import generate_distance_sq8_ip4_json


def arg_parse():
    """
    Parse arguements to the operator model
    """

    parser = argparse.ArgumentParser(
        description='generate distance_compute_sq8 operator model')

    utils.op_common_parse(parser, "--cores", 'core_num', 2, int, "Core number")
    utils.op_common_parse(parser, "-d", 'dim_in', 256, int, "Feature dimension")
    utils.op_common_parse(parser, "-r", 'compress_ratio', 4, int, "compress_ratio")
    utils.op_common_parse(parser, "-c", 'coarse_centroid_num', 16384, int, "Number of coarse centroid")
    utils.op_common_parse(parser, "-p", 'process_id', 0, int, "Number of process_id")
    utils.op_common_parse(parser, "-pool", "pool_size", 32, int, "Number of pool_size")
    utils.op_common_parse(parser, "-t", 'npu_type', "310", str, "NPU type, 310 or 310P. 310 by default")
    return parser.parse_args()


def generate_distance_flat_ip_maxs_json(core_nums, queries_num, dim, code_num, file_path):
    # write dist_compute_flat_mins json
    dist_flat_ip_maxs_obj = []
    generator = OpJsonGenerator("DistanceFlatIPMaxs")
    generator.add_input("ND", [queries_num, dim], "float16")
    generator.add_input("ND", [queries_num, (code_num + 7) // 8], "uint8")
    generator.add_input("ND", [code_num // 16, dim // 16, 16, 16], "float16")
    generator.add_input("ND", [core_nums, 8], "uint32")
    generator.add_output("ND", [queries_num, code_num], "float16")
    generator.add_output("ND", [queries_num, (code_num + 63) // 64 * 2], "float16")
    generator.add_output("ND", [core_nums, 16], "uint16")
    dist_flat_ip_maxs_obj.append(generator.generate_obj())
    utils.generate_op_config(dist_flat_ip_maxs_obj, file_path)


def generate_distance_flat_subcenters_json(core_nums, queries_num, coarse_centers_nums, picked_nums, dims, file_path):
    code_nums = 64
    dist_flat_obj = []
    generator = OpJsonGenerator("DistanceFlatSubcenters")
    generator.add_input("ND", [queries_num, dims], "float16")
    generator.add_input("ND", [coarse_centers_nums, dims // 16, code_nums, 16], "float16")
    generator.add_input("ND", [coarse_centers_nums, code_nums], "float16")
    generator.add_input("ND", [queries_num, picked_nums], "uint16")
    generator.add_output("ND", [queries_num, picked_nums, code_nums], "float16")
    generator.add_output("ND", [core_nums, 16], "uint16")
    dist_flat_obj.append(generator.generate_obj())
    utils.generate_op_config(dist_flat_obj, file_path)


def generate_distance_sq8_ipx_json(core_nums, query_num, di, segment_nums, file_path):
    # write distance_sq8x_ip json
    # number of code per time
    code_num = 16384
    segment_size = 64
    max_batch = 16
    dist_sq8_obj = []
    generator = OpJsonGenerator("DistanceIVFSQ8IPX")
    generator.add_input("ND", [query_num, di], "float16")
    generator.add_input("ND", [code_num * di], "uint8")
    generator.add_input("ND", [query_num, segment_nums], "uint64")
    generator.add_input("ND", [2, di], "float16")
    generator.add_output("ND", [query_num, segment_nums * segment_size], "float16")
    generator.add_output("ND", [query_num, segment_nums * 2 * (segment_size // max_batch)], "float16")
    generator.add_output("ND", [core_nums, 16], "uint16")
    dist_sq8_obj.append(generator.generate_obj())
    utils.generate_op_config(dist_sq8_obj, file_path)


def generate_corr_compute_json(core_nums, dim, file_path):
    code_num = 1024
    corr_compute_obj = [{
        "op":
            "CorrCompute",
        "input_desc": [{
            "format": "ND",
            "shape": [1, code_num // 16, dim, 16],
            "type": "float16"
        }, {
            "format": "ND",
            "shape": [core_nums, ],
            "type": "uint64"
        }],
        "output_desc": [{
            "format": "ND",
            "shape": [dim, dim],
            "type": "float16"
        }, {
            "format": "ND",
            "shape": [core_nums, 16],
            "type": "uint16"
        }]
    }]

    utils.generate_op_config(corr_compute_obj, file_path)


def generate_ivf_sqt_offline_model():
    utils.set_env()
    args = arg_parse()

    process_ids = args.process_id
    coarse_centroid_nums = args.coarse_centroid_num
    utils.check_param_range(coarse_centroid_nums, [1024, 2048, 4096, 8192, 16384, 32768], "coarse_centroid_num")
    
    ratio = args.compress_ratio
    if ratio <= 1:
        raise ValueError("not support compress_ratio %d" % ratio)

    din = args.dim_in
    utils.check_param_range(din, [256], "din")

    soc_version = utils.get_soc_version_from_npu_type(args.npu_type)
    core_nums = utils.get_core_num_by_npu_type(args.core_num, args.npu_type)
    dout = din // ratio
    utils.check_param_range(dout, [32, 64, 128], "dout")

    work_dir = '.'
    config_path = utils.get_config_path(work_dir)

    search_page_sizes_l1 = (64, 32, 16, 8, 4, 2, 1)
    search_page_sizes_l2 = (128, 64, 32, 16, 8, 4, 2, 1)
    picked_nums_l2 = (8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64)
    search_page_sizes_l3 = (32, 16, 8, 4, 2, 1)
    segment_size = (24, 36, 48, 60, 72, 84, 96, 120, 144, 156, 168, 192, 216, 240, 360, 480, 600, 720, 840, 960, 1020)

    flat_ip_op_name = "distance_flat_ip_maxs_op{}_pid{}"
    flat_l2_op_name = "distance_flat_l2_mins_op{}_pid{}"
    subcenter_op_name = "distance_flat_subcenters_op{}_{}_pid{}"
    distance_ipx_op_name = "distance_ivf_sq8_ipx_op_pid{}_{}_pid{}"

    map_args = []
    utils.check_pool_size(args.pool_size)

    p = Pool(args.pool_size)

    for page_size in search_page_sizes_l1:
        flat_l2_op_name_ = flat_l2_op_name.format(page_size, process_ids)
        file_l2_path_ = os.path.join(config_path, '%s.json' % flat_l2_op_name_)
        utils.generate_dist_flat_l2_mins_json(core_nums, page_size, din, coarse_centroid_nums, file_l2_path_)
        map_args.append((flat_l2_op_name_, soc_version))

        flat_ip_op_name_ = flat_ip_op_name.format(page_size, process_ids)
        file_ip_path_ = os.path.join(config_path, '%s.json' % flat_ip_op_name_)
        generate_distance_flat_ip_maxs_json(core_nums, page_size, din, coarse_centroid_nums, file_ip_path_)
        map_args.append((flat_ip_op_name_, soc_version))

    for page_size in search_page_sizes_l2:
        for pick_num in picked_nums_l2:
            op_name_ = subcenter_op_name.format(page_size, pick_num, process_ids)
            file_path_ = os.path.join(config_path, '%s.json' % op_name_)
            generate_distance_flat_subcenters_json(core_nums, page_size, coarse_centroid_nums,
                                                   pick_num, din, file_path_)
            map_args.append((op_name_, soc_version))

    for page_size in search_page_sizes_l3:
        for seg in segment_size:
            op_name_ = distance_ipx_op_name.format(page_size, seg, process_ids)
            file_path_ = os.path.join(config_path, '{}.json'.format(op_name_))
            generate_distance_sq8_ipx_json(core_nums, page_size, dout, seg, file_path_)
            map_args.append((op_name_, soc_version))
    for dim in [din, dout]:
        op_name_ = "distance_ivf_sq8_ip_op{}_pid{}".format(dim, process_ids)
        file_path_ = os.path.join(config_path, '{}.json'.format(op_name_))
        generate_distance_sq8_ip4_json(core_nums, 1, dim, file_path_)
        map_args.append((op_name_, soc_version))

    p.starmap(utils.atc_model, map_args)

    for dim in [din]:
        op_name_ = "corr_compute_op{}_pid{}".format(dim, process_ids)
        file_path_ = os.path.join(config_path, '{}.json'.format(op_name_))
        generate_corr_compute_json(core_nums, dim, file_path_)
        utils.atc_model(op_name_, soc_version)


if __name__ == '__main__':
    generate_ivf_sqt_offline_model()
