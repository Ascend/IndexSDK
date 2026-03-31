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


def arg_parse():
    """
    Parse arguments to the operator model
    """

    parser = argparse.ArgumentParser(
        description='generate aicore operator model')

    utils.op_common_parse(parser, "-d", 'dim', 128, int, "dimension")
    utils.op_common_parse(parser, "-hash_len", 'hash_len', 256, int, "hash_len")
    utils.op_common_parse(parser, "-top_k", 'top_k', 32, int, "number of topk")
    utils.op_common_parse(parser, "-data_base", 'data_base', 1000000, int, "data_base")
    utils.op_common_parse(parser, "-degree", 'degree', 64, int, "degree")
    utils.op_common_parse(parser, "-p", 'process_id', 0, int, "Number of process_id")
    utils.op_common_parse(parser, "-t", 'npu_type', "Ascend950PR", str, "NPU type, Ascend950PR by default")

    return parser.parse_args()


def generate_search_distance_json(dim, data_base, top_k, batch, degree, hash_len, file_path):
    # write dist_compute_flat_mins json
    search_distance_obj = []
    generator = OpJsonGenerator("Cagra")
    generator.add_input("ND", [batch, dim], "float32")
    generator.add_input("ND", [data_base, degree], "uint32")
    generator.add_input("ND", [batch, hash_len], "uint32")
    generator.add_input("ND", [dim, data_base], "float32")
    
    generator.add_output("ND", [batch, top_k], "float32")
    generator.add_output("ND", [batch, top_k], "uint32")
    search_distance_obj.append(generator.generate_obj())
    utils.generate_op_config(search_distance_obj, file_path)



def generate_search_distance_offline_model():
    utils.set_env()
    args = arg_parse()
    process_id = args.process_id
    dim = args.dim
    data_base = args.data_base
    topk = args.top_k
    degree = args.degree
    hash_len = args.hash_len

    soc_version = utils.get_soc_version_from_npu_type(args.npu_type)
    search_page_sizes = (32, 16, 8, 4, 2, 1)
    work_dir = '.'
    config_path = utils.get_config_path(work_dir)

    for page_size in search_page_sizes:
        op_name_ = f"ascendc_cagra_search_distance_op_pid{process_id}"
        file_path_ = os.path.join(config_path, f"{op_name_}.json")
        generate_search_distance_json(dim, data_base, topk, page_size, degree, hash_len, file_path_)
        utils.atc_model(op_name_, soc_version)


if __name__ == '__main__':
    generate_search_distance_offline_model()