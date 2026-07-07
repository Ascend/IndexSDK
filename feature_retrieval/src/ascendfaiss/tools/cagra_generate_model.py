#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
-------------------------------------------------------------------------
This file is part of the IndexSDK project.
Copyright (c) 2026 Huawei Technologies Co.,Ltd.

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

JSON_FILE = "{}.json"
_BIT = 6
_HASH_SIZE = 2
_PRE_COMPUTE_SIZE = 4


def arg_parse():
    """
    Parse arguments to the operator model
    """

    parser = argparse.ArgumentParser(description='generate aicore operator model')

    utils.op_common_parse(parser, "-d", 'dim', 128, int, "dimension")
    utils.op_common_parse(parser, "-data_base", 'data_base', 1000000, int, "data_base")
    utils.op_common_parse(parser, "-degree", 'degree', 64, int, "degree")
    utils.op_common_parse(parser, "-topK", 'topK', 64, int, "topK")
    utils.op_common_parse(parser, "-p", 'process_id', 0, int, "Number of process_id")
    utils.op_common_parse(parser, "-t", 'npu_type', "Ascend950PR", str, "NPU type, Ascend950PR by default")

    return parser.parse_args()


def generate_search_distance_json(dim, data_base, code_size, batch, degree, rotated_size, topK, file_path):
    # write dist_compute_flat_mins json
    search_distance_obj = []
    generator = OpJsonGenerator("CagraRabitq")
    generator.add_input("ND", [batch, dim], "float32")
    generator.add_input("ND", [data_base, degree], "uint32")
    generator.add_input("ND", [data_base, _HASH_SIZE], "uint32")
    generator.add_input("ND", [data_base, dim], "float32")
    generator.add_input("ND", [batch * _PRE_COMPUTE_SIZE], "float32")
    generator.add_input("ND", [data_base * code_size], "uint8")
    generator.add_input("ND", [batch * rotated_size], "uint8")

    generator.add_output("ND", [batch, topK], "float32")
    generator.add_output("ND", [batch, topK], "uint32")
    search_distance_obj.append(generator.generate_obj())
    utils.generate_op_config(search_distance_obj, file_path)


def generate_search_distance_offline_model():
    utils.set_env()
    args = arg_parse()
    process_id = args.process_id
    dim = args.dim
    data_base = args.data_base
    degree = args.degree
    topK = args.topK

    offset = (dim + 7) // 8
    rotated_size = offset * _BIT
    code_size = offset * _BIT + 16

    soc_version = utils.get_soc_version_from_npu_type(args.npu_type)
    search_page_sizes = (64, 32, 16, 8, 4, 2, 1)
    work_dir = '.'
    config_path = utils.get_config_path(work_dir)
    cagra_rabitq_op_name = "ascendc_cagra_search_distance_op{}_pid{}"

    for page_size in search_page_sizes:
        op_name_ = cagra_rabitq_op_name.format(page_size, process_id)
        file_path_ = os.path.join(config_path, JSON_FILE.format(op_name_))
        generate_search_distance_json(dim, data_base, code_size, page_size, degree, rotated_size, topK, file_path_)
        utils.atc_model(op_name_, soc_version)


if __name__ == '__main__':
    generate_search_distance_offline_model()
