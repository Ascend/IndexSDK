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

# number of code per time in distance_compute_flat_min64
_QUERY_NUM = 1024


def arg_parse():
    """
    Parse arguements to the operator model
    """

    parser = argparse.ArgumentParser(
        description='generate aicore operator model')

    utils.op_common_parse(parser, "--cores", 'core_num', 56, int, "Core number")
    utils.op_common_parse(parser, "-b", 'batch_size', 8, int, "size of batch")
    utils.op_common_parse(parser, "-m", 'm', 4, int, "number of sub space")
    utils.op_common_parse(parser, "-d", 'dim', 128, int, "number of dim")
    utils.op_common_parse(parser, "-k", 'ksub', 256, int, "number of sub space center")
    utils.op_common_parse(parser, "-p", 'process_id', 0, int, "Number of process_id")
    utils.op_common_parse(parser, "-t", 'npu_type', "Ascend950PR", str, "NPU type, Ascend950PR by default")

    return parser.parse_args()


def generate_subspace_distance_json(batch_size, m, dim, ksub, file_path):
    # write dist_compute_flat_mins json
    subspace_distance_obj = []
    generator = OpJsonGenerator("AscendcIvfpqSubspaceDistance")
    generator.add_input("ND", [batch_size, dim], "float32")
    generator.add_input("ND", [m, ksub, int(dim / m)], "float32")

    generator.add_output("ND", [batch_size, m, ksub], "float32")
    generator.add_output("ND", [batch_size, m, int(ksub / 64 * 2)], "float32")
    generator.add_output("ND", [batch_size, m, int(ksub / 16)], "uint16")
    subspace_distance_obj.append(generator.generate_obj())
    utils.generate_op_config(subspace_distance_obj, file_path)


def generate_subspace_distance_offline_model():
    utils.set_env()
    args = arg_parse()
    process_id = args.process_id
    batch_size = args.batch_size
    m = args.m
    dim = args.dim
    ksub = args.ksub

    utils.check_param_range(m, [4], "m")
    utils.check_param_range(ksub, [256], "ksub")
    soc_version = utils.get_soc_version_from_npu_type(args.npu_type)

    core_num = utils.get_core_num_by_npu_type(args.core_num, args.npu_type)

    work_dir = '.'
    config_path = utils.get_config_path(work_dir)

    op_name_ = f"ascendc_ivfpq_subspace_distance_op_pid{process_id}"
    file_path_ = os.path.join(config_path, f"{op_name_}.json")

    generate_subspace_distance_json(batch_size, m, dim, ksub, file_path_)
    utils.atc_model(op_name_, soc_version)


if __name__ == '__main__':
    generate_subspace_distance_offline_model()
