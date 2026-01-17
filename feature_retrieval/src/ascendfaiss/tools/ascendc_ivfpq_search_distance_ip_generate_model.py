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
import common as utils
from common import OpJsonGenerator


def arg_parse():
    """
    Parse arguements to the operator model
    """

    parser = argparse.ArgumentParser(
        description='generate aicore operator model')

    utils.op_common_parse(parser, "--cores", 'core_num', 56, int, "Core number")
    utils.op_common_parse(parser, "-m", 'm', 4, int, "number of sub space")
    utils.op_common_parse(parser, "-k", 'ksub', 256, int, "number of sub space center")
    utils.op_common_parse(parser, "-topK", 'topK', 320, int, "number of topk")
    utils.op_common_parse(parser, "-maxSize", 'maxSize', 16384, int, "number of maxSize")
    utils.op_common_parse(parser, "-b", 'blockNum', 56, int, "Number of block")
    utils.op_common_parse(parser, "-p", 'process_id', 0, int, "Number of process_id")
    utils.op_common_parse(parser, "-t", 'npu_type', "910_9579", str, "NPU type, 910_9579 by default")

    return parser.parse_args()


def generate_search_distance_ip_json(m, ksub, blockNum, blockSize, topK, maxSize, file_path):
    # write dist_compute_flat_mins json
    search_distance_obj = []
    generator = OpJsonGenerator("AscendcIvfpqSearchDistanceIP")
    generator.add_input("ND", [m, ksub,], "float32")
    generator.add_input("ND", [blockSize * m], "uint8")
    generator.add_input("ND", [blockNum], "int64")
    generator.add_input("ND", [blockNum], "int64")
    generator.add_input("ND", [topK, maxSize], "int32")

    generator.add_output("ND", [blockNum, blockSize], "float32")
    generator.add_output("ND", [blockNum, topK], "int32")
    generator.add_output("ND", [blockNum, topK], "float32")
    generator.add_output("ND", [blockNum, 16], "uint16")
    search_distance_obj.append(generator.generate_obj())
    utils.generate_op_config(search_distance_obj, file_path)



def generate_search_distance_ip_offline_model():
    utils.set_env()
    args = arg_parse()
    process_id = args.process_id
    m = args.m
    topk = args.topK
    maxSize = args.maxSize
    ksub = args.ksub
    blockNum = args.blockNum

    utils.check_param_range(m, [4], "m")
    utils.check_param_range(ksub, [256], "ksub")
    soc_version = utils.get_soc_version_from_npu_type(args.npu_type)

    core_num = utils.get_core_num_by_npu_type(args.core_num, args.npu_type)

    work_dir = '.'
    blockSize = 16384 * 16
    config_path = utils.get_config_path(work_dir)

    op_name_ = f"ascendc_ivfpq_search_distance_ip_op_pid{process_id}"
    file_path_ = os.path.join(config_path, f"{op_name_}.json")
    generate_search_distance_json(m, ksub, blockNum, blockSize, topk, maxSize, file_path_)
    utils.atc_model(op_name_, soc_version)



if __name__ == '__main__':
    generate_search_distance_ip_offline_model()
