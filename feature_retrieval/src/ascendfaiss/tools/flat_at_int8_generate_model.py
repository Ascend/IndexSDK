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
import sys
import common as utils
from common import OpJsonGenerator

# number of code per time in distance_compute_flat_min64
_QUERY_NUM = 1024
_SUBCLUSTER_CODE_NUM = 64


def arg_parse():
    """
    Parse arguements to the operator model
    """

    parser = argparse.ArgumentParser(
        description='generate aicore operator model')

    utils.op_common_parse(parser, "--cores", 'core_num', 8, int, "Core number")
    utils.op_common_parse(parser, "-d", 'dim', 256, int, "Feature dimension")
    utils.op_common_parse(parser, "-c", 'code_num', 16384, int, "Number vector of base")
    utils.op_common_parse(parser, "-p", 'process_id', 0, int, "Number of process_id")
    utils.op_common_parse(parser, "--soc-version", 'soc_version', "Ascend310P3", str,
                          "soc_version, only support Ascend310P3.")
    utils.op_common_parse(parser, "-t", 'npu_type', "310P", str, "NPU type. 310P by default")
    return parser.parse_args()


def generate_l2_norm_typing_int8_at_json(core_num, page_num, dim, file_path):
    # write dist_compute_flat_mins json
    l2_norm_typing_int8_obj = []
    generator = OpJsonGenerator("L2NormTypingInt8")
    generator.add_input("ND", [page_num, dim], "int8")
    generator.add_input("ND", [256, 16], "int32")
    generator.add_input("ND", [core_num, 8], "uint32")
    generator.add_output("ND", [page_num], "int32")
    generator.add_output("ND", [page_num // 32, dim // 32, 32, 32], "int8")
    l2_norm_typing_int8_obj.append(generator.generate_obj())
    utils.generate_op_config(l2_norm_typing_int8_obj, file_path)


def generate_distance_l2_mins_int8_at_json(core_num, query_num, code_num, dim, file_path):
    # write dist_compute_int8_mins json
    dist_l2_mins_int8_obj = []
    generator = OpJsonGenerator("DistanceL2MinsInt8At")
    generator.add_input("ND", [query_num // 32, dim // 32, 32, 32], "int8")
    generator.add_input("ND", [code_num // 512, dim // 32, 512, 32], "int8")
    generator.add_input("ND", [query_num], "int32")
    generator.add_input("ND", [code_num // 512, 1, 512, 16], "int32")
    generator.add_output("ND", [query_num, code_num], "float16")
    generator.add_output("ND", [query_num, code_num // 64 * 2], "float16")
    generator.add_output("ND", [core_num, 16], "uint16")
    dist_l2_mins_int8_obj.append(generator.generate_obj())
    utils.generate_op_config(dist_l2_mins_int8_obj, file_path)


def generate_distance_flat_l2_at_json(core_num, dim, file_path):
    # write dist_compute_flat_l2_at json
    dist_flat_l2_at_obj = []
    generator = OpJsonGenerator("DistanceFlatL2At")
    generator.add_input("ND", [1, dim], "uint8")
    generator.add_input("ND", [8, ], "uint32")
    generator.add_input("ND",
                [_SUBCLUSTER_CODE_NUM // _SUBCLUSTER_CODE_NUM, dim // 16, _SUBCLUSTER_CODE_NUM, 16], "float16")
    generator.add_input("ND", [_SUBCLUSTER_CODE_NUM // _SUBCLUSTER_CODE_NUM, 1, _SUBCLUSTER_CODE_NUM, 16], "float32")
    generator.add_input("ND", [2, dim], "float16")
    generator.add_output("ND", [core_num, _SUBCLUSTER_CODE_NUM, dim], "float16")
    generator.add_output("ND", [core_num, _SUBCLUSTER_CODE_NUM], "int16")
    generator.add_output("ND", [1, ], "uint16")
    generator.add_output("ND", [core_num, 16], "uint16")
    dist_flat_l2_at_obj.append(generator.generate_obj())
    utils.generate_op_config(dist_flat_l2_at_obj, file_path)


def generate_l2_norm_flat_sub_json(dim, file_path):
    # write l2_norm_flat_sub json
    l2_norm_flat_sub_obj = []
    generator = OpJsonGenerator("L2NormFlatSub")
    generator.add_input("ND", [_SUBCLUSTER_CODE_NUM, dim], "float16")
    generator.add_output("ND", [_SUBCLUSTER_CODE_NUM, 16], "float32")
    generator.add_output("ND", [_SUBCLUSTER_CODE_NUM * dim], "float16")
    l2_norm_flat_sub_obj.append(generator.generate_obj())
    utils.generate_op_config(l2_norm_flat_sub_obj, file_path)


def generate_subcents_accum_json(core_num, file_path):
    # write dist_compute_flat_l2_at json
    subcnts_accum_obj = []
    generator = OpJsonGenerator("SubcentAccum")
    generator.add_input("ND", [1, 8, 64, 256], "float16")
    generator.add_input("ND", [1, 8, 64], "int16")
    generator.add_input("ND", [16, ], "uint16")
    generator.add_output("ND", [64, 256], "float16")
    generator.add_output("ND", [64, ], "int16")
    generator.add_output("ND", [core_num, 16], "uint16")
    subcnts_accum_obj.append(generator.generate_obj())
    utils.generate_op_config(subcnts_accum_obj, file_path)


def generate_flat_offline_model():
    utils.set_env()
    args = arg_parse()
    process_id = args.process_id
    dim = args.dim
    utils.check_param_range(dim, [256], "dim")

    code_num = args.code_num
    utils.check_param_range(code_num, [1024, 2048, 4096, 8192, 16384, 32768], "code_num")

    soc_version = args.soc_version
    # 用户输入-t，则以-t为准
    if "-t" in str(sys.argv):
        if args.npu_type != "310P":
            raise RuntimeError("NPU type only support 310P now!")

    soc_version = utils.get_soc_version_from_npu_type(args.npu_type)
    if soc_version != "Ascend310P3":
        raise RuntimeError("soc_version only support Ascend310P3 now!")

    core_num = utils.get_core_num_by_npu_type(args.core_num, args.npu_type)
    work_dir = '.'
    try:
        page_num = 32768 * 8192 / code_num
    except ZeroDivisionError as e:
        raise Exception("Code_num can not be zero!") from e
    config_path = utils.get_config_path(work_dir)

    op_name_ = f"_l2_norm_typing_int8_op_pid{process_id}"
    file_path_ = os.path.join(config_path, f"{op_name_}.json")
    generate_l2_norm_typing_int8_at_json(core_num, page_num, dim, file_path_)
    utils.atc_model(op_name_, soc_version)

    op_name_ = f"int8_l2_mins_at_op_pid{process_id}"
    file_path_ = os.path.join(config_path, f"{op_name_}.json")
    generate_distance_l2_mins_int8_at_json(core_num, _QUERY_NUM, code_num, dim, file_path_)
    utils.atc_model(op_name_, soc_version)

    op_name_ = f"flat_l2_at_op_pid{process_id}"
    file_path_ = os.path.join(config_path, f"{op_name_}.json")
    generate_distance_flat_l2_at_json(core_num, dim, file_path_)
    utils.atc_model(op_name_, soc_version)

    op_name_ = f"l2_norm_flat_sub_op_pid{process_id}"
    file_path_ = os.path.join(config_path, f"{op_name_}.json")
    generate_l2_norm_flat_sub_json(dim, file_path_)
    utils.atc_model(op_name_, soc_version)

    op_name_ = f"subcents_accum_op_pid{process_id}"
    file_path_ = os.path.join(config_path, f"{op_name_}.json")
    generate_subcents_accum_json(core_num, file_path_)
    utils.atc_model(op_name_, soc_version)


if __name__ == '__main__':
    generate_flat_offline_model()
