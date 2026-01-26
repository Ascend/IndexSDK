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
import argparse
import os
import common as utils
from common import OpJsonGenerator


def arg_parse():
    """
    Parse arguements to the operator model
    """
    parser = argparse.ArgumentParser(
        description='generate linear_transform operator model')
    utils.op_common_parse(parser, "--cores", 'core_num', 2, int, "Core number")
    utils.op_common_parse(parser, "-p", 'process_id', 0, int, "Number of process_id")
    utils.op_common_parse(parser, "-t", 'npu_type', "310", str,
        "NPU type, 310 / 310P / 910B1 / 910B2 / 910B3 / 910B4 / 910_{NPU Name} / Ascend950PR. 310 by default")
    return parser.parse_args()


def generate_topk_flat_int64_obj():
    generator = OpJsonGenerator("TopkFlat")
    generator.add_dynamic_input("ND", [-1, -1, -1], [[1, -1], [1, -1], [1, -1]], "float16")
    generator.add_dynamic_input("ND", [-1, -1, -1], [[1, -1], [1, -1], [1, -1]], "float16")
    generator.add_dynamic_input("ND", [-1, -1, -1], [[1, -1], [1, -1], [1, -1]], "uint32")
    generator.add_dynamic_input("ND", [-1, -1, -1], [[1, -1], [1, -1], [1, -1]], "uint16")
    generator.add_dynamic_input("ND", [-1], [[1, -1]], "int64")
    generator.add_dynamic_output("ND", [-1, -1], [[1, -1], [1, -1]], "float16")
    generator.add_dynamic_output("ND", [-1, -1], [[1, -1], [1, -1]], "int64")
    return generator.generate_obj()


def generate_topk_flat_fp32_obj():
    generator = OpJsonGenerator("TopkFlatFp32")
    generator.add_dynamic_input("ND", [-1, -1, -1], [[1, -1], [1, -1], [1, -1]], "float32")
    generator.add_dynamic_input("ND", [-1, -1, -1], [[1, -1], [1, -1], [1, -1]], "float32")
    generator.add_dynamic_input("ND", [-1, -1, -1], [[1, -1], [1, -1], [1, -1]], "uint32")
    generator.add_dynamic_input("ND", [-1, -1, -1], [[1, -1], [1, -1], [1, -1]], "uint16")
    generator.add_dynamic_input("ND", [-1], [[1, -1]], "int64")
    generator.add_dynamic_output("ND", [-1, -1], [[1, -1], [1, -1]], "float32")
    generator.add_dynamic_output("ND", [-1, -1], [[1, -1], [1, -1]], "int64")
    return generator.generate_obj()


def generate_topk_flat_uint32_obj():
    generator = OpJsonGenerator("TopkFlat")
    generator.add_dynamic_input("ND", [-1, -1, -1], [[1, -1], [1, -1], [1, -1]], "float16")
    generator.add_dynamic_input("ND", [-1, -1, -1], [[1, -1], [1, -1], [1, -1]], "float16")
    generator.add_dynamic_input("ND", [-1, -1, -1], [[1, -1], [1, -1], [1, -1]], "uint32")
    generator.add_dynamic_input("ND", [-1, -1, -1], [[1, -1], [1, -1], [1, -1]], "uint16")
    generator.add_dynamic_input("ND", [-1], [[1, -1]], "int64")
    generator.add_dynamic_output("ND", [-1, -1], [[1, -1], [1, -1]], "float16")
    generator.add_dynamic_output("ND", [-1, -1], [[1, -1], [1, -1]], "uint32")
    return generator.generate_obj()


def generate_topk_flat_uint16_obj():
    generator = OpJsonGenerator("TopkFlat")
    generator.add_dynamic_input("ND", [-1, -1, -1], [[1, -1], [1, -1], [1, -1]], "float16")
    generator.add_dynamic_input("ND", [-1, -1, -1], [[1, -1], [1, -1], [1, -1]], "float16")
    generator.add_dynamic_input("ND", [-1, -1, -1], [[1, -1], [1, -1], [1, -1]], "uint32")
    generator.add_dynamic_input("ND", [-1, -1, -1], [[1, -1], [1, -1], [1, -1]], "uint16")
    generator.add_dynamic_input("ND", [-1], [[1, -1]], "int64")
    generator.add_dynamic_output("ND", [-1, -1], [[1, -1], [1, -1]], "float16")
    generator.add_dynamic_output("ND", [-1, -1], [[1, -1], [1, -1]], "uint16")
    return generator.generate_obj()


def generate_topk_multisearch_obj():
    generator = OpJsonGenerator("TopkMultisearch")
    generator.add_dynamic_input("ND", [-1, -1, -1], [[1, -1], [1, -1], [1, -1]], "float16")
    generator.add_dynamic_input("ND", [-1, -1, -1], [[1, -1], [1, -1], [1, -1]], "float16")
    generator.add_dynamic_input("ND", [-1, -1, -1], [[1, -1], [1, -1], [1, -1]], "uint32")
    generator.add_dynamic_input("ND", [-1, -1, -1], [[1, -1], [1, -1], [1, -1]], "uint16")
    generator.add_dynamic_input("ND", [-1], [[1, -1]], "int64")
    generator.add_dynamic_input("ND", [-1], [[1, -1]], "uint32")
    generator.add_dynamic_input("ND", [-1], [[1, -1]], "uint32")
    generator.add_dynamic_input("ND", [-1], [[1, -1]], "uint16")
    generator.add_dynamic_output("ND", [-1, -1, -1], [[1, -1], [1, -1], [1, -1]], "float16")
    generator.add_dynamic_output("ND", [-1, -1, -1], [[1, -1], [1, -1], [1, -1]], "int64")
    return generator.generate_obj()


def generate_topk_ivf_obj():
    generator = OpJsonGenerator("TopkIvf")
    generator.add_dynamic_input("ND", [-1, -1, -1, -1], [[1, -1], [1, -1], [1, -1], [1, -1]], "float16")
    generator.add_dynamic_input("ND", [-1, -1, -1, -1], [[1, -1], [1, -1], [1, -1], [1, -1]], "float16")
    generator.add_dynamic_input("ND", [-1, -1, -1], [[1, -1], [1, -1], [1, -1]], "int64")
    generator.add_dynamic_input("ND", [-1, -1, -1], [[1, -1], [1, -1], [1, -1]], "uint32")
    generator.add_dynamic_input("ND", [-1, -1, -1, -1], [[1, -1], [1, -1], [1, -1], [1, -1]], "uint16")
    generator.add_dynamic_input("ND", [-1], [[1, -1]], "int64")
    generator.add_dynamic_output("ND", [-1, -1], [[1, -1], [1, -1]], "float16")
    generator.add_dynamic_output("ND", [-1, -1], [[1, -1], [1, -1]], "int64")
    return generator.generate_obj()


def generate_topk_ivf_fp32_obj():
    generator = OpJsonGenerator("TopkIvfFp32")
    generator.add_dynamic_input("ND", [-1, -1, -1, -1], [[1, -1], [1, -1], [1, -1], [1, -1]], "float")
    generator.add_dynamic_input("ND", [-1, -1, -1, -1], [[1, -1], [1, -1], [1, -1], [1, -1]], "float")
    generator.add_dynamic_input("ND", [-1, -1, -1], [[1, -1], [1, -1], [1, -1]], "int64")
    generator.add_dynamic_input("ND", [-1, -1, -1], [[1, -1], [1, -1], [1, -1]], "uint32")
    generator.add_dynamic_input("ND", [-1, -1, -1, -1], [[1, -1], [1, -1], [1, -1], [1, -1]], "uint16")
    generator.add_dynamic_input("ND", [-1], [[1, -1]], "int64")
    generator.add_dynamic_output("ND", [-1, -1], [[1, -1], [1, -1]], "float")
    generator.add_dynamic_output("ND", [-1, -1], [[1, -1], [1, -1]], "uint64")
    return generator.generate_obj()


def generate_topk_ivf_rabitq_fp32_obj():
    generator = OpJsonGenerator("TopkIvfRabitqFp32")
    generator.add_dynamic_input("ND", [-1, -1], [[1, -1], [1, -1]], "float")
    generator.add_dynamic_input("ND", [-1, -1], [[1, -1], [1, -1]], "float")
    generator.add_dynamic_input("ND", [-1, -1], [[1, -1], [1, -1]], "int64")
    generator.add_dynamic_input("ND", [-1, -1], [[1, -1], [1, -1]], "uint32")
    generator.add_dynamic_input("ND", [-1, -1], [[1, -1], [1, -1]], "uint16")
    generator.add_dynamic_input("ND", [-1], [[1, -1]], "int64")
    generator.add_dynamic_output("ND", [-1, -1], [[1, -1], [1, -1]], "float")
    generator.add_dynamic_output("ND", [-1, -1], [[1, -1], [1, -1]], "uint64")
    return generator.generate_obj()


def generate_topk_ivfpq_l3_obj():
    generator = OpJsonGenerator("TopkIvfpqL3")
    generator.add_dynamic_input("ND", [-1, -1, -1, -1], [[1, -1], [1, -1], [1, -1], [1, -1]], "int32")
    generator.add_dynamic_input("ND", [-1, -1, -1, -1], [[1, -1], [1, -1], [1, -1], [1, -1]], "float")
    generator.add_dynamic_input("ND", [-1, -1, -1], [[1, -1], [1, -1], [1, -1]], "int64")
    generator.add_dynamic_input("ND", [-1, -1, -1], [[1, -1], [1, -1], [1, -1]], "int64")
    generator.add_dynamic_input("ND", [-1, -1, -1, -1], [[1, -1], [1, -1], [1, -1], [1, -1]], "uint16")
    generator.add_dynamic_input("ND", [-1], [[1, -1]], "int64")
    generator.add_dynamic_output("ND", [-1, -1], [[1, -1], [1, -1]], "float")
    generator.add_dynamic_output("ND", [-1, -1], [[1, -1], [1, -1]], "uint64")
    return generator.generate_obj()


def generate_vec_l2sqr_obj():
    generator = OpJsonGenerator("VecL2Sqr")
    generator.add_dynamic_input("ND", [-1, -1], [[1, -1], [1, -1]], "float16")
    generator.add_dynamic_output("ND", [-1], [[1, -1]], "float16")
    return generator.generate_obj()


def generate_vec_l2sqr_flat_at_obj():
    generator = OpJsonGenerator("VecL2Sqr")
    generator.add_dynamic_input("ND", [-1, -1], [[1, -1], [1, -1]], "float16")
    generator.add_dynamic_output("ND", [-1, -1], [[1, -1], [1, -1]], "float")
    return generator.generate_obj()


def generate_vec_l2sqr_flat_at_int8_obj():
    generator = OpJsonGenerator("VecL2Sqr")
    generator.add_dynamic_input("ND", [-1, -1], [[1, -1], [1, -1]], "int8")
    generator.add_dynamic_output("ND", [-1, -1], [[1, -1], [1, -1]], "int32")
    return generator.generate_obj()


def generate_transdata_shaped_obj():
    generator = OpJsonGenerator("TransdataShaped")
    generator.add_dynamic_input("ND", [-1, -1], [[1, -1], [1, -1]], "float16")
    generator.add_dynamic_input("ND", [-1], [[1, -1]], "int64")
    generator.add_dynamic_output("ND", [-1, -1, -1, -1], [[1, -1], [1, -1], [1, -1], [1, -1]], "float16")
    return generator.generate_obj()


def generate_removedata_shaped_obj():
    generator = OpJsonGenerator("RemovedataShaped")
    generator.add_dynamic_input("ND", [-1], [[1, -1]], "uint64")
    generator.add_dynamic_input("ND", [-1], [[1, -1]], "int64")
    generator.add_dynamic_output("ND", [-1], [[1, -1]], "uint64")
    return generator.generate_obj()


def generate_removedata_attr_obj():
    generator = OpJsonGenerator("RemovedataAttr")
    generator.add_dynamic_input("ND", [-1], [[1, -1]], "uint64")
    generator.add_dynamic_input("ND", [-1], [[1, -1]], "int64")
    generator.add_dynamic_output("ND", [-1], [[1, -1]], "uint64")
    return generator.generate_obj()


def generate_transdata_custom_attr_obj():
    generator = OpJsonGenerator("TransdataCustomAttr")
    generator.add_dynamic_input("ND", [-1], [[1, -1]], "uint8")
    generator.add_dynamic_input("ND", [-1], [[1, -1]], "int64")
    generator.add_dynamic_output("ND", [-1], [[1, -1]], "uint8")
    return generator.generate_obj()


def generate_removedata_custom_attr_obj():
    generator = OpJsonGenerator("RemovedataCustomAttr")
    generator.add_dynamic_input("ND", [-1], [[1, -1]], "uint64")
    generator.add_dynamic_input("ND", [-1], [[1, -1]], "int64")
    generator.add_dynamic_output("ND", [-1], [[1, -1]], "uint64")
    return generator.generate_obj()


def generate_transdata_shaped_int8_obj():
    generator = OpJsonGenerator("TransdataShaped")
    generator.add_dynamic_input("ND", [-1, -1], [[1, -1], [1, -1]], "int8")
    generator.add_dynamic_input("ND", [-1], [[1, -1]], "int64")
    generator.add_dynamic_output("ND", [-1, -1, -1, -1], [[1, -1], [1, -1], [1, -1], [1, -1]], "int8")
    return generator.generate_obj()


def generate_transdata_shaped_uint8_obj():
    generator = OpJsonGenerator("TransdataShaped")
    generator.add_dynamic_input("ND", [-1, -1], [[1, -1], [1, -1]], "uint8")
    generator.add_dynamic_input("ND", [-1], [[1, -1]], "int64")
    generator.add_dynamic_output("ND", [-1, -1, -1, -1], [[1, -1], [1, -1], [1, -1], [1, -1]], "uint8")
    return generator.generate_obj()


def generate_transdata_raw_obj():
    generator = OpJsonGenerator("TransdataRaw")
    generator.add_dynamic_input("ND", [-1, -1, -1, -1], [[1, -1], [1, -1], [1, -1], [1, -1]], "float16")
    generator.add_dynamic_input("ND", [-1], [[1, -1]], "int64")
    generator.add_dynamic_output("ND", [-1, -1], [[1, -1], [1, -1]], "float16")
    return generator.generate_obj()


def generate_transdata_raw_int8_obj():
    generator = OpJsonGenerator("TransdataRaw")
    generator.add_dynamic_input("ND", [-1, -1, -1, -1], [[1, -1], [1, -1], [1, -1], [1, -1]], "int8")
    generator.add_dynamic_input("ND", [-1], [[1, -1]], "int64")
    generator.add_dynamic_output("ND", [-1, -1], [[1, -1], [1, -1]], "int8")
    return generator.generate_obj()


def generate_transdata_raw_uint8_obj():
    generator = OpJsonGenerator("TransdataRaw")
    generator.add_dynamic_input("ND", [-1, -1, -1, -1], [[1, -1], [1, -1], [1, -1], [1, -1]], "uint8")
    generator.add_dynamic_input("ND", [-1], [[1, -1]], "int64")
    generator.add_dynamic_output("ND", [-1, -1], [[1, -1], [1, -1]], "uint8")
    return generator.generate_obj()
    

def generate_km_update_centroids_obj():
    generator = OpJsonGenerator("KmUpdateCentroids")
    generator.add_dynamic_input("ND", [-1, -1], [[1, -1], [1, -1]], "float16")
    generator.add_dynamic_input("ND", [-1], [[1, -1]], "uint64")
    generator.add_dynamic_output("ND", [-1, -1], [[1, -1], [1, -1]], "float16")
    return generator.generate_obj()


def generate_residual_ivf_obj():
    generator = OpJsonGenerator("ResidualIvf")
    generator.add_dynamic_input("ND", [-1, -1], [[1, -1], [1, -1]], "float16")
    generator.add_dynamic_input("ND", [-1, -1], [[1, -1], [1, -1]], "float16")
    generator.add_dynamic_input("ND", [-1, -1], [[1, -1], [1, -1]], "uint64")
    generator.add_dynamic_output("ND", [-1, -1, -1], [[1, -1], [1, -1], [1, -1]], "float16")
    return generator.generate_obj()


def generate_codes_quantify_obj():
    generator = OpJsonGenerator("CodesQuantify")
    generator.add_dynamic_input("ND", [-1, -1], [[1, -1], [1, -1]], "float16")
    generator.add_dynamic_input("ND", [-1], [[1, -1]], "float16")
    generator.add_dynamic_output("ND", [-1, -1], [[1, -1], [1, -1]], "int8")
    return generator.generate_obj()


def generate_topk_ivf_fuzzy_obj():
    generator = OpJsonGenerator("TopkIvfFuzzy")
    generator.add_dynamic_input("ND", [-1, -1], [[1, -1], [1, -1]], "float16")
    generator.add_dynamic_input("ND", [-1, -1], [[1, -1], [1, -1]], "float16")
    generator.add_dynamic_input("ND", [-1, -1], [[1, -1], [1, -1]], "int64")
    generator.add_dynamic_input("ND", [-1, -1], [[1, -1], [1, -1]], "uint32")
    generator.add_dynamic_input("ND", [-1, -1, -1], [[1, -1], [1, -1], [1, -1]], "uint16")
    generator.add_dynamic_input("ND", [-1], [[1, -1]], "int64")
    generator.add_dynamic_output("ND", [-1, -1], [[1, -1], [1, -1]], "float16")
    generator.add_dynamic_output("ND", [-1, -1], [[1, -1], [1, -1]], "int64")
    generator.add_dynamic_output("ND", [-1, -1], [[1, -1], [1, -1]], "float16")
    generator.add_dynamic_output("ND", [-1, -1], [[1, -1], [1, -1]], "int64")
    return generator.generate_obj()


def generate_topk_ivfsqt_l1_obj():
    generator = OpJsonGenerator("TopkIvfsqtL1")
    generator.add_dynamic_input("ND", [-1, -1], [[1, -1], [1, -1]], "float16")
    generator.add_dynamic_input("ND", [-1, -1], [[1, -1], [1, -1]], "float16")
    generator.add_dynamic_input("ND", [-1, -1, -1], [[1, -1], [1, -1], [1, -1]], "uint16")
    generator.add_dynamic_input("ND", [-1], [[1, -1]], "int64")
    generator.add_dynamic_input("ND", [-1, -1], [[1, -1], [1, -1]], "float16")
    generator.add_dynamic_input("ND", [-1, -1], [[1, -1], [1, -1]], "int32")
    generator.add_dynamic_input("ND", [-1, -1], [[1, -1], [1, -1]], "float")
    generator.add_dynamic_output("ND", [-1, -1], [[1, -1], [1, -1]], "float16")
    generator.add_dynamic_output("ND", [-1, -1], [[1, -1], [1, -1]], "uint16")
    generator.add_dynamic_output("ND", [-1, -1], [[1, -1], [1, -1]], "float16")
    return generator.generate_obj()


def generate_topk_ivfsqt_l2_obj():
    generator = OpJsonGenerator("TopkIvfsqtL2")
    generator.add_dynamic_input("ND", [-1, -1], [[1, -1], [1, -1]], "float16")
    generator.add_dynamic_input("ND", [-1, -1, -1], [[1, -1], [1, -1], [1, -1]], "uint16")
    generator.add_dynamic_input("ND", [-1], [[1, -1]], "int64")
    generator.add_dynamic_input("ND", [-1], [[1, -1]], "int32")
    generator.add_dynamic_input("ND", [-1], [[1, -1]], "uint64")
    generator.add_dynamic_input("ND", [-1], [[1, -1]], "int64")
    generator.add_dynamic_input("ND", [-1], [[1, -1]], "uint32")
    generator.add_dynamic_input("ND", [-1, -1], [[1, -1], [1, -1]], "uint16")
    generator.add_dynamic_output("ND", [-1, -1], [[1, -1], [1, -1]], "uint64")
    generator.add_dynamic_output("ND", [-1, -1], [[1, -1], [1, -1]], "int64")
    generator.add_dynamic_output("ND", [-1, -1], [[1, -1], [1, -1]], "uint32")
    return generator.generate_obj()


def generate_aicpu_offline_model():
    utils.set_env()
    args = arg_parse()
    process_id = args.process_id
    soc_version = utils.get_soc_version_from_npu_type(args.npu_type)
    work_dir = '.'

    config_path = utils.get_config_path(work_dir)
    file_prefix = "aicpu_topk_ops{}".format(process_id)
    file_path = os.path.join(config_path, '{}.json'.format(file_prefix))

    ops_list = []
    ops_list.append(generate_topk_flat_uint32_obj())
    ops_list.append(generate_topk_flat_uint16_obj())
    ops_list.append(generate_topk_flat_int64_obj())
    ops_list.append(generate_topk_flat_fp32_obj())
    ops_list.append(generate_topk_multisearch_obj())
    ops_list.append(generate_topk_ivf_obj())
    ops_list.append(generate_topk_ivf_fp32_obj())
    ops_list.append(generate_topk_ivf_rabitq_fp32_obj())
    ops_list.append(generate_topk_ivfpq_l3_obj())
    ops_list.append(generate_vec_l2sqr_obj())
    ops_list.append(generate_vec_l2sqr_flat_at_obj())
    ops_list.append(generate_vec_l2sqr_flat_at_int8_obj())
    ops_list.append(generate_transdata_shaped_obj())
    ops_list.append(generate_removedata_shaped_obj())
    ops_list.append(generate_removedata_attr_obj())
    ops_list.append(generate_transdata_custom_attr_obj())
    ops_list.append(generate_removedata_custom_attr_obj())
    ops_list.append(generate_transdata_shaped_int8_obj())
    ops_list.append(generate_transdata_shaped_uint8_obj())
    ops_list.append(generate_transdata_raw_obj())
    ops_list.append(generate_transdata_raw_int8_obj())
    ops_list.append(generate_transdata_raw_uint8_obj())
    ops_list.append(generate_km_update_centroids_obj())
    ops_list.append(generate_residual_ivf_obj())
    ops_list.append(generate_codes_quantify_obj())
    ops_list.append(generate_topk_ivf_fuzzy_obj())
    ops_list.append(generate_topk_ivfsqt_l1_obj())
    ops_list.append(generate_topk_ivfsqt_l2_obj())

    utils.generate_op_config(ops_list, file_path)
    utils.atc_model(file_prefix, soc_version)

if __name__ == '__main__':
    generate_aicpu_offline_model()
