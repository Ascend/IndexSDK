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
import argparse
import os
import common as utils


def arg_parse():
    """
    Parse arguements to the operator model
    """
    parser = argparse.ArgumentParser(
        description='generate linear_transform operator model')

    parser.add_argument("--cores",
                        dest='core_num',
                        default=8,
                        type=int,
                        help="Core number")

    parser.add_argument("-p",
                        dest='process_id',
                        default=0,
                        type=int,
                        help="Number of process_id")

    return parser.parse_args()


def generate_topk_ivfsp_l1_int64_obj():
    topk_flat_obj = {
        "op": "TopkIvfSpL1",
        "input_desc": [{
            "format": "ND",
            "shape": [-1, -1],
            "shape_range": [[1, -1], [1, -1]],
            "type": "float16"
        }, {
            "format": "ND",
            "shape": [-1, -1],
            "shape_range": [[1, -1], [1, -1]],
            "type": "uint32"
        }, {
            "format": "ND",
            "shape": [-1, -1],
            "shape_range": [[1, -1], [1, -1]],
            "type": "uint16"
        }, {
            "format": "ND",
            "shape": [-1],
            "shape_range": [[1, -1]],
            "type": "int64"
        }],
        "output_desc": [{
            "format": "ND",
            "shape": [-1, -1],
            "shape_range": [[1, -1], [1, -1]],
            "type": "float16"
        }, {
            "format": "ND",
            "shape": [-1, -1],
            "shape_range": [[1, -1], [1, -1]],
            "type": "int64"
        }]
    }

    return topk_flat_obj


def generate_topk_spsq_int64_obj():
    topk_flat_obj = {
        "op": "TopkSPSQ",
        "input_desc": [{
            "format": "ND",
            "shape": [-1, -1, -1],
            "shape_range": [[1, -1], [1, -1], [1, -1]],
            "type": "float16"
        }, {
            "format": "ND",
            "shape": [-1, -1, -1],
            "shape_range": [[1, -1], [1, -1], [1, -1]],
            "type": "float16"
        }, {
            "format": "ND",
            "shape": [-1, -1, -1],
            "shape_range": [[1, -1], [1, -1], [1, -1]],
            "type": "uint32"
        }, {
            "format": "ND",
            "shape": [-1, -1, -1],
            "shape_range": [[1, -1], [1, -1], [1, -1]],
            "type": "uint16"
        }, {
            "format": "ND",
            "shape": [-1],
            "shape_range": [[1, -1]],
            "type": "int64"
        }, {
            "format": "ND",
            "shape": [-1],
            "shape_range": [[1, -1]],
            "type": "int64"
        }],
        "output_desc": [{
            "format": "ND",
            "shape": [-1, -1],
            "shape_range": [[1, -1], [1, -1]],
            "type": "float16"
        }, {
            "format": "ND",
            "shape": [-1, -1],
            "shape_range": [[1, -1], [1, -1]],
            "type": "int64"
        }]
    }

    return topk_flat_obj


def generate_topk_ivf_sp_obj():
    topk_ivf_sp_obj = {
        "op": "TopkIvfSp",
        "input_desc": [{
            "format": "ND",
            "shape": [-1, -1, -1, -1],
            "shape_range": [[1, -1], [1, -1], [1, -1], [1, -1]],
            "type": "float16"
        }, {
            "format": "ND",
            "shape": [-1, -1, -1, -1],
            "shape_range": [[1, -1], [1, -1], [1, -1], [1, -1]],
            "type": "float16"
        }, {
            "format": "ND",
            "shape": [-1, -1, -1],
            "shape_range": [[1, -1], [1, -1], [1, -1]],
            "type": "int64"
        }, {
            "format": "ND",
            "shape": [-1, -1, -1],
            "shape_range": [[1, -1], [1, -1], [1, -1]],
            "type": "uint32"
        }, {
            "format": "ND",
            "shape": [-1, -1, -1, -1],
            "shape_range": [[1, -1], [1, -1], [1, -1], [1, -1]],
            "type": "uint16"
        }, {
            "format": "ND",
            "shape": [-1],
            "shape_range": [[1, -1]],
            "type": "int64"
        }],
        "output_desc": [{
            "format": "ND",
            "shape": [-1, -1],
            "shape_range": [[1, -1], [1, -1]],
            "type": "float16"
        }, {
            "format": "ND",
            "shape": [-1, -1],
            "shape_range": [[1, -1], [1, -1]],
            "type": "int64"
        }]
    }

    return topk_ivf_sp_obj


def generate_vec_l2sqr_sp_obj():
    vec_l2sqr_sp_obj = {
        "op": "VecL2SqrSp",
        "input_desc": [{
            "format": "ND",
            "shape": [-1, -1],
            "shape_range": [[1, -1], [1, -1]],
            "type": "float16"
        }],
        "output_desc": [{
            "format": "ND",
            "shape": [-1],
            "shape_range": [[1, -1]],
            "type": "float16"
        }]
    }

    return vec_l2sqr_sp_obj


def generate_vec_l2sqr_sp_flat_at_obj():
    vec_l2sqr_sp_obj = {
        "op": "VecL2SqrSp",
        "input_desc": [{
            "format": "ND",
            "shape": [-1, -1],
            "shape_range": [[1, -1], [1, -1]],
            "type": "float16"
        }],
        "output_desc": [{
            "format": "ND",
            "shape": [-1, -1],
            "shape_range": [[1, -1], [1, -1]],
            "type": "float"
        }]
    }

    return vec_l2sqr_sp_obj


def generate_vec_l2sqr_sp_flat_at_int8_obj():
    vec_l2sqr_sp_obj = {
        "op": "VecL2SqrSp",
        "input_desc": [{
            "format": "ND",
            "shape": [-1, -1],
            "shape_range": [[1, -1], [1, -1]],
            "type": "int8"
        }],
        "output_desc": [{
            "format": "ND",
            "shape": [-1, -1],
            "shape_range": [[1, -1], [1, -1]],
            "type": "int32"
        }]
    }

    return vec_l2sqr_sp_obj


def generate_transdata_shaped_sp_obj():
    transdata_shaped_sp_obj = {
        "op": "TransdataShapedSp",
        "input_desc": [{
            "format": "ND",
            "shape": [-1, -1],
            "shape_range": [[1, -1], [1, -1]],
            "type": "float16"
        }, {
            "format": "ND",
            "shape": [-1],
            "shape_range": [[1, -1]],
            "type": "int64"
        }],
        "output_desc": [{
            "format": "ND",
            "shape": [-1, -1, -1, -1],
            "shape_range": [[1, -1], [1, -1], [1, -1], [1, -1]],
            "type": "float16"
        }]
    }

    return transdata_shaped_sp_obj


def generate_transdata_shaped_sp_int8_obj():
    transdata_shaped_sp_obj = {
        "op": "TransdataShapedSp",
        "input_desc": [{
            "format": "ND",
            "shape": [-1, -1],
            "shape_range": [[1, -1], [1, -1]],
            "type": "int8"
        }, {
            "format": "ND",
            "shape": [-1],
            "shape_range": [[1, -1]],
            "type": "int64"
        }],
        "output_desc": [{
            "format": "ND",
            "shape": [-1, -1, -1, -1],
            "shape_range": [[1, -1], [1, -1], [1, -1], [1, -1]],
            "type": "int8"
        }]
    }

    return transdata_shaped_sp_obj


def generate_transdata_shaped_sp_uint8_obj():
    transdata_shaped_sp_obj = {
        "op": "TransdataShapedSp",
        "input_desc": [{
            "format": "ND",
            "shape": [-1, -1],
            "shape_range": [[1, -1], [1, -1]],
            "type": "uint8"
        }, {
            "format": "ND",
            "shape": [-1],
            "shape_range": [[1, -1]],
            "type": "int64"
        }],
        "output_desc": [{
            "format": "ND",
            "shape": [-1, -1, -1, -1],
            "shape_range": [[1, -1], [1, -1], [1, -1], [1, -1]],
            "type": "uint8"
        }]
    }

    return transdata_shaped_sp_obj


def generate_transdata_raw_sp_obj():
    transdata_raw_sp_obj = {
        "op": "TransdataRawSp",
        "input_desc": [{
            "format": "ND",
            "shape": [-1, -1, -1, -1],
            "shape_range": [[1, -1], [1, -1], [1, -1], [1, -1]],
            "type": "float16"
        }, {
            "format": "ND",
            "shape": [-1],
            "shape_range": [[1, -1]],
            "type": "int64"
        }],
        "output_desc": [{
            "format": "ND",
            "shape": [-1, -1],
            "shape_range": [[1, -1], [1, -1]],
            "type": "float16"
        }]
    }

    return transdata_raw_sp_obj


def generate_transdata_raw_sp_int8_obj():
    transdata_raw_sp_obj = {
        "op": "TransdataRawSp",
        "input_desc": [{
            "format": "ND",
            "shape": [-1, -1, -1, -1],
            "shape_range": [[1, -1], [1, -1], [1, -1], [1, -1]],
            "type": "int8"
        }, {
            "format": "ND",
            "shape": [-1],
            "shape_range": [[1, -1]],
            "type": "int64"
        }],
        "output_desc": [{
            "format": "ND",
            "shape": [-1, -1],
            "shape_range": [[1, -1], [1, -1]],
            "type": "int8"
        }]
    }

    return transdata_raw_sp_obj


def generate_transdata_raw_sp_uint8_obj():
    transdata_raw_sp_obj = {
        "op": "TransdataRawSp",
        "input_desc": [{
            "format": "ND",
            "shape": [-1, -1, -1, -1],
            "shape_range": [[1, -1], [1, -1], [1, -1], [1, -1]],
            "type": "uint8"
        }, {
            "format": "ND",
            "shape": [-1],
            "shape_range": [[1, -1]],
            "type": "int64"
        }],
        "output_desc": [{
            "format": "ND",
            "shape": [-1, -1],
            "shape_range": [[1, -1], [1, -1]],
            "type": "uint8"
        }]
    }

    return transdata_raw_sp_obj


def generate_topk_multisearch_ivf_obj():
    topk_ivf_obj = {
        "op": "TopkMultisearchIvf",
        "input_desc": [{
            "format": "ND",
            "shape": [-1, -1, -1, -1],
            "shape_range": [[1, -1], [1, -1], [1, -1], [1, -1]],
            "type": "float16"
        }, {
            "format": "ND",
            "shape": [-1, -1, -1, -1],
            "shape_range": [[1, -1], [1, -1], [1, -1], [1, -1]],
            "type": "float16"
        }, {
            "format": "ND",
            "shape": [-1, -1, -1, -1],
            "shape_range": [[1, -1], [1, -1], [1, -1], [1, -1]],
            "type": "int64"
        }, {
            "format": "ND",
            "shape": [-1, -1, -1, -1],
            "shape_range": [[1, -1], [1, -1], [1, -1], [1, -1]],
            "type": "uint32"
        }, {
            "format": "ND",
            "shape": [-1, -1, -1, -1, -1],
            "shape_range": [[1, -1], [1, -1], [1, -1], [1, -1], [1, -1]],
            "type": "uint16"
        }, {
            "format": "ND",
            "shape": [-1],
            "shape_range": [[1, -1]],
            "type": "int64"
        }],
        "output_desc": [{
            "format": "ND",
            "shape": [-1, -1, -1],
            "shape_range": [[1, -1], [1, -1], [1, -1]],
            "type": "float16"
        }, {
            "format": "ND",
            "shape": [-1, -1, -1],
            "shape_range": [[1, -1], [1, -1], [1, -1]],
            "type": "int64"
        }]
    }

    return topk_ivf_obj


def generate_topk_multisearch_ivf_v2_obj():
    topk_ivf_obj = {
        "op": "TopkMultisearchIvfV2",
        "input_desc": [{
            "format": "ND",
            "shape": [-1, -1, -1, -1],
            "shape_range": [[1, -1], [1, -1], [1, -1], [1, -1]],
            "type": "float16"
        }, {
            "format": "ND",
            "shape": [-1, -1, -1, -1],
            "shape_range": [[1, -1], [1, -1], [1, -1], [1, -1]],
            "type": "float16"
        }, {
            "format": "ND",
            "shape": [-1, -1, -1, -1],
            "shape_range": [[1, -1], [1, -1], [1, -1], [1, -1]],
            "type": "int64"
        }, {
            "format": "ND",
            "shape": [-1, -1, -1, -1],
            "shape_range": [[1, -1], [1, -1], [1, -1], [1, -1]],
            "type": "uint32"
        }, {
            "format": "ND",
            "shape": [-1, -1, -1, -1, -1],
            "shape_range": [[1, -1], [1, -1], [1, -1], [1, -1], [1, -1]],
            "type": "uint16"
        }, {
            "format": "ND",
            "shape": [-1],
            "shape_range": [[1, -1]],
            "type": "int64"
        }
        ],
        "output_desc": [{
            "format": "ND",
            "shape": [-1, -1, -1],
            "shape_range": [[1, -1], [1, -1], [1, -1]],
            "type": "float16"
        }, {
            "format": "ND",
            "shape": [-1, -1, -1],
            "shape_range": [[1, -1], [1, -1], [1, -1]],
            "type": "int64"
        }]
    }

    return topk_ivf_obj


def generate_aicpu_offline_model():
    utils.set_env()
    args = arg_parse()
    process_id = args.process_id
    work_dir = '.'

    if args.core_num not in [2, 8]:
        raise RuntimeError("core num only support 2 or 8 now!")

    if args.core_num == 2:
        soc_version = "Ascend310"
    else:
        soc_version = "Ascend310P3"

    config_path = utils.get_config_path(work_dir)
    file_prefix = "aicpu_topk_ops{}".format(process_id)
    file_path = os.path.join(config_path, '{}.json'.format(file_prefix))

    ops_list = []
    ops_list.append(generate_topk_ivf_sp_obj())
    ops_list.append(generate_vec_l2sqr_sp_obj())
    ops_list.append(generate_vec_l2sqr_sp_flat_at_obj())
    ops_list.append(generate_vec_l2sqr_sp_flat_at_int8_obj())
    ops_list.append(generate_transdata_shaped_sp_obj())
    ops_list.append(generate_transdata_shaped_sp_int8_obj())
    ops_list.append(generate_transdata_shaped_sp_uint8_obj())
    ops_list.append(generate_transdata_raw_sp_obj())
    ops_list.append(generate_transdata_raw_sp_int8_obj())
    ops_list.append(generate_transdata_raw_sp_uint8_obj())
    ops_list.append(generate_topk_spsq_int64_obj())
    ops_list.append(generate_topk_ivfsp_l1_int64_obj())
    ops_list.append(generate_topk_multisearch_ivf_obj())
    ops_list.append(generate_topk_multisearch_ivf_v2_obj())

    utils.generate_op_config(ops_list, file_path)
    utils.atc_model(file_prefix, soc_version)

if __name__ == '__main__':
    generate_aicpu_offline_model()
