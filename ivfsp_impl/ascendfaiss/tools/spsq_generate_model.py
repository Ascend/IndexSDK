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
import traceback
import common as utils

# number of code per time in distance_compute_flat_min64
SEARCH_LIST_SIZE = 32768
_LIST_NUM = 1024


def arg_parse():
    """
    Parse arguements to the operator model
    """

    parser = argparse.ArgumentParser(
        description='generate aicore operator model')

    parser.add_argument("--cores",
                        dest='core_num',
                        default=2,
                        type=int,
                        help="Core number")

    parser.add_argument("-d",
                        dest='dim',
                        default=256,
                        type=int,
                        help="Feature dimension")
    parser.add_argument("-d2",
                        dest='low_dim',
                        default=16,
                        type=int,
                        help="Feature dimension")
    parser.add_argument("-k",
                        dest='k',
                        default=256,
                        type=int,
                        help="Feature dimension")

    parser.add_argument("-p",
                        dest='process_id',
                        default=0,
                        type=int,
                        help="Number of process_id")

    parser.add_argument("-z",
                        dest='zregion_height',
                        default=16,
                        type=int,
                        help="the height of 'z' shaped storage format.")

    parser.add_argument("-code_num",
                        dest='code_num',
                        default=16384*34,
                        type=int,
                        help="code_num.")

    return parser.parse_args()


def generate_distance_sp64_sq8_l2_mins_json(code_num, core_num, query_num, dim, low_dim, k, zregion_height, file_path):
    # write dist_compute_sp_mins json
    dist_compute_sp_mins_obj = [
        {
            "op":
                "DistanceSp64IntL2Mins",
            "input_desc": [{
                "format": "ND",
                "shape": [query_num, dim],
                "type": "float16"
            }, {
                "format": "ND",
                "shape": [low_dim*k // zregion_height, dim // 16, zregion_height, 16],
                "type": "float16"
            }, {
                "format": "ND",
                "shape": [k, dim],
                "type": "float16"
            }, {
                "format": "ND",
                "shape": [code_num // zregion_height, low_dim // 16, zregion_height, 16],
                "type": "uint8"
            }, {
                "format": "ND",
                "shape": [code_num],
                "type": "float32"
            }, {
                "format": "ND",
                "shape": [core_num, 16],
                "type": "uint32"
            }, {
                "format": "ND",
                "shape": [core_num, 128],
                "type": "int32"
            }, {
                    "format": "ND",
                    "shape": [core_num, 128],
                    "type": "int32"
                }
            ],
            "output_desc": [{
                "format": "ND",
                "shape": [query_num, code_num],
                "type": "float16"
            }, {
                "format": "ND",
                "shape": [query_num, (code_num + 63) // 64 * 2],
                "type": "float16"
            }, {
                "format": "ND",
                "shape": [16, 16],
                "type": "uint16"
            }]
        }
    ]

    utils.generate_op_config(dist_compute_sp_mins_obj, file_path)


def generate_cid_filter_json(core_num, cid_num, file_path):
    # write cid_filter json
    cid_filter_obj = [
        {
            "op":
                "CidFilter",
            "input_desc": [{
                "format": "ND",
                "shape": [cid_num],
                "type": "uint8"
            }, {
                "format": "ND",
                "shape": [cid_num],
                "type": "int32"
            }, {
                "format": "ND",
                "shape": [cid_num],
                "type": "int32"
            }, {
                "format": "ND",
                "shape": [4, 64],
                "type": "int32"
            }, {
                "format": "ND",
                "shape": [8],
                "type": "int32"
            }, {
                "format": "ND",
                "shape": [4, 128],
                "type": "uint16"
            }, {
                "format": "ND",
                "shape": [4, 128],
                "type": "float16"
            }],
            "output_desc": [{
                "format": "ND",
                "shape": [cid_num // 16],
                "type": "uint16"
            }, {
                "format": "ND",
                "shape": [core_num, 16],
                "type": "uint16"
            }]
        }
    ]

    utils.generate_op_config(cid_filter_obj, file_path)


def generate_distance_masked_sp64_sq8_l2_mins_json(code_num, core_num, query_num, dim,
                                                   low_dim, k, zregion_height, file_path):
    # write dist_compute_sp_mins json
    dist_compute_sp_mins_obj = [
        {
            "op":
                "DistanceMaskedSp64IntL2Mins",
            "input_desc": [{
                "format": "ND",
                "shape": [query_num, dim],
                "type": "float16"
            }, {
                "format": "ND",
                "shape": [low_dim*k // zregion_height, dim // 16, zregion_height, 16],
                "type": "float16"
            }, {
                "format": "ND",
                "shape": [k, dim],
                "type": "float16"
            }, {
                "format": "ND",
                "shape": [code_num // zregion_height, low_dim // 16, zregion_height, 16],
                "type": "uint8"
            }, {
                "format": "ND",
                "shape": [code_num],
                "type": "float32"
            }, {
                "format": "ND",
                "shape": [core_num, 16],
                "type": "uint32"
            }, {
                "format": "ND",
                "shape": [core_num, 128],
                "type": "int32"
            }, {
                    "format": "ND",
                    "shape": [core_num, 128],
                    "type": "int32"
                },
                {"format": "ND",
                    "shape": [query_num, code_num//8],
                    "type": "uint8"}
            ],
            "output_desc": [{
                "format": "ND",
                "shape": [query_num, code_num],
                "type": "float16"
            }, {
                "format": "ND",
                "shape": [query_num, (code_num + 63) // 64 * 2],
                "type": "float16"
            }, {
                "format": "ND",
                "shape": [16, 16],
                "type": "uint16"
            }]
        }
    ]

    utils.generate_op_config(dist_compute_sp_mins_obj, file_path)


def generate_flat_offline_model():
    utils.set_env()
    args = arg_parse()
    core_num = args.core_num
    dim = args.dim
    low_dim = args.low_dim
    k = args.k
    zregion_height = args.zregion_height
    process_id = args.process_id
    code_num = args.code_num
    work_dir = '.'

    config_path = utils.get_config_path(work_dir)

    cid_filter_op_name = "cid_filter_op{}_pid{}"
    sp_sq8_l2_min_tune_op_name = "distance_sp64_sq8_l2_mins_op{}_pid{}"
    masked_sp_sq8_l2_min_op_name = "distance_masked_sp64_int_l2_mins_op{}_pid{}"
    cid_filter_op_name_ = cid_filter_op_name.format(code_num, process_id)


    file_path_ = os.path.join(config_path, '%s.json' % cid_filter_op_name_)
    generate_cid_filter_json(core_num, code_num, file_path_)
    utils.atc_model(cid_filter_op_name_)
    search_page_sizes = (32, 8, 1)
    

    for page_size in search_page_sizes:
       
        op_name_ = sp_sq8_l2_min_tune_op_name.format(page_size, process_id)
        file_path_ = os.path.join(config_path, '%s.json' % op_name_)
        generate_distance_sp64_sq8_l2_mins_json(code_num, core_num,
                                                page_size, dim, low_dim, k, zregion_height, file_path_)
        utils.atc_model(op_name_)

        op_name_ = masked_sp_sq8_l2_min_op_name.format(page_size, process_id)
        file_path_ = os.path.join(config_path, '%s.json' % op_name_)
        generate_distance_masked_sp64_sq8_l2_mins_json(code_num, core_num,
                                                       page_size, dim, low_dim, k, zregion_height, file_path_)
        utils.atc_model(op_name_)
    
        
if __name__ == '__main__':
    generate_flat_offline_model()
