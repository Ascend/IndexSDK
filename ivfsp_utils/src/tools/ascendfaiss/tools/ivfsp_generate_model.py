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
from multiprocessing import Pool

import common as utils


def check_param_range(param, valid_params, param_name):
    if param not in valid_params:
        raise ValueError("not support {0}: {1}, should be in {2}".format(param_name, param, valid_params))


def check_param_multiple(param, param_name):
    if param % 16 != 0:
        raise ValueError("not support {0}: {1}, should be a multiple of 16.".format(param_name, param))


def check_param_not_greater_than(param, compared, param_name):
    if param > compared:
        raise ValueError("not support {0}: {1}, should be <= {2}".format
                        (param_name, param, compared))


def check_pool_size(pool_size):
    if pool_size > 32 or pool_size < 1:
        raise ValueError("input variable pool_size {0}, should be in range[1, 32]".format(pool_size))


def arg_parse():
    """
    Parse arguements to the operator model
    """
    parser = argparse.ArgumentParser(
        description='generate aicore operator model')

    parser.add_argument("--cores",
                        dest='core_num',
                        default=8,
                        type=int,
                        help="Core number")

    parser.add_argument("-d",
                        dest='dim',
                        default=256,
                        type=int,
                        required=True,
                        help="Feature dimension")
    parser.add_argument("-nonzero_num",
                        dest='low_dim',
                        default=32,
                        type=int,
                        required=True,
                        help="Non-zero dimension after dimension reduction")
    parser.add_argument("-nlist",
                        dest='k',
                        default=1024,
                        type=int,
                        required=True,
                        help="centroids number")

    parser.add_argument("-p",
                        dest='process_id',
                        default=0,
                        type=int,
                        help="Number of process_id")

    parser.add_argument("-handle_batch",
                        dest='handle_batch',
                        default=32,
                        type=int,
                        required=True,
                        help="nprobe handle batch.")

    parser.add_argument("-code_num",
                        dest='code_num',
                        default=32768,
                        type=int,
                        required=True,
                        help="feature code search list size.")
    
    parser.add_argument("--pool",
                        dest="pool_size",
                        default=16,
                        type=int,
                        help="Number of pool_size")

    return parser.parse_args()


def generate_distance_compute_qc_json(core_num, query_num, dim, low_dim, k, zregion_height, file_path):
    dist_compute_sp_mins_obj = [{
        "op":
            "DistanceComputeQC",
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
            "shape": [k],
            "type": "float32"
        }],
        "output_desc": [{
            "format": "ND",
            "shape": [query_num, k],
            "type": "float16"
        }, {
            "format": "ND",
            "shape": [core_num, 16],
            "type": "uint16"
        }]
    }]

    utils.generate_op_config(dist_compute_sp_mins_obj, file_path)


def generate_distance_ivf_sp_int_l2_mins_json(code_num, handle_batch, dim, low_dim, k, zregion_height, file_path):
    dist_compute_sp_mins_obj = [{
        "op":
            "DistanceIVFSpIntL2Mins",
        "input_desc": [{
            "format": "ND",
            "shape": [1, dim],
            "type": "float16"
        }, {
            "format": "ND",
            "shape": [low_dim * k // zregion_height, dim // 16, zregion_height, 16],
            "type": "float16"
        },
            {
            "format": "ND",
            "shape": [code_num*low_dim],
            "type": "uint8"
        }, {
            "format": "ND",
            "shape": [code_num],
            "type": "float32"
        }, {
                "format": "ND",
                "shape": [low_dim],
                "type": "float16"
        }, {
                "format": "ND",
                "shape": [low_dim],
                "type": "float16"
        }, {
            "format": "ND",
            "shape": [handle_batch],
            "type": "uint32"
        }, {
            "format": "ND",
            "shape": [handle_batch],
            "type": "uint64"
        }, {
                "format": "ND",
                "shape": [handle_batch],
                "type": "uint64"
        }, {
                "format": "ND",
                "shape": [handle_batch],
                "type": "uint64"
        }
        ],
        "output_desc": [{
            "format": "ND",
            "shape": [handle_batch, code_num],
            "type": "float16"
        }, {
            "format": "ND",
            "shape": [handle_batch, (code_num + 63) // 64 * 2],
            "type": "float16"
        }, {
            "format": "ND",
            "shape": [8, 16],
            "type": "uint16"
        }]
    }]

    utils.generate_op_config(dist_compute_sp_mins_obj, file_path)


def generate_distance_masked_ivf_sp_int_l2_mins_json(code_num, handle_batch, dim, low_dim, k,
                                                     zregion_height, file_path):
    dist_compute_sp_mins_obj = [{
        "op":
            "DistanceMaskedIVFSpIntL2Mins",
        "input_desc": [{
            "format": "ND",
            "shape": [1, dim],
            "type": "float16"
        }, {
            "format": "ND",
            "shape": [low_dim * k // zregion_height, dim // 16, zregion_height, 16],
            "type": "float16"
        }, {
            "format": "ND",
            "shape": [code_num*low_dim],
            "type": "uint8"
        }, {
                "format": "ND",
                "shape": [handle_batch, code_num//8],
                "type": "uint8"
        }, {
            "format": "ND",
            "shape": [code_num],
            "type": "float32"
        }, {
                "format": "ND",
                "shape": [2, low_dim],
                "type": "float16"
        }, {
            "format": "ND",
            "shape": [handle_batch],
            "type": "uint32"
        }, {
            "format": "ND",
            "shape": [handle_batch],
            "type": "uint64"
        }, {
                "format": "ND",
                "shape": [handle_batch],
                "type": "uint64"
        }, {
                "format": "ND",
                "shape": [handle_batch],
                "type": "uint64"
        }
        ],
        "output_desc": [{
            "format": "ND",
            "shape": [handle_batch, code_num],
            "type": "float16"
        }, {
            "format": "ND",
            "shape": [handle_batch, (code_num + 63) // 64 * 2],
            "type": "float16"
        }, {
            "format": "ND",
            "shape": [8, 16],
            "type": "uint16"
        }]
    }]

    utils.generate_op_config(dist_compute_sp_mins_obj, file_path)


def generate_ivf_cid_filter3_json(handle_batch, core_num, cid_num, file_path):
    cid_filter_obj = [{
        "op":
            "IvfCidFilter3",
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
            "shape": [3, handle_batch],
            "type": "uint64"
        }, {
            "format": "ND",
            "shape": [handle_batch],
            "type": "uint32"
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
            "shape": [handle_batch, cid_num // 16],
            "type": "uint16"
        }, {
            "format": "ND",
            "shape": [core_num, 16],
            "type": "uint16"
        }]
    }]

    utils.generate_op_config(cid_filter_obj, file_path)


def generate_matmul_compute_qc_json(core_num, query_num, dim, low_dim, zregion_height, file_path):
    matmul_compute_qc_obj = [{
        "op":
            "MatmulCompute",
        "input_desc": [{
            "format": "ND",
            "shape": [query_num, dim],
            "type": "float16"
        }, {
            "format": "ND",
            "shape": [low_dim // zregion_height, dim // 16, zregion_height, 16],
            "type": "float16"
        }, {
            "format": "ND",
            "shape": [core_num, 8],
            "type": "uint32"
        }],
        "output_desc": [{
            "format": "ND",
            "shape": [query_num, low_dim],
            "type": "float16"
        }, {
            "format": "ND",
            "shape": [core_num, 16],
            "type": "uint16"
        }]
    }]

    utils.generate_op_config(matmul_compute_qc_obj, file_path)


def generate_fp_to_fp16_json(core_num, queries_num, dim, file_path):
    fp_to_fp16_obj = [{
        "op":
            "FpToFp16",
        "input_desc": [{
            "format": "ND",
            "shape": [queries_num, dim],
            "type": "float32"
        }],
        "output_desc": [{
            "format": "ND",
            "shape": [queries_num, dim],
            "type": "float16"
        }, {
            "format": "ND",
            "shape": [core_num, 16],
            "type": "uint16"
        }]
    }]

    utils.generate_op_config(fp_to_fp16_obj, file_path)


def generate_flat_offline_model():   
    utils.set_env()
    args = arg_parse()
    core_num = args.core_num
    dim = args.dim
    check_param_range(dim, [64, 128, 256, 512, 768], "dim")  
    low_dim = args.low_dim
    check_param_multiple(low_dim, "nonzero_num")
    check_param_not_greater_than(low_dim, dim, "nonzero_num")
    check_param_not_greater_than(low_dim, 128, "nonzero_num")
    
    # 空数组用来填装多进程执行的函数参数
    map_args = []
    check_pool_size(args.pool_size)
    
    soc_version = "Ascend310P3"
    if args.core_num == 2:
        soc_version = "Ascend310"
    else:
        soc_version = "Ascend310P3"

    k = args.k
    high_dims = [512, 768] # nlist should be <= 2048 if dim is in high_dims
    if dim not in high_dims:
        check_param_range(k, [256, 512, 1024, 2048, 4096, 8192, 16384], "centroid num")
    else:
        check_param_range(k, [256, 512, 1024, 2048], "centroid num")

    zregion_height = 16
    process_id = args.process_id
    handle_batch = args.handle_batch
    check_param_multiple(handle_batch, "handle_batch")
    handle_batch_min = 16
    handle_batch_max = 240
    if handle_batch < handle_batch_min or handle_batch > handle_batch_max:
        raise ValueError("not support {0}: {1}, {2} should be between {3} and {4}".format
                         ("handle_batch", handle_batch, "handle_batch", handle_batch_min, handle_batch_max))
    code_num = args.code_num
    work_dir = '.'
    config_path = utils.get_config_path(work_dir)

    ivf_cid_filter3_op_name = "ivf_cid_filter3_op{}_pid{}"
    compute_qc_op_name = "distance_compute_qc_op{}_pid{}"
    ivf_sp_sq8_l2_min_op_name = "distance_ivf_sp_int_l2_mins_op{}_pid{}"
    masked_ivf_sp_sq8_l2_min_tune_op_name = "distance_masked_ivf_sp_int_l2_mins_op{}_pid{}"
    fp2fp16_op_name = "fp_to_fp16_op_{}_pid{}"

    ivf_cid_filter3_op_name_ = ivf_cid_filter3_op_name.format(handle_batch, process_id)
    file_path_ = os.path.join(config_path, '%s.json' % ivf_cid_filter3_op_name_)
    generate_ivf_cid_filter3_json(handle_batch, core_num, code_num, file_path_)
    map_args.append((ivf_cid_filter3_op_name_, soc_version))

    op_name_ = masked_ivf_sp_sq8_l2_min_tune_op_name.format(1, process_id)
    file_path_ = os.path.join(config_path, '%s.json' % op_name_)
    generate_distance_masked_ivf_sp_int_l2_mins_json(code_num, handle_batch, dim, low_dim, k, zregion_height,
                                                     file_path_)
    map_args.append((op_name_, soc_version))

    op_name_ = ivf_sp_sq8_l2_min_op_name.format(1, process_id)
    file_path_ = os.path.join(config_path, '%s.json' % op_name_)
    generate_distance_ivf_sp_int_l2_mins_json(code_num, handle_batch, dim, low_dim, k, zregion_height, file_path_)
    map_args.append((op_name_, soc_version))

    search_page_sizes = (1024, 32, 16, 8, 4, 2, 1)
    for page_size in search_page_sizes:
        op_name_ = compute_qc_op_name.format(page_size, process_id)
        file_path_ = os.path.join(config_path, '%s.json' % op_name_)
        generate_distance_compute_qc_json(core_num, page_size, dim, low_dim, k, zregion_height, file_path_)
        map_args.append((op_name_, soc_version))

    search_optimize_page_sizes = (1024, 256, 128, 64, 48, 36, 32, 30, 24, 18, 16, 12, 8, 6, 4, 2, 1)
    for page_size in search_optimize_page_sizes:
        op_name_ = fp2fp16_op_name.format(page_size, process_id)
        file_path_ = os.path.join(config_path, '%s.json' % op_name_)
        generate_fp_to_fp16_json(core_num, page_size, dim, file_path_)
        map_args.append((op_name_, soc_version))
        
    # 使用多进程进行算子生成
    pool = Pool(args.pool_size)
    pool.starmap(utils.atc_model, map_args)
    pool.close()
    pool.join()


if __name__ == '__main__':
    generate_flat_offline_model()
