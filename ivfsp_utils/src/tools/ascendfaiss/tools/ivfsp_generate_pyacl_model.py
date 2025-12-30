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
import subprocess
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

    parser.add_argument("-batch_size",
                        dest='batch_size',
                        default=32768,
                        type=int,
                        help="Feature code train batch size")

    parser.add_argument("-code_num",
                        dest='codebook_batch_size',
                        default=32768,
                        type=int,
                        required=True,
                        help="codebook feature code access batch size")

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
            "type": "float16"
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


def generate_matmul_compute_qc_json(core_num, query_num, dim, low_dim, zregion_height, file_path):
    matmul_compute_qc_obj = [{
        "op":
            "MatmulCompute",
        "input_desc": [{
            "format": "ND",
            "shape": [low_dim, dim],
            "type": "float16"
        }, {
            "format": "ND",
            "shape": [query_num // zregion_height, dim // 16, zregion_height, 16],
            "type": "float16"
        }, {
            "format": "ND",
            "shape": [core_num, 8],
            "type": "uint32"
        }],
        "output_desc": [{
            "format": "ND",
            "shape": [low_dim, query_num],
            "type": "float16"
        }, {
            "format": "ND",
            "shape": [core_num, 16],
            "type": "uint16"
        }]
    }]

    utils.generate_op_config(matmul_compute_qc_obj, file_path)


def generate_matmul_compute_cb_json(core_num, code_num, dim, low_dim, zregion_height, file_path):
    matmul_compute_cb_obj = [{
        "op":
            "MatmulCompute",
        "input_desc": [{
            "format": "ND",
            "shape": [dim, low_dim],
            "type": "float16"
        }, {
            "format": "ND",
            "shape": [code_num // zregion_height, low_dim // 16, zregion_height, 16],
            "type": "float16"
        }, {
            "format": "ND",
            "shape": [core_num, 8],
            "type": "uint32"
        }],
        "output_desc": [{
            "format": "ND",
            "shape": [dim, code_num],
            "type": "float16"
        }, {
            "format": "ND",
            "shape": [core_num, 16],
            "type": "uint16"
        }]
    }]
    utils.generate_op_config(matmul_compute_cb_obj, file_path)


def generate_matmul_compute_ATA_json(core_num, code_num, dim, file_path):
    matmul_compute_ata_obj = [{
        "op":
            "CorrCompute",
        "input_desc": [{
            "format": "ND",
            "shape": [1, code_num//16, dim, 16],
            "type": "float16"
        }, {
            "format": "ND",
            "shape": [core_num],
            "type": "uint64"
        }],
        "output_desc": [{
            "format": "ND",
            "shape": [dim, dim],
            "type": "float16"
        }, {
            "format": "ND",
            "shape": [core_num, 16],
            "type": "uint16"
        }]
    }]

    utils.generate_op_config(matmul_compute_ata_obj, file_path)


def atc_model(json_file, output_path, soc_version="Ascend310"):
    ascend_toolkit_path = os.path.join(os.environ['ASCEND_HOME'], os.environ['ASCEND_VERSION'])
    atc_path = os.path.join(ascend_toolkit_path, 'atc/bin/atc')
    return_code = subprocess.call([
        atc_path, '--singleop=./config/%s.json' % json_file,
        '--soc_version=%s' % soc_version, f'--output={output_path}'
    ], shell=False)

    if return_code:
        raise Exception("Failed to generate op models.")



def generate_flat_offline_model():
    utils.set_env()
    args = arg_parse()
    core_num = args.core_num
    dim = args.dim
    check_param_range(dim, [64, 128, 256, 512, 768], "dim")
    low_dim = args.low_dim
    check_param_multiple(low_dim, "nonzero num")
    check_param_not_greater_than(low_dim, dim, "nonzero_num")
    check_param_not_greater_than(low_dim, 128, "nonzero_num")
    
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
    
    batch_size_limit = 32768 # upper limit for both batch_size and codebook_batch_size
    batch_size = args.batch_size
    check_param_multiple(batch_size, "batch_size")
    check_param_not_greater_than(batch_size, batch_size_limit, "batch_size")
    codebook_batch_size = args.codebook_batch_size
    check_param_multiple(codebook_batch_size, "codebook_batch_size")
    check_param_not_greater_than(codebook_batch_size, batch_size_limit, "codebook_batch_size")
    work_dir = '.'
    output_path = "op_models_pyacl"

    config_path = utils.get_config_path(work_dir)

    compute_qc_op_name = "distance_compute_qc_op{}_pid{}"
    matmul_compute_qc_op_name = "matmul_compute_qc_op{}_pid{}"
    matmul_compute_cb_op_name = "matmul_compute_cb_op{}_pid{}"
    matmul_compute_ata_op_name = "matmul_compute_ata_op{}_pid{}"

    search_page_sizes = (codebook_batch_size,)
    for page_size in search_page_sizes:
        op_name_ = matmul_compute_ata_op_name.format(page_size, process_id)
        file_path_ = os.path.join(config_path, '%s.json' % op_name_)
        generate_matmul_compute_ATA_json(core_num=core_num, code_num=page_size, dim=dim, file_path=file_path_)
        atc_model(op_name_, output_path=output_path, soc_version=soc_version)
        
    search_page_sizes = (codebook_batch_size,)
    for page_size in search_page_sizes:
        op_name_ = matmul_compute_qc_op_name.format(page_size, process_id)
        file_path_ = os.path.join(config_path, '%s.json' % op_name_)
        generate_matmul_compute_qc_json(core_num=core_num, query_num=page_size, dim=dim,
                                        low_dim=low_dim, zregion_height=zregion_height, file_path=file_path_)
        atc_model(op_name_, output_path=output_path, soc_version=soc_version)

    search_page_sizes = (codebook_batch_size,)
    for page_size in search_page_sizes:
        op_name_ = matmul_compute_cb_op_name.format(page_size, process_id)
        file_path_ = os.path.join(config_path, '%s.json' % op_name_)
        generate_matmul_compute_cb_json(core_num=core_num, code_num=page_size, dim=dim, low_dim=low_dim,
                                        zregion_height=zregion_height, file_path=file_path_)
        atc_model(op_name_, output_path=output_path, soc_version=soc_version)
        
    search_page_sizes = (batch_size,)
    for page_size in search_page_sizes:
        op_name_ = compute_qc_op_name.format(page_size, process_id)
        file_path_ = os.path.join(config_path, '%s.json' % op_name_)
        generate_distance_compute_qc_json(core_num, page_size, dim, low_dim, k, zregion_height, file_path_)
        atc_model(op_name_, output_path=output_path, soc_version=soc_version)


if __name__ == '__main__':
    generate_flat_offline_model()
