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
import sys
import stat
import json
import subprocess
from multiprocessing import Pool

########################################### Utility Functions #####################################################


def set_env():
    """
    Set environment variable
    """

    if os.environ.get('MAX_COMPILE_CORE_NUMBER') != '1':
        os.environ['MAX_COMPILE_CORE_NUMBER'] = '1'
        
    if os.environ.get('ASCEND_HOME') is None:
        os.environ['ASCEND_HOME'] = '/usr/local/Ascend'

    if os.environ.get('ASCEND_VERSION') is None:
        os.environ['ASCEND_VERSION'] = 'ascend-toolkit/latest'

    ascend_toolkit_path = os.path.join(os.environ['ASCEND_HOME'], os.environ['ASCEND_VERSION'])
    if not os.path.exists(os.path.join(ascend_toolkit_path, "bin/atc")):
        err_msg = "Please set right ASCEND_HOME, now ASCEND_HOME={os.environ['ASCEND_VERSION']}.\n" \
                  "Usage: export ASCEND_HOME=${driver/ascend-toolkit_install_path}\n" \
                  "export ASCEND_VERSION=ascend-toolkit/latest"
        raise Exception(err_msg)

    os.environ['PATH'] = os.path.join(
        os.environ.get('PATH', ''),
        os.pathsep, os.path.join(ascend_toolkit_path, 'bin')[1:])

    os.environ['LD_LIBRARY_PATH'] = os.path.join(
        os.environ.get('LD_LIBRARY_PATH', ''),
        os.pathsep, os.path.join(ascend_toolkit_path, 'lib64')[1:])
    os.environ['PYTHONPATH'] = os.path.join(
        os.environ.get('PYTHONPATH', ''),
        os.pathsep, os.path.join(ascend_toolkit_path, 'python/site-packages')[1:])
    
    os.environ['ASCEND_OPP_PATH'] = os.path.join(ascend_toolkit_path, 'opp')


def atc_model(json_file, soc_version="Ascend310"):
    """
    Generate Offline Operators
    """
    ascend_toolkit_path = os.path.join(os.environ['ASCEND_HOME'], os.environ['ASCEND_VERSION'])
    atc_path = os.path.join(ascend_toolkit_path, 'bin/atc')
    
    return_code = subprocess.call([
        atc_path, '--singleop=./config/%s.json' % json_file,
        '--soc_version=%s' % soc_version, '--output=op_models', '--log=error'
    ], shell=False)

    if return_code:
        raise Exception("Failed to generate op models.")


def generate_op_config(dist_compute_obj, file_path):
    if os.path.islink(file_path):
        raise RuntimeError(file_path + " can not be soft link!!!")
    obj_str = json.dumps(dist_compute_obj)
    flags = os.O_WRONLY | os.O_CREAT
    modes = stat.S_IWUSR | stat.S_IRUSR | stat.S_IRGRP
    with os.fdopen(os.open(file_path, flags, modes), 'w') as fout:
        fout.write(obj_str)


def generate_work_dir(work_dir='.'):
    # generate directory of config
    config_path = os.path.join(work_dir, 'config')
    if not os.path.exists(config_path):
        os.makedirs(config_path, mode=0o750)

    # generate directory of model
    op_model_path = os.path.join(work_dir, 'op_models')
    if not os.path.exists(op_model_path):
        os.makedirs(op_model_path, mode=0o755)


def get_config_path(work_dir=''):
    generate_work_dir(work_dir)
    return os.path.join(work_dir, 'config')


def get_soc_version_from_npu_type(npu_type):
    if npu_type not in ["310", "310P"]:
        raise RuntimeError("NPU type only support 310 or 310P now!")

    if npu_type == "310P":
        return "Ascend310P3"

    return "Ascend310"


def get_core_num_by_npu_type(core_num, npu_type):
    # 如果用户指定了cores就以用户指定的为准
    if "--cores" in str(sys.argv):
        if core_num not in [2, 8]:
            raise RuntimeError("core_num only support 2 or 8 now!")
        return core_num

    # 310默认2核，310P默认8核
    return 2 if npu_type == "310" else 8


def run_generate_model_task(args, map_args):
    if args.pool_size:
        check_pool_size(args.pool_size)
        pool = Pool(args.pool_size)
        pool.starmap(atc_model, map_args)
    else:
        for arg in map_args:
            atc_model(arg[0], arg[1])


def check_param_range(param, valid_params, param_name):
    if param not in valid_params:
        raise ValueError("not support {0}: {1}, should be in {2}".format(param_name, param, valid_params))


def check_pool_size(pool_size):
    if pool_size > 32 or pool_size < 1:
        raise ValueError("input variable pool_size {0}, should be in range[1, 32]".format(pool_size))
    

def check_search_params(search_params, params_type, nlist_l1_tmp=0, nlist_l2_tmp=0):
    if params_type == "nprobe_l1":
        correct_condition = (search_params > 16) and (search_params % 8 == 0) and (search_params <= nlist_l1_tmp)
        if not correct_condition:
            raise ValueError("input variable nprobe_l1 {0}, should be in range (16, {1}] and divisible by 8.".format(search_params, nlist_l1_tmp)) 
    elif params_type == "nprobe_l2":
        correct_condition = (search_params > 16) and (search_params % 8 == 0) and (search_params <= nlist_l1_tmp * nlist_l2_tmp)
        if not correct_condition:
            raise ValueError("input variable nprobe_l2 {0}, should be in range (16, {1}] and divisible by 8.".format(search_params, nlist_l1_tmp * nlist_l2_tmp))        
    elif params_type == "segment_num_l3":
        correct_condition = (search_params > 100) and (search_params <= 5000) and (search_params % 8 == 0)
        if not correct_condition:
            raise ValueError("input variable segment_num_l3 {0}, should be in range (100, 5000] and divisible by 8.".format(search_params))         
    else:
        raise RuntimeError("Invalid search params input to check_search_params.")
        

######################################### Operator Specific functions #################################################################

def arg_parse():
    """
    Parse arguements to the operator model
    """

    parser = argparse.ArgumentParser(
        description='generate Vstar offline operators')

    parser.add_argument("--dim",
                        dest='dim',
                        default=256,
                        type=int,
                        help="Data Dimension")
    
    parser.add_argument("--nlistL1",
                        dest='nlist_l1',
                        default=1024,
                        type=int,
                        help="L1 Bucket Number")

    parser.add_argument("--subDimL1",
                        dest='sub_dim_l1',
                        default=32,
                        type=int,
                        help="Subspace Dimension L1")

    parser.add_argument("--nProbeL1",
                        dest='nprobe_l1',
                        default=[72],
                        type=int,
                        nargs='+',
                        help="Number of L1 buckets to search from after L1 bucketing")

    parser.add_argument("--nProbeL2",
                        dest='nprobe_l2',
                        default=[64, 296],
                        type=int,
                        nargs='+',
                        help="Number of L2 buckets to search from after L2 bucketing")

    parser.add_argument("--segmentNumL3",
                        dest='segment_num_l3',
                        default=[512, 1000, 1504],
                        type=int,
                        nargs='+',
                        help="Number of L3 Segments (each segment size = 64) to search from using topK")

    parser.add_argument("--pool",
                        dest="pool_size",
                        default=16,
                        type=int,
                        help="Number of pool_size")
    
    return parser.parse_args()


def generate_vstar_compute_l1_json(core_nums: int, batch_size: int, dim: int, nlist_l1: int, sub_dim_l1: int, file_path: str):
    """
    Generate full json for vstar_compute_l1 operator
    """

    vstar_compute_l1_obj = [{
        "op": "VstarComputeL1",
        "input_desc": [
            {
                "name": "x0",
                "param_type": "required",
                "format": "ND",
                "shape": [batch_size, dim],
                "type": "float"
            },
            {
                "name": "x1",
                "param_type": "required",
                "format": "ND",
                "shape": [nlist_l1 * sub_dim_l1 // 16, dim // 16, 16, 16],
                "type": "float16"
            }
        ],
        "output_desc": [
            {
                "name": "y0",
                "param_type": "required",
                "format": "ND",
                "shape": [batch_size, nlist_l1 * sub_dim_l1],
                "type": "float16"
            },
            {
                "name": "y1",
                "param_type": "required",
                "format": "ND",
                "shape": [batch_size, nlist_l1],
                "type": "float16"
            },
            {
                "name": "y2",
                "param_type": "required",
                "format": "ND",
                "shape": [core_nums, 16],
                "type": "uint16"
            }
        ],
        "attr": [
            {
                "name": "subSpaceDim",
                "param_type": "required",
                "type": "int",
                "default_value": 1,
                "value": sub_dim_l1
            }
        ]
    }]

    generate_op_config(vstar_compute_l1_obj, file_path)


def generate_vstar_compute_l2_json(core_nums: int, batch_size: int, nlist_l1: int, nlist_l2: int, sub_dim_l1: int, sub_dim_l2: int, nprobe_l1: int, file_path: str):
    """
    Generate full json for vstar_compute_l2 operator
    """

    vstar_compute_l2_obj = [{
        "op": "VstarComputeL2",
        "input_desc": [
            {
                "name": "queryCode",
                "param_type": "required",
                "format": "ND",
                "shape": [batch_size, nlist_l1 * sub_dim_l1],
                "type": "float16"
            },
            {
                "name": "codebookL2",
                "param_type": "required",
                "format": "ND",
                "shape": [nlist_l1 * nlist_l2 * sub_dim_l2 // 16, sub_dim_l1 // 16, 16, 16],
                "type": "float16"
            },
            {
                "name": "l1KIndices",
                "param_type": "required",
                "format": "ND",
                "shape": [batch_size, nprobe_l1],
                "type": "uint16"
            }
        ],
        "output_desc": [
            {
                "name": "outDists",
                "param_type": "required",
                "format": "ND",
                "shape": [batch_size, nprobe_l1 * nlist_l2],
                "type": "float16"
            },
            {
                "name": "opFlag",
                "param_type": "required",
                "format": "ND",
                "shape": [core_nums, 16],
                "type": "uint16"
            }
        ],
        "attr": [
            {
                "name": "subSpaceDim1",
                "param_type": "required",
                "type": "int",
                "default_value": 1,
                "value": sub_dim_l1
            },
            {
                "name": "subSpaceDim2",
                "param_type": "required",
                "type": "int",
                "default_value": 1,
                "value": sub_dim_l2
            }
        ]
    }]

    generate_op_config(vstar_compute_l2_obj, file_path)


def generate_vstar_compute_l3_json(core_nums: int, batch_size: int, nlist_l1: int, nlist_l2: int, sub_dim_l1: int, nprobe_l2: int, segment_num_l3: int, file_path: str):
    """
    Generate full json for vstar_compute_l3 operator
    """

    seg_size_x = 64
    vc_min_seg_size_x = 64

    vstar_compute_l3_obj = [{
        "op": "VSC3",
        "input_desc": [
            {
                "name": "queryCode",
                "param_type": "required",
                "format": "ND",
                "shape": [batch_size, nlist_l1 * sub_dim_l1],
                "type": "float16"
            },
            {
                "name": "codeWord",
                "param_type": "required",
                "format": "ND",
                "shape": [seg_size_x // 16, sub_dim_l1 // 16, 16, 16],
                "type": "uint8"
            },
            {
                "name": "l2Indices",
                "param_type": "required",
                "format": "ND",
                "shape": [batch_size, nprobe_l2 * 6],
                "type": "uint64"
            },
            {
                "name": "diff1",
                "param_type": "required",
                "format": "ND",
                "shape": [1, sub_dim_l1],
                "type": "float16"
            },
            {
                "name": "diff2",
                "param_type": "required",
                "format": "ND",
                "shape": [1, sub_dim_l1],
                "type": "float16"
            },
            {
                "name": "precompute",
                "param_type": "required",
                "format": "ND",
                "shape": [1, seg_size_x],
                "type": "float16"
            },
            {
                "name": "attr_nlistl1",
                "param_type": "required",
                "format": "ND",
                "shape": [1, nlist_l1],
                "type": "int32"
            },
            {
                "name": "attr_nlistl2",
                "param_type": "required",
                "format": "ND",
                "shape": [1, nlist_l2],
                "type": "int32"
            },
            {
                "name": "attr_segmentl3",
                "param_type": "required",
                "format": "ND",
                "shape": [1, segment_num_l3],
                "type": "int32"
            },          
        ],
        "output_desc": [
            {
                "name": "outDists",
                "param_type": "required",
                "format": "ND",
                "shape": [batch_size, segment_num_l3 * seg_size_x],
                "type": "float16"
            },
            {
                "name": "opFlag",
                "param_type": "required",
                "format": "ND",
                "shape": [core_nums, 16],
                "type": "uint16"
            },
            {
                "name": "vcMin",
                "param_type": "required",
                "format": "ND",
                "shape": [batch_size, 2 * segment_num_l3 * seg_size_x // vc_min_seg_size_x],
                "type": "float16"
            }
        ],
        "attr": [
            {
                "name": "nlist1",
                "param_type": "required",
                "type": "int",
                "default_value": 1,
                "value": nlist_l1
            },
            {
                "name": "nlist2",
                "param_type": "required",
                "type": "int",
                "default_value": 1,
                "value": nlist_l2
            },
            {
                "name": "segmentNum",
                "param_type": "required",
                "type": "int",
                "default_value": 1,
                "value": segment_num_l3
            }
        ]
    }]

    generate_op_config(vstar_compute_l3_obj, file_path)


def generate_vstar_compute_l3_mask_json(core_nums: int, batch_size: int, nlist_l1: int, nlist_l2: int, sub_dim_l1: int, nprobe_l2: int, segment_num_l3: int, file_path: str):
    """
    Generate full json for vstar_compute_l3_mask operator
    """

    seg_size_x = 64
    vc_min_seg_size_x = 64

    vstar_compute_l3_mask_obj = [{
        "op": "VSM3",
        "input_desc": [
            {
                "name": "queryCode",
                "param_type": "required",
                "format": "ND",
                "shape": [batch_size, nlist_l1 * sub_dim_l1],
                "type": "float16"
            },
            {
                "name": "codeWord",
                "param_type": "required",
                "format": "ND",
                "shape": [seg_size_x // 16, sub_dim_l1 // 16, 16, 16],
                "type": "uint8"
            },
            {
                "name": "l2Indices",
                "param_type": "required",
                "format": "ND",
                "shape": [batch_size, nprobe_l2 * 6],
                "type": "uint64"
            },
            {
                "name": "diff1",
                "param_type": "required",
                "format": "ND",
                "shape": [1, sub_dim_l1],
                "type": "float16"
            },
            {
                "name": "diff2",
                "param_type": "required",
                "format": "ND",
                "shape": [1, sub_dim_l1],
                "type": "float16"
            },
            {
                "name": "precompute",
                "param_type": "required",
                "format": "ND",
                "shape": [1, seg_size_x],
                "type": "float16"
            },
            {
                "name": "mask",
                "param_type": "required",
                "format": "ND",
                "shape": [1, seg_size_x],
                "type": "uint8"
            },
            {
                "name": "attr_nlistl1",
                "param_type": "required",
                "format": "ND",
                "shape": [1, nlist_l1],
                "type": "int32"
            },
            {
                "name": "attr_nlistl2",
                "param_type": "required",
                "format": "ND",
                "shape": [1, nlist_l2],
                "type": "int32"
            },
            {
                "name": "attr_segmentl3",
                "param_type": "required",
                "format": "ND",
                "shape": [1, segment_num_l3],
                "type": "int32"
            },
        ],
        "output_desc": [
            {
                "name": "outDists",
                "param_type": "required",
                "format": "ND",
                "shape": [batch_size, segment_num_l3 * seg_size_x],
                "type": "float16"
            },
            {
                "name": "opFlag",
                "param_type": "required",
                "format": "ND",
                "shape": [core_nums, 16],
                "type": "uint16"
            },
            {
                "name": "vcMin",
                "param_type": "required",
                "format": "ND",
                "shape": [batch_size, 2 * segment_num_l3 * seg_size_x // vc_min_seg_size_x],
                "type": "float16"
            }
        ],
        "attr": [
            {
                "name": "nlist1",
                "param_type": "required",
                "type": "int",
                "default_value": 1,
                "value": nlist_l1
            },
            {
                "name": "nlist2",
                "param_type": "required",
                "type": "int",
                "default_value": 1,
                "value": nlist_l2
            },
            {
                "name": "segmentNum",
                "param_type": "required",
                "type": "int",
                "default_value": 1,
                "value": segment_num_l3
            }
        ]
    }]

    generate_op_config(vstar_compute_l3_mask_obj, file_path)
    

def generate_base_add_matmul_json(core_nums: int, dim: int, nlist_l1: int, sub_dim_l1: int, file_path: str):
    """
    Generate full json for base_add_matmul operator
    """
    
    nb = 64
    
    base_add_matmul_obj = [{
        "op": "VstarBaseAddMatMul",
        "input_desc": [
            {
                "name": "a",
                "param_type": "required",
                "format": "ND",
                "shape": [nb, dim],
                "type": "float"
            },
            {
                "name": "b",
                "param_type": "required",
                "format": "ND",
                "shape": [nlist_l1, sub_dim_l1, dim],
                "type": "float"
            }
        ],
        "output_desc": [
            {
                "name": "c",
                "param_type": "required",
                "format": "ND",
                "shape": [nb, nlist_l1, sub_dim_l1],
                "type": "float"
            }
        ]
    }]

    generate_op_config(base_add_matmul_obj, file_path)
    

def generate_ivfsp_l1_ivf_int64_obj():
    topk_flat_obj = {
        "op": "IvfSpTopkL1",
        "input_desc": [{
            "name": "x0",
            "format": "ND",
            "shape": [-1, -1],
            "shape_range": [[1, -1], [1, -1]],
            "type": "float16"
        }, {
            "name": "x1",
            "format": "ND",
            "shape": [-1, -1],
            "shape_range": [[1, -1], [1, -1]],
            "type": "uint32"
        }, {
            "name": "x2",
            "format": "ND",
            "shape": [-1, -1],
            "shape_range": [[1, -1], [1, -1]],
            "type": "uint16"
        }, {
            "name": "x3",
            "format": "ND",
            "shape": [-1],
            "shape_range": [[1, -1]],
            "type": "int64"
        }],
        "output_desc": [{
            "name": "y0",
            "format": "ND",
            "shape": [-1, -1],
            "shape_range": [[1, -1], [1, -1]],
            "type": "float16"
        }, {
            "name": "y1",
            "format": "ND",
            "shape": [-1, -1],
            "shape_range": [[1, -1], [1, -1]],
            "type": "int64"
        }]
    }

    return topk_flat_obj


def generate_ivfsp_l1_ivf_uint16_obj():
    topk_flat_obj = {
        "op": "IvfSpTopkL1",
        "input_desc": [{
            "name": "x0",
            "format": "ND",
            "shape": [-1, -1],
            "shape_range": [[1, -1], [1, -1]],
            "type": "float16"
        }, {
            "name": "x1",
            "format": "ND",
            "shape": [-1, -1],
            "shape_range": [[1, -1], [1, -1]],
            "type": "uint32"
        }, {
            "name": "x2",
            "format": "ND",
            "shape": [-1, -1],
            "shape_range": [[1, -1], [1, -1]],
            "type": "uint16"
        }, {
            "name": "x3",
            "format": "ND",
            "shape": [-1],
            "shape_range": [[1, -1]],
            "type": "int64"
        }],
        "output_desc": [{
            "name": "y0",
            "format": "ND",
            "shape": [-1, -1],
            "shape_range": [[1, -1], [1, -1]],
            "type": "float16"
        }, {
            "name": "y1",
            "format": "ND",
            "shape": [-1, -1],
            "shape_range": [[1, -1], [1, -1]],
            "type": "uint16"
        }]
    }

    return topk_flat_obj


def generate_ivfsp_l2_topk_uint64_obj():
    topk_flat_obj = {
        "op": "IvfSpTopkL2",
        "input_desc": [
            {
                "name": "dists",
                "format": "ND",
                "shape": [-1, -1],
                "shape_range": [[1, -1], [1, -1]],
                "type": "float16"
            },
            {
                "name": "L1Indices",
                "format": "ND",
                "shape": [-1, -1],
                "shape_range": [[1, -1], [1, -1]],
                "type": "uint16"
            },
            {
                "name": "opFlag",
                "format": "ND",
                "shape": [-1, -1],
                "shape_range": [[1, -1], [1, -1]],
                "type": "uint16"
            },
            {
                "name": "addressOffsetOfBucket",
                "format": "ND",
                "shape": [-1],
                "shape_range": [[1, -1]],
                "type": "uint64"
            },
            {
                "name": "attr",
                "format": "ND",
                "shape": [-1],
                "shape_range": [[1, -1]],
                "type": "int64"
            },

        ],
        "output_desc": [
            {
                "name": "distsRes",
                "format": "ND",
                "shape": [-1, -1],
                "shape_range": [[1, -1], [1, -1]],
                "type": "float16"
            },
            {
                "name": "addressOffsetL3",
                "format": "ND",
                "shape": [-1, -1],
                "shape_range": [[1, -1], [1, -1]],
                "type": "uint64"
            },
            {
                "name": "idAddressL3",
                "format": "ND",
                "shape": [-1, -1],
                "shape_range": [[1, -1], [1, -1]],
                "type": "uint64"
            }]
    }

    return topk_flat_obj


def generate_ivfmultisp_l2_topk_uint64_obj():
    topk_flat_obj = {
        "op": "IvfMultiSpTopkL2",
        "input_desc": [
            {
                "name": "dists",
                "format": "ND",
                "shape": [-1, -1],
                "shape_range": [[1, -1], [1, -1]],
                "type": "float16"
            },
            {
                "name": "L1Indices",
                "format": "ND",
                "shape": [-1, -1],
                "shape_range": [[1, -1], [1, -1]],
                "type": "uint16"
            },
            {
                "name": "opFlag",
                "format": "ND",
                "shape": [-1, -1],
                "shape_range": [[1, -1], [1, -1]],
                "type": "uint16"
            },
            {
                "name": "attr",
                "format": "ND",
                "shape": [-1],
                "shape_range": [[1, -1]],
                "type": "int64"
            },

        ],
        "output_desc": [
            {
                "name": "distsRes",
                "format": "ND",
                "shape": [-1, -1],
                "shape_range": [[1, -1], [1, -1]],
                "type": "float16"
            },
            {
                "name": "indicesL2",
                "format": "ND",
                "shape": [-1, -1],
                "shape_range": [[1, -1], [1, -1]],
                "type": "uint64"
            }
        ]
    }

    return topk_flat_obj


def generate_ivfsp_l2_topk_with_mask_uint64_obj():
    topk_flat_obj = {
        "op": "IvfSpTopkL2WithMask",
        "input_desc": [
            {
                "name": "maskBitNpu",
                "format": "ND",
                "shape": [-1],
                "shape_range": [[1, -1]],
                "type": "uint8"
            },
            {
                "name": "dists",
                "format": "ND",
                "shape": [-1, -1],
                "shape_range": [[1, -1], [1, -1]],
                "type": "float16"
            },
            {
                "name": "L1Indices",
                "format": "ND",
                "shape": [-1, -1],
                "shape_range": [[1, -1], [1, -1]],
                "type": "uint16"
            },
            {
                "name": "opFlag",
                "format": "ND",
                "shape": [-1, -1],
                "shape_range": [[1, -1], [1, -1]],
                "type": "uint16"
            },
            {
                "name": "addressOffsetOfBucket",
                "format": "ND",
                "shape": [-1],
                "shape_range": [[1, -1]],
                "type": "uint64"
            },
            {
                "name": "attr",
                "format": "ND",
                "shape": [-1],
                "shape_range": [[1, -1]],
                "type": "int64"
            },

        ],
        "output_desc": [
            {
                "name": "distsRes",
                "format": "ND",
                "shape": [-1, -1],
                "shape_range": [[1, -1], [1, -1]],
                "type": "float16"
            },
            {
                "name": "addressOffsetL3",
                "format": "ND",
                "shape": [-1, -1],
                "shape_range": [[1, -1], [1, -1]],
                "type": "uint64"
            },
            {
                "name": "idAddressL3",
                "format": "ND",
                "shape": [-1, -1],
                "shape_range": [[1, -1], [1, -1]],
                "type": "uint64"
            },
            {
                "name": "maskByteNpu",
                "format": "ND",
                "shape": [-1, -1],
                "shape_range": [[1, -1], [1, -1]],
                "type": "uint8"
            },
            {
                "name": "isMaskOffset",
                "format": "ND",
                "shape": [-1, -1],
                "shape_range": [[1, -1], [1, -1]],
                "type": "uint64"
            }]
    }

    return topk_flat_obj


def generate_ivfsp_l3_topk_int64_obj():
    topk_flat_obj = {
        "op": "IvfSpTopkL3",
        "input_desc": [{
            "name": "x0",
            "format": "ND",
            "shape": [-1, -1],
            "shape_range": [[1, -1], [1, -1]],
            "type": "float16"
        }, {
            "name": "x1",
            "format": "ND",
            "shape": [-1, -1],
            "shape_range": [[1, -1], [1, -1]],
            "type": "float16"
        }, {
            "name": "x2",
            "format": "ND",
            "shape": [-1, -1],
            "shape_range": [[1, -1], [1, -1]],
            "type": "uint64"
        }, {
            "name": "x3",
            "format": "ND",
            "shape": [-1, -1],
            "shape_range": [[1, -1], [1, -1]],
            "type": "uint16"
        }, {
            "name": "x4",
            "format": "ND",
            "shape": [-1],
            "shape_range": [[1, -1]],
            "type": "int64"
        }],
        "output_desc": [{
            "name": "y0",
            "format": "ND",
            "shape": [-1, -1],
            "shape_range": [[1, -1], [1, -1]],
            "type": "float16"
        }, {
            "name": "y1",
            "format": "ND",
            "shape": [-1, -1],
            "shape_range": [[1, -1], [1, -1]],
            "type": "int64"
        }]
    }

    return topk_flat_obj


def generate_ivfmultisp_l3_topk_int64_obj():
    topk_flat_obj = {
        "op": "IvfMultiSpTopkL3",
        "input_desc": [{
            "name": "x0",
            "format": "ND",
            "shape": [-1, -1, -1],
            "shape_range": [[1, -1], [1, -1], [1, -1]],
            "type": "float16"
        }, {
            "name": "x1",
            "format": "ND",
            "shape": [-1, -1, -1],
            "shape_range": [[1, -1], [1, -1], [1, -1]],
            "type": "float16"
        }, {
            "name": "x2",
            "format": "ND",
            "shape": [-1, -1, -1],
            "shape_range": [[1, -1], [1, -1], [1, -1]],
            "type": "uint64"
        }, {
            "name": "x3",
            "format": "ND",
            "shape": [-1, -1, -1],
            "shape_range": [[1, -1], [1, -1], [1, -1]],
            "type": "uint16"
        }, {
            "name": "x4",
            "format": "ND",
            "shape": [-1],
            "shape_range": [[1, -1]],
            "type": "int64"
        }, {
            "name": "x5",
            "format": "ND",
            "shape": [-1, -1, -1],
            "shape_range": [[1, -1], [1, -1], [1, -1]],
            "type": "uint64"
        }],
        "output_desc": [{
            "name": "y0",
            "format": "ND",
            "shape": [-1, -1, -1],
            "shape_range": [[1, -1], [1, -1], [1, -1]],
            "type": "float16"
        }, {
            "name": "y1",
            "format": "ND",
            "shape": [-1, -1, -1],
            "shape_range": [[1, -1], [1, -1], [1, -1]],
            "type": "int64"
        }]
    }

    return topk_flat_obj


def generate_vstar_offline_model():
    """
    Generate json files for all operators and generate offline operators
    """
    
    set_env()
    args = arg_parse()
    
    output_dir = '.'
    batches = [1, 2, 4, 8, 16]
        
    dim = args.dim
    check_param_range(dim, [128, 256, 512, 1024], "dim")
    
    nlist_l1 = args.nlist_l1
    check_param_range(nlist_l1, [256, 512, 1024], "nlist_l1")
    
    if dim in [128, 256, 512]:
        nlist_l2 = 32
    else:
        nlist_l2 = 16
    
    sub_dim_l1 = args.sub_dim_l1
    if sub_dim_l1 >= dim:
        raise ValueError("sub_dim_l1 should be less than dim. Currently sub_dim_l1 = {0}, dim = {1}".format(sub_dim_l1, dim))
    check_param_range(sub_dim_l1, [32, 64, 128], "sub_dim_l1")
    
    sub_dim_l2 = int(sub_dim_l1/2)
    
    nprobe_l1 = args.nprobe_l1
    if 72 not in nprobe_l1:
        nprobe_l1.append(72)
        
    nprobe_l2 = args.nprobe_l2
    if 64 not in nprobe_l2:
        nprobe_l2.append(64)
        
    segment_num_l3 = args.segment_num_l3
    if 512 not in segment_num_l3:
        segment_num_l3.append(512)
    
    for nprobe_l1x in nprobe_l1:
        check_search_params(nprobe_l1x, "nprobe_l1", nlist_l1)
        
    # 获得最小的nprobe_l1，以此为基准对nprobe_l2进行校验
    min_nprobe_l1 = min(nprobe_l1)
    for nprobe_l2x in nprobe_l2:
        check_search_params(nprobe_l2x, "nprobe_l2", min_nprobe_l1, nlist_l2)
            
    for segment_num_l3_x in segment_num_l3:
        check_search_params(segment_num_l3_x, "segment_num_l3")
    

    soc_version = "Ascend310P3"
    core_nums = 8

    work_dir = output_dir
    config_path = get_config_path(work_dir)
    
    # Generate AICPU operators (not multiprocessing for now because this is fast)
    print("Generating AICPU Operators...")
    json_list = []
    json_list.append(generate_ivfsp_l1_ivf_int64_obj())
    json_list.append(generate_ivfsp_l1_ivf_uint16_obj())
    json_list.append(generate_ivfsp_l2_topk_uint64_obj())
    json_list.append(generate_ivfmultisp_l2_topk_uint64_obj())
    json_list.append(generate_ivfsp_l2_topk_with_mask_uint64_obj())
    json_list.append(generate_ivfsp_l3_topk_int64_obj())
    json_list.append(generate_ivfmultisp_l3_topk_int64_obj())
    generate_op_config(json_list, "./config/aicpu_kernel.json")
    atc_model("aicpu_kernel", soc_version)
    print("Generating AICPU Operators completed!")    
    
    # Generate AICore operators
    vstar_compute_l1_op_name = "VstarComputeL1_op{}_{}_{}_{}"
    vstar_compute_l2_op_name = "VstarComputeL2_op{}_{}_{}_{}_{}_{}"
    vstar_compute_l3_op_name = "VSC3_op{}_{}_{}_{}_{}_{}"
    vstar_compute_l3_mask_op_name = "VSM3_mask_op{}_{}_{}_{}_{}_{}"
    base_add_matmul_op_name = "VstarBaseAddMatMul_op{}_{}_{}"

    map_args = []
    check_pool_size(args.pool_size)

    p = Pool(args.pool_size)
    
    for nq_x in batches:
        vstar_compute_l1_op_name_ = vstar_compute_l1_op_name.format(nq_x, dim, nlist_l1, sub_dim_l1)
        file_vstar_compute_l1_path_ = os.path.join(config_path, '%s.json' % vstar_compute_l1_op_name_)
        generate_vstar_compute_l1_json(core_nums, nq_x, dim, nlist_l1, sub_dim_l1, file_vstar_compute_l1_path_)
        map_args.append((vstar_compute_l1_op_name_, soc_version))
                    

    for nq_x in (batches + [64]):
        for nprobe1_x in (nprobe_l1 + [1]):
            vstar_compute_l2_op_name_ = vstar_compute_l2_op_name.format(nq_x, nlist_l1, nlist_l2, sub_dim_l1, sub_dim_l2, nprobe1_x)
            file_vstar_compute_l2_path_ = os.path.join(config_path, '%s.json' % vstar_compute_l2_op_name_)
            generate_vstar_compute_l2_json(core_nums, nq_x, nlist_l1, nlist_l2, sub_dim_l1, sub_dim_l2, nprobe1_x, file_vstar_compute_l2_path_)
            map_args.append((vstar_compute_l2_op_name_, soc_version))
                            

    for nq_x in batches:
        for nprobe2_x in nprobe_l2:
            for segment_num_x in segment_num_l3:
                vstar_compute_l3_op_name_ = vstar_compute_l3_op_name.format(nq_x, nlist_l1,
                                            nlist_l2, sub_dim_l1, nprobe2_x, segment_num_x)
                file_vstar_compute_l3_path_ = os.path.join(config_path, '%s.json' % vstar_compute_l3_op_name_)
                generate_vstar_compute_l3_json(core_nums, nq_x, nlist_l1, nlist_l2,
                    sub_dim_l1, nprobe2_x, segment_num_x, file_vstar_compute_l3_path_)
                map_args.append((vstar_compute_l3_op_name_, soc_version))
                            
    for nq_x in batches:
        for nprobe2_x in nprobe_l2:
            for segment_num_x in segment_num_l3:
                vstar_compute_l3_mask_op_name_ = vstar_compute_l3_mask_op_name.format(nq_x, nlist_l1, nlist_l2,
                                                sub_dim_l1, nprobe2_x, segment_num_x)
                file_vstar_compute_l3_mask_path_ = os.path.join(config_path, '%s.json' % vstar_compute_l3_mask_op_name_)
                generate_vstar_compute_l3_mask_json(core_nums, nq_x, nlist_l1, nlist_l2,
                    sub_dim_l1, nprobe2_x, segment_num_x, file_vstar_compute_l3_mask_path_)
                map_args.append((vstar_compute_l3_mask_op_name_, soc_version))
                            

    base_add_matmul_op_name_ = base_add_matmul_op_name.format(dim, nlist_l1, sub_dim_l1)
    file_base_add_matmul_path_ = os.path.join(config_path, '%s.json' % base_add_matmul_op_name_)
    generate_base_add_matmul_json(core_nums, dim, nlist_l1, sub_dim_l1, file_base_add_matmul_path_)
    map_args.append((base_add_matmul_op_name_, soc_version))
                
    print("Generating AICore Operators...")
    p.starmap(atc_model, map_args)
    print("Generating AICore Operators Completed!")
    

if __name__ == '__main__':
    generate_vstar_offline_model()