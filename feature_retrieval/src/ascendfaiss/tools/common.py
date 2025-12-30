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
import sys
import stat
import json
import subprocess
from multiprocessing import Pool
import common as utils


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
        os.environ.get('PATH', ''), os.pathsep, 'usr/local/python3.7.5/bin',
        os.pathsep, os.path.join(ascend_toolkit_path, 'bin')[1:])

    os.environ['LD_LIBRARY_PATH'] = os.path.join(
        os.environ.get('LD_LIBRARY_PATH', ''),
        os.pathsep, os.path.join(ascend_toolkit_path, 'lib64')[1:])
    os.environ['PYTHONPATH'] = os.path.join(
        os.environ.get('PYTHONPATH', ''),
        os.pathsep, os.path.join(ascend_toolkit_path, 'python/site-packages')[1:])
    
    os.environ['ASCEND_OPP_PATH'] = os.path.join(ascend_toolkit_path, 'opp')


def atc_model(json_file, soc_version="Ascend310"):
    # generate aicore operator model
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


def generate_dist_compute_json(queries_num, dim, code_num, file_path):
    dist_compute_obj = []
    generator = OpJsonGenerator("DistanceComputeFlat")
    generator.add_input("ND", [queries_num, dim], "float16")
    generator.add_input("ND", [code_num // 16, dim // 16, 16, 16], "float16")
    generator.add_input("ND", [code_num], "float16")
    generator.add_output("ND", [queries_num, code_num], "float16")
    generator.add_output("ND", [32], "uint16")
    dist_compute_obj.append(generator.generate_obj())
    generate_op_config(dist_compute_obj, file_path)


def generate_dist_flat_l2_mins_json(core_nums, queries_num, dim, code_num, file_path):
    dist_flat_obj = []
    generator = OpJsonGenerator("DistanceFlatL2Mins")
    generator.add_input("ND", [queries_num, dim], "float16")
    generator.add_input("ND", [code_num // 16, dim // 16, 16, 16], "float16")
    generator.add_input("ND", [code_num], "float16")
    generator.add_output("ND", [queries_num, code_num], "float16")
    generator.add_output("ND", [queries_num, max(code_num // 32 * 2, 256)], "float16")
    generator.add_output("ND", [core_nums, 16], "uint16")
    dist_flat_obj.append(generator.generate_obj())
    generate_op_config(dist_flat_obj, file_path)


def get_int8_l2_norm_json(vectors_num, dim, file_path):
    # write int8_l2_norm json
    l2_norm_obj = []
    generator = OpJsonGenerator("Int8L2Norm")
    generator.add_input("ND", [vectors_num, dim], "int8")
    generator.add_input("ND", [256, 16], "float16")
    generator.add_input("ND", [8], "uint32")
    generator.add_output("ND", [vectors_num], "float16")
    l2_norm_obj.append(generator.generate_obj())
    generate_op_config(l2_norm_obj, file_path)


class OpJsonGenerator:
    def __init__(self, op_name):
        self.op_name = op_name
        self.input_desc = []
        self.output_desc = []

    def add_input(self, data_format: str, data_shape: list, data_type: str):
        self.input_desc.append({"format": data_format, "shape": data_shape, "type": data_type})
    
    def add_dynamic_input(self, data_format: str, data_shape: list, shape_range: list, data_type: str):
        self.input_desc.append({"format": data_format, "shape": data_shape,
                               "shape_range": shape_range, "type": data_type})
    
    def add_output(self, data_format: str, data_shape: list, data_type: str):
        self.output_desc.append({"format": data_format, "shape": data_shape, "type": data_type})
    
    def add_dynamic_output(self, data_format: str, data_shape: list, shape_range: list, data_type: str):
        self.output_desc.append({"format": data_format, "shape": data_shape,
                                "shape_range": shape_range, "type": data_type})

    def generate_obj(self):
        obj = {"op": self.op_name}
        obj['input_desc'] = self.input_desc
        obj['output_desc'] = self.output_desc
        return obj


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
    npu_type_soc_version_dict = {
        "310P": "Ascend310P3",
        "310": "Ascend310",
        "910B1": "Ascend910B1",
        "910B2": "Ascend910B2",
        "910B3": "Ascend910B3",
        "910B4": "Ascend910B4",
        "910_9392": "Ascend910_9392",
        "910_9382": "Ascend910_9382",
        "910_9372": "Ascend910_9372",
        "910_9362": "Ascend910_9362",
    }
    str_keys = [str(key) for key in npu_type_soc_version_dict.keys()]
    if npu_type not in npu_type_soc_version_dict.keys():
        raise RuntimeError("input invalid npu_type, only support %s" % str_keys)

    return npu_type_soc_version_dict[npu_type]


def get_core_num_by_npu_type(core_num, npu_type):
    # 设置默认的core num
    npu_type_core_num_dict = {
        "310": 2,
        "310P": 8,
        "910B1": 48,
        "910B2": 48,
        "910B3": 40,
        "910B4": 40,
        "910_9392": 48,
        "910_9382": 48,
        "910_9372": 40,
        "910_9362": 40,
    }
    if "--cores" in sys.argv:
        if core_num not in npu_type_core_num_dict.values():
            raise RuntimeError("invalid core_num: %d" % core_num)
        return core_num

    str_keys = [str(key) for key in npu_type_core_num_dict.keys()]
    if npu_type not in npu_type_core_num_dict.keys():
        raise RuntimeError("input invalid npu_type, only support %s" % str_keys)

    return npu_type_core_num_dict[npu_type]


def run_generate_model_task(args, map_args):
    if args.pool_size:
        check_pool_size(args.pool_size)
        pool = Pool(args.pool_size)
        pool.starmap(utils.atc_model, map_args)
    else:
        for arg in map_args:
            utils.atc_model(arg[0], arg[1])


def check_param_range(param, valid_params, param_name):
    if param not in valid_params:
        raise ValueError("not support {0}: {1}, should be in {2}".format(param_name, param, valid_params))


def check_pool_size(pool_size):
    if pool_size > 32 or pool_size < 1:
        raise ValueError("input variable pool_size {0}, should be in range[1, 32]".format(pool_size))


def op_common_parse(parser, key_str: str, dest: str, default_value, data_type, help_str: str):
    parser.add_argument(key_str, dest=dest, default=default_value, type=data_type, help=help_str)
