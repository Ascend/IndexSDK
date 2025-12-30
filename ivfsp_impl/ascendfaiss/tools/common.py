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
import stat
import json
import subprocess


def set_env():
    """
    Set environment variable
    """

    if os.environ.get('ASCEND_HOME') is None:
        os.environ['ASCEND_HOME'] = '/usr/local/Ascend'

    if os.environ.get('ASCEND_VERSION') is None:
        os.environ['ASCEND_VERSION'] = 'ascend-toolkit/latest'

    ascend_toolkit_path = os.path.join(os.environ['ASCEND_HOME'], os.environ['ASCEND_VERSION'])
    if not os.path.exists(os.path.join(ascend_toolkit_path, "compiler/bin/atc")):
        err_msg = "Please set right ASCEND_HOME, now ASCEND_HOME={os.environ['ASCEND_VERSION']}.\n" \
                  "Usage: export ASCEND_HOME=${driver/ascend-toolkit_install_path}\n" \
                  "export ASCEND_VERSION=ascend-toolkit/latest"
        raise Exception(err_msg)

    os.environ['PATH'] = os.path.join(
        os.environ.get('PATH', ''), os.pathsep, 'usr/local/python3.7.5/bin',
        os.pathsep, os.path.join(ascend_toolkit_path, 'compiler/ccec_compiler/bin')[1:],
        os.pathsep, os.path.join(ascend_toolkit_path, 'compiler/bin')[1:])

    os.environ['LD_LIBRARY_PATH'] = os.path.join(
        os.environ.get('LD_LIBRARY_PATH', ''),
        os.pathsep, os.path.join(ascend_toolkit_path, 'compiler/lib64')[1:])
    os.environ['PYTHONPATH'] = os.path.join(
        os.environ.get('PYTHONPATH', ''),
        os.pathsep, os.path.join(ascend_toolkit_path, 'compiler/python/site-packages')[1:])
    
    os.environ['ASCEND_OPP_PATH'] = os.path.join(ascend_toolkit_path, 'opp')


def atc_model(json_file, soc_version="Ascend310"):
    # generate aicore operator model
    ascend_toolkit_path = os.path.join(os.environ['ASCEND_HOME'], os.environ['ASCEND_VERSION'])
    atc_path = os.path.join(ascend_toolkit_path, 'compiler/bin/atc')
    return_code = subprocess.call([
        atc_path, '--singleop=./config/%s.json' % json_file,
        '--soc_version=%s' % soc_version, '--output=op_models', '--log=error'
    ], shell=False)

    if return_code:
        raise Exception("Failed to generate op models.")


def generate_op_config(dist_compute_obj, file_path):
    obj_str = json.dumps(dist_compute_obj)
    flags = os.O_WRONLY | os.O_CREAT
    modes = stat.S_IWUSR | stat.S_IRUSR | stat.S_IRGRP
    with os.fdopen(os.open(file_path, flags, modes), 'w') as fout:
        fout.write(obj_str)


def generate_dist_compute_json(queries_num, dim, code_num, file_path):
    dist_compute_obj = [
        {
            "op":
                "DistanceComputeFlat",
            "input_desc": [{
                "format": "ND",
                "shape": [queries_num, dim],
                "type": "float16"
            }, {
                "format": "ND",
                "shape": [code_num // 16, dim // 16, 16, 16],
                "type": "float16"
            }, {
                "format": "ND",
                "shape": [code_num],
                "type": "float16"
            }],
            "output_desc": [{
                "format": "ND",
                "shape": [queries_num, code_num],
                "type": "float16"
            }, {
                "format": "ND",
                "shape": [32],
                "type": "uint16"
            }]
        }
    ]

    generate_op_config(dist_compute_obj, file_path)


def generate_dist_flat_l2_mins_json(core_nums, queries_num, dim, code_num, file_path):
    dist_flat_obj = [
        {
            "op":
                "DistanceFlatL2Mins",
            "input_desc": [{
                "format": "ND",
                "shape": [queries_num, dim],
                "type": "float16"
            }, {
                "format": "ND",
                "shape": [code_num // 16, dim // 16, 16, 16],
                "type": "float16"
            }, {
                "format": "ND",
                "shape": [code_num],
                "type": "float16"
            }],
            "output_desc": [{
                "format": "ND",
                "shape": [queries_num, code_num],
                "type": "float16"
            }, {
                "format": "ND",
                "shape": [queries_num, max(code_num // 32 * 2, 256)],
                "type": "float16"
            }, {
                "format": "ND",
                "shape": [core_nums, 16],
                "type": "uint16"
            }]
        }
    ]

    generate_op_config(dist_flat_obj, file_path)


def get_int8_l2_norm_json(vectors_num, dim, file_path):
    # write int8_l2_norm json
    l2_norm_obj = [
        {
            "op":
                "Int8L2Norm",
            "input_desc": [{
                "format": "ND",
                "shape": [vectors_num, dim],
                "type": "int8"
            }, {
                "format": "ND",
                "shape": [256, 16],
                "type": "float16"
            }, {
                "format": "ND",
                "shape": [8],
                "type": "uint32"
            }],
            "output_desc": [{
                "format": "ND",
                "shape": [vectors_num],
                "type": "float16"
            }]
        }
    ]

    generate_op_config(l2_norm_obj, file_path)


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
