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
import json
import os
import stat
import sys
import logging

logging.basicConfig(level=logging.NOTSET, format="")


def parse_ini_files(ini_files):
    tbe_ops_info = {}
    for ini_file in ini_files:
        parse_ini_to_obj(ini_file, tbe_ops_info)
    return tbe_ops_info


def parse_ini_to_obj(ini_file, tbe_ops_info):
    with open(ini_file) as ini_file:
        lines = ini_file.readlines()
        op = {}
        for line in lines:
            line = line.rstrip()
            if line.startswith("["):
                op_name = line[1:-1]
                op = {}
                tbe_ops_info[op_name] = op
            else:
                key1 = line[:line.index("=")]
                key2 = line[line.index("=") + 1:]
                key1_0, key1_1 = key1.split(".")
                if key1_0 not in op:
                    op[key1_0] = {}
                op[key1_0][key1_1] = key2


def check_op_info(tbe_ops):
    logging.info("\n\n==============check valid for ops info start==============")
    required_op_input_info_keys = ["paramType", "name"]
    required_op_output_info_keys = ["paramType", "name"]
    param_type_valid_value = ["dynamic", "optional", "required"]
    is_valid = True
    for op_key in tbe_ops:
        op = tbe_ops[op_key]

        for op_info_key in op:
            if op_info_key.startswith("input"):
                op_input_info = op[op_info_key]
                missing_keys = []
                for required_op_input_info_key in required_op_input_info_keys:
                    if required_op_input_info_key not in op_input_info:
                        missing_keys.append(required_op_input_info_key)
                if len(missing_keys) > 0:
                    logging.info("op: %s %s missing: %s", op_key, op_info_key, ",".join(missing_keys))
                    is_valid = False
                else:
                    if op_input_info["paramType"] not in param_type_valid_value:
                        logging.info("op: %s %s paramType not valid, valid key:[dynamic, optional, required]",
                                     op_key, op_info_key)
                        is_valid = False
            if op_info_key.startswith("output"):
                op_input_info = op[op_info_key]
                missing_keys = []
                for required_op_input_info_key in required_op_output_info_keys:
                    if required_op_input_info_key not in op_input_info:
                        missing_keys.append(required_op_input_info_key)
                if len(missing_keys) > 0:
                    logging.info("op: %s %s missing: %s", op_key, op_info_key, ",".join(missing_keys))
                    is_valid = False
                else:
                    if op_input_info["paramType"] not in param_type_valid_value:
                        logging.info("op: %s %s paramType not valid, valid key:[fix, range, list]", op_key, op_info_key)
                        is_valid = False
    logging.info("==============check valid for ops info end================\n\n")
    return is_valid


def write_json_file(tbe_ops_info, json_file_path):
    json_file_real_path = os.path.realpath(json_file_path)
    flags = os.O_WRONLY | os.O_CREAT
    modes = stat.S_IWUSR | stat.S_IRUSR | stat.S_IWGRP | stat.S_IRGRP
    with os.fdopen(os.open(json_file_real_path, flags, modes), "w") as f:
        json.dump(tbe_ops_info, f, sort_keys=True, indent=4, separators=(',', ':'))
    logging.info("Compile op info cfg successfully.")


def parse_ini_to_json(ini_files, out_file):
    tbe_ops_info = parse_ini_files(ini_files)
    if not check_op_info(tbe_ops_info):
        logging.info("Compile op info cfg failed.")
        return False
    write_json_file(tbe_ops_info, out_file)
    return True


if __name__ == '__main__':
    args = sys.argv

    outfile_path = "tbe_ops_info.json"
    ini_file_paths = []

    for arg in args:
        if arg.endswith("ini"):
            ini_file_paths.append(arg)
        if arg.endswith("json"):
            outfile_path = arg

    if not ini_file_paths:
        ini_file_paths.append("tbe_ops_info.ini")

    if not parse_ini_to_json(ini_file_paths, outfile_path):
        sys.exit(1)
