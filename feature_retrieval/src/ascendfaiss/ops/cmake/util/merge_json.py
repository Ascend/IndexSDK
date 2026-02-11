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
import logging
import os
import shutil
import sys
import const_var

logging.basicConfig(level=logging.NOTSET, format="")


def insert_op_info(src_file, dst_file):
    if not os.path.exists(src_file):
        logging.error("%s not exist.", src_file)
        return
    if not os.path.exists(dst_file):
        logging.error("%s not exist.", dst_file)
        return

    with open(src_file, 'r') as load_f:
        insert_operator = json.load(load_f)

    all_operators = {}
    if os.path.exists(dst_file):
        if os.path.getsize(dst_file) != 0:
            with open(dst_file, 'r') as load_f:
                all_operators = json.load(load_f)

    for k in insert_operator.keys():
        if k in all_operators.keys():
            logging.info("op replace [', %s, '] success", k)
        else:
            logging.info("op insert [', %s, '] success", k)
        all_operators[k] = insert_operator[k]

    with os.fdopen(os.open(dst_file, const_var.WFLAGS, const_var.WMODES), 'w') as json_file:
        json_file.write(json.dumps(all_operators, indent=4))

    logging.info("insert %s to %s success", src_file, dst_file)

if __name__ == '__main__':
    if len(sys.argv) != 3:
        logging.error(sys.argv)
        logging.error('argv error, ascendc_json_dir tbe_json_dir')
        sys.exit(2)

    npu_types = ['ascend310', 'ascend310p', 'ascend910b', 'ascend910_93', 'ascend950']
    for npu_type in npu_types:
        file_name = "aic-{}-ops-info.json".format(npu_type)
        ascendc_json_file = os.path.join(sys.argv[1], npu_type, file_name)
        tbe_json_file = os.path.join(sys.argv[2], npu_type, file_name)
        # ascendc 存在，tbe不存在，copy
        if os.path.exists(ascendc_json_file) and not os.path.exists(tbe_json_file):
            dstdir = os.path.join(sys.argv[2], npu_type)
            logging.info("copy %s to %s.", ascendc_json_file, dstdir)
            if not os.path.exists(dstdir):
                os.makedirs(dstdir)
            shutil.copy(ascendc_json_file, dstdir)
        # ascendc 存在，tbe存在，insert
        elif os.path.exists(ascendc_json_file) and os.path.exists(tbe_json_file):
            logging.info("insert %s to %s.", ascendc_json_file, tbe_json_file)
            insert_op_info(ascendc_json_file, tbe_json_file)
        else:
            pass
