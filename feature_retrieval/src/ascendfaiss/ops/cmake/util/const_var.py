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


REPLAY_BATCH = 'batch'
REPLAY_ITERATE = 'iterate'
CFG_IMPL_DIR = 'impl_dir'
CFG_OUT_DIR = 'out_dir'
AUTO_GEN_DIR = 'auto_gen_dir'
WFLAGS = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
WMODES = stat.S_IWUSR | stat.S_IRUSR
SOC_MAP_EXT = {
    'ascend310p': 'Ascend310P3', 'ascend310b': 'Ascend310B1', 'ascend910': 'Ascend910A', 'ascend910b': 'Ascend910B1'
}

BIN_CMD = 'opc $1 --main_func={fun} --input_param={param} --soc_version={soc} \
--output=$2 --impl_mode={impl} --simplified_key_mode=0 --op_mode=dynamic\n'
CHK_CMD = '''
if ! test -f $2/{res_file} ; then
  echo "$2/{res_file} not generated!"
  exit 1
fi
'''
ATTR_DEF_VAL = {
    'str' : '', 'int': 0, 'float': 0.0, 'bool': False, 'list_bool': [], 'list_int': [],
    'list_float': [], 'list_list_int': [[]]
}
