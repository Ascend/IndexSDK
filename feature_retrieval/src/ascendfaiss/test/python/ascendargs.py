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
import ast
import argparse
import ascendfaiss


parser = argparse.ArgumentParser(description='Test feature_retrieval Sift1M recall')
parser.add_argument("--data-home", default="/data", help="sift1M home", type=str)
parser.add_argument("--devices", default=None, help="device list, split by ,", type=str)
parser.add_argument("-m", "--metric-type", default="L2", help="metric type", type=str)
parser.add_argument("--normalized", default=None, help="if need nomalized", type=ast.literal_eval)
parser.add_argument("--scaled", default=True, help="if need scaled", type=ast.literal_eval)
parser.add_argument("--by-residual", default=True, help="for ivf index", type=ast.literal_eval)
args = parser.parse_args()

devices = args.devices
metric_name = args.metric_type
metric_type = ascendfaiss.METRIC_INNER_PRODUCT if metric_name == "IP" else ascendfaiss.METRIC_L2
need_norm = metric_type == ascendfaiss.METRIC_INNER_PRODUCT
if args.normalized is not None:
    need_norm = args.normalized
need_scale = args.scaled
by_residual = args.by_residual
data_home = os.path.join(args.data_home, "sift1M")

devices = [0]
if args.devices is not None:
    devices = args.devices.split(",")
    while '' in devices:
        devices.remove('')
    devices = [int(elem) for elem in devices]
if len(devices) == 0:
    raise Exception("wrong arguments, device list is empty")
