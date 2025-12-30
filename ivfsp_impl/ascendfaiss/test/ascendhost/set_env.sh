#!/bin/bash
# -------------------------------------------------------------------------
# This file is part of the IndexSDK project.
# Copyright (c) 2025 Huawei Technologies Co.,Ltd.
#
# IndexSDK is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#          http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

current_path=`pwd`
echo $current_path
export ASCEND_LATEST_INSTALL_PATH=/usr/local/Ascend/ascend-toolkit/latest/
export MX_INDEX_MODELPATH=${current_path}/../../tools/op_models
export LD_LIBRARY_PATH=${current_path}/../../build/ascend/:/usr/local/gtest/lib64/:${current_path}/../../build/ascenddaemon/:/usr/local/protobuf/lib/:$LD_LIBRARY_PATH


export ASCEND_SLOG_PRINT_TO_STDOUT=0
export ASCEND_GLOBAL_LOG_LEVEL=3
