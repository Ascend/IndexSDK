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

export LD_LIBRARY_PATH=${PWD}/beta/lib:${PWD}/lib:/usr/local/Ascend/ascend-toolkit/latest/opp/vendors/customize/op_api/lib:/usr/local/faiss/faiss1.10.0/lib64:/opt/OpenBLAS/lib:$LD_LIBRARY_PATH
export ASCEND_LATEST_INSTALL_PATH=/usr/local/Ascend/ascend-toolkit/latest
export MX_INDEX_MODELPATH=${PWD}/ops/ascend_c/op_models
export ASCEND_SLOG_PRINT_TO_STDOUT=1
export ASCEND_GLOBAL_LOG_LEVEL=3
unset ASCEND_SLOG_PRINT_TO_STDOUT
unset ASCEND_GLOBAL_LOG_LEVEL
echo "MX_INDEX_MODELPATH = " ${MX_INDEX_MODELPATH}
if [[ $# -eq 0 ]]
then
./test/build/test_ascendhost
fi

if [[ $# -eq 1 ]]
then
./test/build/test_ascendhost $1
fi

if [[ $# -eq 2 ]]
then
./test/build/test_ascendhost $1 $2
fi

if [[ $# -eq 3 ]]
then
./test/build/test_ascendhost $1 $2 $3
fi