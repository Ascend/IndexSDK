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
set -e
CURRENT_PATH=$(cd "$(dirname "$0")"; pwd)
echo $CURRENT_PATH

. /usr/local/Ascend/ascend-toolkit/set_env.sh
export LD_LIBRARY_PATH="${CURRENT_PATH}/../pkg/securec/lib":$LD_LIBRARY_PATH
export LD_LIBRARY_PATH="${CURRENT_PATH}/../pkg/hcps/lib":$LD_LIBRARY_PATH
export LD_LIBRARY_PATH="${CURRENT_PATH}/../pkg/memory_bridge/lib":$LD_LIBRARY_PATH
export LD_LIBRARY_PATH="${CURRENT_PATH}/../pkg/vsa/lib":$LD_LIBRARY_PATH
export MX_INDEX_MODELPATH=/home/xyz/newModelPath

cd "${CURRENT_PATH}/build"
rm -rf *
cmake ../
make -j64

cd "${CURRENT_PATH}/build"

./develop_test 2>&1 | tee ../logs/log.log