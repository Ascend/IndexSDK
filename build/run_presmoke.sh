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

readonly CUR_DIR=$(dirname "$(readlink -f "$0")")
readonly RUN_PKG_PATH="${CUR_DIR}/../.."
readonly PRESMOKE_DIR="/home/indexPreSmoke"
export MX_INDEX_INSTALL_PATH=/usr/local/Ascend/mxIndex
export MX_INDEX_MODELPATH=$PRESMOKE_DIR/pkg/modelpath_512
export MX_INDEX_FINALIZE=1

echo "[PRESMOKE_INFO] indexPreSmoke start"

# ============== 1. install run pkg ==============
echo "[PRESMOKE_INFO] start install run pkg..."

if [ -d "$PRESMOKE_DIR/pkg" ]; then
    echo "[PRESMOKE_INFO] test files already exist, removing..."
    rm -rf "$PRESMOKE_DIR/pkg"
fi
echo "[PRESMOKE_INFO] mkdir: $PRESMOKE_DIR/pkg"
mkdir -p "$MX_INDEX_MODELPATH"

if [ -d "/usr/local/Ascend/mxIndex" ]; then
    echo "[PRESMOKE_INFO] mxIndex already exist, uninstalling..."
    if [ -f "/usr/local/Ascend/mxIndex/script/uninstall.sh" ]; then
        bash /usr/local/Ascend/mxIndex/script/uninstall.sh
    else
        echo "[PRESMOKE_WARN] uninstall.sh not found, performing manual cleanup..."
        rm -rf /usr/local/Ascend/mxIndex
    fi
fi

cp "$RUN_PKG_PATH"/Ascend-mindxsdk-mxindex_*_linux-aarch64.run "$PRESMOKE_DIR/pkg"
cd "$PRESMOKE_DIR/pkg"
chmod +x *.run
echo "[PRESMOKE_INFO] start installing run pkg"
./Ascend-mindxsdk-mxindex_*_linux-aarch64.run --install --platform=910B

# ============== 2. generate ops ==============
echo "[PRESMOKE_INFO] start generate ops..."

source /etc/profile
bash /usr/local/Ascend/mxIndex/ops/custom_opp_*.run
cd $MX_INDEX_INSTALL_PATH/tools
python3 aicpu_generate_model.py -t 910B4
python3 flat_generate_model.py -d 512 -t 910B4
cp op_models/* $MX_INDEX_MODELPATH
cd $MX_INDEX_MODELPATH && chmod 644 *

# ============== 3. run test demo ==============
echo "[PRESMOKE_INFO] start run test demo..."

cd "$PRESMOKE_DIR"
./indexDemo

echo "[PRESMOKE_INFO] indexPreSmoke finished"
