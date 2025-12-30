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

if [ $# -lt 1 ] ; then
    echo "Usage:  bash build.sh NPU_TYPE"
    echo "  Examples:  bash build.sh 310 / bash build.sh 310P"
    exit 1
fi
build_target=$1

if [ "$ASCEND_HOME" ]; then
    ASCEND_HOME="$ASCEND_HOME"
else
    ASCEND_HOME=/usr/local/Ascend
fi

if [ "$ASCEND_VERSION" ]; then
    ASCEND_VERSION="$ASCEND_VERSION"
else
    ASCEND_VERSION=ascend-toolkit/latest
fi

if [ ! "$PYTHON" ]; then
    echo "make sure /usr/bin/python3 exists!"
    export PYTHON=/usr/bin/python3
fi

if [ ! -d "${ASCEND_HOME}/${ASCEND_VERSION}/atc/include" ]; then
    echo "Please set right ASCEND_HOME, now ASCEND_HOME=${ASCEND_HOME}"
    echo "Please set right ASCEND_VERSION, now ASCEND_VERSION=${ASCEND_VERSION}"
    echo "Usage: export ASCEND_HOME=\${driver/ascend-toolkit_install_path}"
    echo "       export ASCEND_VERSION=ascend-toolkit/latest"
    exit 1
fi

echo "ASCEND_TOOLKIT_PATH: ${ASCEND_HOME}/${ASCEND_VERSION}"

export ASCEND_TENSOR_COMPLIER_INCLUDE="${ASCEND_HOME}/${ASCEND_VERSION}/atc/include"
export ASCEND_OPP_PATH="${ASCEND_HOME}/${ASCEND_VERSION}/opp"
export PROJECT_PATH="$(pwd)"

echo "ops build_target=${build_target}"

if [ -d "$PROJECT_PATH/build_out" ]; then
    rm -rf "$PROJECT_PATH"/build_out
fi

mkdir -p  $PROJECT_PATH/build_out
cd $PROJECT_PATH/build_out
cmake .. -DCMAKE_CXX_COMPILER=g++ -DNPU_TYPE=${build_target}
make -j

if [ $? -ne 0 ]; then
    echo "[ERROR] build operator faild!"
    exit 1
fi
