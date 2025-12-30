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

#编译安装三方依赖
set -e
readonly CURRENT_PATH="$(cd "$(dirname "$0")"; pwd)"
CMAKE_BUILD_TYPE="$1"
if [ "${CMAKE_BUILD_TYPE}" = "" ]; then
    CMAKE_BUILD_TYPE="Release"
fi
INSTALL_PATH="${CURRENT_PATH}/../install"
OPENSOURCE_PATH="${CURRENT_PATH}/../3rdparty"

[ -d "${INSTALL_PATH}" ] && rm -rf "${INSTALL_PATH}"
mkdir -p "${INSTALL_PATH}"

SECUREC_SRC_PATH="${OPENSOURCE_PATH}/huawei_secure_c/src"
MB_BUILD_PATH="${OPENSOURCE_PATH}/memory_bridge"
echo "Build ${SECUREC_SRC_PATH}"
echo "Build ${MB_BUILD_PATH}"
SECUREC_INSTALL_PATH="${INSTALL_PATH}/securec"
MB_INSTALL_PATH="${INSTALL_PATH}/memory_bridge"

function compile_memory_bridge() {
    [ -d "${MB_INSTALL_PATH}" ] && rm -rf "${MB_INSTALL_PATH}"
    echo "${MB_INSTALL_PATH}"
    if [[ ! -d "${MB_INSTALL_PATH}" ]]; then
            mkdir -p "${MB_INSTALL_PATH}"
    fi
    cd "${MB_BUILD_PATH}"
    dos2unix *
    dos2unix buildscript/*
    echo "bash ${MB_BUILD_PATH}/build.sh -t ${CMAKE_BUILD_TYPE}"
    bash build.sh -t ${CMAKE_BUILD_TYPE}

    cp -rf "${OPENSOURCE_PATH}/memory_bridge/install/memory_bridge/include" "${MB_INSTALL_PATH}/"
    cp -rf "${OPENSOURCE_PATH}/memory_bridge/install/memory_bridge/lib" "${MB_INSTALL_PATH}/"
    cp -rf "${OPENSOURCE_PATH}/memory_bridge/install/memory_bridge/tools" "${MB_INSTALL_PATH}/"
    cp -rf "${OPENSOURCE_PATH}/memory_bridge/intraface" "${MB_INSTALL_PATH}/"
}

compile_memory_bridge
echo "compiled memory_bridge"