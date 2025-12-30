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
readonly CURRENT_PATH=$(cd "$(dirname "$0")"; pwd)
CMAKE_BUILD_TYPE="$1"
COMPILE_VERIFICATION="$2"
USING_COVERAGE="$3"
if [ "${CMAKE_BUILD_TYPE}" = "" ]; then
    CMAKE_BUILD_TYPE="Release"
fi
INSTALL_PATH="${CURRENT_PATH}/../install"
OPENSOURCE_PATH="${CURRENT_PATH}/../3rdparty"
if [ ! -d "${INSTALL_PATH}" ]; then
    mkdir -p "${INSTALL_PATH}"
else
    [ -d "${INSTALL_PATH}" ] && rm -rf "${INSTALL_PATH}"
    mkdir -p "${INSTALL_PATH}"
fi

HCPS_BUILD_PATH="${OPENSOURCE_PATH}/hcps"
echo "Build ${HCPS_BUILD_PATH}"
SECUREC_INSTALL_PATH="${INSTALL_PATH}/securec"
MB_INSTALL_PATH="${INSTALL_PATH}/memory_bridge"
HCPS_INSTALL_PATH="${INSTALL_PATH}/hcps"

function compile_hcps_pier() {
    [ -d "${HCPS_INSTALL_PATH}" ] && rm -rf "${HCPS_INSTALL_PATH}"
    echo "${HCPS_INSTALL_PATH}"
    if [[ ! -d "${HCPS_INSTALL_PATH}" ]]; then
        mkdir -p "${HCPS_INSTALL_PATH}"
    fi

    [ -d "${MB_INSTALL_PATH}" ] && rm -rf "${MB_INSTALL_PATH}"
    echo "${MB_INSTALL_PATH}"
    if [[ ! -d "${MB_INSTALL_PATH}" ]]; then
            mkdir -p "${MB_INSTALL_PATH}"
    fi

    cd "${HCPS_BUILD_PATH}"
    dos2unix *
    dos2unix buildscript/*
    echo "bash ${HCPS_BUILD_PATH}/build.sh -t ${CMAKE_BUILD_TYPE} -c ${USING_COVERAGE}"
    bash build.sh -t ${CMAKE_BUILD_TYPE} -c ${USING_COVERAGE}

    cp -rf "${OPENSOURCE_PATH}/hcps/install/memory_bridge/lib" "${MB_INSTALL_PATH}"
    cp -rf "${OPENSOURCE_PATH}/hcps/install/memory_bridge/include" "${MB_INSTALL_PATH}/"
    cp -rf "${OPENSOURCE_PATH}/hcps/install/memory_bridge/intraface" "${MB_INSTALL_PATH}/"
    cp -rf "${OPENSOURCE_PATH}/hcps/install/memory_bridge/tools" "${MB_INSTALL_PATH}/"
    cp -rf "${OPENSOURCE_PATH}/hcps/install/hcps_pier/include" "${HCPS_INSTALL_PATH}/"
    cp -rf "${OPENSOURCE_PATH}/hcps/install/hcps_pier/lib" "${HCPS_INSTALL_PATH}/"
    cp -rf "${OPENSOURCE_PATH}/hcps/intraface" "${HCPS_INSTALL_PATH}/"
}

function compile_openssl() {
    echo "making openssl patch"
    cmake ${OPENSOURCE_PATH}
}

compile_hcps_pier
echo "compiled hcps_pier"

if [ "${COMPILE_VERIFICATION}" = "ON" ]; then
  compile_openssl
  echo "compiled huawei_openssl"
fi
#compile_memory_bridge
#echo "compiled memory_bridge"