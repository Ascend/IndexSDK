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
INSTALL_PATH="${CURRENT_PATH}/../install"
OPENSOURCE_PATH="${CURRENT_PATH}/../3rdparty"

[ -d "${INSTALL_PATH}" ] && rm -rf "${INSTALL_PATH}"
mkdir -p "${INSTALL_PATH}"

SECUREC_SRC_PATH="${OPENSOURCE_PATH}/huawei_secure_c/src"
echo "Build ${SECUREC_SRC_PATH}"
SECUREC_INSTALL_PATH="${INSTALL_PATH}/securec"

function compile_securec() 
{
    [ -d "${SECUREC_INSTALL_PATH}" ] && rm -rf "${SECUREC_INSTALL_PATH}"
    echo "${SECUREC_INSTALL_PATH}"
    if [[ ! -d "${SECUREC_INSTALL_PATH}" ]]; then
        mkdir -p "${SECUREC_INSTALL_PATH}"
    fi
    cd "${SECUREC_SRC_PATH}"
    make -j 10
    make lib -j 10 # build .a
    cp -rf "${OPENSOURCE_PATH}/huawei_secure_c/include" "${SECUREC_INSTALL_PATH}/"
    cp -rf "${OPENSOURCE_PATH}/huawei_secure_c/lib" "${SECUREC_INSTALL_PATH}/"
}

function compile_openssl() {
    echo "making openssl patch"
    cmake ${OPENSOURCE_PATH}
}

compile_securec
echo "compiled huawei_secure_c"
compile_openssl
echo "compiled huawei_openssl"