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
CMAKE_BUILD_TYPE="$1"
USING_COVERAGE="$2"
USING_XSAN="$3"
COMPILE_VERIFICATION="$4"
CURRENT_PATH=$(cd "$(dirname "$0")"; pwd)
source "${CURRENT_PATH}/build_env.sh"
BUILD_PATH=$(cd "$CURRENT_PATH/../build"; pwd)
CPU_TYPE="$(arch)"
echo "Build ${BUILD_PATH} Type=${CMAKE_BUILD_TYPE}"
if [ ! -d "${BUILD_PATH}" ]; then
    mkdir -p "${BUILD_PATH}"
else
    [ -d "${BUILD_PATH}" ] && rm -rf "${BUILD_PATH}"
    mkdir -p "${BUILD_PATH}"
fi
rm -rf "${OUTPUT_PATH}/vsa"
mkdir -p "${OUTPUT_PATH}/vsa"
cp -rf "${VSA_ROOT_PATH}/interface" "${OUTPUT_PATH}/vsa/include"
echo "copy include success."
cd "${BUILD_PATH}"
if ! cmake "${VSA_ROOT_PATH}" -DCMAKE_INSTALL_PREFIX:STRING="${OUTPUT_PATH}" -DCPU_TYPE="${CPU_TYPE}" \
    -DCMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE}" -DPROJECT_ROOT_PATH="${VSA_ROOT_PATH}" -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
    -DUSING_COVERAGE="${USING_COVERAGE}" -DUSING_XSAN="${USING_XSAN}" -DCOMPILE_VERIFICATION="${COMPILE_VERIFICATION}"; then
    echo "cmake failed."
	exit 1
fi

echo "cmake success."

make clean
if ! make -j 10; then
    echo "make -j 10 failed."
    exit 1
fi
if ! make install; then
    echo "make install failed."
    exit 1
fi
cp -rf "${OUTPUT_PATH}/vsa" "${INSTALL_PATH}/."
echo "make success."
