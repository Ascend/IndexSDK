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
CURRENT_PATH=$(cd "$(dirname "${BASH_SOURCE[0]}")"; pwd)
export HMM_ROOT_PATH=$(cd "${CURRENT_PATH}/.."; pwd)
export OUTPUT_PATH="${HMM_ROOT_PATH}/output"

export HMM_SRC_PATH="${HMM_ROOT_PATH}/src"
export HMM_TEST_PATH="${HMM_ROOT_PATH}/tests"
export HMM_THIRDPART_PATH="${HMM_ROOT_PATH}/3rdparty"
export CPU_TYPE=$(arch)
export OCK_VERSION=23.0.0
export INSTALL_PATH="${HMM_ROOT_PATH}/install"