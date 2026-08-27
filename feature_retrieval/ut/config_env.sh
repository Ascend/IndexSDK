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

GREEN="\033[32m"
YELLOW="\033[33m"
RED="\033[31m"
END="\033[0m"

dependcy_package_install() {
    echo -e "${YELLOW} check gdb, boost, linux_tools ${END}"
    if command -v apt-get >/dev/null 2>&1; then
        apt-get install -y --no-install-recommends \
            gdb \
            libboost-all-dev \
            linux-tools-generic \
            linux-tools-common
    elif command -v yum >/dev/null 2>&1; then
        yum install -y gdb boost-devel perf
    else
        echo -e "${RED} unsupported package manager ${END}"
        exit 1
    fi
    echo -e "${GREEN} gdb, boost, perf ok ${END}"
}

dependcy_package_install
