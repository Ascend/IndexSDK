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
    echo -e "${YELLOW} check gdb ${END}"
    apt-get install -y gdb
    echo -e "${GREEN} gdb ok ${END}"

    echo -e "${YELLOW} check boost ${END}"
    apt-get install -y libboost-all-dev
    echo -e "${GREEN} boost ok ${END}"

    echo -e "${YELLOW} check linux_tools ${END}"
    apt-get install -y linux-tools-generic linux-tools-common
    echo -e "${GREEN} perf ok ${END}"
}

dependcy_package_install
