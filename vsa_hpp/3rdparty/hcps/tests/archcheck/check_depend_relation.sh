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
CURRENT_PATH=$(cd "$(dirname "$0")"; pwd)
cd "${CURRENT_PATH}/../.."
set -e
PKG_ROOT_PATH=$(pwd)
GCC_VERSION=$(gcc --version|grep GCC|grep gcc|awk '{print $NF}')
GCC_TARGET=$(gcc -v 2>&1|grep Target|awk -F':' '{print $2}'|awk '{print $1}')
HPP_INCLUDE_PATH="-I /usr/local/gcc${GCC_VERSION}/include/c++/${GCC_VERSION}"
HPP_INCLUDE_PATH+=" -I /usr/local/gcc${GCC_VERSION}/include/c++/${GCC_VERSION}/${GCC_TARGET}"
HPP_INCLUDE_PATH+=" -I ${ASCEND_HOME_PATH}/include"
HPP_INCLUDE_PATH+=" -I ${PKG_ROOT_PATH}/install/securec/include"
HPP_INCLUDE_PATH+=" -I ${PKG_ROOT_PATH}/install/memory_bridge/include"
HPP_INCLUDE_PATH+=" -I ${PKG_ROOT_PATH}/install/memory_bridge/intraface"

IS_EXISTIS_ERROR="False"
check_hpp_depend()
{
    isEmpty="$(gcc -E -M ${HPP_INCLUDE_PATH} "$1" 2>&1|grep -v "c++0x_warning.h"|grep "fatal error"| sed 's/fatal error:\(.*\)$/\o033[31m\1\o033[0m/')"
    if [ "$isEmpty" != "" ]; then
        echo "$isEmpty"
        IS_EXISTIS_ERROR="True"
    fi
}

check_interface()
{
    inter_file_list="$(find ${PKG_ROOT_PATH}/interface/ -type f -name "*.h")"
    for file_path in ${inter_file_list}
    do
        check_hpp_depend "$file_path"
    done
}
check_intraface()
{
    HPP_INCLUDE_PATH+=" -I ${PKG_ROOT_PATH}/intraface"
    intra_file_list="$(find ${PKG_ROOT_PATH}/intraface/ -type f -name "*.h")"
    for file_path in ${intra_file_list}
    do
        check_hpp_depend "$file_path"
    done
}

HPP_INCLUDE_PATH+=" -I ${PKG_ROOT_PATH}/interface"
check_interface
HPP_INCLUDE_PATH+=" -I ${PKG_ROOT_PATH}/intraface"
check_intraface
if [ "$IS_EXISTIS_ERROR" = "True" ]; then
   echo "check dependency failed!"
   exit 255
fi
