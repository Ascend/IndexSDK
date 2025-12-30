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

targetdir=/usr/local/Ascend/opp
target_custom=0

sourcedir=$PWD/packages

log() {
    cur_date=$(date +"%Y-%m-%d %H:%M:%S")
    echo "[runtime] [$cur_date] "$1
}

if [[ "x${ASCEND_OPP_PATH}" == "x" ]];then
    log "[ERROR] env ASCEND_OPP_PATH no exist"
    exit 1
fi

targetdir="${ASCEND_OPP_PATH}"

if [ ! -d "$targetdir" ];then
    log "[ERROR] $targetdir no exist"
    exit 1
fi

chmod -R +w "$targetdir">/dev/null 2>&1

upgrade()
{
    if [ ! -d "${sourcedir}"/"$1" ]; then
        log "[INFO] no need to upgrade ops $1 files"
        return 0
    fi

    if [ ! -d "${targetdir}"/"$1" ];then
        log "[INFO] create ${targetdir}/$1."
        mkdir -p "${targetdir}"/"$1"
        if [ $? -ne 0 ];then
            log "[ERROR] create ${targetdir}/$1 failed"
            return 1
        fi
    else
        log "[INFO] replace old ops $1 files ......"
    fi

    log "copy new ops $1 files ......"
    cp -rf "${sourcedir}"/"$1"/* "$targetdir"/"$1"/
    if [ $? -ne 0 ];then
        log "[ERROR] copy new $1 files failed"
        return 1
    fi

    return 0
}
log "[INFO] start install ops ..."

echo "[ops_custom]upgrade framework"
upgrade framework
if [ $? -ne 0 ];then
    exit 1
fi

echo "[ops_custom]upgrade op proto"
upgrade op_proto
if [ $? -ne 0 ];then
    exit 1
fi

echo "[ops_custom]upgrade op impl"
upgrade op_impl
if [ $? -ne 0 ];then
    exit 1
fi

changemode()
{
    if [ -d "${targetdir}" ];then
        chmod -R 550 "${targetdir}">/dev/null 2>&1
    fi

    return 0
}
echo "[ops_custom]changemode..."
changemode
if [ $? -ne 0 ];then
    exit 1
fi

chmod -R -w "${targetdir}">/dev/null 2>&1

echo "SUCCESS"
exit 0

