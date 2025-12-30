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

vendor_name=mxIndex
targetdir=/usr/local/Ascend/opp
target_custom=0

sourcedir=$(pwd)/packages
vendordir=vendors/$vendor_name
LOG_SIZE_THRESHOLD=1024000
PACKAGE_LOG_NAME=mxIndex
info_record_path="$HOME/log/mxIndex"
info_record_file="deployment.log"
info_record_file_back="deployment.log.bak"
log_file=$info_record_path/$info_record_file

readonly user_n=$(whoami)
readonly who_path=$(which who)
readonly cut_path=$(which cut)
ip_n=$(${who_path} -m | ${cut_path} -d '(' -f 2 | ${cut_path} -d ')' -f 1)
if [ "${ip_n}" = "" ]; then
    ip_n="localhost"
fi
readonly ip_n

log() {
    if [ ! -d "$info_record_path" ];then
      mkdir -p "$info_record_path"
      chmod 750 "$info_record_path"
    fi
    if [[ ! -f "$log_file" ]];then
      touch "$log_file"
      chmod 640 "$log_file"
    fi
    # print log to log file
    if [[ "$log_file" = "" ]] || [[ "$quiet_flag" = n ]] || [[ "$3" = "y" ]]; then
        echo -e "[${PACKAGE_LOG_NAME}] [$(date +%Y%m%d-%H:%M:%S)] [user: $user_n] [$ip_n] [$1] $2"
    fi

    if [ -f "$log_file" ]; then
        if ! echo -e "[${PACKAGE_LOG_NAME}] [$(date +%Y%m%d-%H:%M:%S)] [$user_n] [$ip_n] [$1] $2" >>"$log_file"
        then
          echo "can not write log, exiting!" >&2
          exit 1
        fi
    else
        echo "log file not exist, exiting!" >&2
        exit 1
    fi
}
if [[ "x${ASCEND_OPP_PATH}" == "x" ]];then
    log "ERROR" "env ASCEND_OPP_PATH no exist" "y"
    exit 1
fi

targetdir="${ASCEND_OPP_PATH}"

if [ ! -d "$targetdir" ];then
    log "ERROR" "$targetdir no exist" "y"
    exit 1
fi

targetdir=$(realpath "$targetdir")
 
cann_owner=$(stat -c '%U' "$targetdir")
current_user=$(whoami)
 
if [ "$current_user" != "$cann_owner" ]; then
    log "ERROR" "Cann owner is not current user." "y"
    exit 126
fi

upgrade()
{
    if [ ! -d "${sourcedir}"/"$vendordir"/"$1" ]; then
        log "INFO" "no need to upgrade ops $1 files" "y"
        return 0
    fi

    if [ ! -d "${targetdir}"/"$vendordir"/"$1" ];then
        log "INFO" "create ${targetdir}/$vendordir/$1."
        mkdir -p "${targetdir}"/"$vendordir"/"$1"
        if [ $? -ne 0 ];then
            log "ERROR" "create ${targetdir}/$vendordir/$1 failed" "y"
            return 1
        fi
    fi

    log "INFO" "copy new ops $1 files ......"
    if [ -d "${targetdir}"/"$vendordir"/"$1"/ ]; then
        chmod -R +w "$targetdir/$vendordir/$1/" >/dev/null 2>&1
    fi
    cp -rf "${sourcedir}"/"$vendordir"/"$1"/* "$targetdir"/"$vendordir"/"$1"/
    if [ $? -ne 0 ];then
        log "ERROR" "copy new $1 files failed" "y"
        return 1
    fi

    return 0
}

upgrade_file()
{
    if [ ! -e "${sourcedir}"/"$vendordir"/"$1" ]; then
        log "INFO" "no need to upgrade ops $1 file"
        return 0
    fi

    log "INFO" "copy new $1 files ......"
    cp -f "${sourcedir}"/"$vendordir"/"$1" "$targetdir"/"$vendordir"/"$1"
    if [ $? -ne 0 ];then
        log "ERROR" "copy new $1 file failed" "y"
        return 1
    fi

    return 0
}

delete_optiling_file()
{
  if [ ! -d "${targetdir}"/vendors ];then
    log "INFO" "$1 not exist, no need to uninstall" "y"
    return 0
  fi
  sys_info=$(uname -m)
  if [ ! -d "${sourcedir}"/"$vendordir"/"$1"/ai_core/tbe/op_tiling/lib/linux/"${sys_info}" ];then
    rm -rf "${sourcedir}"/"$vendordir"/"$1"/ai_core/tbe/op_tiling/liboptiling.so
  fi
  return 0
}

log "INFO" "start install ops ..." "y"

if [ ! -d "${targetdir}"/vendors ];then
    log "ERROR" "${targetdir}/vendors not exist." "y"
    return 1
fi
 
if [ -d "${targetdir}"/"$vendordir" ];then
    chmod -R u+w "${targetdir}"/"$vendordir"
fi
 
config_file="${targetdir}"/vendors/config.ini
if [ ! -f "${config_file}" ]; then
    touch "${config_file}"
    chmod 640 "${config_file}"
    echo "load_priority=$vendor_name" > "${config_file}"
    if [ $? -ne 0 ];then
        log "ERROR" "echo load_priority failed" "y"
        exit 1
    fi
else
    found_vendors="$(grep -w "load_priority" "$config_file" | cut --only-delimited -d"=" -f2-)"
    found_vendor=$(echo "$found_vendors" | sed "s/$vendor_name//g" | tr ',' ' ')
    vendor=$(echo "$found_vendor" | tr -s ' ' ',')
    # 检查vendor是否由大小写字母和数据及下划线组成，且不为空
    if [[ "$vendor" =~ ^[a-zA-Z0-9_]+$ && -n "$vendor" ]]; then
        sed -i "/load_priority=$found_vendors/s@load_priority=$found_vendors@load_priority=$vendor_name,$vendor@g" "$config_file"
    fi
fi

upgrade framework
if [ $? -ne 0 ];then
    exit 1
fi

upgrade op_proto
if [ $? -ne 0 ];then
    exit 1
fi

upgrade_file version.info
if [ $? -ne 0 ];then
    exit 1
fi

delete_optiling_file op_impl
upgrade op_impl
if [ $? -ne 0 ];then
    exit 1
fi

upgrade op_api
if [ $? -ne 0 ];then
    exit 1
fi

config_file="${targetdir}"/vendors/config.ini
if [ ! -f "${config_file}" ]; then
    touch "${config_file}"
    chmod 640 "${config_file}"
    echo "load_priority=$vendor_name" > "${config_file}"
    if [ $? -ne 0 ];then
        echo "echo load_priority failed"
        exit 1
    fi
else
    found_vendors="$(grep -w "load_priority" "$config_file" | cut --only-delimited -d"=" -f2-)"
    found_vendor=$(echo "$found_vendors" | sed "s/$vendor_name//g" | tr ',' ' ')
    vendor=$(echo "$found_vendor" | tr -s ' ' ',')
    if [ "$vendor" != "" ]; then
        sed -i "/load_priority=$found_vendors/s@load_priority=$found_vendors@load_priority=$vendor_name,$vendor@g" "$config_file"
    fi
fi

changemode()
{
    if [ -d "${targetdir}"/"$vendordir" ];then
        chmod -R 550 "${targetdir}"/"$vendordir" >/dev/null 2>&1
    fi

    return 0
}

changemode
if [ $? -ne 0 ];then
    exit 1
fi

chmod -R -w "${targetdir}"/"$vendordir">/dev/null 2>&1
log "INFO" "Install package successfully" "y"
exit 0

