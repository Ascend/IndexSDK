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


arch=$(uname -m)
install_flag="n"
version_flag="n"
upgrade_flag="n"
quiet_flag="n"
tmp_path=$(pwd)
PACKAGE_LOG_NAME=mxIndex
info_record_path="$HOME/log/mxIndex"
info_record_file="deployment.log"
info_record_file_back="deployment.log.bak"
log_file=$info_record_path/$info_record_file
LOG_SIZE_THRESHOLD=1024000
ascend_type=310
MAX_LEN_OF_PATH=1024
MIN_LEN_OF_PATH=0
log_init_flag=n
package_name=""
package_arch=""

current_uid=$(id -u)
readonly current_uid

if [[ "$UID" = "0" ]]; then
    install_path="/usr/local/Ascend"
else
    install_path="${HOME}/Ascend"
fi

function print() {
    # 将关键信息打印到屏幕上
    echo "[${PACKAGE_LOG_NAME}] [$(date +%Y%m%d-%H:%M:%S)] [user: $user_n] [$ip_n] [$1] $2"
}
readonly user_n=$(whoami)
readonly who_path=$(which who)
readonly cut_path=$(which cut)
ip_n=$(${who_path} -m | ${cut_path} -d '(' -f 2 | ${cut_path} -d ')' -f 1)
if [ "${ip_n}" = "" ]; then
    ip_n="localhost"
fi
readonly ip_n

function check_path() {
    local path=$1
    [[ ${#path} -gt ${MAX_LEN_OF_PATH} ]] || [[ ${#path} -le ${MIN_LEN_OF_PATH} ]] && print "ERROR" "${path} length is invalid, either exceeding ${MAX_LEN_OF_PATH} or less than ${MIN_LEN_OF_PATH}, exiting" "$force_exit" && exit 1
    [[ $(echo "$path" | wc -l) -gt 1 ]]  && print "ERROR" "${path} contains newline characters, exiting"  && exit 1
    [[ -n $(echo "$path" | grep -Ev '^[/~][-_.0-9a-zA-Z/]*$') ]]  && print "ERROR" "${path} must start with '/' or '~' and characters only can contain '-_.0-9a-zA-Z/', exiting"  && exit 1
    [[ $(echo "$path" | grep -E "\.\.") ]]  && print "ERROR" "${path} contains .. , exiting"  && exit 1
}

function error_exit() {
    local force_exit=$2
    if [[ "$force_exit" == "T" ]]; then
      print "ERROR" "exiting due to $1"
      log "ERROR"  "run failed on $1"
      exit 1
    fi
    log "ERROR"  "run failed on $1" "y"
    exit 1
}

function log() {
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
        echo -e "[${PACKAGE_LOG_NAME}] [$(date +%Y%m%d-%H:%M:%S)] [user: $USER_NAME] [$ip_n] [$1] $2"
    fi

    if [ -f "$log_file" ]; then
        log_check "$log_file"
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

#请在此处定义各种函数

function is_safe_owned()
{
    local path=$1
    local force_exit=$2
    local allow_group_write=$3

    check_path "$path"

    if [ -L "${path}" ]; then
        error_exit "The $path is a soft link! exiting" "$force_exit"
    fi
    local user_id
    user_id=$(stat -c %u "${path}")
    local group_id
    group_id=$(stat -c %g "${path}")
    if [ -z "${user_id}" ] || [ -z "${group_id}" ]; then
        error_exit "user or group not exist, exiting" "$force_exit"
    fi
    if [ "$(stat -c '%A' "${path}"|cut -c9)" == w ]; then
        error_exit "file $path does not meet with security rules other write, other users have write permission. exiting" "$force_exit"
    fi
    if [[ "$allow_group_write" != T ]] && [ "$(stat -c '%A' "${path}"|cut -c6)" == w ]; then
        error_exit "file $path does not meet with security rules group write, group has write permission. exiting" "$force_exit"
    fi
    if [ "${user_id}" != "0" ] && [ "${user_id}" != "${current_uid}" ]; then
        error_exit "The $path is not owned by root or current user, exiting" "$force_exit"
    fi
    return 0
}

function safe_path_exit()
{
    local path=$1

    check_path "$path"

    while [ ! -e "${path}" ]; do
        path=$(dirname "${path}")
    done
    path=$(realpath -s "$path")
    local cur=${path}
    while true; do
        if [ "${cur}" == '/' ]; then
           break
        fi
        is_safe_owned_exit "${cur}"
        cur=$(dirname "${cur}")
    done
}

function is_safe_owned_exit()
{
    local path=$1

    check_path "$path"

    if [ -L "${path}" ]; then
        print "ERROR" "The $path is a soft link! exiting" && exit 1
    fi
    local user_id
    user_id=$(stat -c %u "${path}")
    local group_id
    group_id=$(stat -c %g "${path}")
    if [ -z "${user_id}" ] || [ -z "${group_id}" ]; then
        print "ERROR" "user or group not exist, exiting" && exit 1
    fi
    if [ "$(stat -c '%A' "${path}"|cut -c9)" == w ]; then
        print "ERROR" "file $path does not meet with security rules other write, other users have write permission, exiting" && exit 1
    fi
    if [ "${user_id}" != "0" ] && [ "${user_id}" != "${current_uid}" ]; then
        print "ERROR" "The $path is not owned by root or current user, exiting" && exit 1
    fi
    return 0
}


function safe_path()
{
    local path=$1
    local force_exit=$2
    local allow_group_write=$3
    check_path "$path"
    while [[ ! -e "${path}" ]]; do
        path=$(dirname "${path}")
    done
    path=$(realpath -s "$path")
    local cur=${path}
    while true; do
        if [[ "${cur}" == '/' ]]; then
           break
        fi
        is_safe_owned "${cur}" "$force_exit" "$allow_group_write"
        cur=$(dirname "${cur}")
    done
}


function safe_change_mode() {
    local mode=$1
    local path=$2
    local allow_group_write=$3
    safe_path "$path" F "$allow_group_write"
    chmod "${mode}" "${path}"
}

### 脚本入参的相关处理函数
function check_script_args() {
    # 检测脚本参数的组合关系
    ######################  check params confilct ###################

    if [[ "${package_arch}" != *"${arch}"* ]];then
        print "ERROR" "the package is ${package_arch} but current is ${arch}, exit."
        error_exit "package_arch"
    fi

    if [ "$install_flag" != "y" ] && [ "$upgrade_flag" != "y" ]; then
        print "ERROR" "parameter error ! Mode is neither install, upgrade."
        error_exit "parameter error ! Mode is neither install, upgrade."
    fi

    if [ "$install_path_flag" = y ]; then
        if [ "$install_flag" = "n" ] && [ "$upgrade_flag" = "n" ]; then
            print "ERROR" "Unsupported separate 'install-path' used independently"
            error_exit "Unsupported separate 'install-path' used independently"
        fi
    fi
}

function make_file() {
    safe_path "${1}" T T
    if touch "${1}" 2>/dev/null
    then
        print "INFO" "create $1 success"
    else
        print "ERROR" "create log file failed"
        exit 1
    fi
    safe_change_mode 640 "${1}" T
}

function log_init() {
    if [ "${log_init_flag}" = "y" ];then
        return
    fi
    # 日志模块初始化
    # 判断输入的安装路径路径是否存在，不存在则创建
    mkdir -p "${info_record_path}" 2>/dev/null
    if [ ! -f "$log_file" ]; then
        make_file "$log_file"
    fi
    chmod 750 "${info_record_path}"
    chmod 640 "${log_file}"
    log "INFO" "LogFile ${log_file}"
    log_init_flag=y
}

function rotate_log() {
    safe_path_exit "$log_file"
    if [ -f "$info_record_path/deployment.log.bak" ] && [ "$UID" = "0" ]; then
      chown -h root:root "$info_record_path/deployment.log.bak"
    fi
    safe_path_exit "$info_record_path/deployment.log.bak"
    mv -f "$log_file" "$info_record_path/deployment.log.bak"
    touch "$log_file" 2>/dev/null
    safe_change_mode 440 "$info_record_path/deployment.log.bak" T
    safe_change_mode 640 "$log_file" T
}

function log_check() {
    local log_size
    log_size=$(find "$log_file" -exec stat -c %s {} \;)
    if [[ "${log_size}" -ge "${LOG_SIZE_THRESHOLD}" ]];then
        rotate_log
    fi
}

log_init

package_content() {
    version_content=$(head -n +1 "$tmp_path/version.info")
    version_number=$(echo "$version_content" | cut -d ":" -f 2)
    package_name="mxIndex-$version_number"

    version_content_withb=$(head -n 2 "$tmp_path/version.info" | tail -n 1)
    version_number_withb=$(echo "$version_content_withb" | cut -d ":" -f 2)

    arch_content=$(head -n 3 "$tmp_path/version.info" | tail -n 1)
    package_arch=$(echo "$arch_content" | cut -d ":" -f 2)
}

# 解析脚本自身的参数
function parse_script_args() {
    log "INFO" "start to run"
    local all_para_len="$*"
    if [[ ${#all_para_len} -gt 1024 ]]; then
        error_exit "The total length of the parameter is too long"
    fi
    local num=0
    while true; do
        if [[ "$1" == "" ]]; then
            break
        fi
        if [[ "${1: 0: 2}" == "--" ]]; then
            num=$((num + 1))
        fi
        if [[ $num -gt 2 ]]; then
            break
        fi
        shift 1
    done

    package_content

    while true; do
        case "$1" in
        --version)
            version_flag=y
            echo "${PACKAGE_LOG_NAME} version:${version_number_withb}"
            exit 0
            ;;
        --check)
            print "INFO" "Check successfully, exit with 0"
            exit 0
            ;;
        --install)
            install_flag=y
            shift
            ;;
        --install-path=*)
            # 去除指定安装目录后所有的 "/"
            install_path=$(echo "$1" | cut -d"=" -f2 | sed "s/\/*$//g")
            install_path_flag=y
            safe_path "$install_path"
            local home_dir="$(echo ~)"
            install_path=$(echo "$install_path" | sed -e "s#^~#${home_dir}#")
            shift
            ;;
        --upgrade)
            upgrade_flag=y
            shift
            ;;
        --platform=*)
            ascend_type=$(echo "$1" | cut -d"=" -f2)
            if [[ $ascend_type != 310 && $ascend_type != 310P && $ascend_type != 910B && $ascend_type != A3 ]]; then
                echo "not support ascend platform"
                log "ERROR" "not support ascend platform"
                exit 1
            fi
            shift
            ;;
        --quiet)
            quiet_flag=y
            msg="user choose quiet mode"
            log "INFO" "${msg}"
            echo  >&2
            shift
            ;;
        -*)
            print "ERROR" "Unsupported parameters: $1"
            exit 1
            ;;
        *)
            if [ "x$1" != "x" ]; then
                print "WARNING" "Unsupported parameters: $1"
                exit 1
            fi
            break
            ;;
        esac
    done
}

function modify_file_permission()
{
    # headers and libs
    local path="$1"
    chmod -R 550 "$path"/device
    chmod -R 550 "$path"/include
    chmod -R 550 "$path"/host
    find "$path/" -type f -name *.h -exec chmod 440 {} +
    find "$path/" -type f -name *.so* -exec chmod 440 {} +
    find "$path/" -type f -name *.a -exec chmod 440 {} +

    find "$path/" -type f -name version.info -exec chmod 440 {} +
    find "$path/" -type f -name filelist.txt -exec chmod 440 {} +

    find "$path/" -type f -name *.py -exec chmod 550 {} +
    find "$path/" -type f -name *.pyc -exec chmod 550 {} +
    find "$path/" -type f -name *.xml -exec chmod 640 {} +
    find "$path/" -type f -name *.json -exec chmod 640 {} +
    find "$path/" -type f -name *.sh -exec chmod 550 {} +
    find "$path/" -type f -name *.run -exec chmod 550 {} +

    chmod 750 {"$path"/ops,"$path"/tools}
    chmod 550 "$path"/script
    chmod 550 "$path"/tools/train
}

function UnTAR()
{
    export GCC_VERSION=$(gcc -dumpversion | cut -c1-2)
    if test x"$GCC_VERSION" = x"4." || test x"$GCC_VERSION" = x"3." || \
        test x"$GCC_VERSION" = x"2." || test x"$GCC_VERSION" = x"1."; then
        if [ "$ascend_type" == "310" ]; then
            tar_package_name="*gcc4.8.5*310.tar.gz"
        elif [ "$ascend_type" = "310P" ]; then
            tar_package_name="*gcc4.8.5*310P.tar.gz"
        elif [ "$ascend_type" = "910B" ]; then
            tar_package_name="*gcc4.8.5*910B.tar.gz"
        elif [ "$ascend_type" = "A3" ]; then
            tar_package_name="*gcc4.8.5*910B.tar.gz"
        fi
    else
        if [ "$ascend_type" == "310" ]; then
            tar_package_name="*gcc7.3.0*310.tar.gz"
        elif [ "$ascend_type" = "310P" ]; then
            tar_package_name="*gcc7.3.0*310P.tar.gz"
        elif [ "$ascend_type" = "910B" ]; then
            tar_package_name="*gcc7.3.0*910B.tar.gz"
        elif [ "$ascend_type" = "A3" ]; then
            tar_package_name="*gcc7.3.0*910B.tar.gz"
        fi
    fi

    tar -xzf $tar_package_name

}

function install_process() {
    log "INFO" "install start"
    log "INFO" "The install path is ${install_path} !"

    if [[ -e "$install_path"/mxIndex/script/uninstall.sh ]]; then
        log "ERROR" "can not install twice, you have already installed Index SDK." "y"
        exit 1
    fi

    if ! mkdir -p "${install_path}/${package_name}"; then
        log "ERROR" "create install dir failed"
        exit 1
    fi

    UnTAR

    cp -rf feature_retrieval/* "${install_path}/${package_name}"
    cp "${tmp_path}"/version.info "${install_path}/${package_name}"
    if ! cd "${install_path}"; then
        log "ERROR" "cd to ${install_path} failed." "y"
        exit 1
    fi

    if [ -e mxIndex ] && [ ! -L mxIndex ];then
        rm "mxIndex"
    fi
    ln -s "${package_name}" "mxIndex"
    cd "${install_path}/${package_name}"
    ! [[ -h lib ]] && ln -sf host/lib lib

    if [ -f filelist.txt ]; then
        rm -f filelist.txt
    fi
    find * | sort > filelist.txt
    sed -i '/filelist.txt/d' filelist.txt
    
    modify_file_permission "${install_path}/${package_name}"

    log "INFO" "Install package successfully" "y"
}

function upgrade_process() {
    log "INFO" "upgrade start"

    # check whether the old version is exist
    if [[ ! -e "$install_path"/mxIndex/script/uninstall.sh ]]; then
        log "ERROR" "There is no Index SDK installed in cur install path, please check it." "y"
        exit 1
    fi

    local doupgrade=n
    if test x"$quiet_flag" = xn; then
       read -t 60 -n1 -re -p "Do you want to upgrade to a newer version provided by this package and the old version will be removed? [Y/N]" answer
       case "${answer}" in
            Y|y)
                doupgrade=y
                ;;
            *)
                doupgrade=n
                ;;
       esac
    else
      doupgrade=y
    fi
    if [[ x"$doupgrade" == "xn" ]]; then
        log  "WARNING" "user reject to upgrade, nothing changed" "y"
        exit 1
    else
        "$install_path"/mxIndex/script/uninstall.sh 2>uninstall_err; res=$?;
        if test "$res" -ne 0; then
            log "ERROR" "uninstall old package failed, ${uninstall_err}" "y"
            exit 1
        fi
        log "INFO" "older version was removed. Installing new version..." "y"
    fi

    install_process

    log "INFO" "upgrade package successfully" "y"
}

function process() {
    if [ "$install_flag" = "y" ]; then
        install_process
    elif [ "$upgrade_flag" = "y" ]; then
        upgrade_process
    fi
}

# 程序开始
function main() {
    parse_script_args "$@"
    check_script_args "$@"
    log_check
    process
}

main "$@"
