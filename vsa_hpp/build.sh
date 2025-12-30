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
CURRENT_PATH=$(cd "$(dirname "$0")"; pwd)

function print_help_info()
{
    Magenta="\e[32m"
    ENDCOLOR="\e[0m"
    echo "$0 options"
    echo "options"
    echo -e "${Magenta}    -t <release|debug|clean>${ENDCOLOR}  BuildThe default value is 'release'"
    echo -e "${Magenta}         relase${ENDCOLOR}               Build in release mode"
    echo -e "${Magenta}         debug${ENDCOLOR}                Build in debug mode"
    echo -e "${Magenta}         clean${ENDCOLOR}                Deleting Temporary Files Generated During Build"
    echo -e "${Magenta}    -c <on|off>${ENDCOLOR}               Instrument code coverage, default is no instrumentation when not specified, [default: 'off']"
    echo -e "${Magenta}    -s <on|off>${ENDCOLOR}               Address sanitizer, default is on when not specified, [default: 'on']"
    echo -e "${Magenta}    --enable_submodule_update${ENDCOLOR} git submodule update --init --recursive --remote"
    echo -e "${Magenta}    -h|--help${ENDCOLOR}                 Show this message"
    echo "Example:"
    echo "  $0"
    echo "  $0 -t release"
    echo "  $0 -t release -v on"
    echo "  $0 -t debug"
    echo "  $0 -t debug -c ON -S ON"
    echo "  $0 -t clean"
}

function build_all()
{
    cd "${CURRENT_PATH}"
    if ! bash buildscript/build_3rdparty.sh "$CMAKE_BUILD_TYPE" "$COMPILE_VERIFICATION" "$USING_COVERAGE"; then
        echo "Failed to build_3rdparty" 1>&2
        return 1
    fi
    if ! bash buildscript/build_src.sh "$CMAKE_BUILD_TYPE" "$USING_COVERAGE" "$USING_XSCAN" "$COMPILE_VERIFICATION"; then
        echo "Failed to build_src" 1>&2
        return 1
    fi
    mkdir -p "${CURRENT_PATH}/output"
}
function read_input_args()
{
    COMPILE_VERIFICATION="FALSE"
    if [ "$1" = "" ] || [ "$1" = "-t" ]; then
        BUILD_TYPE="RELEASE"
    fi
    ENABLE_SUBMODULE_UPDATE="False"
    set -- $(getopt -o h::t:c:s:m: --long xsan:,isa:,os:,compiler:,instrument:,entry-point:,help,ENABLE_SUBMODULE_UPDATE="False" -n "$0" -- "$@")
    while true
    do
        case "$1" in
            -t)
                BUILD_TYPE=$(echo "$2"|tr a-z A-Z|tr -d "'")
                shift
                ;;
            -c)
                USING_COVERAGE=$(echo "$2"|tr a-z A-Z|tr -d "'")
                shift
                ;;
            -h|--help)
                BUILD_TYPE="-H"
                shift
                break
                ;;
            --enable_submodule_update)
                ENABLE_SUBMODULE_UPDATE="True"
                shift
                break
                ;;
            -s)
                USING_XSCAN=$(echo "$2"|tr a-z A-Z|tr -d "'")
                shift
                ;;
            --)
                if [ "$BUILD_TYPE" = "" ]; then
                    BUILD_TYPE="UNKNOWN"
                fi
                shift
                break
                ;;
            *)
                if [ "$BUILD_TYPE" = "" ]; then
                    BUILD_TYPE="UNKNOWN"
                fi
                break
                ;;
        esac
        shift
    done
}
function clean_build_files()
{
    source "${CURRENT_PATH}/buildscript/build_env.sh"
    rm -rf "${VSA_ROOT_PATH}/output"
    mkdir -p "${VSA_ROOT_PATH}/output"
    touch "${VSA_ROOT_PATH}/output/.gitkeep"
    rm -rf "${VSA_ROOT_PATH}/install"
    mkdir -p "${VSA_ROOT_PATH}/install"
    touch "${VSA_ROOT_PATH}/install/.gitkeep"
    rm -rf "${VSA_ROOT_PATH:?}/build"
    mkdir -p "${VSA_ROOT_PATH}/build"
    touch "${VSA_ROOT_PATH}/build/.gitkeep"
    echo "Clean succeed!"
}
function main()
{
    read_input_args $@
    echo "BUILD_TYPE=${BUILD_TYPE}"
    echo "COMPILE_VERIFICATION=${COMPILE_VERIFICATION}"
    if [ "${BUILD_TYPE}" = "RELEASE" ] || [ "${BUILD_TYPE}" = "" ]; then
        CMAKE_BUILD_TYPE="Release"
    elif [ "${BUILD_TYPE}" = "DEBUG" ]; then
        CMAKE_BUILD_TYPE="Debug"
    elif [ "${BUILD_TYPE}" = "-H" ] || [ "${BUILD_TYPE}" = "--HELP" ]; then
        print_help_info
        exit
    elif [ "${BUILD_TYPE}" = "CLEAN" ]; then
        clean_build_files
        exit
    else
        print_help_info
        exit 1
    fi
    if [ "${ENABLE_SUBMODULE_UPDATE}" = "True" ]; then
        git submodule update --init --recursive --remote
    fi
    if ! build_all; then
        exit 1
    fi
}

main "$@"
exit $?