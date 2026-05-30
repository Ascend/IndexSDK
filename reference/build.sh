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

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
BUILD_DIR="${SCRIPT_DIR}/build"

list_test_cases() {
    echo "=========================================="
    echo "Available test cases:"
    echo "=========================================="
    local i=1
    for file in "${SCRIPT_DIR}"/Test*.cpp; do
        if [ -f "$file" ]; then
            local name=$(basename "$file" .cpp)
            echo "  [$i] $name"
            ((i++))
        fi
    done
    echo "=========================================="
    echo "Usage:"
    echo "  $0                  Build all test cases"
    echo "  $0 <name>           Build specific test case"
    echo "  $0 <number>         Build test case by number"
    echo ""
    echo "Examples:"
    echo "  $0                              # Build all"
    echo "  $0 TestAscendIndexFlat          # Build by name"
    echo "  $0 1                            # Build by number"
    echo "=========================================="
}

get_test_name_by_number() {
    local num=$1
    local i=1
    for file in "${SCRIPT_DIR}"/Test*.cpp; do
        if [ -f "$file" ]; then
            if [ "$i" -eq "$num" ]; then
                basename "$file" .cpp
                return 0
            fi
            ((i++))
        fi
    done
    return 1
}

if [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
    list_test_cases
    exit 0
fi

if [ -d "$BUILD_DIR" ]; then
    rm -rf "$BUILD_DIR"
fi
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

cmake ..

if [ -z "$1" ]; then
    echo ""
    echo "No test case specified. Building all test cases..."
    echo ""
    make
else
    TARGET="$1"

    if [[ "$TARGET" =~ ^[0-9]+$ ]]; then
        TARGET=$(get_test_name_by_number "$TARGET")
        if [ $? -ne 0 ]; then
            echo "Error: Invalid test case number: $1"
            list_test_cases
            exit 1
        fi
        echo "Selected test case: $TARGET"
    fi

    if [ ! -f "${SCRIPT_DIR}/${TARGET}.cpp" ]; then
        echo "Error: Test case '${TARGET}.cpp' not found"
        list_test_cases
        exit 1
    fi

    echo ""
    echo "Building test case: $TARGET"
    echo ""
    make "$TARGET"
fi

echo ""
echo "Build success!"
echo "Please check env of MX_INDEX_MODELPATH and LD_LIBRARY_PATH before executing the binary file in directory of build"
