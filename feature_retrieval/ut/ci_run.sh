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

CUR_DIR=$(dirname "$(readlink -f "$0")")
if [ ! -d "${CUR_DIR}/../secondparty/" ];then
    mkdir -p "${CUR_DIR}/../secondparty/"
fi
cp -r "${CUR_DIR}/../../huawei_secure_c" "${CUR_DIR}/../secondparty/"
if [ ! -d "${CUR_DIR}/../opensource/" ];then
    mkdir -p "${CUR_DIR}/../opensource/"
fi
cp -r "${CUR_DIR}/../../mockcpp" "${CUR_DIR}/../opensource/"
SECUREC_HOME="${CUR_DIR}"/../secondparty/huawei_secure_c
ASAN_LOG=asan_log

coverage_option="off"
asan_option="off"

function usage() {
    echo "usage:"
    echo "ci_run.sh [-c on/off]"
    echo "-c:coverage on/off default:off"
    exit
}

function set_env() {
    export LD_LIBRARY_PATH=/usr/local/protobuf/lib:"${LD_LIBRARY_PATH}"
    export LD_LIBRARY_PATH=/usr/local/faiss/lib:"${LD_LIBRARY_PATH}"
    export LD_LIBRARY_PATH=/opt/OpenBLAS/lib:"${LD_LIBRARY_PATH}"
    export LD_LIBRARY_PATH="${CUR_DIR}"/../opensource/AscendCLMock/acl/lib:"${LD_LIBRARY_PATH}"
    export LD_LIBRARY_PATH="${CUR_DIR}"/../opensource/AscendCLMock/securec/lib:"${LD_LIBRARY_PATH}"
}

function build_opensource() {
    pushd ./
    cd "${CUR_DIR}"/../opensource/AscendCLMock/"${1}"
    if [ -d "build" ]; then
        rm -rf build
    fi

    if [ -d "lib" ]; then
        rm -rf lib
    fi

    mkdir build
    mkdir lib

    cd build
    cmake ../ -DSECUREC_HOME="${SECUREC_HOME}"
    make clean
    make -j
    make install
    popd
}

function build_mockcpp() {
    pushd ./
    cd "${CUR_DIR}"/../opensource/mockcpp
    if [ -d "output" ]; then
        echo "skip build mockcpp"
        popd
        return
    fi

    if [ -d "build" ]; then
        rm -rf build
    fi

    mkdir build
    mkdir output

    cd build
    cmake ../ -DCMAKE_INSTALL_PREFIX=../output
    make clean
    make -j
    make install
    popd
}

function build_securec() {
    pushd ./
    cd "${SECUREC_HOME}"
    make CC=gcc
    cd lib
    ln -s libboundscheck.so libsecurec.so
    popd
}

function gen_report() {
    cd "${CUR_DIR}"
    mkdir -p ./output/coverage/summary
    mkdir -p ./output/coverage/report
    
    cd ./build
    echo "========= Testing is running pls wait ========="
    make

    if [ "${1}" == "on" ]; then
        GCC_MAJOR_VERSION=$(gcc -dumpversion | awk '{print $1}')
        GET_ARCH=`arch`
        ASAN_SO=/usr/lib/gcc/$GET_ARCH-linux-gnu/${GCC_MAJOR_VERSION}/libasan.so
        export ASAN_OPTIONS=halt_on_error=0:log_path=${CUR_DIR}/build/${ASAN_LOG}
        LD_PRELOAD=${ASAN_SO} ./TestAscendIndexUT --gtest_output=xml:./test_detail.xml

        if [ -f ${CUR_DIR}/build/${ASAN_LOG}.* ]; then
            echo "asan found error, please check asan log"
            exit 1
        fi
    else
        ctest -V -R TestAscendIndexUT
    fi

    echo "========= Testing finish ========="
    cd ../

    echo "========= LCOV html is collecting ... ===="
    cp ./build/test_detail.xml ./output/coverage/report

    lcov --rc lcov_excl_br_line='(ASCEND_THROW_.*|FAISS_THROW_.*|APP_LOG.*|APPERR_RETURN_.*)' \
         --rc lcov_branch_coverage=1 -c -d ./ --filter branch -o ./output/coverage/summary/total.info \
          --ignore-errors inconsistent --ignore-errors mismatch
    lcov -r ./output/coverage/summary/total.info '*.hpp' '*.inl' '*ut/*' '/usr/include/*' '/usr/local/*' '*opensource*' \
    '*/src/faiss/ascend/*.h' '*/src/faiss/ascend/custom/impl/*.h' '*/ascend/custom/impl/*.h' '*/ascenddaemon/*.h' \
    --rc lcov_branch_coverage=1 -o ./output/coverage/summary/total.info --ignore-errors inconsistent \
         --ignore-errors unused
    genhtml --branch-coverage -o ./output/coverage/report/total ./output/coverage/summary/total.info \
     --ignore-errors inconsistent  --ignore-errors corrupt

    find ./ -type f -name "*.gcda" | xargs rm
    find ./ -type f -name "*.gcno" | xargs rm
    echo "========= LCOV html is collect successfully. ===="
}

function build_uint_test() {
    echo "BuildUniTest"

    [ ! -c "/dev/davinci_manager" ] && mknod "/dev/davinci_manager" c 237 0

    if [ -d "build" ]; then
        rm -rf build
    fi

    if [ -d "output" ]; then
        rm -rf output
    fi

    if [ -d "${CUR_DIR}/../src/faiss" ]; then
        rm -rf "${CUR_DIR}"/../src/faiss
    fi

    ln -s "${CUR_DIR}"/../src/ascendfaiss "${CUR_DIR}"/../src/faiss

    mkdir build
    mkdir output

    cd ./build

    cmake  "${CUR_DIR}"/../src/ascendfaiss/ut \
        -DCMAKE_INSTALL_PREFIX="../output" \
        -DUT_COVERAGE="${coverage_option}" \
        -DASAN_OPTION="${asan_option}" \
        -DSECUREC_HOME="${SECUREC_HOME}"

    make clean
    make -j
    make install
}

while getopts 'c:a:' OPT; do
    case "$OPT" in
        c)
        coverage_option="$OPTARG";;
        a)
        asan_option="$OPTARG";;
        h)
        usage;;
    esac
done

set_env
build_securec
build_mockcpp
build_opensource acl
build_uint_test

if [ "${coverage_option}" == "on" ]; then
    gen_report ${asan_option}
fi

CUR_DIR=$(dirname "$(readlink -f "$0")")
