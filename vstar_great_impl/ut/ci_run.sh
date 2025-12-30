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

find ./ -name "*.sh" -exec dos2unix {} \;
find ./ -name "*.sh" -exec chmod +x {} \;

asan_option="off" # option to turn on address sanitizer for potential memory-releated bugs
use_local_option="off" # option to use local directories and env variables for development, as opposed to those of real ci environments

if [ "${use_local_option}" == "on" ]; then
    source ./use_local.sh
else
    ASCEND_MOCK_DIR=/usr1/opensource/AscendCLMock/
    SECUREC_DIR=/usr1/platform/huawei_secure_c/
    MOCKCPP_DIR=/usr1/opensource/mockcpp/mockcpp/
fi
GTEST_DIR=/usr/local/gtest/
GTEST_INSTALL_DIR=/usr1/opensource/googletest/

ASAN_LOG_DIR=./output/

function modify_source_code_for_ut() {
    sed -i '/if (!ASCEND_310_SOC) {/,/^    }$/d' ../mix-index/src/npu/common/utils/MemorySpace.cpp
    # Remove the limit on the base library size
    sed -i '/ASCEND_THROW_IF_NOT_FMT((NTOTAL_MIN <= n) && (n <= NTOTAL_MAX),/,/input base size.*NTOTAL_MAX);/d' ../mix-index/src/npu/IndexGreat.cpp
    sed -i '/ASCEND_THROW_IF_NOT_FMT((NTOTAL_MIN <= n) && (n <= NTOTAL_MAX),/,/input base size.*NTOTAL_MAX);/d' ../mix-index/src/npu/NpuIndexVStar.cpp
    # Remove the limit on the number of queries
    sed -i '/ASCEND_THROW_IF_NOT_MSG(params\.queryData\.size() >= params\.n \* static_cast<size_t>(dim)/,/Queries vector must have at least n \* dim elements\.\\n");/d' ../mix-index/src/npu/IndexGreat.cpp
    sed -i '/ASCEND_THROW_IF_NOT_MSG(params\.queryData\.size() >= params\.n \* static_cast<size_t>(dim)/,/Queries vector must have at least n \* dim elements\.\\n");/d' ../mix-index/src/npu/NpuIndexVStar.cpp

    # Remove mltisearch related code : The operator stub library does not simulate the dependency functions required by multisearch, causing this feature to fail operation.
    sed -i '/std::vector<std::future<bool> > results;/,/^    }$/d' ../mix-index/src/npu/NpuIndexIVFHSP.cpp
    sed -i '/\/\/ data flow: labelL2Cpu (n \* nProbeL2) ->/{:a;N;/ACL_REQUIRE_OK(ret);/!ba;d}' ../mix-index/src/npu/NpuIndexIVFHSP.cpp
}

function set_env() {
    export LD_LIBRARY_PATH=${ASCEND_MOCK_DIR}/acl/lib:$LD_LIBRARY_PATH
    if [ "${use_local_option}" == "off" ]; then
        export LD_LIBRARY_PATH=/opt/OpenBLAS/lib:$LD_LIBRARY_PATH
        export LD_LIBRARY_PATH=/usr/local/faiss/faiss1.10.0/lib:$LD_LIBRARY_PATH
        export LD_LIBRARY_PATH=/usr/local/gtest/lib:$LD_LIBRARY_PATH
        export LD_LIBRARY_PATH=/usr1/vstar_great_impl/ut/output/lib:$LD_LIBRARY_PATH
    fi
    # this variable prevents the SocUtil instance from calling aclFinalize();
    # because within the stub method, the m_activeContexts variable, upon program termination,
    # is first deallocated by the vector destructor，but aclFinalize() calls ClearAll() which tries to
    # clear m_activeContexts' content again, causing double free
    export MX_INDEX_FINALIZE=0
}

function build_gtest() {
    pushd ./
    cd $GTEST_INSTALL_DIR/
    if [ -d "build" ]; then
        rm -rf build
    fi
    mkdir build
    cd build
    cmake .. -DCMAKE_INSTALL_PREFIX=/usr/local
    make -j
    popd
}

function build_securec() {
    pushd ./
    cd $SECUREC_DIR
    rm -rf lib
    cd src
    make -j
    popd
}

function build_ascendmock() {
    pushd ./
    cd $ASCEND_MOCK_DIR/$1
    if [ -d "build" ]; then
        rm -rf build
    fi
    if [ -d "lib" ]; then
        rm -rf lib
    fi
    mkdir build
    mkdir lib
    cd build
    cmake ../ -DSECUREC_HOME="${SECUREC_DIR}"
    make clean
    make -j
    make install
    popd
}

function build_mockcpp() {
    pushd ./
    cd ${MOCKCPP_DIR}
    if [ -d "build" ]; then
        rm -rf build
    fi
    if [ -d "output" ]; then
        rm -rf output
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

function build_unit_test() {
    pushd ./

    [ ! -c "/dev/davinci_manager" ] && mknod "/dev/davinci_manager" c 237 0

    if [ -d "build" ]; then
        rm -rf build
    fi

    if [ -d "output" ]; then
        rm -rf output
    fi

    mkdir build
    mkdir output

    cd ./build

    echo ACL MOCK DIR = "${ASCEND_MOCK_DIR}"

    cmake  .. \
        -DCMAKE_INSTALL_PREFIX="../output" \
        -DSECUREC_DIR="${SECUREC_DIR}" \
        -DACLMOCK_DIR="${ASCEND_MOCK_DIR}" \
        -DMOCKCPP_DIR="${MOCKCPP_DIR}" \
        -DGTEST_DIR="${GTEST_DIR}" \
        -DASAN_OPTION="${asan_option}" \
        -DUSE_LOCAL_OPTION="${use_local_option}"

    make clean
    make -j
    make install
    popd
}

function gen_report() {
    mkdir -p ./output/coverage/summary
    mkdir -p ./output/coverage/report
    
    cd ./build
    echo "========= Testing is running pls wait ========="
    make

    if [ "${asan_option}" == "on" ]; then
        echo "Running with ASAN turned on..."
        GCC_MAJOR_VERSION=$(gcc -dumpversion | awk '{print $1}')
        GET_ARCH=`arch`
        ASAN_SO=/usr/lib/gcc/$GET_ARCH-redhat-linux/${GCC_MAJOR_VERSION}/libasan.so
        export ASAN_OPTIONS=halt_on_error=1:log_path=${ASAN_LOG_DIR}
        LD_PRELOAD=${ASAN_SO} ./TestAscendIndexUT --gtest_output=xml:./test_detail.xml
    else
        ctest -V -R TestAscendIndexUT
    fi

    echo "========= Testing finish ========="
    cd ../

    echo "========= LCOV html is collecting ... ===="
    cp ./build/test_detail.xml ./output/coverage/report

    lcov --rc lcov_excl_br_line='(ASCEND_THROW_.*|ASCENDSEARCH_THROW_.*|FAISS_THROW_.*|APP_LOG.*|APPERR_RETURN_.*)' \
         --rc lcov_branch_coverage=1 -c -d ./ --filter branch -o ./output/coverage/summary/total.info \
          --ignore-errors inconsistent --ignore-errors mismatch
    lcov -r ./output/coverage/summary/total.info \
        '*.hpp' \
        '*.inl' \
        '*ut/*' \
        '/usr/include/*' \
        '/usr/local/*' \
        '/usr/lib/*' \
        '*opensource*' \
        '*/include/impl/*.h' \
        '*/include/npu/*.h' \
        '*/include/npu/common/*.h' \
        '*/include/npu/common/threadpool/*.h' \
        '*/include/npu/common/utils/*.h' \
        '*/include/utils/*.h' \
        '*/third_party/include/impl/*.h' \
        '*/third_party/include/utils/*.h' \
        '*/src/npu/common/utils/*' \
        '*/src/utils/*' \
        '*/third_party/src/*' \
    --rc lcov_branch_coverage=1 -o ./output/coverage/summary/total.info --ignore-errors inconsistent \
         --ignore-errors unused --ignore-errors empty
    genhtml --branch-coverage -o ./output/coverage/report/total ./output/coverage/summary/total.info \
     --ignore-errors inconsistent  --ignore-errors corrupt --ignore-errors empty

    find ./ -type f -name "*.gcda" | xargs rm
    find ./ -type f -name "*.gcno" | xargs rm
    echo "========= LCOV html is collect successfully. ===="
}

modify_source_code_for_ut

if [ "${use_local_option}" == "off" ]; then
    build_gtest
fi
build_securec
build_mockcpp
build_ascendmock acl
set_env
build_unit_test
gen_report