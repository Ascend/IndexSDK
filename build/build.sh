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

readonly CUR_DIR=$(dirname "$(readlink -f "$0")")
readonly TOP_DIR=${CUR_DIR}/..
readonly MAKESELF_SCRIPT=${TOP_DIR}/makeself/makeself.sh
readonly MAKESELF_HEADER=${TOP_DIR}/makeself/makeself-header.sh
readonly Gcc4_PATH="/opt/rh/devtoolset-7/root/usr/bin"
readonly Gcc7_PATH="/opt/rh/devtoolset-index/root/usr/bin"

cd "${TOP_DIR}"
find ./ -name "*.sh" -exec dos2unix {} \;
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver:$LD_LIBRARY_PATH

function set_env()
{
    export CC=$GCC_HOME/bin/gcc
    export CXX=$GCC_HOME/bin/g++
    export PATH=$GCC_HOME/bin:/usr/local/openmpi/bin:$PATH
    export LD_LIBRARY_PATH=/usr/local/openmpi/lib:$LD_LIBRARY_PATH
}

function build_Retrieval()
{
    local gcc_type="$1"
    cd "${TOP_DIR}" && cp -rf "vstar_great_impl/mix-index/publish_union.sh" ./
    chmod +x publish_union.sh
    ./publish_union.sh "${TOP_DIR}/vstar_great_impl" "${gcc_type}"
    rm -f "${TOP_DIR}/publish_union.sh"
    mkdir -p "${TOP_DIR}/build/output"
    mv "${TOP_DIR}"/Ascend-ascendsearch* "${TOP_DIR}/build/output"
    echo "build Retrieval success"
}

function build_vsa()
{
    export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64/driver:$LD_LIBRARY_PATH
    cd "${TOP_DIR}/vsa_hpp"
    chmod +x build.sh && ./build.sh
    echo "build vsa_hpp success"
}

function build_feature_retrieval()
{
    local gcc_type="$1"
    if [ ! -d "${TOP_DIR}/feature_retrieval/src/ascendfaiss/ops/cmake/util/makeself" ]; then
        mkdir -p "${TOP_DIR}/feature_retrieval/src/ascendfaiss/ops/cmake/util/makeself"
    fi
    cp -rf "${MAKESELF_SCRIPT}" "${MAKESELF_HEADER}" "${TOP_DIR}/feature_retrieval/src/ascendfaiss/ops/cmake/util/makeself/"

    if [ ! -d "${TOP_DIR}/feature_retrieval/secondparty" ]; then
        mkdir -p "${TOP_DIR}/feature_retrieval/secondparty"
    fi
    cp -rf "${TOP_DIR}"/build/output/Ascend-ascendsearch*.tar.gz "${TOP_DIR}/feature_retrieval/secondparty/"
    cp -rf "${TOP_DIR}/vsa_hpp/install/"* "${TOP_DIR}/feature_retrieval/secondparty/"

    cd "${TOP_DIR}/feature_retrieval/build" && bash build.sh "${gcc_type}"
    echo "build feature_retrieval success"
}

function run_ut()
{
    rm -rf "${TOP_DIR}/feature_retrieval/opensource/mockcpp"
    rm -rf "${TOP_DIR}/feature_retrieval/secondparty/huawei_secure_c"
    cd "${TOP_DIR}/feature_retrieval/ut"
    chmod +x *.sh
    ./config_env.sh && ./ci_run.sh -c on -a on
}

function clean()
{
    set +e
    cd "${TOP_DIR}/vsa_hpp" && chmod +x build.sh && ./build.sh -t clean
    cd "${TOP_DIR}/vsa_hpp/3rdparty/hcps" && chmod +x build.sh && ./build.sh -t clean
    cd "${TOP_DIR}/vsa_hpp/3rdparty/hcps/3rdparty/memory_bridge" && chmod +x build.sh && ./build.sh -t clean
    rm -rf "${TOP_DIR}/build/output"
    rm -rf "${TOP_DIR}/feature_retrieval/output"
    rm -rf "${TOP_DIR}/feature_retrieval"/build_gcc*
    rm -rf "${TOP_DIR}/feature_retrieval/secondparty"
    set -e
}

function package()
{
    bash "${CUR_DIR}/package.sh"
}

function prepare()
{
    if [ "${BUILDTYPE}" = "ut" ]; then
        echo "building googletest..."
        cd "${TOP_DIR}/googletest"
        cmake -DBUILD_SHARED_LIBS=ON -DCMAKE_INSTALL_PREFIX="${TOP_DIR}/gtest_install" -DCMAKE_MAKE_PROGRAM=make .
        make -j
        make install

        cd "${TOP_DIR}/mockcpp" && patch -p1 < ../mockcpp_patch/mockcpp-2.7_py3-h3.patch
        CMAKELISTS="${TOP_DIR}/mockcpp/CMakeLists.txt"
        if [ ! -f "$CMAKELISTS" ]; then
            echo "Error: $CMAKELISTS not found!"
            exit 1
        fi
        sed -i '/target_compile_definitions(mockcpp PRIVATE/,/)/s/_GLIBCXX_USE_CXX11_ABI=0/_GLIBCXX_USE_CXX11_ABI=1/' "$CMAKELISTS"
        echo "Successfully updated _GLIBCXX_USE_CXX11_ABI from 0 to 1 in $CMAKELISTS"
    else
        if grep -A1 'cat << EOF  > "\$archname"' "$MAKESELF_HEADER" | tail -n1 | grep -q '^#!/bin/bash$'; then
            echo "makeself patch already applied."
        else
            echo "start patching makeself..."
            cd "${TOP_DIR}/makeself" && patch -p1 < ../makeself_patch/makeself-2.5.0.patch
        fi
    fi
}

function main()
{
    clean
    if [ -d "${Gcc4_PATH}" ]; then
        echo "build with Gcc4 in ${Gcc4_PATH}..."
        export GCC_HOME=/opt/rh/devtoolset-7/root/usr
        set_env
        build_Retrieval Gcc4
        build_vsa
        build_feature_retrieval Gcc4
    fi
    if [ -d "${Gcc7_PATH}" ]; then
        echo "build with Gcc7 in ${Gcc7_PATH}..."
        export GCC_HOME=/opt/rh/devtoolset-index/root/usr
        set_env
        build_Retrieval Gcc7
        build_vsa
        build_feature_retrieval Gcc7
    fi
    if [ ! -d "${Gcc4_PATH}" ] && [ ! -d "${Gcc7_PATH}" ]; then
        echo "build with system default Gcc..."
        build_Retrieval Gcc7
        build_vsa
        build_feature_retrieval Gcc7
    fi
    package
}

BUILDTYPE=$1
prepare
if [ "${BUILDTYPE}" = "ut" ]; then
    run_ut
else
    main
fi
