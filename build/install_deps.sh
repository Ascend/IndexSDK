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
readonly FAISS_INSTALL_PATH="/usr/local/faiss"

# ============== 1. install OpenBLAS ==============
echo "[INSTALL_INFO] start installing OpenBLAS..."

if [ -d "/opt/OpenBLAS" ]; then
    echo "[INSTALL_INFO] OpenBLAS already exist."
else
    cd ${TOP_DIR}
    if [ -f "OpenBLAS-0.3.10.tar.gz" ]; then
        echo "[INSTALL_INFO] Using local cached OpenBLAS-0.3.10.tar.gz"
    else
        wget https://github.com/OpenMathLib/OpenBLAS/archive/v0.3.10.tar.gz -O OpenBLAS-0.3.10.tar.gz
    fi
    tar -xf OpenBLAS-0.3.10.tar.gz && cd OpenBLAS-0.3.10
    make FC=gfortran USE_OPENMP=1 -j && make install
    ln -sf /opt/OpenBLAS/lib/libopenblas.so /usr/lib/libopenblas.so
    cd .. && rm -rf OpenBLAS-0.3.10.tar.gz OpenBLAS-0.3.10
fi

# ============== 2. install faiss ==============
echo "[INSTALL_INFO] start installing faiss..."

if [ -d "/usr/local/faiss" ]; then
    echo "[INSTALL_INFO] faiss already exist."
else
    cd ${TOP_DIR}
    if [ -f "faiss-1.10.0.tar.gz" ]; then
        echo "[INSTALL_INFO] Using local cached faiss-1.10.0.tar.gz"
    else
        wget https://github.com/facebookresearch/faiss/archive/v1.10.0.tar.gz -O faiss-1.10.0.tar.gz
    fi
    tar -xf faiss-1.10.0.tar.gz && cd faiss-1.10.0/faiss
    sed -i "149 i virtual void search_with_filter (idx_t n, const float *x, idx_t k, float *distances, idx_t *labels, const void *mask = nullptr) const {}" Index.h
    sed -i "49 i template <typename IndexT> IndexIDMapTemplate<IndexT>::IndexIDMapTemplate (IndexT *index, std::vector<idx_t> &ids): index (index), own_fields (false) {this->is_trained = index->is_trained; this->metric_type = index->metric_type; this->verbose = index->verbose; this->d = index->d; id_map = ids;}" IndexIDMap.cpp
    sed -i "30 i explicit IndexIDMapTemplate (IndexT *index, std::vector<idx_t> &ids); " IndexIDMap.h
    sed -i "217 i utils/sorting.h" CMakeLists.txt
    cd ..
    cmake -B build . -DFAISS_ENABLE_GPU=OFF -DFAISS_ENABLE_PYTHON=OFF -DBUILD_TESTING=OFF -DBUILD_SHARED_LIBS=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=${FAISS_INSTALL_PATH}
    cd build && make -j && make install
    cd ../.. && rm -f faiss-1.10.0.tar.gz && rm -rf faiss-1.10.0
fi
if [ ! -f "/usr/local/lib/libfaiss.so" ]; then
    cp /usr/local/faiss/lib/libfaiss.so /usr/local/lib
fi

# ============== 3. clone makeself ==============
function install_package_deps()
{
    echo "[INSTALL_INFO] start cloning makeself..."

    if [ -d "${TOP_DIR}/makeself" ]; then
        echo "[INSTALL_INFO] makeself already exist."
    else
        cd ${TOP_DIR}
        git clone -b v2.5.0.x https://gitcode.com/cann-src-third-party/makeself.git makeself_patch
        git clone -b release-2.5.0 https://gitcode.com/gh_mirrors/ma/makeself.git
    fi

    MAKESELF_HEADER="${TOP_DIR}/makeself/makeself-header.sh"
    if grep -A1 'cat << EOF  > "\$archname"' "$MAKESELF_HEADER" | tail -n1 | grep -q '^#!/bin/bash$'; then
        echo "[INSTALL_INFO] makeself patch already applied."
    else
        echo "[INSTALL_INFO] start patching makeself..."
        cd "${TOP_DIR}/makeself" && patch -p1 < ../makeself_patch/makeself-2.5.0.patch
    fi
}

# ============== 4. clone ut deps ==============
function install_ut_deps()
{
    echo "[INSTALL_INFO] start cloning ut deps..."

    # mockcpp
    if [ ! -d "${TOP_DIR}/mockcpp" ]; then
        cd ${TOP_DIR}
        git clone -b v2.7.x-h3 https://gitcode.com/cann-src-third-party/mockcpp.git mockcpp_patch
        git clone -b v2.7 https://gitee.com/sinojelly/mockcpp.git
    fi
    CMAKELISTS="${TOP_DIR}/mockcpp/CMakeLists.txt"
    if [ ! -f "$CMAKELISTS" ]; then
        echo "[INSTALL_ERROR]: $CMAKELISTS not found!"
        exit 1
    fi
    if grep -q '_GLIBCXX_USE_CXX11_ABI=1' "$CMAKELISTS"; then
        echo "[INSTALL_INFO] mockcpp patch already applied, _GLIBCXX_USE_CXX11_ABI already set to 1."
    else 
        cd "${TOP_DIR}/mockcpp" && patch -p1 < ../mockcpp_patch/mockcpp-2.7_py3-h3.patch
        sed -i '/target_compile_definitions(mockcpp PRIVATE/,/)/s/_GLIBCXX_USE_CXX11_ABI=0/_GLIBCXX_USE_CXX11_ABI=1/' "$CMAKELISTS"
        echo "[INSTALL_INFO] successfully updated _GLIBCXX_USE_CXX11_ABI from 0 to 1 in $CMAKELISTS"
    fi

    # huawei_secure_c
    if [ ! -d "${TOP_DIR}/huawei_secure_c" ]; then
        cd ${TOP_DIR}
        git clone -b v1.1.16 https://gitee.com/openeuler/libboundscheck.git huawei_secure_c
    fi

    # googletest
    if [ ! -d "${TOP_DIR}/googletest" ]; then
        cd ${TOP_DIR}
        git clone -b release-1.11.0 https://gitcode.com/GitHub_Trending/go/googletest.git googletest
    fi
    cd "${TOP_DIR}/googletest"
    cmake -DBUILD_SHARED_LIBS=ON -DCMAKE_INSTALL_PREFIX="${TOP_DIR}/gtest_install" -DCMAKE_MAKE_PROGRAM=make .
    make -j
    make install

    # lcov2.0
    if ! lcov --version 2>/dev/null | grep -q '2\.0'; then
        echo "[INSTALL_INFO] lcov 2.0 not found. Installing..."
        cd "${TOP_DIR}"
        apt update && apt install -y libcapture-tiny-perl libdatetime-perl libtimedate-perl
        wget https://github.com/linux-test-project/lcov/releases/download/v2.0/lcov-2.0.tar.gz
        tar -xzf lcov-2.0.tar.gz && cd lcov-2.0
        make install
        cd .. && rm -rf lcov-2.0 lcov-2.0.tar.gz
    else
        echo "[INSTALL_INFO] lcov 2.0 is already installed."
    fi
}

BUILDTYPE=$1
if [ "${BUILDTYPE}" = "ut" ]; then
    install_ut_deps
else
    install_package_deps
fi
