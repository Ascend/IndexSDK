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

readonly ARCH=$(uname -m)
readonly NPU_MODEL=$(npu-smi info | awk 'NR==7 {print $3}')
readonly CUR_DIR=$(dirname "$(readlink -f "$0")")
readonly INDEXSDK_VERSION="26.0.0"
readonly INDEXSDK_REPO_URL="https://gitcode.com/Ascend/IndexSDK/releases/download/v${INDEXSDK_VERSION}"
OPENBLAS_INSTALL_PATH="/opt/OpenBLAS"
FAISS_INSTALL_PATH="/usr/local/faiss"
MX_INDEX_INSTALL_PATH="/usr/local/Ascend"

if [ "${ARCH}" = "aarch64" ]; then
    CMAKE_SCRIPT="cmake-3.24.0-linux-aarch64.sh"
    INDEXSDK_PKG="Ascend-mindxsdk-mxindex_${INDEXSDK_VERSION}_linux-aarch64.run"
elif [ "${ARCH}" = "x86_64" ]; then
    CMAKE_SCRIPT="cmake-3.24.0-linux-x86_64.sh"
    INDEXSDK_PKG="Ascend-mindxsdk-mxindex_${INDEXSDK_VERSION}_linux-x86_64.run"
else
    echo "[ERROR] Unsupported architecture: ${ARCH}"
    exit 1
fi

if [[ "${NPU_MODEL}" == 310P* ]]; then
    readonly PLATFORM="310P"
elif [[ "${NPU_MODEL}" == 910B* ]]; then
    readonly PLATFORM="910B"
else
    readonly PLATFORM="A3"
fi

echo "[INFO] Start deploying for ${PLATFORM} on ${ARCH}..."

# ============== 1. 安装系统依赖 ==============
echo "[INFO] Installing system dependencies..."
apt-get update
apt-get install -y wget gfortran openmpi-bin libopenmpi-dev gcc g++ make pkg-config libblas-dev liblapack-dev libffi-dev libssl-dev zlib1g-dev

python3 -m pip install --upgrade pip
pip3 install numpy==1.26.4 decorator sympy==1.4 cffi==1.15.1 pyyaml pathlib2 protobuf scipy requests attrs psutil faiss-cpu==1.13.2 \
     -i https://repo.huaweicloud.com/repository/pypi/simple

# 检查 CMake 版本，低于 3.24.0 则安装
echo "[INFO] Checking cmake installation..."

if ! command -v cmake &> /dev/null; then
    NEED_INSTALL_CMAKE=1
else
    CURRENT_VERSION=$(cmake --version | head -n1 | grep -oP '\d+\.\d+\.\d+')
    REQUIRED_VERSION="3.24.0"
    if ! dpkg --compare-versions "${CURRENT_VERSION}" ge "${REQUIRED_VERSION}"; then
        NEED_INSTALL_CMAKE=1
        echo "[INFO] Current cmake version (${CURRENT_VERSION}) is lower than required (${REQUIRED_VERSION}). Installing..."
    else
        NEED_INSTALL_CMAKE=0
        echo "[INFO] CMake version (${CURRENT_VERSION}) meets the requirement. Skipping installation."
    fi
fi
if [ "${NEED_INSTALL_CMAKE}" = "1" ]; then
    cd "${CUR_DIR}"
    if [ -f "${CMAKE_SCRIPT}" ]; then
        echo "[INFO] Using cached cmake installer"
    else
        echo "[INFO] Downloading cmake installer..."
        wget https://github.com/Kitware/CMake/releases/download/v3.24.0/${CMAKE_SCRIPT}
    fi
    echo "[INFO] Installing cmake..."
    echo -e "y\nn" | bash ./${CMAKE_SCRIPT} --skip-license --prefix=/usr
fi

# ============== 2. 安装 OpenBLAS ==============
echo "[INFO] Start installing OpenBLAS..."

if [ -d "${OPENBLAS_INSTALL_PATH}" ]; then
    echo "[WARN] OpenBLAS already exists at ${OPENBLAS_INSTALL_PATH}"
    echo "[WARN] Faiss depends on OpenBLAS, recompile Faiss if OpenBLAS upgraded"
    read -p "[INFO] Press Enter to use default path (may overwrite), or enter custom path: " CUSTOM_OPENBLAS_PATH
    if [ -n "${CUSTOM_OPENBLAS_PATH}" ]; then
        OPENBLAS_INSTALL_PATH="${CUSTOM_OPENBLAS_PATH}"
    fi
fi

echo "[INFO] Installing OpenBLAS to ${OPENBLAS_INSTALL_PATH}..."

cd "${CUR_DIR}"
if [ -f "OpenBLAS-0.3.10.tar.gz" ]; then
    echo "[INFO] Using cached OpenBLAS archive"
else
    echo "[INFO] Downloading OpenBLAS..."
    wget https://github.com/OpenMathLib/OpenBLAS/archive/v0.3.10.tar.gz -O OpenBLAS-0.3.10.tar.gz
fi
tar -xf OpenBLAS-0.3.10.tar.gz && cd OpenBLAS-0.3.10
echo "[INFO] Compiling and installing OpenBLAS..."
make FC=gfortran USE_OPENMP=1 -j$(nproc) && make PREFIX=${OPENBLAS_INSTALL_PATH} install
cd .. && rm -rf OpenBLAS-0.3.10

echo "export LD_LIBRARY_PATH=${OPENBLAS_INSTALL_PATH}/lib:\$LD_LIBRARY_PATH" >> /etc/profile
export LD_LIBRARY_PATH=${OPENBLAS_INSTALL_PATH}/lib:$LD_LIBRARY_PATH

# ============== 3. 安装 faiss ==============
echo "[INFO] Start installing faiss..."

if [ -d "${FAISS_INSTALL_PATH}" ]; then
    echo "[WARN] Faiss already exists at ${FAISS_INSTALL_PATH}"
    read -p "[INFO] Press Enter to use default path (may overwrite), or enter custom path: " CUSTOM_FAISS_PATH
    if [ -n "${CUSTOM_FAISS_PATH}" ]; then
        FAISS_INSTALL_PATH="${CUSTOM_FAISS_PATH}"
    fi
fi

echo "[INFO] Installing faiss to ${FAISS_INSTALL_PATH}..."

cd "${CUR_DIR}"
if [ -f "faiss-1.10.0.tar.gz" ]; then
    echo "[INFO] Using cached faiss archive"
else
    echo "[INFO] Downloading faiss..."
    wget https://github.com/facebookresearch/faiss/archive/v1.10.0.tar.gz -O faiss-1.10.0.tar.gz
fi
tar -xf faiss-1.10.0.tar.gz && cd faiss-1.10.0/faiss
sed -i "149 i virtual void search_with_filter (idx_t n, const float *x, idx_t k, float *distances, idx_t *labels, const void *mask = nullptr) const {}" Index.h
sed -i "49 i template <typename IndexT> IndexIDMapTemplate<IndexT>::IndexIDMapTemplate (IndexT *index, std::vector<idx_t> &ids): index (index), own_fields (false) {this->is_trained = index->is_trained; this->metric_type = index->metric_type; this->verbose = index->verbose; this->d = index->d; id_map = ids;}" IndexIDMap.cpp
sed -i "30 i explicit IndexIDMapTemplate (IndexT *index, std::vector<idx_t> &ids); " IndexIDMap.h
sed -i "217 i utils/sorting.h" CMakeLists.txt
cd ..
echo "[INFO] Compiling and installing faiss..."
cmake -B build . -DFAISS_ENABLE_GPU=OFF -DFAISS_ENABLE_PYTHON=OFF -DBUILD_TESTING=OFF -DBUILD_SHARED_LIBS=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=${FAISS_INSTALL_PATH}
cd build && make -j$(nproc) && make install
cd ../.. && rm -rf faiss-1.10.0

echo "export LD_LIBRARY_PATH=${FAISS_INSTALL_PATH}/lib:\$LD_LIBRARY_PATH" >> /etc/profile

# ============== 4. 安装 IndexSDK ==============
echo "[INFO] Start installing IndexSDK..."

if [ -d "${MX_INDEX_INSTALL_PATH}/mxIndex" ]; then
    echo "[WARN] IndexSDK already exists at ${MX_INDEX_INSTALL_PATH}"
    read -p "[INFO] Press Enter to use default path (uninstall existing IndexSDK), or enter custom path: " CUSTOM_INDEX_PATH
    if [ -n "${CUSTOM_INDEX_PATH}" ]; then
        MX_INDEX_INSTALL_PATH="${CUSTOM_INDEX_PATH}"
    else
        bash "${MX_INDEX_INSTALL_PATH}/mxIndex/script/uninstall.sh"
    fi
fi

cd "${CUR_DIR}"
if [ -f "${INDEXSDK_PKG}" ]; then
    echo "[INFO] Using cached IndexSDK installer"
else
    echo "[INFO] Downloading IndexSDK..."
    wget ${INDEXSDK_REPO_URL}/${INDEXSDK_PKG}
fi
echo "[INFO] Installing IndexSDK..."
chmod +x ${INDEXSDK_PKG}
./${INDEXSDK_PKG} --install --install-path=${MX_INDEX_INSTALL_PATH} --platform=${PLATFORM}

echo "export LD_LIBRARY_PATH=${MX_INDEX_INSTALL_PATH}/mxIndex/host/lib:\$LD_LIBRARY_PATH" >> /etc/profile
echo "[INFO] Deployment completed successfully! Please execute: source /etc/profile"
