# 基于 IndexSDK 官方 Dockerfile.910b.ubuntu 适配 Dev Container
ARG CANN_VERSION=9.0.0
FROM swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:${CANN_VERSION}-910b-ubuntu22.04-py3.11

ARG PLATFORM=910B
ARG VERSION=26.0.0
ARG install_path=/usr/local/faiss
ARG ASCEND_BASE=/usr/local/Ascend
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Shanghai

# ========== 1. 更换 apt 源为华为云镜像 ==========
RUN cp -a /etc/apt/sources.list /etc/apt/sources.list.bak && \
    sed -i "s@http://.*archive.ubuntu.com@https://mirrors.huaweicloud.com@g" /etc/apt/sources.list && \
    sed -i "s@http://.*security.ubuntu.com@https://mirrors.huaweicloud.com@g" /etc/apt/sources.list

# ========== 2. 安装系统依赖 ==========
RUN apt update && \
    apt install --no-install-recommends -y \
        ca-certificates wget vim dos2unix net-tools ssh lsof \
        curl gcc g++ make pkg-config unzip autoconf git patch \
        libblas3 liblapack3 gfortran openmpi-bin libopenmpi-dev libxml2 \
        pciutils liblapack-dev libblas-dev libffi-dev libssl-dev \
        zlib1g-dev xz-utils libgmpxx4ldbl \
        libcapture-tiny-perl libdatetime-perl libtimedate-perl \
        python3-pip software-properties-common build-essential && \
    apt clean && rm -rf /var/lib/apt/lists/*

# ========== 3. 配置 pip 为阿里云镜像 ==========
RUN mkdir -p ~/.pip && \
    echo "[global]" > ~/.pip/pip.conf && \
    echo "trusted-host=mirrors.aliyun.com" >> ~/.pip/pip.conf && \
    echo "index-url=https://mirrors.aliyun.com/pypi/simple" >> ~/.pip/pip.conf && \
    echo "timeout=200" >> ~/.pip/pip.conf

# ========== 4. 安装 Python 依赖 ==========
RUN pip3 install -U pip && \
    pip3 install decorator sympy==1.4 cffi==1.15.1 pyyaml pathlib2 protobuf \
                 scipy requests attrs psutil numpy==1.26.4 faiss-cpu==1.13.2 && \
    rm -rf /root/.cache/pip

# ========== 5. 安装 CMake（使用预编译二进制） ==========
RUN wget https://cmake.org/files/v3.24/cmake-3.24.0.tar.gz && \
    tar xf cmake-3.24.0.tar.gz && cd cmake-3.24.0 && \
    ./configure --prefix=/usr && make -j12 && make -j12 install && \
    cd .. && rm -rf cmake-3.24.0*


# ========== 6. 编译安装 OpenBLAS ==========
RUN wget https://github.com/xianyi/OpenBLAS/archive/v0.3.10.tar.gz -O OpenBLAS-0.3.10.tar.gz && \
    tar -xf OpenBLAS-0.3.10.tar.gz && cd OpenBLAS-0.3.10 && \
    make FC=gfortran USE_OPENMP=1 TARGET=ATOM -j12 && make install && \
    ln -s /opt/OpenBLAS/lib/libopenblas.so /usr/lib/libopenblas.so && \
    cd .. && rm -f OpenBLAS-0.3.10.tar.gz && rm -rf OpenBLAS-0.3.10


# ========== 7. 编译安装 Faiss ==========
RUN wget https://github.com/facebookresearch/faiss/archive/v1.10.0.tar.gz -O faiss-1.10.0.tar.gz && \
    tar -xf faiss-1.10.0.tar.gz && cd faiss-1.10.0/faiss && \
    sed -i "149 i virtual void search_with_filter (idx_t n, const float *x, idx_t k, float *distances, idx_t *labels, const void *mask = nullptr) const {}" Index.h && \
    sed -i "49 i template <typename IndexT> IndexIDMapTemplate<IndexT>::IndexIDMapTemplate (IndexT *index, std::vector<idx_t> &ids): index (index), own_fields (false) {this->is_trained = index->is_trained; this->metric_type = index->metric_type; this->verbose = index->verbose; this->d = index->d; id_map = ids;}" IndexIDMap.cpp && \
    sed -i "30 i explicit IndexIDMapTemplate (IndexT *index, std::vector<idx_t> &ids); " IndexIDMap.h &&\
    sed -i "217 i utils/sorting.h" CMakeLists.txt &&\
    cd .. && \
    cmake -B build . -DFAISS_ENABLE_GPU=OFF -DFAISS_ENABLE_PYTHON=OFF \
          -DBUILD_TESTING=OFF -DBUILD_SHARED_LIBS=ON \
          -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=${install_path} && \
    cd build && make -j && make install && \
    cd ../.. && rm -f faiss-1.10.0.tar.gz && rm -rf faiss-1.10.0 && \
    cp ${install_path}/lib/libfaiss.so /usr/local/lib

# ========== 8. 安装 lcov 和 googletest ==========
RUN wget https://github.com/linux-test-project/lcov/releases/download/v2.0/lcov-2.0.tar.gz && \
    tar -xzf lcov-2.0.tar.gz && cd lcov-2.0 && make install && \
    cd .. && rm -rf lcov-2.0.tar.gz lcov-2.0

RUN wget https://github.com/google/googletest/archive/refs/tags/release-1.11.0.tar.gz && \
    cp release-1.11.0.tar.gz /tmp/googletest-release-1.11.0.tar.gz && \
    tar xf release-1.11.0.tar.gz && cd googletest-release-1.11.0 && \
    cmake -DBUILD_SHARED_LIBS=ON -DCMAKE_INSTALL_PREFIX=/usr/local/gtest . && \
    make -j12 && make install && \
    cd .. && rm -rf release-1.11.0.tar.gz googletest-release-1.11.0

# ========== 9. 安装 IndexSDK ==========
RUN ARCH=$(uname -m | sed 's/x86_64/x86_64/; s/aarch64/aarch64/') && \
    wget https://gitcode.com/Ascend/IndexSDK/releases/download/v${VERSION}/Ascend-mindxsdk-mxindex_${VERSION}_linux-${ARCH}.run && \
    bash Ascend-mindxsdk-mxindex_${VERSION}_linux-${ARCH}.run --install --platform=${PLATFORM} && \
    cd ${ASCEND_BASE}/mxIndex/ops && ./custom_opp_${ARCH}.run

# ========== 10. 设置环境变量 ==========
ENV TOOLKIT_PATH=/usr/local/Ascend/ascend-toolkit/latest \
    GLOG_v=2 \
    LD_LIBRARY_PATH=/usr/local/Ascend/mxIndex/host/lib:/usr/local/Ascend/ascend-toolkit/latest/runtime/lib64:/usr/local/faiss/lib:/opt/OpenBLAS/lib:/usr/local/gtest/lib:/usr/local/Ascend/driver/lib64/driver:$LD_LIBRARY_PATH \
    TBE_IMPL_PATH=/usr/local/Ascend/ascend-toolkit/latest/opp/op_impl/built-in/ai_core/tbe \
    PATH=/usr/local/Ascend/ascend-toolkit/latest/atc/ccec_compiler/bin:/usr/local/Ascend/ascend-toolkit/latest/atc/bin:$PATH \
    ASCEND_OPP_PATH=/usr/local/Ascend/ascend-toolkit/latest/opp \
    ASCEND_AICPU_PATH=/usr/local/Ascend/ascend-toolkit/latest \
    PYTHONPATH=/usr/local/Ascend/ascend-toolkit/latest/opp/op_impl/built-in/ai_core/tbe:$PYTHONPATH \
    ASCEND_HOME=/usr/local/Ascend \
    MX_INDEX_MODELPATH=/home/Ascend/modelpath

RUN mkdir -p ${MX_INDEX_MODELPATH}

# ========== 11. 配置 bashrc ==========
RUN sed -i '$a\source /usr/local/Ascend/ascend-toolkit/set_env.sh' ~/.bashrc && \
    sed -i '$a\export LD_LIBRARY_PATH=/usr/local/faiss/lib:$LD_LIBRARY_PATH' ~/.bashrc && \
    sed -i '$a\export LD_LIBRARY_PATH=/opt/OpenBLAS/lib:$LD_LIBRARY_PATH' ~/.bashrc && \
    sed -i '$a\export LD_LIBRARY_PATH=/usr/local/gtest/lib:$LD_LIBRARY_PATH' ~/.bashrc && \
    sed -i '$a\export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64/driver:$LD_LIBRARY_PATH' ~/.bashrc

# ========== 12. 设置工作目录 ==========
WORKDIR /workspace
