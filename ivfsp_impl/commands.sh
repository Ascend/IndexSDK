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
set +eux
source /etc/profile
set -eu

gcc_type=$1
if [ "${gcc_type}" == "Gcc7" ]; then
    gcc_version="gcc7.3.0"
else
    gcc_version="gcc4.8.5"
fi

readonly ARCH="$(uname -m)"

readonly project_root_folder=${PWD}
if [ -f ${project_root_folder}/../build/conf/version.yaml ]; then 
    readonly VERSION=$(sed -n 's/^version:[[:space:]]*//p' "${project_root_folder}"/../../ci/config/config.ini)
else
    VERSION="5.0.RC2"
fi
DIR_PATH=Ascend-ascendsearch_${VERSION}_${gcc_version}-linux-${ARCH}-310P

cd Retrieval_Common
! [[ -h ascendsearch ]] && ln -s . ascendsearch
cd ../ascendfaiss
export Retrieval_PATH=$(pwd)
! [[ -h ascendsearch ]] && ln -s . ascendsearch
rm -rf build
mkdir build && cd build

cmake -D FAISS_HOME=/usr/local/faiss ../

make -j
sed -i "1iset(CMAKE_INSTALL_PREFIX $Retrieval_PATH)" cmake_install.cmake
make install
ls -l 
cd ../
mkdir ${DIR_PATH} && cd ${DIR_PATH}
mkdir include
cp -r $Retrieval_PATH/include/ascendsearch/ include/
cp ../../../Retrieval_Common/ascend/AscendIndex.h include/ascendsearch/ascend/

mkdir lib
cp -r $Retrieval_PATH/host/lib/libascendsearch.so lib/

mkdir ops
mkdir ops/cpukernel
mkdir ops/tbe
cp -r ../ops/cpukernel/op_info_cfg ops/cpukernel/op_info_cfg
cp ../ops/tbe/CMakeLists.txt ops/tbe/CMakeLists.txt
cp -r ../ops/tbe/op_info_cfg/ ops/tbe/op_info_cfg/
cp -r ../ops/tbe/impl ops/tbe/impl
cd ../ops
bash build.sh 310P
cd ../
cd ${DIR_PATH}
cp -r ../ops/build_out/makepkg/packages/op_proto/custom/libcust_op_proto_ascendsearch.so ops/
cp -r ../ops/build_out/makepkg/packages/op_impl/custom/cpu/aicpu_kernel/custom_impl/libcust_aicpu_kernels_ascendsearch.so ops/

mkdir tools
cp ../../../Retrieval_Common/tools/ascendfaiss/tools/aicpu_generate_model.py tools
cp ../../../Retrieval_Common/tools/ascendfaiss/tools/ivfsp_generate_model.py tools
cp ../../../Retrieval_Common/tools/ascendfaiss/tools/ivfsp_generate_pyacl_model.py tools

mkdir train
cd ../train 

cd ../
cd ${DIR_PATH}
cp ../train/readme.md train/
cp ../../../Retrieval_Common/tools/ascendfaiss/train/train.py train/
cp ../train/train.sh train/
cp ../train/SP* train/

mkdir DR
cd DR
mv ../../train/DR/PA* ./
cp ../../../../Retrieval_Common/DR/* ./
cd ../

tar -zcvf ../Ascend-ascendsearch_${VERSION}_${gcc_version}-linux-${ARCH}-310P.tar.gz *
