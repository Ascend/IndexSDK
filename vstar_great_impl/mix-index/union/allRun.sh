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

echo "Builing ascendsearch ..."

cd ../ivfsp_impl/Retrieval_Common/src
! [[ -h ascendsearch ]] && ln -s . ascendsearch
cd ../../ascendfaiss
export Retrieval_PATH=$(pwd)
! [[ -h ascendsearch ]] && ln -s . ascendsearch
cd ../../

cd ./union
rm -rf ./build
rm -rf ./lib
mkdir ./build
cd ./build
cmake -DCMAKE_CXX_COMPILER=g++ -DCMAKE_C_COMPILER=gcc -DACL_ONLINE_MODE=FALSE -DFAISS_HOME=/usr/local/faiss ..
make -j
sed -i "1iset(CMAKE_INSTALL_PREFIX $Retrieval_PATH)" cmake_install.cmake
make install
cd ../..