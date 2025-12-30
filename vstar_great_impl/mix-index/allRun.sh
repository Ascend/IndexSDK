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

echo "Builing indexGreat ..."
cd ./beta
if [ -d "./build" ]; then
    rm -rf ./build
fi
if [ -d "./lib" ]; then
    rm -rf ./lib
fi
mkdir ./build
cd ./build
cmake -DCMAKE_CXX_COMPILER=g++ -DCMAKE_C_COMPILER=gcc -DACL_ONLINE_MODE=FALSE ..
make -j
cd ../..

# echo "Generating ops..."
# cd ./ops/ascend_c
# dos2unix buildAll.sh
# bash buildAll.sh
# cd ../..

dos2unix build.sh
echo "Building Test executable (ascend_test)..."
bash build.sh test # build test
dos2unix run.sh

echo "Test executable building completed. You can run it in now the format specified in allRun.sh."

echo "Running Test..."

bash run.sh