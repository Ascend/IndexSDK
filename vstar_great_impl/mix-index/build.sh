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

if [[ $# -eq 0 ]] || [[ $1 == 'ascend' ]]
then
  echo "Build ascendsearch lib..."
  rm -rf ./build
  rm -rf ./lib/*
  mkdir ./build
  cd ./build
  cmake -DCMAKE_CXX_COMPILER=g++ -DCMAKE_C_COMPILER=gcc -DACL_ONLINE_MODE=FALSE ..
  make -j
  make install
  cd ..
fi

if [[ $# -eq 1 ]] && [[ $1 == 'test' ]]
then
  echo "Build Test..."
  cd ./test
  rm -rf ./build
  mkdir ./build
  cd ./build
  cmake -DCMAKE_CXX_COMPILER=g++ -DCMAKE_C_COMPILER=gcc -DACL_ONLINE_MODE=FALSE ..
  make -j

fi
