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
usage() {
    echo "Usage:"
    echo "run_dt.sh [-o openblas_path] [-f faiss_path] [-p protobuf_host_path] [-g googletest_path] [-m modelpath]"
    exit 1
}

openblas_path="/opt/OpenBLAS/lib"
faiss_path="/usr/local/index_dev/faiss-1.7.1/build/faiss"
protobuf_path="/usr/local/protobuf/lib/"
googletest_path="/usr/local/googletest-release-1.10.0/lib"
modelpath="/home/lijunyu/modelpath"

while getopts 'h:o:f:p:g:m:' OPT; do
    case "$OPT" in
        o)
		openblas_path="$OPTARG";;
        f)
		faiss_path="$OPTARG";;
        p)
		protobuf_path="$OPTARG";;
        g)
		googletest_path="true";;
        m)
		modelpath="$OPTARG";;
        h)
		usage;;
    esac
done

echo "openblas_path: ${openblas_path}"
echo "faiss_path: ${faiss_path}"
echo "protobuf_path: ${protobuf_path}"
echo "googletest_path: ${googletest_path}"
echo "modelpath: ${modelpath}"

CUR_DIR=$(dirname "$(readlink -f "$0")")
cd ${CUR_DIR}
cd ../src/ascendfaiss/

find ./ -name "*.sh" -exec dos2unix {} \;
find ./ -name "*.sh" -exec chmod +x {} \;

mkdir -p build
cd build

cmake .. -DBUILD_ASCENDCTL=OFF -DBUILD_ASCENDDEVICE=OFF -DBUILD_OPS=OFF -DBUILD_TESTS=on -DCOVERAGE=on
make -j

export MX_INDEX_MODELPATH=${modelpath}

CUR_DIR=$(dirname "$(readlink -f "$0")")

export LD_LIBRARY_PATH=${CUR_DIR}/host/lib:${openblas_path}:${faiss_path}:${protobuf_path}:${googletest_path}:$LD_LIBRARY_PATH

export ASCEND_SLOG_PRINT_TO_STDOUT=1

make test

make mxindex-lcov

cp -r cover_report ../../../test/
cp -r ./test/test_detail.xml ../../../test
