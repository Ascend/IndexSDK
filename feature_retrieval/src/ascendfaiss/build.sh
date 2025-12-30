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
FAISS_PATH="/usr/local"
DRIVER_PATH="/usr/local/Ascend"
PROTO_HOST_PATH="/usr/local/protobuf"
PROTO_MINIOS_PATH="/opt/aarch64/protobuf"
generate_model="false"
npu_type="310"

set -e

usage() {
    echo "Usage:"
    echo "build.sh [-g generate_model] [-t npu_type, 310 or 310P] [-f faiss_path] [-p protobuf_host_path] [-a protobuf_minios_path] [-d driver_path]"
    exit -1
}

while getopts 'hd:f:p:a:gt:' OPT; do
    case "$OPT" in
        d) 
		DRIVER_PATH="$OPTARG";;
        f) 
		FAISS_PATH="$OPTARG";;
        p) 
		PROTO_HOST_PATH="$OPTARG";;
        a) 
		PROTO_MINIOS_PATH="$OPTARG";;
        g) 
		generate_model="true";;
        t) 
		npu_type="$OPTARG";;
        h) 
		usage;;
        ?) 
		usage;;
    esac
done

echo "DRIVER_PATH: ${DRIVER_PATH}"
echo "FAISS_PATH: ${FAISS_PATH}"
echo "PROTO_HOST_PATH: ${PROTO_HOST_PATH}"
echo "PROTO_MINIOS_PATH: ${PROTO_MINIOS_PATH}"
echo "generate_model: ${generate_model}"
echo "npu_type: ${npu_type}"

adapt_to_310p()
{
    sed -i 's/-DASCEND_310/-DASCEND_N310/g' acinclude/fa_check_ascend.m4
    sed -i 's/const int CORE_NUM = 2/const int CORE_NUM = 8/g' ops/op_proto/include/ascend_operator.h
}

adapt_to_310()
{
    sed -i 's/-DASCEND_N310/-DASCEND_310/g' acinclude/fa_check_ascend.m4
    sed -i 's/const int CORE_NUM = 8/const int CORE_NUM = 2/g' ops/op_proto/include/ascend_operator.h
}

if [ "${npu_type}" == "310" ]; then
    adapt_to_310
else
    adapt_to_310p
fi

# generate configure file
./autogen.sh
result=$?
if [ ${result} -ne 0 ]; then
    echo "[ERROR] autogen faild!"
    exit 1
fi

# set python interpreter, make sure installing Numpy in advance
if [ ! "$PYTHON" ]; then
    export PYTHON=/usr/bin/python3.7.5
fi

# configure
./configure --with-faiss=${FAISS_PATH} --with-ascenddriver=${DRIVER_PATH} --with-protobuf=${PROTO_HOST_PATH} --with-protobufaarch64=${PROTO_MINIOS_PATH}
result=$?
if [ ${result} -ne 0 ]; then
    echo "[ERROR] check environment failed!"
    exit 1
fi

# run make
export PKG_CONFIG_PATH=${PROTO_HOST_PATH}/lib/pkgconfig
echo "PKG_CONFIG_PATH: ${PKG_CONFIG_PATH}"

make clean && make -j
result=$?
if [ ${result} -ne 0 ]; then
    echo "[ERROR] make failed!"
    exit 1
fi
find ./ -name "*.o" -exec rm {} \;
result=$?
if [ ${result} -ne 0 ]; then
    echo "[ERROR] find .o and rm failed!"
    exit 1
fi
make -f Makefile-device -j
result=$?
if [ ${result} -ne 0 ]; then
    echo "[ERROR] make -f Makefile-device failed!"
    exit 1
fi

# run make install
make install
result=$?
if [ ${result} -ne 0 ];then
    echo "[ERROR] make install faild!"
    exit 1
fi

# registering custom tbe operator
cd ops && bash build.sh ${npu_type}
result=$?
if [ ${result} -ne 0 ];then
    echo "[ERROR] build and install tbe operators faild!"
    exit 1
fi

./build_out/custom_opp_*.run
result=$?
if [ ${result} -ne 0 ];then
    echo "[ERROR] deploy tbe operators faild!"
    exit 1
fi

# convert custom tbe operator to om files
if [ "${generate_model}" == "true" ];then
    cd ../tools
    rm -rf op_models config kernel_meta ../modelpath/*
    if [ "${npu_type}" == "310" ];then
        python run_generate_model.py
    else
        python run_generate_model.py -t 310P
    fi
    result=$?
    if [ ${result} -ne 0 ];then
        echo "[ERROR] generate om models faild!"
        exit 1
    fi
    mv op_models/* ../modelpath
    cd ..
fi
