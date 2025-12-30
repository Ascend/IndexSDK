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

VSTAR_PATH=$1
GCC_VER=$2

readonly VERSION=$(sed -n 's/^version:[[:space:]]*//p' "${VSTAR_PATH}"/../../ci/config/config.ini)

echo "Vstar Path is $VSTAR_PATH"
echo "Version is $VERSION"
CURRENT_PATH=$(pwd)

# 为减少同一代码仓内重复代码文件，对版本级cleanCode扫出的ivfsp_utils仓中的冗余文件进行移除, 在流水线编译中将需要的
# 文件拷贝至原本的移除文件处
copy_dup_files_retrieval_common() {

    # ascend_operator.h
    cp $RETRIEVAL_COMMON_PATH/src/ops/op_proto/include/ascend_operator.h \
       $RETRIEVAL_COMMON_PATH/src/ops_IVFSP/op_proto/include/
    
    # kernel_utils.h
    cp $RETRIEVAL_COMMON_PATH/src/ops/cpukernel/impl/utils/kernel_utils.h \
       $RETRIEVAL_COMMON_PATH/src/ops_IVFSP/cpukernel/impl/utils/
    cp $RETRIEVAL_COMMON_PATH/src/ops/cpukernel/impl/utils/kernel_utils.h \
       $RETRIEVAL_COMMON_PATH/src/ops_vstar/CpuOps/cpukernel/impl/utils/

    # kernel_utils.cpp
    cp $RETRIEVAL_COMMON_PATH/src/ops/cpukernel/impl/utils/kernel_utils.cpp \
       $RETRIEVAL_COMMON_PATH/src/ops_IVFSP/cpukernel/impl/utils/
    cp $RETRIEVAL_COMMON_PATH/src/ops/cpukernel/impl/utils/kernel_utils.cpp \
       $RETRIEVAL_COMMON_PATH/src/ops_vstar/CpuOps/cpukernel/impl/utils/

    # kernel_tensor.h
    cp $RETRIEVAL_COMMON_PATH/src/ops/cpukernel/impl/utils/kernel_tensor.h \
       $RETRIEVAL_COMMON_PATH/src/ops_IVFSP/cpukernel/impl/utils/
    cp $RETRIEVAL_COMMON_PATH/src/ops/cpukernel/impl/utils/kernel_tensor.h \
       $RETRIEVAL_COMMON_PATH/src/ops_vstar/CpuOps/cpukernel/impl/utils/

    # cpu_node_def.h
    cp $RETRIEVAL_COMMON_PATH/src/ops/cpukernel/impl/utils/cpu_node_def.h \
       $RETRIEVAL_COMMON_PATH/src/ops_IVFSP/cpukernel/impl/utils/
    cp $RETRIEVAL_COMMON_PATH/src/ops/cpukernel/impl/utils/cpu_node_def.h \
       $RETRIEVAL_COMMON_PATH/src/ops_vstar/CpuOps/cpukernel/impl/utils/

    # cpu_kernel_utils.h
    cp $RETRIEVAL_COMMON_PATH/src/ops/cpukernel/impl/utils/cpu_kernel_utils.h \
       $RETRIEVAL_COMMON_PATH/src/ops_IVFSP/cpukernel/impl/utils/
    cp $RETRIEVAL_COMMON_PATH/src/ops/cpukernel/impl/utils/cpu_kernel_utils.h \
       $RETRIEVAL_COMMON_PATH/src/ops_vstar/CpuOps/cpukernel/impl/utils/

    # kernel_shared_def.h
    cp $RETRIEVAL_COMMON_PATH/src/ops_vstar/CpuOps/cpukernel/impl/utils/kernel_shared_def.h \
       $RETRIEVAL_COMMON_PATH/src/ops_IVFSP/cpukernel/impl/utils/

}


cp -r ./ivfsp_utils ./ivfsp_impl
mv ./ivfsp_impl/ivfsp_utils ./ivfsp_impl/Retrieval_Common
RETRIEVAL_COMMON_PATH=$CURRENT_PATH/ivfsp_impl/Retrieval_Common
copy_dup_files_retrieval_common

cp -r ./ivfsp_impl ./vstar_great_impl/mix-index
cd ./vstar_great_impl/mix-index

####################

MIX_IDX_MAIN_PATH=$(pwd)
ASCENDFAISS_MAIN_PATH=$MIX_IDX_MAIN_PATH/ivfsp_impl/ascendfaiss
gcc_version="gcc7.3.0"
if [ "${GCC_VER}" = "Gcc4" ]; then
        gcc_version="gcc4.8.5"
fi
readonly ARCH="$(uname -m)"

# 将Vstar算法的AICPU算子文件(op_info_cfg内的配置文件和impl内的实现文件)拷贝进入IVFSP的/ops/cpukernels目录内
copy_vstar_aicpu_ops() {

    # 处理op_info_config文件
    cd $MIX_IDX_MAIN_PATH/ops/ascend_c/CpuOps/cpukernel/op_info_cfg/
    cp -r ./aicpu_kernel ./tmp && cd ./tmp
    for file in ./*; 
    do
        sed -i "s|"libcust_aicpu_kernels.so"|"libcust_aicpu_kernels_ascendsearch.so"|" ${file} # 改成最终的.so包名字
    done
    cp ./* $ASCENDFAISS_MAIN_PATH/ops/cpukernel/op_info_cfg/aicpu_kernel
    cd .. & rm -rf ./tmp

    # 处理impl实现文件
    rm -rf $ASCENDFAISS_MAIN_PATH/ops/cpukernel/impl/utils
    cp $MIX_IDX_MAIN_PATH/ops/ascend_c/CpuOps/cpukernel/impl/*.h $ASCENDFAISS_MAIN_PATH/ops/cpukernel/impl/
    cp $MIX_IDX_MAIN_PATH/ops/ascend_c/CpuOps/cpukernel/impl/*.cpp $ASCENDFAISS_MAIN_PATH/ops/cpukernel/impl/
    cp -r $MIX_IDX_MAIN_PATH/ops/ascend_c/CpuOps/cpukernel/impl/utils $ASCENDFAISS_MAIN_PATH/ops/cpukernel/impl/
}

# 将Retrieval_Common仓内算子相关文件拷贝到Vstar和IVFSP算法源码所在路径，从而避免下面的脚本改动
mkdir $MIX_IDX_MAIN_PATH/ops
cp -r $MIX_IDX_MAIN_PATH/ivfsp_impl/Retrieval_Common/src/ops_vstar $MIX_IDX_MAIN_PATH/ops/ascend_c
cp -r $MIX_IDX_MAIN_PATH/ivfsp_impl/Retrieval_Common/src/ops_IVFSP $ASCENDFAISS_MAIN_PATH/ops
cp -r $MIX_IDX_MAIN_PATH/ivfsp_impl/Retrieval_Common/src/CodeBookTraining_vstar $MIX_IDX_MAIN_PATH/CodeBookTraining

# 开始联合编译
cd union
dos2unix allRun.sh
bash allRun.sh
cd $MIX_IDX_MAIN_PATH

# 编译磁盘检索动态库
cd $MIX_IDX_MAIN_PATH/../DiskIndex
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j
cd $MIX_IDX_MAIN_PATH

echo "Building mix-index package..."
if [ -d "./Ascend-ascendsearch_${VERSION}_${gcc_version}-linux-${ARCH}-310P" ]; then
rm -rf ./Ascend-ascendsearch_${VERSION}_${gcc_version}-linux-${ARCH}-310P
fi
mkdir Ascend-ascendsearch_${VERSION}_${gcc_version}-linux-${ARCH}-310P && cd Ascend-ascendsearch_${VERSION}_${gcc_version}-linux-${ARCH}-310P

# 将ivfsp + (vstar + great)的库拷贝到最终路径
mkdir lib
cp ../union/lib/libascendsearch.so ./lib
cp $MIX_IDX_MAIN_PATH/../DiskIndex/build/src/libdisksearch_opgs.so ./lib # 拷贝磁盘检索适配OpenGauss so到最终路径

# set up ivfsp package
cd $ASCENDFAISS_MAIN_PATH

if [ -d "./IVFSP" ]; then
    rm -rf ./IVFSP
fi
mkdir IVFSP && cd IVFSP

mkdir include
cp -r ../include/ascendsearch/ include/
cp ../../Retrieval_Common/src/ascend/AscendIndex.h include/ascendsearch/ascend/

mkdir ops
mkdir ops/cpukernel
mkdir ops/tbe
copy_vstar_aicpu_ops
cd $ASCENDFAISS_MAIN_PATH/IVFSP

cp -r ../ops/cpukernel/op_info_cfg ops/cpukernel/op_info_cfg
cp ../ops/tbe/CMakeLists.txt ops/tbe/CMakeLists.txt
cp -r ../ops/tbe/op_info_cfg/ ops/tbe/op_info_cfg/
cp -r ../ops/tbe/impl ops/tbe/impl

cd ../ops
num_lines=$(wc -l < CMakeLists.txt)
sed -i "${num_lines}c\)" CMakeLists.txt # 将CMakeLists最后一行注释，不在此处去注册算子
bash build.sh 310P
cd ../IVFSP
cp -r ../ops/build_out/makepkg/packages/op_proto/custom/libcust_op_proto_ascendsearch.so ops/
cp -r ../ops/build_out/makepkg/packages/op_impl/custom/cpu/aicpu_kernel/custom_impl/libcust_aicpu_kernels_ascendsearch.so ops/

mkdir tools
cp ../../Retrieval_Common/src/tools/ascendfaiss/tools/aicpu_generate_model.py tools
cp ../../Retrieval_Common/src/tools/ascendfaiss/tools/ivfsp_generate_model.py tools
cp ../../Retrieval_Common/src/tools/ascendfaiss/tools/ivfsp_generate_pyacl_model.py tools

mkdir train
cd ../train 

cd ../
cd IVFSP
cp ../train/readme.md train/
cp ../../Retrieval_Common/src/tools/ascendfaiss/train/train.py train/
cp ../train/train.sh train/
cp $RETRIEVAL_COMMON_PATH/src/tools/ascendfaiss/train/SP* train/

mkdir DR
cd DR
cp $RETRIEVAL_COMMON_PATH/src/DR/PA* ./
cp ../../../Retrieval_Common/src/DR/* ./
cd ../../
cp -r ./IVFSP $MIX_IDX_MAIN_PATH/Ascend-ascendsearch_${VERSION}_${gcc_version}-linux-${ARCH}-310P
cd $MIX_IDX_MAIN_PATH

# set up mix-search package
if [ -d "./mix-index" ]; then
    rm -rf ./mix-index
fi
mkdir mix-index && cd mix-index 

mkdir include
cp -r ../include ./include/Vstar_include
cp -r ../third_party/include ./include/Great_include

mkdir train && cd ../CodeBookTraining/
cd ../mix-index
cp ../CodeBookTraining/vstar_train_codebook.py ./train
cp ../CodeBookTraining/vstar_trainer.py ./train

mkdir READMEs
cp ../READMEs/indexGreat_README.md ./READMEs
cp ../READMEs/indexVstar_README.md ./READMEs

mkdir -p ops/ascend_c
cp -r ../ops/ascend_c/AscendcOps ./ops/ascend_c/

mkdir tools
cp ../ops/ascend_c/vstar_generate_models.py ./tools/

echo "Building mix-index package finished!"

cd $MIX_IDX_MAIN_PATH

# 将mix-index目录拷贝进入最终包内
mv ./mix-index ./Ascend-ascendsearch_${VERSION}_${gcc_version}-linux-${ARCH}-310P

cd Ascend-ascendsearch_${VERSION}_${gcc_version}-linux-${ARCH}-310P

tar -zcvf $CURRENT_PATH/Ascend-ascendsearch_${VERSION}_${gcc_version}-linux-${ARCH}-310P.tar.gz *

# cleaning

cd $CURRENT_PATH

# cleaning
mv $MIX_IDX_MAIN_PATH/Ascend-ascendsearch_${VERSION}_${gcc_version}-linux-${ARCH}-310P .
rm -rf $MIX_IDX_MAIN_PATH/ivfsp_impl
rm -rf $MIX_IDX_MAIN_PATH/mix-index
rm -rf $CURRENT_PATH/ivfsp_impl/Retrieval_Common
