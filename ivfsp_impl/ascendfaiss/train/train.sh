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

# in ivfsp_train_codebook.py, the parameters below have default value
# you may set those parameters according to the actual situation.
num_iter=20
batch_size=32768
ratio=1.0

# in ivfsp_train_codebook.py, the parameters below have no default value, which need the user to set
# please set those parameters according to the actual situation.
CURRENT_DIR=$(cd $(dirname $0); pwd)
nlist=1024
dim=256
nonzero_num=32
device=0
codebook_batch_size=32768
learn_data_path=${CURRENT_DIR}/../learn.bin
codebook_output_dir=~/codebook_dir
train_model_dir=${CURRENT_DIR}/../op_models_pyacl

if [ ! -d "${codebook_output_dir}" ]; then
    mkdir -m 750 ${codebook_output_dir}
fi

cd "${CURRENT_DIR}" || exit 1

python3 ivfsp_train_codebook.py \
--nlist $nlist \
--dim $dim \
--nonzero_num $nonzero_num \
--num_iter $num_iter \
--device $device \
--batch_size $batch_size \
--code_num $codebook_batch_size \
--ratio $ratio \
--learn_data_path $learn_data_path \
--codebook_output_dir $codebook_output_dir \
--train_model_dir $train_model_dir
