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

soc_version='Ascend310P3'
run_model=$1

if [[ `echo $run_model|grep ".onnx"` != "" ]]
then
  default_save_model=(${run_model//.onnx/ })
  framework=5
  input_format=NCHW
elif [[ `echo $run_model|grep ".pb"` != "" ]]
then
  default_save_model=(${run_model//.pb/ })
  framework=3
  input_format=NHWC
fi
output=${2:-${default_save_model[0]}}
echo $output

input_shape=$3
atc \
  --model=$run_model \
  --framework=$framework \
  --output=$output \
  --input_format=$input_format \
  --input_shape=$input_shape \
  --soc_version=$soc_version \
  --log=info

# clean cache
rm -rf fusion_result.json