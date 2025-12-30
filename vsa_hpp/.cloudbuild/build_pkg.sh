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

export LD_LIBRARY_PATH=$ASCEND_AUTOML_PATH/../${ARCH}-linux/devlib/:$LD_LIBRARY_PATH
source /usr/local/Ascend/ascend-toolkit/set_env.sh

WORKSPACE=$(cd "$(dirname "$0")/../"; pwd)
BUILD_TYPE=${1,,}
cd ${WORKSPACE}
bash build.sh -t ${BUILD_TYPE}
if [ 0 != $? ];then
      echo "Failed to build"
      exit 1
fi

cd ${WORKSPACE}/install
tar -zcvf OCK-Ascend-VSA-23.0.0_${OS}-${ARCH}.tar.gz *
echo "buildVersion=VSA_${localId}_${committer}">${WORKSPACE}/../buildInfo.properties
