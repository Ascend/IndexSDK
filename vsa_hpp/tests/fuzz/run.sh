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
CURRENT_PATH=$(cd "$(dirname "$0")"; pwd)
echo $CURRENT_PATH

. /usr/local/Ascend/ascend-toolkit/set_env.sh
export LD_LIBRARY_PATH="${CURRENT_PATH}/../../install/securec/lib":$LD_LIBRARY_PATH
export LD_LIBRARY_PATH="${CURRENT_PATH}/../../install/hcps/lib":$LD_LIBRARY_PATH
export LD_LIBRARY_PATH="${CURRENT_PATH}/../../install/memory_bridge/lib":$LD_LIBRARY_PATH
export LD_LIBRARY_PATH="${CURRENT_PATH}/../../install/vsa/lib":$LD_LIBRARY_PATH
export LD_LIBRARY_PATH="/usr/local/lib":$LD_LIBRARY_PATH
export MX_INDEX_MODELPATH=/home/pjy/modelpath

#bash ../../build.sh -t release -c on
#mkdir build
cd "${CURRENT_PATH}/build"
rm -rf *
cmake ../
make -j64

cd "${CURRENT_PATH}/build"

./fuzz_test --gtest_filter=FuzzTestOckVsaAnnCreateParam.create_param
mkdir -p ../build/gcovr_report
pwd
lcov --d ../../.. --c --output-file ../build/test.info --rc lcov_branch_coverage=1
lcov --remove ../build/test.info "*7.3.0*" "*/memory_bridge/*" "*/gtest/*" "*9*" "*usr*" "*nop*" "*hop*" "*streams*" -output-file ../build/coverage.info --rc lcov_branch_coverage=1;
genhtml -o ../build/gcovr_report ../build/coverage.info --show-details --legend --rc lcov_branch_coverage=1
