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
find .. -type f -name "*.sh" -exec dos2unix {} \;
CURRENT_PATH=$(cd "$(dirname "$0")"; pwd)
if [ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]; then
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
fi
show_help_info()
{    
    echo "Example:"
    echo "    $1 -h|--help                                    Show this message"
    echo "    $1 --with_submit_report                         Src Build + Test Build + Function Test + Fuzz Test + Performance Test + Api Test + Coverage Report + Submit Performance Report"
    echo "    $1 --disable_fuzz_test                          Src Build + Test Build + Function Test + Performance Test + Api Test + Coverage Report "
    echo "    $1 --disable_perf_test                          Src Build + Test Build + Function Test + Fuzz Test + Api Test + Coverage Report"
    echo "    $1 --disable_api_test                           Src Build + Test Build + Function Test + Fuzz Test + Performance Test + Coverage Report"
    echo "    $1 --disable_fuzz_test --disable_perf_test      Src Build + Test Build + Function Test + Api Test + Coverage Report"
    echo "    $1 --disable_function_test                      Src Build + Test Build + Fuzz Test + Performance Test + Api Test + Coverage Report"
    echo "    $1 --disable_build                              Fuzz Test + Performance Test + Api Test + Coverage Report"
    echo "    $1 --disable_src_build                          Test Build + Fuzz Test + Performance Test + Api Test + Coverage Report"
    echo "    $1 --disable_report                             Src Build + Test Build + Function Test + Fuzz Test + Performance Test + Api Test"
    echo "    $1 --with_filter=POSTIVE_PATTERNS               Specify gtest_filter"
    echo "    $1 --disable_src_build  --disable_fuzz_test --disable_perf_test --disable_api_test --disable_report --with_filter TestOckUEFeature*"
    echo "    $1 --with-nomock                                run nomock test. --with-nomock must be the first argument"
    echo "    $1                                              Src Build + Test Build + Function Test + Fuzz Test + Api Test + Performance Test + Coverage Report"
}
check_with_nomock_input()
{
    while [ $# -ge 1 ]
    do
        if [ "$1" = "--with-nomock" ]; then
            return 1
        fi
        shift
    done
    return 0
}
if [ "$1" = "-h" -o "$1" = "--help" ]; then
    show_help_info $0
    exit
fi
if [ "${MX_INDEX_MODELPATH}" = "" ]; then
    export MX_INDEX_MODELPATH="/opt/ascend/model"
fi
WITH_SUBMIT_REPORT="False"
WITH_FUZZ_TEST="True"
WITH_PERF_TEST="True"
WITH_API_TEST="True"
WITH_FUNCTION_TEST="True"
WITH_BUILD="True"
WITH_SRC_BUILD="True"
WITH_FILTER=""
WITH_REPORT="True"
RUN_NOMOCK="False"
if [ "$1" = "--with-nomock" ]; then
    RUN_NOMOCK="True"
    shift
fi
if ! check_with_nomock_input $@
then
    show_help_info $0
    echo "ERROR: --with-nomock must be the first argument"
    exit
fi

export LD_LIBRARY_PATH="${CURRENT_PATH}/../install/securec/lib":$LD_LIBRARY_PATH
export LD_LIBRARY_PATH="${CURRENT_PATH}/../install/memory_bridge/lib":$LD_LIBRARY_PATH
export LD_LIBRARY_PATH="${CURRENT_PATH}/../install/hcps_pier/lib":$LD_LIBRARY_PATH
export LD_LIBRARY_PATH="${CURRENT_PATH}/../install/vsa_hpp/lib":$LD_LIBRARY_PATH

echo "RUN_NOMOCK is ${RUN_NOMOCK}"
if [ "${RUN_NOMOCK}" = "True" ]; then
  cd "${CURRENT_PATH}/nomock"
  bash run_nomock.sh "$@"
fi

cd "${CURRENT_PATH}/withmock"
bash run_mock.sh "$@"

dos2unix "${CURRENT_PATH}/archcheck/check_depend_relation.sh"
bash "${CURRENT_PATH}/archcheck/check_depend_relation.sh"
