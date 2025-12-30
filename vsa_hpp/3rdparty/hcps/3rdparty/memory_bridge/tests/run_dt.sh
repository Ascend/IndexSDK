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
if [ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]; then
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
fi
if [ "$1" = "-h" -o "$1" = "--help" ]; then
    echo "Example:"
    echo "    $0 -h                                           Show this message"
    echo "    $0 --with_submit_report                         Src Build + Test Build + Function Test + Fuzz Test + Performance Test + Api Test + Coverage Report + Submit Performance Report"
    echo "    $0 --disable_fuzz_test                          Src Build + Test Build + Function Test + Performance Test + Api Test + Coverage Report "
    echo "    $0 --disable_perf_test                          Src Build + Test Build + Function Test + Fuzz Test + Api Test + Coverage Report"
    echo "    $0 --disable_api_test                           Src Build + Test Build + Function Test + Fuzz Test + Performance Test + Coverage Report"
    echo "    $0 --disable_fuzz_test --disable_perf_test      Src Build + Test Build + Function Test + Api Test + Coverage Report"
    echo "    $0 --disable_function_test                      Src Build + Test Build + Fuzz Test + Performance Test + Api Test + Coverage Report"
    echo "    $0 --disable_build                              Fuzz Test + Performance Test + Api Test + Coverage Report"
    echo "    $0 --disable_src_build                          Test Build + Fuzz Test + Performance Test + Api Test + Coverage Report"
    echo "    $0 --disable_report                             Src Build + Test Build + Function Test + Fuzz Test + Performance Test + Api Test"
    echo "    $0 --with_filter=POSTIVE_PATTERNS               Specify gtest_filter"
    echo "    $0 --disable_src_build  --disable_fuzz_test --disable_perf_test --disable_api_test --disable_report --with_filter TestOckUEFeature*"
    echo "    $0                                              Src Build + Test Build + Function Test + Fuzz Test + Api Test + Performance Test + Coverage Report"
    exit
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
while [ $# -ge 1 ]
do
    if [ "$1" = "--with_filter" ]; then
        shift
        WITH_FILTER="$1"
        shift
    elif [ "$1" = "--disable_report" ]; then
        WITH_REPORT="False"
        shift
    elif [ "$1" = "--with_submit_report" ]; then
        WITH_SUBMIT_REPORT="True"
        shift
    elif [ "$1" = "--disable_fuzz_test" ]; then
        WITH_FUZZ_TEST="False"
        shift
    elif [ "$1" = "--disable_perf_test" ]; then
        WITH_PERF_TEST="False"
        shift
    elif [ "$1" = "--disable_function_test" ]; then
        WITH_FUNCTION_TEST="False"
        shift
    elif [ "$1" = "--disable_src_build" ]; then
        WITH_SRC_BUILD="False"
        shift
    elif [ "$1" = "--disable_api_test" ]; then
        WITH_API_TEST="False"
        shift
    elif [ "$1" = "--disable_build" ]; then
        WITH_BUILD="False"
        shift
    else
        break
    fi
done
cd "${CURRENT_PATH:?}"
echo "WITH_BUILD=${WITH_BUILD} WITH_SRC_BUILD=${WITH_SRC_BUILD} WITH_API_TEST=${WITH_API_TEST} WITH_FUNCTION_TEST=${WITH_FUNCTION_TEST} WITH_PERF_TEST=${WITH_PERF_TEST} WITH_FUZZ_TEST=${WITH_FUZZ_TEST} WITH_SUBMIT_REPORT=${WITH_SUBMIT_REPORT}"
# **************************代码构建 Begin **************************
if [ "$WITH_BUILD" = "True" ]; then
    if [ "$WITH_SRC_BUILD" = "False" ]; then
        hdt build --task cmake_test "$@"
    else
        hdt build "$@"
    fi
fi
# **************************代码构建 End ****************************

# **************************测试执行 Begin **************************
GTEST_FILTER=""
HAS_NEGATIVE="False"
add_gtest_negative_filter_fun()
{
    if [ "$1" != "True" ]; then
        if [ "$GTEST_FILTER" = "" ]; then
            GTEST_FILTER=" --gtest_filter=-$2"
        else
            if [ "$HAS_NEGATIVE" = "False" ]; then
                GTEST_FILTER="${GTEST_FILTER}:-$2"
            else            
                GTEST_FILTER="${GTEST_FILTER}:$2"
            fi
        fi
        HAS_NEGATIVE=True
    fi
}
if [ "$WITH_FILTER" ]; then
    GTEST_FILTER=" --gtest_filter=${WITH_FILTER}"
fi

hdt comp -i || true
add_gtest_negative_filter_fun "${WITH_PERF_TEST}" "PTest*"
add_gtest_negative_filter_fun "${WITH_FUZZ_TEST}" "FuzzTest*"
add_gtest_negative_filter_fun "${WITH_FUNCTION_TEST}" "Test*"
add_gtest_negative_filter_fun "${WITH_API_TEST}" "ApiTest*"

echo "GTEST_FILTER=${GTEST_FILTER}"
hdt run --args="--gtest_output=xml:report.xml --ptest_baseline=performance_baseline.json --ptest_report=performance_report.json${GTEST_FILTER}"
# **************************测试执行 End ***************************

# **************************报告生成 Begin **************************
if [ "$WITH_REPORT" = "True" ]; then
    if [ "$WITH_SUBMIT_REPORT" != "True" ]; then
        hdt report --task lcov_report "$@"
    else
        hdt report "$@"
    fi
fi
# **************************报告生成 End ***************************
