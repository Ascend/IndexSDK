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

# ========== 环境配置（可通过环境变量覆盖）==========
: "${FAISS_HOME:=/home/indexSDK/faiss1.14.1}"
: "${GTEST_HOME:=/opt/buildtools/googletest-1.11.0}"
: "${OPENBLAS_HOME:=/opt/OpenBLAS}"

readonly CUR_DIR=$(dirname "$(readlink -f "$0")")
readonly RUN_PKG_PATH="${CUR_DIR}/../.."                     # ci 拉取的 run 包路径
readonly CHANGE_FILE="${CUR_DIR}/../../change.txt"           # ci 生成的 change.txt 文件，记录修改的文件路径
readonly PRESMOKE_DIR="/home/indexSDK/preSmokeTestFiles"     # 预冒烟测试文件目录，pkg 下存放 run 包，modelpath 下存放算子模型
readonly REFERENCE_DIR="${CUR_DIR}/../reference"             # 参考样例目录
export MX_INDEX_INSTALL_PATH=/usr/local/Ascend/mxIndex
export MX_INDEX_MODELPATH="${PRESMOKE_DIR}/modelpath"
export MX_INDEX_FINALIZE=1
export LD_LIBRARY_PATH=${OPENBLAS_HOME}/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=${FAISS_HOME}/lib/:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/Ascend/mxIndex/host/lib/:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64/driver:$LD_LIBRARY_PATH
source /usr/local/Ascend/ascend-toolkit/set_env.sh

SKIP_PRESMOKE_PATHS=(".github/" "build/" "docker/" "docs/" ".codespellrc" "*.sh" "*.md" "*.txt" "*.yaml")
MONITORED_SRC_PATHS=("feature_retrieval/src/ascendfaiss/ops" "feature_retrieval/src/ascendfaiss/tools")
OPS_LIST=("aicpu" "binary_flat" "ivfsqt" "ivfsq8" "flat_at_int8" "flat_at" "int8flat" "sq8" "flat" "mask")
ALGORITHMS=("binary_flat" "ivfsqt" "ivfsq" "ilflat" "cluster" "int8" "sq" "ts" "multi" "flat")
npu-smi info

# 根据文件名判断 ops 或者 tools 目录下的文件是否有修改，若有则需重新生成算子模型
declare -A OP_FILE_PATTERNS=(
    ["aicpu"]="aicpu_generate_model.py cpukernel"
    ["flat"]="flat_generate_model.py distance_flat distance_compute_flat distance_filter"
    ["binary_flat"]="binary_flat_generate_model.py distance_binary distance_flat_hamming"
    ["sq8"]="sq8_generate_model.py distance_sq8 distance_masked_sq8 cid_filter"
    ["int8flat"]="int8flat_generate_model.py ascendc_dist_int8 ascendc_l2_norm distance_int8"
    ["ivfsq8"]="ivfsq8_generate_model.py distance_ivf_sq8"
    ["ivfsqt"]="ivfsqt_generate_model.py distance_flat_ip_maxs distance_flat_subcenters distance_ivf_sq8_ipx"
    ["flat_at"]="flat_at_generate_model.py distance_flat_l2_mins_at l2_norm corr_compute"
    ["flat_at_int8"]="flat_at_int8_generate_model.py l2_norm_typing_int8 int8_at l2_at l2_norm_flat_sub subcent_accum"
    ["mask"]="mask_generate_model.py _mask_"
)

# 每个算子的生成命令
declare -A OP_GEN_CMD=(
    ["aicpu"]="python3 aicpu_generate_model.py -t 310P"
    ["flat"]="python3 flat_generate_model.py -d 256 -t 310P"
    ["binary_flat"]="python3 binary_flat_generate_model.py -d 256"
    ["sq8"]="python3 sq8_generate_model.py -d 256 -t 310P"
    ["int8flat"]="python3 int8flat_generate_model.py -d 256 -t 310P"
    ["ivfsq8"]="python3 ivfsq8_generate_model.py -d 256 -c 8192 -t 310P"
    ["ivfsqt"]="python3 ivfsqt_generate_model.py -d 256 -c 8192 -t 310P"
    ["flat_at"]="python3 flat_at_generate_model.py -d 256 -c 8192 -t 310P"
    ["flat_at_int8"]="python3 flat_at_int8_generate_model.py -d 256 -c 8192 -t 310P"
    ["mask"]="python3 mask_generate_model.py -t 310P"
)

# 根据文件名是否包含算法关键词判断需要运行哪些算法的用例
declare -A ALG_KEYWORDS=(
    ["flat"]="IndexFlat flat"
    ["binary_flat"]="IndexBinaryFlat binary_flat"
    ["sq"]="IndexSQ sq"
    ["int8"]="IndexInt8 int8"
    ["cluster"]="IndexCluster cluster"
    ["ilflat"]="IndexILFlat ilflat"
    ["ivfsq"]="IndexIVFSQ ivfsq flat_at"
    ["ivfsqt"]="IndexIVFSQT ivfsqt flat_at"
    ["ts"]="IndexTS ts binary_flat"
    ["multi"]="MultiIndex multi int8flat sq"
)

# 每个算法的测试用例文件
declare -A ALG_TEST=(                                                  # 包含用例个数(25)
    ["flat"]="TestAscendIndexFlat"                                     # 2
    ["binary_flat"]="TestAscendIndexBinaryFlat"                        # 1
    ["sq"]="TestAscendIndexSQ TestAscendIndexSQMulPerformance"         # 2
    ["int8"]="TestAscendIndexInt8Flat TestAscendIndexInt8FlatWithSQ"   # 4
    ["cluster"]="TestAscendIndexCluster"                               # 1
    ["ilflat"]="TestAscendIndexILFlat"                                 # 1
    ["ivfsq"]="TestAscendIndexIVFSQ"                                   # 1
    ["ivfsqt"]="TestAscendIndexIVFSQT"                                 # 1
    ["ts"]="TestAscendIndexTS"                                         # 8
    ["multi"]="TestAscendMultiSearch"                                  # 4
)

echo "[PRESMOKE_INFO] indexPreSmoke start"

# ============== 1. 根据修改的文件判断需要生成的算子和测试用例 ==============
echo "[PRESMOKE_INFO] start detect algorithms to test..."

NEED_RUN_PRESMOKE=false
NEED_REGENERATE_OPS=false
declare -A OPS_TO_PROCESS   # 存储需要生成的算子
declare -A ALGS_TO_PROCESS  # 存储需要处理的算法

if [[ ! -f "$CHANGE_FILE" ]]; then
    echo "[PRESMOKE_ERROR] $CHANGE_FILE not found!"
    exit 1
    # 调试请注释 exit 1
    cd "$CUR_DIR/.."
    changed=$(git diff master --no-commit-id --name-only)
    echo "$changed" > "$CHANGE_FILE"
fi
echo "[PRESMOKE_INFO] Changed files:"
cat "$CHANGE_FILE"
echo ""

while IFS= read -r file; do
    [[ -z "$file" ]] && continue
    is_skip_file=false
    for skip_path in "${SKIP_PRESMOKE_PATHS[@]}"; do
        if [[ "$file" == $skip_path* ]] || [[ "$(basename "$file")" == $skip_path ]]; then
            is_skip_file=true
            echo "[PRESMOKE_INFO] Skipping presmoke for: $file (matches $skip_path)"
            break
        fi
    done
    if [[ "$is_skip_file" == false ]]; then
        NEED_RUN_PRESMOKE=true
        echo "[PRESMOKE_INFO] Need presmoke for: $file"

        # 检查文件路径是否包含 ops 或 tools 的监控路径
        is_monitored_path=false
        for monitored_path in "${MONITORED_SRC_PATHS[@]}"; do
            if [[ "$file" == *"$monitored_path"* ]]; then
                is_monitored_path=true
                break
            fi
        done

        # 如果是监控路径下的文件，进行算子匹配
        if [[ "$is_monitored_path" == true ]]; then
            for op in "${OPS_LIST[@]}"; do
                patterns="${OP_FILE_PATTERNS[$op]}"
                for pat in $patterns; do
                    if [[ "$file" =~ $pat ]]; then
                        # 标记需要重新生成算子
                        NEED_REGENERATE_OPS=true
                        if [[ -z "${OPS_TO_PROCESS[$op]}" ]]; then
                            OPS_TO_PROCESS["$op"]=1
                            echo "    ✅ Detected op change: $file -> $op"
                        fi
                        break 2 # 跳出 2 层循环
                    fi
                done
            done
        fi

        # 判断需要测试的算法
        for alg in "${ALGORITHMS[@]}"; do
            keywords="${ALG_KEYWORDS[$alg]}"
            for kw in $keywords; do
                if echo "$file" | grep -q "$kw"; then
                    if [[ "$alg" == "ivfsq" && "$file" == *"IVFSQT"* ]]; then
                        echo "    ⏭️ Skipping 'ivfsq' because file looks like 'ivfsqt'"
                        break
                    fi
                    if [[ -z "${ALGS_TO_PROCESS[$alg]}" ]]; then
                        ALGS_TO_PROCESS["$alg"]=1
                        echo "    ✅ Detected keyword '$kw' → algorithm: $alg"
                    else
                        echo "    ⏭️ Already detected algorithm: $alg (from keyword '$kw')"
                    fi
                    break 2
                fi
            done
        done
    fi
done < "$CHANGE_FILE"

if [[ "$NEED_RUN_PRESMOKE" == false ]]; then
    echo "[PRESMOKE_INFO] All changed files are in skip paths (docs, docker, build, etc.)."
    echo "[PRESMOKE_INFO] Skipping presmoke test."
    echo "[PRESMOKE_INFO] indexPreSmoke finished (skipped)"
    exit 0
fi

if [[ "$NEED_REGENERATE_OPS" == false ]]; then
    echo "[PRESMOKE_INFO] No operator-specific changes detected. Skipping operator regeneration."
fi

# 如果没有检测到任何算法，跑默认用例（flat）
if [[ ${#ALGS_TO_PROCESS[@]} -eq 0 ]]; then
    echo "[PRESMOKE_INFO] No algorithm-specific changes detected. Running default test: flat"
    ALGS_TO_PROCESS["flat"]=1
fi

# ============== 2. 安装 run 包 ==============
echo "[PRESMOKE_INFO] start install run pkg..."

if [ -d "$PRESMOKE_DIR/pkg" ]; then
    echo "[PRESMOKE_INFO] test files already exist, removing..."
    rm -rf "$PRESMOKE_DIR/pkg"
fi

if [ -d "/usr/local/Ascend/mxIndex" ]; then
    echo "[PRESMOKE_INFO] mxIndex already exist, uninstalling..."
    if [ -f "/usr/local/Ascend/mxIndex/script/uninstall.sh" ]; then
        bash /usr/local/Ascend/mxIndex/script/uninstall.sh
    else
        echo "[PRESMOKE_WARN] uninstall.sh not found, performing manual cleanup..."
        rm -rf /usr/local/Ascend/mxIndex
    fi
fi

mkdir -p "$PRESMOKE_DIR/pkg"
cp "$RUN_PKG_PATH"/Ascend-mindxsdk-mxindex_*_linux-aarch64.run "$PRESMOKE_DIR/pkg"
cd "$PRESMOKE_DIR/pkg"
chmod +x *.run
echo "[PRESMOKE_INFO] start installing run pkg"
./Ascend-mindxsdk-mxindex_*_linux-aarch64.run --install --platform=310P --quiet

# ============== 3. 编译参考样例 ==============
echo "[PRESMOKE_INFO] start compile test demo..."

if [ ! -d "$REFERENCE_DIR" ]; then
    echo "[PRESMOKE_ERROR] reference not exist!"
    exit 1
fi

if [ ! -d "${GTEST_HOME}/lib" ]; then
    cd "${GTEST_HOME}/googletest-release-1.11.0/"
    cmake -DCMAKE_INSTALL_PREFIX="${GTEST_HOME}" \
        -DBUILD_SHARED_LIBS=ON
    make -j10
    make install
fi

cd "$REFERENCE_DIR"
rm -rf TestAscendIndexIVFPQ.cpp TestAscendIndexIVFRabitQ.cpp
sed -i "6c SET(FAISS_HOME ${FAISS_HOME}  CACHE STRING \"\")" CMakeLists.txt
sed -i "7c SET(GTEST_HOME ${GTEST_HOME}  CACHE STRING \"\")" CMakeLists.txt

# 只编译需要的测试用例
TEST_TARGETS=""
for alg in "${!ALGS_TO_PROCESS[@]}"; do
    test_cases="${ALG_TEST[$alg]}"
    for test_file in $test_cases; do
        TEST_TARGETS="$TEST_TARGETS $test_file"
    done
done

TEST_TARGETS=$(echo $TEST_TARGETS | tr ' ' '\n' | sort -u | tr '\n' ' ')  # 去重
echo "[PRESMOKE_INFO] Building test targets:$TEST_TARGETS"
bash build.sh $TEST_TARGETS

# ============== 4. 生成算子 ==============
echo "[PRESMOKE_INFO] start generate ops..."

bash /usr/local/Ascend/mxIndex/ops/custom_opp_*.run
cd $MX_INDEX_INSTALL_PATH/tools

# 检查是否是新环境（MX_INDEX_MODELPATH 不存在或为空）
if [ ! -d "$MX_INDEX_MODELPATH" ] || [ -z "$(ls -A $MX_INDEX_MODELPATH 2>/dev/null)" ]; then
    echo "[PRESMOKE_INFO] New environment detected, generating all ops..."
    # 新环境，生成所有算子
    for op in "${OPS_LIST[@]}"; do
        gen_cmd="${OP_GEN_CMD[$op]}"
        echo "[PRESMOKE_INFO] Running: $gen_cmd"
        eval "$gen_cmd" || { echo "[PRESMOKE_ERROR] Gen command failed for $op"; exit 1; }
        cp op_models/* $MX_INDEX_MODELPATH
    done
else
    echo "[PRESMOKE_INFO] Using cached ops, regenerating modified ops..."
    # 已有环境，只生成检测到的需要重新生成的算子
    for op in "${!OPS_TO_PROCESS[@]}"; do
        gen_cmd="${OP_GEN_CMD[$op]}"
        echo "[PRESMOKE_INFO] Running: $gen_cmd"
        eval "$gen_cmd" || { echo "[PRESMOKE_ERROR] Gen command failed for $op"; exit 1; }
        cp op_models/* $MX_INDEX_MODELPATH
    done
fi

# ============== 5. run test demo ==============
echo "[PRESMOKE_INFO] start run test demo..."
echo "[PRESMOKE_INFO] Algorithms to process (${#ALGS_TO_PROCESS[@]}):"

# 处理每个检测到的算法
for alg in "${!ALGS_TO_PROCESS[@]}"; do
    test_case="${ALG_TEST[$alg]}"

    echo "▶ Processing algorithm: $alg"
    echo "   - Test file : $test_case"

    # 运行测试用例
    cd "$REFERENCE_DIR"/build
    # 如果包含空格，说明有多个测试用例
    if [[ "$test_case" == *" "* ]]; then
        # 用空格分割，逐个执行
        for test_file in $test_case; do
            if [[ -x "$test_file" ]]; then
                echo "[PRESMOKE_INFO] Running test: $test_file"
                ./"$test_file"
            else
                echo "[PRESMOKE_WARNING] $test_file not found or not executable. Skipping test."
            fi
        done
    else
        if [[ -x "$test_case" ]]; then
            echo "[PRESMOKE_INFO] Running test: $test_case"
            ./"$test_case"
        else
            echo "[PRESMOKE_WARNING] $test_case not found or not executable. Skipping test."
        fi
    fi

    echo "[PRESMOKE_INFO] Finished processing $alg"
    echo ""
done

echo "[PRESMOKE_INFO] indexPreSmoke finished"
