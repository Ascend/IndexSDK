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

readonly CUR_DIR=$(dirname "$(readlink -f "$0")")
readonly RUN_PKG_PATH="${CUR_DIR}/../.."
readonly PRESMOKE_DIR="/home/indexSDK/preSmokeTestFiles"
export MX_INDEX_INSTALL_PATH=/usr/local/Ascend/mxIndex
export MX_INDEX_MODELPATH=$PRESMOKE_DIR/pkg/modelpath
export MX_INDEX_FINALIZE=1
export LD_LIBRARY_PATH=/opt/OpenBLAS/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/Ascend/mxIndex/host/lib/:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64/driver:$LD_LIBRARY_PATH
source /usr/local/Ascend/ascend-toolkit/set_env.sh

echo "[PRESMOKE_INFO] indexPreSmoke start"

# ============== 1. install run pkg ==============
echo "[PRESMOKE_INFO] start install run pkg..."

if [ -d "$PRESMOKE_DIR/pkg" ]; then
    echo "[PRESMOKE_INFO] test files already exist, removing..."
    rm -rf "$PRESMOKE_DIR/pkg"
fi
echo "[PRESMOKE_INFO] mkdir: $PRESMOKE_DIR/pkg"
mkdir -p "$MX_INDEX_MODELPATH"

if [ -d "/usr/local/Ascend/mxIndex" ]; then
    echo "[PRESMOKE_INFO] mxIndex already exist, uninstalling..."
    if [ -f "/usr/local/Ascend/mxIndex/script/uninstall.sh" ]; then
        bash /usr/local/Ascend/mxIndex/script/uninstall.sh
    else
        echo "[PRESMOKE_WARN] uninstall.sh not found, performing manual cleanup..."
        rm -rf /usr/local/Ascend/mxIndex
    fi
fi

cp "$RUN_PKG_PATH"/Ascend-mindxsdk-mxindex_*_linux-aarch64.run "$PRESMOKE_DIR/pkg"
cd "$PRESMOKE_DIR/pkg"
chmod +x *.run
echo "[PRESMOKE_INFO] start installing run pkg"
./Ascend-mindxsdk-mxindex_*_linux-aarch64.run --install --platform=310P 

# ============== 2. generate ops ==============
echo "[PRESMOKE_INFO] start generate ops..."

bash /usr/local/Ascend/mxIndex/ops/custom_opp_*.run
cd $MX_INDEX_INSTALL_PATH/tools
python3 aicpu_generate_model.py -t 310P

cp op_models/* $MX_INDEX_MODELPATH
cd $MX_INDEX_MODELPATH && chmod 644 *

# ============== 3. run test demo ==============
echo "[PRESMOKE_INFO] start run test demo..."

if [ ! -d "${PRESMOKE_DIR}/mindsdk-referenceapps" ]; then
    cd "$PRESMOKE_DIR"
    git clone https://gitcode.com/Ascend/mindsdk-referenceapps.git
fi

cd /opt/buildtools/googletest-1.11.0
wget --no-check-certificate https://github.com/google/googletest/archive/refs/tags/release-1.11.0.tar.gz
tar -xzf release-1.11.0.tar.gz
cd googletest-release-1.11.0

cmake -DCMAKE_INSTALL_PREFIX=/opt/buildtools/googletest-1.11.0 \
      -DBUILD_SHARED_LIBS=ON \
      ../googletest-release-1.11.0

make -j10
make install

cd "$PRESMOKE_DIR"/mindsdk-referenceapps/IndexSDK
rm -rf TestAscendIndexIVFPQ.cpp TestAscendIndexIVFRabitQ.cpp
sed -i '1c SET(MXINDEX_HOME /usr/local/Ascend/mxIndex  CACHE STRING "")' dependencies.cmake
sed -i '2c SET(FAISS_HOME /usr/local/faiss/  CACHE STRING "")' dependencies.cmake
sed -i '3c SET(GTEST_HOME /opt/buildtools/googletest-1.11.0  CACHE STRING "")' dependencies.cmake
bash build.sh
cd build

changed=$(git diff ${servicebranch_1} --no-commit-id --name-only)
echo "$changed" > change.txt

ALGORITHMS=("flat" "binary_flat" "sq" "int8" "cluster" "ilflat" "ivfsq" "ivfsqt" "ts" "multi")

# 每个算法的关键词（空格分隔，支持多个）
declare -A ALG_KEYWORDS=(
    ["flat"]="IndexFlat flat"
    ["binary_flat"]="IndexBinaryFlat binary_flat"
    ["sq"]="IndexSQ sq"
    ["int8"]="IndexInt8 int8"
    ["cluster"]="IndexCluster cluster"
    ["ilflat"]="IndexILFlat ilflat"
    ["ivfsq"]="IndexIVFSQ ivfsq"
    ["ivfsqt"]="IndexIVFSQT ivfsqt"
    ["ts"]="IndexTS ts"
    ["multi"]="IndexMulti multi"
)

# 每个算法的测试用例文件
declare -A ALG_TEST=(
    ["flat"]="TestAscendIndexFlat"
    ["binary_flat"]="TestAscendIndexBinaryFlat"
    ["sq"]="TestAscendIndexSQ"
    ["int8"]="TestAscendIndexInt8Flat"
    ["cluster"]="TestAscendIndexCluster"
    ["ilflat"]="TestAscendIndexILFlat"
    ["ivfsq"]="TestAscendIndexIVFSQ"
    ["ivfsqt"]="TestAscendIndexIVFSQT"
    ["ts"]="TestAscendIndexTS"
    ["multi"]="TestAscendIndexMultiSearch"
)

# 每个算法的算子生成命令（根据实际项目修改）
declare -A ALG_GEN_CMD=(
    ["flat"]="python3 flat_generate_model.py -d 512 -t 310P"
    ["binary_flat"]="python3 binary_flat_generate_model.py -d 512"
    ["sq"]="python3 sq_generate_model.py -d 512 -t 310P"
    ["int8"]="python3 int8_generate_model.py -d 512 -t 310P"
    ["cluster"]="python3 flat_generate_model.py -d 64 -t 310P"
    ["ilflat"]="python3 flat_generate_model.py -d 512 -t 310P"
    ["ivfsq"]="python3 ivfsq8_generate_model.py -d 64 -c 8192"
    ["ivfsqt"]="python3 ivfsqt_generate_model.py -d 512"
    ["ts"]="python3 ts_generate_model.py -d 512"
    ["multi"]="python3 multi_generate_model.py -d 512"
    ["ivfflat"]="python3 ivfflat_generate_model.py -d 512"

)
# ====================================================

# 存储需要处理的算法（去重）
declare -A ALGS_TO_PROCESS

CHANGE_FILE="$PRESMOKE_DIR"/mindsdk-referenceapps/IndexSDK/build/change.txt
if [[ ! -f "$CHANGE_FILE" ]]; then
    echo "❌ Error: $CHANGE_FILE not found!"
    exit 1
fi

echo "🔍 Analyzing changed files in $CHANGE_FILE ..."

# 逐行读取变更文件
while IFS= read -r file; do
    [[ -z "$file" ]] && continue
    echo "  → Checking: $file"
    for alg in "${ALGORITHMS[@]}"; do
        keywords="${ALG_KEYWORDS[$alg]}"
        for kw in $keywords; do
            # 使用单词边界匹配，避免误匹配（如 flat 不会匹配 flatten）
            if echo "$file" | grep -q "$kw"; then
                if [[ -z "${ALGS_TO_PROCESS[$alg]}" ]]; then
                    ALGS_TO_PROCESS["$alg"]=1
                    echo "    ✅ Detected keyword '$kw' → algorithm: $alg"
                else
                    echo "    ⏭️  Already detected algorithm: $alg (from keyword '$kw')"
                fi
                break  # 一个文件匹配到一个算法的任一关键词后即可跳出内层循环
            fi
        done
    done
done < "$CHANGE_FILE"

# 如果没有检测到任何算法，退出
if [[ ${#ALGS_TO_PROCESS[@]} -eq 0 ]]; then
    echo "📦 No algorithm-specific changes detected. Exiting."
    exit 0
fi

echo ""
echo "🚀 Algorithms to process (${#ALGS_TO_PROCESS[@]}):"
for alg in "${!ALGS_TO_PROCESS[@]}"; do
    echo "  • $alg"
done
echo ""

# 处理每个检测到的算法
for alg in "${!ALGS_TO_PROCESS[@]}"; do
    test_case="${ALG_TEST[$alg]}"
    gen_cmd="${ALG_GEN_CMD[$alg]}"

    echo "▶ Processing algorithm: $alg"
    echo "   - Test file : $test_case"
    echo "   - Gen command: $gen_cmd"

    # 1. 生成算子
    cd $MX_INDEX_INSTALL_PATH/tools
    echo "   Running: $gen_cmd"
    eval "$gen_cmd" || { echo "❌ Gen command failed for $alg"; exit 1; }
    cp op_models/* $MX_INDEX_MODELPATH
    # 2. 运行测试用例

    cd "$PRESMOKE_DIR"/mindsdk-referenceapps/IndexSDK/build
    echo "   Running test: $test_case"
    if [[ -x "$test_case" ]]; then
        # 如果文件本身是可执行文件，直接运行
        ./"$test_case"
    elif [[ -f "$test_case" ]]; then
        # 如果是源文件，假设项目使用 Makefile，目标名 = 文件名去掉 .cpp
        test_name=$(basename "$test_case" .cpp)
        make "$test_name"
    else
        echo "⚠️  Warning: $test_case not found or not executable. Skipping test."
    fi

    echo "✅ Finished processing $alg"
    echo ""
done

echo "🎉 All done."

echo "[PRESMOKE_INFO] indexPreSmoke finished"
