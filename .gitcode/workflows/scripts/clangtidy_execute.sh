#!/bin/bash
set -e
serviceName=$1


# 读取变更文件列表
MODIFY_FILE="change.txt"
FILE_ARRAY=()
exec 3< "${MODIFY_FILE}"
while IFS= read -r raw_line <&3 || [[ -n "${raw_line}" ]]; do
    fp=$(echo "${raw_line}" | sed -e 's/^"//' -e 's/"$//' -e 's/^[ \t]*//' -e 's/[ \t]*$//')
    if [[ -n "${fp}" ]]; then
        FILE_ARRAY+=("${fp}")
    fi
done
exec 3<&-

# 过滤只保留源码文件传入检测
CHECK_ARRAY=()
for file in "${FILE_ARRAY[@]}"; do
    if [[ "${file}" =~ [.](cpp|cc|cxx|c)$ ]]; then
        CHECK_ARRAY+=("${file}")
    fi
done

if [[ ${#CHECK_ARRAY[@]} -eq 0 ]]; then
    echo "INFO: 本次变更没有 .cpp / .cc / .cxx / .c 文件，无需代码检查，脚本退出"
    exit 0
fi

echo "===== 开始安装clang与pre-commit并执行静态检测 ====="
# 门禁固定使用 LLVM-22.1.8 的 clang-tidy；只有环境中 clang-tidy 主版本不低于 22 时才复用，
# 否则回退到下载安装固定版本，避免镜像内旧版本导致 check name 非法或结果漂移。
REQUIRED_MAJOR=22
CLANG_TIDY_MAJOR=$(clang-tidy --version 2>/dev/null | sed -nE 's/^.*version ([0-9]+)\..*/\1/p')

if command -v clang-tidy >/dev/null 2>&1 \
    && [ -n "${CLANG_TIDY_MAJOR}" ] \
    && [ "${CLANG_TIDY_MAJOR}" -ge "${REQUIRED_MAJOR}" ]; then
    echo "INFO: 环境中已有 clang-tidy ${CLANG_TIDY_MAJOR}.x（>=${REQUIRED_MAJOR}），跳过 LLVM 工具链下载"
    clang-tidy --version
else
    echo "INFO: clang-tidy 不可用，开始下载安装 LLVM 工具链"
    ARCH=$(uname -m)
    echo "检测到架构: ${ARCH}"

    # 根据架构选择包名
    if [[ "${ARCH}" == "x86_64" ]]; then
        PKG_NAME="LLVM-22.1.8-Linux-X64.tar.xz"
    elif [[ "${ARCH}" == "aarch64" ]]; then
        PKG_NAME="LLVM-22.1.8-Linux-ARM64.tar.xz"
    else
        echo "ERROR: 不支持的架构 ${ARCH}"
        exit 1
    fi

    # 下载地址
    RAW_URL="https://mindcluster.obs.cn-north-4.myhuaweicloud.com/blueImageDependency/clang-tidy/22.1.8/${PKG_NAME}"
    DOWNLOAD_URL="${RAW_URL}"

    echo "准备下载包: ${PKG_NAME}"
    echo "下载链接: ${DOWNLOAD_URL}"
    wget -c --no-check-certificate --no-host-directories "${DOWNLOAD_URL}"

    mkdir -p clang-tools
    # 解压
    tar -Jxf "${PKG_NAME}" -C ./clang-tools --strip-components=1

    # PATH：前置新下载的 clang-tools/bin，确保版本校验不达标（<22）时回退下载的 22.1.8 优先于环境旧版被使用
    export PATH=${ATOMGIT_WORKSPACE}/clang-tools/bin:$PATH

    # 验证
    clang-tidy --version
fi


python3 -m pip install --upgrade pre-commit
echo "===== precommit ====="
python3 -m pre_commit --version
cd ./${serviceName}
echo "待检测C/C++源文件总数：${#CHECK_ARRAY[@]}"
for single_file in "${CHECK_ARRAY[@]}"; do
    echo "----------------------------------------"
    ls -a
    echo "正在检测文件：${single_file}"
    cat ${single_file}

    res=$(find -name "compile_commands.json")
    if [ -z "$res" ];then
        echo "WARNING: 未找到 compile_commands.json，clang-tidy 可能头文件查找失败"
    else
        echo "找到编译数据库：$res"
    fi
    python3 -m pre_commit run clang-tidy --files "${single_file}" </dev/null
    ret=$?
    echo "查看返回码：${ret}"
    if [[ ${ret} -ne 0 ]]; then
        echo "ERROR: 文件 ${single_file} clang-tidy 检测不通过"
        exit ${ret}
    fi
done
echo "===== 全部源文件clang-tidy检测完成 ====="
