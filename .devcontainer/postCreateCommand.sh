#!/bin/bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# IndexSDK is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

# .devcontainer/postCreateCommand.sh
# IndexSDK Dev Container 初始化脚本

set -e

echo "=========================================="
echo "IndexSDK Dev Container post-create setup..."
echo "=========================================="

# 1. 验证 NPU 环境
if command -v npu-smi &> /dev/null; then
    echo "NPU environment detected:"
    npu-smi info || true
else
    echo "Warning: npu-smi not found. NPU may not be available."
fi

# 2. 验证 IndexSDK 安装
if [ -d "/usr/local/Ascend/mxIndex" ]; then
    echo "IndexSDK installed successfully at /usr/local/Ascend/mxIndex"
else
    echo "Warning: IndexSDK not found at /usr/local/Ascend/mxIndex"
fi

echo "=========================================="
echo "IndexSDK Dev Container setup completed!"
echo "=========================================="
