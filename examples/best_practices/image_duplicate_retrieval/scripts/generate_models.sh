#!/usr/bin/env bash
# -------------------------------------------------------------------------
# This file is part of the IndexSDK project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
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

set -euo pipefail

PROJECT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
TOOL_DIR=/usr/local/Ascend/mxIndex/tools
NPU_TYPE="${NPU_TYPE:-910B3}"
BLOCK_SIZE="${BLOCK_SIZE:-16384}"
DIM=128
NLIST=1024
CUSTOM_OPP_VENDOR=/usr/local/Ascend/ascend-toolkit/latest/opp/vendors/mxIndex

source /usr/local/Ascend/ascend-toolkit/set_env.sh

if [[ -d "${CUSTOM_OPP_VENDOR}" ]]; then
    export ASCEND_CUSTOM_OPP_PATH="${CUSTOM_OPP_VENDOR}${ASCEND_CUSTOM_OPP_PATH:+:${ASCEND_CUSTOM_OPP_PATH}}"
    IDXSDK_IMPL="${CUSTOM_OPP_VENDOR}/op_impl/ai_core/tbe/mxIndex_impl"
    export PYTHONPATH="${IDXSDK_IMPL}/dynamic:${IDXSDK_IMPL}:${CUSTOM_OPP_VENDOR}/op_impl/ai_core/tbe:${PYTHONPATH:-}"
fi
export PYTHONPATH="${TOOL_DIR}:${PYTHONPATH:-}"

cd "${PROJECT_DIR}"
python3 "${TOOL_DIR}/aicpu_generate_model.py" -t "${NPU_TYPE}"
python3 "${TOOL_DIR}/int8flat_generate_model.py" -t "${NPU_TYPE}" -d "${DIM}" -code "${BLOCK_SIZE}"
python3 "${TOOL_DIR}/flat_generate_model.py" -t "${NPU_TYPE}" -d "${DIM}"
python3 "${TOOL_DIR}/ivfflat_generate_model.py" -t "${NPU_TYPE}" -d "${DIM}" -c "${NLIST}"
python3 "${TOOL_DIR}/ivfrabitq_generate_model.py" -t "${NPU_TYPE}" -d "${DIM}" -c "${NLIST}" -m L2
