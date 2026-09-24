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

PROJECT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
RESULT_DIR="${PROJECT_DIR}/tmp/business-results"

source /usr/local/Ascend/ascend-toolkit/set_env.sh
export LD_LIBRARY_PATH="/usr/local/Ascend/mxIndex/host/lib:/usr/local/faiss/lib64:/usr/local/faiss/lib:/opt/OpenBLAS/lib:/usr/local/Ascend/driver/lib64/driver:${LD_LIBRARY_PATH:-}"
export MX_INDEX_MODELPATH="${PROJECT_DIR}/op_models"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$(nproc)}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-$(nproc)}"

cd "${PROJECT_DIR}"
mkdir -p "${RESULT_DIR}"
"${PROJECT_DIR}/build/image_dedup_business" --data-dir "${PROJECT_DIR}/tmp"
python3 "${PROJECT_DIR}/python/verify_business.py"
