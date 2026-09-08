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
DATASET_DIR="${PROJECT_DIR}/tmp/sift100m-raw"
BASE_PATH="${DATASET_DIR}/base.first100M.u8bin"
QUERY_PATH="${DATASET_DIR}/query.public.10K.u8bin"
LEARN_PATH="${DATASET_DIR}/learn.first100K.u8bin"
BASE_URL=https://dl.fbaipublicfiles.com/billion-scale-ann-benchmarks/bigann/base.1B.u8bin
QUERY_URL=https://dl.fbaipublicfiles.com/billion-scale-ann-benchmarks/bigann/query.public.10K.u8bin
LEARN_URL=https://dl.fbaipublicfiles.com/billion-scale-ann-benchmarks/bigann/learn.100M.u8bin
BASE_LAST_BYTE=12800000007
BASE_SIZE=12800000008
QUERY_SIZE=1280008
LEARN_LAST_BYTE=12800007
LEARN_SIZE=12800008

mkdir -p "${DATASET_DIR}"

curl -fSL --http1.1 --range "0-${BASE_LAST_BYTE}" --output "${BASE_PATH}.part" "${BASE_URL}"
test "$(stat -c '%s' "${BASE_PATH}.part")" -eq "${BASE_SIZE}"
mv "${BASE_PATH}.part" "${BASE_PATH}"

curl -fSL --http1.1 --output "${QUERY_PATH}.part" "${QUERY_URL}"
test "$(stat -c '%s' "${QUERY_PATH}.part")" -eq "${QUERY_SIZE}"
mv "${QUERY_PATH}.part" "${QUERY_PATH}"

curl -fSL --http1.1 --range "0-${LEARN_LAST_BYTE}" --output "${LEARN_PATH}.part" "${LEARN_URL}"
test "$(stat -c '%s' "${LEARN_PATH}.part")" -eq "${LEARN_SIZE}"
mv "${LEARN_PATH}.part" "${LEARN_PATH}"

echo "SIFT100M is ready at ${DATASET_DIR}"
