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
readonly TOP_DIR=${CUR_DIR}/..
readonly NAME="mxindex"
readonly SDK_VERSION=$(sed -n 's/^version:[[:space:]]*//p' "${TOP_DIR}"/../ci/config/config.ini)
readonly SDK_VERSION_INFO=$(sed -n 's/^version:[[:space:]]*//p' "${TOP_DIR}"/../ci/config/config.ini)

linux_type=$(uname -i)
if [ "${linux_type}" == "aarch64" ];then
    ARCH_NAME="linux-aarch64"
else
    ARCH_NAME="linux-x86_64"
fi

ARCH_DES="${CUR_DIR}/output/${ARCH_NAME}"
RUN_FILENAME="Ascend-mindxsdk-${NAME}_${SDK_VERSION}_${ARCH_NAME}.run"
OUTPUT_DIR="${CUR_DIR}/output"
MAKESELF_SCRIPT=${TOP_DIR}/makeself/makeself.sh
MAKESELF_HEADER=${TOP_DIR}/makeself/makeself-header.sh

function package_index()
{
    mkdir -p "${ARCH_DES}"
    cd ${ARCH_DES} || exit 1

    cp -rf ${TOP_DIR}/feature_retrieval/output/*.tar.gz ./ && chmod 640 *.tar.gz

    cp ${CUR_DIR}/conf/Index/install.sh ./ && dos2unix install.sh && chmod 550 install.sh
    cp ${CUR_DIR}/conf/Index/help.info ./ && chmod 640 help.info

    {
        echo "SDK mxIndex:${SDK_VERSION}"
        echo "mxIndex version:${SDK_VERSION_INFO}"
        echo "Plat:linux ${linux_type}"
    } > version.info
    chmod 640 version.info

    if [ ! -f "$MAKESELF_SCRIPT" ] || [ ! -f "$MAKESELF_HEADER" ]; then
        echo "error: makeself.sh or makeself-header.sh not exist!"
        exit 1
    fi

    bash "$MAKESELF_SCRIPT" \
        --chown \
        --nomd5 \
        --sha256 \
        --nocrc \
        --header "$MAKESELF_HEADER" \
        --help-header help.info \
        --packaging-date '' \
        --tar-extra '--owner=root --group=root' \
        "${ARCH_DES}" \
        "${RUN_FILENAME}" \
        'ASCEND INDEX SDK RUN PACKAGE' \
        ./install.sh

    mkdir -p ${OUTPUT_DIR}
    mv ${RUN_FILENAME} ${OUTPUT_DIR}/

    echo "package complete!"
    echo "output: ${OUTPUT_DIR}/${RUN_FILENAME}"
}

package_index