/*
 * -------------------------------------------------------------------------
 * This file is part of the IndexSDK project.
 * Copyright (c) 2025 Huawei Technologies Co.,Ltd.
 *
 * IndexSDK is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *
 *          http://license.coscl.org.cn/MulanPSL2
 *
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 * -------------------------------------------------------------------------
 */


#include "npu/common/utils/AscendOpDesc.h"
#include "npu/common/utils/AscendUtils.h"

namespace ascendSearchacc {
AscendOpDesc::AscendOpDesc(std::string opName) : opType(std::move(opName))
{
    opAttr = aclopCreateAttr();
}

AscendOpDesc::AscendOpDesc(AscendOpDesc &&desc)
{
    opType = std::move(desc.opType);
    opAttr = desc.opAttr;
    desc.opAttr = nullptr;

    inputDesc = std::move(desc.inputDesc);
    outputDesc = std::move(desc.outputDesc);
}

AscendOpDesc::~AscendOpDesc()
{
    for (auto desc : inputDesc) {
        aclDestroyTensorDesc(desc);
    }

    for (auto desc : outputDesc) {
        aclDestroyTensorDesc(desc);
    }

    if (opAttr) {
        aclopDestroyAttr(opAttr);
        opAttr = nullptr;
    }
}

AscendOpDesc &AscendOpDesc::addInputTensorDesc(aclDataType dataType, int numDims, const int64_t *dims, aclFormat format)
{
    inputDesc.push_back(aclCreateTensorDesc(dataType, numDims, dims, format));
    return *this;
}

AscendOpDesc &AscendOpDesc::addOutputTensorDesc(aclDataType dataType, int numDims, const int64_t *dims,
                                                aclFormat format)
{
    outputDesc.push_back(aclCreateTensorDesc(dataType, numDims, dims, format));
    return *this;
}
}  // namespace ascendSearchacc