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


#include <ascenddaemon/utils/AscendOperator.h>
#include <ascenddaemon/utils/AscendUtils.h>
#include <common/utils/LogUtils.h>

namespace ascendSearch {
AscendOperator::AscendOperator(AscendOpDesc &desc)
    : opDesc(std::move(desc)),
      handle(nullptr),
      numInputs(opDesc.inputDesc.size()),
      numOutputs(opDesc.outputDesc.size())
{
}

bool AscendOperator::init()
{
    auto ret = aclopCreateHandle(opDesc.opType.c_str(),
                                 numInputs, opDesc.inputDesc.data(),
                                 numOutputs, opDesc.outputDesc.data(),
                                 opDesc.opAttr, &handle);
    if (ret != ACL_ERROR_NONE) {
        APP_LOG_ERROR("the result of aclopCreateHandle() is err, opType=%s, ret=%d.\n", opDesc.opType.c_str(), ret);
        return false;
    }
    return true;
}

AscendOperator::~AscendOperator()
{
    if (handle) {
        aclopDestroyHandle(handle);
        handle = nullptr;
    }
}

size_t AscendOperator::getInputNumDims(int index)
{
    ASCEND_THROW_IF_NOT(index >= 0);
    ASCEND_THROW_IF_NOT(index < static_cast<int>(opDesc.inputDesc.size()));
    return aclGetTensorDescNumDims(opDesc.inputDesc[index]);
}

int64_t AscendOperator::getInputDim(int index, int dimIndex)
{
    ASCEND_THROW_IF_NOT(index >= 0);
    ASCEND_THROW_IF_NOT(index < static_cast<int>(opDesc.inputDesc.size()));
    ASCEND_THROW_IF_NOT(dimIndex < static_cast<int>(getInputNumDims(index)));
#ifdef USE_ACL_INTERFACE_V2
    int64_t dimSize;
    ACL_REQUIRE_OK(aclGetTensorDescDimV2(opDesc.inputDesc[index], dimIndex, &dimSize));
    return dimSize;
#else
    return aclGetTensorDescDim(opDesc.outputDesc[index], dimIndex);
#endif
}

size_t AscendOperator::getInputSize(int index)
{
    ASCEND_THROW_IF_NOT(index >= 0);
    ASCEND_THROW_IF_NOT(index < static_cast<int>(opDesc.inputDesc.size()));
    return aclGetTensorDescSize(opDesc.inputDesc[index]);
}

size_t AscendOperator::getOutputNumDims(int index)
{
    ASCEND_THROW_IF_NOT(index >= 0);
    ASCEND_THROW_IF_NOT(index < static_cast<int>(opDesc.outputDesc.size()));
    return aclGetTensorDescNumDims(opDesc.outputDesc[index]);
}

int64_t AscendOperator::getOutputDim(int index, int dimIndex)
{
    ASCEND_THROW_IF_NOT(index >= 0);
    ASCEND_THROW_IF_NOT(index < static_cast<int>(opDesc.outputDesc.size()));
    ASCEND_THROW_IF_NOT(dimIndex < static_cast<int>(getOutputNumDims(index)));
#ifdef USE_ACL_INTERFACE_V2
    int64_t dimSize;
    ACL_REQUIRE_OK(aclGetTensorDescDimV2(opDesc.outputDesc[index], dimIndex, &dimSize));
    return dimSize;
#else
    return aclGetTensorDescDim(opDesc.outputDesc[index], dimIndex);
#endif
}

size_t AscendOperator::getOutputSize(int index)
{
    ASCEND_THROW_IF_NOT(index >= 0);
    ASCEND_THROW_IF_NOT(index < static_cast<int>(opDesc.outputDesc.size()));
    return aclGetTensorDescSize(opDesc.outputDesc[index]);
}

void AscendOperator::exec(std::vector<const aclDataBuffer *>& inputBuffers,
                          std::vector<aclDataBuffer *>& outputBuffers, aclrtStream stream) const
{
    ASCEND_THROW_IF_NOT(handle != nullptr);
    ACL_REQUIRE_OK(aclopExecWithHandle(handle,
                                       numInputs,
                                       inputBuffers.data(),
                                       numOutputs,
                                       outputBuffers.data(),
                                       stream));
}
}  // namespace ascendSearch