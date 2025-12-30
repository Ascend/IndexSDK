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

#include "ock/log/OckHcpsLogger.h"
#include "ock/hcps/nop/l2_norm_op/OckL2NormOpDataBuffer.h"
#include "ock/hcps/nop/l2_norm_op/OckL2NormOpFactory.h"
#include "ock/hcps/nop/l2_norm_op/OckL2NormOpRun.h"
namespace ock {
namespace hcps {
namespace nop {
static void GenerateTransfer(std::vector<OckFloat16> &transfer)
{
    transfer.resize(TRANSFER_SIZE * CUBE_ALIGN, acladapter::OckAscendFp16::FloatToFp16(0));
    for (int i = 0; i < TRANSFER_SIZE / CUBE_ALIGN; ++i) {
        for (int j = 0; j < CUBE_ALIGN; ++j) {
            transfer[i * CUBE_ALIGN * CUBE_ALIGN + j * CUBE_ALIGN + j] = acladapter::OckAscendFp16::FloatToFp16(1);
        }
    }
}

OckHcpsErrorCode OckL2NormOpRun::ComputeNormSync(std::shared_ptr<OckL2NormOpHmoBlock> hmoBlock,
    handler::OckHeteroHandler &handler, std::shared_ptr<OckHeteroStreamBase> streamBase)
{
    if (hmoBlock == nullptr || streamBase == nullptr) {
        return HCPS_ERROR_INVALID_OP_INPUT_PARAM;
    }
    OckL2NormOpMeta opSpec;
    opSpec.dims = hmoBlock->dims;
    static std::shared_ptr<OckL2NormOpFactory> factory = OckL2NormOpFactory::Create(opSpec);
    OckL2NormBufferMeta bufferSpec;
    std::vector<OckFloat16> transfer;
    GenerateTransfer(transfer);
    std::vector<uint32_t> actualNum(SIZE_ALIGN, 0U);
    uint32_t offset = 0;
    uint32_t times = utils::SafeDivUp(hmoBlock->addNum, static_cast<uint32_t>(L2NORM_COMPUTE_BATCH));
    for (uint32_t i = 0; i < times; ++i) {
        uint32_t leftNumInBlock = hmoBlock->addNum - offset;
        actualNum[0] = leftNumInBlock < L2NORM_COMPUTE_BATCH ? leftNumInBlock : L2NORM_COMPUTE_BATCH;
        auto dataBuffer = OckL2NormOpDataBuffer::Create(opSpec, bufferSpec);
        auto ret = dataBuffer->AllocBuffersFromHmoBlock(hmoBlock, handler.HmmMgrPtr(), offset);
        if (ret != hmm::HMM_SUCCESS) {
            OCK_HCPS_LOG_ERROR("alloc buffers from hmoBlock failed, the errorCode is " << ret);
            return ret;
        }
        FillBuffer<OckFloat16>(dataBuffer->InputTransfer(), transfer);
        FillBuffer<uint32_t>(dataBuffer->InputActualNum(), actualNum);
        auto normOp = factory->Create(dataBuffer);
        streamBase->AddOp(normOp);
        OCK_CHECK_RETURN_ERRORCODE(streamBase->WaitExecComplete());
        offset += static_cast<uint32_t>(L2NORM_COMPUTE_BATCH);
    }
    return hmm::HMM_SUCCESS;
}

std::shared_ptr<OckL2NormOpHmoBlock> OckL2NormOpRun::BuildNormHmoBlock(std::shared_ptr<hmm::OckHmmHMObject> data,
    handler::OckHeteroHandler &handler, uint64_t dims, uint64_t addNum, OckHcpsErrorCode &errorCode)
{
    if (data == nullptr) {
        OCK_HCPS_LOG_ERROR("data is nullptr");
        return std::shared_ptr<OckL2NormOpHmoBlock>();
    }
    if (errorCode != hmm::HMM_SUCCESS) {
        return std::shared_ptr<OckL2NormOpHmoBlock>();
    }
    auto hmoBlock = std::make_shared<OckL2NormOpHmoBlock>();
    hmoBlock->dataBase = handler::helper::MakeDeviceHmo(handler,
        utils::SafeRoundUp(addNum, L2NORM_COMPUTE_BATCH) * dims * sizeof(int8_t), errorCode);
    if (errorCode != hmm::HMM_SUCCESS) {
        OCK_HCPS_LOG_ERROR("make device hmo(dataBase) failed, the errorCode is " << errorCode);
        return std::shared_ptr<OckL2NormOpHmoBlock>();
    }
    errorCode = handler.HmmMgrPtr()->CopyHMO(*hmoBlock->dataBase, 0U, *data, 0U, addNum * dims * sizeof(int8_t));
    if (errorCode != hmm::HMM_SUCCESS) {
        OCK_HCPS_LOG_ERROR("copy data from " << *data << " to " << *hmoBlock->dataBase << " failed, the errorCode is "
                           << errorCode);
        return std::shared_ptr<OckL2NormOpHmoBlock>();
    }
    hmoBlock->normResult = handler::helper::MakeDeviceHmo(handler,
        utils::SafeRoundUp(addNum, L2NORM_COMPUTE_BATCH) * sizeof(OckFloat16), errorCode);
    if (errorCode != hmm::HMM_SUCCESS) {
        OCK_HCPS_LOG_ERROR("make device hmo(normResult) failed, the errorCode is " << errorCode);
        return std::shared_ptr<OckL2NormOpHmoBlock>();
    }
    hmoBlock->dims = static_cast<uint32_t>(dims);
    hmoBlock->addNum = static_cast<uint32_t>(addNum);
    return hmoBlock;
}
}
}
}