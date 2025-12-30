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

#include <utility>
#include "ock/hcps/error/OckHcpsErrorCode.h"
#include "ock/hcps/nop/OckNpuOp.h"
#include "ock/hcps/nop/OckNpuSingleOperator.h"
namespace ock {
namespace hcps {
namespace nop {
class OckNpuSingleOperatorImpl : public OckNpuSingleOperator {
public:
    virtual ~OckNpuSingleOperatorImpl()
    {
        auto ret = ClearDataBuffer(inputBuffer);
        ret = ClearDataBuffer(outputBuffer);
        if (ret != hmm::HMM_SUCCESS) {
            OCK_HCPS_LOG_ERROR("Fail to clear data buffer for op " << ascendOp->opType);
        }
    }
    OckNpuSingleOperatorImpl(std::shared_ptr<OckNpuOp> &ascendOperator, std::shared_ptr<OckOpDataBuffer> operatorBuffer,
        acladapter::OckTaskResourceType taskResourceType)
        : ascendOp(ascendOperator), opBuffer(operatorBuffer), resourceType(taskResourceType)
    {}

    acladapter::OckTaskResourceType ResourceType() const override
    {
        return resourceType;
    };

    hmm::OckHmmErrorCode Run(OckHeteroStreamContext &context) override
    {
        auto errCode = hmm::HMM_SUCCESS;
        inputBuffer = CreateDataBuffer<const aclDataBuffer *>(opBuffer->GetInputParams(), errCode);
        OCK_CHECK_RETURN_ERRORCODE(errCode);
        outputBuffer = CreateDataBuffer<aclDataBuffer *>(opBuffer->GetOutputParams(), errCode);
        OCK_CHECK_RETURN_ERRORCODE(errCode);

        if (ascendOp->handle == nullptr) {
            OCK_HCPS_LOG_ERROR("ascend op handler is nullptr!");
            return HCPS_ERROR_ASCEND_OP_HANDLE_NULL;
        }
        if (context.DevRtStream() == nullptr) {
            OCK_HCPS_LOG_ERROR("DevRtStream of context is nullptr!");
            return HCPS_ERROR_STREAM_NULL;
        }
        auto ret = aclopExecWithHandle(ascendOp->handle, static_cast<int>(inputBuffer.size()), inputBuffer.data(),
            static_cast<int>(outputBuffer.size()), outputBuffer.data(), context.DevRtStream());
        if (ret != hmm::HMM_SUCCESS) {
            OCK_HCPS_LOG_ERROR("Execute ascend op " << ascendOp->opType << " failed! The error code is: " << ret);
        }
        return ret;
    }

private:
    static void FillDesc(std::vector<const aclTensorDesc *> &desc, const std::vector<std::vector<int64_t>> &shapes,
        const std::vector<aclDataType> &types, const std::vector<aclFormat> &formats)
    {
        for (size_t i = 0; i < shapes.size(); ++i) {
            desc.push_back(aclCreateTensorDesc(types[i], static_cast<int>(shapes[i].size()), shapes[i].data(),
                formats[i]));
        }
    }

    template <class T>
    std::vector<T> CreateDataBuffer(const std::vector<std::shared_ptr<OckDataBuffer>> &params,
        hmm::OckHmmErrorCode &errCode)
    {
        std::vector<T> dataBuffer;
        for (const auto &param : params) {
            auto *aclBuffer = aclCreateDataBuffer(reinterpret_cast<void *>(param->Addr()), param->GetByteSize());
            if (aclBuffer == nullptr) {
                OCK_HCPS_LOG_ERROR("aclCreateDataBuffer for op " << ascendOp->opType << "  failed!");
                errCode = HCPS_ERROR_ACL_CREATE_DATA_BUFFER_FAILED;
            }
            dataBuffer.emplace_back(aclBuffer);
        }
        return dataBuffer;
    }

    template <class T> hmm::OckHmmErrorCode ClearDataBuffer(std::vector<T> &dataBuffer)
    {
        hmm::OckHmmErrorCode errCode = hmm::HMM_SUCCESS;
        for (auto &item : dataBuffer) {
            errCode = aclDestroyDataBuffer(item);
            if (errCode != hmm::HMM_SUCCESS) {
                OCK_HCPS_LOG_ERROR("ClearDataBuffer failed for op " << ascendOp->opType << "! The error code is: " <<
                    errCode);
            }
        }
        dataBuffer.clear();
        return errCode;
    }

    std::shared_ptr<OckNpuOp> ascendOp;
    std::shared_ptr<OckOpDataBuffer> opBuffer;
    std::vector<const aclDataBuffer *> inputBuffer{};
    std::vector<aclDataBuffer *> outputBuffer{};
    acladapter::OckTaskResourceType resourceType;
};

std::shared_ptr<OckHeteroOperatorBase> OckNpuSingleOperator::Create(std::shared_ptr<OckNpuOp> &ascendOp,
    std::shared_ptr<OckOpDataBuffer> opBuffer, acladapter::OckTaskResourceType resourceType)
{
    return std::make_shared<OckNpuSingleOperatorImpl>(ascendOp, opBuffer, resourceType);
}
} // namespace nop
} // namespace hcps
} // namespace ock