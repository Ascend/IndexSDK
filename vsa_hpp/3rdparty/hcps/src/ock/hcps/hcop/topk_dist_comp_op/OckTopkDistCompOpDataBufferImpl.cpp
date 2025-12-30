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

#include "ock/hcps/nop/OckOpDataBufferGen.h"
#include "ock/hcps/hcop/topk_dist_comp_op/OckTopkDistCompOpDataBuffer.h"
namespace ock {
namespace hcps {
namespace hcop {
class OckTopkDistCompOpDataBufferImpl : public nop::OckOpDataBufferGen<OckTopkDistCompOpDataBuffer> {
public:
    virtual ~OckTopkDistCompOpDataBufferImpl() noexcept = default;
    OckTopkDistCompOpDataBufferImpl(const OckTopkDistCompOpMeta &operatorSpec,
        const OckTopkDistCompBufferMeta &bufferSpecific)
        : opSpec(operatorSpec), bufferSpec(bufferSpecific)
    {
        numDistOps = GetNumDistOps(opSpec, bufferSpec);
        nop::OckTopkFlatBufferMeta topkBufferSpec;
        topkBufferSpec.k = bufferSpec.k;
        topkBufferSpec.blockNum = numDistOps;

        nop::OckDistInt8CosMaxBufferMeta distBufferSpec;
        distBufferSpec.ntotal = bufferSpec.ntotal;

        topkBuffer = nop::OckTopkFlatOpDataBuffer::Create(opSpec.ToTopkOpMeta(), topkBufferSpec);
        for (int64_t i = 0; i < numDistOps; ++i) {
            distBufferList.push_back(nop::OckDistInt8CosMaxOpDataBuffer::Create(opSpec.ToDistOpMeta(), distBufferSpec));
        }
    };
    std::shared_ptr<nop::OckTopkFlatOpDataBuffer> &GetTopkBuffer() override
    {
        return topkBuffer;
    }
    std::vector<std::shared_ptr<nop::OckDistInt8CosMaxOpDataBuffer>> &GetDistBuffers() override
    {
        return distBufferList;
    }
    OckHcpsErrorCode AllocBuffers(const std::shared_ptr<hmm::OckHmmHeteroMemoryMgrBase> &devMgr) override
    {
        if (devMgr == nullptr) {
            OCK_HCPS_LOG_ERROR("devMgr is nullptr");
            return HCPS_ERROR_INVALID_OP_INPUT_PARAM;
        }
        OCK_CHECK_RETURN_ERRORCODE(topkBuffer->AllocBuffers(devMgr));
        for (int64_t i = 0; i < numDistOps; ++i) {
            OCK_CHECK_RETURN_ERRORCODE(AllocDistBuffers(static_cast<uint32_t>(i), devMgr));
        }
        return hmm::HMM_SUCCESS;
    }
    OckHcpsErrorCode AllocBuffersFromHmoGroup(const std::shared_ptr<OckTopkDistCompOpHmoGroup> &hmoGroup,
        const std::shared_ptr<hmm::OckHmmHeteroMemoryMgrBase> &devMgr) override
    {
        if (devMgr == nullptr || hmoGroup == nullptr) {
            OCK_HCPS_LOG_ERROR("Either hmoGroup or devMgr is nullptr");
            return HCPS_ERROR_INVALID_OP_INPUT_PARAM;
        }
        OCK_CHECK_RETURN_ERRORCODE(AllocBufferFromHmo<OckFloat16>(hmoGroup->topkDistsHmo, topkBuffer->OutputDists(),
            std::vector<int64_t>{ opSpec.batch, bufferSpec.k }));
        OCK_CHECK_RETURN_ERRORCODE(AllocBufferFromHmo<int64_t>(hmoGroup->topkLabelsHmo, topkBuffer->OutputLabels(),
            std::vector<int64_t>{ opSpec.batch, bufferSpec.k }));
        for (int64_t i = 0; i < numDistOps; ++i) {
            OCK_CHECK_RETURN_ERRORCODE(AllocDistBuffers(hmoGroup, static_cast<uint32_t>(i), devMgr));
        }
        return hmm::HMM_SUCCESS;
    }
    void FillBuffers(OckTopkDistCompHostData &hostData) override
    {
        for (int64_t i = 0; i < numDistOps; ++i) {
            nop::FillBuffer<int8_t>(distBufferList[i]->InputQueries(), hostData.queries);
            nop::FillBuffer<OckFloat16>(distBufferList[i]->InputQueriesNorm(), hostData.queriesNorm);
            nop::FillBuffer<uint8_t>(distBufferList[i]->InputMask(), hostData.mask);
            if (!hostData.base.empty()) {
                nop::FillBuffer<int8_t>(distBufferList[i]->InputShaped(),
                    hostData.base[i]); // 不同i不同数据
            }
            if (!hostData.baseNorm.empty()) {
                nop::FillBuffer<OckFloat16>(distBufferList[i]->InputCodesNorm(),
                    hostData.baseNorm[i]); // 不同i不同数据
            }
        }
        nop::FillBuffer<int64_t>(topkBuffer->InputAttrs(), hostData.topkAttrs);
        nop::FillBuffer<uint32_t>(topkBuffer->InputSizes(), hostData.sizes);
        nop::FillBuffer<uint16_t>(topkBuffer->InputFlags(), hostData.outFlags);
        nop::FillBuffer<OckFloat16>(topkBuffer->OutputDists(), hostData.outDists);
        nop::FillBuffer<int64_t>(topkBuffer->OutputLabels(), hostData.outLabels);
    };
    void SetHyperParameters(std::shared_ptr<OckTopkDistCompOpHmoGroup> hmoGroup) override
    {
        if (hmoGroup == nullptr) {
            OCK_HCPS_LOG_ERROR("hmoGroup is nullptr");
            return;
        }
        PrepareHyperParameters(hmoGroup);
        nop::FillBuffer<uint32_t>(topkBuffer->InputSizes(), sizes);
        nop::FillBuffer<uint16_t>(topkBuffer->InputFlags(), outFlags);
        nop::FillBuffer<int64_t>(topkBuffer->InputAttrs(), topkAttrs);
    }
    hmm::OckHmmErrorCode PrepareHyperParameters(std::shared_ptr<OckTopkDistCompOpHmoGroup> hmoGroup) override
    {
        OCK_HCPS_LOG_DEBUG("numDistOps = " << numDistOps);
        // 原先是 * DEFAULT_PAGE_BLOCK_NUM
        int64_t pageSize = opSpec.codeBlockSize * opSpec.defaultNumDistOps;
        int64_t pageNum = utils::SafeDivUp(bufferSpec.ntotal, pageSize);
        int64_t pageOffset = bufferSpec.pageId * pageSize;
        int64_t computeNum = std::min(bufferSpec.ntotal - pageOffset, pageSize);
        int64_t idxMaskLen = utils::SafeDivUp(bufferSpec.ntotal, nop::BINARY_BYTE_SIZE);

        sizes.resize(numDistOps * nop::CORE_NUM * nop::SIZE_ALIGN);
        int64_t opSizeHostIdx = 0;
        int64_t offset = 0;
        for (int64_t i = 0; i < numDistOps; ++i) {
            sizes[opSizeHostIdx] =
                static_cast<uint32_t>(std::min(computeNum - offset, opSpec.codeBlockSize)); // opSpec.codeBlockSize;
            sizes[opSizeHostIdx + 1U] =
                static_cast<uint32_t>(pageOffset + offset); // 全局offset（当前block的首行在整个底库里面的行offset）
            sizes[opSizeHostIdx + 2U] = static_cast<uint32_t>(idxMaskLen); // 单查询的全局mask长度
            sizes[opSizeHostIdx + 3U] = hmoGroup->usingMask ? 1U : 0U;     // 是否使用mask
            opSizeHostIdx += nop::CORE_NUM * nop::SIZE_ALIGN;
            offset += opSpec.codeBlockSize;
        }

        // 初始时刻flag置零
        outFlags.resize(numDistOps * nop::FLAG_NUM * nop::FLAG_SIZE, 0U);

        topkAttrs = { 0U, /* asc */
            int64_t(bufferSpec.k),
            nop::BURST_LEN,
            int64_t(numDistOps),
            int64_t(bufferSpec.pageId),
            int64_t(pageNum),
            int64_t(pageSize),
            0U, /* quick heap */
            int64_t(opSpec.codeBlockSize) };

        return hmm::HMM_SUCCESS;
    }

private:
    OckHcpsErrorCode AllocDistBuffers(const std::shared_ptr<OckTopkDistCompOpHmoGroup> &hmoGroup, uint32_t index,
        const std::shared_ptr<hmm::OckHmmHeteroMemoryMgrBase> &devMgr)
    {
        if (index >= numDistOps) {
            OCK_HCPS_LOG_ERROR("Invalid index");
            return HCPS_ERROR_INVALID_OP_INPUT_PARAM;
        }
        // 利用hmo
        OCK_CHECK_RETURN_ERRORCODE(AllocBufferFromHmo<int8_t>(hmoGroup->queriesHmo,
            distBufferList[index]->InputQueries(), std::vector<int64_t>{ opSpec.batch, opSpec.dims }));
        OCK_CHECK_RETURN_ERRORCODE(AllocBufferFromHmo<OckFloat16>(hmoGroup->queriesNormHmo,
            distBufferList[index]->InputQueriesNorm(),
            std::vector<int64_t>{ utils::SafeRoundUp(opSpec.batch, nop::FP16_ALIGN) }));
        OCK_CHECK_RETURN_ERRORCODE(AllocBufferFromHmo<int8_t>(hmoGroup->featuresHmo[index],
            distBufferList[index]->InputShaped(),
            std::vector<int64_t>{ utils::SafeDiv(opSpec.codeBlockSize, nop::CUBE_ALIGN),
            utils::SafeDiv(opSpec.dims, nop::CUBE_ALIGN_INT8), nop::CUBE_ALIGN, nop::CUBE_ALIGN_INT8 }));
        OCK_CHECK_RETURN_ERRORCODE(AllocBufferFromHmo<OckFloat16>(hmoGroup->normsHmo[index],
            distBufferList[index]->InputCodesNorm(), std::vector<int64_t>{ opSpec.codeBlockSize }));

        int64_t idxMaskLen = utils::SafeDivUp(std::max(bufferSpec.ntotal, opSpec.codeBlockSize), nop::BINARY_BYTE_SIZE);
        if (hmoGroup->usingMask) {
            OCK_CHECK_RETURN_ERRORCODE(AllocBufferFromHmo<uint8_t>(hmoGroup->maskHMO,
                distBufferList[index]->InputMask(), std::vector<int64_t>{ opSpec.batch, idxMaskLen }));
        }
        // 不重新分配空间
        distBufferList[index]->InputActualSize() = topkBuffer->InputSizes()->SubBuffer<uint32_t>(index);
        distBufferList[index]->OutputDists() = topkBuffer->InputDists()->SubBuffer<OckFloat16>(index);
        distBufferList[index]->OutputMaxDists() = topkBuffer->InputMinDists()->SubBuffer<OckFloat16>(index);
        distBufferList[index]->OutputFlag() = topkBuffer->InputFlags()->SubBuffer<uint16_t>(index);

        return hmm::HMM_SUCCESS;
    }
    OckHcpsErrorCode AllocDistBuffers(uint32_t index, const std::shared_ptr<hmm::OckHmmHeteroMemoryMgrBase> &devMgr)
    {
        if (index >= numDistOps) {
            OCK_HCPS_LOG_ERROR("Invalid index");
            return HCPS_ERROR_INVALID_OP_INPUT_PARAM;
        }
        int64_t idxMaskLen = utils::SafeDivUp(std::max(bufferSpec.ntotal, opSpec.codeBlockSize), nop::BINARY_BYTE_SIZE);

        OCK_CHECK_RETURN_ERRORCODE(AllocBuffer<int8_t>(devMgr, distBufferList[index]->InputQueries(),
            std::vector<int64_t>{ opSpec.batch, opSpec.dims }));
        OCK_CHECK_RETURN_ERRORCODE(AllocBuffer<uint8_t>(devMgr, distBufferList[index]->InputMask(),
            std::vector<int64_t>{ opSpec.batch, idxMaskLen }));
        OCK_CHECK_RETURN_ERRORCODE(AllocBuffer<int8_t>(devMgr, distBufferList[index]->InputShaped(),
            std::vector<int64_t>{ utils::SafeDiv(opSpec.codeBlockSize, nop::CUBE_ALIGN),
            utils::SafeDiv(opSpec.dims, nop::CUBE_ALIGN_INT8), nop::CUBE_ALIGN, nop::CUBE_ALIGN_INT8 }));
        OCK_CHECK_RETURN_ERRORCODE(AllocBuffer<OckFloat16>(devMgr, distBufferList[index]->InputQueriesNorm(),
            std::vector<int64_t>{ utils::SafeRoundUp(opSpec.batch, nop::FP16_ALIGN) }));
        OCK_CHECK_RETURN_ERRORCODE(AllocBuffer<OckFloat16>(devMgr, distBufferList[index]->InputCodesNorm(),
            std::vector<int64_t>{ opSpec.codeBlockSize }));

        // 不重新分配空间
        distBufferList[index]->InputActualSize() = topkBuffer->InputSizes()->SubBuffer<uint32_t>(index);
        distBufferList[index]->OutputDists() = topkBuffer->InputDists()->SubBuffer<OckFloat16>(index);
        distBufferList[index]->OutputMaxDists() = topkBuffer->InputMinDists()->SubBuffer<OckFloat16>(index);
        distBufferList[index]->OutputFlag() = topkBuffer->InputFlags()->SubBuffer<uint16_t>(index);
        return hmm::HMM_SUCCESS;
    }
    template <typename T>
    static hmm::OckHmmErrorCode AllocBuffer(const std::shared_ptr<hmm::OckHmmHeteroMemoryMgrBase> &devMgr,
        std::shared_ptr<nop::OckDataBuffer> &paramBuffer, const std::vector<int64_t> &shape)
    {
        int64_t bytes = sizeof(T) * std::accumulate(shape.cbegin(), shape.cend(), 1LL, std::multiplies<int64_t>());
        auto hmoRet = devMgr->Alloc(bytes, hmm::OckHmmMemoryAllocatePolicy::DEVICE_DDR_ONLY);
        OCK_CHECK_RETURN_ERRORCODE(hmoRet.first);
        paramBuffer = std::make_shared<nop::OckDataBuffer>(hmoRet.second, shape);
        OCK_HCPS_LOG_DEBUG("buffer allocated with byte size = " << bytes);
        return hmm::HMM_SUCCESS;
    }

    template <typename T>
    OckHcpsErrorCode AllocBufferFromHmo(const std::shared_ptr<hmm::OckHmmSubHMObject> &hmo,
        std::shared_ptr<nop::OckDataBuffer> &paramBuffer, const std::vector<int64_t> &shape)
    {
        if (hmo == nullptr) {
            OCK_HCPS_LOG_ERROR("Input hmo is nullptr");
            return HCPS_ERROR_INVALID_OP_INPUT_PARAM;
        }
        int64_t bytes = sizeof(T) * std::accumulate(shape.cbegin(), shape.cend(), 1LL, std::multiplies<int64_t>());
        if (static_cast<int64_t>(hmo->GetByteSize()) < bytes) {
            OCK_HCPS_LOG_ERROR("hmo byte size (" << hmo->GetByteSize() << ") is smaller than the buffer size (" <<
                bytes << ")! ");
            return HCPS_ERROR_INVALID_OP_HMO_BYTE_SIZE;
        } else {
            OCK_HCPS_LOG_DEBUG("hmo byte size = " << hmo->GetByteSize() << ", buffer size = " << bytes);
            auto buffer = hmo->GetBuffer(hmm::OckHmmHeteroMemoryLocation::DEVICE_DDR, 0, bytes);
            if (buffer.get() == nullptr) {
                OCK_HCPS_LOG_ERROR("get buffer failed!");
                return HCPS_ERROR_GET_BUFFER_FAILED;
            }
            OCK_CHECK_RETURN_ERRORCODE(buffer->ErrorCode());
            paramBuffer = std::make_shared<nop::OckDataBuffer>(buffer, shape);
        }
        return hmm::HMM_SUCCESS;
    }

    OckTopkDistCompOpMeta opSpec{};
    OckTopkDistCompBufferMeta bufferSpec{};
    int64_t numDistOps{ 0 };

    std::vector<uint32_t> sizes{};
    std::vector<uint16_t> outFlags{};
    std::vector<int64_t> topkAttrs{};

    std::shared_ptr<nop::OckTopkFlatOpDataBuffer> topkBuffer{ nullptr };
    std::vector<std::shared_ptr<nop::OckDistInt8CosMaxOpDataBuffer>> distBufferList{};
};

std::shared_ptr<OckTopkDistCompOpDataBuffer> OckTopkDistCompOpDataBuffer::Create(const OckTopkDistCompOpMeta &opSpec,
    const OckTopkDistCompBufferMeta &bufferSpec)
{
    return std::make_shared<OckTopkDistCompOpDataBufferImpl>(opSpec, bufferSpec);
}
} // namespace hcop
} // namespace hcps
} // namespace ock