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


#ifndef OCK_VSA_NEIGHBOR_NEIGHBOR_RELATION_SAMPLE_FEATURE_MGR_H
#define OCK_VSA_NEIGHBOR_NEIGHBOR_RELATION_SAMPLE_FEATURE_MGR_H
#include <cstdint>
#include <utility>
#include <deque>
#include <memory>
#include <vector>
#include "ock/hcps/algo/OckElasticBitSet.h"
#include "ock/hcps/handler/OckHeteroHandler.h"
#include "ock/hcps/handler/OckHandlerHmmHelper.h"
#include "ock/hmm/mgr/OckHmmHMObject.h"
#include "ock/hcps/algo/OckShape.h"

namespace ock {
namespace vsa {
namespace neighbor {
namespace hpp {
namespace relation {
template <typename DataTemp, uint64_t DimSizeTemp> struct OckVsaSampleFeatureMgr {
public:
    OckVsaSampleFeatureMgr(std::shared_ptr<hcps::handler::OckHeteroHandler> heteroHandler, uint64_t maxFeatureNum,
        uint64_t blockSize)
        : handler(heteroHandler), maxFeatureRowCount(maxFeatureNum), blockRowCount(blockSize){};

    hcps::OckHcpsErrorCode AddSampleFeature(std::shared_ptr<hmm::OckHmmHMObject> shapedFeatureInHost,
        std::shared_ptr<hmm::OckHmmHMObject> normInHost, uint32_t validateRowCount)
    {
        if (shapedFeatureInHost.get() == nullptr || normInHost.get() == nullptr) {
            return VSA_ERROR_INVALID_INPUT_PARAM;
        }
        shapedFeatureGroupInHost.push_back(shapedFeatureInHost);
        normGroupInHost.push_back(normInHost);
        groupRowCountInfo.push_back(validateRowCount);
        OCK_CHECK_RETURN_ERRORCODE(
            AddSampleFeatureNotIncludeRowCount(shapedFeatureInHost, normInHost, validateRowCount));
        return hmm::HMM_SUCCESS;
    };

    /*
    @brief 1. 弹出front  2. 根据host数据重新生成 npu数据， 默认每个group是对齐的
    */
    hcps::OckHcpsErrorCode PopFrontGroup(void)
    {
        lastBlockRowCount = 0;
        shapedFeatureBlockListInNpu.clear();
        normBlockListInNpu.clear();

        shapedFeatureGroupInHost.pop_front();
        normGroupInHost.pop_front();
        groupRowCountInfo.pop_front();
        for (uint32_t i = 0; i < shapedFeatureGroupInHost.size(); ++i) {
            OCK_CHECK_RETURN_ERRORCODE(AddSampleFeatureNotIncludeRowCount(shapedFeatureGroupInHost[i],
                normGroupInHost[i], groupRowCountInfo[i]));
        }
        return hmm::HMM_SUCCESS;
    };

    /*
    @brief 删除全是空洞的group
    */
    hcps::OckHcpsErrorCode DropEmptyGroup(uint64_t index)
    {
        lastBlockRowCount = 0;
        shapedFeatureBlockListInNpu.clear();
        normBlockListInNpu.clear();

        shapedFeatureGroupInHost.erase(shapedFeatureGroupInHost.begin() + index);
        normGroupInHost.erase(normGroupInHost.begin() + index);
        groupRowCountInfo.erase(groupRowCountInfo.begin() + index);
        for (uint32_t i = 0; i < shapedFeatureGroupInHost.size(); ++i) {
            OCK_CHECK_RETURN_ERRORCODE(AddSampleFeatureNotIncludeRowCount(shapedFeatureGroupInHost[i],
                normGroupInHost[i], groupRowCountInfo[i]));
        }
        return hmm::HMM_SUCCESS;
    };

    friend std::ostream &operator << (std::ostream &os, const OckVsaSampleFeatureMgr &data)
    {
        os << "blockRowCount:" << data.blockRowCount << ",lastBlockRowCount:" << data.lastBlockRowCount <<
            ",shapedFeatureBlockListInNpu.size(" << data.shapedFeatureBlockListInNpu.size() << ")"
           << ", groupRowCountInfo:";
        utils::PrintContainer(os, data.groupRowCountInfo);
        return os;
    }

    std::shared_ptr<hcps::handler::OckHeteroHandler> handler{ nullptr };
    uint64_t maxFeatureRowCount{ 0 };
    uint64_t blockRowCount{ 0 };
    uint32_t lastBlockRowCount{ 0 };
    std::deque<std::shared_ptr<hmm::OckHmmHMObject>> shapedFeatureBlockListInNpu{};
    std::deque<std::shared_ptr<hmm::OckHmmHMObject>> normBlockListInNpu{};
    std::deque<uint32_t> groupRowCountInfo{};
    std::deque<std::shared_ptr<hmm::OckHmmHMObject>> shapedFeatureGroupInHost{};
    std::deque<std::shared_ptr<hmm::OckHmmHMObject>> normGroupInHost{};

private:
    OckVsaErrorCode AddSampleFeatureNotIncludeRowCount(std::shared_ptr<hmm::OckHmmHMObject> shapedFeatureInHost,
        std::shared_ptr<hmm::OckHmmHMObject> normInHost, uint32_t validateRowCount)
    {
        OckVsaErrorCode errorCode = VSA_SUCCESS;

        uint32_t srcStartPos = 0UL;
        while (srcStartPos < validateRowCount) {
            if (lastBlockRowCount >= blockRowCount || shapedFeatureBlockListInNpu.empty()) {
                shapedFeatureBlockListInNpu.push_back(hcps::handler::helper::MakeDeviceHmo(*handler,
                    blockRowCount * sizeof(DataTemp) * DimSizeTemp, errorCode));
                OCK_CHECK_RETURN_ERRORCODE(errorCode);
                normBlockListInNpu.push_back(
                    hcps::handler::helper::MakeDeviceHmo(*handler, blockRowCount * sizeof(uint16_t), errorCode));
                OCK_CHECK_RETURN_ERRORCODE(errorCode);
                lastBlockRowCount = 0;
            }
            uint32_t addRowCount =
                std::min((uint32_t)(blockRowCount - lastBlockRowCount), (uint32_t)(validateRowCount - srcStartPos));
            errorCode = AddFeatureDataIntoNpuBlocks(*shapedFeatureBlockListInNpu.back(), *shapedFeatureInHost,
                srcStartPos, addRowCount, validateRowCount);
            OCK_CHECK_RETURN_ERRORCODE(errorCode);
            errorCode = AddNormDataIntoNpuBlocks(*normBlockListInNpu.back(), *normInHost, srcStartPos, addRowCount);
            OCK_CHECK_RETURN_ERRORCODE(errorCode);
            srcStartPos += addRowCount;
            lastBlockRowCount += addRowCount;
        }
        return errorCode;
    };

    OckVsaErrorCode AddFeatureDataIntoNpuBlocks(hmm::OckHmmHMObject &dstHmo, const hmm::OckHmmHMObject &srcHmo,
        uint32_t srcStartRow, uint32_t addRowCount, uint32_t validateRowCount)
    {
        OckVsaErrorCode errorCode = VSA_SUCCESS;
        OCK_VSA_HPP_LOG_INFO("AddFeatureDataIntoNpuBlocks addRowCount:" << addRowCount << ",srcStartRow:" <<
            srcStartRow << ",validateRowCount:" << validateRowCount);
        auto buffer = dstHmo.GetBuffer(hmm::OckHmmHeteroMemoryLocation::LOCAL_HOST_MEMORY, 0, dstHmo.GetByteSize());
        if (buffer == nullptr) {
            OCK_VSA_HPP_LOG_ERROR("AddFeatureDataIntoNpuBlocks GetBuffer failed");
            return VSA_ERROR_INVALID_INPUT_PARAM;
        }
        OCK_CHECK_RETURN_ERRORCODE(buffer->ErrorCode());
        hcps::algo::OckShape<DataTemp, DimSizeTemp> dstShape(buffer->Address(), dstHmo.GetByteSize(),
            lastBlockRowCount);
        hcps::algo::OckShape<DataTemp, DimSizeTemp> srcShape(srcHmo.Addr(), srcHmo.GetByteSize(), validateRowCount);
        dstShape.AddSegment(srcShape, srcStartRow, addRowCount);
        errorCode = buffer->FlushData();
        return errorCode;
    }

    OckVsaErrorCode AddNormDataIntoNpuBlocks(hmm::OckHmmHMObject &dstHmo, const hmm::OckHmmHMObject &srcHmo,
        uint32_t srcStartRow, uint32_t addRowCount)
    {
        OckVsaErrorCode errorCode = VSA_SUCCESS;
        auto buffer = dstHmo.GetBuffer(hmm::OckHmmHeteroMemoryLocation::LOCAL_HOST_MEMORY,
            lastBlockRowCount * sizeof(uint16_t), dstHmo.GetByteSize() - lastBlockRowCount * sizeof(uint16_t));
        if (buffer == nullptr) {
            OCK_VSA_HPP_LOG_ERROR("AddNormDataIntoNpuBlocks dstHmo GetBuffer failed");
            return VSA_ERROR_INVALID_INPUT_PARAM;
        }
        uint16_t *pDstData = reinterpret_cast<uint16_t *>(buffer->Address());
        uint16_t *pSrcData = reinterpret_cast<uint16_t *>(srcHmo.Addr());
        errorCode = memcpy_s(pDstData, dstHmo.GetByteSize() - lastBlockRowCount * sizeof(uint16_t),
            pSrcData + srcStartRow, addRowCount * sizeof(uint16_t));
        OCK_CHECK_RETURN_ERRORCODE(errorCode);
        errorCode = buffer->FlushData();
        return errorCode;
    }
};
} // namespace relation
} // namespace hpp
} // namespace neighbor
} // namespace vsa
} // namespace ock
#endif