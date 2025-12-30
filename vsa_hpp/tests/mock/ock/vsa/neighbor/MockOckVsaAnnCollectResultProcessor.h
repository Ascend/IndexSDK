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

#ifndef OCK_MOCK_VSA_ANN_COLLECT_RESULT_PROCESSOR_H
#define OCK_MOCK_VSA_ANN_COLLECT_RESULT_PROCESSOR_H
#include <gtest/gtest.h>
#include "ptest/ptest.h"
#include "ock/vsa/neighbor/base/OckVsaAnnCollectResultProcessor.h"

namespace ock {
namespace vsa {
namespace neighbor {
namespace hpp {

template <typename DataT, uint32_t DimSizeT>
class MockOckVsaAnnCollectResultProcessor : public adapter::OckVsaAnnCollectResultProcessor<DataT, DimSizeT> {
public:
    MockOckVsaAnnCollectResultProcessor(uint32_t topK, uint32_t groupRowCount, uint32_t groupCount)
        : topK(topK), groupRowCount(groupRowCount), groupCount(groupCount)
    {}
    OckVsaErrorCode Init(void) override
    {
        return hmm::HMM_SUCCESS;
    }
    void NotifyResult(std::shared_ptr<adapter::OckVsaAnnFeature> feature,
        std::shared_ptr<hcps::hfo::OckOneSideIdxMap> idxMap, bool usingMask, OckVsaErrorCode &errorCode) override
    {
        idxMap = idxMap;
    }
    std::vector<hcps::algo::FloatNode> NotifyResultEnd(OckVsaErrorCode &errorCode) override
    {
        if (idxMap.get() == nullptr) {
            return std::vector<hcps::algo::FloatNode>();
        }
        OCK_VSA_HPP_LOG_DEBUG("NotifyResultEnd topK=" << topK);
        auto ret = std::vector<hcps::algo::FloatNode>(topK);
        for (uint32_t i = 0; i < topK && i < idxMap->Count(); ++i) {
            ret.emplace_back(idxMap->GetIdx(i), i * 0.001f);
        }
        return ret;
    }
    std::shared_ptr<std::vector<std::vector<hcps::algo::FloatNode>>> GetTopNResults(
        std::shared_ptr<adapter::OckVsaAnnFeatureSet> featureSet, uint32_t topN, OckVsaErrorCode &errorCode) override
    {
        auto ret = std::make_shared<std::vector<std::vector<hcps::algo::FloatNode>>>(groupCount);
        for (uint32_t grpId = 0; grpId < groupCount; ++grpId) {
            for (uint32_t i = 0; i < topN; ++i) {
                ret->at(grpId).emplace_back(i, i * 0.001f);
            }
        }
        return ret;
    }
    std::shared_ptr<::ock::vsa::neighbor::relation::OckVsaNeighborRelationTopNResult> GetSampleCellTopNResult(
        const relation::OckVsaSampleFeatureMgr<DataT, DimSizeT> &sampleFeature, uint32_t topK,
        OckVsaErrorCode &errorCode) override
    {
        auto ret = std::make_shared<::ock::vsa::neighbor::relation::OckVsaNeighborRelationTopNResult>();
        ret->grpTopIds = std::vector<std::vector<uint32_t>>(groupCount);
        ret->grpTopDistances = std::vector<std::vector<float>>(groupCount);
        for (uint32_t grpId = 0; grpId < groupCount; ++grpId) {
            for (uint32_t i = 0; i < topK; ++i) {
                ret->grpTopIds[grpId].push_back(i);
                ret->grpTopDistances[grpId].push_back(i * 0.01f);
            }
        }
        return ret;
    }
    uint32_t topK;
    uint32_t groupRowCount;
    uint32_t groupCount;
    std::shared_ptr<hcps::hfo::OckOneSideIdxMap> idxMap;
};
}  // namespace hpp
}  // namespace neighbor
}  // namespace vsa
}  // namespace ock
#endif