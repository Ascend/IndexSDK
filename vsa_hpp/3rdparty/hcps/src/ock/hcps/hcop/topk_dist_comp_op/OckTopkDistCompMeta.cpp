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

#include "ock/hcps/hcop/topk_dist_comp_op/OckTopkDistCompMeta.h"
#include "ock/log/OckHcpsLogger.h"
#include "ock/utils/OckSafeUtils.h"
namespace ock {
namespace hcps {
namespace hcop {
OckTopkDistCompOpHmoGroup::OckTopkDistCompOpHmoGroup(bool isUsingMask, uint32_t batchSize, uint32_t featureDims,
    uint32_t topK, uint32_t blockRowCount, uint32_t blockNum, uint32_t totalNum, uint32_t groupIdx)
    : usingMask(isUsingMask),
      batch(batchSize),
      dims(featureDims),
      k(topK),
      blockSize(blockRowCount),
      defaultNumBlocks(blockNum),
      ntotal(totalNum),
      groupId(groupIdx)
{}

void OckTopkDistCompOpHmoGroup::PushDataBase(std::shared_ptr<hmm::OckHmmHMObject> feature,
    std::shared_ptr<hmm::OckHmmHMObject> norm)
{
    if (feature == nullptr || norm == nullptr) {
        return;
    }
    this->featuresHmo.emplace_back(feature);
    this->normsHmo.emplace_back(norm);
}

void OckTopkDistCompOpHmoGroup::SetQueryHmos(std::shared_ptr<hmm::OckHmmSubHMObject> queryData,
    std::shared_ptr<hmm::OckHmmSubHMObject> queryNorm, std::shared_ptr<hmm::OckHmmSubHMObject> mask)
{
    if (queryData == nullptr || queryNorm == nullptr) {
        return;
    }
    this->queriesHmo = queryData;
    this->queriesNormHmo = queryNorm;
    this->maskHMO = mask;
}

void OckTopkDistCompOpHmoGroup::SetOutputHmos(std::shared_ptr<hmm::OckHmmSubHMObject> topKDists,
    std::shared_ptr<hmm::OckHmmSubHMObject> topKLabels)
{
    if (topKDists == nullptr || topKLabels == nullptr) {
        return;
    }
    this->topkDistsHmo = topKDists;
    this->topkLabelsHmo = topKLabels;
}

int64_t GetNumDistOps(const OckTopkDistCompOpMeta &opSpec, const OckTopkDistCompBufferMeta &bufferSpec)
{
    if (opSpec.codeBlockSize == 0) {
        OCK_HCPS_LOG_ERROR("codeBlockSize is 0");
        return 0;
    }
    int64_t pageSize = opSpec.codeBlockSize * opSpec.defaultNumDistOps;
    int64_t pageOffset = bufferSpec.pageId * pageSize;
    int64_t computeNum = std::min(bufferSpec.ntotal - pageOffset, pageSize);
    return utils::SafeDivUp(computeNum, opSpec.codeBlockSize);
}

void UpdateMetaFromHmoGroup(OckTopkDistCompOpMeta &opSpec, OckTopkDistCompBufferMeta &bufferSpec,
    const std::shared_ptr<OckTopkDistCompOpHmoGroup> &hmoGroup)
{
    if (hmoGroup == nullptr) {
        return;
    }
    opSpec.defaultNumDistOps = static_cast<int64_t>(hmoGroup->defaultNumBlocks);
    opSpec.batch = static_cast<int64_t>(hmoGroup->batch);
    opSpec.codeBlockSize = static_cast<int64_t>(hmoGroup->blockSize);
    opSpec.dims = static_cast<int64_t>(hmoGroup->dims);
    bufferSpec.k = static_cast<int64_t>(hmoGroup->k);
    bufferSpec.ntotal = static_cast<int64_t>(hmoGroup->ntotal);
    bufferSpec.pageId = static_cast<int64_t>(hmoGroup->groupId);
    auto numDistOps = GetNumDistOps(opSpec, bufferSpec);
    if (numDistOps != static_cast<int64_t>(hmoGroup->featuresHmo.size())) {
        OCK_HCPS_LOG_WARN("numDistOps (" << numDistOps << ") does not agree with hmoGroup->featuresHmo.size()(" <<
            static_cast<int64_t>(hmoGroup->featuresHmo.size()) << ")!");
    }
}
} // namespace hcop
} // namespace hcps
} // namespace ock