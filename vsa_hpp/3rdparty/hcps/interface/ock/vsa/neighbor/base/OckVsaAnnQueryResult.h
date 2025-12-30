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


#ifndef OCK_VSA_ANN_INDEX_QUERY_RESULT_H
#define OCK_VSA_ANN_INDEX_QUERY_RESULT_H
#include <cstdint>
#include <memory>
#include "ock/vsa/OckVsaErrorCode.h"
#include "ock/hcps/algo/OckTopNQueue.h"

namespace ock {
namespace vsa {
namespace neighbor {

template <typename DataTemp, typename KeyTraitTemp>
struct OckVsaAnnQueryResult {
    using DataT = DataTemp;
    using KeyTraitT = KeyTraitTemp;
    using KeyTypeTupleT = typename KeyTraitTemp::KeyTypeTuple;

    OckVsaAnnQueryResult(uint32_t queryNum, uint32_t topK, int64_t *outLabels, float *outDists, uint32_t *validCounts);
    void AddResult(uint32_t queryBatchPos, const std::vector<hcps::algo::FloatNode> &node);
    template <typename _DT, typename _KT>
    friend std::ostream &operator<<(std::ostream &os, const OckVsaAnnQueryResult<_DT, _KT> &data);

    const uint32_t queryCount;
    const uint32_t topk;
    int64_t *labels;
    float *distances;
    uint32_t *validNums;
};
template <typename DataTemp, typename KeyTraitTemp>
OckVsaAnnQueryResult<DataTemp, KeyTraitTemp>::OckVsaAnnQueryResult(
    uint32_t queryNum, uint32_t topK, int64_t *outLabels, float *outDists, uint32_t *validCounts)
    : queryCount(queryNum), topk(topK), labels(outLabels), distances(outDists), validNums(validCounts)
{}
template <typename DataTemp, typename KeyTraitTemp>
void OckVsaAnnQueryResult<DataTemp, KeyTraitTemp>::AddResult(
    uint32_t queryBatchPos, const std::vector<hcps::algo::FloatNode> &node)
{
    if (queryBatchPos >= queryCount) {
        OCK_HMM_LOG_ERROR("QueryBatchPos exceeds max quert count");
        return;
    }
    for (uint32_t i = 0; i < node.size(); ++i) {
        labels[queryBatchPos * topk + i] = static_cast<int64_t>(node[i].idx);
        distances[queryBatchPos * topk + i] = node[i].distance;
    }
    validNums[queryBatchPos] = static_cast<uint32_t>(node.size());
}
template <typename DataTemp, typename KeyTraitTemp>
std::ostream &operator<<(std::ostream &os, const OckVsaAnnQueryResult<DataTemp, KeyTraitTemp> &data)
{
    for (uint32_t bs = 0; bs < data.queryCount; ++bs) {
        for (uint32_t pos = 0; pos < data.validNums[bs]; ++pos) {
            os << "[" << bs << "-" << pos << "]=(idx=" << data.labels[bs * data.topk + pos]
               << ",dis=" << data.distances[bs * data.topk + pos] << ")" << std::endl;
        }
    }
    return os;
}

}  // namespace neighbor
}  // namespace vsa
}  // namespace ock
#endif