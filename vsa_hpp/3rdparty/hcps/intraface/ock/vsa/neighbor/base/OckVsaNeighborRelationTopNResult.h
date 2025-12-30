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


#ifndef OCK_VSA_NEIGHBOR_NEIGHBOR_RELATION_TOPN_RESULT_H
#define OCK_VSA_NEIGHBOR_NEIGHBOR_RELATION_TOPN_RESULT_H
#include <cstdint>
#include <utility>
#include <deque>
#include "ock/acladapter/utils/OckAscendFp16.h"
#include "ock/log/OckHcpsLogger.h"

namespace ock {
namespace vsa {
namespace neighbor {
namespace relation {

struct OckVsaNeighborRelationTopNResult {
public:
    /*
    @brief 根据组号获取TopN结果中关于该组的ID数
    */
    const std::vector<uint32_t> &TopId(uint32_t grpId) const
    {
        return grpTopIds.at(grpId);
    };
    const std::vector<float> &TopDistance(uint32_t grpId) const
    {
        return grpTopDistances.at(grpId);
    };

    std::vector<std::vector<uint32_t>> grpTopIds{};
    std::vector<std::vector<float>> grpTopDistances{};
};

inline std::ostream &operator << (std::ostream &os, const OckVsaNeighborRelationTopNResult &data)
{
    for (uint32_t i = 0; i < data.grpTopIds.size(); ++i) {
        for (uint32_t j = 0; j < data.grpTopIds[i].size(); ++j) {
            os << "[" << i << "][" << j << "]:" << data.grpTopIds[i][j] << "," << data.grpTopDistances[i][j] <<
                std::endl;
        }
    }
    return os;
}

inline std::vector<uint32_t> BuildQuickArithmetic(const std::deque<uint32_t> &validateRowCountVec)
{
    std::vector<uint32_t> result(validateRowCountVec.size() + 1, 0);
    for (size_t i = 0; i < validateRowCountVec.size() + 1; ++i) {
        if (i > 0) {
            result[i] = result[i - 1] + validateRowCountVec[i - 1];
        } else {
            result[i] = 0;
        }
    }
    return result;
}

inline std::pair<uint32_t, uint32_t> CalcGroupPos(uint32_t label, const std::vector<uint32_t> &groupQuickArithmetic)
{
    size_t n = groupQuickArithmetic.size();
    for (size_t i = 1; i < n; i++) {
        if (label < groupQuickArithmetic[i]) {
            return std::make_pair(i - 1, label - groupQuickArithmetic[i - 1]);
        }
    }
    return std::make_pair(n - 1, label - groupQuickArithmetic[n - 1]);
}

inline std::shared_ptr<OckVsaNeighborRelationTopNResult> MakeOckVsaNeighborRelationTopNResult(
    uint32_t* topNlables, OckFloat16* topNdistance, uint32_t topN, const std::deque<uint32_t> &validateRowCountVec)
{
    if (topNlables == nullptr || topNdistance == nullptr) {
        return std::shared_ptr<OckVsaNeighborRelationTopNResult>();
    }
    // 初始化结果
    std::shared_ptr<OckVsaNeighborRelationTopNResult> result = std::make_shared<OckVsaNeighborRelationTopNResult>();
    result->grpTopIds = std::vector<std::vector<uint32_t>>(validateRowCountVec.size());
    result->grpTopDistances = std::vector<std::vector<float>>(validateRowCountVec.size());

    std::vector<uint32_t> groupQuickArithmetic = BuildQuickArithmetic(validateRowCountVec);

    // 将TopN结果转换后放入result中
    for (uint32_t i = 0; i < topN; ++i) {
        uint32_t label = topNlables[i];
        float distance = acladapter::OckAscendFp16::Fp16ToFloat(topNdistance[i]);
        std::pair<uint32_t, uint32_t> pos = CalcGroupPos(label, groupQuickArithmetic);
        result->grpTopIds[pos.first].push_back(pos.second);
        result->grpTopDistances[pos.first].push_back(distance);
    }
    return result;
};

}  // namespace relation
}  // namespace neighbor
}  // namespace vsa
}  // namespace ock
#endif