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


#include <cstdint>
#include <algorithm>
#include "ock/utils/OckSafeUtils.h"
#include "ock/vsa/neighbor/hpp/kernel/impl/OckVsaHPPSelectRate.h"
namespace ock {
namespace vsa {
namespace neighbor {
namespace hpp {
namespace impl {
namespace {
const double SELECT_GROUP_COUNT_POW_GENE = 0.46;
const double SELECT_COV_RATE_GENE = 0.39;
const uint32_t TOPN_BASE_LINE = 200UL;
const double TOPN_BASE_SELECT_RATE_GENE = 0.02;
const double TOPN_BASE_SELECT_BASE_LINE = 1.05;
const double TOPN_BASE_SELECT_WEIGHT_BASE = 0.12;
const double TOPN_BASE_SELECT_WEIGHT_INC = 0.7;
const double TOPN_BASE_SELECT_GROUP_COUNT_POW_GENE = 0.12;
const double TOPN_BASE_SELECT_GROUP_COUNT_INC_SPEED = 8.0;
const double TOPN_BASE_SELECT_GROUP_COUNT_INC_POW_GENE = 0.1;
const double TOPN_BASE_SELECT_GROUP_COUNT_WEIGHT = 0.008;
const double TOPN_BASE_SELECT_GROUP_COUNT_SUB_WEIGHT = 5.4;
const double TOPN_BASE_SELECT_GROUP_COUNT_SUB_POW_GENE = 0.25;
const double TOPN_BASE_GROUP_SLICE_COUNT = 262144.0;
const double SELECT_TOPN_PER_GROUP_GENE = 2.0;
const double HEAD_TOPN_PER_GROUP_GENE = 3.5;
const double SELECT_DATA_RATE_THRESHOLD = 0.45;
const double PURE_SLICE_SELECT_DATA_RATE_THRESHOLD = 0.3;
const double PURE_KEY_COV_RATE_THRESHOLD = 0.2;
const uint32_t PURE_SLICE_SELECT_DATA_SIZE_THRESHOLD = 1ULL * 1024ULL * 1024ULL * 1024ULL;
const uint32_t MIN_SELECT_TOPN_FACTOR = 2UL;
const uint32_t MAX_SELECT_TOPN_PER_GROUP_FACTOR = 5UL;
const double KEY_COV_RATE_THRESHOLD = 0.007;
} // namespace

bool UsingPureFilter(double keyCovRate)
{
    return keyCovRate < KEY_COV_RATE_THRESHOLD;
}
double CalcCoverageDegree(const std::deque<std::shared_ptr<relation::OckVsaNeighborRelationHmoGroup>> &relTable,
    uint32_t groupRowCount)
{
    if (relTable.empty()) {
        return 1.0;
    }
    uint64_t relationCount = 0ULL;
    for (uint32_t i = 0; i < relTable.size(); ++i) {
        relationCount += relTable.at(i)->validateRowCount * (relation::NEIGHBOR_RELATION_COUNT_PER_CELL + 1UL);
    }
    return static_cast<double>(relationCount) / static_cast<double>((relTable.size() * groupRowCount));
}
uint32_t CalcTopNInSampleGroup(uint32_t topN, uint32_t curGroupCount, double keyCovRate, uint32_t groupRowCount,
    const std::deque<std::shared_ptr<relation::OckVsaNeighborRelationHmoGroup>> &relTable)
{
    if (keyCovRate < KEY_COV_RATE_THRESHOLD) {
        keyCovRate = KEY_COV_RATE_THRESHOLD;
    }
    double coverDegree = CalcCoverageDegree(relTable, groupRowCount);          // 盖度系数
    double groupGene = 1.0 + curGroupCount - std::sqrt((double)curGroupCount); // 组数系数
    uint32_t topNTheory =
        uint32_t(relation::SAMPLE_INTERVAL_OF_NEIGHBOR_CELL + topN * coverDegree * groupGene / keyCovRate + 0.5f);
    return std::min(std::min(std::max(topNTheory, static_cast<uint32_t>(curGroupCount * MIN_SELECT_TOPN_FACTOR)),
        static_cast<uint32_t>(curGroupCount) * MAX_SELECT_TOPN_PER_GROUP_FACTOR * topN),
        static_cast<uint32_t>(topN * curGroupCount / keyCovRate));
}
} // namespace impl
} // namespace hpp
} // namespace neighbor
} // namespace vsa
} // namespace ock