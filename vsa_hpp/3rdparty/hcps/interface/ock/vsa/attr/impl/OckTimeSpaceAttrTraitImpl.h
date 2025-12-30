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


#ifndef OCK_VSA_TIME_SPACE_ATTR_TRAITS_IMPL_H
#define OCK_VSA_TIME_SPACE_ATTR_TRAITS_IMPL_H
#include <type_traits>
#include <algorithm>
#include "ock/utils/OckTypeTraits.h"
#include "ock/utils/OckSafeUtils.h"
#include "ock/utils/OckLimits.h"

namespace ock {
namespace vsa {
namespace attr {
inline OckTimeSpaceAttrTrait::OckTimeSpaceAttrTrait(uint32_t maxTokenCount)
    : maxTokenNumber(maxTokenCount), bitSet(hcps::algo::OckElasticBitSet{ maxTokenNumber })
{}

inline OckTimeSpaceAttrTrait OckTimeSpaceAttrTrait::InitFrom(const OckTimeSpaceAttrTrait &other)
{
    return OckTimeSpaceAttrTrait(other.maxTokenNumber);
}

inline void OckTimeSpaceAttrTrait::Add(const KeyType &value)
{
    maxTime = std::max(maxTime, value.time);
    minTime = std::min(minTime, value.time);

    maxTokenId = std::max(maxTokenId, value.tokenId);
    minTokenId = std::min(minTokenId, value.tokenId);

    bitSet.Set(value.tokenId);
}

inline void OckTimeSpaceAttrTrait::Add(const OckTimeSpaceAttrTrait &other)
{
    maxTime = std::max(maxTime, other.maxTime);
    minTime = std::min(minTime, other.minTime);

    maxTokenId = std::max(maxTokenId, other.maxTokenId);
    minTokenId = std::min(minTokenId, other.minTokenId);

    bitSet.OrWith(other.bitSet);
}

inline bool OckTimeSpaceAttrTrait::In(const KeyType &value) const
{
    bool timeIn = value.time <= maxTime && value.time >= minTime;
    bool tokenIn = (value.tokenId <= maxTokenId && value.tokenId >= minTokenId) && bitSet[value.tokenId];
    return timeIn & tokenIn;
}

inline bool OckTimeSpaceAttrTrait::Intersect(const OckTimeSpaceAttrTrait &other) const
{
    bool timeIntersect = std::min(maxTime, other.maxTime) >= std::max(minTime, other.minTime);
    bool tokenIntersect = (std::min(maxTokenId, other.maxTokenId) >= std::max(minTokenId, other.minTokenId)) &&
        bitSet.Intersect(other.bitSet);
    return timeIntersect & tokenIntersect;
}

inline double OckTimeSpaceAttrTrait::CoverageRate(const OckTimeSpaceAttrTrait &other) const
{
    int32_t timeInterMax = std::min(maxTime, other.maxTime);
    int32_t timeInterMin = std::max(minTime, other.minTime);

    if (maxTime == minTime || bitSet.Count() == 0UL) {
        return std::numeric_limits<double>::max();
    }

    double timeCoverRate = static_cast<double>(timeInterMax - timeInterMin) / static_cast<double>(maxTime - minTime);
    double tokenCoverRate =
        static_cast<double>(bitSet.IntersectCount(other.bitSet)) / static_cast<double>(bitSet.Count());

    return timeCoverRate * tokenCoverRate;
}

std::ostream &operator << (std::ostream &os, const OckTimeSpaceAttrTrait &trait)
{
    return os << "{'maxTime':" << trait.maxTime << ", 'minTime':" << trait.minTime << ", 'maxTokenId':" <<
        trait.maxTokenId << ", 'minTokenId':" << trait.minTokenId << ", 'maxTokenNumber':" << trait.maxTokenNumber <<
        "}";
}
} // namespace attr
} // namespace vsa
} // namespace ock
#endif