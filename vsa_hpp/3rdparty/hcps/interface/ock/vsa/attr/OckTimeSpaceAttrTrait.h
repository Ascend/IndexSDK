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


#ifndef OCK_VSA_NEIGHBOR_TIME_SPACE_ATTR_TRAIT_H
#define OCK_VSA_NEIGHBOR_TIME_SPACE_ATTR_TRAIT_H
#include "ock/vsa/attr/OckKeyTrait.h"
namespace ock {
namespace vsa {
namespace attr {
/*
@brief 时空属性，来源于当前Mindx的定义
*/
struct OckTimeSpaceAttr {
    OckTimeSpaceAttr(void): time(0), tokenId(0) {}
    OckTimeSpaceAttr(int32_t timeAttr, uint32_t spaceAttr) : time(timeAttr), tokenId(spaceAttr) {}
    int32_t time;
    uint32_t tokenId;
};

struct OckTimeSpaceAttrTrait : OckKeyTrait {
    using KeyType = OckTimeSpaceAttr;
    using KeyTypeTuple = std::tuple<OckTimeSpaceAttr>;

    explicit OckTimeSpaceAttrTrait() = default;
    explicit OckTimeSpaceAttrTrait(uint32_t maxTokenCount);
    bool operator == (const OckTimeSpaceAttrTrait &other) const
    {
        return (maxTime == other.maxTime && minTime == other.minTime && maxTokenId == other.maxTokenId &&
            minTokenId == other.minTokenId && bitSet == other.bitSet);
    }

    static OckTimeSpaceAttrTrait InitFrom(const OckTimeSpaceAttrTrait &other);

    void Add(const KeyType &value);
    void Add(const OckTimeSpaceAttrTrait &other);
    bool In(const KeyType &value) const;
    bool Intersect(const OckTimeSpaceAttrTrait &other) const;
    double CoverageRate(const OckTimeSpaceAttrTrait &other) const;
    friend inline std::ostream &operator << (std::ostream &os, const OckTimeSpaceAttrTrait &trait);

    int32_t maxTime{ utils::Limits<int32_t>::Min() };
    int32_t minTime{ utils::Limits<int32_t>::Max() };
    uint32_t maxTokenId{ utils::Limits<uint32_t>::Min() };
    uint32_t minTokenId{ utils::Limits<uint32_t>::Max() };
    uint32_t maxTokenNumber{ 2500U };
    hcps::algo::OckElasticBitSet bitSet{ maxTokenNumber };
};
} // namespace attr
} // namespace vsa
} // namespace ock
#include "ock/vsa/attr/impl/OckTimeSpaceAttrTraitImpl.h"
#endif