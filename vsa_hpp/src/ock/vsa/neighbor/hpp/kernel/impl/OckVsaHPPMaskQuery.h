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


#ifndef OCK_VSA_HETERO_PIECEWISE_PROGRESSIVE_MASK_QUERY_H
#define OCK_VSA_HETERO_PIECEWISE_PROGRESSIVE_MASK_QUERY_H
#include <cstdint>
#include <utility>
#include <vector>
#include <deque>
#include <securec.h>
#include "ock/hmm/mgr/OckHmmHMObject.h"
#include "ock/hcps/algo/OckElasticBitSet.h"
namespace ock {
namespace vsa {
namespace neighbor {
namespace hpp {
namespace impl {
class OckVsaHPPGroupMaskBase {
public:
    virtual ~OckVsaHPPGroupMaskBase() noexcept = default;
    virtual bool UsingSlice(uint32_t sliceId) const = 0;
    virtual void MergeValidTags(const hcps::algo::OckElasticBitSet &delTags) = 0;
    virtual uint64_t UsedCount(void) const = 0;
    virtual const hcps::algo::OckRefBitSet &BitSet(void) const = 0;
};
class OckVsaHPPMaskQuery {
public:
    OckVsaHPPMaskQuery(const std::vector<std::shared_ptr<hmm::OckHmmSubHMObject>> &maskDatas, uint32_t groupRowCount,
        uint32_t sliceRowCount);
    bool UsingSlice(uint32_t grpPos, uint32_t sliceId) const;
    const OckVsaHPPGroupMaskBase &GroupQuery(uint32_t grpPos) const;
    void MergeValidTags(const std::deque<std::shared_ptr<hcps::algo::OckElasticBitSet>> &delTags);
    uint64_t UsedCount(void) const;

private:
    std::vector<std::shared_ptr<OckVsaHPPGroupMaskBase>> groupData{};
};
} // namespace impl
} // namespace hpp
} // namespace neighbor
} // namespace vsa
} // namespace ock
#endif