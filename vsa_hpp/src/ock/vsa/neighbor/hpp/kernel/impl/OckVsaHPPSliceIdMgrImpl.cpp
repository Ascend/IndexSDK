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

#include <vector>
#include "ock/vsa/neighbor/hpp/kernel/impl/OckVsaHPPSliceIdMgr.h"
#include "ock/utils/OstreamUtils.h"
namespace ock {
namespace vsa {
namespace neighbor {
namespace hpp {
namespace impl {
class OckVsaHPPSliceIdMgrImpl : public OckVsaHPPSliceIdMgr {
public:
    virtual ~OckVsaHPPSliceIdMgrImpl() noexcept = default;
    explicit OckVsaHPPSliceIdMgrImpl(uint32_t groupCount) : grpSliceIdSet(groupCount) {}
    const std::unordered_set<uint32_t> &SliceSet(uint32_t grpId) const override
    {
        if (grpId >= grpSliceIdSet.size()) {
            return emptySet;
        }
        return grpSliceIdSet.at(grpId);
    }
    std::unordered_set<uint32_t> &SliceSet(uint32_t grpId) override
    {
        if (grpId >= grpSliceIdSet.size()) {
            return emptySet;
        }
        return grpSliceIdSet.at(grpId);
    }
    bool AddSlice(uint32_t grpId, uint32_t sliceId) override
    {
        if (grpId >= grpSliceIdSet.size()) {
            return false;
        }
        auto insertResult = grpSliceIdSet.at(grpId).insert(sliceId);
        return insertResult.second;
    }
    uint32_t SliceCount(void) const override
    {
        uint32_t ret = 0UL;
        for (auto &data : grpSliceIdSet) {
            ret += static_cast<uint32_t>(data.size());
        }
        return ret;
    }
    uint32_t GroupCount(void) const override
    {
        return static_cast<uint32_t>(grpSliceIdSet.size());
    }

private:
    std::unordered_set<uint32_t> emptySet{};
    std::vector<std::unordered_set<uint32_t>> grpSliceIdSet{};
};

std::shared_ptr<OckVsaHPPSliceIdMgr> OckVsaHPPSliceIdMgr::Create(uint32_t groupCount)
{
    return std::make_shared<OckVsaHPPSliceIdMgrImpl>(groupCount);
}
std::ostream &operator << (std::ostream &os, const OckVsaHPPSliceIdMgr &data)
{
    os << "{";
    for (uint32_t grpId = 0; grpId < data.GroupCount(); ++grpId) {
        os << grpId << ":";
        utils::PrintContainer(os, data.SliceSet(grpId));
    }
    os << "}";
    return os;
}
} // namespace impl
} // namespace hpp
} // namespace neighbor
} // namespace vsa
} // namespace ock