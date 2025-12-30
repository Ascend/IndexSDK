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

#include "ock/vsa/neighbor/hpp/kernel/impl/OckVsaHPPMaskQuery.h"
#include "ock/log/OckVsaHppLogger.h"
namespace ock {
namespace vsa {
namespace neighbor {
namespace hpp {
namespace impl {
class OckVsaHPPMaskQueryImpl : public OckVsaHPPGroupMaskBase {
public:
    virtual ~OckVsaHPPMaskQueryImpl() noexcept = default;
    explicit OckVsaHPPMaskQueryImpl(std::shared_ptr<hcps::algo::OckRefBitSet> refBitSet,
        std::shared_ptr<hmm::OckHmmHMOBuffer> bufferData, const uint32_t sliceSize)
        : sliceRowCount(sliceSize), bitSet(refBitSet), buffer(bufferData)
    {}
    bool UsingSlice(uint32_t sliceId) const override
    {
        return bitSet->HasSetBit(sliceId * sliceRowCount, sliceRowCount);
    }
    void MergeValidTags(const hcps::algo::OckElasticBitSet &delTags) override
    {
        bitSet->AndWith(delTags);
    }
    uint64_t UsedCount(void) const override
    {
        return bitSet->Count();
    }
    const hcps::algo::OckRefBitSet &BitSet(void) const override
    {
        return *bitSet;
    }

private:
    const uint32_t sliceRowCount{ 0 };
    std::shared_ptr<hcps::algo::OckRefBitSet> bitSet{ nullptr };
    std::shared_ptr<hmm::OckHmmHMOBuffer> buffer{ nullptr };
};
OckVsaHPPMaskQuery::OckVsaHPPMaskQuery(const std::vector<std::shared_ptr<hmm::OckHmmSubHMObject>> &maskDatas,
    uint32_t groupRowCount, uint32_t sliceRowCount)
{
    for (auto &hmo : maskDatas) {
        auto buffer = hmo->GetBuffer(hmm::OckHmmHeteroMemoryLocation::LOCAL_HOST_MEMORY, 0, hmo->GetByteSize());
        if (buffer == nullptr) {
            OCK_VSA_HPP_LOG_ERROR("maskData GetBuffer failed");
            return;
        }
        groupData.push_back(std::make_shared<OckVsaHPPMaskQueryImpl>(
            std::make_shared<hcps::algo::OckRefBitSet>(groupRowCount, reinterpret_cast<uint64_t *>(buffer->Address())),
            buffer, sliceRowCount));
    }
}
bool OckVsaHPPMaskQuery::UsingSlice(uint32_t grpPos, uint32_t sliceId) const
{
    if (grpPos >= groupData.size()) {
        return false;
    }
    return groupData.at(grpPos)->UsingSlice(sliceId);
}
const OckVsaHPPGroupMaskBase &OckVsaHPPMaskQuery::GroupQuery(uint32_t grpPos) const
{
    return *(groupData.at(grpPos));
}
void OckVsaHPPMaskQuery::MergeValidTags(const std::deque<std::shared_ptr<hcps::algo::OckElasticBitSet>> &delTags)
{
    for (uint32_t i = 0; i < groupData.size() && i < delTags.size(); ++i) {
        groupData[i]->MergeValidTags(*delTags[i]);
    }
}
uint64_t OckVsaHPPMaskQuery::UsedCount(void) const
{
    uint64_t ret = 0ULL;
    for (uint32_t i = 0; i < groupData.size(); ++i) {
        ret += groupData[i]->UsedCount();
    }
    return ret;
}
} // namespace impl
} // namespace hpp
} // namespace neighbor
} // namespace vsa
} // namespace ock