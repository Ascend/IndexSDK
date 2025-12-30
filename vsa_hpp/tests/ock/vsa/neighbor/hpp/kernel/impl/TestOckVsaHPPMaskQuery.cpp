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

#include <gtest/gtest.h>
#include "ock/hcps/algo/OckElasticBitSet.h"
#include "ock/vsa/neighbor/hpp/kernel/impl/OckVsaHPPTopNSliceSelector.h"
#include "ock/acladapter/WithEnvAclMock.h"
#include "ock/hmm/OckHmmFactory.h"

namespace ock {
namespace vsa {
namespace neighbor {
namespace hpp {
namespace impl {
class TestOckVsaHPPMaskQuery : public acladapter::WithEnvAclMock<testing::Test> {
public:
    using BaseT = acladapter::WithEnvAclMock<testing::Test>;
    void SetUp(void) override
    {
        BaseT::SetUp();
        singleDeviceInfo = std::make_shared<hmm::OckHmmDeviceInfo>();
        singleDeviceInfo->deviceId = 0U;
        CPU_SET(1U, &singleDeviceInfo->cpuSet);                                                      // 设置1号CPU核
        CPU_SET(2U, &singleDeviceInfo->cpuSet);                                                      // 设置2号CPU核
        singleDeviceInfo->memorySpec.devSpec.maxDataCapacity = 2ULL * 1024ULL * 1024ULL * 1024ULL;   // 2G
        singleDeviceInfo->memorySpec.devSpec.maxSwapCapacity = 3ULL * 64ULL * 1024ULL * 1024ULL;     // 3 * 64M
        singleDeviceInfo->memorySpec.hostSpec.maxDataCapacity = 2ULL * 1024ULL * 1024ULL * 1024ULL;  // 2G
        singleDeviceInfo->memorySpec.hostSpec.maxSwapCapacity = 3ULL * 64ULL * 1024ULL * 1024ULL;    // 3 * 64M
        singleDeviceInfo->transferThreadNum = 2ULL;                                                  // 2个线程
        InitBitSetValue();
    }

    void TearDown(void) override
    {
        DestroyHmmDeviceMgr();  // 需要提前reset，否则打桩不生效。
        BaseT::TearDown();
    }
    void InitBitSetValue(void)
    {
        for (uint32_t i = 0; i < MASK_BIT_COUNT; ++i) {
            if (i < sliceRowCount) {
                bitSet.Set(i);  // 只是第1个slice为true
            }
        }
    }

    void BuildSingleMgr(void)
    {
        auto factory = hmm::OckHmmFactory::Create();
        auto ret = factory->CreateSingleDeviceMemoryMgr(singleDeviceInfo);
        singleMgr = ret.second;
    }

    void WriteBitSetIntoHMO(const std::shared_ptr<hmm::OckHmmHMObject> &dstHmo)
    {
        memcpy_s(reinterpret_cast<uint8_t *>(dstHmo->Addr()),
            dstHmo->GetByteSize(),
            bitSet.dataHolder,
            bitSet.WordCount() * sizeof(hcps::algo::OckElasticBitSet::WordT));
    }

    void DestroyHmmDeviceMgr(void)
    {
        if (singleMgr.get() != nullptr) {
            singleMgr.reset();
        }
    }
    const uint32_t MASK_BIT_COUNT = 248;
    hcps::algo::OckElasticBitSet bitSet{MASK_BIT_COUNT};
    std::shared_ptr<hmm::OckHmmDeviceInfo> singleDeviceInfo;
    std::shared_ptr<hmm::OckHmmSingleDeviceMgr> singleMgr;
    uint64_t fragThreshold = 2ULL * 1024ULL * 1024ULL;
    int32_t minHmoBytes = 1;
    int32_t maxHmoBytes = 3 * 64 * 1024 * 1024;
    uint32_t sliceRowCount{8UL};  // 与MASK_BIT_COUNT对应， 相对于每个slice 248/8 = 31条数据
};
TEST_F(TestOckVsaHPPMaskQuery, usingSlice)
{
    BuildSingleMgr();
    auto devHmoRet = singleMgr->Alloc(bitSet.WordCount() * sizeof(hcps::algo::OckElasticBitSet::WordT));
    auto hmo = devHmoRet.second;
    WriteBitSetIntoHMO(hmo);
    auto hmoVec = hmm::OckHmmHMObject::CreateSubHmoList(hmo, hmo->GetByteSize());

    OckVsaHPPMaskQuery query(*hmoVec, MASK_BIT_COUNT, sliceRowCount);
    EXPECT_TRUE(query.UsingSlice(0, 0UL));
    EXPECT_FALSE(query.GroupQuery(0).UsingSlice(1UL));
    EXPECT_EQ(query.UsedCount(), 8UL);
}
TEST_F(TestOckVsaHPPMaskQuery, MergeValidTags)
{
    BuildSingleMgr();
    auto devHmoRet = singleMgr->Alloc(bitSet.WordCount() * sizeof(hcps::algo::OckElasticBitSet::WordT));
    // bit 1 ~ 8 setted
    WriteBitSetIntoHMO(devHmoRet.second);
    auto hmoVec = hmm::OckHmmHMObject::CreateSubHmoList(devHmoRet.second, devHmoRet.second->GetByteSize());
    OckVsaHPPMaskQuery query(*hmoVec, MASK_BIT_COUNT, sliceRowCount);

    // 只有第2条被设置为删除
    std::deque<std::shared_ptr<hcps::algo::OckElasticBitSet>> delTags;
    delTags.push_back(std::make_shared<hcps::algo::OckElasticBitSet>(MASK_BIT_COUNT));
    delTags.back()->SetAll();
    delTags.back()->Set(1ULL, false);

    query.MergeValidTags(delTags);

    EXPECT_TRUE(query.UsingSlice(0, 0UL));
    EXPECT_FALSE(query.UsingSlice(0, 1UL));
    EXPECT_FALSE(query.GroupQuery(0).UsingSlice(1UL));
    EXPECT_EQ(query.UsedCount(), 7UL);
}
}  // namespace impl
}  // namespace hpp
}  // namespace neighbor
}  // namespace vsa
}  // namespace ock