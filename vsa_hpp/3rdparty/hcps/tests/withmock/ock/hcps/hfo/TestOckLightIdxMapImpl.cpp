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

#include <cstdlib>
#include <gtest/gtest.h>
#include "ock/hcps/hfo/OckLightIdxMap.h"
#include "ock/hcps/hfo/OckOneSideIdxMap.h"
#include "ock/hmm/mgr/MockOckHmmSingleDeviceMgr.h"

namespace ock {
namespace hcps {
namespace hfo {
class TestOckLightIdxMapImpl : public testing::Test {
public:
    void SetUp(void) override
    {}
    void TearDown(void) override {}
    std::shared_ptr<OckLightIdxMap> MakeLightIdxMap(void)
    {
        return OckLightIdxMap::Create(maxCount, hmmMgr, bulcketCount);
    }
    std::shared_ptr<OckOneSideIdxMap> MakeOneSideIdxMap(void)
    {
        return OckOneSideIdxMap::Create(maxCount, hmmMgr);
    }
    uint64_t maxCount{ 10000ULL };
    uint64_t bulcketCount{ 100ULL };
    hmm::MockOckHmmSingleDeviceMgr hmmMgr;
};

TEST_F(TestOckLightIdxMapImpl, write_invalid_innerIdx)
{
    auto idxMap = MakeLightIdxMap();
    uint64_t innerIdx = maxCount;
    uint64_t outterIdx = 2312312123ULL; // 任意一个数
    idxMap->SetIdxMap(innerIdx, outterIdx);
    EXPECT_NE(idxMap->GetOutterIdx(innerIdx), outterIdx);
    EXPECT_NE(idxMap->GetInnerIdx(outterIdx), innerIdx);
    EXPECT_EQ(idxMap->GetInnerIdx(maxCount), INVALID_IDX_VALUE);
    EXPECT_EQ(idxMap->GetOutterIdx(maxCount), INVALID_IDX_VALUE);
}

TEST_F(TestOckLightIdxMapImpl, read_and_write_single_pair)
{
    auto idxMap = MakeLightIdxMap();
    uint64_t innerIdx = 3ULL;           // 一个小于maxCount的数
    uint64_t outterIdx = 2312312123ULL; // 任意一个数
    idxMap->SetIdxMap(innerIdx, outterIdx);
    EXPECT_EQ(idxMap->GetOutterIdx(innerIdx), outterIdx);
    EXPECT_EQ(idxMap->GetInnerIdx(outterIdx), innerIdx);
    EXPECT_EQ(idxMap->GetInnerIdx(maxCount), INVALID_IDX_VALUE);
    EXPECT_EQ(idxMap->GetOutterIdx(maxCount), INVALID_IDX_VALUE);
}

TEST_F(TestOckLightIdxMapImpl, read_and_write_duplicate_data)
{
    auto idxMap = MakeLightIdxMap();
    uint64_t innerIdx = 3ULL;           // 一个小于maxCount的数
    uint64_t outterIdx = 2312312123ULL; // 任意一个数
    idxMap->SetIdxMap(innerIdx, outterIdx);
    innerIdx++;
    idxMap->SetIdxMap(innerIdx, outterIdx);
    EXPECT_EQ(idxMap->GetOutterIdx(innerIdx), outterIdx);
    EXPECT_EQ(idxMap->GetInnerIdx(outterIdx), innerIdx);
    EXPECT_EQ(idxMap->GetInnerIdx(maxCount), INVALID_IDX_VALUE);
    EXPECT_EQ(idxMap->GetOutterIdx(maxCount), INVALID_IDX_VALUE);
}

TEST_F(TestOckLightIdxMapImpl, read_and_remove_single_pair)
{
    auto idxMap = MakeLightIdxMap();
    uint64_t innerIdx = 3ULL;           // 一个小于maxCount的数
    uint64_t outterIdx = 2312312123ULL; // 任意一个数
    idxMap->SetIdxMap(innerIdx, outterIdx);
    idxMap->SetIdxMap(innerIdx + 1, outterIdx + 1);
    EXPECT_EQ(idxMap->GetOutterIdx(innerIdx), outterIdx);
    EXPECT_EQ(idxMap->GetInnerIdx(outterIdx), innerIdx);
    idxMap->SetRemoved(outterIdx);
    idxMap->SetRemovedByInnerId(innerIdx + 1);
    EXPECT_EQ(idxMap->GetOutterIdx(innerIdx), INVALID_IDX_VALUE);
    EXPECT_EQ(idxMap->GetInnerIdx(outterIdx + 1), INVALID_IDX_VALUE);
}

TEST_F(TestOckLightIdxMapImpl, remove_front_by_inner)
{
    auto idxMap = MakeLightIdxMap();
    uint64_t count = 20ULL; // 一个小于maxCount的数
    for (uint64_t i = 0; i < count; i++) {
        idxMap->SetIdxMap(i, i);
    }
    EXPECT_EQ(idxMap->GetOutterIdx(count - 1), count - 1);
    EXPECT_EQ(idxMap->GetInnerIdx(count - 1), count - 1);
    idxMap->RemoveFrontByInner(10ULL);
    EXPECT_EQ(idxMap->GetOutterIdx(0ULL), INVALID_IDX_VALUE);
    EXPECT_EQ(idxMap->GetOutterIdx(maxCount), INVALID_IDX_VALUE);
    EXPECT_EQ(idxMap->GetInnerIdx(count - 1), 19ULL);
}

TEST_F(TestOckLightIdxMapImpl, read_and_get_outterIdxs)
{
    auto idxMap = MakeLightIdxMap();
    uint64_t count = 10ULL; // 一个小于maxCount的数
    for (uint64_t i = 0; i < count; i++) {
        idxMap->SetIdxMap(i, i);
    }
    std::vector<uint64_t> outterIdxs = idxMap->GetOutterIdxs(0ULL, count);
    EXPECT_EQ(outterIdxs[count - 1], idxMap->GetOutterIdx(count - 1));
    EXPECT_EQ(outterIdxs.size(), count);

    uint64_t innerStartIdx = 0ULL;
    uint64_t *pOutDataAddr = new uint64_t[count];
    idxMap->GetOutterIdxs(innerStartIdx, count, pOutDataAddr);
    EXPECT_EQ(pOutDataAddr[count - 1], idxMap->GetOutterIdx(count - 1));
    delete[] pOutDataAddr;
}

TEST_F(TestOckLightIdxMapImpl, get_outterIdxs_to_oneSideIdxMap)
{
    // GetOutterIdxs(uint64_t innerStartIdx, uint64_t count, OckOneSideIdxMap &outData)
    auto idxMap = MakeLightIdxMap();
    auto oneSideIdxMap = MakeOneSideIdxMap();
    EXPECT_EQ(oneSideIdxMap->Count(), 0ULL);
    uint64_t count = 20ULL; // 一个小于maxCount的数
    for (uint64_t i = 0; i < count; i++) {
        idxMap->SetIdxMap(i, i);
    }
    idxMap->GetOutterIdxs(0ULL, count, *oneSideIdxMap);
    EXPECT_EQ(oneSideIdxMap->GetIdx(count - 1), count - 1);
    EXPECT_EQ(oneSideIdxMap->Count(), count);
    oneSideIdxMap->Add(100ULL);
    auto otherSideIdxMap = MakeOneSideIdxMap();
    otherSideIdxMap->AddFrom(*oneSideIdxMap, 0ULL, 10ULL);
    EXPECT_EQ(oneSideIdxMap->GetIdx(1ULL), otherSideIdxMap->GetIdx(1ULL));
}

TEST_F(TestOckLightIdxMapImpl, delete_by_outter)
{
    auto idxMap = MakeLightIdxMap();
    uint64_t count = 100ULL;
    for (uint64_t i = 0; i < count; i++) {
        idxMap->SetIdxMap(i, i);
    }
    EXPECT_EQ(idxMap->GetInnerIdx(count - 1), count - 1);
    idxMap->Delete(count - 1);
    EXPECT_EQ(idxMap->GetInnerIdx(count - 1), INVALID_IDX_VALUE);
    EXPECT_FALSE(idxMap->InOutterMap(count - 1));

    idxMap->Delete(0UL);
    EXPECT_EQ(idxMap->GetInnerIdx(0UL), INVALID_IDX_VALUE);
    EXPECT_FALSE(idxMap->InOutterMap(0UL));
    EXPECT_EQ(idxMap->InnerValidSize(), 98UL);
}

TEST_F(TestOckLightIdxMapImpl, delete_by_inner)
{
    auto idxMap = MakeLightIdxMap();
    uint64_t count = 100ULL;
    for (uint64_t i = 0; i < count; i++) {
        idxMap->SetIdxMap(i, i);
    }

    EXPECT_EQ(idxMap->GetOutterIdx(5UL), 5UL);
    idxMap->DeleteByInnerId(5UL);
    EXPECT_EQ(idxMap->GetInnerIdx(5UL), INVALID_IDX_VALUE);
    EXPECT_FALSE(idxMap->InOutterMap(5UL));
    EXPECT_EQ(idxMap->InnerValidSize(), count - 1);
}

TEST_F(TestOckLightIdxMapImpl, batch_remove_by_inner)
{
    auto idxMap = MakeLightIdxMap();
    uint64_t count = 10000ULL;
    uint64_t innerStartIdx = 1000ULL;
    for (uint64_t i = 0; i < count; i++) {
        idxMap->SetIdxMap(i, i);
    }
    EXPECT_EQ(idxMap->GetOutterIdx(500ULL), 500ULL);
    idxMap->BatchRemoveByInner(innerStartIdx, 5000ULL);
    bool flag = true;
    for (uint64_t i = 0ULL; i < count; i++) {
        if (i < 1000ULL && (idxMap->GetOutterIdx(i) != i || idxMap->GetInnerIdx(i) != i)) {
            flag = false;
            break;
        }
        if (1000ULL <= i && i < 6000ULL &&
            (idxMap->GetOutterIdx(i) != INVALID_IDX_VALUE || idxMap->GetInnerIdx(i) != INVALID_IDX_VALUE)) {
            flag = false;
            break;
        }
        if (6000ULL <= i && i < 10000ULL && (idxMap->GetOutterIdx(i) != i || idxMap->GetInnerIdx(i) != i)) {
            flag = false;
            break;
        }
    }
    EXPECT_EQ(true, flag);
    EXPECT_EQ(idxMap->InnerValidSize(), 5000ULL);
}
} // namespace hfo
} // namespace hcps
} // namespace ock
