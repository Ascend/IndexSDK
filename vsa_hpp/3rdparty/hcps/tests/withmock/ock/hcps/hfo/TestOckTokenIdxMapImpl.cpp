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
#include "ock/hcps/hfo/OckTokenIdxMap.h"
#include "ock/hmm/mgr/MockOckHmmSingleDeviceMgr.h"

namespace ock {
namespace hcps {
namespace hfo {
class TestOckTokenIdxMapImpl : public testing::Test {
public:
    void SetUp(void) override
    {
        EXPECT_CALL(hmmMgr, AllocateHost(testing::_)).WillRepeatedly(testing::Invoke([](uint64_t byteSize) {
            return new uint8_t[byteSize];
        }));
        EXPECT_CALL(hmmMgr, DeallocateHost(testing::_, testing::_))
            .WillRepeatedly(testing::Invoke([](uint8_t *addr, size_t) { delete[] addr; }));
    }
    void TearDown(void) override
    {}
    std::shared_ptr<OckTokenIdxMap> MakeTokenIdxMap(void)
    {
        return OckTokenIdxMap::Create(maxTokenNumber, hmmMgr, grpId);
    }
    void AddRowIds(OckTokenIdxMap &idxMap, uint32_t tokenId, uint32_t fromRowId, uint32_t count)
    {
        for (uint32_t i = 0; i < count; ++i) {
            idxMap.AddData(tokenId, fromRowId + i);
        }
    }
    void ExpectSpecEqual(const OckTokenIdxMap &lhs, const OckTokenIdxMap &rhs) const
    {
        for (uint32_t i = 0; i < lhs.TokenNum(); ++i) {
            EXPECT_EQ(lhs.RowIds(i).size(), rhs.RowIds(i).size());
        }
    }
    uint32_t maxTokenNumber{100ULL};
    uint32_t tokenIdA{30UL};
    uint32_t tokenIdB{99UL};
    uint32_t grpId{10ULL};
    hmm::MockOckHmmSingleDeviceMgr hmmMgr;
};

TEST_F(TestOckTokenIdxMapImpl, groupId_rw)
{
    auto tokenIdxMap = MakeTokenIdxMap();
    EXPECT_EQ(grpId, tokenIdxMap->GroupId());
    EXPECT_EQ(maxTokenNumber, tokenIdxMap->TokenNum());
    tokenIdxMap->SetGroupId(grpId + 1);
    EXPECT_EQ(grpId + 1UL, tokenIdxMap->GroupId());
}
TEST_F(TestOckTokenIdxMapImpl, delete_data)
{
    uint32_t existsRowIdBegin = 3UL;
    uint32_t existsRowIdCount = 100UL;
    uint32_t notExistsRowId = 1000UL;
    auto tokenIdxMap = MakeTokenIdxMap();
    AddRowIds(*tokenIdxMap, tokenIdA, existsRowIdBegin, existsRowIdCount);
    AddRowIds(*tokenIdxMap, tokenIdB, existsRowIdBegin, existsRowIdCount);
    // 删除不存在的数据场景
    tokenIdxMap->DeleteData(tokenIdA, notExistsRowId);
    EXPECT_EQ(existsRowIdCount, tokenIdxMap->RowIds(tokenIdA).size());
    EXPECT_EQ(existsRowIdCount, tokenIdxMap->RowIds(tokenIdB).size());

    // 删除一个数据场景
    tokenIdxMap->DeleteData(tokenIdA, existsRowIdBegin);
    // 删除整组数据场景
    tokenIdxMap->DeleteToken(tokenIdB);
    EXPECT_EQ(existsRowIdCount - 1, tokenIdxMap->RowIds(tokenIdA).size());
    EXPECT_EQ(0UL, tokenIdxMap->RowIds(tokenIdB).size());
}
TEST_F(TestOckTokenIdxMapImpl, clear_all_data)
{
    uint32_t existsRowIdBegin = 3UL;
    uint32_t existsRowIdCount = 100UL;
    auto tokenIdxMap = MakeTokenIdxMap();
    AddRowIds(*tokenIdxMap, tokenIdA, existsRowIdBegin, existsRowIdCount);
    AddRowIds(*tokenIdxMap, tokenIdB, existsRowIdBegin, existsRowIdCount);

    tokenIdxMap->ClearAll();
    EXPECT_EQ(0UL, tokenIdxMap->RowIds(tokenIdA).size());
    EXPECT_EQ(0UL, tokenIdxMap->RowIds(tokenIdB).size());
}
TEST_F(TestOckTokenIdxMapImpl, copySpec)
{
    uint32_t existsRowIdBegin = 3UL;
    uint32_t existsRowIdCount = 100UL;
    auto tokenIdxMap = MakeTokenIdxMap();
    AddRowIds(*tokenIdxMap, tokenIdA, existsRowIdBegin, existsRowIdCount);
    auto newIdxMap = tokenIdxMap->CopySpec(hmmMgr);

    ExpectSpecEqual(*tokenIdxMap, *newIdxMap);
}

}  // namespace hfo
}  // namespace hcps
}  // namespace ock
