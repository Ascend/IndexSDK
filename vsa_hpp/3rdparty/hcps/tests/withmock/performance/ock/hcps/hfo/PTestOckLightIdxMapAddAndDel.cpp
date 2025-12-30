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
#include <chrono>
#include "ptest/ptest.h"
#include "ock/hcps/hfo/OckLightIdxMap.h"
#include "ock/hmm/mgr/MockOckHmmSingleDeviceMgr.h"

namespace ock {
namespace hcps {
namespace hfo {
class PTestOckLightIdxMapAddAndDel : public testing::Test {
public:
    void SetUp(void) override
    {}
    void TearDown(void) override
    {}
    std::shared_ptr<OckLightIdxMap> MakeLightIdxMap(void)
    {
        return OckLightIdxMap::Create(maxCount, hmmMgr, bulcketCount);
    }
    void AddIdx(OckLightIdxMap &idxMap, uint64_t count)
    {
        for (uint64_t i = 0; i < count; ++i) {
            idxMap.SetIdxMap(i, i);
        }
    }
    uint64_t maxCount{ 30000000ULL };
    uint64_t bulcketCount{ 100000ULL };
    hmm::MockOckHmmSingleDeviceMgr hmmMgr;
};

TEST_F(PTestOckLightIdxMapAddAndDel, performance_test_1_insert_data)
{
    auto idxMap = MakeLightIdxMap();
    // 测试3000万数据下插入50000条数据的性能
    uint64_t count = 50000ULL;
    auto timeGuard = fast::hdt::TestTimeGuard();
    AddIdx(*idxMap, count);
    EXPECT_TRUE(FAST_PTEST().Test("OCK.HCPS.ALGO.IDMap.Add", "UsedTime", timeGuard.ElapsedMicroSeconds()));
}

TEST_F(PTestOckLightIdxMapAddAndDel, performance_test_2_batch_remove)
{
    auto idxMap = MakeLightIdxMap();
    // 测试3000万数据下删除50000条数据的性能
    uint64_t count = 50000ULL;
    AddIdx(*idxMap, count);
    auto timeGuard = fast::hdt::TestTimeGuard();
    idxMap->RemoveFrontByInner(count);
    EXPECT_TRUE(FAST_PTEST().Test("OCK.HCPS.ALGO.IDMap.Delete", "UsedTime", timeGuard.ElapsedMicroSeconds()));
}

TEST_F(PTestOckLightIdxMapAddAndDel, performance_test_3_query_remove)
{
    auto idxMap = MakeLightIdxMap();
    // 测试3000万数据下查询50000条数据的性能
    uint64_t count = 50000ULL;
    AddIdx(*idxMap, count);
    auto timeGuard = fast::hdt::TestTimeGuard();
    for (uint64_t index = 0; index < count; ++index) {
        idxMap->GetOutterIdx(index);
        idxMap->GetInnerIdx(index);
    }
    EXPECT_TRUE(FAST_PTEST().Test("OCK.HCPS.ALGO.IDMap.Query", "UsedTime", timeGuard.ElapsedMicroSeconds()));
}
} // namespace hfo
} // namespace hcps
} // namespace ock
