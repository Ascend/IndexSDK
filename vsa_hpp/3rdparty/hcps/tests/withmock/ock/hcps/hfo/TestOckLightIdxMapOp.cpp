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
#include <cmath>
#include <random>
#include <vector>
#include <bitset>
#include <algorithm>
#include <gtest/gtest.h>
#include "ock/utils/OckContainerBuilder.h"
#include "ock/utils/OckCompareUtils.h"
#include "ock/utils/StrUtils.h"
#include "ock/hcps/WithEnvOckHeteroStream.h"
#include "ock/acladapter/WithEnvAclMock.h"
#include "ock/hcps/hfo/OckLightIdxMap.h"
#include "ock/hmm/mgr/MockOckHmmSingleDeviceMgr.h"

namespace ock {
namespace hcps {
namespace hfo {
namespace {
const uint64_t DIM_SIZE = 256ULL;
}
class TestOckLightIdxMapOp : public WithEnvOckHeteroStream<acladapter::WithEnvAclMock<testing::Test>> {
public:
    using BaseT = WithEnvOckHeteroStream<acladapter::WithEnvAclMock<testing::Test>>;
    using DataT = uint8_t;
    void SetUp(void) override
    {
        BaseT::SetUp();
        CPU_ZERO(&cpuSet);
        uint32_t cpuCount = sysconf(_SC_NPROCESSORS_CONF);
        for (uint32_t i = cpuCount / 2UL; i < cpuCount; ++i) {
            CPU_SET(i, &cpuSet);
        }
    }
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
    uint64_t maxCount{ 300000ULL };
    uint64_t bulcketCount{ 10000ULL };
    hmm::MockOckHmmSingleDeviceMgr hmmMgr;
};
TEST_F(TestOckLightIdxMapOp, create_add_datas_ops)
{
    OckHeteroOperatorTroupe troupe;
    auto idxMap = MakeLightIdxMap();
    uint64_t count = 500ULL;
    uint64_t *outterIdx = new uint64_t[count];
    // 设置outterIdx的值
    for (uint64_t i = 0; i < count; i++) {
        outterIdx[i] = i;
    }
    uint64_t innerStartIdx = 100ULL;
    InitStream();
    auto ret = idxMap->CreateAddDatasOps(count, outterIdx, innerStartIdx);
    troupe.push_back(ret);
    this->stream->AddOps(troupe);
    EXPECT_EQ(hmm::HMM_SUCCESS, this->stream->WaitExecComplete());
    delete[] outterIdx;
    EXPECT_EQ(idxMap->GetOutterIdx(101ULL), 1ULL);
}

TEST_F(TestOckLightIdxMapOp, set_removed_by_inner_startId)
{
    auto idxMap = MakeLightIdxMap();
    AddIdx(*idxMap, 500ULL);
    uint64_t innerStartIdx = 100ULL;
    uint64_t count = 200ULL;
    InitStream();
    auto ops = idxMap->CreateSetRemovedOps(count, innerStartIdx);
    EXPECT_EQ(this->stream->RunOps(*ops, OckStreamExecPolicy::TRY_BEST), hmm::HMM_SUCCESS);
    bool flag = true;
    for (uint64_t i = 0ULL; i < count; i++) {
        if (i < 100ULL && (idxMap->GetOutterIdx(i) != i || idxMap->GetInnerIdx(i) != i)) {
            flag = false;
            break;
        }
        if (100ULL <= i && i < 300ULL &&
            (idxMap->GetOutterIdx(i) != INVALID_IDX_VALUE || idxMap->GetInnerIdx(i) != INVALID_IDX_VALUE)) {
            flag = false;
            break;
        }
        if (300ULL <= i && i < 500ULL && (idxMap->GetOutterIdx(i) != i || idxMap->GetInnerIdx(i) != i)) {
            flag = false;
            break;
        }
    }
    EXPECT_EQ(true, flag);
}

TEST_F(TestOckLightIdxMapOp, set_removed_by_otter_idx)
{
    auto idxMap = MakeLightIdxMap();
    AddIdx(*idxMap, 500ULL);
    uint64_t count = 200ULL;
    uint64_t *outterIdx = new uint64_t[count];
    // 设置outterIdx的值
    for (uint64_t i = 0; i < count; i++) {
        outterIdx[i] = i;
    }
    InitStream();
    auto ops = idxMap->CreateSetRemovedOps(count, outterIdx);
    EXPECT_EQ(this->stream->RunOps(*ops, OckStreamExecPolicy::TRY_BEST), hmm::HMM_SUCCESS);
    delete[] outterIdx;
    bool flag = true;
    for (uint64_t i = 0ULL; i < 500ULL; i++) {
        if (i < count &&
            (idxMap->GetOutterIdx(i) != INVALID_IDX_VALUE || idxMap->GetInnerIdx(i) != INVALID_IDX_VALUE)) {
            flag = false;
            break;
        }
        if (i >= count && (idxMap->GetOutterIdx(i) != i || idxMap->GetInnerIdx(i) != i)) {
            flag = false;
            break;
        }
    }
    EXPECT_EQ(true, flag);
}

TEST_F(TestOckLightIdxMapOp, removed_front_by_inner)
{
    auto idxMap = MakeLightIdxMap();
    AddIdx(*idxMap, 500ULL);
    uint64_t count = 200ULL;
    InitStream();
    auto ops = idxMap->CreateRemoveFrontByInnerOps(count);
    EXPECT_EQ(this->stream->RunOps(*ops, OckStreamExecPolicy::TRY_BEST), hmm::HMM_SUCCESS);
    bool flag = true;
    for (uint64_t i = 0ULL; i < 500ULL; i++) {
        if (i < count &&
            (idxMap->GetOutterIdx(i) != INVALID_IDX_VALUE || idxMap->GetInnerIdx(i) != INVALID_IDX_VALUE)) {
            flag = false;
            break;
        }
        if (i >= count && (idxMap->GetOutterIdx(i) != i || idxMap->GetInnerIdx(i) != i)) {
            flag = false;
            break;
        }
    }
    EXPECT_EQ(true, flag);
}
} // namespace hfo
} // namespace hcps
} // namespace ock
