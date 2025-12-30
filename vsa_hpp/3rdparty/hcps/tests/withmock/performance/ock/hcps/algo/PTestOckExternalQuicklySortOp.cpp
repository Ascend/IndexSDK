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

#include <random>
#include <algorithm>
#include <gtest/gtest.h>
#include "ptest/ptest.h"
#include "ock/utils/StrUtils.h"
#include "ock/utils/OckCompareUtils.h"
#include "ock/hcps/WithEnvOckHeteroHandler.h"
#include "ock/hcps/error/OckHcpsErrorCode.h"
#include "ock/hcps/handler/OckHandlerHmmHelper.h"
#include "ock/hcps/algo/OckBitSet.h"
#include "ock/hcps/hfo/feature/OckHashFeatureGen.h"
#include "ock/hcps/hop/OckSplitGroupOp.h"
#include "ock/hcps/hop/OckExternalQuicklySortOp.h"
#include "ock/hcps/stream/OckHeteroStreamBase.h"

namespace ock {
namespace hcps {
namespace hop {
namespace {
const uint64_t DIM_SIZE = 256UL;
const uint64_t BIT_NUM = 16ULL;
const uint64_t PRECISIONT = (1U << BIT_NUM) - 1U;
const uint64_t DATA_COUNT = 16777216ULL; // 8G
} // namespace
class PTestOckExternalQuicklySortOp : public handler::WithEnvOckHeteroHandler<testing::Test> {
public:
    using BaseT = handler::WithEnvOckHeteroHandler<testing::Test>;
    using FromDataT = int8_t;
    using BitDataT = algo::OckBitSet<DIM_SIZE * BIT_NUM, DIM_SIZE>;
    using OckHashFeatureGenT = hfo::OckHashFeatureGen<int8_t, DIM_SIZE, BIT_NUM, PRECISIONT, BitDataT>;
    void SetUp(void) override
    {
        BaseT::SetUp();
        CPU_ZERO(&cpuSet);
        uint32_t cpuCount = sysconf(_SC_NPROCESSORS_CONF); // 96
        for (uint32_t i = cpuCount / 2UL; i < cpuCount; ++i) {
            CPU_SET(i, &cpuSet);
        }
    }
    void InitRandData(OckHeteroStreamBase &stream)
    {
        uint64_t randDataNum = DIM_SIZE + DIM_SIZE;
        randData.reserve(randDataNum);
        for (uint64_t i = 0; i < randDataNum; ++i) {
            randData.push_back(static_cast<FromDataT>(rand() % (std::numeric_limits<FromDataT>::max())));
        }

        auto ops = MakeOckSplitGroupAtmoicOpsNoReturn<uint64_t, acladapter::OckTaskResourceType::HOST_CPU>(0ULL,
            DATA_COUNT, 512ULL, [this](uint64_t k) {
                uint64_t randDataPffsetPos = rand() % DIM_SIZE;
                memcpy_s(fromData[k], DIM_SIZE, randData.data() + randDataPffsetPos, DIM_SIZE);
            });
        stream.AddOps(*ops);
        stream.WaitExecComplete();
    }
    void HashFeatureGen(OckHeteroStreamBase &stream)
    {
        auto ops = MakeOckSplitGroupAtmoicOpsNoReturn<uint64_t, acladapter::OckTaskResourceType::HOST_CPU>(0ULL,
            DATA_COUNT, 512ULL, [this](uint64_t k) { OckHashFeatureGenT::Gen(fromData[k], extFeatureDatas[k]); });
        stream.AddOps(*ops);
        stream.WaitExecComplete();
    }
    void BuildExtFeaturePtrContainer(OckHeteroStreamBase &stream)
    {
        auto ops = MakeOckSplitGroupAtmoicOpsNoReturn<uint64_t, acladapter::OckTaskResourceType::HOST_CPU>(0ULL,
            DATA_COUNT, 512ULL, [this](uint64_t k) { needSortedExtFeatureData[k] = &extFeatureDatas[k]; });
        stream.AddOps(*ops);
        stream.WaitExecComplete();
    }

    uint64_t splitThreshold{ 102400UL };
    std::vector<FromDataT> randData;
    FromDataT fromData[DATA_COUNT][DIM_SIZE];
    std::vector<BitDataT> extFeatureDatas;
    std::vector<BitDataT *> needSortedExtFeatureData;
    std::vector<BitDataT *> sortedExtFeatureData;
};
TEST_F(PTestOckExternalQuicklySortOp, external_quickly_sort_and_hash_gen)
{
    OckHcpsErrorCode errorCode = hmm::HMM_SUCCESS;
    auto handler = CreateSingleDeviceHandler(errorCode);
    auto stream = handler::helper::MakeStream(*handler, errorCode);
    InitRandData(*stream);

    extFeatureDatas = std::vector<BitDataT>(DATA_COUNT);
    needSortedExtFeatureData = std::vector<BitDataT *>(DATA_COUNT);
    sortedExtFeatureData = std::vector<BitDataT *>(DATA_COUNT);
    auto timeGuard = fast::hdt::TestTimeGuard();
    HashFeatureGen(*stream);
    EXPECT_TRUE(FAST_PTEST().Test("OCK.HCPS.HFO.HashFeature", "UsedTime", timeGuard.ElapsedMicroSeconds()));
    BuildExtFeaturePtrContainer(*stream);

    auto timeGuardSort = fast::hdt::TestTimeGuard();
    ExternalQuicklySort(*stream, needSortedExtFeatureData, sortedExtFeatureData, utils::PtrCompareAdapter(),
        splitThreshold);
    EXPECT_TRUE(FAST_PTEST().Test("OCK.HCPS.ALGO.ExternQuicklySort", "UsedTime", timeGuardSort.ElapsedMicroSeconds()));
}
} // namespace hop
} // namespace hcps
} // namespace ock