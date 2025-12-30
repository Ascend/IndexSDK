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
#include "ock/hcps/hop/OckQuicklySortOp.h"
#include "ock/hcps/hop/OckMergeSortOp.h"
#include "ock/hcps/WithEnvOckHeteroStream.h"
#include "ock/acladapter/WithEnvAclMock.h"

namespace ock {
namespace hcps {
namespace hop {
namespace {
const uint64_t DIM_SIZE = 256ULL;
}
class TestOckSortOp : public WithEnvOckHeteroStream<acladapter::WithEnvAclMock<testing::Test>> {
public:
    using BaseT = WithEnvOckHeteroStream<acladapter::WithEnvAclMock<testing::Test>>;
    using DataT = uint8_t;
    using FeatureT = uint32_t;
    using FeatureVecT = std::vector<uint32_t>;
    void SetUp(void) override
    {
        BaseT::SetUp();
        CPU_ZERO(&cpuSet);
        uint32_t cpuCount = sysconf(_SC_NPROCESSORS_CONF);
        for (uint32_t i = cpuCount / 2UL; i < cpuCount; ++i) {
            CPU_SET(i, &cpuSet);
        }
    }
    std::shared_ptr<FeatureVecT> BuildADatas(uint64_t dataCount = 1024UL * 1024ULL * 4ULL)
    {
        auto ret = std::make_shared<FeatureVecT>(dataCount);
        for (uint64_t i = 0; i < dataCount; ++i) {
            ret->at(i) = static_cast<uint32_t>(rand());
        }
        return ret;
    }

    void BuildDatas(uint64_t dataCountOfA, uint64_t dataCountOfB = 0ULL)
    {
        dataAHolder = BuildADatas(dataCountOfA);
        dataA.reserve(dataCountOfA);
        utils::BuildPtrContainer(*dataAHolder, dataA);

        dataBHolder = BuildADatas(dataCountOfB);
        dataB.reserve(dataCountOfB);
        utils::BuildPtrContainer(*dataBHolder, dataB);

        stdMerged = std::vector<FeatureT *>(dataCountOfA + dataCountOfB);
        quicklyMerged = std::vector<FeatureT *>(dataCountOfA + dataCountOfB);
    }

    void CompareSortResult(void)
    {
        for (uint64_t i = 0; i < stdMerged.size(); ++i) {
            EXPECT_TRUE(*stdMerged[i] == *quicklyMerged[i]);
        }
    }
    void MergeByStdFunction(const cpu_set_t &cpuSet)
    {
        (void)pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuSet);
        std::merge(
            dataA.begin(), dataA.end(), dataB.begin(), dataB.end(), stdMerged.begin(), utils::LessCompareAdapter());
    }
    std::shared_ptr<FeatureVecT> dataAHolder;
    std::shared_ptr<FeatureVecT> dataBHolder;
    std::vector<FeatureT *> dataB;
    std::vector<FeatureT *> dataA;
    std::vector<FeatureT *> stdMerged;
    std::vector<FeatureT *> quicklyMerged;
};
TEST_F(TestOckSortOp, QuicklyMerge)
{
    const uint64_t leftCount = 1024ULL;
    const uint64_t rightCount = 1024ULL;
    BuildDatas(leftCount, rightCount);
    std::sort(dataA.begin(), dataA.end(), utils::LessCompareAdapter());
    std::sort(dataB.begin(), dataB.end(), utils::LessCompareAdapter());

    MergeByStdFunction(cpuSet);

    this->transferThreadNum = 20UL;
    InitStream();
    EXPECT_EQ(utils::ToString(*stream), "'devStream':0");
    
    uint64_t splitThreshold = 128ULL;
    auto ops = MakeOckMergeSortOpList(utils::MakeContainerInfo(dataA.begin(), dataA.end()),
        utils::MakeContainerInfo(dataB.begin(), dataB.end()),
        utils::MakeContainerInfo(quicklyMerged.begin(), quicklyMerged.end()),
        utils::LessCompareAdapter(),
        splitThreshold);
    auto opDeque = OckHeteroOperatorGroupQueue();
    opDeque.push(ops);

    EXPECT_EQ(this->stream->RunOps(opDeque, OckStreamExecPolicy::STOP_IF_ERROR), hmm::HMM_SUCCESS);

    CompareSortResult();
}
}  // namespace hop
}  // namespace hcps
}  // namespace ock
