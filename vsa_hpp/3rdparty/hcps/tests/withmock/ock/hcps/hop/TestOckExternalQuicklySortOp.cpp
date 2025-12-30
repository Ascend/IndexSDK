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
#include <algorithm>
#include <gtest/gtest.h>
#include "ptest/ptest.h"
#include "ock/utils/OckCompareUtils.h"
#include "ock/utils/StrUtils.h"
#include "ock/hcps/hop/OckExternalQuicklySortOp.h"
#include "ock/hcps/stream/OckHeteroStreamBase.h"
#include "ock/hcps/WithEnvOckHeteroStream.h"
#include "ock/acladapter/WithEnvAclMock.h"

namespace ock {
namespace hcps {
namespace hop {
namespace {
const uint64_t DIM_SIZE = 512ULL;
const uint64_t DATA_COUNT = 409600ULL;
}
class TestOckExternalQuicklySortOp : public WithEnvOckHeteroStream<acladapter::WithEnvAclMock<testing::Test>> {
public:
    using BaseT = WithEnvOckHeteroStream<acladapter::WithEnvAclMock<testing::Test>>;
    using DataT = uint32_t;
    void SetUp(void) override
    {
        BaseT::SetUp();
        CPU_ZERO(&cpuSet);
        uint32_t cpuCount = sysconf(_SC_NPROCESSORS_CONF);
        for (uint32_t i = cpuCount / 2UL; i < cpuCount; ++i) {
            CPU_SET(i, &cpuSet);
        }
    }
    void InitRandData()
    {
        inputDatas.reserve(DATA_COUNT);
        inputDatasOri.reserve(DATA_COUNT);
        outputDatas.reserve(DATA_COUNT);
        for (uint64_t i = 0; i < DATA_COUNT; ++i) {
            auto tmpdata = static_cast<DataT>(rand() % std::numeric_limits<DataT>::max());
            inputDatas.push_back(tmpdata);
            inputDatasOri.push_back(tmpdata);
        }
    }

    uint64_t splitThreshold = 102400UL;
    std::vector<DataT> inputDatas;
    std::vector<DataT> inputDatasOri;
    std::vector<DataT> outputDatas;
};
TEST_F(TestOckExternalQuicklySortOp, quickly_sort_even_threshold)
{
    InitRandData();

    (void)pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuSet);
    std::sort(inputDatasOri.begin(), inputDatasOri.end(), utils::LessCompareAdapter());

    InitStream();
    ExternalQuicklySort(*(this->stream), inputDatas, outputDatas, utils::LessCompareAdapter(), splitThreshold);
    bool result = std::equal(inputDatasOri.begin(), inputDatasOri.end(), outputDatas.begin());
    EXPECT_EQ(result, true);
}
TEST_F(TestOckExternalQuicklySortOp, quickly_sort_odd_threshold)
{
    InitRandData();

    (void)pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuSet);
    std::sort(inputDatasOri.begin(), inputDatasOri.end(), utils::LessCompareAdapter());

    InitStream();
    splitThreshold = 9999UL;
    ExternalQuicklySort(*(this->stream), inputDatas, outputDatas, utils::LessCompareAdapter(), splitThreshold);
    bool result = std::equal(inputDatasOri.begin(), inputDatasOri.end(), outputDatas.begin());
    EXPECT_EQ(result, true);
}
} // namespace hfo
} // namespace hcps
} // namespace ock