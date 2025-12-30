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
class TestOckExternalQuicklyMergeOp : public WithEnvOckHeteroStream<acladapter::WithEnvAclMock<testing::Test>> {
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
    void InitRandData(void)
    {
        outSortedData.reserve(DATA_COUNT);
        tmpDatas.reserve(DATA_COUNT);
        sortedGroups.reserve(lines);
        for (uint64_t i = 0; i < lines; ++i) {
            std::vector<DataT> tmp;
            for (uint64_t j = 0; j < DIM_SIZE; j++) {
                DataT tmpData = static_cast<DataT>((rand() % std::numeric_limits<DataT>::max()));
                tmp.push_back(tmpData);
                outSortedDataOri.push_back(tmpData);
                outSortedData.push_back(static_cast<DataT>((rand() % std::numeric_limits<DataT>::max())));
                tmpDatas.push_back(static_cast<DataT>((rand() % std::numeric_limits<DataT>::max())));
            }
            sort(tmp.begin(), tmp.end());
            sortedGroups.push_back(tmp);
        }
    }

    bool CompareSortResult(void)
    {
        ExternalQuicklyMerge(*(this->stream), sortedGroups, outSortedData, tmpDatas, utils::LessCompareAdapter(),
            splitThreshold);
        std::sort(outSortedDataOri.begin(), outSortedDataOri.end());
        return std::equal(outSortedData.begin(), outSortedData.end(), outSortedDataOri.begin());
    }

    uint64_t splitThreshold = 1024UL;
    uint64_t lines = DATA_COUNT / DIM_SIZE;
    std::vector<std::vector<DataT>> sortedGroups;
    std::vector<DataT> outSortedDataOri;
    std::vector<DataT> outSortedData;
    std::vector<DataT> tmpDatas;
};
TEST_F(TestOckExternalQuicklyMergeOp, quickly_merge_result)
{
    InitRandData();

    InitStream();
    bool result = CompareSortResult();
    EXPECT_EQ(result, true);
}
} // namespace hfo
} // namespace hcps
} // namespace ock