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


#include "gtest/gtest.h"
#include "ock/conf/OckSysConf.h"
#include "ock/hmm/mgr/checker/OckHmmHeteroMemoryMgrParamCheck.h"

namespace ock {
namespace hmm {
namespace test {

class TestOckHmmHeteroMemoryMgrParamCheckAlloc : public testing::Test {
public:
    using Check = OckHmmHeteroMemoryMgrParamCheck;

    uint64_t minHmoSize = conf::OckSysConf::HmmConf().minAllocHmoBytes;
    uint64_t maxHmoSize = conf::OckSysConf::HmmConf().maxAllocHmoBytes;
    uint64_t minMallocSize = conf::OckSysConf::HmmConf().minMallocBytes;
    uint64_t maxMallocSize = conf::OckSysConf::HmmConf().maxMallocBytes;
};

TEST_F(TestOckHmmHeteroMemoryMgrParamCheckAlloc, check_minHmoSize_and_maxHmoSize)
{
    EXPECT_EQ(minHmoSize, 0ULL);
    EXPECT_EQ(maxHmoSize, 128ULL * 1024ULL * 1024ULL * 1024ULL);
}

TEST_F(TestOckHmmHeteroMemoryMgrParamCheckAlloc, check_alloc_valid_hmoBytes)
{
    EXPECT_EQ(Check::CheckAlloc(maxHmoSize), HMM_SUCCESS);
}

TEST_F(TestOckHmmHeteroMemoryMgrParamCheckAlloc, check_alloc_invalid_hmoBytes)
{
    EXPECT_EQ(Check::CheckAlloc(minHmoSize), HMM_ERROR_INPUT_PARAM_OUT_OF_RANGE);
    EXPECT_EQ(Check::CheckAlloc(maxHmoSize + 1), HMM_ERROR_INPUT_PARAM_OUT_OF_RANGE);
}

TEST_F(TestOckHmmHeteroMemoryMgrParamCheckAlloc, check_minMallocSize_and_maxMallocSize)
{
    EXPECT_EQ(minMallocSize, 0ULL);
    EXPECT_EQ(maxMallocSize, 128ULL * 1024ULL * 1024ULL * 1024ULL);
}

TEST_F(TestOckHmmHeteroMemoryMgrParamCheckAlloc, check_malloc_valid_size)
{
    EXPECT_EQ(Check::CheckMalloc(maxMallocSize), HMM_SUCCESS);
}

TEST_F(TestOckHmmHeteroMemoryMgrParamCheckAlloc, check_malloc_invalid_size)
{
    EXPECT_EQ(Check::CheckMalloc(minMallocSize), HMM_ERROR_INPUT_PARAM_OUT_OF_RANGE);
    EXPECT_EQ(Check::CheckMalloc(maxMallocSize + 1), HMM_ERROR_INPUT_PARAM_OUT_OF_RANGE);
}
}
}
}