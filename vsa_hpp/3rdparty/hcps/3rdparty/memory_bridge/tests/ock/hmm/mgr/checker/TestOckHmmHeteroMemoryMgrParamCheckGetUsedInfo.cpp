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
#include "ock/hmm/mgr/checker/OckHmmHeteroMemoryMgrParamCheck.h"

namespace ock {
namespace hmm {
namespace test {

class TestOckHmmHeteroMemoryMgrParamCheckGetUsedInfo : public testing::Test {
public:
    using ParamCheck = OckHmmHeteroMemoryMgrParamCheck;

    uint64_t minFragThreshlod = conf::OckSysConf::HmmConf().minFragThreshold;
    uint64_t maxFragThreshlod = conf::OckSysConf::HmmConf().maxFragThreshold;
};

TEST_F(TestOckHmmHeteroMemoryMgrParamCheckGetUsedInfo, valid_fragThreshold)
{
    EXPECT_EQ(ParamCheck::CheckGetUsedInfo(minFragThreshlod), HMM_SUCCESS);
    EXPECT_EQ(ParamCheck::CheckGetUsedInfo(maxFragThreshlod), HMM_SUCCESS);
}

TEST_F(TestOckHmmHeteroMemoryMgrParamCheckGetUsedInfo, invalid_fragThreshold)
{
    EXPECT_EQ(ParamCheck::CheckGetUsedInfo(minFragThreshlod - 1), HMM_ERROR_INPUT_PARAM_OUT_OF_RANGE);
    EXPECT_EQ(ParamCheck::CheckGetUsedInfo(maxFragThreshlod + 1), HMM_ERROR_INPUT_PARAM_OUT_OF_RANGE);
}
}
}
}
