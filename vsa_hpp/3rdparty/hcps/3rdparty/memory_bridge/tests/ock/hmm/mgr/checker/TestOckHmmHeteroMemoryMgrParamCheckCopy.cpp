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
#include "ock/hmm/mgr/WithEnvOckHmmSingleDeviceMgrExt.h"

namespace ock {
namespace hmm {
namespace test {

class TestOckHmmHeteroMemoryMgrParamCheckCopy : public WithEnvOckHmmSingleDeviceMgrExt<testing::Test> {
public:
    using BaseT = WithEnvOckHmmSingleDeviceMgrExt<testing::Test>;
    using ParamCheck = OckHmmHeteroMemoryMgrParamCheck;

    std::pair<std::shared_ptr<OckHmmHMObject>, std::shared_ptr<OckHmmHMObject>> BuildBlankScene()
    {
        this->MockAllocFreeWithNewDelete(*devDataAlloc);
        this->MockAllocFreeWithNewDelete(*hostDataAlloc);
        auto dstHmo = mgr->Alloc(maxHmoSize, OckHmmMemoryAllocatePolicy::DEVICE_DDR_FIRST).second;
        auto srcHmo = mgr->Alloc(maxHmoSize, OckHmmMemoryAllocatePolicy::DEVICE_DDR_FIRST).second;
        return std::make_pair(dstHmo, srcHmo);
    }

    uint64_t minHmoSize = conf::OckSysConf::HmmConf().minAllocHmoBytes;
    uint64_t maxHmoSize = conf::OckSysConf::HmmConf().maxAllocHmoBytes;
};

TEST_F(TestOckHmmHeteroMemoryMgrParamCheckCopy, check_valid_scene)
{
    auto ret = BuildBlankScene();
    EXPECT_EQ(ParamCheck::CheckCopy(*ret.first, minHmoSize, *ret.second, minHmoSize, maxHmoSize - minHmoSize),
              HMM_SUCCESS);
}

TEST_F(TestOckHmmHeteroMemoryMgrParamCheckCopy, check_offset_exceed_scope)
{
    auto ret = BuildBlankScene();
    EXPECT_EQ(ParamCheck::CheckCopy(*ret.first, maxHmoSize, *ret.second, minHmoSize, maxHmoSize - minHmoSize),
              HMM_ERROR_INPUT_PARAM_DST_OFFSET_EXCEED_SCOPE);
    EXPECT_EQ(ParamCheck::CheckCopy(*ret.first, minHmoSize, *ret.second, maxHmoSize, maxHmoSize - minHmoSize),
              HMM_ERROR_INPUT_PARAM_SRC_OFFSET_EXCEED_SCOPE);
}

TEST_F(TestOckHmmHeteroMemoryMgrParamCheckCopy, check_length_exceed_scope)
{
    auto ret = BuildBlankScene();
    EXPECT_EQ(ParamCheck::CheckCopy(*ret.first, minHmoSize + 1, *ret.second, minHmoSize, maxHmoSize - minHmoSize),
              HMM_ERROR_INPUT_PARAM_DST_LENGTH_EXCEED_SCOPE);
    EXPECT_EQ(ParamCheck::CheckCopy(*ret.first, minHmoSize, *ret.second, minHmoSize + 1, maxHmoSize - minHmoSize),
              HMM_ERROR_INPUT_PARAM_SRC_LENGTH_EXCEED_SCOPE);
}
}
}
}
