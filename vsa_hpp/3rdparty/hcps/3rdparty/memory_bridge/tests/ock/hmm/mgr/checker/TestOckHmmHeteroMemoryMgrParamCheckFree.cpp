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

class TestOckHmmHeteroMemoryMgrParamCheckFree : public WithEnvOckHmmSingleDeviceMgrExt<testing::Test> {
public:
    using BaseT = WithEnvOckHmmSingleDeviceMgrExt<testing::Test>;
    using ParamCheck = OckHmmHeteroMemoryMgrParamCheck;

    uint64_t minHmoSize = conf::OckSysConf::HmmConf().minAllocHmoBytes;
    uint64_t maxHmoSize = conf::OckSysConf::HmmConf().maxAllocHmoBytes;
};

TEST_F(TestOckHmmHeteroMemoryMgrParamCheckFree, check_null_hmo_object)
{
    EXPECT_EQ(ParamCheck::CheckFree(nullptr), HMM_ERROR_INPUT_PARAM_EMPTY);
}

TEST_F(TestOckHmmHeteroMemoryMgrParamCheckFree, check_non_null_hmo_object)
{
    this->MockAllocFreeWithNewDelete(*devDataAlloc);
    this->MockAllocFreeWithNewDelete(*hostDataAlloc);
    auto maxHmo = mgr->Alloc(maxHmoSize, OckHmmMemoryAllocatePolicy::DEVICE_DDR_FIRST).second;

    EXPECT_EQ(ParamCheck::CheckFree(maxHmo), HMM_SUCCESS);

    mgr->Free(maxHmo);
}
}  // namespace test
}  // namespace hmm
}  // namespace ock