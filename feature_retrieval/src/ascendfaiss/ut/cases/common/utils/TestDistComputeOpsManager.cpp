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


#include <gtest/gtest.h>
#include <mockcpp/mockcpp.hpp>
#include "ascendfaiss/ascenddaemon/utils/DistComputeOpsManager.h"

using namespace testing;
using namespace std;

namespace ascend {

class TestDistComputeOpsManager : public Test {
public:
    void TearDown() override
    {
        GlobalMockObject::verify();
    }
};

TEST_F(TestDistComputeOpsManager, construct)
{
    int32_t deviceId = 1000;
    MOCKER(&aclrtGetDevice).stubs().with(outBoundP(&deviceId))
                                    .will(returnValue(1))
                                    .then(returnValue(0));
    MOCKER_CPP(&AscendMultiThreadManager::IsMultiThreadMode).stubs().will(returnValue(true));

    DistComputeOpsManager::getShared();
    DistComputeOpsManager::getShared();
    DistComputeOpsManager::getShared();
}
} // namespace ascend