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
#include "ascendfaiss/ascenddaemon/utils/MemorySpace.h"
#include "acl.h"

using namespace testing;
using namespace std;

namespace ascend {

class TestMemorySpace : public Test {
public:
    void TearDown() override
    {
        GlobalMockObject::verify();
    }
};

TEST_F(TestMemorySpace, construct)
{
    int8_t *start = nullptr;
    size_t size = 0;
    string actualMsg = "";

    MemorySpace spaceType = MemorySpace::DEVICE;
    MOCKER_CPP(&aclrtMalloc).stubs().will(returnValue(1))
                                    .then(returnValue(0))
                                    .then(returnValue(1))
                                    .then(returnValue(0));
    try {
        AllocMemorySpace(spaceType, &start, size);
    } catch (std::exception &e) {
        actualMsg = e.what();
    }
    string expectMsg("failed to aclrtMalloc");
    EXPECT_TRUE(actualMsg.find(expectMsg) != string::npos);

    actualMsg = "";
    try {
        AllocMemorySpace(spaceType, &start, size);
    } catch (std::exception &e) {
        actualMsg = e.what();
    }
    EXPECT_TRUE(actualMsg.empty());

    spaceType = MemorySpace::DEVICE_HUGEPAGE;
    try {
        AllocMemorySpace(spaceType, &start, size);
    } catch (std::exception &e) {
        actualMsg = e.what();
    }
    EXPECT_TRUE(actualMsg.find(expectMsg) != string::npos);

    actualMsg = "";
    try {
        AllocMemorySpace(spaceType, &start, size);
    } catch (std::exception &e) {
        actualMsg = e.what();
    }
    EXPECT_TRUE(actualMsg.empty());

    spaceType= static_cast<MemorySpace>(0);
    try {
        AllocMemorySpace(spaceType, &start, size);
    } catch (std::exception &e) {
        actualMsg = e.what();
    }
    expectMsg = string("Unsupported memoryspace type");
    EXPECT_TRUE(actualMsg.find(expectMsg) != string::npos);
}
} // namespace ascend