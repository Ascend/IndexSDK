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
#include "ascendfaiss/ascenddaemon/utils/AscendRWLock.h"

using namespace testing;
using namespace std;

namespace ascend {

class TestAscendRWLock : public Test {
public:
    void TearDown() override
    {
        GlobalMockObject::verify();
    }
};

TEST_F(TestAscendRWLock, construct)
{
    std::mutex mtx;
    AscendRWLock lock(&mtx);

    std::mutex readMtx;
    size_t reaMtxCnt = 0;
    std::mutex writeMtx;
    AscendRWLock writeLock1(&readMtx, &reaMtxCnt, &writeMtx);
    AscendRWLock writeLock2(&readMtx, &reaMtxCnt, &writeMtx);

    // 没有这种类型
    std::mutex newMtx;
    AscendRWLock lockOther(&newMtx);
    lockOther.lockType = WRITE_LOCK + 1;
}
} // namespace ascend