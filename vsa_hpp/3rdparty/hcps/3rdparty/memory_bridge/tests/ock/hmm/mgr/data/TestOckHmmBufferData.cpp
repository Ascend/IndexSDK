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


#include <sstream>
#include "gtest/gtest.h"
#include "ock/utils/StrUtils.h"
#include "ock/hmm/mgr/data/OckHmmBufferData.h"
namespace ock {
namespace hmm {
namespace test {
TEST(TestOckHmmHeteroMemoryLocation, OckHmmBufferData)
{
    uint64_t addr = 3ULL;
    uint64_t byteSize = 5ULL;
    uint64_t offset = 7ULL;
    uint64_t hmoObjId = 11ULL;
    OckHmmBufferData hostData(addr, byteSize, offset, OckHmmHeteroMemoryLocation::LOCAL_HOST_MEMORY, hmoObjId);
    OckHmmBufferData devData(addr, byteSize, offset, OckHmmHeteroMemoryLocation::DEVICE_DDR, hmoObjId);
    EXPECT_EQ(hmoObjId, hostData.hmoObjId);
    EXPECT_EQ(hostData, hostData);
    EXPECT_NE(hostData, devData);
    EXPECT_NE(hostData,
        OckHmmBufferData(addr + 1, byteSize, offset, OckHmmHeteroMemoryLocation::LOCAL_HOST_MEMORY, hmoObjId));
    EXPECT_NE(hostData,
        OckHmmBufferData(addr, byteSize + 1, offset, OckHmmHeteroMemoryLocation::LOCAL_HOST_MEMORY, hmoObjId));
    EXPECT_NE(hostData,
        OckHmmBufferData(addr, byteSize, offset + 1, OckHmmHeteroMemoryLocation::LOCAL_HOST_MEMORY, hmoObjId));
    EXPECT_EQ("{'byteSize':5,'offset':7,'location':LOCAL_HOST_MEMORY}", utils::ToString(hostData));
}
}  // namespace test
}  // namespace hmm
}  // namespace ock
