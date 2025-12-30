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

#include <ostream>
#include <cstdint>
#include <string>
#include <gtest/gtest.h>
#include "securec.h"
#include "ock/utils/StrUtils.h"
#include "ock/tools/topo/TopoDetectResult.h"

namespace ock {
namespace tools {
namespace topo {
namespace tests {

TEST(TestTopoDetectResult, print_correct)
{
    TopoDetectResult topoResult;
    topoResult.deviceInfo.deviceId = {1};
    topoResult.deviceInfo.cpuIds.push_back(CpuIdRange({1}, {3}));
    topoResult.deviceInfo.cpuIds.push_back(CpuIdRange({5}, {5}));
    topoResult.transferBytes = {1};
    topoResult.usedMicroseconds = {1};
    EXPECT_EQ(utils::ToString(topoResult),
        "|1         |1-3,5               |9.3e-10        |0.00093        |HOST_TO_DEVICE   |Normal              |\n");
    topoResult.errorCode = hmm::HMM_ERROR_DEVICE_DATA_SPACE_NOT_ENOUGH;
    EXPECT_EQ(utils::ToString(topoResult),
        "|1         |1-3,5               |-              |-              |HOST_TO_DEVICE   |Error(14071)        |\n");
    EXPECT_EQ(utils::ToString(topoResult.deviceInfo), "1:1-3,5");
}
TEST(TestTopoDetectResult, print_while_empty_cpuids)
{
    TopoDetectResult topoResult;
    topoResult.deviceInfo.deviceId = {1};
    topoResult.transferBytes = {1};
    topoResult.usedMicroseconds = {1};
    EXPECT_EQ(utils::ToString(topoResult),
        "|1         |                    |9.3e-10        |0.00093        |HOST_TO_DEVICE   |Normal              |\n");
}

}  // namespace tests
}  // namespace topo
}  // namespace tools
}  // namespace ock