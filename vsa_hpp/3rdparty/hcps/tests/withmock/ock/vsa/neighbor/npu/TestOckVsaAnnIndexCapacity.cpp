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
#include "ock/vsa/neighbor/npu/impl/OckVsaAnnIndexCapacity.h"
#include "ock/vsa/attr/OckTimeSpaceAttrTrait.h"

namespace ock {
namespace vsa {
namespace neighbor {
namespace npu {
namespace test {
TEST(TestOckVsaAnnIndexCapacity, space)
{
    hmm::OckHmmDeviceId devId = 1UL;
    uint64_t maxFeatureRowCount = 262144ULL * 64ULL;
    uint32_t tokenNum = 2500UL;
    cpu_set_t cpuSet;
    uint32_t extKeyAttrsByteSize = 0UL;
    uint32_t extKeyAttrBlockSize = 262144ULL;
    uint64_t featureBytes = 256ULL;
    uint64_t normBytes = 2ULL;
    auto param = OckVsaAnnCreateParam::Create(cpuSet, devId, maxFeatureRowCount, tokenNum, extKeyAttrsByteSize,
        extKeyAttrBlockSize);
    EXPECT_GE(OckVsaAnnIndexCapacity::DeviceSpace(*param, featureBytes, normBytes), 4773117952ULL);
    EXPECT_LT(OckVsaAnnIndexCapacity::DeviceSpace(*param, featureBytes, normBytes),
        OckVsaAnnIndexCapacity::HostSpace(*param, featureBytes, normBytes));
}
}  // namespace test
}  // namespace npu
}  // namespace neighbor
}  // namespace vsa
}  // namespace ock