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
#include "ock/utils/StrUtils.h"
#include "ock/acladapter/data/OckMemoryCopyKind.h"
namespace ock {
namespace acladapter {
namespace test {
TEST(TestOckMemoryCopyKind, toString)
{
    EXPECT_EQ(utils::ToString(OckMemoryCopyKind::HOST_TO_HOST), "HOST_TO_HOST");
    EXPECT_EQ(utils::ToString(OckMemoryCopyKind::HOST_TO_DEVICE), "HOST_TO_DEVICE");
    EXPECT_EQ(utils::ToString(OckMemoryCopyKind::DEVICE_TO_HOST), "DEVICE_TO_HOST");
    EXPECT_EQ(utils::ToString(OckMemoryCopyKind::DEVICE_TO_DEVICE), "DEVICE_TO_DEVICE");

    uint32_t unknownKind = 100U;
    EXPECT_EQ(utils::ToString(static_cast<OckMemoryCopyKind>(unknownKind)), "UnknownCopyKind(100)");
}
TEST(TestCalcMemoryCopyKind, all)
{
    EXPECT_EQ(CalcMemoryCopyKind(hmm::OckHmmHeteroMemoryLocation::LOCAL_HOST_MEMORY,
                  hmm::OckHmmHeteroMemoryLocation::LOCAL_HOST_MEMORY),
        OckMemoryCopyKind::HOST_TO_HOST);
    EXPECT_EQ(CalcMemoryCopyKind(
                  hmm::OckHmmHeteroMemoryLocation::LOCAL_HOST_MEMORY, hmm::OckHmmHeteroMemoryLocation::DEVICE_DDR),
        OckMemoryCopyKind::HOST_TO_DEVICE);
    EXPECT_EQ(CalcMemoryCopyKind(
                  hmm::OckHmmHeteroMemoryLocation::LOCAL_HOST_MEMORY, hmm::OckHmmHeteroMemoryLocation::DEVICE_HBM),
        OckMemoryCopyKind::HOST_TO_DEVICE);
    EXPECT_EQ(CalcMemoryCopyKind(
                  hmm::OckHmmHeteroMemoryLocation::DEVICE_DDR, hmm::OckHmmHeteroMemoryLocation::LOCAL_HOST_MEMORY),
        OckMemoryCopyKind::DEVICE_TO_HOST);
    EXPECT_EQ(
        CalcMemoryCopyKind(hmm::OckHmmHeteroMemoryLocation::DEVICE_DDR, hmm::OckHmmHeteroMemoryLocation::DEVICE_DDR),
        OckMemoryCopyKind::DEVICE_TO_DEVICE);
    EXPECT_EQ(
        CalcMemoryCopyKind(hmm::OckHmmHeteroMemoryLocation::DEVICE_DDR, hmm::OckHmmHeteroMemoryLocation::DEVICE_HBM),
        OckMemoryCopyKind::DEVICE_TO_DEVICE);
    EXPECT_EQ(CalcMemoryCopyKind(
                  hmm::OckHmmHeteroMemoryLocation::DEVICE_HBM, hmm::OckHmmHeteroMemoryLocation::LOCAL_HOST_MEMORY),
        OckMemoryCopyKind::DEVICE_TO_HOST);
    EXPECT_EQ(
        CalcMemoryCopyKind(hmm::OckHmmHeteroMemoryLocation::DEVICE_HBM, hmm::OckHmmHeteroMemoryLocation::DEVICE_DDR),
        OckMemoryCopyKind::DEVICE_TO_DEVICE);
    EXPECT_EQ(
        CalcMemoryCopyKind(hmm::OckHmmHeteroMemoryLocation::DEVICE_HBM, hmm::OckHmmHeteroMemoryLocation::DEVICE_HBM),
        OckMemoryCopyKind::DEVICE_TO_DEVICE);
}
}  // namespace test
}  // namespace acladapter
}  // namespace ock
