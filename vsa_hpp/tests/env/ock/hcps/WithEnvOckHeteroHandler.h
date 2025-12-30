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


#ifndef OCK_HCPS_PIER_WITH_ENV_OCK_HETERO_STREAM_H
#define OCK_HCPS_PIER_WITH_ENV_OCK_HETERO_STREAM_H
#include <thread>
#include <chrono>
#include "gtest/gtest.h"
#include "mockcpp/mockcpp.hpp"
#include "ock/hcps/handler/OckHeteroHandler.h"
#include "ock/acladapter/WithEnvAclMock.h"
namespace ock {
namespace hcps {
namespace handler {
template <typename _BaseT>
class WithEnvOckHeteroHandler : public acladapter::WithEnvAclMock<_BaseT> {
public:
    using BaseT = acladapter::WithEnvAclMock<_BaseT>;
    void SetUp(void) override
    {
        BaseT::SetUp();
        hmmSpec.devSpec.maxDataCapacity = 256UL * 1024UL * 1024UL;   // 256MB;
        hmmSpec.devSpec.maxSwapCapacity = 64UL * 1024UL * 1024UL;    // 64MB;
        hmmSpec.hostSpec.maxDataCapacity = 256UL * 1024UL * 1024UL;  // 256MB;
        hmmSpec.hostSpec.maxSwapCapacity = 64UL * 1024UL * 1024UL;   // 64MB;
        deviceId = 1ULL;
        CPU_ZERO(&cpuSet);
        CPU_SET(31UL, &cpuSet);
        CPU_SET(32UL, &cpuSet);
        CPU_SET(33UL, &cpuSet);
        CPU_SET(34UL, &cpuSet);
        CPU_SET(35UL, &cpuSet);
        CPU_SET(36UL, &cpuSet);
    }
    void TearDown(void) override
    {
        BaseT::TearDown();
    }
    std::shared_ptr<OckHeteroHandler> CreateSingleDeviceHandler(hmm::OckHmmErrorCode &errorCode)
    {
        return OckHeteroHandler::CreateSingleDeviceHandler(deviceId, cpuSet, hmmSpec, errorCode);
    }
    hmm::OckHmmMemorySpecification hmmSpec;
    hmm::OckHmmDeviceId deviceId;
    cpu_set_t cpuSet;
};
}  // namespace handler
}  // namespace hcps
}  // namespace ock
#endif