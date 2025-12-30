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
#include "ock/hcps/stream/OckHeteroStreamBase.h"
#include "ock/acladapter/WithEnvOckAsyncTaskExecuteService.h"
namespace ock {
namespace hcps {
template <typename _BaseT>
class WithEnvOckHeteroStream : public acladapter::WithEnvOckAsyncTaskExecuteService<_BaseT> {
public:
    using BaseT = acladapter::WithEnvOckAsyncTaskExecuteService<_BaseT>;
    void SetUp(void) override
    {
        BaseT::SetUp();
        resourceType = acladapter::OckTaskResourceType::CPU;
    }
    void TearDown(void) override
    {
        if (stream.get() != nullptr) {
            stream->WaitExecComplete();
        }
        stream.reset();
        if (this->taskService.get() != nullptr) {
            this->taskService->Stop();
        }
        BaseT::TearDown();
    }
    void InitStream(void)
    {
        this->taskService = acladapter::OckAsyncTaskExecuteService::Create(this->deviceId,
            this->cpuSet,
            {{this->resourceType, this->transferThreadNum}, {acladapter::OckTaskResourceType::DEVICE_STREAM, 1ULL}});
        stream = OckHeteroStreamBase::Create(this->taskService).second;
    }
    acladapter::OckTaskResourceType resourceType;
    std::shared_ptr<OckHeteroStreamBase> stream;
};
}  // namespace hcps
}  // namespace ock
#endif