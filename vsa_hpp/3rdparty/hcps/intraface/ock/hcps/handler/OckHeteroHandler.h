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

#ifndef OCK_HCPS_OCK_HETERO_HANDLER_H
#define OCK_HCPS_OCK_HETERO_HANDLER_H
#include <vector>
#include "ock/hmm/mgr/OckHmmHeteroMemoryMgr.h"
#include "ock/acladapter/executor/OckAsyncTaskExecuteService.h"

namespace ock {
namespace hcps {
namespace handler {
class OckHeteroHandler {
public:
    virtual ~OckHeteroHandler() noexcept = default;
    virtual hmm::OckHmmHeteroMemoryMgrBase &HmmMgr(void) = 0;
    virtual std::shared_ptr<hmm::OckHmmHeteroMemoryMgrBase> HmmMgrPtr(void) = 0;
    virtual hmm::OckHmmDeviceId GetDeviceId(void) = 0;
    virtual std::shared_ptr<acladapter::OckAsyncTaskExecuteService> Service(void) = 0;

    static std::shared_ptr<OckHeteroHandler> CreateSingleDeviceHandler(hmm::OckHmmDeviceId deviceId,
        const cpu_set_t &cpuSet, const hmm::OckHmmMemorySpecification &memorySpec, hmm::OckHmmErrorCode &errorCode);

    static std::pair<hmm::OckHmmErrorCode, std::shared_ptr<OckHeteroHandler>> CreateSingleDeviceHandler(
        hmm::OckHmmDeviceId deviceId, const cpu_set_t &cpuSet, const hmm::OckHmmMemorySpecification &memorySpec);
};
}  // namespace handler
}  // namespace hcps
}  // namespace ock
#endif