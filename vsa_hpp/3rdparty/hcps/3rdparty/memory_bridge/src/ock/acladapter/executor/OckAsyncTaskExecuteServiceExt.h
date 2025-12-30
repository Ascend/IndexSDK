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


#ifndef OCK_MEMORY_BRIDGE_OCK_ADAPTER_TASK_ASYNC_TASK_EXECUTE_SERVICE_EXT_H
#define OCK_MEMORY_BRIDGE_OCK_ADAPTER_TASK_ASYNC_TASK_EXECUTE_SERVICE_EXT_H
#include "ock/acladapter/executor/OckAsyncTaskExecuteService.h"
namespace ock {
namespace acladapter {

struct OckDeviceStreamInfo {
    OckDeviceStreamInfo(hmm::OckHmmDeviceId deviceId, OckDevRtStream stream);
    hmm::OckHmmDeviceId deviceId;
    OckDevRtStream stream;
};
class OckAsyncTaskExecuteServiceExt : public OckAsyncTaskExecuteService {
public:
    virtual ~OckAsyncTaskExecuteServiceExt() noexcept = default;

    virtual hmm::OckHmmDeviceId GetDeviceId(void) const = 0;
    virtual const cpu_set_t &GetCpuSet(void) const = 0;

    virtual std::shared_ptr<OckAsyncTaskBase> PickUp(OckTaskResourceType resourceType) = 0;
};
std::ostream &operator<<(std::ostream &os, const OckDeviceStreamInfo &streamInfo);
std::ostream &operator<<(std::ostream &os, const OckAsyncTaskExecuteServiceExt &service);
}  // namespace acladapter
}  // namespace ock
#endif