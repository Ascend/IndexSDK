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


#ifndef OCK_MEMORY_BRIDGE_ACL_ADAPTER_TASK_RUN_THREAD_BASE_H
#define OCK_MEMORY_BRIDGE_ACL_ADAPTER_TASK_RUN_THREAD_BASE_H
#include <memory>
#include "ock/hmm/mgr/OckHmmHMObject.h"
namespace ock {
namespace acladapter {
std::ostream &operator<<(std::ostream &os, const cpu_set_t &data);
class OckTaskRunThreadBase {
public:
    virtual ~OckTaskRunThreadBase() noexcept = default;

    virtual hmm::OckHmmDeviceId DeviceId(void) const = 0;

    virtual cpu_set_t CpuSet(void) const = 0;

    virtual hmm::OckHmmErrorCode Start(void) = 0;

    virtual hmm::OckHmmErrorCode Stop(void) = 0;

    virtual void NotifyStop(void) = 0;

    virtual void NotifyAdded(void) = 0;

    virtual bool Active(void) const = 0;
};
}  // namespace acladapter
}  // namespace ock
#endif
