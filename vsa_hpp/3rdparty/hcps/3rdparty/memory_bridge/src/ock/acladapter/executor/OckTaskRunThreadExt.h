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


#ifndef OCK_MEMORY_BRIDGE_OCK_ADAPTER_TASK_TASK_RUN_THREAD_H
#define OCK_MEMORY_BRIDGE_OCK_ADAPTER_TASK_TASK_RUN_THREAD_H
#include <memory>
#include "ock/acladapter/executor/OckAsyncTaskExecuteServiceExt.h"
#include "ock/acladapter/executor/OckTaskRunThreadGen.h"
namespace ock {
namespace acladapter {

std::shared_ptr<OckTaskRunThreadBase> CreateRunThread(hmm::OckHmmDeviceId deviceId, const cpu_set_t &cpuSet,
    OckAsyncTaskExecuteServiceExt &taskQueue, OckTaskResourceType resourceType = OckTaskResourceType::HMM);

}  // namespace acladapter
}  // namespace ock
#endif