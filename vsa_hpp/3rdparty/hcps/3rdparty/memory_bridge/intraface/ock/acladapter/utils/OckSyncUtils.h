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


#ifndef OCK_MEMORY_BRIDGE_OCK_ADAPTER_TASK_SYNC_UTILS_H
#define OCK_MEMORY_BRIDGE_OCK_ADAPTER_TASK_SYNC_UTILS_H
#include <functional>
#include "ock/acladapter/task/OckAsyncTaskExt.h"
#include "ock/acladapter/executor/OckAsyncTaskExecuteService.h"
#include "ock/acladapter/utils/OckAdapterMemoryGuard.h"
namespace ock {
namespace acladapter {

class OckSyncUtils {
public:
    virtual ~OckSyncUtils() noexcept = default;
    explicit OckSyncUtils(OckAsyncTaskExecuteService &taskService);

    OckAsyncTaskExecuteService &GetService(void);

    std::pair<hmm::OckHmmErrorCode, std::unique_ptr<OckAdapterMemoryGuard>> Malloc(
        size_t byteSize, hmm::OckHmmHeteroMemoryLocation location, uint32_t timeout = 0);

    hmm::OckHmmErrorCode Free(void *addr, hmm::OckHmmHeteroMemoryLocation location, uint32_t timeout = 0);

    hmm::OckHmmErrorCode Copy(
        void *dst, size_t destMax, const void *src, size_t count, OckMemoryCopyKind kind, uint32_t timeout = 0);

    hmm::OckHmmErrorCode CreateStream(OckDevRtStream &stream);
    hmm::OckHmmErrorCode DestroyStream(OckDevRtStream stream);
    hmm::OckHmmErrorCode SynchronizeStream(OckDevRtStream stream, uint32_t timeout = 0);

    hmm::OckHmmErrorCode ExecFun(
        OckTaskResourceType resourceType, std::function<hmm::OckHmmErrorCode()> opFun, uint32_t timeout = 0);
    std::shared_ptr<OckAsyncResultInnerBridge<OckDefaultResult>> ExecFunAsync(
        OckTaskResourceType resourceType, std::function<hmm::OckHmmErrorCode()> opFun);

private:
    OckAsyncTaskExecuteService &service;
};

}  // namespace acladapter
}  // namespace ock
#endif