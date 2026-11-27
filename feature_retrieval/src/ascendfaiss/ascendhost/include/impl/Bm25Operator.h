/*
 * -------------------------------------------------------------------------
 * This file is part of the IndexSDK project.
 * Copyright (c) 2026 Huawei Technologies Co.,Ltd.
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

#ifndef ASCEND_BM25_OPERATOR_H
#define ASCEND_BM25_OPERATOR_H
#include <memory>
#include <vector>

#include "acl/acl.h"
#include "ascenddaemon/utils/AscendOperator.h"
#include "ascendhost/include/impl/Bm25Error.h"
#include "ascendhost/include/impl/Bm25Runtime.h"

struct aclTensor;
struct aclOpExecutor;

namespace ascend::bm25
{
// Generated custom-operator API. The library is resolved only for managed indexes.
struct AclnnApi
{
    using Execute = int (*)(void*, uint64_t, aclOpExecutor*, aclrtStream);
    using FusedPrepare = int (*)(const aclTensor*, const aclTensor*, const aclTensor*, const aclTensor*,
                                 const aclTensor*, const aclTensor*, const aclTensor*, int64_t, int64_t, int64_t,
                                 const aclTensor*, const aclTensor*, const aclTensor*, uint64_t*, aclOpExecutor**);
    using TreePrepare = int (*)(const aclTensor*, const aclTensor*, int64_t, int64_t, const aclTensor*,
                                const aclTensor*, uint64_t*, aclOpExecutor**);
    using SectionPrepare = int (*)(const aclTensor*, const aclTensor*, const aclTensor*, const aclTensor*,
                                   const aclTensor*, const aclTensor*, const aclTensor*, uint64_t*, aclOpExecutor**);

    static const AclnnApi& Load();

    aclTensor* (*createTensor)(const int64_t*, uint64_t, aclDataType, const int64_t*, int64_t, aclFormat,
                               const int64_t*, uint64_t, void*);
    int (*destroyTensor)(const aclTensor*);
    int (*markRepeatable)(aclOpExecutor*);
    int (*destroyExecutor)(aclOpExecutor*);
    FusedPrepare prepareFused;
    TreePrepare prepareTree;
    SectionPrepare prepareSection;
    Execute runFused, runTree, runSection;
};

inline void Require(aclError code, const char* operation) { RequireAcl(code, operation); }
// Owns device allocations even if construction fails before the plan is complete.
struct DeviceAllocations
{
    DeviceAllocations() = default;
    DeviceAllocations(const DeviceAllocations&) = delete;
    DeviceAllocations& operator=(const DeviceAllocations&) = delete;
    DeviceAllocations(DeviceAllocations&&) = delete;
    DeviceAllocations& operator=(DeviceAllocations&&) = delete;
    aclrtStream executionStream = nullptr;
    std::vector<void*> allocations;
    void* Allocate(size_t bytes);
    bool Close();
    ~DeviceAllocations();
};
// Uses the SDK's process-wide model directory initialization. No online fallback.
void InitializeModels();
// Descriptor creation APIs return pointers rather than ACL status codes.
void CheckAclPointer(const void* aclPointer, const char* operation);
}  // namespace ascend::bm25
#endif  // ASCEND_BM25_OPERATOR_H
