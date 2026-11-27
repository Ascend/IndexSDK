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

#ifndef ASCEND_BM25_RUNTIME_H
#define ASCEND_BM25_RUNTIME_H
#include <memory>

#include "acl/acl_rt.h"
#include "ascendhost/include/impl/Bm25Error.h"

namespace ascend::bm25
{
// The caller keeps this context active for construction, Search and destruction.
// resourceOwner retains the SDK resource proxy until all BM25 resources are freed.
struct RuntimeBinding
{
    aclrtContext callerContext = nullptr;
    std::shared_ptr<void> resourceOwner;
};

void RequireAcl(aclError aclStatusCode, const char* operation);

class Runtime
{
   public:
    Runtime() = default;
    Runtime(const Runtime&) = delete;
    Runtime& operator=(const Runtime&) = delete;
    Runtime(Runtime&&) = delete;
    Runtime& operator=(Runtime&&) = delete;
    ~Runtime();
    void Initialize(int deviceId, const RuntimeBinding& runtimeBinding);
    void RequireCurrentContext() const;
    void CloseChecked();
    bool Close() noexcept;
    void RetainResourcesUntilProcessExit() noexcept;

    aclrtContext boundContext = nullptr;
    aclrtStream executionStream = nullptr;

   private:
    // Allocated before acquiring device resources. On unsafe teardown, deliberately
    // retain this lease together with the device handles; no allocation in destructors.
    std::unique_ptr<std::shared_ptr<void>> resourceOwnerLease_;
};
}  // namespace ascend::bm25
#endif  // ASCEND_BM25_RUNTIME_H
