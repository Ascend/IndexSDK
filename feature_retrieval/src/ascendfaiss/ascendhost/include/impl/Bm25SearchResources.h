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

#ifndef ASCEND_BM25_SEARCH_RESOURCES_H
#define ASCEND_BM25_SEARCH_RESOURCES_H
#include <array>
#include <initializer_list>
#include <limits>

#include "acl/acl_rt.h"
#include "ascendhost/include/impl/Bm25Operator.h"

namespace bm25search
{
using ascend::bm25::Check;
using ascend::bm25::DeviceAllocations;
using ascend::bm25::Require;
using ascend::bm25::Runtime;

struct PlannedResult
{
    std::vector<float> scores;
    std::vector<int32_t> ids;
};

// Coordinates stream lifetime with ALL OM resources, including constructor-failure cleanup.
struct PlanResources
{
    PlanResources() = default;
    explicit PlanResources(const std::shared_ptr<Runtime>& sharedRuntime, const ascend::bm25::AclnnApi* api = nullptr)
        : aclnnApi(api), sharedRuntimeLease_(std::make_unique<std::shared_ptr<Runtime>>(sharedRuntime))
    {
        ascend::bm25::CheckArgument(sharedRuntime != nullptr, "BM25 shared runtime is required");
        deviceAllocations.executionStream = sharedRuntime->executionStream;
    }
    PlanResources(const PlanResources&) = delete;
    PlanResources& operator=(const PlanResources&) = delete;
    PlanResources(PlanResources&&) = delete;
    PlanResources& operator=(PlanResources&&) = delete;
    Runtime runtime;
    DeviceAllocations deviceAllocations;
    std::vector<std::unique_ptr<ascend::AscendOperator>> offlineOperators;
    const ascend::bm25::AclnnApi* aclnnApi = nullptr;
    std::vector<aclTensor*> tensors;
    std::vector<aclOpExecutor*> executors;
    std::vector<aclDataBuffer*> dataBuffers;
    std::vector<void*> pinnedHostAllocations;
    struct DeviceMemoryGuard
    {
        unsigned char* baseAddress;
        size_t dataBytes;
    };
    std::vector<DeviceMemoryGuard> memoryGuards;
    bool closed = false;
    aclDataBuffer* CreateDataBuffer(void* deviceMemory, size_t bytes)
    {
        dataBuffers.reserve(dataBuffers.size() + 1);
        auto* buffer = aclCreateDataBuffer(deviceMemory, bytes);
        ascend::bm25::CheckAclPointer(buffer, "aclCreateDataBuffer");
        dataBuffers.push_back(buffer);
        return buffer;
    }
    aclTensor* CreateTensor(void* deviceMemory, aclDataType type, std::initializer_list<int64_t> dimensions)
    {
        Check(aclnnApi && aclnnApi->createTensor, "BM25 ACLNN tensor creation requires a valid API");
        std::array<int64_t, 3> strides{};
        Check(dimensions.size() > 0 && dimensions.size() <= strides.size(), "Invalid BM25 ACLNN tensor rank");
        int64_t stride = 1;
        for (size_t i = dimensions.size(); i > 0; --i)
        {
            strides[i - 1] = stride;
            stride *= dimensions.begin()[i - 1];
        }
        tensors.reserve(tensors.size() + 1);
        auto* tensor = aclnnApi->createTensor(dimensions.begin(), dimensions.size(), type, strides.data(), 0,
                                              ACL_FORMAT_ND, dimensions.begin(), dimensions.size(), deviceMemory);
        ascend::bm25::CheckAclPointer(tensor, "aclCreateTensor");
        tensors.push_back(tensor);
        return tensor;
    }
    void* AllocatePinnedHostMemory(size_t bytes)
    {
        pinnedHostAllocations.reserve(pinnedHostAllocations.size() + 1);
        void* hostMemory = nullptr;
        Require(aclrtMallocHost(&hostMemory, bytes), "allocate pinned plan arena");
        if (!hostMemory) throw std::runtime_error("Null pinned plan arena");
        pinnedHostAllocations.push_back(hostMemory);
        return hostMemory;
    }
    void* AllocateDeviceMemory(size_t bytes)
    {
        constexpr size_t guardBytes = 64;
        constexpr size_t guardOverhead = 2 * guardBytes;
        ascend::bm25::CheckArgument(bytes <= std::numeric_limits<size_t>::max() - guardOverhead,
                                    "Plan allocation size overflow");
        auto* allocationBase = static_cast<unsigned char*>(deviceAllocations.Allocate(bytes + guardOverhead));
        auto* payload = allocationBase + guardBytes;
        std::array<unsigned char, guardBytes> guardPattern{};
        guardPattern.fill(0xa7);
        Require(aclrtMemcpy(allocationBase, guardBytes, guardPattern.data(), guardBytes, ACL_MEMCPY_HOST_TO_DEVICE),
                "initialize leading plan guard");
        Require(aclrtMemcpy(payload + bytes, guardBytes, guardPattern.data(), guardBytes, ACL_MEMCPY_HOST_TO_DEVICE),
                "initialize trailing plan guard");
        memoryGuards.push_back({allocationBase, bytes});
        return payload;
    }
    void CheckGuards() const;
    void CloseChecked();
    bool Close() noexcept;
    ~PlanResources();

   private:
    const Runtime& GetRuntime() const { return sharedRuntimeLease_ ? **sharedRuntimeLease_ : runtime; }
    std::unique_ptr<std::shared_ptr<Runtime>> sharedRuntimeLease_;
};

}  // namespace bm25search
#endif  // ASCEND_BM25_SEARCH_RESOURCES_H
