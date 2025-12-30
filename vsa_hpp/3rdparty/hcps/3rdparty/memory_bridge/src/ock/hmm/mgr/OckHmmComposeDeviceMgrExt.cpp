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

#include <unordered_map>
#include <mutex>
#include <syslog.h>
#include "ock/hmm/mgr/OckHmmErrorCode.h"
#include "ock/hmm/mgr/checker/OckHmmComposeDeviceMgrParamCheck.h"
#include "ock/hmm/mgr/algo/OckHmmComposeDeviceMgrAllocAlgo.h"
#include "ock/hmm/mgr/data/OckHmmHMOObjectIDGenerator.h"
#include "ock/log/OckLogger.h"
#include "ock/acladapter/task/OckMemoryCopyTask.h"
#include "ock/hmm/mgr/OckHmmComposeDeviceMgrExt.h"
namespace ock {
namespace hmm {
class OckHmmComposeDeviceMgrExt : public OckHmmComposeDeviceMgr {
public:
    virtual ~OckHmmComposeDeviceMgrExt() noexcept
    {
        openlog("ComposeDeviceMgr", LOG_CONS | LOG_PID, LOG_USER);
        syslog(LOG_INFO, "OckHmmComposeDeviceMgr is destructed.\n");
        closelog();
    }
    explicit OckHmmComposeDeviceMgrExt(std::vector<std::shared_ptr<OckHmmSingleDeviceMgr>> &deviceMgrVec)
    {
        for (auto &device : deviceMgrVec) {
            deviceMgrMap.insert(std::make_pair(device->GetDeviceId(), device));
        }
        itLast = deviceMgrMap.begin();
    }
    std::shared_ptr<OckHmmResourceUsedInfo> GetUsedInfo(uint64_t fragThreshold, OckHmmDeviceId deviceId) const override
    {
        auto ret = OckHmmComposeDeviceMgrParamCheck::CheckGetUsedInfo(fragThreshold);
        if (ret != HMM_SUCCESS) {
            return std::shared_ptr<OckHmmResourceUsedInfo>();
        }
        auto iter = deviceMgrMap.find(deviceId);
        if (iter == deviceMgrMap.end()) {
            OCK_HMM_LOG_ERROR("the input deviceId(" << deviceId << ") does not exist in composeDeviceMgr!");
            return std::shared_ptr<OckHmmResourceUsedInfo>();
        }
        return iter->second->GetUsedInfo(fragThreshold);
    }
    std::shared_ptr<OckHmmTrafficStatisticsInfo> GetTrafficStatisticsInfo(
        OckHmmDeviceId deviceId, uint32_t maxGapMilliSeconds) override
    {
        if (OckHmmHeteroMemoryMgrParamCheck::CheckGetTrafficStatisticsInfo(maxGapMilliSeconds) != HMM_SUCCESS) {
            return std::make_shared<OckHmmTrafficStatisticsInfo>();
        }
        auto iter = deviceMgrMap.find(deviceId);
        if (iter == deviceMgrMap.end()) {
            OCK_HMM_LOG_ERROR("the input deviceId(" << deviceId << ") does not exist in composeDeviceMgr!");
            return std::shared_ptr<OckHmmTrafficStatisticsInfo>();
        }
        return iter->second->GetTrafficStatisticsInfo(maxGapMilliSeconds);
    }
    const cpu_set_t *GetCpuSet(OckHmmDeviceId deviceId) const override
    {
        auto iter = deviceMgrMap.find(deviceId);
        if (iter == deviceMgrMap.end()) {
            OCK_HMM_LOG_ERROR("the input deviceId(" << deviceId << ") does not exist in composeDeviceMgr!");
            return nullptr;
        }
        return &(iter->second->GetCpuSet());
    }
    const OckHmmMemorySpecification *GetSpecific(OckHmmDeviceId deviceId) const override
    {
        auto iter = deviceMgrMap.find(deviceId);
        if (iter == deviceMgrMap.end()) {
            OCK_HMM_LOG_ERROR("the input deviceId(" << deviceId << ") does not exist in composeDeviceMgr!");
            return nullptr;
        }
        return &(iter->second->GetSpecific());
    }
    std::pair<OckHmmErrorCode, std::shared_ptr<OckHmmHMObject>> Alloc(
        uint64_t hmoBytes, OckHmmMemoryAllocatePolicy policy) override
    {
        auto retCode = OckHmmComposeDeviceMgrParamCheck::CheckAlloc(hmoBytes);
        if (retCode != HMM_SUCCESS) {
            return std::make_pair(retCode, std::shared_ptr<OckHmmHMObject>());
        }
        auto last = GetLast();
        auto ret = OckHmmComposeDeviceMgrAllocAlgo::Alloc(hmoBytes, policy, last, deviceMgrMap);
        if (ret.first == HMM_SUCCESS) {
            UpdateLast(last);
        }
        return ret;
    }
    std::pair<OckHmmErrorCode, std::shared_ptr<OckHmmHMObject>> Alloc(
        OckHmmDeviceId deviceId, uint64_t hmoBytes, OckHmmMemoryAllocatePolicy policy) override
    {
        auto iter = deviceMgrMap.find(deviceId);
        if (iter == deviceMgrMap.end()) {
            return make_pair(HMM_ERROR_INPUT_PARAM_DEVICE_NOT_EXISTS, std::shared_ptr<OckHmmHMObject>());
        }
        return iter->second->Alloc(hmoBytes, policy);
    }
    void Free(std::shared_ptr<OckHmmHMObject> hmo) override
    {
        auto retCode = OckHmmHeteroMemoryMgrParamCheck::CheckFree(hmo);
        if (retCode != HMM_SUCCESS) {
            return;
        }
        OckHmmDeviceId deviceId = OckHmmHMOObjectIDGenerator::ParseDeviceId(hmo->GetId());
        auto iter = deviceMgrMap.find(deviceId);
        if (iter == deviceMgrMap.end()) {
            OCK_HMM_LOG_ERROR("input Hmo " << *hmo << ", deviceId does not exist");
            return;
        }
        iter->second->Free(hmo);
    }
    OckHmmErrorCode CopyHMO(
        OckHmmHMObject &dstHMO, uint64_t dstOffset, OckHmmHMObject &srcHMO, uint64_t srcOffset, size_t length) override
    {
        auto ret = OckHmmComposeDeviceMgrParamCheck::CheckCopy(dstHMO, dstOffset, srcHMO, srcOffset, length);
        if (ret != HMM_SUCCESS) {
            return ret;
        }
        OckHmmDeviceId dstDeviceId = OckHmmHMOObjectIDGenerator::ParseDeviceId(dstHMO.GetId());
        auto iter = deviceMgrMap.find(dstDeviceId);
        if (iter == deviceMgrMap.end()) {
            OCK_HMM_LOG_ERROR("the deviceId of input dstHmo " << dstHMO << " does not exist");
            return HMM_ERROR_INPUT_PARAM_DEVICE_NOT_EXISTS;
        }
        if (length == 0) {
            return HMM_SUCCESS;
        }
        return iter->second->CopyHMO(dstHMO, dstOffset, srcHMO, srcOffset, length);
    }
    std::shared_ptr<OckHmmResourceUsedInfo> GetUsedInfo(uint64_t fragThreshold) const override
    {
        if (deviceMgrMap.empty()) {
            OCK_HMM_LOG_ERROR("deviceMgrMap is empty");
            return std::make_shared<OckHmmResourceUsedInfo>();
        }
        auto ret = OckHmmComposeDeviceMgrParamCheck::CheckGetUsedInfo(fragThreshold);
        if (ret != HMM_SUCCESS) {
            return std::shared_ptr<OckHmmResourceUsedInfo>();
        }
        auto outputUsedInfo = std::make_shared<OckHmmResourceUsedInfo>();
        for (auto iter : deviceMgrMap) {
            auto usedInfo = iter.second->GetUsedInfo(fragThreshold);
            if (usedInfo == nullptr) {
                return std::shared_ptr<OckHmmResourceUsedInfo>();
            }
            outputUsedInfo->devUsedInfo += usedInfo->devUsedInfo;
            outputUsedInfo->hostUsedInfo += usedInfo->hostUsedInfo;
        }
        return outputUsedInfo;
    }
    std::shared_ptr<OckHmmTrafficStatisticsInfo> GetTrafficStatisticsInfo(uint32_t maxGapMilliSeconds) override
    {
        if (OckHmmHeteroMemoryMgrParamCheck::CheckGetTrafficStatisticsInfo(maxGapMilliSeconds) != HMM_SUCCESS) {
            return std::make_shared<OckHmmTrafficStatisticsInfo>();
        }
        auto statisticsInfo = std::make_shared<OckHmmTrafficStatisticsInfo>();
        for (const auto& iter : deviceMgrMap) {
            auto curDevInfo = iter.second->GetTrafficStatisticsInfo(maxGapMilliSeconds);
            statisticsInfo->host2DeviceMovedBytes += curDevInfo->host2DeviceMovedBytes;
            statisticsInfo->device2hostMovedBytes += curDevInfo->device2hostMovedBytes;
            statisticsInfo->host2DeviceSpeed += curDevInfo->host2DeviceSpeed;
            statisticsInfo->device2hostSpeed += curDevInfo->device2hostSpeed;
        }
        return statisticsInfo;
    }
    std::unique_ptr<OckHmmMemoryGuard> Malloc(uint64_t size, OckHmmMemoryAllocatePolicy policy) override
    {
        if (OckHmmHeteroMemoryMgrParamCheck::CheckMalloc(size) != HMM_SUCCESS) {
            return std::unique_ptr<OckHmmMemoryGuard>();
        }
        auto last = GetLast();
        auto ret = OckHmmComposeDeviceMgrAllocAlgo::Malloc(size, policy, last, deviceMgrMap);
        if (ret.get() != nullptr) {
            UpdateLast(last);
        }
        return ret;
    }
    uint8_t *AllocateHost(std::size_t byteCount) override
    {
        if (deviceMgrMap.empty()) {
            return nullptr;
        }
        return deviceMgrMap.begin()->second->AllocateHost(byteCount);
    }
    void DeallocateHost(uint8_t *addr, std::size_t byteCount) override
    {
        if (addr == nullptr) {
            return;
        }
        deviceMgrMap.begin()->second->DeallocateHost(addr, byteCount);
    }

private:
    OckHmmMgrMapContainerT::iterator GetLast(void)
    {
        std::lock_guard<std::mutex> guard(lastAllocIterMutex);
        return itLast;
    }
    void UpdateLast(OckHmmMgrMapContainerT::iterator iter)
    {
        std::lock_guard<std::mutex> guard(lastAllocIterMutex);
        itLast = iter;
    }
    OckHmmMgrMapContainerT deviceMgrMap{};
    std::mutex lastAllocIterMutex{};
    OckHmmMgrMapContainerT::iterator itLast{ nullptr };
};
namespace ext {
std::shared_ptr<OckHmmComposeDeviceMgr> CreateComposeDeviceMgr(
    std::vector<std::shared_ptr<OckHmmSingleDeviceMgr>> &deviceMgrVec)
{
    return std::make_shared<OckHmmComposeDeviceMgrExt>(deviceMgrVec);
}
}  // namespace ext
}  // namespace hmm
}  // namespace ock