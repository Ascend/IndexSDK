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
#include "ock/conf/OckSysConf.h"
#include "ock/utils/StrUtils.h"
#include "ock/hmm/mgr/OckHmmMgrCreator.h"
#include "ock/hmm/mgr/OckHmmHMObjectExt.h"
#include "ock/acladapter/utils/OckSyncUtils.h"
#include "ock/hmm/mgr/data/OckHmmHMOObjectIDGenerator.h"
#include "ock/hmm/mgr/OckHmmMemoryGuardExt.h"
#include "ock/hmm/mgr/checker/OckHmmHeteroMemoryMgrParamCheck.h"
#include "ock/acladapter/task/OckMemoryCopyTask.h"
#include "ock/hmm/mgr/OckHmmSingleDeviceMgrExt.h"

namespace ock {
namespace hmm {
namespace {
const uint64_t MIN_INCREASE_MEMORY_BYTESIZE = 4ULL * 1024U * 1024U * 1024U;
const uint64_t MAX_INCREASE_MEMORY_BYTESIZE = 100ULL * 1024U * 1024U * 1024U;
}
OckHmmSubMemoryAllocSet::OckHmmSubMemoryAllocSet(std::shared_ptr<OckHmmSubMemoryAlloc> devDataAlloc,
    std::shared_ptr<OckHmmSubMemoryAlloc> devSwapAlloc, std::shared_ptr<OckHmmSubMemoryAlloc> hostDataAlloc,
    std::shared_ptr<OckHmmSubMemoryAlloc> hostSwapAlloc)
    : devData(devDataAlloc), devSwap(devSwapAlloc), hostData(hostDataAlloc), hostSwap(hostSwapAlloc)
{}
class OckHmmSingleDeviceMgrExt : public OckHmmSingleDeviceMgr, OckHmmSubMemoryAllocDispatcher {
public:
    virtual ~OckHmmSingleDeviceMgrExt() noexcept
    {
        std::lock_guard<std::mutex> guard(mutex);
        for (auto &data : hmoHolderMap) {
            if (data.first != nullptr) {
                data.first->ForceReleaseMemory();
            }
        }
        openlog("SingleDeviceMgr", LOG_CONS | LOG_PID, LOG_USER);
        syslog(LOG_INFO, "OckHmmSingleDeviceMgr is destructed.\n");
        closelog();
    }
    OckHmmSingleDeviceMgrExt(std::shared_ptr<OckHmmDeviceInfo> devInfo,
        std::shared_ptr<acladapter::OckAsyncTaskExecuteService> taskService,
        std::shared_ptr<OckHmmSubMemoryAllocSet> memAllocSet)
        : service(taskService), deviceInfo(devInfo), cpuSet(std::make_shared<cpu_set_t>(deviceInfo->cpuSet)),
          memorySpec(std::make_shared<OckHmmMemorySpecification>(deviceInfo->memorySpec)), allocSet(memAllocSet),
          csnGenerator(std::make_shared<HmoCsnGenerator>())
    {}
    std::unique_ptr<OckHmmMemoryGuard> Malloc(uint64_t byteSize, OckHmmMemoryAllocatePolicy policy) override
    {
        if (OckHmmHeteroMemoryMgrParamCheck::CheckMalloc(byteSize) != HMM_SUCCESS) {
            return std::unique_ptr<OckHmmMemoryGuard>();
        }
        if (policy == OckHmmMemoryAllocatePolicy::DEVICE_DDR_FIRST) {
            auto ret = MallocBytes(allocSet->devData, byteSize);
            if (ret.get() != nullptr) {
                return ret;
            }
            return MallocBytes(allocSet->hostData, byteSize);
        } else if (policy == OckHmmMemoryAllocatePolicy::DEVICE_DDR_ONLY) {
            return MallocBytes(allocSet->devData, byteSize);
        } else {
            return MallocBytes(allocSet->hostData, byteSize);
        }
    }
    std::pair<OckHmmErrorCode, std::shared_ptr<OckHmmHMObject>> Alloc(
        uint64_t byteSize, OckHmmMemoryAllocatePolicy policy) override
    {
        OckHmmErrorCode retCode = OckHmmHeteroMemoryMgrParamCheck::CheckAlloc(byteSize);
        if (retCode != HMM_SUCCESS) {
            return std::make_pair(retCode, std::shared_ptr<OckHmmHMObject>());
        }
        auto ret = AllocImpl(byteSize, policy);
        if (ret.first == HMM_SUCCESS) {
            std::lock_guard<std::mutex> guard(mutex);
            hmoHolderMap.insert(std::make_pair(ret.second.get(), ret.second));
        }
        return std::make_pair(ret.first, OckHmmHMObjectOutter::Create(ret.second, *this));
    }

    std::shared_ptr<OckHmmSubMemoryAlloc> DevSwapAlloc(void) override
    {
        return allocSet->devSwap;
    }
    std::shared_ptr<OckHmmSubMemoryAlloc> HostSwapAlloc(void) override
    {
        return allocSet->hostSwap;
    }
    std::shared_ptr<OckHmmSubMemoryAlloc> SwapAlloc(OckHmmHeteroMemoryLocation location) override
    {
        if (location == OckHmmHeteroMemoryLocation::LOCAL_HOST_MEMORY) {
            return HostSwapAlloc();
        }
        return DevSwapAlloc();
    }
    void InnerFree(std::shared_ptr<OckHmmHMObjectExt> hmo) override
    {
        FreeImpl(hmo.get());
    }
    void Free(std::shared_ptr<OckHmmHMObject> hmo) override
    {
        auto retCode = OckHmmHeteroMemoryMgrParamCheck::CheckFree(hmo);
        if (retCode != HMM_SUCCESS) {
            return;
        }
        OckHmmHMObjectOutter *pHmoOutter = dynamic_cast<OckHmmHMObjectOutter *>(hmo.get());
        if (pHmoOutter != nullptr) {
            OckHmmHMObjectExt *pHmoExt = pHmoOutter->GetExtHmo();
            if (hmoHolderMap.find(pHmoExt) == hmoHolderMap.end()) {
                OCK_HMM_LOG_WARN("the hmo(" << *hmo << ") does not exist in hmoHolderMap!");
                return;
            }
            FreeImpl(pHmoExt);
        }
    }
    OckHmmErrorCode CopyHMO(
        OckHmmHMObject &dstHMO, uint64_t dstOffset, OckHmmHMObject &srcHMO, uint64_t srcOffset, size_t length) override
    {
        // 在这里做入口参数检查
        auto retCode = OckHmmHeteroMemoryMgrParamCheck::CheckCopy(dstHMO, dstOffset, srcHMO, srcOffset, length);
        if (retCode != HMM_SUCCESS) {
            return retCode;
        }
        auto dstOutterHmo = dynamic_cast<OckHmmHMObjectOutter *>(&dstHMO);
        auto srcOutterHmo = dynamic_cast<OckHmmHMObjectOutter *>(&srcHMO);
        if (dstOutterHmo == nullptr || srcOutterHmo == nullptr) {
            return HMM_ERROR_HMO_OBJECT_NOT_EXISTS;
        }
        OckHmmHMObjectExt *dstHmoExt = dstOutterHmo->GetExtHmo();
        OckHmmHMObjectExt *srcHmoExt = srcOutterHmo->GetExtHmo();
        if (dstHmoExt == nullptr) {
            OCK_HMM_LOG_ERROR("dynamic_cast for dstHmoExt is nullptr");
            return HMM_ERROR_HMO_OBJECT_NOT_EXISTS;
        }
        if (srcHmoExt == nullptr) {
            OCK_HMM_LOG_ERROR("dynamic_cast for srcHmoExt is nullptr");
            return HMM_ERROR_HMO_OBJECT_NOT_EXISTS;
        }
        if (hmoHolderMap.find(dstHmoExt) == hmoHolderMap.end()) {
            OCK_HMM_LOG_ERROR("dstHmo(" << dstHMO << ") does not exist in hmoHolderMap!");
            return HMM_ERROR_HMO_OBJECT_NOT_EXISTS;
        }
        if (hmoHolderMap.find(srcHmoExt) == hmoHolderMap.end()) {
            OCK_HMM_LOG_ERROR("srcHmo(" << srcHMO << ") does not exist in hmoHolderMap!");
            return HMM_ERROR_HMO_OBJECT_NOT_EXISTS;
        }
        if (length == 0) {
            return HMM_SUCCESS;
        }
        auto copyParam = std::make_shared<acladapter::OckMemoryCopyParam>((uint8_t *)dstHMO.Addr() + dstOffset,
            dstHMO.GetByteSize() - dstOffset,
            (uint8_t *)srcHMO.Addr() + srcOffset,
            length,
            acladapter::CalcMemoryCopyKind(srcHMO.Location(), dstHMO.Location()));
        auto bridge = std::make_shared<acladapter::OckAsyncResultInnerBridge<acladapter::OckDefaultResult>>();
        service->AddTask(acladapter::OckMemoryCopyTask::Create(copyParam, bridge));
        auto result = bridge->WaitResult();
        if (result == nullptr) {
            return HMM_ERROR_WAIT_TIME_OUT;
        }
        return result->ErrorCode();
    }
    void FillMemUsedInfo(OckHmmMemoryUsedInfo &usedInfo, const std::shared_ptr<OckHmmMemoryUsedInfoLocal> &devLocal,
        const std::shared_ptr<OckHmmMemoryUsedInfoLocal> &swapDevLocal) const
    {
        usedInfo.usedBytes = devLocal->usedBytes;
        usedInfo.leftBytes = devLocal->leftBytes;
        usedInfo.unusedFragBytes = devLocal->unusedFragBytes;
        usedInfo.swapUsedBytes = swapDevLocal->usedBytes;
        usedInfo.swapLeftBytes = swapDevLocal->leftBytes;
    }
    std::shared_ptr<OckHmmResourceUsedInfo> GetUsedInfo(uint64_t fragThreshold) const override
    {
        if (OckHmmHeteroMemoryMgrParamCheck::CheckGetUsedInfo(fragThreshold) != HMM_SUCCESS) {
            return std::shared_ptr<OckHmmResourceUsedInfo>();
        }
        auto ret = std::make_shared<OckHmmResourceUsedInfo>();
        FillMemUsedInfo(ret->devUsedInfo,
            allocSet->devData->GetUsedInfo(fragThreshold),
            allocSet->devSwap->GetUsedInfo(fragThreshold));
        FillMemUsedInfo(ret->hostUsedInfo,
            allocSet->hostData->GetUsedInfo(fragThreshold),
            allocSet->hostSwap->GetUsedInfo(fragThreshold));
        return ret;
    }
    std::shared_ptr<OckHmmTrafficStatisticsInfo> GetTrafficStatisticsInfo(uint32_t maxGapMilliSeconds) override
    {
        if (OckHmmHeteroMemoryMgrParamCheck::CheckGetTrafficStatisticsInfo(maxGapMilliSeconds) != HMM_SUCCESS) {
            return std::make_shared<OckHmmTrafficStatisticsInfo>();
        }
        return service->TaskStatisticsMgr()->PickUp(deviceInfo->deviceId, maxGapMilliSeconds);
    }
    const OckHmmMemorySpecification &GetSpecific(void) const override
    {
        return *memorySpec;
    }
    const cpu_set_t &GetCpuSet(void) const override
    {
        return *cpuSet;
    }
    OckHmmDeviceId GetDeviceId(void) const override
    {
        return deviceInfo->deviceId;
    }
    uint8_t *AllocateHost(std::size_t byteCount) override
    {
        if (byteCount == 0) {
            OCK_HMM_LOG_ERROR("the byteCount can't be 0.");
            return nullptr;
        }

        uintptr_t addr = allocSet->hostData->Alloc(byteCount);
        if (addr) {
            return reinterpret_cast<uint8_t *>(addr);
        } else {
            return nullptr;
        }
    }
    void DeallocateHost(uint8_t *addr, std::size_t byteCount) override
    {
        if (addr == nullptr) {
            OCK_HMM_LOG_ERROR("the input addr can't be nullptr");
            return;
        }
        allocSet->hostData->Free(uintptr_t(addr), byteCount);
    }

    /* *
     * @brief 在 allocType 指示的位置新增 byteSize 大小的内存分配
     * @allocType 新增内存的位置
     * @byteSize 新增内存的大小, 最小4G, 最大100G
     */
    OckHmmErrorCode IncBindMemory(OckHmmHeteroMemoryLocation allocType, uint64_t byteSize,
        uint32_t timeout = 0) override
    {
        if (byteSize < MIN_INCREASE_MEMORY_BYTESIZE || byteSize > MAX_INCREASE_MEMORY_BYTESIZE) {
            OCK_HMM_LOG_ERROR("Increased memory size (" << byteSize << ") is out of range!");
            return HMM_ERROR_INPUT_PARAM_OUT_OF_RANGE;
        }
        if (allocType != OckHmmHeteroMemoryLocation::LOCAL_HOST_MEMORY) {
            OCK_HMM_LOG_ERROR("memory in " << allocType << " (" << byteSize << ") cannot be added!");
            return HMM_ERROR_INPUT_PARAM_OUT_OF_RANGE;
        }

        // 在 allocType 指示的位置新增 byteSize 大小的内存
        acladapter::OckSyncUtils syncUtils(*service);
        auto incDataMemory = syncUtils.Malloc(byteSize, hmm::OckHmmHeteroMemoryLocation::LOCAL_HOST_MEMORY, timeout);
        if (incDataMemory.first != HMM_SUCCESS) {
            OCK_HMM_LOG_ERROR("malloc " << allocType << " (" << byteSize << ") failed, retCode = " <<
                incDataMemory.first);
            return incDataMemory.first;
        }
        OCK_HMM_LOG_INFO("malloc " << allocType << " (" << byteSize << ") succeed!");

        // 对增量内存加入二次内存分配管理
        allocSet->hostData->IncBindMemoryToMemPool(std::move(incDataMemory.second), byteSize, "HOST_DATA");
        memorySpec->hostSpec.maxDataCapacity += byteSize;
        OCK_HMM_LOG_INFO("SubMemoryAlloc " << allocType << " (" << byteSize << ") succeed!");
        return incDataMemory.first;
    }

private:
    void FreeImpl(OckHmmHMObjectExt *pHmoExt)
    {
        if (pHmoExt != nullptr) {
            pHmoExt->ForceReleaseMemory();
            std::lock_guard<std::mutex> guard(mutex);
            hmoHolderMap.erase(pHmoExt);
        }
    }
    std::pair<OckHmmErrorCode, std::shared_ptr<OckHmmHMObjectExt>> AllocImpl(
        uint64_t byteSize, OckHmmMemoryAllocatePolicy policy)
    {
        if (csnGenerator->UsedCount() >= conf::OckSysConf::HmmConf().maxHMOCountPerDevice) {
            return std::make_pair(HMM_ERROR_HMO_OBJECT_NUM_EXCEED, std::shared_ptr<OckHmmHMObjectExt>());
        }
        if (policy == OckHmmMemoryAllocatePolicy::DEVICE_DDR_FIRST) {
            auto ret = Malloc(allocSet->devData, byteSize);
            if (ret.first == HMM_SUCCESS) {
                return ret;
            }
            ret = Malloc(allocSet->hostData, byteSize);
            if (ret.first == HMM_SUCCESS) {
                return ret;
            }
            return std::make_pair(HMM_ERROR_SPACE_NOT_ENOUGH, std::shared_ptr<OckHmmHMObjectExt>());
        } else if (policy == OckHmmMemoryAllocatePolicy::DEVICE_DDR_ONLY) {
            return Malloc(allocSet->devData, byteSize);
        } else {
            return Malloc(allocSet->hostData, byteSize);
        }
    }
    std::pair<OckHmmErrorCode, std::shared_ptr<OckHmmHMObjectExt>> Malloc(
        std::shared_ptr<OckHmmSubMemoryAlloc> memAlloc, uint64_t byteSize)
    {
        auto addr = memAlloc->Alloc(byteSize);
        if (!addr) {
            return std::make_pair(GetSpaceNotEnoughErroCode(memAlloc.get()), std::shared_ptr<OckHmmHMObjectExt>());
        }
        auto idInfo = csnGenerator->NewId();
        if (!idInfo.first) {
            return std::make_pair(HMM_ERROR_HMO_NO_AVAIBLE_ID, std::shared_ptr<OckHmmHMObjectExt>());
        }
        return std::make_pair(HMM_SUCCESS,
            OckHmmHMObjectExt::Create(OckHmmHMOObjectIDGenerator::Gen(deviceInfo->deviceId, addr, idInfo.second),
                *this,
                service,
                csnGenerator,
                std::make_unique<OckHmmMemoryGuardExt>(memAlloc, addr, byteSize)));
    }
    std::unique_ptr<OckHmmMemoryGuard> MallocBytes(std::shared_ptr<OckHmmSubMemoryAlloc> memAlloc, uint64_t byteSize)
    {
        auto addr = memAlloc->Alloc(byteSize);
        if (!addr) {
            return std::unique_ptr<OckHmmMemoryGuard>();
        }
        return std::make_unique<OckHmmMemoryGuardExt>(memAlloc, addr, byteSize);
    }
    OckHmmErrorCode GetSpaceNotEnoughErroCode(OckHmmSubMemoryAlloc *memAlloc)
    {
        if (memAlloc == allocSet->devData.get()) {
            return HMM_ERROR_DEVICE_DATA_SPACE_NOT_ENOUGH;
        }
        if (memAlloc == allocSet->hostData.get()) {
            return HMM_ERROR_HOST_DATA_SPACE_NOT_ENOUGH;
        }
        return HMM_ERROR_SPACE_NOT_ENOUGH;
    }
    std::shared_ptr<acladapter::OckAsyncTaskExecuteService> service;
    std::shared_ptr<OckHmmDeviceInfo> deviceInfo;
    std::shared_ptr<cpu_set_t> cpuSet;
    std::shared_ptr<OckHmmMemorySpecification> memorySpec;
    std::shared_ptr<OckHmmSubMemoryAllocSet> allocSet;
    std::shared_ptr<HmoCsnGenerator> csnGenerator;
    std::unordered_map<OckHmmHMObjectExt *, std::shared_ptr<OckHmmHMObjectExt>> hmoHolderMap{};
    std::mutex mutex{};
};
namespace ext {
std::shared_ptr<OckHmmSingleDeviceMgr> CreateSingleDeviceMgr(std::shared_ptr<OckHmmDeviceInfo> deviceInfo,
    std::shared_ptr<acladapter::OckAsyncTaskExecuteService> service, std::shared_ptr<OckHmmSubMemoryAllocSet> allocSet)
{
    return std::make_shared<OckHmmSingleDeviceMgrExt>(deviceInfo, service, allocSet);
}
}  // namespace ext
} // namespace hmm
} // namespace ock