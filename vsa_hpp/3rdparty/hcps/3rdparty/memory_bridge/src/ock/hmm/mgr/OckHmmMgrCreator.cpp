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


#include <cstdint>
#include "ock/acladapter/utils/OckSyncUtils.h"
#include "ock/hmm/mgr/OckHmmHeteroMemoryMgr.h"
#include "ock/hmm/mgr/OckHmmSingleDeviceMgrExt.h"
#include "ock/hmm/mgr/OckHmmComposeDeviceMgrExt.h"
#include "ock/hmm/mgr/OckHmmMgrCreator.h"
namespace ock {
namespace hmm {
std::string MemoryNameToString(OckHmmMemoryName name)
{
    switch (name) {
        case OckHmmMemoryName::DEVICE_DATA:
            return "DEVICE_DATA";
        case OckHmmMemoryName::DEVICE_SWAP:
            return "DEVICE_SWAP";
        case OckHmmMemoryName::HOST_DATA:
            return "HOST_DATA";
        case OckHmmMemoryName::HOST_SWAP:
            return "HOST_SWAP";
        default:
            return "";
    }
}
std::pair<OckHmmErrorCode, std::shared_ptr<OckHmmSingleDeviceMgr>> OckHmmMgrCreator::Create(
    const OckHmmDeviceInfo &deviceInfo, uint32_t timeout)
{
    auto service = acladapter::OckAsyncTaskExecuteService::Create(
        deviceInfo.deviceId, deviceInfo.cpuSet, {{acladapter::OckTaskResourceType::HMM, deviceInfo.transferThreadNum}});
    if (service->StartErrorCode() != HMM_SUCCESS) {
        return std::make_pair(service->StartErrorCode(), std::shared_ptr<OckHmmSingleDeviceMgr>());
    }
    return Create(deviceInfo, service, timeout);
}

std::pair<OckHmmErrorCode, std::shared_ptr<OckHmmSingleDeviceMgr>> OckHmmMgrCreator::Create(
    const OckHmmDeviceInfo &deviceInfo, std::shared_ptr<acladapter::OckAsyncTaskExecuteService> service,
    uint32_t timeout)
{
    acladapter::OckSyncUtils syncUtils(*service);
    auto devDataMemory = syncUtils.Malloc(deviceInfo.memorySpec.devSpec.maxDataCapacity,
        hmm::OckHmmHeteroMemoryLocation::DEVICE_DDR, timeout);
    if (devDataMemory.first != HMM_SUCCESS) {
        OCK_HMM_LOG_ERROR("malloc device data area(" << deviceInfo.memorySpec.devSpec.maxDataCapacity <<
            ") failed, deviceId = " << deviceInfo.deviceId << ", retCode = " << devDataMemory.first);
        return std::make_pair(devDataMemory.first, std::shared_ptr<OckHmmSingleDeviceMgr>());
    }
    auto devSwapMemory = syncUtils.Malloc(deviceInfo.memorySpec.devSpec.maxSwapCapacity,
        hmm::OckHmmHeteroMemoryLocation::DEVICE_DDR, timeout);
    if (devSwapMemory.first != HMM_SUCCESS) {
        OCK_HMM_LOG_ERROR("malloc device swap area(" << deviceInfo.memorySpec.devSpec.maxSwapCapacity <<
            ") failed, deviceId = " << deviceInfo.deviceId << ", retCode = " << devSwapMemory.first);
        return std::make_pair(devSwapMemory.first, std::shared_ptr<OckHmmSingleDeviceMgr>());
    }
    auto hostDataMemory = syncUtils.Malloc(deviceInfo.memorySpec.hostSpec.maxDataCapacity,
        hmm::OckHmmHeteroMemoryLocation::LOCAL_HOST_MEMORY, timeout);
    if (hostDataMemory.first != HMM_SUCCESS) {
        OCK_HMM_LOG_ERROR("malloc host data area(" << deviceInfo.memorySpec.hostSpec.maxDataCapacity <<
            ") failed, retCode = " << hostDataMemory.first);
        return std::make_pair(hostDataMemory.first, std::shared_ptr<OckHmmSingleDeviceMgr>());
    }
    auto hostSwapMemory = syncUtils.Malloc(deviceInfo.memorySpec.hostSpec.maxSwapCapacity,
        hmm::OckHmmHeteroMemoryLocation::LOCAL_HOST_MEMORY, timeout);
    if (hostSwapMemory.first != HMM_SUCCESS) {
        OCK_HMM_LOG_ERROR("malloc host swap area(" << deviceInfo.memorySpec.hostSpec.maxSwapCapacity <<
            ") failed, retCode = " << hostSwapMemory.first);
        return std::make_pair(hostSwapMemory.first, std::shared_ptr<OckHmmSingleDeviceMgr>());
    }
    auto allocSet =
        std::make_shared<OckHmmSubMemoryAllocSet>(OckHmmSubMemoryAlloc::Create(std::move(devDataMemory.second),
        deviceInfo.memorySpec.devSpec.maxDataCapacity, MemoryNameToString(OckHmmMemoryName::DEVICE_DATA)),
        OckHmmSubMemoryAlloc::Create(std::move(devSwapMemory.second), deviceInfo.memorySpec.devSpec.maxSwapCapacity,
        MemoryNameToString(OckHmmMemoryName::DEVICE_SWAP)),
        OckHmmSubMemoryAlloc::Create(std::move(hostDataMemory.second), deviceInfo.memorySpec.hostSpec.maxDataCapacity,
        MemoryNameToString(OckHmmMemoryName::HOST_DATA)),
        OckHmmSubMemoryAlloc::Create(std::move(hostSwapMemory.second), deviceInfo.memorySpec.hostSpec.maxSwapCapacity,
        MemoryNameToString(OckHmmMemoryName::HOST_SWAP)));
    return std::make_pair(HMM_SUCCESS,
        ext::CreateSingleDeviceMgr(std::make_shared<OckHmmDeviceInfo>(deviceInfo), service, allocSet));
}

std::pair<OckHmmErrorCode, std::shared_ptr<OckHmmComposeDeviceMgr>> OckHmmMgrCreator::Create(
    std::shared_ptr<OckHmmDeviceInfoVec> deviceInfoVec, uint32_t timeout)
{
    std::vector<std::shared_ptr<OckHmmSingleDeviceMgr>> deviceHmmMgr;
    for (auto &device : *deviceInfoVec) {
        auto devRet = OckHmmMgrCreator::Create(device, timeout);
        if (devRet.first != HMM_SUCCESS) {
            return std::make_pair(devRet.first, std::shared_ptr<OckHmmComposeDeviceMgr>());
        }
        deviceHmmMgr.push_back(devRet.second);
    }
    return std::make_pair(HMM_SUCCESS, ext::CreateComposeDeviceMgr(deviceHmmMgr));
}

std::pair<OckHmmErrorCode, std::shared_ptr<OckHmmComposeDeviceMgr>> OckHmmMgrCreator::Create(
    std::shared_ptr<OckHmmDeviceInfoVec> deviceInfoVec, std::shared_ptr<acladapter::OckAsyncTaskExecuteService> service,
    uint32_t timeout)
{
    std::vector<std::shared_ptr<OckHmmSingleDeviceMgr>> deviceHmmMgr;
    for (auto &device : *deviceInfoVec) {
        auto devRet = OckHmmMgrCreator::Create(device, service, timeout);
        if (devRet.first != HMM_SUCCESS) {
            return std::make_pair(devRet.first, std::shared_ptr<OckHmmComposeDeviceMgr>());
        }
        deviceHmmMgr.push_back(devRet.second);
    }
    return std::make_pair(HMM_SUCCESS, ext::CreateComposeDeviceMgr(deviceHmmMgr));
}
} // namespace hmm
} // namespace ock