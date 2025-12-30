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

#include <vector>
#include "ock/acladapter/utils/OckSyncUtils.h"
#include "ock/tools/topo/TopoTestPackage.h"

namespace ock {
namespace tools {
namespace topo {
TopoTestPackage::Package::Package(
    uint32_t pkgSize, uint8_t *devAddress, uint8_t *hostAddress, uint8_t *devDstAddress, uint8_t *hostDstAddress)
    : packageSize(pkgSize), deviceAddr(devAddress), hostAddr(hostAddress), deviceDstAddr(devDstAddress),
      hostDstAddr(hostDstAddress)
{}
class TopoTestPackageImpl : public TopoTestPackage {
public:
    virtual ~TopoTestPackageImpl() noexcept
    {
        acladapter::OckSyncUtils syncUtils(service);
        for (auto addrPtr : addrInDevice) {
            syncUtils.Free(addrPtr, hmm::OckHmmHeteroMemoryLocation::DEVICE_DDR);
        }
        for (auto addrPtr : addrInHost) {
            syncUtils.Free(addrPtr, hmm::OckHmmHeteroMemoryLocation::LOCAL_HOST_MEMORY);
        }
    }
    explicit TopoTestPackageImpl(const TopoDetectParam &parameter, acladapter::OckAsyncTaskExecuteService &taskService)
        : errorCode(hmm::HMM_SUCCESS), param(parameter), service(taskService)
    {
        // 2倍线程数再+1，使得每个线程都能并发起来
        const uint32_t preparePackageCount = param.ThreadNumPerDevice() * 2U + 1U;
        acladapter::OckSyncUtils syncUtils(service);
        for (uint32_t i = 0; i < preparePackageCount; ++i) {
            auto devRet = syncUtils.Malloc(param.PackageBytesPerTransfer(),
                hmm::OckHmmHeteroMemoryLocation::DEVICE_DDR);
            if (devRet.first != hmm::HMM_SUCCESS) {
                errorCode = devRet.first;
                break;
            }
            addrInDevice.push_back(devRet.second->ReleaseGuard());
            // 申请host侧内存时赋予一个超时等待，最大等待时间10秒，避免默认最大等待3600秒
            auto hostRet = syncUtils.Malloc(
                param.PackageBytesPerTransfer(), hmm::OckHmmHeteroMemoryLocation::LOCAL_HOST_MEMORY, 10000U);
            if (hostRet.first != hmm::HMM_SUCCESS) {
                errorCode = hostRet.first;
                break;
            }
            addrInHost.push_back(hostRet.second->ReleaseGuard());
        }
    }
    Package GetPackage(uint32_t taskId) override
    {
        uint32_t pos = static_cast<uint32_t>(addrInDevice.size() == 0 ? taskId : taskId % addrInDevice.size());
        uint32_t dstPos =
            static_cast<uint32_t>(addrInDevice.size() == 0 ? (taskId + 1U) : (taskId + 1U) % addrInDevice.size());
        return Package(static_cast<uint32_t>(param.PackageBytesPerTransfer()),
            addrInDevice.at(pos),
            addrInHost.at(pos),
            addrInDevice.at(dstPos),
            addrInHost.at(dstPos));
    }
    hmm::OckHmmErrorCode GetError(void) const override
    {
        return errorCode;
    }

private:
    hmm::OckHmmErrorCode errorCode;
    std::vector<uint8_t *> addrInDevice{};
    std::vector<uint8_t *> addrInHost{};
    const TopoDetectParam &param;
    acladapter::OckAsyncTaskExecuteService &service;
};
std::shared_ptr<TopoTestPackage> TopoTestPackage::Create(
    const TopoDetectParam &param, acladapter::OckAsyncTaskExecuteService &service)
{
    return std::make_shared<TopoTestPackageImpl>(param, service);
}
}  // namespace topo
}  // namespace tools
}  // namespace ock