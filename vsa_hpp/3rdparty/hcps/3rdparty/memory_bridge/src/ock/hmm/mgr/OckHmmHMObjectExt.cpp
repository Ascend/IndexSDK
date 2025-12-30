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

#include <mutex>
#include <memory>
#include <atomic>
#include <limits>
#include <unordered_map>
#include "ock/log/OckLogger.h"
#include "ock/utils/OstreamUtils.h"
#include "ock/hmm/mgr/task/OckDefaultAsyncResult.h"
#include "ock/hmm/mgr/data/OckHmmHMOObjectIDGenerator.h"
#include "ock/hmm/mgr/data/OckHmmBufferData.h"
#include "ock/hmm/mgr/OckHmmHMOBufferExt.h"
#include "ock/hmm/mgr/checker/OckHmmHeteroMemoryMgrParamCheck.h"
#include "ock/hmm/mgr/task/OckGetBufferTask.h"
#include "ock/hmm/mgr/OckHmmHMObjectExt.h"
namespace ock {
namespace hmm {
class OckHmmHMObjectExtImpl : public OckHmmHMObjectExt {
public:
    virtual ~OckHmmHMObjectExtImpl() noexcept
    {
        OCK_HMM_LOG_DEBUG("~OckHmmHMObjectExtImpl" << *this);
        csnGenerator->DelId(OckHmmHMOObjectIDGenerator::ParseCsnId(dataInfo.hmoObjId));
    }
    OckHmmHMObjectExtImpl(OckHmmHMOObjectID objectId, OckHmmSubMemoryAllocDispatcher &memAllocDispatcher,
        std::shared_ptr<acladapter::OckAsyncTaskExecuteService> taskService, std::shared_ptr<HmoCsnGenerator> csnGen,
        std::unique_ptr<OckHmmMemoryGuard> &&memGuard)
        : allocDispatcher(memAllocDispatcher),
          dataInfo(memGuard->Addr(), memGuard->ByteSize(), 0U, memGuard->Location(), objectId),
          service(taskService), csnGenerator(csnGen), memoryGuard(std::move(memGuard)), released(false)
    {
        OCK_HMM_LOG_DEBUG("OckHmmHMObjectExtImpl" << *this);
    }

    OckHmmHMOObjectID GetId(void) const override
    {
        return dataInfo.hmoObjId;
    }
    OckHmmDeviceId IntimateDeviceId(void) const override
    {
        return OckHmmHMOObjectIDGenerator::ParseDeviceId(dataInfo.hmoObjId);
    }
    uint64_t GetByteSize(void) const override
    {
        return dataInfo.byteSize;
    }
    OckHmmHeteroMemoryLocation Location(void) const override
    {
        return dataInfo.location;
    }
    void ForceReleaseMemory(void) override
    {
        std::lock_guard<std::mutex> guard(mutex);
        memoryGuard.reset();
        released.store(true);
    }
    std::shared_ptr<OckHmmHMOBuffer> GetBuffer(
        OckHmmHeteroMemoryLocation location, uint64_t offset, uint64_t length, uint32_t timeout) override
    {
        auto task = GetBufferAsync(location, offset, length);
        return task->WaitResult(timeout);
    }
    std::shared_ptr<OckHmmAsyncResult<OckHmmHMOBuffer>> GetBufferAsync(
        OckHmmHeteroMemoryLocation location, uint64_t offset, uint64_t length) override
    {
        std::lock_guard<std::mutex> guard(mutex);
        if (location == dataInfo.location) {
            return std::make_shared<OckDefaultAsyncResult<OckHmmHMOBuffer, OckHmmHMOBuffer, OckHmmHMOBuffer>>(
                OckHmmHMOBufferOutter::Create(OckHmmHMOBufferExt::Create(offset, length, *this)));
        }
        auto bridge = std::make_shared<acladapter::OckAsyncResultInnerBridge<OckHmmHMOBufferExt>>();
        service->AddTask(OckGetBufferTask::Create(
            std::make_shared<OckGetBufferParam>(allocDispatcher.SwapAlloc(location), offset, length, *this), bridge));

        return std::make_shared<OckDefaultAsyncResult<OckHmmHMOBuffer,
            acladapter::OckAsyncResultInnerBridge<OckHmmHMOBufferExt>,
            OckHmmHMOBufferOutter>>(bridge);
    }
    void ReleaseBuffer(std::shared_ptr<OckHmmHMOBuffer> buffer) override
    {
        OckHmmHMOBufferOutter *outBuffer = dynamic_cast<OckHmmHMOBufferOutter *>(buffer.get());
        if (outBuffer == nullptr) {
            OCK_HMM_LOG_WARN("Invalid buffer.");
            return;
        }

        outBuffer->ReleaseData();
    }
    uintptr_t Addr(void) const override
    {
        return dataInfo.addr;
    }
    std::shared_ptr<acladapter::OckAsyncTaskExecuteService> Service(void) override
    {
        return service;
    }
    bool Released(void) const override
    {
        return released.load();
    }

private:
    OckHmmSubMemoryAllocDispatcher &allocDispatcher;
    const OckHmmBufferData dataInfo;  // 复制一份数据，减少对mutex的使用
    std::shared_ptr<acladapter::OckAsyncTaskExecuteService> service;
    std::shared_ptr<HmoCsnGenerator> csnGenerator;
    std::unique_ptr<OckHmmMemoryGuard> memoryGuard;
    std::atomic<bool> released;
    mutable std::mutex mutex{};
};
std::ostream &operator<<(std::ostream &os, const OckHmmHMObject &hmo)
{
    OckHmmHMOObjectID objectId = hmo.GetId();
    return os << "{'objectId':" << objectId << ",'csnId':" << OckHmmHMOObjectIDGenerator::ParseCsnId(objectId)
              << ",'size':" << hmo.GetByteSize()
              << ",'deviceId':" << OckHmmHMOObjectIDGenerator::ParseDeviceId(objectId)
              << ",'location':" << hmo.Location() << "}";
}
std::shared_ptr<OckHmmHMObjectExt> OckHmmHMObjectExt::Create(OckHmmHMOObjectID objectId,
    OckHmmSubMemoryAllocDispatcher &allocDispatcher, std::shared_ptr<acladapter::OckAsyncTaskExecuteService> service,
    std::shared_ptr<HmoCsnGenerator> csnGenerator, std::unique_ptr<OckHmmMemoryGuard> &&memoryGuard)
{
    return std::make_shared<OckHmmHMObjectExtImpl>(
        objectId, allocDispatcher, service, csnGenerator, std::move(memoryGuard));
}

OckHmmHMObjectOutter::~OckHmmHMObjectOutter() noexcept
{
    OCK_HMM_LOG_DEBUG("~OckHmmHMObjectOutter" << *this);
    allocDispatcher.InnerFree(hmo);
}
OckHmmHMObjectOutter::OckHmmHMObjectOutter(
    std::shared_ptr<OckHmmHMObjectExt> hmoExt, OckHmmSubMemoryAllocDispatcher &memAllocDispatcher)
    : hmo(hmoExt), allocDispatcher(memAllocDispatcher)
{
    OCK_HMM_LOG_DEBUG("OckHmmHMObjectOutter" << *this);
}
OckHmmHMOObjectID OckHmmHMObjectOutter::GetId(void) const
{
    if (hmo.get() == nullptr || hmo->Released()) {
        return 0ULL;
    }
    return hmo->GetId();
}

OckHmmDeviceId OckHmmHMObjectOutter::IntimateDeviceId(void) const
{
    if (hmo.get() == nullptr || hmo->Released()) {
        return std::numeric_limits<uint16_t>::max();
    }
    return hmo->IntimateDeviceId();
}
uint64_t OckHmmHMObjectOutter::GetByteSize(void) const
{
    if (hmo.get() == nullptr || hmo->Released()) {
        return 0ULL;
    }
    return hmo->GetByteSize();
}
OckHmmHeteroMemoryLocation OckHmmHMObjectOutter::Location(void) const
{
    if (hmo.get() == nullptr || hmo->Released()) {
        return OckHmmHeteroMemoryLocation::DEVICE_DDR;
    }
    return hmo->Location();
}
std::shared_ptr<OckHmmHMOBuffer> OckHmmHMObjectOutter::GetBuffer(
    OckHmmHeteroMemoryLocation location, uint64_t offset, uint64_t length, uint32_t timeout)
{
    if (hmo.get() == nullptr || hmo->Released()) {
        OCK_HMM_LOG_ERROR("the hmo(" << *hmo << ") does not exist!");
        return std::shared_ptr<OckHmmHMOBuffer>();
    }
    if (OckHmmHeteroMemoryMgrParamCheck::CheckGetBuffer(location, offset, length, *hmo, allocDispatcher) !=
        HMM_SUCCESS) {
        return std::shared_ptr<OckHmmHMOBuffer>();
    }
    return hmo->GetBuffer(location, offset, length, timeout);
}
std::shared_ptr<OckHmmAsyncResult<OckHmmHMOBuffer>> OckHmmHMObjectOutter::GetBufferAsync(
    OckHmmHeteroMemoryLocation location, uint64_t offset, uint64_t length)
{
    if (hmo.get() == nullptr || hmo->Released()) {
        OCK_HMM_LOG_ERROR("the hmo(" << *hmo << ") does not exist!");
        return std::shared_ptr<OckHmmAsyncResult<OckHmmHMOBuffer>>();
    }
    if (OckHmmHeteroMemoryMgrParamCheck::CheckGetBuffer(location, offset, length, *hmo, allocDispatcher) !=
        HMM_SUCCESS) {
        return std::shared_ptr<OckHmmAsyncResult<OckHmmHMOBuffer>>();
    }
    return hmo->GetBufferAsync(location, offset, length);
}
void OckHmmHMObjectOutter::ReleaseBuffer(std::shared_ptr<OckHmmHMOBuffer> buffer)
{
    if (hmo.get() == nullptr || hmo->Released()) {
        OCK_HMM_LOG_ERROR("the hmo(" << *hmo << ") does not exist!");
        return;
    }
    hmo->ReleaseBuffer(buffer);
}
OckHmmHMObjectExt *OckHmmHMObjectOutter::GetExtHmo(void)
{
    return hmo.get();
}
uintptr_t OckHmmHMObjectOutter::Addr(void) const
{
    if (hmo.get() == nullptr || hmo->Released()) {
        return 0ULL;
    }
    return hmo->Addr();
}
std::shared_ptr<OckHmmHMObject> OckHmmHMObjectOutter::Create(
    std::shared_ptr<OckHmmHMObjectExt> hmo, OckHmmSubMemoryAllocDispatcher &allocDispatcher)
{
    return std::make_shared<OckHmmHMObjectOutter>(hmo, allocDispatcher);
}
std::ostream &operator<<(std::ostream &os, const std::shared_ptr<OckHmmSubHMObject> &hmo)
{
    return utils::Print(os, hmo);
}
std::ostream &operator<<(std::ostream &os, const std::shared_ptr<OckHmmHMObject> &hmo)
{
    return utils::Print(os, hmo);
}
std::ostream &operator<<(std::ostream &os, const std::shared_ptr<OckHmmHMOBuffer> &buffer)
{
    return utils::Print(os, buffer);
}
}  // namespace hmm
}  // namespace ock