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
#include <atomic>
#include "ock/log/OckLogger.h"
#include "ock/hmm/mgr/data/OckHmmBufferData.h"
#include "ock/hmm/mgr/data/OckHmmHMOObjectIDGenerator.h"
#include "ock/acladapter/utils/OckSyncUtils.h"
#include "ock/hmm/mgr/OckHmmHMOBufferExt.h"
namespace ock {
namespace hmm {
class OckHmmHMOBufferDftGen : public OckHmmHMOBufferExt {
public:
    ~OckHmmHMOBufferDftGen() noexcept override = default;
    explicit OckHmmHMOBufferDftGen(std::shared_ptr<OckHmmBufferData> bufferData) : data(bufferData)
    {}
    uintptr_t Address(void) const override
    {
        return data->addr;
    }
    uint64_t Size(void) const override
    {
        return data->byteSize;
    }
    uint64_t Offset(void) const override
    {
        return data->offset;
    }
    OckHmmHMOObjectID GetId(void) const override
    {
        return data->hmoObjId;
    }
    OckHmmHeteroMemoryLocation Location(void) const override
    {
        return data->location;
    }

protected:
    std::shared_ptr<OckHmmBufferData> data;
};
class OckHmmHMOBufferExtCopyImpl : public OckHmmHMOBufferDftGen {
public:
    ~OckHmmHMOBufferExtCopyImpl() noexcept override
    {
        OCK_HMM_LOG_DEBUG("~OckHmmHMOBufferExtCopyImpl" << *this);
        ReleaseData();
    }
    OckHmmHMOBufferExtCopyImpl(OckHmmErrorCode retCode, std::unique_ptr<OckHmmMemoryGuard> &&memGuard,
        uint64_t offset, OckHmmHMObjectExt &hmoExt)
        : OckHmmHMOBufferDftGen(std::make_shared<OckHmmBufferData>(
            memGuard->Addr(), memGuard->ByteSize(), offset, memGuard->Location(), hmoExt.GetId())),
          errorCode(retCode), memoryGuard(std::move(memGuard)), hmo(hmoExt), released(false)
    {
        OCK_HMM_LOG_DEBUG("OckHmmHMOBufferExtCopyImpl" << *this);
    }

    OckHmmErrorCode ErrorCode(void) const override
    {
        return errorCode;
    }
    OckHmmErrorCode FlushData(void) override
    {
        // 这里增加锁，防止flushData的时候，源数据的内存被释放
        std::lock_guard<std::mutex> guard(mutex);
        if (memoryGuard.get() == nullptr) {
            OCK_HMM_LOG_WARN("Can not flush. " << *data << " has released.");
            return HMM_ERROR_HMO_BUFFER_RELEASED;
        }
        auto service = hmo.Service();
        acladapter::OckSyncUtils syncUtils(*service);
        uint8_t *buffAddr = (uint8_t *)memoryGuard->Addr();
        uint8_t *hmoAddr = (uint8_t *)hmo.Addr();
        return syncUtils.Copy((void *)(hmoAddr + data->offset),
            memoryGuard->ByteSize(),
            (void *)buffAddr,
            memoryGuard->ByteSize(),
            acladapter::CalcMemoryCopyKind(data->location, hmo.Location()));
    }
    void ReleaseData(void) override
    {
        // 这里增加锁，防止内存被释放的时候，还有人在读取内存数据
        std::lock_guard<std::mutex> guard(mutex);
        if (memoryGuard.get() == nullptr) {
            return;
        }
        OCK_HMM_LOG_DEBUG("Release data " << data << " has released.");
        memoryGuard.reset();
        released.store(true);
    }
    OckHmmHMObjectExt &GetHmo(void) override
    {
        return hmo;
    }
    bool Released(void) const override
    {
        return released.load();
    }

private:
    const OckHmmErrorCode errorCode;
    std::unique_ptr<OckHmmMemoryGuard> memoryGuard;
    OckHmmHMObjectExt &hmo;
    std::atomic<bool> released;
    // 这里的mutex主要lock数据释放、Flush操作，防止并发释放、Flush， 对于数据读操作， 不用加锁
    mutable std::mutex mutex{};
};

class OckHmmHMOBufferExtRefImpl : public OckHmmHMOBufferDftGen {
public:
    ~OckHmmHMOBufferExtRefImpl() noexcept override
    {
        OCK_HMM_LOG_DEBUG("~OckHmmHMOBufferExtRefImpl" << *this);
    }
    OckHmmHMOBufferExtRefImpl(uint64_t offset, uint64_t byteSize, OckHmmHMObjectExt &hmoExt)
        : OckHmmHMOBufferDftGen(std::make_shared<OckHmmBufferData>(hmoExt.Addr() + offset, byteSize, offset,
        hmoExt.Location(), hmoExt.GetId())),
          hmo(hmoExt),
          released(false)
    {
        OCK_HMM_LOG_DEBUG("OckHmmHMOBufferExtRefImpl" << *this);
    }
    OckHmmErrorCode ErrorCode(void) const override
    {
        return HMM_SUCCESS;
    }
    OckHmmErrorCode FlushData(void) override
    {
        return HMM_SUCCESS;
    }
    void ReleaseData(void) override
    {
        released.store(true);
    }
    OckHmmHMObjectExt &GetHmo(void) override
    {
        return hmo;
    }
    bool Released(void) const override
    {
        return released.load();
    }

private:
    OckHmmHMObjectExt &hmo;
    std::atomic<bool> released;
};

std::shared_ptr<OckHmmHMOBufferExt> OckHmmHMOBufferExt::Create(OckHmmErrorCode errorCode,
    std::unique_ptr<OckHmmMemoryGuard> &&memoryGuard, uint64_t offset, OckHmmHMObjectExt &hmo)
{
    return std::make_shared<OckHmmHMOBufferExtCopyImpl>(errorCode, std::move(memoryGuard), offset, hmo);
}
std::shared_ptr<OckHmmHMOBufferExt> OckHmmHMOBufferExt::Create(
    uint64_t offset, uint64_t byteSize, OckHmmHMObjectExt &hmo)
{
    return std::make_shared<OckHmmHMOBufferExtRefImpl>(offset, byteSize, hmo);
}

OckHmmHMOBufferOutter::OckHmmHMOBufferOutter(std::shared_ptr<OckHmmHMOBufferExt> bufferExt) : buffer(bufferExt)
{}
uintptr_t OckHmmHMOBufferOutter::Address(void) const
{
    if (buffer.get() == nullptr || buffer->Released()) {
        return 0UL;
    }
    return buffer->Address();
}
uint64_t OckHmmHMOBufferOutter::Size(void) const
{
    if (buffer.get() == nullptr || buffer->Released()) {
        return 0UL;
    }
    return buffer->Size();
}
uint64_t OckHmmHMOBufferOutter::Offset(void) const
{
    if (buffer.get() == nullptr || buffer->Released()) {
        return 0UL;
    }
    return buffer->Offset();
}
OckHmmHeteroMemoryLocation OckHmmHMOBufferOutter::Location(void) const
{
    if (buffer.get() == nullptr || buffer->Released()) {
        return OckHmmHeteroMemoryLocation::DEVICE_DDR;
    }
    return buffer->Location();
}
OckHmmHMOObjectID OckHmmHMOBufferOutter::GetId(void) const
{
    if (buffer.get() == nullptr || buffer->Released()) {
        return 0UL;
    }
    return buffer->GetId();
}
OckHmmErrorCode OckHmmHMOBufferOutter::FlushData(void)
{
    if (buffer.get() == nullptr || buffer->Released()) {
        return HMM_ERROR_HMO_BUFFER_NOT_ALLOCED;
    }
    if (buffer->GetHmo().Released()) {
        return HMM_ERROR_HMO_OBJECT_NOT_EXISTS;
    }
    return buffer->FlushData();
}
void OckHmmHMOBufferOutter::ReleaseData(void)
{
    if (buffer.get() == nullptr || buffer->Released()) {
        return;
    }
    buffer->ReleaseData();
}
OckHmmErrorCode OckHmmHMOBufferOutter::ErrorCode(void) const
{
    if (buffer.get() == nullptr || buffer->Released()) {
        return HMM_ERROR_WAIT_TIME_OUT;
    }
    return buffer->ErrorCode();
}
OckHmmHMOBufferExt *OckHmmHMOBufferOutter::Inner(void)
{
    return buffer.get();
}
std::shared_ptr<OckHmmHMOBuffer> OckHmmHMOBufferOutter::Create(std::shared_ptr<OckHmmHMOBufferExt> buffer)
{
    return std::make_shared<OckHmmHMOBufferOutter>(buffer);
}

std::ostream &operator<<(std::ostream &os, const OckHmmHMOBufferBase &buffer)
{
    return os << "{'size':" << buffer.Size() << "}";
}
std::ostream &operator<<(std::ostream &os, const OckHmmHMOBuffer &buffer)
{
    return os << "{'objId':" << buffer.GetId()
              << ",'deviceId':" << OckHmmHMOObjectIDGenerator::ParseDeviceId(buffer.GetId())
              << ",'location':" << buffer.Location() << ",'offset':" << buffer.Offset() << ",'size':" << buffer.Size()
              << ",'errorCode':" << buffer.ErrorCode() << "}";
}
}  // namespace hmm
}  // namespace ock