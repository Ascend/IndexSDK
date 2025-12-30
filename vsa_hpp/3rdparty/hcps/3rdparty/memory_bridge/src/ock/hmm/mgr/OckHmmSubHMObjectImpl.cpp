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
#include "ock/log/OckLogger.h"
#include "ock/hmm/mgr/OckHmmHMObject.h"
namespace ock {
namespace hmm {
class OckHmmSubHMObjectImpl : public OckHmmSubHMObject {
public:
    virtual ~OckHmmSubHMObjectImpl() noexcept = default;
    OckHmmSubHMObjectImpl(std::shared_ptr<OckHmmHMObject> hmoObj, uint64_t offset, uint64_t size)
        : hmo(hmoObj), subOffset(offset), subLength(size)
    {}
    uint64_t GetByteSize(void) const override
    {
        return subLength;
    }
    uintptr_t Addr(void) const override
    {
        return hmo->Addr() + subOffset;
    }
    OckHmmHeteroMemoryLocation Location(void) const override
    {
        return hmo->Location();
    }
    std::shared_ptr<OckHmmHMOBuffer> GetBuffer(
        OckHmmHeteroMemoryLocation location, uint64_t offset, uint64_t length, uint32_t timeout) override
    {
        if (offset + length > subLength) {
            OCK_HMM_LOG_ERROR("GetBuffer addr exceed scope. location(" << location << ")offset(" << offset << ")length("
                                                                       << length << ") subHmo(" << *this << ")");
            return std::shared_ptr<OckHmmHMOBuffer>();
        }
        return hmo->GetBuffer(location, subOffset + offset, length, timeout);
    }

    std::shared_ptr<OckHmmAsyncResult<OckHmmHMOBuffer>> GetBufferAsync(
        OckHmmHeteroMemoryLocation location, uint64_t offset, uint64_t length) override
    {
        if (offset + length > subLength) {
            OCK_HMM_LOG_ERROR("GetBufferAsync addr exceed scope. location("
                              << location << ")offset(" << offset << ")length(" << length << ") subHmo(" << *this
                              << ")");
            return std::shared_ptr<OckHmmAsyncResult<OckHmmHMOBuffer>>();
        }
        return hmo->GetBufferAsync(location, subOffset + offset, length);
    }

    void ReleaseBuffer(std::shared_ptr<OckHmmHMOBuffer> buffer) override
    {
        hmo->ReleaseBuffer(buffer);
    }

    std::shared_ptr<OckHmmHMObject> hmo;
    uint64_t subOffset;
    uint64_t subLength;
};

std::shared_ptr<OckHmmSubHMObject> OckHmmSubHMObject::CreateSubHmo(
    std::shared_ptr<OckHmmSubHMObject> subHmoObject, uint64_t offset, uint64_t length)
{
    if (subHmoObject.get() == nullptr) {
        OCK_HMM_LOG_ERROR("the subHmo input is a nullptr");
        return std::shared_ptr<OckHmmSubHMObject>();
    }
    if (subHmoObject->GetByteSize() == 0) {
        OCK_HMM_LOG_ERROR("the subHmo input has been released.");
        return std::shared_ptr<OckHmmSubHMObject>();
    }
    if (length == 0) {
        OCK_HMM_LOG_ERROR("the length(" << length << ") is not allowed.");
        return std::shared_ptr<OckHmmSubHMObject>();
    }
    if (offset + length > subHmoObject->GetByteSize()) {
        OCK_HMM_LOG_ERROR("CreateSubHmo failed offset(" << offset << ")length(" << length << ") hmo=" << *subHmoObject);
        return std::shared_ptr<OckHmmSubHMObject>();
    }
    auto subHmoImpl = dynamic_cast<OckHmmSubHMObjectImpl *>(subHmoObject.get());
    if (subHmoImpl == nullptr) {
        OCK_HMM_LOG_ERROR("subHmoObject :" << *subHmoObject << " dynamic cast to OckHmmSubHMObjectImpl failed!");
        return std::shared_ptr<OckHmmSubHMObject>();
    }
    return std::make_shared<OckHmmSubHMObjectImpl>(subHmoImpl->hmo, subHmoImpl->subOffset + offset, length);
}

std::shared_ptr<OckHmmSubHMObject> OckHmmHMObject::CreateSubHmo(
    std::shared_ptr<OckHmmHMObject> hmoObject, uint64_t offset, uint64_t length)
{
    if (hmoObject.get() == nullptr) {
        OCK_HMM_LOG_ERROR("the hmo object input is a nullptr");
        return std::shared_ptr<OckHmmSubHMObject>();
    }
    if (hmoObject->GetByteSize() == 0) {
        OCK_HMM_LOG_ERROR("the hmo object input has been released.");
        return std::shared_ptr<OckHmmSubHMObject>();
    }
    if (length == 0) {
        OCK_HMM_LOG_ERROR("the length(" << length << ") is not allowed.");
        return std::shared_ptr<OckHmmSubHMObject>();
    }
    if (offset + length > hmoObject->GetByteSize()) {
        OCK_HMM_LOG_ERROR("CreateSubHmo failed offset(" << offset << ")length(" << length << ") hmo=" << *hmoObject);
        return std::shared_ptr<OckHmmSubHMObject>();
    }
    return std::make_shared<OckHmmSubHMObjectImpl>(hmoObject, offset, length);
}
std::shared_ptr<std::vector<std::shared_ptr<OckHmmSubHMObject>>> OckHmmHMObject::CreateSubHmoList(
    std::shared_ptr<OckHmmHMObject> hmo, uint64_t subHmoBytes)
{
    auto ret = std::make_shared<std::vector<std::shared_ptr<OckHmmSubHMObject>>>();
    if (hmo.get() == nullptr) {
        OCK_HMM_LOG_ERROR("the hmo input is a nullptr.");
        return ret;
    }
    if (hmo->GetByteSize() == 0) {
        OCK_HMM_LOG_ERROR("the hmo input has been released.");
        return ret;
    }
    if (subHmoBytes == 0) {
        OCK_HMM_LOG_ERROR("the subHmoBytes(" << subHmoBytes << ") is not allowed.");
        return ret;
    }
    for (uint64_t startPos = 0; startPos < hmo->GetByteSize(); startPos += subHmoBytes) {
        ret->push_back(
            OckHmmHMObject::CreateSubHmo(hmo, startPos, std::min(subHmoBytes, hmo->GetByteSize() - startPos)));
    }
    return ret;
}
std::ostream &operator<<(std::ostream &os, const OckHmmSubHMObject &hmo)
{
    os << "{'size':" << hmo.GetByteSize() << ", 'location':" << hmo.Location() << "}";
    return os;
}
}  // namespace hmm
}  // namespace ock