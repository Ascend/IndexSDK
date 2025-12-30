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

#include "ock/hcps/handler/OckHandlerHmmHelper.h"
#include "ock/log/OckHcpsLogger.h"
namespace ock {
namespace hcps {
namespace handler {
namespace helper {
std::shared_ptr<hmm::OckHmmHMObject> MakeHmo(hmm::OckHmmHeteroMemoryMgrBase &hmmMgr, uint64_t hmoBytes,
    hmm::OckHmmMemoryAllocatePolicy policy, hmm::OckHmmErrorCode &errorCode)
{
    if (errorCode != hmm::HMM_SUCCESS) {
        return std::shared_ptr<hmm::OckHmmHMObject>();
    }
    auto ret = hmmMgr.Alloc(hmoBytes, policy);
    if (ret.first != hmm::HMM_SUCCESS) {
        errorCode = ret.first;
    }
    return ret.second;
}
std::shared_ptr<hmm::OckHmmHMObject> MakeHostHmo(
    hmm::OckHmmHeteroMemoryMgrBase &hmmMgr, uint64_t hmoBytes, hmm::OckHmmErrorCode &errorCode)
{
    if (errorCode != hmm::HMM_SUCCESS) {
        return std::shared_ptr<hmm::OckHmmHMObject>();
    }
    return MakeHmo(hmmMgr, hmoBytes, hmm::OckHmmMemoryAllocatePolicy::LOCAL_HOST_ONLY, errorCode);
}
std::shared_ptr<hmm::OckHmmHMObject> MakeHostHmo(
    OckHeteroHandler &handler, uint64_t hmoBytes, hmm::OckHmmErrorCode &errorCode)
{
    if (errorCode != hmm::HMM_SUCCESS) {
        return std::shared_ptr<hmm::OckHmmHMObject>();
    }
    return MakeHostHmo(handler.HmmMgr(), hmoBytes, errorCode);
}
std::shared_ptr<hmm::OckHmmHMObject> CopyToHostHmo(OckHeteroHandler &handler,
    std::shared_ptr<hmm::OckHmmHMObject> srcHmo, hmm::OckHmmErrorCode &errorCode)
{
    if (srcHmo.get() == nullptr) {
        return std::shared_ptr<hmm::OckHmmHMObject>();
    }
    return CopyToHostHmo(handler, *srcHmo, errorCode);
}
std::shared_ptr<hmm::OckHmmHMObject> CopyToHostHmo(OckHeteroHandler &handler, hmm::OckHmmHMObject &srcHmo,
    hmm::OckHmmErrorCode &errorCode)
{
    if (errorCode != hmm::HMM_SUCCESS) {
        return std::shared_ptr<hmm::OckHmmHMObject>();
    }
    errorCode = hcps::handler::helper::UseIncBindMemory(handler, srcHmo.GetByteSize(), "CopyToHostHmo");
    if (errorCode != hmm::HMM_SUCCESS) {
        return std::shared_ptr<hmm::OckHmmHMObject>();
    }
    auto newHmo = MakeHostHmo(handler, srcHmo.GetByteSize(), errorCode);
    if (errorCode != hmm::HMM_SUCCESS) {
        return std::shared_ptr<hmm::OckHmmHMObject>();
    }
    errorCode = handler.HmmMgr().CopyHMO(*newHmo, 0ULL, srcHmo, 0ULL, srcHmo.GetByteSize());
    if (errorCode != hmm::HMM_SUCCESS) {
        return std::shared_ptr<hmm::OckHmmHMObject>();
    }
    return newHmo;
}
std::deque<std::shared_ptr<hmm::OckHmmHMObject>> MakeHostHmoDeque(
    hmm::OckHmmHeteroMemoryMgrBase &hmmMgr, uint64_t hmoBytes, uint64_t count, hmm::OckHmmErrorCode &errorCode)
{
    if (errorCode != hmm::HMM_SUCCESS) {
        return std::deque<std::shared_ptr<hmm::OckHmmHMObject>>();
    }
    std::deque<std::shared_ptr<hmm::OckHmmHMObject>> ret;
    for (uint64_t i = 0; i < count; ++i) {
        ret.push_back(MakeHostHmo(hmmMgr, hmoBytes, errorCode));
    }
    return ret;
}
std::shared_ptr<hmm::OckHmmHMObject> MakeDeviceHmo(
    hmm::OckHmmHeteroMemoryMgrBase &hmmMgr, uint64_t hmoBytes, hmm::OckHmmErrorCode &errorCode)
{
    if (errorCode != hmm::HMM_SUCCESS) {
        return std::shared_ptr<hmm::OckHmmHMObject>();
    }
    return MakeHmo(hmmMgr, hmoBytes, hmm::OckHmmMemoryAllocatePolicy::DEVICE_DDR_ONLY, errorCode);
}
std::shared_ptr<hmm::OckHmmHMObject> MakeDeviceHmo(
    OckHeteroHandler &handler, uint64_t hmoBytes, hmm::OckHmmErrorCode &errorCode)
{
    if (errorCode != hmm::HMM_SUCCESS) {
        return std::shared_ptr<hmm::OckHmmHMObject>();
    }
    return MakeDeviceHmo(handler.HmmMgr(), hmoBytes, errorCode);
}
std::deque<std::shared_ptr<hmm::OckHmmHMObject>> MakeDeviceHmoDeque(
    hmm::OckHmmHeteroMemoryMgrBase &hmmMgr, uint64_t hmoBytes, uint64_t count, hmm::OckHmmErrorCode &errorCode)
{
    if (errorCode != hmm::HMM_SUCCESS) {
        return std::deque<std::shared_ptr<hmm::OckHmmHMObject>>();
    }
    std::deque<std::shared_ptr<hmm::OckHmmHMObject>> ret;
    for (uint64_t i = 0; i < count; ++i) {
        ret.push_back(MakeDeviceHmo(hmmMgr, hmoBytes, errorCode));
    }
    return ret;
}
std::shared_ptr<OckHeteroStreamBase> MakeStream(
    OckHeteroHandler &handler, hmm::OckHmmErrorCode &errorCode, OckDevStreamType streamType)
{
    if (errorCode != hmm::HMM_SUCCESS) {
        return std::shared_ptr<OckHeteroStreamBase>();
    }
    auto streamRet = OckHeteroStreamBase::Create(handler.Service(), streamType);
    errorCode = streamRet.first;
    return streamRet.second;
}
hmm::OckHmmErrorCode UseIncBindMemory(std::shared_ptr<hmm::OckHmmHeteroMemoryMgrBase> hmmMgr, uint64_t incByteSize,
    std::string incLocationName)
{
    auto errorCode = hmm::HMM_SUCCESS;
    auto usedInfo = hmmMgr->GetUsedInfo(FRAGMENT_SIZE_THRESHOLD);
    uint64_t availableSize = (usedInfo->hostUsedInfo.leftBytes >= usedInfo->hostUsedInfo.unusedFragBytes) ?
        (usedInfo->hostUsedInfo.leftBytes - usedInfo->hostUsedInfo.unusedFragBytes) :
        0ULL;
    if (availableSize <= incByteSize) {
        hmm::OckHmmSingleDeviceMgr *singleMgr = dynamic_cast<hmm::OckHmmSingleDeviceMgr *>(hmmMgr.get());
        if (singleMgr == nullptr) {
            OCK_HCPS_LOG_ERROR("[singleMgr] " << incLocationName << "increase host memory = " << incByteSize <<
                ", host UsedInfo: " << *usedInfo);
            return hmm::HMM_ERROR_UNKOWN_INNER_ERROR;
        }
        if (incByteSize < MIN_INCREASE_MEMORY_BYTESIZE) {
            incByteSize = MIN_INCREASE_MEMORY_BYTESIZE;
        }
        errorCode = singleMgr->IncBindMemory(hmm::OckHmmHeteroMemoryLocation::LOCAL_HOST_MEMORY, incByteSize);
        if (errorCode != hmm::HMM_SUCCESS) {
            OCK_HCPS_LOG_ERROR(incLocationName << " need host memory = " << incByteSize << ", host UsedInfo: " <<
                *usedInfo << " errorCode=" << errorCode);
            return errorCode;
        } else {
            usedInfo = hmmMgr->GetUsedInfo(FRAGMENT_SIZE_THRESHOLD);
            OCK_HCPS_LOG_INFO(incLocationName << " increase host memory = " << incByteSize << ", host UsedInfo: " <<
                *usedInfo);
        }
    }
    return errorCode;
}
hmm::OckHmmErrorCode UseIncBindMemory(OckHeteroHandler &handler, uint64_t incByteSize, std::string incLocationName)
{
    return UseIncBindMemory(handler.HmmMgrPtr(), incByteSize, incLocationName);
}
void MergeMultiHMObjectsToOne(hcps::handler::OckHeteroHandler &handler,
    const std::vector<std::shared_ptr<hmm::OckHmmHMObject>> &hmoVector, std::shared_ptr<hmm::OckHmmHMObject> hmObject,
    hmm::OckHmmErrorCode &errorCode)
{
    if (hmObject == nullptr || errorCode != hmm::HMM_SUCCESS) {
        return;
    }
    uint64_t dstOffset = 0UL;
    for (size_t i = 0; i < hmoVector.size(); ++i) {
        errorCode =
            handler.HmmMgr().CopyHMO(*hmObject, dstOffset, *hmoVector.at(i), 0ULL, hmoVector.at(i)->GetByteSize());
        dstOffset += hmoVector.at(i)->GetByteSize();
    }
}
std::shared_ptr<hmm::OckHmmHMObject> MergeMultiHMObjectsToHost(hcps::handler::OckHeteroHandler &handler,
    const std::vector<std::shared_ptr<hmm::OckHmmHMObject>> &hmoVector, hmm::OckHmmErrorCode &errorCode)
{
    if (hmoVector.size() == 0 || errorCode != hmm::HMM_SUCCESS) {
        return nullptr;
    }
    uint64_t deviceHMObjectByteSize = hmoVector.size() * hmoVector.front()->GetByteSize();
    std::shared_ptr<hmm::OckHmmHMObject> hostHMObject =
        hcps::handler::helper::MakeHostHmo(handler, deviceHMObjectByteSize, errorCode);
    if (errorCode != hmm::HMM_SUCCESS) {
        return std::shared_ptr<hmm::OckHmmHMObject>();
    }
    MergeMultiHMObjectsToOne(handler, hmoVector, hostHMObject, errorCode);
    return hostHMObject;
}
std::shared_ptr<hmm::OckHmmHMObject> MergeMultiHMObjectsToDevice(hcps::handler::OckHeteroHandler &handler,
    const std::vector<std::shared_ptr<hmm::OckHmmHMObject>> &hmoVector, hmm::OckHmmErrorCode &errorCode)
{
    if (hmoVector.size() == 0 || errorCode != hmm::HMM_SUCCESS) {
        return nullptr;
    }
    uint64_t hostHMObjectByteSize = hmoVector.size() * hmoVector.front()->GetByteSize();
    std::shared_ptr<hmm::OckHmmHMObject> deviceHMObject =
        hcps::handler::helper::MakeDeviceHmo(handler, hostHMObjectByteSize, errorCode);
    if (errorCode != hmm::HMM_SUCCESS) {
        return std::shared_ptr<hmm::OckHmmHMObject>();
    }
    MergeMultiHMObjectsToOne(handler, hmoVector, deviceHMObject, errorCode);
    return deviceHMObject;
}
} // namespace helper
} // namespace handler
} // namespace hcps
} // namespace ock