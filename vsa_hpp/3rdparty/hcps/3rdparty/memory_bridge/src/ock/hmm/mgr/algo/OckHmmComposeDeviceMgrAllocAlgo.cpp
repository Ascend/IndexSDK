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

#include "ock/hmm/mgr/algo/OckHmmComposeDeviceMgrAllocAlgo.h"
namespace ock {
namespace hmm {
namespace {
using OckHmmHMObjectRetcodePair = std::pair<OckHmmErrorCode, std::shared_ptr<OckHmmHMObject>>;
template <typename ReturnT>
ReturnT MakeEmptyResult(void)
{
    return ReturnT();
}
template <>
OckHmmHMObjectRetcodePair MakeEmptyResult()
{
    return std::make_pair(HMM_ERROR_SPACE_NOT_ENOUGH, std::shared_ptr<OckHmmHMObject>());
}
template <typename ReturnT>
bool IsEmptyResult(const ReturnT &data)
{
    return data.get() == nullptr;
}
template <>
bool IsEmptyResult(const OckHmmHMObjectRetcodePair &data)
{
    return data.first != HMM_SUCCESS;
}
template <typename ReturnT>
struct AllocFunBridge {
    ReturnT operator()(std::shared_ptr<OckHmmSingleDeviceMgr> &mgr, uint64_t size, OckHmmMemoryAllocatePolicy policy)
    {
        return mgr->Malloc(size, policy);
    }
};
template <>
OckHmmHMObjectRetcodePair AllocFunBridge<OckHmmHMObjectRetcodePair>::operator()(
    std::shared_ptr<OckHmmSingleDeviceMgr> &mgr, uint64_t size, OckHmmMemoryAllocatePolicy policy)
{
    return mgr->Alloc(size, policy);
}
template <typename ReturnT>
ReturnT ByMove(ReturnT &data)
{
    return std::move(data);
}
template <>
OckHmmHMObjectRetcodePair ByMove(OckHmmHMObjectRetcodePair &data)
{
    return data;
}
template <typename ReturnT>
ReturnT ScanAllocImpl(uint64_t hmoBytes, OckHmmMemoryAllocatePolicy policy, OckHmmMgrMapContainerT::iterator startIter,
    OckHmmMgrMapContainerT::iterator endIter, OckHmmMgrMapContainerT::iterator &retIter)
{
    retIter = endIter;
    AllocFunBridge<ReturnT> allocFun;
    for (auto iter = startIter; iter != endIter; ++iter) {
        auto tmpRet = allocFun(iter->second, hmoBytes, policy);
        if (!IsEmptyResult(tmpRet)) {
            retIter = iter;
            return ByMove(tmpRet);
        }
    }
    return MakeEmptyResult<ReturnT>();
}
template <typename ReturnT>
ReturnT AllocImpl(uint64_t hmoBytes, OckHmmMemoryAllocatePolicy policy, OckHmmMgrMapContainerT::iterator &lastIter,
    OckHmmMgrMapContainerT &container)
{
    if (lastIter != container.end()) {
        lastIter++;
    } else {
        lastIter = container.begin();
    }
    OckHmmMgrMapContainerT::iterator retIter = lastIter;
    if (lastIter != container.end()) {
        auto backRet = ScanAllocImpl<ReturnT>(hmoBytes, policy, lastIter, container.end(), retIter);
        if (!IsEmptyResult(backRet)) {
            lastIter = retIter;
            return ByMove(backRet);
        }
    }
    auto frontRet = ScanAllocImpl<ReturnT>(hmoBytes, policy, container.begin(), lastIter, retIter);
    lastIter = retIter;
    return ByMove(frontRet);
}
}  // namespace
std::unique_ptr<OckHmmMemoryGuard> OckHmmComposeDeviceMgrAllocAlgo::Malloc(uint64_t hmoBytes,
    OckHmmMemoryAllocatePolicy policy, OckHmmMgrMapContainerT::iterator &lastIter, OckHmmMgrMapContainerT &mgrMap)
{
    if (policy == OckHmmMemoryAllocatePolicy::DEVICE_DDR_ONLY) {
        return AllocImpl<std::unique_ptr<OckHmmMemoryGuard>>(
            hmoBytes, OckHmmMemoryAllocatePolicy::DEVICE_DDR_ONLY, lastIter, mgrMap);
    } else if (policy == OckHmmMemoryAllocatePolicy::DEVICE_DDR_FIRST) {
        auto ret = AllocImpl<std::unique_ptr<OckHmmMemoryGuard>>(
            hmoBytes, OckHmmMemoryAllocatePolicy::DEVICE_DDR_ONLY, lastIter, mgrMap);
        if (ret.get() != nullptr) {
            return ret;
        }
    }
    return AllocImpl<std::unique_ptr<OckHmmMemoryGuard>>(
        hmoBytes, OckHmmMemoryAllocatePolicy::LOCAL_HOST_ONLY, lastIter, mgrMap);
}
std::pair<OckHmmErrorCode, std::shared_ptr<OckHmmHMObject>> OckHmmComposeDeviceMgrAllocAlgo::Alloc(uint64_t hmoBytes,
    OckHmmMemoryAllocatePolicy policy, OckHmmMgrMapContainerT::iterator &lastIter, OckHmmMgrMapContainerT &mgrMap)
{
    if (policy == OckHmmMemoryAllocatePolicy::DEVICE_DDR_ONLY) {
        return AllocImpl<OckHmmHMObjectRetcodePair>(
            hmoBytes, OckHmmMemoryAllocatePolicy::DEVICE_DDR_ONLY, lastIter, mgrMap);
    }
    if (policy == OckHmmMemoryAllocatePolicy::DEVICE_DDR_FIRST) {
        auto ret = AllocImpl<OckHmmHMObjectRetcodePair>(
            hmoBytes, OckHmmMemoryAllocatePolicy::DEVICE_DDR_ONLY, lastIter, mgrMap);
        if (ret.first == HMM_SUCCESS) {
            return ret;
        }
    }
    return AllocImpl<OckHmmHMObjectRetcodePair>(
        hmoBytes, OckHmmMemoryAllocatePolicy::LOCAL_HOST_ONLY, lastIter, mgrMap);
}
}  // namespace hmm
}  // namespace ock