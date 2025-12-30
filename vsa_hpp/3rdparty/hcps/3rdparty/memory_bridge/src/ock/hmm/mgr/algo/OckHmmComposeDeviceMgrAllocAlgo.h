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

#ifndef OCK_MEMORY_BRIDGE_HMM_COMPOSE_DEVICE_MGR_ALLOC_ALGO_H
#define OCK_MEMORY_BRIDGE_HMM_COMPOSE_DEVICE_MGR_ALLOC_ALGO_H
#include <unordered_map>
#include <memory>
#include "ock/hmm/mgr/OckHmmHeteroMemoryMgr.h"
namespace ock {
namespace hmm {

using OckHmmMgrMapContainerT = std::unordered_map<OckHmmDeviceId, std::shared_ptr<OckHmmSingleDeviceMgr>>;
class OckHmmComposeDeviceMgrAllocAlgo {
public:
    static std::unique_ptr<OckHmmMemoryGuard> Malloc(uint64_t size, OckHmmMemoryAllocatePolicy policy,
        OckHmmMgrMapContainerT::iterator &lastIter, OckHmmMgrMapContainerT &mgrMap);
    static std::pair<OckHmmErrorCode, std::shared_ptr<OckHmmHMObject>> Alloc(uint64_t hmoBytes,
        OckHmmMemoryAllocatePolicy policy, OckHmmMgrMapContainerT::iterator &lastIter, OckHmmMgrMapContainerT &mgrMap);
};
}  // namespace hmm
}  // namespace ock
#endif