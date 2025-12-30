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


#ifndef OCK_MEMORY_BRIDGE_HMM_HETERO_MGR_COMPOSE_DEVICE_PARAM_CHECKER_H
#define OCK_MEMORY_BRIDGE_HMM_HETERO_MGR_COMPOSE_DEVICE_PARAM_CHECKER_H
#include "ock/hmm/mgr/OckHmmHeteroMemoryMgr.h"
#include "ock/hmm/mgr/checker/OckHmmHeteroMemoryMgrParamCheck.h"
namespace ock {
namespace hmm {

class OckHmmComposeDeviceMgrParamCheck {
public:
    static OckHmmErrorCode CheckGetUsedInfo(const uint64_t fragThreshold);
    static OckHmmErrorCode CheckMalloc(const uint64_t size);
    static OckHmmErrorCode CheckAlloc(const uint64_t hmoBytes);
    static OckHmmErrorCode CheckCopy(const OckHmmHMObject &dstHMO, const uint64_t dstOffset,
        const OckHmmHMObject &srcHMO, const uint64_t srcOffset, const size_t length);
};

}  // namespace hmm
}  // namespace ock
#endif