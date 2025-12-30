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

#include "ock/hmm/mgr/checker/OckHmmComposeDeviceMgrParamCheck.h"
#include "ock/hmm/mgr/data/OckHmmHMOObjectIDGenerator.h"
#include "ock/log/OckLogger.h"
namespace ock {
namespace hmm {

OckHmmErrorCode OckHmmComposeDeviceMgrParamCheck::CheckGetUsedInfo(const uint64_t fragThreshold)
{
    // 校验fragThreshold
    return OckHmmHeteroMemoryMgrParamCheck::CheckGetUsedInfo(fragThreshold);
}

OckHmmErrorCode OckHmmComposeDeviceMgrParamCheck::CheckMalloc(const uint64_t size)
{
    return OckHmmHeteroMemoryMgrParamCheck::CheckMalloc(size);
}

OckHmmErrorCode OckHmmComposeDeviceMgrParamCheck::CheckAlloc(const uint64_t hmoBytes)
{
    return OckHmmHeteroMemoryMgrParamCheck::CheckAlloc(hmoBytes);
}

OckHmmErrorCode OckHmmComposeDeviceMgrParamCheck::CheckCopy(const OckHmmHMObject &dstHMO, const uint64_t dstOffset,
    const OckHmmHMObject &srcHMO, const uint64_t srcOffset, const size_t length)
{
    return OckHmmHeteroMemoryMgrParamCheck::CheckCopy(dstHMO, dstOffset, srcHMO, srcOffset, length);
}

}  // namespace hmm
}  // namespace ock