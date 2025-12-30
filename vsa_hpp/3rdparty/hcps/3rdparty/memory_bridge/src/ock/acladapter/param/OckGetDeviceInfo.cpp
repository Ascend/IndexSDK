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


#include "ock/acladapter/param/OckGetDeviceInfo.h"
namespace ock {
namespace acladapter {

void OckGetDeviceInfo::Init(const hmm::OckHmmDeviceId deviceId)
{
    auto ret = aclrtSetDevice(deviceId);
    if (ret != ACL_SUCCESS) {
        OCK_HMM_LOG_ERROR("aclrtSetDevice failed. deviceId=" << deviceId << " ret=" << ret);
    }
}

void OckGetDeviceInfo::FreeResources(const hmm::OckHmmDeviceId deviceId)
{
    auto ret = aclrtResetDevice(deviceId);
    if (ret != ACL_SUCCESS) {
        OCK_HMM_LOG_ERROR("aclrtResetDevice failed. deviceId=" << deviceId << " ret=" << ret);
        return;
    }
}

aclError OckGetDeviceInfo::GetDeviceCount(uint32_t *count)
{
    return aclrtGetDeviceCount(count);
}

aclError OckGetDeviceInfo::GetMemInfo(aclrtMemAttr attr, size_t *free, size_t *total)
{
    return aclrtGetMemInfo(attr, free, total);
}
}
}