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


#include "DistComputeOpsManager.h"
#include "ThreadSafeDistComputeOpsManager.h"
#include "AscendUtils.h"

namespace ascend {
std::shared_ptr<DistComputeOpsManager> &DistComputeOpsManager::getShared()
{
    static std::mutex getMtx;
    static std::map<int, std::shared_ptr<DistComputeOpsManager>> mngs;
    int deviceId = 0;
    auto ret = aclrtGetDevice(&deviceId);
    if (ret != ACL_SUCCESS) {
        APP_LOG_ERROR("failed to get device, ret: %d", ret);
        deviceId = 0;
    }
    auto lock = AscendMultiThreadManager::GetWriteLock(getMtx);
    if (mngs.find(deviceId) == mngs.end()) {
        if (AscendMultiThreadManager::IsMultiThreadMode()) {
            mngs[deviceId] = std::make_shared<ThreadSafeDistComputeOpsManager>();
        } else {
            mngs[deviceId] = std::make_shared<DistComputeOpsManager>();
        }
    }
    return mngs[deviceId];
}
}