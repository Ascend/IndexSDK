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


#ifndef OCK_MEMORY_BRIDGE_HMM_HETERO_MGR_CREATOR_H
#define OCK_MEMORY_BRIDGE_HMM_HETERO_MGR_CREATOR_H
#include <cstdint>
#include <utility>
#include "ock/hmm/mgr/OckHmmHeteroMemoryMgr.h"
#include "ock/acladapter/executor/OckAsyncTaskExecuteService.h"

namespace ock {
namespace hmm {
enum class OckHmmMemoryName : uint32_t {
    DEVICE_DATA = 0,
    DEVICE_SWAP = 1,
    HOST_DATA = 2,
    HOST_SWAP = 3,
};
class OckHmmMgrCreator {
public:
    static std::pair<OckHmmErrorCode, std::shared_ptr<OckHmmSingleDeviceMgr>> Create(const OckHmmDeviceInfo &deviceInfo,
        uint32_t timeout);
    static std::pair<OckHmmErrorCode, std::shared_ptr<OckHmmSingleDeviceMgr>> Create(const OckHmmDeviceInfo &deviceInfo,
        std::shared_ptr<acladapter::OckAsyncTaskExecuteService> service, uint32_t timeout);
    static std::pair<OckHmmErrorCode, std::shared_ptr<OckHmmComposeDeviceMgr>> Create(
        std::shared_ptr<OckHmmDeviceInfoVec> deviceInfoVec, uint32_t timeout);
    static std::pair<OckHmmErrorCode, std::shared_ptr<OckHmmComposeDeviceMgr>> Create(
        std::shared_ptr<OckHmmDeviceInfoVec> deviceInfoVec,
        std::shared_ptr<acladapter::OckAsyncTaskExecuteService> service, uint32_t timeout);
};
} // namespace hmm
} // namespace ock
#endif