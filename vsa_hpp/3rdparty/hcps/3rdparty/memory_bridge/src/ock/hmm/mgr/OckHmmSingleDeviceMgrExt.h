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


#ifndef OCK_MEMORY_BRIDGE_HMM_SINGLE_DEVICE_MGR_EXT_H
#define OCK_MEMORY_BRIDGE_HMM_SINGLE_DEVICE_MGR_EXT_H
#include <cstdint>
#include "ock/hmm/mgr/OckHmmHeteroMemoryMgr.h"
#include "ock/hmm/mgr/OckHmmSubMemoryAlloc.h"
#include "ock/acladapter/executor/OckAsyncTaskExecuteService.h"
namespace ock {
namespace hmm {
namespace ext {
std::shared_ptr<OckHmmSingleDeviceMgr> CreateSingleDeviceMgr(std::shared_ptr<OckHmmDeviceInfo> deviceInfo,
    std::shared_ptr<acladapter::OckAsyncTaskExecuteService> service, std::shared_ptr<OckHmmSubMemoryAllocSet> allocSet);
}  // namespace ext
}  // namespace hmm
}  // namespace ock
#endif