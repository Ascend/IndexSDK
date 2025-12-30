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


#ifndef OCK_MEMORY_BRIDGE_HMM_COMPOSE_DEVICE_MGR_EXT_H
#define OCK_MEMORY_BRIDGE_HMM_COMPOSE_DEVICE_MGR_EXT_H
#include <vector>
#include <memory>
#include "ock/hmm/mgr/OckHmmHeteroMemoryMgr.h"
namespace ock {
namespace hmm {
namespace ext {
std::shared_ptr<OckHmmComposeDeviceMgr> CreateComposeDeviceMgr(
    std::vector<std::shared_ptr<OckHmmSingleDeviceMgr>> &deviceMgrVec);
}  // namespace ext
}  // namespace hmm
}  // namespace ock
#endif