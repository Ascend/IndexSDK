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

#ifndef OCK_MEMORY_BRIDGE_OCK_HMM_FACTORY_H
#define OCK_MEMORY_BRIDGE_OCK_HMM_FACTORY_H

#include "ock/hmm/mgr/OckHmmHeteroMemoryMgr.h"

namespace ock {
namespace hmm {
namespace {
const uint32_t MAX_CREATOR_TIMEOUT_MILLISECONDS = 5000U;
}
class OckHmmFactory {
public:
    virtual ~OckHmmFactory() noexcept = default;
    /*
    @brief 创建异构内存管理对象, 对象创建时会完成内存预分配等动作， 如果预分配失败会返回失败。
    @result 当std::pair<...>.first不为HMM_SUCCESS, std::pair<...>.second不可信
    */
    virtual std::pair<OckHmmErrorCode, std::shared_ptr<OckHmmSingleDeviceMgr>> CreateSingleDeviceMemoryMgr(
        std::shared_ptr<OckHmmDeviceInfo> deviceInfo, uint32_t timeout = MAX_CREATOR_TIMEOUT_MILLISECONDS) = 0;

    /*
    @brief 创建多设备组合的异构内存管理对象, 对象创建时会完成内存预分配等动作， 如果预分配失败会返回失败。
    @result 当std::pair<...>.first不为HMM_SUCCESS, std::pair<...>.second不可信
    */
    virtual std::pair<OckHmmErrorCode, std::shared_ptr<OckHmmComposeDeviceMgr>> CreateComposeMemoryMgr(
        std::shared_ptr<OckHmmDeviceInfoVec> deviceInfoVec, uint32_t timeout = MAX_CREATOR_TIMEOUT_MILLISECONDS) = 0;

    /*
    @brief 创建共享内存模式的异构内存管理对象, 对象创建时会完成内存预分配等动作， 如果预分配失败会返回失败。
    @result 当std::pair<...>.first不为HMM_SUCCESS, std::pair<...>.second不可信
    */
    virtual std::pair<OckHmmErrorCode, std::shared_ptr<OckHmmShareDeviceMgr>> CreateShareMemoryMgr(
        std::shared_ptr<OckHmmPureDeviceInfoVec> deviceInfoVec,
        std::shared_ptr<OckHmmMemoryCapacitySpecification> hostSpec) = 0;

    /*
    @brief 创建一个OckHmmFactory工厂, 当前OckHmmFactory本身不会占用资源，
    只有使用本对象创建MemoryMgr的时候才会占用资源。
    */
    static std::shared_ptr<OckHmmFactory> Create(void);
};
}  // namespace hmm
}  // namespace ock
#endif