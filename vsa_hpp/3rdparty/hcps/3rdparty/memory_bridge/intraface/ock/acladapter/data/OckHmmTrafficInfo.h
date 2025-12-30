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


#ifndef OCK_MEMORY_BRIDGE_OCK_ADAPTER_OCK_HMM_TRAFFIC_INFO_H
#define OCK_MEMORY_BRIDGE_OCK_ADAPTER_OCK_HMM_TRAFFIC_INFO_H
#include <memory>
#include <ostream>
#include <chrono>
#include "ock/hmm/mgr/OckHmmHMObject.h"
#include "ock/acladapter/data/OckMemoryCopyKind.h"
namespace ock {
namespace acladapter {

struct OckHmmTrafficInfo {
    OckHmmTrafficInfo(hmm::OckHmmDeviceId deviceIdx, uint64_t movedByteCount, OckMemoryCopyKind memCopyKind,
        const std::chrono::steady_clock::time_point &startPoint);

    hmm::OckHmmDeviceId deviceId;                     // 设备ID
    uint64_t movedBytes;                              // 移动字节数
    OckMemoryCopyKind copyKind;                       // 移动种类
    std::chrono::steady_clock::time_point startTime;  // 移动的开始时间
    std::chrono::steady_clock::time_point endTime;    // 移动的结束时间
};

bool operator==(
    const OckHmmTrafficInfo &lhs, const OckHmmTrafficInfo &rhs);  // 只比较设备ID、数据量和移动种类，不比较时间
bool operator!=(
    const OckHmmTrafficInfo &lhs, const OckHmmTrafficInfo &rhs);  // 只比较设备ID、数据量和移动种类，不比较时间
std::ostream &operator<<(std::ostream &os, const OckHmmTrafficInfo &data);

}  // namespace acladapter
}  // namespace ock
#endif