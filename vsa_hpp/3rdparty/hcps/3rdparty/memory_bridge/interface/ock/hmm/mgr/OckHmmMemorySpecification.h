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


#ifndef OCK_MEMORY_BRIDGE_HMO_SPECIFICATION_H
#define OCK_MEMORY_BRIDGE_HMO_SPECIFICATION_H
#include <ostream>

namespace ock {
namespace hmm {
struct OckHmmMemoryCapacitySpecification {
    uint64_t maxDataCapacity {0ULL};  // 数据区容量
    uint64_t maxSwapCapacity {0ULL};  // 交互区容量
} __attribute__((packed));

struct OckHmmMemorySpecification {
    OckHmmMemoryCapacitySpecification devSpec{};   // 设备规格
    OckHmmMemoryCapacitySpecification hostSpec{};  // host规格
} __attribute__((packed));

struct OckHmmMemoryUsedInfo {
    OckHmmMemoryUsedInfo(void);
    OckHmmMemoryUsedInfo &operator+=(const OckHmmMemoryUsedInfo &other);

    uint64_t usedBytes;        // 已经使用的内存大小
    uint64_t unusedFragBytes;  // 未使用的碎片内存大小
    uint64_t leftBytes;        // 剩余内存大小
    uint64_t swapUsedBytes;    // swap使用空间(buffer)内存大小
    uint64_t swapLeftBytes;    // swap剩余空间(buffer)内存大小
} __attribute__((packed));

struct OckHmmResourceUsedInfo {
    OckHmmMemoryUsedInfo devUsedInfo{};   // 设备内存资源使用信息
    OckHmmMemoryUsedInfo hostUsedInfo{};  // host内存资源使用信息
} __attribute__((packed));

bool operator==(const OckHmmMemoryCapacitySpecification &lhs, const OckHmmMemoryCapacitySpecification &rhs);
bool operator==(const OckHmmMemorySpecification &lhs, const OckHmmMemorySpecification &rhs);
bool operator==(const OckHmmMemoryUsedInfo &lhs, const OckHmmMemoryUsedInfo &rhs);
bool operator==(const OckHmmResourceUsedInfo &lhs, const OckHmmResourceUsedInfo &rhs);
bool operator!=(const OckHmmMemoryCapacitySpecification &lhs, const OckHmmMemoryCapacitySpecification &rhs);
bool operator!=(const OckHmmMemorySpecification &lhs, const OckHmmMemorySpecification &rhs);
bool operator!=(const OckHmmMemoryUsedInfo &lhs, const OckHmmMemoryUsedInfo &rhs);
bool operator!=(const OckHmmResourceUsedInfo &lhs, const OckHmmResourceUsedInfo &rhs);

std::ostream &operator<<(std::ostream &os, const OckHmmMemoryCapacitySpecification &memorySpec);
std::ostream &operator<<(std::ostream &os, const OckHmmMemoryUsedInfo &usedInfo);
std::ostream &operator<<(std::ostream &os, const OckHmmMemorySpecification &memorySpec);
std::ostream &operator<<(std::ostream &os, const OckHmmResourceUsedInfo &usedInfo);
}  // namespace hmm
}  // namespace ock
#endif