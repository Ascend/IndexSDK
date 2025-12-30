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

#ifndef OCK_MEMORY_BRIDGE_OCK_HMM_CONF_H
#define OCK_MEMORY_BRIDGE_OCK_HMM_CONF_H
#include <cstdint>
#include <ostream>

namespace ock {
namespace conf {

const uint32_t MaxHMOIDNumberPerDevice = 1024U * 1024U;  // 每个设备的最大HMO的ID个数，1个HMO占用1bit空间
struct OckHmmConf {
    OckHmmConf(void);
    uint32_t maxHMOCountPerDevice;    // 每个设备的最大HMO的个数, MaxHMOCountPerDevice远小于MaxHMOIDNumberPerDevice(单位: 个)
    uint64_t defaultFragThreshold;    // 缺省的内存碎片大小阈值(单位: Bytes)
    uint64_t maxWaitTimeMilliSecond;  // 任务最大的等待超时时间(用户即使设置为timeout=0， 仍然会使用最多的等待超时)(单位: 毫秒)
    uint64_t subMemoryBaseSize;         // 二次分配的内存块的最小尺寸（单位：bytes)
    uint64_t subMemoryLargerBaseSize;   // 二次分配的内存块的稍大基础尺寸（单位：bytes)
    uint64_t subMemoryLargestBaseSize;  // 二次分配的内存块的最大基础尺寸（单位：bytes)
    uint64_t maxAllocHmoBytes;        // OckHmmHeteroMemoryMgrBase::Alloc最大字节数
    uint64_t minAllocHmoBytes;        // OckHmmHeteroMemoryMgrBase::Alloc最小字节数
    uint64_t maxMallocBytes;          // Malloc最大字节数
    uint64_t minMallocBytes;          // Malloc最小字节数
    uint64_t maxFragThreshold;        // fragThreshold最大取值
    uint64_t minFragThreshold;        // fragThreshold最小取值
    uint32_t maxMaxGapMilliSeconds;   // maxGapMilliSeconds最大取值
    uint32_t minMaxGapMilliSeconds;   // maxGapMilliSeconds最小取值
};
bool operator==(const OckHmmConf &lhs, const OckHmmConf &rhs);
bool operator!=(const OckHmmConf &lhs, const OckHmmConf &rhs);
std::ostream &operator<<(std::ostream &os, const OckHmmConf &hmmConf);

}  // namespace conf
}  // namespace ock
#endif