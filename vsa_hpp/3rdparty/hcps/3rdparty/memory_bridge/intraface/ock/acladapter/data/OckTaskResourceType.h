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


#ifndef OCK_ADAPTER_OCK_TASK_RESOURCE_TYPE_H
#define OCK_ADAPTER_OCK_TASK_RESOURCE_TYPE_H
#include <cstdint>
#include <ostream>
#include <limits>
#include <unordered_map>
namespace ock {
namespace acladapter {

enum class OckTaskResourceType : uint32_t {
    MEMORY_TRANSFER = 1U,   // 内存传输任务, 主要使用PCIe资源和Host/Device的内存读写IO
    DEVICE_MEMORY_OP = 2U,  // Device上的内存申请与释放操作任务,  主要占用队列资源和内存IO资源
    HOST_MEMORY_OP = 4U,    // Host上的内存申请与释放操作任务,  主要占用队列资源和内存IO资源
    MEMORY_OP = DEVICE_MEMORY_OP | HOST_MEMORY_OP,  // Host或Device上的内存申请与释放操作
    HMM = MEMORY_OP | MEMORY_TRANSFER,              // Host与Device上的内存管理操作
    DEVICE_AI_CUBE = 8U,                            // Device上的计算任务, 主要占用AICube计算资源
    DEVICE_AI_VECTOR = 16U,                         // Device上的计算任务, 主要占用AI VECTOR计算资源
    DEVICE_AI_CPU = 32U,                            // Device上的计算任务, 主要占用AI CPU计算资源
    HOST_CPU = 64U,                                 // Host上的计算任务, 主要占用HOST CPU计算资源
    HOST_STREAM = 128U,  // HOST上的任务分发与创建， 主要占用线程资源， CPU计算动作比较少
    CPU = HOST_CPU | DEVICE_AI_CPU,                 // Device或Host上的计算任务, 主要占用CPU计算资源
    OP_TASK = HMM | DEVICE_AI_CUBE | DEVICE_AI_VECTOR | DEVICE_AI_CPU | HOST_CPU | CPU,  // 同步算子任务
    DEVICE_STREAM = 256U,                    // Device上的异步流/异步流任务的创建, 主要占用队列资源
    OP_SYNC_TASK = OP_TASK | DEVICE_STREAM,  // 异步算子任务
    ALL = std::numeric_limits<uint32_t>::max(),  // 支持以上所有
};
inline OckTaskResourceType operator|(OckTaskResourceType lhs, OckTaskResourceType rhs)
{
    return static_cast<OckTaskResourceType>(static_cast<uint32_t>(lhs) | static_cast<uint32_t>(rhs));
}
inline OckTaskResourceType operator&(OckTaskResourceType lhs, OckTaskResourceType rhs)
{
    return static_cast<OckTaskResourceType>(static_cast<uint32_t>(lhs) & static_cast<uint32_t>(rhs));
}
inline bool RhsInLhs(OckTaskResourceType lhs, OckTaskResourceType rhs)
{
    return (static_cast<uint32_t>(lhs) & static_cast<uint32_t>(rhs)) == static_cast<uint32_t>(rhs);
}
using OckTaskThreadNumberMap = std::unordered_map<OckTaskResourceType, uint32_t>;
uint32_t CalcRelatedThreadNumber(const OckTaskThreadNumberMap &taskNumberMap, OckTaskResourceType resourceType);
std::ostream &operator<<(std::ostream &os, OckTaskResourceType resourceType);
std::ostream &operator<<(std::ostream &os, const OckTaskThreadNumberMap &data);
}  // namespace acladapter
}  // namespace ock
#endif