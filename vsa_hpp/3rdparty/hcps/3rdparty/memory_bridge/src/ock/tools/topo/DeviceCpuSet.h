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

#ifndef OCK_MEMORY_BRIDGE_OCK_TOOL_TOPO_DECTECT_DEVICE_CPU_SET_H
#define OCK_MEMORY_BRIDGE_OCK_TOOL_TOPO_DECTECT_DEVICE_CPU_SET_H
#include <iostream>
#include <ostream>
#include <cstdint>
#include <unordered_set>
#include <vector>
#include <memory>
#include "ock/hmm/mgr/OckHmmHMObject.h"

namespace ock {
namespace tools {
namespace topo {

struct CpuIdRange {
    CpuIdRange(uint32_t startIdx, uint32_t endIdx);
    uint32_t startId;
    uint32_t endId;
};
struct DeviceCpuSet {
    DeviceCpuSet(void) = default;
    DeviceCpuSet(hmm::OckHmmDeviceId deviceIdx);

    cpu_set_t GetCpuSet(void) const;
    bool operator==(const DeviceCpuSet &other) const;

    static std::pair<bool, std::shared_ptr<DeviceCpuSet>> ParseFrom(const std::string &buff);

    hmm::OckHmmDeviceId deviceId{ 0 };
    std::vector<CpuIdRange> cpuIds{};
};
struct DeviceCpuSetHash {
    size_t operator()(const DeviceCpuSet &data) const;
};
std::ostream &operator<<(std::ostream &os, const CpuIdRange &range);
std::ostream &operator<<(std::ostream &os, const DeviceCpuSet &deviceInfo);
std::ostream &operator<<(std::ostream &os, const std::vector<CpuIdRange> &rangeVec);

bool FromString(CpuIdRange &result, const std::string &data) noexcept;

}  // namespace topo
}  // namespace tools
}  // namespace ock
#endif