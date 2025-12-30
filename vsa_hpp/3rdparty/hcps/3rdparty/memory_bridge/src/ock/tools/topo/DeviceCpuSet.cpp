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

#include "ock/utils/StrUtils.h"
#include "ock/tools/topo/DeviceCpuSet.h"

namespace ock {
namespace tools {
namespace topo {
CpuIdRange::CpuIdRange(uint32_t startIdx, uint32_t endIdx) : startId(startIdx), endId(endIdx)
{}

DeviceCpuSet::DeviceCpuSet(hmm::OckHmmDeviceId deviceIdx) : deviceId(deviceIdx)
{}
bool DeviceCpuSet::operator==(const DeviceCpuSet &other) const
{
    return deviceId == other.deviceId;
}
cpu_set_t DeviceCpuSet::GetCpuSet(void) const
{
    cpu_set_t cpuSet;
    CPU_ZERO(&cpuSet);
    for (auto &data : cpuIds) {
        for (uint32_t i = data.startId; i <= data.endId; ++i) {
            CPU_SET(i, &cpuSet);
        }
    }
    return cpuSet;
}
size_t DeviceCpuSetHash::operator()(const DeviceCpuSet &data) const
{
    return std::hash<hmm::OckHmmDeviceId>()(data.deviceId);
}
std::pair<bool, std::shared_ptr<DeviceCpuSet>> DeviceCpuSet::ParseFrom(const std::string &buff)
{
    // 请在这里支持多种设备信息描述的格式
    // 1:1,2,3
    // 2:1-3,4,5
    // 3:4-5,1,2
    const size_t devSplitPartNumber = {2};
    std::vector<std::string> devTokens;
    utils::Split(buff, ":", devTokens);
    if (devTokens.size() != devSplitPartNumber) {
        return std::make_pair(false, std::shared_ptr<DeviceCpuSet>());
    }
    hmm::OckHmmDeviceId deviceId = 0;
    if (!utils::FromString(deviceId, devTokens.front())) {
        return std::make_pair(false, std::shared_ptr<DeviceCpuSet>());
    }

    auto deviceInfo = std::make_shared<DeviceCpuSet>(deviceId);

    std::vector<std::string> cpuIdToken;
    utils::Split(devTokens.back(), ",", cpuIdToken);
    for (auto &token : cpuIdToken) {
        CpuIdRange range(0, 0);
        if (!FromString(range, token)) {
            return std::make_pair(false, std::shared_ptr<DeviceCpuSet>());
        }
        deviceInfo->cpuIds.push_back(range);
    }
    return std::make_pair(true, deviceInfo);
}
std::ostream &operator<<(std::ostream &os, const CpuIdRange &range)
{
    if (range.startId == range.endId) {
        return os << range.startId;
    }
    return os << range.startId << "-" << range.endId;
}
std::ostream &operator<<(std::ostream &os, const DeviceCpuSet &deviceInfo)
{
    os << deviceInfo.deviceId << ":" << deviceInfo.cpuIds;
    return os;
}
std::ostream &operator<<(std::ostream &os, const std::vector<CpuIdRange> &rangeVec)
{
    for (auto iter = rangeVec.begin(); iter != rangeVec.end(); ++iter) {
        if (iter != rangeVec.begin()) {
            os << ",";
        }
        os << *iter;
    }
    return os;
}
bool FromString(CpuIdRange &result, const std::string &data) noexcept
{
    // 这里需要支持多种格式
    // 1,2,3
    // 1-3,4,5
    // 4-5,1,2,31-37
    std::vector<size_t> splitPartNumber = {1, 2};
    uint32_t startId = 0;
    std::vector<std::string> cpuIdRange;
    utils::Split(data, "-", cpuIdRange);
    auto ret = utils::FromString(startId, cpuIdRange.front());
    if (!ret) {
        return false;
    }
    result.startId = startId;
    if (cpuIdRange.size() == splitPartNumber.front()) {
        if (data.find("-") != std::string::npos) {
            std::cerr << "cpuId can not less than 0" << std::endl;
            return false;
        }
        result.endId = startId;
    } else if (cpuIdRange.size() == splitPartNumber.back()) {
        uint32_t endId = 0;
        ret = utils::FromString(endId, cpuIdRange.back());
        if (!ret) {
            return false;
        }
        result.endId = endId;
    } else {
        return false;
    }
    return true;
}
}  // namespace topo
}  // namespace tools
}  // namespace ock