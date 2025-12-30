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


#ifndef OCK_MEMORY_BRIDGE_TRAFFIC_STATISTICS_INFO_H
#define OCK_MEMORY_BRIDGE_TRAFFIC_STATISTICS_INFO_H
#include <ostream>

namespace ock {
namespace hmm {
struct OckHmmTrafficStatisticsInfo {
    OckHmmTrafficStatisticsInfo(uint64_t h2dBytes, uint64_t d2hBytes, double h2dSpeed, double d2hSpeed);
    OckHmmTrafficStatisticsInfo() = default;
    uint64_t host2DeviceMovedBytes {0ULL};
    uint64_t device2hostMovedBytes {0ULL};
    double host2DeviceSpeed {0};  // 单位MB/s
    double device2hostSpeed {0};  // 单位MB/s
};

bool operator==(const OckHmmTrafficStatisticsInfo &lhs, const OckHmmTrafficStatisticsInfo &rhs);
bool operator!=(const OckHmmTrafficStatisticsInfo &lhs, const OckHmmTrafficStatisticsInfo &rhs);
std::ostream &operator<<(std::ostream &os, const OckHmmTrafficStatisticsInfo &data);
}  // namespace hmm
}  // namespace ock
#endif