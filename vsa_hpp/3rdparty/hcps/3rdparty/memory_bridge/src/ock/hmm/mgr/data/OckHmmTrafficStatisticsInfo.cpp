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

#include "ock/hmm/mgr/OckHmmTrafficStatisticsInfo.h"
#include "ock/utils/OckSafeUtils.h"
namespace ock {
namespace hmm {

OckHmmTrafficStatisticsInfo::OckHmmTrafficStatisticsInfo(
    uint64_t h2dBytes, uint64_t d2hBytes, double h2dSpeed, double d2hSpeed)
    : host2DeviceMovedBytes(h2dBytes), device2hostMovedBytes(d2hBytes), host2DeviceSpeed(h2dSpeed),
      device2hostSpeed(d2hSpeed) {}


bool operator==(const OckHmmTrafficStatisticsInfo &lhs, const OckHmmTrafficStatisticsInfo &rhs)
{
    return lhs.host2DeviceMovedBytes == rhs.host2DeviceMovedBytes &&
            lhs.device2hostMovedBytes == rhs.device2hostMovedBytes &&
            utils::SafeFloatEqual(lhs.host2DeviceSpeed, rhs.host2DeviceSpeed) &&
            utils::SafeFloatEqual(lhs.device2hostSpeed, rhs.device2hostSpeed);
}
bool operator!=(const OckHmmTrafficStatisticsInfo &lhs, const OckHmmTrafficStatisticsInfo &rhs)
{
    return !(lhs == rhs);
}

std::ostream &operator<<(std::ostream &os, const OckHmmTrafficStatisticsInfo &data)
{
    return os << "{'host2DeviceMovedBytes':" << data.host2DeviceMovedBytes
              << ",'device2hostMovedBytes':" << data.device2hostMovedBytes
              << ",'host2DeviceSpeed':" << data.host2DeviceSpeed << ",'device2hostSpeed':" << data.device2hostSpeed
              << "}";
}

}  // namespace hmm
}  // namespace ock