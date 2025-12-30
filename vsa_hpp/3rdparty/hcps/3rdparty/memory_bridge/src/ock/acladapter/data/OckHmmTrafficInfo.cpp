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

#include <mutex>
#include <list>
#include "ock/acladapter/data/OckHmmTrafficInfo.h"
namespace ock {
namespace acladapter {

OckHmmTrafficInfo::OckHmmTrafficInfo(hmm::OckHmmDeviceId deviceIdx, uint64_t movedByteCount,
    OckMemoryCopyKind memCopyKind, const std::chrono::steady_clock::time_point &startPoint)
    : deviceId(deviceIdx), movedBytes(movedByteCount), copyKind(memCopyKind), startTime(startPoint),
      endTime(std::chrono::steady_clock::now())
{}
bool operator==(const OckHmmTrafficInfo &lhs, const OckHmmTrafficInfo &rhs)
{
    return lhs.deviceId == rhs.deviceId && lhs.copyKind == rhs.copyKind && lhs.movedBytes == rhs.movedBytes;
}
bool operator!=(const OckHmmTrafficInfo &lhs, const OckHmmTrafficInfo &rhs)
{
    return !(lhs == rhs);
}
std::ostream &operator<<(std::ostream &os, const OckHmmTrafficInfo &data)
{
    return os << "{'deviceId':" << data.deviceId << ",'movedBytes':" << data.movedBytes
              << ",'copyKind':" << data.copyKind << ",'startTime':" << data.startTime.time_since_epoch().count()
              << ",'endTime':" << data.endTime.time_since_epoch().count() << "}";
}

}  // namespace acladapter
}  // namespace ock