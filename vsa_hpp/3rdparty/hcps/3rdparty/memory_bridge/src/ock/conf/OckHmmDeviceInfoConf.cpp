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


#include "ock/conf/OckHmmDeviceInfoConf.h"

namespace ock {
namespace conf {

OckHmmDeviceInfoConf::OckHmmDeviceInfoConf(void)
    : deviceId(ParamRange<hmm::OckHmmDeviceId>(0U, 15U)),
      transferThreadNum(ParamRange<uint32_t>(1U, 128U)),
      deviceBaseCapacity(ParamRange<uint64_t>(1024ULL * 1024ULL * 1024ULL, 512ULL * 1024ULL * 1024ULL * 1024ULL)),
      deviceBufferCapacity(ParamRange<uint64_t>(64ULL * 1024ULL * 1024ULL, 8ULL * 1024ULL * 1024ULL * 1024ULL)),
      hostBaseCapacity(ParamRange<uint64_t>(1024ULL * 1024ULL * 1024ULL, 512ULL * 1024ULL * 1024ULL * 1024ULL)),
      hostBufferCapacity(ParamRange<uint64_t>(64ULL * 1024ULL * 1024ULL, 8ULL * 1024ULL * 1024ULL * 1024ULL))
{}

bool operator==(const OckHmmDeviceInfoConf &lhs, const OckHmmDeviceInfoConf &rhs)
{
    return lhs.deviceId == rhs.deviceId &&
           lhs.transferThreadNum == rhs.transferThreadNum &&
           lhs.deviceBaseCapacity == rhs.deviceBaseCapacity &&
           lhs.deviceBufferCapacity == rhs.deviceBufferCapacity &&
           lhs.hostBaseCapacity == rhs.hostBaseCapacity &&
           lhs.hostBufferCapacity == rhs.hostBufferCapacity;
}

bool operator!=(const OckHmmDeviceInfoConf &lhs, const OckHmmDeviceInfoConf &rhs)
{
    return !(lhs == rhs);
}

std::ostream &operator<<(std::ostream &os, const OckHmmDeviceInfoConf &deviceInfoConf)
{
    return os << "'DeviceIdRange':" << deviceInfoConf.deviceId
              << ", 'TransferThreadNumRange':" << deviceInfoConf.transferThreadNum
              << ", 'DeviceBaseCapacityRange':" << deviceInfoConf.deviceBaseCapacity
              << ", 'DeviceBufferCapacityRange':" << deviceInfoConf.deviceBufferCapacity
              << ", 'HostBaseCapacityRange':" << deviceInfoConf.hostBaseCapacity
              << ", 'HostBufferCapacityRange':" << deviceInfoConf.hostBufferCapacity;
}
}
}