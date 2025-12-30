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


#ifndef MEMORY_BRIDGE_OCK_HMM_DEVICE_INFO_CONF_H
#define MEMORY_BRIDGE_OCK_HMM_DEVICE_INFO_CONF_H

#include <cstdint>
#include <ostream>
#include "ock/hmm/mgr/OckHmmHeteroMemoryMgr.h"
#include "ock/conf/ParamRange.h"

namespace ock {
namespace conf {

struct OckHmmDeviceInfoConf {
    OckHmmDeviceInfoConf(void);

    ParamRange<hmm::OckHmmDeviceId> deviceId;
    ParamRange<uint32_t> transferThreadNum;
    ParamRange<uint64_t> deviceBaseCapacity;
    ParamRange<uint64_t> deviceBufferCapacity;
    ParamRange<uint64_t> hostBaseCapacity;
    ParamRange<uint64_t> hostBufferCapacity;
};

bool operator==(const OckHmmDeviceInfoConf &lhs, const OckHmmDeviceInfoConf &rhs);
bool operator!=(const OckHmmDeviceInfoConf &lhs, const OckHmmDeviceInfoConf &rhs);
std::ostream &operator<<(std::ostream &os, const OckHmmDeviceInfoConf &deviceInfoConf);

}  // namespace conf
}  // namespace ock
#endif
