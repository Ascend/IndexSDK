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

#include "ock/hmm/mgr/data/OckHmmHMOObjectIDGenerator.h"
namespace ock {
namespace hmm {
namespace {
const uint32_t LOCAL_DEVICE_ID_OFFSET = 48;
const uint32_t LOCAL_ADDR_OFFSET = 32;
const uint64_t LOCAL_DEVICE_ID_MASK = 0xFFFF000000000000UL;
const uint64_t LOCAL_ADDR_MASK = 0x0000FFFF00000000UL;
const uint64_t LOCAL_CSN_ID_MASK = 0x00000000FFFFFFFFUL;
}  // namespace

hmm::OckHmmHMOObjectID OckHmmHMOObjectIDGenerator::Gen(hmm::OckHmmDeviceId deviceId, uintptr_t addr, uint32_t csnId)
{
    uint64_t ret = (uint64_t(deviceId) << LOCAL_DEVICE_ID_OFFSET) & LOCAL_DEVICE_ID_MASK;
    ret += (uint64_t(addr) << LOCAL_ADDR_OFFSET) & LOCAL_ADDR_MASK;
    ret += csnId;
    return ret;
}
bool OckHmmHMOObjectIDGenerator::Valid(hmm::OckHmmHMOObjectID hmoObjectId, hmm::OckHmmDeviceId deviceId, uintptr_t addr)
{
    if (((uint64_t(deviceId) << LOCAL_DEVICE_ID_OFFSET) & LOCAL_DEVICE_ID_MASK) !=
        (hmoObjectId & LOCAL_DEVICE_ID_MASK)) {
        return false;
    }
    if (((uint64_t(addr) << LOCAL_ADDR_OFFSET) & LOCAL_ADDR_MASK) != (hmoObjectId & LOCAL_ADDR_MASK)) {
        return false;
    }
    return true;
}
hmm::OckHmmDeviceId OckHmmHMOObjectIDGenerator::ParseDeviceId(hmm::OckHmmHMOObjectID hmoObjectId)
{
    return hmm::OckHmmDeviceId((hmoObjectId & LOCAL_DEVICE_ID_MASK) >> LOCAL_DEVICE_ID_OFFSET);
}
uint32_t OckHmmHMOObjectIDGenerator::ParseCsnId(hmm::OckHmmHMOObjectID hmoObjectId)
{
    return uint32_t(hmoObjectId & LOCAL_CSN_ID_MASK);
}
}  // namespace hmm
}  // namespace ock