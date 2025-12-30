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


#ifndef OCK_MEMORY_BRIDGE_OCK_HMM_OCK_HMO_OBJECTID_GENERATOR_H
#define OCK_MEMORY_BRIDGE_OCK_HMM_OCK_HMO_OBJECTID_GENERATOR_H
#include <cstdint>
#include "ock/hmm/mgr/OckHmmHMObject.h"
namespace ock {
namespace hmm {

class OckHmmHMOObjectIDGenerator {
public:
    static hmm::OckHmmHMOObjectID Gen(hmm::OckHmmDeviceId deviceId, uintptr_t addr, uint32_t csnId);
    static bool Valid(hmm::OckHmmHMOObjectID hmoObjectId, hmm::OckHmmDeviceId deviceId, uintptr_t addr);
    static hmm::OckHmmDeviceId ParseDeviceId(hmm::OckHmmHMOObjectID hmoObjectId);
    static uint32_t ParseCsnId(hmm::OckHmmHMOObjectID hmoObjectId);
};

}  // namespace hmm
}  // namespace ock
#endif