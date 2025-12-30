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


#ifndef OCK_MEMORY_BRIDGE_HMO_MEMORY_LOCATION_H
#define OCK_MEMORY_BRIDGE_HMO_MEMORY_LOCATION_H
#include <cstdint>
#include <ostream>
namespace ock {
namespace hmm {
enum class OckHmmHeteroMemoryLocation : uint32_t {
    LOCAL_HOST_MEMORY = 0,
    DEVICE_HBM = 1,
    DEVICE_DDR = 2,
};

enum class OckHmmMemoryAllocatePolicy : uint32_t { DEVICE_DDR_FIRST = 0, DEVICE_DDR_ONLY = 1, LOCAL_HOST_ONLY = 2 };

std::ostream &operator<<(std::ostream &os, OckHmmHeteroMemoryLocation location);
std::ostream &operator<<(std::ostream &os, OckHmmMemoryAllocatePolicy policy);
}  // namespace hmm
}  // namespace ock
#endif