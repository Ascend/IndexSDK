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

#include "ock/hmm/mgr/OckHmmHeteroMemoryLocation.h"

namespace ock {
namespace hmm {
std::ostream &operator<<(std::ostream &os, OckHmmHeteroMemoryLocation location)
{
    switch (location) {
        case OckHmmHeteroMemoryLocation::LOCAL_HOST_MEMORY:
            os << "LOCAL_HOST_MEMORY";
            break;
        case OckHmmHeteroMemoryLocation::DEVICE_HBM:
            os << "DEVICE_HBM";
            break;
        case OckHmmHeteroMemoryLocation::DEVICE_DDR:
            os << "DEVICE_DDR";
            break;
        default:
            os << "UnknownLocation(value=" << static_cast<uint32_t>(location) << ")";
            break;
    }
    return os;
}
std::ostream &operator<<(std::ostream &os, OckHmmMemoryAllocatePolicy policy)
{
    switch (policy) {
        case OckHmmMemoryAllocatePolicy::DEVICE_DDR_FIRST:
            os << "DEVICE_DDR_FIRST";
            break;
        case OckHmmMemoryAllocatePolicy::DEVICE_DDR_ONLY:
            os << "DEVICE_DDR_ONLY";
            break;
        case OckHmmMemoryAllocatePolicy::LOCAL_HOST_ONLY:
            os << "LOCAL_HOST_ONLY";
            break;
        default:
            os << "UnknownPolicy(value=" << static_cast<uint32_t>(policy) << ")";
            break;
    }
    return os;
}
}  // namespace hmm
}  // namespace ock