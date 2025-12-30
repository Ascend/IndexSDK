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

#ifndef OCK_VSA_NPU_ANN_INDEX_CAPACITY_H
#define OCK_VSA_NPU_ANN_INDEX_CAPACITY_H
#include "ock/vsa/neighbor/npu/OckVsaAnnNpuIndex.h"
#include "ock/vsa/neighbor/npu/OckVsaAnnNpuBlockGroup.h"

namespace ock {
namespace vsa {
namespace neighbor {
namespace npu {

class OckVsaAnnIndexCapacity {
public:
    static uint64_t HostSpace(const OckVsaAnnCreateParam &param, uint64_t sizeOfFeature, uint64_t sizeOfNorm);
    static uint64_t DeviceSpace(const OckVsaAnnCreateParam &param, uint64_t sizeOfFeature, uint64_t sizeOfNorm);
};
}  // namespace npu
}  // namespace neighbor
}  // namespace vsa
}  // namespace ock
#endif