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

#include "ock/vsa/neighbor/npu/impl/OckVsaAnnIndexCapacity.h"

namespace ock {
namespace vsa {
namespace neighbor {
namespace npu {

uint64_t OckVsaAnnIndexCapacity::HostSpace(
    const OckVsaAnnCreateParam &param, uint64_t sizeOfFeature, uint64_t sizeOfNorm)
{
    // HOST也存一份数据，方便快速delete
    uint64_t idxMapByteSize = param.MaxFeatureRowCount() * 32ULL;            // 每条数据的idx大概占用32字节
    uint64_t hostTempVariableByteSize = 2ULL * 1024ULL * 1024ULL * 1024ULL;  // 2GB临时变量空间
    return DeviceSpace(param, sizeOfFeature, sizeOfNorm) + idxMapByteSize + hostTempVariableByteSize;
}
uint64_t OckVsaAnnIndexCapacity::DeviceSpace(
    const OckVsaAnnCreateParam &param, uint64_t sizeOfFeature, uint64_t sizeOfNorm)
{
    const uint64_t rowCount = param.MaxFeatureRowCount() + param.BlockRowCount();
    uint64_t featureBlockByteSize = rowCount * sizeOfFeature;
    uint64_t normByteSize = rowCount * sizeOfNorm;
    uint64_t keyTraitByteSize =
        rowCount * (OckVsaAnnRawBlockInfo::KeyAttrTimeBytes() + OckVsaAnnRawBlockInfo::KeyAttrQuotientBytes() +
                       OckVsaAnnRawBlockInfo::KeyAttrRemainderBytes());
    uint64_t extKeyTraitByteSize = rowCount * param.ExtKeyAttrByteSize();
    uint64_t maskByteSize = param.MaxFeatureRowCount() / __CHAR_BIT__;
    // 临时变量, 包括距离结果2字节、掩码中间结果64/8、TopN结果等
    uint64_t tempVariableByteSize = rowCount * (sizeof(float) + sizeof(uint64_t));
    return featureBlockByteSize + normByteSize + keyTraitByteSize + extKeyTraitByteSize + maskByteSize +
           tempVariableByteSize;
}
}  // namespace npu
}  // namespace neighbor
}  // namespace vsa
}  // namespace ock