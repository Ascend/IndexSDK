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


#ifndef OCK_VSA_NPU_ANN_INDEX_CONTRUCT_H
#define OCK_VSA_NPU_ANN_INDEX_CONTRUCT_H
#include "ock/vsa/neighbor/npu/impl/OckVsaAnnNpuIndexSystem.h"
#include "ock/acladapter/executor/OckAsyncTaskExecuteService.h"

namespace ock {
namespace vsa {
namespace neighbor {
namespace npu {
template <typename DataTemp, uint64_t DimSizeTemp, uint64_t NormTypeByteSize, typename KeyTraitTemp>
OckVsaAnnNpuIndex<DataTemp, DimSizeTemp, NormTypeByteSize, KeyTraitTemp>::OckVsaAnnNpuIndex(
    std::shared_ptr<hcps::handler::OckHeteroHandler> heteroHandler, const KeyTraitTemp &defaultTrait,
    std::shared_ptr<OckVsaAnnCreateParam> spec)
    : dftTrait(defaultTrait), hisGroupCount(0UL), handler(heteroHandler),
      innerIdCvt(adapter::OckVsaHPPInnerIdConvertor::CalcBitCount(spec->GroupRowCount())),
      grpPosMap(spec->MaxGroupCount()),
      idMapMgr(hcps::hfo::OckLightIdxMap::Create(spec->MaxFeatureRowCount(), handler->HmmMgr(), BUCKET_NUM)),
      param(spec), ockSyncUtils(std::make_shared<acladapter::OckSyncUtils>(*(handler->Service())))
{}

template <typename DataTemp, uint64_t DimSizeTemp, uint64_t NormTypeByteSize, typename KeyTraitTemp>
OckVsaErrorCode OckVsaAnnNpuIndex<DataTemp, DimSizeTemp, NormTypeByteSize, KeyTraitTemp>::Init(void) noexcept
{
    return hmm::HMM_SUCCESS;
}
}  // namespace npu
}  // namespace neighbor
}  // namespace vsa
}  // namespace ock
#endif