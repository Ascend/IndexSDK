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


#ifndef OCK_VSA_HETERO_PIECEWISE_PROGRESSIVE_ANN_INDEX_CONSTRUCT_H
#define OCK_VSA_HETERO_PIECEWISE_PROGRESSIVE_ANN_INDEX_CONSTRUCT_H
#include "ock/vsa/neighbor/hpp/impl/OckVsaAnnHPPIndexSystem.h"

namespace ock {
namespace vsa {
namespace neighbor {
namespace hpp {
template <typename DataT, uint64_t DimSizeT, uint64_t NormTypeByteSizeT, typename KeyTrait>
OckVsaAnnHPPIndex<DataT, DimSizeT, NormTypeByteSizeT, KeyTrait>::OckVsaAnnHPPIndex(
    std::shared_ptr<hcps::handler::OckHeteroHandler> heteroHandler, const KeyTrait &defaultTrait,
    std::shared_ptr<OckVsaAnnCreateParam> parameter, std::shared_ptr<OckVsaAnnCreateParam> npuParameter,
    std::shared_ptr<OckVsaAnnCreateParam> kernelParameter)
    : maxGroupCountInNpu(npuParameter->MaxGroupCount()),
      param(parameter),
      npuIndex(
        std::make_shared<ock::vsa::neighbor::npu::OckVsaAnnNpuIndex<DataT, DimSizeT, NormTypeByteSizeT, KeyTrait>>(
            heteroHandler, defaultTrait, npuParameter)),
      hppKernel(std::make_shared<OckVsaHPPKernelExt<DataT, DimSizeT, NormTypeByteSizeT, KeyTrait>>(heteroHandler,
          defaultTrait, kernelParameter)),
      handler(heteroHandler)
{
    OCK_VSA_HPP_LOG_INFO("OckVsaAnnHPPIndex: maxFeatureNumber="
                         << param->MaxFeatureRowCount() << " maxGroupCount=" << param->MaxGroupCount()
                         << " npu.maxGroupCount=" << npuParameter->MaxGroupCount()
                         << " kernel.maxGroupCount=" << kernelParameter->MaxGroupCount()
                         << *(handler->HmmMgr().GetUsedInfo(64ULL * 1024ULL * 1024ULL)));
}
template <typename DataT, uint64_t DimSizeT, uint64_t NormTypeByteSizeT, typename KeyTrait>
OckVsaErrorCode OckVsaAnnHPPIndex<DataT, DimSizeT, NormTypeByteSizeT, KeyTrait>::Init(void) noexcept
{
    auto retCode = npuIndex->Init();
    if (retCode != hmm::HMM_SUCCESS) {
        return retCode;
    }
    return hppKernel->Init();
}
}  // namespace hpp
}  // namespace neighbor
}  // namespace vsa
}  // namespace ock
#endif