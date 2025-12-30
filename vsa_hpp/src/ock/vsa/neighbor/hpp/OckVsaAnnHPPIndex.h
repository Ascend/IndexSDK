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


#ifndef OCK_VSA_HETERO_PIECEWISE_PROGRESSIVE_ANN_INDEX_H
#define OCK_VSA_HETERO_PIECEWISE_PROGRESSIVE_ANN_INDEX_H
#include "ock/vsa/neighbor/hpp/impl/OckVsaAnnHPPIndexSystem.h"

namespace ock {
namespace vsa {
namespace neighbor {
namespace hpp {
template <typename Data, uint64_t DimSizeT, uint64_t NormTypeByteSizeT, typename KeyTrait>
class OckVsaAnnHPPIndex : public OckVsaAnnIndexBase<Data, DimSizeT, NormTypeByteSizeT, KeyTrait> {
public:
    using BaseT = OckVsaAnnIndexBase<Data, DimSizeT, NormTypeByteSizeT, KeyTrait>;
    using DataT = Data;
    using KeyTraitT = KeyTrait;
    using KeyTypeTupleT = typename KeyTrait::KeyTypeTuple;
    virtual ~OckVsaAnnHPPIndex() noexcept = default;
    explicit OckVsaAnnHPPIndex(std::shared_ptr<hcps::handler::OckHeteroHandler> heteroHandler,
        const KeyTrait &defaultTrait, std::shared_ptr<OckVsaAnnCreateParam> parameter,
        std::shared_ptr<OckVsaAnnCreateParam> npuParameter, std::shared_ptr<OckVsaAnnCreateParam> kernelParameter);
    OckVsaErrorCode Init(void) noexcept;

    OckVsaErrorCode AddFeature(const OckVsaAnnAddFeatureParam<DataT, KeyTrait> &featureParam) override;

    OckVsaErrorCode Search(const OckVsaAnnQueryCondition<DataT, DimSizeT, KeyTrait> &queryCond,
        OckVsaAnnQueryResult<DataT, KeyTrait> &outResult) override;

    uint64_t GetFeatureNum(void) const override;
    uint64_t MaxFeatureRowCount(void) const;
    OckVsaErrorCode GetFeatureByLabel(uint64_t count, const int64_t *labels, DataT *features) const override;
    OckVsaErrorCode GetFeatureAttrByLabel(uint64_t count, const int64_t *labels,
        KeyTypeTupleT *attributes) const override;
    uintptr_t GetCustomAttrByBlockId(uint32_t blockId, OckVsaErrorCode &errorCode) const override;
    uint32_t GetCustomAttrBlockCount(void) const override;

    OckVsaErrorCode DeleteFeatureByLabel(uint64_t count, const int64_t *labels) override;
    OckVsaErrorCode DeleteFeatureByToken(uint64_t count, const uint32_t *tokens) override;

    std::shared_ptr<OckVsaHPPKernelExt<DataT, DimSizeT, NormTypeByteSizeT, KeyTrait>> Kernel(void);
    std::shared_ptr<hcps::handler::OckHeteroHandler> Handler(void);

private:
    OckVsaErrorCode SearchSingle(const OckVsaAnnSingleBatchQueryCondition<DataT, DimSizeT, KeyTrait> &queryCond,
        OckFloatTopNQueue &outResult) const;

    const uint32_t maxGroupCountInNpu{ 0 };
    std::shared_ptr<OckVsaAnnCreateParam> param{ nullptr };
    mutable std::shared_ptr<npu::OckVsaAnnNpuIndex<DataT, DimSizeT, NormTypeByteSizeT, KeyTrait>> npuIndex{ nullptr };
    mutable std::shared_ptr<OckVsaHPPKernelExt<DataT, DimSizeT, NormTypeByteSizeT, KeyTrait>> hppKernel{ nullptr };
    mutable std::shared_ptr<hcps::handler::OckHeteroHandler> handler{ nullptr };
    bool isThresholdInitialised{ false };
};
} // namespace hpp
} // namespace neighbor
} // namespace vsa
} // namespace ock
#include "ock/vsa/neighbor/hpp/impl/OckVsaAnnHPPIndexConstruct.h"
#include "ock/vsa/neighbor/hpp/impl/OckVsaAnnHPPInderyDeleteFeature.h"
#include "ock/vsa/neighbor/hpp/impl/OckVsaAnnHPPIndexAddFeature.h"
#include "ock/vsa/neighbor/hpp/impl/OckVsaAnnHPPIndexQueryFeature.h"
#endif