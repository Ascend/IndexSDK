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


#ifndef OCK_VSA_HETERO_PIECEWISE_PROGRESSIVE_SLICE_ASSEMBLE_CONTEXT_H
#define OCK_VSA_HETERO_PIECEWISE_PROGRESSIVE_SLICE_ASSEMBLE_CONTEXT_H
#include "ock/vsa/neighbor/base/OckVsaAnnCreateParam.h"
#include "ock/vsa/neighbor/base/OckVsaAnnIndexBase.h"
#include "ock/vsa/neighbor/npu/OckVsaAnnNpuIndex.h"
#include "ock/vsa/neighbor/hpp/kernel/impl/OckVsaHPPSliceIdMgr.h"
#include "ock/vsa/neighbor/hpp/kernel/impl/OckVsaHPPMaskQuery.h"

namespace ock {
namespace vsa {
namespace neighbor {
namespace hpp {
namespace impl {
template <typename DataT, uint64_t DimSizeT> struct OckVsaHPPAssembleData {
    OckVsaHPPAssembleData(std::shared_ptr<adapter::OckVsaAnnFeature> features);
    OckVsaHPPAssembleData(std::shared_ptr<adapter::OckVsaAnnFeature> features, const OckVsaHPPGroupMaskBase &groupMask);
    hcps::algo::OckShape<DataT, DimSizeT> shapedFeature;
    hcps::algo::OckRefBitSet maskData;
    std::shared_ptr<adapter::OckVsaAnnFeature> feature;
};
template <typename DataT, uint64_t DimSizeT> struct OckVsaHPPAssembleDataContextSimple {
    OckVsaHPPAssembleDataContextSimple(const std::deque<std::shared_ptr<hmm::OckHmmHMObject>> &features,
        const std::deque<std::shared_ptr<hmm::OckHmmHMObject>> &norms, const impl::OckVsaHPPMaskQuery &maskQuery,
        const hcps::hfo::OckLightIdxMap &idxMap, const adapter::OckVsaHPPInnerIdConvertor &innerIdTransformer,
        const OckVsaAnnCreateParam &parameter, const std::deque<uint32_t> &groupIds);

    const hcps::hfo::OckLightIdxMap &idxMgr;
    const adapter::OckVsaHPPInnerIdConvertor &innerIdConvertor;
    const OckVsaAnnCreateParam &param;
    const std::deque<uint32_t> &groupIdDeque; // 这里表示grpDatas中的每个下标对应的innerIdx中的groupId

    std::deque<OckVsaHPPAssembleData<DataT, DimSizeT>> grpDatas{};
};

template <typename DataT, uint64_t DimSizeT>
struct OckVsaHPPAssembleDataContext : public OckVsaHPPAssembleDataContextSimple<DataT, DimSizeT> {
    OckVsaHPPAssembleDataContext(OckVsaHPPSliceIdMgr &sliceIdManager,
        const std::deque<std::shared_ptr<hmm::OckHmmHMObject>> &features,
        const std::deque<std::shared_ptr<hmm::OckHmmHMObject>> &norms, const impl::OckVsaHPPMaskQuery &maskQuery,
        const hcps::hfo::OckLightIdxMap &idxMap, const adapter::OckVsaHPPInnerIdConvertor &innerIdTransformer,
        const OckVsaAnnCreateParam &parameter, const std::deque<uint32_t> &groupIds);

    OckVsaHPPSliceIdMgr &sliceIdMgr;
};

template <typename DataT, uint64_t DimSizeT>
std::ostream &operator << (std::ostream &os, const OckVsaHPPAssembleData<DataT, DimSizeT> &data)
{
    return os << "{'maskData.Count': " << data.maskData.Count() << "}";
}
template <typename DataT, uint64_t DimSizeT>
std::ostream &operator << (std::ostream &os, const OckVsaHPPAssembleDataContextSimple<DataT, DimSizeT> &data)
{
    os << " param:" << data.param << " groupIdDeque:";
    utils::PrintContainer(os, data.groupIdDeque) << " grpDatas: ";
    return utils::PrintContainer(os, data.grpDatas);
}
template <typename DataT, uint64_t DimSizeT>
std::ostream &operator << (std::ostream &os, const OckVsaHPPAssembleDataContext<DataT, DimSizeT> &data)
{
    os << " sliceIdMgr: sliceCount=" << data.sliceIdMgr.SliceCount() << " groupCount=" <<
        data.sliceIdMgr.GroupCount() << " param:" << data.param << " groupIdDeque:";
    utils::PrintContainer(os, data.groupIdDeque) << " grpDatas: ";
    return utils::PrintContainer(os, data.grpDatas);
}
template <typename DataT, uint64_t DimSizeT>
OckVsaHPPAssembleData<DataT, DimSizeT>::OckVsaHPPAssembleData(std::shared_ptr<adapter::OckVsaAnnFeature> features)
    : shapedFeature(features->feature->Addr(), features->feature->GetByteSize(), features->validateRowCount),
      maskData(features->mask->GetByteSize() * __CHAR_BIT__, reinterpret_cast<uint64_t *>(features->mask->Addr())),
      feature(features)
{}
template <typename DataT, uint64_t DimSizeT>
OckVsaHPPAssembleData<DataT, DimSizeT>::OckVsaHPPAssembleData(std::shared_ptr<adapter::OckVsaAnnFeature> features,
    const OckVsaHPPGroupMaskBase &groupMask)
    : shapedFeature(features->feature->Addr(), features->feature->GetByteSize(), features->validateRowCount),
      maskData(groupMask.BitSet()),
      feature(features)
{}
template <typename DataT, uint64_t DimSizeT>
OckVsaHPPAssembleDataContextSimple<DataT, DimSizeT>::OckVsaHPPAssembleDataContextSimple(
    const std::deque<std::shared_ptr<hmm::OckHmmHMObject>> &features,
    const std::deque<std::shared_ptr<hmm::OckHmmHMObject>> &norms, const impl::OckVsaHPPMaskQuery &maskQuery,
    const hcps::hfo::OckLightIdxMap &idxMap, const adapter::OckVsaHPPInnerIdConvertor &innerIdTransformer,
    const OckVsaAnnCreateParam &parameter, const std::deque<uint32_t> &groupIds)
    : idxMgr(idxMap), innerIdConvertor(innerIdTransformer), param(parameter), groupIdDeque(groupIds)
{
    for (uint32_t i = 0; i < features.size(); ++i) {
        grpDatas.emplace_back(std::make_shared<adapter::OckVsaAnnFeature>(features.at(i), norms.at(i),
            std::shared_ptr<hmm::OckHmmHMObject>(), param.GroupRowCount(), param.GroupRowCount()),
            maskQuery.GroupQuery(i));
    }
}
template <typename DataT, uint64_t DimSizeT>
OckVsaHPPAssembleDataContext<DataT, DimSizeT>::OckVsaHPPAssembleDataContext(OckVsaHPPSliceIdMgr &sliceIdManager,
    const std::deque<std::shared_ptr<hmm::OckHmmHMObject>> &features,
    const std::deque<std::shared_ptr<hmm::OckHmmHMObject>> &norms, const impl::OckVsaHPPMaskQuery &maskQuery,
    const hcps::hfo::OckLightIdxMap &idxMap, const adapter::OckVsaHPPInnerIdConvertor &innerIdTransformer,
    const OckVsaAnnCreateParam &parameter, const std::deque<uint32_t> &groupIds)
    : OckVsaHPPAssembleDataContextSimple<DataT, DimSizeT>(features, norms, maskQuery, idxMap, innerIdTransformer,
    parameter, groupIds),
      sliceIdMgr(sliceIdManager)
{}
} // namespace impl
} // namespace hpp
} // namespace neighbor
} // namespace vsa
} // namespace ock
#endif