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


#ifndef OCK_VSA_HETERO_PIECEWISE_PROGRESSIVE_KERNEL_TOPN_SLICE_SELECTOR_H
#define OCK_VSA_HETERO_PIECEWISE_PROGRESSIVE_KERNEL_TOPN_SLICE_SELECTOR_H
#include "ock/hcps/stream/OckHeteroOperatorBase.h"
#include "ock/hcps/stream/OckHeteroStreamBase.h"
#include "ock/hcps/hop/OckExternalQuicklySortOp.h"
#include "ock/vsa/neighbor/hpp/kernel/impl/OckVsaHPPKernelSystem.h"
#include "ock/vsa/neighbor/base/OckVsaHPPInnerIdConvertor.h"
#include "ock/vsa/neighbor/hpp/kernel/impl/OckVsaHPPMaskQuery.h"
#include "ock/vsa/neighbor/hpp/kernel/impl/OckVsaHPPSelectRate.h"
#include "ock/vsa/neighbor/hpp/kernel/impl/OckVsaHPPSliceIdMgr.h"
#include "ock/vsa/neighbor/hpp/kernel/impl/OckVsaHPPSearchStaticInfo.h"
#include "ock/vsa/neighbor/hpp/kernel/impl/OckVsaHPPAssembleContext.h"
#include "ock/vsa/neighbor/base/OckVsaNeighborRelationTopNResult.h"
namespace ock {
namespace vsa {
namespace neighbor {
namespace hpp {
namespace impl {
template <typename DataT, uint32_t DimSizeT>
void AddSelectMayBeRowsByPrimaryResultOp(hcps::OckHeteroStreamBase &stream,
    const ::ock::vsa::neighbor::relation::OckVsaNeighborRelationTopNResult &primaryResult,
    OckVsaHPPAssembleDataContext<DataT, DimSizeT> &context, uint32_t topK,
    std::deque<std::shared_ptr<relation::OckVsaNeighborRelationHmoGroup>> &relationTable)
{
    for (uint32_t grpId = 0; grpId < relationTable.size(); ++grpId) {
        stream.AddOp(hcps::OckSimpleHeteroOperator<acladapter::OckTaskResourceType::HOST_CPU>::Create(
            [&primaryResult, &context, topK, &relationTable, grpId](hcps::OckHeteroStreamContext &) {
                relationTable[grpId]->SelectRelatedRowIds(primaryResult.TopId(grpId), context.grpDatas[grpId].maskData,
                    context.param.GroupRowCount(), topK, context.sliceIdMgr.SliceSet(grpId));
                return hmm::HMM_SUCCESS;
            }));
    }
}
} // namespace impl
} // namespace hpp
} // namespace neighbor
} // namespace vsa
} // namespace ock
#endif