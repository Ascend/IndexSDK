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


#ifndef OCK_VSA_HETERO_PIECEWISE_PROGRESSIVE_SLICE_ASSEMBLE_WRITER_H
#define OCK_VSA_HETERO_PIECEWISE_PROGRESSIVE_SLICE_ASSEMBLE_WRITER_H
#include "ock/vsa/neighbor/hpp/kernel/impl/OckVsaHPPKernelSystem.h"
#include "ock/vsa/neighbor/hpp/kernel/impl/OckVsaHPPSliceIdMgr.h"
#include "ock/vsa/neighbor/hpp/kernel/impl/OckVsaHPPAssembleContext.h"
#include "ock/hcps/nop/OckOpConst.h"
namespace ock {
namespace vsa {
namespace neighbor {
namespace hpp {
namespace impl {
template <typename DataT, uint64_t DimSizeT, uint64_t NormTypeByteSizeT>
struct OckVsaHPPFeatureWriter : public OckVsaHPPAssembleData<DataT, DimSizeT> {
    OckVsaHPPFeatureWriter(std::shared_ptr<adapter::OckVsaAnnFeature> features,
        hcps::handler::OckHeteroHandler &handler, uint32_t rowCountPerBlock);

    void AddWholeSlice(const OckVsaHPPAssembleDataContextSimple<DataT, DimSizeT> &context,
        const OckVsaHPPAssembleData<DataT, DimSizeT> &grpData, uint32_t grpId, uint32_t sliceId);
    void AddSliceByMaskFilter(const OckVsaHPPAssembleDataContextSimple<DataT, DimSizeT> &context,
        const OckVsaHPPAssembleData<DataT, DimSizeT> &grpData, uint32_t grpId, uint32_t sliceId, uint32_t &startPos);

    void MergeOtherCubeAligned(const OckVsaHPPFeatureWriter<DataT, DimSizeT, NormTypeByteSizeT> &other,
        uint32_t &otherStartPos, const uint32_t maxRowCount);
    void MergeOtherUnCubeAligned(const OckVsaHPPFeatureWriter<DataT, DimSizeT, NormTypeByteSizeT> &other,
        uint32_t &otherStartPos, const uint32_t maxRowCount);

    std::shared_ptr<hcps::hfo::OckOneSideIdxMap> idxMap;

private:
    // 供MergeOtherCubeAligned, MergeOtherUnCubeAligned调用
    void MergeOtherImpl(const OckVsaHPPFeatureWriter<DataT, DimSizeT, NormTypeByteSizeT> &other,
        uint32_t &otherStartPos, const uint32_t mergeOtherCount);
};
template <typename DataT, uint64_t DimSizeT, uint64_t NormTypeByteSizeT>
OckVsaHPPFeatureWriter<DataT, DimSizeT, NormTypeByteSizeT>::OckVsaHPPFeatureWriter(
    std::shared_ptr<adapter::OckVsaAnnFeature> features, hcps::handler::OckHeteroHandler &handler,
    uint32_t rowCountPerBlock)
    : OckVsaHPPAssembleData<DataT, DimSizeT>(features),
      idxMap(hcps::hfo::OckOneSideIdxMap::Create(rowCountPerBlock, handler.HmmMgr()))
{
    this->maskData.UnSetAll();
}
template <typename DataT, uint64_t DimSizeT, uint64_t NormTypeByteSizeT>
void OckVsaHPPFeatureWriter<DataT, DimSizeT, NormTypeByteSizeT>::AddWholeSlice(
    const OckVsaHPPAssembleDataContextSimple<DataT, DimSizeT> &context,
    const OckVsaHPPAssembleData<DataT, DimSizeT> &grpData, uint32_t dequeId, uint32_t sliceId)
{
    this->shapedFeature.AddSegment(grpData.shapedFeature, sliceId * context.param.SliceRowCount(),
        context.param.SliceRowCount());

    auto errorCode = memcpy_s(reinterpret_cast<int8_t *>(this->feature->norm->Addr()) +
        this->feature->validateRowCount * NormTypeByteSizeT,
        this->feature->norm->GetByteSize() - this->feature->validateRowCount * NormTypeByteSizeT,
        reinterpret_cast<int8_t *>(grpData.feature->norm->Addr()) +
        sliceId * NormTypeByteSizeT * context.param.SliceRowCount(),
        NormTypeByteSizeT * context.param.SliceRowCount());
    if (errorCode != hmm::HMM_SUCCESS) {
        OCK_VSA_HPP_LOG_ERROR("memcpy_s from count(" <<
            this->feature->norm->GetByteSize() - this->feature->validateRowCount * NormTypeByteSizeT << ") to count(" <<
            NormTypeByteSizeT * context.param.SliceRowCount() << ") failed, errorCode is " << errorCode);
        return;
    }

    this->maskData.CopyFrom(grpData.maskData,
        (sliceId * context.param.SliceRowCount()) / (sizeof(uint64_t) * __CHAR_BIT__),
        this->feature->validateRowCount / (sizeof(uint64_t) * __CHAR_BIT__),
        context.param.SliceRowCount() / (sizeof(uint64_t) * __CHAR_BIT__));
    context.idxMgr.GetOutterIdxs(
        context.innerIdConvertor.ToIdx(context.groupIdDeque[dequeId], sliceId * context.param.SliceRowCount()),
        context.param.SliceRowCount(), *idxMap);

    this->feature->validateRowCount += context.param.SliceRowCount();
}
/*
@brief 这里的组Id是innerIdx中的grpId
*/
template <typename DataT, uint64_t DimSizeT, uint64_t NormTypeByteSizeT>
void OckVsaHPPFeatureWriter<DataT, DimSizeT, NormTypeByteSizeT>::AddSliceByMaskFilter(
    const OckVsaHPPAssembleDataContextSimple<DataT, DimSizeT> &context,
    const OckVsaHPPAssembleData<DataT, DimSizeT> &grpData, uint32_t grpId, uint32_t sliceId, uint32_t &pos)
{
    uint64_t sliceWordStartPos =
        ((uint64_t)sliceId * context.param.SliceRowCount()) / (sizeof(uint64_t) * __CHAR_BIT__);
    uint16_t *dstNormStartAddr = reinterpret_cast<uint16_t *>(this->feature->norm->Addr());
    uint16_t *srcNormStartAddr = reinterpret_cast<uint16_t *>(grpData.feature->norm->Addr());
    if (dstNormStartAddr == nullptr || srcNormStartAddr == nullptr) {
        return;
    }
    while (pos < context.param.SliceRowCount()) {
        if ((pos % (sizeof(uint64_t) * __CHAR_BIT__) == 0) &&
            grpData.maskData.Data()[sliceWordStartPos + pos / (sizeof(uint64_t) * __CHAR_BIT__)] == 0) {
            pos += static_cast<uint32_t>((sizeof(uint64_t) * __CHAR_BIT__));
        } else {
            if (grpData.maskData.At(sliceId * context.param.SliceRowCount() + pos)) {
                this->shapedFeature.AddFrom(grpData.shapedFeature, sliceId * context.param.SliceRowCount() + pos);
                dstNormStartAddr[this->feature->validateRowCount] =
                    srcNormStartAddr[sliceId * context.param.SliceRowCount() + pos];
                this->idxMap->Add(context.idxMgr.GetOutterIdx(context.innerIdConvertor.ToIdx(
                    context.groupIdDeque[grpId], sliceId * context.param.SliceRowCount() + pos)));
                this->feature->validateRowCount++;
            }
            if (this->feature->validateRowCount >= context.param.BlockRowCount()) {
                break;
            }
            pos++;
        }
    }
}
template <typename DataT, uint64_t DimSizeT, uint64_t NormTypeByteSizeT>
void OckVsaHPPFeatureWriter<DataT, DimSizeT, NormTypeByteSizeT>::MergeOtherCubeAligned(
    const OckVsaHPPFeatureWriter<DataT, DimSizeT, NormTypeByteSizeT> &other, uint32_t &otherStartPos,
    const uint32_t maxRowCount)
{
    // 这里的16未来需要直接修改为Shape的方法来获取
    uint32_t curLeftSpaceRowCount = maxRowCount - this->feature->validateRowCount;
    uint32_t otherLeftAlignRowCount =
        utils::SafeRoundDown(other.feature->validateRowCount, static_cast<uint32_t>(hcps::nop::CUBE_ALIGN)) -
        otherStartPos;
    uint32_t mergeOtherCount = std::min(curLeftSpaceRowCount, otherLeftAlignRowCount);

    this->MergeOtherImpl(other, otherStartPos, mergeOtherCount);
}
template <typename DataT, uint64_t DimSizeT, uint64_t NormTypeByteSizeT>
void OckVsaHPPFeatureWriter<DataT, DimSizeT, NormTypeByteSizeT>::MergeOtherImpl(
    const OckVsaHPPFeatureWriter<DataT, DimSizeT, NormTypeByteSizeT> &other, uint32_t &otherStartPos,
    const uint32_t mergeOtherCount)
{
    if (mergeOtherCount == 0) {
        return;
    }
    OCK_VSA_HPP_LOG_DEBUG("Before AddSegment: shapedFeatrue.ValidateRowCount(" <<
        this->shapedFeature.ValidateRowCount() << ") this->idxMap(" << *(this->idxMap) <<
        ")feature->validateRowCount=" << this->feature->validateRowCount << " other.feature->validateRowCount:" <<
        other.feature->validateRowCount << " otherStartPos:" << otherStartPos << " mergeOtherCount=" <<
        mergeOtherCount);
    this->shapedFeature.AddSegment(other.shapedFeature, otherStartPos, mergeOtherCount);
    auto errorCode = memcpy_s(reinterpret_cast<int8_t *>(this->feature->norm->Addr()) +
        this->feature->validateRowCount * NormTypeByteSizeT,
        this->feature->norm->GetByteSize() - this->feature->validateRowCount * NormTypeByteSizeT,
        reinterpret_cast<int8_t *>(other.feature->norm->Addr()) + otherStartPos * NormTypeByteSizeT,
        mergeOtherCount * NormTypeByteSizeT);
    if (errorCode != hmm::HMM_SUCCESS) {
        OCK_VSA_HPP_LOG_ERROR("memcpy_s from count(" <<
            this->feature->norm->GetByteSize() - this->feature->validateRowCount * NormTypeByteSizeT << ") to count(" <<
            mergeOtherCount * NormTypeByteSizeT << ") failed, errorCode is " << errorCode);
        return;
    }
    this->idxMap->AddFrom(*(other.idxMap), otherStartPos, mergeOtherCount);
    otherStartPos += mergeOtherCount;
    this->feature->validateRowCount += mergeOtherCount;
    OCK_VSA_HPP_LOG_DEBUG("After AddSegment: shapedFeatrue.ValidateRowCount(" <<
        this->shapedFeature.ValidateRowCount() << ") this->idxMap(" << *(this->idxMap) <<
        ")feature->validateRowCount=" << this->feature->validateRowCount << " mergeOtherCount=" << mergeOtherCount);
}
template <typename DataT, uint64_t DimSizeT, uint64_t NormTypeByteSizeT>
void OckVsaHPPFeatureWriter<DataT, DimSizeT, NormTypeByteSizeT>::MergeOtherUnCubeAligned(
    const OckVsaHPPFeatureWriter<DataT, DimSizeT, NormTypeByteSizeT> &other, uint32_t &otherStartPos,
    const uint32_t maxRowCount)
{
    // 这里的16未来需要直接修改为Shape的方法来获取
    uint32_t curLeftSpaceRowCount = maxRowCount - this->feature->validateRowCount;
    uint32_t otherLeftRowCount = other.feature->validateRowCount - otherStartPos;
    uint32_t mergeOtherCount = std::min(curLeftSpaceRowCount, otherLeftRowCount);

    this->MergeOtherImpl(other, otherStartPos, mergeOtherCount);
}
} // namespace impl
} // namespace hpp
} // namespace neighbor
} // namespace vsa
} // namespace ock
#endif