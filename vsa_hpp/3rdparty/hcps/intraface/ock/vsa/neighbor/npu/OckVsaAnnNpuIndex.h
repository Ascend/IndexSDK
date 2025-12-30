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


#ifndef OCK_VSA_NPU_ANN_INDEX_H
#define OCK_VSA_NPU_ANN_INDEX_H
#include "ock/vsa/neighbor/npu/impl/OckVsaAnnNpuIndexSystem.h"
#include "ock/hcps/nop/l2_norm_op/OckL2NormOpRun.h"
#include "ock/hcps/hcop/topk_dist_comp_op/OckTopkDistCompOpRun.h"
#include "ock/hcps/nop/trans_data_custom_attr_op/OckTransDataCustomAttrOpRun.h"
#include "ock/hcps/nop/trans_data_shaped_op/OckTransDataShapedOpRun.h"
#include "ock/hcps/nop/dist_mask_gen_op/OckDistMaskGenOpRun.h"
#include "ock/hcps/nop/dist_mask_with_extra_gen_op/OckDistMaskWithExtraGenOpRun.h"
#include "ock/acladapter/utils/OckSyncUtils.h"
#include "ock/vsa/neighbor/base/OckVsaHPPInnerIdConvertor.h"
#include "ock/acladapter/utils/OckAscendFp16.h"

namespace ock {
namespace vsa {
namespace neighbor {
namespace npu {
template <typename DataTemp, uint64_t DimSizeTemp, uint64_t NormTypeByteSize, typename KeyTraitTemp>
class OckVsaAnnNpuIndex : public OckVsaAnnIndexBase<DataTemp, DimSizeTemp, NormTypeByteSize, KeyTraitTemp> {
public:
    using BaseT = OckVsaAnnIndexBase<DataTemp, DimSizeTemp, NormTypeByteSize, KeyTraitTemp>;
    using DataT = DataTemp;
    using KeyTraitT = KeyTraitTemp;
    using KeyTypeTupleT = typename KeyTraitTemp::KeyTypeTuple;
    virtual ~OckVsaAnnNpuIndex() noexcept = default;
    OckVsaAnnNpuIndex(std::shared_ptr<hcps::handler::OckHeteroHandler> heteroHandler, const KeyTraitTemp &defaultTrait,
        std::shared_ptr<OckVsaAnnCreateParam> spec);
    OckVsaErrorCode Init(void) noexcept;

    OckVsaErrorCode AddFeature(const OckVsaAnnAddFeatureParam<DataTemp, KeyTraitTemp> &featureParam) override;

    OckVsaErrorCode Search(const OckVsaAnnQueryCondition<DataT, DimSizeTemp, KeyTraitTemp> &queryCond,
        OckVsaAnnQueryResult<DataT, KeyTraitTemp> &outResult) override;

    uint64_t GetFeatureNum(void) const override;
    uint64_t MaxFeatureRowCount(void) const;
    // block数量接口
    OckVsaErrorCode GetFeatureByLabel(uint64_t count, const int64_t *labels, DataTemp *features) const override;
    OckVsaErrorCode GetFeatureAttrByLabel(uint64_t count, const int64_t *labels,
        KeyTypeTupleT *attributes) const override;
    uintptr_t GetCustomAttrByBlockId(uint32_t blockId, OckVsaErrorCode &errorCode) const override;
    uint32_t GetCustomAttrBlockCount(void) const override;

    OckVsaErrorCode DeleteFeatureByLabel(uint64_t count, const int64_t *labels) override;
    OckVsaErrorCode DeleteFeatureByToken(uint64_t count, const uint32_t *tokens) override;

    /*
    @brief 一个时空属性在NPU中存储的字节大小
    */
    static uint64_t KeyAttrByteSize(void);
    std::shared_ptr<hcps::OckHeteroOperatorGroupQueue> CreateDeleteFeatureOp(const std::vector<uint64_t> &deleteIds,
        const std::vector<uint64_t> &copyIds);
    /*
    @brief 内部接口。弹出第一个BlockGroup
    */
    std::shared_ptr<OckVsaAnnRawBlockInfo> PopFrontBlockGroup(
            std::shared_ptr<OckVsaAnnNpuBlockGroup>& outBlockGroup, std::vector<uint64_t> &outterLabels,
        std::deque<std::shared_ptr<hcps::hfo::OckTokenIdxMap>> &tokenToRowIdsMap, OckVsaErrorCode &errorCode);

    /*
    @brief 内部接口, 一次查询一个bs的topk数据
    */
    OckVsaErrorCode Search(const OckVsaAnnSingleBatchQueryCondition<DataT, DimSizeTemp, KeyTraitTemp> &queryCond,
        OckFloatTopNQueue &outResult);
    /*
    @brief 内部接口, 一次查询多个bs的topk数据
    @param outResult， 调用者提前根据queryCond中的queryBatchCount分配好了内存, 直接使用就可以
    */
    OckVsaErrorCode Search(const OckVsaAnnQueryCondition<DataT, DimSizeTemp, KeyTraitTemp> &queryCond,
        std::vector<std::shared_ptr<OckFloatTopNQueue>> &outResult);
    /*
    @brief 根据查询条件(共attrFeature.size() * rowCountPerHmo条数据)，计算mask结果
    @param queryCond查询条件
    @param attrFeatureGroups 过滤属性的底库存放位置(多个过滤底库)
    @param rowCountPerGroup 每个底库的数据条数。调用者确保rowCountPerGroup能被262144 * 64整除
    @result 返回Mask结果, 返回结果存放在NPU侧
    */
    OckVsaErrorCode GetMaskResult(
        const OckVsaAnnSingleBatchQueryCondition<DataTemp, DimSizeTemp, KeyTraitTemp> &queryCond,
        const std::deque<OckVsaAnnKeyAttrInfo> &attrFeatureGroups, uint32_t rowCountPerGroup,
        std::shared_ptr<hmm::OckHmmHMObject> maskHmo);

private:
    void UpdateGroupPosMap(void);

    bool IsGroupFull(std::shared_ptr<OckVsaAnnNpuBlockGroup> group);
    OckVsaErrorCode PrepareOneFeatureBlock(void);

    OckVsaErrorCode CopyDataToHmo(std::shared_ptr<hmm::OckHmmHMObject> dstHmo, const void *srcData,
        uint64_t startByteSize, uint64_t byteSize);
    OckVsaErrorCode AddTimeSpaceAttr(uint64_t addNum, const KeyTypeTupleT *attributes,
        std::shared_ptr<OckVsaAnnNpuBlockGroup> curGrp);
    OckVsaErrorCode AddBaseFeature(const uint64_t addNum, const DataTemp *features,
        std::shared_ptr<OckVsaAnnNpuBlockGroup> curGrp);
    OckVsaErrorCode AddVectorsImpl(const DataTemp *features, const uint64_t addNum,
        std::shared_ptr<OckVsaAnnNpuBlockGroup> curGrp);
    OckVsaErrorCode AddCustomAttr(const uint64_t addNum, const uint8_t *customAttr,
        std::shared_ptr<OckVsaAnnNpuBlockGroup> curGrp);

    OckVsaErrorCode CopySingleFeature(adapter::OckVsaHPPIdx grpPosInfo, DataTemp *features) const;
    OckVsaErrorCode CopySingleAttr(adapter::OckVsaHPPIdx grpPosInfo, KeyTypeTupleT *attributes) const;

    OckVsaErrorCode SearchImpl(int batch,
        const OckVsaAnnSingleBatchQueryCondition<DataT, DimSizeTemp, KeyTraitTemp> &queryCond,
        std::vector<std::shared_ptr<OckFloatTopNQueue>> &outResult, uint32_t offset);
    OckVsaErrorCode GenerateQueryNorm(std::shared_ptr<hcps::nop::OckL2NormOpHmoBlock> hmoNormGroup);
    OckVsaErrorCode DoSearchProcess(uint32_t batch,
        const OckVsaAnnSingleBatchQueryCondition<DataT, DimSizeTemp, KeyTraitTemp> &queryCond,
        std::shared_ptr<hmm::OckHmmHMObject> topkDistHmo, std::shared_ptr<hmm::OckHmmHMObject> topkLabelsHmo);

    void CalculateInnerIdx(uint64_t *topkLabel, uint64_t count, uint64_t *innerIdx);
    OckVsaErrorCode GenerateQueryAttr(uint32_t batch,
        const OckVsaAnnSingleBatchQueryCondition<DataT, DimSizeTemp, KeyTraitTemp> &queryCond,
        std::vector<std::shared_ptr<hmm::OckHmmHMObject>> &queryTimeHmoVec,
        std::vector<std::shared_ptr<hmm::OckHmmHMObject>> &queryTokenIdHmoVec);
    OckVsaErrorCode GenerateSingleQueryAttr(const KeyTraitTemp *attrFilter,
        std::vector<std::shared_ptr<hmm::OckHmmHMObject>> &queryTimeHmoVec,
        std::vector<std::shared_ptr<hmm::OckHmmHMObject>> &queryTokenIdHmoVec, uint32_t maxTokenNumber,
        bool enableTimeFilter);
    OckVsaErrorCode GenerateMask(
        const OckVsaAnnSingleBatchQueryCondition<DataTemp, DimSizeTemp, KeyTraitTemp> &queryCond,
        const std::deque<OckVsaAnnKeyAttrInfo> &attrFeatureGroups, std::shared_ptr<hmm::OckHmmHMObject> maskHmo,
        uint32_t queryBatch, uint64_t maskLenAligned);
    OckVsaErrorCode GenerateMaskWithExtra(
        const OckVsaAnnSingleBatchQueryCondition<DataTemp, DimSizeTemp, KeyTraitTemp> &queryCond,
        const std::deque<OckVsaAnnKeyAttrInfo> &attrFeatureGroups, std::shared_ptr<hmm::OckHmmHMObject> maskHmo,
        uint32_t queryBatch, uint64_t maskLenAligned);
    std::shared_ptr<hcps::nop::OckDistMaskGenOpHmoGroups> PrepareMaskData(
        const std::vector<std::shared_ptr<hmm::OckHmmHMObject>> &queryTimeHmoVec,
        const std::vector<std::shared_ptr<hmm::OckHmmHMObject>> &queryTokenIdHmoVec,
        std::shared_ptr<hmm::OckHmmSubHMObject> maskHmo, const std::deque<OckVsaAnnKeyAttrInfo> &attrFeature,
        uint64_t maskLen, uint32_t blockNumPerGroup);
    std::shared_ptr<hcps::nop::OckDistMaskWithExtraGenOpHmoGroups> PrepareMaskDataWithExtra(
        const std::vector<std::shared_ptr<hmm::OckHmmHMObject>> &queryTimeHmoVec,
        const std::vector<std::shared_ptr<hmm::OckHmmHMObject>> &queryTokenIdHmoVec,
        std::shared_ptr<hmm::OckHmmSubHMObject> maskHmo, std::shared_ptr<hmm::OckHmmSubHMObject> extraMaskHmo,
        const std::deque<OckVsaAnnKeyAttrInfo> &attrFeature, uint64_t maskLen, uint32_t blockNumPerGroup);
    void PushDataToResult(int batch, std::vector<std::shared_ptr<OckFloatTopNQueue>>::iterator outResult,
        const OckVsaAnnSingleBatchQueryCondition<DataT, DimSizeTemp, KeyTraitTemp> &queryCond, uint64_t *topkLabel,
        OckFloat16 *topkDist);
    void ChangeBitSetToVector(const KeyTraitTemp *attrFilter, uint32_t maxTokenNumber,
        std::vector<uint8_t> &tokenIdVec);
    void AddDatas(uint32_t count, OckFloatTopNQueue &outResult, uint64_t *idx, OckFloat16 *distance);
    void AddIds(uint64_t addNum, OckVsaAnnAddFeatureParam<DataTemp, KeyTraitTemp> leftFeatureParam,
        std::shared_ptr<OckVsaAnnNpuBlockGroup> curGrp);

    OckVsaErrorCode LoadTokenDataToHost(std::deque<std::shared_ptr<hmm::OckHmmHMObject>> &quotientQueue,
        std::deque<std::shared_ptr<hmm::OckHmmHMObject>> &reminderQueue);
    void UpdateOckTokenIdxMap(const std::vector<uint64_t> &innerIds, const std::vector<uint64_t> &copyIds,
        std::deque<std::shared_ptr<hmm::OckHmmHMObject>> &quotientQueue,
        std::deque<std::shared_ptr<hmm::OckHmmHMObject>> &reminderQueue,
        uint64_t count);
    void DeleteBackTokenIdxMap(std::deque<std::shared_ptr<hmm::OckHmmHMObject>> &quotientQueue,
                            std::deque<std::shared_ptr<hmm::OckHmmHMObject>> &reminderQueue,
                            uint64_t count);
    std::shared_ptr<hcps::OckHeteroOperatorBase> BuildDeleteAttrOp(const std::vector<uintptr_t> &src,
        const std::vector<uintptr_t> &dst, uint64_t deleteSize, uint32_t dataType, uint8_t copyNum);
    OckVsaErrorCode DeleteAttrByIds(const std::vector<uint64_t> &innerIds, const std::vector<uint64_t> &copyIds,
        std::shared_ptr<hcps::OckHeteroOperatorGroup> attrGroupOp);
    OckVsaErrorCode DeleteCustomAttrByIds(const std::vector<uint64_t> &innerIds, const std::vector<uint64_t> &copyIds,
        std::shared_ptr<hcps::OckHeteroOperatorGroup> customAttrGroupOp);
    OckVsaErrorCode DeleteShapedByIds(const std::vector<uint64_t> &innerIds, const std::vector<uint64_t> &copyIds,
        std::shared_ptr<hcps::OckHeteroOperatorGroup> shapedGroupOp);
    OckVsaErrorCode DeleteNormByIds(const std::vector<uint64_t> &innerIds, const std::vector<uint64_t> &copyIds);
    OckVsaErrorCode DeleteInvalidBlock(uint64_t oldFeatureNum, uint64_t removeCount);
    OckVsaErrorCode DeleteRun(std::vector<uint64_t> &deleteChaosIds, std::vector<uint64_t> &copyIds, uint64_t totalNum,
        uint64_t count);
    OckVsaErrorCode TransOffsetInBaseToInnerIdx(uint64_t offset, uint64_t &innerIdx);
    OckVsaErrorCode TransInnerIdxToOffsetInBase(uint64_t innerIdx, uint64_t &offset);

    uint64_t GetByteSize(void) const;
    /*
    @grpId 就是blockGroups的下标， 因此blockGroups中的数据变化后 tokenIdxVectorMap中的grpId需要跟随变更
    @innerIdx uint64_t 是一个特殊的值, 它是由grpId + offset组成, 意味着 grpId变化后，idMapMgr中的innerIdx需要变化
    */
    const KeyTraitTemp dftTrait;
    uint32_t hisGroupCount{ 0UL }; // 历史所有Group量, 包括删除的, 与MaxGroupCount配合， 达到循环使用ID功效
    std::shared_ptr<hcps::handler::OckHeteroHandler> handler;
    adapter::OckVsaHPPInnerIdConvertor innerIdCvt;
    std::deque<uint32_t> groupIdDeque{};                               // 队列中各元素所属的groupId
    std::vector<uint32_t> grpPosMap{};                                 // 根据grpId获取该group在哪个deque位置
    std::deque<std::shared_ptr<OckVsaAnnNpuBlockGroup>> blockGroups{}; // block信息
    std::deque<std::deque<std::shared_ptr<hcps::hfo::OckTokenIdxMap>>> tokenIdxVectorMap{}; // 各组TokenId与内部idx的映射关系
    std::shared_ptr<hcps::hfo::OckLightIdxMap> idMapMgr;                      // id映射数据
    std::shared_ptr<OckVsaAnnCreateParam> param;
    std::shared_ptr<acladapter::OckSyncUtils> ockSyncUtils;
    bool isThresholdInitialised{ false };
};
} // namespace npu
} // namespace neighbor
} // namespace vsa
} // namespace ock
#include "ock/vsa/neighbor/npu/impl/OckVsaAnnNpuIndexContruct.h"
#include "ock/vsa/neighbor/npu/impl/OckVsaAnnNpuIndexDeleteFeature.h"
#include "ock/vsa/neighbor/npu/impl/OckVsaAnnNpuIndexAddFeature.h"
#include "ock/vsa/neighbor/npu/impl/OckVsaAnnNpuIndexQueryFeature.h"
#endif