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


#ifndef OCK_VSA_HETERO_PIECEWISE_PROGRESSIVE_KERNEL_EXT_H
#define OCK_VSA_HETERO_PIECEWISE_PROGRESSIVE_KERNEL_EXT_H
#include "ock/vsa/neighbor/hpp/kernel/impl/OckVsaHPPKernelSystem.h"
#include "ock/vsa/neighbor/hpp/kernel/impl/OckVsaHPPSearchStaticInfo.h"
#include "ock/vsa/neighbor/hpp/kernel/impl/OckVsaHPPMaskQuery.h"
#include "ock/acladapter/utils/OckSyncUtils.h"

namespace ock {
namespace vsa {
namespace neighbor {
namespace hpp {
const uint32_t MINIMUM_DIMENSION = 64ULL;
template <typename Data, uint64_t DimSizeT, uint64_t NormTypeByteSizeT = sizeof(Data) * 2ULL,
    typename KeyTrait = attr::OckMultiKeyTrait<attr::OckMaxMinTrait<uint32_t>>,
    uint64_t BitPerDimT = EXT_FEATURE_SCALE_OUT_BITS>
class OckVsaHPPKernelExt {
public:
    using DataT = Data;
    using HmoDeque = std::deque<std::shared_ptr<hmm::OckHmmHMObject>>;
    using KeyTraitT = KeyTrait;
    using KeyTypeTupleT = typename KeyTrait::KeyTypeTuple;
    using OckVsaVectorHashT = OckVsaVectorHash<(DimSizeT) / (__CHAR_BIT__ * sizeof(uint64_t))>;
    static_assert(DimSizeT % MINIMUM_DIMENSION == 0, "DimSizeT must be divisible by 64ULL");
    static uint64_t DimSize(void)
    {
        return DimSizeT;
    }
    static uint64_t NormTypeByteSize(void)
    {
        return NormTypeByteSizeT;
    }

    virtual ~OckVsaHPPKernelExt() noexcept = default;
    explicit OckVsaHPPKernelExt(std::shared_ptr<hcps::handler::OckHeteroHandler> heteroHandler,
        const KeyTrait &defaultTrait, std::shared_ptr<OckVsaAnnCreateParam> parameter);
    /*
    @brief 初始化，完成变量的空间的分配
    */
    OckVsaErrorCode Init(void);
    /*
    @brief 获取指定label的数据
    */
    OckVsaErrorCode GetFeatureByLabel(uint64_t label, DataT *feature);
    OckVsaErrorCode GetFeatureAttrByLabel(uint64_t label, typename KeyTrait::KeyTypeTuple *feature);
    uintptr_t GetCustomAttrByBlockId(uint32_t blockId, OckVsaErrorCode &errorCode) const;
    uint32_t GetCustomAttrBlockCount(void) const;
    /*
    @brief 查询底库组上有效数据条数, 当groupId不存在， 直接返回0
    */
    uint32_t ValidRowCount(uint32_t groupId) const;
    /*
    @brief 查询底库组上有效数据条数
    */
    uint64_t ValidRowCount(void) const;
    /*
    @brief 当前有效的Group数
    */
    uint32_t GroupCount(void) const;
    /*
    @brief 当前最老的Group中的有效数据条数
    */
    uint64_t ValidRowCountInOldestGroup(void) const;
    /*
    @brief 当前HPP支持的最大Group数，超过此值后AddFeature会强制删除最老的Feature。
    */
    uint32_t MaxGroupCount(void) const;
    /*
    @brief 增加底库, 接口调用完后， 本接口会混乱向量数据，本接口调用完毕后，调用方应该做如下操作
    1. 根据返回的数据顺序重新排列时空属性
    2. 根据返回的结果数据重新排列AttrFilter数据
    特别地：当空间不足时(GroupCount等于MaxGroupCount)，会主动删除最老的Group
    @param hostHmo 底库数据为未分形的数据，一次输入的数据条数必须等于_GroupRowCountT。这些数据由HPP接管
    @param hostNormaHmo 底库数据为未分形的数据，一次输入的数据条数必须等于_GroupRowCountT。这些数据由HPP接管
    @param hostAttrFilterHmo  底库数据为未分形的数据，一次输入的数据条数必须等于_GroupRowCountT。这些数据由HPP接管
    @param outterLabels 对外的label, 特别地，用户的label值不能超过maxFeatureRowCount值,
    需要确保outterLabels.size()大于等于_GroupRowCountT
    @return 错误码
    */
    OckVsaErrorCode AddFeature(npu::OckVsaAnnRawBlockInfo &blockInfo, const std::vector<uint64_t> &outterLabels,
        std::deque<std::shared_ptr<hcps::hfo::OckTokenIdxMap>> tokenToRowIdsMap,
        std::shared_ptr<relation::OckVsaNeighborRelationHmoGroup> neighborRelationGroup);
    /*
    @brief 删除指定位置的group，当整组数据都被打上了删除标签，调用一次删除接口， 删除_GroupRowCountT条数据
    */
    OckVsaErrorCode DeleteEmptyGroup(uint64_t index);
    /*
    @brief 将指定数据打上删除标记，打上删除标记的数据并不会直接删除，仍然会占用内存。
    对应的数据会reset为0(包括底库和attrFilter数据，但norm数据不会修改)
    */
    OckVsaErrorCode SetDroppedByLabel(uint64_t count, const uint64_t *labels);
    OckVsaErrorCode SetDroppedByToken(uint64_t count, const uint32_t *tokens);
    /*
    @brief 删除最老底库中的数据空洞，并记录剩余的有效数据
    */
    OckVsaErrorCode DeleteInvalidFeature(AddFeatureParamMeta<int8_t, attr::OckTimeSpaceAttrTrait> &paramStruct);
    /*
    @brief 还原host侧的customAttr，用于添加到npu侧
    */
    OckVsaErrorCode RestoreCustomAttr(std::shared_ptr<acladapter::OckSyncUtils> syncUtils, uint32_t featureOffset,
        std::vector<uint8_t> &customAttr, uint32_t addFeatureNum);
    /*
    @brief 获取删除数据空洞后的有效数据
    */
    OckVsaErrorCode GetValidData(AddFeatureParamMeta<int8_t, attr::OckTimeSpaceAttrTrait> &paramStruct);
    /*
    @brief 根据MaskData与查询条件，计算需要NPU计算的数据
    @param queryCond 查询条件中的数据维度要等于_DimSizeT，本接口暂不支持多批次。
    @param maskDatas maskDatas的大小为GroupCount，为整个HPP管理的HMO对应对应的mask。
    maskData[0]的高位(对应0x80)代表第一条数据 maxData[0]=0x98(对应的二进制码为 1001 1000)
    代表第1条、4条、5条数据是需要的，其他数据不需要。
    maskData应该根据本HPP的GetAttrFilter的结果生成
    */
    OckVsaErrorCode Search(const OckVsaAnnSingleBatchQueryCondition<DataT, DimSizeT, KeyTrait> &queryCond,
        const std::vector<std::shared_ptr<hmm::OckHmmSubHMObject>> &maskDatas,
        std::shared_ptr<hcps::OckHeteroOperatorBase> rawSearchOp, OckFloatTopNQueue &outResult,
        OckVsaHPPSearchStaticInfo &stInfo);

    /*
    @brief 内部使用， 主要是用于mask生成时使用。
    */
    std::deque<npu::OckVsaAnnKeyAttrInfo> GetAllFeatureAttrs(void) const;

private:
    void SetDroppedByLabel(uint64_t outterIdx);
    void AddSetDroppedByTokenOp(hcps::OckHeteroOperatorGroup &ops, uint32_t token);
    void UpdateGroupPosMap(void);
    OckVsaErrorCode GenerateSortedExtFeature(npu::OckVsaAnnRawBlockInfo &blockInfo);
    std::shared_ptr<hcps::OckHeteroOperatorBase> CreateSetDroppedByTokenOp(uint32_t grpId, uint32_t token,
        const hcps::hfo::OckTokenIdxMap &idxMap);
    OckVsaErrorCode SearchByFullFilter(const OckVsaAnnSingleBatchQueryCondition<DataT, DimSizeT, KeyTrait> &queryCond,
        const impl::OckVsaHPPMaskQuery &maskQuery, std::shared_ptr<hcps::OckHeteroOperatorBase> rawSearchOp,
        OckFloatTopNQueue &outResult,
        std::shared_ptr<adapter::OckVsaAnnCollectResultProcessor<DataT, DimSizeT>> processor,
        OckVsaHPPSearchStaticInfo &stInfo);

    std::shared_ptr<OckVsaAnnCreateParam> param;
    KeyTrait dftTrait;
    uint32_t hisGroupCount{ 0UL }; // 历史所有Group量, 包括删除的, 与MaxGroupCount配合， 达到循环使用ID功效
    adapter::OckVsaHPPInnerIdConvertor innerIdConvertor;           // 内部ID转换器
    std::deque<std::shared_ptr<hmm::OckHmmHMObject>> usedFeatures{}; // 存放在Host上的底库, 分形特征
    std::deque<std::shared_ptr<hmm::OckHmmHMObject>> unusedFeatures{};
    std::deque<std::shared_ptr<hmm::OckHmmHMObject>> usedNorms{}; // 存放在Host上的底库的Norm值
    std::deque<std::shared_ptr<hmm::OckHmmHMObject>> unusedNorms{};
    std::deque<std::shared_ptr<hmm::OckHmmHMObject>> usedAttrTimeFilters{}; // 存放在Dev上
    std::deque<std::shared_ptr<hmm::OckHmmHMObject>> unusedAttrTimeFilters{};
    std::deque<std::shared_ptr<hmm::OckHmmHMObject>> usedAttrQuotientFilters{}; // 存放在Dev上
    std::deque<std::shared_ptr<hmm::OckHmmHMObject>> unusedAttrQuotientFilters{};
    std::deque<std::shared_ptr<hmm::OckHmmHMObject>> usedAttrRemainderFilters{}; // 存放在Dev上
    std::deque<std::shared_ptr<hmm::OckHmmHMObject>> unusedAttrRemainderFilters{};
    std::deque<std::shared_ptr<hmm::OckHmmHMObject>> usedCustomerAttrs{}; // 存放在Dev上
    std::deque<std::shared_ptr<hmm::OckHmmHMObject>> unusedCustomerAttrs{};
    std::deque<std::deque<std::shared_ptr<hcps::hfo::OckTokenIdxMap>>>
        tokenIdxVectorMap{}; // 各组TokenId与内部idx的映射关系
    std::deque<std::shared_ptr<relation::OckVsaNeighborRelationHmoGroup>> usedNeighborRelationGroups{};
    std::shared_ptr<relation::OckVsaSampleFeatureMgr<DataT, DimSizeT>> sampleFeatureMgr{ nullptr };
    // 分组信息
    std::vector<uint64_t> tmpOutterLables{}; // 临时的外部标签数据
    std::deque<uint32_t> groupIdDeque{};     // 队列中各元素所属的groupId
    std::vector<uint32_t> grpPosMap{};       // 根据grpId获取该group在哪个deque位置
    std::deque<std::shared_ptr<hcps::algo::OckElasticBitSet>> usedValidTags{};
    std::deque<std::shared_ptr<hcps::algo::OckElasticBitSet>> unusedValidTags{};
    std::deque<std::shared_ptr<hmm::OckHmmHMObject>> unusedDistHMOs{}; // 存放在Host上的临时底库空间
    std::shared_ptr<hcps::hfo::OckLightIdxMap> idMapMgr{ nullptr };
    std::shared_ptr<hcps::handler::OckHeteroHandler> handler;
};
} // namespace hpp
} // namespace neighbor
} // namespace vsa
} // namespace ock
#include "ock/vsa/neighbor/hpp/kernel/impl/OckVsaHPPKernelExtConstruct.h"
#include "ock/vsa/neighbor/hpp/kernel/impl/OckVsaHPPKernelExtAddFeature.h"
#include "ock/vsa/neighbor/hpp/kernel/impl/OckVsaHPPKernelExtDelFeature.h"
#include "ock/vsa/neighbor/hpp/kernel/impl/OckVsaHPPKernelExtQueryFeature.h"
#endif