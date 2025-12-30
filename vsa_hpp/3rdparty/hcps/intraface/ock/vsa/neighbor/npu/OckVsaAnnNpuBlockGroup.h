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


#ifndef OCK_VSA_NPU_ANN_INDEX_BLOCK_GROUP_H
#define OCK_VSA_NPU_ANN_INDEX_BLOCK_GROUP_H
#include <vector>
#include "ock/hcps/hop/OckSplitGroupOp.h"
#include "ock/hmm/mgr/OckHmmHMObject.h"
#include "ock/hcps/handler/OckHeteroHandler.h"
#include "ock/vsa/neighbor/base/OckVsaAnnFeatureSet.h"
#include "ock/vsa/attr/OckTimeSpaceAttrTrait.h"
#include "ock/hcps/stream/OckHeteroStreamBase.h"

namespace ock {
namespace vsa {
namespace neighbor {
namespace npu {
const int OPS_DATA_TYPE_TIMES = 2;
const uint8_t OPS_DATA_PADDING_VAL = 64;
const float Fp16Mistake = 0.001F;
const double distanceThreshold = 1.0 + Fp16Mistake;
const uint64_t BUCKET_NUM = 16384ULL;
struct OckVsaAnnKeyAttrInfo {
    static uint64_t KeyAttrTimeBytes(void);
    static uint64_t KeyAttrQuotientBytes(void);
    static uint64_t KeyAttrRemainderBytes(void);
    static uint64_t KeyAttrAllBytes(void);
    static void AttrCvt(attr::OckTimeSpaceAttr &toData, uintptr_t attrTime, uintptr_t attrQuotient,
        uintptr_t attrRemainder);
    std::shared_ptr<hmm::OckHmmHMObject> keyAttrTime{ nullptr };      // 时空属性的时间 4字节
    std::shared_ptr<hmm::OckHmmHMObject> keyAttrQuotient{ nullptr };  // 时空属性的空间的商(第几个字节) 4字节
    std::shared_ptr<hmm::OckHmmHMObject> keyAttrRemainder{ nullptr }; // 时空属性的空间的余数2字节(1个有效字节)
    std::shared_ptr<hmm::OckHmmHMObject> extKeyAttr{ nullptr };       // 用户自定义扩展属性数据
};

struct OckVsaAnnRawBlockInfo : public OckVsaAnnKeyAttrInfo {
    uint64_t GetByteSize(void) const;             //  feature 和 norm 的 ByteSize
    std::shared_ptr<hmm::OckHmmHMObject> feature{ nullptr }; // 未分形的底库数据
    std::shared_ptr<hmm::OckHmmHMObject> norm{ nullptr };    // norms数据
    uint32_t rowCount{ 0 };                            // 数据总条数
};
/*
@brief 特别地这里每个block的空间都是一样大的(maxRowCount一样)
*/
struct OckVsaAnnNpuBlockGroup {
    uint32_t BlockCount(void) const;
    uint64_t GetByteSize(void) const;
    std::vector<std::shared_ptr<hmm::OckHmmHMObject>> features{};     // 分形后的底库数据
    std::vector<std::shared_ptr<hmm::OckHmmHMObject>> norms{};        // norms数据
    std::vector<std::shared_ptr<hmm::OckHmmHMObject>> keyAttrsTime{}; // 时空属性时间 4字节
    std::vector<std::shared_ptr<hmm::OckHmmHMObject>> keyAttrsQuotient{}; // 时空属性的空间的商(第几个字节) 4字节
    std::vector<std::shared_ptr<hmm::OckHmmHMObject>> keyAttrsRemainder{}; // 时空属性的空间的余数2字节(1个有效字节)
    std::vector<std::shared_ptr<hmm::OckHmmHMObject>> extKeyAttrs{}; // 用户自定义扩展属性数据
    uint32_t rowCount{ 0 };                                        // 数据条数
    uint32_t lastBlockRowCount{ 0 };                               // 最后一个Block的数据条数
    uint32_t lastCustomBlockRowCount{ 0 };                         // 最后一个custom Block的数据条数
};

template <typename DataTemp, uint32_t DimSizeTemp>
std::shared_ptr<hmm::OckHmmHMObject> MergeMultiShapedHMOToUnshapedHMO(hcps::handler::OckHeteroHandler &handler,
    const std::vector<std::shared_ptr<hmm::OckHmmHMObject>> &shapedHMOVector, hmm::OckHmmErrorCode &errorCode)
{
    uint64_t shapedHMOVectorByteSize = shapedHMOVector.size() * shapedHMOVector.front()->GetByteSize();
    std::shared_ptr<hmm::OckHmmHMObject> unshapedHMO =
        hcps::handler::helper::MakeHostHmo(handler, shapedHMOVectorByteSize, errorCode);
    if (errorCode != hmm::HMM_SUCCESS) {
        return std::shared_ptr<hmm::OckHmmHMObject>();
    }
    DataTemp *dstFeatureAddr = reinterpret_cast<DataTemp *>(unshapedHMO->Addr());
    uint64_t rowCountPerBlock = shapedHMOVector.front()->GetByteSize() / sizeof(DataTemp) / DimSizeTemp;
    auto ops = hcps::hop::MakeOckSplitGroupAtmoicOps<uint64_t, acladapter::OckTaskResourceType::HOST_CPU>(0ULL,
        shapedHMOVector.size(), 1UL, [&shapedHMOVector, dstFeatureAddr, rowCountPerBlock, &handler](uint64_t blockId) {
            OckVsaErrorCode ret = hmm::HMM_SUCCESS;
            auto hostHmo = hcps::handler::helper::CopyToHostHmo(handler, shapedHMOVector.at(blockId), ret);
            OCK_CHECK_RETURN_ERRORCODE(ret);
            hcps::algo::OckShape<DataTemp, DimSizeTemp> srcShape(hostHmo->Addr(), hostHmo->GetByteSize(),
                rowCountPerBlock);
            srcShape.Restore(dstFeatureAddr + blockId * rowCountPerBlock * DimSizeTemp);
            return ret;
        });
    auto stream = hcps::handler::helper::MakeStream(handler, errorCode);
    if (errorCode != hmm::HMM_SUCCESS) {
        return std::shared_ptr<hmm::OckHmmHMObject>();
    }
    stream->AddOps(*ops);
    errorCode = stream->WaitExecComplete();
    return unshapedHMO;
}

/*
@brief 将NPU上的多个BLOCK数据Merge到CPU端
特别地：
1. 到CPU端后的数据是未分形的数据，这里需要注意性能。需要考虑多线程操作
2. 可以直接认为每个BLOCK的数据是满的
3. 这里提供的模板参数_DataT和_DimSizeT主要是为了方便分形使用
4. 需要注意的是，返回的OckVsaAnnRawBlockInfo.feature 是未分形的数据
    norm和keyAttrs和extKeyAttrs保留NPU中原有的分形排列，之间将多个block数据拼接在一起就可以了
*/
template <typename DataTemp, uint32_t DimSizeTemp>
std::shared_ptr<OckVsaAnnRawBlockInfo> LoadGroupBlocksIntoHostImpl(hcps::handler::OckHeteroHandler &handler,
    const OckVsaAnnNpuBlockGroup &blockGroup, hmm::OckHmmErrorCode &errorCode)
{
    if (errorCode != hmm::HMM_SUCCESS) {
        return std::shared_ptr<OckVsaAnnRawBlockInfo>();
    }
    std::shared_ptr<OckVsaAnnRawBlockInfo> blockInfo = std::make_shared<OckVsaAnnRawBlockInfo>();
    blockInfo->feature =
        MergeMultiShapedHMOToUnshapedHMO<DataTemp, DimSizeTemp>(handler, blockGroup.features, errorCode);
    blockInfo->norm = hcps::handler::helper::MergeMultiHMObjectsToHost(handler, blockGroup.norms, errorCode);
    blockInfo->keyAttrTime =
        hcps::handler::helper::MergeMultiHMObjectsToHost(handler, blockGroup.keyAttrsTime, errorCode);
    blockInfo->keyAttrQuotient =
        hcps::handler::helper::MergeMultiHMObjectsToHost(handler, blockGroup.keyAttrsQuotient, errorCode);
    blockInfo->keyAttrRemainder =
        hcps::handler::helper::MergeMultiHMObjectsToHost(handler, blockGroup.keyAttrsRemainder, errorCode);
    blockInfo->extKeyAttr =
        hcps::handler::helper::MergeMultiHMObjectsToHost(handler, blockGroup.extKeyAttrs, errorCode);
    if (errorCode != hmm::HMM_SUCCESS) {
        return std::shared_ptr<OckVsaAnnRawBlockInfo>();
    }
    blockInfo->rowCount = blockGroup.rowCount;
    return blockInfo;
}

template <typename DataTemp, uint32_t DimSizeTemp>
std::shared_ptr<OckVsaAnnRawBlockInfo> LoadGroupBlocksIntoHost(hcps::handler::OckHeteroHandler &handler,
    const OckVsaAnnNpuBlockGroup &blockGroup, hmm::OckHmmErrorCode &errorCode)
{
    if (errorCode != hmm::HMM_SUCCESS) {
        return std::shared_ptr<OckVsaAnnRawBlockInfo>();
    }
    uint64_t deviceHMObjectByteSize = blockGroup.GetByteSize() * 2ULL;
    errorCode = hcps::handler::helper::UseIncBindMemory(handler, deviceHMObjectByteSize, "LoadGroupBlocksIntoHost");
    auto usedInfo = handler.HmmMgr().GetUsedInfo(10ULL * 1024ULL * 1024ULL);
    OCK_HCPS_LOG_INFO("LoadGroupBlocksIntoHost after" << *usedInfo);
    if (errorCode != hmm::HMM_SUCCESS) {
        return std::shared_ptr<OckVsaAnnRawBlockInfo>();
    }
    return LoadGroupBlocksIntoHostImpl<DataTemp, DimSizeTemp>(handler, blockGroup, errorCode);
}
} // namespace npu
} // namespace neighbor
} // namespace vsa
} // namespace ock
#endif