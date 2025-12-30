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


#ifndef OCK_VSA_ANN_COLLECT_RESULT_PROCESSOR_H
#define OCK_VSA_ANN_COLLECT_RESULT_PROCESSOR_H
#include <cstdint>
#include "ock/vsa/OckVsaErrorCode.h"
#include "ock/hmm/mgr/OckHmmHMObject.h"
#include "ock/hcps/hfo/OckOneSideIdxMap.h"
#include "ock/hcps/algo/OckTopNQueue.h"
#include "ock/hcps/handler/OckHeteroHandler.h"
#include "ock/vsa/neighbor/base/OckVsaAnnFeatureSet.h"
#include "ock/vsa/neighbor/base/OckVsaNeighborRelationTopNResult.h"

namespace ock {
namespace vsa {
namespace neighbor {
namespace adapter {

// 储存用于计算主小区topN的信息
struct OckVsaNeighborSampleInfo {
    OckVsaNeighborSampleInfo(uint64_t blockSize, uint32_t lastBlockSize,
        std::deque<std::shared_ptr<hmm::OckHmmHMObject>> &shapedBlocksListInNpu,
        std::deque<std::shared_ptr<hmm::OckHmmHMObject>> &normBlocksListInNpu, std::deque<uint32_t> &groupSizeInfo)
        : blockRowCount(blockSize),
          lastBlockRowCount(lastBlockSize),
          shapedFeatureBlockListInNpu(shapedBlocksListInNpu),
          normBlockListInNpu(normBlocksListInNpu),
          groupRowCountInfo(groupSizeInfo)
    {}
    uint64_t blockRowCount;
    uint32_t lastBlockRowCount;
    std::deque<std::shared_ptr<hmm::OckHmmHMObject>> shapedFeatureBlockListInNpu;
    std::deque<std::shared_ptr<hmm::OckHmmHMObject>> normBlockListInNpu;
    std::deque<uint32_t> groupRowCountInfo;
};
template <typename DataTemp, uint64_t DimSizeTemp>
class OckVsaAnnCollectResultProcessor {
public:
    virtual ~OckVsaAnnCollectResultProcessor() noexcept = default;

    virtual OckVsaErrorCode Init(void) = 0;

    /*
    @brief 通知收集到的结果，这些数据都存放在host端，
    这些hmo的内存可能比实际需要的稍微大一些，具体里面存放了多少数据，以rowsCount为准
    processor需要根据收集的feature进行topN计算。这里会多次调用NotifyResult，应该针对多次调用的NotifyResult进行TopN计算
    警告：这里收到通知后应该及时处理，NotifyResult函数体内，不要有执行时间比较长的操作。
    @param feature 底库信息
    @param idxMap 这部分底库数据对应的外部的标签
    @param usingMask 是否使用Mask
    */
    virtual void NotifyResult(std::shared_ptr<OckVsaAnnFeature> feature,
        std::shared_ptr<hcps::hfo::OckOneSideIdxMap> idxMap, bool usingMask, OckVsaErrorCode &errorCode) = 0;
    /*
    @brief 通知结束，汇总topN结果返回(topN为 CreateNPUProcessor传入的topN值)
    */
    virtual std::vector<hcps::algo::FloatNode> NotifyResultEnd(OckVsaErrorCode &errorCode) = 0;

    /*
    @brief 获取TopN结果
    这里没有填写查询向量，
    因为实现此接口的用户很清楚查询向量是什么，同时也清楚对查询向量做预处理来加快速度
    这里需要根据传入的centroidSet中的数据分组获取TopN结果，每组的TopN独立计算TopN
    返回的TopN结果应该是排序好的TopN结果，每个TopN内部按距离由近到远排列
    */
    virtual std::shared_ptr<std::vector<std::vector<hcps::algo::FloatNode>>> GetTopNResults(
        std::shared_ptr<OckVsaAnnFeatureSet> featureSet, uint32_t topK, OckVsaErrorCode &errorCode) = 0;

    /*
    @brief 返回的TopN结果应该是排序好的TopN结果，每个TopN内部按距离由近到远排列
    */
    virtual std::shared_ptr<relation::OckVsaNeighborRelationTopNResult> GetSampleCellTopNResult(
            const OckVsaNeighborSampleInfo& sampleInfo, uint32_t topK,
        OckVsaErrorCode &errorCode) = 0;

    /*
    @brief 生成基于NPU的Processor
    @param handler
    @param queryCond 查询向量，这里只有一个查询向量。 NotifyResult 和 GetTopNResults 都需要根据queryCond进行计算
    @param topN
    */
    static std::shared_ptr<OckVsaAnnCollectResultProcessor> CreateNPUProcessor(
        std::shared_ptr<hcps::handler::OckHeteroHandler> handler, const std::vector<DataTemp> &queryCond,
        uint32_t topN);
};
}  // namespace adapter
}  // namespace neighbor
}  // namespace vsa
}  // namespace ock
#endif