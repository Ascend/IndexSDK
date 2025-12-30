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


#ifndef OCK_HCPS_HCOP_TOPK_DIST_COMP_META
#define OCK_HCPS_HCOP_TOPK_DIST_COMP_META
#include <vector>
#include "ock/hmm/mgr/OckHmmHMObject.h"
#include "ock/hcps/nop/OckOpConst.h"
#include "ock/hcps/nop/topk_flat_op/OckTopkFlatMeta.h"
#include "ock/hcps/nop/dist_int8_cos_max_op/OckDistInt8CosMaxMeta.h"
namespace ock {
namespace hcps {
namespace hcop {
// 组合算子处理的底库向量数：ntotal
struct OckTopkDistCompOpMeta {
    int64_t defaultNumDistOps{ nop::DEFAULT_PAGE_BLOCK_NUM }; // topk和dist配比
    int64_t batch{ 1 }; // 仅支持 { 64, 48, 36, 32, 24, 18, 16, 12, 8, 6, 4, 2, 1 }
    int64_t codeBlockSize{ nop::FLAT_DEFAULT_DIST_COMPUTE_BATCH };
    int64_t dims{ 256 };

    nop::OckTopkFlatOpMeta ToTopkOpMeta() const
    {
        nop::OckTopkFlatOpMeta topkOpSpec;
        topkOpSpec.outLabelsDataType = "int64"; // 组合算子仅关心此种情况
        topkOpSpec.batch = batch;
        topkOpSpec.codeBlockSize = codeBlockSize;
        return topkOpSpec;
    }
    nop::OckDistInt8CosMaxOpMeta ToDistOpMeta() const
    {
        nop::OckDistInt8CosMaxOpMeta distOpSpec;
        distOpSpec.batch = batch;
        distOpSpec.dims = dims;
        distOpSpec.codeBlockSize = codeBlockSize;
        return distOpSpec;
    }
};

struct OckTopkDistCompBufferMeta {
    int64_t k{ 1 };
    // 作用1：dist算子中非首个inqry的mask的offset的计算
    // 作用2：topk算子的attr(和topk的迭代更新有关)
    int64_t ntotal{ nop::FLAT_DEFAULT_DIST_COMPUTE_BATCH * nop::DEFAULT_PAGE_BLOCK_NUM };
    int64_t pageId{ 0 };
};

// 注意：暂无法处理各group长度不同的场景
struct OckTopkDistCompOpHmoGroup {
    OckTopkDistCompOpHmoGroup() = default;
    OckTopkDistCompOpHmoGroup(bool isUsingMask, uint32_t batchSize, uint32_t featureDims, uint32_t topK,
        uint32_t blockRowCount, uint32_t blockNum, uint32_t totalNum, uint32_t groupIdx);
    void PushDataBase(std::shared_ptr<hmm::OckHmmHMObject> feature, std::shared_ptr<hmm::OckHmmHMObject> norm);
    void SetQueryHmos(std::shared_ptr<hmm::OckHmmSubHMObject> queryData,
                      std::shared_ptr<hmm::OckHmmSubHMObject> queryNorm,
                      std::shared_ptr<hmm::OckHmmSubHMObject> mask);
    void SetOutputHmos(std::shared_ptr<hmm::OckHmmSubHMObject> topKDists,
                       std::shared_ptr<hmm::OckHmmSubHMObject> topKLabels);

    std::vector<std::shared_ptr<hmm::OckHmmHMObject>> featuresHmo{}; // 【本组特有】int8_t
    std::vector<std::shared_ptr<hmm::OckHmmHMObject>> normsHmo{};    // 【本组特有】OckFloat16
    std::shared_ptr<hmm::OckHmmSubHMObject> queriesHmo{ nullptr };               // 【各组共用】int8_t
    std::shared_ptr<hmm::OckHmmSubHMObject> queriesNormHmo{ nullptr };           // 【各组共用】OckFloat16
    std::shared_ptr<hmm::OckHmmSubHMObject> maskHMO{ nullptr };                  // 【全局】
    std::shared_ptr<hmm::OckHmmSubHMObject> topkDistsHmo{ nullptr };             // 【各组迭代更新】OckFloat16
    std::shared_ptr<hmm::OckHmmSubHMObject> topkLabelsHmo{ nullptr };            // 【各组迭代更新】uint64_t
    bool usingMask;
    uint32_t batch;            // 查询向量的个数
    uint32_t dims;             // 底库与查询向量的维度
    uint32_t k;                // topk的k
    uint32_t blockSize;        // 每个block的数据条数
    uint32_t defaultNumBlocks; // 每个group的block个数(最后一个group除外)
    uint32_t ntotal;           // 整个底库的数据条数
    uint32_t groupId;          // 组号 【本组特有】
};

int64_t GetNumDistOps(const OckTopkDistCompOpMeta &opSpec, const OckTopkDistCompBufferMeta &bufferSpec);
void UpdateMetaFromHmoGroup(OckTopkDistCompOpMeta &opSpec, OckTopkDistCompBufferMeta &bufferSpec,
    const std::shared_ptr<OckTopkDistCompOpHmoGroup> &hmoGroup);
} // namespace hcop
} // namespace hcps
} // namespace ock
#endif // OCK_HCPS_HCOP_TOPK_DIST_COMP_META