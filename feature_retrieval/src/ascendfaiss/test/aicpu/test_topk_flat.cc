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


#include <vector>

#include "gtest/gtest.h"
#ifndef private
#define private public
#define protected public
#endif
#include "cpu_kernel_utils.h"
#include "cpu_nodedef_builder.h"
#undef private
#undef protected

#include "topk_flat_cpu_kernel.h"
#include "utils.h"

using namespace std;
using namespace aicpu;
using namespace aicpu::unittest;

#define CREATE_NODEDEF(shapes, dataTypes, datas, idx)                               \
  auto nodeDef##idx = NodeDefBuilder::CreateNodeDef();                              \
  NodeDefBuilder(nodeDef##idx.get(), "TopkFlat", "TopkFlat")                        \
      .Input({"indists", dataTypes[0], shapes[0], datas[0]})                        \
      .Input({"vmdists", dataTypes[1], shapes[1], datas[1]})                        \
      .Input({"size", dataTypes[2], shapes[2], datas[2]})                           \
      .Input({"opflag", dataTypes[3], shapes[3], datas[3]})                         \
      .Input({"attr", dataTypes[4], shapes[4], datas[4]})                           \
      .Output({"outdists", dataTypes[5], shapes[5], datas[5]})                      \
      .Output({"outlabels", dataTypes[6], shapes[6], datas[6]})

class TopKTest : public testing::Test {};

TEST_F(TopKTest, TestTopK_1) {
    int64_t blockNum = 4;
    int64_t nq = 48;
    int64_t blockSize = 262144;
    int64_t burstLen = 64;
    int64_t burstSize = blockSize / burstLen * 2;
    int64_t coreNum = 2;
    int64_t sizeAlign = 8;
    int64_t flagSize = 16;
    int64_t attrSize = 9;
    int64_t topk = 20;
    int64_t pageId = 0;
    int64_t pageNum = 1;
    int64_t pageSize = blockSize * 16;
    int64_t asc = 0;
    int64_t quickHeap = 1;
    uint32_t actualNum = 213568;

    // data types
    vector<DataType> dataTypes = { DT_FLOAT16,
                                   DT_FLOAT16,
                                   DT_UINT32,
                                   DT_UINT16,
                                   DT_INT64,
                                   DT_FLOAT16,
                                   DT_INT64 };
    // shapes
    vector<vector<int64_t>> shapes = { {blockNum, nq, blockSize},
                                       {blockNum, nq, burstSize},
                                       {blockNum, coreNum, sizeAlign},
                                       {blockNum, coreNum, flagSize},
                                       {attrSize},
                                       {nq, topk},
                                       {nq, topk} };
    // data
    vector<float16_t> indists(blockNum * nq * blockSize);
    Utils::SetRandomValue<float16_t>(indists.data(), indists.size());
    // topk算子会修改indists的值，后续需要计算gt因此先备份一下
    vector<float16_t> indistsBak(indists.begin(), indists.end());

    vector<float16_t> vmdists(blockNum * nq * burstSize);
    vector<uint16_t> vmdistsIndex(blockNum * nq * burstSize / 2);
    vector<float16_t> vmdistsValue(blockNum * nq * burstSize / 2);
    for (auto blockIdx = 0; blockIdx < blockNum; blockIdx++) {
        for (auto qIdx = 0; qIdx < nq; qIdx++) {
            auto curBlockBurstSize = burstSize / 2;
            if (blockIdx == blockNum - 1) {
                curBlockBurstSize = actualNum / burstLen;
            }
            for (auto burstIdx = 0; burstIdx < curBlockBurstSize; burstIdx++) {
                int64_t offsetIndists = Utils::GetOffset(shapes[0], {blockIdx, qIdx, burstIdx * burstLen});
                auto maxIdx = Utils::Argmax(indists.begin() + offsetIndists,
                                            indists.begin() + offsetIndists + burstLen);
                int64_t offsetVmdists = Utils::GetOffset(shapes[1], {blockIdx, qIdx, burstIdx * 2});
                vmdists[offsetVmdists] = indists[offsetIndists + maxIdx];
                vmdists[offsetVmdists + 1] = *(reinterpret_cast<float16_t *>(&maxIdx));
                vmdistsIndex[offsetVmdists / 2] = maxIdx;
                vmdistsValue[offsetVmdists / 2] = indists[offsetIndists + maxIdx];
            }
        }
    }

    vector<uint32_t> opSize(blockNum * coreNum * sizeAlign, 0);
    for (auto blockIdx = 0; blockIdx < blockNum; blockIdx++) {
        auto offset = Utils::GetOffset(shapes[2], {blockIdx, 0, 0});
        if (blockIdx == blockNum - 1) {
            opSize[offset] = actualNum;
        } else {
            opSize[offset] = blockSize;
        }
    }

    vector<uint16_t> opflag(blockNum * coreNum * flagSize, 0);
    for (auto blockIdx = 0; blockIdx < blockNum; blockIdx++) {
        for (auto coreIdx = 0; coreIdx < coreNum; coreIdx++) {
            auto offset = Utils::GetOffset(shapes[3], {blockIdx, coreIdx, 0});
            opflag[offset] = 1;
        }
    }

    vector<int64_t> attrs = {asc, topk, burstLen, blockNum, pageId, pageNum, pageSize, quickHeap, blockSize};

    vector<float16_t> outdists(nq * topk, 0.0);
    vector<int64_t> outlabels(nq * topk, 0);

    vector<void *> datas = {(void *)indists.data(),
                            (void *)vmdists.data(), 
                            (void *)opSize.data(),
                            (void *)opflag.data(),
                            (void *)attrs.data(),
                            (void *)outdists.data(),
                            (void *)outlabels.data() };

    CREATE_NODEDEF(shapes, dataTypes, datas, 0);
    CpuKernelContext ctx(DEVICE);
    EXPECT_EQ(ctx.Init(nodeDef0.get()), 0);
    TopkFlatCpuKernel topK;
    double start = Utils::GetMillisecs();
    EXPECT_EQ(topK.Compute(ctx), 0);
    double end = Utils::GetMillisecs();
    std::cout << "topk compute cust:" << end - start << " ms" << std::endl;

    vector<float16_t> gtdists(nq * topk, 0.0);
    vector<int64_t> gtlabels(nq * topk, 0);
    Utils::InitGTDistAndLabel(asc, gtdists, gtlabels);
    for (auto qIdx = 0; qIdx < nq; qIdx++) {
        vector<float16_t> dists(blockNum * blockSize, 0.0);
        for (auto blockIdx = 0; blockIdx < blockNum; blockIdx++) {
            auto codeNum = blockSize;
            if (blockIdx == blockNum - 1) {
                codeNum = actualNum;
            }
            auto offset = Utils::GetOffset(shapes[0], {blockIdx, qIdx});
            memcpy(dists.data() + blockSize * blockIdx, indistsBak.data() + offset, codeNum * sizeof(float16_t));
        }

        vector<int64_t> labels(dists.size());
        iota(labels.begin(), labels.end(), 0);
        if (asc != 0) {
            sort(labels.begin(), labels.end(),
                 [&](size_t index1, size_t index2) { return dists[index1] < dists[index2]; });
        } else {
            sort(labels.begin(), labels.end(),
                 [&](size_t index1, size_t index2) { return dists[index1] > dists[index2]; });
        }
        auto offset = Utils::GetOffset(shapes[6], {qIdx});

        for (auto i = 0; i < topk; i++) {
            gtdists[offset + i] = dists[labels[i]];
            gtlabels[offset + i] = labels[i];
        }
    }

    ASSERT_TRUE(Utils::VerifySearchResult(nq, topk, blockSize, indistsBak.data(), outdists.data(),
                                          outlabels.data(), gtdists.data(), gtlabels.data()));
}
