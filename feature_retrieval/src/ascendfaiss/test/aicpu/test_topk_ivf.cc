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

#include "topk_ivf_cpu_kernel.h"
#include "utils.h"

using namespace std;
using namespace aicpu;
using namespace aicpu::unittest;

#define CREATE_NODEDEF(shapes, dataTypes, datas, idx)                               \
  auto nodeDef##idx = NodeDefBuilder::CreateNodeDef();                              \
  NodeDefBuilder(nodeDef##idx.get(), "TopkIvf", "TopkIvf")                          \
      .Input({"indists", dataTypes[0], shapes[0], datas[0]})                        \
      .Input({"vmdists", dataTypes[1], shapes[1], datas[1]})                        \
      .Input({"ids", dataTypes[2], shapes[2], datas[2]})                            \
      .Input({"size", dataTypes[3], shapes[3], datas[3]})                           \
      .Input({"opflag", dataTypes[4], shapes[4], datas[4]})                         \
      .Input({"attr", dataTypes[5], shapes[5], datas[5]})                           \
      .Output({"outdists", dataTypes[6], shapes[6], datas[6]})                      \
      .Output({"outlabels", dataTypes[7], shapes[7], datas[7]})

class TopKIvfTest : public testing::Test {};

TEST_F(TopKIvfTest, TestTopKIvf_1) {
    int64_t blockNum = 16;
    int64_t nq = 48;
    int64_t handleBatch = 4;
    int64_t blockSize = 8192;//262144;
    int64_t burstLen = 32;
    int64_t burstSize = blockSize / burstLen * 2;
    int64_t coreNum = 2;
    int64_t sizeAlign = 8;
    int64_t flagSize = 32;
    int64_t attrSize = 5;
    int64_t topk = 20;

    ASSERT_TRUE(sizeAlign >= handleBatch);

    // data types
    vector<DataType> dataTypes = { DT_FLOAT16,
                                   DT_FLOAT16,
                                   DT_INT64,
                                   DT_UINT32,
                                   DT_UINT16,
                                   DT_INT64,
                                   DT_FLOAT16,
                                   DT_INT64 };
    // shapes
    vector<vector<int64_t>> shapes = { {nq, blockNum, handleBatch, blockSize},
                                       {nq, blockNum, handleBatch, burstSize},
                                       {nq, blockNum, handleBatch},
                                       {nq, blockNum, sizeAlign},
                                       {nq, blockNum, coreNum, flagSize},
                                       {attrSize},
                                       {nq, topk},
                                       {nq, topk} };
    // data
    vector<float16_t> indists(nq * blockNum * handleBatch * blockSize, 0.0);
    Utils::SetRandomValue<float16_t>(indists.data(), indists.size());
    vector<float16_t> indistsBak(indists.begin(), indists.end());

    vector<float16_t> vmdists(nq * blockNum * handleBatch * burstSize, 0.0);
    for (auto qIdx = 0; qIdx < nq; ++qIdx) {
        for (auto blockIdx = 0; blockIdx < blockNum; ++blockIdx) {
            for (auto handleIdx = 0; handleIdx < handleBatch; ++handleIdx) {
                for (auto burstIdx = 0; burstIdx < burstSize / 2; ++burstIdx) {
                    int64_t offsetIndists = Utils::GetOffset(shapes[0], {qIdx, blockIdx, handleIdx, burstIdx * burstLen});
                    auto maxIdx = Utils::Argmax(indists.begin() + offsetIndists, indists.begin() + offsetIndists + burstLen);
                    int64_t offsetVmdists = Utils::GetOffset(shapes[1], {qIdx, blockIdx, handleIdx, burstIdx * 2});
                    vmdists[offsetVmdists] = indists[offsetIndists + maxIdx];
                    vmdists[offsetVmdists + 1] = 0; // no use
                }
            }
        }
    }

    size_t total = blockNum * handleBatch * blockSize;
    vector<int64_t> labels(total, 0);
    for (auto blockIdx = 0; blockIdx < blockNum; blockIdx++) {
        auto start = Utils::GetOffset(shapes[0], {0, blockIdx});
        auto end = Utils::GetOffset(shapes[0], {0, blockIdx + 1});
        for (auto idx = start; idx < end; ++idx) {
            labels[idx] = idx;
        }
    }
    vector<int64_t> ids(nq * blockNum * handleBatch);
    for (auto qIdx = 0; qIdx < nq; ++qIdx) {
        for (auto blockIdx = 0; blockIdx < blockNum; blockIdx++) {
            for (auto handleIdx = 0; handleIdx < handleBatch; ++handleIdx) {
                auto offsetLabel = Utils::GetOffset(shapes[0], {0, blockIdx, handleIdx});
                auto offsetIds = Utils::GetOffset(shapes[2], {qIdx, blockIdx, handleIdx});
                ids[offsetIds] = reinterpret_cast<int64_t>(labels.data() + offsetLabel);
            }
        }
    }

    vector<uint32_t> opSize(nq * blockNum * sizeAlign, 0);
    for (auto qIdx = 0; qIdx < nq; ++qIdx) {
        for (auto blockIdx = 0; blockIdx < blockNum; blockIdx++) {
            auto offset = Utils::GetOffset(shapes[3], {qIdx, blockIdx});
            for (auto handleIdx = 0; handleIdx < handleBatch; ++handleIdx) {
                opSize[offset + handleIdx] = blockSize;
            }
        }
    }

    vector<uint16_t> opflag(nq * blockNum * coreNum * flagSize, 0);
    for (auto qIdx = 0; qIdx < nq; ++qIdx) {
        for (auto blockIdx = 0; blockIdx < blockNum; blockIdx++) {
            for (auto coreIdx = 0; coreIdx < coreNum; coreIdx++) {
                auto offset = Utils::GetOffset(shapes[4], {qIdx, blockIdx, coreIdx});
                opflag[offset] = 1;
            }
        }
    }

    vector<int64_t> attrs = {0, topk, burstLen, blockNum, coreNum};

    vector<float16_t> outdists(nq * topk, 0.0);
    vector<int64_t> outlabels(nq * topk, 0);

    vector<void *> datas = {(void *)indists.data(),
                            (void *)vmdists.data(), 
                            (void *)ids.data(),
                            (void *)opSize.data(),
                            (void *)opflag.data(),
                            (void *)attrs.data(),
                            (void *)outdists.data(),
                            (void *)outlabels.data() };

    CREATE_NODEDEF(shapes, dataTypes, datas, 0);
    CpuKernelContext ctx(DEVICE);
    EXPECT_EQ(ctx.Init(nodeDef0.get()), 0);
    TopkIvfCpuKernel topK;
    topK.Compute(ctx);

    vector<float16_t> gtdists(nq * topk, 0.0);
    vector<int64_t> gtlabels(nq * topk, 0);
    for (auto qIdx = 0; qIdx < nq; qIdx++) {
        std::vector<float16_t> dists(total, 0);
        auto offsetDist = Utils::GetOffset(shapes[0], {qIdx});
        memcpy(dists.data(), indistsBak.data() + offsetDist, total * sizeof(float16_t));
        std::vector<int64_t> labels(dists.size());
        std::iota(labels.begin(), labels.end(), 0);
        std::sort(labels.begin(), labels.end(),
                  [&](size_t index1, size_t index2) { return dists[index1] > dists[index2]; });
        auto offset = Utils::GetOffset(shapes[6], {qIdx});
        for (auto i = 0; i < topk; i++) {
            gtdists[offset + i] = dists[labels[i]];
            gtlabels[offset + i] = labels[i];
        }
    }
    ASSERT_TRUE(Utils::VerifySearchResult(nq, topk, blockSize, indistsBak.data(), outdists.data(),
                                          outlabels.data(), gtdists.data(), gtlabels.data()));
}
