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

#include "topk_ivf_fuzzy_cpu_kernel.h"
#include "utils.h"

using namespace std;
using namespace aicpu;
using namespace aicpu::unittest;

#define CREATE_NODEDEF(shapes, dataTypes, datas, idx)                               \
  auto nodeDef##idx = NodeDefBuilder::CreateNodeDef();                              \
  NodeDefBuilder(nodeDef##idx.get(), "TopkIvfFuzzy", "TopkIvfFuzzy")                \
      .Input({"indists", dataTypes[0], shapes[0], datas[0]})                        \
      .Input({"vmdists", dataTypes[1], shapes[1], datas[1]})                        \
      .Input({"ids", dataTypes[2], shapes[2], datas[2]})                            \
      .Input({"size", dataTypes[3], shapes[3], datas[3]})                           \
      .Input({"opflag", dataTypes[4], shapes[4], datas[4]})                         \
      .Input({"attr", dataTypes[5], shapes[5], datas[5]})                           \
      .Output({"outdists", dataTypes[6], shapes[6], datas[6]})                      \
      .Output({"outlabels", dataTypes[7], shapes[7], datas[7]})                     \
      .Output({"popdists", dataTypes[8], shapes[8], datas[8]})                      \
      .Output({"poplabels", dataTypes[9], shapes[9], datas[9]})

class TopKIvfFuzzyTest : public testing::Test {};

TEST_F(TopKIvfFuzzyTest, TestTopKIvfFuzzy_1) {
    int64_t nq = 1;
    int64_t l3SegNum = 4;
    int64_t l3SegSize = 64;
    int64_t burstLen = 16;
    int64_t burstSize = l3SegNum * l3SegSize / burstLen * 2;
    int64_t coreNum = 2;
    int64_t flagSize = 16;
    int64_t attrSize = 8;
    int64_t topk = 10;
    int64_t kHeapRatio = 2;
    int64_t kBufRatio = 2;
    int64_t queryBatch = 2;

    // data types
    vector<DataType> dataTypes = { DT_FLOAT16,
                                   DT_FLOAT16,
                                   DT_INT64,
                                   DT_UINT32,
                                   DT_UINT16,
                                   DT_INT64,
                                   DT_FLOAT16,
                                   DT_INT64,
                                   DT_FLOAT16,
                                   DT_INT64 };
    // shapes
    vector<vector<int64_t>> shapes = { {nq, l3SegNum * l3SegSize},
                                       {nq, burstSize},
                                       {nq, l3SegNum},
                                       {nq, l3SegNum},
                                       {nq, coreNum, flagSize},
                                       {attrSize},
                                       {nq, topk},
                                       {nq, topk},
                                       {nq, topk / kBufRatio},
                                       {nq, topk / kBufRatio} };
    // data
    vector<float16_t> indists(nq * l3SegNum * l3SegSize);
    Utils::SetRandomValue<float16_t>(indists.data(), indists.size());

    vector<float16_t> vmdists(nq * burstSize);
    for (auto qIdx = 0; qIdx < nq; ++qIdx) {
        for (auto burstIdx = 0; burstIdx < burstSize / 2; ++burstIdx) {
            int64_t offsetIndists = Utils::GetOffset(shapes[0], {qIdx, burstIdx * burstLen});
            auto maxIdx = Utils::Argmax(indists.begin() + offsetIndists, indists.begin() + offsetIndists + burstLen);
            int64_t offsetVmdists = Utils::GetOffset(shapes[1], {qIdx, burstIdx * 2});
            vmdists[offsetVmdists] = indists[offsetIndists + maxIdx];
            vmdists[offsetVmdists + 1] = 0; // no use
        }
    }

    size_t total = l3SegNum * l3SegSize;
    vector<int64_t> labels(total, 0);
    for (auto idx = 0; idx < total; ++idx) {
        labels[idx] = idx;
    }
    vector<int64_t> ids(nq * l3SegNum);
    for (auto qIdx = 0; qIdx < nq; ++qIdx) {
        for (auto sIdx = 0; sIdx < l3SegNum; ++sIdx) {
            auto offsetIds = Utils::GetOffset(shapes[2], {qIdx, sIdx});
            ids[offsetIds] = reinterpret_cast<int64_t>(labels.data() + sIdx * l3SegSize);
        }
    }

    std::vector<uint32_t> opSize(nq * l3SegNum, l3SegSize);

    std::vector<uint16_t> opflag(nq * coreNum * flagSize, 1);

    std::vector<int64_t> attrs = {0, topk, burstLen, l3SegNum, l3SegSize, kHeapRatio, kBufRatio, queryBatch};

    std::vector<float16_t> outdists(nq * topk, 0.0);
    std::vector<int64_t> outlabels(nq * topk, 0);
    std::vector<float16_t> popdists(nq * (topk / kBufRatio), 0.0);
    std::vector<int64_t> poplabels(nq * (topk / kBufRatio), 0);

    vector<void *> datas = {(void *)indists.data(),
                            (void *)vmdists.data(),
                            (void *)ids.data(),
                            (void *)opSize.data(),
                            (void *)opflag.data(),
                            (void *)attrs.data(),
                            (void *)outdists.data(),
                            (void *)outlabels.data(),
                            (void *)popdists.data(),
                            (void *)poplabels.data() };

    CREATE_NODEDEF(shapes, dataTypes, datas, 0);
    CpuKernelContext ctx(DEVICE);
    EXPECT_EQ(ctx.Init(nodeDef0.get()), 0);
    TopkIvfFuzzyCpuKernel topK;
    EXPECT_EQ(topK.Compute(ctx), 0);

    // for print results:
    // printf("indists:\n");
    // Utils::PrintData(indists.data(), indists.size());
    // printf("vmdists:\n");
    // Utils::PrintData(vmdists.data(), vmdists.size());
    // printf("outdists:\n");
    // Utils::PrintData(outdists.data(), outdists.size());
    // printf("outlabels:\n");
    // Utils::PrintData(outlabels.data(), outlabels.size());
    // printf("popdists:\n");
    // Utils::PrintData(popdists.data(), popdists.size());
    // printf("poplabels:\n");
    // Utils::PrintData(poplabels.data(), poplabels.size());

    for (auto qIdx = 0; qIdx < nq; ++qIdx) {
        auto offsetIndists = Utils::GetOffset(shapes[0], {qIdx});
        auto offsetIndistsEnd = Utils::GetOffset(shapes[0], {qIdx + 1});
        auto offsetOutDists = Utils::GetOffset(shapes[6], {qIdx});
        for (auto i = 0; i < topk / kHeapRatio; ++i) {
            auto dist = outdists[offsetOutDists + i];
            int count = 0;
            for (auto j = offsetIndists; j < offsetIndistsEnd; ++j) {
                if (indists[j] > dist) {
                    ++count;
                }
            }
            ASSERT_TRUE(count < burstSize / 2);
        }
    }
}
