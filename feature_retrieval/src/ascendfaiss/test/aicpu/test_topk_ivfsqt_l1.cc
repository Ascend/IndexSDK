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

#include "topk_ivfsqt_l1_cpu_kernel.h"
#include "utils.h"

using namespace std;
using namespace aicpu;
using namespace aicpu::unittest;

#define CREATE_NODEDEF(shapes, dataTypes, datas, idx)                               \
  auto nodeDef##idx = NodeDefBuilder::CreateNodeDef();                              \
  NodeDefBuilder(nodeDef##idx.get(), "TopkIvfsqtL1", "TopkIvfsqtL1")                \
      .Input({"indists", dataTypes[0], shapes[0], datas[0]})                        \
      .Input({"vmdists", dataTypes[1], shapes[1], datas[1]})                        \
      .Input({"opflag", dataTypes[2], shapes[2], datas[2]})                         \
      .Input({"attr", dataTypes[3], shapes[3], datas[3]})                           \
      .Input({"queryIn", dataTypes[4], shapes[4], datas[4]})                        \
      .Input({"compressIndex", dataTypes[5], shapes[5], datas[5]})                  \
      .Input({"compressValue", dataTypes[6], shapes[6], datas[6]})                  \
      .Output({"outdists", dataTypes[7], shapes[7], datas[7]})                      \
      .Output({"outlabels", dataTypes[8], shapes[8], datas[8]})                     \
      .Output({"queryOut", dataTypes[9], shapes[9], datas[9]})

class TopKIvfsqtL1Test : public testing::Test {};

TEST_F(TopKIvfsqtL1Test, TopKIvfsqtL1Test_1) {
    int64_t nq = 2048;
    int64_t numLists = 16384;
    int64_t burstLen = 64;
    int64_t burstSize = numLists / burstLen * 2;
    int64_t coreNum = 8;
    int64_t flagSize = 16;
    int64_t attrSize = 6;
    int64_t topk = 48;
    int64_t queryBatch = 32;

    int64_t dimIn = 256;
    int64_t dimOut = 64;
    int64_t ratio = dimIn / dimOut;

    // data types
    vector<DataType> dataTypes = { DT_FLOAT16,            // indists
                                   DT_FLOAT16,            // vmdists
                                   DT_UINT16,             // opflag
                                   DT_INT64,              // attr
                                   DT_FLOAT16,            // queryIn
                                   DT_INT32,              // compressIndex
                                   DT_FLOAT,              // compressValue
                                   DT_FLOAT16,            // outdists
                                   DT_UINT16,             // outlabels
                                   DT_FLOAT16 };          // queryOut
    // shapes
    vector<vector<int64_t>> shapes = { {nq, numLists},
                                       {nq, burstSize},
                                       {nq, coreNum, flagSize},
                                       {attrSize},
                                       {nq, dimIn},
                                       {dimOut, ratio},
                                       {ratio, dimOut},
                                       {nq, topk},
                                       {nq, topk},
                                       {nq, dimOut}};
    // data
    vector<float16_t> indists(nq * numLists);
    Utils::SetRandomValue<float16_t>(indists.data(), indists.size());

    vector<float16_t> vmdists(nq * burstSize);
    uint16_t *vmlabel = reinterpret_cast<uint16_t *>(vmdists.data());
    for (auto qIdx = 0; qIdx < nq; ++qIdx) {
        for (auto burstIdx = 0; burstIdx < burstSize / 2; ++burstIdx) {
            int64_t offsetIndists = Utils::GetOffset(shapes[0], {qIdx, burstIdx * burstLen});
            auto minIdx = Utils::Argmin(indists.begin() + offsetIndists, indists.begin() + offsetIndists + burstLen);
            int64_t offsetVmdists = Utils::GetOffset(shapes[1], {qIdx, burstIdx * 2});
            vmdists[offsetVmdists] = indists[offsetIndists + minIdx];
            vmlabel[offsetVmdists + 1] = minIdx; // no use
        }
    }

    std::vector<uint16_t> opflag(nq * coreNum * flagSize, 1);

    std::vector<int64_t> attrs = {1, topk, burstLen, numLists, queryBatch, 1};

    vector<float16_t> queryIn(nq * dimIn);
    Utils::SetRandomValue<float16_t>(queryIn.data(), queryIn.size());

    vector<int32_t> compressIndex(dimOut * ratio);
    std::iota(compressIndex.begin(), compressIndex.end(), 0);

    vector<float> compressValue(ratio * dimOut);
    Utils::SetRandomValue<float>(compressValue.data(), compressValue.size());

    std::vector<float16_t> outdists(nq * topk, 0.0);
    std::vector<uint16_t> outlabels(nq * topk, 0);
    vector<float16_t> queryOut(nq * dimOut);

    vector<float16_t> indistsBak = indists;

    vector<void *> datas = {(void *)indists.data(),
                            (void *)vmdists.data(),
                            (void *)opflag.data(),
                            (void *)attrs.data(),
                            (void *)queryIn.data(),
                            (void *)compressIndex.data(), 
                            (void *)compressValue.data(),
                            (void *)outdists.data(),
                            (void *)outlabels.data(),
                            (void *)queryOut.data() };

    CREATE_NODEDEF(shapes, dataTypes, datas, 0);
    CpuKernelContext ctx(DEVICE);
    EXPECT_EQ(ctx.Init(nodeDef0.get()), 0);
    TopkIvfsqtL1CpuKernel topK;

    double start = Utils::GetMillisecs();
    EXPECT_EQ(topK.Compute(ctx), 0);
    double end = Utils::GetMillisecs();
    std::cout << "compute time cost: " << end - start << " ms" << std::endl;

    auto checkVectroCmp = [](std::vector<float16_t> &dist1, std::vector<float16_t> &dist2) {
        if (dist1.size() != dist2.size()) {
            return false;
        }
        std::sort(dist1.begin(), dist1.end());
        std::sort(dist2.begin(), dist2.end());
        size_t size = dist1.size();
        const float eps = 0.001;
        for (size_t i = 0; i < size; i++) {
            if (std::fabs(dist1[i] - dist2[i]) > eps) {
                return false;
            }
        }
        return true;
    };

    for (int i = 0; i < nq; i++) {
        std::cout << "indists [" << i << "]: top" << topk << std::endl;
        int start = i * numLists;
        int end = (i + 1) * numLists;
        std::vector<float16_t> indistsI(indistsBak.begin() + start, indistsBak.begin() + end);
        std::sort(indistsI.begin(), indistsI.end(), std::less<float16_t>());
        for (int j = 0; j < topk; j++) {
            std::cout << indistsI[j] << " ";
        }
        std::cout << std::endl;
        std::cout << "outdists [" << i << "]: top" << topk << std::endl;
        std::vector<float16_t> indistsTopk(indistsI.begin(), indistsI.begin() + topk);

        start = i * topk;
        end = (i + 1) * topk;
        std::vector<float16_t> outdistsTopk(outdists.begin() + start, outdists.begin() + end);
        for (int j = 0; j < topk; j++) {
            std::cout << outdistsTopk[j] << " ";
        }
        std::cout << std::endl;
        EXPECT_TRUE(checkVectroCmp(indistsTopk, outdistsTopk));
        std::cout << std::endl;
    }

    const float ratioDenominator = 1.0 / ratio;
    const float epsilon = 1e-6;
    vector<float16_t> queryCompute(nq * dimOut);
    for (int i = 0; i < nq; i++) {
        const float16_t *xSlice = queryIn.data() + i * dimIn;
        float16_t *resSlice = queryCompute.data() + i * dimOut;
        float denominator = 0;
        for (int j = 0; j < dimOut; ++j) {
            for (int k = 0; k < ratio; ++k) {
                resSlice[j] +=*(xSlice + compressIndex[j * ratio + k]) * (1 - compressValue[k * dimOut + j]);
            }
            resSlice[j] *= ratioDenominator;
            denominator += (resSlice[j] * resSlice[j]);
        }
        if (denominator > epsilon) {
            const float invNr = 1.0 / sqrt(denominator);
            for (int j = 0; j < dimOut; j++) {
                resSlice[j] *= invNr;
            }
        }
    }

    for (int i = 0; i < nq; i++) {
        for (int j = 0; j < dimOut; j++) {
            EXPECT_TRUE(fabs(queryOut[i * dimOut + j] - queryCompute[i * dimOut + j]) < 1e-3);
        }
    }
}
