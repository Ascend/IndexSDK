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


#include <arm_fp16.h>
#include <ascenddaemon/impl/AuxIndexStructures.h>
#include <ascenddaemon/utils/AscendUtils.h>
#include <ascenddaemon/utils/StaticUtils.h>
#include <ascenddaemon/impl/IndexIVFPQ.h>
#include <ascenddaemon/utils/AscendTensor.h>
#include <sys/time.h>
#include <gtest/gtest.h>
#include <vector>
#include <iostream>
#include <algorithm>
#include <limits>

namespace {
static unsigned long CurrentTime()
{
    const int secToMs = 1000000;
    struct timeval tv = {0};
    gettimeofday(&tv, nullptr);
    return (tv.tv_sec * secToMs + tv.tv_usec);  // get time in microsecond
}

TEST(TestIndexIVFPQ, Search)
{
    // create Index
    int numList = 8192;
    int dim = 512;
    int pqM = 64;
    int nprobe = 64;
    int bitsPerCode = 8;
    int numSubQuantizerCodes = ascend::utils::pow2(bitsPerCode);
    ascend::IndexIVFPQ index(numList, dim, pqM, bitsPerCode, nprobe);

    unsigned int seed = time(nullptr);
    srand(seed);
    // update coarseCents and pqCents
    std::vector<float16_t> coarseCentsVec(numList * dim);
    for (size_t i = 0; i < coarseCentsVec.size(); i++) {
        coarseCentsVec[i] = (float16_t)(rand() / ((float)RAND_MAX));
    }
    std::vector<float16_t> pqCentsVec(dim * numSubQuantizerCodes);
    for (size_t i = 0; i < pqCentsVec.size(); i++) {
        pqCentsVec[i] = (float16_t)(rand() / ((float)RAND_MAX));
    }

    ascend::AscendTensor<float16_t, 2> coarseCents(coarseCentsVec.data(),
        {numList, dim});
    ascend::AscendTensor<float16_t, 2> pqCents(pqCentsVec.data(),
        {pqM, dim / pqM * numSubQuantizerCodes});

    index.updateCoarseCentroidsData(coarseCents);
    index.updatePQCentroidsData(pqCents);
    std::cout << "update pq and coarse centroids success" << std::endl;

    // addVectors
    int listLength = 1700;  // num of vectors every list
    int idx = 0;
    std::vector<unsigned char> codes(listLength * pqM);
    for (size_t j = 0; j < codes.size(); j++) {
        codes[j] = (unsigned char)(255 * (rand() / (RAND_MAX + 0.0f)));
    }
    for (int i = 0; i < numList; i++) {
        std::vector<ascend::Index::idx_t> indices(listLength);
        for (size_t j = 0; j < indices.size(); j++) {
            indices[j] = idx;
            idx++;
        }
        index.addVectors(i, codes.data(), indices.data(), listLength);
    }
    std::cout << "add vectors success" << std::endl;

    // search
    const std::vector<int> numQueries{1, 31, 100};
    const int topK = 1000;
    int codesPerQuant = dim / pqM;
    for (auto qnum: numQueries) {
        std::vector<float16_t> queries(qnum * dim);
        std::vector<ascend::Index::idx_t> queryIds(qnum);
        for (int i = 0; i < qnum; i++) {
            int listId = rand() % (numList - 1);
            int subId = rand() % (index.getListLength(listId) - 1);
            queryIds[i] = *(index.getListIndices(listId).data() + subId);
            auto err = memcpy_s(queries.data() + i * dim, dim * sizeof(float16_t),
                                coarseCentsVec.data() + listId * dim, dim * sizeof(float16_t));
            if (err != EOK) {
                std::cout << "memcpy error, quit!" << std::endl;
                return;
            }

            for (int j = 0; j < pqM; j++) {
                unsigned char indexId = *(index.getListVectors(listId).data() + subId * pqM + j);
                for (int k = 0; k < codesPerQuant; k++) {
                    queries[i * dim + j * codesPerQuant + k] +=
                        pqCentsVec[j * numSubQuantizerCodes * codesPerQuant + indexId * codesPerQuant + k];
                }
            }
        }

        std::vector<float16_t> resDists(qnum * topK);
        std::vector<ascend::Index::idx_t> resIndices(qnum * topK);
        index.search(qnum, queries.data(), topK, resDists.data(), resIndices.data());

        std::cout << "----- start search, num_query is " << qnum << "-----" << std::endl;
        double cost = 0.0;
        double min = std::numeric_limits<double>::max();
        double max = 0.0;

        const int loopSearch = 10;
        for (int i = 0; i < loopSearch; i++) {
            unsigned long long start = CurrentTime();
            index.search(qnum, queries.data(), topK, resDists.data(), resIndices.data());
            unsigned long long end = CurrentTime();
            double cur_cost = (end - start) / 1000.0;
            min = std::min(min, cur_cost);
            max = std::max(max, cur_cost);
            cost += cur_cost;
        }

        for (int i = 0; i < qnum; i++) {
            int minDistIndex = std::min_element(resDists.begin() + i * topK, resDists.begin() + (i + 1) * topK)
                - (resDists.begin() + i * topK);
            std::cout << "query_Id: " << queryIds[i] << ", ";
            std::cout << "results_ID: " << resIndices[i * topK + minDistIndex] <<
                ", distance: " << resDists[i * topK + minDistIndex] << std::endl;
        }
        std::cout << "search cost time: " << cost / loopSearch << "ms in average, " << "max:"
            << max << "ms, min:" << min << "ms."<< std::endl;
    }
}

TEST(TestIndexIVFPQ, AddRemove)
{
    // create Index
    int numList = 8192;
    int dim = 512;
    int pqM = 64;
    int nprobe = 64;
    int bitsPerCode = 8;
    ascend::IndexIVFPQ indexIVFPQ(numList, dim, pqM, bitsPerCode, nprobe);

    int num = 100;  // num of vectors every list
    for (int i = 0; i < numList; i++) {
        std::vector<unsigned char> codes(num * pqM);
        for (size_t j = 0; j < codes.size(); j++) {
            codes[j] = (unsigned char)(255 * (rand() / (RAND_MAX + 0.0f)));
        }
        std::vector<ascend::Index::idx_t> indices(num);
        for (size_t j = 0; j < indices.size(); j++) {
            indices[j] = j;
        }
        indexIVFPQ.addVectors(i, codes.data(), indices.data(), num);

        auto& indexCodes = indexIVFPQ.getListVectors(i);
        auto& indexIndices = indexIVFPQ.getListIndices(i);
        EXPECT_EQ(memcmp(codes.data(), indexCodes.data(), indexCodes.size()), 0);
        EXPECT_EQ(memcmp(indices.data(), indexIndices.data(), indexIndices.size()), 0);
        EXPECT_EQ(indexIVFPQ.getListLength(i), num);
    }

    int rangeDel = 20;  // delete ids in ID range
    ascend::Index::idx_t idmin = static_cast<ascend::Index::idx_t>(50 * (rand() / (RAND_MAX + 0.0f)));  // idmin is 50
    ascend::IDSelectorRange idRange(idmin, idmin + rangeDel);
    auto xRange = indexIVFPQ.removeIds(idRange);
    EXPECT_EQ(xRange, rangeDel * numList);

    for (int i = 0; i < numList; i++) {
        EXPECT_EQ(indexIVFPQ.getListLength(i), num - rangeDel);
    }

    ascend::Index::idx_t idSet[5] = {3002, 3004, 3006, 3008, 3010};  // delete ids not existed in Index
    ascend::IDSelectorBatch idBatch(5, idSet);
    auto xSet = indexIVFPQ.removeIds(idBatch);
    EXPECT_EQ(xSet, 0);
    for (int i = 0; i < numList; i++) {
        EXPECT_EQ(indexIVFPQ.getListLength(i), num - rangeDel);
    }
}
} // namespace

int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);

    ascend::DeviceScope deviceScope;
    return RUN_ALL_TESTS();
}