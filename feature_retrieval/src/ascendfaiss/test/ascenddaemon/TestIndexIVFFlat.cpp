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


#include <gtest/gtest.h>
#include <iostream>
#include <omp.h>
#include <cmath>

#include "ascenddaemon/impl/AuxIndexStructures.h"
#include "ascenddaemon/impl/IndexIVFFlat.h"

namespace {
TEST(TestIndexIVFFlat, Search)
{
    // create Index
    const int numList = 2048;
    const int dim = 128;
    const int nprobe = 64;
    ascend::IndexIVFFlat indexIVFFlat(numList, dim, nprobe);

    unsigned int seed = time(nullptr);
    srand(seed);
    std::vector<float16_t> coarseCentsVec(numList * dim);
#pragma omp parallel for
    for (size_t i = 0; i < coarseCentsVec.size(); i++) {
        coarseCentsVec[i] = (float16_t)(rand() / ((float)RAND_MAX));
    }

    ascend::AscendTensor<float16_t, 2> coarseCents(coarseCentsVec.data(), { numList, dim });
    indexIVFFlat.updateCoarseCentroidsData(coarseCents);
    std::cout << "update trained value and coarse centroids success" << std::endl;

    // addVectors
    int listLength = 100; // num of vectors every list
    int idx = 0;
    std::vector<float16_t> rawData(listLength * dim);
#pragma omp parallel for
    for (size_t j = 0; j < rawData.size(); ++j) {
        rawData[j] = (float16_t)(rand() / ((float)RAND_MAX));
    }

    for (int i = 0; i < numList; i++) {
        std::vector<ascend::Index::idx_t> indices(listLength);
        for (size_t j = 0; j < indices.size(); j++) {
            indices[j] = idx;
            idx++;
        }

        indexIVFFlat.addVectors(i, listLength, rawData.data(), indices.data());
    }
    std::cout << "add vectors success" << std::endl;

    // search
    const std::vector<int> numQueries { 1, 31, 50 };
    const int topK = 1000;
    for (auto qnum : numQueries) {
        std::vector<float16_t> queries(qnum * dim);
        std::vector<ascend::Index::idx_t> queryIds(qnum);

        for (int i = 0; i < qnum; i++) {
            int listId = rand() % (numList - 1);
            int subId = rand() % (indexIVFFlat.getListLength(listId) - 1);
            queryIds[i] = *(indexIVFFlat.getListIndices(listId).data() + subId);

            std::vector<unsigned char> reshapedData;
            indexIVFFlat.getListVectorsReshaped(listId, reshapedData);
            float16_t *pReshaped = reinterpret_cast<float16_t *>(reshapedData.data());
            for (int j = 0; j < dim; j++) {
                queries[i * dim + j] += *(pReshaped + subId * dim + j);
            }
        }

        std::vector<float16_t> resDists(qnum * topK);
        std::vector<ascend::Index::idx_t> resIndices(qnum * topK);
        indexIVFFlat.search(qnum, queries.data(), topK, resDists.data(), resIndices.data());

        std::cout << "----- start search, num_query is " << qnum << "-----" << std::endl;
        double cost = 0.0;
        double min = std::numeric_limits<double>::max();
        double max = 0.0;

        const int loopSearch = 100;
        for (int i = 0; i < loopSearch; i++) {
            double start = ascend::utils::getMillisecs();
            indexIVFFlat.search(qnum, queries.data(), topK, resDists.data(), resIndices.data());
            double end = ascend::utils::getMillisecs();
            double cur_cost = end - start;
            min = std::min(min, cur_cost);
            max = std::max(max, cur_cost);
            cost += cur_cost;
        }
      
        std::cout << "search cost time: " << cost / (loopSearch * qnum) << "ms in average, "
                  << "max:" << max / qnum << "ms, min:" << min / qnum << "ms." << std::endl;
    }
}

TEST(TestIndexIVFFlat, AddRemove)
{
    // create Index
    const int numList = 2048;
    const int dim = 128;
    const int nprobe = 64;
    ascend::IndexIVFFlat indexIVFFlat(numList, dim, nprobe);

    std::vector<float16_t> coarseCent(numList * dim);
    for (size_t i = 0; i < coarseCent.size(); i++) {
        coarseCent[i] = (float16_t)(i * 0.5f);
    }

    ascend::AscendTensor<float16_t, ascend::DIMS_2> coarseCentroidsData(coarseCent.data(), { numList, dim });
    indexIVFFlat.updateCoarseCentroidsData(coarseCentroidsData);

    double tas = ascend::utils::getMillisecs();
    int num = 100; // num of vectors every list
    for (int i = 0; i < numList; i++) {
        std::vector<float16_t> base(num * dim);
#pragma omp parallel for
        for (size_t j = 0; j < base.size(); ++j) {
            base[j] = (float16_t)(rand() / ((float)RAND_MAX));
        }

        std::vector<ascend::Index::idx_t> indices(num);
        for (size_t j = 0; j < indices.size(); j++) {
            indices[j] = j;
        }

        indexIVFFlat.addVectors(i, num, base.data(), indices.data());

        std::vector<unsigned char> indexCodes;
        indexIVFFlat.getListVectorsReshaped(i, indexCodes);
        float16_t *pCodes = reinterpret_cast<float16_t *>(indexCodes.data());
        auto &indexIndices = indexIVFFlat.getListIndices(i);

        ASSERT_EQ(memcmp(base.data(), pCodes, indexIndices.size()), 0);
        ASSERT_EQ(memcmp(indices.data(), indexIndices.data(), indexIndices.size()), 0);
        ASSERT_EQ(indexIVFFlat.getListLength(i), num);
    }
    std::cout << "add cost: " << ascend::utils::getMillisecs() - tas << "ms" << std::endl;

    // delete range
    double trrs = ascend::utils::getMillisecs();
    const int rangeDel = 20;                                       // delete ids in ID range
    ascend::Index::idx_t idmin = static_cast<ascend::Index::idx_t>(50 * (rand() / (RAND_MAX + 0.0f))); // idmin is 50
    ascend::IDSelectorRange idRange(idmin, idmin + rangeDel);
    auto xRange = indexIVFFlat.removeIds(idRange);
    EXPECT_EQ(xRange, rangeDel * numList);
    std::cout << "remove range cost: " << ascend::utils::getMillisecs() - trrs << "ms" << std::endl;

    for (int i = 0; i < numList; i++) {
        EXPECT_EQ(indexIVFFlat.getListLength(i), num - rangeDel);
    }

    // delete batch
    double trbs = ascend::utils::getMillisecs();
    ascend::Index::idx_t idSet[5] = {3002, 3004, 3006, 3008, 3010}; // delete ids not existed in Index
    ascend::IDSelectorBatch idBatch(5, idSet);
    auto xSet = indexIVFFlat.removeIds(idBatch);
    EXPECT_EQ(xSet, 0);
    std::cout << "remove batch cost: " << ascend::utils::getMillisecs() - trbs << "ms" << std::endl;

    for (int i = 0; i < numList; i++) {
        EXPECT_EQ(indexIVFFlat.getListLength(i), num - rangeDel);
    }
}
} // namespace

int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);

    ascend::DeviceScope deviceScope;
    return RUN_ALL_TESTS();
}