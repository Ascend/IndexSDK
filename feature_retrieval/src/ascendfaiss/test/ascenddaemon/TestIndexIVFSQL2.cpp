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


#include <ascenddaemon/impl/IndexIVFSQL2.h>
#include <ascenddaemon/impl/AuxIndexStructures.h>
#include <gtest/gtest.h>
#include <iostream>
#include <omp.h>
#include <cmath>

namespace {
TEST(TestIndexIVFSQL2, Search)
{
    // create Index
    const float min = 1.0f;
    const float diff = 5.0f;
    const int numList = 16384;
    const int dim = 128;
    const int nprobe = 64;
    ascend::IndexIVFSQL2 indexIVFSQ(numList, dim, true, nprobe);

    unsigned int seed = time(nullptr);
    srand(seed);
    std::vector<float16_t> coarseCentsVec(numList * dim);
#pragma omp parallel for
    for (size_t i = 0; i < coarseCentsVec.size(); i++) {
        coarseCentsVec[i] = (float16_t)fmod(((float)rand()) + min, diff + min);
    }

    std::vector<float16_t> vmin(dim);
    std::vector<float16_t> vdiff(dim);
    for (int i = 0; i < dim; i++) {
        vmin[i] = (float16_t)fmod((float)rand(), min);
        vdiff[i] = (float16_t)fmod((float)rand(), diff);
    }

    ascend::AscendTensor<float16_t, 2> coarseCents(coarseCentsVec.data(), { numList, dim });
    ascend::AscendTensor<float16_t, ascend::DIMS_1> trainedMin(vmin.data(), { dim });
    ascend::AscendTensor<float16_t, ascend::DIMS_1> trainedDiff(vdiff.data(), { dim });
    indexIVFSQ.updateCoarseCentroidsData(coarseCents);
    indexIVFSQ.updateTrainedValue(trainedMin, trainedDiff);
    std::cout << "update trained value and coarse centroids success" << std::endl;

    // addVectors
    int listLength = 1700; // num of vectors every list
    int idx = 0;
    std::vector<unsigned char> codes(listLength * dim);
    std::vector<float> precomp(listLength, 0);
#pragma omp parallel for
    for (size_t j = 0; j < codes.size(); j++) {
        codes[j] = (unsigned char)(255 * (rand() / (RAND_MAX + 0.0f)));
    }

#pragma omp parallel for
    for (size_t i = 0; i < precomp.size(); i++) {
        int offset = i * dim;
        for (int j = 0; j < dim; j++) {
            float val = (codes[offset + j] + 0.5f) * (float)vdiff[j] / 255.0f + (float)vmin[j];
            precomp[i] += val * val;
        }
    }

    for (int i = 0; i < numList; i++) {
        std::vector<ascend::Index::idx_t> indices(listLength);
        for (size_t j = 0; j < indices.size(); j++) {
            indices[j] = idx;
            idx++;
        }

        indexIVFSQ.addVectors(i, listLength, codes.data(), indices.data(), precomp.data());
    }
    std::cout << "add vectors success" << std::endl;

    // search
    const std::vector<int> numQueries { 1, 127, 300 };
    const int topK = 1000;
    for (auto qnum : numQueries) {
        std::vector<float16_t> queries(qnum * dim);
        std::vector<ascend::Index::idx_t> queryIds(qnum);
        for (int i = 0; i < qnum; i++) {
            int listId = rand() % (numList - 1);
            int subId = rand() % (indexIVFSQ.getListLength(listId) - 1);
            queryIds[i] = *(indexIVFSQ.getListIndices(listId).data() + subId);
            auto err = memcpy_s(queries.data() + i * dim, dim * sizeof(float16_t), coarseCentsVec.data() + listId * dim,
                dim * sizeof(float16_t));
            ASSERT_EQ(err, EOK);

            std::vector<unsigned char> reshapedData;
            indexIVFSQ.getListVectorsReshaped(listId, reshapedData);

            for (int j = 0; j < dim; j++) {
                float16_t val = (float16_t)((reshapedData[subId * dim + j] + 0.5f) * vdiff[j] / 255.0f + vmin[j]);
                queries[i * dim + j] += val;
            }
        }

        std::vector<float16_t> resDists(qnum * topK);
        std::vector<ascend::Index::idx_t> resIndices(qnum * topK);
        indexIVFSQ.search(qnum, queries.data(), topK, resDists.data(), resIndices.data());

        std::cout << "----- start search, num_query is " << qnum << "-----" << std::endl;
        double cost = 0.0;
        double min = std::numeric_limits<double>::max();
        double max = 0.0;

        const int loopSearch = 100;
        for (int i = 0; i < loopSearch; i++) {
            double start = ascend::utils::getMillisecs();
            indexIVFSQ.search(qnum, queries.data(), topK, resDists.data(), resIndices.data());
            double end = ascend::utils::getMillisecs();
            double cur_cost = end - start;
            min = std::min(min, cur_cost);
            max = std::max(max, cur_cost);
            cost += cur_cost;
        }

        for (int i = 0; i < qnum; i++) {
            std::cout << "query: " << queryIds[i] << ", results_id: " << resIndices[i * topK] << ", distance : " <<
                resDists[i * topK] << std::endl;

            EXPECT_EQ(queryIds[i], resIndices[i * topK]);
        }
        std::cout << "search cost time: " << cost / (loopSearch * qnum) << "ms in average, "
                  << "max:" << max / qnum << "ms, min:" << min / qnum << "ms." << std::endl;
    }
}

TEST(TestIndexIVFSQL2, AddRemove)
{
    // create Index
    const int numList = 16384;
    const int dim = 128;
    const int nprobe = 64;
    ascend::IndexIVFSQL2 indexIVFSQ(numList, dim, true, nprobe);

    std::vector<float16_t> coarseCent(numList * dim);
    for (size_t i = 0; i < coarseCent.size(); i++) {
        coarseCent[i] = (float16_t)(i * 0.5f);
    }

    std::vector<float16_t> vmin(dim);
    std::vector<float16_t> vdiff(dim);
    for (int i = 0; i < dim; i++) {
        vmin[i] = 1.0f;
        vdiff[i] = 50.0f;
    }

    ascend::AscendTensor<float16_t, ascend::DIMS_2> coarseCentroidsData(coarseCent.data(), { numList, dim });
    ascend::AscendTensor<float16_t, ascend::DIMS_1> trainedMin(vmin.data(), { dim });
    ascend::AscendTensor<float16_t, ascend::DIMS_1> trainedDiff(vdiff.data(), { dim });
    indexIVFSQ.updateCoarseCentroidsData(coarseCentroidsData);
    indexIVFSQ.updateTrainedValue(trainedMin, trainedDiff);

    double tas = ascend::utils::getMillisecs();
    int num = 100; // num of vectors every list
    for (int i = 0; i < numList; i++) {
        std::vector<uint8_t> codes(num * dim);
        for (size_t j = 0; j < codes.size(); j++) {
            codes[j] = (uint8_t)(255 * (rand() / (RAND_MAX + 0.0f)));
        }
        std::vector<ascend::Index::idx_t> indices(num);
        std::vector<float> precomp(num);
        for (size_t j = 0; j < indices.size(); j++) {
            indices[j] = j;
            precomp[j] = j * 1.0f;
        }

        indexIVFSQ.addVectors(i, num, codes.data(), indices.data(), precomp.data());

        std::vector<unsigned char> indexCodes;
        indexIVFSQ.getListVectorsReshaped(i, indexCodes);
        auto &indexIndices = indexIVFSQ.getListIndices(i);
        auto &indexPre = indexIVFSQ.getListPrecompute(i);
        ASSERT_EQ(memcmp(codes.data(), indexCodes.data(), indexCodes.size()), 0);
        ASSERT_EQ(memcmp(indices.data(), indexIndices.data(), indexIndices.size()), 0);
        ASSERT_EQ(memcmp(precomp.data(), indexPre.data(), indexPre.size()), 0);
        ASSERT_EQ(indexIVFSQ.getListLength(i), num);
    }
    std::cout << "add cost: " << ascend::utils::getMillisecs() - tas << "ms" << std::endl;

    double trrs = ascend::utils::getMillisecs();
    const int rangeDel = 20;                                       // delete ids in ID range
    ascend::Index::idx_t idmin = static_cast<ascend::Index::idx_t>(50 * (rand() / (RAND_MAX + 0.0f))); // idmin is 50
    ascend::IDSelectorRange idRange(idmin, idmin + rangeDel);
    auto xRange = indexIVFSQ.removeIds(idRange);
    EXPECT_EQ(xRange, rangeDel * numList);
    std::cout << "remove range cost: " << ascend::utils::getMillisecs() - trrs << "ms" << std::endl;

    for (int i = 0; i < numList; i++) {
        EXPECT_EQ(indexIVFSQ.getListLength(i), num - rangeDel);
    }

    double trbs = ascend::utils::getMillisecs();
    ascend::Index::idx_t idSet[5] = {3002, 3004, 3006, 3008, 3010}; // delete ids not existed in Index
    ascend::IDSelectorBatch idBatch(5, idSet);
    auto xSet = indexIVFSQ.removeIds(idBatch);
    EXPECT_EQ(xSet, 0);
    std::cout << "remove batch cost: " << ascend::utils::getMillisecs() - trbs << "ms" << std::endl;

    for (int i = 0; i < numList; i++) {
        EXPECT_EQ(indexIVFSQ.getListLength(i), num - rangeDel);
    }
}
} // namespace

int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);

    ascend::DeviceScope deviceScope;
    return RUN_ALL_TESTS();
}