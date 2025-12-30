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


#include <unistd.h>
#include <cstdlib>
#include <string>
#include <iostream>
#include <limits.h>
#include "Utils.h"

TEST(IndexVstar, singlesearch) {
    auto index = CreateIndexVstar();

    std::string codebook_path = "../codebook.bin";
    CreateCodebook(codebook_path);

    index->AddCodeBooks(codebook_path);

    std::string IndexPath = "../Vstar_128_mixed.index";
    
    std::vector<float> baseData = GenRandData(g_dim, 256);
    index->AddVectors(baseData);

    index->WriteIndex(IndexPath);
    index->LoadIndex(IndexPath);

    // cover if (codebookAdded)
    index->AddCodeBooks(codebook_path);

    // cover AddCodeBooks(const NpuIndexVStar *loadedIndex)
    auto index2 = CreateIndexVstar();
    index2->AddCodeBooks(index.get());

    std::vector<float> queryData = GenRandData(g_dim, 10);

    int batchSize = 5;
    int topK = 10;
    int gtK = 1;
    int gtDim = index->GetDim();
    int nq = 10;
    int nb = index->GetNTotal();
    std::vector<float> dists(nq * topK);
    std::vector<int64_t> labels(nq * topK);
    ascendSearchacc::SearchImplParams params(batchSize, queryData, topK, dists, labels);

    index->SetTopK(5);
    {
        // warm up
        printf("=================warm up start==================\n");
        for (size_t i = 0; i < 100; i++) {
            index->Search(params);
        }
        printf("================warm up end===================\n");
    }

    for (size_t i = 0; i < nq; i += batchSize) {
        index->Search(params);
    }

    // masksearch
    std::cout << "mask single search start" << std::endl;
    size_t maskDim = (nb + 7) / 8;
    std::vector<uint8_t>mask(nb * maskDim, 0);

    for (size_t i = 0; i < 100; i++) {
        index->Search(params, mask);
    }
    // deletevectors
    int64_t vector_id_to_delete = 1;
    index->DeleteVectors(vector_id_to_delete);

    std::vector<int64_t> ids_to_delete = {2, 3, 4};
    index->DeleteVectors(ids_to_delete);

    int64_t start_id = 5;
    int64_t end_id = 10;
    index->DeleteVectors(start_id, end_id);

    std::remove(codebook_path.c_str());
    std::remove(IndexPath.c_str());
}

TEST(IndexVstar, multisearch) {
    size_t multiIndexNum = 2;
    std::vector<std::shared_ptr<ascendSearchacc::NpuIndexVStar>> indexes;

    auto first_index = CreateIndexVstar();

    std::string codebook_path = "../codebook.bin";
    CreateCodebook(codebook_path);

    first_index->AddCodeBooks(codebook_path);

    std::vector<float> baseData = GenRandData(g_dim, 256);
    first_index->AddVectors(baseData);

    std::string IndexPath = "../multi.index";
    first_index->WriteIndex(IndexPath);

    indexes.push_back(first_index);

    std::vector<float> queryData = GenRandData(g_dim, 10);

    for (int i = 1; i < multiIndexNum; ++i) {
        auto index = CreateIndexVstar();
        index->LoadIndex(IndexPath, indexes[0].get());
        indexes.push_back(index);
    }

    int batchSize = 1;
    int topK = 10;
    int gtK = 1;
    int dim = first_index->GetDim();
    int nq = static_cast<int>(queryData.size() / dim);
    std::vector<float> dists(nq * topK * multiIndexNum);
    std::vector<int64_t> labels(nq * topK * multiIndexNum);
    ascendSearchacc::SearchImplParams params(batchSize, queryData, topK, dists, labels);

    std::vector<float> dists_merge(nq * topK);
    std::vector<int64_t> labels_merge(nq * topK);
    ascendSearchacc::SearchImplParams params_merge(batchSize, queryData, topK, dists_merge, labels_merge);

    std::cout << "After creating searcmImplParams..." << std::endl;

    std::vector<ascendSearchacc::NpuIndexVStar*> raw_indexes;
    for (const auto& index_ptr : indexes) {
        if (index_ptr) {
            raw_indexes.push_back(index_ptr.get());
        } else {
            std::cerr << "Warning: Found null shared_ptr in indexes, skipping." << std::endl;
        }
    }

    raw_indexes[0]->MultiSearch(raw_indexes, params, false);

    // merge
    raw_indexes[0]->MultiSearch(raw_indexes, params_merge, true);

    std::remove(codebook_path.c_str());
    std::remove(IndexPath.c_str());
}

TEST(IndexVstar, multisaerchwithmask) {
    size_t multiIndexNum = 2;
    std::vector<std::shared_ptr<ascendSearchacc::NpuIndexVStar>> indexes;

    auto first_index = CreateIndexVstar();

    std::string codebook_path = "../codebook.bin";
    CreateCodebook(codebook_path);

    first_index->AddCodeBooks(codebook_path);

    std::vector<float> baseData = GenRandData(g_dim, 256);
    first_index->AddVectors(baseData);

    std::string IndexPath = "../multi.index";
    first_index->WriteIndex(IndexPath);

    indexes.push_back(first_index);

    std::vector<float> queryData = GenRandData(g_dim, 10);

    for (int i = 1; i < multiIndexNum; ++i) {
        auto index = CreateIndexVstar();
        index->LoadIndex(IndexPath, indexes[0].get());
        indexes.push_back(index);
    }

    int batchSize = 1;
    int topK = 10;
    int gtK = 1;
    int dim = first_index->GetDim();
    int nb = first_index->GetNTotal();
    int nq = static_cast<int>(queryData.size() / dim);

    std::vector<float> dists(nq * topK * multiIndexNum);
    std::vector<int64_t> labels(nq * topK * multiIndexNum);
    ascendSearchacc::SearchImplParams params(batchSize, queryData, topK, dists, labels);

    std::vector<float> dists_merge(nq * topK);
    std::vector<int64_t> labels_merge(nq * topK);
    ascendSearchacc::SearchImplParams params_merge(batchSize, queryData, topK, dists_merge, labels_merge);

    size_t maskDim = (nb + 7) / 8;
    std::vector<uint8_t> mask(nb * maskDim, 0);

    std::cout << "After creating searcmImplParams..." << std::endl;

    std::vector<ascendSearchacc::NpuIndexVStar*> raw_indexes;
    for (const auto& index_ptr : indexes) {
        if (index_ptr) {
            raw_indexes.push_back(index_ptr.get());
        } else {
            std::cerr << "Warning: Found null shared_ptr in indexes, skipping." << std::endl;
        }
    }

    raw_indexes[0]->MultiSearch(raw_indexes, params, mask, false);

    // merge
    raw_indexes[0]->MultiSearch(raw_indexes, params_merge, mask, true);

    std::remove(codebook_path.c_str());
    std::remove(IndexPath.c_str());
}

TEST(IndexVstar, AddVectorsWithIds) {
    auto index = CreateIndexVstar();
    auto index2 = CreateIndexVstar();
    
    std::string codebook_path = "../codebook.bin";
    CreateCodebook(codebook_path);

    index->AddCodeBooks(codebook_path);
    index2->AddCodeBooks(codebook_path);

    std::vector<float> baseData = GenRandData(g_dim, 256);

    int dim = index->GetDim();
    size_t num_vectors = baseData.size() / dim;
    
    std::vector<int64_t> ids(num_vectors);
    for (size_t i = 0; i < num_vectors; i++) {
        ids[i] = 1000 + i;
    }
    index->AddVectorsWithIds(baseData, ids);
    
    std::string IndexPath = "../Vstar_mixed_with_ids.index";
    index->WriteIndex(IndexPath);
    index->LoadIndex(IndexPath);

    // search
    std::vector<float> queryData = GenRandData(g_dim, 10);

    int batchSize = 5;
    int topK = 10;
    int gtK = 1;
    int gtDim = index->GetDim();
    int nq = 10;
    int nb = index->GetNTotal();
    std::vector<float> dists(nq * topK);
    std::vector<int64_t> labels(nq * topK);
    ascendSearchacc::SearchImplParams params(batchSize, queryData, topK, dists, labels);

    {
        // warm up
        printf("=================warm up start==================\n");
        for (size_t i = 0; i < 100; i++) {
            index->Search(params);
        }
        printf("================warm up end===================\n");
    }

    for (size_t i = 0; i < nq; i += batchSize) {
        index->Search(params);
    }
    // searchwithmask
    size_t maskDim = (nb + 7) / 8;
    std::vector<uint8_t> mask(nb * maskDim, 0);
    index->SetTopK(5);
    // batchsize > 1；KQueryNum <= KModeMaxSearch
    for (size_t i = 0, j = 0; i < nq; i += batchSize, j++) {
        index->Search(params, mask);
    }

    index->Reset();
    std::remove(codebook_path.c_str());
    std::remove(IndexPath.c_str());
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}