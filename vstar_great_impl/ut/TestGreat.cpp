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

TEST(IndexGreat, AKMode) {
    int dim;
    int nTotal;
    auto index = CreateIndexGreat("AKMode");

    std::string codebook_path = "../codebook.bin";
    CreateCodebook(codebook_path);
    index->AddCodeBooks(codebook_path);

    std::string AModeIndexPath = "../great_vstar_128_mixed.index";
    std::string KModeIndexPath = "../great_hnsw_128_mixed.index";

    std::vector<float> baseData = GenRandData(g_dim, 256);
    nTotal = index->GetNTotal();

    index->AddVectors(baseData);

    index->WriteIndex(AModeIndexPath, KModeIndexPath);
    
    std::vector<float> queryData = GenRandData(g_dim, 10);

    // batchsize > 1；KQueryNum <= KModeMaxSearch
    std::cout << "---------- start searching----------" << std::endl;
    int batchSize = 5;
    int topK = 10;
    int gtK = 1;
    int gtDim = index->GetDim();
    int nb = index->GetNTotal();
    size_t nq = queryData.size();
    std::vector<float> dists(nq * topK);
    std::vector<int64_t> labels(nq * topK);

    // setting search hyperparameters
    ascendSearchacc::IndexGreatSearchParams params;
    params.mode = "AKMode";
    params.nProbeL1 = 72;
    params.nProbeL2 = 64;
    params.l3SegmentNum = 512;
    params.ef = 150;
    index->SetSearchParams(params);

    // batchsize > 1；KQueryNum <= KModeMaxSearch
    ascendSearchacc::SearchImplParams searchparams1(batchSize, queryData, topK, dists, labels);
    auto greatSearchParam = index->GetSearchParams();
    for (size_t i = 0, j = 0; i < nq; i += batchSize, j++) {
        index->Search(searchparams1);
    }

    // batchsize = 1
    ascendSearchacc::SearchImplParams searchparams2(1, queryData, topK, dists, labels);
    for (size_t i = 0, j = 0; i < nq; i += 1, j++) {
        index->Search(searchparams2);
    }

    // batchsize > 1；KQueryNum > KModeMaxSearch
    ascendSearchacc::SearchImplParams searchparams3(521, queryData, topK, dists, labels);
    for (size_t i = 0, j = 0; i < nq; i += 521, j++) {
        index->Search(searchparams3);
    }

    std::remove(codebook_path.c_str());
    std::remove(AModeIndexPath.c_str());
    std::remove(KModeIndexPath.c_str());
}

TEST(IndexGreat, AKMode_SearchWithMask) {
    int dim;
    int nTotal;
    auto index = CreateIndexGreat("AKMode");

    std::string codebook_path = "../codebook.bin";
    CreateCodebook(codebook_path);
    index->AddCodeBooks(codebook_path);

    std::vector<float> baseData = GenRandData(g_dim, 256);
    nTotal = index->GetNTotal();

    index->AddVectors(baseData);

    std::vector<float> queryData = GenRandData(g_dim, 10);

    // batchsize > 1；KQueryNum <= KModeMaxSearch
    std::cout << "---------- start searching----------" << std::endl;
    int batchSize = 5;
    int topK = 10;
    int gtK = 1;
    int gtDim = index->GetDim();
    int nb = index->GetNTotal();
    size_t nq = queryData.size();
    std::vector<float> dists(nq * topK);
    std::vector<int64_t> labels(nq * topK);

    // setting search hyperparameters
    ascendSearchacc::IndexGreatSearchParams params;
    params.mode = "AKMode";
    params.nProbeL1 = 72;
    params.nProbeL2 = 64;
    params.l3SegmentNum = 512;
    params.ef = 150;
    index->SetSearchParams(params);

    size_t maskDim = (nb + 7) / 8;
    std::vector<uint8_t> mask1(nb * maskDim, 0);
    std::vector<uint8_t> mask2(521 * maskDim, 0);

    // batchsize > 1；KQueryNum <= KModeMaxSearch
    ascendSearchacc::SearchImplParams searchparams1(batchSize, queryData, topK, dists, labels);
    for (size_t i = 0, j = 0; i < nq; i += batchSize, j++) {
        index->SearchWithMask(searchparams1, mask1);
    }

    // batchsize = 1 KQueryNum <= KModeMaxSearch
    ascendSearchacc::SearchImplParams searchparams2(1, queryData, topK, dists, labels);
    for (size_t i = 0, j = 0; i < nq; i += 1, j++) {
        index->SearchWithMask(searchparams2, mask1);
    }

    // batchsize > 1；KQueryNum > KModeMaxSearch
    ascendSearchacc::SearchImplParams searchparams3(521, queryData, topK, dists, labels);
    for (size_t i = 0, j = 0; i < nq; i += 521, j++) {
        index->SearchWithMask(searchparams3, mask2);
    }

    index->Reset();
    std::remove(codebook_path.c_str());
}

TEST(IndexGreat, KMode) {
    int nb;
    int dim;
    auto index = CreateIndexGreat("KMode");
    nb = index->GetNTotal();

    std::string KModeIndexPath = "../kmode.index";
    
    std::vector<float> baseData = GenRandData(g_dim, 256);

    index->AddVectors(baseData);
    dim = index->GetDim();
    nb = index->GetNTotal();

    index->WriteIndex(KModeIndexPath);

    std::vector<float> queryData = GenRandData(g_dim, 10);

    std::cout << "---------- start searching----------" << std::endl;
    int batchSize = 5;
    int topK = 10;
    int gtK = 1;
    int gtDim = 128;
    size_t nq = queryData.size();
    std::vector<float> dists(nq * topK);
    std::vector<int64_t> labels(nq * topK);

    // setting search hyperparameters
    ascendSearchacc::IndexGreatSearchParams params;
    params.mode = "KMode";
    params.nProbeL1 = 72;
    params.nProbeL2 = 64;
    params.l3SegmentNum = 512;
    params.ef = 150;
    index->SetSearchParams(params);

    // KQueryNum <= KModeMaxSearch
    ascendSearchacc::SearchImplParams searchparams1(batchSize, queryData, topK, dists, labels);
    auto greatSearchParam = index->GetSearchParams();

    {
        // warm up
        printf("=================warm up start==================\n");
        for (size_t i = 0; i < 100; i++) {
            index->Search(searchparams1);
        }
        printf("================warm up end===================\n");
    }

    for (size_t i = 0, j = 0; i < nq; i += batchSize, j++) {
        index->Search(searchparams1);
    }

    // KQueryNum > KModeMaxSearch
    ascendSearchacc::SearchImplParams searchparams2(261, queryData, topK, dists, labels);
    for (size_t i = 0, j = 0; i < nq; i += 261, j++) {
        index->Search(searchparams2);
    }

    index->Reset();
    std::remove(KModeIndexPath.c_str());
}

TEST(IndexGreat, KMode_SearchWithMask) {
    auto index = CreateIndexGreat("KMode");

    // 加载底库数据
    std::vector<float> baseData = GenRandData(g_dim, 256);

    // 将基础数据添加到索引中
    index->AddVectors(baseData);

    // 加载查询数据
    std::vector<float> queryData = GenRandData(g_dim, 10);

    int batchSize = 5;
    int topK = 10;
    int gtK = 1;
    int gtDim = index->GetDim();
    int nb = index->GetNTotal();
    size_t nq = queryData.size();
    std::vector<float> dists(nq * topK);
    std::vector<int64_t> labels(nq * topK);

    // setting search hyperparameters
    ascendSearchacc::IndexGreatSearchParams params;
    params.mode = "KMode";
    params.nProbeL1 = 72;
    params.nProbeL2 = 64;
    params.l3SegmentNum = 512;
    params.ef = 150;
    index->SetSearchParams(params);

    // masksearch
    size_t maskDim = (nb + 7) / 8;
    std::vector<uint8_t> mask1(nb * maskDim, 0);
    std::vector<uint8_t> mask2(521 * maskDim, 0);

    // KQueryNum <= KModeMaxSearch =======================================================================
    ascendSearchacc::SearchImplParams searchparams1(batchSize, queryData, topK, dists, labels);
    for (size_t i = 0, j = 0; i < nq; i += batchSize, j++) {
        index->SearchWithMask(searchparams1, mask1);
    }

    // KQueryNum > KModeMaxSearch =======================================================================
    ascendSearchacc::SearchImplParams searchparams2(521, queryData, topK, dists, labels);
    for (size_t i = 0, j = 0; i < nq; i += 521, j++) {
        index->SearchWithMask(searchparams2, mask2);
    }

    index->Reset();
}


TEST(IndexGreat, KMode_AddVectorsWithIds) {
    // KMode
    auto index = CreateIndexGreat("KMode");

    std::vector<float> baseData = GenRandData(g_dim, 256);
    int dim = index->GetDim();
    size_t num_vectors = baseData.size() / dim;
    
    std::vector<int64_t> ids(num_vectors);
    for (size_t i = 0; i < num_vectors; ++i) {
        ids[i] = 1000 + i;
    }

    index->AddVectorsWithIds(baseData, ids);
}

TEST(IndexGreat, AKMode_AddVectorWithIds) {
    // AKMode
    auto index = CreateIndexGreat("AKMode");

    std::vector<float> baseData = GenRandData(g_dim, 256);
    int dim = index->GetDim();
    size_t num_vectors = baseData.size() / dim;
    
    std::vector<int64_t> ids(num_vectors);
    for (size_t i = 0; i < num_vectors; ++i) {
        ids[i] = 1000 + i;
    }

    std::string codebook_path = "../codebook.bin";
    CreateCodebook(codebook_path);
    index->AddCodeBooks(codebook_path);
    index->AddVectorsWithIds(baseData, ids);

    std::vector<float> queryData = GenRandData(g_dim, 10);

    int batchSize = 5;
    int topK = 10;
    int gtK = 1;
    int gtDim = index->GetDim();
    int nq = 10;
    int nb = index->GetNTotal();

    std::vector<float> dists(nq * topK);
    std::vector<int64_t> labels(nq * topK);

    // setting search hyperparameters
    ascendSearchacc::IndexGreatSearchParams params;
    params.mode = "AKMode";
    params.nProbeL1 = 72;
    params.nProbeL2 = 64;
    params.l3SegmentNum = 512;
    params.ef = 150;
    index->SetSearchParams(params);

    ascendSearchacc::SearchImplParams searchparams(batchSize, queryData, topK, dists, labels);

    for (size_t i = 0, j = 0; i < nq; i += batchSize, j++) {
        index->Search(searchparams);
    }

    // masksearch
    size_t maskDim = (nb + 7) / 8;
    std::vector<uint8_t> mask(nb * maskDim, 0);
    // KQueryNum <= KModeMaxSearch
    for (size_t i = 0, j = 0; i < nq; i += batchSize, j++) {
        index->SearchWithMask(searchparams, mask);
    }
    
    std::remove(codebook_path.c_str());
}

TEST(IndexGreat, KMode_initialize) {
    std::vector<int> deviceList {0};
    auto index = std::make_shared<ascendSearchacc::IndexGreat>("KMode", deviceList, true);
}

TEST(IndexGreat, AKMode_initilaize) {
    std::vector<int> deviceList {0};
    auto index = std::make_shared<ascendSearchacc::IndexGreat>("AKMode", deviceList, true);
}

TEST(IndexGreat, Error_initialize_1) {
    std::vector<int> deviceList {0};
    try {
        auto index = std::make_shared<ascendSearchacc::IndexGreat>("GMode", deviceList, true);
        FAIL() << "Expected AscendException but no exception was thrown";
    } catch (const ascendSearchacc::AscendException& e) {
        std::string errorMsg(e.what());
        EXPECT_NE(errorMsg.find("Search Mode mismatched. Choose between 'KMode' and 'AKMode'."), std::string::npos);
    }
}

TEST(IndexGreat, Error_initialize_2) {
    try {
        auto index = CreateIndexGreat("GMode");
        FAIL() << "Expected AscendException but no exception was thrown";
    } catch (const ascendSearchacc::AscendException& e) {
        std::string errorMsg(e.what());
        EXPECT_NE(errorMsg.find("Search Mode mismatched. Choose between 'KMode' and 'AKMode'."), std::string::npos);
    }
}