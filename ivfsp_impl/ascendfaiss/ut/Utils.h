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


#ifndef UTILS_INCLUDED
#define UTILS_INCLUDED

#include <algorithm>
#include <cstdio>
#include <iostream>
#include <fstream>
#include <vector>
#include <random>

#include <gtest/gtest.h>
#include "securec.h"

#include "AscendMultiIndexSearch.h"
#include "ascend/custom/AscendIndexIVFSPSQ.h"

constexpr int32_t K_MAX_CAMERA_NUM = 128;
constexpr int MASK_LEN = 8;
constexpr int TIME_STAMP_START = 200;
constexpr int TIME_STAMP_END = 500;

constexpr int g_nShards = 3; // MultiIndex场景下index数
constexpr int g_handleBatch = 64; // 检索支持一次处理的batch大小
constexpr int g_searchListSize = 32768; // 检索支持一次扫描的最大桶数
constexpr int g_ntotal = 100; // 底库向量大小
constexpr int g_k = 100; // 检索topK
constexpr int g_dim = 128; // 检索维度
constexpr int g_nProbe = 64; // 检索时扫描的桶数
constexpr int g_nonzeroNum = 32; // IVFSP算法使用的降维维度 (此处，由128维降至32维)
constexpr int g_nlist = 256; // IVFSP算法使用的桶数
constexpr int g_device = 0; // NPU device id
constexpr unsigned long long g_resourceSize = 2LLU * 1024 * 1024 * 1024; // NPU全局内存池大小
constexpr bool g_filterable = true; // 是否开启filter功能

const std::vector<uint32_t> g_timeStamp {TIME_STAMP_START, TIME_STAMP_END};

struct IDFilter {
    IDFilter()
    {
        int ret = memset_s(cameraIdMask, K_MAX_CAMERA_NUM / MASK_LEN,
            static_cast<uint8_t>(0), K_MAX_CAMERA_NUM / MASK_LEN);
        if (ret == -1) {
            std::cerr << "Error when creating IDFilter with memset; ret = " << ret << std::endl;
        }
        timeRange[0] = 0;
        timeRange[1] = -1;
    }

    // 一个IDFilter对象是可以涵盖处理所有cid in [0, 127] 共128个camera
    uint8_t cameraIdMask[K_MAX_CAMERA_NUM / MASK_LEN] = {0};
    uint32_t timeRange[2] = {0};
};

/*
    batch即searchNum， 一条被检索的特征向量，传递一个IDFilter对象
    std::vector<int> &cids, 是一个固定的128元素的向量，其值从0到127
*/
inline void ConstructCidFilter(IDFilter *idFilters, int batch, const std::vector<int> &cids,
    const std::vector<uint32_t> &timestamps)
{
    for (int i = 0; i < batch; ++i) {
        for (auto current_cid : cids) {
            int g = current_cid / MASK_LEN;
            int k = current_cid % MASK_LEN;
            idFilters[i].cameraIdMask[g] += (1 << k);
        }
        idFilters[i].timeRange[0] = timestamps[0]; // start
        idFilters[i].timeRange[1] = timestamps[1]; // end
    }
}

/**
 * Return a vector of size (dim * nb) filled with values sampled from uniform distribution [0, 1]
 */
template <typename T = float>
std::vector<T> GenRandData(size_t dim, size_t nb)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);
    std::vector<T> result(dim * nb);
    for (auto& num : result) {
        num = dis(gen);
    }
    return result;
}

/**
 * Generate a random codebook
 */
inline std::vector<float> GenRandCodeBook(int dim, int nonzeroNum, int nlist)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);
    std::vector<float> codebook(static_cast<size_t>(dim) * nonzeroNum * nlist);
    for (auto& num : codebook) {
        num = dis(gen);
    }
    return codebook;
}

/**
 * Create an index instance based on global variables
 */
inline std::shared_ptr<faiss::ascendSearch::AscendIndexIVFSPSQ> CreateIndex()
{
    faiss::ascendSearch::AscendIndexIVFSPSQConfig ivfspsqConfig({g_device}, g_resourceSize);
    ivfspsqConfig.filterable = g_filterable;
    ivfspsqConfig.handleBatch = g_handleBatch;
    ivfspsqConfig.searchListSize = g_searchListSize;

    auto index = std::make_shared<faiss::ascendSearch::AscendIndexIVFSPSQ>(g_dim, g_nonzeroNum, g_nlist,
        g_nlist, faiss::ScalarQuantizer::QuantizerType::QT_8bit, faiss::METRIC_L2, false, ivfspsqConfig);

    return index;
}

template <typename T>
std::vector<T> CreateIotaVector(size_t vecSize, size_t startFrom = 0)
{
    std::vector<T> vec(vecSize);
    std::iota(vec.begin(), vec.end(), startFrom);
    return vec;
}

inline faiss::ascendSearch::AscendIndexCodeBookTrainerConfig CreateCodeBookTrainConfig(
    const float *learnDataPtr, size_t learnDataLength, bool trainAndAdd = true)
{
    faiss::ascendSearch::AscendIndexCodeBookTrainerConfig config;
    config.numIter = 1;
    config.device = g_device;
    config.batchSize = g_searchListSize;
    config.codeNum = g_searchListSize;
    config.verbose = true;
    config.memLearnData = learnDataPtr;
    config.memLearnDataSize = learnDataLength;
    config.trainAndAdd = trainAndAdd;
    return config;
}

inline faiss::ascendSearch::AscendIndexCodeBookTrainerConfig CreateCodeBookTrainConfig(
    const std::string &learnDataPath, const std::string &codebookPath, int device, float sampleRatio
)
{
    faiss::ascendSearch::AscendIndexCodeBookTrainerConfig config;
    config.numIter = 1;
    config.device = device;
    config.batchSize = g_searchListSize;
    config.codeNum = g_searchListSize;
    config.verbose = true;
    config.codeBookOutputDir = codebookPath;
    config.learnDataPath = learnDataPath;
    config.trainAndAdd = false;
    config.ratio = sampleRatio;
    return config;
}

inline void WriteCodeBookWithMeta(std::vector<float> &codebook, std::vector<char> &magicNumber,
    std::vector<uint8_t> &version, int dim, int nonzeroNum, int nlist, int blankDataSize,
    std::string codebookPath)
{
    std::ofstream outFile(codebookPath, std::ios::binary);
    // 码本需要固定的metadata格式
    std::vector<uint8_t> blankData(blankDataSize);
    outFile.write(reinterpret_cast<char *>(magicNumber.data()), magicNumber.size() * sizeof(magicNumber[0]));
    outFile.write(reinterpret_cast<char *>(version.data()), version.size() * sizeof(version[0]));
    outFile.write(reinterpret_cast<char *>(&dim), sizeof(dim));
    outFile.write(reinterpret_cast<char *>(&nonzeroNum), sizeof(nonzeroNum));
    outFile.write(reinterpret_cast<char *>(&nlist), sizeof(nlist));
    outFile.write(reinterpret_cast<char *>(blankData.data()), blankData.size());
    outFile.write(reinterpret_cast<char *>(codebook.data()), sizeof(float) * codebook.size());
    outFile.close();
}

inline void TestSearch(faiss::ascendSearch::AscendIndexIVFSPSQ *index, std::vector<float> &queryData,
    size_t dim, size_t queryNum, size_t k, bool useMask = false)
{
    std::cout << "=================== Start Searching ===================" << std::endl;
    std::vector<float> dist(k * queryNum, 0);
    std::vector<int64_t> label(k * queryNum, 0);
    const size_t batchSize = 64;
    
    std::vector<uint8_t> mask;
    if (useMask) {
        mask.resize((g_ntotal + 7) / 8, 0);
    }
    for (size_t i = 0; i < queryNum; i += batchSize) {
        size_t curBatchSize = std::min(batchSize, queryNum - i);
        if (useMask) {
            index->search_with_masks(curBatchSize, queryData.data() + i * dim, k, dist.data() + i * k,
                label.data() + i * k, mask.data());
        } else {
            index->search(curBatchSize, queryData.data() + i * dim, k, dist.data() + i * k, label.data() + i * k);
        }
    }
    std::cout << "=================== Searching Finished ================" << std::endl;
}

inline void TestSearchWithFilter(faiss::ascendSearch::AscendIndexIVFSPSQ *index, std::vector<float> &queryData,
    size_t dim, size_t queryNum, size_t k, const std::vector<uint32_t>& timeStamp, std::vector<int>& searchCID,
    bool useL1Distance = false)
{
    std::cout << "=================== Start Searching With Filter ===================" << std::endl;
    std::vector<float> dist(k * queryNum, 0);
    std::vector<int64_t> label(k * queryNum, 0);
    const size_t batchSize = 64;
    
    IDFilter idFilters[batchSize];
    void *pFilter = &idFilters[0];
    ConstructCidFilter(idFilters, batchSize, searchCID, timeStamp);

    std::vector<float> l1Distance;
    if (useL1Distance) {
        l1Distance.resize(g_nlist, 0.0f);
    }

    for (size_t i = 0; i < queryNum; i += batchSize) {
        size_t curBatchSize = std::min(batchSize, queryNum - i);
        if (useL1Distance) {
            index->search_with_filter(curBatchSize, queryData.data() + i * dim, k, dist.data() + i * k,
                label.data() + i * k, l1Distance.data(), pFilter);
        } else {
            index->search_with_filter(curBatchSize, queryData.data() + i * dim, k, dist.data() + i * k,
                label.data() + i * k, pFilter);
        }
    }
    std::cout << "=================== Searching With Filter Finished ================" << std::endl;
}

#endif // UTILS_INCLUDED