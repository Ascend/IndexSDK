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
#include <queue>
#include <utility>
#include <cstdint>
#include <vector>
#include <string>
#include <stdexcept>
#include <cmath>
#include <limits>
#include <cstring>
#include <sys/stat.h>
#include <gtest/gtest.h>
#include "securec.h"

#include "npu/IndexGreat.h"
#include "npu/NpuIndexVStar.h"
#include "npu/NpuIndexIVFHSP.h"

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
inline void WriteAll(std::ofstream& ofs, const void* p, size_t n)
{
    ofs.write(reinterpret_cast<const char*>(p), static_cast<std::streamsize>(n));
    if (!ofs) {
        throw std::runtime_error("write failed");
    }
}

inline void BuildCodeBooks(int32_t dim, int32_t nlistL1, int32_t subSpaceDimL1,
                           int32_t nlistL2, int32_t subSpaceDimL2,
                           std::vector<float>& l1, std::vector<float>& l2)
{
    const size_t L1 = static_cast<size_t>(nlistL1) * static_cast<size_t>(subSpaceDimL1) * static_cast<size_t>(dim);
    const size_t L2 = static_cast<size_t>(nlistL1) * static_cast<size_t>(nlistL2)
                    * static_cast<size_t>(subSpaceDimL2) * static_cast<size_t>(subSpaceDimL1);
    l1.resize(L1);
    l2.resize(L2);

    const size_t strideBd = static_cast<size_t>(subSpaceDimL1) * static_cast<size_t>(dim);
    for (size_t i = 0; i < L1; ++i) {
        const size_t a  = i / strideBd;
        const size_t r1 = i % strideBd;
        const size_t b  = r1 / static_cast<size_t>(dim);
        const size_t d  = r1 % static_cast<size_t>(dim);

        const float v = std::sin(0.017f * static_cast<float>(a)
                               + 0.023f * static_cast<float>(b)
                               + 0.031f * static_cast<float>(d));
        l1[i] = v;
    }

    const size_t strideBks = static_cast<size_t>(nlistL2) * static_cast<size_t>(subSpaceDimL2) * static_cast<size_t>(subSpaceDimL1);
    const size_t strideKs  = static_cast<size_t>(subSpaceDimL2) * static_cast<size_t>(subSpaceDimL1);
    for (size_t i = 0; i < L2; ++i) {
        const size_t a  = i / strideBks;
        const size_t r1 = i % strideBks;
        const size_t b  = r1 / strideKs;
        const size_t r2 = r1 % strideKs;
        const size_t k  = r2 / static_cast<size_t>(subSpaceDimL1);
        const size_t s  = r2 % static_cast<size_t>(subSpaceDimL1);

        const float v = std::cos(0.013f * static_cast<float>(a)
                               + 0.019f * static_cast<float>(b)
                               + 0.029f * static_cast<float>(k)
                               + 0.037f * static_cast<float>(s));
        l2[i] = v;
    }
}

inline bool BuildAndSaveCodebookBin(const std::string& codeBooksPath,
                                    int32_t dim, int32_t nlistL1, int32_t subSpaceDimL1,
                                    int32_t nlistL2, int32_t subSpaceDimL2)
{
    static_assert(sizeof(int32_t) == 4, "int32_t must be 4 bytes");
    static_assert(sizeof(float)   == 4, "float must be 4 bytes");

    long long l2prod64 = 1LL * nlistL2 * subSpaceDimL2;
    if (l2prod64 > std::numeric_limits<int32_t>::max() || l2prod64 < std::numeric_limits<int32_t>::min()) {
        throw std::runtime_error("nlistL2 * subSpaceDimL2 overflows int32");
    }
    std::vector<float> codeBooksL1;
    std::vector<float> codeBooksL2;
    BuildCodeBooks(dim, nlistL1, subSpaceDimL1, nlistL2, subSpaceDimL2, codeBooksL1, codeBooksL2);

    const size_t l1Expect = static_cast<size_t>(nlistL1) * subSpaceDimL1 * dim;
    const size_t l2Expect = static_cast<size_t>(nlistL1) * nlistL2 * subSpaceDimL2 * subSpaceDimL1;
    if (codeBooksL1.size() != l1Expect) {
        throw std::runtime_error("L1 size mismatch");
    }
    if (codeBooksL2.size() != l2Expect) {
        throw std::runtime_error("L2 size mismatch");
    }

    std::ofstream ofs(codeBooksPath, std::ios::binary);
    if (!ofs) {
        throw std::runtime_error("open failed: " + codeBooksPath);
    }

    const char fourcc[4] = { 'C', 'B', 'S', 'P' };
    WriteAll(ofs, fourcc, sizeof(fourcc));

    WriteAll(ofs, &dim, sizeof(int32_t));
    WriteAll(ofs, &nlistL1, sizeof(int32_t));
    WriteAll(ofs, &subSpaceDimL1, sizeof(int32_t));
    const int32_t l2prod = static_cast<int32_t>(l2prod64);
    WriteAll(ofs, &l2prod, sizeof(int32_t));

    WriteAll(ofs, codeBooksL1.data(), sizeof(float) * codeBooksL1.size());

    WriteAll(ofs, codeBooksL2.data(), sizeof(float) * codeBooksL2.size());

    ofs.close();
    if (!ofs) {
        throw std::runtime_error("close failed (disk full?)");
    }

    std::cout << "Wrote codebook: " << codeBooksPath << std::endl;
    return true;
}

inline void CreateCodebook(const std::string& codeBookPath)
{
    int32_t nlistL2 = 32;
    int32_t subSpaceDimL2 = 16;

    std::cout << "Generating codebook..." << std::endl;
    BuildAndSaveCodebookBin(codeBookPath, g_dim, g_nlist, g_nonzeroNum, nlistL2, subSpaceDimL2);
}

/**
 * Create a vstar index
 */
inline std::shared_ptr<ascendSearchacc::NpuIndexVStar> CreateIndexVstar()
{
    ascendSearchacc::IndexVstarInitParams AModeInit;
    AModeInit.dim = g_dim;
    AModeInit.nlistL1 = g_nlist;
    AModeInit.subSpaceDimL1 = g_nonzeroNum;
    AModeInit.deviceList = { g_device };
    AModeInit.verbose = true;

    auto index = std::make_shared<ascendSearchacc::NpuIndexVStar>(AModeInit);

    return index;
}

/**
 * Create a great index
 */
inline std::shared_ptr<ascendSearchacc::IndexGreat> CreateIndexGreat(std::string mode)
{
    ascendSearchacc::KModeInitParams KModeInit;
    KModeInit.dim = g_dim;
    KModeInit.convPQM = 128;
    KModeInit.evaluationType = 0;
    KModeInit.R = 50;

    ascendSearchacc::IndexVstarInitParams AModeInit;
    AModeInit.dim = g_dim;
    AModeInit.nlistL1 = g_nlist;
    AModeInit.subSpaceDimL1 = g_nonzeroNum;
    AModeInit.deviceList = { g_device };
    AModeInit.verbose = true;

    ascendSearchacc::IndexGreatInitParams GreatInit;
    if (mode == "KMode") {
        GreatInit.mode = mode;
        GreatInit.KInitParams = KModeInit;
        GreatInit.verbose = true;
    } else if (mode == "AKMode") {
        GreatInit.mode = mode;
        GreatInit.AInitParams = AModeInit;
        GreatInit.KInitParams = KModeInit;
        GreatInit.verbose = true;
    }

    auto index = std::make_shared<ascendSearchacc::IndexGreat>(mode, GreatInit);

    return index;
}

#endif // UTILS_INCLUDED