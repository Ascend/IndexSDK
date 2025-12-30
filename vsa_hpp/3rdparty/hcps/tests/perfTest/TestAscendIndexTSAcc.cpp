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

#include <bitset>
#include <cstdlib>
#include <ctime>
#include <exception>
#include <functional>
#include <queue>
#include <random>
#include <sys/time.h>
#include <unistd.h>
#include <vector>
#include <set>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <thread>
#include <memory>
#include <fstream>
#include <algorithm>
#include <unordered_map>
#include <unordered_set>
#include <iostream>
#include <cstring>
#include <gtest/gtest.h>
#include "ock/log/OckHmmLogHandler.h"
#include "index/AscendIndexTS.h"
#include "securec.h"

const int DIM = 256;
using FeatureAttr = faiss::ascend::FeatureAttr;
using AttrFilter = faiss::ascend::AttrFilter;
inline double GetMillisecs()
{
    struct timeval tv = { 0, 0 };
    gettimeofday(&tv, nullptr);
    return tv.tv_sec * 1e3 + tv.tv_usec * 1e-3;
}

template <typename U, typename V> constexpr auto SafeDivUp(U a, V b) -> decltype(a + b)
{
    return (b == 0) ? std::numeric_limits<decltype(a + b)>::max() : ((a + b - 1) / b);
}

std::vector<int> RandomPosGen(int maxData = 255, int minData = 0, int posNum = 20)
{
    int seed = 333;
    srand(seed);
    // 生成 posNum 个随机整数并输出
    std::vector<int> randomResult;
    randomResult.resize(posNum);
    std::cout << "random position: ";
    for (int i = 0; i < posNum; i++) {
        int num = rand() % (maxData - minData + 1) + minData;
        randomResult[i] = num;
        std::cout << num << " ";
    }
    std::cout << std::endl;
    return randomResult;
}

std::vector<int8_t> TwoDimRandomInt8Gen(int lines, int columns)
{
    int seed = 333;
    srand(seed);
    std::cout << "Start Random base data!" << std::endl;
    std::vector<int8_t> randomData;
    randomData.resize(lines * columns);
    for (int i = 0; i < lines * columns; i++) {
        randomData[i] = static_cast<int8_t>(rand() % (std::numeric_limits<int8_t>::max()));
    }
    std::cout << "Random base data generated finished!" << std::endl;
    return randomData;
}

void ReadBase(std::vector<int8_t> &features, uint64_t count)
{
    std::ifstream inFile;
    inFile.open("/home/xyz/VGG2C/VGG2C_extended_int8_bin/base.bin", std::ios::binary);
    if (!inFile.is_open()) {
        std::cerr << "The file can not be opened" << std::endl;
        return;
    }

    uint64_t blockSize = 262144ULL;
    uint64_t nBlocksRaw = 24ULL;     // 只要前一半
    uint64_t nBlocksOneCopy = 12ULL; // 只要前一半

    std::vector<int8_t> rawData(DIM * blockSize * nBlocksRaw);
    inFile.read((char *)(rawData.data()), DIM * blockSize * nBlocksRaw);
    // 只取前 256 维数据
    std::vector<int8_t> oneCopyData(DIM * blockSize * nBlocksOneCopy);
    for (uint64_t i = 0; i < blockSize * nBlocksOneCopy; i++) {
        memcpy_s(oneCopyData.data() + i * DIM, DIM, rawData.data() + i * 2UL * DIM, DIM);
    }

    // 拷贝原始 oneCopyData
    memcpy_s(features.data(), DIM * blockSize * nBlocksOneCopy, oneCopyData.data(), DIM * blockSize * nBlocksOneCopy);

    // 随机生成基础数据
    uint64_t nBlocks = count / blockSize;                               // e.g.(262144*128*2)/262144=256
    int needExtendNum = nBlocks / nBlocksOneCopy;                       // 需要扩展的次数 e.g. 256/12=21...4
    uint64_t nTail = nBlocks - needExtendNum * nBlocksOneCopy;          // e.g. 256-21*12=4
    int needRandDim = 20;                                               // 需要随机的维数
    std::vector<int> randomPos = RandomPosGen(DIM - 1, 0, needRandDim); // 随机选取 20 个位置
    std::vector<int8_t> RandomData = TwoDimRandomInt8Gen(needExtendNum, needRandDim);

    for (int i = 0; i < needExtendNum - 1; i++) {
        // 对原始数据的随机 20 个位置的元素进行随机替换
        for (int j = 0; j < needRandDim; j++) {
            int theRandomPos = randomPos[j];
            int8_t baseData = RandomData[i * needRandDim + j];
            // 遍历每个向量
            for (int k = 0; k < blockSize * nBlocksOneCopy; k++) {
                int rawD = oneCopyData[k * DIM + theRandomPos];
                oneCopyData[k * DIM + theRandomPos] = ((rawD + baseData) + 128UL) % 255UL - 128UL;
            }
        }
        // 加入 262144 * 12 条随机向量
        memcpy_s(features.data() + (i + 1) * DIM * blockSize * nBlocksOneCopy, DIM * blockSize * nBlocksOneCopy,
            oneCopyData.data(), DIM * blockSize * nBlocksOneCopy);
        std::cout << "Added " << (i + 1) << " features!" << std::endl;
    }

    // 拷贝 262144 * 4 条向量
    for (int j = 0; j < needRandDim; j++) {
        int theRandomPos = randomPos[j];
        int8_t baseData = RandomData[(needExtendNum - 1) * needRandDim + j];
        std::cout << "theRandomPos = " << theRandomPos << "\t"
                  << "baseData = " << baseData << std::endl;
        // 遍历每个向量
        for (int k = 0; k < blockSize * nTail; k++) {
            int rawD = oneCopyData[k * DIM + theRandomPos];
            oneCopyData[k * DIM + theRandomPos] = ((rawD + baseData) + 128UL) % 255UL - 128UL;
        }
    }
    // 加入 262144 * 12 条随机向量
    memcpy_s(features.data() + (needExtendNum)*DIM * blockSize * nBlocksOneCopy, DIM * blockSize * nTail,
        oneCopyData.data(), DIM * blockSize * nTail);
    std::cout << "Added " << needExtendNum << " features!" << std::endl;
}

template <typename DataT> DataT *ReadDatFile(std::string filepath, uint64_t &theFileLength)
{
    std::ifstream file;
    file.open(filepath, std::ios::in);
    if (!file.is_open()) {
        std::cout << "can not open the file " << filepath << "\n";
        return nullptr;
    }
    file.seekg(0, std::ios::end);   // 指针定位到文件末尾
    auto fileLength = file.tellg(); // 指定定位到文件开始
    file.seekg(0, std::ios::beg);
    theFileLength = static_cast<uint64_t>(fileLength);
    std::cout << filepath << " fileLength: " << fileLength << '\n';

    char *buffer = new char[fileLength];
    file.read(buffer, fileLength);
    file.close();
    return reinterpret_cast<DataT *>(buffer);
}

namespace test {
class TestAscendIndexTSAcc : public testing::Test {
public:
    void SetUp(void) override
    {
        int errorCode;
        tsIndex = std::make_shared<faiss::ascend::AscendIndexTS>();
        errorCode = tsIndex->Init(deviceId, DIM, tokenNum, faiss::ascend::AlgorithmType::FLAT_HPP_COS_INT8,
            faiss::ascend::MemoryStrategy::HPP, customAttrLen, customAttrBlockSize, maxFeatureCount);
        ASSERT_EQ(errorCode, 0);
    }

    void AscendTSHPPAddFeatures(int64_t ntotal, uint32_t addNum, std::vector<int8_t> &features,
        std::vector<FeatureAttr> &attrs)
    {
        int64_t validNum = 0;
        double addTotalTime = 0;
        for (size_t i = 0; i < addNum; i++) {
            std::vector<int64_t> labels;
            for (int64_t j = 0; j < ntotal; ++j) {
                labels.emplace_back(j + i * ntotal);
            }
            std::cout << "add batch " << i << "features" << std::endl;

            // 性能测试，每次添加相同底库、时空属性
            std::vector<uint8_t> addFeatures(features.begin(), features.begin() + (ntotal * DIM));
            std::vector<FeatureAttr> addAttrs(attrs.begin(), attrs.begin() + ntotal);
            std::vector<uint8_t> customAttr(ntotal * customAttrLen, 1);
            double addStart = GetMillisecs();
            auto ret =
                tsIndex->AddFeature(ntotal, addFeatures.data(), addAttrs.data(), labels.data(), customAttr.data());
            double addEnd = GetMillisecs();
            EXPECT_EQ(ret, 0);

            addTotalTime += addEnd - addStart;
            tsIndex->GetFeatureNum(&validNum);
            EXPECT_EQ(validNum, (i + 1) * ntotal);
            std::cout << "---------------------------GetFeatureNum is " << validNum << std::endl;
        }

        std::cout << "add " << addNum * ntotal << " data, cost time " << addTotalTime << "ms" << std::endl;
        totalNum += addNum * ntotal;
    }

    void AscendTSHPPGetFeatureNum()
    {
        int64_t featureNum = 0;
        tsIndex->GetFeatureNum(&featureNum);
        EXPECT_EQ(featureNum, totalNum);
    }

    void AscendTSHPPGetCustomAttrByBlockId()
    {
        uint8_t *customAttr0;
        uint32_t existingBlockId = 0;
        double getCustomAttrByBlockIdStart = GetMillisecs();
        auto ret = tsIndex->GetCustomAttrByBlockId(existingBlockId, customAttr0);
        double getCustomAttrByBlockIdEnd = GetMillisecs();
        EXPECT_EQ(ret, 0);
        EXPECT_NE(customAttr0, nullptr);
        std::cout << "GetCustomAttrByBlockId cost time " << getCustomAttrByBlockIdEnd - getCustomAttrByBlockIdStart <<
            "ms" << std::endl;

        uint8_t *customAttr1;
        uint32_t notExistingBlockId = totalNum / 262144 + 10;
        ret = tsIndex->GetCustomAttrByBlockId(notExistingBlockId, customAttr1);
        EXPECT_NE(ret, 0);
    }

    void AscendTSHPPGetFeatureByLabel(uint64_t count)
    {
        std::vector<int64_t> labels;
        for (int64_t i = 0; i < count; ++i) {
            labels.push_back(i);
        }
        std::vector<int8_t> featureOut(DIM * count);

        double getStart = GetMillisecs();
        auto ret = tsIndex->GetFeatureByLabel(count, labels.data(), featureOut.data());
        double getEnd = GetMillisecs();
        EXPECT_EQ(ret, 0);
        EXPECT_NE(featureOut.data(), nullptr);

        std::cout << "Get " << count << " features by label, cost time " << getEnd - getStart << "ms" << std::endl;
    }

    void AscendTSHPPSearch(std::vector<int8_t> &querys, std::vector<int> &batchSizes, int topk = 200, int queryNum = 50)
    {
        std::vector<int32_t> filtVec{ 1, 10, 49 };
        // 测试多个过滤比, 多个batchSize查询
        for (auto &batchSize : batchSizes) {
            std::vector<float> distances(batchSize * topk, -1);
            std::vector<int64_t> labelRes(batchSize * topk, -1);
            std::vector<uint32_t> validNum(batchSize, -1);

            // 每个batchSize测多个过滤比
            for (int t = 0; t < filtVec.size(); ++t) {
                uint32_t setlen = (uint32_t)(((tokenNum + 7) / 8));
                std::vector<uint8_t> bitSet(setlen, 0);
                // 00000111   -> 0,1,2
                bitSet[0] = (0x1 << 0) | (0x1 << 1) | (0x1 << 2);

                AttrFilter filter{};
                filter.timesStart = 0;
                filter.timesEnd = filtVec[t];
                filter.tokenBitSet = bitSet.data();
                filter.tokenBitSetLen = setlen;
                std::vector<AttrFilter> queryFilters(batchSize, filter);

                // 单 batch 查询，遍历 queryNum 次
                double searchTime = 0;
                std::vector<int8_t> queryCur(querys.begin(), querys.begin() + batchSize * DIM);
                for (int i = 0; i < queryNum; i++) {
                    double searchTimeBegin = GetMillisecs();

                    auto ret = tsIndex->Search(batchSize, queryCur.data(), queryFilters.data(), false, topk,
                        labelRes.data(), distances.data(), validNum.data());

                    double searchTimeEnd = GetMillisecs();
                    EXPECT_EQ(ret, 0);
                    searchTime += searchTimeEnd - searchTimeBegin;
                }
                std::cout << "BatchSize is " << batchSize << ", filter is " <<
                    double(filter.timesEnd + 1) / double(100UL) << ", average qps is " <<
                    batchSize * 1000UL * queryNum / searchTime << std::endl;
            }
        }
    }

    void AscendTSHPPSearchWithExtraMask(std::vector<int8_t> &querys, std::vector<int> &batchSizes, int topk = 200,
        int queryNum = 50)
    {
        std::vector<int32_t> attrFilterVec{ 1, 9, 49 };
        // 测试多个过滤比, 多个batch
        for (auto &batchSize : batchSizes) {
            std::vector<float> distances(batchSize * topk, -1);
            std::vector<int64_t> labelRes(batchSize * topk, -1);
            std::vector<uint32_t> validNum(batchSize, -1);

            for (int t = 0; t < attrFilterVec.size(); ++t) {
                uint32_t setlen = (uint32_t)(((tokenNum + 7) / 8));
                std::vector<uint8_t> bitSet(setlen, 0);
                // 00000111   -> 0,1,2
                bitSet[0] = (0x1 << 0) | (0x1 << 1) | (0x1 << 2);

                AttrFilter filter{};
                filter.timesStart = 0;
                filter.timesEnd = attrFilterVec[t];
                filter.tokenBitSet = bitSet.data();
                filter.tokenBitSetLen = setlen;
                std::vector<AttrFilter> queryFilters(batchSize, filter);

                // 单 batch 查询，遍历 queryNum 次
                double searchWithExtraMaskTime = 0;
                std::vector<int8_t> queryCur(querys.begin(), querys.begin() + batchSize * DIM);
                for (int i = 0; i < queryNum; i++) {
                    // searchWithExtraMask不带extraMask生成的性能
                    std::vector<uint8_t> extraMask(SafeDivUp(totalNum, 8UL) * batchSize, 255UL);
                    double searchWithExtraMaskTimeBegin = GetMillisecs();

                    auto ret = tsIndex->SearchWithExtraMask(batchSize, queryCur.data(), queryFilters.data(), false,
                        topk, extraMask.data(), extraMask.size() / batchSize, false, labelRes.data(), distances.data(),
                        validNum.data());

                    double searchWithExtraMaskTimeEnd = GetMillisecs();
                    EXPECT_EQ(ret, 0);
                    searchWithExtraMaskTime += searchWithExtraMaskTimeEnd - searchWithExtraMaskTimeBegin;
                }
                std::cout << "[extra mask] BatchSize is " << batchSize << ", filter is " <<
                    double(filter.timesEnd + 1) / double(100UL) << ", average qps is " <<
                    batchSize * 1000UL * queryNum / searchWithExtraMaskTime << std::endl;
            }
        }
    }

    std::shared_ptr<faiss::ascend::AscendIndexTS> tsIndex;
    uint32_t deviceId = 5;
    uint32_t tokenNum = 2500;
    uint32_t customAttrLen = 22;
    uint32_t customAttrBlockSize = 262144;
    uint64_t maxFeatureCount = 262144ULL * 64ULL * 26ULL;
    uint64_t totalNum = 0;
};

class TestData {
public:
    void SetNTotal(size_t totalNum)
    {
        this->ntotal = totalNum;
    }

    void FeatureAttrGenerator()
    {
        size_t n = ntotal;
        attrs.resize(ntotal);
        for (size_t i = 0; i < n; ++i) {
            attrs[i].time = int32_t(i % 4UL);
            attrs[i].tokenId = int32_t(i % 4UL);
        }
    }

    std::vector<FeatureAttr> &GetFeaturesAttrs()
    {
        return attrs;
    }

    void ReadQueryBase(std::string fileName, std::vector<int8_t> &querys, int queryNum = 1)
    {
        uint64_t fileLength = 0ULL;
        int8_t *querysBuffer = ReadDatFile<int8_t>(fileName, fileLength);

        querys.resize(queryNum * DIM);
        for (int i = 0; i < queryNum; i++) {
            memcpy_s(querys.data() + i * DIM, DIM, querysBuffer + i * 2UL * DIM, DIM);
        }
        for (int i = 0; i < 20UL; ++i) {
            std::cout << "Read from file, querys[" << i << "] is " << int(querys[i]) << std::endl;
        }
    }

private:
    size_t ntotal = 0;
    std::vector<int8_t> features;
    std::vector<FeatureAttr> attrs;
};

TEST_F(TestAscendIndexTSAcc, perf)
{
    uint32_t addNum = 64;
    int64_t ntotal = 262144LL * 1; // 262144; // 1000000;

    // prepare base data
    std::vector<int8_t> features(DIM * ntotal * addNum);
    ReadBase(features, ntotal * addNum);

    // 时空属性
    std::unique_ptr<TestData> testData = std::make_unique<TestData>();
    testData->SetNTotal(ntotal * addNum);
    testData->FeatureAttrGenerator();
    std::vector<FeatureAttr> &attrs = testData->GetFeaturesAttrs();

    // prepare query data
    std::vector<int8_t> querys;
    testData->ReadQueryBase("/home/xyz/VGG2C/VGG2C_extended_int8_bin/query.bin", querys);

    AscendTSHPPAddFeatures(ntotal, addNum, features, attrs);
    AscendTSHPPGetFeatureNum();
    AscendTSHPPGetCustomAttrByBlockId();

    uint64_t count = 500;
    AscendTSHPPGetFeatureByLabel(count);

    std::vector<int> batchSizes{ 1, 2, 4, 8 };
    AscendTSHPPSearch(querys, batchSizes);
    AscendTSHPPSearchWithExtraMask(querys, batchSizes);
    std::cout << "finish test" << std::endl;
}
} // end of namespace test
