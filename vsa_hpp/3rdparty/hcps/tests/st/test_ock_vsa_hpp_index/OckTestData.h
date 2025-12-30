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

#ifndef HCPS_PIER_TESTS_ST_TEST_OCK_VSA_HPP_INDEX_TEST_DATA_H
#define HCPS_PIER_TESTS_ST_TEST_OCK_VSA_HPP_INDEX_TEST_DATA_H
#include <cstdint>
#include <vector>
#include <random>
#include <chrono>
#include <climits>
#include <fstream>
#include <iostream>
#include <memory>
#include "OckTestUtils.h"
#include "ock/vsa/attr/OckTimeSpaceAttrTrait.h"

namespace ock {
namespace vsa {
namespace neighbor {
namespace test {
const uint64_t GROUPROWCOUNT = 262144ULL * 64ULL;   // 一个 group 包含的数据个数，用于随机向量生成
const int SEED = 333;
class OckTestData {
public:
    void Init(uint64_t dimension, uint64_t nTotal, std::string rawBaseDataName, uint32_t queryBatch,
        std::string randomExtendDataSavePath)
    {
        dim = dimension;
        ntotal = nTotal;
        rawBaseDataPath = rawBaseDataName;
        batch = queryBatch;
        randomExtendDataPath = randomExtendDataSavePath;
        std::cout << "dim = " << dim << ", ntotal = " << ntotal << ", rawBaseDataPath = " <<
            rawBaseDataPath << ", batch = " << batch << ", randomExtendDataPath = " << randomExtendDataPath <<
            std::endl;
    }

    /* *
     * @brief 数据打乱，输入为一维 vector，按照 dim 转为二维，再打乱
     */
    void RandomShuffleOneDimVector(std::vector<int8_t> &inputFeatures)
    {
        // 打乱顺序 oneCopfeaturesyData （1） 一维转二维；（2）random_shuffle （3）二维转一维
        uint64_t dataCol = dim;                                                              // 列数
        uint64_t dataRow = inputFeatures.size() / dim;                                       // 行数
        std::vector<std::vector<int8_t>> douFeatures(dataRow, std::vector<int8_t>(dataCol)); // 定义二维数组
        for (uint64_t i = 0; i < dataRow; i++) {
            for (uint64_t j = 0; j < dataCol; j++) {
                douFeatures[i][j] = inputFeatures[i * dim + j];
            }
        }
        std::cout << "Conversion from one-dimensional vector to two-dimensional vector completed!" << std::endl;
        std::srand(seed);
        std::random_shuffle(douFeatures.begin(), douFeatures.end());
        std::cout << "Data shuffle completed!" << std::endl;
        for (uint64_t i = 0; i < dataRow; i++) {
            for (uint64_t j = 0; j < dataCol; j++) {
                inputFeatures[i * dim + j] = douFeatures[i][j];
            }
        }
        std::cout << "Conversion from two-dimensional vector to one-dimensional vector completed!" << std::endl;
    }

    /* *
     * @brief               每个 group 内调用 GenOneGroupRandFeatures 生成随机向量
     * @brief               重复每个 group 生成 ntotal 条 features
     */
    void GenRandFeatures()
    {
        features.resize(dim * ntotal);

        auto start = std::chrono::high_resolution_clock::now();
        std::vector<uint8_t> oneGroupFeatures = GenOneGroupRandFeatures();
        auto end = std::chrono::high_resolution_clock::now();
        double timeSpan = std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count();
        std::cout << "Generate one group random feature took " << timeSpan << " seconds." << std::endl;

        uint64_t needExtendNum = ntotal / GROUPROWCOUNT;
        uint64_t nTail = ntotal - needExtendNum * GROUPROWCOUNT;

        std::cout << "needExtendNum = " << needExtendNum << "; nTail = " << nTail << std::endl;
        std::cout << "oneGroupFeatures size = " << oneGroupFeatures.size() << "; features size = " << features.size() <<
            std::endl;

        for (uint64_t i = 0; i < needExtendNum; i++) {
            for (uint64_t j = 0; j < GROUPROWCOUNT; j++) {
                memcpy_s(features.data() + dim * GROUPROWCOUNT * i + j * dim, dim, oneGroupFeatures.data() + j * dim,
                    dim);
            }
        }
        if (nTail != 0) {
            for (uint64_t j = 0; j < nTail; j++) {
                memcpy_s(features.data() + dim * GROUPROWCOUNT * needExtendNum + j * dim, dim,
                    oneGroupFeatures.data() + j * dim, dim);
            }
        }
    }

    /* *
     * @brief ReadBaseRandomExtend      从 datasetCount*datasetDim 底库数据中读取 ntotal*dim 的数据
     * @datasetDim                      数据集原始数据的维数
     */
    void ReadBase(uint64_t datasetDim)
    {
        uint64_t fileLength = 0ULL;
        int8_t *featuresBuffer = ReadDatFile<int8_t>(rawBaseDataPath, fileLength);

        uint64_t datasetCount = fileLength / datasetDim;
        if (datasetDim < dim || datasetCount < ntotal) {
            std::cout << "The dimensions(" << datasetDim << ") or number(" << datasetCount <<
                ") of the dataset are too small.";
            return;
        }

        features.resize(dim * ntotal);
        for (uint64_t i = 0; i < ntotal; i++) {
            memcpy_s(features.data() + i * dim, dim, featuresBuffer + i * datasetDim, dim);
        }
    }

    void ReadBaseDirectExtend(uint64_t count) {}

    /* *
     * @brief ReadBaseRandomExtend      底库数据读取，选取指定维度数量 needRandDim，随机扩展到指定的 count 条向量
     * @dataDim                         数据集原始数据的维数
     * @dataCount                       数据集原始数据条数
     */
    void ReadBaseRandomExtend(int needRandDim = 5, uint64_t dataDim = 512ULL, uint64_t dataCount = 262144ULL * 12ULL,
        bool isSave = false)
    {
        std::ifstream inFile;
        inFile.open(rawBaseDataPath, std::ios::binary);
        if (!inFile.is_open()) {
            std::cerr << "The file can not be opened" << std::endl;
            return;
        }

        std::vector<int8_t> rawData(dataDim * dataCount);
        inFile.read((char *)(rawData.data()), dataDim * dataCount);
        std::vector<int8_t> oneCopyData(dim * dataCount);
        std::vector<int8_t> oneCopyDataTmp(dim * dataCount);
        for (uint64_t i = 0; i < dataCount; i++) {
            memcpy_s(oneCopyData.data() + i * dim, dim, rawData.data() + i * dataDim, dim);
        }

        // 写入features
        features.resize(dim * ntotal);
        memcpy_s(features.data(), dim * dataCount, oneCopyData.data(), dim * dataCount);
        memcpy_s(oneCopyDataTmp.data(), dim * dataCount, oneCopyData.data(), dim * dataCount);

        // 随机生成基础数据
        int needExtendNum = ntotal / dataCount;                       // 需要扩展的次数 e.g. 256/12=21...4
        uint64_t nTail = ntotal - needExtendNum * dataCount;          // e.g. 262144*256 - 262144*21*12 = 4 * 262144
        std::vector<int> randomPos = RandomPosGen(dim - 1, 0, needRandDim); // 随机选取 needRandDim 个位置
        std::vector<int8_t> RandomData = TwoDimRandomInt8Gen(needExtendNum, needRandDim);

        for (int i = 0; i < needExtendNum - 1; i++) {
            memcpy_s(oneCopyData.data(), dim * dataCount, oneCopyDataTmp.data(), dim * dataCount);
            // 对原始数据的随机 20 个位置的元素进行随机替换
            for (int j = 0; j < needRandDim; j++) {
                int theRandomPos = randomPos[j];
                int8_t baseData = RandomData[i * needRandDim + j];
                modifyDatasetValuePos(oneCopyData, dataCount, baseData, theRandomPos);
            }
            // 加入 262144 * 12 条随机向量
            memcpy_s(features.data() + (i + 1) * dim * dataCount, dim * dataCount, oneCopyData.data(), dim * dataCount);
            std::cout << "Added " << (i + 1) << " features!" << std::endl;
        }

        // 拷贝 262144 * 4 条向量
        memcpy_s(oneCopyData.data(), dim * dataCount, oneCopyDataTmp.data(), dim * dataCount);
        for (int j = 0; j < needRandDim; j++) {
            int theRandomPos = randomPos[j];
            int8_t baseData = RandomData[(needExtendNum - 1) * needRandDim + j];
            modifyDatasetValuePos(oneCopyData, nTail, baseData, theRandomPos);
        }
        memcpy_s(features.data() + needExtendNum * dim * dataCount, dim * nTail, oneCopyData.data(),
            dim * dataCount);

        // 保存底库数据
        if (isSave) {
            SaveFeaturesData(randomExtendDataPath, features);
        }
    }

    void FeatureLabelsGenerator()
    {
        for (int64_t j = 0; j < ntotal; ++j) {
            labels.push_back(j);
        }
    }

    void CustomAttrGenerator(uint32_t extKeyAttrsByteSize = 1)
    {
        customAttr.resize(ntotal * extKeyAttrsByteSize, 255U);
    }

    // 生成底库属性数据
    void FeatureAttrGenerator(int32_t maxTime = 100, int32_t maxTokenId = 3)
    {
        size_t n = ntotal;
        attrs.resize(ntotal);
        for (size_t i = 0; i < n; ++i) {
            attrs[i].time = int32_t(i % maxTime);
            attrs[i].tokenId = int32_t(i % maxTokenId);
        }
    }

    /* *
     * @brief ReadQueryBase       	    读取 queryNum 个查询向量到 queryFeature, 维度为 dim
     * @param[in] fileName              测试数据集路径
     * @param[in] queryDim              数据集原始维度
     */
    void ReadQueryBase(std::string fileName, uint64_t queryNum = 1, uint64_t queryDim = 256ULL)
    {
        uint64_t fileLength = 0ULL;
        int8_t *queriesBuffer = ReadDatFile<int8_t>(fileName, fileLength);

        uint64_t datasetCount = fileLength / queryDim;
        if (queryDim < dim || datasetCount < queryNum) {
            std::cout << "The dimensions(" << queryDim << ") or number(" << datasetCount <<
                ") of the test set are too small.";
            return;
        }

        uint64_t queryByteNum = queryNum * dim;
        queryFeature.resize(queryByteNum);
        for (int i = 0; i < queryNum; i++) {
            memcpy_s(queryFeature.data() + i * dim, dim, queriesBuffer + i * queryDim, dim);
        }
    }

    // 新建过滤条件
    void AttrFilterGenerator(uint32_t tokenNum = 2500)
    {
        attr::OckTimeSpaceAttrTrait filter(tokenNum);
        filter.minTime = 0;
        filter.maxTime = 98L;
        filter.bitSet.Set(0U);
        filter.bitSet.Set(1U);
        filter.bitSet.Set(2U);
        attrFilter.resize(batch, filter);
    }

    // 用户自定义属性过滤条件
    void ExtraMaskGenerator(int &errorCode)
    {
        extraMaskIsAtDevice = true;
        extraMaskLenEachQuery = ntotal / __CHAR_BIT__;
        uint64_t extraMaskLenAllQuery = extraMaskLenEachQuery * batch;
        void *aclExtraMask = nullptr;
        errorCode = aclrtMalloc(&aclExtraMask, extraMaskLenAllQuery, ACL_MEM_MALLOC_HUGE_FIRST);
        if (errorCode != 0) {
            std::cout << "aclrtMalloc failed." << " ret=" << errorCode << std::endl;
            return;
        } else {
            extraMask = reinterpret_cast<uint8_t *>(aclExtraMask);
            CopyMemoryToOne(extraMask, extraMaskLenAllQuery, 100UL, errorCode);
            std::cout << "aclrtMalloc succeed." << " ret=" << errorCode << std::endl;
        }
    }

    // labels 和 distances 读取
    template <typename LabelDataT> void ReadLabelsBase(std::string fileName, uint32_t topk = 200, uint64_t queryNum = 1)
    {
        uint64_t fileLength = 0ULL;
        LabelDataT *labelsBuffer = ReadDatFile<LabelDataT>(fileName, fileLength);

        uint64_t datasetCount = fileLength / topk;
        if (datasetCount < queryNum) {
            std::cout << "The number(" << datasetCount << ") of the labels data set are too small.";
            return;
        }

        queryLabel.resize(queryNum * topk, -1);
        for (int i = 0; i < queryNum; i++) {
            memcpy_s(queryLabel.data() + i * topk, topk * sizeof(LabelDataT), labelsBuffer + i * topk,
                topk * sizeof(LabelDataT));
        }
    }

    template <typename DistanceDataT>
    void ReadDistanceBase(std::string fileName, uint32_t topk = 200, uint64_t queryNum = 1)
    {
        uint64_t fileLength = 0ULL;
        DistanceDataT *distancesBuffer = ReadDatFile<DistanceDataT>(fileName, fileLength);

        uint64_t datasetCount = fileLength / topk;
        if (datasetCount < queryNum) {
            std::cout << "The number(" << datasetCount << ") of the distance data set are too small.";
            return;
        }

        queryDistances.resize(queryNum * topk, -1);
        for (int i = 0; i < queryNum; i++) {
            memcpy_s(queryDistances.data() + i * topk, topk * sizeof(DistanceDataT), distancesBuffer + i * topk,
                topk * sizeof(DistanceDataT));
        }
    }

    // 底库数据落盘
    template <typename DataT> void SaveFeaturesData(std::string fileName, const std::vector<DataT> &features)
    {
        std::ofstream outFileStream;
        outFileStream.open(fileName, std::ios::binary);
        if (outFileStream.is_open()) {
            outFileStream.write(reinterpret_cast<const char *>(features.data()), features.size() * sizeof(DataT));
            outFileStream.close();
        }
        std::cout << "Saved " << features.size() << " data!" << std::endl;
    }

    // 配置参数
    uint64_t dim{ 256ULL };
    uint64_t ntotal{ 16777216ULL };
    int seed{ SEED };

    // 底库数据
    std::vector<int8_t> features;
    std::vector<int64_t> labels;
    std::vector<attr::OckTimeSpaceAttr> attrs;
    std::vector<uint8_t> customAttr;
    std::string rawBaseDataPath;
    std::string randomExtendDataPath = "/home/liulianguang/data/dataTemp/VGG2C/tmp/base.dat";

    // 查询数据
    uint32_t batch{ 1U };
    std::vector<int8_t> queryFeature;
    std::vector<attr::OckTimeSpaceAttrTrait> attrFilter;
    bool shareAttrFilter{ false };
    uint8_t *extraMask = nullptr;
    uint64_t extraMaskLenEachQuery{ 0 };
    bool extraMaskIsAtDevice{ false };
    bool enableTimeFilter{ true };
    std::vector<float> queryDistances;
    std::vector<int64_t> queryLabel;

    // mindx查询结果数据

private:
    /* *
     * @brief .dat 文件读取，返回 DataT * 类型的指针
     */
    template <typename DataT>
    DataT *ReadDatFile(std::string filepath, uint64_t &theFileLength)
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

    std::vector<int> RandomPosGen(int maxData = 255, int minData = 0, int posNum = 20)
    {
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

    // 修改数据集指定位置的数值
    void modifyDatasetValuePos(std::vector<int8_t> &baseData, uint64_t dataNum, int8_t modifyData, int theRandomPos)
    {
        int offsetData = std::numeric_limits<int8_t>::max() + 1;
        int maxData = std::numeric_limits<uint8_t>::max();
        for (int k = 0; k < dataNum; k++) {
            int8_t rawD = baseData[k * dim + theRandomPos];
            baseData[k * dim + theRandomPos] = ((rawD + modifyData) + offsetData) % maxData - offsetData;
        }
    }

    // aclrtMemcpy 给 device 上的内存块赋值
    void CopyMemoryToOne(uint8_t *memPtr, uint64_t byteCount, uint64_t memsetTimes, int &ret)
    {
        uint64_t byteCountEach = byteCount / memsetTimes;
        uint8_t *arr = new uint8_t[byteCountEach];
        memset_s(arr, byteCountEach, 0xff, byteCountEach);
        for (uint64_t i = 0; i < memsetTimes; i++) {
            ret = aclrtMemcpy(reinterpret_cast<void *>(memPtr + i * byteCountEach), byteCountEach,
                reinterpret_cast<void *>(arr), byteCountEach, ACL_MEMCPY_HOST_TO_DEVICE);
            if (ret != 0) {
                std::cout << "aclrtMemcpy failed."
                          << " ret=" << ret << std::endl;
                return;
            }
        }
    }

    // 生成一个 group 的随机数 262144 * 64
    std::vector<uint8_t> GenOneGroupRandFeatures(void)
    {
        uint64_t featureCounts = GROUPROWCOUNT * dim;
        std::vector<uint8_t> randFeatures(featureCounts);
        std::mt19937 gen(seed);
        std::uniform_int_distribution<> dis(0, UCHAR_MAX);

        for (int n = 0; n < featureCounts; ++n) {
            randFeatures[n] = static_cast<uint8_t>(dis(gen));
        }
        return randFeatures;
    }

    int fastRandMax = 0x7FFF;
};
} // namespace test
} // namespace neighbor
} // namespace vsa
} // namespace ock
#endif // HCPS_PIER_TESTS_ST_TEST_OCK_VSA_HPP_INDEX_TEST_DATA_H
