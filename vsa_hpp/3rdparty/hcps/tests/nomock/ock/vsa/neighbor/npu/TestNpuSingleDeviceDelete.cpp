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

#include <iostream>
#include <fstream>
#include <random>
#include <chrono>
#include <vector>
#include <numeric>
#include <cassert>
#include <string>
#include "gtest/gtest.h"
#include "acl/acl.h"
#include "ock/log/OckHcpsLogger.h"
#include "ock/vsa/attr/OckTimeSpaceAttrTrait.h"
#include "ock/vsa/neighbor/base/OckVsaAnnFactory.h"
#include "ock/vsa/neighbor/base/OckVsaAnnCreateParam.h"
#include "ock/vsa/neighbor/base/OckVsaAnnAddFeatureParam.h"

namespace ock {
namespace vsa {
namespace neighbor {
namespace npu {
const uint64_t DIMS = 256ULL;
class TestNpuSingleDeviceDelete : public testing::Test {
public:
    using msTime = std::chrono::time_point<std::chrono::high_resolution_clock>;
    void SetUp(void) override
    {
        BuildDeviceInfo();
        aclrtSetDevice(deviceInfo->deviceId);
    }

    void TearDown(void) override
    {
        aclrtResetDevice(deviceInfo->deviceId);
    }

    void BuildIndexBase()
    {
        OckVsaErrorCode errorCode = hmm::HMM_SUCCESS;
        auto facReg = OckVsaAnnIndexFactoryRegister<int8_t, DIMS, 2U, attr::OckTimeSpaceAttrTrait>::Instance();
        auto fac = facReg.GetFactory("NPUTS");
        auto param = OckVsaAnnCreateParam::Create(deviceInfo->cpuSet, deviceInfo->deviceId, maxFeatureRowCount,
            tokenNum, extKeyAttrsByteSize);
        indexBase = fac->Create(param, dftTrait, errorCode);
    }

    void ReadDatFile(std::string filepath, char *buffer, uint64_t length)
    {
        std::ifstream file;
        file.open(filepath, std::ios::in | std::ios::binary);
        if (!file.is_open()) {
            std::cout << "Can not open the file " << filepath << "\n";
        }
        file.seekg(0, std::ios::end);       // 指针定位到文件末尾
        uint64_t fileLength = file.tellg(); // 指定定位到文件开始
        file.seekg(0, std::ios::beg);
        std::cout << "FileLength: " << fileLength << '\n';

        file.read(buffer, std::min(length, fileLength));
        file.close();
        // 如果文件长度不够，填充随机数据
        if (length > fileLength) {
            buffer += fileLength;
            for (uint64_t i = 0; i < length - fileLength; i++) {
                *buffer = static_cast<char>(rand() % std::numeric_limits<uint8_t>::max());
                buffer++;
            }
        }
        return;
    }

    template <typename DataT> void PrintVector(const DataT *data, uint64_t printNum)
    {
        std::cout << "Vector:"
                  << "\n";
        for (uint64_t i = 0; i < printNum; i++) {
            std::cout << static_cast<int>(data[i]) << ", ";
        }
        std::cout << std::endl;
    }

    void AddData(uint64_t count, std::string featurePath)
    {
        features.reserve(count * DIMS);
        customAttr.reserve(count * extKeyAttrsByteSize);
        ReadDatFile(featurePath, reinterpret_cast<char *>(features.data()), count * DIMS * sizeof(int8_t));
        for (uint64_t i = 0; i < count; ++i) {
            labels.push_back(i);
            attrs.push_back((attr::OckTimeSpaceAttr(int32_t(i % tokenNum), uint32_t(i % tokenNum))));
        }
        std::cout << "Add features success" << std::endl;
    }

    void PrepareOutputData()
    {
        outLabels.resize(queryNum * topK, -1);
        outDistances.resize(queryNum * topK, -1);
        validNums.resize(queryNum, 1);
    }

    void GenerateQueryData(uint64_t count)
    {
        // 生成查询向量
        for (uint32_t i = 0; i < count * DIMS; ++i) {
            queryFeature.push_back(static_cast<int8_t>(i % 8U + 1));
        }
        // 生成时空过滤条件
        int32_t minTime = 0;
        int32_t maxTime = 100;
        attrFilter = std::make_shared<attr::OckTimeSpaceAttrTrait>(tokenNum);
        attrFilter->minTime = minTime;
        attrFilter->maxTime = maxTime;
        attrFilter->bitSet.SetAll();
    }

    std::vector<int64_t> GenerateRangeLabels(int64_t start, int64_t end)
    {
        std::vector<int64_t> labelVec;
        for (size_t i = start; i <= end; i++) {
            labelVec.push_back(i);
        }
        return labelVec;
    }

    std::vector<int64_t> GenerateSampleLabels(int64_t start, int64_t end, int64_t num)
    {
        // 确保离散采样效率
        assert(num <= (end - start) / 2UL);
        std::random_device seed;
        std::ranlux48 engine(seed());
        std::uniform_int_distribution<> distrib(start, end);
        std::unordered_set<int64_t> labelSet;
        while (labelSet.size() < num) {
            labelSet.insert(distrib(engine));
        }
        return std::vector<int64_t>(std::make_move_iterator(labelSet.begin()), std::make_move_iterator(labelSet.end()));
    }

    bool ComparePtrValue(int8_t *inputPtr, int8_t *outputPtr, uint64_t vecLen)
    {
        for (uint64_t i = 0UL; i < vecLen; i++) {
            if (*(inputPtr + i) != *(outputPtr + i)) {
                std::cout << i << std::endl;
                PrintVector(inputPtr + i, DIMS);
                PrintVector(outputPtr + i, DIMS);
                return false;
            }
        }
        return true;
    }

    void LabelBatchDelete(uint64_t &rowCount, uint64_t &curNum, uint64_t start, std::vector<int8_t> &featuresA,
        std::vector<int8_t> &featuresB)
    {
        auto tailLabels = std::move(GenerateRangeLabels(rowCount - batchSize, rowCount - 1ULL));
        auto deleteLabels = std::move(GenerateRangeLabels(start, start + batchSize - 1UL));
        errorCode = indexBase->GetFeatureByLabel(batchSize, tailLabels.data(), featuresA.data());
        EXPECT_EQ(errorCode, hmm::HMM_SUCCESS);
        indexBase->DeleteFeatureByLabel(deleteLabels.size(), deleteLabels.data());
        curNum -= batchSize;
        EXPECT_EQ(indexBase->GetFeatureNum(), curNum);
        errorCode = indexBase->GetFeatureByLabel(batchSize, tailLabels.data(), featuresB.data());
        EXPECT_EQ(errorCode, hmm::HMM_SUCCESS);
        EXPECT_TRUE(ComparePtrValue(featuresA.data(), featuresB.data(), batchSize * DIMS));
        errorCode = indexBase->GetFeatureByLabel(batchSize, deleteLabels.data(), featuresA.data());
        EXPECT_EQ(errorCode, VSA_ERROR_LABEL_NOT_EXIST);
        rowCount -= batchSize;
    }

    void LabelSearchDelete(OckVsaAnnQueryCondition<int8_t, DIMS, attr::OckTimeSpaceAttrTrait> &queryCondition,
        OckVsaAnnQueryResult<int8_t, attr::OckTimeSpaceAttrTrait> &outResult)
    {
        std::vector<int64_t> deleteLabels;
        std::vector<int64_t> topLabels;
        auto ret = indexBase->Search(queryCondition, outResult);
        EXPECT_EQ(ret, hmm::HMM_SUCCESS);
        for (size_t i = 0; i < topK; i++) {
            if (i < topK / 3UL) {
                deleteLabels.push_back(outResult.labels[i]);
            }
            topLabels.push_back(outResult.labels[i]);
        }
        errorCode = indexBase->DeleteFeatureByLabel(deleteLabels.size(), deleteLabels.data());
        EXPECT_EQ(errorCode, hmm::HMM_SUCCESS);
        ret = indexBase->Search(queryCondition, outResult);
        EXPECT_EQ(ret, hmm::HMM_SUCCESS);

        CheckLabelDelResult(outResult, deleteLabels, topLabels);
    }

    void TokenSearchDelete(std::vector<std::tuple<attr::OckTimeSpaceAttr>> &topAttrs,
        std::vector<int64_t> &deleteLabels,
        OckVsaAnnQueryCondition<int8_t, DIMS, attr::OckTimeSpaceAttrTrait> &queryCondition,
        OckVsaAnnQueryResult<int8_t, attr::OckTimeSpaceAttrTrait> &outResult)
    {
        std::unordered_set<uint32_t> deleteTokenSet;
        for (size_t i = 0; i < topK / 3UL; i++) {
            deleteTokenSet.insert(std::get<0>(topAttrs[i]).tokenId);
        }

        std::vector<uint32_t> deleteTokens(deleteTokenSet.begin(), deleteTokenSet.end());
        PrintVector(deleteTokens.data(), deleteTokens.size());
        errorCode = indexBase->DeleteFeatureByToken(deleteTokens.size(), deleteTokens.data());
        EXPECT_EQ(errorCode, hmm::HMM_SUCCESS);
        EXPECT_EQ(indexBase->Search(queryCondition, outResult), hmm::HMM_SUCCESS);

        std::unordered_set<int64_t> lastRes;
        for (size_t i = 0; i < topK; i++) {
            lastRes.insert(outResult.labels[i]);
        }
        for (size_t i = 0; i < topK / 3UL; i++) {
            EXPECT_EQ(lastRes.find(deleteLabels[i]), lastRes.end());
        }
    }

    void CheckLabelDelResult(OckVsaAnnQueryResult<int8_t, attr::OckTimeSpaceAttrTrait> &outResult,
        std::vector<int64_t> &deleteLabels, std::vector<int64_t> &topLabels)
    {
        std::unordered_set<int64_t> lastRes;
        for (size_t i = 0; i < topK; i++) {
            lastRes.insert(outResult.labels[i]);
        }
        for (size_t i = 0; i < topK / 3UL; i++) {
            EXPECT_EQ(lastRes.find(deleteLabels[i]), lastRes.end());
        }
        for (size_t i = topK / 3UL; i < topK; i++) {
            EXPECT_TRUE(lastRes.find(topLabels[i]) != lastRes.end());
        }
    }

    void PreProcess()
    {
        // 创建index对象
        BuildIndexBase();

        // 数据准备
        AddData(nTotal, "/home/liulianguang/data/ExtendData/Deep1B-256-int8/train.dat");
        GenerateQueryData(1UL);
        PrepareOutputData();
        // 准备待添加的底库数据
        auto addFeatureParam = OckVsaAnnAddFeatureParam<int8_t, attr::OckTimeSpaceAttrTrait>(nTotal, features.data(),
            reinterpret_cast<attr::OckTimeSpaceAttrTrait::KeyTypeTuple *>(attrs.data()), labels.data(),
            customAttr.data());
        EXPECT_EQ(std::get<0>(*(addFeatureParam.attributes + 2U)).tokenId, 2U);
        EXPECT_EQ(addFeatureParam.count, nTotal);
        // 添加底库
        EXPECT_EQ(indexBase->AddFeature(addFeatureParam), hmm::HMM_SUCCESS);
        std::cout << "Add to device success" << std::endl;
    }

    void ExtractLabels(std::vector<int64_t> &deleteLabels, std::vector<int64_t> &topLabels,
        OckVsaAnnQueryResult<int8_t, attr::OckTimeSpaceAttrTrait> &outResult)
    {
        deleteLabels.clear();
        topLabels.clear();
        for (size_t i = 0; i < topK; i++) {
            if (i < topK / 3UL) {
                deleteLabels.push_back(outResult.labels[i]);
            }
            topLabels.push_back(outResult.labels[i]);
        }
    }

    std::shared_ptr<hmm::OckHmmDeviceInfo> deviceInfo;
    // 当前一个group包含64个block
    uint64_t groupNum{ 3ULL };
    uint64_t nTotal{ 262144ULL * 64ULL * groupNum };
    // maxFeatureRowCount不能超卡上内存
    uint64_t maxFeatureRowCount{ 262144ULL * 64ULL * 8ULL };
    uint32_t tokenNum{ 2500UL };
    uint32_t extKeyAttrsByteSize{ 20UL };
    attr::OckTimeSpaceAttrTrait dftTrait{ tokenNum };
    std::shared_ptr<neighbor::OckVsaAnnIndexBase<int8_t, DIMS, 2U, attr::OckTimeSpaceAttrTrait>> indexBase;
    // 底库数据
    std::vector<int8_t> features;
    std::vector<int64_t> labels;
    std::vector<attr::OckTimeSpaceAttr> attrs;
    std::vector<uint8_t> customAttr;
    // 查询数据
    std::vector<int8_t> queryFeature;
    std::shared_ptr<attr::OckTimeSpaceAttrTrait> attrFilter;
    bool shareAttrFilter{ true };
    uint32_t topK{ 128UL };
    uint32_t queryNum{ 1UL };
    uint8_t *extraMask = nullptr;
    uint64_t extraMaskLenEachQuery{ 0 };
    bool extraMaskIsAtDevice{ false };
    bool enableTimeFilter{ true };
    // 结果数据
    std::vector<int64_t> outLabels;
    std::vector<float> outDistances;
    std::vector<uint32_t> validNums;
    uint64_t batchSize = 1000ULL;
    ock::vsa::OckVsaErrorCode errorCode = hmm::HMM_SUCCESS;

private:
    void BuildDeviceInfo()
    {
        deviceInfo = std::make_shared<hmm::OckHmmDeviceInfo>();
        deviceInfo->deviceId = 2U;
        CPU_ZERO(&deviceInfo->cpuSet);
        CPU_SET(1U, &deviceInfo->cpuSet);                                                      // 设置1号CPU核
        CPU_SET(2U, &deviceInfo->cpuSet);                                                      // 设置2号CPU核
        deviceInfo->memorySpec.devSpec.maxDataCapacity = 20ULL * 1024ULL * 1024ULL * 1024ULL;  // 10G
        deviceInfo->memorySpec.devSpec.maxSwapCapacity = 20ULL * 64ULL * 1024ULL * 1024ULL;    // 2 * 64M
        deviceInfo->memorySpec.hostSpec.maxDataCapacity = 20ULL * 1024ULL * 1024ULL * 1024ULL; // 10G
        deviceInfo->memorySpec.hostSpec.maxSwapCapacity = 20ULL * 64ULL * 1024ULL * 1024ULL;   // 2 * 64M
        deviceInfo->transferThreadNum = 2ULL;                                                  // 2个线程
    }
};

TEST_F(TestNpuSingleDeviceDelete, delete_by_label_correctly)
{
    uint64_t rowCount = nTotal;
    uint64_t curNum = nTotal;
    std::vector<int64_t> tailLabels;
    std::vector<int64_t> deleteLabels;
    std::vector<int8_t> featuresA(batchSize * DIMS);
    std::vector<int8_t> featuresB(batchSize * DIMS);
    /* -------------------------添加底库---------------------------------------- */
    PreProcess();
    // 准备查询数据
    auto queryCondition = OckVsaAnnQueryCondition<int8_t, DIMS, attr::OckTimeSpaceAttrTrait>(queryNum,
        queryFeature.data(), attrFilter.get(), shareAttrFilter, topK, extraMask, extraMaskLenEachQuery,
        extraMaskIsAtDevice, enableTimeFilter);
    auto outResult = OckVsaAnnQueryResult<int8_t, attr::OckTimeSpaceAttrTrait>(queryNum, topK, outLabels.data(),
        outDistances.data(), validNums.data());
    /* ----------------------------------------------------------------- */

    // 删除第一个group
    // rowCount代表最后一个外部label值
    std::cout << "Test 1" << std::endl;
    deleteLabels = std::move(GenerateRangeLabels(0ULL, 262144ULL * 64ULL - 1ULL));
    errorCode = indexBase->DeleteFeatureByLabel(deleteLabels.size(), deleteLabels.data());
    EXPECT_EQ(errorCode, hmm::HMM_SUCCESS);
    EXPECT_EQ(indexBase->GetFeatureNum(), 262144UL * 64UL * (groupNum - 1UL));

    // 删除最后一个group
    std::cout << "Test 2" << std::endl;
    deleteLabels = std::move(GenerateRangeLabels(nTotal - 262144UL * 64UL, nTotal - 1ULL));
    errorCode = indexBase->DeleteFeatureByLabel(deleteLabels.size(), deleteLabels.data());
    EXPECT_EQ(errorCode, hmm::HMM_SUCCESS);
    rowCount -= 262144UL * 64UL;
    EXPECT_EQ(indexBase->GetFeatureNum(), 262144UL * 64UL * (groupNum - 2UL));

    // 连续删除且重叠，不跨block
    std::cout << "Test 3" << std::endl;
    curNum = 262144UL * 64UL;
    LabelBatchDelete(rowCount, curNum, 262144ULL * 64ULL, featuresA, featuresB);

    // 连续删除，跨block
    std::cout << "Test 4" << std::endl;
    LabelBatchDelete(rowCount, curNum, 262144ULL * 64ULL + 262140ULL, featuresA, featuresB);

    // 离散删除
    std::cout << "Test 5" << std::endl;
    tailLabels = std::move(GenerateRangeLabels(rowCount - batchSize, rowCount - 1ULL));
    deleteLabels = std::move(GenerateSampleLabels(262144ULL * 64ULL + 262144ULL * 10ULL,
        262144ULL * 64ULL + 262144ULL * 30ULL, batchSize));
    errorCode = indexBase->GetFeatureByLabel(batchSize, tailLabels.data(), featuresA.data());
    EXPECT_EQ(errorCode, hmm::HMM_SUCCESS);
    indexBase->DeleteFeatureByLabel(deleteLabels.size(), deleteLabels.data());
    curNum -= batchSize;
    EXPECT_EQ(indexBase->GetFeatureNum(), curNum);
    errorCode = indexBase->GetFeatureByLabel(batchSize, tailLabels.data(), featuresB.data());
    EXPECT_EQ(errorCode, hmm::HMM_SUCCESS);
    EXPECT_TRUE(ComparePtrValue(featuresA.data(), featuresB.data(), batchSize));
    errorCode = indexBase->GetFeatureByLabel(batchSize, deleteLabels.data(), featuresA.data());
    EXPECT_EQ(errorCode, VSA_ERROR_LABEL_NOT_EXIST);

    // Search删除
    std::cout << "Test 6" << std::endl;
    std::vector<int64_t> topLabels;
    LabelSearchDelete(queryCondition, outResult);

    // 再次Search删除（验ID映射）
    std::cout << "Test 7" << std::endl;
    LabelSearchDelete(queryCondition, outResult);
}

TEST_F(TestNpuSingleDeviceDelete, delete_hybrid_correctly)
{
    uint64_t rowCount = nTotal;
    std::vector<int64_t> deleteLabels;
    std::vector<int64_t> topLabels;
    std::vector<std::tuple<attr::OckTimeSpaceAttr>> topAttrs(topK);
    ock::vsa::OckVsaErrorCode errorCode = hmm::HMM_SUCCESS;
    /* -------------------------添加底库---------------------------------------- */
    PreProcess();
    // 准备查询数据
    auto queryCondition = OckVsaAnnQueryCondition<int8_t, DIMS, attr::OckTimeSpaceAttrTrait>(queryNum,
        queryFeature.data(), attrFilter.get(), shareAttrFilter, topK, extraMask, extraMaskLenEachQuery,
        extraMaskIsAtDevice, enableTimeFilter);
    auto outResult = OckVsaAnnQueryResult<int8_t, attr::OckTimeSpaceAttrTrait>(queryNum, topK, outLabels.data(),
        outDistances.data(), validNums.data());
    /* ----------------------------------------------------------------- */

    // Search删除
    std::cout << "Test 1: delete by token" << std::endl;
    auto ret = indexBase->Search(queryCondition, outResult);
    EXPECT_EQ(ret, hmm::HMM_SUCCESS);
    ExtractLabels(deleteLabels, topLabels, outResult);
    errorCode = indexBase->GetFeatureAttrByLabel(topK / 3UL, deleteLabels.data(), topAttrs.data());
    for (size_t i = 0; i < topK / 3UL; i++) {
        EXPECT_EQ(std::get<0>(topAttrs[i]).tokenId, deleteLabels[i] % tokenNum);
    }
    EXPECT_EQ(errorCode, hmm::HMM_SUCCESS);
    TokenSearchDelete(topAttrs, deleteLabels, queryCondition, outResult);
    
    std::cout << "Test 2: delete by label" << std::endl;
    ExtractLabels(deleteLabels, topLabels, outResult);
    errorCode = indexBase->DeleteFeatureByLabel(deleteLabels.size(), deleteLabels.data());
    EXPECT_EQ(errorCode, hmm::HMM_SUCCESS);
    EXPECT_EQ(indexBase->Search(queryCondition, outResult), hmm::HMM_SUCCESS);
    CheckLabelDelResult(outResult, deleteLabels, topLabels);

    // 验证tokenMap更新正确性
    std::cout << "Test 3: delete by token again" << std::endl;
    ExtractLabels(deleteLabels, topLabels, outResult);
    errorCode = indexBase->GetFeatureAttrByLabel(topK / 3UL, deleteLabels.data(), topAttrs.data());
    EXPECT_EQ(errorCode, hmm::HMM_SUCCESS);
    TokenSearchDelete(topAttrs, deleteLabels, queryCondition, outResult);
}
} // namespace npu
} // namespace neighbor
} // namespace vsa
} // namespace ock
