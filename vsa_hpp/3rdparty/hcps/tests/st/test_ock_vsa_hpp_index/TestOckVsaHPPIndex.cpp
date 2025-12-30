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
#include <random>
#include <iostream>
#include <sys/time.h>
#include "acl/acl.h"
#include "ock/vsa/neighbor/hpp/OckVsaAnnHppSetup.h"
#include "ock/vsa/neighbor/base/OckVsaAnnFactory.h"
#include "ock/vsa/neighbor/base/OckVsaAnnCreateParam.h"
#include "ock/vsa/neighbor/base/OckVsaAnnAddFeatureParam.h"
#include "ock/vsa/attr/OckTimeSpaceAttrTrait.h"
#include "OckTestData.h"

#include "OckTestUtils.h"

namespace ock {
namespace vsa {
namespace neighbor {
namespace test {
const uint64_t DIM_SIZE = 256; // 256: dim
const float ERRORRANGE = 1e-4;
class TestOckVsaHPPIndex : public testing::Test {
public:
    void SetUp(void) override
    {
        aclrtSetDevice(deviceId);
        InitCpuSetVec();
    }
    void TearDown(void) override
    {
        aclrtResetDevice(deviceId);
    }
    void PrepareData()
    {
        testData.Init(DIM_SIZE, ntotal, trainDataSetName, batch, randomExtendDataSavePath);
        testData.ReadBase(dataSetDim / 2ULL);
        testData.FeatureLabelsGenerator();
        testData.FeatureAttrGenerator();
        testData.CustomAttrGenerator(extKeyAttrsByteSize);

        testData.ReadQueryBase(testDataSetName, queryNum, dataSetDim);
        testData.AttrFilterGenerator(tokenNum);

        outLabels.resize(queryNum * topk, -1);
        outDistances.resize(queryNum * topk, -1);
        validNums.resize(queryNum, 1);
    }
    void BuildIndexBase()
    {
        ock::vsa::neighbor::hpp::SetUpHPPTsFactory();
        OckVsaErrorCode errorCode = hmm::HMM_SUCCESS;
        auto factoryRegister =
            OckVsaAnnIndexFactoryRegister<int8_t, DIM_SIZE, 2U, attr::OckTimeSpaceAttrTrait>::Instance();
        auto factory = factoryRegister.GetFactory("HPPTS");
        auto param = OckVsaAnnCreateParam::Create(cpuSet, deviceId, maxFeatureRowCount, tokenNum, extKeyAttrsByteSize);
        indexBase = factory->Create(param, dftTrait, errorCode);
        EXPECT_NE(indexBase.get(), nullptr);
        std::cout << "Index build success!" << std::endl;
    }

    void Add()
    {
        double tb = OckTestUtils::GetMilliSecs();
        int errorCode = vsa::VSA_SUCCESS;
        for (uint32_t i = 0; i < addNum; i++) {
            std::vector<int8_t> addFeatures(testData.features.begin() + (i * ntotalPerAdd * DIM_SIZE),
                testData.features.begin() + ((i + 1) * ntotalPerAdd * DIM_SIZE));
            std::vector<attr::OckTimeSpaceAttr> addAttrs(testData.attrs.begin() + (i * ntotalPerAdd),
                testData.attrs.begin() + ((i + 1) * ntotalPerAdd));
            std::vector<uint8_t> customAttr(testData.customAttr.begin() + (i * ntotalPerAdd * extKeyAttrsByteSize),
                testData.customAttr.begin() + ((i + 1) * ntotalPerAdd * extKeyAttrsByteSize));
            std::vector<int64_t> labels(testData.labels.begin() + i * ntotalPerAdd,
                testData.labels.begin() + (i + 1) * ntotalPerAdd);
            auto addFeatureParam = OckVsaAnnAddFeatureParam<int8_t, attr::OckTimeSpaceAttrTrait>(ntotalPerAdd,
                addFeatures.data(), reinterpret_cast<attr::OckTimeSpaceAttrTrait::KeyTypeTuple *>(addAttrs.data()),
                labels.data(), customAttr.data());
            errorCode = indexBase->AddFeature(addFeatureParam);
            if (errorCode != vsa::VSA_SUCCESS) {
                std::cout << "There's error, while AddFeature. ret value =" << errorCode << std::endl;
                break;
            }
        }
        EXPECT_EQ(errorCode, vsa::VSA_SUCCESS);
        double te = OckTestUtils::GetMilliSecs();
        double timeConvert = 60.0;
        std::cout << "add features time cost: " << (te - tb) / 1e3 / timeConvert << "min" << std::endl;
    }

    void Search()
    {
        int errorCode = vsa::VSA_SUCCESS;
        double queryTotalTime = 0;
        for (int i = 0; i < queryNum; i++) {
            std::vector<int8_t> queryCur(testData.queryFeature.begin() + i * DIM_SIZE,
                testData.queryFeature.begin() + (i + 1) * DIM_SIZE);

            auto queryCondition = OckVsaAnnQueryCondition<int8_t, DIM_SIZE, attr::OckTimeSpaceAttrTrait>(batch,
                queryCur.data(), testData.attrFilter.data(), testData.shareAttrFilter, topk, testData.extraMask,
                testData.extraMaskLenEachQuery, testData.extraMaskIsAtDevice, testData.enableTimeFilter);
            auto outResult = OckVsaAnnQueryResult<int8_t, attr::OckTimeSpaceAttrTrait>(batch, topk,
                outLabels.data() + i * topk, outDistances.data() + i * topk, validNums.data() + i);

            double t0 = OckTestUtils::GetMilliSecs();
            errorCode = indexBase->Search(queryCondition, outResult);
            EXPECT_EQ(errorCode, vsa::VSA_SUCCESS);
            double t1 = OckTestUtils::GetMilliSecs();
            queryTotalTime += (t1 - t0);
            if (errorCode != 0) {
                std::cout << "There's error, while Search. ret value =" << errorCode << std::endl;
                return;
            }
        }
        searchQPS = queryNum * 1e3 / queryTotalTime;
        std::cout << "The device search QPS = " << searchQPS << std::endl;
    }

    void SearchWithExtraMask()
    {
        int errorCode = vsa::VSA_SUCCESS;

        // extraMask 生成
        testData.ExtraMaskGenerator(errorCode);
        double queryTotalTime = 0;
        for (int i = 0; i < queryNum; i = i + batch) {
            std::vector<int8_t> queryCur(testData.queryFeature.begin() + i * DIM_SIZE,
                testData.queryFeature.begin() + (i + batch) * DIM_SIZE);

            auto queryCondition = OckVsaAnnQueryCondition<int8_t, DIM_SIZE, attr::OckTimeSpaceAttrTrait>(batch,
                queryCur.data(), testData.attrFilter.data(), testData.shareAttrFilter, topk, testData.extraMask,
                testData.extraMaskLenEachQuery, testData.extraMaskIsAtDevice, testData.enableTimeFilter);
            auto outResult = OckVsaAnnQueryResult<int8_t, attr::OckTimeSpaceAttrTrait>(batch, topk,
                outLabels.data() + i * topk, outDistances.data() + i * topk, validNums.data() + i);

            double t0 = OckTestUtils::GetMilliSecs();
            errorCode = indexBase->Search(queryCondition, outResult);
            EXPECT_EQ(errorCode, vsa::VSA_SUCCESS);
            double t1 = OckTestUtils::GetMilliSecs();
            queryTotalTime += (t1 - t0);
            if (errorCode != 0) {
                std::cout << "There's error, while Search. ret value =" << errorCode << std::endl;
                return;
            }
        }
        searchQPS = queryNum * 1e3 / queryTotalTime;
        std::cout << "The device search QPS = " << searchQPS << std::endl;
    }

    float CalcRecallRate(int sampleTopN)
    {
        testData.ReadLabelsBase<int64_t>(labelsBaseName, topk, queryNum);
        testData.ReadDistanceBase<float>(labelsDistanceName, topk, queryNum);

        std::vector<float> recalls;
        for (uint64_t i = 0; i < queryNum; i++) {
            bool accCorrect = true;
            std::vector<int64_t> labelsHCP(outLabels.begin() + i * topk, outLabels.begin() + (i + 1) * topk);
            std::vector<int64_t> labelsMindX(testData.queryLabel.begin() + i * topk,
                testData.queryLabel.begin() + (i + 1) * topk);
            std::vector<float> distHCP(outDistances.begin() + i * topk, outDistances.begin() + (i + 1) * topk);
            std::vector<float> distMindX(testData.queryDistances.begin() + i * topk,
                testData.queryDistances.begin() + (i + 1) * topk);
            for (uint32_t j = 0; j < validNums[i]; j++) {
                if (labelsHCP[j] != labelsMindX[j]) {
                    accCorrect = false;
                    break;
                }
            }
            if (accCorrect) {
                recalls.push_back(1.0f);
                std::cout << "[Success] topk Label is the Same!" << std::endl;
            } else {
                std::unordered_map<int64_t, uint32_t> labelsHCPSet =
                    OckTestUtils::VecToMap<int64_t, uint32_t>(labelsHCP);
                auto curSearchRecall = OckTestUtils::CalcRecallWithDist(labelsHCPSet, labelsMindX, distHCP, distMindX,
                    ERRORRANGE, minNNCosValue);
                recalls.push_back(curSearchRecall);
                OckTestUtils::CalcErrorWithDist(distHCP, distMindX, curSearchRecall, ERRORRANGE);
                std::cout << "------------------------------------------" << std::endl;
            }
        }
        float sum = std::accumulate(recalls.begin(), recalls.end(), 0.0);
        float mean = sum / recalls.size();
        float max = *std::max_element(recalls.begin(), recalls.end());
        float min = *std::min_element(recalls.begin(), recalls.end());
        std::cout << "===================================================" << std::endl;
        std::cout << "sampleTopNNum = " << sampleTopN << "Valid recall number = " << recalls.size() <<
            ", mean recall = " << mean << ", max recall = " << max << ", min recall = " << min << std::endl;
        std::cout << "===================================================" << std::endl;

        searchRecall = mean;
        return searchRecall;
    }

    uint32_t addNum{ 128ULL * 2ULL };
    uint64_t ntotalPerAdd{ 262144ULL };
    uint64_t ntotal{ ntotalPerAdd * addNum };
    uint32_t batch{ 1U };
    uint64_t maxFeatureRowCount{ 262144ULL * 128ULL * 8ULL };
    uint32_t deviceId{ 2U };
    cpu_set_t cpuSet;
    uint32_t tokenNum{ 2500 };
    uint32_t extKeyAttrsByteSize{ 22 };
    attr::OckTimeSpaceAttrTrait dftTrait{ tokenNum };
    std::shared_ptr<OckVsaAnnIndexBase<int8_t, DIM_SIZE, 2U, attr::OckTimeSpaceAttrTrait>> indexBase;
    OckTestData testData; // 底库数据与查询数据
    uint32_t topk{ 200 };
    int sampleTopNNum = 1; // sampleTopN 的系数个数·

    // 数据集
    int seed{ test::SEED };
    uint64_t queryNum = 500ULL;
    std::string testDataSetName = "/home/xyz/VGG2C/VGG2C_extended_int8_bin/query.bin";
    std::string trainDataSetName = "/home/liulianguang/data/dataTemp/VGG2C/base.dat";
    uint64_t dataCount = 262144ULL * 12ULL; // 数据集实际条数
    uint64_t dataSetDim = 512;              // 数据集实际维度
    int needRandDim = 5;                    // 数据扩展时修改的维度数
    bool isSave = false;
    std::string randomExtendDataSavePath = "/home/liulianguang/data/dataTemp/VGG2C/tmp/base.dat";

    // 结果数据
    std::vector<int64_t> outLabels;
    std::vector<float> outDistances;
    std::vector<uint32_t> validNums;
    float searchQPS = 0.0f;
    float searchRecall = 0.0f;

    // 比较的结果数据
    std::string labelsBaseName =
        "/home/liulianguang/data/dataTemp/VGG2C/unshuffle0115/labelResMindX_feature67108864_query500_top200.dat";
    std::string labelsDistanceName =
        "/home/liulianguang/data/dataTemp/VGG2C/unshuffle0115/distancesMindX_feature67108864_query500_top200.dat";
    // 标准
    float minNNCosValue = 0.8f; // 小于该值的数据不纳入召回率的计算
    double leastAcc = 0.999;
    double leastQPS = 2.0;

private:
    void InitCpuSetVec()
    {
        CPU_ZERO(&cpuSet);
        for (uint32_t i = 64U; i < 80U; ++i) {
            CPU_SET(i, &cpuSet);
        }
    }
};

TEST_F(TestOckVsaHPPIndex, acc_and_perf)
{
    // 准备数据
    PrepareData();
    // 创建index
    BuildIndexBase();
    // 添加底库
    Add();

    // Search 测试
    float recall = 0.0f;
    for (int n = 1; n <= sampleTopNNum; n++) {
        OckTestUtils::setEnvValue<int>("TOPN_RATIO", n);
        Search();
        recall = CalcRecallRate(n);
        EXPECT_GT(searchRecall, leastAcc);
        EXPECT_GT(searchQPS, leastQPS);
    }

    // SearchWithExtraMask 测试（全 1 下, SearchWithExtraMask 与 Search 的召回率相等）
    SearchWithExtraMask();
    float recallWithExtraMask = CalcRecallRate(sampleTopNNum);
    EXPECT_EQ(recall, recallWithExtraMask);
    EXPECT_GT(searchQPS, leastQPS);
}
} // namespace test
} // namespace neighbor
} // namespace vsa
} // namespace ock