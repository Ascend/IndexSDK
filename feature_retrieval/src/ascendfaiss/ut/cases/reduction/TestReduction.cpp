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


#include <vector>
#include <string>
#include <securec.h>
#include <gtest/gtest.h>
#include <mockcpp/mockcpp.hpp>
#include "ut/Common.h"
#include "faiss/ascend/custom/IReduction.h"
#include "faiss/ascend/AscendNNInference.h"
#include "faiss/ascend/impl/AscendNNInferenceImpl.h"
#include "faiss/ascend/custom/PcarReduction.h"
#include "ascenddaemon/utils/ModelInference.h"
#include "ascenddaemon/utils/ModelExecuter.h"
#include "ascenddaemon/impl/Index.h"
#include "acl/acl.h"

using namespace ascend;
using namespace testing;

namespace {
int StubInfer(ascend::ModelInference *, size_t n, char *inputData, char *outputData)
{
    std::vector<char> inferResult;
    // dimIn / dimOut = 4
    int num = 4;
    for (size_t i = 0; i < n; i++) {
        if (i % num == 0) {
            inferResult.emplace_back(*(inputData + i));
        }
    }
    auto ret = memcpy_s(outputData, n / num, inferResult.data(), n / num);
    EXPECT_EQ(ret, EOK);
    return 0;
}

struct CheckNNInitItem {
    std::vector<int> deviceList;
    char *model;
    uint64_t modelSize;
    std::string str;
    std::string method;
};

struct CheckNNInferItem {
    size_t ntotal;
    float *inputData;
    float *outputData;
    std::string str;
};

struct CheckPcarInitItem {
    int dimIn;
    int dimOut;
    float eigenPower;
    bool randomRotation;
    std::string str;
};

struct CheckPcarTrainItem {
    Index::idx_t n;
    float *x;
    std::string str;
};

struct CheckPcarReduceItem {
    Index::idx_t n;
    float *x;
    float *res;
    std::string str;
};

class TestNNInit : public TestWithParam<CheckNNInitItem> {
};
class TestNNInfer : public TestWithParam<CheckNNInferItem> {
};
class TestPcarInit : public TestWithParam<CheckPcarInitItem> {
};
class TestPcarTrain : public TestWithParam<CheckPcarTrainItem> {
};
class TestPcarReduce : public TestWithParam<CheckPcarReduceItem> {
};

constexpr int DIM_IN = 256;
constexpr int DIM_OUT = 64;
constexpr int BATCH = 64;
constexpr int INPUT_TYPE = 1;
constexpr int OUTPUT_TYPE = 1;
constexpr size_t NTOTAL = 100;
constexpr float EIGENPOWER = 0;
constexpr int DEVICELISTNUM = 33;
std::string g_nnom = "reduce.om";
std::vector<float> data(NTOTAL * DIM_IN);
std::vector<float> output(NTOTAL * DIM_OUT);
std::vector<int> device(DEVICELISTNUM);

const CheckNNInitItem NNINITITEMS[] = {
    { { }, const_cast<char *>(g_nnom.data()), g_nnom.size(), "device list should be > 0 and <= 32", "NN" },
    { device, const_cast<char *>(g_nnom.data()), g_nnom.size(), "device list should be > 0 and <= 32", "NN" },
    { { 0 }, nullptr, g_nnom.size(), "model can not be nullptr", "NN" },
    { { 0 }, const_cast<char *>(g_nnom.data()), 0, "modelSize should be in (0, 128MB].", "NN" },
    { { 0 }, const_cast<char *>(g_nnom.data()), 129 * 1024 * 1024, "modelSize should be in (0, 128MB].", "NN" },
    { { 0, 0 }, const_cast<char *>(g_nnom.data()), g_nnom.size(),
        "some device IDs are the same, please check it {0,0,}", "NN" },
    { { 0 }, const_cast<char *>(g_nnom.data()), g_nnom.size(), "Unsupported typeName, not in {NN, PCAR}.", "N" }
};

const CheckNNInferItem NNINFERITEMS[] = {
    { 100000, nullptr, output.data(), "inputData can not be nullptr" },
    { 100000, data.data(), nullptr, "outputData can not be nullptr" },
    { 0, data.data(), output.data(), "n must be > 0 and < 1000000000" },
    { 1000000000, data.data(), output.data(), "n must be > 0 and < 1000000000" }
};

const CheckPcarInitItem PCARINITITEMS[] = {
    { 256, 0, 0, false, "The output dim of matrix for PCA should be > 0." },
    { 256, 512, 0, false, "The input dim of matrix for PCA should be greater than or equal to output dim." },
    { 256, 64, 1, false, "The value of eigenPower should be >= -0.5 and <= 0." },
    { 256, 64, -1, false, "The value of eigenPower should be >= -0.5 and <= 0." },
    { 256, 0, 0, true, "The output dim of matrix for PCA should be > 0." },
    { 256, 512, 0, true, "The input dim of matrix for PCA should be greater than or equal to output dim." },
    { 256, 64, 1, true, "The value of eigenPower should be >= -0.5 and <= 0." },
    { 256, 64, -1, true, "The value of eigenPower should be >= -0.5 and <= 0." }
};

const CheckPcarTrainItem PCARTRAINITEMS[] = {
    { 100000, nullptr, "x can not be nullptr." },
    { 0, data.data(), "n must be > 0 and < 1000000000" },
    { 1000000001, data.data(), "n must be > 0 and < 1000000000" }
};

const CheckPcarReduceItem PACRREDUCEITEMS[] = {
    { 100000, nullptr, output.data(), "x can not be nullptr." },
    { 100000, data.data(), nullptr, "res can not be nullptr." },
    { 0, data.data(), output.data(), "n must be > 0 and < 1000000000" },
    { 1000000001, data.data(), output.data(), "n must be > 0 and < 1000000000" }
};

TEST_P(TestNNInit, NNInitInvalidInput)
{
    std::string msg;
    for (int i = 0; i < DEVICELISTNUM; i++) {
        device.emplace_back(i);
    }

    CheckNNInitItem item = GetParam();
    std::vector<int> deviceList = item.deviceList;
    char *model = item.model;
    uint64_t modelSize = item.modelSize;
    std::string str = item.str;
    std::string method = item.method;

    try {
        faiss::ascend::ReductionConfig reductionConfig(deviceList, model, modelSize);
        faiss::ascend::IReduction* reduction = CreateReduction(method, reductionConfig);
        delete reduction;
        faiss::ascend::AscendNNInferenceImpl reduceConfig(deviceList, model, modelSize);
    } catch(std::exception &e) {
        msg = e.what();
    }
    EXPECT_TRUE(msg.find(str) != std::string::npos);
}

TEST_P(TestNNInfer, DISABLED_NNInferInvalidInput)
{
    MOCKER_CPP(&aclmdlLoadFromMem).stubs().will(returnValue(0));
    MOCKER_CPP(&aclmdlGetDesc).stubs().will(returnValue(0));
    MOCKER_CPP(&aclmdlGetInputDataType).stubs().will(returnValue(1));
    MOCKER_CPP(&aclmdlGetOutputDataType).stubs().will(returnValue(1));
    MOCKER_CPP(&ascend::ModelExecuter::getInputDim).stubs().with(eq(0), eq(0)).will(returnValue(64));
    MOCKER_CPP(&ascend::ModelExecuter::getInputDim).stubs().with(eq(0), eq(1)).will(returnValue(256));
    MOCKER_CPP(&ascend::ModelExecuter::getOutputDim).stubs().will(returnValue(64));
    faiss::ascend::AscendNNInference dimInference({ 0 }, g_nnom.data(), g_nnom.size());
    FeatureGenerator(data);
    std::string msg;

    CheckNNInferItem item = GetParam();
    size_t ntotal = item.ntotal;
    float *inputData = item.inputData;
    float *outputData = item.outputData;
    std::string str = item.str;

    try {
        dimInference.infer(ntotal, reinterpret_cast<char *>(inputData), reinterpret_cast<char *>(outputData));
    } catch(std::exception &e) {
        msg = e.what();
    }
    EXPECT_TRUE(msg.find(str) != std::string::npos);

    faiss::ascend::ReductionConfig reductionConfig({ 0 }, g_nnom.data(), g_nnom.size());
    std::string method = "NN";
    faiss::ascend::IReduction* reduction = CreateReduction(method, reductionConfig);
    reduction->train(ntotal, inputData);
    try {
        reduction->reduce(ntotal, inputData, outputData);
    } catch(std::exception &e) {
        msg = e.what();
    }
    delete reduction;
    EXPECT_TRUE(msg.find(str) != std::string::npos);
    GlobalMockObject::verify();
}

TEST(TestAscendIReduction, DISABLED_NNReduce)
{
    MOCKER_CPP(&aclmdlLoadFromMem).stubs().will(returnValue(0));
    MOCKER_CPP(&aclmdlGetDesc).stubs().will(returnValue(0));
    MOCKER_CPP(&ascend::ModelInference::Infer).stubs().will(invoke(StubInfer));
    MOCKER_CPP(&aclmdlGetInputDataType).stubs().will(returnValue(1));
    MOCKER_CPP(&aclmdlGetOutputDataType).stubs().will(returnValue(1));
    MOCKER_CPP(&ascend::ModelExecuter::getInputDim).stubs().with(eq(0), eq(0)).will(returnValue(64));
    MOCKER_CPP(&ascend::ModelExecuter::getInputDim).stubs().with(eq(0), eq(1)).will(returnValue(256));
    MOCKER_CPP(&ascend::ModelExecuter::getOutputDim).stubs().will(returnValue(64));

    FeatureGenerator(data);
    // NN IReduction
    faiss::ascend::AscendNNInference dimInference({ 0 }, g_nnom.data(), g_nnom.size());
    std::vector<float> outputData(NTOTAL * DIM_OUT);

    EXPECT_EQ(dimInference.getDimIn(), DIM_IN);
    EXPECT_EQ(dimInference.getInputType(), INPUT_TYPE);
    EXPECT_EQ(dimInference.getDimBatch(), BATCH);
    dimInference.infer(NTOTAL, reinterpret_cast<char *>(data.data()), reinterpret_cast<char *>(outputData.data()));
    EXPECT_EQ(dimInference.getDimOut(), DIM_OUT);
    EXPECT_EQ(dimInference.getOutputType(), OUTPUT_TYPE);

    faiss::ascend::ReductionConfig reductionConfig({ 0 }, g_nnom.data(), g_nnom.size());
    std::vector<float> output(NTOTAL * DIM_OUT);
    std::string method = "NN";
    faiss::ascend::IReduction* reduction = CreateReduction(method, reductionConfig);
    reduction->train(NTOTAL, data.data());
    reduction->reduce(NTOTAL, data.data(), output.data());
    delete reduction;
    GlobalMockObject::verify();
}

TEST_P(TestPcarInit, PcarInitInvalidInput)
{
    CheckPcarInitItem item = GetParam();
    int pcarDimIn = item.dimIn;
    int pcarDimOut = item.dimOut;
    float eigenPower = item.eigenPower;
    bool flag = item.randomRotation;
    std::string str = item.str;

    std::string msg;
    // 输出维度不大于0
    try {
        faiss::ascend::PcarReduction reductionConfig(pcarDimIn, pcarDimOut, eigenPower, flag);
    } catch(std::exception &e) {
        msg = e.what();
    }
    EXPECT_TRUE(msg.find(str) != std::string::npos);
}

TEST_P(TestPcarTrain, PcarTrainInvalidInput)
{
    std::string msg;
    faiss::ascend::PcarReduction pcarConfig(DIM_IN, DIM_OUT, EIGENPOWER, false);

    FeatureGenerator(data);

    CheckPcarTrainItem item = GetParam();
    Index::idx_t n = item.n;
    float *x = item.x;
    std::string str = item.str;

    try {
        pcarConfig.train(n, x);
    } catch(std::exception &e) {
        msg = e.what();
    }
    EXPECT_TRUE(msg.find(str) != std::string::npos);
}

TEST_P(TestPcarReduce, PcarReduceInvalidInput)
{
    std::string msg;
    faiss::ascend::PcarReduction pcarConfig(DIM_IN, DIM_OUT, EIGENPOWER, false);

    FeatureGenerator(data);

    CheckPcarReduceItem item = GetParam();
    Index::idx_t n = item.n;
    float *x = item.x;
    float *res = item.res;
    std::string str = item.str;

    try {
        pcarConfig.reduce(n, x, res);
    } catch(std::exception &e) {
        msg = e.what();
    }
    EXPECT_TRUE(msg.find(str) != std::string::npos);
}

TEST(TestAscendIReduction, PCARReduce)
{
    FeatureGenerator(data);
	// PCAR IReduction
    std::vector<bool> rotation = { true, false };
    for (size_t i = 0; i < rotation.size(); i++) {
        faiss::ascend::ReductionConfig reductionConfig(DIM_IN, DIM_OUT, 0, rotation[i]);
        std::string method = "PCAR";
        faiss::ascend::IReduction* reduction = CreateReduction(method, reductionConfig);
        reduction->train(NTOTAL, data.data());
        std::vector<float> outputData(NTOTAL * DIM_OUT);
        reduction->reduce(NTOTAL, data.data(), outputData.data());
        delete reduction;
    }
}

INSTANTIATE_TEST_CASE_P(IReductionCheckGroup, TestNNInit, ::testing::ValuesIn(NNINITITEMS));
INSTANTIATE_TEST_CASE_P(IReductionCheckGroup, TestNNInfer, ::testing::ValuesIn(NNINFERITEMS));
INSTANTIATE_TEST_CASE_P(IReductionCheckGroup, TestPcarInit, ::testing::ValuesIn(PCARINITITEMS));
INSTANTIATE_TEST_CASE_P(IReductionCheckGroup, TestPcarTrain, ::testing::ValuesIn(PCARTRAINITEMS));
INSTANTIATE_TEST_CASE_P(IReductionCheckGroup, TestPcarReduce, ::testing::ValuesIn(PACRREDUCEITEMS));

}