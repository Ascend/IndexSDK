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
#include <mockcpp/mockcpp.hpp>
#include <map>
#include <thread>
#include "ascendfaiss/ascenddaemon/utils/ModelExecuter.h"

using namespace testing;
using namespace std;

namespace ascend {

class TestModelExecuter : public Test {
public:
    void TearDown() override
    {
        GlobalMockObject::verify();
    }
};

TEST_F(TestModelExecuter, construct)
{
    // 打桩aclmdlLoadFromMem返回1
    MOCKER_CPP(&aclmdlLoadFromMem).stubs().will(returnValue(1))
                                            .then(returnValue(0));
    string actualMsg;
    void *model = nullptr;
    size_t modelSize = 100;
    try {
        ModelExecuter modelExecuter(model, modelSize);
    } catch (std::exception &e) {
        actualMsg = e.what();
    }
    string expectMsg("load model from memory failed");
    EXPECT_TRUE(actualMsg.find(expectMsg) != string::npos);

    // 打桩aclmdlLoadFromMem返回0
    // 打桩aclmdlCreateDesc返回nullptr
    aclmdlDesc *desc = nullptr;
    aclmdlDesc newDesc;
    MOCKER_CPP(&aclmdlCreateDesc).stubs().will(returnValue(desc))
                                            .then(returnValue(&newDesc));

    try {
        ModelExecuter modelExecuter(model, modelSize);
    } catch (std::exception &e) {
        actualMsg = e.what();
    }
    expectMsg = std::string("create model description failed");
    EXPECT_TRUE(actualMsg.find(expectMsg) != string::npos);

    // 打桩aclmdlCreateDesc返回非nullptr
    // 打桩aclmdlGetDesc返回非0
    MOCKER_CPP(&aclmdlGetDesc).stubs().will(returnValue(1))
                                        .then(returnValue(0));
    try {
        ModelExecuter modelExecuter(model, modelSize);
    } catch (std::exception &e) {
        actualMsg = e.what();
    }
    expectMsg = std::string("get model description failed");
    EXPECT_TRUE(actualMsg.find(expectMsg) != string::npos);

    size_t num = 1;
    MOCKER_CPP(&aclmdlGetNumInputs).stubs().will(returnValue(num));
    MOCKER_CPP(&aclmdlGetNumOutputs).stubs().will(returnValue(num));
    MOCKER_CPP(&aclmdlGetInputSizeByIndex).stubs().will(returnValue(num));
    MOCKER_CPP(&aclmdlGetOutputSizeByIndex).stubs().will(returnValue(num));
    MOCKER_CPP(&aclmdlDestroyDesc).stubs().will(returnValue(0));
    MOCKER_CPP(&aclmdlUnload).stubs().will(returnValue(1))
                                        .then(returnValue(0));

    ModelExecuter modelExecuter(model, modelSize);
}

TEST_F(TestModelExecuter, getInputNumDims)
{
    // 打桩aclmdlLoadFromMem返回0
    MOCKER_CPP(&aclmdlLoadFromMem).stubs().will(returnValue(0));
    // 打桩aclmdlCreateDesc返回非nullptr
    aclmdlDesc newDesc;
    MOCKER_CPP(&aclmdlCreateDesc).stubs().will(returnValue(&newDesc));
    // 打桩aclmdlGetDesc返回0
    MOCKER_CPP(&aclmdlGetDesc).stubs().will(returnValue(0));
    
    size_t dims = 0;
    // 打桩aclmdlGetNumInputs返回0
    MOCKER_CPP(&aclmdlGetNumInputs).stubs().will(returnValue(dims));
    // 打桩aclmdlGetNumOutputs返回0
    MOCKER_CPP(&aclmdlGetNumOutputs).stubs().will(returnValue(dims));
    // 析构没有问题
    MOCKER_CPP(&aclmdlUnload).stubs().will(returnValue(0));
    MOCKER_CPP(&aclmdlDestroyDesc).stubs().will(returnValue(0));
    // 打桩aclmdlGetInputDims返回非0
    MOCKER_CPP(&aclmdlGetInputDims).stubs().will(returnValue(1))
                                            .then(returnValue(0));
    void *model = nullptr;
    size_t modelSize = 100;
    string actualMsg;
    int index = 0;
    try {
        ModelExecuter modelExecuter(model, modelSize);
        modelExecuter.getInputNumDims(index);
    } catch (std::exception &e) {
        actualMsg = e.what();
    }
    std::string expectMsg = std::string("get input dims failed");
    EXPECT_TRUE(actualMsg.find(expectMsg) != string::npos);

    actualMsg = "";
    try {
        ModelExecuter modelExecuter(model, modelSize);
        modelExecuter.getInputNumDims(index);
    } catch (std::exception &e) {
        actualMsg = e.what();
    }
    EXPECT_TRUE(actualMsg.empty());
}

TEST_F(TestModelExecuter, deconstruct)
{
    void *model;
    size_t modelSize = 100;
    string actualMsg;
    // 打桩aclmdlLoadFromMem返回0
    MOCKER_CPP(&aclmdlLoadFromMem).stubs().will(returnValue(0));
    // 打桩aclmdlCreateDesc返回非nullptr
    aclmdlDesc newDesc;
    MOCKER_CPP(&aclmdlCreateDesc).stubs().will(returnValue(&newDesc));
    // 打桩aclmdlGetDesc返回0
    MOCKER_CPP(&aclmdlGetDesc).stubs().will(returnValue(0));
    
    size_t dims = 0;
    // 打桩aclmdlGetNumInputs返回0
    MOCKER_CPP(&aclmdlGetNumInputs).stubs().will(returnValue(dims));
    // 打桩aclmdlGetNumOutputs返回0
    MOCKER_CPP(&aclmdlGetNumOutputs).stubs().will(returnValue(dims));
    // 析构没有问题
    MOCKER_CPP(&aclmdlUnload).stubs().will(returnValue(1));
    MOCKER_CPP(&aclmdlDestroyDesc).stubs().will(returnValue(0));
    ModelExecuter modelExecuter(model, modelSize);
}

TEST_F(TestModelExecuter, getOutputNumDims)
{
    // 打桩aclmdlLoadFromMem返回0
    MOCKER_CPP(&aclmdlLoadFromMem).stubs().will(returnValue(0));
    // 打桩aclmdlCreateDesc返回非nullptr
    aclmdlDesc newDesc;
    MOCKER_CPP(&aclmdlCreateDesc).stubs().will(returnValue(&newDesc));
    // 打桩aclmdlGetDesc返回0
    MOCKER_CPP(&aclmdlGetDesc).stubs().will(returnValue(0));
    
    size_t dims = 0;
    // 打桩aclmdlGetNumInputs返回0
    MOCKER_CPP(&aclmdlGetNumInputs).stubs().will(returnValue(dims));
    // 打桩aclmdlGetNumOutputs返回0
    MOCKER_CPP(&aclmdlGetNumOutputs).stubs().will(returnValue(dims));
    // 析构没有问题
    MOCKER_CPP(&aclmdlUnload).stubs().will(returnValue(0));
    MOCKER_CPP(&aclmdlDestroyDesc).stubs().will(returnValue(0));
    // 打桩aclmdlGetOutputDims返回非0
    MOCKER_CPP(&aclmdlGetOutputDims).stubs().will(returnValue(1))
                                            .then(returnValue(0));

    void *model = nullptr;
    size_t modelSize = 100;
    string actualMsg;
    int index = 0;
    try {
        ModelExecuter modelExecuter(model, modelSize);
        modelExecuter.getOutputNumDims(index);
    } catch (std::exception &e) {
        actualMsg = e.what();
    }
    std::string expectMsg = std::string("get output dims failed");
    EXPECT_TRUE(actualMsg.find(expectMsg) != string::npos);

    actualMsg = "";
    try {
        ModelExecuter modelExecuter(model, modelSize);
        modelExecuter.getOutputNumDims(index);
    } catch (std::exception &e) {
        actualMsg = e.what();
    }
    EXPECT_TRUE(actualMsg.empty());
}

static aclmdlIODims g_stubDims;
static aclError g_stubAclmdlGetDimsRet = ACL_SUCCESS;
aclError StubAclmdlGetDims(const aclmdlDesc *modelDesc, size_t index, aclmdlIODims *dims)
{
    *dims = g_stubDims;
    return g_stubAclmdlGetDimsRet;
}

TEST_F(TestModelExecuter, getInputDim1)
{
    // 打桩aclmdlLoadFromMem返回0
    MOCKER_CPP(&aclmdlLoadFromMem).stubs().will(returnValue(0));
    // 打桩aclmdlCreateDesc返回非nullptr
    aclmdlDesc newDesc;
    MOCKER_CPP(&aclmdlCreateDesc).stubs().will(returnValue(&newDesc));
    // 打桩aclmdlGetDesc返回0
    MOCKER_CPP(&aclmdlGetDesc).stubs().will(returnValue(0));

    size_t dims = 0;
    // 打桩aclmdlGetNumInputs返回0
    MOCKER_CPP(&aclmdlGetNumInputs).stubs().will(returnValue(dims));
    // 打桩aclmdlGetNumOutputs返回0
    MOCKER_CPP(&aclmdlGetNumOutputs).stubs().will(returnValue(dims));
    // 析构没有问题
    MOCKER_CPP(&aclmdlUnload).stubs().will(returnValue(0));
    MOCKER_CPP(&aclmdlDestroyDesc).stubs().will(returnValue(0));
    // 打桩aclmdlGetOutputDims返回非0
    MOCKER_CPP(&aclmdlGetOutputDims).stubs().will(returnValue(1))
                                            .then(returnValue(0));
    void *model = nullptr;
    size_t modelSize = 100;
    string actualMsg;
    size_t index = 0;
    size_t dimIndex = 0;
    {
        g_stubDims.dimCount = 0;
        g_stubAclmdlGetDimsRet = ACL_ERROR_FAILURE;
        MOCKER(aclmdlGetInputDims).stubs().will(invoke(StubAclmdlGetDims));
        try {
            ModelExecuter modelExecuter(model, modelSize);
            modelExecuter.getInputDim(index, dimIndex);
        } catch (std::exception &e) {
            actualMsg = e.what();
        }
        std::string expectMsg = std::string("get input dims failed");
        EXPECT_TRUE(actualMsg.find(expectMsg) != string::npos);
    }

    {
        g_stubDims.dimCount = 0;
        g_stubAclmdlGetDimsRet = ACL_SUCCESS;
        MOCKER(aclmdlGetInputDims).stubs().will(invoke(StubAclmdlGetDims));
        try {
            ModelExecuter modelExecuter(model, modelSize);
            modelExecuter.getInputDim(index, dimIndex);
        } catch (std::exception &e) {
            actualMsg = e.what();
        }
        std::string expectMsg = std::string("get input dim failed");
        EXPECT_TRUE(actualMsg.find(expectMsg) != string::npos);
    }
}

TEST_F(TestModelExecuter, getInputDim2)
{
    // 打桩aclmdlLoadFromMem返回0
    MOCKER_CPP(&aclmdlLoadFromMem).stubs().will(returnValue(0));
    // 打桩aclmdlCreateDesc返回非nullptr
    aclmdlDesc newDesc;
    MOCKER_CPP(&aclmdlCreateDesc).stubs().will(returnValue(&newDesc));
    // 打桩aclmdlGetDesc返回0
    MOCKER_CPP(&aclmdlGetDesc).stubs().will(returnValue(0));

    size_t dims = 0;
    // 打桩aclmdlGetNumInputs返回0
    MOCKER_CPP(&aclmdlGetNumInputs).stubs().will(returnValue(dims));
    // 打桩aclmdlGetNumOutputs返回0
    MOCKER_CPP(&aclmdlGetNumOutputs).stubs().will(returnValue(dims));
    // 析构没有问题
    MOCKER_CPP(&aclmdlUnload).stubs().will(returnValue(0));
    MOCKER_CPP(&aclmdlDestroyDesc).stubs().will(returnValue(0));
    // 打桩aclmdlGetOutputDims返回非0
    MOCKER_CPP(&aclmdlGetOutputDims).stubs().will(returnValue(1))
                                            .then(returnValue(0));
    void *model = nullptr;
    size_t modelSize = 100;
    string actualMsg;
    size_t index = 0;
    size_t dimIndex = 0;
    g_stubDims.dimCount = 1;
    g_stubAclmdlGetDimsRet = ACL_SUCCESS;
    MOCKER(aclmdlGetInputDims).stubs().will(invoke(StubAclmdlGetDims));
    try {
        ModelExecuter modelExecuter(model, modelSize);
        modelExecuter.getInputDim(index, dimIndex);
    } catch (std::exception &e) {
        actualMsg = e.what();
    }
    EXPECT_TRUE(actualMsg.empty());
}

TEST_F(TestModelExecuter, getOutputDim1)
{
    // 打桩aclmdlLoadFromMem返回0
    MOCKER_CPP(&aclmdlLoadFromMem).stubs().will(returnValue(0));
    // 打桩aclmdlCreateDesc返回非nullptr
    aclmdlDesc newDesc;
    MOCKER_CPP(&aclmdlCreateDesc).stubs().will(returnValue(&newDesc));
    // 打桩aclmdlGetDesc返回0
    MOCKER_CPP(&aclmdlGetDesc).stubs().will(returnValue(0));

    size_t dims = 0;
    // 打桩aclmdlGetNumInputs返回0
    MOCKER_CPP(&aclmdlGetNumInputs).stubs().will(returnValue(dims));
    // 打桩aclmdlGetNumOutputs返回0
    MOCKER_CPP(&aclmdlGetNumOutputs).stubs().will(returnValue(dims));
    // 析构没有问题
    MOCKER_CPP(&aclmdlUnload).stubs().will(returnValue(0));
    MOCKER_CPP(&aclmdlDestroyDesc).stubs().will(returnValue(0));

    void *model = nullptr;
    size_t modelSize = 100;
    string actualMsg;
    size_t index = 0;
    size_t dimIndex = 0;
    {
        g_stubDims.dimCount = 0;
        g_stubAclmdlGetDimsRet = ACL_ERROR_FAILURE;
        MOCKER(aclmdlGetOutputDims).stubs().will(invoke(StubAclmdlGetDims));
        try {
            ModelExecuter modelExecuter(model, modelSize);
            modelExecuter.getOutputDim(index, dimIndex);
        } catch (std::exception &e) {
            actualMsg = e.what();
        }
        std::string expectMsg = std::string("get output dims failed");
        EXPECT_TRUE(actualMsg.find(expectMsg) != string::npos);
    }

    {
        g_stubDims.dimCount = 0;
        g_stubAclmdlGetDimsRet = ACL_SUCCESS;
        MOCKER(aclmdlGetOutputDims).stubs().will(invoke(StubAclmdlGetDims));
        try {
            ModelExecuter modelExecuter(model, modelSize);
            modelExecuter.getOutputDim(index, dimIndex);
        } catch (std::exception &e) {
            actualMsg = e.what();
        }
        std::string expectMsg = std::string("get output dim failed");
        EXPECT_TRUE(actualMsg.find(expectMsg) != string::npos);
    }
}

TEST_F(TestModelExecuter, getOutputDim2)
{
    // 打桩aclmdlLoadFromMem返回0
    MOCKER_CPP(&aclmdlLoadFromMem).stubs().will(returnValue(0));
    // 打桩aclmdlCreateDesc返回非nullptr
    aclmdlDesc newDesc;
    MOCKER_CPP(&aclmdlCreateDesc).stubs().will(returnValue(&newDesc));
    // 打桩aclmdlGetDesc返回0
    MOCKER_CPP(&aclmdlGetDesc).stubs().will(returnValue(0));

    size_t dims = 0;
    // 打桩aclmdlGetNumInputs返回0
    MOCKER_CPP(&aclmdlGetNumInputs).stubs().will(returnValue(dims));
    // 打桩aclmdlGetNumOutputs返回0
    MOCKER_CPP(&aclmdlGetNumOutputs).stubs().will(returnValue(dims));
    // 析构没有问题
    MOCKER_CPP(&aclmdlUnload).stubs().will(returnValue(0));
    MOCKER_CPP(&aclmdlDestroyDesc).stubs().will(returnValue(0));

    void *model = nullptr;
    size_t modelSize = 100;
    string actualMsg;
    size_t index = 0;
    size_t dimIndex = 0;

    g_stubDims.dimCount = 1;
    g_stubAclmdlGetDimsRet = ACL_SUCCESS;
    MOCKER(aclmdlGetOutputDims).stubs().will(invoke(StubAclmdlGetDims));
    try {
        ModelExecuter modelExecuter(model, modelSize);
        modelExecuter.getOutputDim(index, dimIndex);
    } catch (std::exception &e) {
        actualMsg = e.what();
    }
    EXPECT_TRUE(actualMsg.empty());
}

TEST_F(TestModelExecuter, releaseResource)
{
    // 打桩aclmdlLoadFromMem返回0
    MOCKER_CPP(&aclmdlLoadFromMem).stubs().will(returnValue(0));
    // 打桩aclmdlCreateDesc返回非nullptr
    aclmdlDesc newDesc;
    MOCKER_CPP(&aclmdlCreateDesc).stubs().will(returnValue(&newDesc));
    // 打桩aclmdlGetDesc返回0
    MOCKER_CPP(&aclmdlGetDesc).stubs().will(returnValue(0));

    size_t dims = 0;
    MOCKER_CPP(&aclmdlGetNumInputs).stubs().will(returnValue(dims));
    MOCKER_CPP(&aclmdlGetNumOutputs).stubs().will(returnValue(dims));
    // 析构没有问题
    MOCKER_CPP(&aclmdlUnload).stubs().will(returnValue(0));
    MOCKER_CPP(&aclmdlDestroyDesc).stubs().will(returnValue(0));
    MOCKER_CPP(&aclDestroyDataBuffer).stubs().will(returnValue(0));
    MOCKER_CPP(&aclmdlDestroyDataset).stubs().will(returnValue(0));

    void *model = nullptr;
    size_t modelSize = 100;
    ModelExecuter modelExecuter(model, modelSize);

    aclmdlDataset *inputPtr = nullptr;
    aclmdlDataset *outputPtr = nullptr;
    aclDataBuffer *inputDbPtr = nullptr;
    aclDataBuffer *outputDbPtr = nullptr;

    aclmdlDataset input;
    aclmdlDataset output;
    aclDataBuffer inputDb;
    aclDataBuffer outputDb;
    modelExecuter.releaseResource(inputPtr, outputPtr, inputDbPtr, outputDbPtr);
    modelExecuter.releaseResource(&input, outputPtr, inputDbPtr, outputDbPtr);
    modelExecuter.releaseResource(&input, &output, inputDbPtr, outputDbPtr);
    modelExecuter.releaseResource(&input, &output, &inputDb, outputDbPtr);
    modelExecuter.releaseResource(&input, &output, &inputDb, &outputDb);

    size_t index = 0;
    modelExecuter.getInputDataType(index);
    modelExecuter.getOutputDataType(index);
}

TEST_F(TestModelExecuter, execute1)
{
    void *model = nullptr;
    size_t modelSize = 100;
    string actualMsg;
    // 打桩aclmdlLoadFromMem返回0
    MOCKER_CPP(&aclmdlLoadFromMem).stubs().will(returnValue(0));
    // 打桩aclmdlCreateDesc返回非nullptr
    aclmdlDesc newDesc;
    MOCKER_CPP(&aclmdlCreateDesc).stubs().will(returnValue(&newDesc));
    // 打桩aclmdlGetDesc返回0
    MOCKER_CPP(&aclmdlGetDesc).stubs().will(returnValue(0));
    
    size_t dims = 0;
    // 打桩aclmdlGetNumInputs返回0
    MOCKER_CPP(&aclmdlGetNumInputs).stubs().will(returnValue(dims));
    // 打桩aclmdlGetNumOutputs返回0
    MOCKER_CPP(&aclmdlGetNumOutputs).stubs().will(returnValue(dims));
    // 析构没有问题
    MOCKER_CPP(&aclmdlUnload).stubs().will(returnValue(0));
    MOCKER_CPP(&aclmdlDestroyDesc).stubs().will(returnValue(0));

    aclmdlDataset *dataPtr = nullptr;
    aclmdlDataset dataset;
    ModelExecuter modelExecuter(model, modelSize);
    MOCKER_CPP(&aclmdlCreateDataset).stubs().will(returnValue(dataPtr))
                                            .then(returnValue(&dataset))
                                            .then(returnValue(dataPtr))
                                            .then(returnValue(&dataset));

    MOCKER(&ModelExecuter::releaseResource).stubs();
    aclDataBuffer *bufferPtr = nullptr;
    MOCKER_CPP(&aclCreateDataBuffer).stubs().will(returnValue(bufferPtr));

    void *inputData = nullptr;
    void *outputData = nullptr;
    try {
        modelExecuter.execute(inputData, outputData);
    } catch (std::exception &e) {
        actualMsg = e.what();
    }
    std::string expectMsg = std::string("create intput dataset failed");
    EXPECT_TRUE(actualMsg.find(expectMsg) != string::npos);

    try {
        modelExecuter.execute(inputData, outputData);
    } catch (std::exception &e) {
        actualMsg = e.what();
    }
    expectMsg = std::string("create output dataset failed");
    EXPECT_TRUE(actualMsg.find(expectMsg) != string::npos);

    modelExecuter.inputSizes.resize(1);
    try {
        modelExecuter.execute(inputData, outputData);
    } catch (std::exception &e) {
        actualMsg = e.what();
    }
    expectMsg = std::string("create input data buffer failed");
    EXPECT_TRUE(actualMsg.find(expectMsg) != string::npos);
}

TEST_F(TestModelExecuter, execute2)
{
    void *model = nullptr;
    string actualMsg;
    // 打桩aclmdlLoadFromMem返回0
    MOCKER_CPP(&aclmdlLoadFromMem).stubs().will(returnValue(0));
    // 打桩aclmdlCreateDesc返回非nullptr
    aclmdlDesc newDesc;
    MOCKER_CPP(&aclmdlCreateDesc).stubs().will(returnValue(&newDesc));
    // 打桩aclmdlGetDesc返回0
    MOCKER_CPP(&aclmdlGetDesc).stubs().will(returnValue(0));
    
    size_t dims = 0;
    // 打桩aclmdlGetNumInputs返回0
    MOCKER_CPP(&aclmdlGetNumInputs).stubs().will(returnValue(dims));
    // 打桩aclmdlGetNumOutputs返回0
    MOCKER_CPP(&aclmdlGetNumOutputs).stubs().will(returnValue(dims));
    // 析构没有问题
    MOCKER_CPP(&aclmdlUnload).stubs().will(returnValue(0));
    MOCKER_CPP(&aclmdlDestroyDesc).stubs().will(returnValue(0));

    aclmdlDataset dataset;
    ModelExecuter modelExecuter(model, 0);
    modelExecuter.inputSizes.resize(1);
    modelExecuter.outputSizes.resize(1);
    MOCKER_CPP(&aclmdlCreateDataset).stubs().will(returnValue(&dataset));

    MOCKER(&ModelExecuter::releaseResource).stubs();
    aclDataBuffer *bufferPtr = nullptr;
    aclDataBuffer newBufferPtr;
    MOCKER_CPP(&aclCreateDataBuffer).stubs().will(returnValue(&newBufferPtr))
                                            .then(returnValue(bufferPtr))
                                            .then(returnValue(&newBufferPtr));

    MOCKER_CPP(&aclmdlAddDatasetBuffer).stubs().will(returnValue(1));

    void *inputData = nullptr;
    void *outputData = nullptr;
    try {
        modelExecuter.execute(inputData, outputData);
    } catch (std::exception &e) {
        actualMsg = e.what();
    }
    std::string expectMsg = std::string("create output data buffer failed");
    EXPECT_TRUE(actualMsg.find(expectMsg) != string::npos);

    try {
        modelExecuter.execute(inputData, outputData);
    } catch (std::exception &e) {
        actualMsg = e.what();
    }
    expectMsg = std::string("add input dataset buffer failed");
    EXPECT_TRUE(actualMsg.find(expectMsg) != string::npos);
}

TEST_F(TestModelExecuter, execute3)
{
    void *model = nullptr;
    string actualMsg;
    // 打桩aclmdlLoadFromMem返回0
    MOCKER_CPP(&aclmdlLoadFromMem).stubs().will(returnValue(0));
    // 打桩aclmdlCreateDesc返回非nullptr
    aclmdlDesc newDesc;
    MOCKER_CPP(&aclmdlCreateDesc).stubs().will(returnValue(&newDesc));
    // 打桩aclmdlGetDesc返回0
    MOCKER_CPP(&aclmdlGetDesc).stubs().will(returnValue(0));
    
    size_t dims = 0;
    // 打桩aclmdlGetNumInputs返回0
    MOCKER_CPP(&aclmdlGetNumInputs).stubs().will(returnValue(dims));
    // 打桩aclmdlGetNumOutputs返回0
    MOCKER_CPP(&aclmdlGetNumOutputs).stubs().will(returnValue(dims));
    // 析构没有问题
    MOCKER_CPP(&aclmdlUnload).stubs().will(returnValue(0));
    MOCKER_CPP(&aclmdlDestroyDesc).stubs().will(returnValue(0));

    MOCKER_CPP(&aclDestroyDataBuffer).stubs().will(returnValue(0));
    MOCKER_CPP(&aclmdlDestroyDataset).stubs().will(returnValue(0));

    aclmdlDataset dataset;
    ModelExecuter modelExecuter(model, 0);
    modelExecuter.inputSizes.resize(1);
    modelExecuter.outputSizes.resize(1);
    MOCKER_CPP(&aclmdlCreateDataset).stubs().will(returnValue(&dataset));

    MOCKER(&ModelExecuter::releaseResource).stubs();
    aclDataBuffer newBufferPtr;
    MOCKER_CPP(&aclCreateDataBuffer).stubs().will(returnValue(&newBufferPtr));

    MOCKER_CPP(&aclmdlAddDatasetBuffer).stubs().will(returnValue(0))
                                                .then(returnValue(1))
                                                .then(returnValue(0));
    MOCKER_CPP(&aclmdlExecute).stubs().will(returnValue(1)).then(returnValue(0));
    void *inputData = nullptr;
    void *outputData = nullptr;
    try {
        modelExecuter.execute(inputData, outputData);
    } catch (std::exception &e) {
        actualMsg = e.what();
    }
    std::string expectMsg = std::string("add output dataset buffer failed");
    EXPECT_TRUE(actualMsg.find(expectMsg) != string::npos);

    try {
        modelExecuter.execute(inputData, outputData);
    } catch (std::exception &e) {
        actualMsg = e.what();
    }
    expectMsg = std::string("execute model failed, modelId");
    EXPECT_TRUE(actualMsg.find(expectMsg) != string::npos);

    actualMsg = "";
    try {
        modelExecuter.execute(inputData, outputData);
    } catch (std::exception &e) {
        actualMsg = e.what();
    }
    EXPECT_TRUE(actualMsg.empty());
}
} // namespace ascend