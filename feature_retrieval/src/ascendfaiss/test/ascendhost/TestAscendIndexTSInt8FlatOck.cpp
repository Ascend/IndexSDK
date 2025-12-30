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
#include <iostream>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <random>
#include <sys/time.h>
#include <unistd.h>
#include <vector>
#include "faiss/ascend/AscendIndexTS.h"
#include "faiss/ascend/custom/IReduction.h"
#include "faiss/ascend/AscendIndexFlat.h"
#include "faiss/ascend/AscendIndexInt8Flat.h"
#include "TestData.h"
#include "securec.h"

using namespace std;
using namespace faiss::ascend;

namespace {

independent_bits_engine<std::mt19937, 8, uint8_t> bitsEngine(1);
string g_dataDir;
string g_nnModelPath;
unique_ptr<TestData> g_testData;
constexpr int g_deviceId = 0;
constexpr uint32_t g_tokenNum = 2500;
constexpr size_t g_deviceCapacity = static_cast<size_t>(1) * 1024 * 1024 * 1024;
constexpr size_t g_deviceBuffer = static_cast<size_t>(9) * 64 * 1024 * 1024;
constexpr size_t g_hostCapacity = static_cast<size_t>(20) * 1024 * 1024 * 1024;
constexpr size_t g_reduceDimIn = 512;
constexpr size_t g_reduceDimOut = 256;
constexpr size_t KB = 1024;

void FeatureGenerator(std::vector<int8_t> &features)
{
    size_t n = features.size();
    for (size_t i = 0; i < n; ++i) {
        features[i] = bitsEngine() - 128;
    }
}

// 之前用例均默认为4取余
void FeatureAttrGenerator(std::vector<FeatureAttr> &attrs, int32_t power = 4)
{
    size_t n = attrs.size();
    for (size_t i = 0; i < n; ++i) {
        attrs[i].time = int32_t(i % power);
        attrs[i].tokenId = int32_t(i % power);
    }
}

string GetNNOmStr(const string &modelPath)
{
    std::ifstream ifs(modelPath.c_str(), std::ios::binary);
    std::stringstream buffer;
    buffer << ifs.rdbuf();
    std::string nnOm(buffer.str());
    ifs.close();
    return nnOm;
}

void LoadTestData()
{
    static bool loadFlag = false;
    if (!loadFlag) {
        string baseFile = g_dataDir + "base.bin";
        string trainFile = g_dataDir + "base.bin";
        string queryFile = g_dataDir + "query.bin";
        string gtFile = g_dataDir + "gt.bin";
        g_testData = make_unique<TestDataSift1B>(baseFile, trainFile, queryFile, gtFile);
        g_testData->SetDim(g_reduceDimIn);
        printf("check dim[%d], start load data!\r\n", g_testData->GetDim());
        g_testData->SetScale(0);
        g_testData->FetchData();
        printf("load data end!\r\n");
        loadFlag = true;
    }
}

pair<vector<int8_t>, vector<int8_t>> ReduceData(faiss::ascend::IReduction* reduction)
{
    vector<float> reduceOutBases(g_testData->GetBaseSize() * g_reduceDimOut);
    reduction->reduce(g_testData->GetBaseSize(), g_testData->GetBase().data(), reduceOutBases.data());
    float maxValue = *std::max_element(reduceOutBases.begin(), reduceOutBases.end());
    float minValue = *std::min_element(reduceOutBases.begin(), reduceOutBases.end());
    float valRange = maxValue - minValue;
    cout << "get maxVal[" << to_string(maxValue) << "]" << endl;
    cout << "get minVal[" << to_string(minValue) << "]" << endl;
    vector<int8_t> reduceOutNormBases(reduceOutBases.size());
    auto float2Int8 = [&valRange] (float x) {
        return x / valRange * 127.0;
    };
    std::transform(reduceOutBases.begin(), reduceOutBases.end(), reduceOutNormBases.begin(), float2Int8);
    reduceOutBases.clear();

    vector<float> reduceOutQuerys(g_testData->GetQuerySize() * g_reduceDimOut);
    reduction->reduce(g_testData->GetQuerySize(), g_testData->GetQuery().data(), reduceOutQuerys.data());
    vector<int8_t> reduceOutNormQuerys(reduceOutQuerys.size());
    std::transform(reduceOutQuerys.begin(), reduceOutQuerys.end(), reduceOutNormQuerys.begin(), float2Int8);
    reduceOutQuerys.clear();

    return make_pair(move(reduceOutNormBases), move(reduceOutNormQuerys));
}

void AddFeatureAttrByPower(faiss::ascend::AscendIndexTS &tsIndex, const vector<int8_t> &reduceOutNormBases, int32_t power = 4)
{
    vector<int64_t> label(g_testData->GetBaseSize());
    for (size_t i = 0; i < label.size(); i++) {
        label[i] = i;
    }
    vector<FeatureAttr> attrs(g_testData->GetBaseSize());
    FeatureAttrGenerator(attrs, power);
    int64_t addLeft = g_testData->GetBaseSize();
    int64_t addNum = 0;
    while(addLeft > 0) {
        int64_t curAddNum = std::min(static_cast<int64_t>(1000000), addLeft);
        auto ret = tsIndex.AddFeature(curAddNum, reduceOutNormBases.data() + (addNum * g_reduceDimOut),
            attrs.data() + addNum, label.data() + addNum);
        ASSERT_EQ(ret, 0);
        addNum += curAddNum;
        addLeft -= curAddNum;
        cout << "addNum[" << addNum << "]" << endl;
    }
}

void TrainPcarReduction(faiss::ascend::IReduction* reduction)
{
    uniform_int_distribution<uint32_t> distrb(0, 3302154);
    default_random_engine e;
    int trainNum = 100000;
    vector<float> trainData(trainNum * g_reduceDimIn);
    for (int i = 0; i < trainNum; i++) {
        uint32_t randVal = distrb(e);
        (void)memcpy_s(trainData.data() + i * g_reduceDimIn, (trainData.size() - i * g_reduceDimIn) * sizeof(float),
            g_testData->GetBase().data() + randVal * g_reduceDimIn, g_reduceDimIn * sizeof(float));
    }
    reduction->train(trainNum, trainData.data());
}

pair<vector<int8_t>, vector<int8_t>> ReduceDataAndQuantBySq(faiss::ascend::IReduction* reduction)
{
    vector<float> reduceOutBases(g_testData->GetBaseSize() * g_reduceDimOut);
    reduction->reduce(g_testData->GetBaseSize(), g_testData->GetBase().data(), reduceOutBases.data());
    faiss::ScalarQuantizer sq(g_reduceDimOut, faiss::ScalarQuantizer::QuantizerType::QT_8bit);;
    sq.train(g_testData->GetBaseSize(), reduceOutBases.data());
    cout << "sq dim[" << sq.d << "] [" << sq.code_size << "]" << endl;
    vector<int8_t> reduceOutNormBases(reduceOutBases.size());
    vector<uint8_t> tempOutBases(reduceOutBases.size());
    sq.compute_codes(reduceOutBases.data(), tempOutBases.data(), g_testData->GetBaseSize());
    auto uint8Toint8 = [] (auto val) -> int8_t {
        return static_cast<int8_t>(static_cast<int32_t>(val) - 128);
    };
    transform(tempOutBases.begin(), tempOutBases.end(), reduceOutNormBases.data(), uint8Toint8);
    reduceOutBases.clear();
    tempOutBases.clear();

    vector<float> reduceOutQuerys(g_testData->GetQuerySize() * g_reduceDimOut);
    reduction->reduce(g_testData->GetQuerySize(), g_testData->GetQuery().data(), reduceOutQuerys.data());
    vector<int8_t> reduceOutNormQuerys(reduceOutQuerys.size());
    vector<uint8_t> tempOutQuerys(reduceOutQuerys.size());
    sq.compute_codes(reduceOutQuerys.data(), tempOutQuerys.data(), g_testData->GetQuerySize());
    transform(tempOutQuerys.begin(), tempOutQuerys.end(), reduceOutNormQuerys.data(), uint8Toint8);
    reduceOutQuerys.clear();
    tempOutQuerys.clear();

    return make_pair(move(reduceOutNormBases), move(reduceOutNormQuerys));
}

void SearchNoMask(faiss::ascend::AscendIndexTS &tsIndex, const vector<int8_t> &query)
{
    uint32_t setlen = static_cast<uint32_t>((g_tokenNum + 7) / 8);
    vector<uint8_t> bitSet(setlen, 0xff);
    AttrFilter filter;
    filter.timesStart = 0;
    filter.timesEnd = 3;
    filter.tokenBitSet = bitSet.data();
    filter.tokenBitSetLen = bitSet.size();
    vector<AttrFilter> queryFilters(g_testData->GetQuerySize(), filter);
    vector<float> distances(g_testData->GetQuerySize() * g_testData->GetGTK(), -1);
    vector<int64_t> labelRes(g_testData->GetQuerySize() * g_testData->GetGTK(), -1);
    vector<uint32_t> validnum(g_testData->GetQuerySize(), 1);
    tsIndex.Search(g_testData->GetQuerySize(), query.data(), queryFilters.data(), false, g_testData->GetGTK(), labelRes.data(),
        distances.data(), validnum.data());
    g_testData->EvaluateResult(labelRes);
}

void CheckExtraMaskResult(faiss::ascend::AscendIndexTS &tsIndex, const vector<int8_t> &reduceOutNormQuerys, const vector<AttrFilter> &queryFilters,
    const vector<int64_t> &labelRes, uint8_t maskVal)
{
    size_t maskLen = (g_testData->GetBaseSize() + 7) / 8; 
    cout << "***** start search with extra mask " << to_string(maskVal) << " *****" << endl;
    vector<uint8_t> extraMask254(maskLen * g_testData->GetQuerySize(), maskVal);
    vector<int64_t> labelResMask(g_testData->GetQuerySize() * g_testData->GetGTK(), -1);
    vector<float> distances(g_testData->GetQuerySize() * g_testData->GetGTK(), -1);
    vector<uint32_t> validnum(g_testData->GetQuerySize(), 1);
    auto ret = tsIndex.SearchWithExtraMask(g_testData->GetQuerySize(), reduceOutNormQuerys.data(), queryFilters.data(), false, g_testData->GetGTK(), extraMask254.data(),
        maskLen, false, labelResMask.data(), distances.data(), validnum.data());
    ASSERT_EQ(ret, 0);
    size_t validNumAll = 0;
    for (size_t iq = 0; iq < g_testData->GetQuerySize(); iq++) {
        for (size_t ik = 0; ik < static_cast<size_t>(g_testData->GetGTK()); ik++) {
            int64_t curGtLabel = labelRes[iq * g_testData->GetGTK() + ik];
            uint8_t idTemp = curGtLabel % 8;
            idTemp = 1 << idTemp;
            // 被extraMask过滤
            if ((idTemp & maskVal) == 0) {
                continue;
            }
            validNumAll++;
            auto it = find(labelResMask.begin() + iq * g_testData->GetGTK(), labelResMask.begin() + (iq + 1) * g_testData->GetGTK(), curGtLabel);
            if (it == labelResMask.begin() + (iq + 1) * g_testData->GetGTK()) {
                printf("Error! Label[%zu][%zu] = %ld not found!\r\n", iq, ik, curGtLabel);
                ASSERT_TRUE(false);
                return;
            }
        }
    }
    printf("check result success, validNumAll[%zu], acturalNum[%zu], filterPower[%f]\r\n", validNumAll, labelRes.size(), static_cast<float>(validNumAll)/labelRes.size());
    cout << "***** end search with extra mask " << to_string(maskVal) << " *****" << endl;
}

} // end of namespace

TEST(TestAscendIndexTSInt8FlatOck, SetHeteroParam_range)
{
    struct InputOutput {
        size_t deviceCapacity { 0 };
        size_t deviceBuffer { 0 };
        size_t hostCapacity { 0 };
        int ret { 0 };
    };

    const int innerErrCode = 2010;
    const int paramErrCode = 2001;
    const size_t minDeviceBuffer = 262144 * g_reduceDimOut * 2;
    vector<InputOutput> inputOutputs {
        { 0, g_deviceBuffer, g_hostCapacity, innerErrCode },
        { KB * KB * KB - 1, g_deviceBuffer, g_hostCapacity, innerErrCode },
        { KB * KB * KB, g_deviceBuffer, g_hostCapacity, 0 },
        { g_deviceCapacity, g_deviceBuffer, g_hostCapacity, 0 },
        { 60 * KB * KB * KB, g_deviceBuffer, g_hostCapacity, innerErrCode },

        { g_deviceCapacity, 0, g_hostCapacity, paramErrCode },
        { g_deviceCapacity, minDeviceBuffer - 1, g_hostCapacity, paramErrCode },
        { g_deviceCapacity, minDeviceBuffer, g_hostCapacity, 0 },
        { g_deviceCapacity, g_deviceBuffer, g_hostCapacity, 0 },
        { g_deviceCapacity, 50 * KB * KB * KB, g_hostCapacity, innerErrCode },

        { g_deviceCapacity, g_deviceBuffer, 0, innerErrCode },
        { g_deviceCapacity, g_deviceBuffer, KB * KB * KB - 1, innerErrCode },
        { g_deviceCapacity, g_deviceBuffer, KB * KB * KB, 0 },
        { g_deviceCapacity, g_deviceBuffer, g_hostCapacity, 0 },
        { g_deviceCapacity, g_deviceBuffer, 501 * KB * KB * KB, innerErrCode },

        { 25 * KB * KB * KB, 25 * KB * KB * KB, g_hostCapacity, innerErrCode },
        { 8 * KB * KB * KB, 8 * KB * KB * KB, g_hostCapacity, 0 },
    };

    unique_ptr<AscendIndexTS> tsIndex;
    int ret = 0;
    for (const auto &io : inputOutputs) {
        if (ret == 0) {
            tsIndex = make_unique<AscendIndexTS>();
            ret = tsIndex->Init(g_deviceId, g_reduceDimOut, g_tokenNum, AlgorithmType::FLAT_COS_INT8,
                MemoryStrategy::HETERO_MEMORY);
            ASSERT_EQ(ret, 0);
        }
        ret = tsIndex->SetHeteroParam(io.deviceCapacity, io.deviceBuffer, io.hostCapacity);
        ASSERT_TRUE(ret == io.ret) << "real ret:" << ret << " expect ret:" << io.ret << " deviceCapcity:" <<
            io.deviceCapacity << " deviceBuffer:" << io.deviceBuffer << " hostCapacity:" << io.hostCapacity;
    }
}

TEST(TestAscendIndexTSInt8FlatOck, SetHeteroParam)
{
    unique_ptr<AscendIndexTS> tsIndex = make_unique<AscendIndexTS>();
    // 未init不能设置异构参数
    auto ret = tsIndex->SetHeteroParam(53125423, 1000000, g_hostCapacity);
    ASSERT_NE(ret, 0);

    // 未设置异构模式，依旧不能设置异构参数
    tsIndex->Init(g_deviceId, g_reduceDimOut, g_tokenNum, AlgorithmType::FLAT_COS_INT8, MemoryStrategy::PURE_DEVICE_MEMORY);
    ret = tsIndex->SetHeteroParam(53125423, 1000000, g_hostCapacity);
    ASSERT_NE(ret, 0);

    tsIndex = make_unique<AscendIndexTS>();
    tsIndex->Init(g_deviceId, g_reduceDimOut, g_tokenNum, AlgorithmType::FLAT_COS_INT8, MemoryStrategy::HETERO_MEMORY);
    // 未成功设置参数，不能使用add接口
    vector<int8_t> feature(256);
    FeatureAttr attr { 1, 5 };
    int64_t label = 0;
    ret = tsIndex->AddFeature(1, feature.data(), &attr, &label);
    ASSERT_NE(ret, 0);

    // 设置合理内存，设置成功
    size_t deviceCapacity = static_cast<size_t>(20) * g_reduceDimOut * 262144;
    size_t deviceBuffer= static_cast<size_t>(3) * g_reduceDimOut * 262144;
    ret = tsIndex->SetHeteroParam(deviceCapacity, deviceBuffer, g_hostCapacity);
    ASSERT_EQ(ret, 0);
    // 设置完成以后，无法再次设置
    ret = tsIndex->SetHeteroParam(deviceCapacity, deviceBuffer, g_hostCapacity);
    ASSERT_NE(ret, 0);
    // 设置完成以后，可以执行add
    ret = tsIndex->AddFeature(1, feature.data(), &attr, &label);
    ASSERT_EQ(ret, 0);

    tsIndex = make_unique<AscendIndexTS>();
    // 不支持其他算法使用异构内存策略
    ret = tsIndex->Init(g_deviceId, g_reduceDimOut, g_tokenNum, AlgorithmType::FLAT_HAMMING, MemoryStrategy::HETERO_MEMORY);
    ASSERT_NE(ret, 0);
    ret = tsIndex->Init(g_deviceId, g_reduceDimOut, g_tokenNum, AlgorithmType::FLAT_IP_FP16, MemoryStrategy::HETERO_MEMORY);
    ASSERT_NE(ret, 0);
}

TEST(TestAscendIndexTSInt8FlatOck, Int8flat_Vgg2C_NoMask)
{
    LoadTestData();

    cout << "***** start reduce *****" << endl;
    vector<int8_t> int8DataBases(g_testData->GetBaseSize() * g_reduceDimIn);
    auto float2Int8 = [] (float x) {
        return static_cast<int8_t>(x);
    };
    std::transform(g_testData->GetBase().begin(), g_testData->GetBase().end(), int8DataBases.begin(), float2Int8);

    vector<int8_t> int8Querys(g_testData->GetQuerySize() * g_reduceDimIn);
    std::transform(g_testData->GetQuery().begin(), g_testData->GetQuery().end(), int8Querys.begin(), float2Int8);
    cout << "***** end reduce *****" << endl;

    cout << "***** start create index *****" << endl;
    faiss::ascend::AscendIndexInt8FlatConfig conf( {g_deviceId} );
    faiss::ascend::AscendIndexInt8Flat index(g_reduceDimIn, faiss::METRIC_INNER_PRODUCT, conf);
    cout << "***** end create index *****" << endl;

    cout << "***** start add *****" << endl;
    index.add(g_testData->GetBaseSize(), int8DataBases.data());
    cout << "***** end add *****" << endl;

    vector<float> distances(g_testData->GetQuerySize() * g_testData->GetGTK(), -1);
    vector<int64_t> labelRes(g_testData->GetQuerySize() * g_testData->GetGTK(), -1);
    cout << "***** start search *****" << endl;
    index.search(g_testData->GetQuerySize(), int8Querys.data(), g_testData->GetGTK(), distances.data(), labelRes.data());
    cout << "***** end search *****" << endl;

    g_testData->EvaluateResult(labelRes);
}

TEST(TestAscendIndexTSInt8FlatOck, PCAR_Reduce_Ts_Ock_Vgg2C_NoMask)
{
    LoadTestData();

    cout << "***** start create IReduction *****" << endl;
    faiss::ascend::ReductionConfig reductionConfig(g_reduceDimIn, g_reduceDimOut, 0, true);
    faiss::ascend::IReduction* reduction = CreateReduction("PCAR", reductionConfig);
    TrainPcarReduction(reduction);
    cout << "***** end create IReduction *****" << endl;

    cout << "***** start reduce *****" << endl;
    auto [reduceOutNormBases, reduceOutNormQuerys] = ReduceData(reduction);
    delete reduction;
    cout << "***** end reduce *****" << endl;

    cout << "***** start create index *****" << endl;
    faiss::ascend::AscendIndexTS tsIndex;
    auto ret = tsIndex.Init(g_deviceId, g_reduceDimOut, g_tokenNum, faiss::ascend::AlgorithmType::FLAT_COS_INT8, faiss::ascend::MemoryStrategy::HETERO_MEMORY);
    ASSERT_EQ(ret, 0);
    ret = tsIndex.SetHeteroParam(g_deviceCapacity, g_deviceBuffer, g_hostCapacity);
    ASSERT_EQ(ret, 0);
    cout << "***** end create index *****" << endl;

    cout << "***** start add *****" << endl;
    AddFeatureAttrByPower(tsIndex, reduceOutNormBases);
    cout << "***** end add *****" << endl;

    cout << "***** start search *****" << endl;
    SearchNoMask(tsIndex, reduceOutNormQuerys);
    cout << "***** end search *****" << endl;
}

TEST(TestAscendIndexTSInt8FlatOck, PCAR_Reduce_SQ_Ts_Ock_Vgg2C_NoMask)
{
    LoadTestData();

    cout << "***** start create IReduction *****" << endl;
    faiss::ascend::ReductionConfig reductionConfig(g_reduceDimIn, g_reduceDimOut, 0, true);
    faiss::ascend::IReduction* reduction = CreateReduction("PCAR", reductionConfig);
    TrainPcarReduction(reduction);
    cout << "***** end create IReduction *****" << endl;


    cout << "***** start reduce *****" << endl;
    auto [reduceOutNormBases, reduceOutNormQuerys] = ReduceDataAndQuantBySq(reduction);
    delete reduction;
    cout << "***** end reduce *****" << endl;

    cout << "***** start create index *****" << endl;
    faiss::ascend::AscendIndexTS tsIndex;
    auto ret = tsIndex.Init(g_deviceId, g_reduceDimOut, g_tokenNum, faiss::ascend::AlgorithmType::FLAT_COS_INT8, faiss::ascend::MemoryStrategy::HETERO_MEMORY);
    ASSERT_EQ(ret, 0);
    ret = tsIndex.SetHeteroParam(g_deviceCapacity, g_deviceBuffer, g_hostCapacity);
    ASSERT_EQ(ret, 0);
    cout << "***** end create index *****" << endl;

    cout << "***** start add *****" << endl;
    AddFeatureAttrByPower(tsIndex, reduceOutNormBases);
    cout << "***** end add *****" << endl;

    cout << "***** start search *****" << endl;
    SearchNoMask(tsIndex, reduceOutNormQuerys);
    cout << "***** end search *****" << endl;

}

TEST(TestAscendIndexTSInt8FlatOck, NN_Reduce_SQ_Ts_Ock_Vgg2C_NoMask)
{
    LoadTestData();

    cout << "***** start load reduce om model *****" << endl;
    string om = GetNNOmStr(g_nnModelPath);
    faiss::ascend::ReductionConfig reductionConfig({g_deviceId}, om.data(), om.size());
    faiss::ascend::IReduction* reduction = CreateReduction("NN", reductionConfig);
    cout << "***** end load reduce om model *****" << endl;

    cout << "***** start reduce *****" << endl;
    auto [reduceOutNormBases, reduceOutNormQuerys] = ReduceDataAndQuantBySq(reduction);
    delete reduction;
    cout << "***** end reduce *****" << endl;

    cout << "***** start create index *****" << endl;
    faiss::ascend::AscendIndexTS tsIndex;
    auto ret = tsIndex.Init(g_deviceId, g_reduceDimOut, g_tokenNum, faiss::ascend::AlgorithmType::FLAT_COS_INT8, faiss::ascend::MemoryStrategy::HETERO_MEMORY);
    ASSERT_EQ(ret, 0);
    ret = tsIndex.SetHeteroParam(g_deviceCapacity, g_deviceBuffer, g_hostCapacity);
    ASSERT_EQ(ret, 0);
    cout << "***** end create index *****" << endl;

    cout << "***** start add *****" << endl;
    AddFeatureAttrByPower(tsIndex, reduceOutNormBases);
    cout << "***** end add *****" << endl;

    cout << "***** start search *****" << endl;
    SearchNoMask(tsIndex, reduceOutNormQuerys);
    cout << "***** end search *****" << endl;
}


TEST(TestAscendIndexTSInt8FlatOck, NN_Reduce_Ts_Ock_Vgg2C_NoMask)
{
    LoadTestData();

    cout << "***** start load reduce om model *****" << endl;
    string om = GetNNOmStr(g_nnModelPath);
    faiss::ascend::ReductionConfig reductionConfig({g_deviceId}, om.data(), om.size());
    faiss::ascend::IReduction* reduction = CreateReduction("NN", reductionConfig);
    cout << "***** end load reduce om model *****" << endl;

    cout << "***** start reduce *****" << endl;
    auto [reduceOutNormBases, reduceOutNormQuerys] = ReduceData(reduction);
    delete reduction;
    cout << "***** end reduce *****" << endl;

    cout << "***** start create index *****" << endl;
    faiss::ascend::AscendIndexTS tsIndex;
    auto ret = tsIndex.Init(g_deviceId, g_reduceDimOut, g_tokenNum, faiss::ascend::AlgorithmType::FLAT_COS_INT8, faiss::ascend::MemoryStrategy::HETERO_MEMORY);
    ASSERT_EQ(ret, 0);
    ret = tsIndex.SetHeteroParam(g_deviceCapacity, g_deviceBuffer, g_hostCapacity);
    ASSERT_EQ(ret, 0);
    cout << "***** end create index *****" << endl;

    cout << "***** start add *****" << endl;
    AddFeatureAttrByPower(tsIndex, reduceOutNormBases);
    cout << "***** end add *****" << endl;

    cout << "***** start search *****" << endl;
    SearchNoMask(tsIndex, reduceOutNormQuerys);
    cout << "***** end search *****" << endl;
}


TEST(TestAscendIndexTSInt8FlatOck, NN_Reduce_Ts_Ock_Vgg2C_Mask)
{
    LoadTestData();

    cout << "***** start load reduce om model *****" << endl;
    string om = GetNNOmStr(g_nnModelPath);
    faiss::ascend::ReductionConfig reductionConfig({g_deviceId}, om.data(), om.size());
    faiss::ascend::IReduction* reduction = CreateReduction("NN", reductionConfig);
    cout << "***** end load reduce om model *****" << endl;

    cout << "***** start reduce *****" << endl;
    auto [reduceOutNormBases, reduceOutNormQuerys] = ReduceData(reduction);
    delete reduction;
    cout << "***** end reduce *****" << endl;

    cout << "***** start create index *****" << endl;
    faiss::ascend::AscendIndexTS tsIndex;
    auto ret = tsIndex.Init(g_deviceId, g_reduceDimOut, g_tokenNum, faiss::ascend::AlgorithmType::FLAT_COS_INT8, faiss::ascend::MemoryStrategy::HETERO_MEMORY);
    ASSERT_EQ(ret, 0);
    ret = tsIndex.SetHeteroParam(g_deviceCapacity, g_deviceBuffer, g_hostCapacity);
    ASSERT_EQ(ret, 0);
    cout << "***** end create index *****" << endl;

    cout << "***** start add *****" << endl;
    int power = 10;
    AddFeatureAttrByPower(tsIndex, reduceOutNormBases, power);
    cout << "***** end add *****" << endl;

    cout << "***** start search *****" << endl;
    uint32_t setlen = static_cast<uint32_t>((g_tokenNum + 7) / 8);
    vector<uint8_t> bitSet(setlen, 0xff);
    AttrFilter filter;
    filter.timesStart = 0;
    filter.timesEnd = 9;
    filter.tokenBitSet = bitSet.data();
    filter.tokenBitSetLen = bitSet.size();
    vector<AttrFilter> queryFilters(g_testData->GetQuerySize(), filter);
    vector<float> distances(g_testData->GetQuerySize() * g_testData->GetGTK(), -1);
    vector<int64_t> labelRes(g_testData->GetQuerySize() * g_testData->GetGTK(), -1);
    vector<uint32_t> validnum(g_testData->GetQuerySize(), 1);
    tsIndex.Search(g_testData->GetQuerySize(), reduceOutNormQuerys.data(), queryFilters.data(), false, g_testData->GetGTK(), labelRes.data(),
        distances.data(), validnum.data());
    g_testData->EvaluateResult(labelRes);
    cout << "***** end search *****" << endl;


    cout << "***** start search with mask *****" << endl;
    // 前面生成的tokenId 是0/1/2/3.../9, 这里过滤成1/2/3..../9，第一个bit是11111110
    vector<uint8_t> tokenBitSet(setlen, 0xff);
    tokenBitSet[0] = 0b11111110;
    cout << "first bit[" << to_string(tokenBitSet[0]) << "] should be 254" << endl;
    filter.timesStart = 0;
    filter.timesEnd = 8;
    filter.tokenBitSet = tokenBitSet.data();
    filter.tokenBitSetLen = tokenBitSet.size();
    vector<AttrFilter> queryFiltersMask(g_testData->GetQuerySize(), filter);
    vector<float> distancesMask(g_testData->GetQuerySize() * g_testData->GetGTK(), -1);
    vector<int64_t> labelResMask(g_testData->GetQuerySize() * g_testData->GetGTK(), -1);
    vector<uint32_t> validnumMask(g_testData->GetQuerySize(), 1);
    tsIndex.Search(g_testData->GetQuerySize(), reduceOutNormQuerys.data(), queryFiltersMask.data(), false, g_testData->GetGTK(), labelResMask.data(),
        distancesMask.data(), validnumMask.data());
    size_t validNumAll = 0;
    for (size_t iq = 0; iq < g_testData->GetQuerySize(); iq++) {
        for (size_t ik = 0; ik < static_cast<size_t>(g_testData->GetGTK()); ik++) {
            int64_t curGtLabel = labelRes[iq * g_testData->GetGTK() + ik];
            int64_t idTemp = curGtLabel % power;
            // 0是因为被时间过滤，9是被token过滤
            if ((idTemp == 0) || (idTemp == 9)) {
                continue;
            }
            validNumAll++;
            auto it = find(labelResMask.begin() + iq * g_testData->GetGTK(), labelResMask.begin() + (iq + 1) * g_testData->GetGTK(), curGtLabel);
            if (it == labelResMask.begin() + (iq + 1) * g_testData->GetGTK()) {
                printf("Error! Label[%zu][%zu] = %ld not found!\r\n", iq, ik, curGtLabel);
                ASSERT_TRUE(false);
                return;
            }
        }
    }
    printf("check result success, validNumAll[%zu], acturalNum[%zu], filterPower[%lf]\r\n", validNumAll, labelRes.size(), static_cast<float>(validNumAll)/labelRes.size());
    cout << "***** end search with mask *****" << endl;
}

TEST(TestAscendIndexTSInt8FlatOck, NN_Reduce_Ts_Vgg2C_NoMask)
{
    LoadTestData();

    cout << "***** start load reduce om model *****" << endl;
    string om = GetNNOmStr(g_nnModelPath);
    faiss::ascend::ReductionConfig reductionConfig({g_deviceId}, om.data(), om.size());
    faiss::ascend::IReduction* reduction = CreateReduction("NN", reductionConfig);
    cout << "***** end load reduce om model *****" << endl;

    cout << "***** start reduce *****" << endl;
    auto [reduceOutNormBases, reduceOutNormQuerys] = ReduceData(reduction);
    delete reduction;
    cout << "***** end reduce *****" << endl;

    cout << "***** start create index *****" << endl;
    faiss::ascend::AscendIndexTS tsIndex;
    auto ret = tsIndex.Init(g_deviceId, g_reduceDimOut, g_tokenNum, faiss::ascend::AlgorithmType::FLAT_COS_INT8);
    ASSERT_EQ(ret, 0);
    cout << "***** end create index *****" << endl;

    cout << "***** start add *****" << endl;
    AddFeatureAttrByPower(tsIndex, reduceOutNormBases);
    cout << "***** end add *****" << endl;

    cout << "***** start search *****" << endl;
    SearchNoMask(tsIndex, reduceOutNormQuerys);
    cout << "***** end search *****" << endl;
}

TEST(TestAscendIndexTSInt8FlatOck, Ts_Vgg2C_NoMask)
{
    LoadTestData();

    cout << "***** start reduce *****" << endl;
    vector<int8_t> int8DataBases(g_testData->GetBaseSize() * g_reduceDimIn);
    auto float2Int8 = [] (float x) {
        return static_cast<int8_t>(x);
    };
    std::transform(g_testData->GetBase().begin(), g_testData->GetBase().end(), int8DataBases.begin(), float2Int8);

    vector<int8_t> int8Querys(g_testData->GetQuerySize() * g_reduceDimIn);
    std::transform(g_testData->GetQuery().begin(), g_testData->GetQuery().end(), int8Querys.begin(), float2Int8);
    cout << "***** end reduce *****" << endl;

    cout << "***** start create index *****" << endl;
    faiss::ascend::AscendIndexTS tsIndex;
    auto ret = tsIndex.Init(g_deviceId, g_reduceDimIn, g_tokenNum, faiss::ascend::AlgorithmType::FLAT_COS_INT8);
    ASSERT_EQ(ret, 0);
    cout << "***** end create index *****" << endl;

    cout << "***** start add *****" << endl;
    {
        vector<int64_t> label(g_testData->GetBaseSize());
        for (size_t i = 0; i < label.size(); i++) {
            label[i] = i;
        }
        vector<FeatureAttr> attrs(g_testData->GetBaseSize());
        FeatureAttrGenerator(attrs);
        int64_t addLeft = g_testData->GetBaseSize();
        int64_t addNum = 0;
        while(addLeft > 0) {
            int64_t curAddNum = std::min(static_cast<int64_t>(1000000), addLeft);
            tsIndex.AddFeature(curAddNum, int8DataBases.data() + (addNum * g_reduceDimIn),
                attrs.data() + addNum, label.data() + addNum);
            addNum += curAddNum;
            addLeft -= curAddNum;
            cout << "addNum[" << addNum << "]" << endl;
        }
    }
    cout << "***** end add *****" << endl;

    cout << "***** start search *****" << endl;
    SearchNoMask(tsIndex, int8Querys);
    cout << "***** end search *****" << endl;
}

TEST(TestAscendIndexTSInt8FlatOck, NN_Reduce_Ts_Ock_Vgg2C_ExtraMask)
{
    LoadTestData();

    cout << "***** start load reduce om model *****" << endl;
    string om = GetNNOmStr(g_nnModelPath);
    faiss::ascend::ReductionConfig reductionConfig({g_deviceId}, om.data(), om.size());
    faiss::ascend::IReduction* reduction = CreateReduction("NN", reductionConfig);
    cout << "***** end load reduce om model *****" << endl;

    cout << "***** start reduce *****" << endl;
    auto [reduceOutNormBases, reduceOutNormQuerys] = ReduceData(reduction);
    delete reduction;
    cout << "***** end reduce *****" << endl;

    cout << "***** start create index *****" << endl;
    faiss::ascend::AscendIndexTS tsIndex;
    auto ret = tsIndex.Init(g_deviceId, g_reduceDimOut, g_tokenNum, faiss::ascend::AlgorithmType::FLAT_COS_INT8, faiss::ascend::MemoryStrategy::HETERO_MEMORY);
    ASSERT_EQ(ret, 0);
    ret = tsIndex.SetHeteroParam(g_deviceCapacity, g_deviceBuffer, g_hostCapacity);
    ASSERT_EQ(ret, 0);
    cout << "***** end create index *****" << endl;

    cout << "***** start add *****" << endl;
    int power = 10;
    AddFeatureAttrByPower(tsIndex, reduceOutNormBases, power);
    cout << "***** end add *****" << endl;

    cout << "***** start search with extra mask 255 *****" << endl;
    uint32_t setlen = static_cast<uint32_t>((g_tokenNum + 7) / 8);
    vector<uint8_t> bitSet(setlen, 0xff);
    AttrFilter filter;
    filter.timesStart = 0;
    filter.timesEnd = 9;
    filter.tokenBitSet = bitSet.data();
    filter.tokenBitSetLen = bitSet.size();
    vector<AttrFilter> queryFilters(g_testData->GetQuerySize(), filter);
    vector<float> distances(g_testData->GetQuerySize() * g_testData->GetGTK(), -1);
    vector<int64_t> labelRes(g_testData->GetQuerySize() * g_testData->GetGTK(), -1);
    vector<uint32_t> validnum(g_testData->GetQuerySize(), 1);
    size_t maskLen = (g_testData->GetBaseSize() + 7) / 8;
    vector<uint8_t> extraMask(maskLen * g_testData->GetQuerySize(), 0xff);
    ret = tsIndex.SearchWithExtraMask(g_testData->GetQuerySize(), reduceOutNormQuerys.data(), queryFilters.data(), false, g_testData->GetGTK(), extraMask.data(),
        maskLen, false, labelRes.data(), distances.data(), validnum.data());
    ASSERT_EQ(ret, 0);
    g_testData->EvaluateResult(labelRes);
    cout << "***** end search with extra mask 255 *****" << endl;

    CheckExtraMaskResult(tsIndex, reduceOutNormQuerys, queryFilters, labelRes, 0x7e);
    CheckExtraMaskResult(tsIndex, reduceOutNormQuerys, queryFilters, labelRes, 0xae);
    CheckExtraMaskResult(tsIndex, reduceOutNormQuerys, queryFilters, labelRes, 0x59);
    CheckExtraMaskResult(tsIndex, reduceOutNormQuerys, queryFilters, labelRes, 0x3b);
    CheckExtraMaskResult(tsIndex, reduceOutNormQuerys, queryFilters, labelRes, 0x16);
}

TEST(TestAscendIndexTSInt8FlatOck, GetFeatureNum)
{
    int64_t ntotal = 1000000;
    uint32_t addTime = 30;
    uint32_t dim = 256;
    uint32_t tokenNum = 2500;

    cout << "***** start create index *****" << endl;
    faiss::ascend::AscendIndexTS tsIndex;
    auto ret = tsIndex.Init(g_deviceId, dim, tokenNum, faiss::ascend::AlgorithmType::FLAT_COS_INT8, faiss::ascend::MemoryStrategy::HETERO_MEMORY);
    ASSERT_EQ(ret, 0);
    ret = tsIndex.SetHeteroParam(g_deviceCapacity, g_deviceBuffer, g_hostCapacity);
    ASSERT_EQ(ret, 0);
    cout << "***** end create index *****" << endl;

    std::vector<int8_t> features(ntotal * dim);
    printf("[add -----------]\n");
    FeatureGenerator(features);
    std::vector<FeatureAttr> attrs(ntotal);
    FeatureAttrGenerator(attrs);

    int64_t addTotal = 0;
    vector<uint32_t> addNumVec = {
        5, 89, 101, 1543, 14234, 3421, 2, 98745, 177777, 684795,
        1, 10, 100, 1000, 10000, 100000, 1000000, 999999, 888888, 777777,
        456456, 456456, 789789, 789789, 123123, 123123, 1, 1, 2, 3
    };
    for (size_t i = 0; i < addTime; i++) {
        uint32_t curAddNum = addNumVec[i];
        vector<int64_t> labels;
        for (int64_t j = addTotal; j < addTotal + curAddNum; j++) {
            labels.push_back(j);
        }

        tsIndex.AddFeature(curAddNum, features.data(), attrs.data(), labels.data());
        addTotal += curAddNum;
        int64_t resNum = 0;
        auto res = tsIndex.GetFeatureNum(&resNum);
        EXPECT_EQ(res, 0);
        EXPECT_EQ(addTotal, resNum);
    }
}

TEST(TestAscendIndexTSInt8FlatOck, GetFeatureByLabel)
{
    int64_t ntotal = 1000000;
    uint32_t addNum = 10;
    uint32_t dim = 256;
    uint32_t tokenNum = 2500;

    cout << "***** start create index *****" << endl;
    faiss::ascend::AscendIndexTS tsIndex;
    auto ret = tsIndex.Init(g_deviceId, dim, tokenNum, faiss::ascend::AlgorithmType::FLAT_COS_INT8, faiss::ascend::MemoryStrategy::HETERO_MEMORY);
    ASSERT_EQ(ret, 0);
    ret = tsIndex.SetHeteroParam(g_deviceCapacity, g_deviceBuffer, g_hostCapacity);
    ASSERT_EQ(ret, 0);
    cout << "***** end create index *****" << endl;

    std::vector<int8_t> features(ntotal * dim);
    printf("[add -----------]\n");
    FeatureGenerator(features);
    for (size_t i = 0; i < addNum; i++) {
        std::vector<int64_t> labels;
        for (int64_t j = 0; j < ntotal; ++j) {
            labels.emplace_back(j + i * ntotal);
        }
        std::vector<FeatureAttr> attrs(ntotal);
        FeatureAttrGenerator(attrs);
        auto ts0 = TestData::GetMillisecs();
        tsIndex.AddFeature(ntotal, features.data(), attrs.data(), labels.data());
        auto te0 = TestData::GetMillisecs();
        printf("add %ld cost %f ms\n", ntotal, te0 - ts0);
    }

    vector<int64_t> labels;
    for (int64_t i = ntotal * addNum - 1; i >= 0; i-=262144) {
        labels.emplace_back(i);
    }
    vector<int8_t> featuresOut(labels.size() * dim, -1);
    ret = tsIndex.GetFeatureByLabel(labels.size(), labels.data(), featuresOut.data());
    ASSERT_EQ(ret, 0);
    for (size_t ilb = 0; ilb < labels.size(); ilb++) {
        auto lb = labels[ilb];
        int64_t lb100 = lb % ntotal;
        for (uint32_t i = 0; i < dim; i++) {
            if (features[lb100 * dim + i] != featuresOut[ilb * dim + i]) {
                ASSERT_EQ(features[lb100 * dim + i], featuresOut[ilb * dim + i]);
            }
        }
    }
}

bool IsAttrSame(const FeatureAttr &lAttr, const FeatureAttr &rAttr)
{
    return (lAttr.time == rAttr.time) && (lAttr.tokenId == rAttr.tokenId);
}

TEST(TestAscendIndexTSInt8FlatOck, GetFeatureAttrByLabel)
{
    int64_t ntotal = 1000000;
    uint32_t addNum = 10;
    uint32_t dim = 256;
    uint32_t tokenNum = 2500;

    cout << "***** start create index *****" << endl;
    faiss::ascend::AscendIndexTS tsIndex;
    auto ret = tsIndex.Init(g_deviceId, dim, tokenNum, faiss::ascend::AlgorithmType::FLAT_COS_INT8, faiss::ascend::MemoryStrategy::HETERO_MEMORY);
    ASSERT_EQ(ret, 0);
    ret = tsIndex.SetHeteroParam(g_deviceCapacity, g_deviceBuffer, g_hostCapacity);
    ASSERT_EQ(ret, 0);
    cout << "***** end create index *****" << endl;

    std::vector<int8_t> features(ntotal * dim);
    printf("[add -----------]\n");
    FeatureGenerator(features);
    std::vector<FeatureAttr> attrs(ntotal);
    FeatureAttrGenerator(attrs, 321);
    for (size_t i = 0; i < addNum; i++) {
        std::vector<int64_t> labels;
        for (int64_t j = 0; j < ntotal; ++j) {
            labels.emplace_back(j + i * ntotal);
        }
        auto ts0 = TestData::GetMillisecs();
        tsIndex.AddFeature(ntotal, features.data(), attrs.data(), labels.data());
        auto te0 = TestData::GetMillisecs();
        printf("add %ld cost %f ms\n", ntotal, te0 - ts0);
    }

    vector<int64_t> labels;
    for (int64_t i = ntotal * addNum - 1; i >= 0; i-=262144) {
        labels.emplace_back(i);
    }
    vector<FeatureAttr> attrsOut(labels.size());
    ret = tsIndex.GetFeatureAttrByLabel(labels.size(), labels.data(), attrsOut.data());
    ASSERT_EQ(ret, 0);
    for (size_t ilb = 0; ilb < labels.size(); ilb++) {
        auto lb = labels[ilb];
        int64_t lb100 = lb % 1000000;
        ASSERT_TRUE(IsAttrSame(attrs[lb100], attrsOut[ilb]));
    }
}

TEST(TestAscendIndexTSInt8FlatOck, DeleteFeatureByLabel)
{
    int64_t ntotal = 1000000;
    uint32_t addNum = 10;
    uint32_t dim = 256;
    faiss::ascend::AscendIndexTS tsIndex;
    auto ret = tsIndex.Init(g_deviceId, dim, g_tokenNum, faiss::ascend::AlgorithmType::FLAT_COS_INT8, faiss::ascend::MemoryStrategy::HETERO_MEMORY);
    ASSERT_EQ(ret, 0);
    ret = tsIndex.SetHeteroParam(g_deviceCapacity, g_deviceBuffer, g_hostCapacity);
    ASSERT_EQ(ret, 0);

    std::vector<int8_t> features(ntotal * dim);
    printf("[add -----------]\n");
    FeatureGenerator(features);
    std::vector<FeatureAttr> attrs(ntotal);
    FeatureAttrGenerator(attrs, 312);
    for (size_t i = 0; i < addNum; i++) {
        std::vector<int64_t> labels;
        for (int64_t j = 0; j < ntotal; ++j) {
            labels.emplace_back(j + i * ntotal);
        }
        auto ts0 = TestData::GetMillisecs();
        tsIndex.AddFeature(ntotal, features.data(), attrs.data(), labels.data());
        auto te0 = TestData::GetMillisecs();
        printf("add %ld cost %f ms\n", ntotal, te0 - ts0);
    }

    int64_t resNum = 0;
    ret = tsIndex.GetFeatureNum(&resNum);
    ASSERT_EQ(ret, 0);
    ASSERT_EQ(resNum, ntotal * addNum);

    // 删除dev侧内存
    int64_t delLabel = 5;
    ret = tsIndex.DeleteFeatureByLabel(1, &delLabel);
    ASSERT_EQ(ret, 0);

    ret = tsIndex.GetFeatureNum(&resNum);
    ASSERT_EQ(ret, 0);
    ASSERT_EQ(resNum, ntotal * addNum - 1);

    // 删除dev侧buffer内存
    delLabel = ntotal * addNum - 3;
    ret = tsIndex.DeleteFeatureByLabel(1, &delLabel);
    ASSERT_EQ(ret, 0);

    ret = tsIndex.GetFeatureNum(&resNum);
    ASSERT_EQ(ret, 0);
    ASSERT_EQ(resNum, ntotal * addNum - 2);

    // 删除host侧内存
    delLabel = ntotal * addNum - 3 - 262144;
    ret = tsIndex.DeleteFeatureByLabel(1, &delLabel);
    ASSERT_EQ(ret, 0);

    ret = tsIndex.GetFeatureNum(&resNum);
    ASSERT_EQ(ret, 0);
    ASSERT_EQ(resNum, ntotal * addNum - 3);

    vector<int64_t> labels;
    for (int64_t i = ntotal * addNum - 1; i > 0; i-=262144) {
        labels.emplace_back(i);
        labels.emplace_back(i - 1);
    }
    vector<int8_t> featuresOut(labels.size() * dim, -1);
    ret = tsIndex.GetFeatureByLabel(labels.size(), labels.data(), featuresOut.data());
    ASSERT_EQ(ret, 0);
    for (size_t ilb = 0; ilb < labels.size(); ilb++) {
        auto lb = labels[ilb];
        int64_t lb100 = lb % 1000000;
        for (uint32_t i = 0; i < dim; i++) {
            if (features[lb100 * dim + i] != featuresOut[ilb * dim + i]) {
                printf("ilb[%zu], lb[%ld], lb100[%ld], i[%u]\r\n", ilb, lb, lb100, i);
                ASSERT_EQ(features[lb100 * dim + i], featuresOut[ilb * dim + i]);
            }
        }
    }
}

TEST(TestAscendIndexTSInt8FlatOck, DeleteFeatureByToken)
{
    int64_t ntotal = 1000000;
    uint32_t addNum = 10;
    uint32_t dim = 256;
    faiss::ascend::AscendIndexTS tsIndex;
    auto ret = tsIndex.Init(g_deviceId, dim, g_tokenNum, faiss::ascend::AlgorithmType::FLAT_COS_INT8, faiss::ascend::MemoryStrategy::HETERO_MEMORY);
    ASSERT_EQ(ret, 0);
    ret = tsIndex.SetHeteroParam(g_deviceCapacity, g_deviceBuffer, g_hostCapacity);
    ASSERT_EQ(ret, 0);

    std::vector<int8_t> features(ntotal * dim);
    printf("[add -----------]\n");
    FeatureGenerator(features);
    std::vector<FeatureAttr> attrs(ntotal);
    FeatureAttrGenerator(attrs, 2500);
    for (size_t i = 0; i < addNum; i++) {
        std::vector<int64_t> labels;
        for (int64_t j = 0; j < ntotal; ++j) {
            labels.emplace_back(j + i * ntotal);
        }
        auto ts0 = TestData::GetMillisecs();
        tsIndex.AddFeature(ntotal, features.data(), attrs.data(), labels.data());
        auto te0 = TestData::GetMillisecs();
        printf("add %ld cost %f ms\n", ntotal, te0 - ts0);
    }

    vector<uint32_t> delToken{ 5 };
    ret = tsIndex.DeleteFeatureByToken(delToken.size(), delToken.data());
    ASSERT_EQ(ret, 0);
    int64_t validNum = 0;
    tsIndex.GetFeatureNum(&validNum);
    ASSERT_EQ(validNum, ntotal * addNum / 2500 * 2499);

    vector<int64_t> labels;
    for (int64_t i = ntotal * addNum - 1; i > 0; i-=262144) {
        labels.emplace_back(i);
        labels.emplace_back(i - 1);
    }
    vector<int8_t> featuresOut(labels.size() * dim, -1);
    ret = tsIndex.GetFeatureByLabel(labels.size(), labels.data(), featuresOut.data());
    ASSERT_EQ(ret, 0);
    for (size_t ilb = 0; ilb < labels.size(); ilb++) {
        auto lb = labels[ilb];
        int64_t lb100 = lb % ntotal;
        for (uint32_t i = 0; i < dim; i++) {
            if (features[lb100 * dim + i] != featuresOut[ilb * dim + i]) {
                printf("ilb[%zu], lb[%ld], lb100[%ld], i[%u]\r\n", ilb, lb, lb100, i);
                ASSERT_EQ(features[lb100 * dim + i], featuresOut[ilb * dim + i]);
            }
        }
    }
}

void SearchWithQPS(IReduction &reduction, AscendIndexTS &tsIndex, int64_t ntotal, uint32_t addNum)
{
    vector<uint32_t> searchBatch = { 1, 2, 4, 8, 16, 32, 64, 128, 256 };
    size_t topK = 200;
    vector<float> searchData(searchBatch.back() * g_reduceDimOut);
    vector<int8_t> searchDataReduced(searchBatch.back() * g_reduceDimOut);
    uint32_t setlen = static_cast<uint32_t>((g_tokenNum + 7) / 8);
    vector<uint8_t> bitSet(setlen, 0xff);
    AttrFilter filter;
    filter.timesStart = 0;
    filter.timesEnd = 9;
    filter.tokenBitSet = bitSet.data();
    filter.tokenBitSetLen = bitSet.size();
    vector<AttrFilter> queryFilters(searchBatch.back(), filter);
    vector<float> distances(searchBatch.back() * topK, -1);
    vector<int64_t> labelRes(searchBatch.back() * topK, -1);
    vector<uint32_t> validnum(searchBatch.back(), 1);
    int loopTimes = 10;
    for (auto batch : searchBatch) {
        if (batch == searchBatch.front()) {
            reduction.reduce(batch, g_testData->GetQuery().data(), searchData.data());
            auto float2Int8 = [] (float f) {
                return f / 800 * 127;
            };
            std::transform(searchData.begin(), searchData.begin() + batch * g_reduceDimOut,
                searchDataReduced.begin(), float2Int8);
            auto ret = tsIndex.Search(batch, searchDataReduced.data(), queryFilters.data(), false, topK, labelRes.data(),
                distances.data(), validnum.data());
            ASSERT_EQ(ret, 0);
        }
        auto ts = TestData::GetMillisecs();
        for (int i = 0; i < loopTimes; i++) {
            reduction.reduce(batch, g_testData->GetQuery().data(), searchData.data());
            auto float2Int8 = [] (float f) {
                return f / 800 * 127;
            };
            std::transform(searchData.begin(), searchData.begin() + batch * g_reduceDimOut,
                searchDataReduced.begin(), float2Int8);
            auto ret = tsIndex.Search(batch, searchDataReduced.data(), queryFilters.data(), false, topK, labelRes.data(),
                distances.data(), validnum.data());
            ASSERT_EQ(ret, 0);
        }
        auto te = TestData::GetMillisecs();
        printf("base: %ld,  dim: %ld,  batch: %u,  topk%zu,  QPS:%7.2Lf\n", ntotal * addNum, g_reduceDimOut, batch, topK,
            static_cast<long double>(1000.0 * batch * loopTimes / (te - ts)));
    }
}

TEST(TestAscendIndexTSInt8FlatOck, NN_Reduce_Ts_Ock_Base_2Y_QPS)
{
    LoadTestData();

    cout << "***** start load reduce om model *****" << endl;
    string om = GetNNOmStr(g_nnModelPath);
    ReductionConfig reductionConfig({g_deviceId}, om.data(), om.size());
    IReduction* reduction = CreateReduction("NN", reductionConfig);
    cout << "***** end load reduce om model *****" << endl;

    cout << "***** start reduce *****" << endl;
    auto [reduceOutNormBases, reduceOutNormQuerys] = ReduceData(reduction);
    cout << "***** end reduce *****" << endl;

    cout << "***** start create index *****" << endl;
    AscendIndexTS tsIndex;
    constexpr size_t deviceCapacity = static_cast<size_t>(33792) * 1024 * 1024;
    constexpr size_t deviceBuffer = static_cast<size_t>(8) * 64 * 1024 * 1024;
    auto ret = tsIndex.Init(g_deviceId, g_reduceDimOut, g_tokenNum, AlgorithmType::FLAT_COS_INT8, MemoryStrategy::HETERO_MEMORY);
    ASSERT_EQ(ret, 0);
    ret = tsIndex.SetHeteroParam(deviceCapacity, deviceBuffer, g_hostCapacity);
    ASSERT_EQ(ret, 0);
    cout << "***** end create index *****" << endl;

    cout << "***** start add *****" << endl;
    int64_t ntotal = 1000000;
    uint32_t addNum = 200;
    std::vector<FeatureAttr> attrs(ntotal);
    FeatureAttrGenerator(attrs, 321);
    for (size_t i = 0; i < addNum; i++) {
        std::vector<int64_t> labels;
        for (int64_t j = 0; j < ntotal; ++j) {
            labels.emplace_back(j + i * ntotal);
        }
        auto ts0 = TestData::GetMillisecs();
        tsIndex.AddFeature(ntotal, reduceOutNormBases.data(), attrs.data(), labels.data());
        auto te0 = TestData::GetMillisecs();
        printf("add %ld cost %f ms\n", ntotal, te0 - ts0);
    }
    cout << "***** end add *****" << endl;

    cout << "***** start search *****" << endl;
    SearchWithQPS(*reduction, tsIndex, ntotal, addNum);
    cout << "***** end search *****" << endl;
    delete reduction;
}

int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);

    if (argc >= 3) {
        g_dataDir = argv[1];
        g_nnModelPath = argv[2];
    }

    cout << "set g_dataDir = " << g_dataDir << endl;
    cout << "set g_nnModelPath = " << g_nnModelPath << endl;

    return RUN_ALL_TESTS();
}