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
#include <numeric>
#include <gtest/gtest.h>
#include <mockcpp/mockcpp.hpp>
#include "AscendIndexTS.h"
#include "IndexInt8FlatCosAicpu.h"
#include "ErrorCode.h"
#include "stub/hmm/HmmMock.h"
#include "stub/hmm/AscendHMOMock.h"

using namespace testing;
using namespace std;
using namespace faiss::ascend;

namespace ascend {

constexpr uint32_t VAILD_DEVICE_ID = 0;
constexpr uint32_t VALID_DIM = 256;
constexpr uint32_t VALID_TOKEN_NUM = 2500;
constexpr AlgorithmType VALID_ALGO_TYPE = AlgorithmType::FLAT_COS_INT8;
constexpr MemoryStrategy VALID_MEM_STRATEGY = MemoryStrategy::HETERO_MEMORY;
constexpr size_t KB = 1024;
constexpr size_t VALID_DEVICE_CAPACITY = KB * KB * KB; // 1073741824
constexpr size_t VALID_DEVICE_BUFFER = 9 * 64 * KB * KB; // 603979776
constexpr size_t VALID_HOST_CAPACITY = 20 * KB * KB * KB; // 21474836480
constexpr uint8_t BIT_LEN = 8;

class TestTSInt8FlatOck : public Test {
public:
    void SetUp() override
    {
        // 这里不直接make_shared创建智能指针是因为EXPECT_CALL会到导致智能指针循环依赖；
        // 因此智能指针不释放内存，而是在TearDown中手动释放
        HmmMock::defaultHmm = shared_ptr<HmmMock>(new HmmMock(), [] (auto) {});
        EXPECT_CALL(*HmmMock::defaultHmm, CreateHmm).WillRepeatedly(Return(HmmMock::defaultHmm));
        EXPECT_CALL(*HmmMock::defaultHmm, Init).WillRepeatedly(Return(APP_ERR_OK));

        // aclrtCreateStream是fake接口太耗时（100ms），大部分用例中不需要使用它；
        // 需要使用的用例应该在用例开头显示调用GlobalMockObject::verify来取消stub
        MOCKER_CPP(&aclrtCreateStream).stubs().will(returnValue(0));
    }

    void TearDown() override
    {
        delete HmmMock::defaultHmm.get();
        HmmMock::defaultHmm.reset();

        GlobalMockObject::verify();
    }
};


TEST_F(TestTSInt8FlatOck, Init_with_memoryStrategy)
{
    struct InputOutput {
        AlgorithmType algoType { AlgorithmType::FLAT_COS_INT8 };
        MemoryStrategy memoryStrategy { MemoryStrategy::PURE_DEVICE_MEMORY };
        APP_ERROR ret { APP_ERR_OK };
    };

    // given
    vector<InputOutput> inputOutputs {
        { VALID_ALGO_TYPE, VALID_MEM_STRATEGY, APP_ERR_OK },
        { VALID_ALGO_TYPE, MemoryStrategy::PURE_DEVICE_MEMORY, APP_ERR_OK },
        { AlgorithmType::FLAT_HAMMING, MemoryStrategy::PURE_DEVICE_MEMORY, APP_ERR_OK },
        { AlgorithmType::FLAT_IP_FP16, MemoryStrategy::PURE_DEVICE_MEMORY, APP_ERR_OK },
        { AlgorithmType::FLAT_HAMMING, VALID_MEM_STRATEGY, APP_ERR_INVALID_PARAM },
        { AlgorithmType::FLAT_IP_FP16, VALID_MEM_STRATEGY, APP_ERR_INVALID_PARAM },
    };

    shared_ptr<AscendIndexTS> tsIndex;

    // when
    for (const auto &io : inputOutputs) {
        tsIndex = make_shared<AscendIndexTS>();
        auto ret = tsIndex->Init(VAILD_DEVICE_ID, VALID_DIM, VALID_TOKEN_NUM, io.algoType, io.memoryStrategy);
        ASSERT_EQ(ret, io.ret) << "algoType: " << static_cast<int>(io.algoType) <<
            " memoryStrategy: " << static_cast<int>(io.memoryStrategy);
    }
}

TEST_F(TestTSInt8FlatOck, SetHeteroParam)
{
    shared_ptr<AscendIndexTS> tsIndex = make_shared<AscendIndexTS>();
    // 未init不能设置异构参数
    auto ret = tsIndex->SetHeteroParam(VALID_DEVICE_CAPACITY, VALID_HOST_CAPACITY, VALID_HOST_CAPACITY);
    ASSERT_EQ(ret, APP_ERR_ILLEGAL_OPERATION);

    // 未设置异构模式，依旧不能设置异构参数
    ret = tsIndex->Init(VAILD_DEVICE_ID, VALID_DIM, VALID_TOKEN_NUM, VALID_ALGO_TYPE,
        MemoryStrategy::PURE_DEVICE_MEMORY);
    ASSERT_EQ(ret, APP_ERR_OK);
    ret = tsIndex->SetHeteroParam(VALID_DEVICE_CAPACITY, VALID_HOST_CAPACITY, VALID_HOST_CAPACITY);
    ASSERT_EQ(ret, APP_ERR_ILLEGAL_OPERATION);

    // 未成功设置参数，不能使用add接口
    tsIndex = make_unique<AscendIndexTS>();
    ret = tsIndex->Init(VAILD_DEVICE_ID, VALID_DIM, VALID_TOKEN_NUM, VALID_ALGO_TYPE, VALID_MEM_STRATEGY);
    ASSERT_EQ(ret, APP_ERR_OK);
    vector<int8_t> feature(VALID_DIM);
    FeatureAttr attr { 1, 5 };
    int64_t label = 0;
    ret = tsIndex->AddFeature(1, feature.data(), &attr, &label);
    ASSERT_EQ(ret, APP_ERR_ILLEGAL_OPERATION);
    // 设置合理内存，设置成功
    ret = tsIndex->SetHeteroParam(VALID_DEVICE_CAPACITY, VALID_DEVICE_BUFFER, VALID_HOST_CAPACITY);
    ASSERT_EQ(ret, APP_ERR_OK);
    // 设置完成以后，无法再次设置
    ret = tsIndex->SetHeteroParam(VALID_DEVICE_CAPACITY, VALID_DEVICE_BUFFER, VALID_HOST_CAPACITY);
    ASSERT_EQ(ret, APP_ERR_ILLEGAL_OPERATION);
}

TEST_F(TestTSInt8FlatOck, SetHeteroParam_para_range)
{
    struct InputOutput {
        size_t deviceCapacity { 0 };
        size_t deviceBuffer { 0 };
        size_t hostCapacity { 0 };
        APP_ERROR ret { APP_ERR_OK };
    };

    const size_t minDeviceBuffer = 262144 * VALID_DIM * 2;
    vector<InputOutput> inputOutputs {
        { VALID_DEVICE_CAPACITY, VALID_DEVICE_BUFFER, VALID_HOST_CAPACITY, APP_ERR_OK },

        { VALID_DEVICE_CAPACITY, 0, VALID_HOST_CAPACITY, APP_ERR_INVALID_PARAM },
        { VALID_DEVICE_CAPACITY, minDeviceBuffer - 1, VALID_HOST_CAPACITY, APP_ERR_INVALID_PARAM },
        { VALID_DEVICE_CAPACITY, minDeviceBuffer, VALID_HOST_CAPACITY, APP_ERR_OK },
        { VALID_DEVICE_CAPACITY, VALID_DEVICE_BUFFER, VALID_HOST_CAPACITY, APP_ERR_OK },

        { VALID_DEVICE_CAPACITY, VALID_DEVICE_BUFFER, KB * KB * KB, APP_ERR_OK },
        { VALID_DEVICE_CAPACITY, VALID_DEVICE_BUFFER, VALID_HOST_CAPACITY, APP_ERR_OK },

        { 8 * KB * KB * KB, 8 * KB * KB * KB, VALID_HOST_CAPACITY, APP_ERR_OK },
    };

    shared_ptr<AscendIndexTS> tsIndex;
    for (const auto &io : inputOutputs) {
        tsIndex = make_shared<AscendIndexTS>();
        auto ret = tsIndex->Init(VAILD_DEVICE_ID, VALID_DIM, VALID_TOKEN_NUM, VALID_ALGO_TYPE, VALID_MEM_STRATEGY);
        ASSERT_EQ(ret, APP_ERR_OK);
        ret = tsIndex->SetHeteroParam(io.deviceCapacity, io.deviceBuffer, io.hostCapacity);
        ASSERT_EQ(ret, io.ret) << " deviceCapcity:" << io.deviceCapacity << " deviceBuffer:" << io.deviceBuffer <<
            " hostCapacity:" << io.hostCapacity;
    }
}

void PrepareForSearch(AscendIndexTS &tsIndex, AscendHMOMock &ascendHMOMock)
{
    const size_t blockSize = 524288;
    std::shared_ptr<AscendHMO> ascendHMOMockShared(&ascendHMOMock, [] (auto) {});
    EXPECT_CALL(*HmmMock::defaultHmm, CreateHmo(blockSize)).Times(1)
        .WillOnce(Return(make_pair(APP_ERR_OK, ascendHMOMockShared)));
    EXPECT_CALL(ascendHMOMock, Empty).WillRepeatedly(Return(false));
    EXPECT_CALL(ascendHMOMock, ValidateBuffer).Times(1);
    EXPECT_CALL(ascendHMOMock, GetAddress).WillOnce(Return(0));
    EXPECT_CALL(ascendHMOMock, FlushData).WillOnce(Return(APP_ERR_OK));
    EXPECT_CALL(ascendHMOMock, InvalidateBuffer).WillOnce(Return(APP_ERR_OK));
    EXPECT_CALL(ascendHMOMock, Clear).Times(1);

    auto ret = tsIndex.Init(VAILD_DEVICE_ID, VALID_DIM, VALID_TOKEN_NUM, VALID_ALGO_TYPE, VALID_MEM_STRATEGY);
    EXPECT_EQ(ret, APP_ERR_OK);
    ret = tsIndex.SetHeteroParam(0, VALID_DEVICE_BUFFER, VALID_HOST_CAPACITY);
    EXPECT_EQ(ret, APP_ERR_OK);

    uint32_t ntotal = 8;
    vector<int8_t> baseData(ntotal * VALID_DIM);
    vector<FeatureAttr> attrs(ntotal);
    vector<int64_t> label(ntotal);
    iota(label.begin(), label.end(), 0);
    ret = tsIndex.AddFeature(ntotal, baseData.data(), attrs.data(), label.data());
    EXPECT_EQ(ret, APP_ERR_OK);
}

// 实际场景下，deviceCapacity侧受限ock约束最小值为1G，但本用例设deviceCapacity为0，测试使用host测内存流程
TEST_F(TestTSInt8FlatOck, AddFeature_all_in_host)
{
    AscendHMOMock ascendHMOMock;
    AscendIndexTS tsIndex;
    PrepareForSearch(tsIndex, ascendHMOMock);
}

TEST_F(TestTSInt8FlatOck, Search)
{
    AscendHMOMock ascendHMOMock;
    std::shared_ptr<AscendHMO> ascendHMOMockShared(&ascendHMOMock, [] (auto) {});
    AscendIndexTS tsIndex;
    PrepareForSearch(tsIndex, ascendHMOMock);

    EXPECT_CALL(ascendHMOMock, ValidateBufferAsync).WillOnce(Return(APP_ERR_OK));
    EXPECT_CALL(ascendHMOMock, IsHostHMO).WillRepeatedly(Return(true));
    EXPECT_CALL(ascendHMOMock, ValidateBuffer).Times(1);
    EXPECT_CALL(ascendHMOMock, GetAddress).WillOnce(Return(0));

    EXPECT_CALL(ascendHMOMock, InvalidateBuffer).WillOnce(Return(APP_ERR_OK));

    uint32_t setLen = (VALID_TOKEN_NUM + 7) / BIT_LEN;
    std::vector<uint8_t> bitSet(setLen, 0);
    AttrFilter filter;
    filter.timesStart = 0;
    filter.timesEnd = 0;
    filter.tokenBitSet = bitSet.data();
    filter.tokenBitSetLen = setLen;

    uint32_t searchNum = 1;
    vector<int8_t> features(searchNum * VALID_DIM);
    uint32_t topK = 1;
    vector<int64_t> labels(searchNum * topK);
    vector<float> distances(searchNum * topK);
    uint32_t validNum = 0;
    auto ret = tsIndex.Search(searchNum, features.data(), &filter, false, topK,
        labels.data(), distances.data(), &validNum);
    EXPECT_EQ(ret, APP_ERR_OK);
}

TEST_F(TestTSInt8FlatOck, GetFeatureByLabel)
{
    AscendHMOMock ascendHMOMock;
    std::shared_ptr<AscendHMO> ascendHMOMockShared(&ascendHMOMock, [] (auto) {});
    AscendIndexTS tsIndex;
    PrepareForSearch(tsIndex, ascendHMOMock);

    int64_t getNum = 1;
    vector<int64_t> labels(getNum);
    vector<int8_t> features(getNum * VALID_DIM);
    // hostData作为模拟底库，每个维度值为100，因为底库被分形，所以实际长度为CUBE_ALIGN*VALID_DIM
    const int8_t val = 100;
    vector<int8_t> hostData(getNum * CUBE_ALIGN * VALID_DIM, val);
    EXPECT_CALL(ascendHMOMock, GetAddress)
        .Times(VALID_DIM / CUBE_ALIGN_INT8)
        .WillRepeatedly(Return(reinterpret_cast<uintptr_t>(hostData.data())));
    EXPECT_CALL(ascendHMOMock, ValidateBuffer).Times(VALID_DIM / CUBE_ALIGN_INT8);
    EXPECT_CALL(ascendHMOMock, InvalidateBuffer).WillOnce(Return(APP_ERR_OK));
    auto ret = tsIndex.GetFeatureByLabel(getNum, labels.data(), features.data());
    EXPECT_EQ(ret, APP_ERR_OK);
    vector<int8_t> realHostData(getNum * VALID_DIM, val);
    EXPECT_EQ(features, realHostData);
}

TEST_F(TestTSInt8FlatOck, DeleteFeatureByLabel)
{
    AscendHMOMock ascendHMOMock;
    std::shared_ptr<AscendHMO> ascendHMOMockShared(&ascendHMOMock, [] (auto) {});
    AscendIndexTS tsIndex;
    PrepareForSearch(tsIndex, ascendHMOMock);

    EXPECT_CALL(ascendHMOMock, CopyTo).Times(VALID_DIM / CUBE_ALIGN_INT8).WillRepeatedly(Return(APP_ERR_OK));

    uint32_t ntotal = 8;
    int64_t baseNum = 0;
    auto ret = tsIndex.GetFeatureNum(&baseNum);
    EXPECT_EQ(ret, APP_ERR_OK);
    EXPECT_EQ(baseNum, ntotal); // PrepareForSearch中设置底库为8

    int64_t delCount = 1;
    int64_t delLabel = 0;
    ret = tsIndex.DeleteFeatureByLabel(delCount, &delLabel);
    EXPECT_EQ(ret, APP_ERR_OK);

    ret = tsIndex.GetFeatureNum(&baseNum);
    EXPECT_EQ(ret, APP_ERR_OK);
    EXPECT_EQ(baseNum, ntotal - delCount);
}


} // namespace ascend
