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

#include <random>
#include <vector>
#include <algorithm>
#include <cstring>
#include <cassert>
#include <fstream>
#include <gtest/gtest.h>
#include "ock/log/OckHcpsLogger.h"
#include "ock/hcps/algo/OckShape.h"

namespace ock {
namespace hcps {
namespace algo {
namespace test {
namespace {
constexpr uint32_t DIM = 256UL;
constexpr uint32_t CUBE_ALIGN = 16UL;
} // namespace

class TestOckShape : public testing::Test {
public:
    // Default: Test 64KB data, when dimSize = 256, dataType = int8, dataNum = 64 * 1024 / 256 = 256
    explicit TestOckShape(uint32_t dataNum = 256) : dataNum(dataNum), dimSize(DIM), byteSize(dataNum * DIM) {}

    void SetUp() override
    {
        dataBase.reserve(byteSize);
        data_ = static_cast<int8_t *>(dataBase.data());
    }

    // 随机生成num条int8向量
    void InitRandData(uint32_t num, int8_t minValue = -20, int8_t maxValue = 64)
    {
        if (num > dataNum) {
            OCK_HCPS_LOG_ERROR("The size is " << num * dimSize << ", larger than byteSize, which is " << byteSize);
            return;
        }
        // 确保数据在类型范围内
        assert(minValue >= std::numeric_limits<int8_t>::min());
        assert(maxValue <= std::numeric_limits<int8_t>::max());
        std::random_device seed;
        std::ranlux48 engine(seed());
        std::uniform_int_distribution<> distrib(minValue, maxValue);

        for (uint32_t i = 0; i < num; ++i) {
            for (uint32_t j = 0; j < dimSize; ++j) {
                dataBase.emplace_back(distrib(engine));
            }
        }
    }

    void InitSameData(uint32_t num = 100)
    {
        int8_t tmpNum;
        int maxInt8 = std::numeric_limits<int8_t>::max();
        for (uint32_t i = 0; i < num; ++i) {
            tmpNum = i % maxInt8;
            for (uint32_t j = 0; j < dimSize; ++j) {
                dataBase.emplace_back(tmpNum);
            }
        }
    }

    void InitRandVec()
    {
        if (dataBase.size() + dimSize * sizeof(int8_t) > byteSize) {
            OCK_HCPS_LOG_ERROR("The size is " << dataBase.size() << ", larger than byteSize, which is " << byteSize);
            return;
        }
        for (uint32_t j = 0; j < dimSize; ++j) {
            dataBase.push_back((int8_t)(rand() % std::numeric_limits<int8_t>::max()));
        }
    }

    // 以大Z小z的方式读取原矩阵，检查只分形前rows行时的结果是否正确
    void ReadMatrixByZz(int8_t *matrix, std::vector<int8_t> &data, uint32_t rows)
    {
        rows = ((rows + CUBE_ALIGN - 1) / CUBE_ALIGN) * CUBE_ALIGN;
        uint32_t i = 0;
        uint32_t j = 0;
        uint32_t k = 0;
        uint32_t cnt = 0;
        uint32_t dimAlign = (CUBE_ALIGN * 2) / sizeof(int8_t);
        uint32_t dimAlignNum = dimSize / dimAlign;
        while (i < rows) {
            for (k = 0; k < dimAlign; k++) {
                data.push_back(*(matrix + i * dimSize + j * dimAlign + k));
            }
            i++;
            if (i % CUBE_ALIGN == 0 && j != dimAlignNum - 1) {
                i = cnt * CUBE_ALIGN;
                j++;
            } else if (i % CUBE_ALIGN == 0 && j == dimAlignNum - 1) {
                cnt++;
                i = cnt * CUBE_ALIGN;
                j = 0;
            }
        }
    }

    bool CheckGetVec(int64_t num, int8_t *fetchPtr)
    {
        for (uint32_t i = 0; i < dimSize; i++) {
            if (*(fetchPtr + i) != num) {
                return false;
            }
        }
        return true;
    }

    bool ComparePtrValue(int8_t *inputPtr, int8_t *outputPtr, uint32_t vecLen)
    {
        for (uint32_t i = 0; i < vecLen; i++) {
            if (*(inputPtr + i) != *(outputPtr + i)) {
                return false;
            }
        }
        return true;
    }

    uint32_t dataNum;
    uint32_t dimSize;
    std::vector<int8_t> dataBase;
    int8_t *data_;
    uint64_t byteSize;
};

TEST_F(TestOckShape, add_single_data_correctly)
{
    const uint32_t initVecNum = 3UL;
    InitRandData(initVecNum);
    int8_t *outPtr = new int8_t[byteSize]{};
    OckShape<int8_t, DIM, CUBE_ALIGN> ockShape(reinterpret_cast<uintptr_t>(outPtr), byteSize);
    ockShape.AddData(data_);
    EXPECT_EQ(ockShape.ValidateRowCount(), 1U);

    std::vector<int8_t> fetchData(dimSize);
    ockShape.GetData(0, fetchData.data());
    bool result = ComparePtrValue(fetchData.data(), data_, dimSize);
    delete[] outPtr;
    EXPECT_EQ(result, true);
}

TEST_F(TestOckShape, add_multi_data_correctly)
{
    const uint32_t initVecNum = 32UL;
    std::vector<int8_t> matrixZ;
    InitRandData(initVecNum);
    int8_t *outPtr = new int8_t[byteSize]{};
    OckShape<int8_t, DIM, CUBE_ALIGN> ockShape(reinterpret_cast<uintptr_t>(outPtr), byteSize);
    data_ = static_cast<int8_t *>(dataBase.data());

    ockShape.AddData(data_, initVecNum);

    ReadMatrixByZz(data_, matrixZ, initVecNum);
    bool result = ComparePtrValue(matrixZ.data(), outPtr, matrixZ.size());
    EXPECT_EQ(result, true);
    delete[] outPtr;
}

TEST_F(TestOckShape, get_data_correctly)
{
    const uint32_t initVecNum = 32UL;
    int64_t rowId = rand() % initVecNum;
    int64_t num = rowId % std::numeric_limits<int8_t>::max();
    InitSameData(initVecNum);
    int8_t *outPtr = new int8_t[byteSize]{};
    int8_t *fetchPtr = new int8_t[dimSize]{};
    OckShape<int8_t, DIM, CUBE_ALIGN> ockShape(reinterpret_cast<uintptr_t>(outPtr), byteSize);

    ockShape.AddData(data_, initVecNum);
    ockShape.GetData(rowId, fetchPtr);

    // InitSameData的初始化方法保证第rowId行存储的值全为num
    bool result = CheckGetVec(num, fetchPtr);
    EXPECT_EQ(result, true);
    delete[] outPtr;
    delete[] fetchPtr;
}

TEST_F(TestOckShape, add_from_other_block_correctly)
{
    const uint32_t initVecNum = 32UL;
    int64_t rowId = rand() % initVecNum;
    InitRandData(initVecNum);
    std::vector<int8_t> blockA(byteSize);
    std::vector<int8_t> blockB(byteSize);
    OckShape<int8_t, DIM, CUBE_ALIGN> ockShapeA(reinterpret_cast<uintptr_t>(blockA.data()), byteSize);
    OckShape<int8_t, DIM, CUBE_ALIGN> ockShapeB(reinterpret_cast<uintptr_t>(blockB.data()), byteSize);

    // 先将源矩阵分形到blockA
    ockShapeA.AddData(data_, initVecNum);
    // 从block1中随机提取第row条数据并存储到blockB
    ockShapeB.AddFrom(ockShapeA, rowId);

    EXPECT_EQ(ockShapeB.ValidateRowCount(), 1UL);

    // 分别提取blockA的第rowId条向量和block2的最后一条向量（测试用例中即第一条向量, 下标0）
    std::vector<int8_t> fetchDataA(dimSize);
    std::vector<int8_t> fetchDataB(dimSize);
    ockShapeA.GetData(rowId, fetchDataA.data());
    ockShapeB.GetData(0, fetchDataB.data());
    bool result = ComparePtrValue(fetchDataA.data(), fetchDataB.data(), dimSize);

    EXPECT_EQ(result, true);
}

TEST_F(TestOckShape, add_segment_and_get_vectors_successfully)
{
    const uint32_t initVecNum = 32UL;
    InitRandData(initVecNum);
    std::vector<int8_t> blockA(byteSize);
    std::vector<int8_t> blockB(byteSize);
    OckShape<int8_t, DIM, CUBE_ALIGN> ockShapeA(reinterpret_cast<uintptr_t>(blockA.data()), byteSize);
    OckShape<int8_t, DIM, CUBE_ALIGN> ockShapeB(reinterpret_cast<uintptr_t>(blockB.data()), byteSize);
    ockShapeB.AddData(data_, initVecNum);

    ockShapeA.AddSegment(ockShapeB, 0U, CUBE_ALIGN);
    EXPECT_EQ(ockShapeA.ValidateRowCount(), CUBE_ALIGN);

    std::vector<int8_t> res(CUBE_ALIGN * DIM);
    for (size_t i = 0; i < CUBE_ALIGN; ++i) {
        ockShapeA.GetData(i, res.data() + i * DIM);
    }
    EXPECT_EQ(ComparePtrValue(res.data(), data_, res.size()), true);
}

TEST_F(TestOckShape, add_segment_while_rowCount_not_aligned)
{
    const uint32_t initVecNum = 32UL;
    InitRandData(initVecNum);
    std::vector<int8_t> blockA(byteSize);
    std::vector<int8_t> blockB(byteSize);
    OckShape<int8_t, DIM, CUBE_ALIGN> ockShapeA(reinterpret_cast<uintptr_t>(blockA.data()), byteSize);
    OckShape<int8_t, DIM, CUBE_ALIGN> ockShapeB(reinterpret_cast<uintptr_t>(blockB.data()), byteSize);
    ockShapeB.AddData(data_, initVecNum);

    uint32_t addNum = CUBE_ALIGN + 3U;
    ockShapeA.AddSegment(ockShapeB, 0U, addNum);
    EXPECT_EQ(ockShapeA.ValidateRowCount(), addNum);

    std::vector<int8_t> res(addNum * DIM);
    for (size_t i = 0; i < addNum; ++i) {
        ockShapeA.GetData(i, res.data() + i * DIM);
    }
    EXPECT_EQ(ComparePtrValue(res.data(), data_, res.size()), true);
}

TEST_F(TestOckShape, add_segment_while_all_not_aligned)
{
    const uint32_t initVecNum = 32UL;
    InitRandData(initVecNum);
    std::vector<int8_t> blockA(byteSize);
    std::vector<int8_t> blockB(byteSize);
    OckShape<int8_t, DIM, CUBE_ALIGN> ockShapeA(reinterpret_cast<uintptr_t>(blockA.data()), byteSize);
    OckShape<int8_t, DIM, CUBE_ALIGN> ockShapeB(reinterpret_cast<uintptr_t>(blockB.data()), byteSize);
    ockShapeB.AddData(data_, initVecNum);
    ockShapeA.AddData(data_, 3U);

    uint32_t addNum = CUBE_ALIGN + 3U;
    ockShapeA.AddSegment(ockShapeB, 3U, addNum);
    EXPECT_EQ(ockShapeA.ValidateRowCount(), addNum + 3U);

    std::vector<int8_t> res((addNum + 3U) * DIM);
    for (size_t i = 0; i < addNum + 3U; ++i) {
        ockShapeA.GetData(i, res.data() + i * DIM);
    }
    EXPECT_EQ(ComparePtrValue(res.data(), data_, res.size()), true);
}

TEST_F(TestOckShape, output_class_shape_successfully)
{
    const uint32_t initVecNum = 23UL;
    InitRandData(initVecNum);
    std::vector<int8_t> block(byteSize);
    OckShape<int8_t, DIM, CUBE_ALIGN> ockShape(reinterpret_cast<uintptr_t>(block.data()), byteSize);
    ockShape.AddData(data_, initVecNum);
    // 捕获输出
    testing::internal::CaptureStdout();
    std::cout << ockShape << std::endl;
    // 获取输出
    std::string output = testing::internal::GetCapturedStdout();

    constexpr int bufferLen = 200;
    char buffer[bufferLen];
    sprintf_s(buffer, bufferLen, "{'size':%d, 'validateRowCount':%d}\n",
        static_cast<int>(ockShape.ByteSize()), static_cast<int>(ockShape.ValidateRowCount()));
    std::string formattedStr = buffer;
    EXPECT_EQ(formattedStr == output, true);
}

TEST_F(TestOckShape, unreshape_to_origin_matrix_successfully)
{
    const uint32_t initVecNum = 200UL;
    InitRandData(initVecNum);
    std::vector<int8_t> block(byteSize);
    OckShape<int8_t, DIM, CUBE_ALIGN> ockShape(reinterpret_cast<uintptr_t>(block.data()), byteSize);
    ockShape.AddData(data_, initVecNum);

    std::vector<int8_t> matrix(byteSize);
    ockShape.Restore(matrix.data());

    bool result = ComparePtrValue(data_, matrix.data(), initVecNum);
    EXPECT_EQ(result, true);
}

TEST_F(TestOckShape, copy_from_other_block_successfully)
{
    const uint32_t initVecNum = 32UL;
    int64_t rowId = rand() % initVecNum;
    InitRandData(initVecNum);
    std::vector<int8_t> blockA(byteSize);
    std::vector<int8_t> blockB(byteSize);
    OckShape<int8_t, DIM, CUBE_ALIGN> ockShapeA(reinterpret_cast<uintptr_t>(blockA.data()), byteSize);
    OckShape<int8_t, DIM, CUBE_ALIGN> ockShapeB(reinterpret_cast<uintptr_t>(blockB.data()), byteSize);

    ockShapeB.AddData(data_, initVecNum);
    ockShapeA.AddData(data_, 5UL);
    ockShapeA.CopyFrom(0UL, ockShapeB, rowId);
    int8_t *fetchPtr = new int8_t[dimSize]{};
    ockShapeA.GetData(0UL, fetchPtr);

    EXPECT_EQ(ComparePtrValue(fetchPtr, data_ + rowId * dimSize, dimSize), true);
    delete[] fetchPtr;
}

TEST_F(TestOckShape, init_from_validateRowCount_successfully)
{
    const uint32_t initVecNum = 32UL;
    InitRandData(initVecNum);
    std::vector<int8_t> blockA(byteSize);
    OckShape<int8_t, DIM, CUBE_ALIGN> ockShapeA(reinterpret_cast<uintptr_t>(blockA.data()), byteSize);
    ockShapeA.AddData(data_, initVecNum);
    OckShape<int8_t, DIM, CUBE_ALIGN> ockShapeB(reinterpret_cast<uintptr_t>(blockA.data()), byteSize, initVecNum);
    ockShapeB.AddData(data_);
    
    int8_t *fetchPtr = new int8_t[dimSize]{};
    ockShapeB.GetData(initVecNum, fetchPtr);
    EXPECT_EQ(ComparePtrValue(fetchPtr, data_, dimSize), true);
    delete[] fetchPtr;
}
} // namespace test
} // namespace algo
} // namespace hcps
} // namespce ock
