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
#include <gtest/gtest.h>
#include "ock/hcps/algo/OckSpliterSegement.h"
#include "ock/hcps/algo/OckQuicklyExternSortSpliter.h"

namespace ock {
namespace hcps {
namespace algo {
// 测试用例类
class TestOckQuicklyExternSortSpliter : public testing::Test {
public:
    explicit TestOckQuicklyExternSortSpliter() : dataCount(16384) // 16384 = 1024*16
    {}
    void InitRandData(uint64_t num = 16384)
    {
        this->dataCount = num;
        inputDatas.reserve(dataCount);
        outputDatas.reserve(dataCount);
        for (uint32_t i = 0; i < this->dataCount; ++i) {
            // rand()返回一随机数值的范围在0至RAND_MAX 间
            // 取除以 int32 的最大值后的余数，生成在 int32 范围内的随机整数
            inputDatas.push_back((int32_t)(rand() % std::numeric_limits<int32_t>::max()));
            outputDatas.push_back((int32_t)(rand() % std::numeric_limits<int32_t>::max()));
        }
    }
    // 使用 QuicklySort 生成的合并记录 mergeSeg 和 复制记录 copySeg 进行排序
    void QuicklySortImpl(std::vector<int32_t> &inputDatas, std::vector<int32_t> &outputDatas,
        std::shared_ptr<spliter::VectorSortSpliterResult<int32_t>> mergeResult)
    {
        // 分割，各片段排序
        for (auto sortSegItem : mergeResult->sortSeg) {
            std::sort(sortSegItem.begin, sortSegItem.end);
        }
        // 合并
        int64_t mergeNum = mergeResult->mergeSeg.size();     // 合并次数 + 1
        for (int64_t level = 0; level < mergeNum; level++) { // 跳过第一层
            for (auto mergeSegItem : mergeResult->mergeSeg[level]) {
                // merge 后的数据会直接写入新的位置 mergeSegItem.toBegin
                std::merge(mergeSegItem.aBegin, mergeSegItem.aEnd, mergeSegItem.bBegin, mergeSegItem.bEnd,
                    mergeSegItem.result);
            }
        }
    }
    void QuicklySortSwap(std::vector<int32_t> &inputDatas, std::vector<int32_t> &outputDatas)
    {
        outputDatas.swap(inputDatas);
    }
    void QuicklySort(std::vector<int32_t> &inputDatas, std::vector<int32_t> &outputDatas,
        uint64_t splitThreshold = 1024)
    {
        // calculation：计算分割和合并细节
        auto spliterResult = CalcQuicklyExternSortSpliterSegments(inputDatas, outputDatas, splitThreshold);
        // sort
        QuicklySortImpl(inputDatas, outputDatas, spliterResult);
        // swap
        if (spliterResult->isSwap)
            QuicklySortSwap(inputDatas, outputDatas);
    }
    bool CompareSortResult(uint64_t splitThreshold)
    {
        std::vector<int32_t> inputDatasTmp(inputDatas);        // 复制一份inputDatas，防止inputDatas被改变
        QuicklySort(inputDatas, outputDatas, splitThreshold);  // 调用排序算法
        std::sort(inputDatasTmp.begin(), inputDatasTmp.end()); // 调用标准库的排序算法
        if (inputDatasTmp == outputDatas) {
            return true;
        } else {
            return false;
        }
    }

    uint32_t dataCount;
    std::vector<int32_t> inputDatas;
    std::vector<int32_t> outputDatas;
};

TEST_F(TestOckQuicklyExternSortSpliter, less_than_threshold)
{
    uint64_t dataCount = 100;
    uint64_t splitThreshold = 1024;
    InitRandData(dataCount);
    bool result = CompareSortResult(splitThreshold);
    EXPECT_EQ(result, 1);
};

TEST_F(TestOckQuicklyExternSortSpliter, equal_to_threshold)
{
    uint64_t splitThreshold = 1024;
    uint64_t dataCount = splitThreshold;
    InitRandData(dataCount);
    bool result = CompareSortResult(splitThreshold);
    EXPECT_EQ(result, 1);
};

TEST_F(TestOckQuicklyExternSortSpliter, twenty_times_threshold)
{
    uint64_t splitThreshold = 1024;
    uint64_t splitThresholdTimes = 20;
    uint64_t dataCount = splitThreshold * splitThresholdTimes;
    InitRandData(dataCount);
    bool result = CompareSortResult(splitThreshold);
    EXPECT_EQ(result, 1);
};

TEST_F(TestOckQuicklyExternSortSpliter, ten_times_threshold_with_bias)
{
    uint64_t splitThreshold = 1024;
    uint64_t splitThresholdTimes = 10;
    uint64_t splitThresholdBias = 111;
    uint64_t dataCount = splitThreshold * splitThresholdTimes + splitThresholdBias;
    InitRandData(dataCount);
    bool result = CompareSortResult(splitThreshold);
    EXPECT_EQ(result, 1);
};

TEST_F(TestOckQuicklyExternSortSpliter, other_threshold)
{
    uint64_t splitThreshold = 10243;
    uint64_t splitThresholdTimes = 10;
    uint64_t splitThresholdBias = 111;
    uint64_t dataCount = splitThreshold * splitThresholdTimes + splitThresholdBias;
    InitRandData(dataCount);
    bool result = CompareSortResult(splitThreshold);
    EXPECT_EQ(result, 1);
};
} // namespace algo
} // namespace hcps
} // namespace ock