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
#include "ock/hcps/algo/OckQuicklyExternMergeSpliter.h"

namespace ock {
namespace hcps {
namespace algo {
// 测试用例类
class TestOckQuicklyExternMergeSpliter : public testing::Test {
public:
    explicit TestOckQuicklyExternMergeSpliter() {}
    void InitRandData(int line, int *columns, int datacount)
    {
        this->lines = line;
        this->datacounts = datacount;
        outSortedData.reserve(datacounts);
        tmpDatas.reserve(datacounts);
        for (int i = 0; i < this->lines; ++i) {
            std::vector<int32_t> tmp;
            for (int j = 0; j < columns[i]; j++) {
                int32_t tmpData = static_cast<int32_t>((rand() % std::numeric_limits<int32_t>::max()));
                tmp.push_back(tmpData);
                sort(tmp.begin(), tmp.end());
                outSortedDataOri.push_back(tmpData);
                outSortedData.push_back(static_cast<int32_t>((rand() % std::numeric_limits<int32_t>::max())));
                tmpDatas.push_back(static_cast<int32_t>((rand() % std::numeric_limits<int32_t>::max())));
            }
            sortedGroups.push_back(tmp);
        }
    }

    void QuicklySortImpl(spliter::VectorExternMergeSpliterResult<int32_t> &mergeResult)
    {
        // 合并
        int64_t mergeNum = mergeResult.mergeSeg.size();
        for (int64_t level = 0; level < mergeNum; level++) {
            for (auto mergeSegItem : mergeResult.mergeSeg[level]) {
                // merge 后的数据会直接写入新的位置 mergeSegItem.toBegin
                std::merge(mergeSegItem.aBegin, mergeSegItem.aEnd, mergeSegItem.bBegin, mergeSegItem.bEnd,
                    mergeSegItem.result);
            }
        }
    }

    void QuicklySort(std::vector<std::vector<int32_t>> &sortedGroups, std::vector<int32_t> &outSortedData,
        std::vector<int32_t> &tmpDatas)
    {
        // calculation：计算分割和合并细节
        auto mergeResult = CalcQuicklyExternMergeSpliterSegments(sortedGroups, outSortedData, tmpDatas);
        // sort
        QuicklySortImpl(*mergeResult);
        // swap
        if (mergeResult->isSwap)
            outSortedData.swap(tmpDatas);
    }

    bool CompareSortResult(void)
    {
        QuicklySort(sortedGroups, outSortedData, tmpDatas);          // 调用排序算法
        std::sort(outSortedDataOri.begin(), outSortedDataOri.end()); // 调用标准库的排序算法
        if (outSortedDataOri == outSortedData) {
            return true;
        } else {
            return false;
        }
    }

    int lines = 9;
    int datacounts;
    std::vector<std::vector<int32_t>> sortedGroups;
    std::vector<int32_t> outSortedDataOri;
    std::vector<int32_t> outSortedData;
    std::vector<int32_t> tmpDatas;
};

// TEST(第一个参数是test case 的名字，第二个参数是test case里面的 test的名字)
TEST_F(TestOckQuicklyExternMergeSpliter, odd_segments)
{
    // 定义二维数组的行数 LINES，和每行的元素个数 columns[LINES]
    static const int LINES = 9;
    int columns[LINES] = {90, 95, 100, 111, 115, 123, 120, 120, 150};
    int datacounts = std::accumulate(std::begin(columns), std::end(columns), 0, std::plus<int>());
    InitRandData(LINES, columns, datacounts);
    bool result = CompareSortResult();
    EXPECT_EQ(result, 1);
}

TEST_F(TestOckQuicklyExternMergeSpliter, even_segments)
{
    // 定义二维数组的行数 LINES，和每行的元素个数 columns[LINES]
    static const int LINES = 8;
    int columns[LINES] = {90, 95, 100, 111, 115, 123, 120, 120};
    int datacounts = std::accumulate(std::begin(columns), std::end(columns), 0, std::plus<int>());
    InitRandData(LINES, columns, datacounts);
    bool result = CompareSortResult();
    EXPECT_EQ(result, 1);
}

TEST_F(TestOckQuicklyExternMergeSpliter, two_segments)
{
    // 定义二维数组的行数 LINES，和每行的元素个数 columns[LINES]
    static const int LINES = 2;
    int columns[LINES] = {91, 1};
    int datacounts = std::accumulate(std::begin(columns), std::end(columns), 0, std::plus<int>());
    InitRandData(LINES, columns, datacounts);
    bool result = CompareSortResult();
    EXPECT_EQ(result, 1);
}

TEST_F(TestOckQuicklyExternMergeSpliter, one_segments)
{
    // 定义二维数组的行数 LINES，和每行的元素个数 columns[LINES]
    static const int LINES = 1;
    int columns[LINES] = {91};
    int datacounts = std::accumulate(std::begin(columns), std::end(columns), 0, std::plus<int>());
    InitRandData(LINES, columns, datacounts);
    bool result = CompareSortResult();
    EXPECT_EQ(result, 1);
}

TEST_F(TestOckQuicklyExternMergeSpliter, large_odd_segments)
{
    // 定义二维数组的行数 LINES，和每行的元素个数 columns[LINES]
    static const int LINES = 555;
    int numMax = 10;
    int numBias = 2;
    int columns[LINES];
    for (int i = 0; i < LINES; i++) {
        columns[i] = static_cast<int>((rand() % numMax + numBias));
    }
    int datacounts = std::accumulate(std::begin(columns), std::end(columns), 0, std::plus<int>());
    InitRandData(LINES, columns, datacounts);
    bool result = CompareSortResult();
    EXPECT_EQ(result, 1);
}

TEST_F(TestOckQuicklyExternMergeSpliter, large_even_segments)
{
    // 定义二维数组的行数 LINES，和每行的元素个数 columns[LINES]
    static const int LINES = 1000;
    int columns[LINES];
    int numMax = 100;
    int numBias = 2;
    for (int i = 0; i < LINES; i++) {
        columns[i] = static_cast<int>((rand() % numMax + numBias));
    }
    int datacounts = std::accumulate(std::begin(columns), std::end(columns), 0, std::plus<int>());
    InitRandData(LINES, columns, datacounts);
    bool result = CompareSortResult();
    EXPECT_EQ(result, 1);
}
} // namespace algo
} // namespace hcps
} // namespace ock