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

#ifndef OCK_HCPS_ALGO_QUICKLY_EXTERN_MERGE_SPLITER_IMPL_H
#define OCK_HCPS_ALGO_QUICKLY_EXTERN_MERGE_SPLITER_IMPL_H
#include <cstdint>
#include <memory>
#include <vector>
#include <cmath>
#include "ock/hcps/algo/OckSpliterSegement.h"

namespace ock {
namespace hcps {
namespace algo {
namespace spliter {
struct MergeParam {
    uint64_t dataCount;      // 数据量
    uint64_t splitThreshold; // 分割阈值
    uint64_t mergeTimes;     // 合并倍数，两两合并，合并的片段倍数关系为 2
    uint64_t mergeSegNum;    // 初始分割的片段数
    uint64_t mergeNum;       // 合并次数
    bool segNumStatus;       // 合并状态
};
template <typename _Tp>
void QuicklySortMergeTwoDim(spliter::VectorExternMergeSpliterResult<_Tp> &SpliterResult,
    std::vector<spliter::VectorExternMergeSegment<_Tp>> &firstMergeSegInTmp,
    std::vector<std::vector<_Tp>> &sortedGroups, std::vector<_Tp> &tmpData, MergeParam &mergeParam);
template <typename _Tp>
void QuicklySortMergeFirst(spliter::VectorExternMergeSpliterResult<_Tp> &SpliterResult,
    std::vector<spliter::VectorExternMergeSegment<_Tp>> firstMergeSegInTmp, std::vector<_Tp> &outSortedData,
    std::vector<_Tp> &tmpData, MergeParam &mergeParam);
template <typename _Tp>
void QuicklySortMergeLast(spliter::VectorExternMergeSpliterResult<_Tp> &SpliterResult, std::vector<_Tp> &outSortedData,
    std::vector<_Tp> &tmpData, MergeParam &mergeParam);
template <typename _Tp>
void QuicklySortMergeLastImpl(spliter::VectorExternMergeSpliterResult<_Tp> &SpliterResult,
    std::vector<_Tp> &outSortedData, std::vector<_Tp> &tmpData, MergeParam &mergeParam);

template <typename _Tp>
void QuicklySortMergeTwoDim(spliter::VectorExternMergeSpliterResult<_Tp> &SpliterResult,
    std::vector<spliter::VectorExternMergeSegment<_Tp>> &firstMergeSegInTmp,
    std::vector<std::vector<_Tp>> &sortedGroups, std::vector<_Tp> &tmpData, MergeParam &mergeParam)
{
    // 第一次合并：二维 sortedGroups -> 一维 tmpData
    SpliterResult.mergeSeg[0].resize(mergeParam.mergeSegNum);
    uint64_t curNum = 0;
    for (uint64_t j = 0; j < mergeParam.mergeSegNum; j++) {
        SpliterResult.mergeSeg[0][j].aBegin = sortedGroups[j * mergeParam.mergeTimes].begin();
        SpliterResult.mergeSeg[0][j].aEnd = sortedGroups[j * mergeParam.mergeTimes].end();

        if (mergeParam.segNumStatus && j == mergeParam.mergeSegNum - 1) { // 奇数个片段，且最后一个片段时，特殊处理
            SpliterResult.mergeSeg[0][j].bBegin = SpliterResult.mergeSeg[0][j].aEnd;
            SpliterResult.mergeSeg[0][j].bEnd = SpliterResult.mergeSeg[0][j].aEnd;
        } else if (j == mergeParam.mergeSegNum - 1 || mergeParam.mergeSegNum == 1) { // 最后两个片段，特殊处理
            SpliterResult.mergeSeg[0][j].bBegin = sortedGroups[j * mergeParam.mergeTimes + 1].begin();
            SpliterResult.mergeSeg[0][j].bEnd = sortedGroups[j * mergeParam.mergeTimes + 1].end();
        } else {
            SpliterResult.mergeSeg[0][j].bBegin = sortedGroups[j * mergeParam.mergeTimes + 1].begin();
            SpliterResult.mergeSeg[0][j].bEnd = sortedGroups[j * mergeParam.mergeTimes + 1].end();
        }

        SpliterResult.mergeSeg[0][j].result = tmpData.begin() + curNum;

        // 使用 firstMergeSegInTmp 记录在 TmpData 中的迭代器
        firstMergeSegInTmp[j].aBegin = tmpData.begin() + curNum;
        firstMergeSegInTmp[j].aEnd =
            firstMergeSegInTmp[j].aBegin + (SpliterResult.mergeSeg[0][j].aEnd - SpliterResult.mergeSeg[0][j].aBegin);
        firstMergeSegInTmp[j].bBegin = firstMergeSegInTmp[j].aEnd;
        if (mergeParam.segNumStatus && j == mergeParam.mergeSegNum - 1) {
            firstMergeSegInTmp[j].bEnd = firstMergeSegInTmp[j].bBegin;
        } else {
            firstMergeSegInTmp[j].bEnd =
                firstMergeSegInTmp[j].aEnd + (SpliterResult.mergeSeg[0][j].bEnd - SpliterResult.mergeSeg[0][j].bBegin);
        }
        firstMergeSegInTmp[j].result = tmpData.begin() + curNum;

        // 更新已从二维转为一维的数字数量
        curNum += (SpliterResult.mergeSeg[0][j].aEnd - SpliterResult.mergeSeg[0][j].aBegin) +
            (SpliterResult.mergeSeg[0][j].bEnd - SpliterResult.mergeSeg[0][j].bBegin);
    }
}

template <typename _Tp>
void QuicklySortMergeFirst(spliter::VectorExternMergeSpliterResult<_Tp> &SpliterResult,
    std::vector<spliter::VectorExternMergeSegment<_Tp>> firstMergeSegInTmp, std::vector<_Tp> &outSortedData,
    std::vector<_Tp> &tmpData, MergeParam &mergeParam)
{
    // 第一次在 tmpData 和 outSortedData 合并，利用 firstMergeSegInTmp 的值
    uint32_t mergeNumSP = 2;
    if (mergeParam.mergeNum >= mergeNumSP) {
        mergeParam.segNumStatus = (mergeParam.mergeSegNum % mergeParam.mergeTimes == 0) ?
            false :
            true; // 片段个数奇偶状态（奇数个片段需要特殊处理）
        mergeParam.mergeSegNum =
            ceil(mergeParam.mergeSegNum / static_cast<float>(mergeParam.mergeTimes)); // 合并后的片段数目
        SpliterResult.mergeSeg[1].resize(mergeParam.mergeSegNum);
        uint64_t curNumFirst = 0;
        for (uint64_t j = 0; j < mergeParam.mergeSegNum; j++) {
            SpliterResult.mergeSeg[1][j].aBegin = firstMergeSegInTmp[j * mergeParam.mergeTimes].aBegin;
            SpliterResult.mergeSeg[1][j].aEnd = firstMergeSegInTmp[j * mergeParam.mergeTimes].bEnd;

            if (mergeParam.segNumStatus && j == mergeParam.mergeSegNum - 1) { // 奇数个片段，且最后一个片段时，特殊处理
                SpliterResult.mergeSeg[1][j].bBegin = firstMergeSegInTmp[j * mergeParam.mergeTimes].aEnd;
                SpliterResult.mergeSeg[1][j].bEnd = firstMergeSegInTmp[j * mergeParam.mergeTimes].aEnd;
            } else if (j == mergeParam.mergeSegNum - 1 || mergeParam.mergeSegNum == 1) { // 最后两个片段，特殊处理
                SpliterResult.mergeSeg[1][j].bBegin = firstMergeSegInTmp[j * mergeParam.mergeTimes + 1].result;
                SpliterResult.mergeSeg[1][j].bEnd = tmpData.end();
            } else {
                SpliterResult.mergeSeg[1][j].bBegin = firstMergeSegInTmp[j * mergeParam.mergeTimes + 1].aBegin;
                SpliterResult.mergeSeg[1][j].bEnd = firstMergeSegInTmp[j * mergeParam.mergeTimes + 1].bEnd;
            }

            // 偶数次合并，写回 input，fromBegin 还在 output 位置
            SpliterResult.mergeSeg[1][j].result = outSortedData.begin() + curNumFirst;
            curNumFirst += SpliterResult.mergeSeg[1][j].bEnd - SpliterResult.mergeSeg[1][j].aBegin;
        }
    }
}

template <typename _Tp>
void QuicklySortMergeLast(spliter::VectorExternMergeSpliterResult<_Tp> &SpliterResult, std::vector<_Tp> &outSortedData,
    std::vector<_Tp> &tmpData, MergeParam &mergeParam)
{
    // 后 mergeNum-1 次合并：与 CalcQuicklyExternSortSpliterSegments 合并逻辑相同
    for (uint64_t i = 2; i < mergeParam.mergeNum; i++) {
        mergeParam.segNumStatus = (mergeParam.mergeSegNum % mergeParam.mergeTimes == 0) ?
            false :
            true; // 片段个数奇偶状态（奇数个片段需要特殊处理）
        mergeParam.mergeSegNum =
            ceil(mergeParam.mergeSegNum / static_cast<float>(mergeParam.mergeTimes)); // 合并后的片段数目
        QuicklySortMergeLastImpl(SpliterResult, outSortedData, tmpData, mergeParam, i);
    }
}

template <typename _Tp>
void QuicklySortMergeLastImpl(spliter::VectorExternMergeSpliterResult<_Tp> &SpliterResult,
    std::vector<_Tp> &outSortedData, std::vector<_Tp> &tmpData, MergeParam &mergeParam, uint64_t layerNum)
{
    uint64_t curNumSecond = 0;
    uint64_t i = layerNum;
    for (uint64_t j = 0; j < mergeParam.mergeSegNum; j++) {
        // 上一批合并数据的写入位置，为本次数据合并的开始位置
        // 起始位置具有 2 倍关系
        spliter::VectorExternMergeSegment<_Tp> curSpliterResult;
        curSpliterResult.aBegin = SpliterResult.mergeSeg[i - 1][j * mergeParam.mergeTimes].result;

        // segmentNum != 2^n 时，只有以下的 Pos 元素受影响；length结果由 Pos 计算得到，也受影响
        if (mergeParam.segNumStatus && j == mergeParam.mergeSegNum - 1) { // 奇数个片段，且最后一个片段时，特殊处理
            if (i % mergeParam.mergeTimes == 1) {
                curSpliterResult.aEnd = tmpData.end();
                curSpliterResult.bBegin = tmpData.end();
                curSpliterResult.bEnd = tmpData.end();
            } else {
                curSpliterResult.aEnd = outSortedData.end();
                curSpliterResult.bBegin = outSortedData.end();
                curSpliterResult.bEnd = outSortedData.end();
            }
        } else if (j == mergeParam.mergeSegNum - 1 || mergeParam.mergeSegNum == 1) { // 最后两个片段，特殊处理
            curSpliterResult.aEnd = SpliterResult.mergeSeg[i - 1][j * mergeParam.mergeTimes].result +
                (SpliterResult.mergeSeg[i - 1][j * mergeParam.mergeTimes].bEnd -
                SpliterResult.mergeSeg[i - 1][j * mergeParam.mergeTimes].aBegin);
            curSpliterResult.bBegin = SpliterResult.mergeSeg[i - 1][j * mergeParam.mergeTimes + 1].result;
            if (i % mergeParam.mergeTimes == 1) {
                curSpliterResult.bEnd = tmpData.end();
            } else {
                curSpliterResult.bEnd = outSortedData.end();
            }
        } else {
            curSpliterResult.aEnd = SpliterResult.mergeSeg[i - 1][j * mergeParam.mergeTimes].result +
                (SpliterResult.mergeSeg[i - 1][j * mergeParam.mergeTimes].bEnd -
                SpliterResult.mergeSeg[i - 1][j * mergeParam.mergeTimes].aBegin);
            curSpliterResult.bBegin = SpliterResult.mergeSeg[i - 1][j * mergeParam.mergeTimes + 1].result;
            curSpliterResult.bEnd = SpliterResult.mergeSeg[i - 1][j * mergeParam.mergeTimes + 1].result +
                (SpliterResult.mergeSeg[i - 1][j * mergeParam.mergeTimes + 1].bEnd -
                SpliterResult.mergeSeg[i - 1][j * mergeParam.mergeTimes + 1].aBegin);
        }

        if (i % mergeParam.mergeTimes == 1) {
            // 偶数次合并，写回 input，fromBegin 还在 output 位置
            curSpliterResult.result = outSortedData.begin() + curNumSecond;
        } else {
            // 奇数次合并，写入 output，fromBegin 还在 input 位置
            curSpliterResult.result = tmpData.begin() + curNumSecond;
        }
        SpliterResult.mergeSeg[i].push_back(curSpliterResult);
        curNumSecond += SpliterResult.mergeSeg[i][j].bEnd - SpliterResult.mergeSeg[i][j].aBegin;
    }
}
}

template <typename _Tp>
std::shared_ptr<spliter::VectorExternMergeSpliterResult<_Tp>> CalcQuicklyExternMergeSpliterSegments(
    std::vector<std::vector<_Tp>> &sortedGroups, std::vector<_Tp> &outSortedData, std::vector<_Tp> &tmpData)
{
    uint64_t segmentNum = sortedGroups.size();  // 片段个数
    uint64_t mergeNum = segmentNum > 1 ? ceil(log2(segmentNum)) : 1; // 合并次数
    uint64_t mergeTimes = 2;                    // 合并倍数，两两合并，合并的片段倍数关系为 2

    // 统计元素个数
    uint64_t dataCount = 0;
    for (uint64_t i = 0; i < segmentNum; i++) {
        dataCount += sortedGroups[i].size();
    }

    spliter::MergeParam mergeParam; // 计算参数
    mergeParam.dataCount = dataCount;
    mergeParam.mergeTimes = mergeTimes;
    mergeParam.mergeSegNum = segmentNum;
    mergeParam.mergeNum = mergeNum;
    mergeParam.segNumStatus = (mergeParam.mergeSegNum % mergeParam.mergeTimes == 0) ?
        false :
        true; // 片段个数奇偶状态（奇数个片段需要特殊处理）
    mergeParam.mergeSegNum =
        ceil(mergeParam.mergeSegNum / static_cast<float>(mergeParam.mergeTimes)); // 合并后的片段数目

    auto SpliterResult = std::make_shared<spliter::VectorExternMergeSpliterResult<_Tp>>(); // 计算结果
    SpliterResult->isSwap = false;
    SpliterResult->mergeSeg.resize(mergeNum); // 设置 mergeSeg 行数

    std::vector<spliter::VectorExternMergeSegment<_Tp>> firstMergeSegInTmp;
    firstMergeSegInTmp.resize(mergeParam.mergeSegNum);

    spliter::QuicklySortMergeTwoDim(*SpliterResult, firstMergeSegInTmp, sortedGroups, tmpData, mergeParam);

    spliter::QuicklySortMergeFirst(*SpliterResult, firstMergeSegInTmp, outSortedData, tmpData, mergeParam);
    spliter::QuicklySortMergeLast(*SpliterResult, outSortedData, tmpData, mergeParam);

    // isSwap
    // 奇数次合并，结果还在 tmpdatas 处，进行一次 swap
    SpliterResult->isSwap = (mergeNum % mergeParam.mergeTimes == 1) ? true : false;
    return SpliterResult;
}
} // namespace algo
} // namespace hcps
} // namespace ock
#endif // HCPS_PIER_OCKQUICKLYEXTERNMERGESPLITERSEGMENTSIMPL_H
