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


#ifndef OCK_HCPS_ALGO_QUICKLY_EXTERN_SORT_SPLITER_IMPL_H
#define OCK_HCPS_ALGO_QUICKLY_EXTERN_SORT_SPLITER_IMPL_H
#include <cstdint>
#include <memory>
#include <vector>
#include <cmath>
#include "ock/utils/OckCompareUtils.h"
#include "ock/hcps/algo/OckSpliterSegement.h"
#include "ock/utils/OckSafeUtils.h"
namespace ock {
namespace hcps {
namespace algo {
namespace spliter {
template <typename _Tp> struct VectorSortSegmentPos {
    typename std::vector<_Tp>::iterator dataBegin{ nullptr };
    uint64_t startPos{ 0 };
    uint64_t endPos{ 0 };
};
template <typename _Tp> struct VectorInplaceMergeSegment {
    typename std::vector<_Tp>::iterator fromBegin{ nullptr };
    uint64_t startPos{ 0 };
    uint64_t middlePos{ 0 };
    uint64_t endPos{ 0 };
    typename std::vector<_Tp>::iterator toBegin{ nullptr };
};
template <typename _Tp> struct VectorSortSpliterResultPos {
    std::vector<VectorSortSegmentPos<_Tp>> sortSeg{};
    std::vector<std::vector<VectorInplaceMergeSegment<_Tp>>> mergeSeg{};
    bool isSwap{ false };
};

enum SegNumStatus {
    evenSegNum = 0,
    oddSegNum = 1
};
struct SortSpliterParam {
    uint64_t dataCount{ 0 };        // 数据量
    uint64_t splitThreshold{ 0 };   // 分割阈值
    uint64_t mergeTimes{ 0 };       // 合并倍数，两两合并，合并的片段倍数关系为 2
    uint64_t mergeSegNum{ 0 };      // 初始分割的片段数
    uint64_t mergeNum{ 0 };         // 合并次数
    SegNumStatus segNumStatus{ evenSegNum }; // 合并状态
};

template <typename _Tp>
void QuicklySortSeg(std::vector<_Tp> &inputDatas, std::vector<VectorSortSegmentPos<_Tp>> &sortSeg,
    SortSpliterParam &mergeParam);
template <typename _Tp>
void QuicklySortMerge(std::vector<_Tp> &inputDatas, std::vector<_Tp> &outputDatas,
    VectorSortSpliterResultPos<_Tp> &mergeResult, SortSpliterParam &mergeParam);
template <typename _Tp>
void QuicklySortMergeImpl(std::vector<_Tp> &inputDatas, std::vector<_Tp> &outputDatas,
    VectorSortSpliterResultPos<_Tp> &mergeResult, SortSpliterParam &mergeParam, uint64_t layerNum);
template <typename _Tp>
void ConvertInnerVectorSortSegmentToOutterSegment(std::vector<VectorSortSegmentPos<_Tp>> &innerSegs,
    std::vector<VectorSortSegment<_Tp>> &outterSegs);
template <typename _Tp>
void ConvertVectorInplaceMergeSegmentToVectorExternMergeSegment(
    std::vector<VectorInplaceMergeSegment<_Tp>> &innerMergeSegs,
    std::vector<VectorExternMergeSegment<_Tp>> &outterMergeSegs);
template <typename _Tp>
void ConvertVectorInplaceMergeSegmentToOutterMergeSegment(
    std::vector<std::vector<VectorInplaceMergeSegment<_Tp>>> &innerMergeSegs,
    std::vector<std::vector<VectorExternMergeSegment<_Tp>>> &outterMergeSegs);

/**
 * @brief 计算分割细节
 */
template <typename _Tp>
void QuicklySortSeg(std::vector<_Tp> &inputDatas, std::vector<VectorSortSegmentPos<_Tp>> &sortSeg,
    SortSpliterParam &mergeParam)
{
    for (uint64_t i = 0; i < mergeParam.mergeSegNum; i++) {
        sortSeg[i].dataBegin = inputDatas.begin() + i * mergeParam.splitThreshold; // 第 i 块的起始迭代器
        sortSeg[i].startPos = 0;
        sortSeg[i].endPos = (i == mergeParam.mergeSegNum - 1) ? mergeParam.dataCount - i * mergeParam.splitThreshold :
                                                                mergeParam.splitThreshold;
    }
}

/**
 * @brief 计算合并细节
 */
template <typename _Tp>
void QuicklySortMerge(std::vector<_Tp> &inputDatas, std::vector<_Tp> &outputDatas,
    VectorSortSpliterResultPos<_Tp> &mergeResult, SortSpliterParam &mergeParam)
{
    // 计算合并片段
    for (uint64_t i = 0; i <= mergeParam.mergeNum; i++) {
        mergeParam.segNumStatus = (mergeParam.mergeSegNum % mergeParam.mergeTimes == 0) ? evenSegNum : oddSegNum;
        mergeParam.mergeSegNum = static_cast<uint64_t>(ceil(static_cast<float>(mergeParam.mergeSegNum) /
            static_cast<float>(mergeParam.mergeTimes))); // 合并后的片段数目
        mergeResult.mergeSeg[i].resize(mergeParam.mergeSegNum);
        QuicklySortMergeImpl(inputDatas, outputDatas, mergeResult, mergeParam, i);
    }
}

/**
 * @brief 计算单次的合并细节
 */
template <typename _Tp>
void QuicklySortMergeImpl(std::vector<_Tp> &inputDatas, std::vector<_Tp> &outputDatas,
    VectorSortSpliterResultPos<_Tp> &mergeResult, SortSpliterParam &mergeParam, uint64_t layerNum)
{
    for (uint64_t j = 0; j < mergeParam.mergeSegNum; j++) {
        // 上一批合并数据的写入位置，为本次数据合并的开始位置
        // 起始位置具有 2 倍关系
        if (layerNum == 0) {
            mergeResult.mergeSeg[layerNum][j].fromBegin = mergeResult.sortSeg[j * mergeParam.mergeTimes].dataBegin;
        } else {
            mergeResult.mergeSeg[layerNum][j].fromBegin =
                mergeResult.mergeSeg[layerNum - 1][j * mergeParam.mergeTimes].toBegin;
        }
        mergeResult.mergeSeg[layerNum][j].startPos = 0;

        // segmentNum != 2^n 时，只有以下的 Pos 元素受影响；length结果由 Pos 计算得到，也受影响
        if (mergeParam.segNumStatus == oddSegNum &&
            j == mergeParam.mergeSegNum - 1) {
            mergeResult.mergeSeg[layerNum][j].middlePos = 0;
            mergeResult.mergeSeg[layerNum][j].endPos = mergeParam.dataCount -
                (mergeResult.mergeSeg[layerNum][j].fromBegin - mergeResult.mergeSeg[layerNum][0].fromBegin);
        } else if (j == mergeParam.mergeSegNum - 1 || mergeParam.mergeSegNum == 1) { // 最后两个片段，特殊处理
            mergeResult.mergeSeg[layerNum][j].middlePos =
                static_cast<uint64_t>(pow(static_cast<double>(mergeParam.mergeTimes), static_cast<double>(layerNum))) *
                mergeParam.splitThreshold;
            mergeResult.mergeSeg[layerNum][j].endPos = mergeParam.dataCount -
                (mergeResult.mergeSeg[layerNum][j].fromBegin - mergeResult.mergeSeg[layerNum][0].fromBegin);
        } else {
            mergeResult.mergeSeg[layerNum][j].middlePos =
                static_cast<uint64_t>(pow(static_cast<double>(mergeParam.mergeTimes), static_cast<double>(layerNum))) *
                mergeParam.splitThreshold;
            mergeResult.mergeSeg[layerNum][j].endPos =
                static_cast<uint64_t>(pow(static_cast<double>(mergeParam.mergeTimes), static_cast<double>(layerNum))) *
                mergeParam.splitThreshold * mergeParam.mergeTimes;
        }

        if (layerNum % mergeParam.mergeTimes == 1) {
            // 偶数次合并，写回 input，fromBegin 还在 output 位置
            uint64_t curBeginPosIndex = mergeResult.mergeSeg[layerNum][j].fromBegin - outputDatas.begin();
            mergeResult.mergeSeg[layerNum][j].toBegin = inputDatas.begin() + curBeginPosIndex;
        } else {
            // 奇数次合并，写入 output，fromBegin 还在 input 位置
            uint64_t curBeginPosIndex = mergeResult.mergeSeg[layerNum][j].fromBegin - inputDatas.begin();
            mergeResult.mergeSeg[layerNum][j].toBegin = outputDatas.begin() + curBeginPosIndex;
        }
    }
}

/**
 * @brief VectorSortSegmentPos 结果转为 VectorSortSegment
 */
template <typename _Tp>
void ConvertInnerVectorSortSegmentToOutterSegment(std::vector<VectorSortSegmentPos<_Tp>> &innerSegs,
    std::vector<VectorSortSegment<_Tp>> &outterSegs)
{
    for (auto &innerSeg : innerSegs) {
        VectorSortSegment<_Tp> outSeg;
        outSeg.begin = innerSeg.dataBegin;
        std::advance(outSeg.begin, innerSeg.startPos);

        outSeg.end = innerSeg.dataBegin;
        std::advance(outSeg.end, innerSeg.endPos);

        outterSegs.push_back(outSeg);
    }
}

/**
 * @brief VectorInplaceMergeSegment 结果转为 VectorExternMergeSegment
 */
template <typename _Tp>
void ConvertVectorInplaceMergeSegmentToVectorExternMergeSegment(
    std::vector<VectorInplaceMergeSegment<_Tp>> &innerMergeSegs,
    std::vector<VectorExternMergeSegment<_Tp>> &outterMergeSegs)
{
    for (auto &innerMergeSeg : innerMergeSegs) {
        VectorExternMergeSegment<_Tp> outterMergeSeg;
        outterMergeSeg.aBegin = innerMergeSeg.fromBegin;
        std::advance(outterMergeSeg.aBegin, innerMergeSeg.startPos);
        outterMergeSeg.aEnd = innerMergeSeg.fromBegin;
        std::advance(outterMergeSeg.aEnd, innerMergeSeg.middlePos);
        outterMergeSeg.bBegin = innerMergeSeg.fromBegin;
        std::advance(outterMergeSeg.bBegin, innerMergeSeg.middlePos);
        outterMergeSeg.bEnd = innerMergeSeg.fromBegin;
        std::advance(outterMergeSeg.bEnd, innerMergeSeg.endPos);
        outterMergeSeg.result = innerMergeSeg.toBegin;

        outterMergeSegs.push_back(outterMergeSeg);
    }
}

/**
 * @brief VectorInplaceMergeSegment 结果转为 VectorExternMergeSegment
 */
template <typename _Tp>
void ConvertSpliterResultPosToSpliterResult(std::vector<std::vector<VectorInplaceMergeSegment<_Tp>>> &innerMergeSegs,
    std::vector<std::vector<VectorExternMergeSegment<_Tp>>> &outterMergeSegs)
{
    for (auto &innerMergeSeg : innerMergeSegs) {
        std::vector<VectorExternMergeSegment<_Tp>> outterMergeSeg;
        ConvertVectorInplaceMergeSegmentToVectorExternMergeSegment(innerMergeSeg, outterMergeSeg);
        outterMergeSegs.push_back(outterMergeSeg);
    }
}
} // namespace spliter

/**
 * @brief 计算分割和和合并细节
 */
template <typename _Tp>
std::shared_ptr<spliter::VectorSortSpliterResult<_Tp>> CalcQuicklyExternSortSpliterSegments(
    std::vector<_Tp> &inputDatas, std::vector<_Tp> &outputDatas, uint64_t splitThreshold)
{
    uint64_t dataCount = inputDatas.size();
    uint64_t segmentNum = utils::SafeDiv((dataCount + splitThreshold - 1UL), splitThreshold); // 分割的片段数量，向上取整
    uint64_t mergeNum = static_cast<uint64_t>(ceil(log2(static_cast<double>(segmentNum)))); // 合并次数
    uint64_t mergeTimes = 2UL; // 合并倍数，两两合并，合并的片段倍数关系为 2

    spliter::SortSpliterParam mergeParam; // 计算参数
    mergeParam.dataCount = dataCount;
    mergeParam.splitThreshold = splitThreshold;
    mergeParam.mergeTimes = mergeTimes;
    mergeParam.mergeSegNum = segmentNum;
    mergeParam.mergeNum = mergeNum;

    auto spliterResult = std::make_shared<spliter::VectorSortSpliterResultPos<_Tp>>(); // 计算结果
    spliterResult->isSwap = false;
    spliterResult->sortSeg.resize(segmentNum);
    spliterResult->mergeSeg.resize(mergeNum + 1); // 设置 mergeSeg 行数 = mergeNum+1; 多存储开始未合并的片段

    // 分割
    spliter::QuicklySortSeg(inputDatas, spliterResult->sortSeg, mergeParam);
    // 合并
    spliter::QuicklySortMerge(inputDatas, outputDatas, *spliterResult, mergeParam);
    // isSwap
    // 偶数次合并，结果还在 inputdatas 处，额外进行一次复制
    spliterResult->isSwap = (mergeNum % mergeParam.mergeTimes == 0) ? true : false;

    auto spliterOutResult = std::make_shared<spliter::VectorSortSpliterResult<_Tp>>();
    ConvertInnerVectorSortSegmentToOutterSegment(spliterResult->sortSeg, spliterOutResult->sortSeg);
    ConvertSpliterResultPosToSpliterResult(spliterResult->mergeSeg, spliterOutResult->mergeSeg);

    return spliterOutResult;
}
} // namespace algo
} // namespace hcps
} // namespace ock
#endif
