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


#ifndef OCK_HCPS_ALGO_SPLITER_SEGMENT_H
#define OCK_HCPS_ALGO_SPLITER_SEGMENT_H
#include <cstdint>
#include <memory>
#include <vector>
#include "ock/utils/OckCompareUtils.h"
namespace ock {
namespace hcps {
namespace algo {
namespace spliter {
template <typename _Tp> struct VectorSortSegment {
    typename std::vector<_Tp>::iterator begin{ nullptr };
    typename std::vector<_Tp>::iterator end{ nullptr };
};
template <typename _Tp> struct VectorExternMergeSegment {
    typename std::vector<_Tp>::iterator aBegin{ nullptr };
    typename std::vector<_Tp>::iterator aEnd{ nullptr };
    typename std::vector<_Tp>::iterator bBegin{ nullptr };
    typename std::vector<_Tp>::iterator bEnd{ nullptr };
    typename std::vector<_Tp>::iterator result{ nullptr };
};
template <typename _Tp> struct VectorExternMergeSpliterResult {
    std::vector<std::vector<VectorExternMergeSegment<_Tp>>> mergeSeg{}; // 记录合并过程
    bool isSwap{ false };                                               // 最终outputDatas与tmpDatas间是否需要swap
};
template <typename _Tp> struct VectorSortSpliterResult {
    std::vector<VectorSortSegment<_Tp>> sortSeg{};                      // 记录分割过程
    std::vector<std::vector<VectorExternMergeSegment<_Tp>>> mergeSeg{}; // 记录合并过程
    bool isSwap{ false }; // 最终outputDatas与inputDatas间是否需要swap
};
} // namespace spliter
} // namespace algo
} // namespace hcps
} // namespace ock
#endif
