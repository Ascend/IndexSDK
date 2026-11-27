/*
 * -------------------------------------------------------------------------
 * This file is part of the IndexSDK project.
 * Copyright (c) 2026 Huawei Technologies Co.,Ltd.
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

#ifndef ASCEND_BM25_SEARCH_PLAN_H
#define ASCEND_BM25_SEARCH_PLAN_H
// Compute the leaf layout, merge levels and memory bounds before allocating device memory.
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

#include "ascendhost/include/impl/Bm25Error.h"
namespace bm25large
{
constexpr uint64_t DOCUMENTS_PER_LEAF = 8192, MERGE_FAN_IN = 32, MAX_LEAF_COUNT = 65536,
                   INT32_DIMENSION_LIMIT = 2147483647;
constexpr uint64_t MAX_QUERY_COUNT = 10000;
constexpr uint64_t MAX_QUERY_WORD_CAPACITY = 16777216;
// Keep in sync with kTerms in bm25_fused_score_top_k.cpp.
constexpr uint32_t MAX_QUERY_WORDS_PER_QUERY = 64;
constexpr uint64_t MAX_TOP_K = 32;
inline void CheckPlanArgument(bool condition, const char* errorMessage)
{
    if (!condition) throw ascend::bm25::InvalidArgument(errorMessage);
}
inline uint64_t CheckedAdd(uint64_t left, uint64_t right)
{
    CheckPlanArgument(left <= UINT64_MAX - right, "Plan addition overflow");
    return left + right;
}
inline uint64_t CheckedMultiply(uint64_t left, uint64_t right)
{
    CheckPlanArgument(!right || left <= UINT64_MAX / right, "Plan product overflow");
    return left * right;
}
inline uint64_t AlignTo32Bytes(uint64_t byteCount) { return CheckedMultiply(CheckedAdd(byteCount, 31) / 32, 32); }
struct QueryPacketSizes
{
    size_t wordOffsetBytes, inputBytes, idOffsetBytes, outputBytes;
};
inline QueryPacketSizes CalculateQueryPacketSizes(uint64_t queryCount, uint64_t wordCapacity, uint64_t topK,
                                                  const char* invalidCapacityMessage)
{
    CheckPlanArgument(queryCount >= 1 && queryCount <= MAX_QUERY_COUNT && wordCapacity <= MAX_QUERY_WORD_CAPACITY &&
                          topK >= 1 && topK <= MAX_TOP_K,
                      invalidCapacityMessage);
    const auto wordOffsetBytes = AlignTo32Bytes(CheckedMultiply(CheckedAdd(queryCount, 1), sizeof(int64_t)));
    const auto wordBytes = CheckedMultiply(std::max<uint64_t>(1, wordCapacity), sizeof(int32_t));
    const auto inputBytes = AlignTo32Bytes(CheckedAdd(wordOffsetBytes, wordBytes));
    const auto resultCount = CheckedMultiply(queryCount, topK);
    const auto idOffsetBytes = AlignTo32Bytes(CheckedMultiply(resultCount, sizeof(float)));
    const auto outputBytes = AlignTo32Bytes(CheckedAdd(idOffsetBytes, CheckedMultiply(resultCount, sizeof(int32_t))));
    CheckPlanArgument(
        inputBytes <= std::numeric_limits<size_t>::max() && outputBytes <= std::numeric_limits<size_t>::max(),
        "Packet size overflow");
    return {static_cast<size_t>(wordOffsetBytes), static_cast<size_t>(inputBytes), static_cast<size_t>(idOffsetBytes),
            static_cast<size_t>(outputBytes)};
}
struct Plan
{
    uint64_t documentCount, paddedDocumentCount, leafCount, queryCount, topK, vocabSize, postingCount, inputBytes,
        outputBytes, indexBytes, candidateBytes = 0, plannedDeviceBytes;
    std::vector<uint64_t> levelListCounts;
    Plan(uint64_t documentCount, uint64_t vocabSize, uint64_t postingCount, uint64_t queryCount, uint64_t topK,
         uint64_t queryWordCapacity)
        : documentCount(documentCount),
          queryCount(queryCount),
          topK(topK),
          vocabSize(vocabSize),
          postingCount(postingCount)
    {
        constexpr const char* invalidDimensions = "Unsupported index/query/K dimensions";
        CheckPlanArgument(documentCount >= topK && documentCount <= INT32_DIMENSION_LIMIT && vocabSize >= 1 &&
                              vocabSize < INT32_DIMENSION_LIMIT && postingCount >= 1 &&
                              postingCount <= INT32_DIMENSION_LIMIT,
                          invalidDimensions);
        const auto packetSizes = CalculateQueryPacketSizes(queryCount, queryWordCapacity, topK, invalidDimensions);
        inputBytes = packetSizes.inputBytes;
        outputBytes = packetSizes.outputBytes;
        leafCount = CheckedAdd(documentCount, DOCUMENTS_PER_LEAF - 1) / DOCUMENTS_PER_LEAF;
        paddedDocumentCount = CheckedMultiply(leafCount, DOCUMENTS_PER_LEAF);
        CheckPlanArgument(leafCount <= MAX_LEAF_COUNT && paddedDocumentCount <= INT32_DIMENSION_LIMIT &&
                              queryCount <= INT32_DIMENSION_LIMIT / paddedDocumentCount,
                          "Padded index/Q*N exceeds explicit INT32 contract");
        levelListCounts.push_back(leafCount);
        do
        {
            levelListCounts.push_back(CheckedAdd(levelListCounts.back(), MERGE_FAN_IN - 1) / MERGE_FAN_IN);
        } while (levelListCounts.back() > 1);
        indexBytes = CheckedAdd(
            CheckedAdd(CheckedMultiply(postingCount, 8), CheckedMultiply(paddedDocumentCount, 4)),
            CheckedAdd(CheckedMultiply(vocabSize, 4), CheckedMultiply(CheckedMultiply(vocabSize, leafCount + 1), 8)));
        for (auto levelListCount : levelListCounts)
            candidateBytes = CheckedAdd(
                candidateBytes, CheckedMultiply(CheckedMultiply(CheckedMultiply(queryCount, levelListCount), topK), 8));
        // Five index arenas, score/id pair per tree level, input/output/placeholder arenas.
        plannedDeviceBytes =
            CheckedAdd(CheckedAdd(indexBytes, candidateBytes),
                       CheckedAdd(CheckedAdd(inputBytes, outputBytes),
                                  CheckedAdd(4, CheckedMultiply(8 + 2 * levelListCounts.size(), 128))));
    }
};
}  // namespace bm25large

namespace bm25plan
{
inline std::vector<int64_t> BuildPostingDirectory(const std::vector<int64_t>& termOffsets,
                                                  const std::vector<int32_t>& postingDocIds, size_t documentCount,
                                                  size_t leafCount)
{
    if (!documentCount || !leafCount || leafCount > documentCount || termOffsets.size() < 2 ||
        termOffsets.front() != 0 || termOffsets.back() != static_cast<int64_t>(postingDocIds.size()))
        throw std::runtime_error("Invalid directory dimensions");
    const size_t vocabSize = termOffsets.size() - 1;
    if (vocabSize > SIZE_MAX / (leafCount + 1)) throw std::runtime_error("Directory size overflow");
    std::vector<int64_t> postingDirectory(vocabSize * (leafCount + 1));
    for (size_t termId = 0; termId < vocabSize; ++termId)
    {
        const int64_t postingBegin = termOffsets[termId];
        const int64_t postingEnd = termOffsets[termId + 1];
        if (postingBegin < 0 || postingEnd < postingBegin || postingEnd > static_cast<int64_t>(postingDocIds.size()))
            throw std::runtime_error("Invalid directory term boundaries");
        auto postingCursor = postingDocIds.begin() + postingBegin;
        auto postingEndIterator = postingDocIds.begin() + postingEnd;
        for (size_t leafIndex = 0; leafIndex <= leafCount; ++leafIndex)
        {
            const int64_t leafBoundaryDocId =
                static_cast<int64_t>(documentCount) * static_cast<int64_t>(leafIndex) / static_cast<int64_t>(leafCount);
            postingCursor = std::lower_bound(postingCursor, postingEndIterator, leafBoundaryDocId);
            postingDirectory[termId * (leafCount + 1) + leafIndex] =
                static_cast<int64_t>(postingCursor - postingDocIds.begin());
        }
        if (postingDirectory[termId * (leafCount + 1)] != postingBegin ||
            postingDirectory[termId * (leafCount + 1) + leafCount] != postingEnd)
            throw std::runtime_error("Directory endpoints disagree with index");
    }
    return postingDirectory;
}
}  // namespace bm25plan

#endif  // ASCEND_BM25_SEARCH_PLAN_H
