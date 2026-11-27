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

#ifndef ASCEND_BM25_QUERY_PACKET_H
#define ASCEND_BM25_QUERY_PACKET_H
// Pack validated queries into the session-owned input arena.
#include <cstddef>
#include <cstdint>
#include <cstring>

#include "ascendhost/include/impl/Bm25Error.h"
#include "ascendhost/include/impl/Bm25SearchPlan.h"
#include "ascendhost/include/index/AscendIndexBm25.h"

namespace bm25search
{
inline void CheckQueryArgument(bool condition, const char* message)
{
    if (!condition) throw ascend::bm25::InvalidArgument(message);
}
}  // namespace bm25search

namespace bm25plan
{
struct PacketLayout
{
    size_t queryCount, wordCapacity, wordOffsetBytes, inputBytes, idOffsetBytes, outputBytes, topK;
    PacketLayout(size_t queryCount, size_t wordCapacity, size_t topK)
        : queryCount(queryCount), wordCapacity(wordCapacity), topK(topK)
    {
        const auto sizes =
            bm25large::CalculateQueryPacketSizes(queryCount, wordCapacity, topK, "Invalid fixed packet capacity");
        wordOffsetBytes = sizes.wordOffsetBytes;
        inputBytes = sizes.inputBytes;
        idOffsetBytes = sizes.idOffsetBytes;
        outputBytes = sizes.outputBytes;
    }
    void Pack(const bm25search::Queries& queries, void* packetMemory) const
    {
        bm25search::CheckQueryArgument(packetMemory && queries.size() == queryCount,
                                       "Fixed session requires its declared query count");
        size_t totalQueryWords = 0;
        // Validate the complete request before modifying memory used by the plan.
        for (const auto& queryWords : queries)
        {
            bm25search::CheckQueryArgument(queryWords.size() <= bm25large::MAX_QUERY_WORDS_PER_QUERY &&
                                               queryWords.size() <= wordCapacity - totalQueryWords,
                                           "Query exceeds fixed word capacity or per-query term limit");
            totalQueryWords += queryWords.size();
            for (int32_t wordId : queryWords)
                bm25search::CheckQueryArgument(wordId >= 0 && wordId < INT32_MAX, "Invalid public word ID");
        }
        auto* packetBytes = static_cast<unsigned char*>(packetMemory);
        std::memset(packetBytes, 0, inputBytes);  // Includes unused words, offsets and alignment padding.
        int64_t wordEndOffset = 0;
        for (size_t queryIndex = 0; queryIndex < queries.size(); ++queryIndex)
        {
            const auto& queryWords = queries[queryIndex];
            if (!queryWords.empty())
                std::memcpy(packetBytes + wordOffsetBytes + static_cast<size_t>(wordEndOffset) * 4, queryWords.data(),
                            queryWords.size() * 4);
            wordEndOffset += static_cast<int64_t>(queryWords.size());
            std::memcpy(packetBytes + (queryIndex + 1) * 8, &wordEndOffset, 8);
        }
    }
};
}  // namespace bm25plan

#endif  // ASCEND_BM25_QUERY_PACKET_H
