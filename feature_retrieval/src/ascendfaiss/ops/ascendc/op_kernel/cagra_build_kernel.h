/*
 * Copyright (c) 2026 Huawei Technologies Co.,Ltd.
 * Licensed under Mulan PSL v2.
 */

#ifndef ASCENDC_CAGRA_BUILD_KERNEL_H
#define ASCENDC_CAGRA_BUILD_KERNEL_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "simt_api/device_warp_functions.h"

using namespace AscendC;

namespace
{
constexpr uint32_t kThreadsPerBlock = 512;
constexpr uint32_t kDistanceTeamSize = 8;
constexpr uint32_t kCachedDistanceDim = 128;
constexpr uint32_t kHighDimTile = 128;
constexpr uint32_t kHighDimRhsBatch = 4;
constexpr uint32_t kBatchedDistanceMaxDim = 512;
constexpr uint32_t kTeamsPerBlock = kThreadsPerBlock / kDistanceTeamSize;
constexpr uint32_t kPruneTeamSize = 32;
constexpr uint32_t kPruneTeamsPerBlock = kThreadsPerBlock / kPruneTeamSize;
constexpr uint32_t kMaxPruneIntermediateDegree = 128;
constexpr uint32_t kPruneSlotsPerLane = kMaxPruneIntermediateDegree / kPruneTeamSize;
constexpr uint32_t kCachedValuesPerLane = kCachedDistanceDim / kDistanceTeamSize;
constexpr uint32_t kSizeFieldCount = 4;
constexpr uint32_t kForwardNewSizeIndex = 0;
constexpr uint32_t kForwardOldSizeIndex = 1;
constexpr uint32_t kReverseNewSizeIndex = 2;
constexpr uint32_t kReverseOldSizeIndex = 3;
static_assert((kDistanceTeamSize & (kDistanceTeamSize - 1)) == 0);
static_assert(kThreadsPerBlock % kDistanceTeamSize == 0);
static_assert(kThreadsPerBlock % kPruneTeamSize == 0);
static_assert(kMaxPruneIntermediateDegree % kPruneTeamSize == 0);
static_assert(kPruneSlotsPerLane == 4, "update the explicit selected masks when changing the prune degree limit");
constexpr uint32_t kInvalidNode = 0xffffffffU;
constexpr uint32_t kOldMask = 0x80000000U;
constexpr uint32_t kNodeMask = 0x7fffffffU;
constexpr float kFloatMax = 3.402823466e+38F;

__simt_callee__ __aicore__ inline uint32_t RawNode(uint32_t node) { return node & kNodeMask; }

__simt_callee__ __aicore__ inline bool IsNew(uint32_t node) { return node != kInvalidNode && (node & kOldMask) == 0; }

__simt_callee__ __aicore__ inline uint32_t Mix32(uint32_t value)
{
    value ^= value >> 16;
    value *= 0x7feb352dU;
    value ^= value >> 15;
    value *= 0x846ca68bU;
    value ^= value >> 16;
    return value;
}

__simt_callee__ __aicore__ inline float L2Distance(__gm__ const float *data, uint32_t lhs, uint32_t rhs, uint32_t dim)
{
    float distance = 0.0F;
    uint64_t lhsOffset = static_cast<uint64_t>(lhs) * dim;
    uint64_t rhsOffset = static_cast<uint64_t>(rhs) * dim;
    for (uint32_t d = 0; d < dim; ++d)
    {
        float delta = data[lhsOffset + d] - data[rhsOffset + d];
        distance += delta * delta;
    }
    return distance;
}

__simt_callee__ __aicore__ inline bool Better(float lhsDistance, uint32_t lhsNode, float rhsDistance, uint32_t rhsNode)
{
    return lhsDistance < rhsDistance || (lhsDistance == rhsDistance && RawNode(lhsNode) < RawNode(rhsNode));
}

__simt_callee__ __aicore__ inline bool RowContains(__gm__ const uint32_t *row, uint32_t size, uint32_t node)
{
    for (uint32_t i = 0; i < size; ++i)
    {
        if (RawNode(row[i]) == node)
        {
            return true;
        }
    }
    return false;
}

__simt_callee__ __aicore__ inline uint32_t CombinedNode(__gm__ const uint32_t *forward, __gm__ const uint32_t *reverse,
                                                        uint32_t forwardSize, uint32_t index, uint32_t row,
                                                        uint32_t sampleDegree)
{
    if (index < forwardSize)
    {
        return forward[static_cast<uint64_t>(row) * sampleDegree + index];
    }
    return reverse[static_cast<uint64_t>(row) * sampleDegree + index - forwardSize];
}

__simt_callee__ __aicore__ inline bool IsFirstCombinedOccurrence(__gm__ const uint32_t *forward,
                                                                 __gm__ const uint32_t *reverse, uint32_t forwardSize,
                                                                 uint32_t index, uint32_t row, uint32_t sampleDegree)
{
    // A graph row is unique, therefore both the sampled forward list and the generated reverse
    // list are individually unique. Only a forward/reverse overlap needs to be removed.
    if (index < forwardSize)
    {
        return true;
    }
    const uint32_t node = RawNode(reverse[static_cast<uint64_t>(row) * sampleDegree + index - forwardSize]);
    const uint64_t forwardOffset = static_cast<uint64_t>(row) * sampleDegree;
    for (uint32_t previous = 0; previous < forwardSize; ++previous)
    {
        if (RawNode(forward[forwardOffset + previous]) == node)
        {
            return false;
        }
    }
    return true;
}

__simt_callee__ __aicore__ inline void AppendCandidate(uint32_t target, uint32_t candidate, float distance,
                                                       __gm__ uint32_t *candidateIds, __gm__ float *candidateDistances,
                                                       __gm__ uint32_t *candidateCounts, uint32_t candidateDegree)
{
    uint32_t slot = AscendC::Simt::AtomicAdd(candidateCounts + target, 1U);
    if (slot < candidateDegree)
    {
        uint64_t offset = static_cast<uint64_t>(target) * candidateDegree + slot;
        candidateIds[offset] = candidate;
        candidateDistances[offset] = distance;
    }
}

__simt_callee__ __aicore__ inline void FindBestHighDimCandidate(
    __gm__ const float *data, __gm__ const uint32_t *forward, __gm__ const uint32_t *reverse, uint32_t forwardSize,
    uint32_t combinedSize, uint32_t row, uint32_t sampleDegree, uint32_t lhs, uint32_t nodeNum, uint32_t dim,
    uint32_t laneId, uint32_t &bestNode, float &bestDistance)
{
    bestNode = kInvalidNode;
    bestDistance = kFloatMax;
    const uint64_t lhsOffset = static_cast<uint64_t>(lhs) * dim;
    for (uint32_t batchBegin = 0; batchBegin < combinedSize; batchBegin += kHighDimRhsBatch)
    {
        uint32_t rhsNodes[kHighDimRhsBatch];
        float partialDistances[kHighDimRhsBatch];
        for (uint32_t batchIndex = 0; batchIndex < kHighDimRhsBatch; ++batchIndex)
        {
            rhsNodes[batchIndex] = kInvalidNode;
            partialDistances[batchIndex] = 0.0F;
            const uint32_t rhsIndex = batchBegin + batchIndex;
            if (rhsIndex >= combinedSize)
            {
                continue;
            }
            const uint32_t rhs = CombinedNode(forward, reverse, forwardSize, rhsIndex, row, sampleDegree);
            if (rhs < nodeNum && lhs != rhs &&
                IsFirstCombinedOccurrence(forward, reverse, forwardSize, rhsIndex, row, sampleDegree))
            {
                rhsNodes[batchIndex] = rhs;
            }
        }

        for (uint32_t dimBegin = 0; dimBegin < dim; dimBegin += kHighDimTile)
        {
            const uint32_t dimEnd = dimBegin + kHighDimTile < dim ? dimBegin + kHighDimTile : dim;
            float lhsValues[kHighDimTile / kDistanceTeamSize];
            for (uint32_t d = dimBegin + laneId, slot = 0; d < dimEnd; d += kDistanceTeamSize, ++slot)
            {
                lhsValues[slot] = data[lhsOffset + d];
            }
            for (uint32_t batchIndex = 0; batchIndex < kHighDimRhsBatch; ++batchIndex)
            {
                const uint32_t rhs = rhsNodes[batchIndex];
                if (rhs == kInvalidNode)
                {
                    continue;
                }
                const uint64_t rhsOffset = static_cast<uint64_t>(rhs) * dim;
                for (uint32_t d = dimBegin + laneId, slot = 0; d < dimEnd; d += kDistanceTeamSize, ++slot)
                {
                    const float delta = lhsValues[slot] - data[rhsOffset + d];
                    partialDistances[batchIndex] += delta * delta;
                }
            }
        }

        for (uint32_t batchIndex = 0; batchIndex < kHighDimRhsBatch; ++batchIndex)
        {
            const uint32_t rhs = rhsNodes[batchIndex];
            if (rhs == kInvalidNode)
            {
                continue;
            }
            float distance = partialDistances[batchIndex];
            for (uint32_t offset = kDistanceTeamSize / 2; offset > 0; offset >>= 1)
            {
                distance += __shfl_xor(distance, offset, kDistanceTeamSize);
            }
            if (Better(distance, rhs, bestDistance, bestNode))
            {
                bestDistance = distance;
                bestNode = rhs;
            }
        }
    }
}

__simt_callee__ __aicore__ inline void FindBestHighDimCandidatePaired(
    __gm__ const float *data, __gm__ const uint32_t *forward, __gm__ const uint32_t *reverse, uint32_t forwardSize,
    uint32_t combinedSize, uint32_t row, uint32_t sampleDegree, uint32_t lhs, uint32_t nodeNum, uint32_t dim,
    uint32_t laneId, uint32_t &bestNode, float &bestDistance)
{
    bestNode = kInvalidNode;
    bestDistance = kFloatMax;
    const uint64_t lhsOffset = static_cast<uint64_t>(lhs) * dim;
    for (uint32_t rhsIndex = 0; rhsIndex < combinedSize; rhsIndex += 2)
    {
        uint32_t rhs0 = CombinedNode(forward, reverse, forwardSize, rhsIndex, row, sampleDegree);
        if (rhs0 >= nodeNum || lhs == rhs0 ||
            !IsFirstCombinedOccurrence(forward, reverse, forwardSize, rhsIndex, row, sampleDegree))
        {
            rhs0 = kInvalidNode;
        }

        uint32_t rhs1 = kInvalidNode;
        if (rhsIndex + 1 < combinedSize)
        {
            rhs1 = CombinedNode(forward, reverse, forwardSize, rhsIndex + 1, row, sampleDegree);
            if (rhs1 >= nodeNum || lhs == rhs1 ||
                !IsFirstCombinedOccurrence(forward, reverse, forwardSize, rhsIndex + 1, row, sampleDegree))
            {
                rhs1 = kInvalidNode;
            }
        }
        if (rhs0 == kInvalidNode && rhs1 == kInvalidNode)
        {
            continue;
        }

        const uint64_t rhsOffset0 = rhs0 == kInvalidNode ? 0 : static_cast<uint64_t>(rhs0) * dim;
        const uint64_t rhsOffset1 = rhs1 == kInvalidNode ? 0 : static_cast<uint64_t>(rhs1) * dim;
        float distance0 = 0.0F;
        float distance1 = 0.0F;
        if (rhs0 != kInvalidNode && rhs1 != kInvalidNode)
        {
            for (uint32_t d = laneId; d < dim; d += kDistanceTeamSize)
            {
                const float lhsValue = data[lhsOffset + d];
                const float delta0 = lhsValue - data[rhsOffset0 + d];
                const float delta1 = lhsValue - data[rhsOffset1 + d];
                distance0 += delta0 * delta0;
                distance1 += delta1 * delta1;
            }
        }
        else if (rhs0 != kInvalidNode)
        {
            for (uint32_t d = laneId; d < dim; d += kDistanceTeamSize)
            {
                const float delta0 = data[lhsOffset + d] - data[rhsOffset0 + d];
                distance0 += delta0 * delta0;
            }
        }
        else
        {
            for (uint32_t d = laneId; d < dim; d += kDistanceTeamSize)
            {
                const float lhsValue = data[lhsOffset + d];
                const float delta1 = lhsValue - data[rhsOffset1 + d];
                distance1 += delta1 * delta1;
            }
        }
        for (uint32_t offset = kDistanceTeamSize / 2; offset > 0; offset >>= 1)
        {
            distance0 += __shfl_xor(distance0, offset, kDistanceTeamSize);
            distance1 += __shfl_xor(distance1, offset, kDistanceTeamSize);
        }
        if (rhs0 != kInvalidNode && Better(distance0, rhs0, bestDistance, bestNode))
        {
            bestDistance = distance0;
            bestNode = rhs0;
        }
        if (rhs1 != kInvalidNode && Better(distance1, rhs1, bestDistance, bestNode))
        {
            bestDistance = distance1;
            bestNode = rhs1;
        }
    }
}

__simt_vf__ __aicore__ LAUNCH_BOUND(kThreadsPerBlock) inline void NndInitSimt(__gm__ const float *data,
                                                                              __gm__ uint32_t *graph,
                                                                              __gm__ float *distances, uint32_t nodeNum,
                                                                              uint32_t dim, uint32_t intermediateDegree,
                                                                              uint32_t blockId, uint32_t blockNum)
{
    uint32_t threadId = AscendC::Simt::GetThreadIdx<0>();
    uint32_t stride = blockNum * kThreadsPerBlock;
    for (uint32_t row = blockId * kThreadsPerBlock + threadId; row < nodeNum; row += stride)
    {
        uint64_t rowOffset = static_cast<uint64_t>(row) * intermediateDegree;
        for (uint32_t rank = 0; rank < intermediateDegree; ++rank)
        {
            graph[rowOffset + rank] = kInvalidNode;
            distances[rowOffset + rank] = kFloatMax;
        }

        for (uint32_t rank = 0; rank < intermediateDegree; ++rank)
        {
            uint32_t candidate = Mix32(row * 0x9e3779b9U + rank * 0x85ebca6bU + 0x27d4eb2dU) % nodeNum;
            while (candidate == row || RowContains(graph + rowOffset, rank, candidate))
            {
                candidate = (candidate + 1U) % nodeNum;
            }
            float distance = L2Distance(data, row, candidate, dim);
            uint32_t position = rank;
            while (position > 0 &&
                   Better(distance, candidate, distances[rowOffset + position - 1], graph[rowOffset + position - 1]))
            {
                graph[rowOffset + position] = graph[rowOffset + position - 1];
                distances[rowOffset + position] = distances[rowOffset + position - 1];
                --position;
            }
            graph[rowOffset + position] = candidate;
            distances[rowOffset + position] = distance;
        }
    }
}

__simt_vf__ __aicore__ LAUNCH_BOUND(kThreadsPerBlock) inline void NndSampleReverseSimt(
    __gm__ const uint32_t *graph, __gm__ uint32_t *forwardNew, __gm__ uint32_t *forwardOld, __gm__ uint32_t *reverseNew,
    __gm__ uint32_t *reverseOld, __gm__ uint32_t *sizes, uint32_t nodeNum, uint32_t intermediateDegree,
    uint32_t sampleDegree, uint32_t blockId, uint32_t blockNum)
{
    // The host must zero the complete sizes buffer before every launch. Forward-size fields are written by the owning
    // row, while reverse-size fields receive atomic additions from other rows and therefore cannot be cleared here.
    uint32_t threadId = AscendC::Simt::GetThreadIdx<0>();
    uint32_t stride = blockNum * kThreadsPerBlock;
    for (uint32_t row = blockId * kThreadsPerBlock + threadId; row < nodeNum; row += stride)
    {
        uint32_t newSize = 0;
        uint32_t oldSize = 0;
        uint32_t newSeen = 0;
        uint32_t oldSeen = 0;
        uint64_t graphOffset = static_cast<uint64_t>(row) * intermediateDegree;
        uint64_t sampleOffset = static_cast<uint64_t>(row) * sampleDegree;
        for (uint32_t rank = 0; rank < intermediateDegree; ++rank)
        {
            uint32_t flaggedNode = graph[graphOffset + rank];
            uint32_t node = RawNode(flaggedNode);
            if (node >= nodeNum)
            {
                continue;
            }
            if (IsNew(flaggedNode))
            {
                ++newSeen;
                if (newSize < sampleDegree)
                {
                    forwardNew[sampleOffset + newSize++] = node;
                }
                else
                {
                    uint32_t slot = Mix32(row ^ node ^ (rank * 0x9e3779b9U)) % newSeen;
                    if (slot < sampleDegree)
                    {
                        forwardNew[sampleOffset + slot] = node;
                    }
                }
            }
            else
            {
                ++oldSeen;
                if (oldSize < sampleDegree)
                {
                    forwardOld[sampleOffset + oldSize++] = node;
                }
                else
                {
                    uint32_t slot = Mix32(row ^ node ^ (rank * 0x85ebca6bU)) % oldSeen;
                    if (slot < sampleDegree)
                    {
                        forwardOld[sampleOffset + slot] = node;
                    }
                }
            }
        }
        sizes[static_cast<uint64_t>(row) * kSizeFieldCount + kForwardNewSizeIndex] = newSize;
        sizes[static_cast<uint64_t>(row) * kSizeFieldCount + kForwardOldSizeIndex] = oldSize;
        for (uint32_t i = 0; i < newSize; ++i)
        {
            uint32_t node = forwardNew[sampleOffset + i];
            uint32_t reverseSlot = AscendC::Simt::AtomicAdd(
                sizes + static_cast<uint64_t>(node) * kSizeFieldCount + kReverseNewSizeIndex, 1U);
            if (reverseSlot < sampleDegree)
            {
                reverseNew[static_cast<uint64_t>(node) * sampleDegree + reverseSlot] = row;
            }
        }
        for (uint32_t i = 0; i < oldSize; ++i)
        {
            uint32_t node = forwardOld[sampleOffset + i];
            uint32_t reverseSlot = AscendC::Simt::AtomicAdd(
                sizes + static_cast<uint64_t>(node) * kSizeFieldCount + kReverseOldSizeIndex, 1U);
            if (reverseSlot < sampleDegree)
            {
                reverseOld[static_cast<uint64_t>(node) * sampleDegree + reverseSlot] = row;
            }
        }
    }
}

__simt_vf__ __aicore__ LAUNCH_BOUND(kThreadsPerBlock) inline void NndLocalJoinSimt(
    __gm__ const float *data, __gm__ const uint32_t *forwardNew, __gm__ const uint32_t *forwardOld,
    __gm__ const uint32_t *reverseNew, __gm__ const uint32_t *reverseOld, __gm__ const uint32_t *sizes,
    __gm__ uint32_t *candidateIds, __gm__ float *candidateDistances, __gm__ uint32_t *candidateCounts, uint32_t nodeNum,
    uint32_t dim, uint32_t sampleDegree, uint32_t candidateDegree, uint32_t blockId, uint32_t blockNum)
{
    // The host must zero candidateCounts before every launch.
    uint32_t threadId = AscendC::Simt::GetThreadIdx<0>();
    uint32_t laneId = threadId % kDistanceTeamSize;
    uint32_t teamId = threadId / kDistanceTeamSize;
    uint32_t stride = blockNum * kTeamsPerBlock;
    for (uint32_t center = blockId * kTeamsPerBlock + teamId; center < nodeNum; center += stride)
    {
        uint64_t sizeOffset = static_cast<uint64_t>(center) * kSizeFieldCount;
        uint32_t forwardNewSize = sizes[sizeOffset + kForwardNewSizeIndex] < sampleDegree
                                      ? sizes[sizeOffset + kForwardNewSizeIndex]
                                      : sampleDegree;
        uint32_t forwardOldSize = sizes[sizeOffset + kForwardOldSizeIndex] < sampleDegree
                                      ? sizes[sizeOffset + kForwardOldSizeIndex]
                                      : sampleDegree;
        uint32_t reverseNewSize = sizes[sizeOffset + kReverseNewSizeIndex] < sampleDegree
                                      ? sizes[sizeOffset + kReverseNewSizeIndex]
                                      : sampleDegree;
        uint32_t reverseOldSize = sizes[sizeOffset + kReverseOldSizeIndex] < sampleDegree
                                      ? sizes[sizeOffset + kReverseOldSizeIndex]
                                      : sampleDegree;
        uint32_t newSize = forwardNewSize + reverseNewSize;
        uint32_t oldSize = forwardOldSize + reverseOldSize;

        for (uint32_t i = 0; i < newSize; ++i)
        {
            uint32_t lhs = CombinedNode(forwardNew, reverseNew, forwardNewSize, i, center, sampleDegree);
            if (lhs >= nodeNum ||
                !IsFirstCombinedOccurrence(forwardNew, reverseNew, forwardNewSize, i, center, sampleDegree))
            {
                continue;
            }

            // Reuse the complete lhs vector for dimensions up to 128. Medium dimensions batch four rhs
            // candidates; larger dimensions stream pairs without a private lhs tile.
            float lhsValues[kCachedValuesPerLane] = {0.0F};
            uint64_t lhsOffset = static_cast<uint64_t>(lhs) * dim;
            if (dim <= kCachedDistanceDim)
            {
                for (uint32_t d = laneId, slot = 0; d < dim; d += kDistanceTeamSize, ++slot)
                {
                    lhsValues[slot] = data[lhsOffset + d];
                }
            }

            uint32_t bestNewNode = kInvalidNode;
            float bestNewDistance = kFloatMax;
            if (dim <= kCachedDistanceDim)
            {
                for (uint32_t j = 0; j < newSize; ++j)
                {
                    uint32_t rhs = CombinedNode(forwardNew, reverseNew, forwardNewSize, j, center, sampleDegree);
                    if (rhs >= nodeNum || lhs == rhs ||
                        !IsFirstCombinedOccurrence(forwardNew, reverseNew, forwardNewSize, j, center, sampleDegree))
                    {
                        continue;
                    }
                    float distance = 0.0F;
                    uint64_t rhsOffset = static_cast<uint64_t>(rhs) * dim;
                    for (uint32_t d = laneId, slot = 0; d < dim; d += kDistanceTeamSize, ++slot)
                    {
                        float delta = lhsValues[slot] - data[rhsOffset + d];
                        distance += delta * delta;
                    }
                    for (uint32_t offset = kDistanceTeamSize / 2; offset > 0; offset >>= 1)
                    {
                        distance += __shfl_xor(distance, offset, kDistanceTeamSize);
                    }
                    if (Better(distance, rhs, bestNewDistance, bestNewNode))
                    {
                        bestNewDistance = distance;
                        bestNewNode = rhs;
                    }
                }
            }
            else
            {
                if (dim <= kBatchedDistanceMaxDim)
                {
                    FindBestHighDimCandidate(data, forwardNew, reverseNew, forwardNewSize, newSize, center,
                                             sampleDegree, lhs, nodeNum, dim, laneId, bestNewNode, bestNewDistance);
                }
                else
                {
                    FindBestHighDimCandidatePaired(data, forwardNew, reverseNew, forwardNewSize, newSize, center,
                                                   sampleDegree, lhs, nodeNum, dim, laneId, bestNewNode,
                                                   bestNewDistance);
                }
            }
            if (laneId == 0 && bestNewNode < nodeNum)
            {
                AppendCandidate(lhs, bestNewNode, bestNewDistance, candidateIds, candidateDistances, candidateCounts,
                                candidateDegree);
            }

            uint32_t bestOldNode = kInvalidNode;
            float bestOldDistance = kFloatMax;
            if (dim <= kCachedDistanceDim)
            {
                for (uint32_t j = 0; j < oldSize; ++j)
                {
                    uint32_t rhs = CombinedNode(forwardOld, reverseOld, forwardOldSize, j, center, sampleDegree);
                    if (rhs >= nodeNum || lhs == rhs ||
                        !IsFirstCombinedOccurrence(forwardOld, reverseOld, forwardOldSize, j, center, sampleDegree))
                    {
                        continue;
                    }
                    float distance = 0.0F;
                    uint64_t rhsOffset = static_cast<uint64_t>(rhs) * dim;
                    for (uint32_t d = laneId, slot = 0; d < dim; d += kDistanceTeamSize, ++slot)
                    {
                        float delta = lhsValues[slot] - data[rhsOffset + d];
                        distance += delta * delta;
                    }
                    for (uint32_t offset = kDistanceTeamSize / 2; offset > 0; offset >>= 1)
                    {
                        distance += __shfl_xor(distance, offset, kDistanceTeamSize);
                    }
                    if (Better(distance, rhs, bestOldDistance, bestOldNode))
                    {
                        bestOldDistance = distance;
                        bestOldNode = rhs;
                    }
                }
            }
            else
            {
                if (dim <= kBatchedDistanceMaxDim)
                {
                    FindBestHighDimCandidate(data, forwardOld, reverseOld, forwardOldSize, oldSize, center,
                                             sampleDegree, lhs, nodeNum, dim, laneId, bestOldNode, bestOldDistance);
                }
                else
                {
                    FindBestHighDimCandidatePaired(data, forwardOld, reverseOld, forwardOldSize, oldSize, center,
                                                   sampleDegree, lhs, nodeNum, dim, laneId, bestOldNode,
                                                   bestOldDistance);
                }
            }
            if (laneId == 0 && bestOldNode < nodeNum)
            {
                AppendCandidate(lhs, bestOldNode, bestOldDistance, candidateIds, candidateDistances, candidateCounts,
                                candidateDegree);
            }
        }
    }
}

__simt_callee__ __aicore__ inline bool IsSampledNew(uint32_t node, __gm__ const uint32_t *forwardNew,
                                                    uint32_t sampleSize)
{
    for (uint32_t i = 0; i < sampleSize; ++i)
    {
        if (forwardNew[i] == node)
        {
            return true;
        }
    }
    return false;
}

__simt_vf__ __aicore__ LAUNCH_BOUND(kThreadsPerBlock) inline void NndUpdateSimt(
    __gm__ const uint32_t *graph, __gm__ const float *distances, __gm__ const uint32_t *forwardNew,
    __gm__ const uint32_t *sizes, __gm__ const uint32_t *candidateIds, __gm__ const float *candidateDistances,
    __gm__ const uint32_t *candidateCounts, __gm__ uint32_t *nextGraph, __gm__ float *nextDistances,
    __gm__ uint32_t *updateCount, uint32_t nodeNum, uint32_t intermediateDegree, uint32_t sampleDegree,
    uint32_t candidateDegree, uint32_t blockId, uint32_t blockNum)
{
    // The host must zero updateCount before every launch.
    uint32_t threadId = AscendC::Simt::GetThreadIdx<0>();
    uint32_t stride = blockNum * kThreadsPerBlock;
    for (uint32_t row = blockId * kThreadsPerBlock + threadId; row < nodeNum; row += stride)
    {
        uint64_t graphOffset = static_cast<uint64_t>(row) * intermediateDegree;
        uint64_t sampleOffset = static_cast<uint64_t>(row) * sampleDegree;
        uint32_t sampleSize = sizes[static_cast<uint64_t>(row) * kSizeFieldCount + kForwardNewSizeIndex];
        if (sampleSize > sampleDegree)
        {
            sampleSize = sampleDegree;
        }
        for (uint32_t rank = 0; rank < intermediateDegree; ++rank)
        {
            uint32_t flaggedNode = graph[graphOffset + rank];
            uint32_t rawNode = RawNode(flaggedNode);
            if (IsNew(flaggedNode) && IsSampledNew(rawNode, forwardNew + sampleOffset, sampleSize))
            {
                flaggedNode = rawNode | kOldMask;
            }
            nextGraph[graphOffset + rank] = flaggedNode;
            nextDistances[graphOffset + rank] = distances[graphOffset + rank];
        }

        uint32_t count = candidateCounts[row];
        if (count > candidateDegree)
        {
            count = candidateDegree;
        }
        uint32_t inserted = 0;
        uint64_t candidateOffset = static_cast<uint64_t>(row) * candidateDegree;
        for (uint32_t i = 0; i < count; ++i)
        {
            uint32_t node = candidateIds[candidateOffset + i];
            float distance = candidateDistances[candidateOffset + i];
            if (node >= nodeNum || node == row ||
                !Better(distance, node, nextDistances[graphOffset + intermediateDegree - 1],
                        nextGraph[graphOffset + intermediateDegree - 1]))
            {
                continue;
            }
            uint32_t position = intermediateDegree;
            bool duplicate = false;
            for (uint32_t rank = 0; rank < intermediateDegree; ++rank)
            {
                uint32_t currentNode = nextGraph[graphOffset + rank];
                if (RawNode(currentNode) == node)
                {
                    duplicate = true;
                    break;
                }
                if (position == intermediateDegree &&
                    Better(distance, node, nextDistances[graphOffset + rank], currentNode))
                {
                    position = rank;
                }
            }
            if (duplicate || position == intermediateDegree)
            {
                continue;
            }
            for (uint32_t rank = intermediateDegree - 1; rank > position; --rank)
            {
                nextGraph[graphOffset + rank] = nextGraph[graphOffset + rank - 1];
                nextDistances[graphOffset + rank] = nextDistances[graphOffset + rank - 1];
            }
            nextGraph[graphOffset + position] = node;
            nextDistances[graphOffset + position] = distance;
            ++inserted;
        }
        if (inserted != 0)
        {
            AscendC::Simt::AtomicAdd(updateCount, inserted);
        }
    }
}

__simt_vf__ __aicore__ LAUNCH_BOUND(kThreadsPerBlock) inline void CagraPruneReverseSimt(
    __gm__ const uint32_t *intermediateGraph, __gm__ uint32_t *prunedGraph, __gm__ uint32_t *reverseGraph,
    __gm__ uint32_t *reverseCounts, uint32_t nodeNum, uint32_t intermediateDegree, uint32_t outputDegree,
    uint32_t blockId, uint32_t blockNum)
{
    // The host must zero reverseCounts before this launch.
    uint32_t threadId = AscendC::Simt::GetThreadIdx<0>();
    uint32_t laneId = threadId % kPruneTeamSize;
    uint32_t teamId = threadId / kPruneTeamSize;
    uint32_t stride = blockNum * kPruneTeamsPerBlock;
    // One 32-thread team cooperates on a row. Keeping 16 independent teams in every
    // 512-thread block preserves row-level concurrency on A5 while parallelizing detour scans.
    for (uint32_t row = blockId * kPruneTeamsPerBlock + teamId; row < nodeNum; row += stride)
    {
        uint64_t outputOffset = static_cast<uint64_t>(row) * outputDegree;
        uint64_t intermediateOffset = static_cast<uint64_t>(row) * intermediateDegree;
        // A5 CAGRA supports M up to 128. Each lane owns kPruneSlotsPerLane ranks and keeps their
        // candidate/detour values in registers so output selection does not reread them from GM.
        uint32_t laneCandidates[kPruneSlotsPerLane];
        uint32_t laneDetours[kPruneSlotsPerLane];
        for (uint32_t slot = 0; slot < kPruneSlotsPerLane; ++slot)
        {
            laneCandidates[slot] = kInvalidNode;
            laneDetours[slot] = 0;
            uint32_t rank = laneId + slot * kPruneTeamSize;
            if (rank < intermediateDegree)
            {
                laneCandidates[slot] = RawNode(intermediateGraph[intermediateOffset + rank]);
                if (laneCandidates[slot] >= nodeNum)
                {
                    laneDetours[slot] = kInvalidNode;
                }
            }
        }

        // Load every D-neighbor row once with coalesced lanes and reuse it for all later
        // A->B candidates. This reduces expensive GM reads per row from O(M^3) to O(M^2).
        for (uint32_t viaRank = 0; viaRank + 1 < intermediateDegree; ++viaRank)
        {
            uint32_t via = RawNode(intermediateGraph[intermediateOffset + viaRank]);
            if (via >= nodeNum)
            {
                continue;
            }
            uint64_t viaOffset = static_cast<uint64_t>(via) * intermediateDegree;
            uint32_t viaNeighbors[kPruneSlotsPerLane];
            for (uint32_t slot = 0; slot < kPruneSlotsPerLane; ++slot)
            {
                viaNeighbors[slot] = kInvalidNode;
                uint32_t rank = laneId + slot * kPruneTeamSize;
                if (rank < intermediateDegree)
                {
                    viaNeighbors[slot] = RawNode(intermediateGraph[viaOffset + rank]);
                }
            }
            for (uint32_t candidateRank = viaRank + 1; candidateRank < intermediateDegree; ++candidateRank)
            {
                uint32_t candidate = RawNode(intermediateGraph[intermediateOffset + candidateRank]);
                if (candidate >= nodeNum)
                {
                    continue;
                }
                uint32_t found = 0;
                for (uint32_t slot = 0; slot < kPruneSlotsPerLane; ++slot)
                {
                    found |= (viaNeighbors[slot] == candidate);
                }
                uint32_t matches = __popc(__ballot(found != 0));
                if (candidateRank % kPruneTeamSize == laneId)
                {
                    uint32_t slot = candidateRank / kPruneTeamSize;
                    laneDetours[slot] += matches;
                }
            }
        }
        uint32_t selectedMask0 = 0;
        uint32_t selectedMask1 = 0;
        uint32_t selectedMask2 = 0;
        uint32_t selectedMask3 = 0;
        for (uint32_t outRank = 0; outRank < outputDegree; ++outRank)
        {
            uint32_t bestRank = kInvalidNode;
            uint32_t bestDetours = kInvalidNode;
            for (uint32_t slot = 0; slot < kPruneSlotsPerLane; ++slot)
            {
                uint32_t rank = laneId + slot * kPruneTeamSize;
                if (rank >= intermediateDegree)
                {
                    continue;
                }
                uint32_t mask = 1U << (rank % kPruneTeamSize);
                bool selected = rank < kPruneTeamSize       ? (selectedMask0 & mask) != 0
                                : rank < 2 * kPruneTeamSize ? (selectedMask1 & mask) != 0
                                : rank < 3 * kPruneTeamSize ? (selectedMask2 & mask) != 0
                                                            : (selectedMask3 & mask) != 0;
                uint32_t candidate = laneCandidates[slot];
                uint32_t detours = laneDetours[slot];
                if (selected || candidate >= nodeNum)
                {
                    continue;
                }
                if (detours < bestDetours || (detours == bestDetours && rank < bestRank))
                {
                    bestDetours = detours;
                    bestRank = rank;
                }
            }

            for (uint32_t offset = kPruneTeamSize / 2; offset > 0; offset >>= 1)
            {
                uint32_t otherDetours = __shfl_xor(bestDetours, offset, kPruneTeamSize);
                uint32_t otherRank = __shfl_xor(bestRank, offset, kPruneTeamSize);
                if (otherDetours < bestDetours || (otherDetours == bestDetours && otherRank < bestRank))
                {
                    bestDetours = otherDetours;
                    bestRank = otherRank;
                }
            }

            uint32_t selectedBit = 1U << (bestRank % kPruneTeamSize);
            if (bestRank < kPruneTeamSize)
            {
                selectedMask0 |= selectedBit;
            }
            else if (bestRank < 2 * kPruneTeamSize)
            {
                selectedMask1 |= selectedBit;
            }
            else if (bestRank < 3 * kPruneTeamSize)
            {
                selectedMask2 |= selectedBit;
            }
            else
            {
                selectedMask3 |= selectedBit;
            }

            if (laneId == 0)
            {
                uint32_t bestNode = bestRank < intermediateDegree
                                        ? RawNode(intermediateGraph[intermediateOffset + bestRank])
                                        : kInvalidNode;
                if (bestNode >= nodeNum)
                {
                    bestNode = (row + outRank + 1U) % nodeNum;
                    while (bestNode == row || RowContains(prunedGraph + outputOffset, outRank, bestNode))
                    {
                        bestNode = (bestNode + 1U) % nodeNum;
                    }
                }
                prunedGraph[outputOffset + outRank] = bestNode;
                uint32_t reverseSlot = AscendC::Simt::AtomicAdd(reverseCounts + bestNode, 1U);
                if (reverseSlot < outputDegree)
                {
                    reverseGraph[static_cast<uint64_t>(bestNode) * outputDegree + reverseSlot] = row;
                }
            }
        }
    }
}

__simt_vf__ __aicore__ LAUNCH_BOUND(kThreadsPerBlock) inline void CagraMergeSimt(
    __gm__ const uint32_t *prunedGraph, __gm__ const uint32_t *reverseGraph, __gm__ const uint32_t *reverseCounts,
    __gm__ uint32_t *graph, uint32_t nodeNum, uint32_t outputDegree, uint32_t blockId, uint32_t blockNum)
{
    uint32_t threadId = AscendC::Simt::GetThreadIdx<0>();
    uint32_t stride = blockNum * kThreadsPerBlock;
    for (uint32_t row = blockId * kThreadsPerBlock + threadId; row < nodeNum; row += stride)
    {
        uint64_t rowOffset = static_cast<uint64_t>(row) * outputDegree;
        for (uint32_t rank = 0; rank < outputDegree; ++rank)
        {
            graph[rowOffset + rank] = prunedGraph[rowOffset + rank];
        }
        uint32_t protectedDegree = outputDegree / 2;
        uint32_t reverseSize = reverseCounts[row] < outputDegree ? reverseCounts[row] : outputDegree;
        uint32_t writePosition = protectedDegree;
        for (uint32_t i = 0; i < reverseSize && writePosition < outputDegree; ++i)
        {
            uint32_t candidate = reverseGraph[rowOffset + i];
            bool duplicate = false;
            for (uint32_t rank = 0; rank < outputDegree; ++rank)
            {
                duplicate |= (graph[rowOffset + rank] == candidate);
            }
            if (candidate < nodeNum && candidate != row && !duplicate)
            {
                graph[rowOffset + writePosition] = candidate;
                ++writePosition;
            }
        }
    }
}
}  // namespace

#ifdef CAGRA_NND_INIT_KERNEL_SOURCE
extern "C" __global__ __aicore__ void cagra_nnd_init(GM_ADDR data, GM_ADDR graph, GM_ADDR distances, GM_ADDR workspace,
                                                     GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    AscendC::Simt::VF_CALL<NndInitSimt>(
        AscendC::Simt::Dim3{kThreadsPerBlock, 1, 1}, reinterpret_cast<__gm__ const float *>(data),
        reinterpret_cast<__gm__ uint32_t *>(graph), reinterpret_cast<__gm__ float *>(distances), tilingData.nodeNum,
        tilingData.dim, tilingData.intermediateDegree, static_cast<uint32_t>(GetBlockIdx()), tilingData.blockNum);
}
#endif

#ifdef CAGRA_NND_SAMPLE_REVERSE_KERNEL_SOURCE
extern "C" __global__ __aicore__ void cagra_nnd_sample_reverse(GM_ADDR graph, GM_ADDR forwardNew, GM_ADDR forwardOld,
                                                               GM_ADDR reverseNew, GM_ADDR reverseOld, GM_ADDR sizes,
                                                               GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    AscendC::Simt::VF_CALL<NndSampleReverseSimt>(
        AscendC::Simt::Dim3{kThreadsPerBlock, 1, 1}, reinterpret_cast<__gm__ const uint32_t *>(graph),
        reinterpret_cast<__gm__ uint32_t *>(forwardNew), reinterpret_cast<__gm__ uint32_t *>(forwardOld),
        reinterpret_cast<__gm__ uint32_t *>(reverseNew), reinterpret_cast<__gm__ uint32_t *>(reverseOld),
        reinterpret_cast<__gm__ uint32_t *>(sizes), tilingData.nodeNum, tilingData.intermediateDegree,
        tilingData.sampleDegree, static_cast<uint32_t>(GetBlockIdx()), tilingData.blockNum);
}
#endif

#ifdef CAGRA_NND_LOCAL_JOIN_KERNEL_SOURCE
extern "C" __global__ __aicore__ void cagra_nnd_local_join(GM_ADDR data, GM_ADDR forwardNew, GM_ADDR forwardOld,
                                                           GM_ADDR reverseNew, GM_ADDR reverseOld, GM_ADDR sizes,
                                                           GM_ADDR candidateIds, GM_ADDR candidateDistances,
                                                           GM_ADDR candidateCounts, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    AscendC::Simt::VF_CALL<NndLocalJoinSimt>(
        AscendC::Simt::Dim3{kThreadsPerBlock, 1, 1}, reinterpret_cast<__gm__ const float *>(data),
        reinterpret_cast<__gm__ const uint32_t *>(forwardNew), reinterpret_cast<__gm__ const uint32_t *>(forwardOld),
        reinterpret_cast<__gm__ const uint32_t *>(reverseNew), reinterpret_cast<__gm__ const uint32_t *>(reverseOld),
        reinterpret_cast<__gm__ const uint32_t *>(sizes), reinterpret_cast<__gm__ uint32_t *>(candidateIds),
        reinterpret_cast<__gm__ float *>(candidateDistances), reinterpret_cast<__gm__ uint32_t *>(candidateCounts),
        tilingData.nodeNum, tilingData.dim, tilingData.sampleDegree, tilingData.candidateDegree,
        static_cast<uint32_t>(GetBlockIdx()), tilingData.blockNum);
}
#endif

#ifdef CAGRA_NND_UPDATE_KERNEL_SOURCE
extern "C" __global__ __aicore__ void cagra_nnd_update(GM_ADDR graph, GM_ADDR distances, GM_ADDR forwardNew,
                                                       GM_ADDR sizes, GM_ADDR candidateIds, GM_ADDR candidateDistances,
                                                       GM_ADDR candidateCounts, GM_ADDR nextGraph,
                                                       GM_ADDR nextDistances, GM_ADDR updateCount, GM_ADDR workspace,
                                                       GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    AscendC::Simt::VF_CALL<NndUpdateSimt>(
        AscendC::Simt::Dim3{kThreadsPerBlock, 1, 1}, reinterpret_cast<__gm__ const uint32_t *>(graph),
        reinterpret_cast<__gm__ const float *>(distances), reinterpret_cast<__gm__ const uint32_t *>(forwardNew),
        reinterpret_cast<__gm__ const uint32_t *>(sizes), reinterpret_cast<__gm__ const uint32_t *>(candidateIds),
        reinterpret_cast<__gm__ const float *>(candidateDistances),
        reinterpret_cast<__gm__ const uint32_t *>(candidateCounts), reinterpret_cast<__gm__ uint32_t *>(nextGraph),
        reinterpret_cast<__gm__ float *>(nextDistances), reinterpret_cast<__gm__ uint32_t *>(updateCount),
        tilingData.nodeNum, tilingData.intermediateDegree, tilingData.sampleDegree, tilingData.candidateDegree,
        static_cast<uint32_t>(GetBlockIdx()), tilingData.blockNum);
}
#endif

#ifdef CAGRA_PRUNE_REVERSE_KERNEL_SOURCE
extern "C" __global__ __aicore__ void cagra_prune_reverse(GM_ADDR intermediateGraph, GM_ADDR prunedGraph,
                                                          GM_ADDR reverseGraph, GM_ADDR reverseCounts,
                                                          GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    AscendC::Simt::VF_CALL<CagraPruneReverseSimt>(
        AscendC::Simt::Dim3{kThreadsPerBlock, 1, 1}, reinterpret_cast<__gm__ const uint32_t *>(intermediateGraph),
        reinterpret_cast<__gm__ uint32_t *>(prunedGraph), reinterpret_cast<__gm__ uint32_t *>(reverseGraph),
        reinterpret_cast<__gm__ uint32_t *>(reverseCounts), tilingData.nodeNum, tilingData.intermediateDegree,
        tilingData.outputDegree, static_cast<uint32_t>(GetBlockIdx()), tilingData.blockNum);
}
#endif

#ifdef CAGRA_MERGE_KERNEL_SOURCE
extern "C" __global__ __aicore__ void cagra_merge(GM_ADDR prunedGraph, GM_ADDR reverseGraph, GM_ADDR reverseCounts,
                                                  GM_ADDR graph, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    AscendC::Simt::VF_CALL<CagraMergeSimt>(
        AscendC::Simt::Dim3{kThreadsPerBlock, 1, 1}, reinterpret_cast<__gm__ const uint32_t *>(prunedGraph),
        reinterpret_cast<__gm__ const uint32_t *>(reverseGraph),
        reinterpret_cast<__gm__ const uint32_t *>(reverseCounts), reinterpret_cast<__gm__ uint32_t *>(graph),
        tilingData.nodeNum, tilingData.outputDegree, static_cast<uint32_t>(GetBlockIdx()), tilingData.blockNum);
}
#endif

#endif
