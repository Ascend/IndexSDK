/* Copyright (c) 2026 Huawei Technologies Co.,Ltd. Licensed under Mulan PSL v2. */
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "simt_api/asc_simt.h"

namespace
{
constexpr uint32_t kThreads = 32;
constexpr int32_t kMaxId = 2147483647;
constexpr float kMaxFloat = 3.402823466e+38F;

__simt_callee__ __aicore__ inline bool Better(float lhsScore, int32_t lhsId, float rhsScore, int32_t rhsId)
{
    return lhsScore > rhsScore || (lhsScore == rhsScore && lhsId < rhsId);
}

__simt_vf__ __aicore__ LAUNCH_BOUND(kThreads) inline void MergeSectionsSimt(
    __gm__ const float* leftScores, __gm__ const int32_t* leftIds, __gm__ const float* rightScores,
    __gm__ const int32_t* rightIds, __gm__ const int32_t* rightRange, __gm__ float* topScores, __gm__ int32_t* topIds,
    uint32_t queryCount, uint32_t k, uint32_t block, uint32_t blockCount)
{
    const int32_t offset = rightRange[0], count = rightRange[1];
    const uint32_t rightLimit = count > static_cast<int32_t>(k) ? k : static_cast<uint32_t>(count);
    const uint32_t lane = threadIdx.x;
    for (uint32_t q = block * kThreads + lane; q < queryCount; q += blockCount * kThreads)
    {
        const uint32_t base = q * k;
        bool valid = offset >= static_cast<int32_t>(k) && count > 0 && count <= kMaxId - offset;
        for (uint32_t rank = 0; valid && rank < k; ++rank)
        {
            const uint32_t i = base + rank;
            const float leftScore = leftScores[i], rightScore = rightScores[i];
            const int32_t leftId = leftIds[i], rightId = rightIds[i];
            valid = leftScore >= 0.0F && leftScore <= kMaxFloat && rightScore >= 0.0F && rightScore <= kMaxFloat &&
                    leftId >= 0 && leftId < offset && rightId >= 0 &&
                    (rank < rightLimit ? rightId < count : rightId >= count && rightScore == 0.0F);
            if (valid && rank > 0)
                valid = Better(leftScores[i - 1], leftIds[i - 1], leftScore, leftId) &&
                        Better(rightScores[i - 1], rightIds[i - 1], rightScore, rightId);
        }

        uint32_t left = 0, right = 0;
        for (uint32_t rank = 0; rank < k; ++rank)
        {
            float score = -1.0F;
            int32_t id = -1;
            if (valid)
            {
                const bool takeRight = right < rightLimit &&
                                       (left == k || Better(rightScores[base + right], rightIds[base + right] + offset,
                                                            leftScores[base + left], leftIds[base + left]));
                if (takeRight)
                {
                    score = rightScores[base + right];
                    id = rightIds[base + right++] + offset;
                }
                else
                {
                    score = leftScores[base + left];
                    id = leftIds[base + left++];
                }
            }
            topScores[base + rank] = score;
            topIds[base + rank] = id;
        }
    }
}
}  // namespace

extern "C" __global__ __aicore__ void bm25_merge_sections(GM_ADDR left_scores, GM_ADDR left_ids, GM_ADDR right_scores,
                                                          GM_ADDR right_ids, GM_ADDR right_range, GM_ADDR top_scores,
                                                          GM_ADDR top_ids, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(data, tiling);
    asc_vf_call<MergeSectionsSimt>(
        dim3{kThreads, 1, 1}, reinterpret_cast<__gm__ const float*>(left_scores),
        reinterpret_cast<__gm__ const int32_t*>(left_ids), reinterpret_cast<__gm__ const float*>(right_scores),
        reinterpret_cast<__gm__ const int32_t*>(right_ids), reinterpret_cast<__gm__ const int32_t*>(right_range),
        reinterpret_cast<__gm__ float*>(top_scores), reinterpret_cast<__gm__ int32_t*>(top_ids), data.queryCount,
        data.k, static_cast<uint32_t>(AscendC::GetBlockIdx()), data.blockNum);
}
