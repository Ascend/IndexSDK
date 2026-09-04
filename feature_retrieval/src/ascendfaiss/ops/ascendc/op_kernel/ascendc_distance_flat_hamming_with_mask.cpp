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

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/matmul.h"
#include "op_kernel_common.h"

using namespace AscendC;
using namespace matmul;

namespace
{
constexpr uint32_t HAMMING_CUBE_ALIGN = 4;
constexpr uint32_t BUFFER_ALIGN_BYTES = 32;
constexpr uint32_t BINARY_BYTE_BITS = 8;
constexpr uint32_t BITS_PER_DIM_BLOCK = 32;
constexpr uint32_t MAX_QUERY_TILE = 128;
constexpr uint32_t MAX_EXPANDED_ELEMENTS = 32 * 1024;
constexpr uint32_t MAX_BASE_GROUP_BATCH = 4;
constexpr uint32_t MAX_BURSTS_PER_TILE = 32;
constexpr uint32_t BURSTS_PER_COPY_ALIGN = BUFFER_ALIGN_BYTES / (2 * sizeof(uint16_t));
constexpr uint32_t MAX_LINE_BYTES = MAX_BURSTS_PER_TILE * 2 * sizeof(uint16_t);
}  // namespace

namespace IndexOps
{
class AscendcDistanceFlatHammingWithMask
{
   public:
    __aicore__ inline explicit AscendcDistanceFlatHammingWithMask(
        const AscendcDistanceFlatHammingWithMaskTilingData &tilingData)
        : queryNum(tilingData.queryNum),
          codeSize(tilingData.codeSize),
          dim(tilingData.dim),
          blockSize(tilingData.blockSize),
          zRegionHeight(tilingData.zRegionHeight),
          burstLen(tilingData.burstLen),
          codeTile(tilingData.codeTile),
          maskRows(tilingData.maskRows),
          cubeTilingIp(tilingData.cubeTilingIp),
          coreIdx(static_cast<uint32_t>(GetBlockIdx())),
          coreNum(static_cast<uint32_t>(GetBlockNum()))
    {
    }

    __aicore__ inline void Init(GM_ADDR query, GM_ADDR base, GM_ADDR actualSize, GM_ADDR mask, GM_ADDR baseMask,
                                GM_ADDR dist, GM_ADDR maxDist, GM_ADDR flag, GM_ADDR workspace)
    {
        // TopK consumes a single host-managed flag. The host publishes it with a same-stream D2D copy after this
        // kernel, so the per-core operator flag output is intentionally unused.
        (void)flag;
        queryGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t *>(query));
        baseGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t *>(base));
        actualSizeGlobal = reinterpret_cast<__gm__ uint32_t *>(actualSize);
        maskTensor.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t *>(mask));
        baseMaskTensor.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t *>(baseMask));
        distGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(dist));
        maxDistRawTensor.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t *>(maxDist));

        actualNum = actualSizeGlobal[0];
        skipMask = actualSizeGlobal[1] != 0;
        useBaseMask = actualSizeGlobal[2] != 0;
        maskLen = blockSize / Utils::MASK_BIT_NUM;
        dimBlockNum = codeSize / HAMMING_CUBE_ALIGN;
        queryTile = dim == 0 ? 0 : MAX_EXPANDED_ELEMENTS / dim;
        if (queryTile > MAX_QUERY_TILE)
        {
            queryTile = MAX_QUERY_TILE;
        }
        if (queryTile > queryNum)
        {
            queryTile = queryNum;
        }
        queryExpandTile = queryTile;
        baseGroupBatch = (zRegionHeight == 0 || dim == 0) ? 0 : MAX_EXPANDED_ELEMENTS / (zRegionHeight * dim);
        if (baseGroupBatch > MAX_BASE_GROUP_BATCH)
        {
            baseGroupBatch = MAX_BASE_GROUP_BATCH;
        }
        workspaceOk = InitWorkspace(workspace);
        ComputeCoreProcInfo();
        InitBuffers();
    }

    __aicore__ inline void Process()
    {
        REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), matmulObjIp);
        if (!workspaceOk || queryNum == 0 || codeSize == 0 || dim == 0 || codeTile == 0 || codeProcNum == 0 ||
            queryTile == 0 || baseGroupBatch == 0)
        {
            return;
        }

        matmulObjIp.Init(&cubeTilingIp);
        ExpandAllQueries();
        uint32_t queryLoopNum = (queryNum + queryTile - 1) / queryTile;
        for (uint32_t codeOffset = codeOffsetBegin; codeOffset < codeOffsetEnd; codeOffset += codeTile)
        {
            uint32_t codeProcTile = codeOffsetEnd - codeOffset;
            if (codeProcTile > codeTile)
            {
                codeProcTile = codeTile;
            }
            ExpandBaseTile(codeOffset, codeProcTile);
            matmulObjIp.SetTensorB(baseExpandedGlobal, true);
            for (uint32_t queryLoopIdx = 0; queryLoopIdx < queryLoopNum; ++queryLoopIdx)
            {
                uint32_t queryOffset = queryLoopIdx * queryTile;
                uint32_t queryProcNum = queryNum - queryOffset;
                if (queryProcNum > queryTile)
                {
                    queryProcNum = queryTile;
                }
                matmulObjIp.SetTensorA(queryExpandedGlobal[static_cast<uint64_t>(queryOffset) * dim]);
                matmulObjIp.SetTail(queryProcNum, codeProcTile, dim);
                matmulObjIp.IterateAll<false>(innerProductGlobal, 0, false, true);
                matmulObjIp.WaitIterateAll();
                WriteDistanceAndMax(queryOffset, queryProcNum, codeOffset, codeProcTile);
            }
        }
        matmulObjIp.End();
    }

   private:
    using MatmulTypeQuery = MatmulType<TPosition::GM, CubeFormat::ND, int8_t>;
    using MatmulTypeBase = MatmulType<TPosition::GM, CubeFormat::ND, int8_t, true>;
    using MatmulTypeScore = MatmulType<TPosition::GM, CubeFormat::ND, int32_t>;
    Matmul<MatmulTypeQuery, MatmulTypeBase, MatmulTypeScore> matmulObjIp;

    __aicore__ inline uint32_t AlignBufferBytes(uint32_t bytes) const
    {
        return ((bytes + BUFFER_ALIGN_BYTES - 1) / BUFFER_ALIGN_BYTES) * BUFFER_ALIGN_BYTES;
    }

    __aicore__ inline bool InitWorkspace(GM_ADDR workspace)
    {
        GM_ADDR userWorkspace = GetUserWorkspace(workspace);
        if (userWorkspace == nullptr)
        {
            return false;
        }
        uint64_t queryBytes = static_cast<uint64_t>(queryNum) * dim * sizeof(int8_t);
        uint64_t baseBytes = static_cast<uint64_t>(codeTile) * dim * sizeof(int8_t);
        uint64_t scoreBytes = static_cast<uint64_t>(queryTile) * codeTile * sizeof(int32_t);
        uint64_t perCoreBytes = queryBytes + baseBytes + scoreBytes;
        GM_ADDR coreWorkspace = userWorkspace + static_cast<uint64_t>(coreIdx) * perCoreBytes;
        queryExpandedGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t *>(coreWorkspace));
        baseExpandedGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t *>(coreWorkspace + queryBytes));
        innerProductGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(coreWorkspace + queryBytes + baseBytes));
        return true;
    }

    __aicore__ inline void ComputeCoreProcInfo()
    {
        uint32_t codeUnit = burstLen * BURSTS_PER_COPY_ALIGN;
        uint32_t validCodeNum = actualNum < blockSize ? actualNum : blockSize;
        uint32_t codeUnitNum = (validCodeNum + codeUnit - 1) / codeUnit;
        uint32_t usedCoreNum = codeUnitNum < coreNum ? codeUnitNum : coreNum;
        if (usedCoreNum == 0 || coreIdx >= usedCoreNum)
        {
            codeOffsetBegin = 0;
            codeOffsetEnd = 0;
            codeProcNum = 0;
            return;
        }
        uint32_t unitPerCore = codeUnitNum / usedCoreNum;
        uint32_t unitTail = codeUnitNum % usedCoreNum;
        uint32_t beginUnit = 0;
        uint32_t unitCount = unitPerCore;
        if (coreIdx < unitTail)
        {
            beginUnit = coreIdx * (unitPerCore + 1);
            unitCount = unitPerCore + 1;
        }
        else
        {
            beginUnit = unitTail * (unitPerCore + 1) + (coreIdx - unitTail) * unitPerCore;
        }
        codeOffsetBegin = beginUnit * codeUnit;
        codeProcNum = unitCount * codeUnit;
        codeOffsetEnd = codeOffsetBegin + codeProcNum;
    }

    __aicore__ inline void InitBuffers()
    {
        uint32_t queryPackedBytes = queryExpandTile * codeSize * sizeof(uint8_t);
        uint32_t queryElemCount = queryExpandTile * dim;
        uint32_t baseBatchPackedBytes = baseGroupBatch * zRegionHeight * codeSize * sizeof(uint8_t);
        uint32_t baseBatchElemCount = baseGroupBatch * zRegionHeight * dim;
        uint32_t sharedPackedBytes = queryPackedBytes > baseBatchPackedBytes ? queryPackedBytes : baseBatchPackedBytes;
        uint32_t sharedElemCount = queryElemCount > baseBatchElemCount ? queryElemCount : baseBatchElemCount;
        uint32_t scoreElemCount = codeTile;
        pipe.InitBuffer(queryPackedBuf, AlignBufferBytes(sharedPackedBytes));
        pipe.InitBuffer(maskBuf, AlignBufferBytes(codeTile / BINARY_BYTE_BITS));
        pipe.InitBuffer(baseMaskBuf, AlignBufferBytes(codeTile / BINARY_BYTE_BITS));
        pipe.InitBuffer(expandedInt8Buf, AlignBufferBytes(sharedElemCount * sizeof(int8_t)));
        pipe.InitBuffer(expandedHalfBuf, AlignBufferBytes(sharedElemCount * sizeof(half)));
        pipe.InitBuffer(scoreBuf, AlignBufferBytes(scoreElemCount * sizeof(int32_t)));
        pipe.InitBuffer(scoreHalfBuf, AlignBufferBytes(scoreElemCount * sizeof(half)));
        pipe.InitBuffer(maxLineBuf, MAX_LINE_BYTES);
    }

    __aicore__ inline void ExpandPackedToInt8(LocalTensor<uint8_t> &packedLocal, LocalTensor<int8_t> &int8Local,
                                              LocalTensor<half> &halfLocal, uint32_t packedCount)
    {
        uint32_t elemCount = packedCount * BINARY_BYTE_BITS;
        Duplicate(halfLocal, static_cast<half>(1.0), elemCount);
        pipe_barrier(PIPE_V);
        Select(halfLocal, packedLocal, halfLocal, static_cast<half>(-1.0), SELMODE::VSEL_TENSOR_SCALAR_MODE, elemCount);
        pipe_barrier(PIPE_V);
        Cast(int8Local, halfLocal, RoundMode::CAST_ROUND, elemCount);
    }

    __aicore__ inline void ExpandQueries(uint32_t queryOffset, uint32_t queryProcNum)
    {
        LocalTensor<uint8_t> queryPackedLocal = queryPackedBuf.Get<uint8_t>();
        LocalTensor<int8_t> queryInt8Local = expandedInt8Buf.Get<int8_t>();
        LocalTensor<half> queryHalfLocal = expandedHalfBuf.Get<half>();
        uint32_t packedCount = queryProcNum * codeSize;
        uint32_t elemCount = queryProcNum * dim;
        DataCopy(queryPackedLocal, queryGlobal[static_cast<uint64_t>(queryOffset) * codeSize], packedCount);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        ExpandPackedToInt8(queryPackedLocal, queryInt8Local, queryHalfLocal, packedCount);
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
        DataCopy(queryExpandedGlobal[static_cast<uint64_t>(queryOffset) * dim], queryInt8Local, elemCount);
        set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID2);
        wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID2);
    }

    __aicore__ inline void ExpandAllQueries()
    {
        uint32_t queryLoopNum = (queryNum + queryExpandTile - 1) / queryExpandTile;
        for (uint32_t queryLoopIdx = 0; queryLoopIdx < queryLoopNum; ++queryLoopIdx)
        {
            uint32_t queryOffset = queryLoopIdx * queryExpandTile;
            uint32_t queryProcNum = queryNum - queryOffset;
            if (queryProcNum > queryExpandTile)
            {
                queryProcNum = queryExpandTile;
            }
            ExpandQueries(queryOffset, queryProcNum);
        }
    }

    __aicore__ inline void ExpandBaseTile(uint32_t codeOffset, uint32_t codeProcTile)
    {
        LocalTensor<uint8_t> basePackedLocal = queryPackedBuf.Get<uint8_t>();
        LocalTensor<int8_t> baseInt8Local = expandedInt8Buf.Get<int8_t>();
        LocalTensor<half> baseHalfLocal = expandedHalfBuf.Get<half>();
        uint32_t baseGroupElemCount = zRegionHeight * BITS_PER_DIM_BLOCK;
        uint32_t groupNum = codeProcTile / zRegionHeight;
        uint32_t packedBytesPerGroup = zRegionHeight * codeSize;
        for (uint32_t groupIdx = 0; groupIdx < groupNum; groupIdx += baseGroupBatch)
        {
            uint32_t groupProcNum = groupNum - groupIdx;
            if (groupProcNum > baseGroupBatch)
            {
                groupProcNum = baseGroupBatch;
            }
            uint32_t baseIdx = codeOffset + groupIdx * zRegionHeight;
            uint32_t h1 = baseIdx / zRegionHeight;
            uint64_t packedOffset = static_cast<uint64_t>(h1) * packedBytesPerGroup;
            uint32_t packedCount = groupProcNum * packedBytesPerGroup;
            DataCopy(basePackedLocal, baseGlobal[packedOffset], packedCount);
            set_flag(PIPE_MTE2, PIPE_V, EVENT_ID3);
            wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID3);
            ExpandPackedToInt8(basePackedLocal, baseInt8Local, baseHalfLocal, packedCount);
            set_flag(PIPE_V, PIPE_MTE3, EVENT_ID4);
            wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID4);

            const uint16_t blockCount = static_cast<uint16_t>(zRegionHeight);
            const uint16_t blockLen = 1;  // One 32-byte dimension block per base row.
            const uint16_t srcStride = 0;
            const uint16_t dstStride = static_cast<uint16_t>((dim - BITS_PER_DIM_BLOCK) / BUFFER_ALIGN_BYTES);
            DataCopyParams copyParams{blockCount, blockLen, srcStride, dstStride};
            for (uint32_t localGroupIdx = 0; localGroupIdx < groupProcNum; ++localGroupIdx)
            {
                for (uint32_t dimBlock = 0; dimBlock < dimBlockNum; ++dimBlock)
                {
                    uint32_t localOffset = (localGroupIdx * dimBlockNum + dimBlock) * baseGroupElemCount;
                    uint64_t workspaceOffset = (static_cast<uint64_t>(groupIdx + localGroupIdx) * zRegionHeight) * dim +
                                               dimBlock * BITS_PER_DIM_BLOCK;
                    DataCopy(baseExpandedGlobal[workspaceOffset], baseInt8Local[localOffset], copyParams);
                }
            }
            set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID5);
            wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID5);
        }
    }

    __aicore__ inline void ApplyMask(uint32_t queryIdx, uint32_t codeOffset, uint32_t codeProcTile,
                                     LocalTensor<half> &scoreHalfLocal)
    {
        LocalTensor<uint8_t> maskLocal = maskBuf.Get<uint8_t>();
        uint32_t maskQueryIdx = (maskRows == 1) ? 0 : queryIdx;
        uint64_t maskOffset = static_cast<uint64_t>(maskQueryIdx) * maskLen + codeOffset / Utils::MASK_BIT_NUM;
        DataCopy(maskLocal, maskTensor[maskOffset], codeProcTile / Utils::MASK_BIT_NUM);
        LocalTensor<uint8_t> baseMaskLocal = baseMaskBuf.Get<uint8_t>();
        if (useBaseMask)
        {
            DataCopy(baseMaskLocal, baseMaskTensor[codeOffset / Utils::MASK_BIT_NUM],
                     codeProcTile / Utils::MASK_BIT_NUM);
        }
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID3);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID3);

        if (useBaseMask)
        {
            LocalTensor<int16_t> maskInt16Local = maskLocal.ReinterpretCast<int16_t>();
            LocalTensor<int16_t> baseMaskInt16Local = baseMaskLocal.ReinterpretCast<int16_t>();
            And(maskInt16Local, maskInt16Local, baseMaskInt16Local,
                codeProcTile / Utils::MASK_BIT_NUM / sizeof(int16_t));
            pipe_barrier(PIPE_V);
        }

        // The mask stores one bit per base vector. Each VSEL repeat consumes one 128-element half-vector segment;
        // rejected candidates are replaced with HALF_MIN before the per-burst maximum is calculated.
        BinaryRepeatParams param{1, 1, 1, 8, 8, 8};
        constexpr uint32_t distOffset = Utils::SELECT_REPEAT_TIME * Utils::VIC_HALF_FULL_MASK;
        constexpr uint32_t maskRepeatOffset = distOffset / Utils::MASK_BIT_NUM;
        uint32_t selectRepeatNum = codeProcTile / Utils::VIC_HALF_FULL_MASK;
        uint32_t selectLoopTime = selectRepeatNum / Utils::SELECT_REPEAT_TIME;
        uint8_t selectRemainder = static_cast<uint8_t>(selectRepeatNum % Utils::SELECT_REPEAT_TIME);
        for (uint32_t i = 0; i < selectLoopTime; ++i)
        {
            Select(scoreHalfLocal[i * distOffset], maskLocal[i * maskRepeatOffset], scoreHalfLocal[i * distOffset],
                   Utils::HALF_MIN, SELMODE::VSEL_TENSOR_SCALAR_MODE, Utils::VIC_HALF_FULL_MASK,
                   Utils::SELECT_REPEAT_TIME, param);
        }
        if (selectRemainder != 0)
        {
            Select(scoreHalfLocal[selectLoopTime * distOffset], maskLocal[selectLoopTime * maskRepeatOffset],
                   scoreHalfLocal[selectLoopTime * distOffset], Utils::HALF_MIN, SELMODE::VSEL_TENSOR_SCALAR_MODE,
                   Utils::VIC_HALF_FULL_MASK, selectRemainder, param);
        }
        pipe_barrier(PIPE_V);
    }

    __aicore__ inline void WriteDistanceAndMax(uint32_t queryOffset, uint32_t queryProcNum, uint32_t codeOffset,
                                               uint32_t codeProcTile)
    {
        LocalTensor<int32_t> scoreLocal = scoreBuf.Get<int32_t>();
        LocalTensor<float> scoreFloatLocal = scoreLocal.ReinterpretCast<float>();
        LocalTensor<half> scoreHalfLocal = scoreHalfBuf.Get<half>();
        LocalTensor<half> maxHalfLocal = maxLineBuf.Get<half>(MAX_LINE_BYTES / sizeof(half));
        LocalTensor<uint16_t> maxRawLocal = maxHalfLocal.ReinterpretCast<uint16_t>();
        uint32_t burstNum = codeProcTile / burstLen;
        uint32_t srcRepStride = burstLen * sizeof(half) / BUFFER_ALIGN_BYTES;

        for (uint32_t queryLocalIdx = 0; queryLocalIdx < queryProcNum; ++queryLocalIdx)
        {
            DataCopy(scoreLocal, innerProductGlobal[static_cast<uint64_t>(queryLocalIdx) * codeTile], codeProcTile);
            set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
            Cast(scoreFloatLocal, scoreLocal, RoundMode::CAST_NONE, codeProcTile);
            Cast(scoreHalfLocal, scoreFloatLocal, RoundMode::CAST_NONE, codeProcTile);

            uint32_t globalQueryIdx = queryOffset + queryLocalIdx;
            if (!skipMask)
            {
                ApplyMask(globalQueryIdx, codeOffset, codeProcTile, scoreHalfLocal);
            }
            if (codeOffset + codeProcTile > actualNum)
            {
                uint32_t validTileNum = actualNum > codeOffset ? actualNum - codeOffset : 0;
                Duplicate(scoreHalfLocal[validTileNum], Utils::HALF_MIN, codeProcTile - validTileNum);
                pipe_barrier(PIPE_V);
            }
            WholeReduceMax(maxHalfLocal, scoreHalfLocal, burstLen, burstNum, 1, 1, srcRepStride);
            set_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
            wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
            DataCopy(distGlobal[static_cast<uint64_t>(globalQueryIdx) * blockSize + codeOffset], scoreHalfLocal,
                     codeProcTile);
            uint32_t firstBurstIdx = codeOffset / burstLen;
            uint64_t gmOffset = static_cast<uint64_t>(globalQueryIdx) * (blockSize / burstLen * 2) + firstBurstIdx * 2;
            DataCopy(maxDistRawTensor[gmOffset], maxRawLocal, burstNum * 2);
            set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID2);
            wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID2);
        }
    }

   private:
    uint32_t queryNum{0};
    uint32_t codeSize{0};
    uint32_t dim{0};
    uint32_t blockSize{0};
    uint32_t zRegionHeight{0};
    uint32_t burstLen{0};
    uint32_t codeTile{0};
    uint32_t maskRows{0};
    uint32_t actualNum{0};
    uint32_t maskLen{0};
    uint32_t dimBlockNum{0};
    uint32_t queryTile{0};
    uint32_t queryExpandTile{0};
    uint32_t baseGroupBatch{0};
    uint32_t codeOffsetBegin{0};
    uint32_t codeOffsetEnd{0};
    uint32_t codeProcNum{0};
    uint32_t coreIdx{0};
    uint32_t coreNum{1};
    bool workspaceOk{false};
    bool skipMask{false};
    bool useBaseMask{false};
    TCubeTiling cubeTilingIp;

    TPipe pipe;
    TBuf<> queryPackedBuf;
    TBuf<> maskBuf;
    TBuf<> baseMaskBuf;
    TBuf<> expandedInt8Buf;
    TBuf<> expandedHalfBuf;
    TBuf<> scoreBuf;
    TBuf<> scoreHalfBuf;
    TBuf<> maxLineBuf;

    GlobalTensor<uint8_t> queryGlobal;
    GlobalTensor<uint8_t> baseGlobal;
    GlobalTensor<uint8_t> maskTensor;
    GlobalTensor<uint8_t> baseMaskTensor;
    __gm__ uint32_t *actualSizeGlobal{nullptr};
    GlobalTensor<half> distGlobal;
    GlobalTensor<uint16_t> maxDistRawTensor;
    GlobalTensor<int8_t> queryExpandedGlobal;
    GlobalTensor<int8_t> baseExpandedGlobal;
    GlobalTensor<int32_t> innerProductGlobal;
};
}  // namespace IndexOps

extern "C" __global__ __aicore__ void ascendc_distance_flat_hamming_with_mask(GM_ADDR query, GM_ADDR base,
                                                                              GM_ADDR actualSize, GM_ADDR mask,
                                                                              GM_ADDR baseMask, GM_ADDR dist,
                                                                              GM_ADDR maxDist, GM_ADDR flag,
                                                                              GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    IndexOps::AscendcDistanceFlatHammingWithMask op(tilingData);
    op.Init(query, base, actualSize, mask, baseMask, dist, maxDist, flag, workspace);
    op.Process();
}
