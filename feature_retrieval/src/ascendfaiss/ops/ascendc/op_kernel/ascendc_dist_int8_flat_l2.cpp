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


#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/matmul.h"
#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"
#include "op_kernel_common.h"

using namespace AscendC;
using namespace matmul;
using namespace Utils;

namespace {
// 基础数据
constexpr uint32_t BUFFER_DEPTH = 1;
constexpr uint32_t BLOCK_BYBE_SIZE = 32;
constexpr uint32_t BURST_LEN = 64;
constexpr uint32_t FLAG_SIZE = 16;
constexpr uint32_t CORE_PROC_CODE_GRAIN = 256;
constexpr uint8_t IDX_COMP_OFFSET = 1;
constexpr uint8_t IDX_MASK_LEN = 2;
constexpr uint8_t IDX_USE_MASK = 3;
}

namespace IndexOps {
class AscendcDistInt8FlatL2 {
public:
    __aicore__ inline AscendcDistInt8FlatL2(const AscendcDistInt8FlatL2TilingData &tilingData, bool useFp16)
        : queryNum(tilingData.querySize), dim(tilingData.dim), codeBlockSize(tilingData.codeBlockSize),
        availableCoreNum(tilingData.aivNum), blockIdx(GetBlockIdx()), useFp16(useFp16),
        cubeTilingSquare(tilingData.cubeTilingSquare), cubeTilingIp(tilingData.cubeTilingIp) {}

    // 初始化，为输入输出分配内存
    __aicore__ inline void Init(GM_ADDR queries, GM_ADDR mask, GM_ADDR codes,
        GM_ADDR norm, GM_ADDR actualSize, GM_ADDR distance, GM_ADDR distanceMin,
        GM_ADDR flag);

    __aicore__ inline void Process();

private:
    __aicore__ inline void GetShapeInfo(GM_ADDR actualSize);
    // 预分配内存
    __aicore__ inline void InitQueBuffer();

    __aicore__ inline bool IsParaValid();

    __aicore__ inline void SetFlag();

    __aicore__ inline void ComputeCoreProcInfo();

    __aicore__ inline void ComputeLoopParas();

    __aicore__ inline void SetBuffer(GM_ADDR queries, GM_ADDR mask, GM_ADDR codes,
        GM_ADDR norm, GM_ADDR distance, GM_ADDR distanceMin, GM_ADDR flag);

    __aicore__ inline void ProcessEachQueryLoop(uint32_t queryLoopIdx);

    __aicore__ inline void ComputeQuerySquare(uint32_t queryOffset, uint32_t queryProcNum,
        LocalTensor<int32_t>& querySquareLocal, bool isLastQuery);

    __aicore__ inline void ProcessEachCodeLoop(uint32_t queryLoopIdx, uint32_t codeLoopIdx, uint32_t queryProcNum,
        bool isLastQuery, LocalTensor<int32_t>& querySquareLocal);

    __aicore__ inline void CopyInNorm(uint32_t codeLoopIdx);

    __aicore__ inline void ComputeDistance(uint32_t queryLoopIdx, uint32_t codeLoopIdx, uint32_t queryProcNum,
        uint32_t codeProcNum, LocalTensor<int32_t>& querySquareLocal);

    __aicore__ inline void ComputeNormSum(uint32_t queryProcNum,
        LocalTensor<int32_t>& distanceLocal, LocalTensor<int32_t>& querySquareLocal);

    __aicore__ inline void ComputeInnerProduct(uint32_t codeLoopIdx, bool isLastQuery);

    __aicore__ inline void ComputeL2Distance(uint32_t queryProcNum, LocalTensor<int32_t>& distIntTensor);

    __aicore__ inline void ComputeDistanceMin(uint32_t queryProcNum, uint32_t codeProcNum,
        LocalTensor<half>& distanceTensor, LocalTensor<half>& distanceMinTensor);

    __aicore__ inline void CopyOutDist(uint32_t queryLoopIdx, uint32_t codeLoopIdx,
        uint32_t queryProcNum, LocalTensor<half>& distanceTensor);

    __aicore__ inline void CopyOutDistMin(uint32_t queryLoopIdx, uint32_t codeLoopIdx,
        uint32_t queryProcNum, LocalTensor<half>& distanceMinTensor);

    __aicore__ inline void CastDistInt2Half(LocalTensor<int32_t>& srcTensor, LocalTensor<half>& dstTensor);

    __aicore__ inline void DoMask(uint32_t queryLoopIdx, uint32_t codeLoopIdx, uint32_t queryProcNum,
        LocalTensor<half>& distanceTensor);

private:
    using MatmulTypeQueryA = MatmulType<TPosition::GM, CubeFormat::ND, int8_t>;
    using MatmulTypeCode = MatmulType<TPosition::GM, CubeFormat::ND, int8_t, true>;
    using MatMulTypeIp = MatmulType<TPosition::GM, CubeFormat::ND, int32_t>;
    Matmul<MatmulTypeQueryA, MatmulTypeCode, MatMulTypeIp> matmulObjIp;

    using MatmulTypeQueryB = MatmulType<TPosition::GM, CubeFormat::ND, int8_t, true>;
    using MatmulTypeResult = MatmulType<TPosition::LCM, CubeFormat::ND, int32_t>;
    Matmul<MatmulTypeQueryA, MatmulTypeQueryB, MatmulTypeResult> matmulObjSquare;

    TCubeTiling cubeTilingSquare;
    TCubeTiling cubeTilingIp;

    // 输入输出tensor
    GlobalTensor<int8_t> queryTensor;
    GlobalTensor<uint8_t> maskTensor;
    GlobalTensor<int8_t> codesTensor;
    GlobalTensor<int32_t> normTensor;
    GlobalTensor<half> distTensor;
    GlobalTensor<half> minDistTensor;
    GlobalTensor<uint16_t> flagTensor;
    GlobalTensor<int32_t> innerProductTensor;

    TPipe m_pipe;
    TQue<QuePosition::VECIN, BUFFER_DEPTH> normQue;
    TQue<QuePosition::VECIN, BUFFER_DEPTH> maskQue;

    TQue<QuePosition::VECOUT, BUFFER_DEPTH> querySquareQue;
    TQue<QuePosition::VECOUT, BUFFER_DEPTH> queryTmpSquareQue;
    TQue<QuePosition::VECOUT, BUFFER_DEPTH> distanceQue;
    TQue<QuePosition::VECOUT, BUFFER_DEPTH> innerProductQue;
    TQue<QuePosition::VECOUT, BUFFER_DEPTH> distanceMinQue;

    uint32_t queryNum { 0 };
    uint32_t dim { 0 };
    uint32_t codeNum { 0 };
    uint32_t availableCoreNum { 0 };
    uint32_t blockIdx { 0 };
    uint32_t codeBlockSize { 0 };

    bool useFp16 { false };

    // 循环数据
    uint32_t codeOffset { 0 };
    uint32_t codeProcNum { 0 };

    // tiling
    uint32_t querySizeEachLoop { 0 };
    uint32_t codeSizeEachLoop { 0 };
    uint32_t querySizeLastLoop { 0 };
    uint32_t queryAlign { 0 };
    uint32_t queryLoopTimes { 0 };
    uint32_t queryCopyOutOffset { 0 };
    uint32_t queryCopyOutBatchOffset { 0 };
    uint32_t codeSizeLastLoop { 0 };
    uint32_t codeLoopTimes { 0 };
    uint32_t codeBatchSize { 0 };
    uint32_t codeCopyOutBatchOffset { 0 };
    uint32_t codeBurstLen { 0 };

    uint8_t validResultVecRepeats { 0 };
    uint32_t onceNumMaskLen { 0 };
    uint32_t maskBlockOffset { 0 };
    uint32_t maskLen { 0 };
    uint32_t useMask { 0 };
    float scale { 0.0 };
};

// *******************************************************
// 算子实现

__aicore__ inline void AscendcDistInt8FlatL2::Init(GM_ADDR queries, GM_ADDR mask, GM_ADDR codes,
    GM_ADDR norm, GM_ADDR actualSize, GM_ADDR distance, GM_ADDR distanceMin, GM_ADDR flag)
{
    GetShapeInfo(actualSize);

    ComputeCoreProcInfo();

    ComputeLoopParas();

    SetBuffer(queries, mask, codes, norm, distance, distanceMin, flag);

    InitQueBuffer();
}

__aicore__ inline void AscendcDistInt8FlatL2::GetShapeInfo(GM_ADDR actualSize)
{
    codeNum = *(reinterpret_cast<__gm__ uint32_t*>(actualSize));
    maskBlockOffset = *(reinterpret_cast<__gm__ uint32_t*>(actualSize) + IDX_COMP_OFFSET);
    maskLen = *(reinterpret_cast<__gm__ uint32_t*>(actualSize) + IDX_MASK_LEN);
    useMask = *(reinterpret_cast<__gm__ uint32_t*>(actualSize) + IDX_USE_MASK);

    // 数据从uint32_t类型转换为float16类型时，为避免精度丢失，需要在转换时对数据进行缩放
    // 缩放参数m_scale根据dim大小改变，计算公式保持与tik一致，64、128、4数值含义待理清
    uint32_t scaleTmp = Min(dim / 64, Max(dim / 128 + 1, static_cast<uint32_t>(4)));
    scale = 0.01f / scaleTmp;  // scale最大值为0.01
}

__aicore__ inline void AscendcDistInt8FlatL2::ComputeCoreProcInfo()
{
    uint32_t taskNum = DivUp(codeNum, CORE_PROC_CODE_GRAIN);  // 底库切分，循环任务数
    uint32_t usedCoreNum = Min(taskNum, availableCoreNum);
    uint32_t batchNumPerCoreProcess = taskNum / usedCoreNum;  // 每个核分到的任务数
    uint32_t batchTail = taskNum % usedCoreNum;   // 剩余任务数
    uint32_t codeNumEachCoreHead = batchNumPerCoreProcess + 1;  // 将剩余任务数分到前batchTail个核上，每个核处理总任务数

    if (blockIdx >= usedCoreNum) {  // 底库数量小于可用核数时，当前核未处理数据
        codeProcNum = 0;
    } else if (blockIdx != usedCoreNum - 1) {
        if (blockIdx < batchTail) {
            codeOffset = (blockIdx * codeNumEachCoreHead) * CORE_PROC_CODE_GRAIN;
            codeProcNum = codeNumEachCoreHead * CORE_PROC_CODE_GRAIN;
        } else {
            codeOffset = (batchTail * codeNumEachCoreHead +
                (blockIdx - batchTail) * batchNumPerCoreProcess) * CORE_PROC_CODE_GRAIN;
            codeProcNum = batchNumPerCoreProcess * CORE_PROC_CODE_GRAIN;
        }
    } else {
        codeOffset = (batchTail * codeNumEachCoreHead +
            (blockIdx - batchTail) * batchNumPerCoreProcess) * CORE_PROC_CODE_GRAIN;
        codeProcNum = codeNum - codeOffset;
    }
}

__aicore__ inline void AscendcDistInt8FlatL2::ComputeLoopParas()
{
    querySizeEachLoop = cubeTilingIp.M;
    queryLoopTimes = DivUp(queryNum, querySizeEachLoop);  // batch循环次数
    querySizeLastLoop = queryNum - (queryLoopTimes - 1) * querySizeEachLoop;   // batch剩余次数
    queryAlign = AlignUp(querySizeEachLoop, CUBE_ALIGN);
    queryCopyOutOffset = querySizeEachLoop * codeBlockSize;
    codeBurstLen = (codeBlockSize + BURST_LEN - 1) / BURST_LEN;
    queryCopyOutBatchOffset = querySizeEachLoop * codeBurstLen;

    codeSizeEachLoop = cubeTilingIp.N;
    codeLoopTimes = DivUp(codeProcNum, codeSizeEachLoop);
    codeSizeLastLoop = codeProcNum - (codeLoopTimes - 1) * codeSizeEachLoop;
    codeBatchSize = AlignUp(CORE_PROC_CODE_GRAIN / BURST_LEN, CUBE_ALIGN / BURST_BLOCK_RATIO);
    codeCopyOutBatchOffset = DivUp(codeSizeEachLoop, BURST_LEN);

    onceNumMaskLen = codeSizeEachLoop / MASK_BIT_NUM;
    validResultVecRepeats = codeSizeEachLoop * querySizeEachLoop / VIC_HALF_FULL_MASK;
}

__aicore__ inline void AscendcDistInt8FlatL2::SetBuffer(GM_ADDR queries, GM_ADDR mask, GM_ADDR codes,
    GM_ADDR norm, GM_ADDR distance, GM_ADDR distanceMin, GM_ADDR flag)
{
    queryTensor.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(queries));
    maskTensor.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(mask) +
        (maskBlockOffset + codeOffset) / MASK_BIT_NUM);
    codesTensor.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(codes) + codeOffset * dim);
    normTensor.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(norm) + codeOffset);
    distTensor.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(distance) + codeOffset);
    minDistTensor.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(distanceMin)
         + (codeOffset / BURST_LEN) * BURST_BLOCK_RATIO);
    flagTensor.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t*>(flag) + blockIdx * CUBE_ALIGN);

    // 异步matmul使用
    GM_ADDR workSpace = GetSysWorkSpacePtr();
    GM_ADDR userWorkSpace = GetUserWorkspace(workSpace);
    if (userWorkSpace == nullptr) {
        return;
    }
    innerProductTensor.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(userWorkSpace) +
        blockIdx * querySizeEachLoop * CORE_PROC_CODE_GRAIN);
}

__aicore__ inline void AscendcDistInt8FlatL2::InitQueBuffer()
{
    if (codeProcNum != 0) {
        m_pipe.InitBuffer(querySquareQue, BUFFER_DEPTH, queryAlign * sizeof(int32_t));
        m_pipe.InitBuffer(queryTmpSquareQue, BUFFER_DEPTH, queryAlign * queryAlign * sizeof(int32_t));
        m_pipe.InitBuffer(normQue, BUFFER_DEPTH, codeSizeEachLoop * sizeof(int32_t));
        m_pipe.InitBuffer(distanceQue, BUFFER_DEPTH, querySizeEachLoop * codeSizeEachLoop * sizeof(int32_t));
        m_pipe.InitBuffer(innerProductQue, BUFFER_DEPTH, querySizeEachLoop * codeSizeEachLoop * sizeof(int32_t));
        m_pipe.InitBuffer(distanceMinQue, BUFFER_DEPTH,
            codeBatchSize * querySizeEachLoop * BURST_BLOCK_RATIO * sizeof(half));
        m_pipe.InitBuffer(maskQue, BUFFER_DEPTH, querySizeEachLoop * codeSizeEachLoop / MASK_BIT_NUM * sizeof(uint8_t));
    } else {
        constexpr uint32_t defaultSize = 32;
        m_pipe.InitBuffer(querySquareQue, BUFFER_DEPTH, defaultSize);
        m_pipe.InitBuffer(queryTmpSquareQue, BUFFER_DEPTH, defaultSize);
        m_pipe.InitBuffer(normQue, BUFFER_DEPTH, defaultSize);
        m_pipe.InitBuffer(distanceQue, BUFFER_DEPTH, defaultSize);
        m_pipe.InitBuffer(innerProductQue, BUFFER_DEPTH, defaultSize);
        m_pipe.InitBuffer(distanceMinQue, BUFFER_DEPTH, defaultSize);
        m_pipe.InitBuffer(maskQue, BUFFER_DEPTH, defaultSize);
    }
}

__aicore__ inline void AscendcDistInt8FlatL2::Process()
{
    REGIST_MATMUL_OBJ(&m_pipe, GetSysWorkSpacePtr(), matmulObjSquare, matmulObjIp);

    if (!IsParaValid()) {
        SetFlag();
        return;
    }

    // 本次处理任务数为0，直接设置flag返回
    if (codeProcNum == 0) {
        SetFlag();
        return;
    }

    matmulObjSquare.Init(&cubeTilingSquare);
    matmulObjIp.Init(&cubeTilingIp);

    for (uint32_t i = 0; i < queryLoopTimes; i++) {
        ProcessEachQueryLoop(i);
    }

    matmulObjSquare.End();
    matmulObjIp.End();

    SetFlag();
}

__aicore__ inline bool AscendcDistInt8FlatL2::IsParaValid()
{
    // dim大于等于64且必须是32的倍数
    if ((dim % 32 != 0) || (dim < 64)) {
        return false;
    }

    // codeBlockSize必须是16的倍数
    if (codeBlockSize % 16 != 0) {
        return false;
    }

    return true;
}

__aicore__ inline void AscendcDistInt8FlatL2::SetFlag()
{
    pipe_barrier(PIPE_MTE3);
    TBuf<> flagBuf;
    m_pipe.InitBuffer(flagBuf, DEFAULT_C0_SIZE);
    LocalTensor<uint16_t> flagLocal = flagBuf.Get<uint16_t>(DEFAULT_C0_SIZE / sizeof(uint16_t));
    flagLocal.SetValue(0, 1);
    set_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
    DataCopy(flagTensor, flagLocal, flagLocal.GetSize());
}

__aicore__ inline void AscendcDistInt8FlatL2::ProcessEachQueryLoop(uint32_t queryLoopIdx)
{
    bool isLastQuery = (queryLoopIdx == queryLoopTimes - 1);
    uint32_t queryProcNum = isLastQuery ? querySizeLastLoop : querySizeEachLoop;
    uint32_t queryOffset = queryLoopIdx * querySizeEachLoop * dim;

    auto querySquareLocal = querySquareQue.AllocTensor<int32_t>();
    ComputeQuerySquare(queryOffset, queryProcNum, querySquareLocal, isLastQuery);

    matmulObjIp.SetTensorA(queryTensor[queryOffset]);
    for (uint32_t i = 0; i < codeLoopTimes; i++) {
        ProcessEachCodeLoop(queryLoopIdx, i, queryProcNum, isLastQuery, querySquareLocal);
    }
    querySquareQue.FreeTensor(querySquareLocal);
}

__aicore__ inline void AscendcDistInt8FlatL2::ComputeQuerySquare(uint32_t queryOffset, uint32_t queryProcNum,
    LocalTensor<int32_t>& querySquareLocal, bool isLastQuery)
{
    auto queryTmpLocal = queryTmpSquareQue.AllocTensor<int32_t>();
    if (isLastQuery) {
        matmulObjSquare.SetTail(querySizeLastLoop, querySizeLastLoop, dim);
    }

    matmulObjSquare.SetTensorA(queryTensor[queryOffset]);
    matmulObjSquare.SetTensorB(queryTensor[queryOffset], true);
    matmulObjSquare.IterateAll(queryTmpLocal);
    queryTmpSquareQue.EnQue<int32_t>(queryTmpLocal);

    {
        auto queryTmpLocal = queryTmpSquareQue.DeQue<int32_t>();
        set_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);

        for (uint32_t i = 0; i < queryProcNum; i++) {
            auto querySquareSum = queryTmpLocal.GetValue(i * querySizeEachLoop + i);
            querySquareLocal.SetValue(i, querySquareSum);
        }
        queryTmpSquareQue.FreeTensor(queryTmpLocal);
    }
}

__aicore__ inline void AscendcDistInt8FlatL2::ProcessEachCodeLoop(uint32_t queryLoopIdx, uint32_t codeLoopIdx,
    uint32_t queryProcNum, bool isLastQuery, LocalTensor<int32_t>& querySquareLocal)
{
    ComputeInnerProduct(codeLoopIdx, isLastQuery);

    CopyInNorm(codeLoopIdx);

    uint32_t codeProcNum = (codeLoopIdx == codeLoopTimes - 1) ? codeSizeLastLoop : codeSizeEachLoop;
    ComputeDistance(queryLoopIdx, codeLoopIdx, queryProcNum, codeProcNum, querySquareLocal);
}

__aicore__ inline void AscendcDistInt8FlatL2::ComputeInnerProduct(uint32_t codeLoopIdx, bool isLastQuery)
{
    if (isLastQuery) {
        matmulObjIp.SetTail(querySizeLastLoop, codeSizeEachLoop, dim);
    }

    uint32_t codeOffset = codeLoopIdx * codeSizeEachLoop * dim;
    matmulObjIp.SetTensorB(codesTensor[codeOffset], true);

    matmulObjIp.IterateAll<false>(innerProductTensor, 0, false, true);
}

__aicore__ inline void AscendcDistInt8FlatL2::CopyInNorm(uint32_t codeLoopIdx)
{
    uint32_t copyOffset = codeLoopIdx * codeSizeEachLoop;
    LocalTensor<int32_t> norm = normQue.AllocTensor<int32_t>();
    DataCopy(norm, normTensor[copyOffset], codeSizeEachLoop);
    normQue.EnQue<int32_t>(norm);
}

__aicore__ inline void AscendcDistInt8FlatL2::ComputeDistance(uint32_t queryLoopIdx, uint32_t codeLoopIdx,
    uint32_t queryProcNum, uint32_t codeProcNum, LocalTensor<int32_t>& querySquareLocal)
{
    auto distanceIntTensor = distanceQue.AllocTensor<int32_t>();
    ComputeNormSum(queryProcNum, distanceIntTensor, querySquareLocal);

    ComputeL2Distance(queryProcNum, distanceIntTensor);

    // 精度转换
    auto distanceTensor = LocalTensor<half>();
    CastDistInt2Half(distanceIntTensor, distanceTensor);

    DoMask(queryLoopIdx, codeLoopIdx, queryProcNum, distanceTensor);

    auto distanceMinTensor = distanceMinQue.AllocTensor<half>();
    ComputeDistanceMin(queryProcNum, codeProcNum, distanceTensor, distanceMinTensor);

    // wait for compute distance over
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);

    CopyOutDist(queryLoopIdx, codeLoopIdx, queryProcNum, distanceTensor);
    CopyOutDistMin(queryLoopIdx, codeLoopIdx, queryProcNum, distanceMinTensor);

    distanceQue.FreeTensor(distanceIntTensor);
    distanceMinQue.FreeTensor(distanceMinTensor);

    // wait for copy distance to gm over
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID2);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID2);
}

__aicore__ inline void AscendcDistInt8FlatL2::ComputeNormSum(uint32_t queryProcNum,
    LocalTensor<int32_t>& distanceLocal, LocalTensor<int32_t>& querySquareLocal)
{
    // y^2
    auto norm = normQue.DeQue<int32_t>();
    for (uint32_t i = 0; i < queryProcNum; i++) {
        auto querySquare = querySquareLocal.GetValue(i);
        set_flag(PIPE_S, PIPE_V, EVENT_ID0);

        wait_flag(PIPE_S, PIPE_V, EVENT_ID0);
        Adds(distanceLocal[i * codeSizeEachLoop], norm, querySquare, codeSizeEachLoop);
    }

    normQue.FreeTensor(norm);
}

__aicore__ inline void AscendcDistInt8FlatL2::ComputeL2Distance(uint32_t queryProcNum,
    LocalTensor<int32_t>& distIntTensor)
{
    matmulObjIp.WaitIterateAll();
    auto innerProductLocal = innerProductQue.AllocTensor<int32_t>();
    DataCopy(innerProductLocal, innerProductTensor, queryProcNum * codeSizeEachLoop);

    set_flag(PIPE_MTE2, PIPE_S, EVENT_ID6);
    wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID6);

    int32_t neg2 = -2;  // 计算-2xy
    Muls(innerProductLocal, innerProductLocal, neg2, queryProcNum * codeSizeEachLoop);
    Add(distIntTensor, distIntTensor, innerProductLocal, queryProcNum * codeSizeEachLoop);

    innerProductQue.FreeTensor(innerProductLocal);
}

__aicore__ inline void AscendcDistInt8FlatL2::CastDistInt2Half(LocalTensor<int32_t>& srcTensor,
    LocalTensor<half>& dstTensor)
{
    auto distFloatLocal = srcTensor.ReinterpretCast<float>();
    Cast(distFloatLocal, srcTensor, RoundMode::CAST_NONE, querySizeEachLoop * codeSizeEachLoop);

    Muls(distFloatLocal, distFloatLocal, scale, querySizeEachLoop * codeSizeEachLoop);

    dstTensor = distFloatLocal.ReinterpretCast<half>();
    Cast(dstTensor, distFloatLocal, RoundMode::CAST_NONE, querySizeEachLoop * codeSizeEachLoop);
}

__aicore__ inline void AscendcDistInt8FlatL2::DoMask(uint32_t queryLoopIdx, uint32_t codeLoopIdx,
    uint32_t queryProcNum, LocalTensor<half>& distanceTensor)
{
    if (useMask == 0) {
        return;
    }

    auto maskLocal = maskQue.AllocTensor<uint8_t>();

    uint64_t maskOffset = queryLoopIdx * querySizeEachLoop * static_cast<uint64_t>(maskLen) +
        codeLoopIdx * codeSizeEachLoop / MASK_BIT_NUM;

    for (uint32_t i = 0; i < queryProcNum; i++) {
        DataCopy(maskLocal[i * onceNumMaskLen], maskTensor[i * static_cast<uint64_t>(maskLen) + maskOffset],
            onceNumMaskLen);
    }
    maskQue.EnQue(maskLocal);
    maskLocal = maskQue.DeQue<uint8_t>();
    BinaryRepeatParams param {
        1,
        1,
        1,
        8,
        8,
        8
    };

    uint32_t selectLoopTimes = validResultVecRepeats / SELECT_REPEAT_TIME;
    uint8_t selectRemainderTimes = static_cast<uint8_t>(validResultVecRepeats % SELECT_REPEAT_TIME);
    for (uint32_t i = 0; i < selectLoopTimes; i++) {
        Select(distanceTensor[i * SELECT_REPEAT_TIME * VIC_HALF_FULL_MASK],
               maskLocal[i * SELECT_REPEAT_TIME * VIC_HALF_FULL_MASK / MASK_BIT_NUM],
               distanceTensor[i * SELECT_REPEAT_TIME * VIC_HALF_FULL_MASK],
               HALF_MAX,
               SELMODE::VSEL_TENSOR_SCALAR_MODE,
               VIC_HALF_FULL_MASK,
               static_cast<uint8_t>(SELECT_REPEAT_TIME),
               param);
    }
    if (selectRemainderTimes != 0) {
        Select(distanceTensor[selectLoopTimes * SELECT_REPEAT_TIME * VIC_HALF_FULL_MASK],
               maskLocal[selectLoopTimes * SELECT_REPEAT_TIME * VIC_HALF_FULL_MASK / MASK_BIT_NUM],
               distanceTensor[selectLoopTimes * SELECT_REPEAT_TIME * VIC_HALF_FULL_MASK],
               HALF_MAX,
               SELMODE::VSEL_TENSOR_SCALAR_MODE,
               VIC_HALF_FULL_MASK,
               selectRemainderTimes,
               param);
    }

    maskQue.FreeTensor(maskLocal);
}

__aicore__ inline void AscendcDistInt8FlatL2::ComputeDistanceMin(uint32_t queryProcNum, uint32_t codeProcNum,
    LocalTensor<half>& distanceTensor, LocalTensor<half>& distanceMinTensor)
{
    half zero = 0.0;
    Duplicate(distanceMinTensor, zero, codeBatchSize * querySizeEachLoop * BURST_BLOCK_RATIO);
    constexpr auto srcRepStride = BURST_LEN * sizeof(half) / BLOCK_BYBE_SIZE;
    if (codeProcNum / BURST_LEN != 0) {
        for (uint32_t i = 0; i < queryProcNum; i++) {
            WholeReduceMin<half>(distanceMinTensor[i * codeBatchSize * BURST_BLOCK_RATIO],
                                 distanceTensor[i * codeSizeEachLoop],
                                 BURST_LEN,
                                 codeProcNum / BURST_LEN,
                                 1,
                                 1,
                                 srcRepStride);
        }
    }

    pipe_barrier(PIPE_V);

    if (codeProcNum % BURST_LEN != 0) {
        for (uint32_t i = 0; i < queryProcNum; i++) {
            WholeReduceMin<half>(distanceMinTensor[(i * codeBatchSize + codeProcNum / BURST_LEN) * BURST_BLOCK_RATIO],
                                 distanceTensor[i * codeSizeEachLoop + codeProcNum / BURST_LEN * BURST_LEN],
                                 codeProcNum % BURST_LEN,
                                 1,
                                 1,
                                 1,
                                 srcRepStride);
        }
    }
}

__aicore__ inline void AscendcDistInt8FlatL2::CopyOutDist(uint32_t queryLoopIdx, uint32_t codeLoopIdx,
    uint32_t queryProcNum, LocalTensor<half>& distanceTensor)
{
    auto copyOffset = queryLoopIdx * queryCopyOutOffset + codeLoopIdx * codeSizeEachLoop;
    DataCopyParams copyParams;
    copyParams.blockCount = queryProcNum;
    copyParams.blockLen = codeSizeEachLoop * sizeof(half) / BLOCK_BYBE_SIZE;
    copyParams.dstStride = (codeBlockSize - codeSizeEachLoop) * sizeof(half) / BLOCK_BYBE_SIZE;
    DataCopy(distTensor[copyOffset], distanceTensor, copyParams);
}

__aicore__ inline void AscendcDistInt8FlatL2::CopyOutDistMin(uint32_t queryLoopIdx, uint32_t codeLoopIdx,
    uint32_t queryProcNum, LocalTensor<half>& distanceMinTensor)
{
    auto copyOffset = (queryLoopIdx * queryCopyOutBatchOffset + codeLoopIdx * codeCopyOutBatchOffset)
        * BURST_BLOCK_RATIO;
    DataCopyParams copyParams;
    copyParams.blockCount = queryProcNum;
    copyParams.blockLen = codeBatchSize * BURST_BLOCK_RATIO * sizeof(half) / BLOCK_BYBE_SIZE;
    copyParams.dstStride = (codeBurstLen - codeBatchSize) * BURST_BLOCK_RATIO * sizeof(half) / BLOCK_BYBE_SIZE;
    SetAtomicAdd<half>();
    DataCopy(minDistTensor[copyOffset], distanceMinTensor, copyParams);
    SetAtomicNone();
}
} /* namespace IndexOps */


extern "C" __global__ __aicore__ void ascendc_dist_int8_flat_l2(GM_ADDR queries, GM_ADDR mask, GM_ADDR codes,
    GM_ADDR norm, GM_ADDR actualSize, GM_ADDR distance, GM_ADDR distanceMin, GM_ADDR flag, GM_ADDR workspace,
    GM_ADDR tiling)
{
    if (workspace == nullptr) {
        return;
    }

    GET_TILING_DATA(tilingData, tiling);

    IndexOps::AscendcDistInt8FlatL2 op(tilingData, false);
    if ASCEND_IS_AIV {
        op.Init(queries, mask, codes, norm, actualSize, distance, distanceMin, flag);
    }
    op.Process();
}
