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


#include <cstdio>
#include <algorithm>
#include <type_traits>
#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "op_kernel_common.h"

using namespace AscendC;
using namespace matmul;

namespace {
constexpr uint32_t BUFFER_NUM = 1;
constexpr uint32_t BLOCK_BYTE_SIZE = 32;
constexpr uint32_t MIN_BATCH = 64;
constexpr uint32_t CORE_PROC_CODE_GRAIN = 256;
constexpr uint32_t CODE_SIZE = 16384 * Utils::CUBE_ALIGN;
constexpr uint32_t CODE_BATCH_SIZE = (CODE_SIZE + MIN_BATCH - 1) / MIN_BATCH;
}

namespace IndexOps {
// 后续如果DistFlatL2Kernel 需要支持更多类型 可以参考如下通过特化新的类型来完成
template <typename T>
struct TraitsUpgradeType {
    using type = T;
};

template <>
struct TraitsUpgradeType<half> {
    using type = float;
};

template <typename DataType>
class DistanceFlatL2 {
public:
    // L2计算过程中涉及到精度抬升 这个表示抬升得精度类型
    using UpGradeDataType = typename TraitsUpgradeType<DataType>::type;

    __aicore__ inline DistanceFlatL2() = default;
    __aicore__ inline void Init(GM_ADDR inputQueries, GM_ADDR mask, GM_ADDR inputCodes, GM_ADDR inputL2PreNorm,
                                GM_ADDR inputActualSize, GM_ADDR outputL2Dist, GM_ADDR outputL2DistMin,
                                GM_ADDR outputflag, const DistanceFlatL2TilingData *__restrict tiling);

    __aicore__ inline void process();

private:
    __aicore__ inline void processEachQueryLoop(uint32_t queryLoopIdx);

    __aicore__ inline void processEachCodeLoop(uint32_t queryLoopIdx, uint32_t codeLoopIdx, uint32_t queryProcNum,
                                               bool isQueryTail, LocalTensor<UpGradeDataType> &queryL2NormLocal);

    // preNorm(gm) -> code_l2_local(ub)
    __aicore__ inline void CopyInPreNorm(uint32_t codeLoopIdx);

    __aicore__ inline void Compute(uint32_t queryLoopIdx, uint32_t codeLoopIdx, uint32_t queryProcNum,
                                   uint32_t codeProcNum, LocalTensor<UpGradeDataType> &queryL2NormLocal);

    __aicore__ inline void ComputeQueryL2Norm(uint32_t queryOffset, uint32_t queryProcNum,
                                              LocalTensor<UpGradeDataType> &queryL2NormLocal, bool isTail);

    // compute part1:query_local^2 + code_l2_local^2 (vector)
    __aicore__ inline void ComputeL2NormSum(uint32_t queryProcNum, LocalTensor<UpGradeDataType> &l2NormSumLocal,
                                            LocalTensor<UpGradeDataType> &queryL2NormLocal);

    // compute part2:x * y
    __aicore__ inline void ComputeInnerProduct(uint32_t codeLoopIdx, bool isTail);

    // compute l2Distance:part1 - 2*part2
    __aicore__ inline void ComputeL2Distance(uint32_t queryProcNum, LocalTensor<DataType> &l2DistanceTensor,
                                             LocalTensor<UpGradeDataType> &l2DistanceLocal);

    // compute l2DistanceMin
    __aicore__ inline void ComputeL2DistanceMin(uint32_t queryProcNum, uint32_t codeProcNum,
                                                LocalTensor<DataType> &l2DistanceLocal,
                                                LocalTensor<DataType> &l2DistanceLocalMin);

    // l2_distance_local(ub) -> l2_distance(gm)
    __aicore__ inline void CopyOutL2Dist(uint32_t queryLoopIdx, uint32_t codeLoopIdx, uint32_t queryProcNum,
                                         LocalTensor<DataType> &l2DistanceLocal);

    __aicore__ inline void CopyOutL2DistMin(uint32_t queryLoopIdx, uint32_t codeLoopIdx, uint32_t queryProcNum,
                                            LocalTensor<DataType> &l2DistanceLocalMin);

    __aicore__ inline void SetFlagReady();

    __aicore__ inline void GetMaskParameters(GM_ADDR inputActualSize);

    __aicore__ inline void InitShapeInfoFormGm(GM_ADDR inputActualSize);

    __aicore__ inline void InitCoreProcInfo();

    __aicore__ inline void InitGlobalTensor(GM_ADDR inputQueries, GM_ADDR mask, GM_ADDR inputCodes,
                                            GM_ADDR inputL2PreNorm, GM_ADDR outputL2Dist, GM_ADDR outputL2DistMin,
                                            GM_ADDR outputflag);

    __aicore__ inline void InitMemoryQueue();

    __aicore__ inline void InitTilingArgs();

    __aicore__ inline void DoMask(uint32_t queryLoopIdx, uint32_t codeLoopIdx, uint32_t queryProcNum,
                                  uint32_t codeProcNum, LocalTensor<half> &l2DistanceLocal);

private:
    using MatMulTypeQueryA = MatmulType<TPosition::GM, CubeFormat::ND, DataType>;
    using MatMulTypeCode = MatmulType<TPosition::GM, CubeFormat::ND, DataType, true>;
    using MatMulTypeIp = MatmulType<TPosition::GM, CubeFormat::ND, UpGradeDataType>;
    Matmul<MatMulTypeQueryA, MatMulTypeCode, MatMulTypeIp> matmulObjIp;

    using MatMulTypeQueryB = MatmulType<TPosition::GM, CubeFormat::ND, DataType, true>;
    using MatMulTypeResult = MatmulType<TPosition::LCM, CubeFormat::ND, UpGradeDataType>;
    Matmul<MatMulTypeQueryA, MatMulTypeQueryB, MatMulTypeResult> matmulObjL2Norm;

    GlobalTensor<DataType> querysGlobal;
    GlobalTensor<uint8_t> maskGlobal;
    GlobalTensor<DataType> codesGlobal;
    GlobalTensor<DataType> codesPreNormGlobal;
    GlobalTensor<DataType> l2distGlobal;
    GlobalTensor<DataType> l2distMinGlobal;
    GlobalTensor<uint16_t> flagGlobal;
    GlobalTensor<UpGradeDataType> innerProductGloabal;

    TPipe pipe;
    TQue<QuePosition::VECOUT, BUFFER_NUM> queryL2NormQue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> queryDotQueryQue;
    TSCM<QuePosition::GM, BUFFER_NUM> codeInQue;
    TQue<QuePosition::VECIN, BUFFER_NUM> codeL2NormQue;
    TQue<QuePosition::VECIN, BUFFER_NUM> codeL2NormCastQue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> innerProductQue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> l2DistanceQue;
    TQue<QuePosition::VECIN, BUFFER_NUM> l2DistanceLocalQue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> l2DistanceMinQue;
    TQue<QuePosition::VECIN, BUFFER_NUM> maskQue;

    // dims range:32 64 128 256 384 512 1024
    // query range:48 36 32 30 24 18 16 12 8 6 4 2 1
    // code range: (0, 16384]
    // shape info
    uint32_t codeSize{ 0 };
    uint32_t querySize{ 0 };
    uint32_t dimSize{ 0 };

    // core info
    uint32_t usedCoreNum{ 0 };
    uint32_t coreId{ 0 };
    uint32_t coreNum{ 0 };
    uint32_t corePorcCodeSize{ 0 };
    uint32_t coreProcCodeOffset{ 0 };

    // tiling args
    uint32_t querySizeEachLoop{ 0 };
    uint32_t querySizeLastLoop{ 0 };
    uint32_t queryAlign{ 0 };
    uint32_t queryLoopTimes{ 0 };
    uint32_t queryCopyOutOffset{ 0 };
    uint32_t queryCopyOutBatchOffset{ 0 };
    uint32_t codeSizeEachLoop{ 0 };
    uint32_t codeSizeLastLoop{ 0 };
    uint32_t codeLoopTimes{ 0 };
    uint32_t codeBatchSize{ 0 };
    uint32_t codeCopyOutBatchOffset{ 0 };

    uint32_t maskLenEachLoop {0};
    uint32_t maskBlockOffset {0};
    uint32_t maskLen {0};
    uint32_t maskFlag {0};
    uint32_t selectLoopTime {0};
    uint8_t selectRemainder {0};

    TCubeTiling cubeTilingIp;
    TCubeTiling cubeTilingL2Norm;

    const DistanceFlatL2TilingData *__restrict pTilingDevice{ nullptr };
};

template <typename DataType>
__aicore__ inline void DistanceFlatL2<DataType>::GetMaskParameters(GM_ADDR inputActualSize)
{
    maskBlockOffset = *(reinterpret_cast<__gm__ uint32_t *>(inputActualSize) + Utils::MASK_BLOCK_OFFSET_IDX);
    maskLen = *(reinterpret_cast<__gm__ uint32_t *>(inputActualSize) + Utils::MASK_LEN_IDX);
    maskFlag = *(reinterpret_cast<__gm__ uint32_t *>(inputActualSize) + Utils::MASK_FLAG_IDX);
}

template <typename DataType>
__aicore__ inline void DistanceFlatL2<DataType>::InitShapeInfoFormGm(GM_ADDR inputActualSize)
{
    GlobalTensor<uint32_t> actualSizeGm;
    actualSizeGm.SetGlobalBuffer((__gm__ uint32_t *)inputActualSize);

    ALLOC_LOCAL_TENSOR(actualSizeLocal, 8, uint32_t, VECIN);
    DataCopy(actualSizeLocal, actualSizeGm, BLOCK_BYTE_SIZE / sizeof(uint32_t));
    set_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);

    wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
    codeSize = actualSizeLocal.GetValue(0);
    querySize = pTilingDevice->querySize;
    dimSize = pTilingDevice->dimSize;
}

template <typename DataType>
__aicore__ inline void DistanceFlatL2<DataType>::InitCoreProcInfo()
{
    coreId = GetBlockIdx();
    coreNum = pTilingDevice->aivNum;
    uint32_t taskNum = Utils::DivUp(codeSize, CORE_PROC_CODE_GRAIN);
    usedCoreNum = Utils::Min(taskNum, coreNum);

    uint32_t taskNumEachCore = taskNum / usedCoreNum;
    uint32_t taskpaddingNumEachCore = taskNumEachCore + 1;
    uint32_t paddingIdx = taskNum % usedCoreNum;

    if (coreId >= usedCoreNum) {
        corePorcCodeSize = 0;
    } else if (coreId != (usedCoreNum - 1)) {
        if (coreId < paddingIdx) {
            corePorcCodeSize = taskpaddingNumEachCore * CORE_PROC_CODE_GRAIN;
            coreProcCodeOffset = (coreId * taskpaddingNumEachCore) * CORE_PROC_CODE_GRAIN;
        } else {
            corePorcCodeSize = taskNumEachCore * CORE_PROC_CODE_GRAIN;
            coreProcCodeOffset =
                (paddingIdx * taskpaddingNumEachCore + (coreId - paddingIdx) * taskNumEachCore) * CORE_PROC_CODE_GRAIN;
        }
    } else {
        coreProcCodeOffset =
            (paddingIdx * taskpaddingNumEachCore + (coreId - paddingIdx) * taskNumEachCore) * CORE_PROC_CODE_GRAIN;
        corePorcCodeSize = codeSize - coreProcCodeOffset;
    }
}

template <typename DataType>
__aicore__ inline void DistanceFlatL2<DataType>::InitGlobalTensor(GM_ADDR inputQueries, GM_ADDR mask,
                                                                  GM_ADDR inputCodes, GM_ADDR inputL2PreNorm,
                                                                  GM_ADDR outputL2Dist, GM_ADDR outputL2DistMin,
                                                                  GM_ADDR outputflag)
{
    querysGlobal.SetGlobalBuffer((__gm__ DataType *)inputQueries);
    maskGlobal.SetGlobalBuffer((__gm__ uint8_t *)mask + (maskBlockOffset + coreProcCodeOffset) / Utils::MASK_BIT_NUM);
    codesGlobal.SetGlobalBuffer((__gm__ DataType *)inputCodes + coreProcCodeOffset * dimSize);
    codesPreNormGlobal.SetGlobalBuffer((__gm__ DataType *)inputL2PreNorm + coreProcCodeOffset);
    l2distGlobal.SetGlobalBuffer((__gm__ DataType *)outputL2Dist + coreProcCodeOffset);
    l2distMinGlobal.SetGlobalBuffer((__gm__ DataType *)outputL2DistMin + (coreProcCodeOffset / MIN_BATCH) * 2);
    flagGlobal.SetGlobalBuffer((__gm__ uint16_t *)outputflag + coreId * Utils::CUBE_ALIGN);

    GM_ADDR workSpace = GetSysWorkSpacePtr();
    GM_ADDR userWorkSpace = GetUserWorkspace(workSpace);
    if (userWorkSpace == nullptr) {
        return;
    }
    innerProductGloabal.SetGlobalBuffer((__gm__ UpGradeDataType *)userWorkSpace +
                                        coreId * querySizeEachLoop * CORE_PROC_CODE_GRAIN);
}

template <typename DataType>
__aicore__ inline void DistanceFlatL2<DataType>::InitTilingArgs()
{
    // query tiling
    querySizeEachLoop = pTilingDevice->querySizeEachLoop;
    queryLoopTimes = pTilingDevice->queryLoopTimes;
    querySizeLastLoop = pTilingDevice->querySizeLastLoop;  // 尾部数据
    queryAlign = AlignUp(querySizeEachLoop, Utils::CUBE_ALIGN);
    queryCopyOutOffset = querySizeEachLoop * CODE_SIZE;
    queryCopyOutBatchOffset = querySizeEachLoop * CODE_BATCH_SIZE;
    codeSizeEachLoop = pTilingDevice->codeSizeEachLoop;
    codeLoopTimes = Utils::DivUp(corePorcCodeSize, codeSizeEachLoop);
    codeSizeLastLoop = corePorcCodeSize - (codeLoopTimes - 1) * codeSizeEachLoop;  // 尾部数据
    codeBatchSize = AlignUp(CORE_PROC_CODE_GRAIN / MIN_BATCH, Utils::CUBE_ALIGN / 2);
    codeCopyOutBatchOffset = Utils::DivUp(codeSizeEachLoop, MIN_BATCH);

    // cube tiling
    cubeTilingIp = pTilingDevice->cubeTilingIp;
    cubeTilingL2Norm = pTilingDevice->cubeTilingL2Norm;

    // 每次循环计算时，mask的长度
    maskLenEachLoop = codeSizeEachLoop / Utils::MASK_BIT_NUM;
    // 每次循环计算时，select的重复次数
    uint32_t validResultVecRepeats = querySizeEachLoop * codeSizeEachLoop / Utils::VIC_HALF_FULL_MASK;
    selectLoopTime = validResultVecRepeats / Utils::SELECT_REPEAT_TIME;
    selectRemainder = static_cast<uint8_t>(validResultVecRepeats % Utils::SELECT_REPEAT_TIME);
}

template <typename DataType>
__aicore__ inline void DistanceFlatL2<DataType>::InitMemoryQueue()
{
    if (corePorcCodeSize != 0) {
        pipe.InitBuffer(queryDotQueryQue, BUFFER_NUM, queryAlign * queryAlign * sizeof(UpGradeDataType));
        pipe.InitBuffer(queryL2NormQue, BUFFER_NUM, queryAlign * sizeof(UpGradeDataType));
        pipe.InitBuffer(codeInQue, BUFFER_NUM, codeSizeEachLoop * dimSize * sizeof(DataType));
        pipe.InitBuffer(codeL2NormQue, BUFFER_NUM, codeSizeEachLoop * sizeof(DataType));
        pipe.InitBuffer(codeL2NormCastQue, BUFFER_NUM, codeSizeEachLoop * sizeof(UpGradeDataType));
        pipe.InitBuffer(innerProductQue, BUFFER_NUM, querySizeEachLoop * codeSizeEachLoop * sizeof(UpGradeDataType));
        pipe.InitBuffer(l2DistanceQue, BUFFER_NUM, querySizeEachLoop * codeSizeEachLoop * sizeof(DataType));
        pipe.InitBuffer(l2DistanceLocalQue, BUFFER_NUM, querySizeEachLoop * codeSizeEachLoop * sizeof(UpGradeDataType));
        pipe.InitBuffer(l2DistanceMinQue, BUFFER_NUM, codeBatchSize * querySizeEachLoop * 2 * sizeof(DataType));
        pipe.InitBuffer(maskQue, BUFFER_NUM, querySizeEachLoop * maskLenEachLoop * sizeof(uint8_t));
    } else {
        constexpr uint32_t defaultByteSize = BLOCK_BYTE_SIZE;
        pipe.InitBuffer(queryDotQueryQue, BUFFER_NUM, defaultByteSize);
        pipe.InitBuffer(queryL2NormQue, BUFFER_NUM, defaultByteSize);
        pipe.InitBuffer(codeInQue, BUFFER_NUM, defaultByteSize);
        pipe.InitBuffer(codeL2NormQue, BUFFER_NUM, defaultByteSize);
        pipe.InitBuffer(codeL2NormCastQue, BUFFER_NUM, defaultByteSize);
        pipe.InitBuffer(innerProductQue, BUFFER_NUM, defaultByteSize);
        pipe.InitBuffer(l2DistanceQue, BUFFER_NUM, defaultByteSize);
        pipe.InitBuffer(l2DistanceLocalQue, BUFFER_NUM, defaultByteSize);
        pipe.InitBuffer(l2DistanceMinQue, BUFFER_NUM, defaultByteSize);
        pipe.InitBuffer(maskQue, BUFFER_NUM, defaultByteSize);
    }
}

template <typename DataType>
__aicore__ inline void DistanceFlatL2<DataType>::Init(GM_ADDR inputQueries, GM_ADDR mask, GM_ADDR inputCodes,
                                                      GM_ADDR inputL2PreNorm, GM_ADDR inputActualSize,
                                                      GM_ADDR outputL2Dist, GM_ADDR outputL2DistMin,
                                                      GM_ADDR outputflag,
                                                      const DistanceFlatL2TilingData *__restrict tiling)
{
    pTilingDevice = tiling;
    InitShapeInfoFormGm(inputActualSize);
    GetMaskParameters(inputActualSize);
    InitCoreProcInfo();
    InitTilingArgs();
    InitGlobalTensor(inputQueries, mask, inputCodes, inputL2PreNorm, outputL2Dist, outputL2DistMin, outputflag);
    InitMemoryQueue();
}

template <typename DataType>
__aicore__ inline void DistanceFlatL2<DataType>::CopyInPreNorm(uint32_t codeLoopIdx)
{
    LocalTensor<DataType> codePreNorm = codeL2NormQue.AllocTensor<DataType>();
    auto copyOffset = codeLoopIdx * codeSizeEachLoop;
    DataCopy(codePreNorm, codesPreNormGlobal[copyOffset], codeSizeEachLoop);
    codeL2NormQue.EnQue<DataType>(codePreNorm);
}

template <typename DataType>
__aicore__ inline void DistanceFlatL2<DataType>::ComputeL2NormSum(uint32_t queryProcNum,
                                                                  LocalTensor<UpGradeDataType> &l2NormSumLocal,
                                                                  LocalTensor<UpGradeDataType> &queryL2NormLocal)
{
    auto codePreNorm = codeL2NormQue.DeQue<DataType>();  // y^2
    auto codePreNormLocal = codeL2NormCastQue.AllocTensor<UpGradeDataType>();

    Cast(codePreNormLocal, codePreNorm, RoundMode::CAST_NONE, codeSizeEachLoop);
    set_flag(PIPE_V, PIPE_S, EVENT_ID0);

    wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
    for (auto i = 0; i < queryProcNum; i++) {
        auto querySquare = queryL2NormLocal.GetValue(i);
        set_flag(PIPE_S, PIPE_V, EVENT_ID0);

        wait_flag(PIPE_S, PIPE_V, EVENT_ID0);
        Adds(l2NormSumLocal[i * codeSizeEachLoop], codePreNormLocal, querySquare, codeSizeEachLoop);
    }

    codeL2NormCastQue.FreeTensor(codePreNormLocal);
    codeL2NormQue.FreeTensor(codePreNorm);
}

template <typename DataType>
__aicore__ inline void DistanceFlatL2<DataType>::ComputeInnerProduct(uint32_t codeLoopIdx, bool isTail)
{
    if (isTail) {
        matmulObjIp.SetTail(querySizeLastLoop, codeSizeEachLoop, dimSize);
    }

    auto codeOffset = codeLoopIdx * codeSizeEachLoop * dimSize;
    matmulObjIp.SetTensorB(codesGlobal[codeOffset], true);
    // cube core start matmul
    matmulObjIp.template IterateAll<false>(innerProductGloabal, 0, false, true);
}

template <typename DataType>
__aicore__ inline void DistanceFlatL2<DataType>::ComputeL2Distance(uint32_t queryProcNum,
                                                                   LocalTensor<DataType> &l2DistanceTensor,
                                                                   LocalTensor<UpGradeDataType> &l2DistanceLocal)
{
    // wait cube matmul caculate over
    matmulObjIp.WaitIterateAll();

    auto innerProductLocal = innerProductQue.AllocTensor<UpGradeDataType>();
    DataCopy(innerProductLocal, innerProductGloabal, queryProcNum * codeSizeEachLoop);

    set_flag(PIPE_MTE2, PIPE_S, EVENT_ID6);
    wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID6);

    UpGradeDataType neg2 = -2.0;
    Axpy(l2DistanceLocal, innerProductLocal, neg2, queryProcNum * codeSizeEachLoop);
    Cast(l2DistanceTensor, l2DistanceLocal, RoundMode::CAST_NONE, queryProcNum * codeSizeEachLoop);

    innerProductQue.FreeTensor(innerProductLocal);
}

template <typename DataType>
__aicore__ inline void DistanceFlatL2<DataType>::ComputeL2DistanceMin(uint32_t queryProcNum, uint32_t codeProcNum,
                                                                      LocalTensor<DataType> &l2DistanceLocal,
                                                                      LocalTensor<DataType> &l2DistanceLocalMin)
{
    DataType zero = 0;
    Duplicate(l2DistanceLocalMin, zero, codeBatchSize * queryProcNum * 2);
    // 可以被MIN_BATCH整除部分
    constexpr auto srcRepStride = MIN_BATCH * sizeof(DataType) / BLOCK_BYTE_SIZE;
    if (codeProcNum / MIN_BATCH != 0) {
        for (auto i = 0; i < queryProcNum; i++) {
            WholeReduceMin(l2DistanceLocalMin[i * codeBatchSize * 2], l2DistanceLocal[i * codeSizeEachLoop], MIN_BATCH,
                           codeProcNum / MIN_BATCH, 1, 1, srcRepStride);
        }
    }

    pipe_barrier(PIPE_V);

    // 不可以被MIN_BATCH整除部分
    if (codeProcNum % MIN_BATCH != 0) {
        for (auto i = 0; i < queryProcNum; i++) {
            WholeReduceMin(l2DistanceLocalMin[(i * codeBatchSize + codeProcNum / MIN_BATCH) * 2],
                           l2DistanceLocal[i * codeSizeEachLoop + codeProcNum / MIN_BATCH * MIN_BATCH],
                           codeProcNum % MIN_BATCH, 1, 1, 1, srcRepStride);
        }
    }
}

template <typename DataType>
__aicore__ inline void DistanceFlatL2<DataType>::CopyOutL2Dist(uint32_t queryLoopIdx, uint32_t codeLoopIdx,
                                                               uint32_t queryProcNum,
                                                               LocalTensor<DataType> &l2DistanceLocal)
{
    auto copyOffset = queryLoopIdx * queryCopyOutOffset + codeLoopIdx * codeSizeEachLoop;

    DataCopyParams copyParams;
    copyParams.blockCount = queryProcNum;
    copyParams.blockLen = codeSizeEachLoop * sizeof(DataType) / BLOCK_BYTE_SIZE;
    copyParams.dstStride = (CODE_SIZE - codeSizeEachLoop) * sizeof(DataType) / BLOCK_BYTE_SIZE;
    DataCopy(l2distGlobal[copyOffset], l2DistanceLocal, copyParams);
}

template <typename DataType>
__aicore__ inline void DistanceFlatL2<DataType>::CopyOutL2DistMin(uint32_t queryLoopIdx, uint32_t codeLoopIdx,
                                                                  uint32_t queryProcNum,
                                                                  LocalTensor<DataType> &l2DistanceLocalMin)
{
    auto copyOffset = (queryLoopIdx * queryCopyOutBatchOffset + codeLoopIdx * codeCopyOutBatchOffset) * 2;

    DataCopyParams copyParams;
    copyParams.blockCount = queryProcNum;
    copyParams.blockLen = codeBatchSize * 2 * sizeof(DataType) / BLOCK_BYTE_SIZE;
    copyParams.dstStride = (CODE_BATCH_SIZE - codeBatchSize) * 2 * sizeof(DataType) / BLOCK_BYTE_SIZE;
    SetAtomicAdd<DataType>();
    DataCopy(l2distMinGlobal[copyOffset], l2DistanceLocalMin, copyParams);
    SetAtomicNone();
}

template <typename DataType>
__aicore__ inline void DistanceFlatL2<DataType>::SetFlagReady()
{
    // wait for distance copy over
    pipe_barrier(PIPE_MTE3);

    ALLOC_LOCAL_TENSOR(flagLocal, Utils::CUBE_ALIGN, uint16_t, VECOUT);
    uint16_t zero = 0;
    Duplicate(flagLocal, zero, Utils::CUBE_ALIGN);

    uint16_t readyFlag = 1;
    flagLocal.SetValue(0, readyFlag);
    set_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
    DataCopy(flagGlobal, flagLocal, Utils::CUBE_ALIGN);
}

template <typename DataType>
__aicore__ inline void DistanceFlatL2<DataType>::DoMask(uint32_t queryLoopIdx,
                                                        uint32_t codeLoopIdx,
                                                        uint32_t queryProcNum,
                                                        uint32_t codeProcNum,
                                                        LocalTensor<half> &l2DistanceLocal)
{
    if (maskFlag == 0) {
        return;
    }
    auto maskLocal = maskQue.AllocTensor<uint8_t>();
    // 本次算子已经计算好的query的对应的mask偏移:queryLoopIdx * querySizeEachLoop * maskLen
    // 本次算子已经计算好的code的对应的mask偏移:codeLoopIdx * codeSizeEachLoop / Utils::MASK_BIT_NUM
    uint64_t maskOffset = queryLoopIdx * querySizeEachLoop * static_cast<uint64_t>(maskLen) +
        codeLoopIdx * codeSizeEachLoop / Utils::MASK_BIT_NUM;

    for (uint32_t i = 0; i < queryProcNum; i++) {
        DataCopy(maskLocal[i * maskLenEachLoop], maskGlobal[i * static_cast<uint64_t>(maskLen) + maskOffset],
            maskLenEachLoop);
    }
    maskQue.EnQue(maskLocal);
    maskLocal = maskQue.DeQue<uint8_t>();
    // dstBlkStride,src0BlkStride,src1BlkStride 1
    // dstRepStride,src0RepStride,src1RepStride 8
    BinaryRepeatParams param {
        1,
        1,
        1,
        8,
        8,
        8
    };

    const uint32_t distOffset = Utils::SELECT_REPEAT_TIME * Utils::VIC_HALF_FULL_MASK;
    const uint32_t maskRepateOffset = distOffset / Utils::MASK_BIT_NUM;
    for (uint32_t i = 0; i < selectLoopTime; i++) {
        Select(l2DistanceLocal[i * distOffset], maskLocal[i * maskRepateOffset], l2DistanceLocal[i * distOffset],
            Utils::HALF_MAX, SELMODE::VSEL_TENSOR_SCALAR_MODE, Utils::VIC_HALF_FULL_MASK, Utils::SELECT_REPEAT_TIME,
            param);
    }
    if (selectRemainder != 0) {
        Select(l2DistanceLocal[selectLoopTime * distOffset], maskLocal[selectLoopTime * maskRepateOffset],
            l2DistanceLocal[selectLoopTime * distOffset], Utils::HALF_MAX, SELMODE::VSEL_TENSOR_SCALAR_MODE,
            Utils::VIC_HALF_FULL_MASK, selectRemainder, param);
    }

    maskQue.FreeTensor(maskLocal);
}

template <typename DataType>
__aicore__ inline void DistanceFlatL2<DataType>::Compute(uint32_t queryLoopIdx, uint32_t codeLoopIdx,
                                                         uint32_t queryProcNum, uint32_t codeProcNum,
                                                         LocalTensor<UpGradeDataType> &queryL2NormLocal)
{
    auto l2DistanceTensor = l2DistanceQue.AllocTensor<DataType>();
    {
        {
            auto l2DistanceLocal = l2DistanceLocalQue.AllocTensor<UpGradeDataType>();
            // 160us
            ComputeL2NormSum(queryProcNum, l2DistanceLocal, queryL2NormLocal);
            // ~=10us
            ComputeL2Distance(queryProcNum, l2DistanceTensor, l2DistanceLocal);
            l2DistanceLocalQue.FreeTensor(l2DistanceLocal);
        };
    }

    DoMask(queryLoopIdx, codeLoopIdx, queryProcNum, codeProcNum, l2DistanceTensor);

    auto l2DistanceMinTensor = l2DistanceMinQue.AllocTensor<DataType>();
    // ~=40us
    ComputeL2DistanceMin(queryProcNum, codeProcNum, l2DistanceTensor, l2DistanceMinTensor);

    // wait for compute l2distance over
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);

    // ~=20us
    CopyOutL2Dist(queryLoopIdx, codeLoopIdx, queryProcNum, l2DistanceTensor);
    CopyOutL2DistMin(queryLoopIdx, codeLoopIdx, queryProcNum, l2DistanceMinTensor);

    l2DistanceQue.FreeTensor(l2DistanceTensor);
    l2DistanceMinQue.FreeTensor(l2DistanceMinTensor);

    // wait for copy l2distance to gm over
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID2);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID2);
}

template <typename DataType>
__aicore__ inline void DistanceFlatL2<DataType>::ComputeQueryL2Norm(uint32_t queryOffset, uint32_t queryProcNum,
                                                                    LocalTensor<UpGradeDataType> &queryL2NormLocal,
                                                                    bool isTail)
{
    auto queryDotQueryLocal = queryDotQueryQue.AllocTensor<UpGradeDataType>();
    if (isTail) {
        matmulObjL2Norm.SetTail(querySizeLastLoop, querySizeLastLoop, dimSize);
    }

    matmulObjL2Norm.SetTensorA(querysGlobal[queryOffset]);
    matmulObjL2Norm.SetTensorB(querysGlobal[queryOffset], true);
    matmulObjL2Norm.IterateAll(queryDotQueryLocal);
    queryDotQueryQue.EnQue<UpGradeDataType>(queryDotQueryLocal);

    {
        auto queryDotQueryLocal = queryDotQueryQue.DeQue<UpGradeDataType>();
        set_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);

        wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
        for (auto i = 0; i < queryProcNum; i++) {
            auto querySquareSum = queryDotQueryLocal.GetValue(i * querySizeEachLoop + i);
            queryL2NormLocal.SetValue(i, querySquareSum);
        }
        queryDotQueryQue.FreeTensor(queryDotQueryLocal);
    }
}

template <typename DataType>
__aicore__ inline void DistanceFlatL2<DataType>::processEachCodeLoop(uint32_t queryLoopIdx, uint32_t codeLoopIdx,
                                                                     uint32_t queryProcNum, bool isQueryTail,
                                                                     LocalTensor<UpGradeDataType> &queryL2NormLocal)
{
    auto codeProcNum = (codeLoopIdx != (codeLoopTimes - 1)) ? codeSizeEachLoop : codeSizeLastLoop;
    ComputeInnerProduct(codeLoopIdx, isQueryTail);
    CopyInPreNorm(codeLoopIdx);
    Compute(queryLoopIdx, codeLoopIdx, queryProcNum, codeProcNum, queryL2NormLocal);
}

template <typename DataType>
__aicore__ inline void DistanceFlatL2<DataType>::processEachQueryLoop(uint32_t queryLoopIdx)
{
    bool isQueryTail = (queryLoopIdx == (queryLoopTimes - 1));
    auto queryProcNum = isQueryTail ? querySizeLastLoop : querySizeEachLoop;
    auto queryOffset = queryLoopIdx * querySizeEachLoop * dimSize;

    auto queryL2NormLocal = queryL2NormQue.AllocTensor<UpGradeDataType>();
    ComputeQueryL2Norm(queryOffset, queryProcNum, queryL2NormLocal, isQueryTail);

    matmulObjIp.SetTensorA(querysGlobal[queryOffset]);
    for (auto codeLoopIdx = 0; codeLoopIdx < codeLoopTimes; codeLoopIdx++) {
        processEachCodeLoop(queryLoopIdx, codeLoopIdx, queryProcNum, isQueryTail, queryL2NormLocal);
    }

    queryL2NormQue.FreeTensor(queryL2NormLocal);
}

template <typename DataType>
__aicore__ inline void DistanceFlatL2<DataType>::process()
{
    REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), matmulObjIp, matmulObjL2Norm);
    if (corePorcCodeSize != 0) {
        matmulObjIp.Init(&cubeTilingIp);
        matmulObjL2Norm.Init(&cubeTilingL2Norm);

        for (auto queryLoopIdx = 0; queryLoopIdx < queryLoopTimes; queryLoopIdx++) {
            processEachQueryLoop(queryLoopIdx);
        }

        matmulObjIp.End();
        matmulObjL2Norm.End();
    }
    SetFlagReady();
}
}

extern "C" __global__ __aicore__ void distance_flat_l2(GM_ADDR queries, GM_ADDR mask, GM_ADDR codes, GM_ADDR l2PreNorm,
                                                       GM_ADDR actualNum, GM_ADDR l2Distance, GM_ADDR l2DistanceMin,
                                                       GM_ADDR flag, GM_ADDR workspace, GM_ADDR tiling)
{
    if (workspace == nullptr) {
        return;
    }

    GET_TILING_DATA(tiling_data, tiling);
    const DistanceFlatL2TilingData *__restrict pTilingDevice = &tiling_data;

    IndexOps::DistanceFlatL2<half> op;
    if ASCEND_IS_AIV {
        op.Init(queries, mask, codes, l2PreNorm, actualNum, l2Distance, l2DistanceMin, flag, pTilingDevice);
    }
    op.process();
}