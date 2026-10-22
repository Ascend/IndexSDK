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

#include <algorithm>
#include <cstdio>
#include <type_traits>

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "op_kernel_common.h"

using namespace AscendC;
using namespace matmul;

namespace
{
constexpr uint32_t BUFFER_NUM = 1;
constexpr uint32_t BLOCK_BYTE_SIZE = 32;
constexpr uint32_t MIN_BATCH = 64;
// 每个 burst(64条code)在极值输出里占 2 个元素：[最小值, burst内argmin]
constexpr uint32_t BURST_BLOCK_RATIO = 2;
// 每核处理的code粒度，必须为 MIN_BATCH(64) 的整数倍。
// 【跨文件强约束】必须与 op_host/distance_flat_l2_with950.cpp 的 CODE_PROC_PER_LOOP_MAX 严格相等(均为512)：
// 一轮产出 CORE_PROC_CODE_GRAIN/MIN_BATCH 个burst(极值pair)，其字节数必须 == 该轮目的地址步进，
// 才能消除相邻codeLoop之间的GM地址重叠与行末越界补零。即
//   codeBatchSize(=AlignUp(CORE_PROC_CODE_GRAIN/MIN_BATCH, CUBE_ALIGN/2))
// 必须等于 codeCopyOutBatchOffset(=DivUp(codeSizeEachLoop, MIN_BATCH))，
// 该等式仅在两个常量同为512时成立(均为8)。单独调整任一侧会静默破坏 CopyOutL2DistMin 的
// 输出布局与 host 侧 workspace 大小(queryAlignAll*CODE_PROC_PER_LOOP_MAX)，必须两侧同步修改。
constexpr uint32_t CORE_PROC_CODE_GRAIN = 512;
}  // namespace

namespace IndexOps
{
// 后续如果DistFlatL2Kernel 需要支持更多类型 可以参考如下通过特化新的类型来完成
template <typename T>
struct TraitsUpgradeType
{
    using type = T;
};

template <>
struct TraitsUpgradeType<half>
{
    using type = float;
};

template <typename DataType>
class DistanceFlatL2With950
{
   public:
    // L2计算过程中涉及到精度抬升 这个表示抬升得精度类型
    using UpGradeDataType = typename TraitsUpgradeType<DataType>::type;

    __aicore__ inline DistanceFlatL2With950() = default;
    __aicore__ inline void Init(GM_ADDR inputQueries, GM_ADDR mask, GM_ADDR inputCodes, GM_ADDR inputL2PreNorm,
                                GM_ADDR inputActualSize, GM_ADDR outputL2Dist, GM_ADDR outputL2DistMin,
                                GM_ADDR outputflag, const DistanceFlatL2With950TilingData *__restrict tiling,
                                GM_ADDR usrWorkspace);

    __aicore__ inline void process();

    // flag写入，从kernel入口调用
    __aicore__ inline void SetFlagReady();

   private:
    // code 预归一化(y^2) 每 codeLoop 只加载/抬升一次(gm -> ub)，供该轮所有 query 分块复用。
    __aicore__ inline void LoadCodePreNorm(uint32_t codeLoopIdx, LocalTensor<UpGradeDataType> &codePreNormLocal);

    __aicore__ inline void Compute(uint32_t queryLoopIdx, uint32_t codeLoopIdx, uint32_t queryProcNum,
                                   uint32_t codeProcNum, uint32_t normOffset,
                                   LocalTensor<UpGradeDataType> &queryL2NormAllLocal,
                                   LocalTensor<UpGradeDataType> &codePreNormLocal);

    // 预计算一批 query 的 L2 范数，写入 queryL2NormAllLocal[normOffset .. normOffset+queryProcNum)
    __aicore__ inline void ComputeQueryL2Norm(uint32_t queryOffset, uint32_t queryProcNum, uint32_t normOffset,
                                              LocalTensor<UpGradeDataType> &queryL2NormAllLocal, bool isTail);

    // compute part1:query_local^2 + code_l2_local^2 (vector)
    __aicore__ inline void ComputeL2NormSum(uint32_t queryProcNum, uint32_t normOffset,
                                            LocalTensor<UpGradeDataType> &l2NormSumLocal,
                                            LocalTensor<UpGradeDataType> &queryL2NormAllLocal,
                                            LocalTensor<UpGradeDataType> &codePreNormLocal);

    // compute part2:x * y
    __aicore__ inline void ComputeInnerProduct(uint32_t codeLoopIdx);

    // compute l2Distance:part1 - 2*part2
    __aicore__ inline void ComputeL2Distance(uint32_t queryLoopIdx, uint32_t queryProcNum,
                                             LocalTensor<DataType> &l2DistanceTensor,
                                             LocalTensor<UpGradeDataType> &l2DistanceLocal);

    // compute l2DistanceMin
    __aicore__ inline void ComputeL2DistanceMin(uint32_t queryProcNum, uint32_t codeProcNum,
                                                LocalTensor<DataType> &l2DistanceLocal,
                                                LocalTensor<DataType> &l2DistanceLocalMin);

    // l2_distance_local(ub) -> l2_distance(gm)
    __aicore__ inline void CopyOutL2Dist(uint32_t queryLoopIdx, uint32_t codeLoopIdx, uint32_t queryProcNum,
                                         uint32_t codeProcNum, LocalTensor<DataType> &l2DistanceLocal);

    __aicore__ inline void CopyOutL2DistMin(uint32_t queryLoopIdx, uint32_t codeLoopIdx, uint32_t queryProcNum,
                                            LocalTensor<DataType> &l2DistanceLocalMin);

    __aicore__ inline void GetMaskParameters(GM_ADDR inputActualSize);

    __aicore__ inline void InitShapeInfoFormGm(GM_ADDR inputActualSize);

    __aicore__ inline void InitCoreProcInfo();

    __aicore__ inline void InitGlobalTensor(GM_ADDR inputQueries, GM_ADDR mask, GM_ADDR inputCodes,
                                            GM_ADDR inputL2PreNorm, GM_ADDR outputL2Dist, GM_ADDR outputL2DistMin,
                                            GM_ADDR outputflag, GM_ADDR usrWorkspace);

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
    using MatMulTypeResult = MatmulType<TPosition::GM, CubeFormat::ND, UpGradeDataType>;
    Matmul<MatMulTypeQueryA, MatMulTypeQueryB, MatMulTypeResult> matmulObjL2Norm;

    GlobalTensor<DataType> querysGlobal;
    GlobalTensor<uint8_t> maskGlobal;
    GlobalTensor<DataType> codesGlobal;
    GlobalTensor<DataType> codesPreNormGlobal;
    GlobalTensor<DataType> l2distGlobal;
    GlobalTensor<DataType> l2distMinGlobal;
    GlobalTensor<uint16_t> flagGlobal;
    GlobalTensor<UpGradeDataType> innerProductGloabal;
    GlobalTensor<UpGradeDataType> queryDotQueryGlobal;

    TPipe pipe;
    TQue<QuePosition::VECOUT, BUFFER_NUM> queryL2NormQue;
    TQue<QuePosition::VECIN, BUFFER_NUM> queryDotQueryQue;
    TSCM<QuePosition::GM, BUFFER_NUM> codeInQue;
    TQue<QuePosition::VECIN, BUFFER_NUM> codeL2NormQue;
    TQue<QuePosition::VECIN, BUFFER_NUM> codeL2NormCastQue;
    TQue<QuePosition::VECIN, BUFFER_NUM> innerProductQue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> l2DistanceQue;
    TQue<QuePosition::VECIN, BUFFER_NUM> l2DistanceLocalQue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> l2DistanceMinQue;
    TQue<QuePosition::VECIN, BUFFER_NUM> maskQue;
    // 950 归约符号还原掩码(int16=0x7FFF)：清除 pair 索引槽取负后残留的符号位
    TBuf<QuePosition::VECCALC> signMaskBuf;

    // dims range:32 64 128 256 384 512 1024
    // query range:48 36 32 30 24 18 16 12 8 6 4 2 1
    // code range: (0, 16384]
    // shape info
    uint32_t codeSize{0};   // 本block实际参与计算的code数(来自ActualNum)
    uint32_t blockSize{0};  // 整块block的code数，作为输出行距基准(来自tiling codeSize)
    uint32_t querySize{0};
    uint32_t dimSize{0};

    // core info
    uint32_t usedCoreNum{0};
    uint32_t coreId{0};
    uint32_t coreNum{0};
    uint32_t corePorcCodeSize{0};
    uint32_t coreProcCodeOffset{0};

    // tiling args
    uint32_t querySizeEachLoop{0};
    uint32_t querySizeLastLoop{0};
    uint32_t queryAlign{0};     // 后处理分块的 M 对齐尺寸(=AlignUp(querySizeEachLoop,16))
    uint32_t queryAlignAll{0};  // 全量 query 的 M 对齐尺寸(=AlignUp(querySize,16))，用于 Ip matmul
    uint32_t queryLoopTimes{0};
    uint32_t queryCopyOutOffset{0};
    uint32_t queryCopyOutBatchOffset{0};
    uint32_t codeSizeEachLoop{0};
    uint32_t codeSizeLastLoop{0};
    uint32_t codeLoopTimes{0};
    uint32_t codeBatchSize{0};
    uint32_t codeBatchSizeTotal{0};
    uint32_t codeCopyOutBatchOffset{0};

    uint32_t maskLenEachLoop{0};
    uint32_t maskBlockOffset{0};
    uint32_t maskLen{0};
    uint32_t maskFlag{0};
    uint32_t selectLoopTime{0};
    uint8_t selectRemainder{0};

    TCubeTiling cubeTilingIp;
    TCubeTiling cubeTilingL2Norm;
    GM_ADDR flagBaseAddr{nullptr};

    const DistanceFlatL2With950TilingData *__restrict pTilingDevice{nullptr};
};

template <typename DataType>
__aicore__ inline void DistanceFlatL2With950<DataType>::GetMaskParameters(GM_ADDR inputActualSize)
{
    maskBlockOffset = *(reinterpret_cast<__gm__ uint32_t *>(inputActualSize) + Utils::MASK_BLOCK_OFFSET_IDX);
    maskLen = *(reinterpret_cast<__gm__ uint32_t *>(inputActualSize) + Utils::MASK_LEN_IDX);
    maskFlag = *(reinterpret_cast<__gm__ uint32_t *>(inputActualSize) + Utils::MASK_FLAG_IDX);
}

template <typename DataType>
__aicore__ inline void DistanceFlatL2With950<DataType>::InitShapeInfoFormGm(GM_ADDR inputActualSize)
{
    // 使用直接GM指针解引用(MTE3标量路径)读取codeSize，避免MIX模式下
    // AIC核Vector路径(VECIN/DataCopy)读取GM数据异常的bug
    codeSize = *(reinterpret_cast<__gm__ uint32_t *>(inputActualSize));
    blockSize = pTilingDevice->codeSize;
    querySize = pTilingDevice->querySize;
    dimSize = pTilingDevice->dimSize;
}

template <typename DataType>
__aicore__ inline void DistanceFlatL2With950<DataType>::InitCoreProcInfo()
{
    // MIX 1:2：GetBlockIdx()在AIV上为0..2*aicNum-1，同组的两个AIV映射到同一AIC。
    coreNum = pTilingDevice->aicNum;
    if ASCEND_IS_AIV
    {
        coreId = GetBlockIdx() / 2;
    }
    else
    {
        coreId = GetBlockIdx();
    }
    uint32_t taskNum = Utils::DivUp(codeSize, CORE_PROC_CODE_GRAIN);
    usedCoreNum = Utils::Min(taskNum, coreNum);

    uint32_t taskNumEachCore = taskNum / usedCoreNum;
    uint32_t taskpaddingNumEachCore = taskNumEachCore + 1;
    uint32_t paddingIdx = taskNum % usedCoreNum;

    if (coreId >= usedCoreNum)
    {
        corePorcCodeSize = 0;
    }
    else if (coreId != (usedCoreNum - 1))
    {
        if (coreId < paddingIdx)
        {
            corePorcCodeSize = taskpaddingNumEachCore * CORE_PROC_CODE_GRAIN;
            coreProcCodeOffset = (coreId * taskpaddingNumEachCore) * CORE_PROC_CODE_GRAIN;
        }
        else
        {
            corePorcCodeSize = taskNumEachCore * CORE_PROC_CODE_GRAIN;
            coreProcCodeOffset =
                (paddingIdx * taskpaddingNumEachCore + (coreId - paddingIdx) * taskNumEachCore) * CORE_PROC_CODE_GRAIN;
        }
    }
    else
    {
        coreProcCodeOffset =
            (paddingIdx * taskpaddingNumEachCore + (coreId - paddingIdx) * taskNumEachCore) * CORE_PROC_CODE_GRAIN;
        corePorcCodeSize = codeSize - coreProcCodeOffset;
    }
}

template <typename DataType>
__aicore__ inline void DistanceFlatL2With950<DataType>::InitGlobalTensor(GM_ADDR inputQueries, GM_ADDR mask,
                                                                         GM_ADDR inputCodes, GM_ADDR inputL2PreNorm,
                                                                         GM_ADDR outputL2Dist, GM_ADDR outputL2DistMin,
                                                                         GM_ADDR outputflag, GM_ADDR usrWorkspace)
{
    querysGlobal.SetGlobalBuffer((__gm__ DataType *)inputQueries);
    maskGlobal.SetGlobalBuffer((__gm__ uint8_t *)mask + (maskBlockOffset + coreProcCodeOffset) / Utils::MASK_BIT_NUM);
    codesGlobal.SetGlobalBuffer((__gm__ DataType *)inputCodes + coreProcCodeOffset * dimSize);
    codesPreNormGlobal.SetGlobalBuffer((__gm__ DataType *)inputL2PreNorm + coreProcCodeOffset);
    l2distGlobal.SetGlobalBuffer((__gm__ DataType *)outputL2Dist + coreProcCodeOffset);
    l2distMinGlobal.SetGlobalBuffer((__gm__ DataType *)outputL2DistMin + (coreProcCodeOffset / MIN_BATCH) * 2);
    flagBaseAddr = outputflag;
    // AIV block 0统一写全部flag，其余核不写。
    flagGlobal.SetGlobalBuffer((__gm__ uint16_t *)outputflag, pTilingDevice->aivNum * Utils::CUBE_ALIGN);

    // 直接从kernel入口的workspace参数获取用户workspace地址。
    // MIX 1:2模式下，同cube组（AIC + 2个AIV）必须共享同一块workspace，
    // 因为matmul由AIV发起、AIC执行Cube计算，C张量输出地址必须一致。
    uint32_t workIdx = 2 * coreId;

    // workspace分配保持一致；query 范数中间结果仍按后处理分块(queryAlign)分配。
    uint32_t ipElemCnt = queryAlignAll * codeSizeEachLoop;
    uint32_t normElemCnt = queryAlign * queryAlign;
    auto userWork = (__gm__ UpGradeDataType *)usrWorkspace + workIdx * (ipElemCnt + normElemCnt);
    innerProductGloabal.SetGlobalBuffer(userWork, ipElemCnt);
    queryDotQueryGlobal.SetGlobalBuffer(userWork + ipElemCnt, normElemCnt);
}

template <typename DataType>
__aicore__ inline void DistanceFlatL2With950<DataType>::InitTilingArgs()
{
    // query tiling
    querySizeEachLoop = pTilingDevice->querySizeEachLoop;
    queryLoopTimes = pTilingDevice->queryLoopTimes;
    querySizeLastLoop = pTilingDevice->querySizeLastLoop;  // 尾部数据
    queryAlign = AlignUp(querySizeEachLoop, Utils::CUBE_ALIGN);
    // matmul 的 M = 全量 query 的对齐尺寸，与 host 侧 cubeTilingIp.SetShape 一致
    queryAlignAll = AlignUp(querySize, Utils::CUBE_ALIGN);
    codeSizeEachLoop = pTilingDevice->codeSizeEachLoop;
    codeLoopTimes = Utils::DivUp(corePorcCodeSize, codeSizeEachLoop);
    codeSizeLastLoop = corePorcCodeSize - (codeLoopTimes - 1) * codeSizeEachLoop;  // 尾部数据
    codeBatchSize = AlignUp(CORE_PROC_CODE_GRAIN / MIN_BATCH, Utils::CUBE_ALIGN / 2);
    codeCopyOutBatchOffset = Utils::DivUp(codeSizeEachLoop, MIN_BATCH);
    codeBatchSizeTotal = (blockSize + MIN_BATCH - 1) / MIN_BATCH;  // 整块code可分的batch数(输出行距基准)
    queryCopyOutOffset = querySizeEachLoop * blockSize;  // 每轮query在输出中的行偏移(按整块blockSize)
    queryCopyOutBatchOffset = querySizeEachLoop * codeBatchSizeTotal;  // 每轮query在极值输出中的行偏移

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
__aicore__ inline void DistanceFlatL2With950<DataType>::InitMemoryQueue()
{
    // MIX 1:2模式：所有核(AIC+AIV)的pipe buffer尺寸必须一致，否则matmul握手失败

    pipe.InitBuffer(queryDotQueryQue, BUFFER_NUM, queryAlign * queryAlign * sizeof(UpGradeDataType));
    // query 范数缓冲覆盖全部 query(queryAlignAll)，一次分配后分块写入
    pipe.InitBuffer(queryL2NormQue, BUFFER_NUM, queryAlignAll * sizeof(UpGradeDataType));
    // codeInQue 全文件未使用，却占 codeSizeEachLoop*dimSize*sizeof(DataType)=64KB UB，故不再分配。
    pipe.InitBuffer(codeL2NormQue, BUFFER_NUM, codeSizeEachLoop * sizeof(DataType));
    pipe.InitBuffer(codeL2NormCastQue, BUFFER_NUM, codeSizeEachLoop * sizeof(UpGradeDataType));
    pipe.InitBuffer(innerProductQue, BUFFER_NUM, querySizeEachLoop * codeSizeEachLoop * sizeof(UpGradeDataType));
    pipe.InitBuffer(l2DistanceQue, BUFFER_NUM, querySizeEachLoop * codeSizeEachLoop * sizeof(DataType));
    pipe.InitBuffer(l2DistanceLocalQue, BUFFER_NUM, querySizeEachLoop * codeSizeEachLoop * sizeof(UpGradeDataType));
    pipe.InitBuffer(l2DistanceMinQue, BUFFER_NUM, codeBatchSize * querySizeEachLoop * 2 * sizeof(DataType));
    pipe.InitBuffer(maskQue, BUFFER_NUM, querySizeEachLoop * maskLenEachLoop * sizeof(uint8_t));
    // 符号还原掩码：所有核(AIC+AIV)同样分配，避免MIX 1:2握手不一致
    pipe.InitBuffer(signMaskBuf, querySizeEachLoop * codeBatchSize * BURST_BLOCK_RATIO * sizeof(int16_t));
    if ASCEND_IS_AIV
    {
        LocalTensor<int16_t> signMask = signMaskBuf.Get<int16_t>(querySizeEachLoop * codeBatchSize * BURST_BLOCK_RATIO);
        Duplicate(signMask, static_cast<int16_t>(0x7FFF), querySizeEachLoop * codeBatchSize * BURST_BLOCK_RATIO);
    }
}

template <typename DataType>
__aicore__ inline void DistanceFlatL2With950<DataType>::Init(GM_ADDR inputQueries, GM_ADDR mask, GM_ADDR inputCodes,
                                                             GM_ADDR inputL2PreNorm, GM_ADDR inputActualSize,
                                                             GM_ADDR outputL2Dist, GM_ADDR outputL2DistMin,
                                                             GM_ADDR outputflag,
                                                             const DistanceFlatL2With950TilingData *__restrict tiling,
                                                             GM_ADDR usrWorkspace)
{
    pTilingDevice = tiling;
    InitShapeInfoFormGm(inputActualSize);
    GetMaskParameters(inputActualSize);
    InitCoreProcInfo();
    InitTilingArgs();
    InitGlobalTensor(inputQueries, mask, inputCodes, inputL2PreNorm, outputL2Dist, outputL2DistMin, outputflag,
                     usrWorkspace);
}

template <typename DataType>
__aicore__ inline void DistanceFlatL2With950<DataType>::LoadCodePreNorm(uint32_t codeLoopIdx,
                                                                        LocalTensor<UpGradeDataType> &codePreNormLocal)
{
    auto copyOffset = codeLoopIdx * codeSizeEachLoop;
    LocalTensor<DataType> codePreNorm = codeL2NormQue.AllocTensor<DataType>();
    DataCopy(codePreNorm, codesPreNormGlobal[copyOffset], codeSizeEachLoop);
    codeL2NormQue.EnQue<DataType>(codePreNorm);
    codePreNorm = codeL2NormQue.DeQue<DataType>();

    codePreNormLocal = codeL2NormCastQue.AllocTensor<UpGradeDataType>();
    Cast(codePreNormLocal, codePreNorm, RoundMode::CAST_NONE, codeSizeEachLoop);
    codeL2NormQue.FreeTensor(codePreNorm);
}

template <typename DataType>
__aicore__ inline void DistanceFlatL2With950<DataType>::ComputeL2NormSum(
    uint32_t queryProcNum, uint32_t normOffset, LocalTensor<UpGradeDataType> &l2NormSumLocal,
    LocalTensor<UpGradeDataType> &queryL2NormAllLocal, LocalTensor<UpGradeDataType> &codePreNormLocal)
{
    // Ascend950: Scalar/Vector share PIPE_S, no inter-pipe flag needed.
    for (auto i = 0; i < queryProcNum; i++)
    {
        auto querySquare = queryL2NormAllLocal.GetValue(normOffset + i);
        Adds(l2NormSumLocal[i * codeSizeEachLoop], codePreNormLocal, querySquare, codeSizeEachLoop);
    }
}

template <typename DataType>
__aicore__ inline void DistanceFlatL2With950<DataType>::ComputeInnerProduct(uint32_t codeLoopIdx)
{
    // M = 全量 query：每轮只发一次 matmul，code 矩阵从 HBM 仅读一遍。
    // SetTail 按真实 query 行数校正(非仅尾部)，使 matmul tiling 的 M(已按 CUBE_ALIGN 补齐)被校正回真实值。
    matmulObjIp.SetTail(querySize, codeSizeEachLoop, dimSize);

    auto codeOffset = codeLoopIdx * codeSizeEachLoop * dimSize;
    matmulObjIp.SetTensorB(codesGlobal[codeOffset], true);
    // 默认MIX模式由AIV发起Matmul请求，AIC在高阶API内部执行Cube计算。
    matmulObjIp.template IterateAll<false>(innerProductGloabal, 0, false, true);

    matmulObjIp.WaitIterateAll();
    matmulObjIp.End();
}

template <typename DataType>
__aicore__ inline void DistanceFlatL2With950<DataType>::ComputeL2Distance(uint32_t queryLoopIdx, uint32_t queryProcNum,
                                                                          LocalTensor<DataType> &l2DistanceTensor,
                                                                          LocalTensor<UpGradeDataType> &l2DistanceLocal)
{
    auto innerProductLocal = innerProductQue.AllocTensor<UpGradeDataType>();
    // 结果按 M=全量query 存放，需按后处理分块的 query 行偏移取回本块
    auto ipOffset = queryLoopIdx * querySizeEachLoop * codeSizeEachLoop;
    DataCopy(innerProductLocal, innerProductGloabal[ipOffset], queryProcNum * codeSizeEachLoop);

    // Ascend950 MIX 1:2：使用动态事件ID避免与Matmul内部事件冲突
    auto evtMte2S = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_S));
    AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(evtMte2S);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(evtMte2S);

    UpGradeDataType neg2 = -2.0;
    Axpy(l2DistanceLocal, innerProductLocal, neg2, queryProcNum * codeSizeEachLoop);
    Cast(l2DistanceTensor, l2DistanceLocal, RoundMode::CAST_NONE, queryProcNum * codeSizeEachLoop);

    innerProductQue.FreeTensor(innerProductLocal);
}

// 逐burst(64条code)产出 (最小值, burst内argmin) 的交错pair，供 host 的 quick heap 使用。
//
// host(topk_flat_cpu_kernel.cpp, quickTopk_==1)的 stage-one 直接用
//   outdists[0] = vmdists[i*2]、outlabel[0] = burstLen_*i + vmlabel[i*2+1]
// 更新堆，因此 pair 的值槽必须是"该 burst(64条code)的最小距离"，索引槽必须是 burst 内真实 argmin(0..63)，
// 否则 label = burstLen*i + 索引 + 基址 会系统性越过 ntotal。

//   用一次 WholeReduceMax(mask=burstLen, repeatTimes, dstRepStride=1, srcBlkStride=1, srcRepStride=4)
//   直接产出 [最值, burst内下标] 交错 pair，无任何标量热路径。
//   dstRepStride=1 时相邻 pair 紧挨排列，索引槽与值同宽(half→按 uint16 读)。
//
// 【950 关键约束】同源 A/B 诊断已坐实：本 build 上 `WholeReduceMin` 的 [值,索引] pair 输出为静默空实现
//   （值槽/索引槽恒为0），而 `WholeReduceMax` 正常。故这里在"负距离域"上求最小：
//       max(-d) == -min(d)，且 argmax(-d) == argmin(d)。
//   调用方(Compute)已先把源取负；本函数用 WholeReduceMax 产出 [-min, argmin] pair，
//   随后由调用方把值槽还原为 min、索引槽按位清符号位还原为 argmin。
// 尾部 burst（codeProcNum%MIN_BATCH!=0）单独调用一次(repeatTimes=1, mask=tailLen)；
//   主体/尾部写读地址均不重叠，且向量指令按序执行(Cast与ReduceMax同属PIPE_V)，故无需 PipeBarrier（与ip_maxs参考一致）。
// Duplicate 把整轮 pair 清零：尾部/未使用槽保持 0，host 只读 ntotal/burstLen 以内的 burst，不会误用。
template <typename DataType>
__aicore__ inline void DistanceFlatL2With950<DataType>::ComputeL2DistanceMin(uint32_t queryProcNum,
                                                                             uint32_t codeProcNum,
                                                                             LocalTensor<DataType> &l2DistanceLocal,
                                                                             LocalTensor<DataType> &l2DistanceLocalMin)
{
    DataType zero = 0;
    Duplicate(l2DistanceLocalMin, zero, l2DistanceLocalMin.GetSize());

    // 一次归约处理 MIN_BATCH 个元素；srcRepStride 以 32B DataBlock 为单位
    constexpr uint32_t srcRepStride = MIN_BATCH * sizeof(DataType) / BLOCK_BYTE_SIZE;

    uint32_t repeatTimes = codeProcNum / MIN_BATCH;
    if (repeatTimes != 0)
    {
        for (uint32_t i = 0; i < queryProcNum; i++)
        {
            WholeReduceMax<DataType>(l2DistanceLocalMin[i * codeBatchSize * BURST_BLOCK_RATIO],
                                     l2DistanceLocal[i * codeSizeEachLoop], MIN_BATCH, repeatTimes, 1, 1, srcRepStride);
        }
    }

    uint32_t lastNum = codeProcNum % MIN_BATCH;
    if (lastNum != 0)
    {
        for (uint32_t i = 0; i < queryProcNum; i++)
        {
            WholeReduceMax<DataType>(l2DistanceLocalMin[(i * codeBatchSize + repeatTimes) * BURST_BLOCK_RATIO],
                                     l2DistanceLocal[i * codeSizeEachLoop + repeatTimes * MIN_BATCH], lastNum, 1, 1, 1,
                                     srcRepStride);
        }
    }
}

template <typename DataType>
__aicore__ inline void DistanceFlatL2With950<DataType>::CopyOutL2Dist(uint32_t queryLoopIdx, uint32_t codeLoopIdx,
                                                                      uint32_t queryProcNum, uint32_t codeProcNum,
                                                                      LocalTensor<DataType> &l2DistanceLocal)
{
    auto copyOffset = queryLoopIdx * queryCopyOutOffset + codeLoopIdx * codeSizeEachLoop;

    DataCopyParams copyParams;
    copyParams.blockCount = queryProcNum;
    copyParams.blockLen = codeProcNum * sizeof(DataType) / BLOCK_BYTE_SIZE;
    copyParams.srcStride = (codeSizeEachLoop - codeProcNum) * sizeof(DataType) / BLOCK_BYTE_SIZE;
    copyParams.dstStride = (blockSize - codeProcNum) * sizeof(DataType) / BLOCK_BYTE_SIZE;
    DataCopy(l2distGlobal[copyOffset], l2DistanceLocal, copyParams);
}

template <typename DataType>
__aicore__ inline void DistanceFlatL2With950<DataType>::CopyOutL2DistMin(uint32_t queryLoopIdx, uint32_t codeLoopIdx,
                                                                         uint32_t queryProcNum,
                                                                         LocalTensor<DataType> &l2DistanceLocalMin)
{
    // 每轮极值对(pair)数 = codeSizeEachLoop/MIN_BATCH = codeCopyOutBatchOffset = codeBatchSize = 8，
    // 该等式依赖 CORE_PROC_CODE_GRAIN == host CODE_PROC_PER_LOOP_MAX == 512(见文件头常量注释，勿单独调整)；
    // 写入长度 blockLen(codeBatchSize个pair = 16个half = 32B = 1个DataBlock) 与该轮目的地址步进
    // (codeCopyOutBatchOffset个pair = 16个half) 完全一致 ⇒ 各轮写入区间互不重叠，无需atomic累加，
    auto copyOffset = (queryLoopIdx * queryCopyOutBatchOffset + codeLoopIdx * codeCopyOutBatchOffset) * 2;

    DataCopyParams copyParams;
    copyParams.blockCount = queryProcNum;
    copyParams.blockLen = codeBatchSize * 2 * sizeof(DataType) / BLOCK_BYTE_SIZE;
    // UB侧每个query的极值行紧凑排列(codeBatchSize*2个half)，以整块为基准则块间无间隙 ⇒ srcStride必须显式置0
    copyParams.srcStride = 0;
    // 行距须与host侧curBurstsOfBlock一致：每个query一行codeBatchSizeTotal*2个half
    copyParams.dstStride = (codeBatchSizeTotal - codeBatchSize) * 2 * sizeof(DataType) / BLOCK_BYTE_SIZE;
    DataCopy(l2distMinGlobal[copyOffset], l2DistanceLocalMin, copyParams);
}

template <typename DataType>
__aicore__ inline void DistanceFlatL2With950<DataType>::SetFlagReady()
{
    // 每个flag占CUBE_ALIGN(32B=16个uint16)，stride与topk AICPU的FLAG_SIZE=16对齐
    uint32_t flagElemCnt = pTilingDevice->aivNum * Utils::CUBE_ALIGN;

    TBuf<> flagBuf;
    pipe.InitBuffer(flagBuf, flagElemCnt * sizeof(uint16_t));
    LocalTensor<uint16_t> flagLocal = flagBuf.Get<uint16_t>(flagElemCnt);

    uint16_t zero = 0;
    Duplicate(flagLocal, zero, flagElemCnt);

    // Ascend950: Scalar/Vector share PIPE_S, Duplicate(V)→SetValue(S) sequential, no flag needed.
    // 写aivNum个flag（与TopK AICPU的coreNum_一致），每个AIV核一个flag
    uint16_t readyFlag = 1;
    for (uint32_t i = 0; i < pTilingDevice->aivNum; i++)
    {
        flagLocal.SetValue(i * Utils::CUBE_ALIGN, readyFlag);
    }

    // S→MTE3: 等SetValue(S流)完成再DataCopy(MTE3)
    auto evtSMte3 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::S_MTE3));
    AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(evtSMte3);
    AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(evtSMte3);
    DataCopy(flagGlobal, flagLocal, flagElemCnt);

    // 退出kernel前确认ready位已从UB写到GM，供另一条流上的AICPU观察。
    auto evtMte3S = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE3_S));
    AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(evtMte3S);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(evtMte3S);
}

template <typename DataType>
__aicore__ inline void DistanceFlatL2With950<DataType>::DoMask(uint32_t queryLoopIdx, uint32_t codeLoopIdx,
                                                               uint32_t queryProcNum, uint32_t codeProcNum,
                                                               LocalTensor<half> &l2DistanceLocal)
{
    if (maskFlag == 0)
    {
        return;
    }
    auto maskLocal = maskQue.AllocTensor<uint8_t>();
    // 本次算子已经计算好的query的对应的mask偏移:queryLoopIdx * querySizeEachLoop * maskLen
    // 本次算子已经计算好的code的对应的mask偏移:codeLoopIdx * codeSizeEachLoop / Utils::MASK_BIT_NUM
    uint64_t maskOffset = queryLoopIdx * querySizeEachLoop * static_cast<uint64_t>(maskLen) +
                          codeLoopIdx * codeSizeEachLoop / Utils::MASK_BIT_NUM;

    for (uint32_t i = 0; i < queryProcNum; i++)
    {
        DataCopy(maskLocal[i * maskLenEachLoop], maskGlobal[i * static_cast<uint64_t>(maskLen) + maskOffset],
                 maskLenEachLoop);
    }
    maskQue.EnQue(maskLocal);
    maskLocal = maskQue.DeQue<uint8_t>();
    // dstBlkStride,src0BlkStride,src1BlkStride 1
    // dstRepStride,src0RepStride,src1RepStride 8
    BinaryRepeatParams param{1, 1, 1, 8, 8, 8};

    const uint32_t distOffset = Utils::SELECT_REPEAT_TIME * Utils::VIC_HALF_FULL_MASK;
    const uint32_t maskRepateOffset = distOffset / Utils::MASK_BIT_NUM;
    for (uint32_t i = 0; i < selectLoopTime; i++)
    {
        Select(l2DistanceLocal[i * distOffset], maskLocal[i * maskRepateOffset], l2DistanceLocal[i * distOffset],
               Utils::HALF_MAX, SELMODE::VSEL_TENSOR_SCALAR_MODE, Utils::VIC_HALF_FULL_MASK, Utils::SELECT_REPEAT_TIME,
               param);
    }
    if (selectRemainder != 0)
    {
        Select(l2DistanceLocal[selectLoopTime * distOffset], maskLocal[selectLoopTime * maskRepateOffset],
               l2DistanceLocal[selectLoopTime * distOffset], Utils::HALF_MAX, SELMODE::VSEL_TENSOR_SCALAR_MODE,
               Utils::VIC_HALF_FULL_MASK, selectRemainder, param);
    }

    maskQue.FreeTensor(maskLocal);
}

template <typename DataType>
__aicore__ inline void DistanceFlatL2With950<DataType>::Compute(uint32_t queryLoopIdx, uint32_t codeLoopIdx,
                                                                uint32_t queryProcNum, uint32_t codeProcNum,
                                                                uint32_t normOffset,
                                                                LocalTensor<UpGradeDataType> &queryL2NormAllLocal,
                                                                LocalTensor<UpGradeDataType> &codePreNormLocal)
{
    auto l2DistanceTensor = l2DistanceQue.AllocTensor<DataType>();
    {
        {
            auto l2DistanceLocal = l2DistanceLocalQue.AllocTensor<UpGradeDataType>();
            ComputeL2NormSum(queryProcNum, normOffset, l2DistanceLocal, queryL2NormAllLocal, codePreNormLocal);
            ComputeL2Distance(queryLoopIdx, queryProcNum, l2DistanceTensor, l2DistanceLocal);
            l2DistanceLocalQue.FreeTensor(l2DistanceLocal);
        };
    }

    DoMask(queryLoopIdx, codeLoopIdx, queryProcNum, codeProcNum, l2DistanceTensor);

    auto l2DistanceMinTensor = l2DistanceMinQue.AllocTensor<DataType>();
    const uint32_t distanceNum = queryProcNum * codeSizeEachLoop;
    const uint32_t pairNum = queryProcNum * codeBatchSize * BURST_BLOCK_RATIO;
    const DataType negOne = static_cast<DataType>(-1);
    // 950: WholeReduceMin 的 [值,索引] pair 输出为静默空实现（同源 A/B 已坐实），改用 WholeReduceMax 在负距离域求最小。
    // 1) 源取负 → 2) WholeReduceMax 产出 [-min, argmin] → 3) 还原源 → 4) 还原 pair 各槽符号
    Muls(l2DistanceTensor, l2DistanceTensor, negOne, distanceNum);
    ComputeL2DistanceMin(queryProcNum, codeProcNum, l2DistanceTensor, l2DistanceMinTensor);
    // 还原源：CopyOutL2Dist 需要正值全量距离（host stage-two 会回扫 indists）
    Muls(l2DistanceTensor, l2DistanceTensor, negOne, distanceNum);
    // 还原 pair：值槽 -min --(*-1)--> min；索引槽 j --(*-1)--> -j，再按位清符号位 --> j
    Muls(l2DistanceMinTensor, l2DistanceMinTensor, negOne, pairNum);
    {
        auto pairInt16 = l2DistanceMinTensor.template ReinterpretCast<int16_t>();
        auto signMask = signMaskBuf.Get<int16_t>(pairNum);
        And(pairInt16, pairInt16, signMask, pairNum);
    }

    // Ascend950: 两块输出UB均由 Vector 流写入(全距离来自 Cast/Select，极值 pair 来自 WholeReduceMax + 符号还原)，
    // 统一用一次 V_MTE3 保证写入对后续 MTE3 搬运可见（与参考内核 distance_flat_ip_with950 一致）。
    auto evtVMte3 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_MTE3));
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(evtVMte3);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(evtVMte3);
    CopyOutL2Dist(queryLoopIdx, codeLoopIdx, queryProcNum, codeProcNum, l2DistanceTensor);
    CopyOutL2DistMin(queryLoopIdx, codeLoopIdx, queryProcNum, l2DistanceMinTensor);

    l2DistanceQue.FreeTensor(l2DistanceTensor);
    l2DistanceMinQue.FreeTensor(l2DistanceMinTensor);

    // wait for copy l2distance to gm over
    auto evtMte3Mte2 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE3_MTE2));
    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(evtMte3Mte2);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(evtMte3Mte2);
}

template <typename DataType>
__aicore__ inline void DistanceFlatL2With950<DataType>::ComputeQueryL2Norm(
    uint32_t queryOffset, uint32_t queryProcNum, uint32_t normOffset, LocalTensor<UpGradeDataType> &queryL2NormAllLocal,
    bool isTail)
{
    if (isTail)
    {
        matmulObjL2Norm.SetTail(querySizeLastLoop, querySizeLastLoop, dimSize);
    }

    matmulObjL2Norm.SetTensorA(querysGlobal[queryOffset]);
    matmulObjL2Norm.SetTensorB(querysGlobal[queryOffset], true);
    // 统一为flat_ip/int8-950写法：IterateAll<false>(...,0,false,true) + WaitIterateAll + End。
    // 缺少WaitIterateAll会导致AIV在Cube写GM完成前读取queryDotQueryGlobal（竞争）。
    matmulObjL2Norm.template IterateAll<false>(queryDotQueryGlobal, 0, false, true);
    matmulObjL2Norm.WaitIterateAll();
    // 切换到matmulObjIp前必须释放当前Matmul对象的内部资源。
    matmulObjL2Norm.End();

    if ASCEND_IS_AIV
    {
        if (GetSubBlockIdx() == 1)
        {
            auto queryDotQueryLocal = queryDotQueryQue.AllocTensor<UpGradeDataType>();
            DataCopy(queryDotQueryLocal, queryDotQueryGlobal, queryAlign * queryAlign);
            queryDotQueryQue.EnQue<UpGradeDataType>(queryDotQueryLocal);
            queryDotQueryLocal = queryDotQueryQue.DeQue<UpGradeDataType>();
            auto evtMte2S = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_S));
            AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(evtMte2S);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(evtMte2S);
            for (auto i = 0; i < queryProcNum; i++)
            {
                // C矩阵的行距由matmul tiling的N决定(=queryAlign)，与host侧SetShape保持一致
                auto querySquareSum = queryDotQueryLocal.GetValue(i * queryAlign + i);
                queryL2NormAllLocal.SetValue(normOffset + i, querySquareSum);
            }
            queryDotQueryQue.FreeTensor(queryDotQueryLocal);
        }
    }
}

template <typename DataType>
__aicore__ inline void DistanceFlatL2With950<DataType>::process()
{
    // 多个Matmul对象必须与各自的tiling成对注册，且先于所有InitBuffer调用。
    REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), matmulObjIp, &cubeTilingIp, matmulObjL2Norm, &cubeTilingL2Norm);
    InitMemoryQueue();

    // AIC和AIV保持相同的Matmul调用序列；每组奇数编号的AIV执行Vector后处理。
    if (corePorcCodeSize != 0)
    {
        // 1) 预计算全部 query 的 L2 范数：分块顺序调用 L2Norm matmul，后处理子核各分块一次写入
        //    queryL2NormAllLocal 的对应偏移(normOffset)。GM 中间结果 queryDotQueryGlobal 按 cube 组共享，
        //    顺序复用不会冲突。
        LocalTensor<UpGradeDataType> queryL2NormAllLocal;
        if ASCEND_IS_AIV
        {
            if (GetSubBlockIdx() == 1)
            {
                queryL2NormAllLocal = queryL2NormQue.AllocTensor<UpGradeDataType>();
            }
        }
        for (auto queryLoopIdx = 0; queryLoopIdx < queryLoopTimes; queryLoopIdx++)
        {
            bool isQueryTail = (queryLoopIdx == (queryLoopTimes - 1));
            auto queryProcNum = isQueryTail ? querySizeLastLoop : querySizeEachLoop;
            auto queryOffset = queryLoopIdx * querySizeEachLoop * dimSize;
            auto normOffset = queryLoopIdx * querySizeEachLoop;
            ComputeQueryL2Norm(queryOffset, queryProcNum, normOffset, queryL2NormAllLocal, isQueryTail);
        }

        // 2) code 外层循环：M 取「全量 query」，每个 codeLoop 只发一次 matmul，code 矩阵从 HBM 仅读一遍。
        //    后处理仍按 querySizeEachLoop 分块(UB 不涨)，从 Ip GM 结果中按 query 行偏移分块取回。
        matmulObjIp.SetTensorA(querysGlobal[0]);
        for (auto codeLoopIdx = 0; codeLoopIdx < codeLoopTimes; codeLoopIdx++)
        {
            auto codeProcNum = (codeLoopIdx != (codeLoopTimes - 1)) ? codeSizeEachLoop : codeSizeLastLoop;
            // ComputeInnerProduct内部已完成IterateAll/WaitIterateAll/End
            ComputeInnerProduct(codeLoopIdx);
            if ASCEND_IS_AIV
            {
                if (GetSubBlockIdx() == 1)
                {
                    LocalTensor<UpGradeDataType> codePreNormLocal;
                    LoadCodePreNorm(codeLoopIdx, codePreNormLocal);
                    for (auto queryLoopIdx = 0; queryLoopIdx < queryLoopTimes; queryLoopIdx++)
                    {
                        bool isQueryTail = (queryLoopIdx == (queryLoopTimes - 1));
                        auto queryProcNum = isQueryTail ? querySizeLastLoop : querySizeEachLoop;
                        auto normOffset = queryLoopIdx * querySizeEachLoop;
                        Compute(queryLoopIdx, codeLoopIdx, queryProcNum, codeProcNum, normOffset, queryL2NormAllLocal,
                                codePreNormLocal);
                    }
                    codeL2NormCastQue.FreeTensor(codePreNormLocal);
                }
            }
        }

        if ASCEND_IS_AIV
        {
            if (GetSubBlockIdx() == 1)
            {
                queryL2NormQue.FreeTensor(queryL2NormAllLocal);
            }
        }
    }

    if ASCEND_IS_AIV
    {
        if (GetSubBlockIdx() == 1)
        {
            // 最后一轮Vector结果必须先写回GM，不能只等待Scalar到达核间屏障。
            auto evtMte3S = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE3_S));
            AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(evtMte3S);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(evtMte3S);
        }
    }

    SyncAll();
    // flag由AIV block 0统一写全部aivNum个，其余核不写；AIC同样满足GetBlockIdx()==0，
    // 故必须用ASCEND_IS_AIV守卫，避免AIC0重复执行InitBuffer/Duplicate/DataCopy(flagGlobal)。
    if ASCEND_IS_AIV
    {
        if (GetBlockIdx() == 0)
        {
            SetFlagReady();
        }
    }
}
}  // namespace IndexOps

extern "C" __global__ __aicore__ void distance_flat_l2_with950(GM_ADDR queries, GM_ADDR mask, GM_ADDR codes,
                                                               GM_ADDR l2PreNorm, GM_ADDR actualNum, GM_ADDR l2Distance,
                                                               GM_ADDR l2DistanceMin, GM_ADDR flag, GM_ADDR workspace,
                                                               GM_ADDR tiling)
{
    if (workspace == nullptr)
    {
        return;
    }

    // 声明MIX 1:2模型：aicNum个AIC + 2*aicNum个AIV
    // Matmul高阶API在MIX模式下由AIV发起、AIC执行Cube计算。
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);

    GET_TILING_DATA(tiling_data, tiling);
    const DistanceFlatL2With950TilingData *__restrict pTilingDevice = &tiling_data;

    GM_ADDR usrWorkspace = GetUserWorkspace(workspace);

    IndexOps::DistanceFlatL2With950<half> op;
    op.Init(queries, mask, codes, l2PreNorm, actualNum, l2Distance, l2DistanceMin, flag, pTilingDevice, usrWorkspace);
    op.process();
}
