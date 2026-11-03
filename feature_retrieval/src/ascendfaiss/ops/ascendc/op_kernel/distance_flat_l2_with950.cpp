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

#include <algorithm>
#include <cstdio>
#include <type_traits>

// CUBE_ONLY：matmul 仅在 AIC 上执行，AIV 完全不调用 matmul API。
// 只有这样 AIC/AIV 之间才能用裸 CrossCore flag 做显式同步(参考 ascendc_l2_norm_with950 /
// ascendc_ivfpq_subspace_distance)：AIC 写 cube 结果到 GM 槽后置 CUBE_READY，AIV 读完置
// AIV_READ_DONE，AIC 收到后才写下一槽，从而彻底消除「AIC 覆盖 AIV 正在读的槽」的竞争。
#define ASCENDC_CUBE_ONLY

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/matmul.h"
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
// 输出布局与 host 侧 workspace 大小(queryAlignEpi*CODE_PROC_PER_LOOP_MAX)，必须两侧同步修改。
constexpr uint32_t CORE_PROC_CODE_GRAIN = 512;

constexpr uint16_t FLATL2_CUBE_READY_ID = 1;
// AIV 读完本轮结果后置该 flag，AIC 收到后才写下一轮槽位。
// 只有这条反向握手才能保证「AIC 写 GM」与「AIV 读 GM」不重叠：单向 flag 下 AIV 刚通过
// wait 就开始搬运 64KB，而 AIC 已经在打下一轮 fixpipe，AIV 会读到半刷新的矩阵。
// mode2 语义是「同组两个 AIV 都 set 之后，AIC 的 wait 才放行」，因此两个次块都必须置位。
constexpr uint16_t FLATL2_AIV_READ_DONE_ID = 2;
constexpr uint8_t SYNC_MODE2 = 2;
// Ip/L2Norm 中间结果在 GM 上的双缓冲槽数：槽位 = iterIdx % FLATL2_CUBE_SLOT_NUM
constexpr uint32_t FLATL2_CUBE_SLOT_NUM = 2;
}  // namespace

namespace IndexOps
{
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

    // flag写入，从kernel入口调用
    __aicore__ inline void SetFlagReady();

    __aicore__ inline void RegisterMatmul();
    __aicore__ inline void ProcessCube();

    // AIV 侧：初始化 vector 侧缓冲并执行后处理主循环，读槽前等 CUBE_READY、读完置 AIV_READ_DONE
    __aicore__ inline void InitVectorBuffer();
    __aicore__ inline void ProcessVector();

    // 所有核共同收尾：等 MTE3 落盘 + SyncAll + AIV block0 写 ready flag
    __aicore__ inline void Finish();

   private:
    // code 预归一化(y^2) 每 codeLoop 只加载/抬升一次(gm -> ub)，供该轮所有 query 分块复用。
    __aicore__ inline void LoadCodePreNorm(uint32_t codeLoopIdx, LocalTensor<UpGradeDataType> &codePreNormLocal);

    __aicore__ inline void Compute(uint32_t queryLoopIdx, uint32_t codeLoopIdx, uint32_t queryProcNum,
                                   uint32_t codeProcNum, uint32_t normOffset,
                                   LocalTensor<UpGradeDataType> &queryL2NormAllLocal,
                                   LocalTensor<UpGradeDataType> &codePreNormLocal);

    // AIC：计算一批 query 的 L2 范数(query * query^T)，结果写入 queryDotQueryGlobal 当前槽
    __aicore__ inline void ComputeQueryNormCube(uint32_t queryOffset, uint32_t queryProcNum, bool isTail);

    // AIV：从 queryDotQueryGlobal 当前槽取对角线，写入 queryL2NormAllLocal[normOffset .. +queryProcNum)
    __aicore__ inline void QueryNormPostProcess(uint32_t normOffset, uint32_t queryProcNum,
                                                LocalTensor<UpGradeDataType> &queryL2NormAllLocal);

    // 切换 Ip / L2Norm 中间结果的 GM 双缓冲槽位
    __aicore__ inline void SetIpSlot(uint32_t iterIdx);
    __aicore__ inline void SetNormSlot(uint32_t iterIdx);

    // compute part1:query_local^2 + code_l2_local^2 (vector)
    __aicore__ inline void ComputeL2NormSum(uint32_t queryProcNum, uint32_t normOffset,
                                            LocalTensor<UpGradeDataType> &l2NormSumLocal,
                                            LocalTensor<UpGradeDataType> &queryL2NormAllLocal,
                                            LocalTensor<UpGradeDataType> &codePreNormLocal);

    // compute part2:x * y。M 取本轮 query 分块(queryProcNum)，与 host cubeTilingIp 的 M 一致
    __aicore__ inline void ComputeInnerProduct(uint32_t codeLoopIdx, uint32_t queryProcNum, uint32_t codeProcNum);

    // compute l2Distance:part1 - 2*part2
    __aicore__ inline void ComputeL2Distance(uint32_t queryProcNum, LocalTensor<DataType> &l2DistanceTensor,
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

    // Ip / L2Norm 中间结果在 GM 上的双缓冲基址与单槽元素数
    __gm__ UpGradeDataType *ipSlotBase{nullptr};
    __gm__ UpGradeDataType *normSlotBase{nullptr};
    uint32_t ipSlotElemCnt{0};
    uint32_t normSlotElemCnt{0};

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
    uint32_t queryAlignAll{0};  // 全量 query 的 M 对齐尺寸(=AlignUp(querySize,16))，用于 query 范数缓冲
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

    ipSlotElemCnt = queryAlign * codeSizeEachLoop;
    normSlotElemCnt = queryAlign * queryAlign;
    auto groupBase = (__gm__ UpGradeDataType *)usrWorkspace +
                     workIdx * (FLATL2_CUBE_SLOT_NUM * ipSlotElemCnt + FLATL2_CUBE_SLOT_NUM * normSlotElemCnt);
    ipSlotBase = groupBase;
    normSlotBase = groupBase + FLATL2_CUBE_SLOT_NUM * ipSlotElemCnt;
    SetIpSlot(0);
    SetNormSlot(0);
}

template <typename DataType>
__aicore__ inline void DistanceFlatL2With950<DataType>::SetIpSlot(uint32_t iterIdx)
{
    innerProductGloabal.SetGlobalBuffer(ipSlotBase + (iterIdx % FLATL2_CUBE_SLOT_NUM) * ipSlotElemCnt, ipSlotElemCnt);
}

template <typename DataType>
__aicore__ inline void DistanceFlatL2With950<DataType>::SetNormSlot(uint32_t iterIdx)
{
    queryDotQueryGlobal.SetGlobalBuffer(normSlotBase + (iterIdx % FLATL2_CUBE_SLOT_NUM) * normSlotElemCnt,
                                        normSlotElemCnt);
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
__aicore__ inline void DistanceFlatL2With950<DataType>::InitVectorBuffer()
{
    // AIV 侧后处理所需的 UB 缓冲（CUBE_ONLY 下不再要求与 AIC 缓冲尺寸一致）
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
    // 950 归约符号还原掩码(int16=0x7FFF)：清除 pair 索引槽取负后残留的符号位
    pipe.InitBuffer(signMaskBuf, querySizeEachLoop * codeBatchSize * BURST_BLOCK_RATIO * sizeof(int16_t));
    LocalTensor<int16_t> signMask = signMaskBuf.Get<int16_t>(querySizeEachLoop * codeBatchSize * BURST_BLOCK_RATIO);
    Duplicate(signMask, static_cast<int16_t>(0x7FFF), querySizeEachLoop * codeBatchSize * BURST_BLOCK_RATIO);
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
__aicore__ inline void DistanceFlatL2With950<DataType>::ComputeInnerProduct(uint32_t codeLoopIdx, uint32_t queryProcNum,
                                                                            uint32_t codeProcNum)
{
    matmulObjIp.SetTail(queryProcNum, codeProcNum, dimSize);

    auto codeOffset = codeLoopIdx * codeSizeEachLoop * dimSize;
    matmulObjIp.SetTensorB(codesGlobal[codeOffset], true);

    matmulObjIp.template IterateAll<false>(innerProductGloabal, 0, false, true);
    matmulObjIp.End();
}

template <typename DataType>
__aicore__ inline void DistanceFlatL2With950<DataType>::ComputeL2Distance(uint32_t queryProcNum,
                                                                          LocalTensor<DataType> &l2DistanceTensor,
                                                                          LocalTensor<UpGradeDataType> &l2DistanceLocal)
{
    auto innerProductLocal = innerProductQue.AllocTensor<UpGradeDataType>();
    // C 矩阵即本轮 query 分块(M=queryProcNum)的结果，从偏移 0 读回
    DataCopy(innerProductLocal, innerProductGloabal, queryProcNum * codeSizeEachLoop);

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
            ComputeL2Distance(queryProcNum, l2DistanceTensor, l2DistanceLocal);
            l2DistanceLocalQue.FreeTensor(l2DistanceLocal);
        };
    }

    DoMask(queryLoopIdx, codeLoopIdx, queryProcNum, codeProcNum, l2DistanceTensor);

    auto l2DistanceMinTensor = l2DistanceMinQue.AllocTensor<DataType>();
    const uint32_t distanceNum = queryProcNum * codeSizeEachLoop;
    const uint32_t pairNum = queryProcNum * codeBatchSize * BURST_BLOCK_RATIO;
    const DataType negOne = static_cast<DataType>(-1);

    Muls(l2DistanceTensor, l2DistanceTensor, negOne, distanceNum);
    ComputeL2DistanceMin(queryProcNum, codeProcNum, l2DistanceTensor, l2DistanceMinTensor);

    Muls(l2DistanceTensor, l2DistanceTensor, negOne, distanceNum);

    Muls(l2DistanceMinTensor, l2DistanceMinTensor, negOne, pairNum);
    {
        auto pairInt16 = l2DistanceMinTensor.template ReinterpretCast<int16_t>();
        auto signMask = signMaskBuf.Get<int16_t>(pairNum);
        And(pairInt16, pairInt16, signMask, pairNum);
    }

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
__aicore__ inline void DistanceFlatL2With950<DataType>::ComputeQueryNormCube(uint32_t queryOffset,
                                                                             uint32_t queryProcNum, bool isTail)
{
    if (isTail)
    {
        matmulObjL2Norm.SetTail(querySizeLastLoop, querySizeLastLoop, dimSize);
    }

    matmulObjL2Norm.SetTensorA(querysGlobal[queryOffset]);
    matmulObjL2Norm.SetTensorB(querysGlobal[queryOffset], true);

    matmulObjL2Norm.template IterateAll<false>(queryDotQueryGlobal, 0, false, true);
    matmulObjL2Norm.End();
}

template <typename DataType>
__aicore__ inline void DistanceFlatL2With950<DataType>::QueryNormPostProcess(
    uint32_t normOffset, uint32_t queryProcNum, LocalTensor<UpGradeDataType> &queryL2NormAllLocal)
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

template <typename DataType>
__aicore__ inline void DistanceFlatL2With950<DataType>::RegisterMatmul()
{
    // 多个Matmul对象必须与各自的tiling成对注册，且先于所有InitBuffer调用。
    REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), matmulObjIp, &cubeTilingIp, matmulObjL2Norm, &cubeTilingL2Norm);
}

template <typename DataType>
__aicore__ inline void DistanceFlatL2With950<DataType>::ProcessCube()
{
    if (corePorcCodeSize == 0)
    {
        return;
    }

    // 阶段1：逐 query 分块计算 L2 范数(query * query^T)
    for (auto queryLoopIdx = 0; queryLoopIdx < queryLoopTimes; queryLoopIdx++)
    {
        bool isQueryTail = (queryLoopIdx == (queryLoopTimes - 1));
        auto queryProcNum = isQueryTail ? querySizeLastLoop : querySizeEachLoop;
        auto queryOffset = queryLoopIdx * querySizeEachLoop * dimSize;

        AscendC::CrossCoreWaitFlag(FLATL2_AIV_READ_DONE_ID);
        SetNormSlot(queryLoopIdx);
        ComputeQueryNormCube(queryOffset, queryProcNum, isQueryTail);
        AscendC::CrossCoreSetFlag<SYNC_MODE2, PIPE_FIX>(FLATL2_CUBE_READY_ID);
    }

    // 阶段2：code 外层 + query 内层，M 取本轮 query 分块(queryProcNum)
    uint32_t iterIdx = 0;
    for (auto codeLoopIdx = 0; codeLoopIdx < codeLoopTimes; codeLoopIdx++)
    {
        auto codeProcNum = (codeLoopIdx != (codeLoopTimes - 1)) ? codeSizeEachLoop : codeSizeLastLoop;
        for (auto queryLoopIdx = 0; queryLoopIdx < queryLoopTimes; queryLoopIdx++)
        {
            bool isQueryTail = (queryLoopIdx == (queryLoopTimes - 1));
            auto queryProcNum = isQueryTail ? querySizeLastLoop : querySizeEachLoop;
            auto queryOffset = queryLoopIdx * querySizeEachLoop * dimSize;

            AscendC::CrossCoreWaitFlag(FLATL2_AIV_READ_DONE_ID);
            SetIpSlot(iterIdx);
            matmulObjIp.SetTensorA(querysGlobal[queryOffset]);
            ComputeInnerProduct(codeLoopIdx, queryProcNum, codeProcNum);
            AscendC::CrossCoreSetFlag<SYNC_MODE2, PIPE_FIX>(FLATL2_CUBE_READY_ID);
            iterIdx++;
        }
    }
}

template <typename DataType>
__aicore__ inline void DistanceFlatL2With950<DataType>::ProcessVector()
{
    if (corePorcCodeSize == 0)
    {
        return;
    }

    LocalTensor<UpGradeDataType> queryL2NormAllLocal;
    if (GetSubBlockIdx() == 1)
    {
        queryL2NormAllLocal = queryL2NormQue.AllocTensor<UpGradeDataType>();
    }

    // 阶段1：query 范数
    for (auto queryLoopIdx = 0; queryLoopIdx < queryLoopTimes; queryLoopIdx++)
    {
        bool isQueryTail = (queryLoopIdx == (queryLoopTimes - 1));
        auto queryProcNum = isQueryTail ? querySizeLastLoop : querySizeEachLoop;
        auto normOffset = queryLoopIdx * querySizeEachLoop;

        AscendC::CrossCoreSetFlag<SYNC_MODE2, PIPE_MTE2>(FLATL2_AIV_READ_DONE_ID);
        AscendC::CrossCoreWaitFlag(FLATL2_CUBE_READY_ID);
        SetNormSlot(queryLoopIdx);
        if (GetSubBlockIdx() == 1)
        {
            QueryNormPostProcess(normOffset, queryProcNum, queryL2NormAllLocal);
        }
    }

    // 阶段2：内积 + L2 距离后处理；code 外层 + query 内层，与 AIC 顺序一致
    uint32_t iterIdx = 0;
    for (auto codeLoopIdx = 0; codeLoopIdx < codeLoopTimes; codeLoopIdx++)
    {
        auto codeProcNum = (codeLoopIdx != (codeLoopTimes - 1)) ? codeSizeEachLoop : codeSizeLastLoop;
        // 每个 code 分块只加载一次 code 预归一化，供本轮所有 query 分块复用
        LocalTensor<UpGradeDataType> codePreNormLocal;

        for (auto queryLoopIdx = 0; queryLoopIdx < queryLoopTimes; queryLoopIdx++)
        {
            bool isQueryTail = (queryLoopIdx == (queryLoopTimes - 1));
            auto queryProcNum = isQueryTail ? querySizeLastLoop : querySizeEachLoop;
            auto normOffset = queryLoopIdx * querySizeEachLoop;

            AscendC::CrossCoreSetFlag<SYNC_MODE2, PIPE_MTE2>(FLATL2_AIV_READ_DONE_ID);
            AscendC::CrossCoreWaitFlag(FLATL2_CUBE_READY_ID);
            SetIpSlot(iterIdx);
            if (GetSubBlockIdx() == 1)
            {
                if (queryLoopIdx == 0)
                {
                    LoadCodePreNorm(codeLoopIdx, codePreNormLocal);
                }
                Compute(queryLoopIdx, codeLoopIdx, queryProcNum, codeProcNum, normOffset, queryL2NormAllLocal,
                        codePreNormLocal);
            }
            iterIdx++;
        }

        if (GetSubBlockIdx() == 1)
        {
            codeL2NormCastQue.FreeTensor(codePreNormLocal);
        }
    }

    if (GetSubBlockIdx() == 1)
    {
        queryL2NormQue.FreeTensor(queryL2NormAllLocal);
    }
}

template <typename DataType>
__aicore__ inline void DistanceFlatL2With950<DataType>::Finish()
{
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
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);

    GET_TILING_DATA(tiling_data, tiling);
    const DistanceFlatL2With950TilingData *__restrict pTilingDevice = &tiling_data;

    GM_ADDR usrWorkspace = GetUserWorkspace(workspace);

    IndexOps::DistanceFlatL2With950<half> op;
    op.Init(queries, mask, codes, l2PreNorm, actualNum, l2Distance, l2DistanceMin, flag, pTilingDevice, usrWorkspace);

    if ASCEND_IS_AIC
    {
        op.RegisterMatmul();
        op.ProcessCube();
    }
    if ASCEND_IS_AIV
    {
        op.InitVectorBuffer();
        op.ProcessVector();
    }
    // 所有核共同收尾（SyncAll 是全局屏障，AIC/AIV 都必须参与）
    op.Finish();
}
