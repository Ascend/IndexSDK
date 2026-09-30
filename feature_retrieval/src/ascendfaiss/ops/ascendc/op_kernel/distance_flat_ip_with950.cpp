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
#include "op_kernel_common.h"

using namespace AscendC;
using namespace matmul;

namespace
{
constexpr uint32_t BURST_LEN_HIGH = 64;
}

namespace IndexOps
{
class DistanceFlatIPWith950
{
   public:
    __aicore__ inline DistanceFlatIPWith950(const DistanceFlatIPWith950TilingData &tilingData)
        : queryNum(tilingData.queryNum),
          codeNum(tilingData.codeNum),
          dim(tilingData.dim),
          vecCoreNum(tilingData.vecCoreNum),
          cubeCoreNum(tilingData.cubeCoreNum),
          blockIdx(GetBlockIdx()),
          tiling(tilingData.cubeTilingData)
    {
    }

    __aicore__ inline void Init(GM_ADDR query, GM_ADDR mask, GM_ADDR shaped, GM_ADDR actualSize, GM_ADDR dist,
                                GM_ADDR maxDist, GM_ADDR flag, GM_ADDR usrWorkspace);

    __aicore__ inline void Process();

   private:
    __aicore__ inline void SetFlag(uint32_t aivBlkIdx);

    __aicore__ inline void ComputeLoopParameters();

    __aicore__ inline void CopyDist2GM(const LocalTensor<half> &distTensor, uint32_t codeMoveOffset,
                                       uint32_t codeMoveNum, uint32_t queryMoveOffset, uint32_t queryMoveNum);

    __aicore__ inline void DistanceComputeLoop(uint32_t aicoreMoveOffset, uint32_t aicoreCodeNum,
                                               uint32_t queryMoveOffset, uint32_t queryMoveNum, bool isTail);

    __aicore__ inline void CubeComputeLoop(uint32_t codeMoveOffset, uint32_t codeMoveNum, uint32_t queryMoveOffset,
                                           uint32_t queryMoveNum, bool isTail);

    __aicore__ inline void ComputeExtremum(const LocalTensor<half> &dist, uint32_t codeMoveNum, uint32_t queryMoveNum,
                                           LocalTensor<half> &distExtremum);

    __aicore__ inline void CopyDistExtremum2GM(const LocalTensor<half> &distExtremum, uint32_t codeMoveOffset,
                                               uint32_t codeMoveNum, uint32_t queryMoveOffset, uint32_t queryMoveNum);

    __aicore__ inline void GetMaskParameters(GM_ADDR actualSize);

    __aicore__ inline void DoMask(uint32_t codeMoveOffset, uint32_t codeMoveNum, uint32_t queryMoveOffset,
                                  uint32_t queryMoveNum, LocalTensor<half> &dist);

   private:
    uint32_t queryNum;
    uint32_t codeNum;
    uint32_t dim;
    uint32_t vecCoreNum;
    uint32_t cubeCoreNum;
    uint32_t blockIdx;  // cube组号0..27
    TCubeTiling tiling;

    uint32_t burstSizeOfBlock{0};
    uint32_t burstSizeEachLoop{0};

    uint32_t actualCodeNum{0};
    uint32_t burstLen{0};
    uint32_t codeNumEachCore{0};
    uint32_t aicoreCodeMoveOffset{0};
    uint32_t queryNumEachLoop{0};
    uint32_t codeNumEachLoop{0};
    uint32_t maskLenEachLoop{0};
    uint32_t maskBlockOffset{0};
    uint32_t maskLen{0};
    uint32_t maskFlag{0};
    uint32_t selectLoopTime{0};
    uint8_t selectRemainder{0};

    // A/B矩阵均从GM输入，op_host中我们设置每次循环的计算量给singleCore的参数
    // cube的tiling的具体参数会自动计算，它自己控制L1等空间的使用
    using MatmulTypeA = MatmulType<TPosition::GM, CubeFormat::ND, half>;
    // true:B矩阵需要转置
    using MatmulTypeB = MatmulType<TPosition::GM, CubeFormat::ND, half, true>;
    // Ascend950: Cube输出写GM，AIV从GM搬回UB做后处理
    using MatmulTypeC = MatmulType<TPosition::GM, CubeFormat::ND, half>;
    Matmul<MatmulTypeA, MatmulTypeB, MatmulTypeC> matmulObj;

    GlobalTensor<half> queryGM;
    GlobalTensor<uint8_t> maskGM;
    GlobalTensor<half> codeGM;
    GlobalTensor<half> distGM;
    GlobalTensor<uint16_t> flagGM;
    GlobalTensor<half> distMaxGM;

    // matmul结果暂存于GM user workspace，每个block分配一段独立缓冲区
    GlobalTensor<half> distGm;

    TPipe pipe;

    TSCM<TPosition::GM, 1> querySCM;
    TQue<QuePosition::VECIN, 1> maskQue;
    TQue<QuePosition::VECOUT, 1> distQueue;
    TQue<QuePosition::VECOUT, 1> distExtremumQueue;
};

__aicore__ inline void DistanceFlatIPWith950::SetFlag(uint32_t aivBlkIdx)
{
    // 声明KERNEL_TYPE_MIX_AIC_1_2后GetBlockIdx()返回cube组号0..cubeCoreNum-1
    // 参考distance_ivf_flat_ip_fp32：SyncAll后由blockIdx==0统一写所有vecCoreNum个flag
    // 每个flag占CUBE_ALIGN(32B=16个uint16)，stride与topk AICPU的FLAG_SIZE=16对齐
    uint32_t flagElemCnt = vecCoreNum * Utils::CUBE_ALIGN;  // 56 * 16 = 896 uint16_t
    TBuf<> flagBuf;
    pipe.InitBuffer(flagBuf, flagElemCnt * sizeof(uint16_t));
    LocalTensor<uint16_t> flagLocal = flagBuf.Get<uint16_t>(flagElemCnt);
    uint16_t padValue = 0;
    Duplicate(flagLocal, padValue, flagElemCnt);
    for (uint32_t i = 0; i < vecCoreNum; i++)
    {
        flagLocal.SetValue(i * Utils::CUBE_ALIGN, 1);
    }
    // S→MTE3: 等SetValue(S流)完成再DataCopy(MTE3)，参考L2的set_flag(PIPE_S, PIPE_MTE3, EVENT_ID0)
    auto evtSMte3 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::S_MTE3));
    AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(evtSMte3);
    AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(evtSMte3);
    DataCopy(flagGM, flagLocal, flagElemCnt);
}

__aicore__ inline void DistanceFlatIPWith950::GetMaskParameters(GM_ADDR actualSize)
{
    maskBlockOffset = *(reinterpret_cast<__gm__ uint32_t *>(actualSize) + Utils::MASK_BLOCK_OFFSET_IDX);
    maskLen = *(reinterpret_cast<__gm__ uint32_t *>(actualSize) + Utils::MASK_LEN_IDX);
    maskFlag = *(reinterpret_cast<__gm__ uint32_t *>(actualSize) + Utils::MASK_FLAG_IDX);
}

__aicore__ inline void DistanceFlatIPWith950::ComputeLoopParameters()
{
    // 标量直读GM：actualCodeNum从GM矩阵的第0个元素读取
    // 避免UB搬运+wait_flag方向错误（910B写法S→MTE2方向反了，950会挂死）
    // 此处actualSize指针由Init传入，需要用tilingData中的actualSize
    // 注意：这个函数在Init中被调用，actualSize GM地址通过GetMaskParameters同一个指针
    // 所以我们在Init中先保存actualSize指针
    // 实际实现：在Init中直接scalar读取actualCodeNum

    burstLen = BURST_LEN_HIGH;
    // 从tiling中获取query和code分块的参数，在op_host中设置的
    queryNumEachLoop = tiling.M;
    codeNumEachLoop = tiling.N;

    // 按cube组级别均分底库（28份，同组AIC+2AIV共享同一段底库和workspace）
    // 声明KERNEL_TYPE_MIX_AIC_1_2后GetBlockIdx()返回cube组号0..27
    // 2个AIV通过GetSubBlockIdx()分割query段做后处理，写不同最终GM地址
    uint32_t totalLoopTime = actualCodeNum / codeNumEachLoop;
    uint32_t leftCode = actualCodeNum % codeNumEachLoop;
    uint32_t blockCount = cubeCoreNum;  // 28个cube组分活
    uint32_t eachBlockLoopTime = totalLoopTime / blockCount;
    uint32_t leftLoopTime = totalLoopTime % blockCount;
    if (leftLoopTime != 0)
    {
        if (blockIdx < leftLoopTime)
        {
            codeNumEachCore = (eachBlockLoopTime + 1) * codeNumEachLoop;
            aicoreCodeMoveOffset = blockIdx * codeNumEachCore;
        }
        else
        {
            codeNumEachCore = eachBlockLoopTime * codeNumEachLoop;
            aicoreCodeMoveOffset =
                leftLoopTime * (eachBlockLoopTime + 1) * codeNumEachLoop + (blockIdx - leftLoopTime) * codeNumEachCore;
        }
    }
    else
    {
        codeNumEachCore = eachBlockLoopTime * codeNumEachLoop;
        aicoreCodeMoveOffset = blockIdx * codeNumEachCore;
    }
    if (blockIdx == blockCount - 1)
    {
        codeNumEachCore = codeNumEachCore + leftCode;
        aicoreCodeMoveOffset = actualCodeNum - codeNumEachCore;
    }

    // cpp申请空间的时候保证对齐，codeNum能被burstLen整除
    burstSizeOfBlock = codeNum / burstLen * Utils::BURST_BLOCK_RATIO;
    burstSizeEachLoop = codeNumEachLoop / burstLen * Utils::BURST_BLOCK_RATIO;

    maskLenEachLoop = codeNumEachLoop / Utils::MASK_BIT_NUM;
    uint32_t validResultVecRepeats = queryNumEachLoop * codeNumEachLoop / Utils::VIC_HALF_FULL_MASK;
    selectLoopTime = validResultVecRepeats / Utils::SELECT_REPEAT_TIME;
    selectRemainder = static_cast<uint8_t>(validResultVecRepeats % Utils::SELECT_REPEAT_TIME);
}

__aicore__ inline void DistanceFlatIPWith950::Init(GM_ADDR query, GM_ADDR mask, GM_ADDR shaped, GM_ADDR actualSize,
                                                   GM_ADDR dist, GM_ADDR maxDist, GM_ADDR flag, GM_ADDR usrWorkspace)
{
    // 标量直读GM：所有核（AIC和AIV）都执行，避免AIC未初始化导致flagGM空指针
    actualCodeNum = *(reinterpret_cast<__gm__ uint32_t *>(actualSize));

    // AIV核：GetBlockIdx()对AIV返回 2*cubeGroupIdx（0,2,4,...,54），除以2映射回cube组号(0-27)
    // 同cube组的2个AIV共享同一个GetBlockIdx()值，通过GetSubBlockIdx()区分
    if ASCEND_IS_AIV
    {
        blockIdx = GetBlockIdx() / 2;
    }

    // Mask参数与循环参数计算：所有核共用相同循环参数
    GetMaskParameters(actualSize);
    ComputeLoopParameters();

    // GM地址映射：所有核（AIC和AIV）均需执行
    queryGM.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(query), queryNum * dim);
    // mask是已经在searchPaged中对query偏移后的。maskBlockOffset是当前searchPaged中，已经计算过的底库的偏移量
    maskGM.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t *>(mask) + maskBlockOffset / Utils::MASK_BIT_NUM);
    codeGM.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(shaped), dim * codeNum);
    distGM.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(dist), queryNum * codeNum);
    // flag：vecCoreNum(56)个flag区域，每个CUBE_ALIGN(32B=16个uint16)，由blockIdx==0统一写
    flagGM.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t *>(flag), vecCoreNum * Utils::CUBE_ALIGN);
    distMaxGM.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(maxDist), queryNum * burstSizeOfBlock);

    // matmul GM输出缓冲区：从user workspace中按cube组号分配（28份，同组AIC+AIV共享）
    // AIC写matmul结果到distGm[blockIdx]，paired AIV从同一distGm[blockIdx]读取
    // 2个AIV按GetSubBlockIdx分割query段做后处理，写不同最终GM地址
    // workspace参数已经是GetUserWorkspace(workspace)返回的用户workspace地址
    distGm.SetGlobalBuffer(
        reinterpret_cast<__gm__ half *>(usrWorkspace) + blockIdx * queryNumEachLoop * codeNumEachLoop,
        queryNumEachLoop * codeNumEachLoop);
}

__aicore__ inline void DistanceFlatIPWith950::Process()
{
    // 950 MIX 1:2：tiling必须传给REGIST_MATMUL_OBJ（第4参数），不调用Init
    // 参考distance_ivf_flat_ip_fp32.cpp/rotate_and_l2_at_fp32.cpp
    REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), matmulObj, &tiling);

    // UB缓冲区：所有核均需初始化（参考distance_ivf_flat_ip_fp32.cpp，无AIV/AIC守卫）
    // AIC跑matmul(cube指令)，AIV跑后处理(vector指令)，由matmul API内部调度
    pipe.InitBuffer(distQueue, 1, queryNumEachLoop * codeNumEachLoop * sizeof(half));
    pipe.InitBuffer(distExtremumQueue, 1, queryNumEachLoop * burstSizeEachLoop * sizeof(half));
    pipe.InitBuffer(maskQue, 1, queryNumEachLoop * maskLenEachLoop * sizeof(uint8_t));

    // 底库较少时，某些block可能无计算量，空活核直接到SyncAll
    // 参考rotate_and_l2_at_fp32.cpp：if (this->vecLength > 0) { ... } else { 直接SyncAll }
    if (codeNumEachCore > 0)
    {
        // query分块的循环计算次数（AIC和AIV同步执行相同的循环结构）
        uint32_t queryLoopTime = queryNum / queryNumEachLoop;
        for (uint32_t queryLoopIdx = 0; queryLoopIdx < queryLoopTime; queryLoopIdx++)
        {
            DistanceComputeLoop(aicoreCodeMoveOffset, codeNumEachCore, queryLoopIdx * queryNumEachLoop,
                                queryNumEachLoop, false);
        }

        uint32_t queryLastNum = queryNum % queryNumEachLoop;
        if (queryLastNum > 0)
        {
            DistanceComputeLoop(aicoreCodeMoveOffset, codeNumEachCore, queryLoopTime * queryNumEachLoop, queryLastNum,
                                true);
        }
    }

    // 参考distance_ivf_flat_ip_fp32：SyncAll后由blockIdx==0统一写所有vecCoreNum个flag
    SyncAll();
    if (blockIdx == 0)
    {
        SetFlag(0);
    }
}

__aicore__ inline void DistanceFlatIPWith950::DistanceComputeLoop(uint32_t aicoreMoveOffset, uint32_t aicoreCodeNum,
                                                                  uint32_t queryMoveOffset, uint32_t queryMoveNum,
                                                                  bool isTail)
{
    // SetTensorA：所有核均执行，确保matmul握手一致
    GlobalTensor<half> curQueryGM = queryGM[queryMoveOffset * dim];
    matmulObj.SetTensorA(curQueryGM);

    // code分块的循环计算次数
    uint32_t codeLoopTime = aicoreCodeNum / codeNumEachLoop;
    for (uint32_t codeLoopIdx = 0; codeLoopIdx < codeLoopTime; codeLoopIdx++)
    {
        CubeComputeLoop(aicoreMoveOffset + codeLoopIdx * codeNumEachLoop, codeNumEachLoop, queryMoveOffset,
                        queryMoveNum, isTail);
    }

    uint32_t codeLastNum = aicoreCodeNum % codeNumEachLoop;
    if (codeLastNum > 0)
    {
        CubeComputeLoop(aicoreMoveOffset + codeLoopTime * codeNumEachLoop, codeLastNum, queryMoveOffset, queryMoveNum,
                        true);
    }
}

__aicore__ inline void DistanceFlatIPWith950::CopyDist2GM(const LocalTensor<half> &distTensor, uint32_t codeMoveOffset,
                                                          uint32_t codeMoveNum, uint32_t queryMoveOffset,
                                                          uint32_t queryMoveNum)
{
    uint32_t startOffset = queryMoveOffset * codeNum + codeMoveOffset;
    uint16_t nBlock = static_cast<uint16_t>(Utils::DivUp(codeMoveNum, Utils::CUBE_ALIGN));
    DataCopyParams copyParam = {static_cast<uint16_t>(queryMoveNum), nBlock,
                                static_cast<uint16_t>((codeNumEachLoop / Utils::CUBE_ALIGN) - nBlock),
                                static_cast<uint16_t>((codeNum - codeMoveNum) / Utils::CUBE_ALIGN)};
    DataCopy(distGM[startOffset], distTensor, copyParam);
}

__aicore__ inline void DistanceFlatIPWith950::ComputeExtremum(const LocalTensor<half> &dist, uint32_t codeMoveNum,
                                                              uint32_t queryMoveNum, LocalTensor<half> &distExtremum)
{
    half zero = 0;
    Duplicate(distExtremum, zero, distExtremum.GetSize());
    uint32_t srcRepStride = burstLen * sizeof(half) / DEFAULT_C0_SIZE;
    uint32_t repeatTimes = codeMoveNum / burstLen;
    if (repeatTimes > 0)
    {
        for (uint32_t j = 0; j < queryMoveNum; j++)
        {
            WholeReduceMax(distExtremum[j * burstSizeEachLoop], dist[j * codeNumEachLoop], burstLen, repeatTimes, 1, 1,
                           srcRepStride);
        }
    }

    uint32_t lastNum = codeMoveNum % burstLen;
    if (lastNum > 0)
    {
        for (uint32_t j = 0; j < queryMoveNum; j++)
        {
            WholeReduceMax(distExtremum[j * burstSizeEachLoop + repeatTimes * Utils::BURST_BLOCK_RATIO],
                           dist[j * codeNumEachLoop + repeatTimes * burstLen], lastNum, 1, 1, 1, srcRepStride);
        }
    }
}

__aicore__ inline void DistanceFlatIPWith950::CopyDistExtremum2GM(const LocalTensor<half> &distExtremum,
                                                                  uint32_t codeMoveOffset, uint32_t codeMoveNum,
                                                                  uint32_t queryMoveOffset, uint32_t queryMoveNum)
{
    uint32_t dstOffset = queryMoveOffset * burstSizeOfBlock + codeMoveOffset / burstLen * Utils::BURST_BLOCK_RATIO;
    uint32_t burstNum = Utils::DivUp(codeMoveNum, burstLen);
    uint32_t blocks = Utils::DivUp(burstNum * Utils::BURST_BLOCK_RATIO, Utils::CUBE_ALIGN);
    DataCopyParams copyParam = {static_cast<uint16_t>(queryMoveNum), static_cast<uint16_t>(blocks),
                                static_cast<uint16_t>(burstSizeEachLoop / Utils::CUBE_ALIGN - blocks),
                                static_cast<uint16_t>(burstSizeOfBlock / Utils::CUBE_ALIGN - blocks)};
    DataCopy(distMaxGM[dstOffset], distExtremum, copyParam);
}

__aicore__ inline void DistanceFlatIPWith950::DoMask(uint32_t codeMoveOffset, uint32_t codeMoveNum,
                                                     uint32_t queryMoveOffset, uint32_t queryMoveNum,
                                                     LocalTensor<half> &dist)
{
    if (maskFlag == 0)
    {
        return;
    }
    auto maskLocal = maskQue.AllocTensor<uint8_t>();
    uint64_t maskOffset = queryMoveOffset * static_cast<uint64_t>(maskLen) + codeMoveOffset / Utils::MASK_BIT_NUM;

    for (uint32_t i = 0; i < queryMoveNum; i++)
    {
        DataCopy(maskLocal[i * maskLenEachLoop], maskGM[i * static_cast<uint64_t>(maskLen) + maskOffset],
                 maskLenEachLoop);
    }
    maskQue.EnQue(maskLocal);
    maskLocal = maskQue.DeQue<uint8_t>();

    BinaryRepeatParams param{1, 1, 1, 8, 8, 8};

    const uint32_t distOffset = Utils::SELECT_REPEAT_TIME * Utils::VIC_HALF_FULL_MASK;
    const uint32_t maskRepateOffset = distOffset / Utils::MASK_BIT_NUM;
    for (uint32_t i = 0; i < selectLoopTime; i++)
    {
        Select(dist[i * distOffset], maskLocal[i * maskRepateOffset], dist[i * distOffset], Utils::HALF_MIN,
               SELMODE::VSEL_TENSOR_SCALAR_MODE, Utils::VIC_HALF_FULL_MASK, Utils::SELECT_REPEAT_TIME, param);
    }
    if (selectRemainder != 0)
    {
        Select(dist[selectLoopTime * distOffset], maskLocal[selectLoopTime * maskRepateOffset],
               dist[selectLoopTime * distOffset], Utils::HALF_MIN, SELMODE::VSEL_TENSOR_SCALAR_MODE,
               Utils::VIC_HALF_FULL_MASK, selectRemainder, param);
    }

    maskQue.FreeTensor(maskLocal);
}

__aicore__ inline void DistanceFlatIPWith950::CubeComputeLoop(uint32_t codeMoveOffset, uint32_t codeMoveNum,
                                                              uint32_t queryMoveOffset, uint32_t queryMoveNum,
                                                              bool isTail)
{
    // === 所有核：matmul计算 ===
    GlobalTensor<half> curLoopCodeGM = codeGM[codeMoveOffset * dim];
    matmulObj.SetTensorB(curLoopCodeGM, true);
    if (isTail)
    {
        matmulObj.SetTail(queryMoveNum, codeMoveNum, dim);
    }
    // GM输出模式：AIC驱动异步matmul，结果写入GM
    matmulObj.IterateAll<false>(distGm, 0, false, true);
    matmulObj.WaitIterateAll();
    // End在每次WaitIterateAll后调用，参考distance_ivf_flat_ip_fp32.cpp的WaitGemmQB
    // 不调用End会导致下一次IterateAll挂起（matmul内部资源未释放）
    matmulObj.End();

    // === AIV核：从GM搬回UB做后处理（mask + extremum + 写回GM）===
    // 只让subBlockIdx==1的AIV跑后处理，避免同组2个AIV写同一GM地址
    // 另一个AIV空转，等待matmul握手完成即可
    if ASCEND_IS_AIV
    {
        if (GetSubBlockIdx() == 1)
        {
            LocalTensor<half> distTensor = distQueue.AllocTensor<half>();
            LocalTensor<half> distExtremumensor = distExtremumQueue.AllocTensor<half>();

            // GM→UB搬运matmul结果
            DataCopy(distTensor, distGm, queryMoveNum * codeMoveNum);
            // MTE2→S: 等DataCopy(GM→UB)完成，参考L2的set_flag(PIPE_MTE2, PIPE_S, EVENT_ID6)
            auto evtMte2S = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_S));
            AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(evtMte2S);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(evtMte2S);

            DoMask(codeMoveOffset, codeMoveNum, queryMoveOffset, queryMoveNum, distTensor);

            ComputeExtremum(distTensor, codeMoveNum, queryMoveNum, distExtremumensor);

            // V→MTE3: 等Vector(DoMask+WholeReduceMax)完成再搬回GM，参考rotate_and_l2_at_fp32.cpp
            auto evtVMte3 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_MTE3));
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(evtVMte3);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(evtVMte3);

            CopyDist2GM(distTensor, codeMoveOffset, codeMoveNum, queryMoveOffset, queryMoveNum);
            CopyDistExtremum2GM(distExtremumensor, codeMoveOffset, codeMoveNum, queryMoveOffset, queryMoveNum);

            distQueue.FreeTensor(distTensor);
            distExtremumQueue.FreeTensor(distExtremumensor);

            // MTE3→MTE2: 等UB→GM搬运完成再进入下一轮MTE2，参考L2的set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID2)
            auto evtMte3Mte2 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE3_MTE2));
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(evtMte3Mte2);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(evtMte3Mte2);
        }  // end if (GetSubBlockIdx() == 1)
    }  // end if ASCEND_IS_AIV
}

}  // namespace IndexOps

extern "C" __global__ __aicore__ void distance_flat_ip_with_950(GM_ADDR query, GM_ADDR mask, GM_ADDR shaped,
                                                                GM_ADDR actualSize, GM_ADDR dist, GM_ADDR maxDist,
                                                                GM_ADDR flag, GM_ADDR workspace, GM_ADDR tiling)
{
    if (GetSysWorkSpacePtr() == nullptr)
    {
        return;
    }
    // 声明MIX 1:2模型：28 AIC + 56 AIV，GetBlockIdx()返回cube组号0..cubeCoreNum-1
    // 参考distance_ivf_flat_ip_fp32.cpp/rotate_and_l2_at_fp32.cpp
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    GET_TILING_DATA(tilingData, tiling);
    // 必须从extern函数的workspace参数获取用户workspace，不能用GetSysWorkSpacePtr()
    // GetUserWorkspace(GetSysWorkSpacePtr())返回的是系统workspace地址，Fixpipe写会报161
    GM_ADDR usrWorkspace = GetUserWorkspace(workspace);
    IndexOps::DistanceFlatIPWith950With950 op(tilingData);
    op.Init(query, mask, shaped, actualSize, dist, maxDist, flag, usrWorkspace);
    op.Process();
}
