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

#ifndef ASCENDC_IVFPQ_SEARCH_DISTANCE_TOPK_H
#define ASCENDC_IVFPQ_SEARCH_DISTANCE_TOPK_H
#include <limits>
#include "kernel_operator.h"

using namespace AscendC;

namespace IndexOps {

constexpr uint32_t IVFPQ_FLAG_ALIGN = 16;
constexpr uint32_t IVFPQ_THREAD_NUM = 1024;
constexpr uint8_t IVFPQ_UNROLL_FACTOR = 4;
constexpr uint32_t IVFPQ_CODE_BLOCK_SIZE = 262144;
constexpr uint32_t IVFPQ_BLOCK_MAX_SIZE = 16384;

constexpr float IVFPQ_MAX_FLOAT = std::numeric_limits<float>::max();
constexpr float IVFPQ_MIN_FLOAT = -std::numeric_limits<float>::max();

template <uint8_t NumSubQuantizers>
__aicore__ inline float SimtSearchTableSumV1(
    __ubuf__ float *queryPQ, __gm__ uint8_t *codeBase, int64_t curBlockIndex, uint32_t taskIdx, uint32_t ksub)
{
    uint8_t codeBookIndices[IVFPQ_UNROLL_FACTOR];
    float sumDist = 0;
    uint8_t codeBookIndex = 0;
    uint8_t m = 0;
#pragma unroll
    for (; m + IVFPQ_UNROLL_FACTOR <= NumSubQuantizers; m += IVFPQ_UNROLL_FACTOR) {
// 先批量读取indices，让后续访问可以并行
#pragma unroll
        for (uint8_t i = 0; i < IVFPQ_UNROLL_FACTOR; i++) {
            codeBookIndices[i] = codeBase[curBlockIndex + taskIdx * NumSubQuantizers + m + i];
        }
// 再批量访问queryPQ和累加，减少数据依赖
#pragma unroll
        for (int i = 0; i < IVFPQ_UNROLL_FACTOR; i++) {
            sumDist += queryPQ[(m + i) * ksub + codeBookIndices[i]];
        }
    }

// 处理剩余部分
#pragma unroll
    for (; m < NumSubQuantizers; m++) {
        codeBookIndex = codeBase[curBlockIndex + taskIdx * NumSubQuantizers + m];
        sumDist += queryPQ[m * ksub + codeBookIndex];
    }
    return sumDist;
}

template <int NumSubQuantizers>
__simt_vf__ __aicore__ LAUNCH_BOUND(IVFPQ_THREAD_NUM) inline void SimtSearchComputeV1(__ubuf__ float *queryPQ,
    __gm__ uint8_t *codeBase, __ubuf__ float *distance, __gm__ int64_t *codeSize, __gm__ int64_t *codeOffset,
    uint32_t ksub, uint32_t codeBlockNum, uint32_t perCoreInnerBlockDealSize, float distResultInitValue,
    uint32_t innerBlockindex, uint32_t blockIndex, uint32_t usedAivNum)
{
    uint32_t curBlockThreadIdx = static_cast<uint32_t>(Simt::GetThreadIdx<0>());
    uint32_t threadCount = static_cast<uint32_t>(Simt::GetThreadNum<0>());

    uint32_t curBlockIdx1 = Simt::GetBlockIdx();
    uint32_t blockCount = Simt::GetBlockNum();

    uint32_t curBlockNum = codeSize[blockIndex * usedAivNum + curBlockIdx1];

    uint32_t startTaskId = perCoreInnerBlockDealSize * innerBlockindex;
    uint32_t endtaskId = Simt::Min(perCoreInnerBlockDealSize * (innerBlockindex + 1), curBlockNum);

    // 当前block的起始地址
    int64_t curBlockIndex = codeOffset[blockIndex * usedAivNum + curBlockIdx1];

    for (uint32_t taskIdx = curBlockThreadIdx + startTaskId; taskIdx < endtaskId; taskIdx += threadCount) {
        uint32_t dstIndex = taskIdx - startTaskId;
        distance[dstIndex] = SimtSearchTableSumV1<NumSubQuantizers>(queryPQ, codeBase, curBlockIndex, taskIdx, ksub);
    }

    AscendC::Simt::ThreadBarrier();
}

class AscendcIvfpqSearchDistanceTopK {
public:
    __aicore__ inline AscendcIvfpqSearchDistanceTopK(){};
    __aicore__ inline ~AscendcIvfpqSearchDistanceTopK(){};
    __aicore__ inline void Init(GM_ADDR queryPQ, GM_ADDR codeBase, GM_ADDR codeOffset, GM_ADDR codeSize,
        GM_ADDR distResult, GM_ADDR topk, GM_ADDR topkIndex, GM_ADDR topkValue, GM_ADDR flag, GM_ADDR workspace,
        const AscendcIvfpqSearchDistanceTopKTilingData *__restrict tilingData, TPipe *tPipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ParseTilingData(const AscendcIvfpqSearchDistanceTopKTilingData *__restrict tilingData);
    __aicore__ inline void InitReduceUbTensor();
    __aicore__ inline void WholeBlockTopK(LocalTensor<float> dist_local_value, LocalTensor<int32_t> dist_local_index,
        LocalTensor<float> src_local_value, LocalTensor<int32_t> src_local_index, LocalTensor<uint8_t> tmp_local_whole,
        LocalTensor<uint8_t> tmp_local_single, int innerBlockindex);
    __aicore__ inline void SingleBlockTopKAndIndexAlign(LocalTensor<float> dst_local_value,
        LocalTensor<int32_t> dst_local_index, LocalTensor<float> distResultUb, LocalTensor<uint8_t> tmp_local,
        int innerBlockindex);

private:
    TPipe pipe;

    GlobalTensor<float> queryPQGm;
    GlobalTensor<uint8_t> codeBaseGm;
    GlobalTensor<int64_t> codeOffsetGm;
    GlobalTensor<int64_t> codeSizeGm;
    GlobalTensor<float> distResultGm;
    GlobalTensor<uint16_t> flagGm;

    GlobalTensor<int32_t> topkIndexGm;
    GlobalTensor<float> topkValueGm;

    TQue<AscendC::TPosition::VECIN, 1> distResultQueue, queryPQUbQueue;

    AscendC::TBuf<AscendC::TPosition::VECCALC> top_k_tmp_buf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> top_k_result_value_buf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> top_k_result_index_buf;

    AscendC::TBuf<AscendC::TPosition::VECCALC> top_k_tmp_whole_buf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> top_k_tmp_single_buf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> top_k_result_value_buf_whole;
    AscendC::TBuf<AscendC::TPosition::VECCALC> top_k_result_index_buf_whole;

    AscendC::TBuf<AscendC::TPosition::VECCALC> top_k_finish_local_buf_whole;

    AscendC::TBuf<AscendC::TPosition::VECCALC> flag_buf;

    int64_t blockIdx = 0;
    uint32_t aivNum = 0;
    uint32_t usedAivNum = 0;
    uint32_t headAivNum = 0;

    uint32_t reduceMode = 0;
    uint32_t subSpaceNum = 0;
    uint32_t ksub = 0;
    uint32_t codeBaseSize = 0;
    uint32_t codeBlockSize = 0;
    uint32_t codeBlockNum = 0;
    float distResultInitValue = 0;
    uint32_t perCoreInnerBlockDealSize = 0;

    uint32_t min_size = 0;
    uint32_t single_core_total_block = 0;
    uint32_t min_size_whole = 0;
    uint32_t min_size_single = 0;

    TopkTiling topkTilingData;
    TopkTiling topkTilingDataWhole;
    TopkTiling topkTilingDataSingle;

    uint32_t topk = 0;
    uint32_t topk_outter_num = 0;

    uint32_t top_k_result_value_buf_num = 0;
    uint32_t top_k_result_value_single_num = 0;

    bool isLargest;
};

__aicore__ inline void AscendcIvfpqSearchDistanceTopK::Init(GM_ADDR queryPQ, GM_ADDR codeBase, GM_ADDR codeOffset,
    GM_ADDR codeSize, GM_ADDR topk, GM_ADDR distResult, GM_ADDR topkIndex, GM_ADDR topkValue, GM_ADDR flag,
    GM_ADDR workspace, const AscendcIvfpqSearchDistanceTopKTilingData *__restrict tilingData, TPipe *tPipe)
{
    blockIdx = GetBlockIdx();
    if (tPipe == nullptr || tilingData == nullptr) {
        return;
    }
    pipe = *tPipe;

    this->queryPQGm.SetGlobalBuffer((__gm__ float *)queryPQ);
    this->codeBaseGm.SetGlobalBuffer((__gm__ uint8_t *)codeBase);
    this->codeOffsetGm.SetGlobalBuffer((__gm__ int64_t *)codeOffset);
    this->codeSizeGm.SetGlobalBuffer((__gm__ int64_t *)codeSize);
    this->distResultGm.SetGlobalBuffer((__gm__ float *)distResult);
    this->topkIndexGm.SetGlobalBuffer((__gm__ int32_t *)topkIndex);
    this->topkValueGm.SetGlobalBuffer((__gm__ float *)topkValue);
    this->flagGm.SetGlobalBuffer((__gm__ uint16_t *)flag);

    ParseTilingData(tilingData);
    InitReduceUbTensor();
}

__aicore__ inline void AscendcIvfpqSearchDistanceTopK::ParseTilingData(
    const AscendcIvfpqSearchDistanceTopKTilingData *__restrict tilingData)
{
    this->codeBlockSize = 0;
    for (int i = 0; i < tilingData->codeBlockNum; i++) {
        int64_t codeBlockSizeTmp = codeSizeGm.GetValue(i);
        if (this->codeBlockSize < codeBlockSizeTmp) {
            this->codeBlockSize = codeBlockSizeTmp;
        }
    }
    this->codeBlockSize = (this->codeBlockSize + IVFPQ_BLOCK_MAX_SIZE - 1) /
            IVFPQ_BLOCK_MAX_SIZE * IVFPQ_BLOCK_MAX_SIZE;
    this->usedAivNum = tilingData->usedAivNum;

    this->reduceMode = tilingData->reduceMode;
    this->subSpaceNum = tilingData->subSpaceNum;
    this->ksub = tilingData->ksub;
    this->codeBaseSize = tilingData->codeBaseSize;
    this->codeBlockNum = tilingData->codeBlockNum;
    this->perCoreInnerBlockDealSize = tilingData->perCoreInnerBlockDealSize;

    if (reduceMode == 0) {
        this->distResultInitValue = IVFPQ_MAX_FLOAT;
        this->isLargest = false;
    } else {
        this->distResultInitValue = IVFPQ_MIN_FLOAT;
        this->isLargest = true;
    }

    this->min_size = tilingData->minSize;

    this->min_size_whole = tilingData->minSizeWhole;
    this->min_size_single = tilingData->minSizeSingle;
    this->single_core_total_block = tilingData->singleCoretotalBlock;

    this->topkTilingData = tilingData->topkTilingData;              // TopK操作的第一组tiling数据
    this->topkTilingDataWhole = tilingData->topkTilingDataWhole;    // TopK操作的第一组tiling数据
    this->topkTilingDataSingle = tilingData->topkTilingDataSingle;  // TopK操作的第一组tiling数据

    this->topk = tilingData->topk;
    this->topk_outter_num = tilingData->topkOutterNum;
    this->top_k_result_value_buf_num = topk * (single_core_total_block + 1);
    this->top_k_result_value_single_num = topk * single_core_total_block;
}

__aicore__ inline void AscendcIvfpqSearchDistanceTopK::InitReduceUbTensor()
{
    pipe.InitBuffer(top_k_result_value_buf, top_k_result_value_buf_num * sizeof(float));    // 4.5k
    pipe.InitBuffer(top_k_result_index_buf, top_k_result_value_buf_num * sizeof(int32_t));  // 4.5k

    pipe.InitBuffer(top_k_result_value_buf_whole, topk * sizeof(float));    // 1.125k
    pipe.InitBuffer(top_k_result_index_buf_whole, topk * sizeof(int32_t));  // 1.125k

    pipe.InitBuffer(queryPQUbQueue, 1, this->ksub * this->subSpaceNum * sizeof(float));    // 4k
    pipe.InitBuffer(distResultQueue, 1, this->perCoreInnerBlockDealSize * sizeof(float));  // 64k

    pipe.InitBuffer(flag_buf, IVFPQ_FLAG_ALIGN * sizeof(uint16_t));  // TopK临时缓冲区

    pipe.InitBuffer(top_k_tmp_buf, this->min_size * sizeof(uint8_t));                // TopK临时缓冲区
    pipe.InitBuffer(top_k_tmp_whole_buf, this->min_size_whole * sizeof(uint8_t));    // TopK临时缓冲区
    pipe.InitBuffer(top_k_tmp_single_buf, this->min_size_single * sizeof(uint8_t));  // TopK临时缓冲区
}

__aicore__ inline void AscendcIvfpqSearchDistanceTopK::SingleBlockTopKAndIndexAlign(LocalTensor<float> dst_local_value,
    LocalTensor<int32_t> dst_local_index, LocalTensor<float> distResultUb, LocalTensor<uint8_t> tmp_local,
    int innerBlockindex)
{
    static constexpr TopKConfig topkConfig{TopKAlgo::RADIX_SELECT, TopKOrder::UNSET, false};
    for (uint32_t index = 0; index < this->single_core_total_block; index++) {
        LocalTensor<int32_t> src_local_index;  // 源索引缓冲区（不需要输入）
        LocalTensor<bool> src_local_finish;    // 源完成标志（不需要输入）

        TopKInfo topKInfo;
        topKInfo.outter = 1;                     // 外层循环次数
        topKInfo.inner = this->topk_outter_num;  // 内层块大小
        topKInfo.n = this->topk_outter_num;      // 数据总数
                                                 // TopK临时缓冲区

        // log_probs_gm(行339-349): 执行第一次TopK操作，从当前ub中选出top_k候选
        AscendC::TopK<float, false, false, false, AscendC::TopKMode::TOPK_NORMAL, topkConfig>(
            dst_local_value[index * topk],
            dst_local_index[index * topk],
            distResultUb[index * this->topk_outter_num],
            src_local_index,
            src_local_finish,
            tmp_local,
            this->topk,
            this->topkTilingData,
            topKInfo,
            this->isLargest);
    }
    AscendC::PipeBarrier<PIPE_V>();

    for (uint32_t index = 0; index < this->single_core_total_block; index++) {
        AscendC::Adds(dst_local_index[index * this->topk],
            dst_local_index[index * this->topk],
            index * this->topk_outter_num + innerBlockindex * this->perCoreInnerBlockDealSize,
            this->topk);
    }
    AscendC::PipeBarrier<PIPE_V>();
}

__aicore__ inline void AscendcIvfpqSearchDistanceTopK::WholeBlockTopK(LocalTensor<float> dist_local_value,
    LocalTensor<int32_t> dist_local_index, LocalTensor<float> src_local_value, LocalTensor<int32_t> src_local_index,
    LocalTensor<uint8_t> tmp_local_whole, LocalTensor<uint8_t> tmp_local_single, int innerBlockindex)
{
    static constexpr TopKConfig topkConfigSort{TopKAlgo::RADIX_SELECT, TopKOrder::UNSET, true};
    if (innerBlockindex > 0) {
        // 需要两次对比
        TopKInfo topKInfoWhole;
        topKInfoWhole.outter = 1;
        topKInfoWhole.inner = this->top_k_result_value_buf_num;
        topKInfoWhole.n = this->top_k_result_value_buf_num;
        LocalTensor<bool> src_local_finish_whole;

        AscendC::TopK<float, true, false, false, AscendC::TopKMode::TOPK_NORMAL, topkConfigSort>(dist_local_value,
            dist_local_index,
            src_local_value,
            src_local_index,
            src_local_finish_whole,
            tmp_local_whole,
            topk,
            this->topkTilingDataWhole,
            topKInfoWhole,
            this->isLargest);
    } else {
        // 只对比当前轮次
        TopKInfo topKInfoWhole;
        topKInfoWhole.outter = 1;                             // 外层循环次数
        topKInfoWhole.inner = top_k_result_value_single_num;  // 内层块大小
        topKInfoWhole.n = top_k_result_value_single_num;      // 数据总数
                                                              // TopK临时缓冲区
        LocalTensor<bool> src_local_finish_whole;

        AscendC::TopK<float, true, false, false, AscendC::TopKMode::TOPK_NORMAL, topkConfigSort>(dist_local_value,
            dist_local_index,
            src_local_value,
            src_local_index,
            src_local_finish_whole,
            tmp_local_single,
            topk,
            this->topkTilingDataSingle,
            topKInfoWhole,
            this->isLargest);
    }
    AscendC::PipeBarrier<PIPE_ALL>();
}

__aicore__ inline void AscendcIvfpqSearchDistanceTopK::Process()
{
    LocalTensor<float> queryPQEnqueUb = queryPQUbQueue.AllocTensor<float>();
    AscendC::DataCopy(queryPQEnqueUb, queryPQGm, this->ksub * this->subSpaceNum);
    queryPQUbQueue.EnQue(queryPQEnqueUb);
    LocalTensor<float> queryPQDequeUb = queryPQUbQueue.DeQue<float>();
    __ubuf__ float *queryPQUb = (__ubuf__ float *)queryPQDequeUb.GetPhyAddr();

    uint32_t perCoreBlockNum = (codeBlockNum + usedAivNum - 1) / usedAivNum;
    uint32_t perCoreInnerBlockLoopNum =
        (codeBlockSize + this->perCoreInnerBlockDealSize - 1) / this->perCoreInnerBlockDealSize;

    LocalTensor<float> dst_local_value = top_k_result_value_buf.Get<float>();                  // TopK结果值
    LocalTensor<int32_t> dst_local_index = top_k_result_index_buf.Get<int32_t>();              // TopK结果索引
    LocalTensor<float> dst_local_value_whole = top_k_result_value_buf_whole.Get<float>();      // TopK结果值
    LocalTensor<int32_t> dst_local_index_whole = top_k_result_index_buf_whole.Get<int32_t>();  // TopK结果索引

    LocalTensor<uint8_t> tmp_local = top_k_tmp_buf.Get<uint8_t>();
    LocalTensor<uint8_t> tmp_local_single = top_k_tmp_single_buf.Get<uint8_t>();
    LocalTensor<uint8_t> tmp_local_whole = top_k_tmp_whole_buf.Get<uint8_t>();
    LocalTensor<uint16_t> flag_local = flag_buf.Get<uint16_t>();
    AscendC::Duplicate<uint16_t>(flag_local, 1, IVFPQ_FLAG_ALIGN);

    __gm__ uint8_t *codeBase = (__gm__ uint8_t *)this->codeBaseGm.GetPhyAddr();
    __gm__ int64_t *codeSize = (__gm__ int64_t *)this->codeSizeGm.GetPhyAddr();
    __gm__ int64_t *codeOffset = (__gm__ int64_t *)this->codeOffsetGm.GetPhyAddr();

    for (int blockIndex = 0; blockIndex < perCoreBlockNum; blockIndex++) {
        uint32_t blockOffsetNum = blockIdx + blockIndex * usedAivNum;
        for (int innerBlockindex = 0; innerBlockindex < perCoreInnerBlockLoopNum; innerBlockindex++) {
            LocalTensor<float> distResultUb = distResultQueue.AllocTensor<float>();
            AscendC::Duplicate<float>(distResultUb, this->distResultInitValue, this->perCoreInnerBlockDealSize);
            AscendC::PipeBarrier<PIPE_V>();
            __ubuf__ float *distResultUbPtr = (__ubuf__ float *)distResultUb.GetPhyAddr();
            if (this->subSpaceNum == 4) {
                Simt::VF_CALL<SimtSearchComputeV1<4>>(Simt::Dim3{IVFPQ_THREAD_NUM, 1, 1},
                    queryPQUb,
                    codeBase,
                    distResultUbPtr,
                    codeSize,
                    codeOffset,
                    this->ksub,
                    this->codeBlockNum,
                    this->perCoreInnerBlockDealSize,
                    this->distResultInitValue,
                    innerBlockindex,
                    blockIndex,
                    usedAivNum);
                AscendC::PipeBarrier<PIPE_ALL>();
            }
            SingleBlockTopKAndIndexAlign(dst_local_value, dst_local_index, distResultUb, tmp_local, innerBlockindex);
            WholeBlockTopK(dst_local_value_whole,
                dst_local_index_whole,
                dst_local_value,
                dst_local_index,
                tmp_local_whole,
                tmp_local_single,
                innerBlockindex);

            AscendC::DataCopy(
                dst_local_value[top_k_result_value_single_num], dst_local_value_whole, topk);  // 从GM->VECIN搬运40字节
            AscendC::DataCopy(
                dst_local_index[top_k_result_value_single_num], dst_local_index_whole, topk);  // 从GM->VECIN搬运40字节

            AscendC::DataCopy(distResultGm[(blockOffsetNum)*IVFPQ_CODE_BLOCK_SIZE +
                                           codeBlockSize / perCoreInnerBlockLoopNum * innerBlockindex],
                distResultUb,
                this->perCoreInnerBlockDealSize);
            AscendC::SyncAll();
            distResultQueue.FreeTensor(distResultUb);
        }
        AscendC::DataCopy(topkValueGm[(blockOffsetNum)*topk], dst_local_value_whole, topk);
        AscendC::DataCopy(topkIndexGm[(blockOffsetNum)*topk], dst_local_index_whole, topk);
        AscendC::DataCopy(flagGm[blockOffsetNum * IVFPQ_FLAG_ALIGN], flag_local, IVFPQ_FLAG_ALIGN);
    }
    queryPQUbQueue.FreeTensor(queryPQDequeUb);
}

}  // namespace IndexOps
#endif
