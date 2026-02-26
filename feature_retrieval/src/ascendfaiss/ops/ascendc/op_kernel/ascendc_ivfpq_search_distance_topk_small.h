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

#ifndef ASCENDC_IVFPQ_SEARCH_DISTANCE_TOPK_SMALL_H
#define ASCENDC_IVFPQ_SEARCH_DISTANCE_TOPK_SMALL_H
#include <limits>

#include "kernel_operator.h"

using namespace AscendC;

namespace IndexOps {

template <uint8_t NumSubQuantizers>
__aicore__ inline float SimtSearchTableSumSmallV1(__ubuf__ float *queryPQ, __gm__ uint8_t *codeBase,
                                                  int64_t curBlockIndex, uint32_t taskIdx, uint32_t ksub)
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
__simt_vf__ __aicore__ LAUNCH_BOUND(IVFPQ_THREAD_NUM) inline void SimtSearchComputeSmallV1(
    __ubuf__ float *queryPQ, __gm__ uint8_t *codeBase, __ubuf__ float *distance, __gm__ int64_t *codeSize,
    __gm__ int64_t *codeOffset, uint32_t ksub, uint32_t codeBlockNum, uint32_t perCoreInnerBlockDealSize,
    float distResultInitValue, uint32_t innerBlockindex, uint32_t blockIndex, uint32_t usedAivNum)
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
        distance[dstIndex] =
            SimtSearchTableSumSmallV1<NumSubQuantizers>(queryPQ, codeBase, curBlockIndex, taskIdx, ksub);
    }

    AscendC::Simt::ThreadBarrier();
}

__simt_vf__ __aicore__ LAUNCH_BOUND(512) inline void SimtSearchGatherSmallV1(__ubuf__ int32_t *topkMergeEndDisIndex,
                                                                             __gm__ int64_t *labelOffsetGm,
                                                                             __gm__ uint64_t *labelBaseGm,
                                                                             __gm__ uint64_t *topkMergeEndLabel,
                                                                             uint32_t topk)
{
    uint32_t curBlockThreadIdx = static_cast<uint32_t>(Simt::GetThreadIdx<0>());
    uint32_t threadCount = static_cast<uint32_t>(Simt::GetThreadNum<0>());

    for (uint32_t i = curBlockThreadIdx; i < topk; i += threadCount) {
        int32_t topkIndex = topkMergeEndDisIndex[i];
        int32_t labelBlockId = topkIndex / IVFPQ_CODE_BLOCK_SIZE;
        int32_t labelInnerBlockId = topkIndex % IVFPQ_CODE_BLOCK_SIZE;
        int64_t labelOffset = labelOffsetGm[labelBlockId];
        topkMergeEndLabel[i] = labelBaseGm[labelOffset + labelInnerBlockId];
    }

    AscendC::Simt::ThreadBarrier();
}

class AscendcIvfpqSearchDistanceTopKSmall {
public:
    __aicore__ inline AscendcIvfpqSearchDistanceTopKSmall(){};
    __aicore__ inline ~AscendcIvfpqSearchDistanceTopKSmall(){};
    __aicore__ inline void Init(GM_ADDR queryPQ, GM_ADDR codeBase, GM_ADDR codeOffset, GM_ADDR codeSize, GM_ADDR topk,
                                GM_ADDR labelBase, GM_ADDR labelOffset, GM_ADDR distResult, GM_ADDR topkIndex,
                                GM_ADDR topkValue, GM_ADDR topkLabelFinal, GM_ADDR topkValueFinal, GM_ADDR flag,
                                GM_ADDR workspace,
                                const AscendcIvfpqSearchDistanceTopKTilingData *__restrict tilingData, TPipe *tPipe);
    __aicore__ inline void Process();
    __aicore__ inline void ProcessMerge();

private:
    __aicore__ inline void ParseTilingData(const AscendcIvfpqSearchDistanceTopKTilingData *__restrict tilingData);
    __aicore__ inline void InitReduceUbTensor();
    __aicore__ inline void WholeBlockTopK(LocalTensor<float> dist_local_value, LocalTensor<int32_t> dist_local_index,
                                          LocalTensor<float> src_local_value, LocalTensor<int32_t> src_local_index,
                                          LocalTensor<uint8_t> tmp_local_whole, LocalTensor<uint8_t> tmp_local_single,
                                          int innerBlockindex);
    __aicore__ inline void SingleBlockTopKAndIndexAlign(LocalTensor<float> dst_local_value,
                                                        LocalTensor<int32_t> dst_local_index,
                                                        LocalTensor<float> distResultUb, LocalTensor<uint8_t> tmp_local,
                                                        int innerBlockindex);
    __aicore__ inline void DistCompute(__ubuf__ float *queryPQUb, __gm__ uint8_t *codeBase, __gm__ int64_t *codeSize,
                                       __gm__ int64_t *codeOffset, __ubuf__ float *distResultUbPtr, int innerBlockindex,
                                       int blockIndex);

private:
    TPipe pipe;

    GlobalTensor<float> queryPQGm;
    GlobalTensor<uint8_t> codeBaseGm;
    GlobalTensor<int64_t> codeOffsetGm;
    GlobalTensor<int64_t> codeSizeGm;

    GlobalTensor<uint64_t> labelBaseGm;
    GlobalTensor<int64_t> labelOffsetGm;

    GlobalTensor<float> distResultGm;

    GlobalTensor<int32_t> topkIndexGm;
    GlobalTensor<float> topkValueGm;

    GlobalTensor<uint64_t> topkLabelFinalGm;
    GlobalTensor<float> topkValueFinalGm;

    GlobalTensor<uint16_t> flagGm;

    TQue<AscendC::TPosition::VECIN, 1> distResultQueue, queryPQUbQueue;

    AscendC::TBuf<AscendC::TPosition::VECCALC> top_k_tmp_buf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> top_k_result_value_buf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> top_k_result_index_buf;

    AscendC::TBuf<AscendC::TPosition::VECCALC> top_k_tmp_whole_buf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> top_k_tmp_single_buf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> top_k_result_value_buf_whole;
    AscendC::TBuf<AscendC::TPosition::VECCALC> top_k_result_index_buf_whole;

    AscendC::TBuf<AscendC::TPosition::VECCALC> flag_buf;

    // 从gm获取block内的topk结果
    TQue<AscendC::TPosition::VECIN, 1> top_k_merge_src_value_que;
    TQue<AscendC::TPosition::VECIN, 1> top_k_merge_src_index_que;

    // 用于存储首轮topk的结果，包括尾块
    AscendC::TBuf<AscendC::TPosition::VECCALC> top_k_merge_begin_dst_value_buf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> top_k_merge_begin_dst_index_buf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> top_k_merge_begin_temp_buf;

    // 用于存储尾轮归并结果
    AscendC::TBuf<AscendC::TPosition::VECCALC> top_k_merge_end_dst_value_buf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> top_k_merge_end_dst_index_buf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> top_k_merge_end_temp_buf;

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

    TopkTiling topkTilingDataMergeBegin;
    TopkTiling topkTilingDataMergeBeginTail;
    TopkTiling topkTilingDataMergeEnd;

    uint32_t mergeBeginTopNumInLoop = 0;
    uint32_t mergeBeginTailTopNumInLoop = 0;
    uint32_t mergeBeginBlockLoopTime = 0;
    uint32_t mergeBeginTailBlockLoopTime = 0;
    uint32_t minSizeMergeBegin = 0;
    uint32_t minSizeMergeBeginTail = 0;
    uint32_t minSizeMergeEnd = 0;

    bool isLargest;

    uint32_t batch = 0;
};

__aicore__ inline void AscendcIvfpqSearchDistanceTopKSmall::Init(
    GM_ADDR queryPQ, GM_ADDR codeBase, GM_ADDR codeOffset, GM_ADDR codeSize, GM_ADDR topk, GM_ADDR labelBase,
    GM_ADDR labelOffset, GM_ADDR distResult, GM_ADDR topkIndex, GM_ADDR topkValue, GM_ADDR topkLabelFinal,
    GM_ADDR topkValueFinal, GM_ADDR flag, GM_ADDR workspace,
    const AscendcIvfpqSearchDistanceTopKTilingData *__restrict tilingData, TPipe *tPipe)
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

    this->labelBaseGm.SetGlobalBuffer((__gm__ uint64_t *)labelBase);
    this->labelOffsetGm.SetGlobalBuffer((__gm__ int64_t *)labelOffset);
    this->topkLabelFinalGm.SetGlobalBuffer((__gm__ uint64_t *)topkLabelFinal);
    this->topkValueFinalGm.SetGlobalBuffer((__gm__ float *)topkValueFinal);

    ParseTilingData(tilingData);
    InitReduceUbTensor();
}

__aicore__ inline void AscendcIvfpqSearchDistanceTopKSmall::ParseTilingData(
    const AscendcIvfpqSearchDistanceTopKTilingData *__restrict tilingData)
{
    this->codeBlockSize = 0;
    for (int i = 0; i < tilingData->batch * tilingData->codeBlockNum; i++) {
        int64_t codeBlockSizeTmp = codeSizeGm.GetValue(i);
        if (this->codeBlockSize < codeBlockSizeTmp) {
            this->codeBlockSize = codeBlockSizeTmp;
        }
    }

    this->codeBlockSize =
        (this->codeBlockSize + IVFPQ_BLOCK_MAX_SIZE - 1) / IVFPQ_BLOCK_MAX_SIZE * IVFPQ_BLOCK_MAX_SIZE;
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

    this->topkTilingData = tilingData->topkTilingData;
    this->topkTilingDataWhole = tilingData->topkTilingDataWhole;
    this->topkTilingDataSingle = tilingData->topkTilingDataSingle;

    this->topk = tilingData->topk;
    this->topk_outter_num = tilingData->topkOutterNum;
    this->top_k_result_value_buf_num = topk * (single_core_total_block + 1);
    this->top_k_result_value_single_num = topk * single_core_total_block;

    this->topkTilingDataMergeBegin = tilingData->topkTilingDataMergeBegin;
    this->topkTilingDataMergeBeginTail = tilingData->topkTilingDataMergeBeginTail;
    this->topkTilingDataMergeEnd = tilingData->topkTilingDataMergeEnd;

    this->mergeBeginTopNumInLoop = tilingData->mergeBeginTopNumInLoop;
    this->mergeBeginTailTopNumInLoop = tilingData->mergeBeginTailTopNumInLoop;
    this->mergeBeginBlockLoopTime = tilingData->mergeBeginBlockLoopTime;
    this->mergeBeginTailBlockLoopTime = tilingData->mergeBeginTailBlockLoopTime;
    this->minSizeMergeBegin = tilingData->minSizeMergeBegin;
    this->minSizeMergeBeginTail = tilingData->minSizeMergeBeginTail;
    this->minSizeMergeEnd = tilingData->minSizeMergeEnd;
    this->batch = tilingData->batch;
}

__aicore__ inline void AscendcIvfpqSearchDistanceTopKSmall::InitReduceUbTensor()
{
    pipe.InitBuffer(top_k_result_value_buf, top_k_result_value_buf_num * sizeof(float));
    pipe.InitBuffer(top_k_result_index_buf, top_k_result_value_buf_num * sizeof(int32_t));

    pipe.InitBuffer(top_k_result_value_buf_whole, topk * sizeof(float));
    pipe.InitBuffer(top_k_result_index_buf_whole, topk * sizeof(int32_t));

    pipe.InitBuffer(queryPQUbQueue, 1, this->ksub * this->subSpaceNum * sizeof(float));
    pipe.InitBuffer(distResultQueue, 1, this->perCoreInnerBlockDealSize * sizeof(float));

    pipe.InitBuffer(flag_buf, IVFPQ_FLAG_ALIGN * sizeof(uint16_t));

    pipe.InitBuffer(top_k_tmp_buf, this->min_size * sizeof(uint8_t));
    pipe.InitBuffer(top_k_tmp_whole_buf, this->min_size_whole * sizeof(uint8_t));
    pipe.InitBuffer(top_k_tmp_single_buf, this->min_size_single * sizeof(uint8_t));
}

__aicore__ inline void AscendcIvfpqSearchDistanceTopKSmall::SingleBlockTopKAndIndexAlign(
    LocalTensor<float> dst_local_value, LocalTensor<int32_t> dst_local_index, LocalTensor<float> distResultUb,
    LocalTensor<uint8_t> tmp_local, int innerBlockindex)
{
    static constexpr TopKConfig topkConfig{TopKAlgo::RADIX_SELECT, TopKOrder::UNSET, false};
    for (uint32_t index = 0; index < this->single_core_total_block; index++) {
        LocalTensor<int32_t> src_local_index; // 源索引缓冲区（不需要输入）
        LocalTensor<bool> src_local_finish;   // 源完成标志（不需要输入）

        TopKInfo topKInfo;
        topKInfo.outter = 1;                    // 外层循环次数
        topKInfo.inner = this->topk_outter_num; // 内层块大小
        topKInfo.n = this->topk_outter_num;     // 数据总数

        AscendC::TopK<float, false, false, false, AscendC::TopKMode::TOPK_NORMAL, topkConfig>(
            dst_local_value[index * topk], dst_local_index[index * topk], distResultUb[index * this->topk_outter_num],
            src_local_index, src_local_finish, tmp_local, this->topk, this->topkTilingData, topKInfo, this->isLargest);
    }
    AscendC::PipeBarrier<PIPE_V>();

    for (uint32_t index = 0; index < this->single_core_total_block; index++) {
        AscendC::Adds(dst_local_index[index * this->topk], dst_local_index[index * this->topk],
                      index * this->topk_outter_num + innerBlockindex * this->perCoreInnerBlockDealSize, this->topk);
    }
    AscendC::PipeBarrier<PIPE_V>();
}

__aicore__ inline void AscendcIvfpqSearchDistanceTopKSmall::WholeBlockTopK(
    LocalTensor<float> dist_local_value, LocalTensor<int32_t> dist_local_index, LocalTensor<float> src_local_value,
    LocalTensor<int32_t> src_local_index, LocalTensor<uint8_t> tmp_local_whole, LocalTensor<uint8_t> tmp_local_single,
    int innerBlockindex)
{
    static constexpr TopKConfig topkConfigSort{TopKAlgo::RADIX_SELECT, TopKOrder::UNSET, false};
    if (innerBlockindex > 0) {
        // 需要两次对比
        TopKInfo topKInfoWhole;
        topKInfoWhole.outter = 1;
        topKInfoWhole.inner = this->top_k_result_value_buf_num;
        topKInfoWhole.n = this->top_k_result_value_buf_num;
        LocalTensor<bool> src_local_finish_whole;

        AscendC::TopK<float, true, false, false, AscendC::TopKMode::TOPK_NORMAL, topkConfigSort>(
            dist_local_value, dist_local_index, src_local_value, src_local_index, src_local_finish_whole,
            tmp_local_whole, topk, this->topkTilingDataWhole, topKInfoWhole, this->isLargest);
    } else {
        // 只对比当前轮次
        TopKInfo topKInfoWhole;
        topKInfoWhole.outter = 1;                            // 外层循环次数
        topKInfoWhole.inner = top_k_result_value_single_num; // 内层块大小
        topKInfoWhole.n = top_k_result_value_single_num;     // 数据总数
        LocalTensor<bool> src_local_finish_whole;

        AscendC::TopK<float, true, false, false, AscendC::TopKMode::TOPK_NORMAL, topkConfigSort>(
            dist_local_value, dist_local_index, src_local_value, src_local_index, src_local_finish_whole,
            tmp_local_single, topk, this->topkTilingDataSingle, topKInfoWhole, this->isLargest);
    }
    AscendC::PipeBarrier<PIPE_ALL>();
}

__aicore__ inline void
AscendcIvfpqSearchDistanceTopKSmall::DistCompute(__ubuf__ float *queryPQUb, __gm__ uint8_t *codeBase,
                                                 __gm__ int64_t *codeSize, __gm__ int64_t *codeOffset,
                                                 __ubuf__ float *distResultUbPtr, int innerBlockindex, int blockIndex)
{
    if (this->subSpaceNum == 2) {
        Simt::VF_CALL<SimtSearchComputeSmallV1<2>>(Simt::Dim3{IVFPQ_THREAD_NUM, 1, 1}, queryPQUb, codeBase,
                                                   distResultUbPtr, codeSize, codeOffset, this->ksub,
                                                   this->codeBlockNum, this->perCoreInnerBlockDealSize,
                                                   this->distResultInitValue, innerBlockindex, blockIndex, usedAivNum);
    } else if (this->subSpaceNum == 4) {
        Simt::VF_CALL<SimtSearchComputeSmallV1<4>>(Simt::Dim3{IVFPQ_THREAD_NUM, 1, 1}, queryPQUb, codeBase,
                                                   distResultUbPtr, codeSize, codeOffset, this->ksub,
                                                   this->codeBlockNum, this->perCoreInnerBlockDealSize,
                                                   this->distResultInitValue, innerBlockindex, blockIndex, usedAivNum);
    } else if (this->subSpaceNum == 8) {
        Simt::VF_CALL<SimtSearchComputeSmallV1<8>>(Simt::Dim3{IVFPQ_THREAD_NUM, 1, 1}, queryPQUb, codeBase,
                                                   distResultUbPtr, codeSize, codeOffset, this->ksub,
                                                   this->codeBlockNum, this->perCoreInnerBlockDealSize,
                                                   this->distResultInitValue, innerBlockindex, blockIndex, usedAivNum);
    } else if (this->subSpaceNum == 16) {
        Simt::VF_CALL<SimtSearchComputeSmallV1<16>>(Simt::Dim3{IVFPQ_THREAD_NUM, 1, 1}, queryPQUb, codeBase,
                                                    distResultUbPtr, codeSize, codeOffset, this->ksub,
                                                    this->codeBlockNum, this->perCoreInnerBlockDealSize,
                                                    this->distResultInitValue, innerBlockindex, blockIndex, usedAivNum);
    }
    AscendC::PipeBarrier<PIPE_ALL>();
}

__aicore__ inline void AscendcIvfpqSearchDistanceTopKSmall::Process()
{
    uint32_t perCoreBlockNum = (codeBlockNum + usedAivNum - 1) / usedAivNum;
    uint32_t perCoreInnerBlockLoopNum =
        (codeBlockSize + this->perCoreInnerBlockDealSize - 1) / this->perCoreInnerBlockDealSize;

    LocalTensor<float> dst_local_value = top_k_result_value_buf.Get<float>();                 // TopK结果值
    LocalTensor<int32_t> dst_local_index = top_k_result_index_buf.Get<int32_t>();             // TopK结果索引
    LocalTensor<float> dst_local_value_whole = top_k_result_value_buf_whole.Get<float>();     // TopK结果值
    LocalTensor<int32_t> dst_local_index_whole = top_k_result_index_buf_whole.Get<int32_t>(); // TopK结果索引

    LocalTensor<uint8_t> tmp_local = top_k_tmp_buf.Get<uint8_t>();
    LocalTensor<uint8_t> tmp_local_single = top_k_tmp_single_buf.Get<uint8_t>();
    LocalTensor<uint8_t> tmp_local_whole = top_k_tmp_whole_buf.Get<uint8_t>();

    for (int batchIndex = 0; batchIndex < batch; batchIndex++) {
        LocalTensor<float> queryPQEnqueUb = queryPQUbQueue.AllocTensor<float>();
        AscendC::DataCopy(queryPQEnqueUb, queryPQGm[batchIndex * this->ksub * this->subSpaceNum],
                          this->ksub * this->subSpaceNum);
        queryPQUbQueue.EnQue(queryPQEnqueUb);
        LocalTensor<float> queryPQDequeUb = queryPQUbQueue.DeQue<float>();
        __ubuf__ float *queryPQUb = (__ubuf__ float *)queryPQDequeUb.GetPhyAddr();
        __gm__ uint8_t *codeBase = (__gm__ uint8_t *)this->codeBaseGm.GetPhyAddr();
        __gm__ int64_t *codeSize = (__gm__ int64_t *)this->codeSizeGm[batchIndex * codeBlockNum].GetPhyAddr();
        __gm__ int64_t *codeOffset = (__gm__ int64_t *)this->codeOffsetGm[batchIndex * codeBlockNum].GetPhyAddr();

        for (int blockIndex = 0; blockIndex < perCoreBlockNum; blockIndex++) {
            uint32_t blockOffsetNum = blockIdx + blockIndex * usedAivNum;
            int64_t codeBlockSizeTmp = codeSizeGm.GetValue(batchIndex * codeBlockNum + blockOffsetNum);
            if (codeBlockSizeTmp != 0) {
                for (int innerBlockindex = 0; innerBlockindex < perCoreInnerBlockLoopNum; innerBlockindex++) {
                    LocalTensor<float> distResultUb = distResultQueue.AllocTensor<float>();
                    AscendC::Duplicate<float>(distResultUb, this->distResultInitValue, this->perCoreInnerBlockDealSize);
                    AscendC::PipeBarrier<PIPE_V>();
                    __ubuf__ float *distResultUbPtr = (__ubuf__ float *)distResultUb.GetPhyAddr();
                    DistCompute(queryPQUb, codeBase, codeSize, codeOffset, distResultUbPtr, innerBlockindex,
                                blockIndex);
                    SingleBlockTopKAndIndexAlign(dst_local_value, dst_local_index, distResultUb, tmp_local,
                                                 innerBlockindex);
                    WholeBlockTopK(dst_local_value_whole, dst_local_index_whole, dst_local_value, dst_local_index,
                                   tmp_local_whole, tmp_local_single, innerBlockindex);

                    AscendC::DataCopy(dst_local_value[top_k_result_value_single_num], dst_local_value_whole,
                                      topk);
                    AscendC::DataCopy(dst_local_index[top_k_result_value_single_num], dst_local_index_whole,
                                      topk);
                    distResultQueue.FreeTensor(distResultUb);
                }
                AscendC::DataCopy(topkValueGm[batchIndex * topk * codeBlockNum + (blockOffsetNum)*topk],
                                  dst_local_value_whole, topk);
                AscendC::DataCopy(topkIndexGm[batchIndex * topk * codeBlockNum + (blockOffsetNum)*topk],
                                  dst_local_index_whole, topk);
            } else {
                int32_t eventIDMTE3ToV = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE3_V));
                AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(eventIDMTE3ToV);
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(eventIDMTE3ToV);

                AscendC::Duplicate<float>(dst_local_value_whole, this->distResultInitValue, topk);
                AscendC::Duplicate<int32_t>(dst_local_index_whole, 0, topk);

                int32_t eventIDVToMTE3 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_MTE3));
                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(eventIDVToMTE3);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(eventIDVToMTE3);

                AscendC::DataCopy(topkValueGm[batchIndex * topk * codeBlockNum + (blockOffsetNum)*topk],
                                  dst_local_value_whole, topk);
                AscendC::DataCopy(topkIndexGm[batchIndex * topk * codeBlockNum + (blockOffsetNum)*topk],
                                  dst_local_index_whole, topk);
            }
        }
        queryPQUbQueue.FreeTensor(queryPQDequeUb);
    }
    AscendC::SyncAll();
}

__aicore__ inline void AscendcIvfpqSearchDistanceTopKSmall::ProcessMerge()
{
    static constexpr TopKConfig topkConfig{TopKAlgo::RADIX_SELECT, TopKOrder::UNSET, false};
    static constexpr TopKConfig topkConfigSort{TopKAlgo::RADIX_SELECT, TopKOrder::UNSET, true};

    pipe.Reset();

    pipe.InitBuffer(top_k_merge_src_value_que, 1, topk * mergeBeginTopNumInLoop * sizeof(float));
    pipe.InitBuffer(top_k_merge_src_index_que, 1, topk * mergeBeginTopNumInLoop * sizeof(int32_t));

    pipe.InitBuffer(top_k_merge_begin_dst_value_buf,
                    topk * (mergeBeginBlockLoopTime + mergeBeginTailBlockLoopTime) * sizeof(float));
    pipe.InitBuffer(top_k_merge_begin_dst_index_buf,
                    topk * (mergeBeginBlockLoopTime + mergeBeginTailBlockLoopTime) * sizeof(int32_t));
    pipe.InitBuffer(top_k_merge_begin_temp_buf, minSizeMergeBegin * sizeof(uint8_t));

    pipe.InitBuffer(top_k_merge_end_dst_value_buf, topk * sizeof(float));
    pipe.InitBuffer(top_k_merge_end_dst_index_buf, topk * sizeof(int32_t));
    pipe.InitBuffer(top_k_merge_end_temp_buf, minSizeMergeEnd * sizeof(uint8_t));

    LocalTensor<float> top_k_merge_begin_dst_value_buf_local = top_k_merge_begin_dst_value_buf.Get<float>();
    LocalTensor<int32_t> top_k_merge_begin_dst_index_buf_local = top_k_merge_begin_dst_index_buf.Get<int32_t>();

    LocalTensor<float> top_k_merge_end_dst_value_buf_local = top_k_merge_end_dst_value_buf.Get<float>();
    LocalTensor<int32_t> top_k_merge_end_dst_index_buf_local = top_k_merge_end_dst_index_buf.Get<int32_t>();

    LocalTensor<uint8_t> top_k_merge_begin_temp_buf_local = top_k_merge_begin_temp_buf.Get<uint8_t>();
    LocalTensor<uint8_t> top_k_merge_end_temp_buf_local = top_k_merge_end_temp_buf.Get<uint8_t>();

    LocalTensor<bool> src_local_finish; // 源完成标志（不需要输入）

    for (int batchIndex = blockIdx; batchIndex < batch; batchIndex += usedAivNum) {
        __gm__ uint64_t *labelBaseGmAddr = (__gm__ uint64_t *)this->labelBaseGm.GetPhyAddr();
        __gm__ int64_t *labelOffsetGmAddr =
            (__gm__ int64_t *)this->labelOffsetGm[batchIndex * codeBlockNum].GetPhyAddr();
        __gm__ uint64_t *topkLabelFinalGmAddr =
            (__gm__ uint64_t *)this->topkLabelFinalGm[batchIndex * topk].GetPhyAddr();

        for (int i = 0; i < (mergeBeginBlockLoopTime + mergeBeginTailBlockLoopTime); i++) {
            int64_t codeBlockSizeTmp = codeSizeGm.GetValue(batchIndex * codeBlockNum + i * mergeBeginBlockLoopTime);
            if (codeBlockSizeTmp != 0) {
                LocalTensor<float> top_k_merge_src_value = top_k_merge_src_value_que.AllocTensor<float>();
                LocalTensor<int32_t> top_k_merge_src_index = top_k_merge_src_index_que.AllocTensor<int32_t>();

                AscendC::DataCopy(top_k_merge_src_value,
                                  topkValueGm[batchIndex * codeBlockNum * topk + i * topk * mergeBeginTopNumInLoop],
                                  topk * mergeBeginTopNumInLoop);
                AscendC::DataCopy(top_k_merge_src_index,
                                  topkIndexGm[batchIndex * codeBlockNum * topk + i * topk * mergeBeginTopNumInLoop],
                                  topk * mergeBeginTopNumInLoop);

                top_k_merge_src_value_que.EnQue(top_k_merge_src_value);
                top_k_merge_src_index_que.EnQue(top_k_merge_src_index);

                LocalTensor<float> top_k_merge_src_value_deque = top_k_merge_src_value_que.DeQue<float>();
                LocalTensor<int32_t> top_k_merge_src_index_deque = top_k_merge_src_index_que.DeQue<int32_t>();

                for (int indexId = 0; indexId < mergeBeginTopNumInLoop; indexId++) {
                    AscendC::Adds(top_k_merge_src_index_deque[indexId * this->topk],
                                  top_k_merge_src_index_deque[indexId * this->topk],
                                  IVFPQ_CODE_BLOCK_SIZE * (i * mergeBeginTopNumInLoop + indexId), this->topk);
                }
                AscendC::PipeBarrier<PIPE_V>();

                if (i != mergeBeginBlockLoopTime) {
                    TopKInfo topKInfo;
                    topKInfo.outter = 1;
                    topKInfo.inner = topk * mergeBeginTopNumInLoop;
                    topKInfo.n = topk * mergeBeginTopNumInLoop;
                    AscendC::TopK<float, true, false, false, AscendC::TopKMode::TOPK_NORMAL, topkConfig>(
                        top_k_merge_begin_dst_value_buf_local[i * topk],
                        top_k_merge_begin_dst_index_buf_local[i * topk], top_k_merge_src_value_deque,
                        top_k_merge_src_index_deque, src_local_finish, top_k_merge_begin_temp_buf_local, topk,
                        this->topkTilingDataMergeBegin, topKInfo, this->isLargest);
                } else {
                    TopKInfo topKInfo;
                    topKInfo.outter = 1;
                    topKInfo.inner = topk * mergeBeginTailTopNumInLoop;
                    topKInfo.n = topk * mergeBeginTailTopNumInLoop;
                    AscendC::TopK<float, true, false, false, AscendC::TopKMode::TOPK_NORMAL, topkConfig>(
                        top_k_merge_begin_dst_value_buf_local[i * topk],
                        top_k_merge_begin_dst_index_buf_local[i * topk], top_k_merge_src_value_deque,
                        top_k_merge_src_index_deque, src_local_finish, top_k_merge_begin_temp_buf_local, topk,
                        this->topkTilingDataMergeBeginTail, topKInfo, this->isLargest);
                }
                AscendC::PipeBarrier<PIPE_V>();
                top_k_merge_src_value_que.FreeTensor(top_k_merge_src_value_deque);
                top_k_merge_src_index_que.FreeTensor(top_k_merge_src_index_deque);
            } else {
                AscendC::Duplicate<float>(top_k_merge_begin_dst_value_buf_local[i * topk], this->distResultInitValue,
                                          topk);
                AscendC::Duplicate<int32_t>(top_k_merge_begin_dst_index_buf_local[i * topk], 0, topk);
                AscendC::PipeBarrier<PIPE_V>();
            }
        }
        TopKInfo topKInfo;
        topKInfo.outter = 1;
        topKInfo.inner = topk * (mergeBeginBlockLoopTime + mergeBeginTailBlockLoopTime);
        topKInfo.n = topk * (mergeBeginBlockLoopTime + mergeBeginTailBlockLoopTime);

        AscendC::TopK<float, true, false, false, AscendC::TopKMode::TOPK_NORMAL, topkConfigSort>(
            top_k_merge_end_dst_value_buf_local, top_k_merge_end_dst_index_buf_local,
            top_k_merge_begin_dst_value_buf_local, top_k_merge_begin_dst_index_buf_local, src_local_finish,
            top_k_merge_end_temp_buf_local, topk, this->topkTilingDataMergeEnd, topKInfo, this->isLargest);
        AscendC::PipeBarrier<PIPE_ALL>();

        __ubuf__ int32_t *top_k_merge_end_dst_index_buf_local_addr =
            (__ubuf__ int32_t *)top_k_merge_end_dst_index_buf_local.GetPhyAddr();
        Simt::VF_CALL<SimtSearchGatherSmallV1>(Simt::Dim3{512, 1, 1}, top_k_merge_end_dst_index_buf_local_addr,
                                               labelOffsetGmAddr, labelBaseGmAddr, topkLabelFinalGmAddr, topk);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::DataCopy(topkValueFinalGm[batchIndex * topk], top_k_merge_end_dst_value_buf_local, topk);
    }

    if (blockIdx == 0) {
        AscendC::InitGlobalMemory(flagGm, static_cast<uint64_t>(IVFPQ_FLAG_ALIGN), (uint16_t)1);
    }
}

} // namespace IndexOps
#endif
