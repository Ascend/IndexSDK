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

#include <limits>
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"

using namespace AscendC;

namespace kernels {
constexpr float IVFRABITQ_MAX_FLOAT = std::numeric_limits<float>::max();
constexpr float IVFRABITQ_MIN_FLOAT = std::numeric_limits<float>::lowest();
constexpr int32_t METRIC_L2 = 0;
constexpr int32_t METRIC_INNER_PRODUCT = 1;

template <typename T>
class DistanceIVFRabitqL2FP32 {
public:
    TPipe *pipe_;
    DistanceIVFRabitqL2FP32TilingData tiling_;

public:
    __aicore__ inline DistanceIVFRabitqL2FP32(TPipe *pipe_in)
    {
        pipe_ = pipe_in;
    };
    __aicore__ inline void Init(GM_ADDR queryl2, GM_ADDR querylut, GM_ADDR centroidslut, GM_ADDR queryid,
                                GM_ADDR centroidsid, GM_ADDR centroidsl2, GM_ADDR base, GM_ADDR offset,
                                GM_ADDR actual_size, GM_ADDR indexl2, GM_ADDR indexl1, GM_ADDR indexl2offset,
                                GM_ADDR indexl1offset, GM_ADDR result, GM_ADDR min_result, GM_ADDR flag,
                                GM_ADDR usrWorkspace, const DistanceIVFRabitqL2FP32TilingData *tiling_data)
    {
        this->blockIdx = get_block_idx();
        tiling_ = *tiling_data;
        this->dimLength = tiling_.dimLength;  // 这个dimLength应该是原向量的dim
        this->codeBlockLength = tiling_.codeBlockLength;
        this->lutDimLength = tiling_.lutDimLength;  // lut的维度 值应该固定256
        this->lutLength = tiling_.lutLength;        // = this->dimLength / 8

        this->lutTileNum = tiling_.lutTileNum;
        this->lutTileLength = tiling_.lutTileLength;          // 每次处理多少lut
        this->lastLutTileLength = tiling_.lastLutTileLength;  // 最后一次处理多少lut

        this->metric = tiling_.metricType;
        if (this->metric == METRIC_L2) {
            this->reduceCmpInitValue = IVFRABITQ_MAX_FLOAT;
        } else {
            this->reduceCmpInitValue = IVFRABITQ_MIN_FLOAT;
        }

        this->maskNum = this->dimLength / this->mask;
        this->multiplier = static_cast<T>(2) / sqrt(static_cast<T>(this->dimLength));

        queryid_gm_.SetGlobalBuffer((__gm__ uint32_t *)queryid);
        uint32_t cur_queryid = queryid_gm_.GetValue(this->blockIdx);  // 获取当前coreid对应的查询id
        queryl2_gm_.SetGlobalBuffer((__gm__ T *)queryl2);
        cur_queryl2 = queryl2_gm_.GetValue(cur_queryid);  // 根据查询id获取当前core对应的查询的l2模长
        cur_queryl2 = cur_queryl2 * (-1);
        querylut_gm_.SetGlobalBuffer((__gm__ T *)querylut + this->lutLength * this->lutDimLength * cur_queryid,
                                     this->lutLength * this->lutDimLength);  // 根据查询id获取当前core需要的查询lut

        centroidsid_gm_.SetGlobalBuffer((__gm__ uint32_t *)centroidsid);
        uint32_t cur_centroidsid = centroidsid_gm_.GetValue(this->blockIdx);  // 获取当前coreid对应的质心id
        centroidsl2_gm_.SetGlobalBuffer((__gm__ T *)centroidsl2);
        cur_centroidsl2 = centroidsl2_gm_.GetValue(this->blockIdx);  // 获取当前coreid对应的质心到query的l2距离
        cur_centroidslut_gm_.SetGlobalBuffer((__gm__ T *)centroidslut + this->lutLength * this->lutDimLength * cur_centroidsid,
                                             this->lutLength * this->lutDimLength);  // 根据质心id获取当前core需要的质心lut

        offset_gm_.SetGlobalBuffer((__gm__ uint64_t *)offset);
        actual_size_gm_.SetGlobalBuffer((__gm__ uint32_t *)actual_size);
        uint64_t code_offset = offset_gm_.GetValue(this->blockIdx);
        this->actualSizeVal = actual_size_gm_.GetValue(this->blockIdx);
        code_gm_.SetGlobalBuffer((__gm__ uint8_t *)(base + code_offset));

        indexl1_offset_gm_.SetGlobalBuffer((__gm__ uint64_t *)indexl1offset);
        indexl2_offset_gm_.SetGlobalBuffer((__gm__ uint64_t *)indexl2offset);
        uint64_t l1Offset = indexl1_offset_gm_.GetValue(this->blockIdx);
        uint64_t l2Offset = indexl2_offset_gm_.GetValue(this->blockIdx);
        indexl1_gm_.SetGlobalBuffer((__gm__ T *)(indexl1 + l1Offset));
        indexl2_gm_.SetGlobalBuffer((__gm__ T *)(indexl2 + l2Offset));

        result_gm_.SetGlobalBuffer((__gm__ T *)result + this->blockIdx * this->codeBlockLength);
        min_res_gm_.SetGlobalBuffer((__gm__ T *)min_result +
                                    (this->codeBlockLength + this->mask - 1) / this->mask * 2 * this->blockIdx);
        flag_gm_.SetGlobalBuffer((__gm__ uint16_t *)flag + this->blockIdx * 32);

        this->codeTileLength = tiling_.codeTileLength;  // 每次处理多少codes

        this->codeTileNum = (actualSizeVal + this->codeTileLength - 1) / this->codeTileLength;
        this->lastCodeTileLength = actualSizeVal % this->codeTileLength;
        if (this->lastCodeTileLength == 0) {
            this->lastCodeTileLength = this->codeTileLength;
        }

        this->distTileLength = tiling_.distTileLength;  // 每次处理多少dist

        this->distTileNum = (actualSizeVal + this->distTileLength - 1) / this->distTileLength;
        this->lastDistTileLength = actualSizeVal % this->distTileLength;
        if (this->lastDistTileLength == 0) {
            this->lastDistTileLength = this->distTileLength;
        }

        pipe_->InitBuffer(in_querylut_que, 1, 
                          RoundUp(this->lutLength * this->lutDimLength * sizeof(T), 64)); // querylut全部拉取，该空间可以存储q-c
        pipe_->InitBuffer(codes_in_que, 1, RoundUp(this->dimLength / 8 * this->codeTileLength, 64));
        pipe_->InitBuffer(codes_half_que, 1, RoundUp(this->dimLength / 8 * this->codeTileLength * sizeof(half), 64));
        pipe_->InitBuffer(codes_int32_que, 1, RoundUp(this->dimLength / 8 * this->codeTileLength * sizeof(int32_t), 64));
        pipe_->InitBuffer(idx_calc_que, 1, RoundUp(this->dimLength / 8 * sizeof(int32_t), 64));
        pipe_->InitBuffer(select_out_que, 1, RoundUp(this->dimLength / 8 * this->codeTileLength * sizeof(T), 64));
        pipe_->InitBuffer(ip_out_que, 1, RoundUp(this->codeTileLength * sizeof(T), 64));
    }

    __aicore__ inline int32_t RoundUp(int32_t length, int32_t align)
    {
        return (length + align - 1) / align * align;
    }

    __aicore__ inline void SecondInit()
    {
        pipe_->Reset();
        pipe_->InitBuffer(in_mat_res_que, 1, RoundUp(this->distTileLength * sizeof(T), 64));
        pipe_->InitBuffer(in_indexl1_que, 1, RoundUp(this->distTileLength * sizeof(T), 64));
        pipe_->InitBuffer(in_indexl2_que, 1, RoundUp(this->distTileLength * sizeof(T), 64));
        pipe_->InitBuffer(dist_result_que, 1, RoundUp(this->distTileLength * sizeof(T), 32));
        pipe_->InitBuffer(min_result_que, 1, RoundUp(this->distTileLength * this->mask * sizeof(T), 32));
    }

    __aicore__ inline void Process()
    {
        if (this->codeTileNum > 0) {
            QuerySubtractCentroid();  // 先计算q-c
            PipeBarrier<PIPE_ALL>();
            LookUpTableAndSum();  // 查表求和
            PipeBarrier<PIPE_ALL>();
            SecondInit();
            DistCompute();   // 计算距离
        }
        PipeBarrier<PIPE_ALL>();
        AscendC::InitGlobalMemory(flag_gm_, static_cast<uint64_t>(32), (uint16_t)1);
    }

private:
    __aicore__ inline void QuerySubtractCentroid()
    {  // 计算q-c
        CopyInForQueryLUT();
        LocalTensor<T> querylutLocal = in_querylut_que.DeQue<T>();
        for (int32_t i = 0; i < this->lutTileNum; i++) {
            int32_t computeLutLength = (i == this->lutTileNum - 1) ? this->lastLutTileLength : this->lutTileLength;
            CopyInForCentroidsLUT(i, computeLutLength);
            LocalTensor<T> centroidslutLocal = select_out_que.DeQue<T>();
            Sub(querylutLocal[i * this->lutDimLength * this->lutTileLength],
                querylutLocal[i * this->lutDimLength * this->lutTileLength], centroidslutLocal,
                computeLutLength * this->lutDimLength);
            select_out_que.FreeTensor(centroidslutLocal);
        }
        PipeBarrier<PIPE_V>();
        for (int i = 0; i < this->lutLength; i++) {
            this->sumQuery += querylutLocal.GetValue(i * this->lutDimLength + this->lutDimLength - 1);
        }
        this->sumQuery = this->sumQuery / sqrt(static_cast<T>(this->dimLength)) * (T)(-1);
        PipeBarrier<PIPE_ALL>();
        in_querylut_que.EnQue(querylutLocal);  // 将q-c结果放入in_querylut_que
    }

    __aicore__ inline void CopyInForQueryLUT()
    {
        LocalTensor<T> querylutLocal = in_querylut_que.AllocTensor<T>();
        DataCopy(querylutLocal, querylut_gm_, this->lutLength * this->lutDimLength);
        in_querylut_que.EnQue(querylutLocal);
    }

    __aicore__ inline void CopyInForCentroidsLUT(int32_t progress, int32_t copyLength)
    {
        LocalTensor<T> centroidslutLocal = select_out_que.AllocTensor<T>();
        DataCopy(centroidslutLocal, cur_centroidslut_gm_[progress * this->lutDimLength * this->lutTileLength],
                 copyLength * this->lutDimLength);
        select_out_que.EnQue(centroidslutLocal);
    }

    __aicore__ inline void LookUpTableAndSum()
    {
        LocalTensor<T> querylutLocal = in_querylut_que.DeQue<T>();
        LocalTensor<int32_t> idxLocal = idx_calc_que.AllocTensor<int32_t>();
        CreateVecIndex(idxLocal, 0, this->dimLength / 8);
        PipeBarrier<PIPE_V>();
        ShiftLeft(idxLocal, idxLocal, 8, this->dimLength / 8); // 每行 LUT 为 2^8 个，因此每行的偏移值需要左移 8
        PipeBarrier<PIPE_ALL>();
        for (int32_t i = 0; i < this->codeTileNum; i++) {
            int32_t copyLength = this->codeTileLength;
            if (i == this->codeTileNum - 1) {
                copyLength = this->lastCodeTileLength;
            }
            CopyInForCode(i, copyLength);
            SelectAndReduceSum(i, copyLength, querylutLocal, idxLocal);
        }
        PipeBarrier<PIPE_ALL>();
        in_querylut_que.FreeTensor(querylutLocal);
        idx_calc_que.FreeTensor(idxLocal);
    }

    __aicore__ inline void CopyInForCode(int32_t progress, int32_t copyLength)
    {
        LocalTensor<uint8_t> codesLocal = codes_in_que.AllocTensor<uint8_t>();

        uint8_t paddingBytes = RoundUp(this->dimLength / 8 * copyLength, 32) - this->dimLength / 8 * copyLength;
        DataCopyPadExtParams<uint8_t> padParams{true, 0, paddingBytes, 0};
        DataCopyExtParams copyParams1{1, (uint32_t)(this->dimLength / 8 * copyLength * sizeof(uint8_t)), 0, 0, 0};
        DataCopyPad(codesLocal, code_gm_[this->dimLength / 8 * progress * this->codeTileLength],
                    copyParams1, padParams);
        codes_in_que.EnQue(codesLocal);
    }

    __aicore__ inline void SelectAndReduceSum(int32_t progress, int32_t copyLength, LocalTensor<T> vecsLocal, LocalTensor<int32_t> idxLocal)
    {
        LocalTensor<uint8_t> codesLocal = codes_in_que.DeQue<uint8_t>();
        LocalTensor<half> codesHalfLocal = codes_half_que.AllocTensor<half>();
        LocalTensor<int32_t> codesInt32Local = codes_int32_que.AllocTensor<int32_t>();
        
        LocalTensor<T> selResultLocal = select_out_que.AllocTensor<T>();
        LocalTensor<T> ipResultLocal = ip_out_que.AllocTensor<T>();

        uint32_t codeSize = this->dimLength / 8 * copyLength;
        Cast(codesHalfLocal, codesLocal, RoundMode::CAST_NONE, codeSize);
        PipeBarrier<PIPE_V>();
        Cast(codesInt32Local, codesHalfLocal, RoundMode::CAST_ROUND, codeSize);
        PipeBarrier<PIPE_V>();
        uint32_t addNum = this->dimLength / 8;
        int32_t addTileNum = (addNum + 63) / 64;
        uint8_t repeatStride = addNum / 8;
        for (int32_t k = 0; k < addTileNum; k++) {
            uint64_t maskLen = 64;
            if (k == addTileNum - 1) {
                maskLen = addNum % 64;
                if (maskLen == 0) {
                    maskLen = 64;
                }
            }
            for (int32_t i = 0; i < copyLength; i += 255) {
                int32_t len = copyLength - i;
                if (len > 255)
                    len = 255;
                // dimLength是64的倍数, dimLength / 8 是 32B 对齐
                Add(codesInt32Local[i * this->dimLength / 8 + k * 64], idxLocal[k * 64],
                    codesInt32Local[i * this->dimLength / 8 + k * 64], maskLen, len, {1, 1, 1, repeatStride, 0, repeatStride});       
            }
        }
        PipeBarrier<PIPE_V>();
        ShiftLeft(codesInt32Local, codesInt32Local, 2, codeSize); //sizeof(float) = 2^2，因此字节偏移需要左移 2
        PipeBarrier<PIPE_V>();
        Gather(selResultLocal, vecsLocal, codesInt32Local.ReinterpretCast<uint32_t>(), (uint32_t)0, codeSize);
        PipeBarrier<PIPE_V>();

        codes_int32_que.FreeTensor(codesInt32Local);
        if (this->dimLength / 8 > 64) {
            LocalTensor<uint8_t> tmpLocal = codes_int32_que.AllocTensor<uint8_t>();
            uint32_t shape[] = {static_cast<uint32_t>(this->codeTileLength), static_cast<uint32_t>(this->dimLength / 8)};
            AscendC::ReduceSum<float, AscendC::Pattern::Reduce::AR, true>(ipResultLocal, selResultLocal, tmpLocal, shape, true);
            PipeBarrier<PIPE_V>();
            codes_int32_que.FreeTensor(tmpLocal);
        } else {
            for (int32_t i = 0; i < copyLength; i += 255) {
                int32_t len = copyLength - i;
                if (len > 255)
                    len = 255;
                WholeReduceSum<T>(ipResultLocal[i], selResultLocal[this->dimLength / 8 * i], this->dimLength / 8, len, 1, 1, repeatStride);
            }
            PipeBarrier<PIPE_V>();
        }

        PipeBarrier<PIPE_ALL>();
        DataCopyExtParams copyParams{1, (uint32_t)(copyLength * sizeof(T)), 0, 0, 0};
        DataCopyPad(result_gm_[progress * this->codeTileLength], ipResultLocal, copyParams);
        ip_out_que.FreeTensor(ipResultLocal);
        select_out_que.FreeTensor(selResultLocal);
        codes_in_que.FreeTensor(codesLocal);
        codes_half_que.FreeTensor(codesHalfLocal);
    }

    __aicore__ inline void DistCompute()
    {
        LocalTensor<T> minResultLocal = min_result_que.AllocTensor<T>();
        Duplicate(minResultLocal, this->reduceCmpInitValue, this->distTileLength * this->mask);
        this->minNum = 0;
        for (int32_t i = 0; i < this->distTileNum; i++) {
            int32_t copyLength = (i == this->distTileNum - 1) ? this->lastDistTileLength : this->distTileLength;
            CopyInForMatResAndIndexL1(i, copyLength);  // 拷贝内积结果和L1因子
            Compute(i, copyLength, minResultLocal);
            CopyOutForDistRes(i, copyLength);
        }
        PipeBarrier<PIPE_ALL>();
        min_result_que.FreeTensor(minResultLocal);
    }

    __aicore__ inline void CopyInForMatResAndIndexL1(int32_t progress, int32_t copyLength)
    {
        LocalTensor<T> matResLocal = in_mat_res_que.AllocTensor<T>();
        LocalTensor<T> indexL1Local = in_indexl1_que.AllocTensor<T>();

        DataCopyExtParams copyParams{1, (uint32_t)(copyLength * sizeof(T)), 0, 0, 0};
        uint8_t paddingNum = (RoundUp(copyLength * sizeof(T), 32) - copyLength * sizeof(T)) / 4;
        DataCopyPadExtParams<T> padParams{true, 0, paddingNum, 0};

        DataCopyPad(matResLocal, result_gm_[progress * this->distTileLength], copyParams, padParams);
        DataCopyPad(indexL1Local, indexl1_gm_[progress * this->distTileLength], copyParams, padParams);
        in_mat_res_que.EnQue(matResLocal);
        in_indexl1_que.EnQue(indexL1Local);
    }

    __aicore__ inline void CopyInForIndexL2(int32_t progress, int32_t copyLength)
    {
        LocalTensor<T> indexL2Local = in_indexl2_que.AllocTensor<T>();
        DataCopyExtParams copyParams{1, (uint32_t)(copyLength * sizeof(T)), 0, 0, 0};
        uint8_t paddingNum = (RoundUp(copyLength * sizeof(T), 32) - copyLength * sizeof(T)) / 4;
        DataCopyPadExtParams<T> padParams{true, 0, paddingNum, 0};
        DataCopyPad(indexL2Local, indexl2_gm_[progress * this->distTileLength], copyParams, padParams);
        in_indexl2_que.EnQue(indexL2Local);
    }

    __aicore__ inline void Compute(int32_t progress, int32_t copyLength, LocalTensor<T> minResultLocal)
    {
        LocalTensor<T> distResultLocal = dist_result_que.AllocTensor<T>();
        LocalTensor<T> ipResLocal = in_mat_res_que.DeQue<T>();
        LocalTensor<T> indexL1Local = in_indexl1_que.DeQue<T>();

        // dist l2: ||o - q||^2 = ||o - c||^2 + ||q - c||^2 - 2 <q - c, o - c>
        Muls(ipResLocal, ipResLocal, this->multiplier, copyLength);
        PipeBarrier<PIPE_V>();
        Adds(ipResLocal, ipResLocal, this->sumQuery, copyLength);
        PipeBarrier<PIPE_V>();
        Mul(ipResLocal, ipResLocal, indexL1Local, copyLength);  // 复用ipResLocal保存临时结果
        CopyInForIndexL2(progress, copyLength);
        PipeBarrier<PIPE_V>();
        LocalTensor<T> indexL2Local = in_indexl2_que.DeQue<T>();  // 拷贝L2距离
        Sub(distResultLocal, indexL2Local, ipResLocal, copyLength);
        PipeBarrier<PIPE_V>();
        Adds(distResultLocal, distResultLocal, cur_centroidsl2, copyLength);
        PipeBarrier<PIPE_V>();
        if (this->metric == METRIC_INNER_PRODUCT) { // dist ip: - 2 <q, o> = ||o - q||^2 - ||q||^2 - ||o||^2
            PipeBarrier<PIPE_V>();
            Adds(distResultLocal, distResultLocal, this->cur_queryl2, copyLength);
            PipeBarrier<PIPE_V>();
            Muls(distResultLocal, distResultLocal, static_cast<float>(-0.5), copyLength);
            PipeBarrier<PIPE_V>();
        }

        int copyNum = copyLength / this->mask;
        int lastCodesLength = copyLength % this->mask;
        for (int32_t i = 0; i < copyNum; i += 255) {
            int32_t num = copyNum - i;
            if (num > 255)
                num = 255;
            Copy(minResultLocal[(progress % this->mask) * this->distTileLength + i * this->mask],
                 distResultLocal[i * this->mask], this->mask, num, {1, 1, 8, 8});
        }
        if (lastCodesLength != 0) {
            Copy(minResultLocal[(progress % this->mask) * this->distTileLength + (copyLength - lastCodesLength)],
                 distResultLocal[copyLength - lastCodesLength], lastCodesLength, 1, {1, 1, 8, 8});
        }

        ComputeMinResult(progress, minResultLocal);

        dist_result_que.EnQue(distResultLocal);  // 结果入队
        in_mat_res_que.FreeTensor(ipResLocal);
        in_indexl1_que.FreeTensor(indexL1Local);
        in_indexl2_que.FreeTensor(indexL2Local);
    }

    __aicore__ inline void ComputeMinResult(int32_t progress, LocalTensor<T> minResultLocal)
    {
        if (progress == this->distTileNum - 1) {
            int32_t minLength =
                (this->actualSizeVal - this->minNum * this->mask * this->distTileLength + this->mask - 1) / this->mask;
            PipeBarrier<PIPE_V>();
            for (int32_t i = 0; i < minLength; i += 255) {
                int32_t len = minLength - i;
                if (len > 255)
                    len = 255;
                if (this->metric == METRIC_L2) {
                    WholeReduceMin(minResultLocal[i * 2], minResultLocal[i * this->mask], this->mask, len, 1, 1, 8);
                } else {
                    WholeReduceMax(minResultLocal[i * 2], minResultLocal[i * this->mask], this->mask, len, 1, 1, 8);
                }
            }
            min_result_que.EnQue(minResultLocal);
            PipeBarrier<PIPE_V>();
            CopyOutMin(progress, minLength);
        } else if ((progress + 1) % this->mask == 0) {  // 当progress为63时，才会进入，需要运行64次才会进行一次计算操作
            int32_t minLength = this->distTileLength;
            PipeBarrier<PIPE_V>();
            // minResultLocal现在有this->mask * this->distTileLength 个float元素，
            // WholeReduceMin执行的操作：每隔8个dataBlock（8*32B）迭代一次，计算出一个最小值，迭代this->tileLength次。
            // 每次迭代的元素个数是 64个float(8*32B=4B*64)，每次迭代的参与计算的元素个数为this->mask(64)。
            // 每64个数取一个最小值和其索引，得到this->tileLength个数和对应的索引
            for (int32_t i = 0; i < minLength; i += 255) {
                int32_t len = minLength - i;
                if (len > 255)
                    len = 255;
                if (this->metric == METRIC_L2) {
                    WholeReduceMin(minResultLocal[i * 2], minResultLocal[i * this->mask], this->mask, len, 1, 1, 8);
                } else {
                    WholeReduceMax(minResultLocal[i * 2], minResultLocal[i * this->mask], this->mask, len, 1, 1, 8);
                }
            }
            min_result_que.EnQue(minResultLocal);
            PipeBarrier<PIPE_V>();
            CopyOutMin(progress, minLength);
            PipeBarrier<PIPE_ALL>();
            Duplicate(minResultLocal, this->reduceCmpInitValue, this->distTileLength * this->mask);
            PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline void CopyOutForDistRes(int32_t progress, int32_t copyLength)
    {
        LocalTensor<T> distResultLocal = dist_result_que.DeQue<T>();
        DataCopyExtParams copyParams{1, (uint32_t)(copyLength * sizeof(T)), 0, 0, 0};
        DataCopyPad(result_gm_[progress * this->distTileLength], distResultLocal, copyParams);
        dist_result_que.FreeTensor(distResultLocal);
    }

    __aicore__ inline void CopyOutMin(int32_t progress, int32_t minLength)
    {
        LocalTensor<T> minResultLocal = min_result_que.DeQue<T>();
        int32_t minOffset = (this->codeBlockLength + this->mask - 1) / this->mask;
        DataCopyExtParams copyParams{1, (uint32_t)(minLength * 2 * sizeof(T)), 0, 0, 0};
        DataCopyPad(min_res_gm_[this->minNum * this->distTileLength * 2], minResultLocal, copyParams);
        this->minNum = this->minNum + 1;
    }

private:
    TQue<QuePosition::VECIN, 1> in_querylut_que;
    TQue<QuePosition::VECIN, 1> codes_in_que;
    TQue<QuePosition::VECIN, 1> select_out_que;
    TQue<QuePosition::VECCALC, 1> codes_half_que, codes_int32_que, idx_calc_que;
    TQue<QuePosition::VECOUT, 1> ip_out_que;

    TQue<QuePosition::VECIN, 1> in_mat_res_que, in_indexl1_que, in_indexl2_que;
    TQue<QuePosition::VECOUT, 1> dist_result_que, min_result_que;

    GlobalTensor<T> queryl2_gm_;
    GlobalTensor<T> querylut_gm_;
    GlobalTensor<T> cur_centroidslut_gm_;
    GlobalTensor<uint32_t> queryid_gm_;
    GlobalTensor<uint32_t> centroidsid_gm_;
    GlobalTensor<T> centroidsl2_gm_;

    GlobalTensor<uint8_t> code_gm_;
    GlobalTensor<uint64_t> offset_gm_;
    GlobalTensor<uint32_t> actual_size_gm_;

    GlobalTensor<T> indexl1_gm_;
    GlobalTensor<uint64_t> indexl1_offset_gm_;
    GlobalTensor<T> indexl2_gm_;
    GlobalTensor<uint64_t> indexl2_offset_gm_;

    GlobalTensor<T> result_gm_;
    GlobalTensor<T> min_res_gm_;
    GlobalTensor<uint16_t> flag_gm_;

    int32_t blockIdx;
    uint32_t actualSizeVal;
    int32_t lutLength;
    int32_t lutDimLength;

    int32_t lutTileLength;
    int32_t lutTileNum;
    int32_t lastLutTileLength;

    int32_t codeTileNum;
    int32_t codeTileLength;
    int32_t lastCodeTileLength;

    int32_t distTileNum;
    int32_t distTileLength;
    int32_t lastDistTileLength;

    int32_t dimLength;
    int32_t codeBlockLength;
    int32_t minNum;

    int32_t metric;

    int32_t mask = 64;
    int32_t maskNum;
    float32_t oneFloat = 1.0;
    float32_t zeroFloat = 0.0;
    float32_t reduceCmpInitValue = 0.0;
    T cur_centroidsl2;
    T cur_queryl2;
    T multiplier = 0.0;
    T sumQuery = 0.0;
};

}

extern "C" __global__ __aicore__ void distance_ivf_rabitq_l2_fp32(
    GM_ADDR queryl2, GM_ADDR querylut, GM_ADDR centroidslut, GM_ADDR queryid, GM_ADDR centroidsid, GM_ADDR centroidsl2,
    GM_ADDR base, GM_ADDR offset, GM_ADDR actual_size, GM_ADDR indexl2, GM_ADDR indexl1, GM_ADDR indexl2offset,
    GM_ADDR indexl1offset, GM_ADDR result, GM_ADDR min_result, GM_ADDR flag, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    GM_ADDR usrWorkspace = GetUserWorkspace(workspace);
    TPipe pipe;
    kernels::DistanceIVFRabitqL2FP32<float32_t> op(&pipe);
    op.Init(queryl2, querylut, centroidslut, queryid, centroidsid, centroidsl2, base, offset, actual_size, indexl2,
            indexl1, indexl2offset, indexl1offset, result, min_result, flag, usrWorkspace, &tiling_data);
    op.Process();
}