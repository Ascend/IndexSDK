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

#include "ivf_sp_topk_l2_with_mask_kernels.h"
#include <algorithm>
#include <string>
#include <cstring>
#include <map>
#include <numeric>
#include "cpu_kernel.h"
#include "cust_cpu_utils.h"
#include "utils/cpu_kernel_utils.h"
#include "utils/kernel_tensor.h"
#include "utils/kernel_utils.h"
#include "utils/kernel_shared_def.h"
#include "securec.h"

namespace  {
    const char *IVF_SP_TOPK_L2_WITH_MASK = "IvfSpTopkL2WithMask";
    const uint32_t THREAD_CNT = 6;
    // 该算子中，所有输入输出的xxxOffset变量，均以所选的桶 (nProbe) 为单位，且存储2个值:
    // 每个桶当前变量的的起始地址和结束地址; 以该常量作为每个桶的偏移量（每次偏移两个单位
    const int STORE_INTERVAL = 2;
}

namespace aicpu  {
    uint32_t IvfSpTopkL2WithMaskCpuKernel::Compute(CpuKernelContext &ctx)
    {
        Inputs inputs;
        Outputs outputs;
        auto ret = GetInOutAndCheck(ctx, inputs, outputs);
        if (ret != KERNEL_STATUS_OK) {
            KERNEL_LOG_ERROR("Failed to get inputs or outputs");
            return ret;
        }

        ret = CheckInputShapes(inputs);
        if (ret != KERNEL_STATUS_OK) {
            KERNEL_LOG_ERROR("Failed to check input shapes");
            return ret;
        }
        UpdateOutputsShape(outputs);

        InitTopkHeap(outputs);

        auto funcLess = [](const float16_t a, const float16_t b) -> bool const { return a < b; };
        auto funcGreater = [](const float16_t a, const float16_t b) -> bool const { return a > b; };

        auto computeFunc = [&](size_t start, size_t end) {
            if (asc_ != 0) {
                // put greatest one to top of heap
                DoCompute(start, end, inputs, outputs, funcGreater);
            } else {
                // put least one to top of heap
                DoCompute(start, end, inputs, outputs, funcLess);
            }
        };

        uint32_t core = std::min({ CpuKernelUtils::GetCPUNum(ctx), static_cast<uint32_t>(nq_), THREAD_CNT });
        CpuKernelUtils::ParallelFor(ctx, nq_, nq_ / core, computeFunc);

        return KERNEL_STATUS_OK;
    }

    uint32_t IvfSpTopkL2WithMaskCpuKernel::GetInOutAndCheck(const CpuKernelContext &ctx,
                                                            Inputs &inputs, Outputs &outputs) const
    {
        KERNEL_LOG_INFO("IvfSpTopkL2WithMaskCpuKernel GetInOutAndCheck begin");

        inputs.maskBitNpu = ctx.Input(INPUT_NUM0);
        inputs.dists = ctx.Input(INPUT_NUM1);
        inputs.L1Indices = ctx.Input(INPUT_NUM2);
        inputs.opFlag = ctx.Input(INPUT_NUM3);
        inputs.addressOffsetOfBucket = ctx.Input(INPUT_NUM4);
        inputs.attr = ctx.Input(INPUT_NUM5);

        outputs.distsRes = ctx.Output(INPUT_NUM0);
        outputs.addressOffsetL3 = ctx.Output(INPUT_NUM1);
        outputs.idAddressL3 = ctx.Output(INPUT_NUM2);
        outputs.maskByteNpu = ctx.Output(INPUT_NUM3);
        outputs.isMaskOffset = ctx.Output(INPUT_NUM4);

        KERNEL_CHECK_NULLPTR(inputs.maskBitNpu, KERNEL_STATUS_PARAM_INVALID, "Get input[0], name[maskBitNpu] failed");
        KERNEL_CHECK_NULLPTR(inputs.dists, KERNEL_STATUS_PARAM_INVALID, "Get input[0], name[indists] failed");
        KERNEL_CHECK_NULLPTR(inputs.L1Indices, KERNEL_STATUS_PARAM_INVALID, "Get input[1], name[L1Indices] failed");
        KERNEL_CHECK_NULLPTR(inputs.opFlag, KERNEL_STATUS_PARAM_INVALID, "Get input[2], name[opFlag] failed");
        KERNEL_CHECK_NULLPTR(inputs.addressOffsetOfBucket, KERNEL_STATUS_PARAM_INVALID,
                             "Get input[3], name[baseAddressOffsetOfBucket] failed");
        KERNEL_CHECK_NULLPTR(inputs.attr, KERNEL_STATUS_PARAM_INVALID, "Get input[6], name[attr] failed");
        KERNEL_CHECK_NULLPTR(outputs.distsRes, KERNEL_STATUS_PARAM_INVALID, "Get output[0], name[distsRes] failed");
        KERNEL_CHECK_NULLPTR(outputs.addressOffsetL3, KERNEL_STATUS_PARAM_INVALID,
                             "Get output[2], name[addressOffsetL3] failed");
        KERNEL_CHECK_NULLPTR(outputs.maskByteNpu, KERNEL_STATUS_PARAM_INVALID,
                             "Get output[3], name[maskByteNpu] failed");
        KERNEL_CHECK_NULLPTR(outputs.isMaskOffset, KERNEL_STATUS_PARAM_INVALID,
                             "Get output[4], name[isMaskOffset] failed");

        KERNEL_LOG_INFO("Shape of input[0][indists] is %s",
                        ShapeToString(inputs.dists->GetTensorShape()->GetDimSizes()).c_str());
        KERNEL_LOG_INFO("Shape of input[1][L1Indices] is %s",
                        ShapeToString(inputs.L1Indices->GetTensorShape()->GetDimSizes()).c_str());
        KERNEL_LOG_INFO("Shape of input[2][opFlag] is %s",
                        ShapeToString(inputs.opFlag->GetTensorShape()->GetDimSizes()).c_str());
        KERNEL_LOG_INFO("Shape of input[3][addressOffsetOfBucket] is %s",
                        ShapeToString(inputs.addressOffsetOfBucket->GetTensorShape()->GetDimSizes()).c_str());
        KERNEL_LOG_INFO("Shape of input[4][attr] is %s",
                        ShapeToString(inputs.attr->GetTensorShape()->GetDimSizes()).c_str());

        return KERNEL_STATUS_OK;
    }

    uint32_t IvfSpTopkL2WithMaskCpuKernel::CheckInputShapes(const Inputs &inputs)
    {
        KERNEL_LOG_INFO("topk2CpuKernel CheckInputShapes begin");

        auto shapeIndists = inputs.dists->GetTensorShape();
        auto shapeL1Indices = inputs.L1Indices->GetTensorShape();
        auto shapeOpFlag = inputs.opFlag->GetTensorShape();
        auto addressOffset = inputs.addressOffsetOfBucket->GetTensorShape();
        auto shapeAttr = inputs.attr->GetTensorShape();

        KERNEL_CHECK_TRUE(shapeIndists->GetDims() == INPUT_NUM2, KERNEL_STATUS_PARAM_INVALID,
                          "Dims of input[0][indists] must be 2");
        KERNEL_CHECK_TRUE(shapeL1Indices->GetDims() == INPUT_NUM2, KERNEL_STATUS_PARAM_INVALID,
                          "Dims of input[1][size] must be 2");
        KERNEL_CHECK_TRUE(shapeOpFlag->GetDims() == INPUT_NUM2, KERNEL_STATUS_PARAM_INVALID,
                          "Dims of input[2][size] must be 2");
        KERNEL_CHECK_TRUE(addressOffset->GetDims() == INPUT_NUM1, KERNEL_STATUS_PARAM_INVALID,
                          "Dims of input[3][size] must be 2");
        KERNEL_CHECK_TRUE(shapeAttr->GetDims() == INPUT_NUM1, KERNEL_STATUS_PARAM_INVALID,
                          "Dims of input[6][size] must be 1");

        auto attr = static_cast<int64_t *>(inputs.attr->GetData());

        auto nq0 = shapeIndists->GetDimSize(INPUT_NUM0);
        auto nsize = shapeIndists->GetDimSize(INPUT_NUM1); // nprobeL1 *nListL2
        auto coreNum0 = shapeOpFlag->GetDimSize(INPUT_NUM0);
        flagSize_ = shapeOpFlag->GetDimSize(INPUT_NUM1);
        nq_ = nq0;
        nsize_ = nsize;
        coreNum_ = coreNum0;

        asc_ = *(attr + 0); // 0 for put greatest one to top of heap, 1 for put least one to top of heap
        nProbeL2 = *(attr + 1);
        nListL2_ = *(attr + 2);
        segmentSize_ = *(attr + 3);
        bucketNum = *(attr + 4);
        subSpaceDim2 = *(attr + 5);
        segmentNumL3 = *(attr + 6);

        KERNEL_CHECK_TRUE(nProbeL2 > 0, KERNEL_STATUS_PARAM_INVALID,
                          "k must ge 0");

        return KERNEL_STATUS_OK;
    }

    void IvfSpTopkL2WithMaskCpuKernel::UpdateOutputsShape(Outputs &outputs)
    {
        KERNEL_LOG_INFO("topk2CpuKernel UpdateOutputsShape begin");

        auto shapeOutdists = outputs.distsRes->GetTensorShape();
        std::vector<int64_t> dimOutdists;
        dimOutdists.push_back(nq_);
        dimOutdists.push_back(nProbeL2);
        shapeOutdists->SetDimSizes(dimOutdists);

        auto shapeAddressOffsetL3 = outputs.addressOffsetL3->GetTensorShape();
        std::vector<int64_t> dimAddressOffsetL3;
        dimAddressOffsetL3.push_back(nq_);
        dimAddressOffsetL3.push_back(nProbeL2 * 6);
        shapeAddressOffsetL3->SetDimSizes(dimAddressOffsetL3);

        auto shapeIdAddressL3 = outputs.idAddressL3->GetTensorShape();
        std::vector<int64_t> dimIdAddressL3;
        dimIdAddressL3.push_back(nq_);
        dimIdAddressL3.push_back(nProbeL2 * 2);
        shapeIdAddressL3->SetDimSizes(dimIdAddressL3);
    }

    void IvfSpTopkL2WithMaskCpuKernel::InitTopkHeap(Outputs &outputs) const
    {
        uint16_t *outdists = static_cast<uint16_t *>(outputs.distsRes->GetData());
        if (asc_ != 0) {
            std::fill_n(outdists, nq_ * nProbeL2, 0x7bff);
        } else {
            std::fill_n(outdists, nq_ * nProbeL2, 0xfbff);
        }
    }

    template <typename C>
    void IvfSpTopkL2WithMaskCpuKernel::DoCompute(size_t start, size_t end,
                                                 const Inputs &inputs, Outputs &outputs, C &&cmp)
    {
        KernelTensor<uint8_t> inmaskBitNpu(inputs.maskBitNpu);
        KernelTensor<float16_t> indists(inputs.dists);
        KernelTensor<uint16_t> L1Indices(inputs.L1Indices);
        KernelTensor<uint16_t> opFlag(inputs.opFlag);

        KernelTensor<uint64_t> addressOffsetOfBucket(inputs.addressOffsetOfBucket);

        KernelTensor<float16_t> outdists(outputs.distsRes);
        KernelTensor<uint64_t> outOffset(outputs.addressOffsetL3);
        KernelTensor<uint64_t> outIdAddress(outputs.idAddressL3);

        KernelTensor<uint8_t> outmaskByteNpu(outputs.maskByteNpu);
        KernelTensor<uint64_t> outIsMaskOffset(outputs.isMaskOffset);

        auto flagPtr = opFlag.GetSubTensorDim0(0);
        for (int64_t j = 0; j < coreNum_; j++) {
            WAITING_FLAG_READY(*(flagPtr + j * flagSize_), TIMEOUT_CHECK_TICK, TIMEOUT_MS);
        }

        for (size_t j = start; j < end; j++) {
            ComputeBlock<C>(j, inmaskBitNpu, indists, L1Indices, addressOffsetOfBucket, outdists, outOffset,
                            outIdAddress, outmaskByteNpu, outIsMaskOffset, cmp);
        }
    }

    template <typename C>
    void IvfSpTopkL2WithMaskCpuKernel::ComputeBlock(size_t n,
                                                    KernelTensor<uint8_t> &inmaskBitNpuTensor,
                                                    KernelTensor<float16_t> &indistsTensor,
                                                    KernelTensor<uint16_t> &l1IndicesTensor,
                                                    KernelTensor<uint64_t> &offsetOfBucketTensor,
                                                    KernelTensor<float16_t> &outdistsTensor,
                                                    KernelTensor<uint64_t> &outOffsetTensor,
                                                    KernelTensor<uint64_t> &outIdAddressTensor,
                                                    KernelTensor<uint8_t> &outmaskByteNpuTensor,
                                                    KernelTensor<uint64_t> &outIsMaskOffsetTensor,
                                                    C &&cmp)
    {
        uint8_t* inmaskBit = inmaskBitNpuTensor.GetSubTensorDim0(0);
        float16_t *indists = indistsTensor.GetSubTensorDim0(n);
        uint16_t *l1Indices = l1IndicesTensor.GetSubTensorDim0(n);
        uint64_t* offsetOfBucket = offsetOfBucketTensor.GetSubTensorDim0(0);
        uint64_t* inNormL2OffsetOfBucket = offsetOfBucket + bucketNum * 2;
        uint64_t* inIdOffsetOfBucket = offsetOfBucket + bucketNum * 4;

        float16_t *outdists = outdistsTensor.GetSubTensorDim0(n);
        uint64_t* outOffsetOfQuery = outOffsetTensor.GetSubTensorDim0(n);
        uint64_t* outIdAddressOfQuery = outIdAddressTensor.GetSubTensorDim0(n);
        uint8_t* outmaskByte = outmaskByteNpuTensor.GetSubTensorDim0(0);

        uint64_t *outlabel = outOffsetOfQuery;
        std::fill_n(outlabel, nProbeL2, 0xffffffff);

        uint64_t *outsumSegmentNum = outOffsetOfQuery + nProbeL2;
        uint64_t *outBaseOffsetMask = outOffsetOfQuery + nProbeL2 * 2;
        uint64_t *outNormL2OffsetMask = outOffsetOfQuery + nProbeL2 * 4;
        uint64_t *outIdOffsetMask = outIdAddressOfQuery;

        int64_t idx = 0;
        // process tail data
        while (idx < nsize_) {
            if (cmp(outdists[0], indists[idx])) {
                uint64_t globalL2ID = l1Indices[idx / nListL2_] * nListL2_ + idx % nListL2_;
                UpdateHeap<C>(outdists, outlabel, nProbeL2, indists[idx], globalL2ID, cmp);
            }
            ++idx;
        }

        // sort heap results
        if (nProbeL2 > 1) {
            std::vector<std::pair<float16_t, uint64_t>> vecMask(nProbeL2);
            auto pairCompare = [](const std::pair<float16_t, uint64_t>& a, const std::pair<float16_t, uint64_t>& b) {
                return a.first > b.first;
            };
            for (int i = 0; i < nProbeL2; ++i) {
                vecMask[i] = std::make_pair(outdists[i], outlabel[i]);
            }
            std::sort(vecMask.begin(), vecMask.end(), pairCompare);
            for (int i = 0; i < nProbeL2; ++i) {
                outdists[i] = vecMask[i].first;
                outlabel[i] = vecMask[i].second;
            }
        }

        // get selected buckets's Offset into outOffset
        uint64_t minNormL2OffsetMask =  inNormL2OffsetOfBucket[outlabel[0] * STORE_INTERVAL];
        uint64_t maxNormL2OffsetMask = inNormL2OffsetOfBucket[outlabel[0] * STORE_INTERVAL + 1];

        for (int i = 0; i < nProbeL2; ++i) {
            outNormL2OffsetMask[i * STORE_INTERVAL] = inNormL2OffsetOfBucket[outlabel[i] * STORE_INTERVAL];
            outNormL2OffsetMask[i * STORE_INTERVAL + 1] = inNormL2OffsetOfBucket[outlabel[i] * STORE_INTERVAL + 1];
            minNormL2OffsetMask = std::min(minNormL2OffsetMask, outNormL2OffsetMask[i * STORE_INTERVAL]);
            maxNormL2OffsetMask = std::max(maxNormL2OffsetMask, outNormL2OffsetMask[i * STORE_INTERVAL + 1]);
        }

        const uint64_t totalSegNum = (maxNormL2OffsetMask - minNormL2OffsetMask) / segmentSize_;
        std::vector<uint64_t> bitSet((totalSegNum + 63) / 64, 0);

        int segCnt = 0;
        for (int i = 0; i < nProbeL2; ++i) {
            auto startOffset = outNormL2OffsetMask[i * STORE_INTERVAL];
            auto endOffset = outNormL2OffsetMask[i * STORE_INTERVAL + 1];
            auto startIdx = (startOffset - minNormL2OffsetMask) / segmentSize_;
            auto endIdx = (endOffset - minNormL2OffsetMask) / segmentSize_;
            uint64_t  actualK0Mask = startIdx;
            uint64_t  actualK1Mask = startIdx;
            bool flag = false;
            for (auto k = startIdx; k < endIdx; k++) {
                if (segCnt == segmentNumL3) {
                    break;
                }
                auto k0Mask = k / 64;
                auto k1Mask = k % 64;
                if ((bitSet[k0Mask] & (1L << k1Mask)) == 0) {
                    // 如果当前segment没有被占领
                    bitSet[k0Mask] =  bitSet[k0Mask] | (1L << k1Mask); // 进行标记
                    segCnt++;
                    flag = true;
                    actualK1Mask = k + 1;
                } else {
                    // 如果当前segment被占领
                    if (!flag) {
                        actualK0Mask++;
                    } else {
                        actualK1Mask = k;
                        break;
                    }
                }
            }
            if (actualK0Mask > actualK1Mask) {
                actualK1Mask = actualK0Mask;
            }
            // 更新
            outBaseOffsetMask[i * STORE_INTERVAL] = (actualK0Mask - startIdx) * segmentSize_ * subSpaceDim2 +
                offsetOfBucket[outlabel[i] * STORE_INTERVAL];
            outBaseOffsetMask[i * STORE_INTERVAL + 1] = (actualK1Mask - startIdx) * segmentSize_ * subSpaceDim2 +
                offsetOfBucket[outlabel[i] * STORE_INTERVAL];
            outNormL2OffsetMask[i * STORE_INTERVAL] = actualK0Mask * segmentSize_ + minNormL2OffsetMask;
            outNormL2OffsetMask[i * STORE_INTERVAL + 1] = actualK1Mask * segmentSize_ + minNormL2OffsetMask;

            outIdOffsetMask[i * STORE_INTERVAL] = (actualK0Mask - startIdx) * segmentSize_ * sizeof(int64_t) +
                inIdOffsetOfBucket[outlabel[i] * STORE_INTERVAL];
            outIdOffsetMask[i * STORE_INTERVAL + 1] = (actualK1Mask - startIdx) * segmentSize_ * sizeof(int64_t) +
                inIdOffsetOfBucket[outlabel[i] * STORE_INTERVAL];
            if (outIdOffsetMask[i * STORE_INTERVAL] > inIdOffsetOfBucket[outlabel[i] * STORE_INTERVAL + 1]) {
                outIdOffsetMask[i * STORE_INTERVAL] = inIdOffsetOfBucket[outlabel[i] * STORE_INTERVAL + 1];
            }
            if (outIdOffsetMask[i * STORE_INTERVAL + 1] > inIdOffsetOfBucket[outlabel[i] * STORE_INTERVAL + 1]) {
                outIdOffsetMask[i * STORE_INTERVAL + 1] = inIdOffsetOfBucket[outlabel[i] * STORE_INTERVAL + 1];
            }
            {
                uint64_t start = outIdOffsetMask[STORE_INTERVAL * i];
                uint64_t end = outIdOffsetMask[STORE_INTERVAL * i + 1];
                uint64_t offset = (end - start) / sizeof(uint64_t);
                std::vector<int64_t> realIds(offset, 0);
                int64_t *idStart = static_cast<int64_t*>(reinterpret_cast<void *>(start));

                if (offset > 0) {
                    auto ret = memcpy_s(realIds.data(), sizeof(int64_t) * offset, idStart, sizeof(int64_t) * offset);
                    if (ret != 0) {
                        KERNEL_LOG_ERROR("Copy data shaped error %d", ret);
                        return;
                    }
                    for (size_t j = 0; j < offset; j++) {
                        uint64_t realIdX = realIds[j] / 8;
                        uint64_t realIdY = realIds[j] % 8;
                        if (inmaskBit[realIdX] & (1 << realIdY)) {
                            outmaskByte[outNormL2OffsetMask[i * STORE_INTERVAL] + j] = 1;
                        } else {
                            outmaskByte[outNormL2OffsetMask[i * STORE_INTERVAL] + j] = 0;
                        }
                    }
                }
            }
        }

        // calculate sumSegmentNum
        outsumSegmentNum[0] = (outNormL2OffsetMask[1] - outNormL2OffsetMask[0]) / segmentSize_;
        for (int i = 1; i < nProbeL2; ++i) {
            outsumSegmentNum[i] = (outNormL2OffsetMask[i * STORE_INTERVAL + 1] -
            outNormL2OffsetMask[i * STORE_INTERVAL]) / segmentSize_ +
                    outsumSegmentNum[i - 1] ;
        }
    }

    template <typename C>
    void IvfSpTopkL2WithMaskCpuKernel::UpdateHeap(float16_t *dists, uint64_t *label, int64_t len, float16_t pushDist,
                                                  uint64_t index, C &&cmp)
    {
        size_t i = UpdateHeapImpl(dists, label, len, pushDist, cmp);
        dists[i] = pushDist;
        label[i] = index;
    }

    REGISTER_CPU_KERNEL(IVF_SP_TOPK_L2_WITH_MASK, IvfSpTopkL2WithMaskCpuKernel);
} // namespace aicpu
