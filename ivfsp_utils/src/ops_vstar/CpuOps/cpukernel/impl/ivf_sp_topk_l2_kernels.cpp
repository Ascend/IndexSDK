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

#include "ivf_sp_topk_l2_kernels.h"
#include <algorithm>
#include <string>
#include <map>
#include <numeric>
#include "cpu_kernel.h"
#include "cust_cpu_utils.h"
#include "utils/cpu_kernel_utils.h"
#include "utils/kernel_tensor.h"
#include "utils/kernel_utils.h"
#include "utils/kernel_shared_def.h"

namespace  {
    const char *IVF_SP_TOPK_L2 = "IvfSpTopkL2";
    const uint32_t THREAD_CNT = 6;
    // 该算子中，所有输入输出的xxxOffset变量，均以所选的桶 (nProbe) 为单位，且存储2个值:
    // 每个桶当前变量的的起始地址和结束地址; 以该常量作为每个桶的偏移量(每次偏移两个单位)
    const int STORE_INTERVAL = 2;
}

namespace aicpu  {
    uint32_t IvfSpTopkL2CpuKernel::Compute(CpuKernelContext &ctx)
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

    uint32_t IvfSpTopkL2CpuKernel::GetInOutAndCheck(const CpuKernelContext &ctx, Inputs &inputs, Outputs &outputs) const
    {
        KERNEL_LOG_INFO("IvfSpTopkL2CpuKernel GetInOutAndCheck begin");

        inputs.dists = ctx.Input(INPUT_NUM0);
        inputs.L1Indices = ctx.Input(INPUT_NUM1);
        inputs.opFlag = ctx.Input(INPUT_NUM2);
        inputs.addressOffsetOfBucket = ctx.Input(INPUT_NUM3);
        inputs.attr = ctx.Input(INPUT_NUM4);

        outputs.distsRes = ctx.Output(INPUT_NUM0);
        outputs.addressOffsetL3 = ctx.Output(INPUT_NUM1);
        outputs.idAddressL3 = ctx.Output(INPUT_NUM2);

        KERNEL_CHECK_NULLPTR(inputs.dists, KERNEL_STATUS_PARAM_INVALID, "Get input[0], name[indists] failed");
        KERNEL_CHECK_NULLPTR(inputs.L1Indices, KERNEL_STATUS_PARAM_INVALID, "Get input[1], name[L1Indices] failed");
        KERNEL_CHECK_NULLPTR(inputs.opFlag, KERNEL_STATUS_PARAM_INVALID, "Get input[2], name[opFlag] failed");
        KERNEL_CHECK_NULLPTR(inputs.addressOffsetOfBucket, KERNEL_STATUS_PARAM_INVALID,
                             "Get input[3], name[baseAddressOffsetOfBucket] failed");
        KERNEL_CHECK_NULLPTR(inputs.attr, KERNEL_STATUS_PARAM_INVALID, "Get input[6], name[attr] failed");
        KERNEL_CHECK_NULLPTR(outputs.distsRes, KERNEL_STATUS_PARAM_INVALID, "Get output[0], name[distsRes] failed");
        KERNEL_CHECK_NULLPTR(outputs.addressOffsetL3, KERNEL_STATUS_PARAM_INVALID,
                             "Get output[2], name[addressOffsetL3] failed");

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

    uint32_t IvfSpTopkL2CpuKernel::CheckInputShapes(const Inputs &inputs)
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

    void IvfSpTopkL2CpuKernel::UpdateOutputsShape(Outputs &outputs)
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

    void IvfSpTopkL2CpuKernel::InitTopkHeap(Outputs &outputs) const
    {
        uint16_t *outdists = static_cast<uint16_t *>(outputs.distsRes->GetData());
        if (asc_ != 0) {
            std::fill_n(outdists, nq_ * nProbeL2, 0x7bff);
        } else {
            std::fill_n(outdists, nq_ * nProbeL2, 0xfbff);
        }
    }

    template <typename C>
    void IvfSpTopkL2CpuKernel::DoCompute(size_t start, size_t end, const Inputs &inputs, Outputs &outputs, C &&cmp)
    {
        KernelTensor<float16_t> indists(inputs.dists);
        KernelTensor<uint16_t> L1Indices(inputs.L1Indices);
        KernelTensor<uint16_t> opFlag(inputs.opFlag);

        KernelTensor<uint64_t> addressOffsetOfBucket(inputs.addressOffsetOfBucket);

        KernelTensor<float16_t> outdists(outputs.distsRes);
        KernelTensor<uint64_t> outOffset(outputs.addressOffsetL3);
        KernelTensor<uint64_t> outIdAddress(outputs.idAddressL3);

        auto flagPtr = opFlag.GetSubTensorDim0(0);
        for (int64_t j = 0; j < coreNum_; j++) {
            WAITING_FLAG_READY(*(flagPtr + j * flagSize_), TIMEOUT_CHECK_TICK, TIMEOUT_MS);
        }

        for (size_t j = start; j < end; j++) {
            ComputeBlock<C>(j, indists, L1Indices, addressOffsetOfBucket, outdists, outOffset, outIdAddress, cmp);
        }
    }

    template <typename C>
    void IvfSpTopkL2CpuKernel::ComputeBlock(size_t n,
                                            KernelTensor<float16_t> &indistsTensor,
                                            KernelTensor<uint16_t> &l1IndicesTensor,
                                            KernelTensor<uint64_t> &offsetOfBucketTensor,
                                            KernelTensor<float16_t> &outdistsTensor,
                                            KernelTensor<uint64_t> &outOffsetTensor,
                                            KernelTensor<uint64_t> &outIdAddressTensor,
                                            C &&cmp)
    {
        float16_t *indists = indistsTensor.GetSubTensorDim0(n);
        uint16_t *l1Indices  = l1IndicesTensor.GetSubTensorDim0(n);
        uint64_t* offsetOfBucket = offsetOfBucketTensor.GetSubTensorDim0(0);
        uint64_t* inNormL2OffsetOfBucket = offsetOfBucket + bucketNum * 2;
        uint64_t* inIdOffsetOfBucket = offsetOfBucket + bucketNum * 4;

        float16_t *outdists = outdistsTensor.GetSubTensorDim0(n);
        uint64_t* outOffsetOfQuery = outOffsetTensor.GetSubTensorDim0(n);
        uint64_t* outIdAddressOfQuery = outIdAddressTensor.GetSubTensorDim0(n);

        uint64_t *outlabel = outOffsetOfQuery;
        std::fill_n(outlabel, nProbeL2, 0xffffffff);

        uint64_t *outsumSegmentNum = outOffsetOfQuery + nProbeL2;
        uint64_t *outBaseOffset = outOffsetOfQuery + nProbeL2 * 2;
        uint64_t *outNormL2Offset = outOffsetOfQuery + nProbeL2 * 4;
        uint64_t *outIdOffset = outIdAddressOfQuery;

        int64_t idx = 0;

        // process tail data
        while (idx < nsize_) {
            if (cmp(outdists[0], indists[idx])) {
                uint64_t globalL2ID = l1Indices [idx / nListL2_] * nListL2_ + idx % nListL2_;
                UpdateHeap<C>(outdists, outlabel, nProbeL2, indists[idx], globalL2ID, cmp);
            }
            ++idx;
        }

        // sort heap results
        if (nProbeL2 > 1) {
            std::vector<std::pair<float16_t, uint64_t>> vec(nProbeL2);
            auto pairCompare = [](const std::pair<float16_t, uint64_t>& a, const std::pair<float16_t, uint64_t>& b) {
                return a.first > b.first;
            };
            for (int i = 0; i < nProbeL2; ++i) {
                vec[i] = std::make_pair(outdists[i], outlabel[i]);
            }
            std::sort(vec.begin(), vec.end(), pairCompare);
            for (int i = 0; i < nProbeL2; ++i) {
                outdists[i] = vec[i].first;
                outlabel[i] = vec[i].second;
            }
        }

        // get selected buckets's Offset into outOffset
        uint64_t minNormL2Offset =  inNormL2OffsetOfBucket[outlabel[0] * STORE_INTERVAL];
        uint64_t maxNormL2Offset = inNormL2OffsetOfBucket[outlabel[0] * STORE_INTERVAL + 1];

        for (int i = 0; i < nProbeL2; ++i) {
            outNormL2Offset[i * STORE_INTERVAL] = inNormL2OffsetOfBucket[outlabel[i] * STORE_INTERVAL];
            outNormL2Offset[i * STORE_INTERVAL + 1] = inNormL2OffsetOfBucket[outlabel[i] * STORE_INTERVAL + 1];
            minNormL2Offset = std::min(minNormL2Offset, outNormL2Offset[i * STORE_INTERVAL]);
            maxNormL2Offset = std::max(maxNormL2Offset, outNormL2Offset[i * STORE_INTERVAL + 1]);
        }

        const uint64_t totalSegNum = (maxNormL2Offset - minNormL2Offset) / segmentSize_;
        std::vector<uint64_t> bitSet((totalSegNum + 63) / 64, 0);

        int segCnt = 0;
        for (int i = 0; i < nProbeL2; ++i) {
            auto startOffset = outNormL2Offset[i * STORE_INTERVAL];
            auto endOffset = outNormL2Offset[i * STORE_INTERVAL + 1];
            auto startIdx = (startOffset - minNormL2Offset) / segmentSize_;
            auto endIdx = (endOffset - minNormL2Offset) / segmentSize_;
            uint64_t  actualK0 = startIdx;
            uint64_t  actualK1 = startIdx;
            bool flag = false;
            for (auto k = startIdx; k < endIdx; k++) {
                if (segCnt == segmentNumL3) {
                    break;
                }
                auto k0 = k / 64;
                auto k1 = k % 64;
                if ((bitSet[k0] & (1L << k1)) == 0) {
                    // 如果当前segment没有被占领
                    bitSet[k0] =  bitSet[k0] | (1L << k1); // 进行标记
                    segCnt++;
                    flag = true;
                    actualK1 = k + 1;
                } else {
                    // 如果当前segment被占领
                    if (!flag) {
                        actualK0++;
                    } else {
                        actualK1 = k;
                        break;
                    }
                }
            }
            if (actualK0 > actualK1) {
                actualK1 = actualK0;
            }
            // 更新
            outBaseOffset[i * STORE_INTERVAL] = (actualK0 - startIdx) * segmentSize_ * subSpaceDim2 +
                offsetOfBucket[outlabel[i] * STORE_INTERVAL];
            outBaseOffset[i * STORE_INTERVAL + 1] = (actualK1 - startIdx) * segmentSize_ * subSpaceDim2 +
                offsetOfBucket[outlabel[i] * STORE_INTERVAL];
            outNormL2Offset[i * STORE_INTERVAL] = actualK0 * segmentSize_ + minNormL2Offset;
            outNormL2Offset[i * STORE_INTERVAL + 1] = actualK1 * segmentSize_ + minNormL2Offset;

            outIdOffset[i * STORE_INTERVAL] = (actualK0 - startIdx) * segmentSize_ * sizeof(int64_t) +
                inIdOffsetOfBucket[outlabel[i] * STORE_INTERVAL];
            outIdOffset[i * STORE_INTERVAL + 1] = (actualK1 - startIdx) * segmentSize_ * sizeof(int64_t) +
                inIdOffsetOfBucket[outlabel[i] * STORE_INTERVAL];
            if (outIdOffset[i * STORE_INTERVAL] > inIdOffsetOfBucket[outlabel[i] * STORE_INTERVAL + 1]) {
                outIdOffset[i * STORE_INTERVAL] = inIdOffsetOfBucket[outlabel[i] * STORE_INTERVAL + 1];
            }
            if (outIdOffset[i * STORE_INTERVAL + 1] > inIdOffsetOfBucket[outlabel[i] * STORE_INTERVAL + 1]) {
                outIdOffset[i * STORE_INTERVAL + 1] = inIdOffsetOfBucket[outlabel[i] * STORE_INTERVAL + 1];
            }
        }

        // calculate sumSegmentNum
        outsumSegmentNum[0] = (outNormL2Offset[1] - outNormL2Offset[0]) / segmentSize_;
        for (int i = 1; i < nProbeL2; ++i) {
            outsumSegmentNum[i] = (outNormL2Offset[i * STORE_INTERVAL + 1] -
            outNormL2Offset[i * STORE_INTERVAL]) / segmentSize_ +
                outsumSegmentNum[i - 1] ;
        }
    }

    template <typename C>
    void IvfSpTopkL2CpuKernel::UpdateHeap(float16_t *dists, uint64_t *label, int64_t len, float16_t pushDist,
                                          uint64_t index, C &&cmp)
    {
        size_t i = UpdateHeapImpl(dists, label, len, pushDist, cmp);
        dists[i] = pushDist;
        label[i] = index;
    }

    REGISTER_CPU_KERNEL(IVF_SP_TOPK_L2, IvfSpTopkL2CpuKernel);
} // namespace aicpu
