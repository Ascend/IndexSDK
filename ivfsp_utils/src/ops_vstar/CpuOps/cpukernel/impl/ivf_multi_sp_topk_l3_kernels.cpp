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


#include "ivf_multi_sp_topk_l3_kernels.h"

#include <algorithm>
#include <string>
#include <map>
#include <cstring>
#include "securec.h"

#include "cpu_kernel.h"
#include "utils/cpu_kernel_utils.h"
#include "utils/kernel_tensor.h"
#include "utils/kernel_utils.h"
#include "utils/kernel_shared_def.h"

namespace {
    const char *IVF_MULTI_SP_TOPK_L3 = "IvfMultiSpTopkL3";
    const uint32_t THREAD_CNT = 6;
}

namespace aicpu {
    uint32_t IvfMultiSpTopkL3CpuKernel::Compute(CpuKernelContext &ctx)
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

        if (labelType_ == DT_INT64) {
            InitTopkHeap<int64_t>(outputs);
        } else if (labelType_ == DT_UINT16) {
            InitTopkHeap<uint16_t>(outputs);
        } else {
            KERNEL_LOG_ERROR("Invalid datatype");
        }

        auto funcLess = [](const float16_t a, const float16_t b) -> bool const { return a < b; };
        auto funcGreater = [](const float16_t a, const float16_t b) -> bool const { return a > b; };
#ifdef AICPU_UTEST
        uint32_t core = 1;
#else
        uint32_t core = std::min({ CpuKernelUtils::GetCPUNum(ctx), static_cast<uint32_t>(indexNum_), THREAD_CNT });
#endif

        auto computeFunc = [&](size_t start, size_t end) {
            (void)end; // end is unuseful in this function
            if (asc_ != 0) {
                // put greatest one to top of heap
                if (labelType_ == DT_INT64) {
                    DoCompute<int64_t>(core, start, inputs, outputs, funcGreater);
                } else if (labelType_ == DT_UINT16) {
                    DoCompute<uint16_t>(core, start, inputs, outputs, funcGreater);
                } else {
                    KERNEL_LOG_ERROR("Invalid datatype");
                }
            } else {
                // put least one to top of heap
                if (labelType_ == DT_INT64) {
                    DoCompute<int64_t>(core, start, inputs, outputs, funcLess);
                } else if (labelType_ == DT_UINT16) {
                    DoCompute<uint16_t>(core, start, inputs, outputs, funcLess);
                } else {
                    KERNEL_LOG_ERROR("Invalid datatype");
                }
            }
        };
#ifdef AICPU_UTEST
        computeFunc(0, nq_);
#else
        CpuKernelUtils::ParallelFor(ctx, core, 1, computeFunc);
#endif

        return KERNEL_STATUS_OK;
    }

    uint32_t IvfMultiSpTopkL3CpuKernel::GetInOutAndCheck(const CpuKernelContext &ctx,
                                                         Inputs &inputs, Outputs &outputs) const
    {
        KERNEL_LOG_INFO("IvfMultiSpTopkL3CpuKernel GetInOutAndCheck begin");

        inputs.indists = ctx.Input(INPUT_NUM0);
        inputs.vmdists = ctx.Input(INPUT_NUM1);
        inputs.idaddress = ctx.Input(INPUT_NUM2);
        inputs.opflag = ctx.Input(INPUT_NUM3);
        inputs.attr = ctx.Input(INPUT_NUM4);
        inputs.l2indices = ctx.Input(INPUT_NUM5);

        outputs.outdists = ctx.Output(INPUT_NUM0);
        outputs.outlabels = ctx.Output(INPUT_NUM1);

        KERNEL_CHECK_NULLPTR(inputs.indists, KERNEL_STATUS_PARAM_INVALID, "Get input[0], name[indists] failed");
        KERNEL_CHECK_NULLPTR(inputs.vmdists, KERNEL_STATUS_PARAM_INVALID, "Get input[1], name[vmdists] failed");
        KERNEL_CHECK_NULLPTR(inputs.idaddress, KERNEL_STATUS_PARAM_INVALID, "Get input[2], name[idaddress] failed");
        KERNEL_CHECK_NULLPTR(inputs.opflag, KERNEL_STATUS_PARAM_INVALID, "Get input[3], name[opflag] failed");
        KERNEL_CHECK_NULLPTR(inputs.attr, KERNEL_STATUS_PARAM_INVALID, "Get input[4], name[attr] failed");
        KERNEL_CHECK_NULLPTR(inputs.l2indices, KERNEL_STATUS_PARAM_INVALID, "Get input[5], name[l2indices] failed");
        KERNEL_CHECK_NULLPTR(outputs.outdists, KERNEL_STATUS_PARAM_INVALID, "Get output[0], name[outdists] failed");
        KERNEL_CHECK_NULLPTR(outputs.outlabels, KERNEL_STATUS_PARAM_INVALID, "Get output[1], name[outlabels] failed");

        KERNEL_LOG_INFO("Shape of input[0][indists] is %s",
                        ShapeToString(inputs.indists->GetTensorShape()->GetDimSizes()).c_str());
        KERNEL_LOG_INFO("Shape of input[1][vmdists] is %s",
                        ShapeToString(inputs.vmdists->GetTensorShape()->GetDimSizes()).c_str());
        KERNEL_LOG_INFO("Shape of input[2][idaddress] is %s",
                        ShapeToString(inputs.idaddress->GetTensorShape()->GetDimSizes()).c_str());
        KERNEL_LOG_INFO("Shape of input[3][opflag] is %s",
                        ShapeToString(inputs.opflag->GetTensorShape()->GetDimSizes()).c_str());
        KERNEL_LOG_INFO("Shape of input[4][attr] is %s",
                        ShapeToString(inputs.attr->GetTensorShape()->GetDimSizes()).c_str());
        KERNEL_LOG_INFO("Shape of input[5][l2indices] is %s",
                        ShapeToString(inputs.l2indices->GetTensorShape()->GetDimSizes()).c_str());

        return KERNEL_STATUS_OK;
    }

    uint32_t IvfMultiSpTopkL3CpuKernel::CheckInputShapes(const Inputs &inputs)
    {
        KERNEL_LOG_INFO("IvfSpTopkL3puKernel CheckInputShapes begin");

        auto shapeIndists = inputs.indists->GetTensorShape();
        auto shapeVmdists = inputs.vmdists->GetTensorShape();
        auto shapeIdaddress = inputs.idaddress->GetTensorShape();
        auto shapeOpflag = inputs.opflag->GetTensorShape();
        auto shapeAttr = inputs.attr->GetTensorShape();
        auto shapeL2indices = inputs.l2indices->GetTensorShape();

        KERNEL_CHECK_TRUE(shapeIndists->GetDims() == INPUT_NUM3, KERNEL_STATUS_PARAM_INVALID,
                          "Dims of input[0][indists] must be 3");
        KERNEL_CHECK_TRUE(shapeVmdists->GetDims() == INPUT_NUM3, KERNEL_STATUS_PARAM_INVALID,
                          "Dims of input[0][vmdists] must be 3");
        KERNEL_CHECK_TRUE(shapeIdaddress->GetDims() == INPUT_NUM3, KERNEL_STATUS_PARAM_INVALID,
                          "Dims of input[0][idaddress] must be 3");
        KERNEL_CHECK_TRUE(shapeOpflag->GetDims() == INPUT_NUM3, KERNEL_STATUS_PARAM_INVALID,
                          "Dims of input[0][opflag] must be 3");
        KERNEL_CHECK_TRUE(shapeAttr->GetDims() == INPUT_NUM1, KERNEL_STATUS_PARAM_INVALID,
                          "Dims of input[0][attr] must be 1");
        KERNEL_CHECK_TRUE(shapeL2indices->GetDims() == INPUT_NUM3, KERNEL_STATUS_PARAM_INVALID,
                          "Dims of input[0][l2indices] must be 3");

        auto nq0 = shapeIndists->GetDimSize(INPUT_NUM1);
        auto nq1 = shapeVmdists->GetDimSize(INPUT_NUM1);
        auto indexNum = shapeIndists->GetDimSize(INPUT_NUM0);
        KERNEL_CHECK_TRUE(nq0 == nq1, KERNEL_STATUS_PARAM_INVALID, "Nq of inputs must be same");
        nq_ = nq0;
        indexNum_ = indexNum;

        auto coreNum0 = shapeOpflag->GetDimSize(INPUT_NUM1);

        coreNum_ = coreNum0;

        flagSize_ = shapeOpflag->GetDimSize(INPUT_NUM2);

        auto attrCount = shapeAttr->GetDimSize(INPUT_NUM0);
        KERNEL_CHECK_TRUE(attrCount == 7, KERNEL_STATUS_PARAM_INVALID, "Num of attrs must be %d",
                          7);

        auto attr = static_cast<int64_t *>(inputs.attr->GetData());
        asc_ = *(attr + 0);
        k_ = *(attr + 1);
        burstLen_ = *(attr + 2);
        l3SegNum_ = *(attr + 3);
        l3SegSize_ = *(attr + 4);
        nprobeL2_ = *(attr + 5);
        quickTopk_ = *(attr + 6);

        KERNEL_CHECK_TRUE(k_ > 0 && burstLen_ > 0 && asc_ >= 0 && l3SegNum_ > 0 && l3SegSize_ > 0,
                          KERNEL_STATUS_PARAM_INVALID, "Value of asc, k, bustLen, l3SegNum, l3SegSize must ge 0");

        return KERNEL_STATUS_OK;
    }

    void IvfMultiSpTopkL3CpuKernel::UpdateOutputsShape(Outputs &outputs)
    {
        KERNEL_LOG_INFO("IvfSpTopkL3puKernel UpdateOutputsShape begin");

        auto shapeOutdists = outputs.outdists->GetTensorShape();
        std::vector<int64_t> dimOutdists;
        dimOutdists.push_back(indexNum_);
        dimOutdists.push_back(nq_);
        dimOutdists.push_back(k_);
        shapeOutdists->SetDimSizes(dimOutdists);

        auto shapeOutlabels = outputs.outlabels->GetTensorShape();
        std::vector<int64_t> dimOutlabels;
        dimOutlabels.push_back(indexNum_);
        dimOutlabels.push_back(nq_);
        dimOutlabels.push_back(k_);
        shapeOutlabels->SetDimSizes(dimOutlabels);

        labelType_ = outputs.outlabels->GetDataType();
    }

    template <typename T, typename C>
    void IvfMultiSpTopkL3CpuKernel::DoCompute(size_t tcnt, size_t start,
                                              const Inputs &inputs, Outputs &outputs, C &&cmp)
    {
        KernelTensor<float16_t> indists(inputs.indists);
        KernelTensor<float16_t> vmdists(inputs.vmdists);
        KernelTensor<uint64_t> idaddress(inputs.idaddress);
        KernelTensor<uint64_t> l2indices(inputs.l2indices);
        KernelTensor<uint16_t> opflag(inputs.opflag);

        KernelTensor<float16_t> outdists(outputs.outdists);
        KernelTensor<T> outlabels(outputs.outlabels);

        for (int64_t indexidx = start; indexidx < indexNum_; indexidx += tcnt) {
            auto flagPtr = opflag.GetSubTensorDim1(indexidx, 0);
            for (int64_t j = 0; j < coreNum_; j++) {
                WAITING_FLAG_READY(*(flagPtr + j * flagSize_), TIMEOUT_CHECK_TICK, TIMEOUT_MS);
            }
            bool reorder = 1; // reorder only last page and last block
            for (size_t j = 0; j < (size_t)nq_; j++) {
                ComputeBlock<T, C>(indexidx, j, indists, vmdists, idaddress,
                                   l2indices, outdists, outlabels, reorder, cmp);
            }
        }
    }

    template<typename T>
    void IvfMultiSpTopkL3CpuKernel::InitTopkHeap(Outputs &outputs) const
    {
        uint16_t *outdists = static_cast<uint16_t *>(outputs.outdists->GetData());
        T *outlabels = static_cast<T *>(outputs.outlabels->GetData());
        // Set initial outlables vaule -1
        std::fill_n(outlabels, indexNum_ * nq_ * k_, 0xffffffffffffffff);
        if (asc_ != 0) {
            std::fill_n(outdists, indexNum_ * nq_ * k_, 0x7bff);
        } else {
            std::fill_n(outdists, indexNum_ * nq_ * k_, 0xfbff);
        }
    }

    template <typename T, typename C>
    void IvfMultiSpTopkL3CpuKernel::ComputeBlock(int64_t indexidx, size_t n, KernelTensor<float16_t> &indistsTensor,
                                                 KernelTensor<float16_t> &vmdistsTensor,
                                                 KernelTensor<uint64_t> &idaddressTensor,
                                                 KernelTensor<uint64_t> &l2indicesTensor,
                                                 KernelTensor<float16_t> &outdistsTensor,
                                                 KernelTensor<T> &outlabelsTensor, bool reorder, C &&cmp)
    {
        float16_t *indists = indistsTensor.GetSubTensorDim1(indexidx, n);
        float16_t *vmdists = vmdistsTensor.GetSubTensorDim1(indexidx, n);
        uint64_t *idaddress = idaddressTensor.GetSubTensorDim1(indexidx, n);
        float16_t *outdists = outdistsTensor.GetSubTensorDim1(indexidx, n);
        T *outlabel = outlabelsTensor.GetSubTensorDim1(indexidx, n);

        auto ret = 0;

        std::vector<int64_t> realIds(l3SegNum_ * l3SegSize_, -1);
        int curTotalMulti = 0;
        for (int i = 0; i < nprobeL2_; i++) {
            uint64_t start = idaddress[2 * i];
            uint64_t end = idaddress[2 * i + 1];
            uint64_t offset = (end - start) / sizeof(uint64_t);
            if (offset == 0) {
                continue;
            }
            uint64_t curProbeSegNum = (offset + l3SegSize_ - 1) / l3SegSize_;

            int64_t *idStart = static_cast<int64_t*>(reinterpret_cast<void *>(start));
            size_t cpySize = std::min(static_cast<int>(offset),
                static_cast<int>(l3SegNum_ * l3SegSize_ - curTotalMulti));
            ret = memcpy_s(realIds.data() + curTotalMulti, sizeof(int64_t) * cpySize,
                           idStart, sizeof(int64_t) * cpySize);
            if (ret != 0) {
                KERNEL_LOG_ERROR("Copy data shaped error %d", ret);
                return;
            }

            // 解决无效id对应的dists
            if (curProbeSegNum * l3SegSize_ - offset > 0) {
                std::vector<float16_t> fillDist(curProbeSegNum * l3SegSize_ - offset, 99.99);
                ret = memcpy_s(indists + curTotalMulti + offset,
                               sizeof(float16_t) * (curProbeSegNum * l3SegSize_ - offset),
                               fillDist.data(), sizeof(float16_t) * (curProbeSegNum * l3SegSize_ - offset));
                if (ret != 0) {
                    KERNEL_LOG_ERROR("Copy data shaped error %d", ret);
                    return;
                }
            }

            curTotalMulti += curProbeSegNum * l3SegSize_;

            if (curTotalMulti >= l3SegNum_ * l3SegSize_) {
                break;
            }
        }
        if (curTotalMulti < l3SegNum_ * l3SegSize_) {
            std::vector<float16_t> fillDist(l3SegNum_ * l3SegSize_ - curTotalMulti, 99.99);
            ret = memcpy_s(indists + curTotalMulti, sizeof(float16_t) * (l3SegNum_ * l3SegSize_ - curTotalMulti), fillDist.data(),
                           sizeof(float16_t) * (l3SegNum_ * l3SegSize_ - curTotalMulti));
            if (ret != 0) {
                KERNEL_LOG_ERROR("Copy data shaped error %d", ret);
                return;
            }
        }

        int64_t baseOffset = 0;
        int64_t ntotal = l3SegNum_ * l3SegSize_;
        int64_t idxMulti = 0;
        int64_t burstIdx = 0;
        int64_t burstSize = ntotal / burstLen_;
        if (!quickTopk_) {
            for (int64_t i = burstIdx; i < burstSize; ++i) {
                if (!cmp(outdists[0], vmdists[i * 2])) { // vmdists[i*2] is dists, vmdists[i*2+1] is label
                    // skip one burst
                    idxMulti += burstLen_;
                    continue;
                }
                for (int64_t j = 0; j < burstLen_ && idxMulti < ntotal; ++j, ++idxMulti) {
                    if (cmp(outdists[0], indists[idxMulti])) {
                        outdists[0] = indists[idxMulti];
                        outlabel[0] = static_cast<T>(realIds[baseOffset + idxMulti]);
                        UpdateHeap<T, C>(outdists, outlabel, k_, 0, cmp);
                    }
                }
            }
        } else {
        }
        // process tail data
        while (idxMulti < ntotal) {
            if (cmp(outdists[0], indists[idxMulti])) {
                outdists[0] = indists[idxMulti];
                outlabel[0] = static_cast<T>(realIds[baseOffset + idxMulti]);
                UpdateHeap<T, C>(outdists, outlabel, k_, 0, cmp);
            }
            ++idxMulti;
        }
        // reorder dist result in last block
        if (reorder) {
            for (int64_t j = k_ - 1; j >= 1; --j) {
                std::swap(outdists[0], outdists[j]);
                std::swap(outlabel[0], outlabel[j]);
                UpdateHeap<T, C>(outdists, outlabel, j, 0, cmp);
            }
        }
    }

    REGISTER_CPU_KERNEL(IVF_MULTI_SP_TOPK_L3, IvfMultiSpTopkL3CpuKernel);
} // namespace aicpu