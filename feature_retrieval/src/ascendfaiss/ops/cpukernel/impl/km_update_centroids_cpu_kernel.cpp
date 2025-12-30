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


#include "km_update_centroids_cpu_kernel.h"

#include <algorithm>
#include <securec.h>
#include <string>

#include "cpu_kernel.h"
#include "cpu_kernel_utils.h"
#include "kernel_tensor.h"
#include "kernel_utils.h"
#include "Random.h"

namespace {
const char* KM_UPDATE_CENTROIDS = "KmUpdateCentroids";
const int64_t RAND_SEED = 1234;
const float16_t EPS = 1 / 1024.0;
}

namespace aicpu {
uint32_t KmUpdateCentroidsCpuKernel::Compute(CpuKernelContext &ctx)
{
    Inputs inputs;
    Outputs outputs;
    auto ret = GetInOutAndCheck(ctx, inputs, outputs);
    if (ret != KERNEL_STATUS_OK) {
        KERNEL_LOG_ERROR("Failed to get inputs or outputs");
        return ret;
    }

    ret = CheckInOutShapes(inputs, outputs);
    if (ret != KERNEL_STATUS_OK) {
        KERNEL_LOG_ERROR("Failed to check input shapes");
        return ret;
    }

    DoCompute(ctx, inputs, outputs);

    return KERNEL_STATUS_OK;
}

uint32_t KmUpdateCentroidsCpuKernel::GetInOutAndCheck(const CpuKernelContext &ctx,
    Inputs &inputs, Outputs &outputs) const
{
    KERNEL_LOG_INFO("KmUpdateCentroidsCpuKernel GetInOutAndCheck begin");

    inputs.codes = ctx.Input(INPUT_NUM0);
    inputs.assign = ctx.Input(INPUT_NUM1);
    outputs.centroids = ctx.Output(INPUT_NUM0);

    KERNEL_CHECK_NULLPTR(inputs.codes, KERNEL_STATUS_PARAM_INVALID, "Get input[0], name[codes] failed");
    KERNEL_CHECK_NULLPTR(inputs.assign, KERNEL_STATUS_PARAM_INVALID, "Get input[1], name[assign] failed");
    KERNEL_CHECK_NULLPTR(outputs.centroids, KERNEL_STATUS_PARAM_INVALID, "Get output[0], name[centroids] failed");

    KERNEL_LOG_INFO("Shape of input[0][codes] is %s",
        ShapeToString(inputs.codes->GetTensorShape()->GetDimSizes()).c_str());
    KERNEL_LOG_INFO("Shape of input[1][assign] is %s",
        ShapeToString(inputs.assign->GetTensorShape()->GetDimSizes()).c_str());
    KERNEL_LOG_INFO("Shape of output[0][centroids] is %s",
        ShapeToString(outputs.centroids->GetTensorShape()->GetDimSizes()).c_str());

    return KERNEL_STATUS_OK;
}

uint32_t KmUpdateCentroidsCpuKernel::CheckInOutShapes(const Inputs &inputs, const Outputs &outputs)
{
    KERNEL_LOG_INFO("KmUpdateCentroidsCpuKernel CheckInputShapes begin");

    auto shapeCodes = inputs.codes->GetTensorShape();
    auto shapeAssign = inputs.assign->GetTensorShape();
    auto shapeCentroids = outputs.centroids->GetTensorShape();

    KERNEL_CHECK_TRUE(shapeCodes->GetDims() == INPUT_NUM2,
        KERNEL_STATUS_PARAM_INVALID, "Dims of input[0][codes] must be 2");
    KERNEL_CHECK_TRUE(shapeAssign->GetDims() == INPUT_NUM1,
        KERNEL_STATUS_PARAM_INVALID, "Dims of input[1][assign] must be 1");
    KERNEL_CHECK_TRUE(shapeCentroids->GetDims() == INPUT_NUM2,
        KERNEL_STATUS_PARAM_INVALID, "Dims of output[0][centroids] must be 2");

    nq_ = shapeCodes->GetDimSize(INPUT_NUM0);
    dim_ = shapeCodes->GetDimSize(INPUT_NUM1);
    k_ = shapeCentroids->GetDimSize(INPUT_NUM0);

    return KERNEL_STATUS_OK;
}

void KmUpdateCentroidsCpuKernel::DoAccumulate(size_t start, size_t end, size_t kFrozen, std::vector<size_t> &hassign,
    float16_t *centroids, KernelTensor<float16_t> &codes, KernelTensor<uint64_t> &assign)
{
    const float16_t *xi = codes.GetSubTensorDim0(0);

    // Centers accumulate each vector in the clusters and counts the numbers of vectors
    for (size_t i = 0; i < static_cast<size_t>(nq_); i++) {
        uint64_t ci = *(assign.GetSubTensorDim0(i));
        if (ci >= uint64_t(k_) + uint64_t(kFrozen)) {
            KERNEL_LOG_ERROR("internal error \n");
        }
        ci -= kFrozen;
        if (ci >= start && ci < end) {
            float16_t *c = centroids + ci * dim_;
            hassign[ci]++;
            for (size_t j = 0; j < size_t(dim_); j++) {
                c[j] += xi[j];
            }
        }
        xi += dim_;
    }
}

void KmUpdateCentroidsCpuKernel::GetCentroids(size_t start, size_t end, float16_t *centroids,
    std::vector<size_t> &hassign)
{
    // Traverse each cluster center and accumulate vectors in it
    for (size_t ci = start; ci < end; ci++) {
        float16_t *c = centroids + ci * dim_;
        size_t ni = hassign[ci];
        if (ni != 0) {
            for (size_t j = 0; j < size_t(dim_); j++) {
                c[j] /= static_cast<float16_t>(ni);
            }
        }
    }
}

void KmUpdateCentroidsCpuKernel::RepalceCentroids(std::vector<size_t> &hassign, float16_t *centroids)
{
    // Take care of void clusters
    size_t nsplit = 0;
    ascend::RandomGenerator rng(RAND_SEED);
    for (size_t ci = 0; ci < size_t(k_); ci++) {
        // Deviation points, which need to redefine a centroid
        if (hassign[ci] == 0) {
            size_t cj = 0;
            for (cj = 0; true; cj = (cj + 1) % k_) {
                // probability to pick this cluster for split
                float p = (hassign[cj] - 1.0) / (float)(nq_ - k_);
                float r = rng.RandFloat();
                if (r < p) {
                    break;
                }
            }
            // Replace ci with cj
            auto err = memcpy_s(centroids + ci * dim_, sizeof(*centroids) * dim_,
                centroids + cj * dim_, sizeof(*centroids) * dim_);
            if (err != EOK) {
                KERNEL_LOG_ERROR("memset_s failed. %d", err);
            }

            // Add disturbance to distinguish between cj and ci and improve the effect
            for (size_t j = 0; j < size_t(dim_); j++) {
                // divide j by 2, judge even by the remainder
                if (j % 2 == 0) {
                    centroids[ci * dim_ + j] *= 1 + EPS;
                    centroids[cj * dim_ + j] *= 1 - EPS;
                } else {
                    centroids[ci * dim_ + j] *= 1 - EPS;
                    centroids[cj * dim_ + j] *= 1 + EPS;
                }
            }
            // divide by 2, assume even split of the cluster
            hassign[ci] = hassign[cj] / 2;
            hassign[cj] -= hassign[ci];
            nsplit++;
        }
    }
}

void KmUpdateCentroidsCpuKernel::DoCompute(const CpuKernelContext &ctx, Inputs &inputs, Outputs &outputs)
{
    // Currently, only fp16 is supported
    KernelTensor<float16_t> codes(inputs.codes);
    KernelTensor<uint64_t> assign(inputs.assign);

    KernelTensor<float16_t> outvecsTensor(outputs.centroids);

    // If service extension is required, kFrozen can be used as an operator parameter
    size_t kFrozen = 0;
    k_ -= kFrozen;

    float16_t *centroids = outvecsTensor.GetSubTensorDim0(kFrozen);

    std::vector<size_t> hassign(k_);

    auto err = memset_s(centroids, sizeof(float16_t) * dim_ * k_, 0, sizeof(float16_t) * dim_ * k_);
    if (err != EOK) {
        KERNEL_LOG_ERROR("memset_s failed. %d", err);
    }

    auto accumulateFunc = [&](size_t start, size_t end) {
        DoAccumulate(start, end, kFrozen, hassign, centroids, codes, assign);
    };
    auto getCentroidsFunc = [&](size_t start, size_t end) {
        GetCentroids(start, end, centroids, hassign);
    };
#ifdef AICPU_UTEST
    accumulateFunc(0, k_);
    getCentroidsFunc(0, k_);
#else
    uint32_t core = std::min({CpuKernelUtils::GetCPUNum(ctx), static_cast<uint32_t>(k_)});
    CpuKernelUtils::ParallelFor(ctx, k_, k_ / core, accumulateFunc);
    CpuKernelUtils::ParallelFor(ctx, k_, k_ / core, getCentroidsFunc);
#endif

    RepalceCentroids(hassign, centroids);
}

REGISTER_CPU_KERNEL(KM_UPDATE_CENTROIDS, KmUpdateCentroidsCpuKernel);
} // namespace aicpu
