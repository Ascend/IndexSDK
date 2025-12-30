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

#include "ascend/custom/PcarReduction.h"

#include <faiss/VectorTransform.h>

#include "common/utils/LogUtils.h"

namespace faiss {
namespace ascend {
const float NO_WHITENING = 0;
const float FULL_WHITENING = -0.5;
class PcarReductionImpl {
public:
    PcarReductionImpl(int dimIn, int dimOut, float eigenPower, bool randomRotation)
    {
        APP_LOG_INFO("PcarReductionImpl operation started, dimIn=%d, dimOut=%d.\n", dimIn, dimOut);
        FAISS_THROW_IF_NOT_MSG(dimOut > 0,
            "The output dim of matrix for PCA should be > 0.\n");
        FAISS_THROW_IF_NOT_MSG(dimIn >= dimOut,
            "The input dim of matrix for PCA should be greater than or equal to output dim.\n");
        FAISS_THROW_IF_NOT_MSG(eigenPower >= FULL_WHITENING && eigenPower <= NO_WHITENING,
            "The value of eigenPower should be >= -0.5 and <= 0.\n");
        
        this->cpuLtrans = std::make_shared<faiss::PCAMatrix>(dimIn, dimOut, eigenPower, randomRotation);

        APP_LOG_INFO("PcarReductionImpl operation finished.\n");
    }

    ~PcarReductionImpl() {}

    void train(idx_t n, const float *x) const
    {
        APP_LOG_INFO("PcarReductionImpl train operation started, with %ld vector(s).\n", n);
        FAISS_THROW_IF_NOT_MSG(x, "x can not be nullptr.");
        FAISS_THROW_IF_NOT_FMT((n > 0) && (n < MAX_N), "n must be > 0 and < %ld", MAX_N);
        if (this->cpuLtrans == nullptr) {
            return;
        }
        this->cpuLtrans->train(n, x);
        APP_LOG_INFO("PcarReductionImpl train operation finished.\n");
    }

    void reduce(idx_t n, const float *x, float *res) const
    {
        APP_LOG_INFO("PcarReductionImpl reduce operation started, with %ld vector(s).\n", n);
        FAISS_THROW_IF_NOT_FMT((n > 0) && (n < MAX_N), "n must be > 0 and < %ld", MAX_N);
        FAISS_THROW_IF_NOT_MSG(x, "x can not be nullptr.");
        FAISS_THROW_IF_NOT_MSG(res, "res can not be nullptr.");
        this->cpuLtrans->apply_noalloc(n, x, res);
        APP_LOG_INFO("PcarReductionImpl reduce operation finished.\n");
    }

protected:
    std::shared_ptr<faiss::LinearTransform> cpuLtrans;
};

PcarReduction::PcarReduction(int dimIn, int dimOut, float eigenPower, bool randomRotation)
{
    APP_LOG_INFO("PcarReduction operation started, dimIn=%d, dimOut=%d, eigenPower=%.4f, randomRotation=%d.\n", dimIn,
        dimOut, eigenPower, randomRotation);
    pcarReductionImpl = std::make_shared<PcarReductionImpl>(dimIn, dimOut, eigenPower, randomRotation);
    APP_LOG_INFO("PcarReduction operation finished.\n");
}

PcarReduction::~PcarReduction() {}

void PcarReduction::train(idx_t n, const float *x) const
{
    APP_LOG_INFO("PcarReduction train operation started, with %ld vector(s).\n", n);
    pcarReductionImpl->train(n, x);
    APP_LOG_INFO("PcarReduction train operation finished.\n");
}

void PcarReduction::reduce(idx_t n, const float *x, float *res) const
{
    APP_LOG_INFO("PcarReduction reduce operation started, with %ld vector(s).\n", n);
    pcarReductionImpl->reduce(n, x, res);
    APP_LOG_INFO("PcarReduction reduce operation finished.\n");
}
} // ascend
} // faiss
