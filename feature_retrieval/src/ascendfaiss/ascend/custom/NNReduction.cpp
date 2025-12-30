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

#include "ascend/custom/NNReduction.h"
#include "ascend/AscendNNInference.h"
#include "common/utils/LogUtils.h"

namespace faiss {
namespace ascend {
class NNReductionImpl {
public:
    NNReductionImpl(const NNReductionImpl&) = delete;
    
    NNReductionImpl& operator=(const NNReductionImpl&) = delete;

    NNReductionImpl(std::vector<int> deviceList, const char *model, uint64_t modelSize)
    {
        APP_LOG_INFO("NNReductionImpl operation started.\n");
        this->nnInference = new AscendNNInference(deviceList, model, modelSize);
        APP_LOG_INFO("NNReductionImpl operation finished.\n");
    }

    ~NNReductionImpl()
    {
        delete this->nnInference;
        this->nnInference = nullptr;
    }

    void train(idx_t /* n */, const float * /* x */) const
    {
        APP_LOG_INFO("NNReductionImpl does not require training operation.\n");
    }

    void reduce(idx_t n, const float *x, float *res) const
    {
        APP_LOG_INFO("NNReductionImpl reduce operation started, with %ld vector(s).\n", n);
        this->nnInference->infer((size_t)n, (char *)x, (char *)res);
        APP_LOG_INFO("NNReNNReductionImplduction reduce operation finished.\n");
    }

protected:
    AscendNNInference *nnInference;
};

NNReduction::NNReduction(std::vector<int> deviceList, const char *model, uint64_t modelSize)
{
    APP_LOG_INFO("NNReduction operation started.\n");
    nnReductionImpl = std::make_shared<NNReductionImpl>(deviceList, model, modelSize);
    APP_LOG_INFO("NNReduction operation finished.\n");
}

NNReduction::~NNReduction() {}

void NNReduction::train(idx_t /* n */, const float * /* x */) const
{
    APP_LOG_INFO("NNReduction does not require training operation.\n");
}

void NNReduction::reduce(idx_t n, const float *x, float *res) const
{
    APP_LOG_INFO("NNReduction reduce operation started, with %ld vector(s).\n", n);
    nnReductionImpl->reduce(n, x, res);
    APP_LOG_INFO("NNReduction reduce operation finished.\n");
}
} // ascend
} // faiss
