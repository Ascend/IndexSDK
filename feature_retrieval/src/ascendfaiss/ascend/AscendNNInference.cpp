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


#include "AscendNNInference.h"

#include "ascend/impl/AscendNNInferenceImpl.h"

namespace faiss {
namespace ascend {
AscendNNInference::AscendNNInference(std::vector<int> deviceList, const char *model, uint64_t modelSize)
    : impl_(std::make_shared<AscendNNInferenceImpl>(deviceList, model, modelSize))
{}

AscendNNInference::~AscendNNInference() {}

void AscendNNInference::infer(size_t n, const char *inputData, char *outputData) const
{
    impl_->infer(n, inputData, outputData);
}

int AscendNNInference::getInputType() const
{
    return impl_->getInputType();
}

int AscendNNInference::getOutputType() const
{
    return impl_->getOutputType();
}

int AscendNNInference::getDimIn() const
{
    return impl_->getDimIn();
}

int AscendNNInference::getDimOut() const
{
    return impl_->getDimOut();
}

int AscendNNInference::getDimBatch() const
{
    return impl_->getDimBatch();
}
} // ascend
} // faiss