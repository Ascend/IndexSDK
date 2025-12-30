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


#include "AscendIndexFlatATInt8.h"

#include "ascend/custom/impl/AscendIndexFlatATInt8Impl.h"

namespace faiss {
namespace ascend {
// implementation of AscendIndexFlatATInt8
AscendIndexFlatATInt8::AscendIndexFlatATInt8(int dims, int baseSize, AscendIndexFlatATInt8Config config)
    : AscendIndex(dims, MetricType::METRIC_L2, config),
    impl_(std::make_shared<AscendIndexFlatATInt8Impl>(dims, baseSize, config, this))
{
    AscendIndex::impl_ = impl_;
}

AscendIndexFlatATInt8::~AscendIndexFlatATInt8() {}

void AscendIndexFlatATInt8::reset()
{
    impl_->reset();
}

void AscendIndexFlatATInt8::sendMinMax(float qMin, float qMax)
{
    impl_->sendMinMax(qMin, qMax);
}

void AscendIndexFlatATInt8::searchInt8(idx_t n, const int8_t* x, idx_t k,
    float* distances, idx_t* labels) const
{
    impl_->searchInt8(n, x, k, distances, labels);
}

void AscendIndexFlatATInt8::clearAscendTensor()
{
    impl_->clearAscendTensor();
}
} // ascend
} // faiss
