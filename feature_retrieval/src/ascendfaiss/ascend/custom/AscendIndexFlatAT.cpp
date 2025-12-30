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


#include "AscendIndexFlatAT.h"

#include "ascend/custom/impl/AscendIndexFlatATImpl.h"

namespace faiss {
namespace ascend {
// implementation of AscendIndexFlatAT
AscendIndexFlatAT::AscendIndexFlatAT(int dims, int baseSize, AscendIndexFlatATConfig config)
    : AscendIndex(dims, MetricType::METRIC_L2, config),
    impl_(std::make_shared<AscendIndexFlatATImpl>(dims, baseSize, config, this))
{
    AscendIndex::impl_ = impl_;
}

AscendIndexFlatAT::~AscendIndexFlatAT() {}

void AscendIndexFlatAT::reset()
{
    impl_->reset();
}

void AscendIndexFlatAT::search(idx_t n, const float* x, idx_t k,
    float* distances, idx_t* labels, const SearchParameters *) const
{
    impl_->search(n, x, k, distances, labels);
}

void AscendIndexFlatAT::clearAscendTensor()
{
    impl_->clearAscendTensor();
}
} // ascend
} // faiss
