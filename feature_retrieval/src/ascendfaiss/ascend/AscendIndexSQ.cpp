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

#include "AscendIndexSQ.h"
#include "ascend/impl/AscendIndexSQImpl.h"

namespace faiss {
namespace ascend {
AscendIndexSQ::AscendIndexSQ(const faiss::IndexScalarQuantizer *index, AscendIndexSQConfig config)
    : AscendIndex(0, faiss::METRIC_L2, config),
      impl_(std::make_shared<AscendIndexSQImpl>(this, index, config))
{
    AscendIndex::impl_ = impl_;
}

AscendIndexSQ::AscendIndexSQ(const faiss::IndexIDMap *index, AscendIndexSQConfig config)
    : AscendIndex(0, faiss::METRIC_L2, config),
      impl_(std::make_shared<AscendIndexSQImpl>(this, index, config))
      
{
    AscendIndex::impl_ = impl_;
}

AscendIndexSQ::AscendIndexSQ(int dims, faiss::ScalarQuantizer::QuantizerType qType, faiss::MetricType metric,
    AscendIndexSQConfig config)
    : AscendIndex(0, faiss::METRIC_L2, config),
      impl_(std::make_shared<AscendIndexSQImpl>(this, dims, qType, metric, config))
{
    AscendIndex::impl_ = impl_;
}

AscendIndexSQ::~AscendIndexSQ() {}

void AscendIndexSQ::copyFrom(const faiss::IndexScalarQuantizer *index)
{
    impl_->copyFrom(index);
}

void AscendIndexSQ::copyFrom(const faiss::IndexIDMap *index)
{
    impl_->copyFrom(index);
}

void AscendIndexSQ::copyTo(faiss::IndexScalarQuantizer *index) const
{
    impl_->copyTo(index);
}

void AscendIndexSQ::copyTo(faiss::IndexIDMap *index) const
{
    impl_->copyTo(index);
}

void AscendIndexSQ::getBase(int deviceId, char* xb) const
{
    impl_->getBase(deviceId, xb);
}

size_t AscendIndexSQ::getBaseSize(int deviceId) const
{
    return impl_->getBaseSize(deviceId);
}

void AscendIndexSQ::getIdxMap(int deviceId, std::vector<idx_t> &idxMap) const
{
    impl_->getIdxMap(deviceId, idxMap);
}

void AscendIndexSQ::train(idx_t n, const float *x)
{
    impl_->train(n, x);
}

void AscendIndexSQ::search_with_masks(idx_t n, const float *x, idx_t k,
    float *distances, idx_t *labels, const void *mask) const
{
    impl_->search_with_masks(n, x, k, distances, labels, mask);
}

void AscendIndexSQ::search_with_filter(idx_t n, const float *x, idx_t k,
    float *distances, idx_t *labels, const void *filters) const
{
    impl_->search_with_filter(n, x, k, distances, labels, filters);
}
} // ascend
} // faiss