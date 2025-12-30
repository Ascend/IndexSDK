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


#include "AscendIndexFlat.h"

#include <faiss/IndexFlat.h>

#include "ascend/impl/AscendIndexImpl.h"
#include "ascend/impl/AscendIndexFlatImpl.h"

namespace faiss {
namespace ascend {
// implementation of AscendIndexFlat
AscendIndexFlat::AscendIndexFlat(const faiss::IndexFlat *index, AscendIndexFlatConfig config)
    : AscendIndex(0, faiss::METRIC_L2, config),
    impl_(std::make_shared<AscendIndexFlatImpl>(index, config, this))
{
    AscendIndex::impl_ = impl_;
}

AscendIndexFlat::AscendIndexFlat(const faiss::IndexIDMap *index, AscendIndexFlatConfig config)
    : AscendIndex(0, faiss::METRIC_L2, config),
    impl_(std::make_shared<AscendIndexFlatImpl>(index, config, this))
{
    AscendIndex::impl_ = impl_;
}

AscendIndexFlat::AscendIndexFlat(int dims, faiss::MetricType metric, AscendIndexFlatConfig config)
    : AscendIndex(0, faiss::METRIC_L2, config),
    impl_(std::make_shared<AscendIndexFlatImpl>(dims, metric, config, this))
{
    AscendIndex::impl_ = impl_;
}

AscendIndexFlat::~AscendIndexFlat() {}


void AscendIndexFlat::copyFrom(const faiss::IndexFlat *index)
{
    impl_->copyFrom(index);
}

void AscendIndexFlat::copyFrom(const faiss::IndexIDMap *index)
{
    impl_->copyFrom(index);
}

void AscendIndexFlat::copyTo(faiss::IndexFlat *index) const
{
    impl_->copyTo(index);
}

void AscendIndexFlat::copyTo(faiss::IndexIDMap *index) const
{
    impl_->copyTo(index);
}

size_t AscendIndexFlat::getBaseSize(int deviceId) const
{
    return impl_->getBaseSize(deviceId);
}

void AscendIndexFlat::getIdxMap(int deviceId, std::vector<idx_t> &idxMap) const
{
    impl_->getIdxMap(deviceId, idxMap);
}

void AscendIndexFlat::getBase(int deviceId, char* xb) const
{
    impl_->getBase(deviceId, xb);
}

void AscendIndexFlat::search_with_masks(idx_t n, const float *x, idx_t k,
    float *distances, idx_t *labels, const void *mask) const
{
    impl_->search_with_masks(n, x, k, distances, labels, mask);
}

void AscendIndexFlat::search_with_masks(idx_t n, const uint16_t *x, idx_t k,
    float *distances, idx_t *labels, const void *mask) const
{
    impl_->search_with_masks_fp16(n, x, k, distances, labels, mask);
}

// implementation of AscendIndexFlatL2
AscendIndexFlatL2::AscendIndexFlatL2(faiss::IndexFlatL2 *index, AscendIndexFlatConfig config)
    : AscendIndexFlat(index, config)
{}

AscendIndexFlatL2::AscendIndexFlatL2(int dims, AscendIndexFlatConfig config)
    : AscendIndexFlat(dims, faiss::METRIC_L2, config)
{}

void AscendIndexFlatL2::copyFrom(faiss::IndexFlat *index)
{
    AscendIndexFlat::copyFrom(index);
}

void AscendIndexFlatL2::copyTo(faiss::IndexFlat *index)
{
    AscendIndexFlat::copyTo(index);
}
} // ascend
} // faiss