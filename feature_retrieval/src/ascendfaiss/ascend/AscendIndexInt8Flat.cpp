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


#include "AscendIndexInt8Flat.h"
#include "ascend/impl/AscendIndexInt8FlatImpl.h"

namespace faiss {
namespace ascend {

AscendIndexInt8Flat::AscendIndexInt8Flat(int dims, faiss::MetricType metric, AscendIndexInt8FlatConfig config)
    : AscendIndexInt8(0, faiss::METRIC_L2, config),
    impl_(std::make_shared<AscendIndexInt8FlatImpl>(dims, metric, config, this))
{
    AscendIndexInt8::impl_ = impl_;
}

AscendIndexInt8Flat::AscendIndexInt8Flat(const faiss::IndexScalarQuantizer *index, AscendIndexInt8FlatConfig config)
    : AscendIndexInt8(0, faiss::METRIC_L2, config),
    impl_(std::make_shared<AscendIndexInt8FlatImpl>(index, config, this))
{
    AscendIndexInt8::impl_ = impl_;
}

AscendIndexInt8Flat::AscendIndexInt8Flat(const faiss::IndexIDMap *index, AscendIndexInt8FlatConfig config)
    : AscendIndexInt8(0, faiss::METRIC_L2, config),
    impl_(std::make_shared<AscendIndexInt8FlatImpl>(index, config, this))
{
    AscendIndexInt8::impl_ = impl_;
}


AscendIndexInt8Flat::~AscendIndexInt8Flat() {}

size_t AscendIndexInt8Flat::getBaseSize(int deviceId) const
{
    return impl_->getBaseSize(deviceId);
}

void AscendIndexInt8Flat::getBase(int deviceId, std::vector<int8_t> &xb) const
{
    impl_->getBase(deviceId, xb);
}

void AscendIndexInt8Flat::getIdxMap(int deviceId, std::vector<idx_t> &idxMap) const
{
    impl_->getIdxMap(deviceId, idxMap);
}

void AscendIndexInt8Flat::reset()
{
    impl_->reset();
}

void AscendIndexInt8Flat::copyFrom(const faiss::IndexScalarQuantizer *index)
{
    impl_->copyFrom(index);
}

void AscendIndexInt8Flat::copyFrom(const faiss::IndexIDMap *index)
{
    impl_->copyFrom(index);
}

void AscendIndexInt8Flat::copyTo(faiss::IndexScalarQuantizer *index) const
{
    impl_->copyTo(index);
}

void AscendIndexInt8Flat::copyTo(faiss::IndexIDMap *index) const
{
    impl_->copyTo(index);
}

void AscendIndexInt8Flat::search_with_masks(idx_t n, const int8_t *x, idx_t k,
    float *distances, idx_t *labels, const void *mask) const
{
#ifndef HOSTCPU
    size_t reqMsgSize = static_cast<size_t>(d) * static_cast<size_t>(n) +
        static_cast<size_t>(n) * static_cast<size_t>(::ascend::utils::divUp(this->ntotal, BINARY_BYTE_SIZE));
    FAISS_THROW_IF_NOT_FMT(reqMsgSize < MAX_SEARCH_WITH_MASK_REQ_MESSAGE_SIZE,
        "search_with_masks' request message size (dim * n + ⌈ntotal / 8⌉ * n = %zu) must be < %zu.",
        reqMsgSize, MAX_SEARCH_WITH_MASK_REQ_MESSAGE_SIZE);
#endif

    impl_->search_with_masks(n, x, k, distances, labels, mask);
}

void AscendIndexInt8Flat::setPageSize(uint16_t pageBlockNum)
{
    impl_->setPageSize(pageBlockNum);
}

} // ascend
} // faiss