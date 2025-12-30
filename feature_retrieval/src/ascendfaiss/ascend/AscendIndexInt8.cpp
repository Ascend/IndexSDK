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

#include <faiss/impl/FaissAssert.h>

#include "ascend/impl/AscendIndexInt8Impl.h"
#include "AscendIndexInt8.h"


namespace faiss {
namespace ascend {
AscendIndexInt8::AscendIndexInt8(int dims, faiss::MetricType metric, AscendIndexInt8Config config)
{
    // params will be set in AscendIndexInt8Impl
    VALUE_UNUSED(dims);
    VALUE_UNUSED(metric);
    VALUE_UNUSED(config);
}

AscendIndexInt8::~AscendIndexInt8() {}

void AscendIndexInt8::train(idx_t n, const int8_t *x)
{
    FAISS_THROW_IF_NOT_MSG(impl_ != nullptr, "impl_ is nullptr!");

    impl_->train(n, x);
}

void AscendIndexInt8::updateCentroids(idx_t n, const int8_t *x)
{
    FAISS_THROW_IF_NOT_MSG(impl_ != nullptr, "impl_ is nullptr!");

    impl_->updateCentroids(n, x);
}

void AscendIndexInt8::updateCentroids(idx_t n, const char *x)
{
    FAISS_THROW_IF_NOT_MSG(impl_ != nullptr, "impl_ is nullptr!");

    impl_->updateCentroids(n, x);
}

void AscendIndexInt8::add(idx_t n, const char *x)
{
    FAISS_THROW_IF_NOT_MSG(impl_ != nullptr, "impl_ is nullptr!");

    impl_->add(n, x);
}

void AscendIndexInt8::add_with_ids(idx_t n, const char *x, const idx_t *ids)
{
    FAISS_THROW_IF_NOT_MSG(impl_ != nullptr, "impl_ is nullptr!");

    impl_->add_with_ids(n, x, ids);
}

void AscendIndexInt8::add(idx_t n, const int8_t *x)
{
    FAISS_THROW_IF_NOT_MSG(impl_ != nullptr, "impl_ is nullptr!");

    impl_->add(n, x);
}

void AscendIndexInt8::add_with_ids(idx_t n, const int8_t *x, const idx_t *ids)
{
    FAISS_THROW_IF_NOT_MSG(impl_ != nullptr, "impl_ is nullptr!");

    impl_->add_with_ids(n, x, ids);
}

size_t AscendIndexInt8::remove_ids(const faiss::IDSelector &sel)
{
    FAISS_THROW_IF_NOT_MSG(impl_ != nullptr, "impl_ is nullptr!");

    return impl_->remove_ids(sel);
}

void AscendIndexInt8::assign(idx_t n, const int8_t *x, idx_t *labels, idx_t k)
{
    FAISS_THROW_IF_NOT_MSG(impl_ != nullptr, "impl_ is nullptr!");

    impl_->assign(n, x, labels, k);
}

void AscendIndexInt8::search(idx_t n, const int8_t *x, idx_t k, float *distances,
    idx_t *labels) const
{
    FAISS_THROW_IF_NOT_MSG(impl_ != nullptr, "impl_ is nullptr!");

    impl_->search(n, x, k, distances, labels);
}

void AscendIndexInt8::search(idx_t n, const char *x, idx_t k, float *distances,
    idx_t *labels) const
{
    FAISS_THROW_IF_NOT_MSG(impl_ != nullptr, "impl_ is nullptr!");

    impl_->search(n, x, k, distances, labels);
}

void AscendIndexInt8::reserveMemory(size_t numVecs)
{
    FAISS_THROW_IF_NOT_MSG(impl_ != nullptr, "impl_ is nullptr!");

    impl_->reserveMemory(numVecs);
}

size_t AscendIndexInt8::reclaimMemory()
{
    FAISS_THROW_IF_NOT_MSG(impl_ != nullptr, "impl_ is nullptr!");

    return impl_->reclaimMemory();
}

std::vector<int> AscendIndexInt8::getDeviceList() const
{
    FAISS_THROW_IF_NOT_MSG(impl_ != nullptr, "impl_ is nullptr!");

    return impl_->getDeviceList();
}

IndexImplBase& AscendIndexInt8::GetIndexImplBase() const
{
    IndexImplBase *base = dynamic_cast<IndexImplBase *>(this->impl_.get());
    if (base == nullptr) {
        FAISS_THROW_MSG("cast impl_ to IndexImplBase failed");
    }
    return *base;
}

int AscendIndexInt8::getDim() const
{
    FAISS_THROW_IF_NOT_MSG(impl_ != nullptr, "impl_ is nullptr!");

    return impl_->getDim();
}

faiss::idx_t AscendIndexInt8::getNTotal() const
{
    FAISS_THROW_IF_NOT_MSG(impl_ != nullptr, "impl_ is nullptr!");

    return impl_->getNTotal();
}

bool AscendIndexInt8::isTrained() const
{
    FAISS_THROW_IF_NOT_MSG(impl_ != nullptr, "impl_ is nullptr!");

    return impl_->isTrained();
}

faiss::MetricType AscendIndexInt8::getMetricType() const
{
    FAISS_THROW_IF_NOT_MSG(impl_ != nullptr, "impl_ is nullptr!");

    return impl_->getMetricType();
}
} // namespace ascend
} // namespace faiss
