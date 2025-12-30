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
#include "ascend/impl/AscendIndexImpl.h"
#include "AscendIndex.h"


namespace faiss {
namespace ascend {
AscendIndex::AscendIndex(int dims, faiss::MetricType metric, AscendIndexConfig config)
{
    // params will be set in AscendIndexImpl
    VALUE_UNUSED(dims);
    VALUE_UNUSED(metric);
    VALUE_UNUSED(config);
}

AscendIndex::~AscendIndex() {}

void AscendIndex::add(idx_t n, const float* x)
{
    FAISS_THROW_IF_NOT_MSG(impl_ != nullptr, "impl_ is nullptr!");

    impl_->add(n, x);
}

void AscendIndex::add_with_ids(idx_t n, const float* x,
                               const idx_t* ids)
{
    FAISS_THROW_IF_NOT_MSG(impl_ != nullptr, "impl_ is nullptr!");

    impl_->add_with_ids(n, x, ids);
}

size_t AscendIndex::remove_ids(const faiss::IDSelector& sel)
{
    FAISS_THROW_IF_NOT_MSG(impl_ != nullptr, "impl_ is nullptr!");

    return impl_->remove_ids(sel);
}

void AscendIndex::search(idx_t n, const float* x, idx_t k,
                         float* distances, idx_t* labels, const SearchParameters *) const
{
    FAISS_THROW_IF_NOT_MSG(impl_ != nullptr, "impl_ is nullptr!");

    impl_->search(n, x, k, distances, labels);
}

void AscendIndex::reserveMemory(size_t numVecs)
{
    FAISS_THROW_IF_NOT_MSG(impl_ != nullptr, "impl_ is nullptr!");

    impl_->reserveMemory(numVecs);
}

size_t AscendIndex::reclaimMemory()
{
    FAISS_THROW_IF_NOT_MSG(impl_ != nullptr, "impl_ is nullptr!");
    return impl_->reclaimMemory();
}

void AscendIndex::reset()
{
    FAISS_THROW_IF_NOT_MSG(impl_ != nullptr, "impl_ is nullptr!");

    impl_->reset();
}

std::vector<int> AscendIndex::getDeviceList()
{
    FAISS_THROW_IF_NOT_MSG(impl_ != nullptr, "impl_ is nullptr!");

    return impl_->getDeviceList();
}

IndexImplBase& AscendIndex::GetIndexImplBase() const
{
    IndexImplBase *base = dynamic_cast<IndexImplBase *>(this->impl_.get());
    FAISS_THROW_IF_NOT_MSG(base != nullptr, "cast impl_ to IndexImplBase failed");
    return *base;
}

void AscendIndex::add(idx_t n, const uint16_t *x)
{
    FAISS_THROW_IF_NOT_MSG(impl_ != nullptr, "impl_ is nullptr!");

    impl_->add(n, x);
}

void AscendIndex::add_with_ids(idx_t n, const uint16_t *x, const idx_t *ids)
{
    FAISS_THROW_IF_NOT_MSG(impl_ != nullptr, "impl_ is nullptr!");

    impl_->add_with_ids(n, x, ids);
}

void AscendIndex::search(idx_t n, const uint16_t *x, idx_t k, float *distances, idx_t *labels) const
{
    FAISS_THROW_IF_NOT_MSG(impl_ != nullptr, "impl_ is nullptr!");

    impl_->search(n, x, k, distances, labels);
}

}  // namespace ascend
}  // namespace faiss
