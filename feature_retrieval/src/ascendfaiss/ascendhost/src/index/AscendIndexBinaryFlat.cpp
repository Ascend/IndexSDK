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

#include "ascendhost/include/index/AscendIndexBinaryFlat.h"
#include "ascendhost/include/impl/AscendIndexBinaryFlatImpl.h"

namespace faiss {
namespace ascend {
void AscendIndexBinaryFlat::setRemoveFast(bool removeFast)
{
    static std::mutex tex;
    std::unique_lock<std::mutex> lock(tex);
    static bool repeatedUse = false;
    FAISS_THROW_IF_NOT_MSG(!repeatedUse, "setRemoveFast should not be used repeatedly!");
    if (!repeatedUse) {
        AscendIndexBinaryFlatImpl::setRemoveFast(removeFast);
        repeatedUse = true;
    }
}

AscendIndexBinaryFlat::AscendIndexBinaryFlat(const faiss::IndexBinaryFlat *index, AscendIndexBinaryFlatConfig config,
    bool usedFloat)
    : IndexBinary(index == nullptr ? 0 : index->d), impl_(new AscendIndexBinaryFlatImpl(index, config, usedFloat))
{
    impl_->Initialize();
}

AscendIndexBinaryFlat::AscendIndexBinaryFlat(const faiss::IndexBinaryIDMap *index, AscendIndexBinaryFlatConfig config,
    bool usedFloat) : IndexBinary(index == nullptr || index->index == nullptr ? 0 : index->index->d),
    impl_(new AscendIndexBinaryFlatImpl(index, config, usedFloat))
{
    impl_->Initialize();
}

AscendIndexBinaryFlat::AscendIndexBinaryFlat(int dims, AscendIndexBinaryFlatConfig config, bool usedFloat)
    : IndexBinary(dims), impl_(new AscendIndexBinaryFlatImpl(dims, config, usedFloat))
{
    impl_->Initialize();
}

void AscendIndexBinaryFlat::add(idx_t n, const uint8_t *x)
{
    impl_->add_with_ids(n, x, nullptr);
    this->ntotal += n;
}

void AscendIndexBinaryFlat::add_with_ids(idx_t n, const uint8_t *x, const idx_t *xids)
{
    impl_->add_with_ids(n, x, xids);
    this->ntotal += n;
}

size_t AscendIndexBinaryFlat::remove_ids(const faiss::IDSelector &sel)
{
    idx_t res = static_cast<idx_t>(impl_->remove_ids(sel));
    this->ntotal -= res;
    return res;
}

void AscendIndexBinaryFlat::search(idx_t n, const uint8_t *x, idx_t k, int32_t *distances, idx_t *labels,
    const SearchParameters *params) const
{
    FAISS_THROW_IF_NOT_MSG(!params, "search params not supported for ascend index");
    impl_->search(n, x, k, distances, labels);
}

void AscendIndexBinaryFlat::search(idx_t n, const float *x, idx_t k, float *distances, idx_t *labels) const
{
    impl_->search(n, x, k, distances, labels);
}

void AscendIndexBinaryFlat::reset()
{
    impl_->reset();
    this->ntotal = 0;
}

void AscendIndexBinaryFlat::copyFrom(const faiss::IndexBinaryFlat *index)
{
    impl_->copyFrom(index);
    this->d = index->d;
    this->code_size = index->code_size;
    this->ntotal = index->ntotal;
}

void AscendIndexBinaryFlat::copyFrom(const faiss::IndexBinaryIDMap *index)
{
    impl_->copyFrom(index);
    auto binaryPtr = dynamic_cast<const faiss::IndexBinaryFlat *>(index->index);
    this->d = binaryPtr->d;
    this->code_size = binaryPtr->code_size;
    this->ntotal = binaryPtr->ntotal;
}

void AscendIndexBinaryFlat::copyTo(faiss::IndexBinaryFlat *index) const
{
    impl_->copyTo(index);
}

void AscendIndexBinaryFlat::copyTo(faiss::IndexBinaryIDMap *index) const
{
    impl_->copyTo(index);
}
} /* namespace ascend */
} /* namespace faiss */