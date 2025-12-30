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


#include "AscendIndexIVF.h"

#include "ascend/impl/AscendIndexIVFImpl.h"

namespace faiss {
namespace ascend {
// implementation of AscendIndexIVF
AscendIndexIVF::AscendIndexIVF(int dims, faiss::MetricType metric, int nlist, AscendIndexIVFConfig config)
    : AscendIndex(dims, metric, config)
{
    VALUE_UNUSED(nlist);
}

AscendIndexIVF::~AscendIndexIVF() {}

// Copy what we need from the CPU equivalent
void AscendIndexIVF::copyFrom(const faiss::IndexIVF *index)
{
    FAISS_THROW_IF_NOT_MSG(impl_ != nullptr, "impl_ is nullptr!");

    impl_->copyFrom(index);
}

// / Copy what we have to the CPU equivalent
void AscendIndexIVF::copyTo(faiss::IndexIVF *index) const
{
    FAISS_THROW_IF_NOT_MSG(impl_ != nullptr, "impl_ is nullptr!");

    impl_->copyTo(index);
}

void AscendIndexIVF::reset()
{
    FAISS_THROW_IF_NOT_MSG(impl_ != nullptr, "impl_ is nullptr!");

    impl_->reset();
}

// Sets the number of list probes per query
void AscendIndexIVF::setNumProbes(int nprobes)
{
    FAISS_THROW_IF_NOT_MSG(impl_ != nullptr, "impl_ is nullptr!");

    impl_->setNumProbes(nprobes);
}

uint32_t AscendIndexIVF::getListLength(int listId) const
{
    FAISS_THROW_IF_NOT_MSG(impl_ != nullptr, "impl_ is nullptr!");

    return impl_->getListLength(listId);
}

void AscendIndexIVF::getListCodesAndIds(int listId, std::vector<uint8_t> &codes, std::vector<ascend_idx_t> &ids) const
{
    FAISS_THROW_IF_NOT_MSG(impl_ != nullptr, "impl_ is nullptr!");

    impl_->getListCodesAndIds(listId, codes, ids);
}

void AscendIndexIVF::reserveMemory(size_t numVecs)
{
    FAISS_THROW_IF_NOT_MSG(impl_ != nullptr, "impl_ is nullptr!");

    impl_->reserveMemory(numVecs);
}

size_t AscendIndexIVF::reclaimMemory()
{
    FAISS_THROW_IF_NOT_MSG(impl_ != nullptr, "impl_ is nullptr!");
    return impl_->reclaimMemory();
}

int AscendIndexIVF::getNumLists() const
{
    FAISS_THROW_IF_NOT_MSG(impl_ != nullptr, "impl_ is nullptr!");

    return impl_->getNumLists();
}

int AscendIndexIVF::getNumProbes() const
{
    FAISS_THROW_IF_NOT_MSG(impl_ != nullptr, "impl_ is nullptr!");

    return impl_->getNumProbes();
}

} // namespace ascend
} // namespace faiss
