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

#include "AscendIndexIVFRaBitQ.h"

#include "ascend/impl/AscendIndexIVFRaBitQImpl.h"

namespace faiss {
namespace ascend {
// implementation of AscendIndexIVF
AscendIndexIVFRaBitQ::AscendIndexIVFRaBitQ(int dims, faiss::MetricType metric, int nlist,
    AscendIndexIVFRaBitQConfig config) : AscendIndexIVF(dims, metric, nlist, config),
    impl_(std::make_shared<AscendIndexIVFRaBitQImpl>(this, dims, nlist, metric, config))
{
    AscendIndexIVF::impl_ = impl_;
    AscendIndex::impl_ = impl_;
}

AscendIndexIVFRaBitQ::~AscendIndexIVFRaBitQ() {}

void AscendIndexIVFRaBitQ::train(idx_t n, const float *x)
{
    FAISS_THROW_IF_NOT_MSG(impl_ != nullptr, "impl_ is nullptr!");
    impl_->train(n, x);
}

// Copy what we need from a CPU IndexIVFRaBitQ
void AscendIndexIVFRaBitQ::copyFrom(const faiss::IndexIVFRaBitQ *index)
{
    APP_LOG_INFO("AscendIndexIVFRaBitQ copyFrom (IndexIVFRaBitQ) operation started.\n");
    FAISS_THROW_IF_NOT_MSG(impl_ != nullptr, "impl_ is nullptr!");
    
    // Call implementation class copyFrom to handle IVFRaBitQ data
    impl_->copyFrom(index);
    
    APP_LOG_INFO("AscendIndexIVFRaBitQ copyFrom (IndexIVFRaBitQ) operation finished.\n");
}

// Copy what we have to a CPU IndexIVFRaBitQ
void AscendIndexIVFRaBitQ::copyTo(faiss::IndexIVFRaBitQ *index) const
{
    APP_LOG_INFO("AscendIndexIVFRaBitQ copyTo (IndexIVFRaBitQ) operation started.\n");
    FAISS_THROW_IF_NOT_MSG(impl_ != nullptr, "impl_ is nullptr!");
    
    // Call implementation class copyTo to handle IVFRaBitQ data
    impl_->copyTo(index);
    
    APP_LOG_INFO("AscendIndexIVFRaBitQ copyTo (IndexIVFRaBitQ) operation finished.\n");
}

void AscendIndexIVFRaBitQ::remove_ids(size_t n, const idx_t* ids)
{
    FAISS_THROW_IF_NOT_MSG(impl_ != nullptr, "impl_ is nullptr!");
    impl_->deleteImpl(static_cast<int>(n), ids);
}

std::vector<idx_t> AscendIndexIVFRaBitQ::update(idx_t n, const float* x, const idx_t* ids)
{
    FAISS_THROW_IF_NOT_MSG(impl_ != nullptr, "impl_ is nullptr!");
    return impl_->update(n, x, ids);
}
} // namespace ascend
} // namespace faiss
