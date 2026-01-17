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

#include "AscendIndexIVFPQ.h"

#include "ascend/impl/AscendIndexIVFPQImpl.h"

namespace faiss {
namespace ascend {

// implementation of AscendIndexIVF
AscendIndexIVFPQ::AscendIndexIVFPQ(int dims, faiss::MetricType metric, int nlist, int msubs, int nbits,
                                   AscendIndexIVFPQConfig config)
    : AscendIndexIVF(dims, metric, nlist, config),
      impl_(std::make_shared<AscendIndexIVFPQImpl>(this, dims, nlist, msubs, nbits, metric, config))
{
    AscendIndexIVF::impl_ = impl_;
    AscendIndex::impl_ = impl_;
}

AscendIndexIVFPQ::~AscendIndexIVFPQ()
{
}

void AscendIndexIVFPQ::copyFrom(const faiss::IndexIVFPQ* index)
{
    FAISS_THROW_IF_NOT_MSG(impl_ != nullptr, "impl_ is nullptr!");
    impl_->copyFrom(index);
}

void AscendIndexIVFPQ::copyTo(faiss::IndexIVFPQ* index) const
{
    FAISS_THROW_IF_NOT_MSG(impl_ != nullptr, "impl_ is nullptr!");
    impl_->copyTo(index);
}

void AscendIndexIVFPQ::train(idx_t n, const float* x)
{
    FAISS_THROW_IF_NOT_MSG(impl_ != nullptr, "impl_ is nullptr!");
    impl_->train(n, x);
}

void AscendIndexIVFPQ::remove_ids(size_t n, const idx_t* ids)
{
    FAISS_THROW_IF_NOT_MSG(impl_ != nullptr, "impl_ is nullptr!");
    impl_->deleteImpl(static_cast<int>(n), ids);
}
}  // namespace ascend
}  // namespace faiss
