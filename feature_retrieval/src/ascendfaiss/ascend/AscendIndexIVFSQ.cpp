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


#include "AscendIndexIVFSQ.h"

#include "ascend/impl/AscendIndexIVFSQImpl.h"

namespace faiss {
namespace ascend {
AscendIndexIVFSQ::AscendIndexIVFSQ(const faiss::IndexIVFScalarQuantizer *index, AscendIndexIVFSQConfig config)
    : AscendIndexIVF(0, faiss::METRIC_L2, 0, config),
    impl_(std::make_shared<AscendIndexIVFSQImpl>(this, index, config))
{
    AscendIndexIVF::impl_ = impl_;
    AscendIndex::impl_ = impl_;
}

AscendIndexIVFSQ::AscendIndexIVFSQ(int dims, int nlist,
    faiss::ScalarQuantizer::QuantizerType qtype, faiss::MetricType metric,
    bool encodeResidual, AscendIndexIVFSQConfig config)
    : AscendIndexIVF(dims, metric, nlist, config),
    impl_(std::make_shared<AscendIndexIVFSQImpl>(this, dims, nlist, qtype, metric, encodeResidual, config))
{
    AscendIndexIVF::impl_ = impl_;
    AscendIndex::impl_ = impl_;
}

AscendIndexIVFSQ::AscendIndexIVFSQ(int dims, int nlist, faiss::MetricType metric,
    AscendIndexIVFSQConfig config) : AscendIndexIVF(dims, metric, nlist, config) {}

AscendIndexIVFSQ::~AscendIndexIVFSQ() {}

void AscendIndexIVFSQ::copyFrom(const faiss::IndexIVFScalarQuantizer *index)
{
    FAISS_THROW_IF_NOT_MSG(impl_ != nullptr, "impl_ is nullptr!");
    impl_->copyFrom(index);
}

void AscendIndexIVFSQ::copyTo(faiss::IndexIVFScalarQuantizer *index) const
{
    FAISS_THROW_IF_NOT_MSG(impl_ != nullptr, "impl_ is nullptr!");
    impl_->copyTo(index);
}

void AscendIndexIVFSQ::train(idx_t n, const float *x)
{
    FAISS_THROW_IF_NOT_MSG(impl_ != nullptr, "impl_ is nullptr!");
    impl_->train(n, x);
}
} // ascend
} // faiss
