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

#include "AscendIndexIVFRabitQ.h"

#include "ascend/impl/AscendIndexIVFRabitQImpl.h"

namespace faiss {
namespace ascend {
// implementation of AscendIndexIVF
AscendIndexIVFRabitQ::AscendIndexIVFRabitQ(int dims, faiss::MetricType metric, int nlist,
    AscendIndexIVFRabitQConfig config) : AscendIndexIVF(dims, metric, nlist, config),
    impl_(std::make_shared<AscendIndexIVFRabitQImpl>(this, dims, nlist, metric, config))
{
    AscendIndexIVF::impl_ = impl_;
    AscendIndex::impl_ = impl_;
}

AscendIndexIVFRabitQ::~AscendIndexIVFRabitQ() {}

void AscendIndexIVFRabitQ::train(idx_t n, const float *x)
{
    FAISS_THROW_IF_NOT_MSG(impl_ != nullptr, "impl_ is nullptr!");
    impl_->train(n, x);
}
} // namespace ascend
} // namespace faiss
