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


#include "AscendIndex.h"

#include "ascend/impl/AscendIndexImpl.h"

namespace faiss {
namespace ascendSearch {
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
    impl_->add(n, x);
}

void AscendIndex::add_with_ids(idx_t n, const float* x,
                               const idx_t* ids)
{
    impl_->add_with_ids(n, x, ids);
}

size_t AscendIndex::remove_ids(const faiss::IDSelector& sel)
{
    return impl_->remove_ids(sel);
}

void AscendIndex::search(idx_t n, const float* x, idx_t k,
    float* distances, idx_t* labels, const SearchParameters *params) const
{
    VALUE_UNUSED(params);
    impl_->search(n, x, k, distances, labels);
}

void AscendIndex::reserveMemory(size_t numVecs)
{
    impl_->reserveMemory(numVecs);
}

size_t AscendIndex::reclaimMemory()
{
    return impl_->reclaimMemory();
}

void AscendIndex::reset()
{
    impl_->reset();
}

std::vector<int> AscendIndex::getDeviceList()
{
    return impl_->getDeviceList();
}
}  // namespace ascendSearch
}  // namespace faiss
