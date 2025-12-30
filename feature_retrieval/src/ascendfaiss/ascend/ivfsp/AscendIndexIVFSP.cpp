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


#include "AscendIndexIVFSP.h"
#include "ascend/ivfsp/AscendIndexIVFSPImpl.h"

namespace faiss {
namespace ascend {

std::shared_ptr<AscendIndexIVFSP> AscendIndexIVFSP::loadAllData(const AscendIndexIVFSPConfig &config,
    const uint8_t *data, size_t dataLen, const AscendIndexIVFSP *codeBookSharedIdx)
{
    auto codeBookSharedImpl = (codeBookSharedIdx != nullptr) ? codeBookSharedIdx->impl_ : nullptr;
    auto index = std::shared_ptr<AscendIndexIVFSP>(new AscendIndexIVFSP());
    index->impl_ = AscendIndexIVFSPImpl::loadAllData(index, config, data, dataLen, codeBookSharedImpl);
    index->AscendIndex::impl_ = index->impl_;
    return index;
}

AscendIndexIVFSP::AscendIndexIVFSP(int dims, int nonzeroNum, int nlist, const char *codeBookPath,
    faiss::ScalarQuantizer::QuantizerType qType, faiss::MetricType metric, AscendIndexIVFSPConfig config)
    : AscendIndex(0, metric, config),
    impl_(std::make_shared<AscendIndexIVFSPImpl>(this, dims, nonzeroNum, nlist, qType, metric, config))
{
    impl_->addCodeBook(codeBookPath);
    AscendIndex::impl_ = impl_;
}

AscendIndexIVFSP::AscendIndexIVFSP(int dims, int nonzeroNum, int nlist, const AscendIndexIVFSP &codeBookSharedIdx,
    faiss::ScalarQuantizer::QuantizerType qType, faiss::MetricType metric, AscendIndexIVFSPConfig config)
    : AscendIndex(0, metric, config),
    impl_(std::make_shared<AscendIndexIVFSPImpl>(this, dims, nonzeroNum, nlist, qType, metric,
        config))
{
    impl_->addCodeBook(*codeBookSharedIdx.impl_);
    AscendIndex::impl_ = impl_;
}

AscendIndexIVFSP::AscendIndexIVFSP(int dims, int nonzeroNum, int nlist,
    faiss::ScalarQuantizer::QuantizerType qType, faiss::MetricType metric, AscendIndexIVFSPConfig config)
    : AscendIndex(0, metric, config),
    impl_(std::make_shared<AscendIndexIVFSPImpl>(this, dims, nonzeroNum, nlist, qType, metric, config))
{
    AscendIndex::impl_ = impl_;
}


AscendIndexIVFSP::AscendIndexIVFSP() : AscendIndex(0, faiss::MetricType::METRIC_INNER_PRODUCT, AscendIndexConfig()) {}

AscendIndexIVFSP::~AscendIndexIVFSP() {}

void AscendIndexIVFSP::saveAllData(uint8_t *&data, size_t &dataLen) const
{
    impl_->saveAllData(data, dataLen);
}

void AscendIndexIVFSP::trainCodeBook(const AscendIndexCodeBookInitParams &codeBookInitParams) const
{
    impl_->trainCodeBook(codeBookInitParams);
}

void AscendIndexIVFSP::trainCodeBookFromMem(const AscendIndexCodeBookInitFromMemParams
    &codeBookInitFromMemParams) const
{
    impl_->trainCodeBookFromMem(codeBookInitFromMemParams);
}

void AscendIndexIVFSP::addCodeBook(const char *codeBookPath)
{
    impl_->addCodeBook(codeBookPath);
}

// `x` need to be resident on CPU
// Handles paged adds if the add set is too large;
void AscendIndexIVFSP::add(idx_t n, const float *x)
{
    impl_->add(n, x);
}

// `x` and `ids` need to be resident on the CPU;
// Handles paged adds if the add set is too large;
void AscendIndexIVFSP::add_with_ids(idx_t n, const float *x, const idx_t *ids)
{
    impl_->add_with_ids(n, x, ids);
}

size_t AscendIndexIVFSP::remove_ids(const faiss::IDSelector &sel)
{
    return impl_->remove_ids(sel);
}

void AscendIndexIVFSP::reset()
{
    impl_->reset();
}

void AscendIndexIVFSP::loadAllData(const char *dataPath)
{
    impl_->loadAllData(dataPath);
}

void AscendIndexIVFSP::saveAllData(const char *dataPath)
{
    impl_->saveAllData(dataPath);
}

void AscendIndexIVFSP::setVerbose(bool verbose)
{
    this->verbose = verbose;
    impl_->pIVFSPSQ->verbose = verbose;
}

void AscendIndexIVFSP::setNumProbes(int nprobes)
{
    impl_->setNumProbes(nprobes);
}

void AscendIndexIVFSP::search(idx_t n, const float *x, idx_t k,
    float *distances, idx_t *labels, const SearchParameters *) const
{
    impl_->search(n, x, k, distances, labels);
}

void AscendIndexIVFSP::search_with_filter(idx_t n, const float *x, idx_t k,
    float *distances, idx_t *labels, const void *filters) const
{
    impl_->search_with_filter(n, x, k, distances, labels, filters);
}

}
}