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


#include "AscendIndexIVFSQT.h"

#include "ascend/custom/impl/AscendIndexIVFSQTImpl.h"

namespace faiss {
namespace ascend {
const int DEFAULT_DIM = 256;
const int DEFAULT_NLIST = 16384;

AscendIndexIVFSQT::AscendIndexIVFSQT(const faiss::IndexIVFScalarQuantizer *index, AscendIndexIVFSQTConfig config)
    : AscendIndexIVFSQ((index == nullptr) ? DEFAULT_DIM : index->d, (index == nullptr) ? DEFAULT_NLIST : index->nlist,
                       MetricType::METRIC_INNER_PRODUCT, config),
    impl_(std::make_shared<AscendIndexIVFSQTImpl>(this, index, config))
{
    AscendIndexIVFSQ::impl_ = impl_;
    AscendIndexIVF::impl_ = impl_;
    AscendIndex::impl_ = impl_;
}

AscendIndexIVFSQT::AscendIndexIVFSQT(int dimIn, int dimOut, int nlist, faiss::ScalarQuantizer::QuantizerType qtype,
    faiss::MetricType metric, AscendIndexIVFSQTConfig config)
    : AscendIndexIVFSQ(dimIn, nlist, metric, config),
    impl_(std::make_shared<AscendIndexIVFSQTImpl>(this, dimIn, dimOut, nlist, qtype, metric, config))
{
    AscendIndexIVFSQ::impl_ = impl_;
    AscendIndexIVF::impl_ = impl_;
    AscendIndex::impl_ = impl_;
}

AscendIndexIVFSQT::~AscendIndexIVFSQT() {}

void AscendIndexIVFSQT::copyTo(faiss::IndexIVFScalarQuantizer *index)
{
    impl_->copyTo(index);
}

void AscendIndexIVFSQT::copyFrom(const faiss::IndexIVFScalarQuantizer *index)
{
    impl_->copyFrom(index);
}

void AscendIndexIVFSQT::train(idx_t n, const float *x)
{
    impl_->train(n, x);
}

void AscendIndexIVFSQT::fineTune(size_t n, const float *x)
{
    impl_->fineTune(n, x);
}

void AscendIndexIVFSQT::setFuzzyK(int value)
{
    impl_->setFuzzyK(value);
}

void AscendIndexIVFSQT::setThreshold(float value)
{
    impl_->setThreshold(value);
}

int AscendIndexIVFSQT::getFuzzyK() const
{
    return impl_->getFuzzyK();
}

float AscendIndexIVFSQT::getThreshold() const
{
    return impl_->getThreshold();
}

void AscendIndexIVFSQT::update(bool cleanData)
{
    impl_->update(cleanData);
}

void AscendIndexIVFSQT::updateTParams(int l2Probe, int l3SegmentNum)
{
    impl_->updateTParams(l2Probe, l3SegmentNum);
}

void AscendIndexIVFSQT::setLowerBound(int lowerBound)
{
    impl_->setLowerBound(lowerBound);
}

int AscendIndexIVFSQT::getLowerBound() const
{
    return impl_->getLowerBound();
}

void AscendIndexIVFSQT::setMergeThres(int mergeThres)
{
    impl_->setMergeThres(mergeThres);
}

int AscendIndexIVFSQT::getMergeThres() const
{
    return impl_->getMergeThres();
}

void AscendIndexIVFSQT::setMemoryLimit(float memoryLimit)
{
    impl_->setMemoryLimit(memoryLimit);
}

void AscendIndexIVFSQT::setAddTotal(size_t addTotal)
{
    impl_->setAddTotal(addTotal);
}

void AscendIndexIVFSQT::setPreciseMemControl(bool preciseMemControl)
{
    impl_->setPreciseMemControl(preciseMemControl);
}

void AscendIndexIVFSQT::reset()
{
    impl_->reset();
}

void AscendIndexIVFSQT::setNumProbes(int nprobes)
{
    impl_->setNumProbes(nprobes);
}

size_t AscendIndexIVFSQT::remove_ids(const faiss::IDSelector &sel)
{
    return impl_->remove_ids(sel);
}

float AscendIndexIVFSQT::getQMin() const
{
    return impl_->getQMin();
}

float AscendIndexIVFSQT::getQMax() const
{
    return impl_->getQMax();
}

uint32_t AscendIndexIVFSQT::getListLength(int listId) const
{
    return impl_->getListLength(listId);
}

void AscendIndexIVFSQT::getListCodesAndIds(
    int listId, std::vector<uint8_t> &codes, std::vector<ascend_idx_t> &ids) const
{
    impl_->getListCodesAndIds(listId, codes, ids);
}

void AscendIndexIVFSQT::setSearchParams(int nprobe, int l2Probe, int l3SegmentNum)
{
    impl_->setSearchParams(nprobe, l2Probe, l3SegmentNum);
}

void AscendIndexIVFSQT::setSortMode(int mode)
{
    impl_->setSortMode(mode);
}

void AscendIndexIVFSQT::setUseCpuUpdate(int numThreads)
{
    impl_->setUseCpuUpdate(numThreads);
}
} // ascend
} // faiss