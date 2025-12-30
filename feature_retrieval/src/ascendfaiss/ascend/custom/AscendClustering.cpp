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


#include "AscendClustering.h"
#include "AscendClusteringImpl.h"


namespace faiss {
namespace ascend {
AscendClustering::AscendClustering(int d, int k, MetricType metricType, AscendClusteringConfig config)
    : Clustering(d, k),
    impl_(std::make_shared<AscendClusteringImpl>(d, k, metricType, config, this)) {}

AscendClustering::~AscendClustering() {}

void AscendClustering::Add(idx_t n, const float *x) const
{
    impl_->add(n, x);
}

// Use fast add to transfer data to dev
void AscendClustering::Train(int niter, float *centroids, bool clearData) const
{
    impl_->train(niter, centroids, clearData);
}

#ifdef HOSTCPU
void AscendClustering::DistributedTrain(
    int niter, float *centroids, const std::vector<int> &deviceList, bool clearData) const
{
    impl_->distributedTrain(niter, centroids, deviceList, clearData);
}
#endif

void AscendClustering::ComputeCorr(float *corr, bool clearData) const
{
    impl_->computeCorr(corr, clearData);
}

size_t AscendClustering::GetNTotal() const
{
    return impl_->getNTotal();
}

void AscendClustering::SubClusAddInt8(idx_t n, const uint8_t *x, int seq, int k) const
{
    impl_->subClusAddInt8(n, x, seq, k);
}

void AscendClustering::SubClusExecInt8(int niter, uint16_t *labels, float *centroids,
    int batchSubNlist, size_t batchNTotal) const
{
    impl_->subClusExecInt8(niter, labels, centroids, batchSubNlist, batchNTotal);
}

void AscendClustering::UpdateVdm(uint16_t *vmin, uint16_t *vdiff) const
{
    impl_->updateVDM(vmin, vdiff);
}

int AscendClustering::GetSubBucketNum(int seq) const
{
    return impl_->getSubBucketNum(seq);
}
}
}