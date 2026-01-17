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


#ifndef ASCEND_INDEX_CLUSTERING
#define ASCEND_INDEX_CLUSTERING

#include <memory>

#include <faiss/Clustering.h>

namespace faiss {
namespace ascend {
const int64_t CLUSTERING_DEFAULT_MEM = 0x2000000; // 0x2000000 mean 32M(resource mem pool's size)
struct AscendClusteringConfig {
    inline AscendClusteringConfig() : deviceList({ 0 }), resourceSize(CLUSTERING_DEFAULT_MEM) {}

    inline explicit AscendClusteringConfig(std::initializer_list<int> devices,
        int64_t resources = CLUSTERING_DEFAULT_MEM)
        : deviceList(devices), resourceSize(resources) {}

    inline explicit AscendClusteringConfig(std::vector<int> devices, int64_t resources = CLUSTERING_DEFAULT_MEM)
        : deviceList(devices), resourceSize(resources) {}

    // Ascend devices mask on which the index is resident
    std::vector<int> deviceList;
    int64_t resourceSize;
};

class AscendClusteringImpl;
class AscendClustering : public Clustering {
public:
    AscendClustering(int d, int k, MetricType metricType, AscendClusteringConfig config);

    virtual ~AscendClustering();

    void Add(idx_t n, const float *x) const;
    void AddFp32(idx_t n, const float *x) const;

    void Train(int niter, float *centroids, bool clearData = true) const;
    void TrainFp32(int niter, float *centroids, bool clearData = true)  const;

#ifdef HOSTCPU
    void DistributedTrain(int niter, float *centroids, const std::vector<int> &deviceList, bool clearData) const;
#endif

    void ComputeCorr(float *corr, bool clearData = false) const;

    size_t GetNTotal() const;

    void SubClusAddInt8(idx_t n, const uint8_t *x, int seq, int k) const;

    void SubClusExecInt8(int niter, uint16_t *labels, float *centroids,
        int batchSubNlist, size_t batchNTotal) const;

    void UpdateVdm(uint16_t *vmin, uint16_t *vdiff) const;

    int GetSubBucketNum(int seq) const;

protected:
    std::shared_ptr<AscendClusteringImpl> impl_;
};
}
}

#endif