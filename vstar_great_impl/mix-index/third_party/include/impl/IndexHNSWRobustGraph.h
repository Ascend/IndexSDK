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


#ifndef GFEATURERETRIEVAL_INDEX_HNSW_GRAPH_H
#define GFEATURERETRIEVAL_INDEX_HNSW_GRAPH_H

#include <vector>
#include <map>

#include <faiss/IndexFlat.h>
#include <faiss/VectorTransform.h>

#include "impl/HNSWRobustGraph.h"
#include "utils/ThreadPool.h"
using idx_t = int64_t;

namespace ascendsearch {
struct IndexHNSWGraph;

const int DEFAULT_OUT_DEG = 32;
const int DEFAULT_WINDOW_SIZE = 8;
const int GLOBAL_DOUBLE = 2;
const int MAX_THREAD_NUM = 260;
const std::vector<int> validDim = {128, 256, 512, 1024};
const int N_TOTAL_MAX = 1e8; // 1亿
const int EF_SEARCH_MAX = 200;
const int EF_CONSTRUCTION_MIN = 200;
const int EF_CONSTRUCTION_MAX = 400;
const int R_MIN = 50;
const int R_MAX = 100;
const int DEFAULT_INDEX_PQ_BITS = 8;
const int HNSW_MAX_LEVEL = 20;

struct HNSWGraphSearchWithFilterParams {
    int64_t nq;
    const float *xq;
    int *filterPtr;
    uint32_t *filters;
    uint32_t *scalaAttrOnCpu;
    uint32_t scalaAttrMaxValue;
    int64_t topK;

    explicit HNSWGraphSearchWithFilterParams();
    explicit HNSWGraphSearchWithFilterParams(int64_t nq, const float *xq, int *filterPtr, uint32_t *filters,
                                             uint32_t *scalaAttrOnCpu, uint32_t scalaAttrMaxValue, int64_t topK)
        : nq(nq),
          xq(xq),
          filterPtr(filterPtr),
          filters(filters),
          scalaAttrOnCpu(scalaAttrOnCpu),
          scalaAttrMaxValue(scalaAttrMaxValue),
          topK(topK)
    {
    }
    ~HNSWGraphSearchWithFilterParams();
};
/* * The HNSW index is a normal random-access index with a HNSW
 * link structure built on top */
struct IndexHNSWGraph : faiss::Index {
    HNSWGraph innerGraph;  // /> the link structure
    int orderType = 0;

    // the sequential storage
    bool ownFields;
    std::shared_ptr<faiss::IndexFlat> storage;

    explicit IndexHNSWGraph(int d, int outDegree = DEFAULT_OUT_DEG, faiss::MetricType metric = faiss::METRIC_L2);
    explicit IndexHNSWGraph(std::shared_ptr<faiss::IndexFlat> storage, int outDegree = DEFAULT_OUT_DEG);

    ~IndexHNSWGraph() override;

    void add(int64_t n, const float *x) override;

    // / Trains the storage if needed
    void train(int64_t n, const float *x) override;

    // / entry point for Search
    void search(int64_t nq, const float *xq, int64_t topK, float *distances, int64_t *labels,
                const faiss::SearchParameters* params = nullptr) const override;

    void reset() override;

    void SearchWithFilter(HNSWGraphSearchWithFilterParams params, float *distances, int64_t *labels) const;

private:
    void AddGraphVertices(size_t n0, size_t n, const float *x, bool presetLevels = false);
    void InitMaxLevel(size_t n0, size_t n, const float *x, bool presetLevels, int &maxLevel);
    size_t GetPeriodHint(size_t flops) const;
};

struct HNSWGraphPQHybridSearchWithMaskParams {
    idx_t n;
    const float *query;
    idx_t topK;
    uint8_t *mask;
    size_t maskDim;
    size_t cpuIdx;
    size_t nbOffset;

    explicit HNSWGraphPQHybridSearchWithMaskParams();
    explicit HNSWGraphPQHybridSearchWithMaskParams(idx_t n, const float *query, idx_t topK, uint8_t *mask,
                                                   size_t maskDim, size_t cpuIdx, size_t nbOffset)
        : n(n), query(query), topK(topK), mask(mask), maskDim(maskDim), cpuIdx(cpuIdx), nbOffset(nbOffset)
    {
    }
};

/* * Flat index topped with with a HNSW structure to access elements
 *  more efficiently.
 */
struct IndexHNSWGraphFlat : IndexHNSWGraph {
    IndexHNSWGraphFlat();
    IndexHNSWGraphFlat(int d, int M, faiss::MetricType metric = faiss::METRIC_L2);
};

struct IndexHNSWGraphPQHybrid : IndexHNSWGraphFlat {
    std::unique_ptr<faiss::Index> p_storage;  // /> high precision storage

    std::unique_ptr<faiss::VectorTransform> vecTrans;  // /> optimized Product Quantization CVPR2013

    IndexHNSWGraphPQHybrid();
    IndexHNSWGraphPQHybrid(int d, int M, int pq_M, int nbits, faiss::MetricType metric = faiss::METRIC_L2);
    ~IndexHNSWGraphPQHybrid();

    void add(idx_t n, const float *x) override;

    // / Train the storage if needed
    void train(idx_t n, const float *x) override;

    void search(idx_t n, const float *x, idx_t k, float *distances, idx_t *labels,
                const faiss::SearchParameters* params = nullptr) const override;

    void SearchWithMask(HNSWGraphPQHybridSearchWithMaskParams params, float *distances, idx_t *labels) const;

    void SaveIndex(std::string filePath);

    std::unique_ptr<IndexHNSWGraphPQHybrid> LoadIndex(std::string filePath);

    int64_t GetDim();

    int64_t GetNTotal();

    void SetIdMap(const std::map<int64_t, int64_t> &idMap);

    const std::map<int64_t, int64_t>& GetIdMap() const;

private:
    std::map<int64_t, int64_t> idMap;
    std::vector<std::shared_ptr<faiss::VisitedTable>> vts;
};
void ThreadFunc(const HNSWGraph &innerGraph, idx_t *idxi, float *simi, faiss::DistanceComputer *fastComputer,
                faiss::DistanceComputer *preciseComputer, idx_t k, faiss::VisitedTable &vt);
}  // namespace ascendsearch

#endif