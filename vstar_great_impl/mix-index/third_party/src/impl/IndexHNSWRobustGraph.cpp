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


#include "impl/IndexHNSWRobustGraph.h"

#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <sys/stat.h>
#include <fstream>

#include <faiss/utils/distances.h>
#include <faiss/utils/sorting.h>
#include <faiss/IndexPQ.h>
#include "utils/AscendSearchAssert.h"
#include "utils/GraphDistanceComputer.h"
#include <utils/IoUtil.h>
#include "utils/VstarIoUtil.h"


namespace ascendsearch {

using MinimaxHeap = HNSWGraph::MinimaxHeap;
using NodeDistFarther = HNSWGraph::NodeDistFarther;

const int HYPER_RANDOMPARAMS = 789;
const int HYPER_PREV_DISPLAY = 10000;
const int HYPER_MODE2 = 2;
const int HYPER_PQBITS = 8;
const int HYPER_VAL256 = 256;


/**************************************************************
 * add / Search blocks of descriptors
 **************************************************************/
namespace {
/* Wrap the distance computer into one that negates the
   distances. This makes supporting INNER_PRODUCE Search easier */
struct NegativeDistanceComputer : faiss::DistanceComputer {
    // owned by this
    faiss::DistanceComputer *basedis;

    explicit NegativeDistanceComputer(faiss::DistanceComputer *basedis) : basedis(basedis)
    {
    }

    void set_query(const float *x) override
    {
        basedis->set_query(x);
    }

    // compute distance of vector i to current query
    float operator()(int64_t i) override
    {
        return -(*basedis)(i);
    }

    // batch computing
    void batch_process(const int64_t *indices, float *dists, size_t count)
    {
        for (size_t i = 0; i < count; ++i) {
            dists[i] = (*basedis)(indices[i]);
        }

        for (size_t i = 0; i < count; ++i) {
            dists[i] = -dists[i];
        }
    }

    // compute distance between two stored vectors
    float symmetric_dis(int64_t i, int64_t j) override
    {
        return -basedis->symmetric_dis(i, j);
    }

    virtual ~NegativeDistanceComputer()
    {
        delete basedis;
    }
};

faiss::DistanceComputer *StorageDistanceComputer(const faiss::IndexFlat *storage)
{
    if (storage->metric_type == faiss::METRIC_INNER_PRODUCT) {
        return std::make_unique<IPDisHnsw>(*storage).release();
    } else {
        return std::make_unique<L2DisHnsw>(*storage).release();
    }
}

faiss::DistanceComputer *StorageDistanceComputerPQ(const faiss::Index *storage)
{
    if (storage->metric_type == faiss::METRIC_INNER_PRODUCT) {
        return std::make_unique<NegativeDistanceComputer>(storage->get_distance_computer()).release();
    } else {
        return storage->get_distance_computer();
    }
}

void AddGraphVerticesBucketSort(HNSWGraph &hnsw, size_t n0, size_t n, std::vector<int> &hist, std::vector<int> &order)
{
    // build histogram
    for (size_t i = 0; i < n; i++) {
        size_t pt_id = i + n0;
        size_t pt_level = hnsw.levels[pt_id] - 1;
        while (pt_level >= hist.size())
            hist.push_back(0);
        hist[pt_level]++;
    }

    // accumulate
    std::vector<int> offsets(hist.size() + 1, 0);
    for (size_t i = 0; i < hist.size() - 1; i++) {
        offsets[i + 1] = offsets[i] + hist[i];
    }

    // bucket sort
    for (size_t i = 0; i < n; i++) {
        int pt_id = i + n0;
        int pt_level = hnsw.levels[pt_id] - 1;
        order[offsets[pt_level]++] = pt_id;
    }
}

void DescribeGraphInfo(HNSWGraph &innerGraph, faiss::IndexFlat *storage)
{
    // Collecting Neighbor Degree Information
    int max = 0;
    int min = 1e6;
    double avg = 0;
    for (int i = 0; i < storage->ntotal; i++) {
        int size = 0;
        size_t begin;
        size_t end;
        innerGraph.GetNeighborPosRange(i, 0, begin, end);
        for (size_t j = begin; j < end; j++) {
            if (innerGraph.neighbors[j] >= 0)
                size += 1;
        }
        max = std::max(size, max);
        min = std::min(size, min);
        avg += size;
    }
    avg = avg / storage->ntotal;
}

}  // namespace

/**************************************************************
 * IndexHNSW implementation
 **************************************************************/
IndexHNSWGraph::IndexHNSWGraph(int d, int outDegree, faiss::MetricType metric)
    : faiss::Index(d, metric), innerGraph(outDegree), ownFields(false), storage(nullptr)
{
}

IndexHNSWGraph::IndexHNSWGraph(std::shared_ptr<faiss::IndexFlat> storage, int outDegree)
    : faiss::Index(storage->d, storage->metric_type),
      innerGraph(outDegree),
      ownFields(false),
      storage(storage)
{
}

IndexHNSWGraph::~IndexHNSWGraph()
{
}

void IndexHNSWGraph::train(int64_t n, const float *x)
{
    ASCENDSEARCH_THROW_IF_NOT_MSG(storage, "Please use IndexHNSWFlat (or variants) instead of IndexHNSW directly");
    // innerGraph structure does not require training
    storage->train(n, x);
    is_trained = true;
}

void IndexHNSWGraph::SearchWithFilter(HNSWGraphSearchWithFilterParams params, float *distances, int64_t *labels) const
{
    ASCENDSEARCH_THROW_IF_NOT(params.topK > 0);
    ASCENDSEARCH_THROW_IF_NOT_MSG(storage, "Inner Storage is null or empty");

    int64_t checkPeriod = GetPeriodHint(innerGraph.maxLevel * d * innerGraph.efSearch);

    for (int64_t i0 = 0; i0 < params.nq; i0 += checkPeriod) {
        int64_t i1 = std::min(i0 + checkPeriod, params.nq);

#pragma omp parallel
        {
            faiss::VisitedTable vt(ntotal);
            graph::FastBitMap mask(params.scalaAttrMaxValue);

            faiss::DistanceComputer *dis = StorageDistanceComputer(storage.get());
            std::unique_ptr<faiss::DistanceComputer> del1(dis);

#pragma omp for
            for (int64_t i = i0; i < i1; i++) {
                int64_t *idxi = labels + i * params.topK;
                float *simi = distances + i * params.topK;
                dis->set_query(params.xq + i * d);

                faiss::maxheap_heapify(params.topK, simi, idxi);

                // filters is the index at which there are 1s
                {
                    mask.clear();
                    int start = params.filterPtr[i];    // filterPtr[0]: 当前query的起始地址
                    int end = params.filterPtr[i + 1];  // filterPtr[1]： 当前query的结束地址
                    for (int m = start; m < end; m++) {
                        mask.set(params.filters[m]);  // filters[m] 传入bit，然后由mask转化成uint32_t整数; e.g. 00010000 =>
                                                      // filters[m] = 4, mask.set(4) => data[0] += 2^4
                        // basically, this says that we should set a mask at filters[m] (a mask indicates trhat we
                        // should keep the neighbor vector!!) if filters[m] == scalaAttrOnCpu(v_x), we keep neighboring
                        // vector v_x in the search
                    }
                }  //

                innerGraph.SearchFilter(*dis, params.topK, mask, params.scalaAttrOnCpu, idxi, simi, vt);
                faiss::maxheap_reorder(params.topK, simi, idxi);
            }
        }
    }

    if (metric_type == faiss::METRIC_INNER_PRODUCT) {
        // we need to revert the negated distances
        for (int64_t i = 0; i < params.topK * params.nq; i++) {
            distances[i] = -distances[i];
        }
    }
}

void IndexHNSWGraph::search(int64_t nq, const float *xq, int64_t topK, float *distances, int64_t *labels,
                            const faiss::SearchParameters*) const
{
    ASCENDSEARCH_THROW_IF_NOT(topK > 0);
    ASCENDSEARCH_THROW_IF_NOT_MSG(storage, "Inner Storage is null or empty");

    int64_t checkPeriod = GetPeriodHint(innerGraph.maxLevel * d * innerGraph.efSearch);

    for (int64_t i0 = 0; i0 < nq; i0 += checkPeriod) {
        int64_t i1 = std::min(i0 + checkPeriod, nq);

#pragma omp parallel
        {
            faiss::VisitedTable vt(ntotal);
            faiss::DistanceComputer *dis = StorageDistanceComputer(storage.get());
            std::unique_ptr<faiss::DistanceComputer> del1(dis);
#pragma omp for
            for (int64_t i = i0; i < i1; i++) {
                int64_t *idxi = labels + i * topK;
                float *simi = distances + i * topK;
                dis->set_query(xq + i * d);

                faiss::maxheap_heapify(topK, simi, idxi);
                innerGraph.Search(*dis, topK, idxi, simi, vt);
                faiss::maxheap_reorder(topK, simi, idxi);
            }
        }
    }

    if (metric_type == faiss::METRIC_INNER_PRODUCT) {
        // we need to revert the negated distances
        for (int64_t i = 0; i < topK * nq; i++) {
            distances[i] = -distances[i];
        }
    }
}

void IndexHNSWGraph::add(int64_t n, const float *x)
{
    ASCENDSEARCH_THROW_IF_NOT_MSG(storage, "Inner storage is nullptr");
    ASCENDSEARCH_THROW_IF_NOT_MSG(is_trained, "please train Index first");
    ASCENDSEARCH_THROW_IF_NOT_MSG(n > 0, "Add Number should not be empty");
    int prevNTotal = ntotal;
    storage->add(n, x);
    ntotal = storage->ntotal;

    AddGraphVertices(prevNTotal, n, x, innerGraph.levels.size() == (size_t)ntotal);

    if (verbose) {
        DescribeGraphInfo(innerGraph, storage.get());
    }
}

void IndexHNSWGraph::InitMaxLevel(size_t n0, size_t n, const float *, bool presetLevels, int &maxLevel)
{
    if (verbose) {
        printf("Initiating...\n");
        printf("AddGraphVertices: adding %zu elements on top of %zu (presetLevels=%d)\n", n, n0,
               static_cast<int>(presetLevels));
    }
    // Determines the level for each node to be inserted and returns maxLevel

    maxLevel = innerGraph.PrepareLevelTab(n, presetLevels);

    if (verbose) {
        printf("AddGraphVertices:  maxLevel = %d\n", maxLevel);
        printf("Initiatiion Complete. Adding...\n");
    }
}

int PreDisplayPrintAndUpdate(int prevDisplay, int i, int i1, int i0)
{
    if (prevDisplay >= 0 && i - i0 > prevDisplay + HYPER_PREV_DISPLAY) {
        prevDisplay = i - i0;
        printf("  %d / %d\r", i - i0, i1 - i0);
        fflush(stdout);
    }
    return prevDisplay;
}

// indexHandler add point
void IndexHNSWGraph::AddGraphVertices(size_t n0, size_t n, const float *x, bool presetLevels)
{
    // Determines the level for each node to be inserted and returns maxLevel
    int maxLevel;
    InitMaxLevel(n0, n, x, presetLevels, maxLevel);

    std::vector<omp_lock_t> locks(ntotal);
    for (size_t i = 0; i < locks.size(); i++) {
        omp_init_lock(&locks[i]);
    }

    // add vectors from highest to lowest level
    std::vector<int> hist;
    std::vector<int> order(n);

    AddGraphVerticesBucketSort(innerGraph, n0, n, hist, order);

    // perform add
    std::mt19937 mt(HYPER_RANDOMPARAMS);

        int i1 = n;
        // construct from highest to lowest level
        for (int ptLevel = hist.size() - 1; ptLevel >= 0; ptLevel--) {
            int i0 = i1 - hist[ptLevel];
            if (verbose) {
                printf("Adding %d elements at level %d\n", i1 - i0, ptLevel);
            }
            // random permutation to get rid of dataset order bias

            for (int j = i0; j < i1; j++) {
                int rand = (mt() & 0x7fffffff) % (i1 - j);

                std::swap(order[j], order[j + rand]);
            }
#pragma omp parallel if (i1 > i0 + 100)
        {
            faiss::VisitedTable vt(ntotal);

            faiss::DistanceComputer *dis = StorageDistanceComputer(storage.get());
            std::unique_ptr<faiss::DistanceComputer> del1(dis);

            int prevDisplay = verbose && omp_get_thread_num() == 0 ? 0 : -1;
            size_t counter = 0;

#pragma omp for schedule(dynamic)
            for (int i = i0; i < i1; i++) {
                int ptId = order[i];

                dis->set_query(x + (ptId - n0) * d);

                innerGraph.AddWithLocks(*dis, ptLevel, ptId, locks, vt);

                prevDisplay = PreDisplayPrintAndUpdate(prevDisplay, i, i1, i0);

                counter++;
            }
        }

        i1 = i0;
    }
    ASCENDSEARCH_ASSERT(i1 == 0);
    for (size_t i = 0; i < locks.size(); i++) {
        omp_destroy_lock(&locks[i]);
    }
}

void IndexHNSWGraph::reset()
{
    innerGraph.Reset();
    if (storage != nullptr) {
        storage->reset();
    }
    ntotal = 0;
}

size_t IndexHNSWGraph::GetPeriodHint(size_t flops) const
{
    return std::max((size_t)10 * 10 * 1000 * 1000 / (flops + 1), (size_t)1);
}

/**************************************************************
 * IndexHNSWFlat implementation
 **************************************************************/
IndexHNSWGraphFlat::IndexHNSWGraphFlat() : IndexHNSWGraph(0)
{
    is_trained = true;
}

IndexHNSWGraphFlat::IndexHNSWGraphFlat(int d, int M, faiss::MetricType metric)
    : IndexHNSWGraph(std::make_shared<faiss::IndexFlat>(d, metric), M)
{
    ownFields = true;
    is_trained = true;
}

/**************************************************************
 * IndexHNSWGraphPQHybrid implementation
 **************************************************************/
IndexHNSWGraphPQHybrid::IndexHNSWGraphPQHybrid() : IndexHNSWGraphFlat()
{
    is_trained = false;
    for (int i = 0; i < MAX_THREAD_NUM; i++) {
        vts.push_back(std::make_shared<faiss::VisitedTable>(N_TOTAL_MAX));
    }
}

IndexHNSWGraphPQHybrid::IndexHNSWGraphPQHybrid(int d, int M, int pq_M, int nbits, faiss::MetricType metric)
    : IndexHNSWGraphFlat(d, M, metric)
{
    for (int i = 0; i < MAX_THREAD_NUM; i++) {
        vts.push_back(std::make_shared<faiss::VisitedTable>(N_TOTAL_MAX));
    }
    p_storage = std::make_unique<faiss::IndexPQ>(d, pq_M, nbits, metric);
    vecTrans = std::make_unique<faiss::OPQMatrix>(d, pq_M);
    ownFields = true;
    is_trained = false;
}

IndexHNSWGraphPQHybrid::~IndexHNSWGraphPQHybrid()
{
}

void IndexHNSWGraphPQHybrid::add(idx_t n, const float *x)
{
    ASCENDSEARCH_THROW_IF_NOT_MSG(p_storage,
                                  "Please use IndexHNSWGraphPQHybrid (or variants) instead of IndexHNSWGraph directly");
    ASCENDSEARCH_THROW_IF_NOT_MSG(n > 0, "Invalid n. The number of Adding vectors should be bigger than 0.");
    float *newX = const_cast<float *>(x);
    if (verbose) {
        printf("--- Build fast sorage start...\n");
    }
    p_storage->add(n, newX);
    if (verbose) {
        printf("--- Build fast sorage finished...\n");
    }
    IndexHNSWGraph::add(n, newX);
}

void IndexHNSWGraphPQHybrid::train(idx_t n, const float *x)
{
    ASCENDSEARCH_THROW_IF_NOT_MSG(p_storage,
                                  "Please use IndexHNSWGraphPQHybrid (or variants) instead of IndexHNSWGraph directly");
    ASCENDSEARCH_THROW_IF_NOT_MSG(n > 0, "Invalid n. The number of Training vectors should be bigger than 0.");
    // innerGraph structure does not require training
    float *newX = const_cast<float *>(x);
    p_storage->train(n, newX);
    is_trained = true;
}

void ThreadFunc(const HNSWGraph &innerGraph, idx_t *idxi, float *simi, faiss::DistanceComputer *fastComputer,
                faiss::DistanceComputer *preciseComputer, idx_t k, faiss::VisitedTable &vt)
{
    innerGraph.HybridSearch(*fastComputer, *preciseComputer, k, idxi, simi, vt);
}

void IndexHNSWGraphPQHybrid::search(idx_t n, const float *x, idx_t k, float *distances, idx_t *labels,
                                    const faiss::SearchParameters*) const
{
    ASCENDSEARCH_THROW_IF_NOT_MSG(p_storage,
                                  "Please use IndexHNSWGraphPQHybrid (or variants) instead of IndexHNSWGraph directly");
    ASCENDSEARCH_THROW_IF_NOT_MSG(storage,
                                  "Please use IndexHNSWGraphPQHybrid (or variants) instead of IndexHNSWGraph directly");
    ASCENDSEARCH_THROW_IF_NOT_MSG(n > 0, "Searching param (number of query) should be bigger than 0.");
    ASCENDSEARCH_THROW_IF_NOT_MSG(k > 0, "Searching param (tok) should be bigger than 0.");
    ASCENDSEARCH_THROW_IF_NOT_MSG(innerGraph.efSearch > 0, "Searching param (efSearch) should be bigger than 0.");
    ASCENDSEARCH_THROW_IF_NOT_MSG(innerGraph.efConstruction > 0,
                                  "Searching param (efConstruction) should be bigger than 0.");

    idx_t check_period =
        faiss::InterruptCallback::get_period_hint(innerGraph.maxLevel * d * innerGraph.efSearch);
    float *newX = const_cast<float *>(x);

    for (idx_t i0 = 0; i0 < n; i0 += check_period) {
        idx_t i1 = std::min(i0 + check_period, n);

        int numThreads = std::min((int)(i1 - i0), omp_get_max_threads());

#pragma omp parallel num_threads(numThreads)
        {
            std::shared_ptr<faiss::VisitedTable> vt = vts.at(omp_get_thread_num());
            vt->advance();

            faiss::DistanceComputer *fastComputer = StorageDistanceComputerPQ(p_storage.get());
            std::unique_ptr<faiss::DistanceComputer> del1(fastComputer);

            faiss::DistanceComputer *preciseComputer = StorageDistanceComputer(storage.get());
            std::unique_ptr<faiss::DistanceComputer> del(preciseComputer);

#pragma omp for
            for (idx_t i = i0; i < i1; i++) {
                idx_t *idxi = labels + i * k;
                float *simi = distances + i * k;
                fastComputer->set_query(newX + i * d);
                preciseComputer->set_query(newX + i * d);

                innerGraph.HybridSearch(*fastComputer, *preciseComputer, k, idxi, simi, *vt);
            }
        }
        faiss::InterruptCallback::check();
    }

    if (metric_type == faiss::METRIC_INNER_PRODUCT) {
        // we need to revert the negated distances
        for (idx_t i = 0; i < k * n; i++) {
            distances[i] = -distances[i];
        }
    }
}

void IndexHNSWGraphPQHybrid::SearchWithMask(HNSWGraphPQHybridSearchWithMaskParams params, float *distances,
                                            idx_t *labels) const
{
    ASCENDSEARCH_THROW_IF_NOT_MSG(p_storage,
                                  "Please use IndexHNSWGraphPQHybrid (or variants) instead of IndexHNSWGraph directly");
    ASCENDSEARCH_THROW_IF_NOT_MSG(storage,
                                  "Please use IndexHNSWGraphPQHybrid (or variants) instead of IndexHNSWGraph directly");
    ASCENDSEARCH_THROW_IF_NOT_MSG(params.n > 0, "Searching param (number of query) should be bigger than 0.");
    ASCENDSEARCH_THROW_IF_NOT_MSG(params.topK > 0, "Searching param (tok) should be bigger than 0.");
    ASCENDSEARCH_THROW_IF_NOT_MSG(params.nbOffset > 0, "Searching param (nbOffset) should be bigger than 0.");
    ASCENDSEARCH_THROW_IF_NOT_MSG(innerGraph.efSearch > 0, "Searching param (efSearch) should be bigger than 0.");
    ASCENDSEARCH_THROW_IF_NOT_MSG(innerGraph.efConstruction > 0,
                                  "Searching param (efConstruction) should be bigger than 0.");

    idx_t check_period =
        faiss::InterruptCallback::get_period_hint(innerGraph.maxLevel * d * innerGraph.efSearch);
    float *newX = const_cast<float *>(params.query);

    for (idx_t i0 = 0; i0 < params.n; i0 += check_period) {
        idx_t i1 = std::min(i0 + check_period, params.n);

        int numThreads = std::min((int)(i1 - i0), omp_get_max_threads());

#pragma omp parallel num_threads(numThreads)
        {
            std::shared_ptr<faiss::VisitedTable> vt = vts.at(omp_get_thread_num());
            vt->advance();

            faiss::DistanceComputer *fastComputer = StorageDistanceComputerPQ(p_storage.get());
            std::unique_ptr<faiss::DistanceComputer> del1(fastComputer);

            faiss::DistanceComputer *preciseComputer = StorageDistanceComputer(storage.get());
            std::unique_ptr<faiss::DistanceComputer> del(preciseComputer);

#pragma omp for
            for (idx_t i = i0; i < i1; i++) {
                idx_t *idxi = labels + i * params.topK;
                float *simi = distances + i * params.topK;
                fastComputer->set_query(newX + i * d);
                preciseComputer->set_query(newX + i * d);

                innerGraph.HybridSearchWithMask(*fastComputer, *preciseComputer, params.topK,
                                                params.mask + i * params.maskDim, params.cpuIdx, params.nbOffset, idxi,
                                                simi, *vt);  // start searching by query
            }
        }
        faiss::InterruptCallback::check();
    }

    if (metric_type == faiss::METRIC_INNER_PRODUCT) {
        // we need to revert the negated distances
        for (idx_t i = 0; i < params.topK * params.n; i++) {
            distances[i] = -distances[i];
        }
    }
}

void IndexHNSWGraphPQHybrid::SaveIndex(std::string filePath)
{
    graph::FileIOWriter writer(filePath);
    // Write Index Type
    uint32_t head = graph::fourcc("AhRp");
    size_t writeSize = writer(&head, sizeof(head), 1);
    ASCENDSEARCH_THROW_IF_NOT_FMT(writeSize == 1, "SaveIndex fail, expected to write %d item, actually write %d item",
                                  1, writeSize);
    graph::WriteIndex(this, &writer);

    int idMapSize = static_cast<int>(idMap.size());
    writeSize = writer(&idMapSize, sizeof(idMapSize), 1);
    ASCENDSEARCH_THROW_IF_NOT_FMT(writeSize == 1, "SaveIndex fail, expected to write %d item, actually write %d item",
                                  1, writeSize);
    writeSize = 0;
    for (const auto &idPair : idMap) {
        writeSize += writer(&idPair.first, sizeof(idPair.first), 1);
        writeSize += writer(&idPair.second, sizeof(idPair.second), 1);
    }
    ASCENDSEARCH_THROW_IF_NOT_FMT(static_cast<int>(writeSize) == idMapSize * 2,
                                  "SaveIndex fail, expected to write %d item, actually write %d item",
                                  idMapSize * 2, writeSize);
}

std::unique_ptr<IndexHNSWGraphPQHybrid> IndexHNSWGraphPQHybrid::LoadIndex(std::string filePath)
{
    graph::FileIOReader reader(filePath);
    uint32_t header;
    auto readSize = reader(&header, sizeof(header), 1);
    ASCENDSEARCH_THROW_IF_NOT_FMT(readSize == 1, "LoadIndex fail, expected to read %d item, actually read %d item",
                                  1, readSize);
    ASCENDSEARCH_THROW_IF_NOT_MSG(header == graph::fourcc("AhRp"), "Index Format Not Format");
    auto index = graph::ReadIndexHNSWGraphPQHybrid(&reader, 0);
    ASCENDSEARCH_THROW_IF_NOT_MSG(index != nullptr, "Returned IndexGreat index shouldn't be a nullptr.");
    int idMapSize = 0;
    readSize = reader(&idMapSize, sizeof(idMapSize), 1);
    ascendSearchacc::loadedValueSanityCheck(idMapSize, 1e8); // nTotal maxValue is 1e8
    ASCENDSEARCH_THROW_IF_NOT_FMT(readSize == 1, "LoadIndex fail, expected to read %d item, actually read %d item",
                                  1, readSize);
    for (int i = 0; i < idMapSize; ++i) {
        int64_t realId = 0;
        int64_t virtualId = 0;
        readSize = reader(&realId, sizeof(realId), 1);
        ASCENDSEARCH_THROW_IF_NOT_FMT(readSize == 1, "LoadIndex fail, expected to read %d item, actually read %d item",
                                      1, readSize);
        readSize = reader(&virtualId, sizeof(virtualId), 1);
        ASCENDSEARCH_THROW_IF_NOT_FMT(readSize == 1, "LoadIndex fail, expected to read %d item, actually read %d item",
                                      1, readSize);
        index->idMap.insert(std::make_pair(realId, virtualId));
    }
    return index;
}

int64_t IndexHNSWGraphPQHybrid::GetDim()
{
    return this->d;
}

int64_t IndexHNSWGraphPQHybrid::GetNTotal()
{
    return this->ntotal;
}

const std::map<int64_t, int64_t>& IndexHNSWGraphPQHybrid::GetIdMap() const
{
    return this->idMap;
}

void IndexHNSWGraphPQHybrid::SetIdMap(const std::map<int64_t, int64_t> &idMap)
{
    this->idMap = idMap;
}

}  // namespace ascendsearch
