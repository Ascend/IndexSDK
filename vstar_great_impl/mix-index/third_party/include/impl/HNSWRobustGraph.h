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


#ifndef GFEATURERETRIEVAL_HNSW_GRAPH_H
#define GFEATURERETRIEVAL_HNSW_GRAPH_H

#include <queue>
#include <unordered_set>
#include <vector>
#include <random>
#include <omp.h>
#include <bitset>
#include <unordered_map>

#include <cmath>

#include <faiss/Index.h>
#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/utils/Heap.h>
#include <faiss/utils/random.h>
#include "utils/FastBitMap.h"

#ifdef DEBUG
#define ASSERT(f) assert(f)
#else
#define ASSERT(f) ((void)0)
#endif
constexpr size_t DISTANCE_BATCH_SIZE = 128;
namespace ascendsearch {

    constexpr float EPSILON = 1e-6; // 浮点比较门槛

    inline bool FloatEqual(float a, float b)
    {
        return fabs(a - b) < EPSILON;
    }


    struct Candidate {
    int64_t id;
    float dist;
    bool isChecked;
    Candidate() = default;
    Candidate(int64_t _id, float _dist, bool _isChecked) : id(_id), dist(_dist), isChecked(_isChecked)
    {
    }
    bool operator<(const Candidate &b) const
    {
        if (!FloatEqual(this->dist, b.dist)) {
            return this->dist < b.dist;
        } else {
            return this->id < b.id;
        }
    }
};

using maxCmp = faiss::CMax<float, int64_t >;
using cmp = faiss::CMax<float, int >;
struct HNSWGraph {
    using Node = std::pair<float, int>;

    /* * Heap structure that allows fast */
    struct MinimaxHeap {
        int n;
        int k;
        int nvalid;

        std::vector<int> ids;
        std::vector<float> dis;

        MinimaxHeap()
        {
        }

        explicit MinimaxHeap(int n) : n(n), k(0), nvalid(0), ids(n), dis(n)
        {
        }

        void Push(int i, float v);

        float Max() const;

        int Size() const;

        void Clear();

        int PopMin(float &minValue);

        int CountBelow(float threshold);
    };

    // / to sort pairs of (id, distance) from nearest to fathest or the reverse
    struct NodeDistCloser {
        float d;
        int id;

        NodeDistCloser(float d, int id) : d(d), id(id)
        {
        }

        bool operator<(const NodeDistCloser &obj1) const
        {
            return d < obj1.d;
        }
    };

    struct NodeDistFarther {
        float d;
        int id;

        NodeDistFarther(float d, int id) : d(d), id(id)
        {
        }

        bool operator<(const NodeDistFarther &obj1) const
        {
            return d > obj1.d;
        }
    };

    // / debug

    // / Open cluster flag
    bool isOpenCluster;

    // / select edge distance ratio
    float thresholdEdge;

    // / 每层的概率数组，加起来和为1.0
    // / assignment probability to each layer (sum=1)
    std::vector<double> assignProbas;

    // / number of neighbors stored per layer (cumulative), should not
    // / be changed after first add
    std::vector<int> cumNeighborCntPerLevel;

    // / level of each vector (base level = 1), capacity = nTotal
    std::vector<int> levels;

    // / offsets[i] is the offset in the neighbors array where vector i is stored
    // / capacity nTotal + 1
    std::vector<size_t> offsets;

    // / neighbors[offsets[i]:offsets[i+1]] is the list of neighbors of vector i
    // / for all levels. this is where all storage goes.
    std::vector<int> neighbors;

    // / entry point in the Search structure (one of the points with maximum
    // / level
    int entryPoint;

    std::mt19937 mt;

    // / maximum level
    int maxLevel;

    // / expansion factor at construction time
    int efConstruction;

    // / expansion factor at Search time
    int efSearch;

    // / during Search: do we check whether the next best distance is good
    // / enough?
    bool check_relative_distance = true;

    // / number of entry points in levels > 0.
    int upperBeam;

    // / use bounded queue during exploration
    bool search_bounded_queue = true;

    // methods that initialize the tree sizes

    // / initialize the assignProbas and cumNeighborCntPerLevel to
    // / have 2*M links on level 0 and M links on levels > 0
    void SetDefaultProbas(int M, float levelMult);

    // / set nb of neighbors for this level (before adding anything)
    void SetNeighborCnt(int level_no, int n);

    // methods that access the tree sizes

    // / nb of neighbors for this level
    int GetNeighborCnt(int layer) const;

    // / cumumlative nb up to (and excluding) this level
    int cum_nb_neighbors(int layer_no) const;

    // / range of entries in the neighbors table of vertex nodeId at layerId
    void GetNeighborPosRange(int64_t nodeId, int layerId, size_t &begin, size_t &end) const;

    // / only mandatory parameter: nb of neighbors
    explicit HNSWGraph(int M = 32);

    // / pick a random level for a new point
    int RandomLevel();

    void AddLinksInLevelGraph(int level, faiss::DistanceComputer &ptdis, int pt_id, int nearest, float d_nearest,
                              omp_lock_t *locks, faiss::VisitedTable &vt);

    /* * add point nodeId on all levels <= nodeLevel and build the link
     * structure for them. */
    void AddWithLocks(faiss::DistanceComputer &distComputer, int nodeLevel, int nodeId, std::vector<omp_lock_t> &locks,
                      faiss::VisitedTable &vt);

    int SearchFromCandidates(faiss::DistanceComputer &qdis, int k, int64_t *I, float *D, MinimaxHeap &candidates,
                             faiss::VisitedTable &vt, int level, MinimaxHeap &candidatesResort, int nres_in = 0) const;

    int SearchFromCandidatesWithStore(faiss::DistanceComputer &qdis, int k, int64_t *I, float *D,
                                      MinimaxHeap &candidates,
                                      faiss::VisitedTable &vt, int level, MinimaxHeap &candidatesResort,
                                      std::unordered_map<int, float> &map, int nres_in = 0) const;

    int SearchFromCandidatesWithFilter(faiss::DistanceComputer &qdis, int k, graph::FastBitMap &filterMask,
                                       uint32_t *scalaAttrOnCpu, int64_t *I, float *D, MinimaxHeap &candidates,
                                       faiss::VisitedTable &vt, int level, MinimaxHeap &candidatesResort,
                                       int nres_in = 0) const;

    /* If the bit is 1, we MASK the base vector. Note this is different from SearchFromCandidatesWithFilter's mask */
    int SearchFromCandidatesWithMask(faiss::DistanceComputer &qdis, int k, uint8_t *mask, size_t cpuIdx,
                                     size_t nbOffset, int64_t *I, float *D, MinimaxHeap &candidates,
                                     faiss::VisitedTable &vt, int level, MinimaxHeap &candidatesResort,
                                     int nres_in = 0) const;

    std::priority_queue<Node> search_from_candidate_unbounded(const Node &node, faiss::DistanceComputer &qdis, int ef,
                                                              faiss::VisitedTable *vt) const;

    std::priority_queue<HNSWGraph::Node> SearcFromCandidateUnboundedWithFilter(const Node &node,
                                                                               faiss::DistanceComputer &qdis,
                                                                               graph::FastBitMap &filterMask,
                                                                               uint32_t *scalaAttrOnCpu, int efS,
                                                                               faiss::VisitedTable *vt) const;

    // / Search interface
    void Search(faiss::DistanceComputer &qdis, int k, int64_t *labels, float *dists, faiss::VisitedTable &vt) const;

    void SearchWithoutUpperBeam(faiss::DistanceComputer &qdis, int k, int64_t *labels, float *dists,
                                faiss::VisitedTable &vt) const;

    void SearchWithUpperBeam(faiss::DistanceComputer &qdis, int k, int64_t *labels, float *dists,
                             faiss::VisitedTable &vt) const;

    void SearchFilter(faiss::DistanceComputer &qdis, int k, graph::FastBitMap &filterMask, uint32_t *scalaAttrOnCpu,
                      int64_t *labels, float *dists, faiss::VisitedTable &vt) const;

    void SearchFilterWithOutUpperBeam(faiss::DistanceComputer &qdis, int k, graph::FastBitMap &filterMask,
                                      uint32_t *scalaAttrOnCpu, int64_t *labels, float *dists,
                                      faiss::VisitedTable &vt) const;

    void SearchFilterWithUpperBeam(faiss::DistanceComputer &qdis, int k, graph::FastBitMap &filterMask,
                                   uint32_t *scalaAttrOnCpu, int64_t *labels, float *dists,
                                   faiss::VisitedTable &vt) const;

    void HybridSearch(faiss::DistanceComputer &fastComputer, faiss::DistanceComputer &preciseComputer,
                      int k, int64_t *I, float *D,
                      faiss::VisitedTable &vt) const;

    void HybridSearchWithMask(faiss::DistanceComputer &fastComputer, faiss::DistanceComputer &preciseComputer, int k,
                              uint8_t *mask,
                              size_t cpuIdx, size_t nbOffset, int64_t *I, float *D, faiss::VisitedTable &vt) const;

    void SearchSingleQueryParallel(faiss::DistanceComputer &qdis, int k, int64_t *labels, float *dists,
                                   std::bitset<50000000> &vt) const;

    int64_t ParallelExpandCandidate(faiss::DistanceComputer &qdis, const int64_t candId, const float &distBound,
                                    std::vector<Candidate> &setL, const int64_t localQueueStart,
                                    int64_t &localQueueSize, const int64_t &localQueueCapacity,
                                    std::bitset<50000000> &vt) const;

    int64_t ParallelAddIntoQueue(std::vector<Candidate> &queue, const int64_t start, int64_t &size,
                                 const int64_t capacity, const Candidate &cand) const;

    void HybridSearchWithoutUpperBeam(faiss::DistanceComputer &fastComputer, faiss::DistanceComputer &preciseComputer,
                                      int k, int64_t *I, float *D, faiss::VisitedTable &vt) const;

    void HybridSearchWithoutUpperBeamWithMask(faiss::DistanceComputer &fastComputer,
                                              faiss::DistanceComputer &preciseComputer, int k,
                                              uint8_t *mask, size_t cpuIdx, size_t nbOffset, int64_t *I, float *D,
                                              faiss::VisitedTable &vt) const;

    void HybridSearchFine(MinimaxHeap &candidatesResort, int ef, faiss::DistanceComputer &preciseComputer,
                          int k, int64_t *I, float *D) const;

    void Reset();

    int PrepareLevelClusters(int d, faiss::MetricType metricType, const float *x, size_t n, bool preset_levels = false);

    int PrepareLevelTab(size_t n, bool preset_levels = false);

    int PushCandidates(faiss::DistanceComputer &qdis, int k, int64_t *I, float *D, MinimaxHeap &candidates,
                       MinimaxHeap &candidatesResort, int nres_in, const int64_t *vertices, size_t vertex_count) const;

    static void ShrinkNeighborList(faiss::DistanceComputer &qdis, std::priority_queue<NodeDistFarther> &input,
                                   std::vector<NodeDistFarther> &output, int max_size, float threshold);
};

}  // namespace ascendsearch

#endif  // GFEATURERETRIEVAL_HNSWGraph_H