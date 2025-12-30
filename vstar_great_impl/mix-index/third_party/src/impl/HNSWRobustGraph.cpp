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

#include "impl/HNSWRobustGraph.h"

#include <algorithm>
#include <memory>
#include <mutex>
#include <stack>

#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/Clustering.h>
#include "utils/AscendSearchAssert.h"
#include "utils/GraphDistanceComputer.h"
#include "securec.h"

#define VALUE_UNUSED(x) (void)(x)
namespace ascendsearch {
const int RANDOM_SEED = 12345;
const int EF_SEARCH_INITIVAL = 16;
const int M_INITIVAL = 2;
const int RANDOM_SEED2 = 456;

using Node = std::pair<float, int>;

/**************************************************************
 * HNSW structure implementation
 **************************************************************/
int HNSWGraph::GetNeighborCnt(int layer) const
{
    return cumNeighborCntPerLevel[layer + 1] - cumNeighborCntPerLevel[layer];
}

void HNSWGraph::SetNeighborCnt(int level_no, int n)
{
    ASCENDSEARCH_THROW_IF_NOT(levels.size() == 0);
    int cur_n = GetNeighborCnt(level_no);
    for (size_t i = level_no + 1; i < cumNeighborCntPerLevel.size(); i++) {
        cumNeighborCntPerLevel[i] += n - cur_n;
    }
}

int HNSWGraph::cum_nb_neighbors(int layer_no) const
{
    return cumNeighborCntPerLevel[layer_no];
}

/**
 * 返回当前节点ID在当前查询层级下所有邻居存储范围
 * @param nodeId 当前节点ID
 * @param layerId 当前查询层级
 * @param begin 返回值：当前节点ID在当前层级邻居节点存储位置的起始位置
 * @param end 返回值：当前节点ID在当前层级邻居节点存储位置的终止位置
 */
void HNSWGraph::GetNeighborPosRange(int64_t nodeId, int layerId, size_t &begin, size_t &end) const
{
    size_t startPos = offsets[nodeId];
    begin = startPos + cum_nb_neighbors(layerId);
    end = startPos + cum_nb_neighbors(layerId + 1);
}

HNSWGraph::HNSWGraph(int M) : mt(RANDOM_SEED)
{
    if (M <= 0) {
        ASCENDSEARCH_THROW_MSG("Invalid maxDegree, maxDegree should be bigger than 0.");
    }
    SetDefaultProbas(M, 1.0 / log(M));
    maxLevel = -1;
    entryPoint = -1;  // entry point
    efSearch = EF_SEARCH_INITIVAL;
    efConstruction = 40;  // Size of the dynamic candidate set. Generally, the value ranges from 200 to 400
    upperBeam = 1;
    offsets.push_back(0);
}

// assign highest level id for a node based on assign probas
int HNSWGraph::RandomLevel()
{
    double f = mt() / float(mt.max());
    // could be a bit faster with bissection
    for (size_t level = 0; level < assignProbas.size(); level++) {
        if (f < assignProbas[level]) {
            return level;
        }
        f -= assignProbas[level];
    }
    // happens with exponentially low probability
    return static_cast<int>(assignProbas.size()) - 1;
}

// Determine the probability function of each layer,
// the probability of the sample point belonging to each layer is determined,
// and from the bottom to the top is gradually decreasing!!!
void HNSWGraph::SetDefaultProbas(int M, float levelMult)
{
    int nn = 0;
    cumNeighborCntPerLevel.push_back(0);
    for (int level = 0;; level++) {
        float proba = exp(-level / levelMult) * (1 - exp(-1 / levelMult));
        if (proba < 1e-9)
            break;
        assignProbas.push_back(proba);
        nn += level == 0 ? M * M_INITIVAL : M;
        cumNeighborCntPerLevel.push_back(nn);
    }
}

void HNSWGraph::Reset()
{
    maxLevel = -1;
    entryPoint = -1;
    offsets.clear();
    offsets.push_back(0);
    levels.clear();
    neighbors.clear();
}

int HNSWGraph::PrepareLevelTab(size_t n, bool preset_levels)
{
    size_t n0 = offsets.size() - 1;

    if (preset_levels) {
        ASCENDSEARCH_ASSERT(n0 + n == levels.size());
    } else {
        ASCENDSEARCH_ASSERT(n0 == levels.size());
        for (size_t i = 0; i < n; i++) {
            int pt_level = RandomLevel();
            levels.push_back(pt_level + 1);
        }
    }

    // 由于每个节点在每一层的邻居数是固定的, 第0层为2M,其余层为M，neighbors数组存储的是某个节点在所有层（该节点进入层往下的所有层）的邻居
    int max_level = 0;
    for (size_t i = 0; i < n; i++) {
        int pt_level = levels[i + n0] - 1;
        if (pt_level > max_level) {
            max_level = pt_level;
        }
        offsets.push_back(offsets.back() + cum_nb_neighbors(pt_level + 1));
        neighbors.resize(offsets.back(), -1);
    }

    return max_level;
}

/* * Enumerate vertices from farthest to nearest from query, keep a
 * neighbor only if there is no previous neighbor that is closer to
 * that vertex than the query.
 */
void HNSWGraph::ShrinkNeighborList(faiss::DistanceComputer &qdis, std::priority_queue<NodeDistFarther> &input,
                                   std::vector<NodeDistFarther> &output, int max_size, float threshold)
{
    while (!input.empty()) {
        NodeDistFarther nodeDistFarther = input.top();
        input.pop();
        float dist_v1_q = nodeDistFarther.d;
        bool good = true;
        for (NodeDistFarther v2 : output) {
            float dist_v1_v2 = qdis.symmetric_dis(v2.id, nodeDistFarther.id);
            if (threshold * dist_v1_v2 < dist_v1_q) {
                good = false;
                break;
            }
        }

        if (good) {
            output.push_back(nodeDistFarther);
            if (output.size() >= (size_t)max_size) {
                return;
            }
        }
    }
}

namespace {
using NodeDistCloser = HNSWGraph::NodeDistCloser;
using NodeDistFarther = HNSWGraph::NodeDistFarther;

/**************************************************************
 * Addition subroutines
 **************************************************************/

// remove neighbors from the list to make it smaller than max_size
void ShrinkNeighborList(faiss::DistanceComputer &qdis, std::priority_queue<NodeDistCloser> &resultSet1, int maxSize,
                        float threshold)
{
    if (resultSet1.size() < (size_t)maxSize) {
        return;
    }
    std::priority_queue<NodeDistFarther> resultSet;
    std::vector<NodeDistFarther> returnlist;

    while (!resultSet1.empty()) {
        resultSet.emplace(resultSet1.top().d, resultSet1.top().id);
        resultSet1.pop();
    }

    HNSWGraph::ShrinkNeighborList(qdis, resultSet, returnlist, maxSize, threshold);

    for (NodeDistFarther curen2 : returnlist) {
        resultSet1.emplace(curen2.d, curen2.id);
    }
}

// add a link between two elements, possibly shrinking the list
// of links to make room for it.
void add_link(HNSWGraph &hnswGraph, faiss::DistanceComputer &qdis, int src, int dest, int level, float threshold)
{
    size_t begin;
    size_t end;
    hnswGraph.GetNeighborPosRange(src, level, begin, end);
    if (hnswGraph.neighbors[end - 1] == -1) {
        // there is enough room, find a slot to add it
        size_t i = end;
        while (i > begin) {
            if (hnswGraph.neighbors[i - 1] != -1)
                break;
            i--;
        }
        hnswGraph.neighbors[i] = dest;
        return;
    }

    // otherwise we let them fight out which to keep

    // copy to resultSet...
    std::priority_queue<NodeDistCloser> resultSet;
    resultSet.emplace(qdis.symmetric_dis(src, dest), dest);
    for (size_t i = begin; i < end; i++) {  // HERE WAS THE BUG
        int neigh = hnswGraph.neighbors[i];
        resultSet.emplace(qdis.symmetric_dis(src, neigh), neigh);
    }

    ShrinkNeighborList(qdis, resultSet, end - begin, threshold);

    // ...and back
    size_t i = begin;
    while (!resultSet.empty()) {
        hnswGraph.neighbors[i++] = resultSet.top().id;
        resultSet.pop();
    }
    // they may have shrunk more than just by 1 element
    while (i < end) {
        hnswGraph.neighbors[i++] = -1;
    }
}

// Search neighbors on a single level, starting from an entry point
void SearchNeighborsInLevelGraph(int level, HNSWGraph &hnsw, faiss::DistanceComputer &distHandler, int entryPoint,
                                 float distForEnterPoint, std::priority_queue<NodeDistCloser> &results,
                                 faiss::VisitedTable &vt)
{
    // top is nearest candidate
    std::priority_queue<NodeDistFarther> candidateQueue;
    candidateQueue.emplace(distForEnterPoint, entryPoint);
    results.emplace(distForEnterPoint, entryPoint);
    vt.set(entryPoint);

    while (!candidateQueue.empty()) {
        // get nearest
        const NodeDistFarther &currEnv = candidateQueue.top();

        if (currEnv.d > results.top().d) {
            break;
        }
        int currNode = currEnv.id;
        candidateQueue.pop();

        // loop over neighbors
        size_t begin;
        size_t end;
        hnsw.GetNeighborPosRange(currNode, level, begin, end);
        for (size_t i = begin; i < end; i++) {
            int nodeId = hnsw.neighbors[i];
            if (nodeId < 0) {
                break;
            }

            if (vt.get(nodeId)) {
                continue;
            }

            vt.set(nodeId);

            float tmpDist = distHandler(nodeId);
            if (results.size() < (size_t)hnsw.efConstruction || results.top().d > tmpDist) {
                results.emplace(tmpDist, nodeId);
                candidateQueue.emplace(tmpDist, nodeId);
            }
            if (results.size() > (size_t)hnsw.efConstruction) {
                results.pop();
            }
        }
    }
    vt.advance();
}

/**************************************************************
 * Searching subroutines
 **************************************************************/

// greedily update a nearest vector at a given level
void greedy_update_nearest(const HNSWGraph &hnsw, faiss::DistanceComputer &qdis, int level, int &nearest,
                           float &dNearest)
{
    int cnt = 0;
    for (;;) {
        int prev_nearest = nearest;

        size_t begin;
        size_t end;
        hnsw.GetNeighborPosRange(nearest, level, begin, end);
        for (size_t i = begin; i < end; i++) {
            int v = hnsw.neighbors[i];
            if (v < 0)
                break;
            float dis = qdis(v);
            cnt++;
            if (dis < dNearest) {
                nearest = v;
                dNearest = dis;
            }
        }
        if (nearest == prev_nearest) {
            return;
        }
    }
}
}  // namespace

// Finds neighbors and builds links with them, starting from an entry
// point. The own neighbor list is assumed to be locked.
void HNSWGraph::AddLinksInLevelGraph(int level, faiss::DistanceComputer &ptdis, int pt_id, int nearest, float d_nearest,
                                     omp_lock_t *locks, faiss::VisitedTable &vt)
{
    // a priority queue to maintain the nodes from farther to closer, top is farther, capacity is efConstruction
    std::priority_queue<NodeDistCloser> linkCandidates;

    SearchNeighborsInLevelGraph(level, *this, ptdis, nearest, d_nearest, linkCandidates, vt);

    // but we can afford only this many neighbors
    int M = GetNeighborCnt(level);

    ::ascendsearch::ShrinkNeighborList(ptdis, linkCandidates, M, thresholdEdge);

    while (!linkCandidates.empty()) {
        int other_id = linkCandidates.top().id;
        // 添加了双向边
        omp_set_lock(&locks[other_id]);
        add_link(*this, ptdis, other_id, pt_id, level, thresholdEdge);
        omp_unset_lock(&locks[other_id]);

        add_link(*this, ptdis, pt_id, other_id, level, thresholdEdge);

        linkCandidates.pop();
    }
}

/**************************************************************
 * Building, parallel, should be thread safety
 **************************************************************/
void HNSWGraph::AddWithLocks(faiss::DistanceComputer &distComputer, int nodeLevel, int nodeId,
                             std::vector<omp_lock_t> &locks, faiss::VisitedTable &vt)
{
    int nearest = entryPoint;
#pragma omp critical
    {
        if (nearest == -1) {
            maxLevel = nodeLevel;
            entryPoint = nodeId;
        }
    }

    if (nearest < 0) {
        return;
    }

    omp_set_lock(&locks[nodeId]);
    float d_nearest = distComputer(nearest);

    // find the nearest node at each currLevel, from highest currLevel to lowest currLevel
    int currLevel = maxLevel;  // currLevel at which we start adding neighbors
    for (; currLevel > nodeLevel; currLevel--) {
        greedy_update_nearest(*this, distComputer, currLevel, nearest, d_nearest);
    }

    for (; currLevel >= 0; currLevel--) {
        AddLinksInLevelGraph(currLevel, distComputer, nodeId, nearest, d_nearest, locks.data(), vt);
    }

    omp_unset_lock(&locks[nodeId]);

    if (nodeLevel > maxLevel) {
        maxLevel = nodeLevel;
        entryPoint = nodeId;
    }
}

/* * Do a BFS on the candidates list */
int HNSWGraph::SearchFromCandidates(faiss::DistanceComputer &qdis, int k, int64_t *I, float *D, MinimaxHeap &candidates,
                                    faiss::VisitedTable &vt, int level, MinimaxHeap &candidatesResort,
                                    int nres_in) const
{
    int nres = nres_in;

    for (int i = 0; i < candidates.Size(); i++) {
        int64_t v1 = candidates.ids[i];
        float d = candidates.dis[i];
        candidatesResort.Push(v1, d);
        ASCENDSEARCH_ASSERT(v1 >= 0);
        if (nres < k) {
            faiss::heap_push<maxCmp>(++nres, D, I, d, v1);
        } else if (d < D[0]) {
            faiss::heap_replace_top<maxCmp>(nres, D, I, d, v1);
        }
        vt.set(v1);
    }

    bool do_dis_check = check_relative_distance;
    int nstep = 0;
    while (candidates.Size() > 0) {
        float d0;
        int v0 = candidates.PopMin(d0);

        if (do_dis_check) {
            // tricky stopping condition: there are more that ef
            // distances that are processed already that are smaller
            // than d0
            int n_dis_below = candidates.CountBelow(d0);
            if (n_dis_below >= efSearch)
                break;
        }

        size_t begin;
        size_t end;
        GetNeighborPosRange(v0, level, begin, end);

        std::vector<int64_t> vertices(DISTANCE_BATCH_SIZE);
        size_t vertex_count = 0;

        for (size_t j = begin; j < end; j++) {
            int v1 = neighbors[j];
            if (v1 < 0)
                break;

            if (vt.get(v1))
                continue;

            vt.set(v1);

            vertices[vertex_count++] = v1;

            if (vertex_count == DISTANCE_BATCH_SIZE) {
                nres = PushCandidates(qdis, k, I, D, candidates, candidatesResort, nres, vertices.data(), vertex_count);
                vertex_count = 0;
            }
        }

        if (vertex_count != 0) {
            nres = PushCandidates(qdis, k, I, D, candidates, candidatesResort, nres, vertices.data(), vertex_count);
        }

        nstep++;
        if (!do_dis_check && nstep > efSearch) {
            break;
        }
        // #endif
    }
    return nres;
}

int HNSWGraph::PushCandidates(faiss::DistanceComputer &qdis, int k, int64_t *I, float *D, MinimaxHeap &candidates,
                              MinimaxHeap &candidatesResort, int nres_in, const int64_t *vertices,
                              size_t vertex_count) const
{
    int nres = nres_in;

    std::vector<float> dists(DISTANCE_BATCH_SIZE);

    for (size_t i = 0; i < vertex_count; ++i) {
        dists[i] = qdis(vertices[i]);
    }

    for (size_t i = 0; i < vertex_count; ++i) {
        if (nres < k)
            faiss::heap_push<maxCmp>(++nres, D, I, dists[i], vertices[i]);
        else if (dists[i] < D[0])
            faiss::heap_replace_top<maxCmp>(nres, D, I, dists[i], vertices[i]);
        candidates.Push(vertices[i], dists[i]);
        candidatesResort.Push(vertices[i], dists[i]);
    }

    return nres;
}

int HNSWGraph::SearchFromCandidatesWithStore(faiss::DistanceComputer &qdis, int k, int64_t *I, float *D,
                                             MinimaxHeap &candidates, faiss::VisitedTable &vt, int level,
                                             MinimaxHeap &candidatesResort, std::unordered_map<int, float> &map,
                                             int nres_in) const
{
    int nres = nres_in;
    int ndis = 0;

    for (int i = 0; i < candidates.Size(); i++) {
        int64_t v1 = candidates.ids[i];
        float d = candidates.dis[i];
        candidatesResort.Push(v1, d);
        ASCENDSEARCH_ASSERT(v1 >= 0);
        if (nres < k) {
            faiss::heap_push<maxCmp>(++nres, D, I, d, v1);
        } else if (d < D[0]) {
            faiss::heap_replace_top<maxCmp>(nres, D, I, d, v1);
        }
        vt.set(v1);
    }

    bool do_dis_check = check_relative_distance;
    int nstep = 0;

    while (candidates.Size() > 0) {
        float d0;
        int v0 = candidates.PopMin(d0);

        if (do_dis_check) {
            // tricky stopping condition: there are more that ef
            // distances that are processed already that are smaller
            // than d0
            int n_dis_below = candidates.CountBelow(d0);
            if (n_dis_below >= efSearch)
                break;
        }

        size_t begin;
        size_t end;
        GetNeighborPosRange(v0, level, begin, end);

        for (size_t j = begin; j < end; j++) {
            int v1 = neighbors[j];
            if (v1 < 0)
                break;

            if (vt.get(v1))
                continue;

            vt.set(v1);
            ndis++;
            float d = 0;
            if (map.find(v1) != map.end())
                d = map.at(v1);
            else
                d = qdis(v1);

            if (nres < k)
                faiss::heap_push<maxCmp>(++nres, D, I, d, (int64_t)v1);
            else if (d < D[0])
                faiss::heap_replace_top<maxCmp>(nres, D, I, d, (int64_t)v1);
            candidates.Push(v1, d);
            candidatesResort.Push(v1, d);
        }

        nstep++;
        if (!do_dis_check && nstep > efSearch) {
            break;
        }
    }
    return nres;
}

int HNSWGraph::SearchFromCandidatesWithFilter(faiss::DistanceComputer &qdis, int k, graph::FastBitMap &filterMask,
                                              uint32_t *scalaAttrOnCpu, int64_t *I, float *D, MinimaxHeap &candidates,
                                              faiss::VisitedTable &vt, int level, MinimaxHeap &candidatesResort,
                                              int nres_in) const
{
    int nres = nres_in;
    int ndis = 0;

    for (int i = 0; i < candidates.Size(); i++) {
        int64_t v1 = candidates.ids[i];
        float d = candidates.dis[i];
        candidatesResort.Push(v1, d);  // push the id distance combination in the maxheap
        ASCENDSEARCH_ASSERT(v1 >= 0);
        bool maskFlag = filterMask.at(scalaAttrOnCpu[v1]);  // scalaAttrOnCpu[v1] gives the correponding bit position of
                                                            // v1?
        if (nres < k && maskFlag) {
            // push the neighbor into the distance (D) and index(label) matrix (I)
            faiss::heap_push<maxCmp>(++nres, D, I, d, v1);
        } else if (d < D[0] && maskFlag) {
            faiss::heap_replace_top<maxCmp>(nres, D, I, d, v1);
        }
        vt.set(v1);
    }

    bool do_dis_check = check_relative_distance;
    int nstep = 0;

    while (candidates.Size() > 0) {
        float d0;
        int v0 = candidates.PopMin(d0);  // v0 is the id of the candidate vector with the minimum distance; that
                                         // distance is stored in d0

        if (do_dis_check) {
            // tricky stopping condition: there are more that ef
            // distances that are processed already that are smaller
            // than d0
            int n_dis_below = candidates.CountBelow(d0);  // count how many distances are smaller than d0
            if (n_dis_below >= efSearch)
                break;
        }

        size_t begin;
        size_t end;
        GetNeighborPosRange(v0, level, begin, end);  // begin and end now stores the starting and ending address of the
                                                     // neighbors of v0 in the given level

        for (size_t j = begin; j < end; j++) {
            int v1 = neighbors[j];  // neighbor of v0
            if (v1 < 0)
                break;

            if (vt.get(v1))
                continue;  // if v1 is a neighbor of v0 that is already visited, then we ignore it

            vt.set(v1);  // indicate v1 has been visited
            ndis++;
            float d = qdis(v1);  // distance of v1 to v0
            bool maskFlag = filterMask.at(scalaAttrOnCpu[v1]);
            if (nres < k && maskFlag)
                faiss::heap_push<maxCmp>(++nres, D, I, d, (int64_t)v1);
            else if (d < D[0] && maskFlag)
                faiss::heap_replace_top<maxCmp>(nres, D, I, d, (int64_t)v1);
            candidates.Push(v1, d);
            candidatesResort.Push(v1, d);
        }

        nstep++;  // count how many candidate vector has been processed
        if (!do_dis_check && nstep > efSearch) {
            break;
        }
    }

    return nres;
}

int HNSWGraph::SearchFromCandidatesWithMask(faiss::DistanceComputer &qdis, int k, uint8_t *mask, size_t cpuIdx,
                                            size_t nbOffset, int64_t *I, float *D, MinimaxHeap &candidates,
                                            faiss::VisitedTable &vt, int level, MinimaxHeap &candidatesResort,
                                            int nres_in) const
{
    int nres = nres_in;
    int ndis = 0;

    for (int i = 0; i < candidates.Size(); i++) {
        int64_t v1 = candidates.ids[i];
        float d = candidates.dis[i];

        int64_t v1_onMask = v1 + cpuIdx * nbOffset;  // shift v1 accordingly by cpuIdx so that we get the correct
                                                     // masking (1 = keep, 0 = mask)
        bool maskFlag = (mask[v1_onMask / 8] >> (v1_onMask & 7)) & 1;  // shift the number to right (v1_onMask % 8) bits

        if (maskFlag)
            candidatesResort.Push(v1, d);  // push the id distance combination in the maxheap
        ASCENDSEARCH_ASSERT(v1 >= 0);

        if (nres < k && maskFlag) {
            // push the neighbor into the distance (D) and index(label) matrix (I)
            faiss::heap_push<maxCmp>(++nres, D, I, d, v1);
        } else if (d < D[0] && maskFlag) {
            faiss::heap_replace_top<maxCmp>(nres, D, I, d, v1);
        }
        vt.set(v1);
    }

    bool do_dis_check = check_relative_distance;
    int nstep = 0;

    while (candidates.Size() > 0) {
        float d0;
        int v0 = candidates.PopMin(d0);  // v0 is the id of the candidate vector with the minimum distance

        if (do_dis_check) {
            // tricky stopping condition: there are more that ef
            // distances that are processed already that are smaller
            // than d0
            int n_dis_below = candidates.CountBelow(d0);
            if (n_dis_below >= efSearch)
                break;
        }

        size_t begin;
        size_t end;
        GetNeighborPosRange(v0, level, begin, end);  // begin and end now stores the starting and ending address of the
                                                     // neighbors of v0 in the given level

        for (size_t j = begin; j < end; j++) {
            int v1 = neighbors[j];  // neighbor of v0

            if (v1 < 0)
                break;

            if (vt.get(v1))
                continue;  // if v1 is a neighbor of v0 that is already visited, then we ignore it

            vt.set(v1);  // indicate v1 has been visited
            ndis++;
            float d = qdis(v1);  // distance of v1 to v0

            int64_t v1_onMask = v1 + cpuIdx * nbOffset;  // shift v1 accordingly by cpuIdx so that we get the correct
                                                         // masking (1 = keep, 0 = mask)
            bool maskFlag = (mask[v1_onMask / 8] >> (v1_onMask & 7)) & 1;  // shift the number to right (v1_onMask % 8)
                                                                           // bits

            // why adjusting the order of D and I if we will do the final sorting in HybridSearchFine?
            if (nres < k && maskFlag)
                faiss::heap_push<maxCmp>(++nres, D, I, d, (int64_t)v1);
            else if (d < D[0] && maskFlag)
                faiss::heap_replace_top<maxCmp>(nres, D, I, d, (int64_t)v1);

            // different from the original code in that if the
            if (maskFlag) {
                candidates.Push(v1, d);
                candidatesResort.Push(v1, d);
            }
        }

        nstep++;
        if (!do_dis_check && nstep > efSearch) {
            break;
        }
    }

    return nres;
}

/**************************************************************
 * Searching
 **************************************************************/
std::priority_queue<HNSWGraph::Node> HNSWGraph::search_from_candidate_unbounded(const Node &node,
                                                                                faiss::DistanceComputer &qdis, int ef,
                                                                                faiss::VisitedTable *vt) const
{
    int ndis = 0;
    std::priority_queue<Node> top_candidates;                                      // from large to small
    std::priority_queue<Node, std::vector<Node>, std::greater<Node> > candidates;  // from small to large

    top_candidates.push(node);
    candidates.push(node);

    vt->set(node.second);

    while (!candidates.empty()) {
        float d0;
        int v0;
        std::tie(d0, v0) = candidates.top();

        if (top_candidates.top().first < d0) {
            break;
        }

        candidates.pop();

        size_t begin;
        size_t end;
        GetNeighborPosRange(v0, 0, begin, end);

        for (size_t j = begin; j < end; ++j) {
            int nodeId = neighbors[j];

            if (nodeId < 0) {
                break;
            }
            if (vt->get(nodeId)) {
                continue;
            }

            vt->set(nodeId);

            float d1 = qdis(nodeId);
            ++ndis;

            if (top_candidates.top().first > d1 || top_candidates.size() < (size_t)ef) {
                candidates.emplace(d1, nodeId);
                top_candidates.emplace(d1, nodeId);
            }
            if (top_candidates.size() > (size_t)ef) {
                top_candidates.pop();
            }
        }
    }

    return top_candidates;
}

void UpdateCandidates(int efS, float dist, int nodeId, graph::FastBitMap &filterMask, uint32_t *scalaAttrOnCpu,
                      std::priority_queue<Node> &topCandidates,
                      std::priority_queue<Node, std::vector<Node>, std::greater<Node> > &candidates)
{
    if (topCandidates.size() < (size_t)efS || topCandidates.top().first > dist) {
        candidates.emplace(dist, nodeId);
        bool maskFlag = filterMask.at(scalaAttrOnCpu[nodeId]);
        if (maskFlag)
            topCandidates.emplace(dist, nodeId);

        if (topCandidates.size() > (size_t)efS) {
            topCandidates.pop();
        }
    }
}

std::priority_queue<HNSWGraph::Node> HNSWGraph::SearcFromCandidateUnboundedWithFilter(const Node &node,
                                                                                      faiss::DistanceComputer &qdis,
                                                                                      graph::FastBitMap &filterMask,
                                                                                      uint32_t *scalaAttrOnCpu, int efS,
                                                                                      faiss::VisitedTable *vt) const
{
    std::priority_queue<Node> topCandidates;                                       // from large to small
    std::priority_queue<Node, std::vector<Node>, std::greater<Node> > candidates;  // from small to large

    bool maskFlag = filterMask.at(scalaAttrOnCpu[node.second]);
    if (maskFlag)
        topCandidates.push(node);
    candidates.push(node);

    vt->set(node.second);

    while (!candidates.empty()) {
        float d0;
        int v0;
        std::tie(d0, v0) = candidates.top();

        if (!topCandidates.empty() && d0 > topCandidates.top().first) {
            break;
        }

        candidates.pop();

        size_t begin;
        size_t end;
        GetNeighborPosRange(v0, 0, begin, end);

        for (size_t j = begin; j < end; ++j) {
            int node = neighbors[j];

            if (node < 0) {
                break;
            }
            if (vt->get(node)) {
                continue;
            }

            vt->set(node);

            float d1 = qdis(node);
            UpdateCandidates(efS, d1, node, filterMask, scalaAttrOnCpu, topCandidates, candidates);
        }
    }

    return topCandidates;
}

void HNSWGraph::SearchWithoutUpperBeam(faiss::DistanceComputer &qdis, int k, int64_t *labels, float *dists,
                                       faiss::VisitedTable &vt) const
{
    //  greedy Search on upper levels
    int nearest = entryPoint;
    float d_nearest = qdis(nearest);

    for (int level = maxLevel; level >= 1; level--) {
        greedy_update_nearest(*this, qdis, level, nearest, d_nearest);
    }
    int ef = std::max(efSearch, k);
    if (search_bounded_queue) {
        MinimaxHeap candidates(ef);
        MinimaxHeap candidatesResort(ef);
        candidates.Push(nearest, d_nearest);
        SearchFromCandidates(qdis, k, labels, dists, candidates, vt, 0, candidatesResort);
    } else {
        std::priority_queue<Node> top_candidates = search_from_candidate_unbounded(Node(d_nearest, nearest), qdis, ef,
                                                                                   &vt);
        while (top_candidates.size() > (size_t)k)
            top_candidates.pop();

        int nres = 0;
        while (!top_candidates.empty()) {
            float d;
            int label;
            std::tie(d, label) = top_candidates.top();
            faiss::heap_push<maxCmp>(++nres, dists, labels, d, (int64_t)label);
            top_candidates.pop();
        }
    }

    vt.advance();
}

void HNSWGraph::SearchWithUpperBeam(faiss::DistanceComputer &qdis, int k, int64_t *labels, float *dists,
                                    faiss::VisitedTable &vt) const
{
    int candidates_size = upperBeam;
    MinimaxHeap candidates(candidates_size);
    MinimaxHeap candidatesResort(candidates_size);

    std::vector<int64_t> I_to_next(candidates_size);
    std::vector<float> D_to_next(candidates_size);

    int nres = 1;
    I_to_next[0] = entryPoint;
    D_to_next[0] = qdis(entryPoint);

    for (int level = maxLevel; level >= 0; level--) {
        candidates.Clear();

        for (int i = 0; i < nres; i++) {
            candidates.Push(I_to_next[i], D_to_next[i]);
        }

        if (level == 0) {
            nres = SearchFromCandidates(qdis, k, labels, dists, candidates, vt, 0,
                                        candidatesResort);  // 只在图的最底层进行search
                                                            // TopK
        } else {
            nres = SearchFromCandidates(qdis, candidates_size, I_to_next.data(), D_to_next.data(), candidates, vt,
                                        level, candidatesResort);
        }
        vt.advance();
    }
}

void HNSWGraph::Search(faiss::DistanceComputer &qdis, int k, int64_t *labels, float *dists,
                       faiss::VisitedTable &vt) const
{
    if (upperBeam == 1) {
        SearchWithoutUpperBeam(qdis, k, labels, dists, vt);
    } else {
        SearchWithUpperBeam(qdis, k, labels, dists, vt);
    }
}

void HNSWGraph::SearchFilterWithOutUpperBeam(faiss::DistanceComputer &qdis, int k, graph::FastBitMap &filterMask,
                                             uint32_t *scalaAttrOnCpu, int64_t *labels, float *dists,
                                             faiss::VisitedTable &vt) const
{
    //  greedy Search on upper levels
    int nearest = entryPoint;
    float d_nearest = qdis(nearest);

    for (int level = maxLevel; level >= 1; level--) {
        greedy_update_nearest(*this, qdis, level, nearest, d_nearest);
    }

    int ef = std::max(efSearch, k);
    if (search_bounded_queue) {
        MinimaxHeap candidates(ef);
        MinimaxHeap candidatesResort(ef);
        candidates.Push(nearest, d_nearest);
        SearchFromCandidatesWithFilter(qdis, k, filterMask, scalaAttrOnCpu, labels, dists, candidates, vt, 0,
                                       candidatesResort);
    } else {
        std::priority_queue<Node> top_candidates =
            SearcFromCandidateUnboundedWithFilter(Node(d_nearest, nearest), qdis, filterMask, scalaAttrOnCpu, ef, &vt);
        while (top_candidates.size() > (size_t)k)
            top_candidates.pop();

        int nres = 0;
        while (!top_candidates.empty()) {
            float d;
            int label;
            std::tie(d, label) = top_candidates.top();
            faiss::heap_push<maxCmp>(++nres, dists, labels, d, (int64_t)label);
            top_candidates.pop();
        }
    }
    vt.advance();
}

void HNSWGraph::SearchFilterWithUpperBeam(faiss::DistanceComputer &qdis, int k, graph::FastBitMap &filterMask,
                                          uint32_t *scalaAttrOnCpu, int64_t *labels, float *dists,
                                          faiss::VisitedTable &vt) const
{
    int candidates_size = upperBeam;
    MinimaxHeap candidates(candidates_size);
    MinimaxHeap candidatesResort(candidates_size);

    std::vector<int64_t> I_to_next(candidates_size);
    std::vector<float> D_to_next(candidates_size);

    int nres = 1;
    I_to_next[0] = entryPoint;
    D_to_next[0] = qdis(entryPoint);

    for (int level = maxLevel; level >= 0; level--) {
        candidates.Clear();

        for (int i = 0; i < nres; i++)
            candidates.Push(I_to_next[i], D_to_next[i]);

        if (level == 0) {
            nres = SearchFromCandidatesWithFilter(qdis, k, filterMask, scalaAttrOnCpu, labels, dists, candidates, vt, 0,
                                                  candidatesResort);
        } else {
            nres = SearchFromCandidates(qdis, candidates_size, I_to_next.data(), D_to_next.data(), candidates, vt,
                                        level, candidatesResort);
        }
        vt.advance();
    }
}

void HNSWGraph::SearchFilter(faiss::DistanceComputer &qdis, int k, graph::FastBitMap &filterMask,
                             uint32_t *scalaAttrOnCpu, int64_t *labels, float *dists, faiss::VisitedTable &vt) const
{
    if (upperBeam == 1) {
        SearchFilterWithOutUpperBeam(qdis, k, filterMask, scalaAttrOnCpu, labels, dists, vt);
    } else {
        SearchFilterWithUpperBeam(qdis, k, filterMask, scalaAttrOnCpu, labels, dists, vt);
    }
}

void HNSWGraph::HybridSearchFine(MinimaxHeap &candidatesResort, int ef, faiss::DistanceComputer &preciseComputer, int k,
                                 int64_t *I, float *D) const
{
    std::vector<int64_t> d(ef);
    float dis;
    // high precision sort
    int sz = candidatesResort.Size();
    for (int i = 0; i < sz; i++) {
        d[i] = candidatesResort.ids[i];
    }

    for (int i = 0; i < sz; i++) {
        dis = preciseComputer(d[i]);  // re-compute the distance of the given base vector and push them back into
                                      // candidateResort
        candidatesResort.Push(d[i], dis);
    }
    // rm duplicate res
    std::unordered_set<long int> hash_map;
    int countK = 0;
    while (countK < k && candidatesResort.Size() != 0) {
        float d0;
        long int v0 = candidatesResort.PopMin(d0);  // v0 is the vector with the smallest distance to the query vector
        if (hash_map.count(v0)) {
            continue;
        }
        hash_map.insert(v0);
        I[countK] = v0;
        D[countK] = d0;
        ++countK;
    }

    if (countK < k) {  // if candidatresResort run out before we found our top k base vectors, we wrap around?
        I[countK] = I[0];
        D[countK] = D[0];
        ++countK;
    }
}

void HNSWGraph::HybridSearchWithoutUpperBeam(faiss::DistanceComputer &fastComputer,
                                             faiss::DistanceComputer &preciseComputer, int k,
                                             int64_t *I, float *D, faiss::VisitedTable &vt) const
{
    int nearest = entryPoint;
    // can it be calculated by fastComputer
    // does it necessary to find the nearest from the top, I mean can it be calculate from the bottom level.
    float d_nearest = preciseComputer(nearest);
    for (int level = maxLevel; level >= 1; level--) {
        greedy_update_nearest(*this, preciseComputer, level, nearest, d_nearest);
    }
    int ef = std::max(efSearch, k);
    if (search_bounded_queue) {
        MinimaxHeap candidates(ef);
        MinimaxHeap candidatesResort(ef);
        candidates.Push(nearest, d_nearest);
        // default branch
        SearchFromCandidates(fastComputer, k, I, D, candidates, vt, 0, candidatesResort);
        HybridSearchFine(candidatesResort, ef, preciseComputer, k, I, D);
    } else {
        std::priority_queue<Node> top_candidates = search_from_candidate_unbounded(Node(d_nearest, nearest),
                                                                                   fastComputer, ef, &vt);
        while (top_candidates.size() > (size_t)k) {
            top_candidates.pop();
        }
        int nres = 0;
        while (!top_candidates.empty()) {
            float d;
            int label;
            std::tie(d, label) = top_candidates.top();
            faiss::heap_push<maxCmp>(++nres, D, I, d, (int64_t)label);
            top_candidates.pop();
        }
    }
    vt.advance();
}

void HNSWGraph::HybridSearchWithoutUpperBeamWithMask(faiss::DistanceComputer &fastComputer,
                                                     faiss::DistanceComputer &preciseComputer,
                                                     int k, uint8_t *mask, size_t cpuIdx, size_t nbOffset, int64_t *I,
                                                     float *D, faiss::VisitedTable &vt) const
{
    int nearest = entryPoint;
    // can it be calculated by fastComputer
    // does it necessary to find the nearest from the top, I mean can it be calculate from the bottom level.
    float d_nearest = preciseComputer(nearest);
    for (int level = maxLevel; level >= 1; level--) {
        greedy_update_nearest(*this, preciseComputer, level, nearest, d_nearest);
    }

    int ef = std::max(efSearch, k);
    if (search_bounded_queue) {
        MinimaxHeap candidates(ef);
        MinimaxHeap candidatesResort(ef);
        candidates.Push(nearest, d_nearest);  // push the entry point (id + distance) with its nearest neighbors (id +
                                              // distance)
        // default branch
        SearchFromCandidatesWithMask(fastComputer, k, mask, cpuIdx, nbOffset, I, D, candidates, vt, 0,
                                     candidatesResort);
        HybridSearchFine(candidatesResort, ef, preciseComputer, k, I, D);
    } else {
        std::priority_queue<Node> top_candidates = search_from_candidate_unbounded(Node(d_nearest, nearest),
                                                                                   fastComputer, ef, &vt);
        while (top_candidates.size() > (size_t)k) {
            top_candidates.pop();
        }
        int nres = 0;
        while (!top_candidates.empty()) {
            float d;
            int label;
            std::tie(d, label) = top_candidates.top();
            faiss::heap_push<maxCmp>(++nres, D, I, d, (int64_t)label);
            top_candidates.pop();
        }
    }
    vt.advance();
}

int64_t HNSWGraph::ParallelAddIntoQueue(std::vector<Candidate> &queue, const int64_t start, int64_t &size,
                                        const int64_t capacity, const Candidate &cand) const
{
    if (size == 0) {
        queue[start + size++] = cand;
        return 0;
    }

    int64_t end = start + size;
    auto it = std::lower_bound(queue.begin() + start, queue.begin() + end, cand);
    int64_t insertLoc = it - queue.begin();
    if (insertLoc == end) {
        if (cand.id == it->id) {
            return capacity;
        }
        if (size >= capacity) {
            --size;
            --end;
        }
    } else {
        if (size < capacity) {
            queue[insertLoc] = cand;
            ++size;
            return size - 1;
        } else {
            return capacity;
        }
    }

    if (end - insertLoc != 0) {
        auto ret = memmove_s(queue.data() + insertLoc + 1, (end - insertLoc) * sizeof(Candidate),
                             queue.data() + insertLoc, (end - insertLoc) * sizeof(Candidate));
        ASCENDSEARCH_THROW_IF_NOT_FMT(ret == 0, "memove error in HNSWGraph::ParallelAddIntoQueue, error is %d",
                                      static_cast<int>(ret));
    }
    queue[insertLoc] = cand;
    ++size;
    return insertLoc - start;
}

int64_t HNSWGraph::ParallelExpandCandidate(faiss::DistanceComputer &qdis, const int64_t candId, const float &distBound,
                                           std::vector<Candidate> &setL, const int64_t localQueueStart,
                                           int64_t &localQueueSize, const int64_t &localQueueCapacity,
                                           std::bitset<50000000> &vt) const
{
    size_t begin = 0;
    size_t end = 0;
    int64_t nk = localQueueCapacity;

    GetNeighborPosRange(candId, 0, begin, end);
    for (size_t i = begin; i < end; i++) {
        int64_t vid = neighbors[i];
        if (vt[vid])
            continue;

        vt[vid] = true;
        float dist = qdis(vid);
        if (dist > distBound)
            continue;

        Candidate cand(vid, dist, false);

        int64_t r = ParallelAddIntoQueue(setL, localQueueStart, localQueueSize, localQueueCapacity, cand);
        if (r < nk)
            nk = r;
    }

    return nk;
}

void HNSWGraph::SearchSingleQueryParallel(faiss::DistanceComputer &qdis, int, int64_t *, float *,
                                          std::bitset<50000000> &vt) const
{
    const std::vector<int64_t> slaveQueuesStarts;
    std::vector<int64_t> slaveQueuesSize;
    const std::vector<int64_t> initIds;
    std::vector<Candidate> setL;
    const int64_t L = 100;

    const int numThread = omp_get_max_threads();

    const int64_t masterQueueStart = slaveQueuesStarts[numThread - 1];
    int64_t &masterQueueSize = slaveQueuesSize[numThread - 1];

    // init params
    for (int64_t i = 0; i < L; i++) {
        vt[initIds[i]] = true;
    }
#pragma omp parallel for
    for (int i = 0; i < L; i++) {
        int vid = initIds[i];
        float dist = qdis(vid);
        setL[masterQueueStart + i] = Candidate(vid, dist, false);
    }
    std::sort(setL.begin() + masterQueueStart, setL.begin() + masterQueueStart + L);

    const float &lastDist = setL[masterQueueStart + L - 1].dist;
    {
        int64_t firstUncheckCandIdx = 0;
        {
            int64_t r;
            auto &cand = setL[masterQueueStart + firstUncheckCandIdx];
            cand.isChecked = true;
            int64_t id = cand.id;
            r = ParallelExpandCandidate(qdis, id, lastDist, setL, masterQueueStart, masterQueueSize, L, vt);
            if (r <= firstUncheckCandIdx) {
                firstUncheckCandIdx = r;
            } else {
                ++firstUncheckCandIdx;
            }
        }
    }
}

void HNSWGraph::HybridSearch(faiss::DistanceComputer &fastComputer, faiss::DistanceComputer &preciseComputer,
                             int k, int64_t *I, float *D, faiss::VisitedTable &vt) const
{
    if (upperBeam == 1) {
        //  greedy Search on upper levels
        HybridSearchWithoutUpperBeam(fastComputer, preciseComputer, k, I, D, vt);
    } else {
        SearchWithUpperBeam(fastComputer, k, I, D, vt);
    }
}

void HNSWGraph::HybridSearchWithMask(faiss::DistanceComputer &fastComputer, faiss::DistanceComputer &preciseComputer,
                                     int k, uint8_t *mask, size_t cpuIdx, size_t nbOffset, int64_t *I, float *D,
                                     faiss::VisitedTable &vt) const
{
    if (upperBeam == 1) {
        //  greedy Search on upper levels
        HybridSearchWithoutUpperBeamWithMask(fastComputer, preciseComputer, k, mask, cpuIdx, nbOffset, I, D, vt);
    } else {
        printf("Search with mask is now only supported without upper beam!\n");
        exit(1);
    }
}

void HNSWGraph::MinimaxHeap::Push(int i, float v)
{
    if (k == n && v >= dis[0]) {
        return;
    } else if (k == n) {
        if (ids[0] != -1) {
            --nvalid;
        }
        faiss::heap_pop<cmp>(k--, dis.data(), ids.data());
    }
    faiss::heap_push<cmp>(++k, dis.data(), ids.data(), v, i);
    ++nvalid;
}

float HNSWGraph::MinimaxHeap::Max() const
{
    return dis[0];
}

int HNSWGraph::MinimaxHeap::Size() const
{
    return nvalid;
}

void HNSWGraph::MinimaxHeap::Clear()
{
    nvalid = k = 0;
}

int HNSWGraph::MinimaxHeap::PopMin(float &minValue)
{
    ASSERT(k > 0);
    int a = k - 1;
    while (a >= 0) {
        if (ids[a] != -1)
            break;
        a--;
    }
    if (a == -1)
        return -1;
    int imin = a;
    float vmin = dis[a];
    a--;
    while (a >= 0) {
        if (ids[a] != -1 && dis[a] < vmin) {
            vmin = dis[a];
            imin = a;
        }
        a--;
    }
    minValue = vmin;
    int ret = ids[imin];
    ids[imin] = -1;
    --nvalid;

    return ret;
}

int HNSWGraph::MinimaxHeap::CountBelow(float threshold)
{
    int n_below = 0;
    for (int i = 0; i < k; i++) {
        if (dis[i] < threshold) {
            n_below++;
        }
    }

    return n_below;
}

}  // namespace ascendsearch