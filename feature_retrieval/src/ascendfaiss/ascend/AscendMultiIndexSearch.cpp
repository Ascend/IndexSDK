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


#include "AscendMultiIndexSearch.h"

#include <algorithm>
#include <unordered_set>

#include "ascend/utils/fp16.h"
#include "ascenddaemon/utils/AscendUtils.h"
#include "ascend/utils/MergeResUtils.h"
#include "ascend/impl/AscendIndexImpl.h"
#include "ascend/impl/AscendIndexInt8Impl.h"
#if defined(HOSTCPU) && defined(BUILD_IVFSP)
#include "ascend/ivfsp/AscendIndexIVFSP.h"
#include "ascend/ivfsp/AscendIndexIVFSPImpl.h"
#endif

namespace faiss {
namespace ascend {
namespace {
const int FILTER_SIZE = 6;
const size_t MAX_INDEX_COUNT = 10000;
const idx_t MAXN = 1024;
const idx_t MAXK = 1024;
}

static bool IsIndexTrained(const AscendIndex &index)
{
    return index.is_trained;
}

static bool IsIndexTrained(const AscendIndexInt8 &index)
{
    return index.isTrained();
}

template<typename IndexT, typename T>
static void CheckParamters(std::vector<IndexT *> indexes, idx_t n, const T *x, idx_t k,
    float *distances, idx_t *labels)
{
    FAISS_THROW_IF_NOT_FMT(indexes.size() > 0 && indexes.size() <= MAX_INDEX_COUNT,
                           "size of indexes (%zu) must be > 0 and <= %zu.", indexes.size(), MAX_INDEX_COUNT);
    FAISS_THROW_IF_NOT_FMT((n > 0) && (n <= MAXN), "n must be > 0 and <= %ld", MAXN);
    FAISS_THROW_IF_NOT_FMT((k > 0) && (k <= MAXK), "k must be > 0 and <= %ld", MAXK);
    FAISS_THROW_IF_NOT_MSG(x, "Invalid x: nullptr.");
    FAISS_THROW_IF_NOT_MSG(distances, "Invalid distances: nullptr.");
    FAISS_THROW_IF_NOT_MSG(labels, "Invalid labels: nullptr.");
    for (size_t i = 0; i < indexes.size(); ++i) {
        FAISS_THROW_IF_NOT_FMT(indexes[i], "Invalid index %zu from given indexes: nullptr.", i);
        FAISS_THROW_IF_NOT_FMT(IsIndexTrained(*(indexes[i])), "Index %zu not trained", i);
    }
}

void SearchPostProcess(std::vector<AscendIndex *> indexes, std::vector<std::vector<float>>& dist,
                       std::vector<std::vector<ascend_idx_t>>& label, int n, int k,
                       float* distances, idx_t* labels, bool merged)
{
    FAISS_THROW_IF_NOT_FMT(indexes.size() > 0, "size of indexes (%zu) must be > 0.", indexes.size());
    FAISS_THROW_IF_NOT_MSG(indexes[0] != nullptr, "indexes[0] is nullptr.");
    APP_LOG_INFO("AscendMultiIndexSearch SearchPostProcess operation started.\n");
    faiss::MetricType metricType = indexes[0]->metric_type;
    // 0. if merged is false, only merge device result
    if (!merged) {
        for (size_t i = 0; i < indexes.size(); ++i) {
            size_t deviceCnt = indexes[0]->getDeviceList().size();
            MergeDeviceResult(dist[i], label[i], n, k, distances + i * n * k, labels + i * n * k,
                deviceCnt, metricType);
        }
        return;
    }

    // 1. init data, to save device merge result
    size_t indexCnt = indexes.size();
    std::vector<std::vector<float>> distResult(indexCnt, std::vector<float>(n * k, 0));
    std::vector<std::vector<idx_t>> labelResult(indexCnt, std::vector<idx_t>(n * k, 0));

    // 2. merge device data
    for (size_t i = 0; i < indexes.size(); ++i) {
        size_t deviceCnt = indexes[0]->getDeviceList().size();
        MergeDeviceResult(dist[i], label[i], n, k, distResult[i].data(), labelResult[i].data(), deviceCnt, metricType);
    }

    // 3. merge index data
    MergeIndexResult(distResult, labelResult, n, k, distances, labels, indexCnt, metricType);
    APP_LOG_INFO("AscendMultiIndexSearch SearchPostProcess operation finished.\n");
}

void SearchPostProcess(std::vector<AscendIndexInt8 *> indexes, std::vector<std::vector<float>>& dist,
                       std::vector<std::vector<ascend_idx_t>>& label, int n, int k,
                       float* distances, idx_t* labels, bool merged)
{
    FAISS_THROW_IF_NOT_FMT(indexes.size() > 0, "size of indexes (%zu) must be > 0.", indexes.size());
    FAISS_THROW_IF_NOT_MSG(indexes[0] != nullptr, "indexes[0] is nullptr.");
    APP_LOG_INFO("AscendMultiIndexSearch SearchPostProcess operation started.\n");
    faiss::MetricType metricType = indexes[0]->getMetricType();
    // 0. if merged is false, only merge device result
    if (!merged) {
        for (size_t i = 0; i < indexes.size(); ++i) {
            size_t deviceCnt = indexes[0]->getDeviceList().size();
            MergeDeviceResult(dist[i], label[i], n, k, distances + i * n * k, labels + i * n * k,
                deviceCnt, metricType);
        }
        return;
    }

    // 1. init data, to save device merge result
    size_t indexCnt = indexes.size();
    std::vector<std::vector<float>> distResult(indexCnt, std::vector<float>(n * k, 0));
    std::vector<std::vector<idx_t>> labelResult(indexCnt, std::vector<idx_t>(n * k, 0));

    // 2. merge device data
    for (size_t i = 0; i < indexes.size(); ++i) {
        size_t deviceCnt = indexes[0]->getDeviceList().size();
        MergeDeviceResult(dist[i], label[i], n, k, distResult[i].data(), labelResult[i].data(), deviceCnt, metricType);
    }

    // 3. merge index data
    MergeIndexResult(distResult, labelResult, n, k, distances, labels, indexCnt, metricType);
    APP_LOG_INFO("AscendMultiIndexSearch SearchPostProcess operation finished.\n");
}
//  call actual index to do search
template<class P, class Q>
void MultiIndexSearch(IndexParam<Q, uint16_t, ascend_idx_t> param, std::vector<P *> indexes)
{
    int n = param.n;
    int dim = param.dim;
    int k = param.k;
    const Q *query = param.query;
    uint16_t *distance = param.distance;
    ascend_idx_t *label = param.label;
    using namespace ::ascend;
    AscendTensor<Q, DIMS_2> tensorDevQueries({ n, dim });
    auto ret = aclrtMemcpy(tensorDevQueries.data(), tensorDevQueries.getSizeInBytes(), query,
                           static_cast<size_t>(n) * static_cast<size_t>(dim) * sizeof(Q), ACL_MEMCPY_HOST_TO_DEVICE);
    FAISS_THROW_IF_NOT_FMT(ret == ACL_SUCCESS, "aclrtMemcpy error %d", ret);

    if (indexes.size() == 1) {
        ret = indexes[0]->search(n, tensorDevQueries.data(), k, distance,
                                 static_cast<::ascend::Index::idx_t *>(label));
    } else {
        ret = indexes[0]->search(indexes, n, tensorDevQueries.data(), k, distance,
                                 static_cast<::ascend::Index::idx_t *>(label));
    }
    FAISS_THROW_IF_NOT_FMT(ret == APP_ERR_OK, "Failed to MultiSearch index, result is %d", ret);
}
template<class P>
void MultiIndexSearchFilter(IndexParam<uint16_t, uint16_t, ascend_idx_t> param, uint32_t filtersSize,
                            std::vector<void *> &filters, std::vector<P *> indexes,
                            bool isMultiFilterSearch)
{
    int n = param.n;
    int dim = param.dim;
    int k = param.k;
    const uint16_t *query = param.query;
    uint16_t *distance = param.distance;
    ascend_idx_t *label = param.label;
    using namespace ::ascend;
    AscendTensor<uint16_t, DIMS_2> tensorDevQueries({ n, dim });
    auto ret = aclrtMemcpy(tensorDevQueries.data(), tensorDevQueries.getSizeInBytes(), query,
                           n * dim * sizeof(uint16_t), ACL_MEMCPY_HOST_TO_DEVICE);
    FAISS_THROW_IF_NOT_FMT(ret == ACL_SUCCESS, "aclrtMemcpy error %d", ret);

    if (indexes.size()  == 1) {
        ret = indexes[0]->searchFilter(n, tensorDevQueries.data(), k, distance,
                                       static_cast<::ascend::Index::idx_t *>(label),
                                       filtersSize, static_cast<uint32_t *>(filters[0]));
    } else {
        ret = indexes[0]->searchFilter(indexes, n, tensorDevQueries.data(), k, distance,
                                       static_cast<::ascend::Index::idx_t *>(label),
                                       filtersSize, filters, isMultiFilterSearch);
    }
    FAISS_THROW_IF_NOT_FMT(ret == APP_ERR_OK, "Failed to MultiSearch index,result is: %d\n", ret);
}
template<class T, class P>
void PrepareMultiSearchData(const std::vector<T *> &indexes,
                            std::unordered_map<int, std::vector<P *>> &deviceIndexMap,
                            std::unordered_map<int, std::vector<int>> &hostIndexIdMap,
                            std::vector<int> &deviceIds)
{
    for (size_t i = 0; i < indexes.size(); ++i) {
        IndexImplBase& indexImpli = indexes[i]->GetIndexImplBase();
        auto deviceList = indexes[i]->getDeviceList();
        for (const auto deviceId : deviceList) {
            deviceIndexMap[deviceId].push_back(static_cast<P *>(indexImpli.GetActualIndex(deviceId, false)));
            hostIndexIdMap[deviceId].push_back(i);
        }
    }

    for (auto iter = deviceIndexMap.begin(); iter != deviceIndexMap.end(); ++iter) {
        deviceIds.push_back(iter->first);
    }
}

// 调用者保证indexes不为空，且内部指针均非空
template<typename T, typename Q>
std::unordered_set<std::shared_ptr<std::shared_lock<std::shared_mutex>>> GetIndexesReadLock(std::vector<Q *> indexes)
{
    std::unordered_set<std::shared_ptr<std::shared_lock<std::shared_mutex>>> lockSet;
    if (!::ascend::AscendMultiThreadManager::IsMultiThreadMode()) {
        return lockSet;
    }

    // 加锁，避免indexes内部index相同但顺序不同时，该接口重入互相死锁的情况
    static std::mutex mtx;
    std::lock_guard<std::mutex> lock(mtx);
    for (size_t i = 0; i < indexes.size(); ++i) {
        const auto &index = static_cast<T &>(indexes[i]->GetIndexImplBase());
        lockSet.insert(index.getReadLock());
    }
    return lockSet;
}

void Search(std::vector<AscendIndex *> indexes, idx_t n, const float *x, idx_t k,
    float *distances, idx_t *labels, bool merged)
{
    APP_LOG_INFO("AscendMultiIndexSearch search operation started: n=%ld, k=%ld.\n", n, k);
    // 1. check the parameters
    CheckParamters(indexes, n, x, k, distances, labels);
    IndexImplBase& indexImpl0 = indexes[0]->GetIndexImplBase();
    // 2. check the param of others with indexes[0]
    for (size_t i = 0; i < indexes.size(); ++i) {
        indexImpl0.CheckIndexParams(indexes[i]->GetIndexImplBase());
    }
#if defined(HOSTCPU) && defined(BUILD_IVFSP)
    if (typeid(*indexes[0]) == typeid(faiss::ascend::AscendIndexIVFSP)) {
        auto deviceId = indexes.front()->getDeviceList().front();
        std::vector<faiss::ascendSearch::AscendIndex*> indexesIVFSPSQ;
        for (AscendIndex* index : indexes) {
            IndexImplBase& indexImpl = index->GetIndexImplBase();
            indexesIVFSPSQ.push_back(indexImpl.GetIVFSPSQPtr());
        }

        if (indexesIVFSPSQ.size() == 1) {
            dynamic_cast<faiss::ascend::AscendIndexIVFSP *>(indexes[0])->search(n, x, k, distances, labels);
        } else {
            AscendIndexIVFSPImpl::SearchMultiIndex(deviceId, indexesIVFSPSQ, n, x, k, distances, labels, merged);
        }
        APP_LOG_INFO("AscendMultiIndexSearch search operation finished.\n");
        return;
    }
#endif

    auto lockSet = GetIndexesReadLock<AscendIndexImpl>(indexes);

    // 3. convert query data from float to fp16, device use fp16 data to search
    std::vector<uint16_t> query(n * indexes[0]->d, 0);
    transform(x, x + n * indexes[0]->d, begin(query), [](float temp) { return fp16(temp).data; });

    // 4. prepare data
    std::unordered_map<int, std::vector<::ascend::Index *>> deviceIndexesMap; // deviceId --> device indexes
    std::unordered_map<int, std::vector<int>> hostIndexIdMap;           // deviceId --> host indexes
    std::vector<int> deviceIds;                                         // idx --> deviceId
    PrepareMultiSearchData<AscendIndex, ::ascend::Index>(indexes, deviceIndexesMap, hostIndexIdMap, deviceIds);

    size_t deviceCnt = deviceIndexesMap.size();
    size_t indexCnt = indexes.size();

    std::vector<std::vector<float>> dist(
        deviceCnt, std::vector<float>(indexCnt * static_cast<size_t>(n) * static_cast<size_t>(k), 0));
    std::vector<std::vector<uint16_t>> distHalf(
        deviceCnt, std::vector<uint16_t>(indexCnt * static_cast<size_t>(n) * static_cast<size_t>(k), 0));
    std::vector<std::vector<ascend_idx_t>> label(
        deviceCnt, std::vector<ascend_idx_t>(indexCnt * static_cast<size_t>(n) * static_cast<size_t>(k), 0));

    std::vector<std::vector<float>> distResult(indexCnt, std::vector<float>());
    std::vector<std::vector<ascend_idx_t>> labelResult(indexCnt, std::vector<ascend_idx_t>());

    auto searchFunctor = [&](int idx) {
        int deviceId = deviceIds[idx];
        auto deviceIndexes = deviceIndexesMap.at(deviceId);
        IndexParam<uint16_t, uint16_t, ascend_idx_t> param(deviceId, n, indexes[0]->d, k);
        param.query = query.data();
        param.distance = distHalf[idx].data();
        param.label = label[idx].data();
        MultiIndexSearch<::ascend::Index, uint16_t>(param, deviceIndexesMap[deviceId]);
        // convert result data from fp16 to float
        transform(begin(distHalf[idx]), end(distHalf[idx]), begin(dist[idx]),
            [](uint16_t temp) { return (float)fp16(temp); });
        // convert offset to label
        std::vector<ascend_idx_t> transLabel(deviceIndexes.size() * static_cast<size_t>(n) * static_cast<size_t>(k), 0);
        for (size_t i = 0; i < deviceIndexes.size(); ++i) {
            const auto id = hostIndexIdMap[deviceId][i];
            IndexImplBase& indexImpl = indexes[id]->GetIndexImplBase();
            std::transform(label[idx].data() + i * n * k, label[idx].data() + (i + 1) * n * k,
                transLabel.data() + i * n * k, [&](ascend_idx_t temp) {
                return (temp != std::numeric_limits<ascend_idx_t>::max()) ?
                       indexImpl.GetIdxFromDeviceMap(idx, temp) : std::numeric_limits<ascend_idx_t>::max();
            });
        }

        // merge result by index
        for (size_t i = 0; i < deviceIndexes.size(); ++i) {
            const auto id = hostIndexIdMap[deviceId][i];
            size_t begin = i * static_cast<size_t>(n) * static_cast<size_t>(k);
            size_t end = (i + 1) * static_cast<size_t>(n) * static_cast<size_t>(k);
            distResult[id].insert(distResult[id].end(), dist[idx].data() + begin, dist[idx].data() + end);
            labelResult[id].insert(labelResult[id].end(), transLabel.data() + begin, transLabel.data() + end);
        }
    };

    // 5. send rpc request
    CALL_PARALLEL_FUNCTOR(deviceCnt, indexImpl0.GetPool(), searchFunctor);

    // 6. post process: default is merge the topk results from devices and indexes
    SearchPostProcess(indexes, distResult, labelResult, n, k, distances, labels, merged);
    APP_LOG_INFO("AscendMultiIndexSearch search operation finished.\n");
}

// only suport AscendIndexSQ and AscendIndexIVFSP
void Search(std::vector<Index *> indexes, idx_t n, const float *x, idx_t k,
    float *distances, idx_t *labels, bool merged)
{
    APP_LOG_INFO("AscendMultiIndexSearch search operation started: n=%ld, k=%ld.\n", n, k);
    FAISS_THROW_IF_NOT_FMT(indexes.size() > 0 && indexes.size() <= MAX_INDEX_COUNT,
                           "size of indexes (%zu) must be > 0 and <= 10000.", indexes.size());
    std::vector<AscendIndex *> ascendIndex(indexes.size());
    transform(indexes.begin(), indexes.end(), ascendIndex.begin(), [](Index* index) {
        return dynamic_cast<AscendIndex *>(index);
    });

    Search(ascendIndex, n, x, k, distances, labels, merged);
    APP_LOG_INFO("AscendMultiIndexSearch search operation finished.\n");
}

void Search(std::vector<AscendIndexInt8 *> indexes, idx_t n, const int8_t *x, idx_t k,
    float *distances, idx_t *labels, bool merged)
{
    APP_LOG_INFO("AscendMultiIndexSearch search operation started: n=%ld, k=%ld.\n", n, k);
    // 1. check the parameters
    CheckParamters(indexes, n, x, k, distances, labels);
    IndexImplBase& indexImpl0 = indexes[0]->GetIndexImplBase();

    // 2. check the param of others with indexes[0]
    for (size_t i = 0; i < indexes.size(); ++i) {
        indexImpl0.CheckIndexParams(indexes[i]->GetIndexImplBase());
    }

    auto lockSet = GetIndexesReadLock<AscendIndexInt8Impl>(indexes);

    // 3. prepare data
    std::unordered_map<int, std::vector<::ascend::IndexInt8 *>> deviceIndexesMap;        // deviceId --> device index
    std::unordered_map<int, std::vector<int>> hostIndexIdMap;           // deviceId --> host indexes
    std::vector<int> deviceIds;
    PrepareMultiSearchData<AscendIndexInt8, ::ascend::IndexInt8>(indexes, deviceIndexesMap, hostIndexIdMap, deviceIds);

    size_t deviceCnt = deviceIndexesMap.size();
    size_t indexCnt = indexes.size();

    std::vector<std::vector<float>> dist(
        deviceCnt, std::vector<float>(indexCnt * static_cast<size_t>(n) * static_cast<size_t>(k), 0));
    std::vector<std::vector<uint16_t>> distHalf(
        deviceCnt, std::vector<uint16_t>(indexCnt * static_cast<size_t>(n) * static_cast<size_t>(k), 0));
    std::vector<std::vector<ascend_idx_t>> label(
        deviceCnt, std::vector<ascend_idx_t>(indexCnt * static_cast<size_t>(n) * static_cast<size_t>(k), 0));

    std::vector<std::vector<float>> distResult(indexCnt, std::vector<float>());
    std::vector<std::vector<ascend_idx_t>> labelResult(indexCnt, std::vector<ascend_idx_t>());

    auto searchFunctor = [&](int idx) {
        int deviceId = deviceIds[idx];
        auto deviceIndexes = deviceIndexesMap.at(deviceId);
        IndexParam<int8_t, uint16_t, ascend_idx_t> param(deviceId, n, indexes[0]->getDim(), k);
        param.query = x;
        param.distance = distHalf[idx].data();
        param.label = label[idx].data();
        MultiIndexSearch<::ascend::IndexInt8, int8_t>(param, deviceIndexesMap[deviceId]);
        if (indexes[0]->getMetricType() == faiss::METRIC_L2) {
            // convert result data from fp16 to float
            auto scaleFunctor = [](int dim) {
                // 0.01, 128, 4 is hyperparameter suitable for UB
                return 0.01 / std::min(dim / 64, std::max(dim / 128 + 1, 4));
            };
            float scale = scaleFunctor(indexes[0]->getDim());
            transform(begin(distHalf[idx]), end(distHalf[idx]), begin(dist[idx]),
                [scale](uint16_t temp) { return std::sqrt((float)fp16(temp) / scale); });
        } else {
            // convert result data from fp16 to float
            transform(begin(distHalf[idx]), end(distHalf[idx]), begin(dist[idx]),
                [](uint16_t temp) { return (float)fp16(temp); });
        }

        // convert offset to label
        std::vector<ascend_idx_t> transLabel(deviceIndexes.size() * static_cast<size_t>(n) * static_cast<size_t>(k), 0);
        for (size_t i = 0; i < deviceIndexes.size(); ++i) {
            const auto id = hostIndexIdMap[deviceId][i];
            IndexImplBase& indexImpl = indexes[id]->GetIndexImplBase();
            std::transform(label[idx].data() + i * n * k, label[idx].data() + (i + 1) * n * k,
                transLabel.data() + i * n * k, [&](ascend_idx_t temp) {
                return (temp != std::numeric_limits<ascend_idx_t>::max()) ?
                    indexImpl.GetIdxFromDeviceMap(idx, temp) : std::numeric_limits<ascend_idx_t>::max();
            });
        }

        // merge result by index
        for (size_t i = 0; i < deviceIndexes.size(); ++i) {
            const auto id = hostIndexIdMap[deviceId][i];
            size_t begin = i * static_cast<size_t>(n) * static_cast<size_t>(k);
            size_t end = (i + 1) * static_cast<size_t>(n) * static_cast<size_t>(k);
            distResult[id].insert(distResult[id].end(), dist[idx].data() + begin, dist[idx].data() + end);
            labelResult[id].insert(labelResult[id].end(), transLabel.data() + begin, transLabel.data() + end);
        }
    };

    // 5. send rpc request
    CALL_PARALLEL_FUNCTOR(deviceCnt, indexImpl0.GetPool(), searchFunctor);

    // 6. post process: default is merge the topk results from devices and indexes
    SearchPostProcess(indexes, distResult, labelResult, n, k, distances, labels, merged);
    APP_LOG_INFO("AscendMultiIndexSearch search operation finished.\n");
}

void MultiSearchWithFilter(std::vector<AscendIndex *> indexes, idx_t n, const float *x, idx_t k,
    float *distances, idx_t *labels, std::vector<void *> &filters, bool merged, bool isMultiFilterSearch)
{
    // convert query data from float to fp16, device use fp16 data to search
    std::vector<uint16_t> query(n * indexes[0]->d, 0);
    transform(x, x + n * indexes[0]->d, begin(query), [](float temp) { return fp16(temp).data; });

    // prepare data
    std::unordered_map<int, std::vector<::ascend::Index *>> deviceIndexMap;        // deviceId --> device indexIds
    std::unordered_map<int, std::vector<int>> hostIndexIdMap;           // deviceId --> host indexes
    std::vector<int> deviceIds;                                         // idx --> deviceId
    PrepareMultiSearchData<AscendIndex, ::ascend::Index>(indexes, deviceIndexMap, hostIndexIdMap, deviceIds);

    size_t deviceCnt = deviceIndexMap.size();
    size_t indexCnt = indexes.size();

    std::vector<std::vector<float>> dist(deviceCnt,
        std::vector<float>(indexCnt * static_cast<size_t>(n) * static_cast<size_t>(k), 0));
    std::vector<std::vector<uint16_t>> distHalf(deviceCnt,
        std::vector<uint16_t>(indexCnt * static_cast<size_t>(n) * static_cast<size_t>(k), 0));
    std::vector<std::vector<ascend_idx_t>> label(deviceCnt,
        std::vector<ascend_idx_t>(indexCnt * static_cast<size_t>(n) * static_cast<size_t>(k), 0));

    std::vector<std::vector<float>> distResult(indexCnt, std::vector<float>());
    std::vector<std::vector<ascend_idx_t>> labelResult(indexCnt, std::vector<ascend_idx_t>());

    auto searchFunctor = [&](int idx) {
        int deviceId = deviceIds[idx];
        auto indexIds = deviceIndexMap.at(deviceId);
        IndexParam<uint16_t, uint16_t, ascend_idx_t> param(deviceId, n, indexes[0]->d, k);
        param.query = query.data();
        param.distance = distHalf[idx].data();
        param.label = label[idx].data();
        MultiIndexSearchFilter<::ascend::Index>(param, n * FILTER_SIZE, filters, deviceIndexMap[deviceId],
                                                isMultiFilterSearch);
        // convert result data from fp16 to float
        transform(begin(distHalf[idx]), end(distHalf[idx]), begin(dist[idx]),
            [](uint16_t temp) { return (float)fp16(temp); });

        // convert offset to label
        std::vector<ascend_idx_t> transLabel(indexIds.size() * static_cast<size_t>(n) * static_cast<size_t>(k), 0);
        for (size_t i = 0; i < indexIds.size(); ++i) {
            const auto id = hostIndexIdMap[deviceId][i];
            IndexImplBase& indexImpl = indexes[id]->GetIndexImplBase();
            std::transform(label[idx].data() + i * n * k, label[idx].data() + (i + 1) * n * k,
                transLabel.data() + i * n * k, [&](ascend_idx_t temp) {
                    return (temp != std::numeric_limits<ascend_idx_t>::max()) ?
                    indexImpl.GetIdxFromDeviceMap(idx, temp) : std::numeric_limits<ascend_idx_t>::max();
            });
        }

        // merge result by index
        for (size_t i = 0; i < indexIds.size(); ++i) {
            const auto id = hostIndexIdMap[deviceId][i];
            auto begin = i * static_cast<size_t>(n) * static_cast<size_t>(k);
            auto end = (i + 1) * static_cast<size_t>(n) * static_cast<size_t>(k);
            distResult[id].insert(distResult[id].end(), dist[idx].data() + begin, dist[idx].data() + end);
            labelResult[id].insert(labelResult[id].end(), transLabel.data() + begin, transLabel.data() + end);
        }
    };

    // send rpc request
    CALL_PARALLEL_FUNCTOR(deviceCnt, indexes[0]->GetIndexImplBase().GetPool(), searchFunctor);

    // post process: default is merge the topk results from devices and indexes
    SearchPostProcess(indexes, distResult, labelResult, n, k, distances, labels, merged);
    APP_LOG_INFO("AscendMultiIndexSearch SearchWithFilter operation finished.\n");
}

void SearchWithFilter(std::vector<AscendIndex *> indexes, idx_t n, const float *x, idx_t k,
    float *distances, idx_t *labels, const void *filters, bool merged)
{
    APP_LOG_INFO("AscendMultiIndexSearch SearchWithFilter operation started: n=%ld, k=%ld.\n", n, k);
    FAISS_THROW_IF_NOT_MSG(filters != nullptr, "Invalid filters nullptr.");
    // 1. check the parameters
    CheckParamters(indexes, n, x, k, distances, labels);

    // 2. check the param of others with indexes[0]
    IndexImplBase& indexImpl0 = indexes[0]->GetIndexImplBase();
    for (size_t i = 0; i < indexes.size(); ++i) {
        indexImpl0.CheckIndexParams(indexes[i]->GetIndexImplBase(), true);
    }
#if defined(HOSTCPU) && defined(BUILD_IVFSP)
    if (typeid(*indexes[0]) == typeid(faiss::ascend::AscendIndexIVFSP)) {
        auto deviceId = indexes.front()->getDeviceList().front();
        std::vector<faiss::ascendSearch::AscendIndex*> indexesIVFSPSQ;
        for (AscendIndex* index: indexes) {
            IndexImplBase& indexImpl = index->GetIndexImplBase();
            indexesIVFSPSQ.push_back(indexImpl.GetIVFSPSQPtr());
        }
        if (indexesIVFSPSQ.size() == 1) {
            dynamic_cast<faiss::ascend::AscendIndexIVFSP *>(indexes[0])->search_with_filter(n, x, k,
                distances, labels, filters);
        } else {
            AscendIndexIVFSPImpl::SearchWithFilterMultiIndex(deviceId, indexesIVFSPSQ,
                n, x, k, distances, labels, filters, merged);
        }
        APP_LOG_INFO("AscendMultiIndexSearch search operation finished.\n");
        return;
    }
#endif

    auto lockSet = GetIndexesReadLock<AscendIndexImpl>(indexes);

    std::vector<void *> multiSearchFilters { const_cast<void*>(filters) };
    MultiSearchWithFilter(indexes, n, x, k, distances, labels, multiSearchFilters, merged, false);
    return;
}

// only suport AscendIndexSQ and AscendIndexIVFSP
void SearchWithFilter(std::vector<Index *> indexes, idx_t n, const float *x, idx_t k,
    float *distances, idx_t *labels, const void *filters, bool merged)
{
    APP_LOG_INFO("AscendMultiIndexSearch SearchWithFilter operation started: n=%ld, k=%ld.\n", n, k);
    FAISS_THROW_IF_NOT_FMT(indexes.size() > 0 && indexes.size() <= MAX_INDEX_COUNT,
                           "size of indexes (%zu) must be > 0 and <= 10000.", indexes.size());
    std::vector<AscendIndex *> ascendIndex(indexes.size());
    transform(indexes.begin(), indexes.end(), ascendIndex.begin(), [](Index* index) {
        return static_cast<AscendIndex *>(index);
    });

    SearchWithFilter(ascendIndex, n, x, k, distances, labels, filters, merged);
    APP_LOG_INFO("AscendMultiIndexSearch SearchWithFilter operation finished.\n");
}

void CheckFilters(idx_t n, void *filters[])
{
    FAISS_THROW_IF_NOT_MSG(filters != nullptr, "Invalid filters nullptr.");
    FAISS_THROW_IF_NOT_FMT((n > 0) && (n <= MAXN), "n must be > 0 and <= %ld", MAXN);
    for (idx_t i = 0; i < n; i++) {
        FAISS_THROW_IF_NOT_FMT(filters[i] != nullptr, "Invalid filters[%ld] nullptr.", i);
    }
}

void SearchWithFilter(std::vector<AscendIndex *> indexes, idx_t n, const float *x, idx_t k,
    float *distances, idx_t *labels, void *filters[], bool merged)
{
    APP_LOG_INFO("AscendMultiIndexSearch SearchWithFilter for different filters started: n=%ld, k=%ld.\n", n, k);
    // 1. check the parameters
    CheckParamters(indexes, n, x, k, distances, labels);
    CheckFilters(n, filters);
    IndexImplBase& indexImpl0 = indexes[0]->GetIndexImplBase();
    // 2. check the param of others with indexes[0]
    for (size_t i = 0; i < indexes.size(); ++i) {
        indexImpl0.CheckIndexParams(indexes[i]->GetIndexImplBase(), true);
    }
#if defined(HOSTCPU) && defined(BUILD_IVFSP)
    if (typeid(*indexes[0]) == typeid(faiss::ascend::AscendIndexIVFSP)) {
        auto deviceId = indexes.front()->getDeviceList().front();
        std::vector<faiss::ascendSearch::AscendIndex*> indexesIVFSPSQ;
        for (AscendIndex* index : indexes) {
            IndexImplBase& indexImpl = index->GetIndexImplBase();
            indexesIVFSPSQ.push_back(indexImpl.GetIVFSPSQPtr());
        }
        AscendIndexIVFSPImpl::SearchWithFilterMultiIndex(deviceId, indexesIVFSPSQ, n, x, k, distances, labels, filters,
            merged);
        APP_LOG_INFO("AscendMultiIndexSearch SearchWithFilter for different filters finished.\n");
        return;
    }
#endif

    auto lockSet = GetIndexesReadLock<AscendIndexImpl>(indexes);

    std::vector<void*> multiSearchFilters(filters, filters + n);
    MultiSearchWithFilter(indexes, n, x, k, distances, labels, multiSearchFilters, merged, true);
}

void SearchWithFilter(std::vector<Index *> indexes, idx_t n, const float *x, idx_t k,
    float *distances, idx_t *labels, void *filters[], bool merged)
{
    APP_LOG_INFO("AscendMultiIndexSearch SearchWithFilter for different filters started: n=%ld, k=%ld.\n", n, k);
    FAISS_THROW_IF_NOT_FMT(indexes.size() > 0 && indexes.size() <= MAX_INDEX_COUNT,
                           "size of indexes (%zu) must be > 0 and <= %zu.", indexes.size(), MAX_INDEX_COUNT);
    std::vector<AscendIndex *> ascendIndex(indexes.size());
    transform(indexes.begin(), indexes.end(), ascendIndex.begin(), [](Index* index) {
        return static_cast<AscendIndex *>(index);
    });

    SearchWithFilter(ascendIndex, n, x, k, distances, labels, filters, merged);
    APP_LOG_INFO("AscendMultiIndexSearch SearchWithFilter for different filters finished.\n");
}
}
}