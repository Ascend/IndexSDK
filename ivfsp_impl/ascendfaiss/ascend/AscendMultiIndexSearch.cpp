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
#include <map>
#include <sys/time.h>
#include "ascendsearch/ascend/utils/fp16.h"
#include "ascendsearch/ascend/utils/AscendUtils.h"
#include "ascendsearch/ascend/utils/MergeResUtils.h"
#include "ascendsearch/ascend/AscendIndex.h"
#include "ascendsearch/ascend/impl/AscendIndexImpl.h"
#include "ascendsearch/ascend/rpc/AscendRpc.h"
#include "common/threadpool/AscendThreadPool.h"

namespace faiss {
namespace ascendSearch {
namespace {
const int FILTER_SIZE = 6;
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
            size_t deviceCnt = indexes[0]->impl_->getDeviceList().size();
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
        size_t deviceCnt = indexes[0]->impl_->getDeviceList().size();
        MergeDeviceResult(dist[i], label[i], n, k, distResult[i].data(), labelResult[i].data(), deviceCnt, metricType);
    }

    // 3. merge index data
    MergeIndexResult(distResult, labelResult, n, k, distances, labels, indexCnt, metricType);
    APP_LOG_INFO("AscendMultiIndexSearch SearchPostProcess operation finished.\n");
}


void Search(std::vector<AscendIndex *> indexes, idx_t n, const float *x, idx_t k,
    float *distances, idx_t *labels, bool merged)
{
    APP_LOG_INFO("AscendMultiIndexSearch search operation started: n=%ld, k=%ld.\n", n, k);
    // 1. check the size of indexes
    FAISS_THROW_IF_NOT_FMT(indexes.size() > 0, "size of indexes (%zu) must be > 0.", indexes.size());
    FAISS_THROW_IF_NOT_FMT((n > 0) && (n < MAX_N), "n must be > 0 and < %ld", MAX_N);
    FAISS_THROW_IF_NOT_FMT((k > 0) && (k <= MAX_K), "k must be > 0 and <= %ld", MAX_K);
    FAISS_THROW_IF_NOT_MSG(x, "Invalid x: nullptr.");
    FAISS_THROW_IF_NOT_MSG(distances, "Invalid distances: nullptr.");
    FAISS_THROW_IF_NOT_MSG(labels, "Invalid labels: nullptr.");
    for (size_t i = 0; i < indexes.size(); ++i) {
        FAISS_THROW_IF_NOT_FMT(indexes[i], "Invalid index %ld from given indexes: nullptr.", i);
        FAISS_THROW_IF_NOT_FMT(indexes[i]->is_trained, "Index %ld not trained", i);
    }
    for (const auto idxPtr : indexes) {
        FAISS_THROW_IF_NOT_MSG(idxPtr, "Invalid index from given indexes: nullptr.");
    }

    // 2. check the param of others with indexes[0] is same
    for (size_t i = 0; i < indexes.size(); ++i) {
        indexes[0]->impl_->checkParamsSame(*(indexes[i]->impl_));
    }

    // 3. convert query data from float to fp16, device use fp16 data to search
    std::vector<uint16_t> query(n * indexes[0]->d, 0);
    transform(x, x + n * indexes[0]->d, begin(query), [](float temp) { return fp16(temp).data; });

    // 4. prepare data
    std::unordered_map<int, rpcContext> ctxMap;                         // deviceId --> context
    std::unordered_map<int, std::vector<int>> deviceIndexIdsMap;        // deviceId --> device indexIds
    std::unordered_map<int, std::vector<int>> hostIndexIdMap;           // deviceId --> host indexes
    for (size_t i = 0; i < indexes.size(); ++i) {
        auto deviceList = indexes[i]->getDeviceList();
        for (const auto deviceId : deviceList) {
            rpcContext ctx = indexes[i]->impl_->contextMap.at(deviceId);
            ctxMap[deviceId] = ctx;
            deviceIndexIdsMap[deviceId].push_back(indexes[i]->impl_->indexMap[ctx]);
            hostIndexIdMap[deviceId].push_back(i);
        }
    }

    std::vector<int> deviceIds;                                         // idx --> deviceId
    for (auto iter = deviceIndexIdsMap.begin(); iter != deviceIndexIdsMap.end(); ++iter) {
        deviceIds.push_back(iter->first);
    }

    size_t deviceCnt = deviceIndexIdsMap.size();
    size_t indexCnt = indexes.size();

    std::vector<std::vector<float>> dist(deviceCnt, std::vector<float>(indexCnt * (size_t)n * (size_t)k, 0));
    std::vector<std::vector<uint16_t>> distHalf(
        deviceCnt, std::vector<uint16_t>(indexCnt * (size_t)n * (size_t)k, 0));
    std::vector<std::vector<ascend_idx_t>> label(
        deviceCnt, std::vector<ascend_idx_t>(indexCnt * (size_t)n * (size_t)k, 0));

    std::vector<std::vector<float>> distResult(indexCnt, std::vector<float>());
    std::vector<std::vector<ascend_idx_t>> labelResult(indexCnt, std::vector<ascend_idx_t>());

    auto searchFunctor = [&](int idx) {
        int deviceId = deviceIds[idx];
        auto ctx = ctxMap.at(deviceId);
        auto indexIds = deviceIndexIdsMap.at(deviceId);
        RpcError ret = RpcMultiIndexSearch(ctx, n, indexes[0]->d, k, query.data(), distHalf[idx].data(),
            label[idx].data(), deviceIndexIdsMap[deviceId]);
        FAISS_THROW_IF_NOT_FMT(ret == RPC_ERROR_NONE, "Multi Index earch implement failed(%d).", ret);

        // convert result data from fp16 to float
        transform(begin(distHalf[idx]), end(distHalf[idx]), begin(dist[idx]),
            [](uint16_t temp) { return (float)fp16(temp); });

        // merge result by index
        for (size_t i = 0; i < indexIds.size(); ++i) {
            const auto id = hostIndexIdMap[deviceId][i];
            size_t begin = i * (size_t)n * (size_t)k;
            size_t end = (i + 1) * (size_t)n * (size_t)k;
            distResult[id].insert(distResult[id].end(), dist[idx].data() + begin, dist[idx].data() + end);
            labelResult[id].insert(labelResult[id].end(), label[idx].data() + begin, label[idx].data() + end);
        }
    };

    // 5. send rpc request
    CALL_PARALLEL_FUNCTOR(deviceCnt, indexes[0]->impl_->pool, searchFunctor);

    // 6. post process: default is merge the topk results from devices and indexes
    SearchPostProcess(indexes, distResult, labelResult, n, k, distances, labels, merged);
    APP_LOG_INFO("AscendMultiIndexSearch search operation finished.\n");
}


void SearchWithFilter(std::vector<AscendIndex *> indexes, idx_t n, const float *x, idx_t k,
    float *distances, idx_t *labels, const void *filters, bool merged)
{
    APP_LOG_INFO("AscendMultiIndexSearch SearchWithFilter operation started: n=%ld, k=%ld.\n", n, k);
    // 1. check the size of indexes
    FAISS_THROW_IF_NOT_FMT(indexes.size() > 0, "size of indexes (%zu) must be > 0.", indexes.size());
    FAISS_THROW_IF_NOT_FMT((n > 0) && (n < MAX_N), "n must be > 0 and < %ld", MAX_N);
    FAISS_THROW_IF_NOT_FMT((k > 0) && (k <= MAX_K), "k must be > 0 and <= %ld", MAX_K);
    FAISS_THROW_IF_NOT_MSG(x, "Invalid x: nullptr.");
    FAISS_THROW_IF_NOT_MSG(distances, "Invalid distances: nullptr.");
    FAISS_THROW_IF_NOT_MSG(labels, "Invalid labels: nullptr.");
    for (size_t i = 0; i < indexes.size(); ++i) {
        FAISS_THROW_IF_NOT_FMT(indexes[i], "Invalid index %ld from given indexes: nullptr.", i);
        FAISS_THROW_IF_NOT_FMT(indexes[i]->is_trained, "Index %ld not trained", i);
    }
    for (const auto idxPtr : indexes) {
        FAISS_THROW_IF_NOT_MSG(idxPtr, "Invalid index from given indexes: nullptr.");
    }
 
    // 2. check the param of others with indexes[0] is same
    for (size_t i = 0; i < indexes.size(); ++i) {
        indexes[0]->impl_->checkParamsSame(*(indexes[i]->impl_));
    }
 
    // 3. convert query data from float to fp16, device use fp16 data to search
    std::vector<uint16_t> query(n * indexes[0]->d, 0);
    transform(x, x + n * indexes[0]->d, begin(query), [](float temp) { return fp16(temp).data; });
 
    // 4. prepare data
    std::unordered_map<int, rpcContext> ctxMap;                         // deviceId --> context
    std::unordered_map<int, std::vector<int>> deviceIndexIdsMap;        // deviceId --> device indexIds
    std::unordered_map<int, std::vector<int>> hostIndexIdMap;           // deviceId --> host indexes
    for (size_t i = 0; i < indexes.size(); ++i) {
        auto deviceList = indexes[i]->impl_->getDeviceList();
        for (const auto deviceId : deviceList) {
            rpcContext ctx = indexes[i]->impl_->contextMap.at(deviceId);
            ctxMap[deviceId] = ctx;
            deviceIndexIdsMap[deviceId].push_back(indexes[i]->impl_->indexMap[ctx]);
            hostIndexIdMap[deviceId].push_back(i);
        }
    }
 
    std::vector<int> deviceIds;                                         // idx --> deviceId
    for (auto iter = deviceIndexIdsMap.begin(); iter != deviceIndexIdsMap.end(); ++iter) {
        deviceIds.push_back(iter->first);
    }
 
    size_t deviceCnt = deviceIndexIdsMap.size();
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
        auto ctx = ctxMap.at(deviceId);
        auto indexIds = deviceIndexIdsMap.at(deviceId);
        RpcError ret = RpcMultiIndexSearchFilter(ctx, n, indexes[0]->d, k, query.data(), distHalf[idx].data(),
            label[idx].data(), n * FILTER_SIZE, static_cast<const uint32_t *>(filters), deviceIndexIdsMap[deviceId]);
        FAISS_THROW_IF_NOT_FMT(ret == RPC_ERROR_NONE, "Multi Index earch implement failed(%d).", ret);
 
        // convert result data from fp16 to float
        transform(begin(distHalf[idx]), end(distHalf[idx]), begin(dist[idx]),
            [](uint16_t temp) { return (float)fp16(temp); });
 
        // merge result by index
        for (size_t i = 0; i < indexIds.size(); ++i) {
            const auto id = hostIndexIdMap[deviceId][i];
            auto begin = i * static_cast<size_t>(n) * static_cast<size_t>(k);
            auto end = (i + 1) * static_cast<size_t>(n) * static_cast<size_t>(k);
            distResult[id].insert(distResult[id].end(), dist[idx].data() + begin, dist[idx].data() + end);
            labelResult[id].insert(labelResult[id].end(), label[idx].data() + begin, label[idx].data() + end);
        }
    };
 
    // 5. send rpc request
    CALL_PARALLEL_FUNCTOR(deviceCnt, indexes[0]->impl_->pool, searchFunctor);
 
    // 6. post process: default is merge the topk results from devices and indexes
    SearchPostProcess(indexes, distResult, labelResult, n, k, distances, labels, merged);
    APP_LOG_INFO("AscendMultiIndexSearch SearchWithFilter operation finished.\n");
}

void SearchWithFilter(std::vector<AscendIndex *> indexes, idx_t n, const float *x, idx_t k,
    float *distances, idx_t *labels, void *filters[], bool merged)
{
    APP_LOG_INFO("AscendMultiIndexSearch SearchWithFilter operation started: n=%ld, k=%ld.\n", n, k);
    // 1. check the size of indexes
    FAISS_THROW_IF_NOT_FMT(indexes.size() > 0, "size of indexes (%zu) must be > 0.", indexes.size());
    FAISS_THROW_IF_NOT_FMT((n > 0) && (n < MAX_N), "n must be > 0 and < %ld", MAX_N);
    FAISS_THROW_IF_NOT_FMT((k > 0) && (k <= MAX_K), "k must be > 0 and <= %ld", MAX_K);
    FAISS_THROW_IF_NOT_MSG(x, "Invalid x: nullptr.");
    FAISS_THROW_IF_NOT_MSG(distances, "Invalid distances: nullptr.");
    FAISS_THROW_IF_NOT_MSG(labels, "Invalid labels: nullptr.");
    for (size_t i = 0; i < indexes.size(); ++i) {
        FAISS_THROW_IF_NOT_FMT(indexes[i], "Invalid index %ld from given indexes: nullptr.", i);
        FAISS_THROW_IF_NOT_FMT(indexes[i]->is_trained, "Index %ld not trained", i);
    }
    for (const auto idxPtr : indexes) {
        FAISS_THROW_IF_NOT_MSG(idxPtr, "Invalid index from given indexes: nullptr.");
    }

    // 2. check the param of others with indexes[0] is same
    for (size_t i = 0; i < indexes.size(); ++i) {
        indexes[0]->impl_->checkParamsSame(*(indexes[i]->impl_));
    }

    // 3. convert query data from float to fp16, device use fp16 data to search
    std::vector<uint16_t> query(n * indexes[0]->d, 0);
    transform(x, x + n * indexes[0]->d, begin(query), [](float temp) { return fp16(temp).data; });

    // 4. prepare data
    std::unordered_map<int, rpcContext> ctxMap;                         // deviceId --> context
    std::unordered_map<int, std::vector<int>> deviceIndexIdsMap;        // deviceId --> device indexIds
    std::unordered_map<int, std::vector<int>> hostIndexIdMap;           // deviceId --> host indexes
    for (size_t i = 0; i < indexes.size(); ++i) {
        auto deviceList = indexes[i]->impl_->getDeviceList();
        for (const auto deviceId : deviceList) {
            rpcContext ctx = indexes[i]->impl_->contextMap.at(deviceId);
            ctxMap[deviceId] = ctx;
            deviceIndexIdsMap[deviceId].push_back(indexes[i]->impl_->indexMap[ctx]);
            hostIndexIdMap[deviceId].push_back(i);
        }
    }

    std::vector<int> deviceIds;                                         // idx --> deviceId
    for (auto iter = deviceIndexIdsMap.begin(); iter != deviceIndexIdsMap.end(); ++iter) {
        deviceIds.push_back(iter->first);
    }

    size_t deviceCnt = deviceIndexIdsMap.size();
    size_t indexCnt = indexes.size();

    std::vector<std::vector<float>> dist(deviceCnt,
        std::vector<float>(indexCnt * static_cast<size_t>(n) * static_cast<size_t>(k), 0));
    std::vector<std::vector<uint16_t>> distHalf(deviceCnt,
        std::vector<uint16_t>(indexCnt * static_cast<size_t>(n) * static_cast<size_t>(k), 0));
    std::vector<std::vector<ascend_idx_t>> label(deviceCnt,
        std::vector<ascend_idx_t>(indexCnt * static_cast<size_t>(n) * static_cast<size_t>(k), 0));

    std::vector<std::vector<float>> distResult(indexCnt, std::vector<float>());
    std::vector<std::vector<ascend_idx_t>> labelResult(indexCnt, std::vector<ascend_idx_t>());
    std::vector<const uint32_t*> filtersUint32t(n);
    for (int i = 0; i < n; i++) {
        filtersUint32t[i] = static_cast<const uint32_t *>(*(filters+i));
    }
    auto searchFunctor = [&](int idx) {
        int deviceId = deviceIds[idx];
        auto ctx = ctxMap.at(deviceId);
        auto indexIds = deviceIndexIdsMap.at(deviceId);
        RpcError ret = RpcMultiIndexSearchFilter(ctx, n, indexes[0]->d, k, query.data(), distHalf[idx].data(),
            label[idx].data(), n * FILTER_SIZE*indexCnt, filtersUint32t.data(), deviceIndexIdsMap[deviceId]);
        FAISS_THROW_IF_NOT_FMT(ret == RPC_ERROR_NONE, "Multi Index earch implement failed(%d).", ret);

        // convert result data from fp16 to float
        transform(begin(distHalf[idx]), end(distHalf[idx]), begin(dist[idx]),
            [](uint16_t temp) { return (float)fp16(temp); });

        // merge result by index
        for (size_t i = 0; i < indexIds.size(); ++i) {
            const auto id = hostIndexIdMap[deviceId][i];
            auto begin = i * static_cast<size_t>(n) * static_cast<size_t>(k);
            auto end = (i + 1) * static_cast<size_t>(n) * static_cast<size_t>(k);
            distResult[id].insert(distResult[id].end(), dist[idx].data() + begin, dist[idx].data() + end);
            labelResult[id].insert(labelResult[id].end(), label[idx].data() + begin, label[idx].data() + end);
        }
    };

    // 5. send rpc request
    CALL_PARALLEL_FUNCTOR(deviceCnt, indexes[0]->impl_->pool, searchFunctor);

    // 6. post process: default is merge the topk results from devices and indexes
    SearchPostProcess(indexes, distResult, labelResult, n, k, distances, labels, merged);
    APP_LOG_INFO("AscendMultiIndexSearch SearchWithFilter operation finished.\n");
}

void SearchWithFilter(std::vector<Index *> indexes, idx_t n, const float *x, idx_t k,
    float *distances, idx_t *labels, const void *filters, bool merged)
{
    APP_LOG_INFO("AscendMultiIndexSearch SearchWithFilter operation started: n=%ld, k=%ld.\n", n, k);
    std::vector<AscendIndex *> sqIndexes(indexes.size());
    transform(indexes.begin(), indexes.end(), sqIndexes.begin(), [](Index* index) {
        return static_cast<AscendIndex *>(index);
    });

    SearchWithFilter(sqIndexes, n, x, k, distances, labels, filters, merged);
    APP_LOG_INFO("AscendMultiIndexSearch SearchWithFilter operation finished.\n");
}
}
}
