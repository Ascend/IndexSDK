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


#include "AscendIndexInt8Impl.h"

#include <algorithm>
#include <string>
#include <set>

#include "common/utils/CommonUtils.h"
#include "ascend/utils/fp16.h"

namespace faiss {
namespace ascend {
namespace {
const size_t KB = 1024;
const size_t RETAIN_SIZE = 2048;
const size_t UNIT_PAGE_SIZE = 64;
const size_t UNIT_VEC_SIZE = 512;
const size_t DEVICE_LIST_SIZE_MAX = 64;

// Default size for which we page add or search
const size_t ADD_PAGE_SIZE = UNIT_PAGE_SIZE * KB * KB - RETAIN_SIZE;

// Or, maximum number of vectors to consider per page of add
const size_t ADD_VEC_SIZE = UNIT_VEC_SIZE * KB;

// search pagesize must be less than 64M, becauseof rpc limitation
const size_t SEARCH_PAGE_SIZE = UNIT_PAGE_SIZE * KB * KB - RETAIN_SIZE;

// Or, maximum number 512K of vectors to consider per page of search
const size_t SEARCH_VEC_SIZE = UNIT_VEC_SIZE * KB;

const int64_t INT8_INDEX_MAX_MEM = 0x400000000;
}

AscendIndexInt8Impl::AscendIndexInt8Impl(int dims, faiss::MetricType metric,
    AscendIndexInt8Config config, AscendIndexInt8 *intf)
    : indexConfig(config), dim(dims), metricType(metric)
{
    APP_LOG_INFO("AscendIndexInt8 construction start");
    FAISS_THROW_IF_NOT_MSG(intf, "Intf is nullptr");

    // dim belong to [64, 1024]
    FAISS_THROW_IF_NOT_MSG(dims >= 64, "Invalid number of dimensions, only support dim >= 64");
    // dim belong to [64, 1024]
    FAISS_THROW_IF_NOT_MSG(dims <= 1024, "Invalid number of dimensions, only support dim <= 1024");
    // multiples of 64
    FAISS_THROW_IF_NOT_MSG(dims % 64 == 0, "Invalid number of dimensions, only support dim mod 64 = 0");

    FAISS_THROW_IF_NOT_FMT(indexConfig.deviceList.size() > 0 && indexConfig.deviceList.size() <= DEVICE_LIST_SIZE_MAX,
                           "device list should be in range (0, %zu]!", DEVICE_LIST_SIZE_MAX);
    FAISS_THROW_IF_NOT_MSG(indexConfig.resourceSize == -1 ||
                           (indexConfig.resourceSize >= 0 && indexConfig.resourceSize <= INT8_INDEX_MAX_MEM),
                           "resourceSize should be -1 or in range [0, 16GB]!");

    std::set<int> uniqueDeviceList(indexConfig.deviceList.begin(), indexConfig.deviceList.end());
    if (uniqueDeviceList.size() != indexConfig.deviceList.size()) {
        std::string deviceListStr;
        for (auto id : indexConfig.deviceList) {
            deviceListStr += std::to_string(id) + ",";
        }
        FAISS_THROW_FMT("some device IDs are the same, please check it {%s}", deviceListStr.c_str());
    }

    intf_ = intf;

    pool = std::make_shared<AscendThreadPool>(indexConfig.deviceList.size());

    APP_LOG_INFO("AscendIndexInt8 construction finished");
}

AscendIndexInt8Impl::~AscendIndexInt8Impl()
{
    clearIndexes();
}

void AscendIndexInt8Impl::add(idx_t n, const char *x)
{
    const int8_t *xi = reinterpret_cast<const int8_t *>(x);
    add_with_ids(n, xi, nullptr);
}

void AscendIndexInt8Impl::add_with_ids(idx_t n, const char *x, const idx_t *ids)
{
    const int8_t *xi = reinterpret_cast<const int8_t *>(x);
    add_with_ids(n, xi, ids);
}

void AscendIndexInt8Impl::add(idx_t n, const int8_t *x)
{
    add_with_ids(n, x, nullptr);
}

void AscendIndexInt8Impl::add_with_ids(idx_t n, const int8_t *x, const idx_t *ids)
{
    auto lock = ::ascend::AscendMultiThreadManager::GetWriteLock(mtx);
    APP_LOG_INFO("AscendIndexInt8 start to add %ld vector(s) with ids.\n", n);
    FAISS_THROW_IF_NOT_MSG(x, "x can not be nullptr.");
    FAISS_THROW_IF_NOT_MSG(trained, "Index not trained");
    FAISS_THROW_IF_NOT_FMT((n > 0) && (n < MAX_N), "n must be > 0 and < %ld", MAX_N);
    FAISS_THROW_IF_NOT_FMT(ntotal + n < MAX_N, "ntotal must be < %ld", MAX_N);

    std::vector<idx_t> tmpIds;
    if (ids == nullptr && addImplRequiresIDs()) {
        tmpIds = std::vector<idx_t>(n);

        for (idx_t i = 0; i < n; ++i) {
            tmpIds[i] = ntotal + i;
        }

        ids = tmpIds.data();
    }

    addPaged(n, x, ids);
    APP_LOG_INFO("AscendIndexInt8 add_with_ids operation finished.\n");
}

void AscendIndexInt8Impl::train(idx_t n, const int8_t *x)
{
    VALUE_UNUSED(n);
    VALUE_UNUSED(x);
}

void AscendIndexInt8Impl::updateCentroids(idx_t n, const int8_t *x)
{
    VALUE_UNUSED(n);
    VALUE_UNUSED(x);
}

void AscendIndexInt8Impl::updateCentroids(idx_t n, const char *x)
{
    APP_LOG_INFO("AscendIndexInt8 updateCentroids operation started.\n");
    const int8_t *xi = reinterpret_cast<const int8_t *>(x);
    updateCentroids(n, xi);
    APP_LOG_INFO("AscendIndexInt8 updateCentroids operation finished.\n");
}

bool AscendIndexInt8Impl::addImplRequiresIDs() const
{
    // default return true
    return true;
}

void AscendIndexInt8Impl::addPaged(int n, const int8_t *x, const idx_t *ids)
{
    APP_LOG_INFO("AscendIndexInt8 addPaged operation started.\n");

    size_t totalSize = static_cast<size_t>(n) * getElementSize();
    if (totalSize > ADD_PAGE_SIZE || static_cast<size_t>(n) > ADD_VEC_SIZE) {
        // How many vectors fit into kAddPageSize?
        size_t maxNumVecsForPageSize = ADD_PAGE_SIZE / getElementSize();
        // Always add at least 1 vector, if we have huge vectors
        maxNumVecsForPageSize = std::max(maxNumVecsForPageSize, static_cast<size_t>(1));

        size_t tileSize = std::min(static_cast<size_t>(n), maxNumVecsForPageSize);

        for (size_t i = 0; i < static_cast<size_t>(n); i += tileSize) {
            size_t curNum = std::min(tileSize, n - i);
            if (this->intf_->verbose) {
                printf("AscendIndexInt8::add: adding %zu:%zu / %d\n", i, i + curNum, n);
            }
            addImpl(curNum, x + i * static_cast<size_t>(dim), ids ? (ids + i) : nullptr);
        }
    } else {
        if (this->intf_->verbose) {
            printf("AscendIndexInt8::add: adding 0:%d / %d\n", n, n);
        }
        addImpl(n, x, ids);
    }
    APP_LOG_INFO("AscendIndexInt8 addPaged operation finished.\n");
}

size_t AscendIndexInt8Impl::remove_ids(const faiss::IDSelector &sel)
{
    auto lock = ::ascend::AscendMultiThreadManager::GetWriteLock(mtx);
    return removeImpl(sel);
}

void AscendIndexInt8Impl::assign(idx_t n, const int8_t *x, idx_t *labels, idx_t k) const
{
    APP_LOG_INFO("AscendIndexInt8 assign operation started.\n");
    FAISS_THROW_IF_NOT_FMT((n > 0) && (n < MAX_N), "n must be > 0 and < %ld", MAX_N);
    FAISS_THROW_IF_NOT_FMT((k > 0) && (k <= MAX_K), "k must be > 0 and <= %ld", MAX_K);
    FAISS_THROW_IF_NOT_MSG(n * k < 1e10, "n * k must be < 1e10");

    std::unique_ptr<float[]> distances = std::make_unique<float[]>(n * k);
    search(n, x, k, distances.get(), labels);
    APP_LOG_INFO("AscendIndexInt8 assign operation finished.\n");
}

void AscendIndexInt8Impl::search(idx_t n, const int8_t *x, idx_t k, float *distances,
    idx_t *labels) const
{
    auto lock = ::ascend::AscendMultiThreadManager::GetReadLock(mtx);
    APP_LOG_INFO("AscendIndexInt8 start to search: searchNum=%ld, topK=%ld.\n", n, k);
    FAISS_THROW_IF_NOT_MSG(x, "x can not be nullptr.");
    FAISS_THROW_IF_NOT_MSG(distances, "distances can not be nullptr.");
    FAISS_THROW_IF_NOT_MSG(labels, "labels can not be nullptr.");
    FAISS_THROW_IF_NOT_FMT((k > 0) && (k <= MAX_K), "k must be > 0 and <= %ld", MAX_K);
    FAISS_THROW_IF_NOT_MSG(trained, "Index not trained");
    FAISS_THROW_IF_NOT_FMT((n > 0) && (n < MAX_N), "n must be > 0 and < %ld", MAX_N);
    FAISS_THROW_IF_NOT_MSG(indexes.size() > 0, "indexes.size must be >0");

    searchPaged(n, x, k, distances, labels);
    APP_LOG_INFO("AscendIndexInt8 search finished.\n");
}

void AscendIndexInt8Impl::search(idx_t n, const char *x, idx_t k, float *distances,
    idx_t *labels) const
{
    search(n, reinterpret_cast<const int8_t *>(x), k, distances, labels);
}

void AscendIndexInt8Impl::searchPaged(int n, const int8_t *x, int k, float *distances, idx_t *labels) const
{
    APP_LOG_INFO("AscendIndexInt8 searchPaged operation started: n=%d, k=%d.\n", n, k);
    size_t totalSize = static_cast<size_t>(n) * static_cast<size_t>(dim) * sizeof(float);
    size_t totalOutSize = static_cast<size_t>(n) * static_cast<size_t>(k) * (sizeof(uint16_t) + sizeof(ascend_idx_t));

    if (totalSize > SEARCH_PAGE_SIZE || static_cast<size_t>(n) > SEARCH_VEC_SIZE || totalOutSize > SEARCH_PAGE_SIZE) {
        // How many vectors fit into searchPageSize?
        size_t maxNumVecsForPageSize = SEARCH_PAGE_SIZE / (static_cast<size_t>(dim) * sizeof(float));

        size_t maxRetVecsForPageSize = SEARCH_PAGE_SIZE / (static_cast<size_t>(k) * (sizeof(uint16_t) +
            sizeof(ascend_idx_t)));

        maxNumVecsForPageSize = std::min(maxNumVecsForPageSize, maxRetVecsForPageSize);

        // Always add at least 1 vector, if we have huge vectors
        maxNumVecsForPageSize = std::max(maxNumVecsForPageSize, static_cast<size_t>(1));

        size_t tileSize = std::min(static_cast<size_t>(n), maxNumVecsForPageSize);

        for (size_t i = 0; i < static_cast<size_t>(n); i += tileSize) {
            size_t curNum = std::min(tileSize, static_cast<size_t>(n) - i);
            searchImpl(curNum, x + i * static_cast<size_t>(dim), k, distances + i * static_cast<size_t>(k),
                labels + i * static_cast<size_t>(k));
        }
    } else {
        searchImpl(n, x, k, distances, labels);
    }
    APP_LOG_INFO("AscendIndexInt8 searchPaged operation finished.\n");
}

void AscendIndexInt8Impl::searchImpl(int n, const int8_t *x, int k, float *distances, idx_t *labels) const
{
    APP_LOG_INFO("AscendIndexInt8 searchImpl operation started: n=%d, k=%d.\n", n, k);
    size_t deviceCnt = indexConfig.deviceList.size();

    // convert query data from float to fp16, device use fp16 data to search
    std::vector<int8_t> query(n * dim, 0);
    transform(x, x + n * dim, begin(query), [](int8_t temp) { return temp; });

    std::vector<std::vector<float>> dist(deviceCnt, std::vector<float>(n * k, 0));
    std::vector<std::vector<uint16_t>> distHalf(deviceCnt, std::vector<uint16_t>(n * k, 0));
    std::vector<std::vector<ascend_idx_t>> label(deviceCnt, std::vector<ascend_idx_t>(n * k, 0));

    auto searchFunctor = [&](int idx) {
        int deviceId = indexConfig.deviceList[idx];
        IndexParam<int8_t, uint16_t, ascend_idx_t> param(deviceId, n, dim, k);
        indexInt8Search(param, query.data(), distHalf[idx].data(), label[idx].data());
        if (metricType == faiss::METRIC_L2) {
            // convert result data from fp16 to float
            auto scaleFunctor = [](int inputDim) {
                // 0.01, 128, 4 is hyperparameter suitable for UB
                return 0.01 / std::min(inputDim / 64, std::max(inputDim / 128 + 1, 4));
            };
            float scale = scaleFunctor(dim);
            transform(begin(distHalf[idx]), end(distHalf[idx]), begin(dist[idx]),
                [scale](uint16_t temp) { return std::sqrt((float)fp16(temp) / scale); });
        } else {
            // convert result data from fp16 to float
            transform(begin(distHalf[idx]), end(distHalf[idx]), begin(dist[idx]),
                [](uint16_t temp) { return (float)fp16(temp); });
        }
    };

    CALL_PARALLEL_FUNCTOR(deviceCnt, pool, searchFunctor);

    // post process: default is merge the topk results from devices
    searchPostProcess(deviceCnt, dist, label, n, k, distances, labels);
    APP_LOG_INFO("AscendIndexInt8 searchImpl operation finished.\n");
}

void AscendIndexInt8Impl::reserveMemory(size_t numVecs)
{
    VALUE_UNUSED(numVecs);

    FAISS_THROW_MSG("reserveMemory not implemented for this type of index");
}

size_t AscendIndexInt8Impl::reclaimMemory()
{
    FAISS_THROW_MSG("reclaimMemory not implemented for this type of index");
    return 0;
}

void AscendIndexInt8Impl::mergeSearchResult(size_t devices, std::vector<std::vector<float>> &dist,
    std::vector<std::vector<ascend_idx_t>> &label, idx_t n, idx_t k, float *distances,
    idx_t *labels) const
{
    APP_LOG_INFO("AscendIndexInt8 mergeSearchResult operation started.\n");
    std::function<bool(float, float)> compFunc = GetCompFunc();

    // merge several topk results into one topk results
    // every topk result need to be reodered in ascending order
    // batch greater then 128 using omp
#pragma omp parallel for if (n > 128) num_threads(::ascend::CommonUtils::GetThreadMaxNums())
    for (idx_t i = 0; i < n; i++) {
        idx_t num = 0;
        idx_t offset = i * k;
        std::vector<int> posit(devices, 0);
        while (num < k) {
            size_t id = 0;
            float disMerged = dist[0][offset + posit[0]];
            ascend_idx_t labelMerged = label[0][offset + posit[0]];
            for (size_t j = 1; j < devices; j++) {
                idx_t pos = offset + posit[j];
                if (static_cast<idx_t>(label[j][pos]) != -1 && compFunc(dist[j][pos], disMerged)) {
                    disMerged = dist[j][pos];
                    labelMerged = label[j][pos];
                    id = j;
                }
            }

            *(distances + offset + num) = disMerged;
            *(labels + offset + num) = (idx_t)labelMerged;
            posit[id]++;
            num++;
        }
    }
    APP_LOG_INFO("AscendIndexInt8 mergeSearchResult operation started.\n");
}

std::function<bool(float, float)> AscendIndexInt8Impl::GetCompFunc() const
{
    std::function<bool(float, float)> compFunc;
    switch (metricType) {
        case faiss::METRIC_INNER_PRODUCT:
            std::greater<float> greaterComp;
            compFunc = greaterComp;
            break;
        case faiss::METRIC_L2:
            std::less<float> lessComp;
            compFunc = lessComp;
            break;
        default:
            FAISS_THROW_MSG("Unsupported metric type");
            break;
    }
    return compFunc;
}

void AscendIndexInt8Impl::searchPostProcess(size_t devices, std::vector<std::vector<float>> &dist,
    std::vector<std::vector<ascend_idx_t>> &label, idx_t n, idx_t k, float *distances,
    idx_t *labels) const
{
    APP_LOG_INFO("AscendIndexInt8 searchPostProcess operation started.\n");
    mergeSearchResult(devices, dist, label, n, k, distances, labels);
    APP_LOG_INFO("AscendIndexInt8 searchPostProcess operation finished.\n");
}

std::shared_ptr<std::shared_lock<std::shared_mutex>> AscendIndexInt8Impl::getReadLock() const
{
    return std::make_shared<std::shared_lock<std::shared_mutex>>(mtx);
}

std::vector<int> AscendIndexInt8Impl::getDeviceList() const
{
    auto lock = ::ascend::AscendMultiThreadManager::GetReadLock(mtx);
    return indexConfig.deviceList;
}

void AscendIndexInt8Impl::initIndexes()
{
    APP_LOG_INFO("AscendIndexInt8 initIndexes start");
    indexes.clear();
    for (size_t i = 0; i < indexConfig.deviceList.size(); i++) {
        int deviceId = indexConfig.deviceList[i];
        indexes[deviceId] = createIndex(deviceId);
    }
    APP_LOG_INFO("AscendIndexInt8 initIndexes finished");
}

void AscendIndexInt8Impl::clearIndexes()
{
    APP_LOG_INFO("AscendIndexInt8Impl clearIndexes start");
    indexes.clear();
    APP_LOG_INFO("AscendIndexInt8Impl clearIndexes finished");
}

void AscendIndexInt8Impl::indexInt8Search(IndexParam<int8_t, uint16_t, ascend_idx_t> param,
                                          const int8_t *query, uint16_t *distance, ascend_idx_t *label) const
{
    auto index = getActualIndex(param.deviceId);
    using namespace ::ascend;
    AscendTensor<int8_t, DIMS_2> tensorDevQueries({ param.n, param.dim });
    auto ret = aclrtMemcpy(tensorDevQueries.data(), tensorDevQueries.getSizeInBytes(), query,
                           static_cast<size_t>(param.n) *  static_cast<size_t>(param.dim) * sizeof(int8_t),
                           ACL_MEMCPY_HOST_TO_DEVICE);
    FAISS_THROW_IF_NOT_FMT(ret == ACL_SUCCESS, "aclrtMemcpy error %d", ret);

    ret = index->search(param.n, tensorDevQueries.data(), param.k, distance, label);
    FAISS_THROW_IF_NOT_FMT(ret == APP_ERR_OK, "Failed to search index result is: %d\n", ret);
}

const std::shared_ptr<AscendThreadPool> AscendIndexInt8Impl::GetPool() const
{
    auto lock = ::ascend::AscendMultiThreadManager::GetReadLock(mtx);
    return this->pool;
}

faiss::idx_t AscendIndexInt8Impl::GetIdxFromDeviceMap(int deviceId, int idxId) const
{
    if (static_cast<size_t>(deviceId) >= idxDeviceMap.size() ||
        static_cast<size_t>(idxId) >= idxDeviceMap[deviceId].size()) {
        // 当前接口外部使用时无效值均使用ascend_idx_t，因此这里异常场景返回ascend_idx_t
        return std::numeric_limits<ascend_idx_t>::max();
    }

    return this->idxDeviceMap.at(deviceId).at(idxId);
}

void AscendIndexInt8Impl::CheckIndexParams(IndexImplBase &index, bool) const
{
    VALUE_UNUSED(index);

    FAISS_THROW_MSG("CheckIndexParams not implemented for this type of index");
}

faiss::ascendSearch::AscendIndexIVFSPSQ* AscendIndexInt8Impl::GetIVFSPSQPtr() const
{
    FAISS_THROW_MSG("GetIVFSPSQPtr not implemented for this type of index, please make sure pass AscendIndexIVFSP");
}

void* AscendIndexInt8Impl::GetActualIndex(int deviceId, bool isNeedSetDevice) const
{
    auto lock = ::ascend::AscendMultiThreadManager::GetReadLock(mtx);
    return getActualIndex(deviceId, isNeedSetDevice);
}

int AscendIndexInt8Impl::getDim() const
{
    auto lock = ::ascend::AscendMultiThreadManager::GetReadLock(mtx);
    return dim;
}

faiss::idx_t AscendIndexInt8Impl::getNTotal() const
{
    auto lock = ::ascend::AscendMultiThreadManager::GetReadLock(mtx);
    return ntotal;
}

bool AscendIndexInt8Impl::isTrained() const
{
    auto lock = ::ascend::AscendMultiThreadManager::GetReadLock(mtx);
    return trained;
}

faiss::MetricType AscendIndexInt8Impl::getMetricType() const
{
    auto lock = ::ascend::AscendMultiThreadManager::GetReadLock(mtx);
    return metricType;
}

} // namespace ascend
} // namespace faiss
