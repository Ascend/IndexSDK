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


#include "AscendIndexImpl.h"

#include <limits>
#include <algorithm>

#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/AuxIndexStructures.h>
#include "ascend/rpc/AscendRpc.h"
#include "ascend/utils/AscendUtils.h"
#include "ascend/utils/fp16.h"
#include "common/threadpool/AscendThreadPool.h"

namespace faiss {
namespace ascendSearch {
namespace {
const size_t KB = 1024;
const size_t RETAIN_SIZE = 2048;
const size_t UNIT_PAGE_SIZE = 64;
const size_t UNIT_VEC_SIZE = 512;
const size_t DEVICE_LIST_SIZE_MAX = 64;
const int MAX_DIM = 2048;
const int DIM_DIVISOR = 16;

// Default size for which we page add or search
const size_t ADD_PAGE_SIZE = UNIT_PAGE_SIZE * KB * KB - RETAIN_SIZE;

// Or, maximum number of vectors to consider per page of add
const size_t ADD_VEC_SIZE = UNIT_VEC_SIZE * KB;

// Default size for which we get base
const size_t GET_PAGE_SIZE = UNIT_PAGE_SIZE * KB * KB - RETAIN_SIZE;

// Or, maximum number of vectors to consider per page of get
const size_t GET_VEC_SIZE = UNIT_VEC_SIZE * KB;

// search pagesize must be less than 64M, becauseof rpc limitation
const size_t SEARCH_PAGE_SIZE = UNIT_PAGE_SIZE * KB * KB - RETAIN_SIZE;

// Or, maximum number 512K of vectors to consider per page of search
const size_t SEARCH_VEC_SIZE = UNIT_VEC_SIZE * KB;
}

AscendIndexImpl::AscendIndexImpl(int dims, faiss::MetricType metric,
                                 AscendIndexConfig config, AscendIndex *intf)
    : indexConfig(config)
{
    APP_LOG_INFO("AscendIndex construction");
    FAISS_THROW_IF_NOT_MSG(intf, "Intf is nullptr");

    FAISS_THROW_IF_NOT_FMT(dims > 0 && dims <= MAX_DIM,
        "Invalid number of dimensions, should be (0, %d]", MAX_DIM);
    FAISS_THROW_IF_NOT_FMT(dims % DIM_DIVISOR == 0,
        "Invalid number of dimensions, should be divisible by %d", DIM_DIVISOR);

    FAISS_THROW_IF_NOT_FMT(indexConfig.deviceList.size() > 0 && indexConfig.deviceList.size() <= DEVICE_LIST_SIZE_MAX,
                           "device list should be in range (0, %zu]!", DEVICE_LIST_SIZE_MAX);
    FAISS_THROW_IF_NOT_MSG(indexConfig.resourceSize == -1 ||
                           (indexConfig.resourceSize >= 0 && indexConfig.resourceSize <= INDEX_MAX_MEM),
                           "resourceSize should be -1 or in range [0, 4096MB]!");

    intf->d = dims;
    intf->metric_type = metric;
    this->intf_ = intf;

    pool = std::make_shared<AscendThreadPool>(indexConfig.deviceList.size());
}

AscendIndexImpl::~AscendIndexImpl()
{
    APP_LOG_INFO("AscendIndex destruction start");
    clearRpcCtx();
    APP_LOG_INFO("AscendIndex destruction finished");
}

void AscendIndexImpl::initRpcCtx()
{
    APP_LOG_INFO("AscendIndex initRpcCtx start");
    indexMap.clear();
    contextMap.clear();
    for (size_t i = 0; i < indexConfig.deviceList.size(); i++) {
        rpcContext ctx;
        int deviceId = indexConfig.deviceList[i];

        RpcError ret = RpcCreateContext(deviceId, &ctx);
        FAISS_THROW_IF_NOT_FMT(ret == RPC_ERROR_NONE, "Connect device (%d) failed, please check device status.",
            deviceId);
        contextMap[deviceId] = ctx;
        APP_LOG_INFO("AscendIndex initRpcCtx finished11111");
        int indexId = 0;
        try {
            indexId = CreateIndex(ctx);
        } catch (std::exception &ex) {
            auto retInside = RpcDestroyContext(ctx);
            if (retInside != RPC_ERROR_NONE) {
                APP_LOG_ERROR("Destroy context failed(%d).", retInside);
            }
            FAISS_THROW_MSG(ex.what());
        }
        indexMap[ctx] = indexId;
    }
    APP_LOG_INFO("AscendIndex initRpcCtx finished");
}

void AscendIndexImpl::clearRpcCtx()
{
    APP_LOG_INFO("AscendIndex clearRpcCtx start");
    for (auto &index : indexMap) {
        DestroyIndex(index.first, index.second);

        auto ret = RpcDestroyContext(index.first);
        if (ret != RPC_ERROR_NONE) {
            APP_LOG_ERROR("Destroy context failed(%d).", ret);
        }
    }

    indexMap.clear();
    contextMap.clear();
    APP_LOG_INFO("AscendIndex clearRpcCtx finished");
}

void AscendIndexImpl::DestroyIndex(rpcContext ctx, int indexId) const
{
    APP_LOG_INFO("AscendIndex DestroyIndex start");
    RpcError ret = RpcDestroyIndex(ctx, indexId);
    if (ret != RPC_ERROR_NONE) {
        APP_LOG_ERROR("Destroy index failed(%d).", ret);
    }
    APP_LOG_INFO("AscendIndex DestroyIndex finished");
}

void AscendIndexImpl::add(idx_t n, const float *x)
{
    APP_LOG_INFO("AscendIndex add operation started.\n");
    add_with_ids(n, x, nullptr);
    APP_LOG_INFO("AscendIndex add operation finished.\n");
}

void AscendIndexImpl::add_with_ids(idx_t n, const float* x, const idx_t* ids)
{
    APP_LOG_INFO("AscendIndex add_with_ids operation started.\n");
    FAISS_THROW_IF_NOT_MSG(x, "x can not be nullptr.");
    FAISS_THROW_IF_NOT_MSG(this->intf_->is_trained, "Index not trained");
    FAISS_THROW_IF_NOT_FMT((n > 0) && (n < MAX_N), "n must be > 0 and < %ld", MAX_N);
    FAISS_THROW_IF_NOT_FMT(this->intf_->ntotal + n < MAX_N, "ntotal must be < %ld", MAX_N);

    std::vector<idx_t> tmpIds;
    if (ids == nullptr && addImplRequiresIDs()) {
        tmpIds = std::vector<idx_t>(n);

        for (idx_t i = 0; i < n; ++i) {
            tmpIds[i] = this->intf_->ntotal + i;
        }

        ids = tmpIds.data();
    }

    addPaged(n, x, ids);
    APP_LOG_INFO("AscendIndex add_with_ids operation finished.\n");
}

void AscendIndexImpl::addPaged(int n, const float* x, const idx_t* ids)
{
    APP_LOG_INFO("AscendIndex addPaged operation started.\n");
    size_t totalSize = static_cast<size_t>(n) * getAddElementSize();
    if (totalSize > ADD_PAGE_SIZE || static_cast<size_t>(n) > ADD_VEC_SIZE) {
        size_t tileSize = getAddPagedSize(n);

        for (size_t i = 0; i < static_cast<size_t>(n); i += tileSize) {
            size_t curNum = std::min(tileSize, n - i);
            if (this->intf_->verbose) {
                printf("AscendIndex::add: adding %zu:%zu / %d\n", i, i + curNum, n);
            }
            addImpl(curNum, x + i * static_cast<size_t>(this->intf_->d), ids ? (ids + i) : nullptr);
        }
    } else {
        if (this->intf_->verbose) {
            printf("AscendIndex::add: adding 0:%d / %d\n", n, n);
        }
        addImpl(n, x, ids);
    }
    APP_LOG_INFO("AscendIndex addPaged operation finished.\n");
}

size_t AscendIndexImpl::getAddPagedSize(int n) const
{
    APP_LOG_INFO("AscendIndex getAddPagedSize operation started.\n");
    size_t maxNumVecsForPageSize = ADD_PAGE_SIZE / getAddElementSize();
    // Always add at least 1 vector, if we have huge vectors
    maxNumVecsForPageSize = std::max(maxNumVecsForPageSize, static_cast<size_t>(1));
    APP_LOG_INFO("AscendIndex getAddPagedSize operation finished.\n");

    return std::min(static_cast<size_t>(n), maxNumVecsForPageSize);
}

size_t AscendIndexImpl::getAddElementSize() const
{
    FAISS_THROW_MSG("getAddElementSize() not implemented for this type of index");
    return 0;
}

void AscendIndexImpl::calcAddMap(int n, std::vector<int> &addMap)
{
    APP_LOG_INFO("AscendIndex calcAddMap operation started.\n");
    size_t devIdx = 0;
    size_t deviceCnt = indexConfig.deviceList.size();
    FAISS_THROW_IF_NOT_FMT(deviceCnt != 0, "Wrong deviceCnt: %zu", deviceCnt);

    for (size_t i = 1; i < deviceCnt; i++) {
        if (idxDeviceMap[i].size() < idxDeviceMap[devIdx].size()) {
            devIdx = i;
            break;
        }
    }
    for (size_t i = 0; i < deviceCnt; i++) {
        addMap[i] += n / static_cast<int>(deviceCnt);
    }
    for (size_t i = 0; i < static_cast<size_t>(n) % deviceCnt; i++) {
        addMap[devIdx % deviceCnt] += 1;
        devIdx += 1;
    }
    APP_LOG_INFO("AscendIndex calcAddMap operation finished.\n");
}

size_t AscendIndexImpl::remove_ids(const faiss::IDSelector& sel)
{
    return removeImpl(sel);
}

size_t AscendIndexImpl::removeImpl(const IDSelector &sel)
{
    APP_LOG_INFO("AscendIndex removeImpl operation not implemented in ascendSearch.\n");
    VALUE_UNUSED(sel);
    return 0;
}

void AscendIndexImpl::removeIdx(const std::vector<std::vector<ascend_idx_t>> &removeMaps) const
{
    APP_LOG_INFO("AscendIndex removeIdx operation not implemented in ascendSearch.\n");
    VALUE_UNUSED(removeMaps);
}

void AscendIndexImpl::search(idx_t n, const float* x, idx_t k,
                             float* distances, idx_t* labels) const
{
    APP_LOG_INFO("AscendIndex start to search: searchNum=%ld, topK=%ld.\n", n, k);
    FAISS_THROW_IF_NOT_MSG(x, "x can not be nullptr.");
    FAISS_THROW_IF_NOT_MSG(distances, "distances can not be nullptr.");
    FAISS_THROW_IF_NOT_MSG(labels, "labels can not be nullptr.");
    FAISS_THROW_IF_NOT_FMT((n > 0) && (n < MAX_N), "n must be > 0 and < %ld", MAX_N);
    FAISS_THROW_IF_NOT_FMT((k > 0) && (k <= MAX_K), "k must be > 0 and <= %ld", MAX_K);
    FAISS_THROW_IF_NOT_MSG(contextMap.size() > 0, "contextMap.size must be >0");
    FAISS_THROW_IF_NOT_MSG(indexMap.size() > 0, "indexMap.size must be >0");

    searchPaged(n, x, k, distances, labels);
    APP_LOG_INFO("AscendIndex search operation finished.\n");
}

size_t AscendIndexImpl::getSearchPagedSize(int n, int k) const
{
    APP_LOG_INFO("AscendIndex getSearchPagedSize operation started.\n");
    // How many vectors fit into searchPageSize?
    size_t maxNumVecsForPageSize = SEARCH_PAGE_SIZE / (static_cast<size_t>(this->intf_->d) * sizeof(float));
    size_t maxRetVecsForPageSize =
        SEARCH_PAGE_SIZE / (static_cast<size_t>(k) * (sizeof(uint16_t) + sizeof(ascend_idx_t)));
    maxNumVecsForPageSize = std::min(maxNumVecsForPageSize, maxRetVecsForPageSize);

    // Always add at least 1 vector, if we have huge vectors
    maxNumVecsForPageSize = std::max(maxNumVecsForPageSize, static_cast<size_t>(1));
    APP_LOG_INFO("AscendIndex getSearchPagedSize operation finished.\n");
    return std::min(static_cast<size_t>(n), maxNumVecsForPageSize);
}

void AscendIndexImpl::searchPaged(int n, const float* x, int k, float* distances, idx_t* labels) const
{
    APP_LOG_INFO("AscendIndex start to searchPaged: n=%d, k=%d.\n", n, k);
    if (n > 0) {
        size_t totalSize = static_cast<size_t>(n) * static_cast<size_t>(this->intf_->d) * sizeof(float);
        size_t totalOutSize =
            static_cast<size_t>(n) * static_cast<size_t>(k) * (sizeof(uint16_t) + sizeof(ascend_idx_t));

        if (totalSize > SEARCH_PAGE_SIZE ||
        static_cast<size_t>(n) > SEARCH_VEC_SIZE || totalOutSize > SEARCH_PAGE_SIZE) {
            size_t tileSize = getSearchPagedSize(n, k);

            for (size_t i = 0; i < static_cast<size_t>(n); i += tileSize) {
                size_t curNum = std::min(tileSize, static_cast<size_t>(n) - i);
                searchImpl(curNum, x + i * static_cast<size_t>(this->intf_->d), k,
                           distances + i * static_cast<size_t>(k), labels + i * static_cast<size_t>(k));
            }
        } else {
            searchImpl(n, x, k, distances, labels);
        }
    }
    APP_LOG_INFO("AscendIndex searchPaged operation finished.\n");
}

void AscendIndexImpl::searchImpl(int n, const float* x, int k, float* distances,
    idx_t* labels) const
{
    APP_LOG_INFO("AscendIndex searchImpl operation started: n=%d, k=%d.\n", n, k);
    size_t deviceCnt = indexConfig.deviceList.size();

    // convert query data from float to fp16, device use fp16 data to search
    std::vector<uint16_t> query(n * this->intf_->d, 0);
    transform(x, x + n * this->intf_->d, begin(query), [](float temp) { return fp16(temp).data; });

    std::vector<std::vector<float>> dist(deviceCnt, std::vector<float>(n * k, 0));
    std::vector<std::vector<uint16_t>> distHalf(deviceCnt, std::vector<uint16_t>(n * k, 0));
    std::vector<std::vector<ascend_idx_t>> label(deviceCnt, std::vector<ascend_idx_t>(n * k, 0));

    auto searchFunctor = [&](int idx) {
        int deviceId = indexConfig.deviceList[idx];
        rpcContext ctx = contextMap.at(deviceId);
        int indexId = indexMap.at(ctx);
        RpcError ret = RpcIndexSearch(ctx, indexId, n, this->intf_->d, k, query.data(),
                                      distHalf[idx].data(), label[idx].data());
        FAISS_THROW_IF_NOT_FMT(ret == RPC_ERROR_NONE, "Search implement failed(%d).", ret);

        // convert result data from fp16 to float
        transform(begin(distHalf[idx]), end(distHalf[idx]),
                  begin(dist[idx]), [](uint16_t temp) { return static_cast<float>(fp16(temp)); });
    };

    CALL_PARALLEL_FUNCTOR(deviceCnt, pool, searchFunctor);

    // post process: default is merge the topk results from devices
    searchPostProcess(deviceCnt, dist, label, n, k, distances, labels);
    APP_LOG_INFO("AscendIndex searchImpl operation finished.\n");
}

size_t AscendIndexImpl::getBaseSize(int deviceId) const
{
    VALUE_UNUSED(deviceId);
    
    FAISS_THROW_MSG("getBaseSize() not implemented for this type of index");
    return 0;
}

void AscendIndexImpl::getBase(int deviceId, char* xb) const
{
    APP_LOG_INFO("AscendIndex getBase operation started.\n");
    FAISS_THROW_IF_NOT_MSG(xb, "xb can not be nullptr.");
    FAISS_THROW_IF_NOT_FMT(contextMap.find(deviceId) != contextMap.end(),
        "deviceId is out of range, deviceId=%d.", deviceId);
    size_t size = getBaseSize(deviceId);
    getBasePaged(deviceId, size, xb);
    APP_LOG_INFO("AscendIndex getBase operation finished.\n");
}

void AscendIndexImpl::getBasePaged(int deviceId, int n, char* codes) const
{
    APP_LOG_INFO("AscendIndex getBasePaged operation started.\n");
    if (n > 0) {
#ifdef HOSTCPU
        // paged is not needed on hostcpu
        getBaseImpl(deviceId, 0, n, codes);
#else
        size_t totalSize = static_cast<size_t>(n) * getBaseElementSize();
        size_t offsetNum = 0;
        if (totalSize > GET_PAGE_SIZE || static_cast<size_t>(n) > GET_VEC_SIZE) {
            size_t tileSize = getBasePagedSize(n);

            for (size_t i = 0; i < static_cast<size_t>(n); i += tileSize) {
                size_t curNum = std::min(tileSize, size_t(n) - i);

                getBaseImpl(deviceId, offsetNum, curNum, codes);
                offsetNum += curNum;
            }
        } else {
            getBaseImpl(deviceId, offsetNum, n, codes);
        }
#endif
    }
    APP_LOG_INFO("AscendIndex getBasePaged operation finished.\n");
}

void AscendIndexImpl::getBaseImpl(int deviceId, int offset, int n, char *x) const
{
    VALUE_UNUSED(deviceId);
    VALUE_UNUSED(offset);
    VALUE_UNUSED(n);
    VALUE_UNUSED(x);

    FAISS_THROW_MSG("AscendIndex not implemented for this type of index");
}

size_t AscendIndexImpl::getBasePagedSize(int n) const
{
    APP_LOG_INFO("AscendIndex getBasePagedSize operation started.\n");
    size_t maxNumVecsForPageSize = GET_PAGE_SIZE / getBaseElementSize();
    // Always add at least 1 vector, if we have huge vectors
    maxNumVecsForPageSize = std::max(maxNumVecsForPageSize, static_cast<size_t>(1));
    APP_LOG_INFO("AscendIndex getBasePagedSize operation finished.\n");

    return std::min(static_cast<size_t>(n), maxNumVecsForPageSize);
}

size_t AscendIndexImpl::getBaseElementSize() const
{
    FAISS_THROW_MSG("getBaseElementSize() not implemented for this type of index");
    return 0;
}

void AscendIndexImpl::getIdxMap(int deviceId, std::vector<idx_t> &idxMap) const
{
    APP_LOG_INFO("AscendIndex getIdxMap operation started.\n");
    FAISS_THROW_IF_NOT_FMT(contextMap.find(deviceId) != contextMap.end(),
        "deviceId is out of range, deviceId=%d.", deviceId);
    size_t deviceNum = indexConfig.deviceList.size();
    for (size_t i = 0; i < deviceNum; i++) {
        if (deviceId == indexConfig.deviceList[i]) {
            idxMap = idxDeviceMap.at(i);
            break;
        }
    }
    APP_LOG_INFO("AscendIndex getIdxMap operation finished.\n");
}

void AscendIndexImpl::reserveMemory(size_t numVecs)
{
    VALUE_UNUSED(numVecs);

    FAISS_THROW_MSG("reserveMemory not implemented for this type of index");
}

size_t AscendIndexImpl::reclaimMemory()
{
    FAISS_THROW_MSG("reclaimMemory not implemented for this type of index");
    return 0;
}

void AscendIndexImpl::reset()
{
    APP_LOG_INFO("AscendIndex reset operation started.\n");
    for (auto &data : indexMap) {
        RpcError ret = RpcIndexReset(data.first, data.second);
        FAISS_THROW_IF_NOT_FMT(ret == RPC_ERROR_NONE, "Reset Index failed(%d).", ret);
    }

    size_t deviceNum = indexConfig.deviceList.size();
    idxDeviceMap.clear();
    idxDeviceMap.resize(deviceNum);

    this->intf_->ntotal = 0;
    APP_LOG_INFO("AscendIndex reset operation finished.\n");
}

std::vector<int> AscendIndexImpl::getDeviceList()
{
    return indexConfig.deviceList;
}

void AscendIndexImpl::checkParamsSame(AscendIndexImpl& index)
{
    VALUE_UNUSED(index);

    FAISS_THROW_MSG("checkParamsSame not implemented for this type of index, maybe the index is not support Search");
}

bool AscendIndexImpl::addImplRequiresIDs() const
{
    // default return true
    return true;
}

void AscendIndexImpl::mergeSearchResult(size_t devices, std::vector<std::vector<float>>& dist,
                                        std::vector<std::vector<ascend_idx_t>>& label, int n, int k,
                                        float* distances, idx_t* labels) const
{
    APP_LOG_INFO("AscendIndex mergeSearchResult operation started.\n");
    std::function<bool(float, float)> compFunc;
    switch (this->intf_->metric_type) {
        case faiss::METRIC_L2:
            std::less<float> lessComp;
            compFunc = lessComp;
            break;
        case faiss::METRIC_INNER_PRODUCT:
            std::greater<float> greaterComp;
            compFunc = std::move(greaterComp);
            break;
        default:
            FAISS_THROW_MSG("Unsupported metric type");
            break;
    }

    // merge several topk results into one topk results
    // every topk result need to be reodered in ascending order
#pragma omp parallel for if (n > 100)
    for (int i = 0; i < n; i++) {
        int num = 0;
        const int offset = i * k;
        std::vector<int> posit(devices, 0);
        while (num < k) {
            size_t id = 0;
            float disMerged = dist[0][offset + posit[0]];
            ascend_idx_t labelMerged = label[0][offset + posit[0]];
            for (size_t j = 1; j < devices; j++) {
                int pos = offset + posit[j];
                if (compFunc(dist[j][pos], disMerged)) {
                    disMerged = dist[j][pos];
                    labelMerged = label[j][pos];
                    id = j;
                }
            }

            *(distances + offset + num) = disMerged;
            *(labels + offset + num) = static_cast<idx_t>(labelMerged);
            posit[id]++;
            num++;
        }
    }
    APP_LOG_INFO("AscendIndex mergeSearchResult operation finished.\n");
}

void AscendIndexImpl::searchPostProcess(size_t devices, std::vector<std::vector<float>>& dist,
                                        std::vector<std::vector<ascend_idx_t>>& label, int n, int k,
                                        float* distances, idx_t* labels) const
{
    APP_LOG_INFO("AscendIndex searchPostProcess operation started.\n");
    // transmit idx per device to referenced value
    size_t deviceCnt = this->indexConfig.deviceList.size();
    std::vector<std::vector<ascend_idx_t>> transLabel(deviceCnt, std::vector<ascend_idx_t>(n * k, 0));
    for (size_t i = 0; i < deviceCnt; i++) {
        transform(begin(label[i]), end(label[i]), begin(transLabel[i]), [&](ascend_idx_t temp) {
            return (temp != std::numeric_limits<ascend_idx_t>::max()) ? idxDeviceMap[i].at(temp) :
                                                                        std::numeric_limits<ascend_idx_t>::max();
        });
    }

    mergeSearchResult(devices, dist, transLabel, n, k, distances, labels);
    APP_LOG_INFO("AscendIndex searchPostProcess operation end.\n");
}
}  // namespace ascendSearch
}  // namespace faiss
