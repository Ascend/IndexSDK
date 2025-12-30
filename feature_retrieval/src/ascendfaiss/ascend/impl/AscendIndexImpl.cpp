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
#include <set>
#include <string>
#include <thread>

#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/impl/IDSelector.h>

#include "AuxIndexStructures.h"
#include "ascend/utils/fp16.h"
#include "common/utils/CommonUtils.h"
#include "ascenddaemon/utils/AscendUtils.h"
#include "AscendRWLock.h"

using ascend::AscendRWLock;

namespace faiss {
namespace ascend {
namespace {
const size_t KB = 1024;
const size_t RETAIN_SIZE = 2048;
const size_t UNIT_PAGE_SIZE = 64;
const size_t UNIT_VEC_SIZE = 512;
const size_t DEVICE_LIST_SIZE_MAX = 64;
const int MAX_DIM = 4096;
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
    : AscendIndexImpl(dims, metric, config, intf, true)
{}

AscendIndexImpl::AscendIndexImpl(int dims, faiss::MetricType metric,
                                 AscendIndexConfig config, AscendIndex *intf, bool enablePool)
    : indexConfig(config)
{
    APP_LOG_INFO("AscendIndex construction");
    
    checkParameters(dims, metric, config, intf);

    intf->d = dims;
    intf->metric_type = metric;
    this->intf_ = intf;

    if (enablePool && indexConfig.deviceList.size() > 1) {
        pool = std::make_shared<AscendThreadPool>(indexConfig.deviceList.size());
    }
}

AscendIndexImpl::~AscendIndexImpl()
{
    APP_LOG_INFO("AscendIndex destruction start");
    clearIndexes();
    APP_LOG_INFO("AscendIndex destruction finished");
}

void AscendIndexImpl::initIndexes()
{
    APP_LOG_INFO("AscendIndex initIndexes start");
    indexes.clear();
    for (size_t i = 0; i < indexConfig.deviceList.size(); i++) {
        int deviceId = indexConfig.deviceList[i];
        indexes[deviceId] = createIndex(deviceId);
    }
    APP_LOG_INFO("AscendIndex initIndexes finished");
}

void AscendIndexImpl::clearIndexes()
{
    APP_LOG_INFO("AscendIndex clearIndexes start");
    indexes.clear();
    APP_LOG_INFO("AscendIndex clearIndexes finished");
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
            addImpl(curNum, x + i * (size_t)this->intf_->d, ids ? (ids + i) : nullptr);
        }
    } else {
        if (this->intf_->verbose) {
            printf("AscendIndex::add: adding 0:%d / %d\n", n, n);
        }
        addImpl(n, x, ids);
    }
    APP_LOG_INFO("AscendIndex addPaged operation finished.\n");
}

void AscendIndexImpl::addPaged(int n, const uint16_t* x, const idx_t* ids)
{
    VALUE_UNUSED(n);
    VALUE_UNUSED(x);
    VALUE_UNUSED(ids);

    FAISS_THROW_MSG("addPaged not implemented for this type of index");
}

size_t AscendIndexImpl::getAddPagedSize(int n) const
{
    APP_LOG_INFO("AscendIndex getAddPagedSize operation started.\n");
    size_t maxNumVecsForPageSize = ADD_PAGE_SIZE / getAddElementSize();
    // Always add at least 1 vector, if we have huge vectors
    maxNumVecsForPageSize = std::max(maxNumVecsForPageSize, (size_t)1);
    APP_LOG_INFO("AscendIndex getAddPagedSize operation finished.\n");

    return std::min((size_t)n, maxNumVecsForPageSize);
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
    auto lock = ::ascend::AscendMultiThreadManager::GetWriteLock(mtx);
    return removeImpl(sel);
}

size_t AscendIndexImpl::removeImpl(const IDSelector &sel)
{
    APP_LOG_INFO("AscendIndex removeImpl operation started.\n");
    size_t deviceCnt = indexConfig.deviceList.size();
    uint32_t removeCnt = 0;

    // 1. remove vector from device, and removeMaps save the id(not index) of pre-removing
    std::vector<std::vector<ascend_idx_t>> removeMaps(deviceCnt, std::vector<ascend_idx_t>());
    for (size_t i = 0; i < deviceCnt; i++) {
        for (int64_t pos = static_cast<int64_t>(idxDeviceMap[i].size()) - 1; pos >= 0; --pos) {
            if (sel.is_member(idxDeviceMap[i][pos])) {
                removeMaps[i].emplace_back(pos);
            }
        }
    }

    // 2. remove the vector in the device
#pragma omp parallel for reduction(+ : removeCnt) num_threads(deviceCnt)
    for (size_t i = 0; i < deviceCnt; i++) {
        if (removeMaps[i].size() == 0) {
            continue;
        }
        uint32_t removeCnt_tmp = 0;
        int deviceId = indexConfig.deviceList[i];
        auto pIndex = getActualIndex(deviceId);
        ::ascend::IDSelectorBatch batch(removeMaps[i].size(),  removeMaps[i].data());
        removeCnt_tmp = pIndex->removeIds(batch);
        removeCnt += removeCnt_tmp;
    }

    // 3. remove the index save in the host
    removeIdx(removeMaps);

    FAISS_THROW_IF_NOT_FMT(this->intf_->ntotal >= removeCnt,
        "removeCnt should be in range [0, %ld], actual=%d.", this->intf_->ntotal, removeCnt);
    this->intf_->ntotal -= static_cast<faiss::idx_t>(removeCnt);
    APP_LOG_INFO("AscendIndex removeImpl operation finished.\n");
    return static_cast<size_t>(removeCnt);
}

void AscendIndexImpl::removeIdx(std::vector<std::vector<ascend_idx_t>> &removeMaps)
{
    APP_LOG_INFO("AscendIndex removeIdx operation started.\n");
    size_t deviceCnt = indexConfig.deviceList.size();
#pragma omp parallel for if (deviceCnt > 1) num_threads(deviceCnt)
    for (size_t i = 0; i < deviceCnt; ++i) {
        // 1. sort by DESC, then delete from the big to small
        std::sort(removeMaps[i].rbegin(), removeMaps[i].rend());

        for (auto pos : removeMaps[i]) {
            int lastIdx = this->idxDeviceMap[i].size() - 1;
            this->idxDeviceMap[i][pos] = this->idxDeviceMap[i][lastIdx];
            this->idxDeviceMap[i].pop_back();
        }
    }
    APP_LOG_INFO("AscendIndex removeIdx operation finished.\n");
}

std::shared_ptr<std::shared_lock<std::shared_mutex>> AscendIndexImpl::getReadLock() const
{
    return std::make_shared<std::shared_lock<std::shared_mutex>>(mtx);
}

void AscendIndexImpl::search(idx_t n, const float* x, idx_t k, float* distances, idx_t* labels) const
{
    searchProcess(n, x, k, distances, labels);
}

void AscendIndexImpl::search(idx_t n, const uint16_t *x, idx_t k, float *distances, idx_t *labels) const
{
    if (!isSupportFp16Search()) {
        VALUE_UNUSED(n);
        VALUE_UNUSED(x);
        VALUE_UNUSED(k);
        VALUE_UNUSED(distances);
        VALUE_UNUSED(labels);
        FAISS_THROW_MSG("search not implemented for this type of index");
    }
    searchProcess(n, x, k, distances, labels);
}

size_t AscendIndexImpl::getSearchPagedSize(int n, int k) const
{
    APP_LOG_INFO("AscendIndex getSearchPagedSize operation started.\n");
    // How many vectors fit into searchPageSize?
    size_t maxNumVecsForPageSize = SEARCH_PAGE_SIZE / ((size_t)this->intf_->d * sizeof(float));
    size_t maxRetVecsForPageSize = SEARCH_PAGE_SIZE / ((size_t)k * (sizeof(uint16_t) + sizeof(ascend_idx_t)));
    maxNumVecsForPageSize = std::min(maxNumVecsForPageSize, maxRetVecsForPageSize);

    // Always add at least 1 vector, if we have huge vectors
    maxNumVecsForPageSize = std::max(maxNumVecsForPageSize, (size_t)1);
    APP_LOG_INFO("AscendIndex getSearchPagedSize operation finished.\n");
    return std::min((size_t)n, maxNumVecsForPageSize);
}

void AscendIndexImpl::searchPaged(int n, const float* x, int k, float* distances, idx_t* labels) const
{
    if (n <= 0) {
        FAISS_THROW_MSG("n must be > 0");
    }
    APP_LOG_INFO("AscendIndex start to searchPaged: n=%d, k=%d.\n", n, k);
    size_t totalSize = static_cast<size_t>(n) * static_cast<size_t>(this->intf_->d) * sizeof(float);
    size_t totalOutSize = static_cast<size_t>(n) * static_cast<size_t>(k) * (sizeof(uint16_t) + sizeof(ascend_idx_t));

    if (totalSize > SEARCH_PAGE_SIZE || static_cast<size_t>(n) > SEARCH_VEC_SIZE || totalOutSize > SEARCH_PAGE_SIZE) {
        size_t tileSize = getSearchPagedSize(n, k);

        for (size_t i = 0; i < static_cast<size_t>(n); i += tileSize) {
            size_t curNum = std::min(tileSize, static_cast<size_t>(n) - i);
            searchImpl(curNum, x + i * static_cast<size_t>(this->intf_->d), k,
                distances + i * static_cast<size_t>(k), labels + i * static_cast<size_t>(k));
        }
    } else {
        searchImpl(n, x, k, distances, labels);
    }
    APP_LOG_INFO("AscendIndex searchPaged operation finished.\n");
}

void AscendIndexImpl::searchImpl(int n, const float* x, int k, float* distances, idx_t* labels) const
{
    // convert query data from float to fp16, device use fp16 data to search
    std::vector<uint16_t> query(n * this->intf_->d, 0);
    transform(x, x + n * this->intf_->d, begin(query), [](float temp) { return fp16(temp).data; });

    searchImpl(n, query.data(), k, distances, labels);
}

void AscendIndexImpl::searchPaged(int n, const uint16_t *x, int k, float *distances, idx_t *labels) const
{
    APP_LOG_INFO("AscendIndex start to searchPaged: n=%d, k=%d.\n", n, k);
    if (n > 0) {
        size_t totalSize = (size_t)n * (size_t)this->intf_->d * sizeof(float);
        size_t totalOutSize = (size_t)n * (size_t)k * (sizeof(uint16_t) + sizeof(ascend_idx_t));

        if (totalSize > SEARCH_PAGE_SIZE || (size_t)n > SEARCH_VEC_SIZE || totalOutSize > SEARCH_PAGE_SIZE) {
            size_t tileSize = getSearchPagedSize(n, k);

            for (size_t i = 0; i < (size_t)n; i += tileSize) {
                size_t curNum = std::min(tileSize, (size_t)(n) - i);
                searchImpl(curNum, x + i * (size_t)this->intf_->d, k,
                           distances + i * (size_t)k, labels + i * (size_t)k);
            }
        } else {
            searchImpl(n, x, k, distances, labels);
        }
    }
    APP_LOG_INFO("AscendIndex searchPaged operation finished.\n");
}

void AscendIndexImpl::searchImpl(int n, const uint16_t *x, int k, float *distances, idx_t *labels) const
{
    APP_LOG_INFO("AscendIndex searchImpl operation started: n=%d, k=%d.\n", n, k);
    size_t deviceCnt = indexConfig.deviceList.size();

    std::vector<std::vector<float>> dist(deviceCnt, std::vector<float>(n * k, 0));
    std::vector<std::vector<uint16_t>> distHalf(deviceCnt, std::vector<uint16_t>(n * k, 0));
    std::vector<std::vector<ascend_idx_t>> label(deviceCnt, std::vector<ascend_idx_t>(n * k, 0));

    auto searchFunctor = [&](int idx) {
        int deviceId = indexConfig.deviceList[idx];
        // search类接口均使用了共享内存，共享内存要求device级别的串行
        IndexParam<uint16_t, uint16_t, ascend_idx_t> param(deviceId, n, this->intf_->d, k);
        param.query = x;
        param.distance = distHalf[idx].data();
        param.label = label[idx].data();
        indexSearch(param);
        transform(begin(distHalf[idx]), end(distHalf[idx]),
                  begin(dist[idx]), [](uint16_t temp) { return (float)fp16(temp); });
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
    FAISS_THROW_IF_NOT_FMT(indexes.find(deviceId) != indexes.end(),
        "deviceId is out of range, deviceId=%d.", deviceId);
    size_t size = getBaseSize(deviceId);
    getBasePaged(deviceId, size, xb);
    APP_LOG_INFO("AscendIndex getBase operation finished.\n");
}

void AscendIndexImpl::getBasePaged(int deviceId, int n, char* codes) const
{
    APP_LOG_INFO("AscendIndex getBasePaged operation started.\n");
    if (n > 0) {
        size_t totalSize = (size_t)n * getBaseElementSize();
        size_t offsetNum = 0;
        if (totalSize > GET_PAGE_SIZE || (size_t)n > GET_VEC_SIZE) {
            size_t tileSize = getBasePagedSize(n);

            for (size_t i = 0; i < (size_t)n; i += tileSize) {
                size_t curNum = std::min(tileSize, size_t(n) - i);

                getBaseImpl(deviceId, offsetNum, curNum, codes);
                offsetNum += curNum;
            }
        } else {
            getBaseImpl(deviceId, offsetNum, n, codes);
        }
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
    maxNumVecsForPageSize = std::max(maxNumVecsForPageSize, (size_t)1);
    APP_LOG_INFO("AscendIndex getBasePagedSize operation finished.\n");

    return std::min((size_t)n, maxNumVecsForPageSize);
}

size_t AscendIndexImpl::getBaseElementSize() const
{
    FAISS_THROW_MSG("getBaseElementSize() not implemented for this type of index");
    return 0;
}

void AscendIndexImpl::getIdxMap(int deviceId, std::vector<idx_t> &idxMap) const
{
    APP_LOG_INFO("AscendIndex getIdxMap operation started.\n");
    FAISS_THROW_IF_NOT_FMT(indexes.find(deviceId) != indexes.end(),
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
    auto lock = ::ascend::AscendMultiThreadManager::GetWriteLock(mtx);
    APP_LOG_INFO("AscendIndex reset operation started.\n");
    size_t deviceNum = indexConfig.deviceList.size();
    for (size_t i = 0; i < deviceNum; ++i) {
        int deviceId = indexConfig.deviceList[i];
        auto pIndex = getActualIndex(deviceId);
        auto ret = pIndex->reset();
        FAISS_THROW_IF_NOT_FMT(ret == ::ascend::APP_ERR_OK,
                               "Failed to reset index result: %d, device id: %d\n", ret, deviceId);
    }
    idxDeviceMap.clear();
    idxDeviceMap.resize(deviceNum);

    this->intf_->ntotal = 0;
    APP_LOG_INFO("AscendIndex reset operation finished.\n");
}

std::vector<int> AscendIndexImpl::getDeviceList() const
{
    auto lock = ::ascend::AscendMultiThreadManager::GetReadLock(mtx);
    return indexConfig.deviceList;
}

const std::shared_ptr<AscendThreadPool> AscendIndexImpl::GetPool() const
{
    auto lock = ::ascend::AscendMultiThreadManager::GetReadLock(mtx);
    return this->pool;
}

faiss::idx_t AscendIndexImpl::GetIdxFromDeviceMap(int deviceId, int idxId) const
{
    if (static_cast<size_t>(deviceId) >= idxDeviceMap.size() ||
        static_cast<size_t>(idxId) >= idxDeviceMap[deviceId].size()) {
        // 当前接口外部使用时无效值均使用ascend_idx_t，因此这里异常场景返回ascend_idx_t
        return std::numeric_limits<ascend_idx_t>::max();
    }

    return this->idxDeviceMap.at(deviceId).at(idxId);
}

void AscendIndexImpl::CheckIndexParams(IndexImplBase &index, bool) const
{
    VALUE_UNUSED(index);

    FAISS_THROW_MSG("CheckIndexParams not implemented for this type of index");
}

faiss::ascendSearch::AscendIndexIVFSPSQ* AscendIndexImpl::GetIVFSPSQPtr() const
{
    FAISS_THROW_MSG("GetIVFSPSQPtr not implemented for this type of index, please make sure pass AscendIndexIVFSP");
}

void* AscendIndexImpl::GetActualIndex(int deviceId, bool isNeedSetDevice) const
{
    auto lock = ::ascend::AscendMultiThreadManager::GetReadLock(mtx);
    return getActualIndex(deviceId, isNeedSetDevice);
}

void AscendIndexImpl::CheckFilterTime(idx_t n, const idx_t* ids) const
{
    if (indexConfig.filterable && ids != nullptr) {
        for (idx_t i = 0; i < n; i++) {
            // Moving the id 41 bits to the right is the symbol bit of the timestamp
            FAISS_THROW_IF_NOT_FMT(((static_cast<uint64_t>(ids[i]) >> 41) & 0x1) == 0,
                "time must be positive, ids[%ld]: %ld", i, ids[i]);
        }
    }
}

bool AscendIndexImpl::addImplRequiresIDs() const
{
    // default return true
    return true;
}

void AscendIndexImpl::mergeSearchResult(size_t devices, std::vector<std::vector<float>>& dist,
                                        std::vector<std::vector<ascend_idx_t>>& label, idx_t n, idx_t k,
                                        float* distances, idx_t* labels) const
{
    APP_LOG_INFO("AscendIndex mergeSearchResult operation started.\n");
    std::function<bool(float, float)> compFunc = GetCompFunc();

    // merge several topk results into one topk results
    // every topk result need to be reodered in ascending order
#pragma omp parallel for if (n > 100) num_threads(::ascend::CommonUtils::GetThreadMaxNums())
    for (idx_t i = 0; i < n; i++) {
        idx_t num = 0;
        const idx_t offset = i * k;
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
    APP_LOG_INFO("AscendIndex mergeSearchResult operation finished.\n");
}

std::function<bool(float, float)> AscendIndexImpl::GetCompFunc() const
{
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
    return compFunc;
}

void AscendIndexImpl::searchPostProcess(size_t devices, std::vector<std::vector<float>>& dist,
                                        std::vector<std::vector<ascend_idx_t>>& label, idx_t n, idx_t k,
                                        float* distances, idx_t* labels) const
{
    APP_LOG_INFO("AscendIndex searchPostProcess operation started.\n");
    // transmit idx per device to referenced value
    size_t deviceCnt = this->indexConfig.deviceList.size();
    std::vector<std::vector<ascend_idx_t>> transLabel(deviceCnt, std::vector<ascend_idx_t>(n * k, 0));
    for (size_t i = 0; i < deviceCnt; i++) {
        transform(begin(label[i]), end(label[i]), begin(transLabel[i]), [&](ascend_idx_t temp) {
            return (temp != std::numeric_limits<ascend_idx_t>::max()  && idxDeviceMap[i].size() > 0) ?
                                                                        idxDeviceMap[i].at(temp) :
                                                                        std::numeric_limits<ascend_idx_t>::max();
        });
    }

    if (deviceCnt == 1) {
        transform(transLabel.front().begin(), transLabel.front().end(), labels,
                  [](ascend_idx_t temp) {return idx_t(temp);});
        transform(dist.front().begin(), dist.front().end(), distances, [](float temp) {return temp;});
    } else {
        mergeSearchResult(devices, dist, transLabel, n, k, distances, labels);
    }
    APP_LOG_INFO("AscendIndex searchPostProcess operation end.\n");
}

void AscendIndexImpl::checkParameters(int dims, faiss::MetricType metric,
    AscendIndexConfig, AscendIndex *intf) const
{
    VALUE_UNUSED(metric);
    APP_LOG_INFO("AscendIndex checkParameters start");

    FAISS_THROW_IF_NOT_MSG(intf != nullptr, "Intf is nullptr");

    FAISS_THROW_IF_NOT_FMT(dims > 0 && dims <= MAX_DIM,
        "Invalid number of dimensions(%d), should be (0, %d]", dims, MAX_DIM);
    FAISS_THROW_IF_NOT_FMT(dims % DIM_DIVISOR == 0,
        "Invalid number of dimensions(%d), should be divisible by %d", dims, DIM_DIVISOR);

    FAISS_THROW_IF_NOT_FMT(indexConfig.deviceList.size() > 0 && indexConfig.deviceList.size() <= DEVICE_LIST_SIZE_MAX,
                           "device list should be in range (0, %zu]!", DEVICE_LIST_SIZE_MAX);
    FAISS_THROW_IF_NOT_FMT(indexConfig.resourceSize == -1 ||
                           (indexConfig.resourceSize >= 0 && indexConfig.resourceSize <= INDEX_MAX_MEM),
                           "resourceSize:%ld should be -1 or in range [0, 4096MB]!", indexConfig.resourceSize);

    std::set<int> uniqueDeviceList(indexConfig.deviceList.begin(), indexConfig.deviceList.end());
    if (uniqueDeviceList.size() != indexConfig.deviceList.size()) {
        std::string deviceListStr;
        for (auto id : indexConfig.deviceList) {
            deviceListStr += std::to_string(id) + ",";
        }
        FAISS_THROW_FMT("some device IDs are the same, please check it {%s}", deviceListStr.c_str());
    }

    APP_LOG_INFO("AscendIndex checkParameters finished");
}

void AscendIndexImpl::indexSearch(IndexParam<uint16_t, uint16_t, ascend_idx_t> &param) const
{
    auto index = getActualIndex(param.deviceId);
    using namespace ::ascend;
    AscendTensor<uint16_t, DIMS_2> tensorDevQueries({ param.n, param.dim });
    auto ret = aclrtMemcpy(tensorDevQueries.data(), tensorDevQueries.getSizeInBytes(),
                           param.query, param.n * param.dim * sizeof(uint16_t), ACL_MEMCPY_HOST_TO_DEVICE);
    FAISS_THROW_IF_NOT_FMT(ret == ACL_SUCCESS,  "aclrtMemcpy error %d", ret);
    ret = index->search(param.n, tensorDevQueries.data(), param.k, param.distance,
                        static_cast<::ascend::Index::idx_t *>(param.label));
    FAISS_THROW_IF_NOT_FMT(ret == APP_ERR_OK,
                           "Failed to search index,deviceId is %d, result is: %d\n", param.deviceId, ret);
}

}  // namespace ascend
}  // namespace faiss
