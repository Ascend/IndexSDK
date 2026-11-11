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

#include "AscendIndexFlatImpl.h"

#include <faiss/IndexFlat.h>
#include <faiss/impl/AuxIndexStructures.h>

#include <algorithm>
#include <limits>
#include <optional>

#include "ascend/utils/fp16.h"
#include "ascenddaemon/utils/StaticUtils.h"
#include "common/Constants.h"
#include "common/utils/CommonUtils.h"
#include "index/IndexFlatIPAicpu.h"
#include "index/IndexFlatL2Aicpu.h"

namespace faiss
{
namespace ascend
{
namespace
{
const size_t KB = 1024;
const size_t RETAIN_SIZE = 2048;
// get pagesize must be less than 32M, becauseof rpc limitation
const size_t PAGE_SIZE = 32U * KB * KB - RETAIN_SIZE;

// Or, maximum number 512K of vectors to consider per page of search
const size_t VEC_SIZE = 512U * KB;

// Default dim in case of nullptr index
const size_t DEFAULT_DIM = 512;

// The value range of dim
const std::vector<int> DIMS = {32, 64, 128, 256, 384, 512, 768, 1024, 1408, 1536, 2048, 3072, 3584, 4096};

const size_t UNIT_PAGE_SIZE = 64;
const size_t UNIT_VEC_SIZE = 512;
const size_t ADD_PAGE_SIZE = UNIT_PAGE_SIZE * KB * KB - RETAIN_SIZE;
const size_t ADD_VEC_SIZE = UNIT_VEC_SIZE * KB;
}  // namespace

// implementation of AscendIndexFlatImpl
AscendIndexFlatImpl::AscendIndexFlatImpl(const faiss::IndexFlat *index, AscendIndexFlatConfig config, AscendIndex *intf)
    : AscendIndexImpl((index == nullptr) ? DEFAULT_DIM : index->d,
                      (index == nullptr) ? faiss::METRIC_L2 : index->metric_type, config, intf),
      flatConfig(config),
      deviceSearchMtx(config.deviceList.size())
{
    FAISS_THROW_IF_NOT_MSG(
        config.shard || config.deviceList.size() == 1 || ::ascend::AscendMultiThreadManager::IsMultiThreadMode(),
        "Flat replica mode requires MX_INDEX_MULTITHREAD=1.");
    // Flat index doesn't need training
    this->intf_->is_trained = true;

    ::ascend::AscendMultiThreadManager::InitGetBaseMtx(config.deviceList, getBaseMtx);

    copyFrom(index);
}

AscendIndexFlatImpl::AscendIndexFlatImpl(const faiss::IndexIDMap *index, AscendIndexFlatConfig config,
                                         AscendIndex *intf)
    : AscendIndexImpl((index == nullptr || index->index == nullptr) ? DEFAULT_DIM : index->index->d,
                      (index == nullptr || index->index == nullptr) ? faiss::METRIC_L2 : index->index->metric_type,
                      config, intf),
      flatConfig(config),
      deviceSearchMtx(config.deviceList.size())
{
    FAISS_THROW_IF_NOT_MSG(
        config.shard || config.deviceList.size() == 1 || ::ascend::AscendMultiThreadManager::IsMultiThreadMode(),
        "Flat replica mode requires MX_INDEX_MULTITHREAD=1.");
    FAISS_THROW_IF_NOT_MSG(index != nullptr && index->index != nullptr, "Invalid index nullptr.");
    // Flat index doesn't need training
    this->intf_->is_trained = true;

    ::ascend::AscendMultiThreadManager::InitGetBaseMtx(config.deviceList, getBaseMtx);

    copyFrom(index);
}

AscendIndexFlatImpl::AscendIndexFlatImpl(int dims, faiss::MetricType metric, AscendIndexFlatConfig config,
                                         AscendIndex *intf)
    : AscendIndexImpl(dims, metric, config, intf), flatConfig(config), deviceSearchMtx(config.deviceList.size())
{
    FAISS_THROW_IF_NOT_MSG(
        config.shard || config.deviceList.size() == 1 || ::ascend::AscendMultiThreadManager::IsMultiThreadMode(),
        "Flat replica mode requires MX_INDEX_MULTITHREAD=1.");
    FAISS_THROW_IF_NOT_MSG(metric == MetricType::METRIC_L2 || metric == MetricType::METRIC_INNER_PRODUCT,
                           "Unsupported metric type");
    FAISS_THROW_IF_NOT_MSG(std::find(DIMS.begin(), DIMS.end(), dims) != DIMS.end(), "Unsupported dims");
    // Flat index doesn't need training
    this->intf_->is_trained = true;

    ::ascend::AscendMultiThreadManager::InitGetBaseMtx(config.deviceList, getBaseMtx);

    initIndexes();

    // initial idxDeviceMap mem space
    idxDeviceMap.clear();
    idxDeviceMap.resize(config.deviceList.size());
}

AscendIndexFlatImpl::~AscendIndexFlatImpl() {}

std::shared_ptr<::ascend::Index> AscendIndexFlatImpl::createIndex(int deviceId)
{
    APP_LOG_INFO("AscendIndexFlat createIndex operation start.\n");
    auto res = aclrtSetDevice(deviceId);
    FAISS_THROW_IF_NOT_FMT(res == 0, "acl set device failed %d, deviceId: %d", res, deviceId);
    std::shared_ptr<::ascend::Index> index = nullptr;
    switch (this->intf_->metric_type)
    {
        case MetricType::METRIC_INNER_PRODUCT:
            index = std::make_shared<::ascend::IndexFlatIPAicpu>(this->intf_->d, indexConfig.resourceSize);
            break;
        case MetricType::METRIC_L2:
            index = std::make_shared<::ascend::IndexFlatL2Aicpu>(this->intf_->d, indexConfig.resourceSize);
            break;
        default:
            ASCEND_THROW_MSG("Unsupported metric type\n");
    }
    auto ret = index->init();
    FAISS_THROW_IF_NOT_FMT(ret == ::ascend::APP_ERR_OK, "Failed to create init index: %d", ret);
    APP_LOG_INFO("AscendIndexFlat createIndex operation finished.\n");
    return index;
}

void AscendIndexFlatImpl::CheckIndexParams(IndexImplBase &index, bool checkFilterable) const
{
    try
    {
        AscendIndexFlatImpl &flatIndex = dynamic_cast<AscendIndexFlatImpl &>(index);

        FAISS_THROW_IF_NOT_MSG(this->intf_->metric_type == flatIndex.intf_->metric_type,
                               "the metric type must be same");
        FAISS_THROW_IF_NOT_MSG(this->intf_->d == flatIndex.intf_->d, "the dim must be same.");
        FAISS_THROW_IF_NOT_MSG(this->flatConfig.deviceList == flatIndex.flatConfig.deviceList,
                               "the deviceList must be same.");
        FAISS_THROW_IF_NOT_MSG(this->flatConfig.shard == flatIndex.flatConfig.shard, "the shard mode must be same.");
        FAISS_THROW_IF_NOT_FMT(flatIndex.flatConfig.deviceList.size() == 1, "the number of deviceList (%zu) must be 1.",
                               flatIndex.flatConfig.deviceList.size());
        if (checkFilterable)
        {
            FAISS_THROW_IF_NOT_MSG(flatConfig.filterable, "the index does not support filterable");
        }
    }
    catch (std::bad_cast &e)
    {
        FAISS_THROW_MSG("the type of index is not same, or the number of deviceList is not 1.");
    }
}

void AscendIndexFlatImpl::copyFrom(const faiss::IndexFlat *index)
{
    auto lock = ::ascend::AscendMultiThreadManager::GetWriteLock(mtx);
    APP_LOG_INFO("AscendIndexFlat copyFrom operation started.\n");
    FAISS_THROW_IF_NOT_MSG(index != nullptr, "Invalid index nullptr.");
    FAISS_THROW_IF_NOT_MSG(
        index->metric_type == MetricType::METRIC_L2 || index->metric_type == MetricType::METRIC_INNER_PRODUCT,
        "Unsupported metric type");
    FAISS_THROW_IF_NOT_MSG(std::find(DIMS.begin(), DIMS.end(), index->d) != DIMS.end(), "Unsupported dims");
    FAISS_THROW_IF_NOT_FMT((index->ntotal >= 0) && (index->ntotal < MAX_N), "ntotal must be >= 0 and < %ld", MAX_N);

    this->intf_->ntotal = 0;
    clearIndexes();

    this->intf_->d = index->d;
    this->intf_->metric_type = index->metric_type;

    initIndexes();

    // initial idxDeviceMap mem space
    idxDeviceMap.clear();
    idxDeviceMap.resize(indexConfig.deviceList.size());

    copyCode(index);
    APP_LOG_INFO("AscendIndexFlat copyFrom operation finished.\n");
}

void AscendIndexFlatImpl::copyFrom(const faiss::IndexIDMap *index)
{
    auto lock = ::ascend::AscendMultiThreadManager::GetWriteLock(mtx);
    APP_LOG_INFO("AscendIndexFlat copyFrom operation started.\n");
    FAISS_THROW_IF_NOT_MSG(index != nullptr && index->index != nullptr, "Invalid index nullptr.");
    FAISS_THROW_IF_NOT_MSG(index->index->metric_type == MetricType::METRIC_L2 ||
                               index->index->metric_type == MetricType::METRIC_INNER_PRODUCT,
                           "Unsupported metric type");
    FAISS_THROW_IF_NOT_MSG(std::find(DIMS.begin(), DIMS.end(), index->index->d) != DIMS.end(), "Unsupported dims");
    FAISS_THROW_IF_NOT_FMT((index->index->ntotal >= 0) && (index->index->ntotal) < MAX_N,
                           "ntotal must be >= 0 and < %ld", MAX_N);
    auto flatPtr = dynamic_cast<faiss::IndexFlat *>(index->index);
    FAISS_THROW_IF_NOT_MSG(flatPtr != nullptr, "Invalid flatIndex nullptr.");

    this->intf_->ntotal = 0;

    clearIndexes();

    this->intf_->d = index->index->d;
    this->intf_->metric_type = index->index->metric_type;

    initIndexes();
    // initial idxDeviceMap mem space
    idxDeviceMap.clear();
    idxDeviceMap.resize(indexConfig.deviceList.size());
    if (index->id_map.data() != nullptr)
    {
        FAISS_THROW_IF_NOT_MSG(index->id_map.size() == static_cast<size_t>(flatPtr->ntotal),
                               "The size of id_map must be equal to ntotal.\n");
    }
    copyCode(flatPtr, index->id_map.data());
    APP_LOG_INFO("AscendIndexFlat copyFrom operation finished.\n");
}

void AscendIndexFlatImpl::copyCode(const faiss::IndexFlat *index, const idx_t *ids)
{
    APP_LOG_INFO("AscendIndexFlat copyCode operation started.\n");
    if (index->ntotal == 0)
    {
        return;
    }
    FAISS_THROW_IF_NOT_MSG(
        index->codes.size() / sizeof(float) == static_cast<size_t>(index->ntotal) * static_cast<size_t>(this->intf_->d),
        "The size of xb must be equal to ntotal * dim.\n");
    if (ids == nullptr)
    {
        // set ids
        std::vector<idx_t> idsInner(index->ntotal);
        for (size_t i = 0; i < idsInner.size(); ++i)
        {
            idsInner[i] = this->intf_->ntotal + static_cast<idx_t>(i);
        }
        add2DeviceFast(index->ntotal, const_cast<float *>(index->get_xb()), idsInner.data());
    }
    else
    {
        add2DeviceFast(index->ntotal, const_cast<float *>(index->get_xb()), ids);
    }
    APP_LOG_INFO("AscendIndexFlat copyCode operation finished.\n");
}

void AscendIndexFlatImpl::add2DeviceFast(int n, const float *codes, const idx_t *ids)
{
    FAISS_THROW_IF_NOT(n > 0);
    auto codeFp16 = new (std::nothrow) uint16_t[static_cast<size_t>(n) * static_cast<size_t>(this->intf_->d)];
    FAISS_THROW_IF_NOT_MSG(codeFp16 != nullptr, "Memory allocation fail for codeFp16");
    std::shared_ptr<uint16_t> transCodes(codeFp16, std::default_delete<uint16_t[]>());
    const int transBatch = 16384;
    int loops = (n + transBatch - 1) / transBatch;

#pragma omp parallel for if (loops >= 10) num_threads(::ascend::CommonUtils::GetThreadMaxNums())
    for (int i = 0; i < loops; i++)
    {
        int transNum = (i < loops - 1) ? transBatch : (n - i * transBatch);
        size_t offset = static_cast<size_t>(i * transBatch) * static_cast<size_t>(this->intf_->d);
        std::transform(codes + offset,
                       codes + offset + static_cast<size_t>(transNum) * static_cast<size_t>(this->intf_->d),
                       transCodes.get() + offset, [](float temp) { return fp16(temp).data; });
    }

    add2DeviceFastFp16(n, transCodes.get(), ids);
}

void AscendIndexFlatImpl::add2DeviceFastFp16(int n, const uint16_t *codes, const idx_t *ids)
{
    APP_LOG_INFO("AscendIndexFlat add2DeviceFastFp16 operation started.\n");
    FAISS_THROW_IF_NOT(n > 0);

    size_t totalSize = static_cast<size_t>(n) * getAddElementSize();
    if (totalSize > ADD_PAGE_SIZE || static_cast<size_t>(n) > ADD_VEC_SIZE)
    {
        size_t tileSize = getAddPagedSize(n);

        for (size_t i = 0; i < static_cast<size_t>(n); i += tileSize)
        {
            size_t curNum = std::min(tileSize, n - i);
            if (this->intf_->verbose)
            {
                printf("AscendIndex::add: adding %zu:%zu / %d\n", i, i + curNum, n);
            }
            add2DeviceFastImpl(curNum, codes + i * static_cast<size_t>(this->intf_->d), ids + i);
        }
    }
    else
    {
        if (this->intf_->verbose)
        {
            printf("AscendIndex::add: adding 0:%d / %d\n", n, n);
        }
        add2DeviceFastImpl(n, codes, ids);
    }
    APP_LOG_INFO("AscendIndexFlat add2DeviceFastFp16 operation finished.\n");
}

void AscendIndexFlatImpl::add2DeviceFastImpl(int n, const uint16_t *transCodes, const idx_t *ids)
{
    APP_LOG_INFO("AscendIndexFlat add2DeviceFastImpl operation started.\n");
    FAISS_THROW_IF_NOT(n > 0);
    size_t deviceCnt = indexConfig.deviceList.size();
    size_t oldSize = idxDeviceMap.front().size();
    std::vector<int> addMap(deviceCnt, 0);
    if (flatConfig.shard)
    {
        calcAddMap(n, addMap);
    }
    else
    {
        std::fill(addMap.begin(), addMap.end(), n);
    }

    int offsum = 0;
    std::vector<int> offsumMap(deviceCnt, 0);
    for (size_t i = 0; i < deviceCnt; i++)
    {
        offsumMap[i] = flatConfig.shard ? offsum : 0;
        int num = addMap.at(i);
        offsum += num;
    }

    auto addFunctor = [&](int idx)
    {
        int num = addMap.at(idx);
        if (num != 0)
        {
            int deviceId = indexConfig.deviceList[idx];
            auto pIndex = getActualIndex(deviceId);
            ::ascend::AscendTensor<float16_t, ::ascend::DIMS_2> vec(
                const_cast<uint16_t *>(transCodes) +
                    static_cast<size_t>(offsumMap[idx]) * static_cast<size_t>(this->intf_->d),
                {num, this->intf_->d});
            auto ret = pIndex->addVectors(vec);
            FAISS_THROW_IF_NOT_FMT(ret == ::ascend::APP_ERR_OK, "Failed to add to index, result is: %d", ret);
            idxDeviceMap[idx].insert(idxDeviceMap[idx].end(), ids + offsumMap[idx], ids + num + offsumMap[idx]);
        }
    };

    CALL_PARALLEL_FUNCTOR(deviceCnt, pool, addFunctor);
    updateIdentityIds(oldSize);
    this->intf_->ntotal += n;
    APP_LOG_INFO("AscendIndexFlat add2DeviceFastImpl operation finished.\n");
}

void AscendIndexFlatImpl::copyTo(faiss::IndexFlat *index) const
{
    auto lock = ::ascend::AscendMultiThreadManager::GetReadLock(mtx);
    APP_LOG_INFO("AscendIndexFlat copyTo operation started.\n");
    FAISS_THROW_IF_NOT(index != nullptr);
    index->reset();
    index->d = this->intf_->d;
    index->ntotal = this->intf_->ntotal;
    index->metric_type = this->intf_->metric_type;
    index->code_size = static_cast<size_t>(this->intf_->d) * sizeof(float);

    if (this->intf_->is_trained && this->intf_->ntotal > 0)
    {
        size_t deviceCnt = flatConfig.shard ? indexConfig.deviceList.size() : 1;
        std::vector<size_t> getOffset(deviceCnt, 0);
        size_t total = 0;
        for (size_t i = 0; i < deviceCnt; i++)
        {
            getOffset[i] = total;
            total += getBaseSize(indexConfig.deviceList[i]);
        }

        std::vector<uint8_t> baseData(total * static_cast<size_t>(this->intf_->d) * sizeof(float));
        auto getFunctor = [this, &baseData, &getOffset](int idx)
        {
            getBase(this->indexConfig.deviceList[idx],
                    reinterpret_cast<char *>(baseData.data() +
                                             getOffset[idx] * static_cast<size_t>(this->intf_->d) * sizeof(float)));
        };
        CALL_PARALLEL_FUNCTOR(deviceCnt, pool, getFunctor);

        index->codes = std::move(baseData);
    }
    APP_LOG_INFO("AscendIndexFlat copyTo operation finished.\n");
}

void AscendIndexFlatImpl::copyTo(faiss::IndexIDMap *index) const
{
    auto lock = ::ascend::AscendMultiThreadManager::GetReadLock(mtx);
    APP_LOG_INFO("AscendIndexFlat copyTo operation started.\n");
    FAISS_THROW_IF_NOT_MSG(index != nullptr, "Invalid index nullptr.");
    index->d = this->intf_->d;
    copyTo(dynamic_cast<faiss::IndexFlat *>(index->index));
    index->ntotal = this->intf_->ntotal;

    std::vector<idx_t> idsFlat;
    for (auto device : indexConfig.deviceList)
    {
        std::vector<idx_t> ids;
        getIdxMap(device, ids);
        idsFlat.insert(idsFlat.end(), ids.begin(), ids.end());
        if (!flatConfig.shard) break;
    }
    index->id_map = std::move(idsFlat);
    APP_LOG_INFO("AscendIndexFlat copyTo operation finished.\n");
}

size_t AscendIndexFlatImpl::getBaseSize(int deviceId) const
{
    auto lock = ::ascend::AscendMultiThreadManager::GetReadLock(mtx);
    APP_LOG_INFO("AscendIndexFlat start to getBaseSize in deivce %d.\n", deviceId);
    uint32_t size = 0;
    auto pIndex = getActualIndex(deviceId);
    size = static_cast<uint32_t>(pIndex->getSize());
    APP_LOG_INFO("AscendIndexFlat getBaseSize finished.\n");
    return static_cast<size_t>(size);
}

void AscendIndexFlatImpl::getIdxMap(int deviceId, std::vector<idx_t> &idxMap) const
{
    auto lock = ::ascend::AscendMultiThreadManager::GetReadLock(mtx);
    APP_LOG_INFO("AscendIndexFlat getIdxMap operation started.\n");
    AscendIndexImpl::getIdxMap(deviceId, idxMap);
    APP_LOG_INFO("AscendIndexFlat getIdxMap operation finished.\n");
}

void AscendIndexFlatImpl::getBase(int deviceId, char *xb) const
{
    auto lock = ::ascend::AscendMultiThreadManager::GetReadLock(mtx);
    // getBase、getBaseEnd接口内部会使用修改成员变量dataVec、attrsVec，因此getBase接口不能并发
    auto getBaseLock = ::ascend::AscendMultiThreadManager::LockGetBaseMtx(deviceId, getBaseMtx);
    APP_LOG_INFO("AscendIndexFlat getBase operation started.\n");
    AscendIndexImpl::getBase(deviceId, xb);
    getActualIndex(deviceId)->getBaseEnd();
    APP_LOG_INFO("AscendIndexFlat getBase operation finished.\n");
}

void AscendIndexFlatImpl::getBaseImpl(int deviceId, int offset, int n, char *x) const
{
    APP_LOG_INFO("AscendIndexFlat getBaseImpl operation started.\n");
    auto pIndex = getActualIndex(deviceId);
    std::vector<uint16_t> baseData(n);
    auto ret = pIndex->getVectors(offset, n, baseData);
    FAISS_THROW_IF_NOT_FMT(ret == ::ascend::APP_ERR_OK, "Failed to get vector index, res is %d", ret);

    auto xi = reinterpret_cast<float *>(x) + static_cast<size_t>(offset) * static_cast<size_t>(this->intf_->d);
    transform(baseData.begin(), baseData.end(), xi, [](uint16_t temp) { return static_cast<float>(fp16(temp)); });
    APP_LOG_INFO("AscendIndexFlat getBaseImpl operation finished.\n");
}

size_t AscendIndexFlatImpl::getBaseElementSize() const
{
    return static_cast<size_t>(this->intf_->d) * sizeof(uint16_t);
}

void AscendIndexFlatImpl::addPaged(int n, const float *x, const idx_t *ids)
{
    APP_LOG_INFO("AscendIndexFlat addPaged operation started n: %d.\n", n);
    add2DeviceFast(n, x, ids);
    APP_LOG_INFO("AscendIndexFlat addPaged operation finished.\n");
}

void AscendIndexFlatImpl::addPaged(int n, const uint16_t *x, const idx_t *ids)
{
    APP_LOG_INFO("AscendIndexFlat addPaged operation started n: %d.\n", n);
    add2DeviceFastFp16(n, x, ids);
    APP_LOG_INFO("AscendIndexFlat addPaged operation finished.\n");
}

void AscendIndexFlatImpl::addImpl(int n, const float *x, const idx_t *ids)
{
    APP_LOG_INFO("AscendIndexFlat addImpl operation started with %d vector(s).\n", n);
    FAISS_THROW_IF_NOT(n > 0);
    if (!flatConfig.shard)
    {
        add2DeviceFast(n, x, ids);
        return;
    }

    size_t deviceCnt = indexConfig.deviceList.size();
    FAISS_THROW_IF_NOT(deviceCnt > 0);
    size_t oldSize = idxDeviceMap.front().size();
    std::vector<int> addMap(deviceCnt, 0);
    calcAddMap(n, addMap);

    uint32_t offsum = 0;
    for (size_t i = 0; i < deviceCnt; i++)
    {
        int num = addMap.at(i);
        if (num == 0)
        {
            continue;
        }
        int deviceId = indexConfig.deviceList[i];

        std::vector<uint16_t> xb(num * this->intf_->d);
        transform(x + offsum * this->intf_->d, x + (offsum + num) * this->intf_->d, std::begin(xb),
                  [](float temp) { return fp16(temp).data; });
        using namespace ::ascend;
        auto pIndex = getActualIndex(deviceId);
        AscendTensor<float16_t, DIMS_2> vec(xb.data(), {num, this->intf_->d});
        auto ret = pIndex->addVectors(vec);
        FAISS_THROW_IF_NOT_FMT(ret == APP_ERR_OK, "Failed to add to index, result is: %d", ret);

        // record ids of new adding vector
        idxDeviceMap[i].insert(idxDeviceMap[i].end(), ids + offsum, ids + num + offsum);
        offsum += static_cast<uint32_t>(num);
    }
    updateIdentityIds(oldSize);
    this->intf_->ntotal += n;
    APP_LOG_INFO("AscendIndexFlat addImpl operation finished.\n");
}

void AscendIndexFlatImpl::updateIdentityIds(size_t start)
{
    const auto &ids = idxDeviceMap.front();
    if (start == 0) identityIds = true;
    if (!identityIds) return;
    for (size_t p = start; p < ids.size(); ++p)
    {
        if (ids[p] != static_cast<idx_t>(p))
        {
            identityIds = false;
            break;
        }
    }
}

size_t AscendIndexFlatImpl::removeImpl(const IDSelector &sel)
{
    size_t removed = AscendIndexImpl::removeImpl(sel);
    updateIdentityIds();
    return removed;
}

size_t AscendIndexFlatImpl::nextDeviceIndex() const
{
    return nextDevice.fetch_add(1, std::memory_order_relaxed) % indexConfig.deviceList.size();
}

size_t AscendIndexFlatImpl::getLogicalRemoveCount(size_t removed) const
{
    if (flatConfig.shard) return removed;
    FAISS_THROW_IF_NOT_MSG(removed % indexConfig.deviceList.size() == 0,
                           "Flat replicas removed different numbers of vectors.");
    for (size_t i = 1; i < idxDeviceMap.size(); ++i)
    {
        FAISS_THROW_IF_NOT_MSG(idxDeviceMap[i] == idxDeviceMap.front(), "Flat replica ID maps differ before removal.");
    }
    return removed / indexConfig.deviceList.size();
}

void AscendIndexFlatImpl::searchImpl(int n, const uint16_t *x, int k, float *distances, idx_t *labels) const
{
    if (flatConfig.shard)
    {
        AscendIndexImpl::searchImpl(n, x, k, distances, labels);
        return;
    }
    size_t idx = nextDeviceIndex();
    int deviceId = indexConfig.deviceList[idx];
    std::vector<uint16_t> distHalf(n * k);
    std::vector<ascend_idx_t> localLabels(n * k);
    IndexParam<uint16_t, uint16_t, ascend_idx_t> param(deviceId, n, this->intf_->d, k);
    param.query = x;
    param.distance = distHalf.data();
    param.label = localLabels.data();
    {
        std::lock_guard<std::mutex> lock(deviceSearchMtx[idx]);
        indexSearch(param);
    }
    for (int i = 0; i < n * k; ++i)
    {
        distances[i] = static_cast<float>(fp16(distHalf[i]));
        labels[i] = localLabels[i] == std::numeric_limits<ascend_idx_t>::max() || idxDeviceMap[idx].empty()
                        ? -1
                        : idxDeviceMap[idx].at(localLabels[i]);
    }
}

void AscendIndexFlatImpl::search_with_masks(idx_t n, const float *x, idx_t k, float *distances, idx_t *labels,
                                            const void *mask) const
{
    auto readLock = flatConfig.shard ? std::optional<std::shared_lock<std::shared_mutex>>()
                                     : ::ascend::AscendMultiThreadManager::GetReadLock(mtx);
    auto writeLock = flatConfig.shard ? ::ascend::AscendMultiThreadManager::GetWriteLock(mtx)
                                      : std::optional<std::unique_lock<std::shared_mutex>>();
    APP_LOG_INFO("AscendIndexFlat search_with_masks operation started: searchNum=%ld, topK=%ld.\n", n, k);
    check(n, x, k, distances, labels);
    FAISS_THROW_IF_NOT_MSG(mask, "mask cannot be nullptr.");

    // convert query data from float to fp16, device use fp16 data to search
    std::vector<float16_t> query(n * this->intf_->d, 0);
    transform(x, x + n * this->intf_->d, begin(query), [](float temp) { return fp16(temp).data; });

    searchWithMaskProcess(n, query.data(), k, distances, labels, mask);
    APP_LOG_INFO("AscendIndexFlat search_with_masks operation finished.\n");
}

void AscendIndexFlatImpl::search_with_masks_fp16(idx_t n, const uint16_t *x, idx_t k, float *distances, idx_t *labels,
                                                 const void *mask) const
{
    auto readLock = flatConfig.shard ? std::optional<std::shared_lock<std::shared_mutex>>()
                                     : ::ascend::AscendMultiThreadManager::GetReadLock(mtx);
    auto writeLock = flatConfig.shard ? ::ascend::AscendMultiThreadManager::GetWriteLock(mtx)
                                      : std::optional<std::unique_lock<std::shared_mutex>>();
    APP_LOG_INFO("AscendIndexFlat search_with_masks_fp16 operation started: searchNum=%ld, topK=%ld.\n", n, k);
    check(n, x, k, distances, labels);
    FAISS_THROW_IF_NOT_MSG(mask, "mask cannot be nullptr.");

    searchWithMaskProcess(n, x, k, distances, labels, mask);
    APP_LOG_INFO("AscendIndexFlat search_with_masks_fp16 operation finished.\n");
}

void AscendIndexFlatImpl::searchWithMaskProcess(idx_t n, const uint16_t *x, idx_t k, float *distances, idx_t *labels,
                                                const void *mask) const
{
    if (!flatConfig.shard)
    {
        size_t idx = nextDeviceIndex();
        std::vector<float16_t> distHalf(n * k);
        std::vector<ascend_idx_t> localLabels(n * k);
        {
            std::lock_guard<std::mutex> lock(deviceSearchMtx[idx]);
            auto pIndex = getActualIndex(indexConfig.deviceList[idx]);
            ::ascend::AscendTensor<float16_t, ::ascend::DIMS_2, int64_t> queries({n, this->intf_->d});
            auto ret = aclrtMemcpy(queries.data(), queries.getSizeInBytes(), x, n * this->intf_->d * sizeof(float16_t),
                                   ACL_MEMCPY_HOST_TO_DEVICE);
            FAISS_THROW_IF_NOT_FMT(ret == ACL_SUCCESS, "aclrtMemcpy error %d", ret);
            ret = pIndex->searchFilter(n, queries.data(), k, distHalf.data(), localLabels.data(),
                                       static_cast<uint8_t *>(const_cast<void *>(mask)));
            FAISS_THROW_IF_NOT_FMT(ret == ::ascend::APP_ERR_OK, "Failed to searchfilter index, result is: %d\n", ret);
        }
        for (idx_t i = 0; i < n * k; ++i)
        {
            distances[i] = static_cast<float>(fp16(distHalf[i]));
            labels[i] = localLabels[i] == std::numeric_limits<ascend_idx_t>::max() || idxDeviceMap[idx].empty()
                            ? -1
                            : idxDeviceMap[idx].at(localLabels[i]);
        }
        return;
    }
    size_t deviceCnt = indexConfig.deviceList.size();

    std::vector<std::vector<float>> dist(deviceCnt, std::vector<float>(n * k, 0));
    std::vector<std::vector<float16_t>> distHalf(deviceCnt, std::vector<float16_t>(n * k, 0));
    std::vector<std::vector<ascend_idx_t>> label(deviceCnt, std::vector<ascend_idx_t>(n * k, 0));

    auto searchFunctor = [&](int idx)
    {
        int deviceId = indexConfig.deviceList[idx];
        std::lock_guard<std::mutex> lock(deviceSearchMtx[idx]);
        auto pIndex = getActualIndex(deviceId);
        using namespace ::ascend;
        AscendTensor<float16_t, DIMS_2, int64_t> tensorDevQueries({n, this->intf_->d});
        auto ret = aclrtMemcpy(tensorDevQueries.data(), tensorDevQueries.getSizeInBytes(), x,
                               n * this->intf_->d * sizeof(float16_t), ACL_MEMCPY_HOST_TO_DEVICE);
        FAISS_THROW_IF_NOT_FMT(ret == ACL_SUCCESS, "aclrtMemcpy error %d", ret);

        ret = pIndex->searchFilter(n, tensorDevQueries.data(), k, distHalf[idx].data(), label[idx].data(),
                                   static_cast<uint8_t *>(const_cast<void *>(mask)));
        FAISS_THROW_IF_NOT_FMT(ret == APP_ERR_OK, "Failed to searchfilter index, result is: %d\n", ret);

        // convert result data from fp16 to float
        transform(begin(distHalf[idx]), end(distHalf[idx]), begin(dist[idx]),
                  [](float16_t temp) { return static_cast<float>(fp16(temp)); });
    };
    CALL_PARALLEL_FUNCTOR(deviceCnt, pool, searchFunctor);

    searchPostProcess(deviceCnt, dist, label, n, k, distances, labels);
}

size_t AscendIndexFlatImpl::getAddElementSize() const { return static_cast<size_t>(this->intf_->d) * sizeof(uint16_t); }

void AscendIndexFlatImpl::selectorToMask(const faiss::IDSelector *sel, idx_t n, std::vector<uint8_t> &mask) const
{
    FAISS_THROW_IF_NOT_MSG(!flatConfig.shard || indexConfig.deviceList.size() == 1,
                           "multi-device search_with_selector requires shard=false.");
    // One selector is shared by the whole batch; mask bits follow base positions, LSB first.
    size_t maskSize = static_cast<size_t>(::ascend::utils::divUp(this->intf_->ntotal, BINARY_BYTE_SIZE));
    FAISS_THROW_IF_NOT_MSG(maskSize > 0, "search_with_selector requires a non-empty index.");
    mask.assign(maskSize * static_cast<size_t>(n), 0);
    const auto &ids = idxDeviceMap.front();
    const auto *bitmap = dynamic_cast<const faiss::IDSelectorBitmap *>(sel);
    if (bitmap != nullptr && identityIds)
    {
        size_t copySize = std::min(maskSize, bitmap->n);
        if (copySize > 0)
        {
            std::copy_n(bitmap->bitmap, copySize, mask.begin());
        }
        if (this->intf_->ntotal % BINARY_BYTE_SIZE != 0)
        {
            mask[maskSize - 1] &= static_cast<uint8_t>((1u << (this->intf_->ntotal % BINARY_BYTE_SIZE)) - 1);
        }
    }
    else
    {
        for (size_t p = 0; p < ids.size(); p++)
        {
            bool selected;
            if (bitmap != nullptr)
            {
                idx_t id = ids[p];
                selected = id >= 0 && static_cast<uint64_t>(id) / BINARY_BYTE_SIZE < bitmap->n &&
                           (bitmap->bitmap[id / BINARY_BYTE_SIZE] & (1u << (id % BINARY_BYTE_SIZE))) != 0;
            }
            else
            {
                selected = sel->is_member(ids[p]);
            }
            if (selected)
            {
                mask[p / BINARY_BYTE_SIZE] |= static_cast<uint8_t>(1u << (p % BINARY_BYTE_SIZE));
            }
        }
    }
    for (idx_t q = 1; q < n; ++q)
    {
        std::copy_n(mask.begin(), maskSize, mask.begin() + static_cast<size_t>(q) * maskSize);
    }
}

void AscendIndexFlatImpl::search_with_selector(idx_t n, const float *x, idx_t k, float *distances, idx_t *labels,
                                               const faiss::IDSelector *sel) const
{
    auto readLock = flatConfig.shard ? std::optional<std::shared_lock<std::shared_mutex>>()
                                     : ::ascend::AscendMultiThreadManager::GetReadLock(mtx);
    auto writeLock = flatConfig.shard ? ::ascend::AscendMultiThreadManager::GetWriteLock(mtx)
                                      : std::optional<std::unique_lock<std::shared_mutex>>();
    APP_LOG_INFO("AscendIndexFlat search_with_selector operation started: searchNum=%ld, topK=%ld.\n", n, k);
    check(n, x, k, distances, labels);
    FAISS_THROW_IF_NOT_MSG(sel, "selector cannot be nullptr.");

    // convert query data from float to fp16, device use fp16 data to search
    std::vector<uint16_t> query(n * this->intf_->d, 0);
    transform(x, x + n * this->intf_->d, begin(query), [](float temp) { return fp16(temp).data; });

    std::vector<uint8_t> mask;
    selectorToMask(sel, n, mask);
    searchWithMaskProcess(n, query.data(), k, distances, labels, mask.data());
    APP_LOG_INFO("AscendIndexFlat search_with_selector operation finished.\n");
}

void AscendIndexFlatImpl::search_with_selector_fp16(idx_t n, const uint16_t *x, idx_t k, float *distances,
                                                    idx_t *labels, const faiss::IDSelector *sel) const
{
    auto readLock = flatConfig.shard ? std::optional<std::shared_lock<std::shared_mutex>>()
                                     : ::ascend::AscendMultiThreadManager::GetReadLock(mtx);
    auto writeLock = flatConfig.shard ? ::ascend::AscendMultiThreadManager::GetWriteLock(mtx)
                                      : std::optional<std::unique_lock<std::shared_mutex>>();
    APP_LOG_INFO("AscendIndexFlat search_with_selector_fp16 operation started: searchNum=%ld, topK=%ld.\n", n, k);
    check(n, x, k, distances, labels);
    FAISS_THROW_IF_NOT_MSG(sel, "selector cannot be nullptr.");

    std::vector<uint8_t> mask;
    selectorToMask(sel, n, mask);
    searchWithMaskProcess(n, x, k, distances, labels, mask.data());
    APP_LOG_INFO("AscendIndexFlat search_with_selector_fp16 operation finished.\n");
}
}  // namespace ascend
}  // namespace faiss
