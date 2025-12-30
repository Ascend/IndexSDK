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


#include "AscendIndexInt8FlatImpl.h"
#include <algorithm>
#include <unordered_set>

#include "AuxIndexStructures.h"
#include "IndexInt8Flat.h"
#include "IndexInt8FlatCosAicpu.h"
#include "IndexInt8FlatL2Aicpu.h"
#include "index_custom/IndexInt8FlatApproxL2Aicpu.h"


namespace faiss {
namespace ascend {
namespace {
const size_t KB = 1024;
const size_t RETAIN_SIZE = 2048;
const size_t UNIT_PAGE_SIZE = 64;
const size_t UNIT_VEC_SIZE = 512;
const size_t ADD_PAGE_SIZE = UNIT_PAGE_SIZE * KB * KB - RETAIN_SIZE;
const size_t ADD_VEC_SIZE = UNIT_VEC_SIZE * KB;
constexpr uint32_t MAX_PAGE_BLOCK_NUM = 144;

// get pagesize must be less than 32M, because of rpc limitation
const size_t PAGE_SIZE = 32U * KB * KB - RETAIN_SIZE;

// Or, maximum number 512K of vectors to consider per page of search
const size_t VEC_SIZE = 512U * KB;

// Default dim in case of nullptr index
const size_t DEFAULT_DIM = 512;

// Search host pipeline
const size_t SEARCH_PAGE_SIZE = 1024;

// The value range of dim
const std::vector<int> DIMS = { 64, 128, 256, 384, 512, 768, 1024 };

// The value range of blockSzie
const std::vector<int> BLOCKSIZES = { 16384, 32768, 65536, 131072, 262144 };
} // namespace

// implementation of AscendIndexFlat
AscendIndexInt8FlatImpl::AscendIndexInt8FlatImpl(int dims, faiss::MetricType metric,
    AscendIndexInt8FlatConfig config, AscendIndexInt8 *intf)
    : AscendIndexInt8Impl(dims, metric, config, intf), int8FlatConfig(config)
{
    FAISS_THROW_IF_NOT_MSG(metric == MetricType::METRIC_L2 || metric == MetricType::METRIC_INNER_PRODUCT,
        "Unsupported metric type");
    FAISS_THROW_IF_NOT_FMT(std::find(DIMS.begin(), DIMS.end(), dims) != DIMS.end(), "Unsupported dims %d", dims);
    FAISS_THROW_IF_NOT_FMT(std::find(BLOCKSIZES.begin(), BLOCKSIZES.end(), config.dBlockSize) != BLOCKSIZES.end(),
        " Unsupported blockSize %u! ", config.dBlockSize);
    FAISS_THROW_IF_NOT_MSG(config.dIndexMode != Int8IndexMode::WITHOUT_NORM_MODE,
        "Unsupported dIndexMode WITHOUT_NORM_MODE");
    FAISS_THROW_IF_NOT_MSG(config.dIndexMode == Int8IndexMode::DEFAULT_MODE ||
        (config.dIndexMode == Int8IndexMode::PIPE_SEARCH_MODE && metric == MetricType::METRIC_L2),
        "Unsupported metric type, should be METRIC_L2");

    ::ascend::AscendMultiThreadManager::InitGetBaseMtx(config.deviceList, getBaseMtx);

    // Flat index doesn't need training
    trained = true;

    pInt8PipeSearchImpl = std::unique_ptr<AscendInt8PipeSearchImpl>(
        new AscendInt8PipeSearchImpl(this->intf_, int8FlatConfig));
    initIndexes();

    // initial idxDeviceMap mem space
    idxDeviceMap.clear();
    idxDeviceMap.resize(config.deviceList.size());
    label2Idx.clear();
    label2Idx.resize(config.deviceList.size());
}

AscendIndexInt8FlatImpl::AscendIndexInt8FlatImpl(const faiss::IndexScalarQuantizer *index,
                                                 AscendIndexInt8FlatConfig config, AscendIndexInt8 *intf)
    : AscendIndexInt8Impl((index == nullptr) ? DEFAULT_DIM : index->d,
                          (index == nullptr) ? faiss::METRIC_L2 : index->metric_type,
                          config,
                          intf),
      int8FlatConfig(config)
{
    FAISS_THROW_IF_NOT_MSG(index != nullptr, "Invalid index nullptr.");
    FAISS_THROW_IF_NOT_FMT(std::find(BLOCKSIZES.begin(), BLOCKSIZES.end(), config.dBlockSize) != BLOCKSIZES.end(),
        " Unsupported blockSize %u!", config.dBlockSize);
    FAISS_THROW_IF_NOT_MSG(config.dIndexMode != Int8IndexMode::WITHOUT_NORM_MODE,
        "Unsupported dIndexMode WITHOUT_NORM_MODE");
    FAISS_THROW_IF_NOT_MSG(config.dIndexMode == Int8IndexMode::DEFAULT_MODE ||
        (config.dIndexMode == Int8IndexMode::PIPE_SEARCH_MODE && index->metric_type == MetricType::METRIC_L2),
        "Unsupported metric type, should be METRIC_L2");
    pInt8PipeSearchImpl = std::unique_ptr<AscendInt8PipeSearchImpl>(
        new AscendInt8PipeSearchImpl(this->intf_, int8FlatConfig));
    ::ascend::AscendMultiThreadManager::InitGetBaseMtx(config.deviceList, getBaseMtx);
    copyFrom(index);
}

AscendIndexInt8FlatImpl::AscendIndexInt8FlatImpl(const faiss::IndexIDMap *index,
                                                 AscendIndexInt8FlatConfig config, AscendIndexInt8 *intf)
    : AscendIndexInt8Impl((index == nullptr || index->index == nullptr) ? DEFAULT_DIM : index->index->d,
                          (index == nullptr || index->index == nullptr) ? faiss::METRIC_L2 : index->index->metric_type,
                          config,
                          intf),
      int8FlatConfig(config)
{
    FAISS_THROW_IF_NOT_FMT(std::find(BLOCKSIZES.begin(), BLOCKSIZES.end(), config.dBlockSize) != BLOCKSIZES.end(),
        " Unsupported blockSize %u!", config.dBlockSize);
    FAISS_THROW_IF_NOT_MSG(index != nullptr, "Invalid index nullptr.");
    FAISS_THROW_IF_NOT_MSG(index->index != nullptr, "Invalid index nullptr.");
    FAISS_THROW_IF_NOT_MSG(config.dIndexMode != Int8IndexMode::WITHOUT_NORM_MODE,
        "Unsupported dIndexMode WITHOUT_NORM_MODE");
    FAISS_THROW_IF_NOT_MSG(config.dIndexMode == Int8IndexMode::DEFAULT_MODE ||
        (config.dIndexMode == Int8IndexMode::PIPE_SEARCH_MODE && index->index->metric_type == MetricType::METRIC_L2),
        "Unsupported metric type, should be METRIC_L2");
    pInt8PipeSearchImpl = std::unique_ptr<AscendInt8PipeSearchImpl>(
        new AscendInt8PipeSearchImpl(this->intf_, int8FlatConfig));
    FAISS_THROW_IF_NOT_MSG(index != nullptr && index->index != nullptr, "Invalid index nullptr.");
    ::ascend::AscendMultiThreadManager::InitGetBaseMtx(config.deviceList, getBaseMtx);
    copyFrom(index);
}

AscendIndexInt8FlatImpl::~AscendIndexInt8FlatImpl()
{
    clearIndexes();
}

size_t AscendIndexInt8FlatImpl::getBaseSize(int deviceId) const
{
    auto lock = ::ascend::AscendMultiThreadManager::GetReadLock(mtx);
    APP_LOG_INFO("AscendIndexInt8Flat getBaseSize operation started.\n");
    FAISS_THROW_IF_NOT_MSG(indexes.find(deviceId) != indexes.end(), "DeviceId out of range.");
    size_t size = indexInt8FlatGetBaseSize(deviceId,  metricType);
    APP_LOG_INFO("AscendIndexInt8Flat getBaseSize operation finished.\n");
    return size;
}

size_t AscendIndexInt8FlatImpl::indexInt8FlatGetBaseSize(int deviceId, faiss::MetricType metric) const
{
    auto *index = getActualIndex(deviceId);
    using namespace ::ascend;
    switch (metric) {
        case MetricType::METRIC_L2: {
            if (auto pIndex = dynamic_cast<IndexInt8FlatL2Aicpu *>(index)) {
                return pIndex->getSize();
            } else if (auto pIndexApprox = dynamic_cast<IndexInt8FlatApproxL2Aicpu *>(index)) {
                return pIndexApprox->getSize();
            } else {
                ASCEND_THROW_MSG("Unsupported index type\n");
            }
            break;
        }
        case MetricType::METRIC_INNER_PRODUCT: {
            auto pIndex = dynamic_cast<IndexInt8FlatCosAicpu *>(index);
            FAISS_THROW_IF_NOT_MSG((pIndex != nullptr), "Unsupported index type.");
            return pIndex->getSize();
            break;
        }
        default: {
            ASCEND_THROW_MSG("Unsupported metric type\n");
        }
    }
}

void AscendIndexInt8FlatImpl::int8FlatGetBase(int deviceId, uint32_t offset, uint32_t num,
                                              std::vector<int8_t> &vectors, faiss::MetricType metric) const
{
        auto *index = getActualIndex(deviceId);
        using namespace ::ascend;
        switch (metric) {
            case MetricType::METRIC_L2: {
                if (auto pIndex = dynamic_cast<IndexInt8FlatL2Aicpu *>(index)) {
                    pIndex->getVectors(offset, num, vectors);
                } else if (auto pIndexApprox = dynamic_cast<IndexInt8FlatApproxL2Aicpu *>(index)) {
                    pIndexApprox->getVectors(offset, num, vectors);
                } else {
                    ASCEND_THROW_MSG("Unsupported index type\n");
                }
                break;
            }
            case MetricType::METRIC_INNER_PRODUCT: {
                auto pIndex = dynamic_cast<IndexInt8FlatCosAicpu *>(index);
                FAISS_THROW_IF_NOT_MSG((pIndex != nullptr), "Unsupported index type.");
                pIndex->getVectors(offset, num, vectors);
                break;
            }
            default: {
                ASCEND_THROW_MSG("Unsupported metric type\n");
            }
        }
}

void AscendIndexInt8FlatImpl::indexInt8FlatAdd(int deviceId, int n, int dim, int8_t *data) const
{
        using namespace ::ascend;
        auto pIndex = getActualIndex(deviceId);
        AscendTensor<int8_t, DIMS_2> vec(data, {n, dim});
        auto ret = pIndex->addVectors(vec);
        FAISS_THROW_IF_NOT_FMT(ret == APP_ERR_OK, "Failed to add to index, result is %d", ret);
}

void AscendIndexInt8FlatImpl::getBase(int deviceId, std::vector<int8_t> &xb) const
{
    auto lock = ::ascend::AscendMultiThreadManager::GetReadLock(mtx);
    // getBase、getBaseEnd接口内部会使用修改成员变量dataVec、attrsVec，因此getBase接口不能并发
    auto getBaseLock = ::ascend::AscendMultiThreadManager::LockGetBaseMtx(deviceId, getBaseMtx);
    APP_LOG_INFO("AscendIndexInt8Flat getBase operation started.\n");
    size_t size = getBaseSize(deviceId);
    FAISS_THROW_IF_NOT_FMT(size > 0, "Get base size is (%zu).", size);
    if (size * static_cast<size_t>(dim) > xb.size()) {
        xb.resize(size * static_cast<size_t>(dim));
    }
    getPaged(deviceId, size, xb);
    getActualIndex(deviceId)->getBaseEnd();
    APP_LOG_INFO("AscendIndexInt8Flat getBase operation finished.\n");
}

void AscendIndexInt8FlatImpl::getBase(int deviceId, int8_t *xb) const
{
    auto lock = ::ascend::AscendMultiThreadManager::GetReadLock(mtx);
    // getBase、getBaseEnd接口内部会使用修改成员变量dataVec、attrsVec，因此getBase接口不能并发
    auto getBaseLock = ::ascend::AscendMultiThreadManager::LockGetBaseMtx(deviceId, getBaseMtx);
    APP_LOG_INFO("AscendIndexInt8Flat getBase operation started.\n");
    FAISS_THROW_IF_NOT_MSG(xb, "xb can not be nullptr.");
    size_t size = getBaseSize(deviceId);
    getPaged(deviceId, size, xb);
    getActualIndex(deviceId)->getBaseEnd();
    APP_LOG_INFO("AscendIndexInt8Flat getBase operation finished.\n");
}

void AscendIndexInt8FlatImpl::getIdxMap(int deviceId, std::vector<idx_t> &idxMap) const
{
    auto lock = ::ascend::AscendMultiThreadManager::GetReadLock(mtx);
    APP_LOG_INFO("AscendIndexInt8Flat getIdxMap operation started.\n");
    FAISS_THROW_IF_NOT_MSG(indexes.find(deviceId) != indexes.end(), "DeviceId out of range.");
    size_t deviceNum = int8FlatConfig.deviceList.size();
    for (size_t i = 0; i < deviceNum; i++) {
        if (deviceId == int8FlatConfig.deviceList[i]) {
            idxMap = idxDeviceMap.at(i);
            break;
        }
    }
    APP_LOG_INFO("AscendIndexInt8Flat getIdxMap operation finished.\n");
}

void AscendIndexInt8FlatImpl::getPaged(int deviceId, int n, std::vector<int8_t> &xb) const
{
    getPaged(deviceId, n, xb.data());
}

void AscendIndexInt8FlatImpl::getPaged(int deviceId, int n, int8_t *xb) const
{
    APP_LOG_INFO("AscendIndexInt8Flat getPaged operation started.\n");
    if (n > 0) {
        size_t totalSize = static_cast<size_t>(n) * static_cast<size_t>(dim) * sizeof(int8_t);
        size_t offsetNum = 0;
        if (totalSize > PAGE_SIZE || static_cast<size_t>(n) > VEC_SIZE) {
            size_t maxNumVecsForPageSize = PAGE_SIZE / (static_cast<size_t>(dim) * sizeof(int8_t));

            // Always add at least 1 vector, if we have huge vectors
            maxNumVecsForPageSize = std::max(maxNumVecsForPageSize, (size_t)1);

            size_t tileSize = std::min(static_cast<size_t>(n), maxNumVecsForPageSize);

            for (size_t i = 0; i < static_cast<size_t>(n); i += tileSize) {
                size_t curNum = std::min(tileSize, n - i);

                getImpl(deviceId, offsetNum, curNum, xb + offsetNum * static_cast<size_t>(dim));
                offsetNum += curNum;
            }
        } else {
            getImpl(deviceId, offsetNum, n, xb);
        }
    }
    APP_LOG_INFO("AscendIndexInt8Flat getPaged operation finished.\n");
}

void AscendIndexInt8FlatImpl::getImpl(int deviceId, int offset, int n, int8_t *x) const
{
    APP_LOG_INFO("AscendIndexInt8Flat getImpl operation started.\n");
    std::vector<int8_t> baseData;
    int8FlatGetBase(deviceId, offset, n, baseData, metricType);
    FAISS_THROW_IF_NOT_FMT(baseData.size() == static_cast<size_t>(n) * static_cast<size_t>(dim),
        "Invalid baseData.size is %zu, expected is %zu.\n",
        baseData.size(), static_cast<size_t>(n) * dim);
    auto ret = memcpy_s(x, n * dim * sizeof(int8_t),
                        baseData.data(), baseData.size() * sizeof(int8_t));
    FAISS_THROW_IF_NOT_FMT(ret == 0, "Memcpy_s failed(%d).", ret);
    APP_LOG_INFO("AscendIndexInt8Flat getImpl operation finished.\n");
}

void AscendIndexInt8FlatImpl::reset()
{
    auto lock = ::ascend::AscendMultiThreadManager::GetWriteLock(mtx);
    APP_LOG_INFO("AscendIndexInt8Flat reset operation started.\n");
    for (auto &data : indexes) {
        auto index = getActualIndex(data.first);
        index->reset();
    }

    idxDeviceMap.clear();
    idxDeviceMap.resize(int8FlatConfig.deviceList.size());

    label2Idx.clear();
    label2Idx.resize(int8FlatConfig.deviceList.size());

    ntotal = 0;
    APP_LOG_INFO("AscendIndexInt8Flat reset operation finished.\n");
}

void AscendIndexInt8FlatImpl::copyFrom(const faiss::IndexScalarQuantizer *index)
{
    auto lock = ::ascend::AscendMultiThreadManager::GetWriteLock(mtx);
    APP_LOG_INFO("AscendIndexInt8Flat copyFrom operation started.\n");
    FAISS_THROW_IF_NOT_MSG(index != nullptr, "Invalid index nullptr.");
    FAISS_THROW_IF_NOT_MSG(std::find(DIMS.begin(), DIMS.end(), index->d) != DIMS.end(), "Unsupported dims");
    FAISS_THROW_IF_NOT_MSG(index->metric_type == MetricType::METRIC_L2 ||
        index->metric_type == MetricType::METRIC_INNER_PRODUCT, "Unsupported metric type");
    FAISS_THROW_IF_NOT_FMT(index->ntotal >= 0 && index->ntotal < MAX_N, "ntotal must be >= 0 and < %ld", MAX_N);

    ntotal = 0;
    clearIndexes();

    dim = index->d;
    metricType = index->metric_type;

    initIndexes();

    // initial idxDeviceMap mem space
    idxDeviceMap.clear();
    idxDeviceMap.resize(int8FlatConfig.deviceList.size());

    label2Idx.clear();
    label2Idx.resize(int8FlatConfig.deviceList.size());

    // The other index might not be trained
    if (!index->is_trained) {
        trained = false;
        ntotal = 0;
        return;
    }

    trained = true;

    // copy cpu index's codes and preCompute to ascend index
    copyCode(index);
    APP_LOG_INFO("AscendIndexInt8Flat copyFrom operation finished.\n");
}

void AscendIndexInt8FlatImpl::copyFrom(const faiss::IndexIDMap *index)
{
    auto lock = ::ascend::AscendMultiThreadManager::GetWriteLock(mtx);
    APP_LOG_INFO("AscendIndexInt8Flat copyFrom operation started.\n");
    FAISS_THROW_IF_NOT_MSG(index != nullptr && index->index != nullptr, "Invalid index nullptr.");
    FAISS_THROW_IF_NOT_MSG(std::find(DIMS.begin(), DIMS.end(), index->index->d) != DIMS.end(), "Unsupported dims");
    FAISS_THROW_IF_NOT_MSG(index->metric_type == MetricType::METRIC_L2 ||
        index->metric_type == MetricType::METRIC_INNER_PRODUCT, "Unsupported metric type");
    FAISS_THROW_IF_NOT_FMT(index->index->ntotal >= 0 && index->index->ntotal < MAX_N,
        "ntotal must be >= 0 and < %ld", MAX_N);

    ntotal = 0;
    clearIndexes();

    dim = index->index->d;
    metricType = index->index->metric_type;

    initIndexes();

    // initial idxDeviceMap mem space
    idxDeviceMap.clear();
    idxDeviceMap.resize(int8FlatConfig.deviceList.size());

    label2Idx.clear();
    label2Idx.resize(int8FlatConfig.deviceList.size());

    // The other index might not be trained
    if (!index->index->is_trained) {
        trained = false;
        ntotal = 0;
        return;
    }

    trained = true;

    // copy cpu index's codes and preCompute to ascend index
    auto sqPtr = dynamic_cast<const faiss::IndexScalarQuantizer *>(index->index);
    FAISS_THROW_IF_NOT_MSG(sqPtr != nullptr, "Invalid sqIndex nullptr.");
    if (index->id_map.data() != nullptr) {
        FAISS_THROW_IF_NOT_MSG(index->id_map.size() == static_cast<size_t>(sqPtr->ntotal),
            "The size of id_map must be equal to ntotal.\n");
    }
    copyCode(sqPtr, index->id_map.data());
    APP_LOG_INFO("AscendIndexInt8Flat copyFrom operation finished.\n");
}

void AscendIndexInt8FlatImpl::copyCode(const faiss::IndexScalarQuantizer *index, const idx_t *ids)
{
    APP_LOG_INFO("AscendIndexInt8Flat copyCode operation started.\n");
    if (index->codes.size() == 0) {
        return;
    }
    FAISS_THROW_IF_NOT_MSG(index->codes.size()
        == static_cast<size_t>(index->ntotal) * static_cast<size_t>(dim),
        "The size of codes must be equal to ntotal * dim.\n");
    if (ids == nullptr) {
        // set ids
        std::vector<idx_t> idsInner(index->ntotal);
        for (size_t i = 0; i < idsInner.size(); ++i) {
            idsInner[i] = ntotal + static_cast<idx_t>(i);
        }
        copyImpl(index->ntotal, reinterpret_cast<const int8_t *>(index->codes.data()), idsInner.data());
    } else {
        copyImpl(index->ntotal, reinterpret_cast<const int8_t *>(index->codes.data()), ids);
    }
    APP_LOG_INFO("AscendIndexInt8Flat copyCode operation finished.\n");
}

void AscendIndexInt8FlatImpl::copyImpl(int n, const int8_t *codes, const idx_t *ids)
{
    APP_LOG_INFO("AscendIndexInt8Flat copyImpl operation started.\n");
    FAISS_THROW_IF_NOT(n > 0);

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
            // 1. compute addMap
            size_t deviceCnt = int8FlatConfig.deviceList.size();
            std::vector<int> addMap(deviceCnt, 0);
            calcAddMap(curNum, addMap);

            // 2. transfer the codes to the device
            add2DeviceFast(curNum, codes + i * static_cast<size_t>(dim), ids + i, addMap);
        }
    } else {
        if (this->intf_->verbose) {
            printf("AscendIndex::add: adding 0:%d / %d\n", n, n);
        }
        // 1. compute addMap
        size_t deviceCnt = int8FlatConfig.deviceList.size();
        std::vector<int> addMap(deviceCnt, 0);
        calcAddMap(n, addMap);

        // 2. transfer the codes to the device
        add2DeviceFast(n, codes, ids, addMap);
    }
    APP_LOG_INFO("AscendIndexInt8Flat copyImpl operation finished.\n");
}

void AscendIndexInt8FlatImpl::calcAddMap(int n, std::vector<int> &addMap)
{
    APP_LOG_INFO("AscendIndexInt8Flat calcAddMap operation started.\n");
    size_t devIdx = 0;
    size_t deviceCnt = int8FlatConfig.deviceList.size();
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
    APP_LOG_INFO("AscendIndexInt8Flat calcAddMap operation finished.\n");
}

void AscendIndexInt8FlatImpl::add2DeviceFast(int n, const int8_t *codes, const idx_t *ids,
    const std::vector<int> &addMap)
{
    APP_LOG_INFO("AscendIndexInt8Flat add2DeviceFast operation started.\n");
    size_t deviceCnt = int8FlatConfig.deviceList.size();

    int offsum = 0;
    std::vector<int> offsumMap(deviceCnt, 0);
    for (size_t i = 0; i < deviceCnt; i++) {
        offsumMap[i] = offsum;
        offsum += addMap.at(i);
    }

    auto addFunctor = [&](int idx) {
        int num = addMap.at(idx);
        if (num == 0) {
            return;
        }
        std::thread insertLabel([&] {
            ::ascend::CommonUtils::attachToCpu(idx + static_cast<int>(deviceCnt));
            ascend_idx_t oriIdx = static_cast<ascend_idx_t>(idxDeviceMap[idx].size());
            idxDeviceMap[idx].insert(idxDeviceMap[idx].end(), ids + offsumMap[idx], ids + num + offsumMap[idx]);
            for (auto i = 0; i < num; i++) {
                idx_t label = *(ids + offsumMap[idx] + i);
                label2Idx[idx][label] = oriIdx + static_cast<ascend_idx_t>(i);
            }
        });
        int deviceId = int8FlatConfig.deviceList[idx];
        int8_t *codeb = const_cast<int8_t *>(codes) +
            static_cast<size_t>(offsumMap[idx]) * static_cast<size_t>(dim);
        try {
            indexInt8FlatAdd(deviceId, num, dim, codeb);
            insertLabel.join();
        } catch (std::exception &e) {
            insertLabel.join();
            ASCEND_THROW_FMT("wait for add functor failed %s", e.what());
        }
    };
    
    std::vector<std::future<void>> functorRets;
    for (int i = 0; i < static_cast<int>(deviceCnt); i++) {
        functorRets.emplace_back(pool->Enqueue(addFunctor, i));
    }
    try {
        for (auto &ret : functorRets) {
            ret.get();
        }
    } catch (std::exception &e) {
        FAISS_THROW_FMT("wait for parallel future failed: %s", e.what());
    }

    ntotal += n;
    APP_LOG_INFO("AscendIndexInt8Flat add2DeviceFast operation finished.\n");
}

void AscendIndexInt8FlatImpl::add2Device(int n, const int8_t *codes, const idx_t *ids,
    const std::vector<int> &addMap)
{
    APP_LOG_INFO("AscendIndexInt8Flat add2Device operation started.\n");
    size_t deviceCnt = int8FlatConfig.deviceList.size();
    uint32_t offsum = 0;
    for (size_t i = 0; i < deviceCnt; i++) {
        int num = addMap.at(i);
        if (num == 0) {
            continue;
        }
        int deviceId = int8FlatConfig.deviceList[i];
        std::vector<int8_t> codeb(num * dim);
        codeb.assign(codes + offsum * dim, codes + (offsum + num) * dim);
        indexInt8FlatAdd(deviceId, num, dim, codeb.data());
        // record ids of new adding vector
        ascend_idx_t oriIdx = static_cast<ascend_idx_t>(idxDeviceMap[i].size());
        idxDeviceMap[i].insert(idxDeviceMap[i].end(), ids + offsum, ids + num + offsum);
        for (auto j = 0; j < num; j++) {
            idx_t label = *(ids + offsum + j);
            label2Idx[i][label] = oriIdx + static_cast<ascend_idx_t>(j);
        }
        offsum += static_cast<uint32_t>(num);
    }
    ntotal += static_cast<faiss::idx_t>(n);
    APP_LOG_INFO("AscendIndexInt8Flat add2Device operation finished.\n");
}

void AscendIndexInt8FlatImpl::copyTo(faiss::IndexScalarQuantizer *index) const
{
    auto lock = ::ascend::AscendMultiThreadManager::GetReadLock(mtx);
    APP_LOG_INFO("AscendIndexInt8Flat copyTo operation started.\n");
    FAISS_THROW_IF_NOT(index != nullptr);
    index->reset();
    index->metric_type = metricType;
    index->d = dim;
    index->ntotal = ntotal;
    index->is_trained = trained;

    if (trained && ntotal > 0) {
        size_t deviceCnt = int8FlatConfig.deviceList.size();
        std::vector<size_t> getOffset(deviceCnt, 0);
        size_t total = 0;
        for (size_t i = 0; i < deviceCnt; i++) {
            getOffset[i] = total;
            total += getBaseSize(int8FlatConfig.deviceList[i]);
        }

        std::vector<uint8_t> baseData(total * static_cast<size_t>(dim));
        auto getFunctor = [this, &baseData, &getOffset] (int idx) {
            getBase(this->int8FlatConfig.deviceList[idx], reinterpret_cast<int8_t *>(
                baseData.data() + getOffset[idx] * static_cast<size_t>(dim)));
        };
        CALL_PARALLEL_FUNCTOR(deviceCnt, pool, getFunctor);

        index->codes = std::move(baseData);
    }
    APP_LOG_INFO("AscendIndexInt8Flat copyTo operation finished.\n");
}

void AscendIndexInt8FlatImpl::copyTo(faiss::IndexIDMap *index) const
{
    auto lock = ::ascend::AscendMultiThreadManager::GetReadLock(mtx);
    APP_LOG_INFO("AscendIndexInt8Flat copyTo operation started.\n");
    FAISS_THROW_IF_NOT_MSG(index != nullptr, "Invalid index nullptr.");

    copyTo(dynamic_cast<faiss::IndexScalarQuantizer *>(index->index));

    std::vector<idx_t> idsInt8;
    for (auto device : int8FlatConfig.deviceList) {
        std::vector<idx_t> ids;
        getIdxMap(device, ids);
        idsInt8.insert(idsInt8.end(), ids.begin(), ids.end());
    }
    index->id_map = std::move(idsInt8);
    APP_LOG_INFO("AscendIndexInt8Flat copyTo operation finished.\n");
}

void AscendIndexInt8FlatImpl::search_with_masks(idx_t n, const int8_t *x, idx_t k,
    float *distances, idx_t *labels, const void *mask) const
{
    auto lock = ::ascend::AscendMultiThreadManager::GetWriteLock(mtx);
    APP_LOG_INFO("AscendIndexInt8Flat search_with_masks operation started: searchNum=%ld, topK=%ld.\n", n, k);
    FAISS_THROW_IF_NOT_FMT((n > 0) && (n < MAX_N), "n must be > 0 and < %ld", MAX_N);
    FAISS_THROW_IF_NOT_FMT((k > 0) && (k <= MAX_K), "k must be > 0 and <= %ld", MAX_K);
    FAISS_THROW_IF_NOT_MSG(x, "x can not be nullptr.");
    FAISS_THROW_IF_NOT_MSG(distances, "distances can not be nullptr.");
    FAISS_THROW_IF_NOT_MSG(labels, "labels can not be nullptr.");
    FAISS_THROW_IF_NOT_MSG(mask, "mask can not be nullptr.");
    FAISS_THROW_IF_NOT_MSG(trained, "Index not trained");

    size_t deviceCnt = int8FlatConfig.deviceList.size();

    // convert query data from float to fp16, device use fp16 data to search
    std::vector<int8_t> query(x, x + n * dim);

    std::vector<std::vector<float>> dist(deviceCnt, std::vector<float>(n * k, 0));
    std::vector<std::vector<uint16_t>> distHalf(deviceCnt, std::vector<uint16_t>(n * k, 0));
    std::vector<std::vector<ascend_idx_t>> label(deviceCnt, std::vector<ascend_idx_t>(n * k, 0));

    auto searchFunctor = [&](int idx) {
        int deviceId = int8FlatConfig.deviceList[idx];
        auto index = getActualIndex(deviceId);
        using namespace ::ascend;
        AscendTensor<int8_t, DIMS_2, idx_t> tensorDevQueries({ n,  dim });
        auto ret = aclrtMemcpy(tensorDevQueries.data(), tensorDevQueries.getSizeInBytes(),
                               query.data(), n *  dim * sizeof(int8_t),
                               ACL_MEMCPY_HOST_TO_DEVICE);
        FAISS_THROW_IF_NOT_FMT(ret == ACL_SUCCESS, "aclrtMemcpy error %d", ret);

        ret = index->search(n, tensorDevQueries.data(), k,  distHalf[idx].data(), label[idx].data(),
                            static_cast<uint8_t *>(const_cast<void *>(mask)));
        FAISS_THROW_IF_NOT_FMT(ret == APP_ERR_OK, "Failed to search index id: %d\n", ret);

        if (metricType == faiss::METRIC_L2) {
            // convert result data from fp16 to float
            auto scaleFunctor = [](int dim) {
                // 0.01, 128, 4 is hyperparameter suitable for UB
                return 0.01 / std::min(dim / 64, std::max(dim / 128 + 1, 4));
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
    APP_LOG_INFO("AscendIndexInt8Flat search_with_masks operation finished.\n");
}

void AscendIndexInt8FlatImpl::addImpl(int n, const int8_t *x, const idx_t *ids)
{
    APP_LOG_INFO("AscendIndexInt8Flat addImpl operation started: n=%d.\n", n);
    FAISS_THROW_IF_NOT(n > 0);

    size_t deviceCnt = int8FlatConfig.deviceList.size();
    FAISS_THROW_IF_NOT_MSG(deviceCnt > 0, "device count should be positive.");

    std::vector<int> addMap(deviceCnt, 0);
    calcAddMap(n, addMap);

    uint32_t offsum = 0;
    for (size_t i = 0; i < deviceCnt; i++) {
        int num = addMap.at(i);
        if (num == 0) {
            continue;
        }
        std::thread insertLabel([&] {
            ascend_idx_t oriIdx = static_cast<ascend_idx_t>(idxDeviceMap[i].size());
            idxDeviceMap[i].insert(idxDeviceMap[i].end(), ids + offsum, ids + num + offsum);
            for (auto j = 0; j < num; j++) {
                idx_t label = *(ids + offsum + j);
                label2Idx[i][label] = oriIdx + static_cast<ascend_idx_t>(j);
            }
        });
        int deviceId = int8FlatConfig.deviceList[i];
        try {
            indexInt8FlatAdd(deviceId, num, dim, const_cast<int8_t *>(x + offsum * dim));
            insertLabel.join();
        } catch (std::exception &e) {
            insertLabel.join();
            ASCEND_THROW_FMT("wait for add functor failed %s", e.what());
        }

        offsum += static_cast<uint32_t>(num);
    }
    ntotal += static_cast<faiss::idx_t>(n);
    APP_LOG_INFO("AscendIndexInt8Flat addImpl operation finished.\n");
}

void AscendIndexInt8FlatImpl::removeSingle(std::vector<std::vector<ascend_idx_t>> &removes,
                                           int deviceNum, ascend_idx_t idx)
{
    APP_LOG_INFO("AscendIndexInt8Flat removeSingle operation started.\n");
    for (int i = 0; i < deviceNum; i++) {
        auto it = label2Idx[i].find(idx);
        if (it != label2Idx[i].end()) {
            removes[i].push_back(it->second);
            break;
        }
    }
    APP_LOG_INFO("AscendIndexInt8Flat removeSingle operation finished.\n");
}

void AscendIndexInt8FlatImpl::searchPostProcess(size_t devices, std::vector<std::vector<float>> &dist,
    std::vector<std::vector<ascend_idx_t>> &label, idx_t n, idx_t k, float *distances,
    idx_t *labels) const
{
    APP_LOG_INFO("AscendIndexInt8Flat searchPostProcess operation started.\n");
    // transmit idx per device to referenced value
    size_t deviceCnt = this->int8FlatConfig.deviceList.size();
    std::vector<std::vector<ascend_idx_t>> transLabel(deviceCnt, std::vector<ascend_idx_t>(n * k, 0));
    for (size_t i = 0; i < deviceCnt; i++) {
        transform(begin(label[i]), end(label[i]), begin(transLabel[i]), [&](ascend_idx_t temp) {
            return (temp != std::numeric_limits<ascend_idx_t>::max() && idxDeviceMap[i].size() > 0) ?
                                                                        idxDeviceMap[i].at(temp) :
                                                                        std::numeric_limits<ascend_idx_t>::max();
        });
    }

    // device数量为1不使用多线程，避免性能波动
    if (deviceCnt == 1) {
        transform(transLabel.front().begin(), transLabel.front().end(), labels,
                  [] (ascend_idx_t label) { return idx_t(label); });
        transform(dist.front().begin(), dist.front().end(), distances, [] (float dist) { return dist; });
    } else {
        mergeSearchResult(devices, dist, transLabel, n, k, distances, labels);
    }
    APP_LOG_INFO("AscendIndexInt8Flat searchPostProcess operation finished.\n");
}

size_t AscendIndexInt8FlatImpl::getElementSize() const
{
    return static_cast<size_t>(dim) * sizeof(int8_t);
}

std::shared_ptr<::ascend::IndexInt8> AscendIndexInt8FlatImpl::createIndex(int deviceId)
{
        APP_LOG_INFO("AscendIndexInt8Flat CreateIndexInt8 started.\n");
        auto res = aclrtSetDevice(deviceId);
        FAISS_THROW_IF_NOT_FMT(res == 0, "acl set device failed %d, deviceId: %d", res, deviceId);
        using namespace ::ascend;
        std::shared_ptr<IndexInt8> index = nullptr;
        if (int8FlatConfig.dIndexMode == Int8IndexMode::DEFAULT_MODE) {
            APP_LOG_INFO("AscendIndexInt8Flat CreateIndexInt8 default mode started.\n");
            switch (metricType) {
                case MetricType::METRIC_L2: {
                    index = std::make_shared<IndexInt8FlatL2Aicpu>(dim, int8FlatConfig.resourceSize,
                                                                   static_cast<int>(int8FlatConfig.dBlockSize));
                    break;
                }
                case MetricType::METRIC_INNER_PRODUCT: {
                    index = std::make_shared<IndexInt8FlatCosAicpu>(dim, int8FlatConfig.resourceSize,
                                                                    static_cast<int>(int8FlatConfig.dBlockSize));
                    break;
                }
                default: {
                    ASCEND_THROW_MSG("Unsupported metric type\n");
                }
            }
            auto ret = index->init();
            FAISS_THROW_IF_NOT_FMT(ret == APP_ERR_OK, "Failed to create index int8 flat:%d", ret);
        } else {
            switch (metricType) {
                case MetricType::METRIC_L2: {
                    index = std::make_shared<IndexInt8FlatApproxL2Aicpu>(dim, int8FlatConfig.resourceSize,
                                                                         static_cast<int>(int8FlatConfig.dBlockSize));
                    break;
                }
                default: {
                    ASCEND_THROW_MSG("Unsupported metric type\n");
                }
            }
            auto ret = index->init();
            FAISS_THROW_IF_NOT_FMT(ret == APP_ERR_OK,  "Failed to create index int8 flat:%d", ret);
        }
        APP_LOG_INFO("AscendIndexInt8Flat CreateIndexInt8 finished.\n");
        return index;
}

size_t AscendIndexInt8FlatImpl::removeImpl(const IDSelector &sel)
{
    APP_LOG_INFO("AscendIndexInt8Flat removeImpl operation started.\n");
    int deviceCnt = static_cast<int>(int8FlatConfig.deviceList.size());
    uint32_t removeCnt = 0;

    // 1. remove vector from device, and removeMaps save the id(not index) of pre-removing
    std::vector<std::vector<ascend_idx_t>> removeMaps(deviceCnt, std::vector<ascend_idx_t>());
    if (auto rangeSel = dynamic_cast<const IDSelectorBatch *>(&sel)) {
        size_t removeSize = rangeSel->set.size();
        FAISS_THROW_IF_NOT_FMT(removeSize <= static_cast<size_t>(ntotal),
            "the size of removed codes should be in range [0, %ld], actual=%zu.", ntotal, removeSize);
        std::vector<ascend_idx_t> removeBatch(removeSize);
        transform(begin(rangeSel->set), end(rangeSel->set), begin(removeBatch),
            [](idx_t temp) { return (ascend_idx_t)temp; });

        for (auto idx : removeBatch) {
            removeSingle(removeMaps, deviceCnt, idx);
        }
    } else if (auto rangeSel = dynamic_cast<const IDSelectorRange *>(&sel)) {
        for (auto idx = rangeSel->imin; idx < rangeSel->imax; ++idx) {
            removeSingle(removeMaps, deviceCnt, idx);
        }
    } else {
        APP_LOG_WARNING("Invalid IDSelector.\n");
        return 0;
    }

    // 2. remove the vector in the device
#pragma omp parallel for reduction(+ : removeCnt) num_threads(deviceCnt)
    for (int i = 0; i < deviceCnt; i++) {
        if (removeMaps[i].size() == 0) {
            continue;
        }
        int deviceId = int8FlatConfig.deviceList[i];
        auto index = getActualIndex(deviceId);
        ::ascend::IDSelectorBatch batch(removeMaps[i].size(), removeMaps[i].data());
        removeCnt += index->removeIds(batch);
    }

    // 3. remove the index save in the host
    removeIdx(removeMaps);

    FAISS_THROW_IF_NOT_FMT(ntotal >= removeCnt,
        "removeCnt should be in range [0, %ld], actual=%d.", ntotal, removeCnt);
    ntotal -= static_cast<faiss::idx_t>(removeCnt);
    APP_LOG_INFO("AscendIndexInt8Flat removeImpl operation finished.\n");
    return (size_t)removeCnt;
}

void AscendIndexInt8FlatImpl::removeIdx(std::vector<std::vector<ascend_idx_t>> &removeMaps)
{
    APP_LOG_INFO("AscendIndexInt8Flat removeIdx operation started.\n");
    size_t deviceCnt = int8FlatConfig.deviceList.size();
#pragma omp parallel for if (deviceCnt > 1) num_threads(deviceCnt)
    for (size_t i = 0; i < deviceCnt; ++i) {
        // 1. sort by DESC, then delete from the big to small
        std::sort(removeMaps[i].rbegin(), removeMaps[i].rend());

        for (auto pos : removeMaps[i]) {
            uint64_t delLabel = this->idxDeviceMap[i][pos];
            size_t lastIdx = this->idxDeviceMap[i].size() - 1;
            auto lastLable = this->idxDeviceMap[i][lastIdx];
            this->idxDeviceMap[i][pos] = lastLable;
            this->idxDeviceMap[i].pop_back();

            this->label2Idx[i][lastLable] = pos;
            this->label2Idx[i].erase(delLabel);
        }
    }
    APP_LOG_INFO("AscendIndexInt8Flat removeIdx operation finished.\n");
}

void AscendIndexInt8FlatImpl::CheckIndexParams(IndexImplBase &index, bool) const
{
    try {
        AscendIndexInt8FlatImpl &int8Index = dynamic_cast<AscendIndexInt8FlatImpl &>(index);

        FAISS_THROW_IF_NOT_MSG(metricType == int8Index.metricType,
            "the metric type must be same.");
        FAISS_THROW_IF_NOT_MSG(dim == int8Index.getDim(), "the dim must be same.");
        FAISS_THROW_IF_NOT_MSG(this->indexConfig.deviceList == int8Index.indexConfig.deviceList,
            "the deviceList must be same.");
        FAISS_THROW_IF_NOT_FMT(int8Index.indexConfig.deviceList.size() == 1,
            "the number of deviceList (%zu) must be 1.", int8Index.indexConfig.deviceList.size());
    } catch (std::bad_cast &e) {
        FAISS_THROW_MSG("the type of index is not same, or the number of deviceList is not 1.");
    }
}

void AscendIndexInt8FlatImpl::searchPaged(int n, const int8_t *x, int k, float *distances, idx_t *labels) const
{
    APP_LOG_INFO("AscendIndexInt8Flat searchPaged operation started: n=%d, k=%d.\n", n, k);
    if (int8FlatConfig.dIndexMode == Int8IndexMode::DEFAULT_MODE) {
        AscendIndexInt8Impl::searchPaged(n, x, k, distances, labels);
        APP_LOG_INFO("AscendIndexInt8Flat indexMode %u searchPaged operation finished.\n", Int8IndexMode::DEFAULT_MODE);
        return;
    }

    searchPagedImpl(n, x, k, distances, labels);

    APP_LOG_INFO("AscendIndexInt8Flat searchPaged operation finished.\n");
}

void AscendIndexInt8FlatImpl::searchPagedImpl(int n, const int8_t *x, int k, float *distances,
    idx_t *labels) const
{
    float scale = 0.01 / std::min(dim / 64, std::max(dim / 128 + 1, 4));
    auto functor = [&](int idx, size_t num, int8_t *query, float *queryNorm, uint16_t *distHalf, ascend_idx_t *label) {
        for (size_t i = 0; i < num; ++i) {
            int32_t res = 0;
            for (int j = 0; j < dim; ++j) {
                res += query[i * static_cast<size_t>(dim) + static_cast<size_t>(j)] *
                       query[i * static_cast<size_t>(dim) + static_cast<size_t>(j)];
            }
            queryNorm[i] = res * scale;
        }
        IndexParam<int8_t, uint16_t, ascend_idx_t> param(indexConfig.deviceList[idx], num, dim, k);
        indexInt8Search(param, query, distHalf, label);
    };

    std::vector<std::future<void>> functorRets;

    size_t prevNum = 0;
    for (size_t i = 0; i < static_cast<size_t>(n); i += SEARCH_PAGE_SIZE) {
        size_t curNum = std::min(SEARCH_PAGE_SIZE, (size_t)(n) - i);
        // prepare data for current page
        pInt8PipeSearchImpl->SearchPipelinePrepare(curNum, k, x, i);

        if (i > 0) {
            // wait for previous page to finish on device
            for (auto &ret : functorRets) {
                ret.get();
            }
            functorRets.clear();
        }

        // call search for current page
        for (size_t idx = 0; idx < indexConfig.deviceList.size(); idx++) {
            functorRets.emplace_back(pool->Enqueue(functor, idx, curNum,
                pInt8PipeSearchImpl->searchPipelineQuery->data(), pInt8PipeSearchImpl->searchPipelineQueryNorm->data(),
                pInt8PipeSearchImpl->searchPipelineDistHalf->at(idx).data(),
                pInt8PipeSearchImpl->searchPipelineLabel->at(idx).data()));
        }

        // run post process for previous page
        if (i > 0) {
            pInt8PipeSearchImpl->SearchPipelineFinish(prevNum, k);
            // post process: default is merge the topk results from devices
            searchPostProcess(int8FlatConfig.deviceList.size(), *pInt8PipeSearchImpl->searchPipelineDistPrev,
                *pInt8PipeSearchImpl->searchPipelineLabelPrev, prevNum, k, distances + (i - SEARCH_PAGE_SIZE) * k,
                labels + (i - SEARCH_PAGE_SIZE) * k);
            pInt8PipeSearchImpl->ClearPrevData();
        }

        pInt8PipeSearchImpl->MoveDataForNext();
        prevNum = curNum;
    }

    // run post process for the last page
    for (auto &ret : functorRets) {
        ret.get();
    }

    size_t lastOffset = (static_cast<size_t>(n) - 1) / SEARCH_PAGE_SIZE * SEARCH_PAGE_SIZE;
    pInt8PipeSearchImpl->SearchPipelineFinish(prevNum, k);
    searchPostProcess(int8FlatConfig.deviceList.size(), *pInt8PipeSearchImpl->searchPipelineDistPrev,
        *pInt8PipeSearchImpl->searchPipelineLabelPrev, prevNum, k, distances + lastOffset * k, labels + lastOffset * k);
    pInt8PipeSearchImpl->ClearPrevData();
}

void AscendIndexInt8FlatImpl::setPageSize(uint16_t pageBlockNum)
{
    auto lock = ::ascend::AscendMultiThreadManager::GetWriteLock(mtx);
    APP_LOG_INFO("AscendIndexInt8Flat setPageSize[%u] started\n", pageBlockNum);
    FAISS_THROW_IF_NOT_FMT(pageBlockNum > 0 && pageBlockNum <= MAX_PAGE_BLOCK_NUM,
        "pageBlockNum[%u] should be in (0, %u]", pageBlockNum, MAX_PAGE_BLOCK_NUM);
    for (const auto &deviceId : int8FlatConfig.deviceList) {
        auto index = getActualIndex(deviceId);
        index->setPageSize(pageBlockNum);
    }
    APP_LOG_INFO("AscendIndexInt8Flat setPageSize finished\n");
}

} // ascend
} // faiss
