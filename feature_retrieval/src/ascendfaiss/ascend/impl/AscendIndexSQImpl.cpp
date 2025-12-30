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


#include "AscendIndexSQImpl.h"

#include <algorithm>

#include <faiss/utils/distances.h>

#include "ascend/utils/fp16.h"
#include "ascendhost/include/index/IndexSQIPAicpu.h"
#include "ascendhost/include/index/IndexSQL2Aicpu.h"
#include "Constants.h"
#include "ascenddaemon/utils/AscendUtils.h"

namespace faiss {
namespace ascend {
namespace {
const int DIM_ALIGN_SIZE = 16;
const int DIM_MAX = 512;

const size_t KB = 1024;
const size_t RETAIN_SIZE = 2048;

// Default size(64MB) for which we page add or search
const size_t ADD_PAGE_SIZE = 64U * KB * KB - RETAIN_SIZE;

// Or, maximum number(512K) of vectors to consider per page of add
const size_t ADD_VEC_SIZE = 512U * KB;

// get pagesize must be less than 32M, becauseof rpc limitation
const size_t PAGE_SIZE = 32U * KB * KB - RETAIN_SIZE;

// Or, maximum number 512K of vectors to consider per page of search
const size_t VEC_SIZE = 512U * KB;

// Default dim in case of nullptr index
const size_t DEFAULT_DIM = 512;

const int FILTER_SIZE = 6;

// The value range of dim
const std::vector<int> DIMS = { 64, 128, 256, 384, 512, 768 };
const std::vector<int> SQ_BLOCK_SIZES = { 8 * 16384, 16 * 16384, 32 * 16384, 64 * 16384 };
} // namespace

AscendIndexSQImpl::AscendIndexSQImpl(AscendIndexSQ *intf, const faiss::IndexScalarQuantizer *index,
    AscendIndexSQConfig config)
    : AscendIndexImpl(((index == nullptr) ? DEFAULT_DIM : index->d),
    ((index == nullptr) ? faiss::METRIC_L2 : index->metric_type), config, intf),
      intf_(intf),
      sqConfig(config)
{
    FAISS_THROW_IF_NOT_MSG(index != nullptr, "Invalid index nullptr.");

    ::ascend::AscendMultiThreadManager::InitGetBaseMtx(config.deviceList, getBaseMtx);

    copyFrom(index);
}

AscendIndexSQImpl::AscendIndexSQImpl(AscendIndexSQ *intf, const faiss::IndexIDMap *index, AscendIndexSQConfig config)
    : AscendIndexImpl(
        (index == nullptr || index->index == nullptr) ? DEFAULT_DIM : index->index->d,
        (index == nullptr || index->index == nullptr) ? faiss::METRIC_L2 : index->index->metric_type,
        config, intf),
      intf_(intf),
      sqConfig(config)
{
    FAISS_THROW_IF_NOT_MSG(index != nullptr, "Invalid index nullptr.");
    FAISS_THROW_IF_NOT_MSG(index->index != nullptr, "Invalid index nullptr.");

    ::ascend::AscendMultiThreadManager::InitGetBaseMtx(config.deviceList, getBaseMtx);

    copyFrom(index);
}

AscendIndexSQImpl::AscendIndexSQImpl(AscendIndexSQ *intf, int dims,
    faiss::ScalarQuantizer::QuantizerType qType, faiss::MetricType metric,
    AscendIndexSQConfig config)
    : AscendIndexImpl(dims, metric, config, intf), intf_(intf), sqConfig(config)
{
    ::ascend::AscendMultiThreadManager::InitGetBaseMtx(config.deviceList, getBaseMtx);
    sq = faiss::ScalarQuantizer(dims, qType);

    checkParams();
    clearIndexes();
    initIndexes();
    // initial idxDeviceMap mem space
    this->idxDeviceMap.clear();
    this->idxDeviceMap.resize(indexConfig.deviceList.size());

    this->intf_->is_trained = false;
}

AscendIndexSQImpl::~AscendIndexSQImpl() {}

std::shared_ptr<::ascend::Index> AscendIndexSQImpl::createIndex(int deviceId)
{
    APP_LOG_INFO("AscendIndexSQ  createIndex operation started, device id: %d\n", deviceId);
    auto res = aclrtSetDevice(deviceId);
    FAISS_THROW_IF_NOT_FMT(res == 0, "acl set device failed %d, deviceId: %d", res, deviceId);
    // index构造和析构会修改算子；全局变量算子管理类DistComputeOpsManager，该对象和device绑定，因此也需要device级别的锁
    // 这里必须保证index在glck析构以后再被调用，否则会造成同一线程重复占用锁而卡住
    std::shared_ptr<::ascend::Index> index = nullptr;
    if (!indexConfig.slim) {
        switch (this->intf_->metric_type) {
            case MetricType::METRIC_INNER_PRODUCT:
                index = std::shared_ptr<::ascend::IndexSQIPAicpu>(
                    new ::ascend::IndexSQIPAicpu(this->intf_->d, indexConfig.filterable,
                                                 indexConfig.resourceSize,
                                                 static_cast<int>(indexConfig.dBlockSize)));
                break;
            case MetricType::METRIC_L2:
                index = std::shared_ptr<::ascend::IndexSQL2Aicpu>(
                    new ::ascend::IndexSQL2Aicpu(this->intf_->d, indexConfig.filterable,
                                                 indexConfig.resourceSize,
                                                 static_cast<int>(indexConfig.dBlockSize)));
                break;
            default:
                ASCEND_THROW_MSG("Unsupported metric type\n");
        }
    } else {
        ASCEND_THROW_MSG("Not Support SQSlim\n");
    }
    auto ret = index->init();
    FAISS_THROW_IF_NOT_FMT(ret == ::ascend::APP_ERR_OK, "Failed to create init index: %d", ret);
    APP_LOG_INFO("AscendIndexSQ createIndex operation finished.\n");

    return index;
}

void AscendIndexSQImpl::checkParams() const
{
    // only support L2 and INNER_PRODUCT
    FAISS_THROW_IF_NOT_MSG(this->intf_->metric_type == MetricType::METRIC_L2 ||
        this->intf_->metric_type == MetricType::METRIC_INNER_PRODUCT,
        "Unsupported metric type");

    // only support SQ8
    FAISS_THROW_IF_NOT_MSG(sq.qtype == faiss::ScalarQuantizer::QT_8bit, "Unsupported qtype");

    FAISS_THROW_IF_NOT_FMT(std::find(DIMS.begin(), DIMS.end(), this->intf_->d) != DIMS.end(),
        "Unsupported dims: %d", this->intf_->d);
    FAISS_THROW_IF_NOT_FMT(std::find(SQ_BLOCK_SIZES.begin(), SQ_BLOCK_SIZES.end(), sqConfig.dBlockSize) !=
        SQ_BLOCK_SIZES.end(), "Unsupported blockSize: %u", sqConfig.dBlockSize);
    FAISS_THROW_IF_NOT_MSG(!sqConfig.slim, "slim only support false.");
}

void AscendIndexSQImpl::CheckIndexParams(IndexImplBase &index, bool checkFilterable) const
{
    auto lock = ::ascend::AscendMultiThreadManager::GetReadLock(mtx);
    try {
        AscendIndexSQImpl& sqIndex = dynamic_cast<AscendIndexSQImpl&>(index);

        FAISS_THROW_IF_NOT_MSG(this->intf_->metric_type == sqIndex.intf_->metric_type, "the metric type must be same.");
        FAISS_THROW_IF_NOT_MSG(sq.qtype == sqIndex.sq.qtype, "the qtype must be same.");
        FAISS_THROW_IF_NOT_MSG(this->intf_->d == sqIndex.intf_->d, "the dim must be same.");
        FAISS_THROW_IF_NOT_MSG(this->sqConfig.deviceList == sqIndex.sqConfig.deviceList,
                               "the deviceList must be same.");
        FAISS_THROW_IF_NOT_FMT(sqIndex.sqConfig.deviceList.size() == 1,
                               "the number of deviceList (%zu) must be 1.", sqIndex.sqConfig.deviceList.size());
        if (checkFilterable) {
            FAISS_THROW_IF_NOT_MSG(sqConfig.filterable, "the index does not support filterable");
        }
    } catch (std::bad_cast &e) {
        FAISS_THROW_MSG("the type of index is not same, or the number of deviceList is not 1.");
    }
}

void AscendIndexSQImpl::copyFrom(const faiss::IndexScalarQuantizer *index)
{
    auto lock = ::ascend::AscendMultiThreadManager::GetWriteLock(mtx);
    APP_LOG_INFO("AscendIndexSQImpl copyFrom operation started.\n");
    FAISS_THROW_IF_NOT_MSG(index != nullptr, "Invalid index nullptr.");
    FAISS_THROW_IF_NOT_FMT((index->ntotal >= 0) && (index->ntotal < MAX_N), "ntotal must be >= 0 and < %ld", MAX_N);
    FAISS_THROW_IF_NOT_FMT(this->intf_->ntotal + index->ntotal < MAX_N, "ntotal must be < %ld", MAX_N);
    sq = index->sq;
    FAISS_THROW_IF_NOT_FMT(static_cast<size_t>(index->d) == sq.code_size,
        "Invalid index, d[%d] must be equal to code_size[%zu]", index->d, sq.code_size);

    this->intf_->ntotal = 0;
    clearIndexes();
    this->intf_->d = index->d;
    this->intf_->metric_type = index->metric_type;

    checkParams();

    initIndexes();

    // initial idxDeviceMap mem space
    this->idxDeviceMap.clear();
    this->idxDeviceMap.resize(indexConfig.deviceList.size());

    // The other index might not be trained
    if (!index->is_trained) {
        this->intf_->is_trained = false;
        this->intf_->ntotal = 0;
        return;
    }

    this->intf_->is_trained = true;

    // copy cpu index's codes and preCompute to ascend index
    copyCode(index);

    // copy train param to ascend index
    updateDeviceSQTrainedValue();
    APP_LOG_INFO("AscendIndexSQImpl copyFrom operation finished.\n");
}

void AscendIndexSQImpl::copyFrom(const faiss::IndexIDMap *index)
{
    auto lock = ::ascend::AscendMultiThreadManager::GetWriteLock(mtx);
    APP_LOG_INFO("AscendIndexSQ copyFrom operation started.\n");
    FAISS_THROW_IF_NOT_MSG(index != nullptr, "Invalid index nullptr.");
    FAISS_THROW_IF_NOT_MSG(index->index != nullptr, "Invalid index nullptr.");
    auto sqPtr = dynamic_cast<faiss::IndexScalarQuantizer *>(index->index);
    FAISS_THROW_IF_NOT_MSG(sqPtr != nullptr, "Invalid sqIndex nullptr.");
    FAISS_THROW_IF_NOT_FMT((sqPtr->ntotal >= 0) && (sqPtr->ntotal < MAX_N), "ntotal must be >= 0 and < %ld", MAX_N);
    FAISS_THROW_IF_NOT_FMT(this->intf_->ntotal + sqPtr->ntotal < MAX_N, "ntotal must be < %ld", MAX_N);
    sq = sqPtr->sq;
    FAISS_THROW_IF_NOT_FMT(static_cast<size_t>(sqPtr->d) == sq.code_size,
        "Invalid index, d[%d] must be equal to code_size[%zu]", sqPtr->d, sq.code_size);

    this->intf_->ntotal = 0;
    clearIndexes();

    this->intf_->d = index->index->d;
    this->intf_->metric_type = index->index->metric_type;

    checkParams();

    initIndexes();

    // initial idxDeviceMap mem space
    this->idxDeviceMap.clear();
    this->idxDeviceMap.resize(indexConfig.deviceList.size());

    // The other index might not be trained
    if (!index->index->is_trained) {
        this->intf_->is_trained = false;
        this->intf_->ntotal = 0;
        return;
    }

    this->intf_->is_trained = true;

    if (index->id_map.data() != nullptr) {
        FAISS_THROW_IF_NOT_MSG(index->id_map.size() == static_cast<size_t>(sqPtr->ntotal),
            "The size of id_map must be equal to ntotal.\n");
    }

    // copy cpu index's codes, ids and preCompute to ascend index
    copyCode(sqPtr, index->id_map.data());

    // copy train param to ascend index
    updateDeviceSQTrainedValue();
    APP_LOG_INFO("AscendIndexSQ copyFrom operation finished.\n");
}

void AscendIndexSQImpl::copyCode(const faiss::IndexScalarQuantizer *index, const idx_t *ids)
{
    APP_LOG_INFO("AscendIndexSQ copyCode operation started.\n");
    if (index->codes.size() == 0) {
        return;
    }

    FAISS_THROW_IF_NOT_MSG(index->codes.size()
        == static_cast<size_t>(index->ntotal) * static_cast<size_t>(this->intf_->d),
        "The size of codes must be equal to ntotal * dim.\n");

    if (ids == nullptr) {
        // set ids
        std::vector<idx_t> idsInner(index->ntotal);
        for (size_t i = 0; i < idsInner.size(); ++i) {
            idsInner[i] = this->intf_->ntotal + static_cast<idx_t>(i);
        }
        copyPaged(index, idsInner.data());
    } else {
        copyPaged(index, ids);
    }

    APP_LOG_INFO("AscendIndexSQ copyCode operation finished.\n");
}

void AscendIndexSQImpl::copyPaged(const faiss::IndexScalarQuantizer *index, const idx_t *ids)
{
    size_t totalSize = static_cast<size_t>(index->ntotal) * getAddElementSize();
    if ((totalSize > ADD_PAGE_SIZE) || (static_cast<size_t>(index->ntotal) > ADD_VEC_SIZE)) {
        size_t tileSize = getAddPagedSize(index->ntotal);

        for (size_t i = 0; i < static_cast<size_t>(index->ntotal); i += tileSize) {
            size_t curNum = std::min(tileSize, index->ntotal - i);
            if (this->intf_->verbose) {
                printf("AscendIndex::add: adding %zu:%zu / %ld\n", i, i + curNum, index->ntotal);
            }
            copyImpl(curNum, index->codes.data() + i * static_cast<size_t>(this->intf_->d), ids ? (ids + i) : nullptr);
        }
    } else {
        if (this->intf_->verbose) {
            printf("AscendIndex::add: adding 0:%ld / %ld\n", index->ntotal, index->ntotal);
        }
        copyImpl(index->ntotal, index->codes.data(), ids);
    }
}

void AscendIndexSQImpl::copyImpl(int n, const uint8_t *codes, const idx_t *ids)
{
    APP_LOG_INFO("AscendIndexSQ copyImpl operation started.\n");
    FAISS_THROW_IF_NOT(n > 0);

    // 1. compute addMap
    size_t deviceCnt = indexConfig.deviceList.size();
    std::vector<int> addMap(deviceCnt, 0);
    calcAddMap(n, addMap);

    // 2. compute the preCompute values
    std::unique_ptr<float[]> preComputeVals = std::make_unique<float[]>(n);
    calcPreCompute(codes, preComputeVals.get(), n);
    // 3. transfer the codes and preCompute to the device
    add2DeviceFast(n, const_cast<uint8_t *>(codes), ids, preComputeVals.get(), addMap);
    APP_LOG_INFO("AscendIndexSQ copyImpl operation finished.\n");
}

void AscendIndexSQImpl::copyTo(faiss::IndexScalarQuantizer *index) const
{
    auto lock = ::ascend::AscendMultiThreadManager::GetReadLock(mtx);
    APP_LOG_INFO("AscendIndexSQ copyTo operation started.\n");
    FAISS_THROW_IF_NOT(index != nullptr);
    index->reset();
    index->metric_type = this->intf_->metric_type;
    index->sq = sq;
    index->code_size = sq.code_size;
    index->d = this->intf_->d;
    index->ntotal = this->intf_->ntotal;
    index->is_trained = this->intf_->is_trained;

    if (this->intf_->is_trained && this->intf_->ntotal > 0) {
        size_t deviceCnt = indexConfig.deviceList.size();
        std::vector<size_t> getOffset(deviceCnt, 0);
        size_t total = 0;
        for (size_t i = 0; i < deviceCnt; i++) {
            getOffset[i] = total;
            total += getBaseSize(indexConfig.deviceList[i]);
        }

        std::vector<uint8_t> baseData(total * static_cast<size_t>(intf_->d));
        auto getFunctor = [this, &baseData, &getOffset] (int idx) {
            getBase(this->indexConfig.deviceList[idx], reinterpret_cast<char *>(
                baseData.data() + getOffset[idx] * static_cast<size_t>(intf_->d)));
        };
        CALL_PARALLEL_FUNCTOR(deviceCnt, pool, getFunctor);

        index->codes = std::move(baseData);
    }

    APP_LOG_INFO("AscendIndexSQ copyTo operation finished.\n");
}

void AscendIndexSQImpl::copyTo(faiss::IndexIDMap *index) const
{
    auto lock = ::ascend::AscendMultiThreadManager::GetReadLock(mtx);
    APP_LOG_INFO("AscendIndexSQ copyTo operation started.\n");
    FAISS_THROW_IF_NOT_MSG(index != nullptr, "Invalid index nullptr.");
    FAISS_THROW_IF_NOT_MSG(index->index != nullptr, "Invalid index nullptr.");

    copyTo(dynamic_cast<faiss::IndexScalarQuantizer *>(index->index));

    std::vector<idx_t> idsSq;
    for (auto device : indexConfig.deviceList) {
        std::vector<idx_t> ids;
        getIdxMap(device, ids);
        idsSq.insert(idsSq.end(), ids.begin(), ids.end());
    }
    index->id_map = std::move(idsSq);
    APP_LOG_INFO("AscendIndexSQ copyTo operation finished.\n");
}

void AscendIndexSQImpl::getBase(int deviceId, char* codes) const
{
    auto lock = ::ascend::AscendMultiThreadManager::GetReadLock(mtx);
    // getBase、getBaseEnd接口内部会使用修改成员变量dataVec、attrsVec，因此getBase接口不能并发
    auto getBaseLock = ::ascend::AscendMultiThreadManager::LockGetBaseMtx(deviceId, getBaseMtx);
    APP_LOG_INFO("AscendIndexSQ getBase operation started.\n");
    AscendIndexImpl::getBase(deviceId, codes);
    getActualIndex(deviceId)->getBaseEnd();
    APP_LOG_INFO("AscendIndexSQ getBase operation finished.\n");
}

size_t AscendIndexSQImpl::getBaseSize(int deviceId) const
{
    auto lock = ::ascend::AscendMultiThreadManager::GetReadLock(mtx);
    APP_LOG_INFO("AscendIndexSQ start to getBaseSize of device %d.\n", deviceId);
    FAISS_THROW_IF_NOT_FMT(indexes.find(deviceId) != indexes.end(),
        "deviceId is out of range, deviceId=%d.", deviceId);
    uint32_t size = 0;
    auto pIndex = getActualIndex(deviceId);
    size = static_cast<uint32_t>(pIndex->getSize());
    APP_LOG_INFO("AscendIndexSQ getBaseSize operation finished.\n");
    return static_cast<size_t>(size);
}

void AscendIndexSQImpl::getIdxMap(int deviceId, std::vector<idx_t> &idxMap) const
{
    auto lock = ::ascend::AscendMultiThreadManager::GetReadLock(mtx);
    APP_LOG_INFO("AscendIndexSQ start to getIdxMap of device %d.\n", deviceId);
    AscendIndexImpl::getIdxMap(deviceId, idxMap);
    APP_LOG_INFO("AscendIndexSQ getIdxMap operation finished.\n");
}

void AscendIndexSQImpl::train(idx_t n, const float *x)
{
    auto lock = ::ascend::AscendMultiThreadManager::GetWriteLock(mtx);
    APP_LOG_INFO("AscendIndexSQ start to train with %ld vector(s).\n", n);
    FAISS_THROW_IF_NOT_MSG(x, "x can not be nullptr.");
    FAISS_THROW_IF_NOT_FMT((n > 0) && (n < MAX_N), "n must be > 0 and < %ld", MAX_N);

    if (this->intf_->is_trained) {
        FAISS_THROW_IF_NOT_MSG(indexes.size() > 0, "indexes.size must be >0");
        return;
    }

    // use the ScalarQuantizer to train data
    sq.train(n, x);

    // update the SQ param to device
    updateDeviceSQTrainedValue();

    this->intf_->is_trained = true;
    APP_LOG_INFO("AscendIndexSQ train operation finished.\n");
}

void AscendIndexSQImpl::search_with_masks(idx_t n, const float *x, idx_t k,
    float *distances, idx_t *labels, const void *mask) const
{
    auto lock = ::ascend::AscendMultiThreadManager::GetReadLock(mtx);
    APP_LOG_INFO("AscendIndexSQ search_with_masks operation started: n=%ld, k=%ld.\n", n, k);
    check(n, x, k, distances, labels);
    FAISS_THROW_IF_NOT_MSG(mask, "mask cannot be nullptr.");
    size_t reqMsgSize = static_cast<size_t>(intf_->d) * static_cast<size_t>(n) * sizeof(uint16_t) +
        static_cast<size_t>(n) * static_cast<size_t>(::ascend::utils::divUp(intf_->ntotal, BINARY_BYTE_SIZE)) *
        sizeof(uint8_t);
    FAISS_THROW_IF_NOT_FMT(reqMsgSize < MAX_SEARCH_WITH_MASK_REQ_MESSAGE_SIZE,
        "search_with_masks' request message size (dim * n * sizeof(uint16_t) + ⌈ntotal / 8⌉ * n = %zu) must be < %zu.",
        reqMsgSize, MAX_SEARCH_WITH_MASK_REQ_MESSAGE_SIZE);
    size_t deviceCnt = indexConfig.deviceList.size();
    // convert query data from float to fp16, device use fp16 data to search
    std::vector<uint16_t> query(n * this->intf_->d, 0);
    transform(x, x + n * this->intf_->d, begin(query), [](float temp) { return fp16(temp).data; });

    std::vector<std::vector<float>> dist(deviceCnt, std::vector<float>(n * k, 0));
    std::vector<std::vector<uint16_t>> distHalf(deviceCnt, std::vector<uint16_t>(n * k, 0));
    std::vector<std::vector<ascend_idx_t>> label(deviceCnt, std::vector<ascend_idx_t>(n * k, 0));

    auto searchFunctor = [&](int idx) {
        int deviceId = indexConfig.deviceList[idx];
        int64_t maskSize = (this->intf_->ntotal + 7) / 8;
        auto pIndex = getActualIndex(deviceId);
        using namespace ::ascend;
        AscendTensor<uint8_t, DIMS_2, int64_t> maskDevice({ n, maskSize });
        auto ret = aclrtMemcpy(maskDevice.data(), maskDevice.getSizeInBytes(), static_cast<const uint8_t *>(mask),
                               static_cast<size_t>(n) * static_cast<size_t>(maskSize) * sizeof(uint8_t),
                               ACL_MEMCPY_HOST_TO_DEVICE);
        FAISS_THROW_IF_NOT_FMT(ret == ACL_SUCCESS,  "aclrtMemcpy error %d", ret);
        AscendTensor<uint16_t, DIMS_2, int64_t> tensorDevQueries({ n, this->intf_->d });
        ret = aclrtMemcpy(tensorDevQueries.data(), tensorDevQueries.getSizeInBytes(),
                          query.data(), n * this->intf_->d * sizeof(uint16_t), ACL_MEMCPY_HOST_TO_DEVICE);
        FAISS_THROW_IF_NOT_FMT(ret == ACL_SUCCESS,  "aclrtMemcpy error %d", ret);
        ret = pIndex->searchFilter(n, tensorDevQueries.data(), k, distHalf[idx].data(),
                                   static_cast<::ascend::idx_t *>(label[idx].data()), maskDevice.data());
        FAISS_THROW_IF_NOT_FMT(ret == APP_ERR_OK, "Failed to searchfilter index,result is: %d\n", ret);
        // convert result data from fp16 to float
        transform(begin(distHalf[idx]), end(distHalf[idx]), begin(dist[idx]),
            [](uint16_t temp) { return (float)fp16(temp); });
    };
    CALL_PARALLEL_FUNCTOR(deviceCnt, pool, searchFunctor);
    // post process: default is merge the topk results from devices
    searchPostProcess(deviceCnt, dist, label, n, k, distances, labels);
    APP_LOG_INFO("AscendIndexSQ search_with_masks operation finished.\n");
}

void AscendIndexSQImpl::search_with_filter(idx_t n, const float *x, idx_t k,
    float *distances, idx_t *labels, const void *filters) const
{
    auto lock = ::ascend::AscendMultiThreadManager::GetReadLock(mtx);
    APP_LOG_INFO("AscendIndexSQ search_with_filter operation started: n=%ld, k=%ld.\n", n, k);
    check(n, x, k, distances, labels);
    FAISS_THROW_IF_NOT_MSG(filters, "filters cannot be nullptr.");
    FAISS_THROW_IF_NOT_MSG(indexConfig.filterable, "search filter not support as sqconfig.filterable = false");
    size_t deviceCnt = indexConfig.deviceList.size();
    // convert query data from float to fp16, device use fp16 data to search
    std::vector<uint16_t> query(n * this->intf_->d, 0);
    transform(x, x + n * this->intf_->d, begin(query), [](float temp) { return fp16(temp).data; });

    std::vector<std::vector<float>> dist(deviceCnt, std::vector<float>(n * k, 0));
    std::vector<std::vector<uint16_t>> distHalf(deviceCnt, std::vector<uint16_t>(n * k, 0));
    std::vector<std::vector<ascend_idx_t>> label(deviceCnt, std::vector<ascend_idx_t>(n * k, 0));

    auto searchFunctor = [&](int idx) {
        int deviceId = indexConfig.deviceList[idx];
        auto pIndex = getActualIndex(deviceId);
        using namespace ::ascend;
        AscendTensor<uint16_t, DIMS_2, int64_t> tensorDevQueries({ n, this->intf_->d });
        auto ret = aclrtMemcpy(tensorDevQueries.data(), tensorDevQueries.getSizeInBytes(),
                               query.data(), n * this->intf_->d * sizeof(uint16_t), ACL_MEMCPY_HOST_TO_DEVICE);
        FAISS_THROW_IF_NOT_FMT(ret == ACL_SUCCESS, "aclrtMemcpy error %d", ret);
        ret = pIndex->searchFilter(n, tensorDevQueries.data(), k, distHalf[idx].data(),
                                   static_cast<::ascend::idx_t *>(label[idx].data()),
                                   static_cast<uint64_t>(n) * FILTER_SIZE,
                                   const_cast<uint32_t *>(static_cast<const uint32_t *>(filters)));
        FAISS_THROW_IF_NOT_FMT(ret == APP_ERR_OK, "Failed to searchfilter result : %d\n", ret);
        transform(begin(distHalf[idx]), end(distHalf[idx]), begin(dist[idx]),
            [](uint16_t temp) { return (float)fp16(temp); });
    };
    CALL_PARALLEL_FUNCTOR(deviceCnt, pool, searchFunctor);
    // post process: default is merge the topk results from devices
    searchPostProcess(deviceCnt, dist, label, n, k, distances, labels);
    APP_LOG_INFO("AscendIndexSQ search_with_filter operation finished.\n");
}

void AscendIndexSQImpl::updateDeviceSQTrainedValue() const
{
    APP_LOG_INFO("AscendIndexSQ updateDeviceSQTrainedValue operation started.\n");
    // convert trained value to fp16, contain vmin and vdiff, so need *2
    std::vector<uint16_t> trainedFp16(this->intf_->d * 2);
    uint16_t *vmin = trainedFp16.data();
    uint16_t *vdiff = trainedFp16.data() + this->intf_->d;

    switch (sq.qtype) {
        case faiss::ScalarQuantizer::QT_8bit:
            transform(begin(sq.trained), end(sq.trained), begin(trainedFp16),
                [](float temp) { return fp16(temp).data; });
            break;
        default:
            FAISS_THROW_FMT("not supportted qtype(%d).", sq.qtype);
            break;
    }
    for (const auto &deviceId : indexConfig.deviceList) {
            auto pIndex = getActualIndex(deviceId);
            using namespace ::ascend;
            AscendTensor<float16_t, DIMS_1> vminTensor(vmin, { this->intf_->d });
            AscendTensor<float16_t, DIMS_1> vdiffTensor(vdiff, { this->intf_->d });
            pIndex->updateTrainedValue(vminTensor, vdiffTensor);
    }
    APP_LOG_INFO("AscendIndexSQ updateDeviceSQTrainedValue operation finished.\n");
}

void AscendIndexSQImpl::indexAddVector(int deviceId, const int ntotal, const uint8_t *codes,
                                       const float *precomputedVal, const idx_t *ids) const
{
    auto pIndex = getActualIndex(deviceId);
    using namespace ::ascend;
    if (ids != nullptr) {
        auto ret = pIndex->addVectorsWithIds(ntotal, codes, reinterpret_cast<const uint64_t *>(ids), precomputedVal);
        FAISS_THROW_IF_NOT_FMT(ret == APP_ERR_OK, "Failed to addwith id to index, result is %d", ret);
    } else {
        auto ret = pIndex->addVectors(ntotal, codes, precomputedVal);
        FAISS_THROW_IF_NOT_FMT(ret == APP_ERR_OK, "Failed to add to index, result is %d", ret);
    }
}

void AscendIndexSQImpl::getBaseImpl(int deviceId, int offset, int n, char *x) const
{
    APP_LOG_INFO("AscendIndexSQ getBaseImpl operation started: deviceId=%d, offset=%d, n=%d.\n",
        deviceId, offset, n);
    std::vector<uint8_t> baseData(n);
    auto pIndex = getActualIndex(deviceId);
    auto ret = pIndex->getVectors(offset, n, baseData);
    FAISS_THROW_IF_NOT_FMT(ret == ::ascend::APP_ERR_OK, "Failed to get vectors, result is %d", ret);
    auto xi = reinterpret_cast<uint8_t *>(x) + static_cast<size_t>(offset) * static_cast<size_t>(this->intf_->d);
    transform(baseData.begin(), baseData.end(), xi, [](uint8_t temp) { return temp; });
    APP_LOG_INFO("AscendIndexSQ getBaseImpl operation finished.\n");
}

size_t AscendIndexSQImpl::getBaseElementSize() const
{
    // element size: codesize + sizeof(preCompute)
    return sq.code_size * sizeof(uint8_t);
}

void AscendIndexSQImpl::add2DeviceFast(int n, const uint8_t *codes, const idx_t *ids, const float *preCompute,
    std::vector<int> &addMap)
{
    APP_LOG_INFO("AscendIndexSQ start to fast add %d vector(s) to device(s).\n", n);
    size_t deviceCnt = indexConfig.deviceList.size();

    int offsum = 0;
    std::vector<int> offsumMap(deviceCnt, 0);
    for (size_t i = 0; i < deviceCnt; i++) {
        offsumMap[i] = offsum;
        int num = addMap.at(i);
        offsum += num;
    }

    auto addFunctor = [&](int idx) {
        int num = addMap.at(idx);
        if (num != 0) {
            int deviceId = indexConfig.deviceList[idx];
            std::vector<uint8_t> codeb(static_cast<size_t>(num) * sq.code_size);
            codeb.assign(codes + offsumMap[idx] * sq.code_size, codes +
                (offsumMap[idx] + num) *sq.code_size);
            const float *preComputeToAdd = preCompute + offsumMap[idx];
            if (!indexConfig.filterable) {
                indexAddVector(deviceId, num, codeb.data(), preComputeToAdd);
            } else {
                std::vector<idx_t> idVec(num);
                idVec.assign(ids + offsumMap[idx], ids + num + offsumMap[idx]);
                indexAddVector(deviceId, num, codeb.data(), preComputeToAdd, idVec.data());
            }
            this->idxDeviceMap[idx].insert(this->idxDeviceMap[idx].end(), ids + offsumMap[idx],
                                           ids + num + offsumMap[idx]);
        }
    };

    CALL_PARALLEL_FUNCTOR(deviceCnt, pool, addFunctor);
    this->intf_->ntotal += n;
    APP_LOG_INFO("AscendIndexSQ add2DeviceFast operation finished.\n");
}

void AscendIndexSQImpl::add2Device(int n, const uint8_t *codes, const idx_t *ids, float *preCompute,
    std::vector<int> &addMap)
{
    APP_LOG_INFO("AscendIndexSQ start to add %d vector(s) to device(s).\n", n);
    size_t deviceCnt = indexConfig.deviceList.size();
    uint32_t offsum = 0;
    for (size_t i = 0; i < deviceCnt; i++) {
        int num = addMap.at(i);
        if (num == 0) {
            continue;
        }
        int deviceId = indexConfig.deviceList[i];

        std::vector<uint8_t> codeb(static_cast<size_t>(num) * sq.code_size);
        codeb.assign(codes + offsum * sq.code_size, codes + (offsum + num) * sq.code_size);
        float *preComputeToAdd = preCompute + offsum;

        if (!indexConfig.filterable) {
            indexAddVector(deviceId, num, codeb.data(), preComputeToAdd);
        } else {
            std::vector<idx_t> idVec(num);
            idVec.assign(ids + offsum, ids + num + offsum);
            indexAddVector(deviceId, num, codeb.data(), preComputeToAdd, idVec.data());
        }

        // record ids of new adding vector
        this->idxDeviceMap[i].insert(this->idxDeviceMap[i].end(), ids + offsum, ids + num + offsum);
        offsum += static_cast<size_t>(num);
    }
    this->intf_->ntotal += n;
    APP_LOG_INFO("AscendIndexSQ add2Device operation finished.\n");
}

void AscendIndexSQImpl::addImpl(int n, const float *x, const idx_t *ids)
{
    APP_LOG_INFO("AscendIndexSQ addImpl operation started: n=%d.\n", n);
    FAISS_THROW_IF_NOT(n > 0);

    // 1. compute addMap
    size_t deviceCnt = indexConfig.deviceList.size();
    std::vector<int> addMap(deviceCnt, 0);
    calcAddMap(n, addMap);

    // 2. compute the sq codes
    std::unique_ptr<uint8_t[]> codes = std::make_unique<uint8_t[]>(static_cast<size_t>(n) * sq.code_size);
    sq.compute_codes(x, codes.get(), n);

    // 3. compute the preCompute values

    std::unique_ptr<float[]> preComputeVals = std::make_unique<float[]>(n);
    calcPreCompute(codes.get(), preComputeVals.get(), n);

    // 4. transfer the codes and preCompute to the device
    add2Device(n, codes.get(), ids, preComputeVals.get(), addMap);
    APP_LOG_INFO("AscendIndexSQ addImpl operation finished.\n");
}

void AscendIndexSQImpl::calcPreCompute(const uint8_t *codes, float *compute, size_t n, float *xMem)
{
    APP_LOG_INFO("AscendIndexSQ calcPreCompute operation started.\n");
    float *x = xMem;
    std::unique_ptr<float[]> tmpPtr;
    if (!x) {
        x = new (std::nothrow) float[n * static_cast<size_t>(this->intf_->d)];
        FAISS_THROW_IF_NOT_MSG(x != nullptr, "Memory allocation fail for x");
        tmpPtr.reset(x);
    }
    sq.decode(codes, x, n);

    fvec_norms_L2sqr(compute, x, this->intf_->d, n);
    APP_LOG_INFO("AscendIndexSQ calcPreCompute operation finished.\n");
}

size_t AscendIndexSQImpl::getAddElementSize() const
{
    // element size: codesize + sizeof(preCompute)
    return sq.code_size * sizeof(uint8_t) + sizeof(float);
}
} // ascend
} // faiss
