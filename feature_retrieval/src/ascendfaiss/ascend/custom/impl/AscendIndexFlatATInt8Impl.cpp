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


#include "ascend/custom/impl/AscendIndexFlatATInt8Impl.h"

#include <algorithm>
#include <omp.h>

#include <faiss/IndexFlat.h>

#include "ascenddaemon/utils/AscendUtils.h"
#include "ascend/utils/fp16.h"
#include "impl/AuxIndexStructures.h"
#include "common/threadpool/AscendThreadPool.h"
#include "common/utils/LogUtils.h"

namespace faiss {
namespace ascend {
namespace {
const size_t KB = 1024;
const size_t RETAIN_SIZE = 2048;
const size_t UNIT_PAGE_SIZE = 64;
const size_t UNIT_VEC_SIZE = 512;
const int ENCODE_MIN_MAX_SIZE = 2;

// search pagesize must be less than 64M, becauseof rpc limitation
const size_t SEARCH_PAGE_SIZE = UNIT_PAGE_SIZE * KB * KB - RETAIN_SIZE;

// Or, maximum number 512K of vectors to consider per page of search
const size_t SEARCH_VEC_SIZE = UNIT_VEC_SIZE * KB;

// the number of base vector must be multible of 1024
const int BASE_ALIGN = 1024;

// the page size of search must be multibleof 32768
const int PAGE_ALIGN = 32768;
const std::vector<int> DIMS = { 256 };
const size_t MAX_BASE_SIZE = 102400;
}
// implementation of AscendIndexFlatATInt8
AscendIndexFlatATInt8Impl::AscendIndexFlatATInt8Impl(int dims, int baseSize, AscendIndexFlatATInt8Config config,
    AscendIndex *intf)
    : AscendIndexImpl(dims, MetricType::METRIC_L2, config, intf), flatConfig(config), baseSize(baseSize)
{
    FAISS_THROW_IF_NOT_MSG(baseSize % BASE_ALIGN == 0, "baseSize must be multiple of 1024");
    FAISS_THROW_IF_NOT_FMT(baseSize > 0 && (size_t)baseSize <= MAX_BASE_SIZE, "baseSize should be in range (0, %zu]!",
        MAX_BASE_SIZE);
    FAISS_THROW_IF_NOT_MSG(std::find(DIMS.begin(), DIMS.end(), dims) != DIMS.end(), "Unsupported dims");
    // Flat index doesn't need training
    this->intf_->is_trained = true;

    initIndexes();

    // initial idxDeviceMap mem space
    idxDeviceMap.clear();
    idxDeviceMap.resize(config.deviceList.size());
}

AscendIndexFlatATInt8Impl::~AscendIndexFlatATInt8Impl() {}

std::shared_ptr<::ascend::Index> AscendIndexFlatATInt8Impl::createIndex(int deviceId)
{
    APP_LOG_INFO("AscendIndexFlatATInt8  createIndex operation started, device id: %d\n", deviceId);
    auto res = aclrtSetDevice(deviceId);
    FAISS_THROW_IF_NOT_FMT(res == 0, "acl set device failed %d, deviceId: %d", res, deviceId);
    std::shared_ptr<::ascend::Index> index =
        std::make_shared<::ascend::IndexFlatATInt8Aicpu>(this->intf_->d, this->baseSize, indexConfig.resourceSize);
    auto ret = index->init();
    FAISS_THROW_IF_NOT_FMT(ret == ACL_SUCCESS, "Create IndexFlatATInt8 failed(%d).", ret);
    APP_LOG_INFO("AscendIndexFlatATInt8 createIndex operation finished.\n");
    return index;
}

void AscendIndexFlatATInt8Impl::reset()
{
    APP_LOG_INFO("AscendIndexFlatATInt8 reset operation started.\n");
    for (auto &data : indexes) {
        int deviceId = data.first;
        auto pIndex = getActualIndex(deviceId);
        auto ret = pIndex->reset();
        FAISS_THROW_IF_NOT_FMT(ret == ::ascend::APP_ERR_OK,
                               "Failed to reset index result : %d,device id: %d\n", ret, deviceId);
    }
    int deviceNum = static_cast<int>(indexConfig.deviceList.size());
    idxDeviceMap.clear();
    idxDeviceMap.resize(deviceNum);

    this->intf_->ntotal = 0;
    APP_LOG_INFO("AscendIndexFlatATInt8 reset operation finished.\n");
}

void AscendIndexFlatATInt8Impl::sendMinMax(float qMin, float qMax)
{
    APP_LOG_INFO("AscendIndexFlatATInt8 sendMinMax operation started.\n");
    FAISS_THROW_IF_NOT_MSG(qMin <= qMax, "qMin should not be greater than qMax.");
    std::vector<float> minMaxTmp;
    minMaxTmp.push_back(qMin);
    minMaxTmp.push_back(qMax);
    std::vector<uint16_t> minMax(ENCODE_MIN_MAX_SIZE);
    transform(minMaxTmp.data(), minMaxTmp.data() + ENCODE_MIN_MAX_SIZE, std::begin(minMax),
        [](float temp) { return fp16(temp).data; });

    for (auto &index : indexes) {
        auto pIndex = getActualIndex(index.first);
        const int encodeMinMaxSize = 2;
        ::ascend::AscendTensor<float16_t, ::ascend::DIMS_1> vec(minMax.data(), { encodeMinMaxSize });
        float16_t qMin = vec[0];
        float16_t qMax = vec[1];
        pIndex->updateQMinMax(qMin, qMax);
    }

    APP_LOG_INFO("AscendIndexFlatATInt8 sendMinMax operation finished.\n");
}

bool AscendIndexFlatATInt8Impl::addImplRequiresIDs() const
{
    return true;
}

void AscendIndexFlatATInt8Impl::addImpl(int n, const float *x, const idx_t *ids)
{
    APP_LOG_INFO("AscendIndexFlatATInt8 addImpl operation started.\n");
    FAISS_THROW_IF_NOT(n > 0);

    auto addFunctor = [&](int idx) {
        int deviceId = indexConfig.deviceList[idx];
        std::vector<uint16_t> xb(n * this->intf_->d);
        transform(x, x + n * this->intf_->d, std::begin(xb), [](float temp) { return fp16(temp).data; });

        auto pIndex = getActualIndex(deviceId);
        using namespace ::ascend;
        AscendTensor<uint16_t, DIMS_2> vec({ n, this->intf_->d });
        auto ret = aclrtMemcpy(vec.data(), vec.getSizeInBytes(), xb.data(), n * this->intf_->d * sizeof(uint16_t),
                               ACL_MEMCPY_HOST_TO_DEVICE);
        FAISS_THROW_IF_NOT_FMT(ret == ACL_SUCCESS, "aclrtMemcpy error %d", ret);
        ret = pIndex->addVectors(vec);
        FAISS_THROW_IF_NOT_FMT(ret == APP_ERR_OK, "Failed to add to index, result is %d", ret);
        // record ids of new adding vector
        idxDeviceMap[idx].insert(idxDeviceMap[idx].end(), ids, ids + n);
    };

    CALL_PARALLEL_FUNCTOR(indexConfig.deviceList.size(), pool, addFunctor);
    this->intf_->ntotal += n;
    APP_LOG_INFO("AscendIndexFlatATInt8 addImpl operation finished.\n");
}

void AscendIndexFlatATInt8Impl::searchPagedInt8(int n, const int8_t *x, int k, float *distances,
    idx_t *labels) const
{
    APP_LOG_INFO("AscendIndexFlatATInt8 start to searchPagedInt8: n=%d, k=%d.\n", n, k);
    size_t deviceCnt = indexConfig.deviceList.size();
    size_t totalSize = static_cast<size_t>(n) * static_cast<size_t>(this->intf_->d) * sizeof(int8_t);
    size_t totalOutSize = (size_t)n * (size_t)k * (sizeof(uint16_t) + sizeof(ascend_idx_t));

    if (totalSize > (deviceCnt * SEARCH_PAGE_SIZE) || (size_t)n > (deviceCnt * SEARCH_VEC_SIZE) ||
        totalOutSize > (deviceCnt * SEARCH_PAGE_SIZE)) {
        size_t tileSize = getSearchPagedSize(n, k) * deviceCnt;

        for (int i = 0; i < n; i += static_cast<int>(tileSize)) {
            size_t curNum = std::min(tileSize, (size_t)(n - i));
            size_t offset = static_cast<size_t>(i) * static_cast<size_t>(k);
            searchImplInt8(curNum, x + static_cast<size_t>(i) * static_cast<size_t>(this->intf_->d), k,
                distances + offset, labels + offset);
        }
    } else {
        searchImplInt8(n, x, k, distances, labels);
    }
    APP_LOG_INFO("AscendIndexFlatATInt8 searchPagedInt8 operation finished.\n");
}

void AscendIndexFlatATInt8Impl::searchImplInt8(int n, const int8_t *x, int k, float *distances,
    idx_t *labels) const
{
    APP_LOG_INFO("AscendIndexFlatATInt8 searchImplInt8 operation started: n=%d, k=%d.\n", n, k);
    size_t deviceCnt = indexConfig.deviceList.size();
    int searchNum = (n + static_cast<int>(deviceCnt) - 1) / static_cast<int>(deviceCnt);
    const int8_t *query = x;
    std::vector<uint16_t> distHalf(static_cast<size_t>(n) * static_cast<size_t>(k), 0);
    std::vector<ascend_idx_t> label(distHalf.size(), 0);

    auto searchFunctor = [&](int idx) {
        int deviceId = indexConfig.deviceList[idx];
        int offset = idx * searchNum;
        int num = std::min(n - offset, searchNum);
        if (num > 0) {
            using namespace ::ascend;
            auto *pIndex = getActualIndex(deviceId);
            AscendTensor<int8_t, DIMS_2> tensorDevQueries({ num, this->intf_->d });
            auto ret = aclrtMemcpy(tensorDevQueries.data(), tensorDevQueries.getSizeInBytes(),
                                   query + static_cast<size_t>(offset) * static_cast<size_t>(this->intf_->d),
                                   static_cast<size_t>(num) * static_cast<size_t>(this->intf_->d) * sizeof(int8_t),
                                   ACL_MEMCPY_HOST_TO_DEVICE);
            FAISS_THROW_IF_NOT_FMT(ret == ACL_SUCCESS, "aclrtMemcpy error %d", ret);
            size_t searchOffset = static_cast<size_t>(offset) * static_cast<size_t>(k);
            ret = pIndex->searchInt8(num, tensorDevQueries.data(), k,
                                     distHalf.data() + searchOffset,
                                     label.data() + searchOffset);
            FAISS_THROW_IF_NOT_FMT(ret == APP_ERR_OK, "Failed to add to index, result is %d", ret);

            // convert result data from fp16 to float
            std::transform(distHalf.data() + searchOffset,
                distHalf.data() + static_cast<size_t>(offset + num) * static_cast<size_t>(k),
                distances + searchOffset,
                [](uint16_t temp) { return (float)fp16(temp); });
            std::transform(label.data() + searchOffset,
                label.data() + static_cast<size_t>(offset + num) * static_cast<size_t>(k),
                labels + searchOffset,
                [&](ascend_idx_t temp) {
                    return (temp != std::numeric_limits<ascend_idx_t>::max()) ?
                        temp : std::numeric_limits<ascend_idx_t>::max();
                });
        }
    };
    CALL_PARALLEL_FUNCTOR(deviceCnt, pool, searchFunctor);
    APP_LOG_INFO("AscendIndexFlatATInt8 searchImpl operation finished.\n");
}

void AscendIndexFlatATInt8Impl::searchInt8(idx_t n, const int8_t *x, idx_t k, float *distances,
    idx_t *labels) const
{
    APP_LOG_INFO("AscendIndexFlatATInt8 start to searchInt8: searchNum=%ld, topK=%ld.\n", n, k);
    FAISS_THROW_IF_NOT_MSG(this->intf_->ntotal == this->baseSize, "ntotal must equal baseSize");
    FAISS_THROW_IF_NOT_MSG(x, "x can not be nullptr.");
    FAISS_THROW_IF_NOT_MSG(distances, "distances can not be nullptr.");
    FAISS_THROW_IF_NOT_MSG(labels, "labels can not be nullptr.");
    FAISS_THROW_IF_NOT_FMT((k > 0) && (k <= MAX_K), "k must be > 0 and <= %ld", MAX_K);
    FAISS_THROW_IF_NOT_MSG(this->intf_->is_trained, "Index not trained");
    FAISS_THROW_IF_NOT_FMT((n > 0) && (n < MAX_N), "n must be > 0 and < %ld", MAX_N);
    FAISS_THROW_IF_NOT_MSG(indexes.size() > 0, "indexes.size must be >0");

    searchPagedInt8(n, x, k, distances, labels);
    APP_LOG_INFO("AscendIndexFlatATInt8 searchInt8 operation finished.\n");
}

void AscendIndexFlatATInt8Impl::searchPaged(int n, const float *x, int k, float *distances, idx_t *labels) const
{
    FAISS_THROW_MSG("searchPaged() not implemented for this type of index");
    VALUE_UNUSED(n);
    VALUE_UNUSED(x);
    VALUE_UNUSED(k);
    VALUE_UNUSED(distances);
    VALUE_UNUSED(labels);
}

void AscendIndexFlatATInt8Impl::searchImpl(int n, const float *x, int k, float *distances, idx_t *labels) const
{
    FAISS_THROW_MSG("searchImpl() not implemented for this type of index");
    VALUE_UNUSED(n);
    VALUE_UNUSED(x);
    VALUE_UNUSED(k);
    VALUE_UNUSED(distances);
    VALUE_UNUSED(labels);
}

size_t AscendIndexFlatATInt8Impl::getAddElementSize() const
{
    return static_cast<size_t>(this->intf_->d) * sizeof(int8_t);
}

size_t AscendIndexFlatATInt8Impl::getBaseElementSize() const
{
    return static_cast<size_t>(this->intf_->d) * sizeof(uint16_t) + sizeof(float);
}

size_t AscendIndexFlatATInt8Impl::removeImpl(const IDSelector &sel)
{
    VALUE_UNUSED(sel);

    FAISS_THROW_MSG("the removeImpl() is not implemented for this type of index");
}

size_t AscendIndexFlatATInt8Impl::getSearchPagedSize(int n, int k) const
{
    return AscendIndexImpl::getSearchPagedSize(n, k) / PAGE_ALIGN * PAGE_ALIGN;
}

void AscendIndexFlatATInt8Impl::clearAscendTensor()
{
    APP_LOG_INFO("AscendIndexFlatATInt8 clearTmpAscendTensor operation started.\n");

    auto clearFunctor = [&](int idx) {
        int deviceId = indexConfig.deviceList[idx];
        auto pIndex = getActualIndex(deviceId);
        pIndex->clearTmpAscendTensor();
    };

    CALL_PARALLEL_FUNCTOR(indexConfig.deviceList.size(), pool, clearFunctor);
    APP_LOG_INFO("AscendIndexFlatATInt8 clearTmpAscendTensor operation finished.\n");
}
} // ascend
} // faiss
