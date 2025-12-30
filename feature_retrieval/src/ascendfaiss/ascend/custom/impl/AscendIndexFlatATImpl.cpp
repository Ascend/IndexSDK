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


#include "ascend/custom/impl/AscendIndexFlatATImpl.h"

#include <algorithm>
#include <omp.h>

#include <faiss/IndexFlat.h>
#include <faiss/impl/AuxIndexStructures.h>

#include "ascenddaemon/utils/AscendUtils.h"
#include "ascend/utils/fp16.h"
#include "common/threadpool/AscendThreadPool.h"
#include "common/utils/LogUtils.h"

namespace faiss {
namespace ascend {
namespace {
const size_t KB = 1024;
const size_t RETAIN_SIZE = 2048;
const size_t UNIT_PAGE_SIZE = 64;
const size_t UNIT_VEC_SIZE = 512;

// search pagesize must be less than 64M, becauseof rpc limitation
const size_t SEARCH_PAGE_SIZE = UNIT_PAGE_SIZE * KB * KB - RETAIN_SIZE;

// Or, maximum number 512K of vectors to consider per page of search
const size_t SEARCH_VEC_SIZE = UNIT_VEC_SIZE * KB;

// the number of base vector must be multible of 1024
const int BASE_ALIGN = 1024;

// the page size of search must be multibleof 32768
const int PAGE_ALIGN = 32768;
const std::vector<int> DIMS = { 64, 128, 256 };
}
// implementation of AscendIndexFlatAT
AscendIndexFlatATImpl::AscendIndexFlatATImpl(int dims, int baseSize, AscendIndexFlatATConfig config, AscendIndex *intf)
    : AscendIndexImpl(dims, MetricType::METRIC_L2, config, intf), flatConfig(config), baseSize(baseSize)
{
    FAISS_THROW_IF_NOT_MSG(baseSize % BASE_ALIGN == 0, "baseSize must be multiple of 1024");
    FAISS_THROW_IF_NOT_MSG(std::find(DIMS.begin(), DIMS.end(), dims) != DIMS.end(), "Unsupported dims");
    // Flat index doesn't need training
    this->intf_->is_trained = true;

    initIndexes();

    // initial idxDeviceMap mem space
    idxDeviceMap.clear();
    idxDeviceMap.resize(config.deviceList.size());
}

AscendIndexFlatATImpl::~AscendIndexFlatATImpl() {}

std::shared_ptr<::ascend::Index> AscendIndexFlatATImpl::createIndex(int deviceId)
{
    APP_LOG_INFO("AscendIndexFlatAT createIndex operation started, device id: %d\n", deviceId);
    auto res = aclrtSetDevice(deviceId);
    FAISS_THROW_IF_NOT_FMT(res == 0, "acl set device failed %d, deviceId: %d", res, deviceId);
    std::shared_ptr<::ascend::Index> index =
        std::make_shared<::ascend::IndexFlatATAicpu>(this->intf_->d, this->baseSize, indexConfig.resourceSize);
    auto ret = index->init();
    FAISS_THROW_IF_NOT_FMT(ret == ::ascend::APP_ERR_OK, "Failed to create IndexFlatATAicpu, result is %d", ret);
    APP_LOG_INFO("AscendIndexFlatAT createIndex operation finished.\n");
    return index;
}

void AscendIndexFlatATImpl::reset()
{
    APP_LOG_INFO("AscendIndexFlatAT reset operation started.\n");
    for (auto &data : indexes) {
        int deviceId = data.first;
        auto pIndex = getActualIndex(deviceId);
        auto ret = pIndex->reset();
        FAISS_THROW_IF_NOT_FMT(ret == ::ascend::APP_ERR_OK,
                               "Failed to reset index result : %d,device id: %d\n", ret, deviceId);
    }

    size_t deviceNum = indexConfig.deviceList.size();
    idxDeviceMap.clear();
    idxDeviceMap.resize(deviceNum);

    this->intf_->ntotal = 0;
    APP_LOG_INFO("AscendIndexFlatAT reset operation finished.\n");
}

void AscendIndexFlatATImpl::addImpl(int n, const float *x, const idx_t *ids)
{
    APP_LOG_INFO("AscendIndexFlatAT addImpl operation started.\n");
    FAISS_THROW_IF_NOT(n > 0);

    auto addFunctor = [&](int idx) {
        int deviceId = indexConfig.deviceList[idx];
        std::vector<uint16_t> xb(n * this->intf_->d);
        transform(x, x + n * this->intf_->d, std::begin(xb), [](float temp) { return fp16(temp).data; });

        auto *pIndex = getActualIndex(deviceId);
        using namespace ::ascend;
        AscendTensor<float16_t, DIMS_2> vec({ n, this->intf_->d });
        auto ret = aclrtMemcpy(vec.data(), vec.getSizeInBytes(), xb.data(), n * this->intf_->d * sizeof(float16_t),
                               ACL_MEMCPY_HOST_TO_DEVICE);
        FAISS_THROW_IF_NOT_FMT(ret == ACL_SUCCESS, "aclrtMemcpy error %d", ret);
        ret = pIndex->addVectors(vec);
        FAISS_THROW_IF_NOT_FMT(ret == APP_ERR_OK, "Failed to add to index, result is %d", ret);
        // record ids of new adding vector
        idxDeviceMap[idx].insert(idxDeviceMap[idx].end(), ids, ids + n);
    };

    CALL_PARALLEL_FUNCTOR(indexConfig.deviceList.size(), pool, addFunctor);
    this->intf_->ntotal += n;
    APP_LOG_INFO("AscendIndexFlatAT addImpl operation finished.\n");
}

void AscendIndexFlatATImpl::searchPaged(int n, const float *x, int k, float *distances, idx_t *labels) const
{
    APP_LOG_INFO("AscendIndexFlatAT start to searchPaged: n=%d, k=%d.\n", n, k);
    size_t deviceCnt = indexConfig.deviceList.size();
    size_t totalSize = static_cast<size_t>(n) * static_cast<size_t>(this->intf_->d) * sizeof(float);
    size_t totalOutSize = (size_t)n * (size_t)k * (sizeof(uint16_t) + sizeof(ascend_idx_t));

    if (totalSize > (deviceCnt * SEARCH_PAGE_SIZE) || (size_t)n > (deviceCnt * SEARCH_VEC_SIZE) ||
        totalOutSize > (deviceCnt * SEARCH_PAGE_SIZE)) {
        size_t tileSize = getSearchPagedSize(n, k) * deviceCnt;

        for (size_t i = 0; i < (size_t)n; i += tileSize) {
            size_t curNum = std::min(tileSize, (size_t)(n)-i);
            searchImpl(curNum, x + i * (size_t)this->intf_->d, k, distances + i * k, labels + i * k);
        }
    } else {
        searchImpl(n, x, k, distances, labels);
    }
    APP_LOG_INFO("AscendIndexFlatAT searchPaged operation finished.\n");
}

void AscendIndexFlatATImpl::searchImpl(int n, const float *x, int k, float *distances, idx_t *labels) const
{
    APP_LOG_INFO("AscendIndexFlatAT searchImpl operation started: n=%d, k=%d.\n", n, k);
    size_t deviceCnt = indexConfig.deviceList.size();
    int searchNum = (n + static_cast<int>(deviceCnt) - 1) / static_cast<int>(deviceCnt);

    // convert query data from float to fp16, device use fp16 data to search
    std::vector<uint16_t> query(n * this->intf_->d, 0);
    transform(x, x + n * this->intf_->d, begin(query), [](float temp) { return fp16(temp).data; });

    std::vector<uint16_t> distHalf(n * k, 0);
    std::vector<ascend_idx_t> label(n * k, 0);

    auto searchFunctor = [&](int idx) {
        int deviceId = indexConfig.deviceList[idx];
        int offset = idx * searchNum;
        int num = std::min(n - offset, searchNum);
        if (num > 0) {
            IndexParam<uint16_t, uint16_t, ascend_idx_t> param(deviceId, num, this->intf_->d, k);
            param.query =  query.data() + offset * this->intf_->d;
            param.distance = distHalf.data() + offset * k;
            param.label = label.data() + offset * k;
            indexSearch(param);
            // convert result data from fp16 to float
            std::transform(distHalf.data() + offset * k, distHalf.data() + (offset + num) * k, distances + offset * k,
                [](uint16_t temp) { return (float)fp16(temp); });
            std::transform(label.data() + offset * k, label.data() + (offset + num) * k, labels + offset * k,
                [&](ascend_idx_t temp) {
                    return (temp != std::numeric_limits<ascend_idx_t>::max()) ?
                        temp :
                        std::numeric_limits<ascend_idx_t>::max();
                });
        }
    };

    CALL_PARALLEL_FUNCTOR(deviceCnt, pool, searchFunctor);
    APP_LOG_INFO("AscendIndexFlatAT searchImpl operation finished.\n");
}

void AscendIndexFlatATImpl::search(idx_t n, const float *x, idx_t k, float *distances,
    idx_t *labels) const
{
    APP_LOG_INFO("AscendIndexFlatAT start to search: searchNum=%ld, topK=%ld.\n", n, k);
    FAISS_THROW_IF_NOT_MSG(this->intf_->ntotal == this->baseSize, "ntotal must equal baseSize");
    AscendIndexImpl::search(n, x, k, distances, labels);
    APP_LOG_INFO("AscendIndexFlatAT search operation finished.\n");
}

size_t AscendIndexFlatATImpl::getAddElementSize() const
{
    return static_cast<size_t>(this->intf_->d) * sizeof(uint16_t);
}

size_t AscendIndexFlatATImpl::getBaseElementSize() const
{
    return static_cast<size_t>(this->intf_->d) * sizeof(uint16_t) + sizeof(float);
}

size_t AscendIndexFlatATImpl::removeImpl(const IDSelector &sel)
{
    VALUE_UNUSED(sel);

    FAISS_THROW_MSG("the removeImpl() is not implemented for this type of index");
}

size_t AscendIndexFlatATImpl::getSearchPagedSize(int n, int k) const
{
    return AscendIndexImpl::getSearchPagedSize(n, k) / PAGE_ALIGN * PAGE_ALIGN;
}

void AscendIndexFlatATImpl::clearAscendTensor()
{
    APP_LOG_INFO("AscendIndexFlatAT clearTmpAscendTensor operation started.\n");

    auto clearFunctor = [&](int idx) {
        int deviceId = indexConfig.deviceList[idx];
        auto pIndex = getActualIndex(deviceId);
        pIndex->clearTmpAscendTensor();
    };

    CALL_PARALLEL_FUNCTOR(indexConfig.deviceList.size(), pool, clearFunctor);
    APP_LOG_INFO("AscendIndexFlatAT clearTmpAscendTensor operation finished.\n");
}
} // ascend
} // faiss
