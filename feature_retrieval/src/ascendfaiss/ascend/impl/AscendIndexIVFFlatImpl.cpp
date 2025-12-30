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


#include "AscendIndexIVFFlatImpl.h"
#include <algorithm>
#include <faiss/utils/distances.h>
#include "ascend/AscendIndexQuantizerImpl.h"

namespace faiss {
namespace ascend {

// Default dim in case of nullptr index
const size_t DEFAULT_DIM = 128;
// Default nlist in case of nullptr index
const size_t DEFAULT_NLIST = 1024;

// The value range of dim
const std::vector<int> DIMS = { 128 };

// The value range of nlist
const std::vector<int> NLISTS = { 1024, 2048, 4096, 8192, 16384, 32768 };
const size_t KB = 1024;
const size_t RETAIN_SIZE = 2048;
const size_t UNIT_PAGE_SIZE = 640;
const size_t ADD_PAGE_SIZE = (UNIT_PAGE_SIZE * KB * KB - RETAIN_SIZE);
const size_t UNIT_VEC_SIZE = 5120;
const size_t ADD_VEC_SIZE = UNIT_VEC_SIZE * KB;
AscendIndexIVFFlatImpl::AscendIndexIVFFlatImpl(AscendIndexIVFFlat *intf, int dims, int nlist,
    faiss::MetricType metric, AscendIndexIVFFlatConfig config)
    : AscendIndexIVFImpl(intf, dims, metric, nlist, config), intf_(intf)
{
    checkParams();
    initIndexes();

    initDeviceAddNumMap();
    centroidsData.resize(nlist);
    this->intf_->is_trained = false;
}

AscendIndexIVFFlatImpl::~AscendIndexIVFFlatImpl() {}

void AscendIndexIVFFlatImpl::checkParams() const
{
    FAISS_THROW_IF_NOT_MSG(this->intf_->metric_type == MetricType::METRIC_INNER_PRODUCT, "Unsupported metric type");
    FAISS_THROW_IF_NOT_MSG(!this->ivfflatConfig.useKmeansPP, "Unsupported NPU clustering");
    FAISS_THROW_IF_NOT_MSG(std::find(NLISTS.begin(), NLISTS.end(), this->nlist) != NLISTS.end(), "Unsupported nlists");
    FAISS_THROW_IF_NOT_MSG(std::find(DIMS.begin(), DIMS.end(), this->intf_->d) != DIMS.end(), "Unsupported dims");
}

void AscendIndexIVFFlatImpl::addL1(int n, const float *x, std::unique_ptr<idx_t[]> &assign)
{
    if (this->ivfflatConfig.useKmeansPP) {
        APP_LOG_ERROR("not support npu clustering!!");
        return;
    } else {
        pQuantizerImpl->cpuQuantizer->assign(n, x, assign.get());
    }
}

void AscendIndexIVFFlatImpl::addImpl(int n, const float *x, const idx_t *ids)
{
    APP_LOG_INFO("AscendIndexIVFFlatImpl addImpl operation started: n=%d.\n", n);
    size_t deviceCnt = indexConfig.deviceList.size();
    std::unique_ptr<idx_t[]> assign = std::make_unique<idx_t[]>(n);
    addL1(n, x, assign);
    size_t dim = static_cast<size_t>(intf_->d);
    for (size_t i = 0; i <  static_cast<size_t>(n); i++) {
        idx_t listId = assign[i];
        FAISS_THROW_IF_NOT(listId >= 0 && listId < this->nlist);
        auto it = assignCounts.find(listId);
        if (it != assignCounts.end()) {
            it->second.Add(const_cast<float *>(x) + i * dim, ids + i);
            deviceAddNumMap[listId][it->second.addDeviceIdx]++;
            continue;
        }
        size_t devIdx = 0;
        for (size_t j = 1; j < deviceCnt; j++) {
            if (deviceAddNumMap[listId][j] < deviceAddNumMap[listId][devIdx]) {
                devIdx = j;
                break;
            }
        }
        deviceAddNumMap[listId][devIdx]++;
        assignCounts.emplace(listId, AscendIVFAddInfo(devIdx, deviceCnt, dim));
        assignCounts.at(listId).Add(const_cast<float *>(x) + i * dim, ids + i);
    }
}

void AscendIndexIVFFlatImpl::copyVectorToDevice(int n)
{
    size_t deviceCnt = indexConfig.deviceList.size();
    auto addFunctor = [&](int idx) {
        int deviceId = indexConfig.deviceList[idx];
        for (auto &centroid : assignCounts) {
            int listId = centroid.first;
            int num = centroid.second.GetAddNum(idx);
            if (num == 0) {
                continue;
            }
            float *codePtr = nullptr;
            ascend_idx_t *idPtr = nullptr;
            centroid.second.GetCodeAndIdPtr(idx, &codePtr, &idPtr);
            IndexParam<float, float, ascend_idx_t> param(deviceId, num, 0, 0);
            param.listId = listId;
            param.query = codePtr;
            param.label = idPtr;
            indexIVFFlatAdd(param);
        }
    };
    CALL_PARALLEL_FUNCTOR(deviceCnt, pool, addFunctor);
    this->intf_->ntotal += n;
    APP_LOG_INFO("AscendIndexIVFSQ addImpl operation finished.\n");
}

size_t AscendIndexIVFFlatImpl::getAddPagedSize(int n) const
{
    APP_LOG_INFO("AscendIndex getAddPagedSize operation started.\n");
    size_t maxNumVecsForPageSize = ADD_PAGE_SIZE / getAddElementSize();
    // Always add at least 1 vector, if we have huge vectors
    maxNumVecsForPageSize = std::max(maxNumVecsForPageSize, static_cast<size_t>(1));
    APP_LOG_INFO("AscendIndex getAddPagedSize operation finished.\n");

    return std::min(static_cast<size_t>(n), maxNumVecsForPageSize);
}

void AscendIndexIVFFlatImpl::addPaged(int n, const float* x, const idx_t* ids)
{
    APP_LOG_INFO("AscendIndexIVFFlatImpl addPaged operation started.\n");
    size_t totalSize = static_cast<size_t>(n) * getAddElementSize();
    size_t addPageSize = ADD_PAGE_SIZE;
    if (totalSize > addPageSize || static_cast<size_t>(n) > ADD_VEC_SIZE * 10) {
        size_t tileSize = getAddPagedSize(n);
        for (size_t i = 0; i < static_cast<size_t>(n); i += tileSize) {
            size_t curNum = std::min(tileSize, n - i);
            if (this->intf_->verbose) {
                printf("AscendIndexIVFFlatImpl::add: adding %zu:%zu / %d\n", i, i + curNum, n);
            }
            addImpl(curNum, x + i * static_cast<size_t>(this->intf_->d), ids ? (ids + i) : nullptr);
        }
    } else {
        if (this->intf_->verbose) {
            printf("AscendIndexIVFFlatImpl::add: adding 0:%d / %d\n", n, n);
        }
        addImpl(n, x, ids);
    }
    copyVectorToDevice(n);
    std::unordered_map<int, AscendIVFAddInfo>().swap(assignCounts); // 释放host侧占用的内存
    APP_LOG_INFO("AscendIndexIVFFlatImpl addPaged operation finished.\n");
}

void AscendIndexIVFFlatImpl::indexIVFFlatAdd(IndexParam<float, float, ascend_idx_t> &param)
{
    auto pIndex = getActualIndex(param.deviceId);
    using namespace ::ascend;
    const float *codes = param.query;
    const ascend_idx_t *ids = param.label;
    auto ret = pIndex->addVectors(param.listId, param.n, codes,
                                  static_cast<const ::ascend::Index::idx_t *>(ids));
    FAISS_THROW_IF_NOT_FMT(ret == APP_ERR_OK, "failed to add to ivf flat, ret: %d", ret);
}

std::shared_ptr<::ascend::Index> AscendIndexIVFFlatImpl::createIndex(int deviceId)
{
    APP_LOG_INFO("AscendIndexIVFFlat  createIndex operation started, device id: %d\n", deviceId);
    auto res = aclrtSetDevice(deviceId);
    FAISS_THROW_IF_NOT_FMT(res == 0, "acl set device failed %d, deviceId: %d", res, deviceId);
    std::shared_ptr<::ascend::IndexIVF> index =
        std::make_shared<::ascend::IndexIVFFlat>(nlist, this->intf_->d, nprobe, indexConfig.resourceSize);
    auto ret = index->init();
    FAISS_THROW_IF_NOT_FMT(ret == ::ascend::APP_ERR_OK, "Failed to create index ivf flat, result is %d", ret);

    APP_LOG_INFO("AscendIndexIVFFlat createIndex operation finished.\n");
    return index;
}

void AscendIndexIVFFlatImpl::updateCoarseCenter(std::vector<float> &centerData)
{
    int deviceCnt = static_cast<int>(indexConfig.deviceList.size());
    for (int i = 0; i < deviceCnt; i++) {
        int deviceId = indexConfig.deviceList[i];
        auto pIndex = getActualIndex(deviceId);
        auto ret = aclrtMemcpy(pIndex->centroidsOnDevice->data(), pIndex->centroidsOnDevice->size() * sizeof(float),
                               centerData.data(), intf_->d * nlist * sizeof(float),
                               ACL_MEMCPY_HOST_TO_DEVICE);
        FAISS_THROW_IF_NOT_FMT(ret == ACL_SUCCESS, "aclrtMemcpy error %d", ret);
    }
}

void AscendIndexIVFFlatImpl::train(idx_t n, const float *x)
{
    APP_LOG_INFO("AscendIndexIVFFlat start to train with %ld vector(s).\n", n);
    FAISS_THROW_IF_NOT_MSG(x, "x can not be nullptr.");
    FAISS_THROW_IF_NOT_FMT((n > 0) && (n < MAX_N), "n must be > 0 and < %ld", MAX_N);

    if (this->intf_->is_trained) {
        FAISS_THROW_IF_NOT_MSG(pQuantizerImpl->cpuQuantizer->is_trained, "cpuQuantizer must be trained");
        FAISS_THROW_IF_NOT_MSG(pQuantizerImpl->cpuQuantizer->ntotal == nlist, "cpuQuantizer.size must be nlist");
        FAISS_THROW_IF_NOT_MSG(indexes.size() > 0, "indexes.size must be >0");
        return;
    }
    FAISS_THROW_IF_NOT_MSG(!ivfConfig.useKmeansPP, "AscendIndexIVFFlat is not support NPU clustering now");
    if (this->intf_->metric_type == MetricType::METRIC_INNER_PRODUCT) {
        APP_LOG_INFO("METRIC_INNER_PRODUCT must set spherical to true in cpu train case\n");
        this->ivfConfig.cp.spherical = true;
    }
    Clustering clus(this->intf_->d, nlist, this->ivfConfig.cp);
    clus.verbose = this->intf_->verbose;
    FAISS_THROW_IF_NOT_MSG(pQuantizerImpl->cpuQuantizer, "cpuQuantizer is not init.");
    clus.train(n, x, *(pQuantizerImpl->cpuQuantizer));
    updateCoarseCenter(clus.centroids);
    this->intf_->is_trained = true;
    APP_LOG_INFO("AscendIndexIVFFlat train operation finished.\n");
}

void AscendIndexIVFFlatImpl::indexSearch(IndexParam<float, float, ascend_idx_t> &param) const
{
    auto pIndex = getActualIndex(param.deviceId);
    auto ret = pIndex->searchImpl(param.n, param.query, param.k, param.distance, param.label);
    FAISS_THROW_IF_NOT_FMT(ret == ::ascend::APP_ERR_OK,
                           "Failed to search index,deviceId is %d, result is: %d\n", param.deviceId, ret);
}

void AscendIndexIVFFlatImpl::searchImpl(int n, const float* x, int k, float* distances, idx_t* labels) const
{
    APP_LOG_INFO("AscendIndex searchImpl operation started: n=%d, k=%d.\n", n, k);
    size_t deviceCnt = indexConfig.deviceList.size();
    std::vector<std::vector<float>> dist(deviceCnt, std::vector<float>(n * k, 0));
    std::vector<std::vector<ascend_idx_t>> label(deviceCnt, std::vector<ascend_idx_t>(n * k, 0));

    auto searchFunctor = [&](int idx) {
        int deviceId = indexConfig.deviceList[idx];
        IndexParam<float, float, ascend_idx_t> param(deviceId, n, this->intf_->d, k);
        param.query = x;
        param.distance = dist[idx].data();
        param.label = label[idx].data();
        indexSearch(param);
    };
    CALL_PARALLEL_FUNCTOR(deviceCnt, pool, searchFunctor);
    searchPostProcess(deviceCnt, dist, label, n, k, distances, labels);
}

size_t AscendIndexIVFFlatImpl::getAddElementSize() const
{
    return static_cast<size_t>(intf_->d) * sizeof(float);
}
} // ascend
} // faiss
