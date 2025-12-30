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

#include "ascendhost/include/index/AscendIndexCluster.h"

#include "ascenddaemon/utils/AscendUtils.h"
#include "ascendhost/include/impl/AscendIndexClusterImpl.h"
#include "common/ErrorCode.h"

using namespace ascend;

namespace faiss {
namespace ascend {

AscendIndexCluster::AscendIndexCluster()
{
    this->pIndexClusterImpl = nullptr;
    this->capacity = 0;
}

APP_ERROR AscendIndexCluster::Init(
    int dim, int capacity, faiss::MetricType metricType, const std::vector<int> &deviceList, int64_t resourceSize)
{
    auto lock = ::ascend::AscendMultiThreadManager::GetWriteLock(mtx);
    APPERR_RETURN_IF_NOT_FMT(capacity > 0 && capacity <= MAX_CAP,
        APP_ERR_INVALID_PARAM,
        "Given capacity should be in value range: (0, %d]. ",
        MAX_CAP);
    APPERR_RETURN_IF_NOT_FMT(
        static_cast<size_t>(dim) * sizeof(float16_t) * static_cast<size_t>(capacity) <= MAX_BASE_SPACE,
        APP_ERR_INVALID_PARAM,
        "The capacity(%d) exceed memory allocation limit, please refer to the manuals to set correct capacity. ",
        capacity);
    APPERR_RETURN_IF_NOT_FMT(metricType == MetricType::METRIC_INNER_PRODUCT,
        APP_ERR_INVALID_PARAM,
        "Unsupported metric type(%d). ",
        metricType);

    APPERR_RETURN_IF_NOT_FMT(resourceSize == -1 || (resourceSize >= MIN_RESOURCE && resourceSize <= MAX_RESOURCE),
        APP_ERR_INVALID_PARAM,
        "resourceSize(%ld) should be -1 or in range [%d Byte, %ld Byte]!",
        resourceSize,
        MIN_RESOURCE,
        MAX_RESOURCE);
    APPERR_RETURN_IF_NOT_LOG(pIndexClusterImpl == nullptr,
        APP_ERR_ILLEGAL_OPERATION,
        "Index is already initialized, mutiple initialization is not allowed. ");
    APPERR_RETURN_IF_NOT_FMT(
        deviceList.size() == 1, APP_ERR_INVALID_PARAM, "the number of deviceList(%zu) is not 1.", deviceList.size());

    this->capacity = capacity;
    this->pIndexClusterImpl = new (std::nothrow) AscendIndexClusterImpl(dim, capacity, deviceList[0],
        (resourceSize == -1 ? MIN_RESOURCE : resourceSize));
    APPERR_RETURN_IF_NOT_LOG(
        this->pIndexClusterImpl, APP_ERR_REQUEST_ERROR, "Inner error, failed to create pIndexClusterImpl. ");

    return this->pIndexClusterImpl->Init();
}

void AscendIndexCluster::Finalize()
{
    auto lock = ::ascend::AscendMultiThreadManager::GetWriteLock(mtx);
    if (this->pIndexClusterImpl != nullptr) {
        this->pIndexClusterImpl->Finalize();
        delete this->pIndexClusterImpl;
        this->pIndexClusterImpl = nullptr;
    }
}

APP_ERROR AscendIndexCluster::AddFeatures(int n, const uint16_t *features, const int64_t *indices)
{
    auto lock = ::ascend::AscendMultiThreadManager::GetWriteLock(mtx);
    APPERR_RETURN_IF_NOT_LOG(
        pIndexClusterImpl != nullptr, APP_ERR_ILLEGAL_OPERATION, "The index must be initialized first. ");
    APPERR_RETURN_IF_NOT_FMT((n) > 0 && (n) <= this->capacity,
        APP_ERR_INVALID_PARAM,
        "The number n should be in range (0, %d]",
        this->capacity);
    APPERR_RETURN_IF_NOT_LOG(features, APP_ERR_INVALID_PARAM, "Features can not be nullptr.");
    APPERR_RETURN_IF_NOT_LOG(indices, APP_ERR_INVALID_PARAM, "Indices can not be nullptr.");
    for (int64_t i = 0; i < static_cast<int64_t>(n); i++) {
        APPERR_RETURN_IF_NOT_FMT(*(indices + i) < this->capacity,
            APP_ERR_INVALID_PARAM, "The indices[%ld](%ld) should be in range [0, %d)",
            i, *(indices + i), this->capacity);
    }
    return this->pIndexClusterImpl->Add(n, features, indices);
}

APP_ERROR AscendIndexCluster::AddFeatures(int n, const float *features, const uint32_t *indices)
{
    auto lock = ::ascend::AscendMultiThreadManager::GetWriteLock(mtx);
    APPERR_RETURN_IF_NOT_LOG(
        pIndexClusterImpl != nullptr, APP_ERR_ILLEGAL_OPERATION, "The index must be initialized first. ");
    APPERR_RETURN_IF_NOT_FMT((n) > 0 && (n) <= this->capacity,
        APP_ERR_INVALID_PARAM,
        "The number n should be in range (0, %d]",
        this->capacity);
    APPERR_RETURN_IF_NOT_LOG(features, APP_ERR_INVALID_PARAM, "Features can not be nullptr.");
    APPERR_RETURN_IF_NOT_LOG(indices, APP_ERR_INVALID_PARAM, "Indices can not be nullptr.");
    for (uint32_t i = 0; i < static_cast<uint32_t>(n); i++) {
        APPERR_RETURN_IF_NOT_FMT(*(indices + i) < static_cast<uint32_t>(this->capacity),
            APP_ERR_INVALID_PARAM, "The indices[%u](%u) should be in range [0, %d)", i, *(indices + i), this->capacity);
    }
    return this->pIndexClusterImpl->Add(n, features, indices);
}

APP_ERROR AscendIndexCluster::ComputeDistanceByThreshold(const std::vector<uint32_t> &queryIdxArr,
    uint32_t codeStartIdx, uint32_t codeNum, float threshold, bool aboveFilter,
    std::vector<std::vector<float>> &resDistArr, std::vector<std::vector<uint32_t>> &resIdxArr)
{
    auto lock = ::ascend::AscendMultiThreadManager::GetWriteLock(mtx);
    APPERR_RETURN_IF_NOT_LOG(
        pIndexClusterImpl != nullptr, APP_ERR_ILLEGAL_OPERATION, "The index must be initialized first. ");
    return this->pIndexClusterImpl->ComputeDistanceByThreshold(
        queryIdxArr, codeStartIdx, codeNum, threshold, aboveFilter, resDistArr, resIdxArr);
}

APP_ERROR AscendIndexCluster::SearchByThreshold(int n, const uint16_t *queries, float threshold, int topk, int *num,
    int64_t *indices, float *distances, unsigned int tableLen, const float *table)
{
    auto lock = ::ascend::AscendMultiThreadManager::GetWriteLock(mtx);

    APPERR_RETURN_IF_NOT_LOG(
        pIndexClusterImpl != nullptr, APP_ERR_ILLEGAL_OPERATION, "The index must be initialized first. ");
    return this->pIndexClusterImpl->SearchByThreshold(n, queries, threshold, topk, num,
        indices, distances, tableLen, table);
}

APP_ERROR AscendIndexCluster::ComputeDistanceByIdx(int n, const uint16_t *queries, const int *num,
    const uint32_t *indices, float *distances, unsigned int tableLen, const float *table)
{
    auto lock = ::ascend::AscendMultiThreadManager::GetWriteLock(mtx);
    APPERR_RETURN_IF_NOT_LOG(
        pIndexClusterImpl != nullptr, APP_ERR_ILLEGAL_OPERATION, "The index must be initialized first. ");
    return this->pIndexClusterImpl->ComputeDistanceByIdx(n, queries, num, indices, distances, tableLen, table);
}

APP_ERROR AscendIndexCluster::GetFeatures(int n, uint16_t *features, const int64_t *indices) const
{
    auto lock = ::ascend::AscendMultiThreadManager::GetWriteLock(mtx);
    APPERR_RETURN_IF_NOT_FMT(n >= 0 && n <= this->capacity, APP_ERR_INVALID_PARAM,
        "The number n should be in range [0, %d]", this->capacity);
    APPERR_RETURN_IF_NOT_LOG(features, APP_ERR_INVALID_PARAM, "Features can not be nullptr.");
    APPERR_RETURN_IF_NOT_LOG(indices, APP_ERR_INVALID_PARAM, "Indices can not be nullptr.");
    APPERR_RETURN_IF_NOT_LOG(
        this->pIndexClusterImpl != nullptr, APP_ERR_INDEX_NOT_INIT, "index must be initialized first!!!\n");
    return this->pIndexClusterImpl->Get(n, features, indices);
}

APP_ERROR AscendIndexCluster::RemoveFeatures(int n, const int64_t *indices)
{
    auto lock = ::ascend::AscendMultiThreadManager::GetWriteLock(mtx);
    APPERR_RETURN_IF_NOT_FMT(n >= 0 && n <= this->capacity, APP_ERR_INVALID_PARAM,
        "The number n should be in range [0, %d]", this->capacity);
    APPERR_RETURN_IF_NOT_LOG(indices, APP_ERR_INVALID_PARAM, "indices can not be nullptr.\n");
    APPERR_RETURN_IF_NOT_LOG(
        this->pIndexClusterImpl != nullptr, APP_ERR_INDEX_NOT_INIT, "index must be initialized first!!!\n");
    return this->pIndexClusterImpl->Remove(n, indices);
}

APP_ERROR AscendIndexCluster::SetNTotal(int n)
{
    auto lock = ::ascend::AscendMultiThreadManager::GetWriteLock(mtx);
    APPERR_RETURN_IF_NOT_FMT(n >= 0 && n <= this->capacity, APP_ERR_INVALID_PARAM,
        "The ntotal should be in range [0, %d].\n", this->capacity);
    APPERR_RETURN_IF_NOT_LOG(
        this->pIndexClusterImpl != nullptr, APP_ERR_INDEX_NOT_INIT, "index must be initialized first!!!\n");
    this->pIndexClusterImpl->SetNTotal(n);
    return APP_ERR_OK;
}

int AscendIndexCluster::GetNTotal() const
{
    auto lock = ::ascend::AscendMultiThreadManager::GetWriteLock(mtx);
    APPERR_RETURN_IF_NOT_LOG(
        this->pIndexClusterImpl != nullptr, APP_ERR_INDEX_NOT_INIT, "index must be initialized first!!!\n");
    return this->pIndexClusterImpl->GetNTotal();
}

} /* namespace ascend */
}