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

#include "ascendhost/include/index/AscendIndexILFlat.h"

#include "ascenddaemon/utils/AscendUtils.h"
#include "ascendhost/include/impl/AscendIndexILFlatImpl.h"
#include "common/ErrorCode.h"

using namespace ascend;

namespace faiss {
namespace ascend {

AscendIndexILFlat::AscendIndexILFlat()
{
    this->pIndexILFlatImpl = nullptr;
    this->capacity = 0;
}

AscendIndexILFlat::~AscendIndexILFlat()
{
    this->Finalize();
}

APP_ERROR AscendIndexILFlat::Init(
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
    APPERR_RETURN_IF_NOT_LOG(pIndexILFlatImpl == nullptr,
        APP_ERR_ILLEGAL_OPERATION,
        "Index is already initialized, mutiple initialization is not allowed. ");
    APPERR_RETURN_IF_NOT_FMT(
        deviceList.size() == 1, APP_ERR_INVALID_PARAM, "the number of deviceList(%zu) is not 1.", deviceList.size());

    this->capacity = capacity;
    this->pIndexILFlatImpl = new (std::nothrow) AscendIndexILFlatImpl(dim, capacity, deviceList[0],
        (resourceSize == -1 ? MIN_RESOURCE : resourceSize));
    APPERR_RETURN_IF_NOT_LOG(
        this->pIndexILFlatImpl, APP_ERR_REQUEST_ERROR, "Inner error, failed to create pIndexILFlatImpl. ");

    return this->pIndexILFlatImpl->Init();
}

void AscendIndexILFlat::Finalize()
{
    auto lock = ::ascend::AscendMultiThreadManager::GetWriteLock(mtx);
    if (this->pIndexILFlatImpl != nullptr) {
        this->pIndexILFlatImpl->Finalize();
        delete this->pIndexILFlatImpl;
        this->pIndexILFlatImpl = nullptr;
    }
}

APP_ERROR AscendIndexILFlat::AddFeatures(int n, const float16_t *features)
{
    auto lock = ::ascend::AscendMultiThreadManager::GetWriteLock(mtx);
    APPERR_RETURN_IF_NOT_LOG(
        pIndexILFlatImpl != nullptr, APP_ERR_ILLEGAL_OPERATION, "The index must be initialized first. ");

    return this->pIndexILFlatImpl->Add(n, features);
}

APP_ERROR AscendIndexILFlat::AddFeatures(int n, const float *features)
{
    auto lock = ::ascend::AscendMultiThreadManager::GetWriteLock(mtx);
    APPERR_RETURN_IF_NOT_LOG(
        pIndexILFlatImpl != nullptr, APP_ERR_ILLEGAL_OPERATION, "The index must be initialized first. ");

    return this->pIndexILFlatImpl->Add(n, features);
}

APP_ERROR AscendIndexILFlat::UpdateFeatures(int n, const float16_t *features, const idx_t *indices)
{
    auto lock = ::ascend::AscendMultiThreadManager::GetWriteLock(mtx);
    APPERR_RETURN_IF_NOT_LOG(
        pIndexILFlatImpl != nullptr, APP_ERR_ILLEGAL_OPERATION, "The index must be initialized first. ");

    return this->pIndexILFlatImpl->Update(n, features, indices);
}

APP_ERROR AscendIndexILFlat::UpdateFeatures(int n, const float *features, const idx_t *indices)
{
    auto lock = ::ascend::AscendMultiThreadManager::GetWriteLock(mtx);
    APPERR_RETURN_IF_NOT_LOG(
        pIndexILFlatImpl != nullptr, APP_ERR_ILLEGAL_OPERATION, "The index must be initialized first. ");

    return this->pIndexILFlatImpl->Update(n, features, indices);
}

APP_ERROR AscendIndexILFlat::SearchByThreshold(int n, const float16_t *queries, float threshold, int topk, int *num,
    idx_t *indices, float *distances, unsigned int tableLen, const float *table)
{
    auto lock = ::ascend::AscendMultiThreadManager::GetWriteLock(mtx);

    APPERR_RETURN_IF_NOT_LOG(
        pIndexILFlatImpl != nullptr, APP_ERR_ILLEGAL_OPERATION, "The index must be initialized first. ");
    return this->pIndexILFlatImpl->SearchByThreshold(n, queries, threshold, topk, num,
        indices, distances, tableLen, table);
}

APP_ERROR AscendIndexILFlat::SearchByThreshold(int n, const float *queries, float threshold, int topk, int *num,
    idx_t *indices, float *distances, unsigned int tableLen, const float *table)
{
    auto lock = ::ascend::AscendMultiThreadManager::GetWriteLock(mtx);

    APPERR_RETURN_IF_NOT_LOG(
        pIndexILFlatImpl != nullptr, APP_ERR_ILLEGAL_OPERATION, "The index must be initialized first. ");
    return this->pIndexILFlatImpl->SearchByThreshold(n, queries, threshold, topk, num,
        indices, distances, tableLen, table);
}

APP_ERROR AscendIndexILFlat::Search(int n, const float16_t *queries, int topk, idx_t *indices, float *distances,
    unsigned int tableLen, const float *table)
{
    auto lock = ::ascend::AscendMultiThreadManager::GetWriteLock(mtx);

    APPERR_RETURN_IF_NOT_LOG(
        pIndexILFlatImpl != nullptr, APP_ERR_ILLEGAL_OPERATION, "The index must be initialized first. ");
    return this->pIndexILFlatImpl->Search(n, queries, topk, indices, distances, tableLen, table);
}

APP_ERROR AscendIndexILFlat::Search(int n, const float *queries, int topk, idx_t *indices, float *distances,
    unsigned int tableLen, const float *table)
{
    auto lock = ::ascend::AscendMultiThreadManager::GetWriteLock(mtx);

    APPERR_RETURN_IF_NOT_LOG(
        pIndexILFlatImpl != nullptr, APP_ERR_ILLEGAL_OPERATION, "The index must be initialized first. ");
    return this->pIndexILFlatImpl->Search(n, queries, topk, indices, distances, tableLen, table);
}

APP_ERROR AscendIndexILFlat::ComputeDistance(int n, const float16_t *queries, float *distances,
    unsigned int tableLen, const float *table)
{
    auto lock = ::ascend::AscendMultiThreadManager::GetWriteLock(mtx);
    APPERR_RETURN_IF_NOT_LOG(
        pIndexILFlatImpl != nullptr, APP_ERR_ILLEGAL_OPERATION, "The index must be initialized first. ");
    return this->pIndexILFlatImpl->ComputeDistance(n, queries, distances, tableLen, table);
}

APP_ERROR AscendIndexILFlat::ComputeDistance(int n, const float *queries, float *distances,
    unsigned int tableLen, const float *table)
{
    auto lock = ::ascend::AscendMultiThreadManager::GetWriteLock(mtx);
    APPERR_RETURN_IF_NOT_LOG(
        pIndexILFlatImpl != nullptr, APP_ERR_ILLEGAL_OPERATION, "The index must be initialized first. ");
    return this->pIndexILFlatImpl->ComputeDistance(n, queries, distances, tableLen, table);
}

APP_ERROR AscendIndexILFlat::ComputeDistanceByIdx(int n, const float16_t *queries, const int *num,
    const idx_t *indices, float *distances, MEMORY_TYPE memoryType, unsigned int tableLen, const float *table)
{
    auto lock = ::ascend::AscendMultiThreadManager::GetWriteLock(mtx);
    APPERR_RETURN_IF_NOT_LOG(
        pIndexILFlatImpl != nullptr, APP_ERR_ILLEGAL_OPERATION, "The index must be initialized first. ");
    return this->pIndexILFlatImpl->ComputeDistanceByIdx(n, queries, num, indices, distances,
        memoryType, tableLen, table);
}

APP_ERROR AscendIndexILFlat::ComputeDistanceByIdx(int n, const float *queries, const int *num,
    const idx_t *indices, float *distances, MEMORY_TYPE memoryType, unsigned int tableLen, const float *table)
{
    auto lock = ::ascend::AscendMultiThreadManager::GetWriteLock(mtx);
    APPERR_RETURN_IF_NOT_LOG(
        pIndexILFlatImpl != nullptr, APP_ERR_ILLEGAL_OPERATION, "The index must be initialized first. ");
    return this->pIndexILFlatImpl->ComputeDistanceByIdx(n, queries, num, indices, distances,
        memoryType, tableLen, table);
}

APP_ERROR AscendIndexILFlat::RemoveFeatures(int n, const idx_t *indices)
{
    auto lock = ::ascend::AscendMultiThreadManager::GetWriteLock(mtx);
    APPERR_RETURN_IF_NOT_FMT(n >= 0 && n <= this->capacity, APP_ERR_INVALID_PARAM,
        "The number n should be in range [0, %d]", this->capacity);
    APPERR_RETURN_IF_NOT_LOG(indices, APP_ERR_INVALID_PARAM, "indices can not be nullptr.\n");
    APPERR_RETURN_IF_NOT_LOG(
        this->pIndexILFlatImpl != nullptr, APP_ERR_INDEX_NOT_INIT, "index must be initialized first!!!\n");
    return this->pIndexILFlatImpl->Remove(n, indices);
}

APP_ERROR AscendIndexILFlat::GetFeatures(int n, float16_t *features, const idx_t *indices) const
{
    auto lock = ::ascend::AscendMultiThreadManager::GetWriteLock(mtx);

    APPERR_RETURN_IF_NOT_LOG(
        this->pIndexILFlatImpl != nullptr, APP_ERR_INDEX_NOT_INIT, "index must be initialized first!!!\n");
    return this->pIndexILFlatImpl->Get(n, features, indices);
}

APP_ERROR AscendIndexILFlat::GetFeatures(int n, float *features, const idx_t *indices) const
{
    auto lock = ::ascend::AscendMultiThreadManager::GetWriteLock(mtx);

    APPERR_RETURN_IF_NOT_LOG(
        this->pIndexILFlatImpl != nullptr, APP_ERR_INDEX_NOT_INIT, "index must be initialized first!!!\n");
    return this->pIndexILFlatImpl->Get(n, features, indices);
}

APP_ERROR AscendIndexILFlat::GetFeaturesOnDevice(int n, float16_t *features, const idx_t *indices) const
{
    auto lock = ::ascend::AscendMultiThreadManager::GetWriteLock(mtx);

    APPERR_RETURN_IF_NOT_LOG(
        this->pIndexILFlatImpl != nullptr, APP_ERR_INDEX_NOT_INIT, "index must be initialized first!!!\n");
    return this->pIndexILFlatImpl->GetDevice(n, features, indices);
}

APP_ERROR AscendIndexILFlat::GetFeaturesOnDevice(int n, float *features, const idx_t *indices) const
{
    auto lock = ::ascend::AscendMultiThreadManager::GetWriteLock(mtx);

    APPERR_RETURN_IF_NOT_LOG(
        this->pIndexILFlatImpl != nullptr, APP_ERR_INDEX_NOT_INIT, "index must be initialized first!!!\n");
    return this->pIndexILFlatImpl->GetDevice(n, features, indices);
}

APP_ERROR AscendIndexILFlat::SetNTotal(int n)
{
    auto lock = ::ascend::AscendMultiThreadManager::GetWriteLock(mtx);
    APPERR_RETURN_IF_NOT_FMT(n >= 0 && n <= this->capacity, APP_ERR_INVALID_PARAM,
        "The ntotal should be in range [0, %d].\n", this->capacity);
    APPERR_RETURN_IF_NOT_LOG(
        this->pIndexILFlatImpl != nullptr, APP_ERR_INDEX_NOT_INIT, "index must be initialized first!!!\n");
    this->pIndexILFlatImpl->SetNTotal(n);
    return APP_ERR_OK;
}

int AscendIndexILFlat::GetNTotal() const
{
    auto lock = ::ascend::AscendMultiThreadManager::GetWriteLock(mtx);
    APPERR_RETURN_IF_NOT_LOG(
        this->pIndexILFlatImpl != nullptr, APP_ERR_INDEX_NOT_INIT, "index must be initialized first!!!\n");
    return this->pIndexILFlatImpl->GetNTotal();
}

} /* namespace ascend */
}