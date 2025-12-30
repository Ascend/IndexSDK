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


#include "ascenddaemon/impl/IndexInt8.h"

namespace ascend {
IndexInt8::IndexInt8(idx_t d, MetricType metric, int64_t resourceSize)
    : dims(d),
      ntotal(0),
      metricType(metric),
      isTrained(false),
      maskData(nullptr),
      maskSearchedOffset(0)
{
    // resourceSize < 0 means use default mem configure
    if (resourceSize == 0) {
        resources.noTempMemory();
    } else if (resourceSize > 0) {
        resources.setTempMemory(resourceSize);
    }

    resources.initialize();
}

IndexInt8::~IndexInt8() {}

APP_ERROR IndexInt8::init()
{
    return APP_ERR_OK;
}

void IndexInt8::train(idx_t n, const int8_t *x)
{
    VALUE_UNUSED(n);
    VALUE_UNUSED(x);
}

APP_ERROR IndexInt8::addVectors(AscendTensor<int8_t, DIMS_2> &rawData)
{
    VALUE_UNUSED(rawData);
    
    return APP_ERR_NOT_IMPLEMENT;
}

size_t IndexInt8::removeIds(const IDSelector &sel)
{
    return removeIdsImpl(sel);
}

APP_ERROR IndexInt8::search(idx_t n, const int8_t *x, idx_t k, float16_t *distances, idx_t *labels, uint8_t *mask)
{
    APPERR_RETURN_IF_NOT_LOG(x, APP_ERR_INVALID_PARAM, "x can not be nullptr.");
    APPERR_RETURN_IF_NOT_LOG(distances, APP_ERR_INVALID_PARAM, "distance can not be nullptr.");
    APPERR_RETURN_IF_NOT_LOG(labels, APP_ERR_INVALID_PARAM, "labels can not be nullptr.");
    APPERR_RETURN_IF_NOT_LOG(this->isTrained, APP_ERR_INVALID_PARAM, "Index not trained.");
    APPERR_RETURN_IF_NOT_FMT(n <= (Index::idx_t)std::numeric_limits<int>::max(),
        APP_ERR_INVALID_PARAM, "indices exceeds max(%d)", std::numeric_limits<int>::max());
    APPERR_RETURN_IF(n == 0 || k == 0, APP_ERR_OK);

    this->maskData = mask;
    this->maskSearchedOffset = 0;

    return searchBatched(n, x, k, distances, labels);
}

APP_ERROR IndexInt8::search(std::vector<IndexInt8*> indexes, idx_t n, const int8_t *x, idx_t k,
    float16_t *distances, idx_t *labels)
{
    APPERR_RETURN_IF_NOT_LOG(x, APP_ERR_INVALID_PARAM, "x can not be nullptr.");
    APPERR_RETURN_IF_NOT_LOG(distances, APP_ERR_INVALID_PARAM, "distance can not be nullptr.");
    APPERR_RETURN_IF_NOT_LOG(labels, APP_ERR_INVALID_PARAM, "labels can not be nullptr.");
    APPERR_RETURN_IF_NOT_LOG(this->isTrained, APP_ERR_INVALID_PARAM, "Index not trained.");
    APPERR_RETURN_IF_NOT_FMT(n <= (Index::idx_t)std::numeric_limits<int>::max(),
        APP_ERR_INVALID_PARAM, "indices exceeds max(%d)", std::numeric_limits<int>::max());
    APPERR_RETURN_IF(n == 0 || k == 0, APP_ERR_OK);

    return searchBatched(indexes, n, x, k, distances, labels);
}

APP_ERROR IndexInt8::searchBatched(std::vector<IndexInt8 *> indexes, int64_t n, const int8_t *x, int64_t k,
    float16_t *distances, idx_t *labels)
{
    APP_ERROR ret = APP_ERR_OK;

    // 1. init result
    initSearchResult(indexes.size(), n, k, distances, labels);

    // 2. search by page
    size_t size = searchBatchSizes.size();
    if (size == 0 || n <= 0) {
        return APP_ERR_INVALID_PARAM;
    }

    int64_t searched = 0;
    for (size_t i = 0; i < size; i++) {
        int64_t batchSize = searchBatchSizes[i];
        if ((n - searched) >= batchSize) {
            int64_t page = (n - searched) / batchSize;
            for (int64_t j = 0; j < page; j++) {
                ret = searchImpl(indexes, n, batchSize, x + searched * this->dims, k,
                    distances + searched * k, labels + searched * k);
                APPERR_RETURN_IF(ret, ret);
                searched += batchSize;
            }
        }
    }

    for (int64_t i = searched; i < n; i++) {
        ret = searchImpl(indexes, n, 1, x + i * this->dims, k, distances + i * k, labels + i * k);
        APPERR_RETURN_IF(ret, ret);
    }

    reorder(indexes.size(), n, k, distances, labels);
    return APP_ERR_OK;
}

void IndexInt8::initSearchResult(int indexesSize, int n, int k, float16_t *distances, idx_t *labels)
{
    APP_LOG_WARNING("the method of searchImpl for initSearchResult is not implement.\n");
    VALUE_UNUSED(indexesSize);
    VALUE_UNUSED(n);
    VALUE_UNUSED(k);
    VALUE_UNUSED(distances);
    VALUE_UNUSED(labels);
}

APP_ERROR IndexInt8::searchImpl(std::vector<IndexInt8 *> indexes, int n, int batchSize, const int8_t *x, int k,
    float16_t *distances, idx_t *labels)
{
    APP_LOG_WARNING("the method of searchImpl for multi index int8 is not implement.\n");

    VALUE_UNUSED(indexes);
    VALUE_UNUSED(n);
    VALUE_UNUSED(batchSize);
    VALUE_UNUSED(x);
    VALUE_UNUSED(k);
    VALUE_UNUSED(distances);
    VALUE_UNUSED(labels);

    return APP_ERR_NOT_IMPLEMENT;
}

void IndexInt8::reorder(int indexNum, int n, int k, float16_t *distances, idx_t *labels)
{
    VALUE_UNUSED(indexNum);
    VALUE_UNUSED(n);
    VALUE_UNUSED(k);
    VALUE_UNUSED(distances);
    VALUE_UNUSED(labels);
}

APP_ERROR IndexInt8::searchBatched(int64_t n, const int8_t *x, int64_t k, float16_t *distance, idx_t *labels)
{
    APP_ERROR ret = APP_ERR_OK;
    int64_t idxMaskLen = static_cast<int64_t>(utils::divUp(this->ntotal, 8));
    size_t size = searchBatchSizes.size();
    if (n > 1 && size > 0) {
        int64_t searched = 0;
        for (size_t i = 0; i < size; i++) {
            int64_t batchSize = searchBatchSizes[i];
            if ((n - searched) < batchSize) {
                continue;
            }

            int64_t page = (n - searched) / batchSize;
            for (int64_t j = 0; j < page; j++) {
                ret = searchImpl(batchSize, x + searched * this->dims, k, distance + searched * k,
                    labels + searched * k);
                APPERR_RETURN_IF(ret, ret);
                searched += batchSize;

                size_t maskDataLen = static_cast<size_t>(idxMaskLen) * static_cast<size_t>(batchSize);
                this->maskSearchedOffset += maskDataLen;
            }
        }

        for (int64_t i = searched; i < n; i++) {
            ret = searchImpl(1, x + i * this->dims, k, distance + i * k, labels + i * k);
            this->maskSearchedOffset += static_cast<size_t>(idxMaskLen);
            APPERR_RETURN_IF(ret, ret);
        }
        return APP_ERR_OK;
    } else {
        return searchImpl(n, x, k, distance, labels);
    }
}

void IndexInt8::reserveMemory(size_t numVecs)
{
    VALUE_UNUSED(numVecs);
    ASCEND_THROW_MSG("reserveMemory not implemented for this type of index");
}

size_t IndexInt8::reclaimMemory()
{
    ASCEND_THROW_MSG("reclaimMemory not implemented for this type of index");
    return 0;
}
} // namespace ascend
