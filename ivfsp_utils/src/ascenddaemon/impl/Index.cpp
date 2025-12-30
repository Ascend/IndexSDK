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


#include <ascenddaemon/impl/Index.h>

#include <set>
#include <algorithm>

#include <ascenddaemon/impl/AuxIndexStructures.h>
#include <ascenddaemon/utils/Limits.h>
#include <common/utils/CommonUtils.h>
#include <common/utils/LogUtils.h>

namespace ascendSearch {
Index::Index(ascendSearch::idx_t d, int64_t resourceSize)
    : dims(d), ntotal(0), isTrained(false)
{
    // resourceSize < 0 means use default mem configure
    if (resourceSize == 0) {
        resources.noTempMemory();
    } else if (resourceSize > 0) {
        resources.setTempMemory(resourceSize);
    }

    resources.initialize();
}

Index::~Index() {}

APP_ERROR Index::init()
{
    return APP_ERR_OK;
}

void Index::train(idx_t n, const float16_t *x)
{
    VALUE_UNUSED(n);
    VALUE_UNUSED(x);
}

size_t Index::removeIds(const IDSelector &sel)
{
    size_t removeCnt = 0;

    try {
        const ascendSearch::IDSelectorBatch &batch = dynamic_cast<const ascendSearch::IDSelectorBatch &>(sel);
        removeCnt = removeIdsBatch(batch);
    } catch (std::bad_cast &e) {
        // ignore
    }

    try {
        const ascendSearch::IDSelectorRange &range = dynamic_cast<const ascendSearch::IDSelectorRange &>(sel);
        removeCnt = removeIdsRange(range);
    } catch (std::bad_cast &e) {
        // ignore
    }

    return removeCnt;
}

size_t Index::removeIdsBatch(const IDSelectorBatch &sel)
{
    std::vector<idx_t> indices(sel.set.begin(), sel.set.end());
    // 1. filter the same id
    std::set<idx_t> filtered;
    for (auto idx : indices) {
        if (idx < ntotal) {
            filtered.insert(idx);
        }
    }

    // 2. sort by DESC, then delete from the big to small
    std::vector<idx_t> sortData(filtered.begin(), filtered.end());
    std::sort(sortData.begin(), sortData.end(), std::greater<idx_t>());

    // 3. move the end data to the locate of delete data
    idx_t oldTotal = this->ntotal;
    for (const auto index : sortData) {
        moveVectorForward(this->ntotal - 1, index);
        --this->ntotal;
    }

    // 4. release the space of unusage
    size_t removedCnt = filtered.size();
    releaseUnusageSpace(oldTotal, removedCnt);

    return removedCnt;
}

size_t Index::removeIdsRange(const IDSelectorRange &sel)
{
    idx_t min = sel.imin;
    idx_t max = sel.imax;

    // 1. check param
    if (min >= max || min >= ntotal) {
        return 0;
    }

    if (max > ntotal) {
        max = ntotal;
    }

    // 2. move the end data to the locate of delete data(delete from back to front)
    idx_t oldTotal = this->ntotal;
    for (idx_t i = 1; i <= max - min; ++i) {
        moveVectorForward(this->ntotal - 1, max - i);
        --this->ntotal;
    }

    // 3. release the space of unusage
    size_t removeCnt = max - min;
    releaseUnusageSpace(oldTotal, removeCnt);
    return removeCnt;
}

void Index::moveVectorForward(idx_t srcIdx, idx_t destIdx)
{
    APP_LOG_WARNING("the method of moveVectorForward is not implement.\n");

    VALUE_UNUSED(srcIdx);
    VALUE_UNUSED(destIdx);
}

void Index::releaseUnusageSpace(int oldTotal, int remove)
{
    APP_LOG_WARNING("the method of releaseUnusageSpace is not implement.\n");

    VALUE_UNUSED(oldTotal);
    VALUE_UNUSED(remove);
}

APP_ERROR Index::search(idx_t n, const float16_t *x, idx_t k, float16_t *distances, idx_t *labels)
{
    APPERR_RETURN_IF_NOT_LOG(x, APP_ERR_INVALID_PARAM, "x can not be nullptr.");
    APPERR_RETURN_IF_NOT_LOG(distances, APP_ERR_INVALID_PARAM, "distance can not be nullptr.");
    APPERR_RETURN_IF_NOT_LOG(labels, APP_ERR_INVALID_PARAM, "labels can not be nullptr.");
    APPERR_RETURN_IF_NOT_FMT(n <= static_cast<idx_t>(std::numeric_limits<int>::max()),
        APP_ERR_INVALID_PARAM, "indices exceeds max(%d)", std::numeric_limits<int>::max());
    APPERR_RETURN_IF(n == 0 || k == 0, APP_ERR_OK);

    return searchBatched(n, x, k, distances, labels);
}

APP_ERROR Index::search(std::vector<Index*> indexes, idx_t n, const float16_t *x, idx_t k,
    float16_t *distances, idx_t *labels)
{
    APPERR_RETURN_IF_NOT_LOG(x, APP_ERR_INVALID_PARAM, "x can not be nullptr.");
    APPERR_RETURN_IF_NOT_LOG(distances, APP_ERR_INVALID_PARAM, "distance can not be nullptr.");
    APPERR_RETURN_IF_NOT_LOG(labels, APP_ERR_INVALID_PARAM, "labels can not be nullptr.");
    APPERR_RETURN_IF_NOT_FMT(n <= static_cast<idx_t>(std::numeric_limits<int>::max()),
        APP_ERR_INVALID_PARAM, "indices exceeds max(%d)", std::numeric_limits<int>::max());
    APPERR_RETURN_IF(n == 0 || k == 0, APP_ERR_OK);
    return searchBatched(indexes, n, x, k, distances, labels);
}

APP_ERROR Index::searchFilter(idx_t n, const float16_t *x, idx_t k, float16_t *distances, idx_t *labels, uint8_t* mask)
{
    APPERR_RETURN_IF_NOT_LOG(x, APP_ERR_INVALID_PARAM, "x can not be nullptr.");
    APPERR_RETURN_IF_NOT_LOG(distances, APP_ERR_INVALID_PARAM, "distance can not be nullptr.");
    APPERR_RETURN_IF_NOT_LOG(labels, APP_ERR_INVALID_PARAM, "labels can not be nullptr.");
    APPERR_RETURN_IF_NOT_LOG(mask, APP_ERR_INVALID_PARAM, "mask can not be nullptr.");
    APPERR_RETURN_IF_NOT_LOG(this->isTrained, APP_ERR_INVALID_PARAM, "Index not trained.");
    APPERR_RETURN_IF_NOT_FMT(n <= static_cast<idx_t>(std::numeric_limits<int>::max()),
        APP_ERR_INVALID_PARAM, "indices exceeds max(%d)", std::numeric_limits<int>::max());
    APPERR_RETURN_IF(n == 0 || k == 0, APP_ERR_OK);

    // 2. call the searchBatched func
    return searchBatched(n, x, k, distances, labels, mask);
}

APP_ERROR Index::searchFilter(idx_t n, const float16_t *x, idx_t k, float16_t *distances, idx_t *labels,
    uint64_t filterSize, uint32_t* filters)
{
    APPERR_RETURN_IF_NOT_LOG(x, APP_ERR_INVALID_PARAM, "x can not be nullptr.");
    APPERR_RETURN_IF_NOT_LOG(distances, APP_ERR_INVALID_PARAM, "distance can not be nullptr.");
    APPERR_RETURN_IF_NOT_LOG(labels, APP_ERR_INVALID_PARAM, "labels can not be nullptr.");
    APPERR_RETURN_IF_NOT_LOG(filters, APP_ERR_INVALID_PARAM, "filters can not be nullptr.");
    APPERR_RETURN_IF_NOT_LOG(this->isTrained, APP_ERR_INVALID_PARAM, "Index not trained.");
    APPERR_RETURN_IF_NOT_FMT(n <= static_cast<idx_t>(std::numeric_limits<int>::max()),
        APP_ERR_INVALID_PARAM, "indices exceeds max(%d)", std::numeric_limits<int>::max());
    APPERR_RETURN_IF(n == 0 || k == 0, APP_ERR_OK);

    // 2. call the searchBatched func
    return searchBatched(n, x, k, distances, labels, filterSize, filters);
}

APP_ERROR Index::searchFilter(std::vector<Index *> indexes, idx_t n, const float16_t *x, idx_t k, float16_t *distances,
    idx_t *labels, uint32_t filterSize, uint32_t *filters)
{
    APPERR_RETURN_IF_NOT_LOG(x, APP_ERR_INVALID_PARAM, "x can not be nullptr.");
    APPERR_RETURN_IF_NOT_LOG(distances, APP_ERR_INVALID_PARAM, "distance can not be nullptr.");
    APPERR_RETURN_IF_NOT_LOG(labels, APP_ERR_INVALID_PARAM, "labels can not be nullptr.");
    APPERR_RETURN_IF_NOT_LOG(this->isTrained, APP_ERR_INVALID_PARAM, "Index not trained.");
    APPERR_RETURN_IF_NOT_FMT(n <= static_cast<idx_t>(std::numeric_limits<int>::max()),
        APP_ERR_INVALID_PARAM, "indices exceeds max(%d)", std::numeric_limits<int>::max());
    APPERR_RETURN_IF(k == 0 || n == 0, APP_ERR_OK);
 
    // 2. call the searchBatched func
    return searchBatched(indexes, n, x, k, distances, labels, filterSize, filters);
}
APP_ERROR Index::searchFilter(std::vector<Index *> indexes, idx_t n, const float16_t *x, idx_t k, float16_t *distances,
    idx_t *labels, uint32_t filterSize, uint32_t **filters)
{
    APPERR_RETURN_IF_NOT_LOG(x, APP_ERR_INVALID_PARAM, "x can not be nullptr.");
    APPERR_RETURN_IF_NOT_LOG(distances, APP_ERR_INVALID_PARAM, "distance can not be nullptr.");
    APPERR_RETURN_IF_NOT_LOG(labels, APP_ERR_INVALID_PARAM, "labels can not be nullptr.");
    APPERR_RETURN_IF_NOT_LOG(this->isTrained, APP_ERR_INVALID_PARAM, "Index not trained.");
    APPERR_RETURN_IF_NOT_FMT(n <= static_cast<idx_t>(std::numeric_limits<int>::max()),
        APP_ERR_INVALID_PARAM, "indices exceeds max(%d)", std::numeric_limits<int>::max());
    APPERR_RETURN_IF(n == 0 || k == 0, APP_ERR_OK);

    // 2. call the searchBatched func
    return searchBatched(indexes, n, x, k, distances, labels, filterSize, filters);
}
APP_ERROR Index::searchBatched(std::vector<Index *> indexes, int n, const float16_t *x, int k, float16_t *distances,
    idx_t *labels, uint32_t filterSize, uint32_t *filters)
{
    VALUE_UNUSED(indexes);
    VALUE_UNUSED(distances);
    VALUE_UNUSED(labels);
    VALUE_UNUSED(filterSize);
    VALUE_UNUSED(filters);
    VALUE_UNUSED(n);
    VALUE_UNUSED(x);
    VALUE_UNUSED(k);
 
    return APP_ERR_ILLEGAL_OPERATRION;
}

APP_ERROR Index::searchBatched(int n, const float16_t *x, int k, float16_t *distance, idx_t *labels)
{
    APP_LOG_INFO("Index::searchBatched start\n");
    APP_ERROR ret = APP_ERR_OK;
    size_t size = searchBatchSizes.size();
    if (n > 1 && size > 0) {
        int searched = 0;
        for (size_t i = 0; i < size; i++) {
            int batchSize = searchBatchSizes[i];
            if ((n - searched) >= batchSize) {
                int page = (n - searched) / batchSize;
                for (int j = 0; j < page; j++) {
                    ret = searchImpl(batchSize, x + searched * this->dims, k, distance + searched * k,
                        labels + searched * k);
                    APPERR_RETURN_IF(ret != APP_ERR_OK, ret);
                    searched += batchSize;
                }
            }
        }

        for (int i = searched; i < n; i++) {
            ret = searchImpl(1, x + i * this->dims, k, distance + i * k, labels + i * k);
            APPERR_RETURN_IF(ret != APP_ERR_OK, ret);
        }
        return APP_ERR_OK;
    } else {
        return searchImpl(n, x, k, distance, labels);
    }
    APP_LOG_INFO("Index::searchBatched end\n");
}

APP_ERROR Index::searchBatched(int n, const float16_t *x, int k, float16_t *distance, idx_t *labels, uint8_t* masks)
{
    uint32_t maskOffset = 0;
    uint32_t maskLen = utils::divUp(ntotal, BIT_OF_UINT8);
    size_t size = searchBatchSizes.size();
    if (n > 1 && size > 0) {
        int searched = 0;
        for (size_t i = 0; i < size; i++) {
            int batchSize = searchBatchSizes[i];
            if ((n - searched) >= batchSize) {
                int page = (n - searched) / batchSize;
                for (int j = 0; j < page; j++) {
                    maskOffset = static_cast<uint32_t>(searched) * maskLen;
                    APP_ERROR ret = searchFilterImpl(batchSize, x + searched * this->dims, k, distance + searched * k,
                        labels + searched * k, masks + maskOffset, maskLen);
                    APPERR_RETURN_IF(ret != APP_ERR_OK, ret);

                    searched += batchSize;
                }
            }
        }

        for (int i = searched; i < n; i++) {
            maskOffset = static_cast<uint32_t>(i) * maskLen;
            APP_ERROR ret = searchFilterImpl(1, x + i * this->dims, k, distance + i * k, labels + i * k,
                masks + maskOffset, maskLen);
            APPERR_RETURN_IF(ret != APP_ERR_OK, ret);
        }
        return APP_ERR_OK;
    } else {
        return searchFilterImpl(n, x, k, distance, labels, masks, maskLen);
    }
}

APP_ERROR Index::searchBatched(int n, const float16_t *x, int k, float16_t *distance, idx_t *labels,
    float16_t *l1distances, uint32_t filterSize, uint32_t* filters)
{
    VALUE_UNUSED(n);
    VALUE_UNUSED(x);
    VALUE_UNUSED(k);
    VALUE_UNUSED(distance);
    VALUE_UNUSED(labels);
    VALUE_UNUSED(l1distances);
    VALUE_UNUSED(filterSize);
    VALUE_UNUSED(filters);

    return APP_ERR_ILLEGAL_OPERATRION;
}

APP_ERROR Index::searchBatched(int n, const float16_t *x, int k, float16_t *distance, idx_t *labels,
    uint64_t filterSize, uint32_t* filters)
{
    VALUE_UNUSED(n);
    VALUE_UNUSED(x);
    VALUE_UNUSED(k);
    VALUE_UNUSED(distance);
    VALUE_UNUSED(labels);
    VALUE_UNUSED(filterSize);
    VALUE_UNUSED(filters);

    return APP_ERR_ILLEGAL_OPERATRION;
}

APP_ERROR Index::searchBatched(std::vector<Index*> indexes, int n, const float16_t *x, int k,
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

    int searched = 0;
    for (size_t i = 0; i < size; i++) {
        int batchSize = searchBatchSizes[i];
        if ((n - searched) >= batchSize) {
            int page = (n - searched) / batchSize;
            for (int j = 0; j < page; j++) {
                ret = searchImpl(indexes, n, batchSize, x + searched * this->dims, k,
                    distances + searched * k, labels + searched * k);
                APPERR_RETURN_IF(ret != APP_ERR_OK, ret);
                searched += batchSize;
            }
        }
    }

    for (int i = searched; i < n; i++) {
        ret = searchImpl(indexes, n, 1, x + i * this->dims, k, distances + i * k, labels + i * k);
        APPERR_RETURN_IF(ret != APP_ERR_OK, ret);
    }

    reorder(indexes.size(), n, k, distances, labels);
    return APP_ERR_OK;
}
APP_ERROR Index::searchBatched(std::vector<Index *> indexes, int n, const float16_t *x, int k, float16_t *distances,
    idx_t *labels, uint32_t filterSize, uint32_t **filters)
{
    VALUE_UNUSED(indexes);
    VALUE_UNUSED(n);
    VALUE_UNUSED(x);
    VALUE_UNUSED(k);
    VALUE_UNUSED(distances);
    VALUE_UNUSED(labels);
    VALUE_UNUSED(filterSize);
    VALUE_UNUSED(filters);

    return APP_ERR_ILLEGAL_OPERATRION;
}

void Index::initSearchResult(int indexesSize, int n, int k, float16_t *distances, idx_t *labels)
{
    APP_LOG_WARNING("the method of searchImpl for init SearchResult is not implement.\n");
    VALUE_UNUSED(indexesSize);
    VALUE_UNUSED(n);
    VALUE_UNUSED(k);
    VALUE_UNUSED(distances);
    VALUE_UNUSED(labels);
}

APP_ERROR Index::searchFilterImpl(std::vector<Index*> indexes, int n, int batchSize, const float16_t *x, int k,
    float16_t *distances, idx_t *labels, AscendTensor<uint8_t, DIMS_2>& maskData, AscendTensor<int, DIMS_1>& maskOffset)
{
    APP_LOG_WARNING("the method of searchFilterImpl for multi index is not implement.\n");
 
    VALUE_UNUSED(indexes);
    VALUE_UNUSED(n);
    VALUE_UNUSED(batchSize);
    VALUE_UNUSED(x);
    VALUE_UNUSED(k);
    VALUE_UNUSED(distances);
    VALUE_UNUSED(labels);
    VALUE_UNUSED(maskData);
    VALUE_UNUSED(maskOffset);
 
    return APP_ERR_NOT_IMPLEMENT;
}

APP_ERROR Index::searchImpl(std::vector<Index*> indexes, int n, int batchSize, const float16_t *x, int k,
                            float16_t *distances, idx_t *labels)
{
    APP_LOG_WARNING("the method of searchImpl for multi index is not implement.\n");

    VALUE_UNUSED(indexes);
    VALUE_UNUSED(n);
    VALUE_UNUSED(batchSize);
    VALUE_UNUSED(x);
    VALUE_UNUSED(k);
    VALUE_UNUSED(distances);
    VALUE_UNUSED(labels);

    return APP_ERR_NOT_IMPLEMENT;
}

void Index::reorder(int indexNum, int n, int k, float16_t *distances, idx_t *labels)
{
    VALUE_UNUSED(indexNum);
    VALUE_UNUSED(n);
    VALUE_UNUSED(k);
    VALUE_UNUSED(distances);
    VALUE_UNUSED(labels);
}


APP_ERROR Index::searchFilterImpl(int n, const float16_t *x, int k, float16_t *distances, idx_t *labels,
    uint8_t* masks, uint32_t maskLen)
{
    APP_LOG_WARNING("the method of searchFilterImpl for multi index is not implement.\n");

    VALUE_UNUSED(n);
    VALUE_UNUSED(x);
    VALUE_UNUSED(k);
    VALUE_UNUSED(distances);
    VALUE_UNUSED(labels);
    VALUE_UNUSED(masks);
    VALUE_UNUSED(maskLen);

    return APP_ERR_NOT_IMPLEMENT;
}

void Index::runDistanceCompute(
    AscendTensor<float16_t, DIMS_2>& queryVecs,
    AscendTensor<float16_t, DIMS_4>& shapedData,
    AscendTensor<float16_t, DIMS_1>& norms,
    AscendTensor<float16_t, DIMS_2>& outDistances,
    AscendTensor<uint16_t, DIMS_1>& flag,
    aclrtStream stream)
{
    AscendOperator *op = nullptr;
    int batch = queryVecs.getSize(0);
    if (distComputeOps.find(batch) != distComputeOps.end()) {
        op = distComputeOps[batch].get();
    }
    ASCEND_THROW_IF_NOT(op);

    std::shared_ptr<std::vector<const aclDataBuffer *>> distOpInput(
        new std::vector<const aclDataBuffer *>(), CommonUtils::AclInputBufferDelete);
    distOpInput->emplace_back(aclCreateDataBuffer(queryVecs.data(), queryVecs.getSizeInBytes()));
    distOpInput->emplace_back(aclCreateDataBuffer(shapedData.data(), shapedData.getSizeInBytes()));
    distOpInput->emplace_back(aclCreateDataBuffer(norms.data(), norms.getSizeInBytes()));

    std::shared_ptr<std::vector<aclDataBuffer *>> distOpOutput(
        new std::vector<aclDataBuffer *>(), CommonUtils::AclOutputBufferDelete);
    distOpOutput->emplace_back(aclCreateDataBuffer(outDistances.data(), outDistances.getSizeInBytes()));
    distOpOutput->emplace_back(aclCreateDataBuffer(flag.data(), flag.getSizeInBytes()));

    op->exec(*distOpInput, *distOpOutput, stream);
}

APP_ERROR Index::resetDistCompOperator(int numLists)
{
    auto distCompOpReset = [&](std::unique_ptr<AscendOperator> &op, int64_t batch) {
        AscendOpDesc desc("DistanceComputeFlat");
        std::vector<int64_t> queryShape({ batch, dims });
        std::vector<int64_t> coarseCentroidsShape({ utils::divUp(numLists, CUBE_ALIGN),
            utils::divUp(dims, CUBE_ALIGN), CUBE_ALIGN, CUBE_ALIGN });
        std::vector<int64_t> preNormsShape({ numLists });
        std::vector<int64_t> distResultShape({ batch, numLists });
        std::vector<int64_t> flagShape({ FLAG_ALIGN });
        desc.addInputTensorDesc(ACL_FLOAT16, queryShape.size(), queryShape.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_FLOAT16, coarseCentroidsShape.size(), coarseCentroidsShape.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_FLOAT16, preNormsShape.size(), preNormsShape.data(), ACL_FORMAT_ND);
        desc.addOutputTensorDesc(ACL_FLOAT16, distResultShape.size(), distResultShape.data(), ACL_FORMAT_ND);
        desc.addOutputTensorDesc(ACL_UINT16, flagShape.size(), flagShape.data(), ACL_FORMAT_ND);

        op.reset();
        op = CREATE_UNIQUE_PTR(AscendOperator, desc);
        return op->init();
    };

    for (auto batch : searchBatchSizes) {
        distComputeOps[batch] = std::unique_ptr<AscendOperator>(nullptr);
        APPERR_RETURN_IF_NOT_LOG(distCompOpReset(distComputeOps[batch], batch),
                                 APP_ERR_ACL_OP_LOAD_MODEL_FAILED, "op init failed");
    }

    return APP_ERR_OK;
}

float16_t Index::fvecNormL2sqr(const float16_t *x, size_t d) const
{
    double res = 0;
    for (size_t i = 0; i < d; i++) {
        res += x[i] * x[i];
    }
    return res;
}

void Index::fvecNormsL2sqr(float16_t *nr, const float16_t *x, size_t d, size_t nx) const
{
#pragma omp parallel for
    for (size_t i = 0; i < nx; i++) {
        nr[i] = fvecNormL2sqr(x + i * d, d);
    }
}

void Index::fvecNormsL2sqrAicpu(AscendTensor<float16_t, DIMS_1> &nr,
                                AscendTensor<float16_t, DIMS_2> &x)
{
    AscendOpDesc desc("VecL2Sqr");
    std::vector<int64_t> shape0 { x.getSize(0), x.getSize(1) };
    std::vector<int64_t> shape1 { nr.getSize(0) };
    desc.addInputTensorDesc(ACL_FLOAT16, shape0.size(), shape0.data(), ACL_FORMAT_ND);
    desc.addOutputTensorDesc(ACL_FLOAT16, shape1.size(), shape1.data(), ACL_FORMAT_ND);
    auto op = CREATE_UNIQUE_PTR(AscendOperator, desc);
    if (!op->init()) {
        APP_LOG_ERROR("vec l2sqr op init failed");
        return;
    }

    std::shared_ptr<std::vector<const aclDataBuffer *>> distOpInput(
        new std::vector<const aclDataBuffer *>, CommonUtils::AclInputBufferDelete);
    distOpInput->emplace_back(aclCreateDataBuffer(x.data(), x.getSizeInBytes()));
    std::shared_ptr<std::vector<aclDataBuffer *>> distOpOutput(
         new std::vector<aclDataBuffer *>, CommonUtils::AclOutputBufferDelete);
    distOpOutput->emplace_back(aclCreateDataBuffer(nr.data(), nr.getSizeInBytes()));
    auto stream = resources.getDefaultStream();
    op->exec(*distOpInput, *distOpOutput, stream);

    auto ret = aclrtSynchronizeStream(stream);
    if (ret != ACL_SUCCESS) {
        APP_LOG_ERROR("vec l2sqr op init run failed");
    }
}

APP_ERROR Index::reserveMemory(size_t numVecs)
{
    VALUE_UNUSED(numVecs);
    APP_LOG_ERROR("reserveMemory not implemented for this type of index");
    return APP_ERR_NOT_IMPLEMENT;
}

size_t Index::reclaimMemory()
{
    ASCEND_THROW_MSG("reclaimMemory not implemented for this type of index");
    return 0;
}
} // namespace ascendSearch
