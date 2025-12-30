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


#include <algorithm>
#include <future>
#include <map>
#include <cstdlib>
#include <vector>
#include <mutex>
#include "ascenddaemon/AscendResourcesProxy.h"
#include "ascenddaemon/utils/AscendOperator.h"
#include "ascenddaemon/utils/AscendTensor.h"
#include "ascenddaemon/utils/AscendUtils.h"
#include "ascenddaemon/utils/DeviceVector.h"
#include "ascenddaemon/utils/Limits.h"
#include "ascenddaemon/utils/TopkOp.h"
#include "ascenddaemon/utils/StaticUtils.h"
#include "common/threadpool/AscendThreadPool.h"
#include "common/utils/CommonUtils.h"
#include "common/utils/SocUtils.h"
#include "ascenddaemon/utils/AscendRWLock.h"
#include "ascenddaemon/impl_device/IndexILFlat.h"

namespace ascend {
namespace {
const int CORE_NUM = faiss::ascend::SocUtils::GetInstance().GetCoreNum();
const int THREADS_CNT = faiss::ascend::SocUtils::GetInstance().GetThreadsCnt();
const int SIZE_ALIGN = 8;
const int ZREGION_HEIGHT = 2; // Features are stored as "Z" layout, the height is aligned by preset value
const int CUBE_ALIGN = 16;
const int FLAG_SIZE = 16; // set flag size as 16 to pad to 32b, the minimum data_move size of UB.
const int BURST_LEN = 64;
const int BIG_BATCH_BURST_LEN = 32;
const int BIG_BATCH_THRESHOLD = 64;
const int ACTUAL_TABLE_LEN = 10000;
const int TABLE_LEN = 10048; // mapping table with redundancy of 48
const int IDX_BURST_LEN = 64;
const int FLAT_BLOCK_SIZE = 16384 * 16;
const int FLAT_COMPUTE_PAGE = FLAT_BLOCK_SIZE * 16;
const double TIMEOUT_MS = 50000;
const int TIMEOUT_CHECK_TICK = 5120;
const int MIN_RESOURCE = 0x8000000; // 0x8000000 mean 128MB
const int64_t MAX_RESOURCE = 0x100000000; // 0x100000000 mean 4096MB
const int FAKE_HUGE_BASE = 20000000; // Fake size setted for tik operators
const int MAX_CAP = 12000000; // Upper limit for capacity
const size_t MAX_BASE_SPACE = 12288000000; // max bytes to store base vectors.
constexpr int UNINITIALIZE_NTOTAL = -1; // uninitialize ntotal
constexpr size_t KB = 1024;
std::vector<int> DIMS = {32, 64, 128, 256, 384, 512, 1024};
std::vector<int> BATCHES = {128, 64, 48, 32, 30, 18, 16, 8, 6, 4, 2, 1};
std::vector<int> COMPUTE_BATCHES = {256, 128, 64, 48, 32, 30, 18, 16, 8, 6, 4, 2, 1};
std::vector<int> COMPUTE_BY_IDX_BATCHES = {256, 128, 64, 48, 32, 16, 8, 6, 4, 2, 1};
}

#define CHECK_FEATURE_PARAMS(N, FEATURES, INDICES)                                                 \
    do {                                                                                           \
        APPERR_RETURN_IF((N) == 0, APP_ERR_OK);                                                    \
        APPERR_RETURN_IF_NOT_FMT((N) >= 0 && (N) <= this->capacity, APP_ERR_INVALID_PARAM,         \
            "The number n should be in range [0, %d]", this->capacity);                            \
        APPERR_RETURN_IF_NOT_LOG(FEATURES, APP_ERR_INVALID_PARAM, "Features can not be nullptr."); \
        APPERR_RETURN_IF_NOT_LOG(INDICES, APP_ERR_INVALID_PARAM, "Indices can not be nullptr.");   \
    } while (false)


#define CHECK_DISTANCE_COMPUTE_PARAMS(N, QUERIES, DISTANCE)                                        \
    do {                                                                                           \
        APPERR_RETURN_IF((N) == 0, APP_ERR_OK);                                                    \
        APPERR_RETURN_IF_NOT_FMT((N) >= 0 && (N) <= this->capacity, APP_ERR_INVALID_PARAM,         \
            "The number n should be greater than 0 and smaller than capacity %d", this->capacity); \
        APPERR_RETURN_IF_NOT_LOG(QUERIES, APP_ERR_INVALID_PARAM, "Queries can not be nullptr.");   \
        APPERR_RETURN_IF_NOT_LOG(DISTANCE, APP_ERR_INVALID_PARAM, "Distance can not be nullptr."); \
    } while (false)

#define CHECK_TABLE_PARAMS(TABLELEN, TABLE)                                                 \
    do {                                                                                    \
        if ((TABLELEN) > 0) {                                                               \
            APPERR_RETURN_IF_NOT_LOG((TABLELEN) == ACTUAL_TABLE_LEN, APP_ERR_INVALID_PARAM, \
                "Unsupported table length. ");                                              \
            APPERR_RETURN_IF_NOT_LOG((TABLE), APP_ERR_INVALID_PARAM,                        \
                "The table pointer cannot be nullptr when tableLen is valid.");             \
        }                                                                                   \
    } while (false)

#define CHECK_TOPK(TOPK)                                                                                       \
    do {                                                                                                       \
        APPERR_RETURN_IF((TOPK) == 0, APP_ERR_OK);                                                             \
        APPERR_RETURN_IF_NOT_LOG((TOPK) >= 0 && (TOPK) <= 1024, APP_ERR_INVALID_PARAM,                         \
            "Invalid parameter, topk should be in range [0, 1024].");                                          \
    } while (false)

#define CHECK_LEGAL_OPERATION(INITIALIZED)                                  \
    do {                                                                    \
        APPERR_RETURN_IF_NOT_LOG((INITIALIZED), APP_ERR_ILLEGAL_OPERATION, \
            "Illegal operation, please initialize the index first. ");      \
    } while (false)

class IndexILFlatImpl {
public:
    IndexILFlatImpl() = default;

    ~IndexILFlatImpl();

    IndexILFlatImpl(int dim, int capacity, int64_t resourceSize);

    APP_ERROR Add(int n, const float16_t *features, const idx_t *indices);

    APP_ERROR Get(int n, float16_t *features, const idx_t *indices);

    APP_ERROR Remove(int n, const idx_t *indices);

    int GetNTotal() const;

    size_t GetMaxCapacity() const;

    void SetNTotal(int n);

    APP_ERROR Init();

    APP_ERROR ResetDistOps();

    APP_ERROR Finalize();

    APP_ERROR Search(int n, const float16_t *queries, int topk, idx_t *indices, float *distances,
        unsigned int tableLen = 0, const float *table = nullptr);

    APP_ERROR SearchByThreshold(int n, const float16_t *queries, float threshold, int topk, int *num, idx_t *indices,
        float *distances, unsigned int tableLen = 0, const float *table = nullptr);

    APP_ERROR ComputeDistByThresholdBatched(int n, const float16_t *queries, float *distances,
        std::tuple<unsigned int, const float *> tableInfo,
        std::tuple<float, int *, idx_t *> filterParams = std::tuple<float, int *, idx_t *>(-2, nullptr, nullptr));

    APP_ERROR ComputeDistanceBatched(int n, const float16_t *queries, float *distances,
        std::tuple<unsigned int, const float *> tableInfo);

    APP_ERROR ComputeDistanceByIdx(int n, const float16_t *queries, const int *num, const idx_t *indices,
        float *distances, unsigned int tableLen = 0, const float *table = nullptr);

protected:
    APP_ERROR SearchImpl(int n, const float16_t *queries, int topk, idx_t *indices, float *distances);

    APP_ERROR SearchPaged(int pageId, AscendTensor<float16_t, DIMS_2> &queries, int topk,
        AscendTensor<idx_t, DIMS_2> &idxs, AscendTensor<float16_t, DIMS_2> &distances);

    APP_ERROR TableMapping(int n, float *distances, unsigned int tableLen, const float *table, int topk = 0);

    APP_ERROR ComputeDistImplBatched(int n, const float16_t *queries, float *distances,
        std::tuple<unsigned int, const float *> tableInfo);

    APP_ERROR ComputeDistByIdxImpl(int n, const float16_t *queries, float *distances,
        std::tuple<int, const int *, const idx_t *> idxInfo, std::tuple<unsigned int, const float *> tableInfo);

    APP_ERROR ComputeDistPaged(int n, const float16_t *queries, float *distances,
        std::tuple<unsigned int, const float *> tableInfo, std::tuple<float, int *, idx_t *> filterParams);

    void ComputeDistWholeBase(AscendTensor<float16_t, DIMS_2> &queryTensor,
        AscendTensor<uint32_t, DIMS_1> &sizeTensor, AscendTensor<float16_t, DIMS_4> &shapedData,
        AscendTensor<uint32_t, DIMS_2> &idxTensor, AscendTensor<float, DIMS_2> &distanceTensor, aclrtStream stream);

    void ComputeDistWholeBaseWithTable(AscendTensor<float16_t, DIMS_2> &queryTensor,
        AscendTensor<uint32_t, DIMS_1> &sizeTensor, AscendTensor<float16_t, DIMS_4> &shapedData,
        AscendTensor<uint32_t, DIMS_2> &idxTensor, AscendTensor<float, DIMS_2> &distanceTensor,
        AscendTensor<float, DIMS_1> &tableTensor, aclrtStream stream);

    void ComputeBlockDist(AscendTensor<float16_t, DIMS_2> &queryTensor,
        AscendTensor<uint8_t, DIMS_2> &mask, AscendTensor<float16_t, DIMS_4> &shapedData,
        AscendTensor<uint32_t, DIMS_2> &size, AscendTensor<float16_t, DIMS_2> &outDistances,
        AscendTensor<float16_t, DIMS_2> &maxDistances, AscendTensor<uint16_t, DIMS_2> &flag, aclrtStream stream);

    void ComputeBlockDistWholeBase(AscendTensor<float16_t, DIMS_2> &queryTensor,
        AscendTensor<float16_t, DIMS_4> &shapedData, AscendTensor<uint32_t, DIMS_1> &baseSize,
        AscendTensor<float, DIMS_1> &distanceTensor, aclrtStream stream);

    void ComputeBlockDistWholeBaseWithTable(AscendTensor<float16_t, DIMS_2> &queryTensor,
        AscendTensor<float16_t, DIMS_4> &shapedData, AscendTensor<uint32_t, DIMS_1> &baseSize,
        AscendTensor<float, DIMS_1> &tableTensor, AscendTensor<float, DIMS_1> &distanceTensor, aclrtStream stream);

    void DistanceFilter(AscendTensor<float, DIMS_1> &distanceTensor, AscendTensor<float, DIMS_1> &thresholdTensor,
        AscendTensor<uint32_t, DIMS_1> &baseSize, AscendTensor<uint32_t, DIMS_1> &initIdx,
        AscendTensor<float, DIMS_2> &filteredDist, AscendTensor<uint32_t, DIMS_2> &filteredIndice,
        AscendTensor<uint32_t, DIMS_1> &filteredCnt, aclrtStream stream);

    size_t GetVecCapacity(size_t vecNum, size_t size) const;

    int dim;
    int pageSize;
    int capacity;
    int blockSize;
    int burstsOfBlock;
    int64_t resourceSize;
    int ntotal;
    int zregionHeight;
    bool isInitialized;
    int blockMaskSize;
    std::vector<int> searchBatchSizes;
    std::vector<int> computeBatchSizes;
    size_t fakeBaseSizeInBytes;
    std::unique_ptr<AscendThreadPool> threadPool;
    AscendResourcesProxy resources;
    std::unique_ptr<DeviceVector<float16_t>> baseSpace;
    std::map<int, std::unique_ptr<AscendOperator>> distComputeOps;
    std::map<int, std::unique_ptr<AscendOperator>> distComputeIdxOps;
    std::map<int, std::unique_ptr<AscendOperator>> distComputeIdxWithTableOps;
    std::map<int, std::unique_ptr<AscendOperator>> distanceFlatIpOps;
    std::map<int, std::unique_ptr<AscendOperator>> distanceFlatIpWithTableOps;
    std::map<int, std::unique_ptr<AscendOperator>> distanceFilterOps;
    TopkOp<std::less<float16_t>, std::less_equal<float16_t>, float16_t, false, uint32_t> topkMaxOp;
    // 用于get接口的锁
    std::mutex getFeaturesMtx;
    size_t getFeatureMtxCnt;
    // 用于remove接口的锁
    std::mutex getRemoveMtx;
    size_t removeMtxCnt;
    // 用于add接口的锁
    std::mutex getAddMtx;
    size_t addMtxCnt;
    // 用于检索接口之间的锁
    mutable std::mutex mtx;

private:
    APP_ERROR ResetDistCompOp(int numLists);
    APP_ERROR ResetDistCompIdxOp();
    APP_ERROR ResetDistCompIdxWithTableOp();
    APP_ERROR ResetDistanceFlatIpOp();
    APP_ERROR ResetDistanceFlatIpWithTableOp();
    APP_ERROR ResetDistanceFilterOp();
};

IndexILFlatImpl::IndexILFlatImpl(int dim, int capacity, int64_t resourceSize)
    : dim(dim), capacity(capacity), resourceSize(resourceSize), isInitialized(false) {}

IndexILFlatImpl::~IndexILFlatImpl() {}

size_t IndexILFlatImpl::GetVecCapacity(size_t vecNum, size_t size) const
{
    size_t minCapacity = 512 * KB;
    const size_t needSize = vecNum * dim;
    const size_t align = 2 * KB * KB;

    // 1. needSize is smaller than minCapacity
    if (needSize < minCapacity) {
        return minCapacity;
    }

    // 2. needSize is smaller than current code.size, don't need to grow
    if (needSize <= size) {
        return size;
    }

    // 3. 2048 is the max code_each_loop, 2048 * dim for operator aligned
    size_t retMemory = utils::roundUp((needSize + 2048 * dim), align);

    // 2048 * dim for operator aligned
    return std::min(retMemory - 2048 * dim, static_cast<size_t>(this->blockSize) * dim);
}

APP_ERROR IndexILFlatImpl::ResetDistOps()
{
    APPERR_RETURN_IF_NOT_OK(ResetDistCompOp(FLAT_BLOCK_SIZE));
    APPERR_RETURN_IF_NOT_OK(ResetDistCompIdxOp());
    APPERR_RETURN_IF_NOT_OK(ResetDistCompIdxWithTableOp());
    APPERR_RETURN_IF_NOT_OK(ResetDistanceFlatIpOp());
    APPERR_RETURN_IF_NOT_OK(ResetDistanceFlatIpWithTableOp());
    APPERR_RETURN_IF_NOT_OK(ResetDistanceFilterOp());
    return APP_ERR_OK;
}

APP_ERROR IndexILFlatImpl::Init()
{
    AscendRWLock writeLock(&mtx);
    APPERR_RETURN_IF_NOT_FMT((size_t)capacity <= GetMaxCapacity(), APP_ERR_INVALID_PARAM,
        "Illegal capacity for dim %d, the upper boundary of capacity is: %zu, ", dim, GetMaxCapacity());
    this->ntotal = 0;
    this->pageSize = FLAT_COMPUTE_PAGE;
    this->searchBatchSizes = BATCHES;
    this->computeBatchSizes = COMPUTE_BATCHES;
    this->blockSize = FLAT_BLOCK_SIZE;
    // get the height of "Z" layout from environment
    this->zregionHeight = ZREGION_HEIGHT;
    // Double the BURST_LEN after round up, hence here we multiply 2
    this->burstsOfBlock = (FLAT_BLOCK_SIZE + BURST_LEN - 1) / BURST_LEN * 2;
    this->fakeBaseSizeInBytes = static_cast<uint64_t>(FAKE_HUGE_BASE) * this->dim * sizeof(float16_t);
    this->threadPool = std::make_unique<AscendThreadPool>(THREADS_CNT);
    this->blockMaskSize = utils::divUp(this->blockSize, SIZE_ALIGN);
    auto ret = ResetDistOps();
    APPERR_RETURN_IF_NOT_FMT(ret == APP_ERR_OK, APP_ERR_INNER_ERROR, "distOps reset error %d", ret);

    try {
        if (resourceSize == 0) {
            this->resources.noTempMemory();
        } else if (resourceSize > 0) {
            this->resources.setTempMemory(resourceSize);
        }
        this->resources.initialize();
    } catch (std::exception &e) {
        APPERR_RETURN_IF_NOT_LOG(false, APP_ERR_INNER_ERROR, "Acl resource initialization failed. ");
    }

    size_t blockAlign = utils::divUp(capacity, this->blockSize);
    size_t blockSizeAlign = utils::divUp(this->blockSize, CUBE_ALIGN);
    int dimAlign = utils::divUp(this->dim, CUBE_ALIGN);
    try {
        this->baseSpace = std::make_unique<DeviceVector<float16_t>>(MemorySpace::DEVICE_HUGEPAGE);
        // 库容比较小的场景下要节约内存
        if (capacity < blockSize) {
            auto conserveCapacity = GetVecCapacity(capacity, baseSpace->size());
            this->baseSpace->resize(conserveCapacity, true);
        } else {
            auto paddedSize = blockAlign * blockSizeAlign * dimAlign * CUBE_ALIGN * CUBE_ALIGN;
            this->baseSpace->resize(paddedSize, true);
        }
    } catch (std::exception &e) {
        APPERR_RETURN_IF_NOT_LOG(false, APP_ERR_INNER_ERROR, "Malloc device memory failed. ");
    }
    isInitialized = true;
    getFeatureMtxCnt = 0;
    removeMtxCnt = 0;
    addMtxCnt = 0;
    return APP_ERR_OK;
}

APP_ERROR IndexILFlatImpl::Finalize()
{
    AscendRWLock writeLock(&mtx);
    CHECK_LEGAL_OPERATION(isInitialized);
    this->baseSpace->clear();
    isInitialized = false;
    return APP_ERR_OK;
}

int IndexILFlatImpl::GetNTotal() const
{
    AscendRWLock writeLock(&mtx);
    return this->ntotal;
}

size_t IndexILFlatImpl::GetMaxCapacity() const
{
    size_t vectorSize = (size_t)dim * sizeof(float16_t);
    return MAX_BASE_SPACE / vectorSize;
}

void IndexILFlatImpl::SetNTotal(int n)
{
    AscendRWLock writeLock(&mtx);
    this->ntotal = n;
}

APP_ERROR IndexILFlatImpl::Add(int n, const float16_t *features, const idx_t *indices)
{
    AscendRWLock lock(&getAddMtx, &addMtxCnt, &mtx);
    CHECK_LEGAL_OPERATION(isInitialized);
    // cube align, save raw data to shaped data
    size_t dimAlign = utils::divUp(this->dim, CUBE_ALIGN);
    auto memErr = EOK;

    idx_t total = 0;
    // 按indice插入
    for (size_t i = 0; i < static_cast<size_t>(n); ++i) {
        auto seq = static_cast<size_t>(*(indices + i));
        APPERR_RETURN_IF_NOT_LOG(seq < static_cast<size_t>(this->capacity), APP_ERR_INVALID_PARAM,
            "Added features should be of indice smaller than capacity.\n");
        auto offset1 = seq / zregionHeight * dimAlign;
        auto offset2 = seq % zregionHeight;
        size_t offset = offset1 * (zregionHeight * CUBE_ALIGN) + offset2 * CUBE_ALIGN;
        auto dataptr = this->baseSpace->data() + offset;

        for (size_t j = 0; j < dimAlign; j++) {
            int padding = (j == dimAlign - 1) ? ((j + 1) * CUBE_ALIGN - this->dim) : 0;
            auto err = memcpy_s(dataptr, (CUBE_ALIGN - padding) * sizeof(float16_t),
                features + i * this->dim + j * CUBE_ALIGN, (CUBE_ALIGN - padding) * sizeof(float16_t));
            ASCEND_EXC_IF_NOT_FMT(err == EOK, memErr = err, "memcpy rawData err, i=%zu, j=%zu, err=%d", i, j, err);
            if (padding > 0) {
                auto err = memset_s(dataptr + (CUBE_ALIGN - padding), sizeof(float16_t) * padding, 0x0,
                    sizeof(float16_t) * padding);
                ASCEND_EXC_IF_NOT_FMT(err == EOK, memErr = err, "memset tmpData err, i=%zu, j=%zu, err=%d", i, j, err);
            }
            dataptr += (zregionHeight * CUBE_ALIGN);
        }
        total = std::max(total, static_cast<idx_t>(seq + 1));
    }
    APPERR_RETURN_IF_NOT_FMT(memErr == EOK, APP_ERR_INNER_ERROR, "Mem operator error %d", (int)memErr);
    // 这里要操作全局变量，单独加锁
    {
        AscendRWLock writeLock(&getAddMtx);
        this->ntotal = std::max(this->ntotal, static_cast<int>(total));
    }
    return APP_ERR_OK;
}

APP_ERROR IndexILFlatImpl::Get(int n, float16_t *features, const idx_t *indices)
{
    AscendRWLock lock(&getFeaturesMtx, &getFeatureMtxCnt, &mtx);
    CHECK_LEGAL_OPERATION(isInitialized);
    AscendTensor<float16_t, DIMS_2> featureResult(features, { n, this->dim });

    size_t dimAlign = utils::divUp(this->dim, CUBE_ALIGN);
    for (size_t i = 0; i < static_cast<size_t>(n); ++i) {
        auto seq = static_cast<size_t>(*(indices + i));
        APPERR_RETURN_IF_NOT_LOG(seq < static_cast<size_t>(this->ntotal), APP_ERR_INVALID_PARAM,
            "Invalid feature to get, the indice should be smaller than ntotal\n");
        float16_t *dataPtr = this->baseSpace->data() + seq / zregionHeight * dimAlign * (zregionHeight * CUBE_ALIGN) +
            seq % zregionHeight * CUBE_ALIGN;

        for (size_t j = 0; j < dimAlign; j++) {
            size_t getOffset = i * this->dim + j * CUBE_ALIGN;
            size_t cpyNum = (j == dimAlign - 1) ? (this->dim - j * CUBE_ALIGN) : CUBE_ALIGN;
            auto err = memcpy_s(featureResult.data() + getOffset, cpyNum * sizeof(float16_t),
                dataPtr + j * zregionHeight * CUBE_ALIGN, cpyNum * sizeof(float16_t));
            APPERR_RETURN_IF_NOT_FMT(err == EOK, APP_ERR_INNER_ERROR,
                "memcpy error, (i=%zu,j=%zu)target buf remains %zu. err=%d\n", i, j, (n - i) * sizeof(float16_t), err);
        }
    }
    return APP_ERR_OK;
}

APP_ERROR IndexILFlatImpl::Remove(int n, const idx_t *indices)
{
    AscendRWLock lock(&getRemoveMtx, &removeMtxCnt, &mtx);
    CHECK_LEGAL_OPERATION(isInitialized);
    // cube align, set removed features as all 0
    size_t dimAlign = utils::divUp(this->dim, CUBE_ALIGN);
    auto memErr = EOK;
    for (int i = 0; i < n; i++) {
        auto seq = static_cast<size_t>(*(indices + i));
        APPERR_RETURN_IF_NOT_LOG(seq < static_cast<size_t>(this->ntotal), APP_ERR_INVALID_PARAM,
            "Invalid feature to remove, the indice should not be greater than ntotal\n");
        size_t offset1 = seq / zregionHeight * dimAlign;
        size_t offset2 = seq % zregionHeight;
        size_t offset = offset1 * zregionHeight * CUBE_ALIGN + offset2 * CUBE_ALIGN;
        auto dataptr = this->baseSpace->data() + offset;

        for (size_t j = 0; j < dimAlign; j++) {
            auto err = memset_s(dataptr, CUBE_ALIGN * sizeof(float16_t), 0x0, CUBE_ALIGN * sizeof(float16_t));
            ASCEND_EXC_IF_NOT_FMT(err == EOK, memErr = err, "inner memset err, i=%zu, j=%zu, err=%d\n", i, j, err);
            dataptr += zregionHeight * CUBE_ALIGN;
        }
    }
    APPERR_RETURN_IF_NOT_FMT(memErr == EOK, APP_ERR_INNER_ERROR, "Mem operator error %d", (int)memErr);
    return APP_ERR_OK;
}

APP_ERROR IndexILFlatImpl::Search(int n, const float16_t *queries, int topk, idx_t *indices, float *distances,
    unsigned int tableLen, const float *table)
{
    CHECK_LEGAL_OPERATION(isInitialized);
    APP_ERROR ret = APP_ERR_OK;
    size_t size = searchBatchSizes.size();
    if (n > 1 && size > 0) {
        int searched = 0;
        for (size_t i = 0; i < size; i++) {
            int batchSize = searchBatchSizes[i];
            if ((n - searched) >= batchSize) {
                size_t page = (n - searched) / batchSize;
                for (size_t j = 0; j < page; j++) {
                    ret = SearchImpl(batchSize, queries + static_cast<size_t>(searched) * this->dim, topk,
                        indices + static_cast<size_t>(searched) * topk,
                        distances + static_cast<size_t>(searched) * topk);
                    APPERR_RETURN_IF(ret, ret);
                    searched += batchSize;
                }
            }
        }
    } else {
        ret = SearchImpl(n, queries, topk, indices, distances);
        APPERR_RETURN_IF(ret, ret);
    }

    return this->TableMapping(n, distances, tableLen, table, topk);
}

APP_ERROR IndexILFlatImpl::SearchImpl(int n, const float16_t *x, int topk, idx_t *indices, float *distances)
{
    AscendTensor<float16_t, DIMS_2> outDistances({ n, topk });
    AscendTensor<idx_t, DIMS_2> outIndices(indices, { n, topk });
    AscendTensor<float16_t, DIMS_2> queries(const_cast<float16_t *>(x), { n, this->dim });

    // 1. init output data
    outDistances.initValue(Limits<float16_t>::getMin());
    outIndices.initValue(std::numeric_limits<idx_t>::max());

    // 2. compute distance by code page
    int pageNum = utils::divUp(this->ntotal, this->pageSize);

    for (int pageId = 0; pageId < pageNum; ++pageId) {
        APP_ERROR ret = SearchPaged(pageId, queries, topk, outIndices, outDistances);
        APPERR_RETURN_IF(ret, ret);
    }

    // 3. reorder the topk results in ascend order
    topkMaxOp.reorder(outDistances, outIndices);

    // 4. transform distance from fp16 to fp32
    for (int i = 0; i < n; i++) {
        std::transform(outDistances[i].data(), outDistances[i].data() + topk, distances + static_cast<size_t>(i) * topk,
            [](float16_t tmpData) { return static_cast<float>(tmpData); });
    }
    return APP_ERR_OK;
}

APP_ERROR IndexILFlatImpl::SearchPaged(int pageId, AscendTensor<float16_t, DIMS_2> &queries, int topk,
    AscendTensor<idx_t, DIMS_2> &idxs, AscendTensor<float16_t, DIMS_2> &distances)
{
    AscendRWLock glock(&AscendGlobalLock::GetInstance(0));
    AscendRWLock lock(&mtx);
    auto streamPtr = this->resources.getDefaultStream();
    auto stream = streamPtr->GetStream();
    auto &mem = this->resources.getMemoryManager();
    int nq = queries.getSize(0);
    int pageOffset = pageId * this->pageSize;
    int blockOffset = pageOffset / this->blockSize;
    int computeNum = std::min(this->ntotal - pageOffset, this->pageSize);
    int blockNum = utils::divUp(computeNum, this->blockSize);
    auto burstLen = (nq >= BIG_BATCH_THRESHOLD) ? BIG_BATCH_BURST_LEN : BURST_LEN;
    // 乘以2，是和算子生成时的shape保持一致
    auto burstOfBlock = (FLAT_BLOCK_SIZE + burstLen - 1) / burstLen * 2;

    AscendTensor<float16_t, DIMS_3> distResult(mem, { blockNum, nq, this->blockSize }, stream);
    AscendTensor<float16_t, DIMS_3> maxDistResult(mem, { blockNum, nq, burstOfBlock }, stream);
    AscendTensor<uint32_t, DIMS_3> opSize(mem, { blockNum, CORE_NUM, SIZE_ALIGN }, stream);
    AscendTensor<uint16_t, DIMS_3> opFlag(mem, { blockNum, CORE_NUM, FLAG_SIZE }, stream);
    opFlag.zero();

    AscendTensor<float16_t, DIMS_2> maxDistances(mem, { nq, topk }, stream);
    AscendTensor<uint32_t, DIMS_2> maxIndices(mem, { nq, topk }, stream);
    maxDistances.initValue(Limits<float16_t>::getMin());
    maxIndices.initValue(std::numeric_limits<uint32_t>::max());

    AscendTensor<uint8_t, DIMS_2> mask(mem, { nq, this->blockMaskSize }, stream);
    // 1. run the topK operator to select the top K async
    bool errorQuit = false;
    auto topkFunctor = [&](int idx) {
        if (idx < THREADS_CNT) {
            CommonUtils::attachToCpu(idx);
        }
        AscendTensor<idx_t, DIMS_1> indices;
        uint32_t offset = pageOffset;
        for (int i = 0; i < blockNum && !errorQuit; ++i) {
            for (int j = 0; j < CORE_NUM; ++j) {
                volatile uint16_t *flagPtr = opFlag[i][j].data();
                WAITING_FLAG_READY(*flagPtr, TIMEOUT_CHECK_TICK, TIMEOUT_MS);
            }
            int size = opSize[i][0][0];
            for (int j = idx; j < nq; j += THREADS_CNT) {
                std::tuple<float16_t *, float16_t *, idx_t *> opOutTp(distResult[i][j].data(),
                    maxDistResult[i][j].data(), indices.data());
                std::tuple<float16_t *, idx_t *, int> topkHeapTp(distances[j].data(), idxs[j].data(), topk);
                std::tuple<float16_t *, uint32_t *> maxHeapTp(maxDistances[j].data(), maxIndices[j].data());

                if (i == 0) {
                    ASCEND_THROW_IF_NOT(topkMaxOp.exec(opOutTp, topkHeapTp, maxHeapTp, size, burstLen));
                } else {
                    ASCEND_THROW_IF_NOT(topkMaxOp.exec(opOutTp, topkHeapTp, size, offset, burstLen));
                }
            }
            offset += blockSize;
        }
    };

    int functorSize = (nq > THREADS_CNT) ? THREADS_CNT : nq;
    std::vector<std::future<void>> topkFunctorRet;
    for (int i = 0; i < functorSize; i++) {
        topkFunctorRet.emplace_back(this->threadPool->Enqueue(topkFunctor, i));
    }

    // 2. run the distance operator to compute the distance
    const int dim1 = utils::divUp(this->blockSize, zregionHeight);
    const int dim2 = utils::divUp(this->dim, CUBE_ALIGN);
    for (int i = 0; i < blockNum; i++) {
        auto baseOffset = static_cast<size_t>(blockOffset + i) * this->blockSize * dim2 * CUBE_ALIGN;
        AscendTensor<float16_t, DIMS_4> shaped(this->baseSpace->data() + baseOffset,
            { dim1, dim2, zregionHeight, CUBE_ALIGN });
        auto dist = distResult[i].view();
        auto maxDist = maxDistResult[i].view();
        auto actualSize = opSize[i].view();
        auto flag = opFlag[i].view();

        int offset = i * this->blockSize;
        auto actual = std::min(static_cast<uint32_t>(computeNum - offset), static_cast<uint32_t>(this->blockSize));
        actualSize[0][0] = actual;
        actualSize[0][3] = 0;  // 3 for IDX_USE_MASK, not use mask

        ComputeBlockDist(queries, mask, shaped, actualSize, dist, maxDist, flag, stream);
    }

    // 3. wait all the op task compute, avoid thread dispatch
    auto ret = synchronizeStream(stream);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR,
        "synchronizeStream failed: %i\n", ret);

    // 4. waiting for topk functor to finish
    int topkWaitIdx = 0;
    try {
        for (auto &ret : topkFunctorRet) {
            topkWaitIdx++;
            ret.get();
        }
    } catch (std::exception &e) {
        errorQuit = true;
        for_each(topkFunctorRet.begin() + topkWaitIdx, topkFunctorRet.end(), [](auto &ret) { ret.wait(); });
        ASCEND_THROW_MSG("wait for topk future failed.");
    }
    return APP_ERR_OK;
}

APP_ERROR IndexILFlatImpl::SearchByThreshold(int n, const float16_t *queries, float threshold, int topk, int *num,
    idx_t *indices, float *distances, unsigned int tableLen, const float *table)
{
    CHECK_LEGAL_OPERATION(isInitialized);
    APP_ERROR ret = this->Search(n, queries, topk, indices, distances, tableLen, table);
    APPERR_RETURN_IF(ret, ret);

    // fiter distances by threshold
    size_t validNum = std::min(topk, this->ntotal);
    for (int i = 0; i < n; i++) {
        int qnum = 0;
        for (size_t j = 0; j < validNum; j++) {
            size_t offset = static_cast<size_t>(i) * topk + j;
            if (*(distances + offset) >= threshold) {
                *(distances + static_cast<size_t>(i) * topk + qnum) = *(distances + offset);
                *(indices + static_cast<size_t>(i) * topk + qnum) = *(indices + offset);
                qnum += 1;
            }
        }
        *(num + i) = qnum;
    }
    return APP_ERR_OK;
}

APP_ERROR IndexILFlatImpl::ComputeDistanceBatched(int n, const float16_t *queries, float *distances,
    std::tuple<unsigned int, const float *> tableInfo)
{
    AscendRWLock writeLock(&mtx);
    CHECK_LEGAL_OPERATION(isInitialized);
    APP_ERROR ret = APP_ERR_OK;
    size_t size = this->computeBatchSizes.size();
    int padNtotal = utils::divUp(this->ntotal, CUBE_ALIGN) * CUBE_ALIGN;
    if (n > 1 && size > 0) {
        size_t searched = 0;
        for (size_t i = 0; i < size; i++) {
            size_t batchSize = this->computeBatchSizes[i];
            if ((n - searched) >= batchSize) {
                int batchNum = (n - searched) / batchSize;
                for (int j = 0; j < batchNum; j++) {
                    ret = ComputeDistImplBatched(batchSize, queries + searched * this->dim,
                        distances + searched * padNtotal, tableInfo);
                    APPERR_RETURN_IF(ret, ret);
                    searched += batchSize;
                }
            }
        }
    } else {
        ret = ComputeDistImplBatched(n, queries, distances, tableInfo);
        APPERR_RETURN_IF(ret, ret);
    }
    return ret;
}

APP_ERROR IndexILFlatImpl::ComputeDistImplBatched(int n, const float16_t *queries, float *distances,
    std::tuple<unsigned int, const float *> tableInfo)
{
    auto streamPtr = this->resources.getDefaultStream();
    auto stream = streamPtr->GetStream();
    auto &mem = this->resources.getMemoryManager();

    unsigned int tableLen = std::get<0>(tableInfo);
    bool needMapping = tableLen > 0 ? true : false;

    AscendTensor<float16_t, DIMS_2> queryTensor(const_cast<float16_t *>(queries), {n, this->dim});
    AscendTensor<uint32_t, DIMS_1> baseSize(mem, { SIZE_ALIGN }, stream);
    int ntotalPad = utils::divUp(this->ntotal, CUBE_ALIGN) * CUBE_ALIGN;
    AscendTensor<float, DIMS_1> distanceTensor(distances, {n * ntotalPad});
    baseSize[0] = ntotalPad;

    const int dim1 = utils::divUp(this->ntotal, zregionHeight);
    const int dim2 = utils::divUp(this->dim, CUBE_ALIGN);
    AscendTensor<float16_t, DIMS_4> shapedData(this->baseSpace->data(), {dim1, dim2, zregionHeight, CUBE_ALIGN});

    if (needMapping) {
        float *table = const_cast<float *>(std::get<1>(tableInfo));
        AscendTensor<float, DIMS_1> tableTensor(table, { TABLE_LEN });
        ComputeBlockDistWholeBaseWithTable(queryTensor, shapedData, baseSize, tableTensor, distanceTensor, stream);
    } else {
        ComputeBlockDistWholeBase(queryTensor, shapedData, baseSize, distanceTensor, stream);
    }
    auto ret = synchronizeStream(stream);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR,
        "synchronizeStream failed: %i\n", ret);
    return APP_ERR_OK;
}

void IndexILFlatImpl::ComputeBlockDistWholeBase(AscendTensor<float16_t, DIMS_2> &queryTensor,
    AscendTensor<float16_t, DIMS_4> &shapedData, AscendTensor<uint32_t, DIMS_1> &baseSize,
    AscendTensor<float, DIMS_1> &distanceTensor, aclrtStream stream)
{
    AscendOperator *op = nullptr;
    int batchSize = queryTensor.getSize(0);
    if (this->distanceFlatIpOps.find(batchSize) != this->distanceFlatIpOps.end()) {
        op = this->distanceFlatIpOps[batchSize].get();
    }
    ASCEND_THROW_IF_NOT(op);

    std::shared_ptr<std::vector<const aclDataBuffer *>> distOpInput(
        new std::vector<const aclDataBuffer *>(), CommonUtils::AclInputBufferDelete);
    distOpInput->emplace_back(aclCreateDataBuffer(queryTensor.data(), queryTensor.getSizeInBytes()));
    distOpInput->emplace_back(aclCreateDataBuffer(shapedData.data(), this->fakeBaseSizeInBytes));
    distOpInput->emplace_back(aclCreateDataBuffer(baseSize.data(), baseSize.getSizeInBytes()));
    std::shared_ptr<std::vector<aclDataBuffer *>> distOpOutput(
        new std::vector<aclDataBuffer *>(), CommonUtils::AclOutputBufferDelete);
    distOpOutput->emplace_back(aclCreateDataBuffer(
        distanceTensor.data(), (size_t)batchSize * FAKE_HUGE_BASE * sizeof(float)));

    op->exec(*distOpInput, *distOpOutput, stream);
}

void IndexILFlatImpl::ComputeBlockDistWholeBaseWithTable(AscendTensor<float16_t, DIMS_2> &queryTensor,
    AscendTensor<float16_t, DIMS_4> &shapedData, AscendTensor<uint32_t, DIMS_1> &baseSize,
    AscendTensor<float, DIMS_1> &tableTensor, AscendTensor<float, DIMS_1> &distanceTensor, aclrtStream stream)
{
    AscendOperator *op = nullptr;
    int batchSize = queryTensor.getSize(0);
    if (this->distanceFlatIpWithTableOps.find(batchSize) != this->distanceFlatIpWithTableOps.end()) {
        op = this->distanceFlatIpWithTableOps[batchSize].get();
    }
    ASCEND_THROW_IF_NOT(op);

    std::shared_ptr<std::vector<const aclDataBuffer *>> distOpInput(
        new std::vector<const aclDataBuffer *>(), CommonUtils::AclInputBufferDelete);
    distOpInput->emplace_back(aclCreateDataBuffer(queryTensor.data(), queryTensor.getSizeInBytes()));
    distOpInput->emplace_back(aclCreateDataBuffer(shapedData.data(), this->fakeBaseSizeInBytes));
    distOpInput->emplace_back(aclCreateDataBuffer(baseSize.data(), baseSize.getSizeInBytes()));
    distOpInput->emplace_back(aclCreateDataBuffer(tableTensor.data(), tableTensor.getSizeInBytes()));

    std::shared_ptr<std::vector<aclDataBuffer *>> distOpOutput(
        new std::vector<aclDataBuffer *>(), CommonUtils::AclOutputBufferDelete);
    distOpOutput->emplace_back(aclCreateDataBuffer(
        distanceTensor.data(), (size_t)batchSize * FAKE_HUGE_BASE * sizeof(float)));

    op->exec(*distOpInput, *distOpOutput, stream);
}

APP_ERROR IndexILFlatImpl::ComputeDistByThresholdBatched(int n, const float16_t *queries, float *distances,
    std::tuple<unsigned int, const float *> tableInfo, std::tuple<float, int *, idx_t *> filterParams)
{
    AscendRWLock writeLock(&mtx);
    CHECK_LEGAL_OPERATION(isInitialized);
    APP_ERROR ret = APP_ERR_OK;
    size_t size = this->computeBatchSizes.size();
    float threshold = std::get<0>(filterParams);
    int *num = std::get<1>(filterParams);
    idx_t *indice = std::get<2>(filterParams);
    int ntotalPad = utils::divUp(this->ntotal, CUBE_ALIGN) * CUBE_ALIGN;

    if (n > 1 && size > 0) {
        size_t searched = 0;
        for (size_t i = 0; i < size; i++) {
            size_t batchSize = this->computeBatchSizes[i];
            if ((n - searched) >= batchSize) {
                int batchNum = (n - searched) / batchSize;
                for (int j = 0; j < batchNum; j++) {
                    auto batchedFilterParams = std::tuple<float, int *, idx_t *>(threshold, num + searched,
                        indice + searched * ntotalPad);
                    ret = ComputeDistPaged(batchSize, queries + searched * this->dim,
                        distances + searched * ntotalPad, tableInfo, batchedFilterParams);
                    APPERR_RETURN_IF(ret, ret);
                    searched += batchSize;
                }
            }
        }
    } else {
        ret = ComputeDistPaged(n, queries, distances, tableInfo, filterParams);
        APPERR_RETURN_IF(ret, ret);
    }
    return ret;
}

APP_ERROR IndexILFlatImpl::ComputeDistPaged(int n, const float16_t *queries, float *distances,
    std::tuple<unsigned int, const float *> tableInfo, std::tuple<float, int *, idx_t *> filterParams)
{
    unsigned int tableLen = std::get<0>(tableInfo);
    float threshold = std::get<0>(filterParams);
    int *idxNum = std::get<1>(filterParams);
    idx_t *indice = std::get<2>(filterParams);

    auto streamPtr = this->resources.getDefaultStream();
    auto stream = streamPtr->GetStream();
    auto &mem = this->resources.getMemoryManager();
    bool needMapping = tableLen > 0 ? true : false;

    const int dim1 = utils::divUp(this->ntotal, zregionHeight);
    const int dim2 = utils::divUp(this->dim, CUBE_ALIGN);
    AscendTensor<float16_t, DIMS_2> queryTensor(const_cast<float16_t *>(queries), {n, this->dim});
    AscendTensor<float16_t, DIMS_4> shapedData(this->baseSpace->data(), {dim1, dim2, zregionHeight, CUBE_ALIGN});
    AscendTensor<uint32_t, DIMS_1> baseSize(mem, { SIZE_ALIGN }, stream);
    AscendTensor<float, DIMS_2> filteredDist(distances, {n, this->ntotal});
    int ntotalPad = utils::divUp(this->ntotal, CUBE_ALIGN) * CUBE_ALIGN;
    AscendTensor<float, DIMS_1> distanceTensor(mem, {n * ntotalPad}, stream);
    baseSize[0] = utils::divUp(this->ntotal, CUBE_ALIGN) * CUBE_ALIGN;

    // 1. call full compare operators
    if (needMapping) {
        float *table = const_cast<float *>(std::get<1>(tableInfo));
        AscendTensor<float, DIMS_1> tableTensor(table, { TABLE_LEN });
        ComputeBlockDistWholeBaseWithTable(queryTensor, shapedData, baseSize, tableTensor, distanceTensor, stream);
    } else {
        ComputeBlockDistWholeBase(queryTensor, shapedData, baseSize, distanceTensor, stream);
    }
    auto ret = synchronizeStream(stream);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR,
        "synchronizeStream failed: %i\n", ret);

    // 2. call distance filter operator
    AscendTensor<uint32_t, DIMS_1> initIdx(mem, {SIZE_ALIGN}, stream);
    AscendTensor<float, DIMS_1> thresholdTensor(mem, {SIZE_ALIGN}, stream);
    AscendTensor<uint32_t, DIMS_2> filteredIndice(indice, {n, this->ntotal});
    AscendTensor<uint32_t, DIMS_1> filteredCnt(mem, {n * SIZE_ALIGN}, stream);
    initIdx[0] = 0;
    baseSize[0] = this->ntotal;
    thresholdTensor[0] = threshold;
    DistanceFilter(distanceTensor, thresholdTensor, baseSize, initIdx, filteredDist,
        filteredIndice, filteredCnt, stream);
    ret = synchronizeStream(stream);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR,
        "synchronizeStream failed: %i\n", ret);

    // 3. postprocess the filtered data, write result to output
    for (int i = 0; i < n; i++) {
        *(idxNum + i) = filteredCnt[i * SIZE_ALIGN].value();
    }
    return ret;
}

void IndexILFlatImpl::DistanceFilter(AscendTensor<float, DIMS_1> &distanceTensor,
    AscendTensor<float, DIMS_1> &thresholdTensor, AscendTensor<uint32_t, DIMS_1> &baseSize,
    AscendTensor<uint32_t, DIMS_1> &initIdx, AscendTensor<float, DIMS_2> &filteredDist,
    AscendTensor<uint32_t, DIMS_2> &filteredIndice, AscendTensor<uint32_t, DIMS_1> &filteredCnt, aclrtStream stream)
{
    AscendOperator *op = nullptr;
    int batchSize = filteredDist.getSize(0);
    if (this->distanceFilterOps.find(batchSize) != this->distanceFilterOps.end()) {
        op = this->distanceFilterOps[batchSize].get();
    }
    ASCEND_THROW_IF_NOT(op);
    std::shared_ptr<std::vector<const aclDataBuffer *>> distOpInput(
        new std::vector<const aclDataBuffer *>(), CommonUtils::AclInputBufferDelete);
    distOpInput->emplace_back(aclCreateDataBuffer(
        distanceTensor.data(), (size_t)batchSize * FAKE_HUGE_BASE * sizeof(float)));
    distOpInput->emplace_back(aclCreateDataBuffer(thresholdTensor.data(), thresholdTensor.getSizeInBytes()));
    distOpInput->emplace_back(aclCreateDataBuffer(baseSize.data(), baseSize.getSizeInBytes()));
    distOpInput->emplace_back(aclCreateDataBuffer(initIdx.data(), initIdx.getSizeInBytes()));

    std::shared_ptr<std::vector<aclDataBuffer *>> distOpOutput(
        new std::vector<aclDataBuffer *>(), CommonUtils::AclOutputBufferDelete);
    distOpOutput->emplace_back(aclCreateDataBuffer(
        filteredDist.data(), (size_t)batchSize * FAKE_HUGE_BASE * sizeof(float)));
    distOpOutput->emplace_back(aclCreateDataBuffer(
        filteredIndice.data(), (size_t)batchSize * FAKE_HUGE_BASE * sizeof(int)));
    distOpOutput->emplace_back(aclCreateDataBuffer(filteredCnt.data(), filteredCnt.getSizeInBytes()));

    op->exec(*distOpInput, *distOpOutput, stream);
}

void IndexILFlatImpl::ComputeBlockDist(AscendTensor<float16_t, DIMS_2> &queryTensor,
    AscendTensor<uint8_t, DIMS_2> &mask,
    AscendTensor<float16_t, DIMS_4> &shapedData, AscendTensor<uint32_t, DIMS_2> &size,
    AscendTensor<float16_t, DIMS_2> &outDistances, AscendTensor<float16_t, DIMS_2> &maxDistances,
    AscendTensor<uint16_t, DIMS_2> &flag, aclrtStream stream)
{
    AscendOperator *op = nullptr;
    int batchSize = queryTensor.getSize(0);
    if (this->distComputeOps.find(batchSize) != this->distComputeOps.end()) {
        op = this->distComputeOps[batchSize].get();
    }
    ASCEND_THROW_IF_NOT(op);

    std::shared_ptr<std::vector<const aclDataBuffer *>> distOpInput(
        new std::vector<const aclDataBuffer *>(), CommonUtils::AclInputBufferDelete);
    distOpInput->emplace_back(aclCreateDataBuffer(queryTensor.data(), queryTensor.getSizeInBytes()));
    distOpInput->emplace_back(aclCreateDataBuffer(mask.data(), mask.getSizeInBytes()));
    distOpInput->emplace_back(aclCreateDataBuffer(shapedData.data(), shapedData.getSizeInBytes()));
    distOpInput->emplace_back(aclCreateDataBuffer(size.data(), size.getSizeInBytes()));

    std::shared_ptr<std::vector<aclDataBuffer *>> distOpOutput(
        new std::vector<aclDataBuffer *>(), CommonUtils::AclOutputBufferDelete);
    distOpOutput->emplace_back(aclCreateDataBuffer(outDistances.data(), outDistances.getSizeInBytes()));
    distOpOutput->emplace_back(aclCreateDataBuffer(maxDistances.data(), maxDistances.getSizeInBytes()));
    distOpOutput->emplace_back(aclCreateDataBuffer(flag.data(), flag.getSizeInBytes()));

    op->exec(*distOpInput, *distOpOutput, stream);
}

APP_ERROR IndexILFlatImpl::ResetDistCompOp(int numLists)
{
    auto distCompOpReset = [&](std::unique_ptr<AscendOperator> &op, int64_t batch) {
        auto burstLen = (batch >= BIG_BATCH_THRESHOLD) ? BIG_BATCH_BURST_LEN : BURST_LEN;
        // 乘以2，是和算子生成时的shape保持一致
        auto burstOfBlock = (FLAT_BLOCK_SIZE + burstLen - 1) / burstLen * 2;
        std::string opName = (batch >= BIG_BATCH_THRESHOLD) ? "DistanceFlatIPMaxsBatch" : "DistanceFlatIPMaxs";
        AscendOpDesc desc(opName);
        std::vector<int64_t> queryShape({ batch, this->dim });
        std::vector<int64_t> maskShape({ batch, blockMaskSize });
        std::vector<int64_t> coarseCentroidsShape({ utils::divUp(numLists, zregionHeight),
            utils::divUp(this->dim, CUBE_ALIGN), zregionHeight, CUBE_ALIGN });
        std::vector<int64_t> sizeShape({ CORE_NUM, SIZE_ALIGN });
        std::vector<int64_t> distResultShape({ batch, numLists });
        std::vector<int64_t> maxResultShape({ batch, burstOfBlock });
        std::vector<int64_t> flagShape({ CORE_NUM, FLAG_SIZE });

        desc.addInputTensorDesc(ACL_FLOAT16, queryShape.size(), queryShape.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_UINT8, maskShape.size(), maskShape.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_FLOAT16, coarseCentroidsShape.size(), coarseCentroidsShape.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_UINT32, sizeShape.size(), sizeShape.data(), ACL_FORMAT_ND);
        desc.addOutputTensorDesc(ACL_FLOAT16, distResultShape.size(), distResultShape.data(), ACL_FORMAT_ND);
        desc.addOutputTensorDesc(ACL_FLOAT16, maxResultShape.size(), maxResultShape.data(), ACL_FORMAT_ND);
        desc.addOutputTensorDesc(ACL_UINT16, flagShape.size(), flagShape.data(), ACL_FORMAT_ND);

        op.reset();
        op = std::make_unique<AscendOperator>(desc);
        return op->init();
    };

    for (auto batch : this->searchBatchSizes) {
        distComputeOps[batch] = std::unique_ptr<AscendOperator>(nullptr);
        APPERR_RETURN_IF_NOT_LOG(distCompOpReset(distComputeOps[batch], batch),
            APP_ERR_ACL_OP_LOAD_MODEL_FAILED, "ResetDistCompOp init failed");
    }
    return APP_ERR_OK;
}

APP_ERROR IndexILFlatImpl::ComputeDistanceByIdx(int n, const float16_t *queries, const int *num, const idx_t *indices,
    float *distances, unsigned int tableLen, const float *table)
{
    AscendRWLock writeLock(&mtx);
    CHECK_LEGAL_OPERATION(isInitialized);
    APP_ERROR ret = APP_ERR_OK;
    std::tuple<unsigned int, const float *> tableInfo(tableLen, table);
    std::tuple<int, const int *, const idx_t *> idxInfo;
    int maxNum = 0;
    for (size_t i = 0; i < static_cast<size_t>(n); i++) {
        int tmpNum = *(num + i);
        APPERR_RETURN_IF_NOT_LOG(tmpNum >= 0 && tmpNum <= ntotal, APP_ERR_INVALID_PARAM,
            "The num of query idx is invalid, it should be in range [0, ntotal]. ");
        maxNum = std::max(maxNum, tmpNum);
    }
    if (maxNum == 0) {
        return APP_ERR_OK;
    }
    for (size_t i = 0; i < static_cast<size_t>(n); i++) {
        int tmpNum = *(num + i);
        for (int j = 0; j < tmpNum; j++) {
            APPERR_RETURN_IF_NOT_LOG(*(indices + i * maxNum + j) < static_cast<idx_t>(this->ntotal),
                APP_ERR_INVALID_PARAM, "The given indice to compare with should be smaller than ntotal");
        }
    }

    // compute batched
    size_t size = COMPUTE_BY_IDX_BATCHES.size();
    if (n > 1 && size > 0) {
        size_t searched = 0;
        for (size_t i = 0; i < size; i++) {
            size_t batchSize = COMPUTE_BY_IDX_BATCHES[i];
            if ((n - searched) >= batchSize) {
                int batchNum = (n - searched) / batchSize;
                for (int j = 0; j < batchNum; j++) {
                    idxInfo = std::make_tuple(maxNum, num + searched, indices + searched * maxNum);
                    ret = ComputeDistByIdxImpl(batchSize, queries + searched * this->dim, distances + searched * maxNum,
                        idxInfo, tableInfo);
                    APPERR_RETURN_IF(ret, ret);
                    searched += batchSize;
                }
            }
        }
    } else {
        idxInfo = std::make_tuple(maxNum, num, indices);
        ret = ComputeDistByIdxImpl(n, queries, distances, idxInfo, tableInfo);
    }
    return APP_ERR_OK;
}

APP_ERROR IndexILFlatImpl::ComputeDistByIdxImpl(int n, const float16_t *queries, float *distances,
    std::tuple<int, const int *, const idx_t *> idxInfo, std::tuple<unsigned int, const float *> tableInfo)
{
    APP_ERROR ret = APP_ERR_OK;
    auto streamPtr = this->resources.getDefaultStream();
    auto stream = streamPtr->GetStream();
    auto &mem = this->resources.getMemoryManager();
    int maxNum = std::get<0>(idxInfo);
    const idx_t *indice = std::get<2>(idxInfo);
    unsigned int tableLen = std::get<0>(tableInfo);
    const float *table = std::get<1>(tableInfo);

    int idxSliceNum = utils::divUp(maxNum, IDX_BURST_LEN);
    int blockNum = utils::divUp(this->capacity, this->blockSize);

    AscendTensor<float16_t, DIMS_2> queryTensor(const_cast<float16_t *>(queries), { n, this->dim });
    AscendTensor<float, DIMS_2> distanceTensor(distances, { n, maxNum });

    AscendTensor<float, DIMS_3> distResult(mem, { idxSliceNum, n, IDX_BURST_LEN }, stream);
    AscendTensor<uint32_t, DIMS_3> idxTensor(mem, { idxSliceNum, n, IDX_BURST_LEN }, stream);

    // 将索引搬运成大z小z，maxNum按64补齐
    int idxCopyNum = 0;
    for (int i = 0; i < idxSliceNum; i++) {
        for (int j = 0; j < n; j++) {
            idxCopyNum = (i == idxSliceNum - 1) ? (maxNum - i * IDX_BURST_LEN) : IDX_BURST_LEN;
            auto err = memcpy_s(idxTensor[i][j].data(), idxCopyNum * sizeof(uint32_t),
                indice + static_cast<size_t>(j) * maxNum + i * IDX_BURST_LEN, idxCopyNum * sizeof(uint32_t));
            APPERR_RETURN_IF_NOT_FMT(
                err == EOK, APP_ERR_INNER_ERROR, "memcpy error, (i=%d,j=%d). err=%d\n", i, j, err);
        }
    }

    const int dim1 = utils::divUp(this->blockSize, zregionHeight);
    const int dim2 = utils::divUp(this->dim, CUBE_ALIGN);

    // 每次调用算子，都传入全量的底库
    AscendTensor<float16_t, DIMS_4> shaped(
        this->baseSpace->data(), { blockNum * dim1, dim2, zregionHeight, CUBE_ALIGN });
    AscendTensor<uint32_t, DIMS_2> sizeTensorList(mem, {idxSliceNum, SIZE_ALIGN}, stream);
    for (int i = 0; i < idxSliceNum; i++) {
        auto index = idxTensor[i].view();
        auto dist = distResult[i].view();
        auto sizeTensor = sizeTensorList[i].view();
        sizeTensor[0] = (i == idxSliceNum - 1) ? (maxNum - i * IDX_BURST_LEN) : IDX_BURST_LEN;
        if (tableLen > 0) {
            AscendTensor<float, DIMS_1> tableTensor(const_cast<float *>(table), { static_cast<int>(TABLE_LEN) });
            ComputeDistWholeBaseWithTable(queryTensor, sizeTensor, shaped, index, dist, tableTensor, stream);
        } else {
            ComputeDistWholeBase(queryTensor, sizeTensor, shaped, index, dist, stream);
        }
    }
    ret = synchronizeStream(stream);

    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR,
        "synchronizeStream default stream failed, error code: %i\n", ret);

    // 拷贝结果到输出空间
    for (int i = 0; i < idxSliceNum; i++) {
        for (int j = 0; j < n; j++) {
            idxCopyNum = (i == idxSliceNum - 1) ? (maxNum - i * IDX_BURST_LEN) : IDX_BURST_LEN;
            auto err = memcpy_s(distanceTensor[j].data() + i * IDX_BURST_LEN, idxCopyNum * sizeof(float),
                distResult[i][j].data(), idxCopyNum * sizeof(float));
            APPERR_RETURN_IF_NOT_FMT(err == EOK, APP_ERR_INNER_ERROR,
                "memcpy error, i = %d, j = %d. err = %d\n", i, j, err);
        }
    }
    return ret;
}

void IndexILFlatImpl::ComputeDistWholeBase(AscendTensor<float16_t, DIMS_2> &queryTensor,
    AscendTensor<uint32_t, DIMS_1> &sizeTensor, AscendTensor<float16_t, DIMS_4> &shapedData,
    AscendTensor<uint32_t, DIMS_2> &idxTensor, AscendTensor<float, DIMS_2> &distanceTensor, aclrtStream stream)
{
    AscendOperator *op = nullptr;
    int batchSize = queryTensor.getSize(0);
    if (this->distComputeIdxOps.find(batchSize) != this->distComputeIdxOps.end()) {
        op = this->distComputeIdxOps[batchSize].get();
    }
    ASCEND_THROW_IF_NOT(op);

    std::shared_ptr<std::vector<const aclDataBuffer *>> distOpInput(
        new std::vector<const aclDataBuffer *>(), CommonUtils::AclInputBufferDelete);

    // construct fake huge input size for basevectors, FLAT_BLOCK_SIZE * IDX_BURST_LEN, the index is 2
    distOpInput->emplace_back(aclCreateDataBuffer(queryTensor.data(), queryTensor.getSizeInBytes()));
    distOpInput->emplace_back(aclCreateDataBuffer(idxTensor.data(), idxTensor.getSizeInBytes()));
    distOpInput->emplace_back(aclCreateDataBuffer(sizeTensor.data(), sizeTensor.getSizeInBytes()));
    distOpInput->emplace_back(aclCreateDataBuffer(shapedData.data(), fakeBaseSizeInBytes));

    std::shared_ptr<std::vector<aclDataBuffer *>> distOpOutput(
        new std::vector<aclDataBuffer *>(), CommonUtils::AclOutputBufferDelete);
    distOpOutput->emplace_back(aclCreateDataBuffer(distanceTensor.data(), distanceTensor.getSizeInBytes()));

    op->exec(*distOpInput, *distOpOutput, stream);
}

void IndexILFlatImpl::ComputeDistWholeBaseWithTable(AscendTensor<float16_t, DIMS_2> &queryTensor,
    AscendTensor<uint32_t, DIMS_1> &sizeTensor, AscendTensor<float16_t, DIMS_4> &shapedData,
    AscendTensor<uint32_t, DIMS_2> &idxTensor, AscendTensor<float, DIMS_2> &distanceTensor,
    AscendTensor<float, DIMS_1> &tableTensor, aclrtStream stream)
{
    AscendOperator *op = nullptr;
    int batchSize = queryTensor.getSize(0);
    if (this->distComputeIdxWithTableOps.find(batchSize) != this->distComputeIdxWithTableOps.end()) {
        op = this->distComputeIdxWithTableOps[batchSize].get();
    }
    ASCEND_THROW_IF_NOT(op);

    std::shared_ptr<std::vector<const aclDataBuffer *>> distOpInput(
        new std::vector<const aclDataBuffer *>(), CommonUtils::AclInputBufferDelete);

    // construct fake huge input size for basevectors, FLAT_BLOCK_SIZE * IDX_BURST_LEN, the index is 2
    distOpInput->emplace_back(aclCreateDataBuffer(queryTensor.data(), queryTensor.getSizeInBytes()));
    distOpInput->emplace_back(aclCreateDataBuffer(idxTensor.data(), idxTensor.getSizeInBytes()));
    distOpInput->emplace_back(aclCreateDataBuffer(sizeTensor.data(), sizeTensor.getSizeInBytes()));
    distOpInput->emplace_back(aclCreateDataBuffer(shapedData.data(), fakeBaseSizeInBytes));
    distOpInput->emplace_back(aclCreateDataBuffer(tableTensor.data(), tableTensor.getSizeInBytes()));

    std::shared_ptr<std::vector<aclDataBuffer *>> distOpOutput(
        new std::vector<aclDataBuffer *>(), CommonUtils::AclOutputBufferDelete);
    distOpOutput->emplace_back(aclCreateDataBuffer(distanceTensor.data(), distanceTensor.getSizeInBytes()));

    op->exec(*distOpInput, *distOpOutput, stream);
}

APP_ERROR IndexILFlatImpl::TableMapping(int n, float *distances, unsigned int tableLen, const float *table, int topk)
{
    if (tableLen > 0) {
        int numEachQuery = 0;
        numEachQuery = std::min(topk, this->ntotal);
        int tableIndex = 0;
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < numEachQuery; ++j) {
                int offset = static_cast<size_t>(i) * topk + j;
                // index = (cos + 1) / 2 * tableLen, 加上0.5，取整，实现四舍五入
                tableIndex = static_cast<int>((*(distances + offset) + 1) / 2 * tableLen + 0.5);
                APPERR_RETURN_IF_NOT_LOG(
                    tableIndex >= 0 && tableIndex < static_cast<int>(TABLE_LEN), APP_ERR_INVALID_TABLE_INDEX,
                    "Invalid index for table mapping, please ensure the correctness of vector normalization.\n");
                *(distances + offset) = *(table + tableIndex);
            }
        }
    }
    return APP_ERR_OK;
}

APP_ERROR IndexILFlatImpl::ResetDistCompIdxOp()
{
    auto distCompIdxOpReset = [&](std::unique_ptr<AscendOperator> &op, int64_t batch) {
        AscendOpDesc desc("DistanceFlatIpByIdx");
        std::vector<int64_t> queryShape({ batch, this->dim });
        std::vector<int64_t> indexShape({ batch, IDX_BURST_LEN });
        std::vector<int64_t> sizeShape({ SIZE_ALIGN });
        std::vector<int64_t> coarseCentroidsShape(
            { utils::divUp(FAKE_HUGE_BASE, zregionHeight), utils::divUp(this->dim, CUBE_ALIGN),
            zregionHeight, CUBE_ALIGN });
        std::vector<int64_t> distResultShape({ batch, IDX_BURST_LEN });

        desc.addInputTensorDesc(ACL_FLOAT16, queryShape.size(), queryShape.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_UINT32, indexShape.size(), indexShape.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_UINT32, sizeShape.size(), sizeShape.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_FLOAT16, coarseCentroidsShape.size(), coarseCentroidsShape.data(), ACL_FORMAT_ND);
        desc.addOutputTensorDesc(ACL_FLOAT, distResultShape.size(), distResultShape.data(), ACL_FORMAT_ND);

        op.reset();
        op = std::make_unique<AscendOperator>(desc);
        return op->init();
    };

    for (auto batch : COMPUTE_BY_IDX_BATCHES) {
        distComputeIdxOps[batch] = std::unique_ptr<AscendOperator>(nullptr);
        APPERR_RETURN_IF_NOT_LOG(distCompIdxOpReset(distComputeIdxOps[batch], batch),
            APP_ERR_ACL_OP_LOAD_MODEL_FAILED, "distCompIdxOpReset init failed");
    }
    return APP_ERR_OK;
}

APP_ERROR IndexILFlatImpl::ResetDistCompIdxWithTableOp()
{
    auto distCompIdxWithTableOpReset = [&](std::unique_ptr<AscendOperator> &op, int64_t batch) {
        AscendOpDesc desc("DistanceFlatIpByIdxWithTable");
        std::vector<int64_t> queryShape({ batch, this->dim });
        std::vector<int64_t> indexShape({ batch, IDX_BURST_LEN });
        std::vector<int64_t> sizeShape({ SIZE_ALIGN });
        std::vector<int64_t> coarseCentroidsShape(
            { utils::divUp(FAKE_HUGE_BASE, zregionHeight), utils::divUp(this->dim, CUBE_ALIGN),
            zregionHeight, (int64_t)CUBE_ALIGN });
        std::vector<int64_t> tableShape({ TABLE_LEN });
        std::vector<int64_t> distResultShape({ batch, IDX_BURST_LEN });

        desc.addInputTensorDesc(ACL_FLOAT16, queryShape.size(), queryShape.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_UINT32, indexShape.size(), indexShape.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_UINT32, sizeShape.size(), sizeShape.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_FLOAT16, coarseCentroidsShape.size(), coarseCentroidsShape.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_FLOAT, tableShape.size(), tableShape.data(), ACL_FORMAT_ND);
        desc.addOutputTensorDesc(ACL_FLOAT, distResultShape.size(), distResultShape.data(), ACL_FORMAT_ND);

        op.reset();
        op = std::make_unique<AscendOperator>(desc);
        return op->init();
    };

    for (auto batch : COMPUTE_BY_IDX_BATCHES) {
        distComputeIdxWithTableOps[batch] = std::unique_ptr<AscendOperator>(nullptr);
        APPERR_RETURN_IF_NOT_LOG(distCompIdxWithTableOpReset(distComputeIdxWithTableOps[batch], batch),
            APP_ERR_ACL_OP_LOAD_MODEL_FAILED, "distCompIdxWithTableOpReset init failed");
    }
    return APP_ERR_OK;
}

APP_ERROR IndexILFlatImpl::ResetDistanceFilterOp()
{
    auto distanceFilterOpReset = [](std::unique_ptr<AscendOperator> &op, int64_t batch) {
        AscendOpDesc desc("DistanceFilter");
        std::vector<int64_t> distanceShape({ batch, FAKE_HUGE_BASE });
        std::vector<int64_t> thresholdShape({ SIZE_ALIGN });
        std::vector<int64_t> distanceNumShape({ SIZE_ALIGN });
        std::vector<int64_t> baseIdxShape({ SIZE_ALIGN });
        std::vector<int64_t> filteredDistShape({ batch, FAKE_HUGE_BASE });
        std::vector<int64_t> filteredIndiceShape({ batch, FAKE_HUGE_BASE });
        std::vector<int64_t> filteredDistCnt({ batch * SIZE_ALIGN });

        desc.addInputTensorDesc(ACL_FLOAT, distanceShape.size(), distanceShape.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_FLOAT, thresholdShape.size(), thresholdShape.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_UINT32, distanceNumShape.size(), distanceNumShape.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_UINT32, baseIdxShape.size(), baseIdxShape.data(), ACL_FORMAT_ND);

        desc.addOutputTensorDesc(ACL_FLOAT, filteredDistShape.size(), filteredDistShape.data(), ACL_FORMAT_ND);
        desc.addOutputTensorDesc(ACL_INT32, filteredIndiceShape.size(), filteredIndiceShape.data(), ACL_FORMAT_ND);
        desc.addOutputTensorDesc(ACL_UINT32, filteredDistCnt.size(), filteredDistCnt.data(), ACL_FORMAT_ND);

        op.reset();
        op = std::make_unique<AscendOperator>(desc);
        return op->init();
    };

    for (auto batch : this->computeBatchSizes) {
        distanceFilterOps[batch] = std::unique_ptr<AscendOperator>(nullptr);
        APPERR_RETURN_IF_NOT_LOG(distanceFilterOpReset(distanceFilterOps[batch], batch),
            APP_ERR_ACL_OP_LOAD_MODEL_FAILED, "distanceFilterOpReset init failed");
    }
    return APP_ERR_OK;
}

APP_ERROR IndexILFlatImpl::ResetDistanceFlatIpWithTableOp()
{
    auto distanceFlatIpWithTableOpReset = [&](std::unique_ptr<AscendOperator> &op, int64_t batch) {
        AscendOpDesc desc("DistanceFlatIpWithTable");
        std::vector<int64_t> queryShape({ batch, this->dim });
        std::vector<int64_t> coarseCentroidsShape({ utils::divUp(FAKE_HUGE_BASE, zregionHeight),
            utils::divUp(this->dim, CUBE_ALIGN), zregionHeight, (int64_t)CUBE_ALIGN });
        std::vector<int64_t> sizeShape({ SIZE_ALIGN });
        std::vector<int64_t> distResultShape({ batch * utils::divUp(FAKE_HUGE_BASE, zregionHeight) * zregionHeight });
        std::vector<int64_t> tableShape({TABLE_LEN});

        desc.addInputTensorDesc(ACL_FLOAT16, queryShape.size(), queryShape.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_FLOAT16, coarseCentroidsShape.size(), coarseCentroidsShape.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_UINT32, sizeShape.size(), sizeShape.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_FLOAT, tableShape.size(), tableShape.data(), ACL_FORMAT_ND);
        desc.addOutputTensorDesc(ACL_FLOAT, distResultShape.size(), distResultShape.data(), ACL_FORMAT_ND);
        op.reset();
        op = std::make_unique<AscendOperator>(desc);
        return op->init();
    };

    for (auto batch : this->computeBatchSizes) {
        distanceFlatIpWithTableOps[batch] = std::unique_ptr<AscendOperator>(nullptr);
        APPERR_RETURN_IF_NOT_LOG(distanceFlatIpWithTableOpReset(distanceFlatIpWithTableOps[batch], batch),
            APP_ERR_ACL_OP_LOAD_MODEL_FAILED, "distanceFlatIpWithTableOpReset init failed");
    }
    return APP_ERR_OK;
}

APP_ERROR IndexILFlatImpl::ResetDistanceFlatIpOp()
{
    auto distanceFlatIpOpReset = [&](std::unique_ptr<AscendOperator> &op, int64_t batch) {
        AscendOpDesc desc("DistanceFlatIp");
        std::vector<int64_t> queryShape({ batch, this->dim });
        std::vector<int64_t> coarseCentroidsShape({ utils::divUp(FAKE_HUGE_BASE, zregionHeight),
            utils::divUp(this->dim, CUBE_ALIGN), zregionHeight, (int64_t)CUBE_ALIGN });
        std::vector<int64_t> sizeShape({ SIZE_ALIGN });
        std::vector<int64_t> distResultShape({ batch * utils::divUp(FAKE_HUGE_BASE, zregionHeight) * zregionHeight });

        desc.addInputTensorDesc(ACL_FLOAT16, queryShape.size(), queryShape.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_FLOAT16, coarseCentroidsShape.size(), coarseCentroidsShape.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_UINT32, sizeShape.size(), sizeShape.data(), ACL_FORMAT_ND);
        desc.addOutputTensorDesc(ACL_FLOAT, distResultShape.size(), distResultShape.data(), ACL_FORMAT_ND);
        op.reset();
        op = std::make_unique<AscendOperator>(desc);
        return op->init();
    };

    for (auto batch : this->computeBatchSizes) {
        distanceFlatIpOps[batch] = std::unique_ptr<AscendOperator>(nullptr);
        APPERR_RETURN_IF_NOT_LOG(distanceFlatIpOpReset(distanceFlatIpOps[batch], batch),
            APP_ERR_ACL_OP_LOAD_MODEL_FAILED, "distanceFlatIpOpReset init failed");
    }
    return APP_ERR_OK;
}

IndexIL::IndexIL() {}

IndexIL::~IndexIL() {}

IndexILFlat::IndexILFlat()
{
    this->pIndexILFlatImpl = nullptr;
}

IndexILFlat::~IndexILFlat()
{
    this->Finalize();
}

int IndexILFlat::GetNTotal() const
{
    if (this->pIndexILFlatImpl != nullptr) {
        return this->pIndexILFlatImpl->GetNTotal();
    }
    APP_LOG_ERROR("index must be init first!!!\n");
    return UNINITIALIZE_NTOTAL;
}

APP_ERROR IndexILFlat::SetNTotal(int n)
{
    APPERR_RETURN_IF_NOT_FMT(n >= 0 && n <= this->capacity, APP_ERR_INVALID_PARAM,
        "The ntotal should be in range [0, %d].\n", this->capacity);
    APPERR_RETURN_IF_NOT_LOG(
        this->pIndexILFlatImpl != nullptr, APP_ERR_INDEX_NOT_INIT, "index must be init first!!!\n");
    this->pIndexILFlatImpl->SetNTotal(n);
    return APP_ERR_OK;
}

APP_ERROR IndexILFlat::AddFeatures(int n, const float16_t *features, const idx_t *indices)
{
    APPERR_RETURN_IF_NOT_LOG(
        this->pIndexILFlatImpl != nullptr, APP_ERR_INDEX_NOT_INIT, "index must be init first!!!\n");
    CHECK_FEATURE_PARAMS(n, features, indices);
    return this->pIndexILFlatImpl->Add(n, features, indices);
}

APP_ERROR IndexILFlat::RemoveFeatures(int n, const idx_t *indices)
{
    APPERR_RETURN_IF(n == 0, APP_ERR_OK);
    APPERR_RETURN_IF_NOT_FMT(n >= 0 && n <= this->capacity, APP_ERR_INVALID_PARAM,
        "the number n should be greater than 0 and smaller than capacity %d", this->capacity);
    APPERR_RETURN_IF_NOT_LOG(indices, APP_ERR_INVALID_PARAM, "indices can not be nullptr.\n");
    APPERR_RETURN_IF_NOT_LOG(
        this->pIndexILFlatImpl != nullptr, APP_ERR_INDEX_NOT_INIT, "index must be init first!!!\n");
    return this->pIndexILFlatImpl->Remove(n, indices);
}

APP_ERROR IndexILFlat::GetFeatures(int n, float16_t *features, const idx_t *indices)
{
    APPERR_RETURN_IF_NOT_LOG(
        this->pIndexILFlatImpl != nullptr, APP_ERR_INDEX_NOT_INIT, "index must be init first!!!\n");
    CHECK_FEATURE_PARAMS(n, features, indices);
    return this->pIndexILFlatImpl->Get(n, features, indices);
}

APP_ERROR IndexILFlat::Init(int dim, int capacity, AscendMetricType metricType, int64_t resourceSize)
{
    AscendRWLock glock(&AscendGlobalLock::GetInstance(0));
    APPERR_RETURN_IF_NOT_FMT(capacity > 0 && capacity <= MAX_CAP,
        APP_ERR_INVALID_PARAM, "Given capacity should be in value range: (0, %d]. ", MAX_CAP);
    APPERR_RETURN_IF_NOT_LOG(
        (size_t)dim * sizeof(float16_t) * (size_t)capacity <= MAX_BASE_SPACE, APP_ERR_INVALID_PARAM,
        "The capacity exceed memory allocation limit, please refer to the manuals to set correct capacity. ");
    APPERR_RETURN_IF_NOT_LOG(metricType == AscendMetricType::ASCEND_METRIC_INNER_PRODUCT,
        APP_ERR_INVALID_PARAM, "Unsupported metric type. ");
    APPERR_RETURN_IF_NOT_LOG(std::find(DIMS.begin(), DIMS.end(), dim) != DIMS.end(), APP_ERR_INVALID_PARAM,
        "Illegal dim, given dim should be in {32, 64, 128, 256, 384, 512, 1024}.");
    APPERR_RETURN_IF_NOT_FMT(resourceSize == -1 || (resourceSize >= MIN_RESOURCE && resourceSize <= MAX_RESOURCE),
        APP_ERR_INVALID_PARAM, "resourceSize(%ld) should be -1 or in range [%d Byte, %ld Byte]!",
        resourceSize, MIN_RESOURCE, MAX_RESOURCE);
    APPERR_RETURN_IF_NOT_LOG(pIndexILFlatImpl == nullptr, APP_ERR_ILLEGAL_OPERATION,
        "Index is already initialized, mutiple initialization is not allowed. ");
    this->dim = dim;
    this->capacity = capacity;
    this->metricType = metricType;
    this->pIndexILFlatImpl = new (std::nothrow)IndexILFlatImpl(dim, capacity, resourceSize);
    APPERR_RETURN_IF_NOT_LOG(this->pIndexILFlatImpl, APP_ERR_INNER_ERROR, "Inner error, failed to create object. ");
    APP_ERROR ret = this->pIndexILFlatImpl->Init();
    if (ret) {
        this->Finalize();
    }
    return ret;
}

APP_ERROR IndexILFlat::Finalize()
{
    if (this->pIndexILFlatImpl != nullptr) {
        this->pIndexILFlatImpl->Finalize();
        delete this->pIndexILFlatImpl;
        this->pIndexILFlatImpl = nullptr;
    }

    return APP_ERR_OK;
}

APP_ERROR IndexILFlat::ComputeDistance(int n, const float16_t *queries, float *distances, unsigned int tableLen,
    const float *table)
{
    AscendRWLock glock(&AscendGlobalLock::GetInstance(0));
    CHECK_DISTANCE_COMPUTE_PARAMS(n, queries, distances);
    CHECK_TABLE_PARAMS(tableLen, table);
    std::tuple<unsigned int, const float *> tableInfo(tableLen, table);
    APPERR_RETURN_IF_NOT_LOG(
        this->pIndexILFlatImpl != nullptr, APP_ERR_INDEX_NOT_INIT, "index must be init first!!!\n");

    return this->pIndexILFlatImpl->ComputeDistanceBatched(n, queries, distances, tableInfo);
}

APP_ERROR IndexILFlat::ComputeDistanceByThreshold(int n, const float16_t *queries, float threshold, int *num,
    idx_t *indices, float *distances, unsigned int tableLen, const float *table)
{
    AscendRWLock glock(&AscendGlobalLock::GetInstance(0));
    CHECK_DISTANCE_COMPUTE_PARAMS(n, queries, distances);
    CHECK_TABLE_PARAMS(tableLen, table);
    APPERR_RETURN_IF_NOT_LOG(num, APP_ERR_INVALID_PARAM, "num can not be nullptr.");
    APPERR_RETURN_IF_NOT_LOG(indices, APP_ERR_INVALID_PARAM, "indices can not be nullptr.");

    std::tuple<float, int *, idx_t *> filterParams(threshold, num, indices);
    std::tuple<unsigned int, const float *> tableInfo(tableLen, table);
    APPERR_RETURN_IF_NOT_LOG(
        this->pIndexILFlatImpl != nullptr, APP_ERR_INDEX_NOT_INIT, "index must be init first!!!\n");

    return this->pIndexILFlatImpl->ComputeDistByThresholdBatched(n, queries, distances, tableInfo, filterParams);
}

APP_ERROR IndexILFlat::Search(int n, const float16_t *queries, int topk, idx_t *indices, float *distances,
    unsigned int tableLen, const float *table)
{
    CHECK_DISTANCE_COMPUTE_PARAMS(n, queries, distances);
    CHECK_TABLE_PARAMS(tableLen, table);
    CHECK_TOPK(topk);
    APPERR_RETURN_IF_NOT_LOG(indices, APP_ERR_INVALID_PARAM, "indices can not be nullptr.");
    APPERR_RETURN_IF_NOT_LOG(
        this->pIndexILFlatImpl != nullptr, APP_ERR_INDEX_NOT_INIT, "index must be init first!!!\n");

    return this->pIndexILFlatImpl->Search(n, queries, topk, indices, distances, tableLen, table);
}

APP_ERROR IndexILFlat::SearchByThreshold(int n, const float16_t *queries, float threshold, int topk, int *num,
    idx_t *indices, float *distances, unsigned int tableLen, const float *table)
{
    CHECK_DISTANCE_COMPUTE_PARAMS(n, queries, distances);
    CHECK_TABLE_PARAMS(tableLen, table);
    CHECK_TOPK(topk);
    APPERR_RETURN_IF_NOT_LOG(num, APP_ERR_INVALID_PARAM, "num can not be nullptr.");
    APPERR_RETURN_IF_NOT_LOG(indices, APP_ERR_INVALID_PARAM, "indices can not be nullptr.");
    APPERR_RETURN_IF_NOT_LOG(
        this->pIndexILFlatImpl != nullptr, APP_ERR_INDEX_NOT_INIT, "index must be init first!!!\n");

    return this->pIndexILFlatImpl->SearchByThreshold(n, queries, threshold, topk, num, indices, distances, tableLen,
        table);
}

APP_ERROR IndexILFlat::ComputeDistanceByIdx(int n, const float16_t *queries, const int *num, const idx_t *indices,
    float *distances, unsigned int tableLen, const float *table)
{
    AscendRWLock glock(&AscendGlobalLock::GetInstance(0));
    CHECK_DISTANCE_COMPUTE_PARAMS(n, queries, distances);
    CHECK_TABLE_PARAMS(tableLen, table);
    APPERR_RETURN_IF_NOT_LOG(indices, APP_ERR_INVALID_PARAM, "indices can not be nullptr.");
    APPERR_RETURN_IF_NOT_LOG(num, APP_ERR_INVALID_PARAM, "num can not be nullptr.");
    APPERR_RETURN_IF_NOT_LOG(
        this->pIndexILFlatImpl != nullptr, APP_ERR_INDEX_NOT_INIT, "index must be init first!!!\n");

    return this->pIndexILFlatImpl->ComputeDistanceByIdx(n, queries, num, indices, distances, tableLen, table);
}
}
