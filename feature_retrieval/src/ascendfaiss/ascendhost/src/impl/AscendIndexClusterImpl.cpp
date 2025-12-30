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


#include "ascendhost/include/impl/AscendIndexClusterImpl.h"

#include <faiss/impl/FaissAssert.h>
#include "threadpool/AscendThreadPool.h"
#include "ascenddaemon/utils/StaticUtils.h"
#include "ascenddaemon/utils/AscendUtils.h"
#include "ascenddaemon/utils/AscendRWLock.h"
#include "ascenddaemon/utils/Limits.h"
#include "common/utils/OpLauncher.h"
#include "ops/cpukernel/impl/utils/kernel_shared_def.h"
#include "ascend/utils/fp16.h"
using namespace ascend;

namespace faiss {
namespace ascend {
std::vector<int> DIMS = {32, 64, 128, 256, 384, 512};
std::vector<int> SEARCH_BATCHES = {128, 64, 48, 32, 30, 18, 16, 8, 6, 4, 2, 1};
std::vector<int> COMPUTE_BATCHES = {64, 48, 32, 30, 18, 16, 8, 6, 4, 2, 1};
std::vector<int> COMPUTE_BY_IDX_BATCHES = {256, 128, 64, 48, 32, 16, 8, 6, 4, 2, 1};
constexpr size_t KB = 1024;
constexpr size_t RETAIN_SIZE = 2048;
constexpr size_t UNIT_PAGE_SIZE = 64;
constexpr size_t ADD_PAGE_SIZE = UNIT_PAGE_SIZE * KB * KB - RETAIN_SIZE;
const int ZREGION_HEIGHT = 2;
const int IDX_BURST_LEN = 64;
const int TABLE_LEN = 10048; // mapping table with redundancy of 48
const int FLAT_BLOCK_SIZE = 16384 * 16;
const int FLAT_COMPUTE_PAGE = FLAT_BLOCK_SIZE * 16;
const int BIG_BATCH_BURST_LEN = 32;
const int BIG_BATCH_THRESHOLD = 64;
const int FLAG_SIZE = 16;
const int MAX_TOPK = 1024;
const int ACTUAL_TABLE_LEN = 10000;
const int CORE_NUM = faiss::ascend::SocUtils::GetInstance().GetCoreNum();
AscendIndexClusterImpl::AscendIndexClusterImpl(int dim, int capacity, int deviceId, int64_t resourceSize)
    : dim(dim), capacity(capacity), deviceId(deviceId), resourceSize(resourceSize), isInitialized(false) {}

AscendIndexClusterImpl::~AscendIndexClusterImpl() {}

APP_ERROR AscendIndexClusterImpl::Init()
{
    APP_LOG_INFO("AscendIndexClusterImpl Init operation started.\n");
    APPERR_RETURN_IF_NOT_FMT(std::find(DIMS.begin(), DIMS.end(), this->dim) != DIMS.end(),
        APP_ERR_INVALID_PARAM,
        "Illegal dim(%d), given dim should be in {32, 64, 128, 256, 384, 512}.", this->dim);

    APPERR_RETURN_IF_NOT_FMT(static_cast<size_t>(capacity) <= GetMaxCapacity(),
        APP_ERR_INVALID_PARAM,
        "Illegal capacity for dim %d, the upper boundary of capacity is: %zu, ", dim, GetMaxCapacity());

    auto ret = aclrtSetDevice(deviceId);
    APP_LOG_INFO("AscendIndexClusterImpl Init on device(%d).\n", deviceId);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_ACL_SET_DEVICE_FAILED, "failed to set device(%d)", ret);
    this->ntotal = 0;
    this->pageSize = FLAT_COMPUTE_PAGE;
    this->computeBatchSizes = COMPUTE_BATCHES;
    this->blockSize = CLUSTER_BLOCK_SIZE;
    this->fakeBaseSizeInBytes =
        static_cast<size_t>(FAKE_HUGE_BASE) * static_cast<size_t>(this->dim) * sizeof(float16_t);
    this->blockMaskSize = utils::divUp(this->blockSize, SIZE_ALIGN);
    pResources = CREATE_UNIQUE_PTR(AscendResourcesProxy);
    pResources->setTempMemory(resourceSize);
    pResources->initialize();

    ret = ResetDistanceFilterOp();
    APPERR_RETURN_IF_NOT_LOG(ret == APP_ERR_OK, APP_ERR_ACL_OP_LOAD_MODEL_FAILED, "op init faild !!!");
    ret = ResetDistanceFlatIpByIdx2Op();
    APPERR_RETURN_IF_NOT_LOG(ret == APP_ERR_OK, APP_ERR_ACL_OP_LOAD_MODEL_FAILED, "op init faild !!!");
    ret = ResetDistCompIdxWithTableOp();
    APPERR_RETURN_IF_NOT_LOG(ret == APP_ERR_OK, APP_ERR_ACL_OP_LOAD_MODEL_FAILED, "op init faild !!!");
    ret = ResetDistCompIdxOp();
    APPERR_RETURN_IF_NOT_LOG(ret == APP_ERR_OK, APP_ERR_ACL_OP_LOAD_MODEL_FAILED, "op init faild !!!");
    ret = ResetTopkCompOp();
    APPERR_RETURN_IF_NOT_LOG(ret == APP_ERR_OK, APP_ERR_ACL_OP_LOAD_MODEL_FAILED, "op init faild !!!");
    ret = ResetDistCompOp(FLAT_BLOCK_SIZE);
    APPERR_RETURN_IF_NOT_LOG(ret == APP_ERR_OK, APP_ERR_ACL_OP_LOAD_MODEL_FAILED, "op init faild !!!");

    size_t blockAlign = static_cast<size_t>(utils::divUp(capacity, this->blockSize));
    size_t blockSizeAlign = static_cast<size_t>(utils::divUp(this->blockSize, CUBE_ALIGN));
    int dimAlign = utils::divUp(this->dim, CUBE_ALIGN);
    try {
        auto paddedSize = blockAlign * blockSizeAlign * static_cast<size_t>(dimAlign * CUBE_ALIGN * CUBE_ALIGN);
        this->baseSpace = CREATE_UNIQUE_PTR(DeviceVector<float16_t>, MemorySpace::DEVICE_HUGEPAGE);
        this->baseSpace->resize(paddedSize, true);
    } catch (std::exception &e) {
        APPERR_RETURN_IF_NOT_FMT(
            false, APP_ERR_ACL_BAD_ALLOC, "Malloc device base memory failed. , error msg[%s]", e.what());
    }
    isInitialized = true;

    APP_LOG_INFO("AscendIndexClusterImpl Init operation end.\n");
    return APP_ERR_OK;
}

void AscendIndexClusterImpl::Finalize()
{
    if (isInitialized) {
        baseSpace->clear();
        isInitialized = false;
    }
}

size_t AscendIndexClusterImpl::GetMaxCapacity() const
{
    size_t vectorSize = static_cast<size_t>(dim) * sizeof(float16_t);
    return MAX_BASE_SPACE / vectorSize;
}

APP_ERROR AscendIndexClusterImpl::Add(int num, const float *featuresFloat, const uint32_t *indices)
{
    APPERR_RETURN_IF_NOT_LOG(
        (isInitialized), APP_ERR_INVALID_PARAM, "Illegal operation, please initialize the index first. ");
    auto ret = aclrtSetDevice(deviceId);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_ACL_SET_DEVICE_FAILED, "failed to set device(%d)", ret);

    size_t tileSize = ADD_PAGE_SIZE / (static_cast<size_t>(this->dim) * sizeof(float16_t));
    tileSize = utils::roundUp(tileSize, CUBE_ALIGN);

    int offset = 0;
    while (offset < num) {
        int copyNum = std::min((num - offset), static_cast<int>(tileSize));
        if (copyNum % CUBE_ALIGN != 0) {
            int copyNumPadding = utils::roundUp(copyNum, CUBE_ALIGN);
            std::vector<float> featuresPadding(copyNumPadding * this->dim, 0.0);

            auto ret = memcpy_s(featuresPadding.data(),
                featuresPadding.size() * sizeof(float),
                featuresFloat + static_cast<uint64_t>(offset) * static_cast<uint64_t>(this->dim),
                copyNum * this->dim * sizeof(float));
            APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "failed to memcpy_s(%d)", ret);

            ret = AddPage(copyNumPadding, featuresPadding.data(), indices[offset]);
            APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "failed to AddPage(%d)", ret);
            offset += copyNumPadding;
        } else {
            auto ret = AddPage(copyNum,
                featuresFloat + static_cast<uint64_t>(offset) * static_cast<uint64_t>(this->dim),
                indices[offset]);
            APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "failed to AddPage(%d)", ret);
            offset += copyNum;
        }
    }
    return APP_ERR_OK;
}

APP_ERROR AscendIndexClusterImpl::Add(int n, const uint16_t *featuresFloat, const int64_t *indices)
{
    APPERR_RETURN_IF_NOT_LOG(
        (isInitialized), APP_ERR_INVALID_PARAM, "Illegal operation, please initialize the index first. ");
    auto ret = aclrtSetDevice(deviceId);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_ACL_SET_DEVICE_FAILED, "failed to set device(%d)", ret);

    size_t tileSize = ADD_PAGE_SIZE / (static_cast<size_t>(this->dim) * sizeof(float16_t));
    tileSize = utils::roundUp(tileSize, CUBE_ALIGN);

    int offset = 0;
    while (offset < n) {
        int copyNum = std::min((n - offset), static_cast<int>(tileSize));
        auto ret = AddPage(copyNum,
            featuresFloat + static_cast<uint64_t>(offset) * static_cast<uint64_t>(this->dim),
            indices[offset]);
        APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "failed to AddPage(%d)", ret);
        offset += copyNum;
    }
    return APP_ERR_OK;
}
APP_ERROR AscendIndexClusterImpl::AddPage(int num, const uint16_t *featuresFloat, int64_t indice)
{
    APP_LOG_INFO("AscendIndexClusterImpl Add operation start.\n");

    std::string opName = "TransdataShaped";
    auto &mem = pResources->getMemoryManager();
    auto streamPtr = pResources->getDefaultStream();
    auto stream = streamPtr->GetStream();

    AscendTensor<float16_t, DIMS_2> baseData(mem, {num, this->dim}, stream);
    auto ret = aclrtMemcpy(baseData.data(),
        baseData.getSizeInBytes(), featuresFloat,
        static_cast<uint64_t>(num) * static_cast<uint64_t>(this->dim) * sizeof(uint16_t),
        ACL_MEMCPY_HOST_TO_DEVICE);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "copy feature to device error(%d)", ret);

    int dimAlign = utils::divUp(this->dim, CUBE_ALIGN);
    int blockAlign = utils::divUp(capacity, this->blockSize);
    int blockSizeAlign = utils::divUp(this->blockSize, ZREGION_HEIGHT);
    int64_t paddedSize = blockAlign * blockSizeAlign * dimAlign * CUBE_ALIGN * ZREGION_HEIGHT;
    APPERR_RETURN_IF_NOT_LOG(paddedSize != 0, APP_ERR_INNER_ERROR, "paddedSize should not be zero.");
    int offsetInBlock = indice % paddedSize;

    AscendTensor<int64_t, DIMS_1> attr(mem, {aicpu::TRANSDATA_SHAPED_ATTR_IDX_COUNT}, stream);
    AscendTensor<float16_t, DIMS_2> src(baseData.data(), {num, this->dim});
    AscendTensor<float16_t, DIMS_4> dst(this->baseSpace->data(),
        {utils::divUp(capacity, ZREGION_HEIGHT), utils::divUp(this->dim, CUBE_ALIGN),
         ZREGION_HEIGHT, CUBE_ALIGN});
    attr[aicpu::TRANSDATA_SHAPED_ATTR_NTOTAL_IDX] = offsetInBlock;

    LaunchOpTwoInOneOut<float16_t, DIMS_2, ACL_FLOAT16, int64_t, DIMS_1, ACL_INT64, float16_t, DIMS_4, ACL_FLOAT16>(
        opName, stream, src, attr, dst);

    ret = synchronizeStream(stream);
    APPERR_RETURN_IF_NOT_FMT(
        ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "synchronizeStream addVector stream failed: %i\n", ret);
    this->ntotal = std::max(this->ntotal, (static_cast<int>(indice) + num));
    APP_LOG_INFO("AscendIndexClusterImpl Add operation end.\n");
    return APP_ERR_OK;
}

APP_ERROR AscendIndexClusterImpl::AddPage(int num, const float *featuresFloat, uint32_t indice)
{
    APP_LOG_INFO("AscendIndexClusterImpl Add operation start.\n");

    uint64_t featureSize = static_cast<uint64_t>(num) * static_cast<uint64_t>(this->dim);
    std::vector<float16_t> features(featureSize);
    std::transform(
        featuresFloat, featuresFloat + featureSize, features.data(), [](float temp) { return fp16(temp).data; });

    std::string opName = "TransdataShaped";
    auto &mem = pResources->getMemoryManager();
    auto streamPtr = pResources->getDefaultStream();
    auto stream = streamPtr->GetStream();

    AscendTensor<float16_t, DIMS_2> baseData(mem, {num, this->dim}, stream);
    auto ret = aclrtMemcpy(baseData.data(),
        baseData.getSizeInBytes(),
        features.data(),
        features.size() * sizeof(float16_t),
        ACL_MEMCPY_HOST_TO_DEVICE);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "copy feature to device error(%d)", ret);

    size_t dimAlign = static_cast<size_t>(utils::divUp(this->dim, CUBE_ALIGN));
    size_t blockAlign = static_cast<size_t>(utils::divUp(capacity, this->blockSize));
    size_t blockSizeAlign = static_cast<size_t>(utils::divUp(this->blockSize, CUBE_ALIGN));
    auto paddedSize = blockAlign * blockSizeAlign * static_cast<size_t>(dimAlign * CUBE_ALIGN * CUBE_ALIGN);
    APPERR_RETURN_IF_NOT_LOG(paddedSize != 0, APP_ERR_INNER_ERROR, "paddedSize should not be zero.");
    size_t offsetInBlock = indice % paddedSize;

    AscendTensor<int64_t, DIMS_1> attr(mem, {aicpu::TRANSDATA_SHAPED_ATTR_IDX_COUNT}, stream);
    AscendTensor<float16_t, DIMS_2> src(baseData.data(), {num, this->dim});
    AscendTensor<float16_t, DIMS_4> dst(this->baseSpace->data(),
        {utils::divUp(capacity, CUBE_ALIGN),
            utils::divUp(this->dim, CUBE_ALIGN),
            CUBE_ALIGN,
            CUBE_ALIGN});
    attr[aicpu::TRANSDATA_SHAPED_ATTR_NTOTAL_IDX] = offsetInBlock;

    LaunchOpTwoInOneOut<float16_t, DIMS_2, ACL_FLOAT16, int64_t, DIMS_1, ACL_INT64, float16_t, DIMS_4, ACL_FLOAT16>(
        opName, stream, src, attr, dst);

    ret = synchronizeStream(stream);
    APPERR_RETURN_IF_NOT_FMT(
        ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "synchronizeStream addVector stream failed: %i\n", ret);
    this->ntotal = std::max(this->ntotal, (static_cast<int>(indice) + num));
    APP_LOG_INFO("AscendIndexClusterImpl Add operation end.\n");
    return APP_ERR_OK;
}

APP_ERROR AscendIndexClusterImpl::ComputeDistanceByThreshold(const std::vector<uint32_t> &queryIdxArr,
    uint32_t codeStartIdx, uint32_t codeNum, float threshold, bool,
    std::vector<std::vector<float>> &resDistArr, std::vector<std::vector<uint32_t>> &resIdxArr)
{
    auto ret = aclrtSetDevice(deviceId);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_ACL_SET_DEVICE_FAILED, "failed to set device(%d)", ret);
    size_t queryNum = queryIdxArr.size();
    APPERR_RETURN_IF_NOT_FMT((queryNum > 0 && queryNum <= static_cast<uint32_t>(this->ntotal)),
        APP_ERR_INVALID_PARAM,
        "queryNum(%zu) must > 0 && <= ntotal(%d)!", queryNum, this->ntotal);
    APPERR_RETURN_IF_NOT_FMT((codeNum > 0 && codeNum <= static_cast<uint32_t>(this->ntotal)),
        APP_ERR_INVALID_PARAM,
        "codeNum(%u) must > 0 && <= ntotal(%d)!", codeNum, this->ntotal);
    APPERR_RETURN_IF_NOT_FMT((codeStartIdx < static_cast<uint32_t>(this->ntotal)),
        APP_ERR_INVALID_PARAM,
        "codeStartIdx(%u) must >= 0 && < ntotal(%d)!", codeStartIdx, this->ntotal);
    APPERR_RETURN_IF_NOT_LOG(
        (isInitialized), APP_ERR_INVALID_PARAM, "Illegal operation, please initialize the index first. ");
    APPERR_RETURN_IF_NOT_FMT(resDistArr.size() == queryNum,
        APP_ERR_INVALID_PARAM,
        "resDistArr size(%zu) not equal to queryIdxArr()",
        resDistArr.size());
    APPERR_RETURN_IF_NOT_FMT(resIdxArr.size() == queryNum,
        APP_ERR_INVALID_PARAM,
        "resIdxArr size(%zu) not equal to queryIdxArr()",
        resIdxArr.size());
    APPERR_RETURN_IF_NOT_FMT(codeStartIdx + codeNum <= static_cast<uint32_t>(this->ntotal),
        APP_ERR_INVALID_PARAM,
        "codeStartIdx(%u) + codeNum(%u) must <= ntotal(%d) !!!",
        codeStartIdx,
        codeNum,
        this->ntotal);
    for (size_t i = 0; i < queryNum; i++) {
        APPERR_RETURN_IF_NOT_FMT((queryIdxArr[i] < static_cast<uint32_t>(this->ntotal)),
            APP_ERR_INVALID_PARAM,
            "queryIdxArr[%zu](%u) must >= 0 && < ntotal(%d)!",
            i,
            queryIdxArr[i],
            this->ntotal);
    }
    return ComputeDistByThresholdBatched(
        queryIdxArr, codeStartIdx, codeNum, threshold, resDistArr, resIdxArr);
}

APP_ERROR AscendIndexClusterImpl::ComputeDistByThresholdBatched(const std::vector<uint32_t> &queryIdxArr,
    uint32_t codeStartIdx, uint32_t codeNum, float threshold, std::vector<std::vector<float>> &resDistArr,
    std::vector<std::vector<uint32_t>> &resIdxArr)
{
    APP_LOG_INFO("AscendIndexClusterImpl ComputeDistByThresholdBatched operation started.\n");
    int ret = 0;
    size_t size = this->computeBatchSizes.size();
    size_t n = queryIdxArr.size();

    // op input requires codeStartIdx down 16-aligned
    uint32_t codeStartIdxPad =
        utils::divDown(codeStartIdx, static_cast<uint32_t>(CUBE_ALIGN)) * static_cast<uint32_t>(CUBE_ALIGN);

    uint32_t codeNumPad = utils::roundUp(codeStartIdx + codeNum, static_cast<uint32_t>(CUBE_ALIGN)) - codeStartIdxPad;
    auto streamPtr = this->pResources->getDefaultStream();
    auto stream = streamPtr->GetStream();
    auto &mem = this->pResources->getMemoryManager();

    uint64_t dataSize = n * codeNumPad;

    std::vector<float> distances;
    distances.reserve(dataSize);
    std::vector<uint32_t> indice;
    indice.reserve(dataSize);

    std::vector<uint64_t> distFilterNum(n, 0);
    for (uint64_t i = 1; i < n; i++) {
        distFilterNum[i] = i * static_cast<uint64_t>(codeNumPad);
    }

    if (n > 1 && size > 0) {
        size_t searched = 0;
        for (size_t i = 0; i < size; i++) {
            size_t batchSize = static_cast<size_t>(this->computeBatchSizes[i]);
            if ((n - searched) >= batchSize) {
                uint32_t batchNum = (n - searched) / batchSize;
                for (uint32_t j = 0; j < batchNum; j++) {
                    AscendTensor<uint32_t, DIMS_1> queryTensor(mem, {static_cast<int>(batchSize)}, stream);
                    ret = aclrtMemcpy(queryTensor.data(),
                        queryTensor.getSizeInBytes(),
                        queryIdxArr.data() + searched,
                        batchSize * sizeof(uint32_t),
                        ACL_MEMCPY_HOST_TO_DEVICE);
                    APPERR_RETURN_IF_NOT_FMT(
                        (ret == ACL_SUCCESS), APP_ERR_ACL_BAD_ALLOC, "copy queryIdxArr to device fail(%d) !!!", ret);

                    ret = ComputeDistPaged(batchSize,
                        queryTensor,
                        codeStartIdxPad,
                        codeNumPad,
                        threshold,
                        distances.data(),
                        indice.data(),
                        distFilterNum.data() + searched);
                    APPERR_RETURN_IF_NOT_FMT(
                        (ret == APP_ERR_OK), APP_ERR_REQUEST_ERROR, "ComputeDistPaged fail (%d) !!!", ret);
                    searched += batchSize;
                }
            }
        }
    } else {
        AscendTensor<uint32_t, DIMS_1> queryTensor(mem, {static_cast<int>(n)}, stream);
        ret = aclrtMemcpy(queryTensor.data(),
            queryTensor.getSizeInBytes(),
            queryIdxArr.data(),
            n * sizeof(uint32_t),
            ACL_MEMCPY_HOST_TO_DEVICE);
        APPERR_RETURN_IF_NOT_FMT(
            (ret == ACL_SUCCESS), APP_ERR_ACL_BAD_ALLOC, "copy queryIdxArr to device fail(%d) !!!", ret);
        ret = ComputeDistPaged(n,
            queryTensor,
            codeStartIdxPad,
            codeNumPad,
            threshold,
            distances.data(),
            indice.data(),
            distFilterNum.data());
        APPERR_RETURN_IF_NOT_FMT((ret == APP_ERR_OK), APP_ERR_REQUEST_ERROR, "ComputeDistPaged fail (%d) !!!", ret);
    }

    std::vector<std::future<void>> functorRet;
    uint32_t threadNum = std::min(n, static_cast<size_t>(MAX_THREAD_NUM));
    AscendThreadPool pool(threadNum);
    auto transformData = [](std::vector<std::vector<float>> &resDistArr,
                             std::vector<std::vector<uint32_t>> &resIdxArr,
                             uint32_t *indice,
                             float *distances,
                             uint32_t queryIdx,
                             uint32_t len,
                             uint32_t codeNumPad,
                             uint32_t codeStartIdx,
                             uint32_t codeNum) {
        resDistArr[queryIdx].clear();
        resIdxArr[queryIdx].clear();
        for (uint64_t j = 0; j < len; j++) {
            uint64_t x = static_cast<uint64_t>(queryIdx) * static_cast<uint64_t>(codeNumPad) + j;
            if (indice[x] >= codeStartIdx && indice[x] < codeStartIdx + codeNum) {
                // Filter the extra distances because ALIGN
                resDistArr[queryIdx].emplace_back(distances[x]);
                resIdxArr[queryIdx].emplace_back(indice[x]);
            }
        }
    };

    for (uint64_t i = 0; i < n; i++) {
        uint32_t len = distFilterNum[i] - i * static_cast<uint64_t>(codeNumPad);
        functorRet.emplace_back(pool.Enqueue(transformData,
            std::ref(resDistArr),
            std::ref(resIdxArr),
            indice.data(),
            distances.data(),
            i,
            len,
            codeNumPad,
            codeStartIdx,
            codeNum));
    }

    uint32_t seartchWait = 0;
    try {
        for (std::future<void> &ret : functorRet) {
            seartchWait++;
            ret.get();
        }
    } catch (std::exception &e) {
        for_each(functorRet.begin() + seartchWait, functorRet.end(), [](std::future<void> &ret) { ret.wait(); });
        APPERR_RETURN_IF_NOT_FMT(
            false, APP_ERR_INNER_ERROR, "translate vector fail, error msg[%s]", e.what());
    }

    APP_LOG_INFO("AscendIndexClusterImpl ComputeDistByThresholdBatched operation end.\n");

    return APP_ERR_OK;
}

APP_ERROR AscendIndexClusterImpl::ComputeDistPaged(uint32_t n, AscendTensor<uint32_t, DIMS_1> &queryTensor,
    uint32_t codeStartIdxPad, uint32_t codeNumPad, float threshold, float *distances, uint32_t *indice,
    uint64_t *distOffset)
{
    uint32_t codeOffset = 0;
    while (codeOffset < codeNumPad) {
        uint32_t computNum = std::min((codeNumPad - codeOffset), static_cast<uint32_t>(COMPUTE_BLOCK_SIZE));
        auto ret = ComputeDistImpl(
            n, queryTensor, codeStartIdxPad + codeOffset, computNum, threshold, distances, indice, distOffset);
        APPERR_RETURN_IF_NOT_FMT((ret == APP_ERR_OK), APP_ERR_REQUEST_ERROR, "ComputeDistImpl fail (%d) !!!", ret);
        codeOffset += computNum;
    }
    return APP_ERR_OK;
}

APP_ERROR AscendIndexClusterImpl::ComputeDistImpl(uint32_t n, AscendTensor<uint32_t, DIMS_1> &queryTensor,
    uint32_t codeStartIdx, uint32_t codeNumPad, float threshold, float *distances, uint32_t *indice,
    uint64_t *distOffset)
{
    APP_LOG_INFO("AscendIndexClusterImpl ComputeDistImpl operation started.\n");
    auto streamPtr = this->pResources->getDefaultStream();
    auto stream = streamPtr->GetStream();
    auto &mem = this->pResources->getMemoryManager();

    const int dim1 = utils::divUp(this->ntotal, CUBE_ALIGN);
    const int dim2 = utils::divUp(this->dim, CUBE_ALIGN);

    // new op
    AscendTensor<uint32_t, DIMS_1> sizeTenser(mem, {SIZE_ALIGN}, stream);
    AscendTensor<float16_t, DIMS_4> shapedData(this->baseSpace->data(), {dim1, dim2, CUBE_ALIGN, CUBE_ALIGN});
    AscendTensor<float, DIMS_1> distanceTensor(mem, {static_cast<int>(n) * static_cast<int>(codeNumPad)}, stream);

    std::vector<uint32_t> baseSizeHost;
    baseSizeHost.emplace_back(codeStartIdx);
    baseSizeHost.emplace_back(codeNumPad);
    baseSizeHost.insert(baseSizeHost.end(), SIZE_ALIGN - baseSizeHost.size(), 0);
    int ret = aclrtMemcpy(sizeTenser.data(),
        sizeTenser.getSizeInBytes(),
        baseSizeHost.data(),
        baseSizeHost.size() * sizeof(uint32_t),
        ACL_MEMCPY_HOST_TO_DEVICE);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "Mem operator error %d", ret);

    ret = DistanceFlatIpByIdx2(queryTensor, sizeTenser, shapedData, distanceTensor, stream);
    APPERR_RETURN_IF_NOT_FMT(ret == APP_ERR_OK, APP_ERR_INNER_ERROR, "DistanceFlatIpByIdx2 run faild(%d) !!!", ret);
    ret = synchronizeStream(stream);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS,
        APP_ERR_ACL_OP_EXEC_FAILED,
        "aclrtSynchronizeAPP_ERR_ACL_OP_EXEC_FAILEDStream DistanceFilter stream failed: %i\n",
        ret);

    // 2. call distance filter operator
    AscendTensor<uint32_t, DIMS_1> initIdx(mem, {SIZE_ALIGN}, stream);
    AscendTensor<float, DIMS_1> thresholdTensor(mem, {SIZE_ALIGN}, stream);
    AscendTensor<float, DIMS_2> filteredDist(mem, {static_cast<int>(n), static_cast<int>(codeNumPad)}, stream);
    AscendTensor<uint32_t, DIMS_2> filteredIndice(mem, {static_cast<int>(n), static_cast<int>(codeNumPad)}, stream);
    AscendTensor<uint32_t, DIMS_1> filteredCnt(mem, {static_cast<int>(n) * static_cast<int>(SIZE_ALIGN)}, stream);
    AscendTensor<uint32_t, DIMS_1> baseSize(mem, {SIZE_ALIGN}, stream);

    std::vector<uint32_t> initIdxHost(SIZE_ALIGN, codeStartIdx);
    ret = aclrtMemcpy(initIdx.data(),
        initIdx.getSizeInBytes(),
        initIdxHost.data(),
        initIdxHost.size() * sizeof(uint32_t),
        ACL_MEMCPY_HOST_TO_DEVICE);
    APPERR_RETURN_IF_NOT_FMT((ret == ACL_SUCCESS), APP_ERR_ACL_BAD_ALLOC, "copy initIdx back to device(%d) !!!", ret);

    std::vector<uint32_t> baseSizeHostFilter(SIZE_ALIGN, codeNumPad);
    ret = aclrtMemcpy(baseSize.data(),
        baseSize.getSizeInBytes(),
        baseSizeHostFilter.data(),
        baseSizeHostFilter.size() * sizeof(uint32_t),
        ACL_MEMCPY_HOST_TO_DEVICE);
    APPERR_RETURN_IF_NOT_FMT((ret == ACL_SUCCESS), APP_ERR_ACL_BAD_ALLOC, "copy baseSize back to device(%d) !!!", ret);

    std::vector<float> thresholdHost(SIZE_ALIGN, threshold);
    ret = aclrtMemcpy(thresholdTensor.data(),
        thresholdTensor.getSizeInBytes(),
        thresholdHost.data(),
        thresholdHost.size() * sizeof(float),
        ACL_MEMCPY_HOST_TO_DEVICE);
    APPERR_RETURN_IF_NOT_FMT(
        (ret == ACL_SUCCESS), APP_ERR_ACL_BAD_ALLOC, "copy thresholdTensor back to device(%d) !!!", ret);

    ret = DistanceFilter(
        distanceTensor, thresholdTensor, baseSize, initIdx, filteredDist, filteredIndice, filteredCnt, stream);
    APPERR_RETURN_IF_NOT_FMT(ret == APP_ERR_OK, APP_ERR_INNER_ERROR, "DistanceFilter run faild(%d) !!!", ret);
    ret = synchronizeStream(stream);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS,
        APP_ERR_ACL_OP_EXEC_FAILED,
        "synchronizeStream DistanceFilter stream failed: %i\n",
        ret);

    std::vector<uint32_t> idxNum(n * SIZE_ALIGN);
    // memcpy data back from dev to host
    ret = aclrtMemcpy(idxNum.data(),
        n * SIZE_ALIGN * sizeof(uint32_t),
        filteredCnt.data(),
        filteredCnt.getSizeInBytes(),
        ACL_MEMCPY_DEVICE_TO_HOST);
    APPERR_RETURN_IF_NOT_FMT((ret == ACL_SUCCESS), APP_ERR_ACL_BAD_ALLOC, "copy outIdxNum back to host(%d) !!!", ret);

    // copy valid data.
    for (uint32_t i = 0; i < n; i++) {
        uint32_t len = idxNum[i * SIZE_ALIGN];
        if (len > 0) {
            ret = aclrtMemcpy(distances + distOffset[i],
                codeNumPad * sizeof(float),
                filteredDist.data() + i * codeNumPad,
                len * sizeof(float),
                ACL_MEMCPY_DEVICE_TO_HOST);
            APPERR_RETURN_IF_NOT_FMT(
                (ret == ACL_SUCCESS), APP_ERR_ACL_BAD_ALLOC, "copy outDistances back to host(%d) !!!", ret);

            ret = aclrtMemcpy(indice + distOffset[i],
                codeNumPad * sizeof(uint32_t),
                filteredIndice.data() + i * codeNumPad,
                len * sizeof(uint32_t),
                ACL_MEMCPY_DEVICE_TO_HOST);
            APPERR_RETURN_IF_NOT_FMT(
                (ret == ACL_SUCCESS), APP_ERR_ACL_BAD_ALLOC, "copy outIndices back to host(%d) !!!", ret);
            distOffset[i] += len;
        }
    }
    APP_LOG_INFO("AscendIndexClusterImpl ComputeDistImpl operation end.\n");

    return ret;
}

APP_ERROR AscendIndexClusterImpl::DistanceFilter(AscendTensor<float, DIMS_1> &distanceTensor,
    AscendTensor<float, DIMS_1> &thresholdTensor, AscendTensor<uint32_t, DIMS_1> &baseSize,
    AscendTensor<uint32_t, DIMS_1> &initIdx, AscendTensor<float, DIMS_2> &filteredDist,
    AscendTensor<uint32_t, DIMS_2> &filteredIndice, AscendTensor<uint32_t, DIMS_1> &filteredCnt, aclrtStream stream)
{
    APP_LOG_INFO("AscendIndexClusterImpl DistanceFilter operation started.\n");
    AscendOperator *op = nullptr;
    int batchSize = filteredDist.getSize(0);
    if (this->distanceFilterOps.find(batchSize) != this->distanceFilterOps.end()) {
        op = this->distanceFilterOps[batchSize].get();
    }
    ASCEND_THROW_IF_NOT(op);
    std::shared_ptr<std::vector<const aclDataBuffer *>> distOpInput(
        new std::vector<const aclDataBuffer *>(), CommonUtils::AclInputBufferDelete);
    distOpInput->emplace_back(aclCreateDataBuffer(
        distanceTensor.data(), static_cast<size_t>(batchSize) * FAKE_HUGE_BASE * sizeof(float)));
    distOpInput->emplace_back(aclCreateDataBuffer(thresholdTensor.data(), thresholdTensor.getSizeInBytes()));
    distOpInput->emplace_back(aclCreateDataBuffer(baseSize.data(), baseSize.getSizeInBytes()));
    distOpInput->emplace_back(aclCreateDataBuffer(initIdx.data(), initIdx.getSizeInBytes()));

    std::shared_ptr<std::vector<aclDataBuffer *>> distOpOutput(
        new std::vector<aclDataBuffer *>(), CommonUtils::AclOutputBufferDelete);
    distOpOutput->emplace_back(aclCreateDataBuffer(
        filteredDist.data(), static_cast<size_t>(batchSize) * FAKE_HUGE_BASE * sizeof(float)));
    distOpOutput->emplace_back(aclCreateDataBuffer(
        filteredIndice.data(), static_cast<size_t>(batchSize) * FAKE_HUGE_BASE * sizeof(int)));
    distOpOutput->emplace_back(aclCreateDataBuffer(filteredCnt.data(), filteredCnt.getSizeInBytes()));

    op->exec(*distOpInput, *distOpOutput, stream);
    APP_LOG_INFO("AscendIndexClusterImpl DistanceFilter operation end.\n");

    return APP_ERR_OK;
}


APP_ERROR AscendIndexClusterImpl::DistanceFlatIpByIdx2(AscendTensor<uint32_t, DIMS_1> &queryOffset,
    AscendTensor<uint32_t, DIMS_1> &size, AscendTensor<float16_t, DIMS_4> &shape,
    AscendTensor<float, DIMS_1> &outTensor, aclrtStream stream)
{
    APP_LOG_INFO("AscendIndexClusterImpl DistanceFlatIpByIdx2 operation started.\n");
    auto ret = aclrtSetDevice(deviceId);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_ACL_SET_DEVICE_FAILED, "failed to set device(%d)", ret);
    AscendOperator *op = nullptr;
    int batchSize = queryOffset.getSize(0);
    if (this->queryCopyOps.find(batchSize) != this->queryCopyOps.end()) {
        op = this->queryCopyOps[batchSize].get();
    }
    ASCEND_THROW_IF_NOT(op);

    std::shared_ptr<std::vector<const aclDataBuffer *>> distOpInput(
        new std::vector<const aclDataBuffer *>(), CommonUtils::AclInputBufferDelete);
    distOpInput->emplace_back(aclCreateDataBuffer(queryOffset.data(), queryOffset.getSizeInBytes()));
    distOpInput->emplace_back(aclCreateDataBuffer(size.data(), size.getSizeInBytes()));
    distOpInput->emplace_back(aclCreateDataBuffer(shape.data(), this->fakeBaseSizeInBytes));

    std::shared_ptr<std::vector<aclDataBuffer *>> distOpOutput(
        new std::vector<aclDataBuffer *>(), CommonUtils::AclOutputBufferDelete);
    distOpOutput->emplace_back(aclCreateDataBuffer(
        outTensor.data(), static_cast<size_t>(batchSize) * FAKE_HUGE_BASE * sizeof(float)));

    op->exec(*distOpInput, *distOpOutput, stream);
    APP_LOG_INFO("AscendIndexClusterImpl DistanceFlatIpByIdx2 operation end.\n");

    return APP_ERR_OK;
}

APP_ERROR AscendIndexClusterImpl::Search(int n, const uint16_t *queries, int topk,
    float *distances, int64_t *labels, unsigned int tableLen, const float *table)
{
    APP_LOG_INFO("AscendIndexClusterImpl Search operation start.\n");
    APPERR_RETURN_IF_NOT_LOG(
        (isInitialized), APP_ERR_INVALID_PARAM, "Illegal operation, please initialize the index first. ");

    if (n == 1) {
        auto ret = SearchImpl(n, queries, topk, labels, distances);
        APPERR_RETURN_IF_NOT_FMT(ret == APP_ERR_OK, APP_ERR_INNER_ERROR,
            "AscendIndexClusterImpl SearchImpl faild(%d)", ret);
        return this->TableMapping(n, distances, tableLen, table, topk);
    }

    size_t size = computeBatchSizes.size();
    int searched = 0;
    for (size_t i = 0; i < size; i++) {
        int batchSize = computeBatchSizes[i];
        if ((n - searched) >= batchSize) {
            int page = (n - searched) / batchSize;
            for (int j = 0; j < page; j++) {
                auto ret = SearchImpl(batchSize, queries + searched * this->dim, topk, labels +
                    searched * topk, distances + searched * topk);
                APPERR_RETURN_IF_NOT_FMT(ret == APP_ERR_OK, APP_ERR_INNER_ERROR,
                    "AscendIndexClusterImpl SearchImpl faild(%d)", ret);
                searched += batchSize;
            }
        }
    }
    APP_LOG_INFO("AscendIndexClusterImpl Search operation end.\n");
    return this->TableMapping(n, distances, tableLen, table, topk);
}

APP_ERROR AscendIndexClusterImpl::TableMapping(int n, float *distances, size_t tableLen,
    const float *table, int topk)
{
    APP_LOG_INFO("AscendIndexClusterImpl TableMapping operation start.\n");
    if (tableLen > 0) {
        int numEachQuery = 0;
        numEachQuery = std::min(topk, this->ntotal);
        int tableIndex = 0;
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < numEachQuery; ++j) {
                int offset = i * topk + j;
                // index = (cos + 1) / 2 * tableLen, 加上0.5，取整，实现四舍五入
                tableIndex = static_cast<int>((*(distances + offset) + 1) / 2 * tableLen + 0.5);
                APPERR_RETURN_IF_NOT_LOG(
                    tableIndex >= 0 && tableIndex < static_cast<int>(TABLE_LEN), APP_ERR_INVALID_TABLE_INDEX,
                    "Invalid index for table mapping, please ensure the correctness of vector normalization.\n");
                *(distances + offset) = *(table + tableIndex);
            }
        }
    }
    APP_LOG_INFO("AscendIndexClusterImpl TableMapping operation end.\n");
    return APP_ERR_OK;
}

APP_ERROR AscendIndexClusterImpl::SearchImpl(int n, const uint16_t *query, int topk, int64_t *labels, float *distances)
{
    APP_LOG_INFO("AscendIndexClusterImpl SearchImpl operation start.\n");

    auto streamPtr = this->pResources->getDefaultStream();
    auto stream = streamPtr->GetStream();
    auto &mem = this->pResources->getMemoryManager();

    AscendTensor<float16_t, DIMS_2> queries(mem, {n, this->dim}, stream);
    auto ret = aclrtMemcpy(queries.data(), queries.getSizeInBytes(), query, n * this->dim * sizeof(uint16_t),
        ACL_MEMCPY_HOST_TO_DEVICE);
    APPERR_RETURN_IF_NOT_FMT((ret == ACL_SUCCESS), APP_ERR_INNER_ERROR, "copy queryIdxArr to device fail(%d)!", ret);

    AscendTensor<int64_t, DIMS_2> outIndices(mem, { n, topk }, stream);
    AscendTensor<float16_t, DIMS_2> outDistances(mem, { n, topk }, stream);

    int pageNum = utils::divUp(this->ntotal, this->pageSize);

    for (int pageId = 0; pageId < pageNum; ++pageId) {
        APP_ERROR ret = SearchPaged(pageId, queries, pageNum, outIndices, outDistances);
        APPERR_RETURN_IF_NOT_FMT(ret == APP_ERR_OK, APP_ERR_INNER_ERROR,
            "AscendIndexClusterImpl SearchPaged faild(%d)", ret);
    }

    std::vector<float16_t> tmpDistances(n * topk);
    ret = aclrtMemcpy(tmpDistances.data(), tmpDistances.size() * sizeof(float16_t), outDistances.data(),
        outDistances.getSizeInBytes(), ACL_MEMCPY_DEVICE_TO_HOST);
    APPERR_RETURN_IF_NOT_FMT((ret == ACL_SUCCESS), APP_ERR_INNER_ERROR, "copy outDistances back to host(%d)", ret);

    std::transform(tmpDistances.begin(), tmpDistances.end(), distances,
        [](float16_t temp) { return static_cast<float>(fp16(temp)); });

    ret = aclrtMemcpy(labels, n * topk * sizeof(int64_t), outIndices.data(), outIndices.getSizeInBytes(),
        ACL_MEMCPY_DEVICE_TO_HOST);
    APPERR_RETURN_IF_NOT_FMT((ret == ACL_SUCCESS), APP_ERR_INNER_ERROR, "copy outIndices back to host(%d)", ret);
    APP_LOG_INFO("AscendIndexClusterImpl SearchImpl operation end.\n");
    return APP_ERR_OK;
}

APP_ERROR AscendIndexClusterImpl::InitAttr(int topk, int burstLen, int blockNum, int pageId, int pageNum,
    AscendTensor<int64_t, DIMS_1>& attrsInput)
{
    std::vector<int64_t> attrs(aicpu::TOPK_FLAT_ATTR_IDX_COUNT);
    attrs[aicpu::TOPK_FLAT_ATTR_ASC_IDX] = 0;
    attrs[aicpu::TOPK_FLAT_ATTR_K_IDX] = topk;
    attrs[aicpu::TOPK_FLAT_ATTR_BURST_LEN_IDX] = burstLen;
    attrs[aicpu::TOPK_FLAT_ATTR_BLOCK_NUM_IDX] = blockNum;
    attrs[aicpu::TOPK_FLAT_ATTR_PAGE_IDX] = pageId;
    attrs[aicpu::TOPK_FLAT_ATTR_PAGE_NUM_IDX] = pageNum;
    attrs[aicpu::TOPK_FLAT_ATTR_PAGE_SIZE_IDX] = this->pageSize;
    attrs[aicpu::TOPK_FLAT_ATTR_QUICK_HEAP] = 1;
    attrs[aicpu::TOPK_FLAT_ATTR_BLOCK_SIZE] = this->blockSize;
    auto ret = aclrtMemcpy(attrsInput.data(), attrsInput.getSizeInBytes(), attrs.data(), attrs.size() * sizeof(int64_t),
        ACL_MEMCPY_HOST_TO_DEVICE);
    APPERR_RETURN_IF_NOT_FMT((ret == ACL_SUCCESS), APP_ERR_INNER_ERROR, "copy attrsInput to device(%d)", ret);
    return APP_ERR_OK;
}

APP_ERROR AscendIndexClusterImpl::SearchPaged(int pageId, AscendTensor<float16_t, DIMS_2> &queries, int pageNum,
    AscendTensor<int64_t, DIMS_2> &outIndices, AscendTensor<float16_t, DIMS_2> &outDistances)
{
    APP_LOG_INFO("AscendIndexClusterImpl SearchPaged operation start.\n");
    auto streamPtr = this->pResources->getDefaultStream();
    auto stream = streamPtr->GetStream();
    auto streamAicpuPtr = this->pResources->getAlternateStreams()[0];
    auto streamAicpu = streamAicpuPtr->GetStream();

    auto &mem = this->pResources->getMemoryManager();
    int batch = queries.getSize(0);
    int topk = outDistances.getSize(1);
    int pageOffset = pageId * this->pageSize;
    int blockOffset = pageOffset / this->blockSize;
    int computeNum = std::min(this->ntotal - pageOffset, this->pageSize);
    int blockNum = utils::divUp(computeNum, this->blockSize);

    auto burstLen = (batch >= BIG_BATCH_THRESHOLD) ? BIG_BATCH_BURST_LEN : BURST_LEN;
    // 乘以2，是和算子生成时的shape保持一致
    auto burstOfBlock = (FLAT_BLOCK_SIZE + burstLen - 1) / burstLen * 2;

    AscendTensor<float16_t, DIMS_3> distResult(mem, { blockNum, batch, this->blockSize }, stream);
    AscendTensor<float16_t, DIMS_3> maxDistResult(mem, { blockNum, batch, burstOfBlock }, stream);
    AscendTensor<uint32_t, DIMS_3> opSize(mem, { blockNum, CORE_NUM, SIZE_ALIGN }, stream);
    AscendTensor<uint16_t, DIMS_3> opFlag(mem, { blockNum, CORE_NUM, FLAG_SIZE }, stream);
    opFlag.zero();

    AscendTensor<uint8_t, DIMS_2> mask(mem, { batch, this->blockMaskSize }, stream);

    AscendTensor<int64_t, DIMS_1> attrsInput(mem, { aicpu::TOPK_FLAT_ATTR_IDX_COUNT }, stream);
    auto ret = InitAttr(topk, burstLen, blockNum, pageId, pageNum, attrsInput);
    APPERR_RETURN_IF_NOT_FMT((ret == ACL_SUCCESS), APP_ERR_INNER_ERROR, "InitAttr failed(%d)", ret);
    runTopkCompute(distResult, maxDistResult, opSize, opFlag, attrsInput, outDistances, outIndices, streamAicpu);

    const int dim1 = utils::divUp(this->blockSize, ZREGION_HEIGHT);
    const int dim2 = utils::divUp(this->dim, CUBE_ALIGN);
    for (int i = 0; i < blockNum; i++) {
        auto baseOffset = static_cast<int64_t>(blockOffset + i) * this->blockSize * dim2 * CUBE_ALIGN;
        AscendTensor<float16_t, DIMS_4> shaped(this->baseSpace->data() + baseOffset,
            { dim1, dim2, ZREGION_HEIGHT, CUBE_ALIGN });
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

    ret = synchronizeStream(stream);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "synchronizeStream failed: %i\n", ret);

    ret = synchronizeStream(streamAicpu);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "synchronizeStream aicpu failed: %i\n", ret);
    APP_LOG_INFO("AscendIndexClusterImpl SearchPaged operation end.\n");
    return APP_ERR_OK;
}

APP_ERROR AscendIndexClusterImpl::CheckParams(int n, const uint16_t *queries, const int *num,
    float *distances, unsigned int tableLen, const float *table)
{
    APPERR_RETURN_IF_NOT_LOG(queries, APP_ERR_INVALID_PARAM, "queries can not be nullptr.");
    APPERR_RETURN_IF_NOT_LOG(num, APP_ERR_INVALID_PARAM, "num can not be nullptr.");
    APPERR_RETURN_IF_NOT_LOG(distances, APP_ERR_INVALID_PARAM, "distances can not be nullptr.");
    APPERR_RETURN_IF_NOT_FMT(n > 0 && n <= this->capacity, APP_ERR_INVALID_PARAM,
        "The number n should be in range (0, %d]", this->capacity);
    if (tableLen > 0) {
        APPERR_RETURN_IF_NOT_FMT(tableLen == ACTUAL_TABLE_LEN, APP_ERR_INVALID_PARAM,
            "table length only support %d", ACTUAL_TABLE_LEN);
        APPERR_RETURN_IF_NOT_LOG(table, APP_ERR_INVALID_PARAM,
            "The table pointer cannot be nullptr when tableLen is valid.");
    }
    return APP_ERR_OK;
}


APP_ERROR AscendIndexClusterImpl::SearchByThreshold(int n, const uint16_t *queries, float threshold, int topk,
    int *num, int64_t *labels, float *distances, unsigned int tableLen, const float *table)
{
    auto res = CheckParams(n, queries, num, distances, tableLen, table);
    APPERR_RETURN_IF_NOT_FMT(res == APP_ERR_OK, APP_ERR_INVALID_PARAM,
        "AscendIndexClusterImpl CheckParams faild(%d)", res);
    APPERR_RETURN_IF_NOT_LOG(topk > 0 && topk <= MAX_TOPK, APP_ERR_INVALID_PARAM,
        "Invalid parameter, topk should be in range (0, 1024].");
    APPERR_RETURN_IF_NOT_LOG(labels, APP_ERR_INVALID_PARAM, "labels can not be nullptr.");
    APPERR_RETURN_IF_NOT_LOG(
        (isInitialized), APP_ERR_INVALID_PARAM, "Illegal operation, please initialize the index first. ");
    APP_ERROR ret = this->Search(n, queries, topk, distances, labels, tableLen, table);

    APPERR_RETURN_IF_NOT_FMT(ret == APP_ERR_OK, APP_ERR_INNER_ERROR, "AscendIndexClusterImpl Search faild(%d)", ret);
    int validNum = std::min(topk, this->ntotal);
    for (int i = 0; i < n; i++) {
        int qnum = 0;
        for (int j = 0; j < validNum; j++) {
            int offset = i * topk + j;
            if (*(distances + offset) >= threshold) {
                *(distances + i * topk + qnum) = *(distances + offset);
                *(labels + i * topk + qnum) = *(labels + offset);
                qnum += 1;
            }
        }
        num[i] = qnum;
    }
    return APP_ERR_OK;
}

void AscendIndexClusterImpl::ComputeBlockDist(AscendTensor<float16_t, DIMS_2> &queryTensor,
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

void AscendIndexClusterImpl::runTopkCompute(AscendTensor<float16_t, DIMS_3> &dists,
                                            AscendTensor<float16_t, DIMS_3> &maxdists,
                                            AscendTensor<uint32_t, DIMS_3> &sizes,
                                            AscendTensor<uint16_t, DIMS_3> &flags,
                                            AscendTensor<int64_t, DIMS_1> &attrs,
                                            AscendTensor<float16_t, DIMS_2> &outdists,
                                            AscendTensor<int64_t, DIMS_2> &outlabel,
                                            aclrtStream stream)
{
    AscendOperator *op = nullptr;
    int batch = dists.getSize(1);
    if (topkComputeOps.find(batch) != topkComputeOps.end()) {
        op = topkComputeOps[batch].get();
    }
    ASCEND_THROW_IF_NOT(op);

    std::shared_ptr<std::vector<const aclDataBuffer *>> topkOpInput(
        new std::vector<const aclDataBuffer *>(), CommonUtils::AclInputBufferDelete);
    topkOpInput->emplace_back(aclCreateDataBuffer(dists.data(), dists.getSizeInBytes()));
    topkOpInput->emplace_back(aclCreateDataBuffer(maxdists.data(), maxdists.getSizeInBytes()));
    topkOpInput->emplace_back(aclCreateDataBuffer(sizes.data(), sizes.getSizeInBytes()));
    topkOpInput->emplace_back(aclCreateDataBuffer(flags.data(), flags.getSizeInBytes()));
    topkOpInput->emplace_back(aclCreateDataBuffer(attrs.data(), attrs.getSizeInBytes()));

    std::shared_ptr<std::vector<aclDataBuffer *>> topkOpOutput(
        new std::vector<aclDataBuffer *>(), CommonUtils::AclOutputBufferDelete);
    topkOpOutput->emplace_back(aclCreateDataBuffer(outdists.data(), outdists.getSizeInBytes()));
    topkOpOutput->emplace_back(aclCreateDataBuffer(outlabel.data(), outlabel.getSizeInBytes()));

    op->exec(*topkOpInput, *topkOpOutput, stream);
}

APP_ERROR AscendIndexClusterImpl::CheckComputeMaxNum(int n, const int *num, const uint32_t *indices, int& maxNum)
{
    for (int i = 0; i < n; i++) {
        int tmpNum = *(num + i);
        APPERR_RETURN_IF_NOT_LOG(tmpNum >= 0 && tmpNum <= this->ntotal, APP_ERR_INVALID_PARAM,
            "The num of query idx is invalid, it should be in range [0, ntotal]. ");
        maxNum = std::max(maxNum, tmpNum);
    }
    if (maxNum == 0) {
        return APP_ERR_OK;
    }
    for (int i = 0; i < n; i++) {
        int tmpNum = *(num + i);
        for (int j = 0; j < tmpNum; j++) {
            APPERR_RETURN_IF_NOT_LOG(*(indices + i * maxNum + j) < static_cast<size_t>(this->ntotal),
                APP_ERR_INVALID_PARAM, "The given indice to compare with should be smaller than ntotal");
        }
    }
    return APP_ERR_OK;
}

APP_ERROR AscendIndexClusterImpl::ComputeDistanceByIdx(int n, const uint16_t *queries, const int *num,
    const uint32_t *indices, float *distances, unsigned int tableLen, const float *table)
{
    APPERR_RETURN_IF_NOT_LOG(
        (isInitialized), APP_ERR_INVALID_PARAM, "Illegal operation, please initialize the index first. ");
    auto res = CheckParams(n, queries, num, distances, tableLen, table);
    APPERR_RETURN_IF_NOT_FMT(res == APP_ERR_OK, APP_ERR_INNER_ERROR,
        "AscendIndexClusterImpl CheckParams faild(%d)", res);
    APPERR_RETURN_IF_NOT_LOG(indices, APP_ERR_INVALID_PARAM, "indices can not be nullptr.");
    int maxNum = 0;
    res = CheckComputeMaxNum(n, num, indices, maxNum);
    APPERR_RETURN_IF_NOT_FMT(res == APP_ERR_OK, APP_ERR_INNER_ERROR,
        "AscendIndexClusterImpl CheckComputeMaxNum faild(%d)", res);
    std::tuple<unsigned int, const float *> tableInfo(tableLen, table);
    std::tuple<int, const int *, const uint32_t *> idxInfo;
    if (n == 1) {
        idxInfo = std::make_tuple(maxNum, num, indices);
        auto ret = ComputeDistByIdxImpl(n, queries, distances, idxInfo, tableInfo);
        APPERR_RETURN_IF_NOT_FMT(ret == APP_ERR_OK, APP_ERR_INNER_ERROR,
            "AscendIndexClusterImpl ComputeDistByIdxImpl faild(%d)", ret);
        return APP_ERR_OK;
    }
    int size = static_cast<int>(COMPUTE_BY_IDX_BATCHES.size());
    int searched = 0;
    for (int i = 0; i < size; i++) {
        int batchSize = COMPUTE_BY_IDX_BATCHES[i];
        if ((n - searched) >= batchSize) {
            int batchNum = (n - searched) / batchSize;
            for (int j = 0; j < batchNum; j++) {
                idxInfo = std::make_tuple(maxNum, num + searched, indices + searched * maxNum);
                auto ret = ComputeDistByIdxImpl(batchSize, queries + searched * this->dim,
                    distances + searched * maxNum, idxInfo, tableInfo);
                APPERR_RETURN_IF_NOT_FMT(ret == APP_ERR_OK, APP_ERR_INNER_ERROR,
                    "AscendIndexClusterImpl ComputeDistByIdxImpl faild(%d)", ret);
                searched += batchSize;
            }
        }
    }
    return APP_ERR_OK;
}

APP_ERROR AscendIndexClusterImpl::CopyDisToHost(int maxNum, int idxCopyNum, float *distances, int n,
    AscendTensor<float, DIMS_3>& distResult)
{
    int idxSliceNum = utils::divUp(maxNum, IDX_BURST_LEN);
    for (int i = 0; i < idxSliceNum; i++) {
        for (int j = 0; j < n; j++) {
            idxCopyNum = (i == idxSliceNum - 1) ? (maxNum - i * IDX_BURST_LEN) : IDX_BURST_LEN;
            auto err = aclrtMemcpy(distances + i * IDX_BURST_LEN + j * maxNum, idxCopyNum * sizeof(float),
                distResult[i][j].data(), idxCopyNum * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
            APPERR_RETURN_IF_NOT_FMT(err == ACL_SUCCESS, APP_ERR_INNER_ERROR,
                "memcpy error, i = %d, j = %d. err = %d\n", i, j, err);
        }
    }
    return APP_ERR_OK;
}

APP_ERROR AscendIndexClusterImpl::MoveData(int& idxCopyNum, int maxNum, int n, const uint32_t *indice,
    AscendTensor<uint32_t, DIMS_3>& idxTensor)
{
    int idxSliceNum = utils::divUp(maxNum, IDX_BURST_LEN);
    for (int i = 0; i < idxSliceNum; i++) {
        for (int j = 0; j < n; j++) {
            idxCopyNum = (i == idxSliceNum - 1) ? (maxNum - i * IDX_BURST_LEN) : IDX_BURST_LEN;
            auto err = aclrtMemcpy(idxTensor[i][j].data(), idxCopyNum * sizeof(uint32_t),
                indice + static_cast<size_t>(j) * maxNum + i * IDX_BURST_LEN, idxCopyNum * sizeof(uint32_t),
                ACL_MEMCPY_HOST_TO_DEVICE);
            APPERR_RETURN_IF_NOT_FMT((err == ACL_SUCCESS), APP_ERR_INNER_ERROR, "copy idxTensor fail(%d)!", err);
        }
    }
    return APP_ERR_OK;
}

APP_ERROR AscendIndexClusterImpl::ComputeDistByIdxImpl(int n, const uint16_t *queries, float *distances,
    std::tuple<int, const int *, const uint32_t *> idxInfo, std::tuple<unsigned int, const float *> tableInfo)
{
    auto streamPtr = this->pResources->getDefaultStream();
    auto stream = streamPtr->GetStream();
    auto &mem = this->pResources->getMemoryManager();
    int maxNum = std::get<0>(idxInfo);
    const uint32_t *indice = std::get<2>(idxInfo);
    unsigned int tableLen = std::get<0>(tableInfo);
    const float *table = std::get<1>(tableInfo);
    int idxSliceNum = utils::divUp(maxNum, IDX_BURST_LEN);

    AscendTensor<float16_t, DIMS_2> queryTensor(mem, {n, this->dim}, stream);
    auto ret = aclrtMemcpy(queryTensor.data(), queryTensor.getSizeInBytes(), queries, n * this->dim * sizeof(uint16_t),
        ACL_MEMCPY_HOST_TO_DEVICE);
    APPERR_RETURN_IF_NOT_FMT((ret == ACL_SUCCESS), APP_ERR_INNER_ERROR, "copy queryIdxArr to device fail(%d)!", ret);

    AscendTensor<float, DIMS_3> distResult(mem, { idxSliceNum, n, IDX_BURST_LEN }, stream);
    AscendTensor<uint32_t, DIMS_3> idxTensor(mem, { idxSliceNum, n, IDX_BURST_LEN }, stream);
    // 将索引搬运成大z小z，maxNum按64补齐
    int idxCopyNum = 0;
    ret = MoveData(idxCopyNum, maxNum, n, indice, idxTensor);
    APPERR_RETURN_IF_NOT_FMT((ret == ACL_SUCCESS), APP_ERR_INNER_ERROR, "MoveData failed(%d)", ret);
    int blockNum = utils::divUp(this->capacity, this->blockSize);
    const int dim1 = utils::divUp(this->blockSize, ZREGION_HEIGHT);
    const int dim2 = utils::divUp(this->dim, CUBE_ALIGN);

    // 每次调用算子，都传入全量的底库
    AscendTensor<float16_t, DIMS_4> shaped(
        this->baseSpace->data(), { blockNum * dim1, dim2, ZREGION_HEIGHT, CUBE_ALIGN });
    AscendTensor<uint32_t, DIMS_2> sizeTensorList(mem, {idxSliceNum, SIZE_ALIGN}, stream);

    for (int i = 0; i < idxSliceNum; i++) {
        auto index = idxTensor[i].view();
        auto dist = distResult[i].view();
        auto sizeTensor = sizeTensorList[i].view();
        sizeTensor[0] = (i == idxSliceNum - 1) ? (maxNum - i * IDX_BURST_LEN) : IDX_BURST_LEN;
        if (tableLen > 0) {
            AscendTensor<float, DIMS_1> tableTensor(mem, {TABLE_LEN}, stream);
            ret = aclrtMemcpy(tableTensor.data(), tableTensor.getSizeInBytes(), table, TABLE_LEN * sizeof(float),
                ACL_MEMCPY_HOST_TO_DEVICE);
            APPERR_RETURN_IF_NOT_FMT(
                (ret == ACL_SUCCESS), APP_ERR_INNER_ERROR, "copy tableTensor to device fail(%d) !!!", ret);
            ComputeDistWholeBaseWithTable(queryTensor, sizeTensor, shaped, index, dist, tableTensor, stream);
        } else {
            ComputeDistWholeBase(queryTensor, sizeTensor, shaped, index, dist, stream);
        }
    }
    ret = synchronizeStream(stream);

    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR,
        "synchronizeStream default stream failed, error code: %i\n", ret);

    // 拷贝结果到输出空间
    ret = CopyDisToHost(maxNum, idxCopyNum, distances, n, distResult);
    APPERR_RETURN_IF_NOT_FMT((ret == ACL_SUCCESS), APP_ERR_INNER_ERROR, "CopyDisToHost failed(%d)", ret);
    return ret;
}

APP_ERROR AscendIndexClusterImpl::Get(int n, uint16_t *features, const int64_t *indices) const
{
    APPERR_RETURN_IF_NOT_LOG(
        (isInitialized), APP_ERR_INVALID_PARAM, "Illegal operation, please initialize the index first. ");

    int64_t dimAlign = utils::divUp(this->dim, CUBE_ALIGN);
    for (int64_t i = 0; i < n; ++i) {
        int64_t seq = *(indices + i);
        APPERR_RETURN_IF_NOT_LOG(seq < this->ntotal, APP_ERR_INVALID_PARAM,
            "Invalid feature to get, the indice should be smaller than ntotal\n");
        float16_t *dataPtr = this->baseSpace->data() + seq / ZREGION_HEIGHT * dimAlign * (ZREGION_HEIGHT * CUBE_ALIGN) +
            seq % ZREGION_HEIGHT * CUBE_ALIGN;

        for (int64_t j = 0; j < dimAlign; j++) {
            int64_t getOffset = i * this->dim + j * CUBE_ALIGN;
            int64_t cpyNum = (j == dimAlign - 1) ? (this->dim - j * CUBE_ALIGN) : CUBE_ALIGN;
            auto err = aclrtMemcpy(features + getOffset, cpyNum * sizeof(uint16_t),
                dataPtr + j * ZREGION_HEIGHT * CUBE_ALIGN, cpyNum * sizeof(float16_t), ACL_MEMCPY_DEVICE_TO_HOST);
            APPERR_RETURN_IF_NOT_FMT(err == EOK, APP_ERR_INNER_ERROR, "aclrtMemcpy error, err=%d\n", err);
        }
    }
    return APP_ERR_OK;
}

APP_ERROR AscendIndexClusterImpl::Remove(int n, const int64_t *indices)
{
    APPERR_RETURN_IF_NOT_LOG(
        (isInitialized), APP_ERR_INVALID_PARAM, "Illegal operation, please initialize the index first. ");

    int64_t dimAlign = utils::divUp(this->dim, CUBE_ALIGN);

    for (int64_t i = 0; i < n; i++) {
        auto seq = *(indices + i);
        APPERR_RETURN_IF_NOT_LOG(seq < this->ntotal, APP_ERR_INVALID_PARAM,
            "Invalid feature to remove, the indice should not be greater than ntotal\n");
        int64_t offset1 = seq / ZREGION_HEIGHT * dimAlign;
        int64_t offset2 = seq % ZREGION_HEIGHT;
        int64_t offset = offset1 * ZREGION_HEIGHT * CUBE_ALIGN + offset2 * CUBE_ALIGN;
        auto dataptr = this->baseSpace->data() + offset;

        for (int64_t j = 0; j < dimAlign; j++) {
            auto err = aclrtMemset(dataptr, CUBE_ALIGN * sizeof(float16_t), 0x0, CUBE_ALIGN * sizeof(float16_t));
            APPERR_RETURN_IF_NOT_FMT(err == EOK, APP_ERR_INNER_ERROR, "aclrtMemset error, err=%d\n", err);
            dataptr += ZREGION_HEIGHT * CUBE_ALIGN;
        }
    }

    return APP_ERR_OK;
}

void AscendIndexClusterImpl::SetNTotal(int n)
{
    this->ntotal = n;
}

int AscendIndexClusterImpl::GetNTotal() const
{
    return this->ntotal;
}

void AscendIndexClusterImpl::ComputeDistWholeBase(AscendTensor<float16_t, DIMS_2> &queryTensor,
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

void AscendIndexClusterImpl::ComputeDistWholeBaseWithTable(AscendTensor<float16_t, DIMS_2> &queryTensor,
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

APP_ERROR AscendIndexClusterImpl::ResetDistCompIdxWithTableOp()
{
    auto distCompIdxWithTableOpReset = [&](std::unique_ptr<AscendOperator> &op, int64_t batch) {
        AscendOpDesc desc("DistanceFlatIpByIdxWithTable");
        std::vector<int64_t> queryShape({ batch, this->dim });
        std::vector<int64_t> indexShape({ batch, IDX_BURST_LEN });
        std::vector<int64_t> sizeShape({ SIZE_ALIGN });
        std::vector<int64_t> coarseCentroidsShape(
            { utils::divUp(FAKE_HUGE_BASE, ZREGION_HEIGHT), utils::divUp(this->dim, CUBE_ALIGN),
            ZREGION_HEIGHT, CUBE_ALIGN });
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
APP_ERROR AscendIndexClusterImpl::ResetTopkCompOp()
{
    auto topkCompOpReset = [&](std::unique_ptr<AscendOperator> &op, int64_t batch) {
        AscendOpDesc desc("TopkFlat");
        auto burstLen = (batch >= BIG_BATCH_THRESHOLD) ? BIG_BATCH_BURST_LEN : BURST_LEN;
        auto burstOfBlock = (FLAT_BLOCK_SIZE + burstLen - 1) / burstLen * 2;
        std::vector<int64_t> shape0 { 0, batch, this->blockSize };
        std::vector<int64_t> shape1 { 0, batch, burstOfBlock };
        std::vector<int64_t> shape2 { 0, CORE_NUM, SIZE_ALIGN };
        std::vector<int64_t> shape3 { 0, CORE_NUM, FLAG_SIZE };
        std::vector<int64_t> shape4 { aicpu::TOPK_FLAT_ATTR_IDX_COUNT };
        std::vector<int64_t> shape5 { batch, 0 };

        desc.addInputTensorDesc(ACL_FLOAT16, shape0.size(), shape0.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_FLOAT16, shape1.size(), shape1.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_UINT32, shape2.size(), shape2.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_UINT16, shape3.size(), shape3.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_INT64, shape4.size(), shape4.data(), ACL_FORMAT_ND);

        desc.addOutputTensorDesc(ACL_FLOAT16, shape5.size(), shape5.data(), ACL_FORMAT_ND);
        desc.addOutputTensorDesc(ACL_INT64, shape5.size(), shape5.data(), ACL_FORMAT_ND);
        op.reset();
        op = CREATE_UNIQUE_PTR(AscendOperator, desc);
        return op->init();
    };

    for (auto batch : SEARCH_BATCHES) {
        topkComputeOps[batch] = std::unique_ptr<AscendOperator>(nullptr);
        APPERR_RETURN_IF_NOT_LOG(topkCompOpReset(topkComputeOps[batch], batch),
                                 APP_ERR_ACL_OP_LOAD_MODEL_FAILED, "topk op init failed");
    }
    return APP_ERR_OK;
}

APP_ERROR AscendIndexClusterImpl::ResetDistCompOp(int numLists)
{
    auto distCompOpReset = [&](std::unique_ptr<AscendOperator> &op, int64_t batch) {
        auto burstLen = (batch >= BIG_BATCH_THRESHOLD) ? BIG_BATCH_BURST_LEN : BURST_LEN;
        // 乘以2，是和算子生成时的shape保持一致
        auto burstOfBlock = (FLAT_BLOCK_SIZE + burstLen - 1) / burstLen * 2;
        std::string opName = (batch >= BIG_BATCH_THRESHOLD) ? "DistanceFlatIPMaxsBatch" : "DistanceFlatIPMaxs";
        AscendOpDesc desc(opName);
        std::vector<int64_t> queryShape({ batch, this->dim });
        std::vector<int64_t> maskShape({ batch, blockMaskSize });
        std::vector<int64_t> coarseCentroidsShape({ utils::divUp(numLists, ZREGION_HEIGHT),
            utils::divUp(this->dim, CUBE_ALIGN), ZREGION_HEIGHT, CUBE_ALIGN });
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

    for (auto batch : SEARCH_BATCHES) {
        distComputeOps[batch] = std::unique_ptr<AscendOperator>(nullptr);
        APPERR_RETURN_IF_NOT_LOG(distCompOpReset(distComputeOps[batch], batch),
            APP_ERR_ACL_OP_LOAD_MODEL_FAILED, "ResetDistCompOp init failed");
    }
    return APP_ERR_OK;
}

APP_ERROR AscendIndexClusterImpl::ResetDistCompIdxOp()
{
    auto distCompIdxOpReset = [&](std::unique_ptr<AscendOperator> &op, int64_t batch) {
        AscendOpDesc desc("DistanceFlatIpByIdx");
        std::vector<int64_t> queryShape({ batch, this->dim });
        std::vector<int64_t> indexShape({ batch, IDX_BURST_LEN });
        std::vector<int64_t> sizeShape({ SIZE_ALIGN });
        std::vector<int64_t> coarseCentroidsShape(
            { utils::divUp(FAKE_HUGE_BASE, ZREGION_HEIGHT), utils::divUp(this->dim, CUBE_ALIGN),
            ZREGION_HEIGHT, CUBE_ALIGN });
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

APP_ERROR AscendIndexClusterImpl::ResetDistanceFlatIpByIdx2Op()
{
    APP_LOG_INFO("AscendIndexClusterImpl ResetDistanceFlatIpByIdx2Op operation started.\n");
    auto queryCopyOpReset = [&](std::unique_ptr<AscendOperator> &op, int64_t batch) {
        AscendOpDesc desc("DistanceFlatIpByIdx2");
        int thisDim = this->dim;
        std::vector<int64_t> queryOffset({batch});
        std::vector<int64_t> size({SIZE_ALIGN});
        std::vector<int64_t> shape(
            {utils::divUp(FAKE_HUGE_BASE, CUBE_ALIGN), utils::divUp(thisDim, CUBE_ALIGN), CUBE_ALIGN, CUBE_ALIGN});

        std::vector<int64_t> out({batch * FAKE_HUGE_BASE});

        desc.addInputTensorDesc(ACL_UINT32, queryOffset.size(), queryOffset.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_UINT32, size.size(), size.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_FLOAT16, shape.size(), shape.data(), ACL_FORMAT_ND);

        desc.addOutputTensorDesc(ACL_FLOAT, out.size(), out.data(), ACL_FORMAT_ND);
        op.reset();
        op = CREATE_UNIQUE_PTR(AscendOperator, desc);
        return op->init();
    };

    for (auto batch : this->computeBatchSizes) {
        queryCopyOps[batch] = std::unique_ptr<AscendOperator>(nullptr);
        APPERR_RETURN_IF_NOT_LOG(
            queryCopyOpReset(queryCopyOps[batch], batch), APP_ERR_ACL_OP_LOAD_MODEL_FAILED, "op init faild !!!");
    }
    APP_LOG_INFO("AscendIndexClusterImpl ResetDistanceFlatIpByIdx2Op operation end.\n");
    return APP_ERR_OK;
}

APP_ERROR AscendIndexClusterImpl::ResetDistanceFilterOp()
{
    APP_LOG_INFO("AscendIndexClusterImpl ResetDistanceFilterOp operation started.\n");
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
        op = CREATE_UNIQUE_PTR(AscendOperator, desc);
        return op->init();
    };

    for (auto batch : this->computeBatchSizes) {
        distanceFilterOps[batch] = std::unique_ptr<AscendOperator>(nullptr);
        APPERR_RETURN_IF_NOT_LOG(distanceFilterOpReset(distanceFilterOps[batch], batch),
            APP_ERR_ACL_OP_LOAD_MODEL_FAILED,
            "op init faild !!!");
    }
    APP_LOG_INFO("AscendIndexClusterImpl ResetDistanceFilterOp operation end.\n");
    return APP_ERR_OK;
}
} // namespace ascend
}