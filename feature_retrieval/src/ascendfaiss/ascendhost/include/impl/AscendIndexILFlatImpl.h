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


#ifndef ASCEND_INDEX_ILFLAT_IMPL_INCLUDED
#define ASCEND_INDEX_ILFLAT_IMPL_INCLUDED

#include <algorithm>
#include <vector>
#include <map>
#include "ascendhost/include/index/AscendIndexILFlat.h"
#include "ascenddaemon/AscendResourcesProxy.h"
#include "ascenddaemon/utils/AscendOperator.h"
#include "ascenddaemon/utils/AscendTensor.h"
#include "ascenddaemon/utils/DeviceVector.h"
#include "common/ErrorCode.h"

using ascend::APP_ERROR;
using ascend::AscendOperator;
using ascend::AscendResourcesProxy;
using ascend::AscendTensor;
using ascend::DIMS_1;
using ascend::DIMS_2;
using ascend::DIMS_3;
using ascend::DIMS_4;
using namespace ascend;
namespace faiss {
namespace ascend {

constexpr int SIZE_ALIGN = 8;
constexpr int CUBE_ALIGN = 16;
constexpr int BURST_LEN = 64;
constexpr int MIN_RESOURCE = 0x8000000; // 0x8000000 mean 128MB
constexpr int64_t MAX_RESOURCE = 0x100000000; // 0x100000000 mean 4096MB
constexpr int FAKE_HUGE_BASE = 20000000; // Fake size setted for tik operators
constexpr int MAX_CAP = 12000000; // Upper limit for capacity
constexpr size_t MAX_BASE_SPACE = 12288000000; // max bytes to store base vectors.
constexpr uint32_t MAX_THREAD_NUM = 32;
constexpr uint8_t BURST_LEN_HIGH = 64;
constexpr int MAX_TOPK = 1024;
constexpr int ACTUAL_TABLE_LEN = 10000;
constexpr size_t KB = 1024;
constexpr size_t RETAIN_SIZE = 2048;
constexpr size_t UNIT_PAGE_SIZE = 64;
constexpr size_t ADD_PAGE_SIZE = UNIT_PAGE_SIZE * KB * KB - RETAIN_SIZE;
class AscendIndexILFlatImpl {
public:
    ~AscendIndexILFlatImpl();

    AscendIndexILFlatImpl(int dim, int capacity, int deviceId, int64_t resourceSize);

    APP_ERROR Init();

    void Finalize();

    void ComputeDistWholeBaseWithTable(AscendTensor<float16_t, DIMS_2> &queryTensor,
        AscendTensor<uint32_t, DIMS_1> &sizeTensor, AscendTensor<float16_t, DIMS_4> &shapedData,
        AscendTensor<idx_t, DIMS_2> &idxTensor, AscendTensor<float, DIMS_2> &distanceTensor,
        AscendTensor<float, DIMS_1> &tableTensor, aclrtStream stream);
    
    void ComputeDistWholeBase(AscendTensor<float16_t, DIMS_2> &queryTensor,
        AscendTensor<uint32_t, DIMS_1> &sizeTensor, AscendTensor<float16_t, DIMS_4> &shapedData,
        AscendTensor<idx_t, DIMS_2> &idxTensor, AscendTensor<float, DIMS_2> &distanceTensor, aclrtStream stream);

    APP_ERROR SearchImpl(int n, const float16_t *queries, int topk, idx_t *indices, float *distances);

    APP_ERROR SearchImpl(int n, const float *queries, int topk, idx_t *indices, float *distances);

    APP_ERROR ComputeDistByIdxImpl(int n, const float16_t *queries, float *distances,
        std::tuple<int, const int *, const idx_t *> idxInfo, std::tuple<unsigned int, const float *> tableInfo);

    APP_ERROR ComputeDistByIdxImpl(int n, const float *queries, float *distances,
        std::tuple<int, const int *, const idx_t *> idxInfo, std::tuple<unsigned int, const float *> tableInfo);

    void ComputeBlockDist(AscendTensor<float16_t, DIMS_2> &queryTensor,
        AscendTensor<uint8_t, DIMS_2> &mask, AscendTensor<float16_t, DIMS_4> &shapedData,
        AscendTensor<uint32_t, DIMS_2> &size, AscendTensor<float16_t, DIMS_2> &outDistances,
        AscendTensor<float16_t, DIMS_2> &maxDistances, AscendTensor<float16_t, DIMS_2> &flag, aclrtStream stream);

    void runTopkCompute(AscendTensor<float16_t, DIMS_3> &dists, AscendTensor<float16_t, DIMS_3> &maxDists,
        AscendTensor<uint32_t, DIMS_3> &sizes, AscendTensor<float16_t, DIMS_3> &flags,
        AscendTensor<int64_t, DIMS_1> &attrs, AscendTensor<float16_t, DIMS_2> &outDists,
        AscendTensor<idx_t, DIMS_2> &outLabel, aclrtStream stream);

    APP_ERROR Remove(int n, const idx_t *indices);

    APP_ERROR Get(int n, float16_t *features, const idx_t *indices) const;

    APP_ERROR Get(int n, float *features, const idx_t *indices) const;

    APP_ERROR GetDevice(int n, float16_t *features, const idx_t *indices) const;

    APP_ERROR GetDevice(int n, float *features, const idx_t *indices) const;

    APP_ERROR Update(int n, const float16_t *features, const idx_t *indices);

    APP_ERROR Update(int n, const float *features, const idx_t *indices);

    void SetNTotal(int n);

    int GetNTotal() const;

    template <typename T>
    APP_ERROR Add(int n, const T *features)
    {
        APPERR_RETURN_IF_NOT_LOG(
            (isInitialized), APP_ERR_INVALID_PARAM, "Illegal operation, please initialize the index first. ");
        APPERR_RETURN_IF_NOT_FMT((n) > 0 && (n) <= this->capacity,
            APP_ERR_INVALID_PARAM,
            "The number n should be in range (0, %d]",
            this->capacity);
        APPERR_RETURN_IF_NOT_LOG(features, APP_ERR_INVALID_PARAM, "Features can not be nullptr.");

        std::vector<idx_t> indices(n);
        for (int i = 0; i < n; ++i) {
            indices[i] = static_cast<idx_t>(ntotal + i);
        }

        for (idx_t i = 0; i < static_cast<idx_t>(n); i++) {
            APPERR_RETURN_IF_NOT_FMT(static_cast<int>(indices[i]) < this->capacity,
                APP_ERR_INVALID_PARAM, "The indices[%ld](%u) should be in range [0, %d)",
                i, indices[i], this->capacity);
        }
        auto ret = aclrtSetDevice(deviceId);
        APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_ACL_SET_DEVICE_FAILED, "failed to set device(%d)", ret);

        size_t tileSize = ADD_PAGE_SIZE / (static_cast<size_t>(this->dim) * sizeof(float16_t));
        tileSize = utils::roundUp(tileSize, CUBE_ALIGN);

        int offset = 0;
        while (offset < n) {
            int copyNum = std::min((n - offset), static_cast<int>(tileSize));
            auto ret = AddPageFp16(copyNum,
                features + static_cast<uint64_t>(offset) * static_cast<uint64_t>(this->dim),
                indices[offset]);
            APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "failed to AddPage(%d)", ret);
            offset += copyNum;
        }
        return APP_ERR_OK;
    }

    template <typename T>
    APP_ERROR SearchByThreshold(int n, const T *queries, float threshold, int topk, int *num,
        idx_t *indice, float *distances, unsigned int tableLen = 0, const float *table = nullptr)
    {
        APPERR_RETURN_IF_NOT_LOG(
            (isInitialized), APP_ERR_INVALID_PARAM, "Illegal operation, please initialize the index first. ");
        APPERR_RETURN_IF_NOT_LOG(num, APP_ERR_INVALID_PARAM, "num can not be nullptr.");

        APP_ERROR ret = this->Search(n, queries, topk, indice, distances, tableLen, table);

        APPERR_RETURN_IF_NOT_FMT(ret == APP_ERR_OK, APP_ERR_INNER_ERROR, "AscendIndexILFlatImpl Search faild(%d)", ret);
        int validNum = std::min(topk, this->ntotal);
        for (int i = 0; i < n; i++) {
            int qnum = 0;
            for (int j = 0; j < validNum; j++) {
                int offset = i * topk + j;
                if (*(distances + offset) >= threshold) {
                    *(distances + i * topk + qnum) = *(distances + offset);
                    *(indice + i * topk + qnum) = *(indice + offset);
                    qnum += 1;
                }
            }
            num[i] = qnum;
        }
        return APP_ERR_OK;
    }

    template <typename T>
    APP_ERROR ComputeDistance(int n, const T *queries, float *distances,
        unsigned int tableLen = 0, const float *table = nullptr)
    {
        APPERR_RETURN_IF_NOT_LOG(
            (isInitialized), APP_ERR_INVALID_PARAM, "Illegal operation, please initialize the index first. ");
        auto ret = CheckComputeParams(n, queries, distances, tableLen, table);
        APPERR_RETURN_IF_NOT_FMT(ret == APP_ERR_OK, APP_ERR_INNER_ERROR,
            "AscendIndexILFlatImpl CheckComputeParams faild(%d)", ret);

        if (n == 1) {
            ret = ComputeDistImplBatched(n, queries, distances, tableLen, table);
            APPERR_RETURN_IF_NOT_FMT(ret == APP_ERR_OK, APP_ERR_INNER_ERROR,
                "AscendIndexILFlatImpl ComputeDistImplBatched faild(%d)", ret);
            return APP_ERR_OK;
        }

        size_t size = this->computeBatchSizes.size();
        int padNtotal = utils::divUp(this->ntotal, CUBE_ALIGN) * CUBE_ALIGN;
        size_t searched = 0;
        for (size_t i = 0; i < size; i++) {
            size_t batchSize = static_cast<size_t>(this->computeBatchSizes[i]);
            if ((static_cast<size_t>(n) - searched) >= batchSize) {
                size_t batchNum = (static_cast<size_t>(n) - searched) / batchSize;
                for (size_t j = 0; j < batchNum; j++) {
                    ret = ComputeDistImplBatched(batchSize, queries + searched * this->dim,
                        distances + searched * padNtotal, tableLen, table);
                    APPERR_RETURN_IF_NOT_FMT(ret == APP_ERR_OK, APP_ERR_INNER_ERROR,
                        "AscendIndexILFlatImpl ComputeDistImplBatched faild(%d)", ret);
                    searched += batchSize;
                }
            }
        }
        return APP_ERR_OK;
    }

    template <typename T>
    APP_ERROR ComputeDistanceByIdx(int n, const T *queries, const int *num, const idx_t *indices, float *distances,
        MEMORY_TYPE memoryType, unsigned int tableLen = 0, const float *table = nullptr)
    {
        APPERR_RETURN_IF_NOT_LOG(
            (isInitialized), APP_ERR_INVALID_PARAM, "Illegal operation, please initialize the index first. ");
        auto res = CheckComputeParams(n, queries, distances, tableLen, table);
        APPERR_RETURN_IF_NOT_FMT(res == APP_ERR_OK, APP_ERR_INNER_ERROR,
            "AscendIndexILFlatImpl CheckComputeParams faild(%d)", res);
        APPERR_RETURN_IF_NOT_LOG(num, APP_ERR_INVALID_PARAM, "num can not be nullptr.");
        APPERR_RETURN_IF_NOT_LOG(indices, APP_ERR_INVALID_PARAM, "indices can not be nullptr.");

        int maxNum = 0;
        res = CheckComputeMaxNum(n, num, indices, maxNum);
        APPERR_RETURN_IF_NOT_FMT(res == APP_ERR_OK, APP_ERR_INNER_ERROR,
            "AscendIndexILFlatImpl CheckComputeMaxNum faild(%d)", res);

        if (memoryType == MEMORY_TYPE::INPUT_DEVICE_OUTPUT_DEVICE ||
            memoryType == MEMORY_TYPE::INPUT_DEVICE_OUTPUT_HOST) {
            isInputDevice = true;
        }
        if (memoryType == MEMORY_TYPE::INPUT_DEVICE_OUTPUT_DEVICE ||
            memoryType == MEMORY_TYPE::INPUT_HOST_OUTPUT_DEVICE) {
            isOutputDevice = true;
        }

        std::tuple<unsigned int, const float *> tableInfo(tableLen, table);
        std::tuple<int, const int *, const idx_t *> idxInfo;
        if (n == 1) {
            idxInfo = std::make_tuple(maxNum, num, indices);
            auto ret = ComputeDistByIdxImpl(n, queries, distances, idxInfo, tableInfo);
            APPERR_RETURN_IF_NOT_FMT(ret == APP_ERR_OK, APP_ERR_INNER_ERROR,
                "AscendIndexILFlatImpl ComputeDistByIdxImpl faild(%d)", ret);
            return APP_ERR_OK;
        }
        int size = static_cast<int>(computeByIdxBatchSizes.size());
        int searched = 0;
        for (int i = 0; i < size; i++) {
            int batchSize = computeByIdxBatchSizes[i];
            if ((n - searched) >= batchSize) {
                int batchNum = (n - searched) / batchSize;
                for (int j = 0; j < batchNum; j++) {
                    idxInfo = std::make_tuple(maxNum, num + searched, indices + searched * maxNum);
                    auto ret = ComputeDistByIdxImpl(batchSize, queries + searched * this->dim,
                        distances + searched * maxNum, idxInfo, tableInfo);
                    APPERR_RETURN_IF_NOT_FMT(ret == APP_ERR_OK, APP_ERR_INNER_ERROR,
                        "AscendIndexILFlatImpl ComputeDistByIdxImpl faild(%d)", ret);
                    searched += batchSize;
                }
            }
        }
        return APP_ERR_OK;
    }

    template <typename T>
    APP_ERROR Search(int n, const T *queries, int topk,
        idx_t *indice, float *distances, unsigned int tableLen, const float *table)
    {
        APP_LOG_INFO("AscendIndexILFlatImpl Search operation start.\n");
        APPERR_RETURN_IF_NOT_LOG(
            (isInitialized), APP_ERR_INVALID_PARAM, "Illegal operation, please initialize the index first. ");

        auto res = CheckSearchParams(n, queries, topk, indice, distances, tableLen, table);
        APPERR_RETURN_IF_NOT_FMT(res == APP_ERR_OK, APP_ERR_INVALID_PARAM,
            "AscendIndexILFlatImpl CheckSearchParams faild(%d)", res);

        if (n == 1) {
            auto ret = SearchImpl(n, queries, topk, indice, distances);
            APPERR_RETURN_IF_NOT_FMT(ret == APP_ERR_OK, APP_ERR_INNER_ERROR,
                "AscendIndexILFlatImpl SearchImpl faild(%d)", ret);
            return this->TableMapping(n, distances, tableLen, table, topk);
        }

        size_t size = searchBatchSizes.size();
        int searched = 0;
        for (size_t i = 0; i < size; i++) {
            int batchSize = searchBatchSizes[i];
            if ((n - searched) >= batchSize) {
                int page = (n - searched) / batchSize;
                for (int j = 0; j < page; j++) {
                    auto ret = SearchImpl(batchSize, queries + searched * this->dim, topk, indice +
                        searched * topk, distances + searched * topk);
                    APPERR_RETURN_IF_NOT_FMT(ret == APP_ERR_OK, APP_ERR_INNER_ERROR,
                        "AscendIndexILFlatImpl SearchImpl faild(%d)", ret);
                    searched += batchSize;
                }
            }
        }
        APP_LOG_INFO("AscendIndexILFlatImpl Search operation end.\n");
        return this->TableMapping(n, distances, tableLen, table, topk);
    }

    template <typename T>
    APP_ERROR CheckSearchParams(int n, const T *queries, int topk, idx_t *indice, float *distances,
        unsigned int tableLen, const float *table)
    {
        APPERR_RETURN_IF_NOT_FMT(n > 0 && n <= this->capacity, APP_ERR_INVALID_PARAM,
            "The number n should be in range (0, %d]", this->capacity);
        APPERR_RETURN_IF_NOT_LOG(queries, APP_ERR_INVALID_PARAM, "queries can not be nullptr.");
        APPERR_RETURN_IF_NOT_LOG(topk > 0 && topk <= MAX_TOPK, APP_ERR_INVALID_PARAM,
            "Invalid parameter, topk should be in range (0, 1024].");
        APPERR_RETURN_IF_NOT_LOG(indice, APP_ERR_INVALID_PARAM, "indice can not be nullptr.");
        APPERR_RETURN_IF_NOT_LOG(distances, APP_ERR_INVALID_PARAM, "distances can not be nullptr.");
        if (tableLen > 0) {
            APPERR_RETURN_IF_NOT_FMT(tableLen == ACTUAL_TABLE_LEN, APP_ERR_INVALID_PARAM,
                "table length only support %d", ACTUAL_TABLE_LEN);
            APPERR_RETURN_IF_NOT_LOG(table, APP_ERR_INVALID_PARAM,
                "The table pointer cannot be nullptr when tableLen is valid.");
        }
        return APP_ERR_OK;
    }

    template <typename T>
    APP_ERROR CheckComputeParams(int n, const T *queries, float *distances,
        unsigned int tableLen, const float *table)
    {
        APPERR_RETURN_IF_NOT_FMT(n > 0 && n <= this->capacity, APP_ERR_INVALID_PARAM,
            "The number n should be in range (0, %d]", this->capacity);
        APPERR_RETURN_IF_NOT_LOG(queries, APP_ERR_INVALID_PARAM, "queries can not be nullptr.");
        APPERR_RETURN_IF_NOT_LOG(distances, APP_ERR_INVALID_PARAM, "distances can not be nullptr.");
        if (tableLen > 0) {
            APPERR_RETURN_IF_NOT_FMT(tableLen == ACTUAL_TABLE_LEN, APP_ERR_INVALID_PARAM,
                "table length only support %d", ACTUAL_TABLE_LEN);
            APPERR_RETURN_IF_NOT_LOG(table, APP_ERR_INVALID_PARAM,
                "The table pointer cannot be nullptr when tableLen is valid.");
        }
        return APP_ERR_OK;
    }

protected:
    int dim;
    int pageSize{0};
    int capacity;
    int blockSize{0};
    int deviceId;
    int64_t resourceSize;
    int ntotal{0};
    bool isInitialized;
    int blockMaskSize{0};
    std::vector<int> computeBatchSizes;
    std::vector<int> searchBatchSizes;
    std::vector<int> computeByIdxBatchSizes;
    size_t fakeBaseSizeInBytes{0};
    bool isInputDevice{false};
    bool isOutputDevice{false};
    std::unique_ptr<AscendResourcesProxy> pResources;
    std::unique_ptr<::ascend::DeviceVector<float16_t>> baseSpace;
    std::map<int, std::unique_ptr<AscendOperator>> queryCopyOps;
    std::map<int, std::unique_ptr<AscendOperator>> distComputeOps;
    std::map<int, std::unique_ptr<AscendOperator>> distanceFlatIpOps;
    std::map<int, std::unique_ptr<AscendOperator>> distComputeIdxOps;
    std::map<int, std::unique_ptr<AscendOperator>> distComputeIdxWithTableOps;
    std::map<int, std::unique_ptr<AscendOperator>> distanceFlatIpWithTableOps;
    std::map<int, std::unique_ptr<::ascend::AscendOperator>> topkComputeOps;

private:
    size_t GetMaxCapacity() const;

    APP_ERROR AddPageFp16(int num, const float16_t *features, idx_t indice);

    APP_ERROR AddPageFp16(int num, const float *features, idx_t indice);

    APP_ERROR SearchPaged(int pageId, AscendTensor<float16_t, DIMS_2> &queries, int pageNum,
        AscendTensor<idx_t, DIMS_2> &idxs, AscendTensor<float16_t, DIMS_2> &distances);

    APP_ERROR searchImplFp16(int n, const float16_t *query, int topk, idx_t *indice, float *distances);

    APP_ERROR TableMapping(int n, float *distances, size_t tableLen, const float *table, int topk);

    APP_ERROR ComputeDistImplBatchedFp16(int n, const float16_t *queries, float *distances,
        unsigned int tableLen, const float *table);

    APP_ERROR ComputeDistImplBatched(int n, const float16_t *queries, float *distances,
        unsigned int tableLen, const float *table);

    APP_ERROR ComputeDistImplBatched(int n, const float *queries, float *distances,
        unsigned int tableLen, const float *table);

    void ComputeBlockDistWholeBase(AscendTensor<float16_t, DIMS_2> &queryTensor,
        AscendTensor<float16_t, DIMS_4> &shapedData, AscendTensor<uint32_t, DIMS_1> &baseSize,
        AscendTensor<float, DIMS_1> &distanceTensor, aclrtStream stream);

    void ComputeBlockDistWholeBaseWithTable(AscendTensor<float16_t, DIMS_2> &queryTensor,
        AscendTensor<float16_t, DIMS_4> &shapedData, AscendTensor<uint32_t, DIMS_1> &baseSize,
        AscendTensor<float, DIMS_1> &tableTensor, AscendTensor<float, DIMS_1> &distanceTensor, aclrtStream stream);

    APP_ERROR CheckComputeMaxNum(int n, const int *num, const idx_t *indices, int& maxNum);

    APP_ERROR ComputeDistByIdxImplFp16(int n, const float16_t *queries, float *distances,
        std::tuple<int, const int *, const idx_t *> idxInfo, std::tuple<unsigned int, const float *> tableInfo);

    APP_ERROR ResetDistCompIdxWithTableOp();
    APP_ERROR ResetDistCompIdxOp();
    APP_ERROR ResetTopkCompOp();
    APP_ERROR ResetDistCompOp(int numLists);
    APP_ERROR ResetDistanceWithTableOp();
    APP_ERROR ResetDistanceFlatIpOp();

    APP_ERROR UpdateFp16(int n, const float16_t *features, const idx_t *indices);
    APP_ERROR InitOps();
    APP_ERROR InitAttr(int topk, int burstLen, int blockNum, int pageId, int pageNum,
        AscendTensor<int64_t, DIMS_1>& attrsInput);
    APP_ERROR CopyDistData(int maxNum, int idxCopyNum, float *distances, int n,
        AscendTensor<float, DIMS_3>& distResult);
    APP_ERROR MoveData(int& idxCopyNum, int maxNum, int n, const idx_t *indice,
        AscendTensor<idx_t, DIMS_3>& idxTensor);
};
}
}
#endif
