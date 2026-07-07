/*
 * -------------------------------------------------------------------------
 * This file is part of the IndexSDK project.
 * Copyright (c) 2026 Huawei Technologies Co.,Ltd.
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

#include "index/AscendIndexCagra.h"

#include <mutex>

#include "common/ErrorCode.h"
#include "impl/AscendIndexCagraImpl.h"
using namespace ascend;
namespace faiss
{
namespace ascend
{
namespace
{
const std::vector<int> VALID_DIM = {64, 128, 256, 512};
const std::vector<int> VALID_GRAPH_DEGREE = {64, 128, 256, 512};
constexpr int64_t MAX_DATA_NUM = 1000000000L;
constexpr int MIN_TOPK = 0;
constexpr int MAX_TOPK = 128;
}  // namespace

AscendIndexCagra::AscendIndexCagra() { pIndexCagraImpl = nullptr; }

AscendIndexCagra::~AscendIndexCagra()
{
    std::lock_guard<std::mutex> lock(mtx);
    if (this->pIndexCagraImpl != nullptr)
    {
        this->pIndexCagraImpl.reset();
        this->pIndexCagraImpl = nullptr;
    }
}

APP_ERROR AscendIndexCagra::Init(int dim, int graphDegree, int dataNum, int topK, const std::vector<int>& deviceList)
{
    std::lock_guard<std::mutex> lock(mtx);
    APPERR_RETURN_IF_NOT_LOG(pIndexCagraImpl == nullptr, APP_ERR_INVALID_PARAM, "pIndexCagraImpl is not nullptr");
    APPERR_RETURN_IF_NOT_LOG(!deviceList.empty(), APP_ERR_INVALID_PARAM, "deviceList cannot be empty");
    APPERR_RETURN_IF_NOT_FMT(std::find(VALID_DIM.begin(), VALID_DIM.end(), dim) != VALID_DIM.end(),
                             APP_ERR_INVALID_PARAM, "Invalid dimension %d, should be in {64, 128, 256, 512}", dim);
    APPERR_RETURN_IF_NOT_FMT(
        std::find(VALID_GRAPH_DEGREE.begin(), VALID_GRAPH_DEGREE.end(), graphDegree) != VALID_GRAPH_DEGREE.end(),
        APP_ERR_INVALID_PARAM, "Invalid graph degree %d, should be in {64, 128, 256, 512}", graphDegree);
    APPERR_RETURN_IF_NOT_FMT(topK > MIN_TOPK && topK <= MAX_TOPK, APP_ERR_INVALID_PARAM,
                             "topK %d, must be in range (%d, %d]", topK, MIN_TOPK, MAX_TOPK);
    APPERR_RETURN_IF_NOT_LOG(deviceList.size() == 1, APP_ERR_INVALID_PARAM, "Only 1 chip supported for Ascend CAGRA");
    APPERR_RETURN_IF_NOT_FMT(dataNum >= 0 && dataNum <= MAX_DATA_NUM, APP_ERR_INVALID_PARAM,
                             "Invalid dataNum %d, should be in range [0, %lld]", dataNum, MAX_DATA_NUM);

    try
    {
        this->pIndexCagraImpl = std::make_shared<AscendIndexCagraImpl>(dim, topK, deviceList);
    }
    catch (const std::bad_alloc& e)
    {
        APPERR_RETURN_IF_NOT_LOG(false, APP_ERR_INNER_ERROR, "Failed to allocate AscendIndexCagraImpl");
    }

    try
    {
        this->pIndexCagraImpl->Init(graphDegree, dataNum);
    }
    catch (const std::bad_alloc& e)
    {
        APPERR_RETURN_IF_NOT_LOG(false, APP_ERR_INNER_ERROR, "Failed to init");
    }

    return APP_ERR_OK;
}

APP_ERROR AscendIndexCagra::Add(const uint32_t* graph, const uint32_t* hash, const float* data)
{
    std::lock_guard<std::mutex> lock(mtx);
    APPERR_RETURN_IF_NOT_LOG(pIndexCagraImpl != nullptr, APP_ERR_INVALID_PARAM, "pIndexCagraImpl is nullptr");
    APPERR_RETURN_IF_NOT_LOG(graph, APP_ERR_INVALID_PARAM, "graph cannot be nullptr");
    APPERR_RETURN_IF_NOT_LOG(hash, APP_ERR_INVALID_PARAM, "hash cannot be nullptr");
    APPERR_RETURN_IF_NOT_LOG(data, APP_ERR_INVALID_PARAM, "data cannot be nullptr");
    return this->pIndexCagraImpl->Add(graph, hash, data);
}

APP_ERROR AscendIndexCagra::QuantizeData(int n, const float* queryData, int ntotal, const float* baseData)
{
    std::lock_guard<std::mutex> lock(mtx);
    APPERR_RETURN_IF_NOT_LOG(pIndexCagraImpl != nullptr, APP_ERR_INVALID_PARAM, "pIndexCagraImpl is nullptr");
    APPERR_RETURN_IF_NOT_LOG(queryData, APP_ERR_INVALID_PARAM, "queryData cannot be nullptr");
    APPERR_RETURN_IF_NOT_LOG(baseData, APP_ERR_INVALID_PARAM, "baseData cannot be nullptr");
    return this->pIndexCagraImpl->QuantizeData(n, queryData, ntotal, baseData);
}

APP_ERROR AscendIndexCagra::Search(int n, const float* queryData, int topK, float* dists, uint32_t* labels)
{
    std::lock_guard<std::mutex> lock(mtx);
    APPERR_RETURN_IF_NOT_LOG(pIndexCagraImpl != nullptr, APP_ERR_INVALID_PARAM, "pIndexCagraImpl is nullptr");
    APPERR_RETURN_IF_NOT_FMT(topK > MIN_TOPK && topK <= MAX_TOPK, APP_ERR_INVALID_PARAM,
                             "topK %d, must be in range (%d, %d]", topK, MIN_TOPK, MAX_TOPK);
    APPERR_RETURN_IF_NOT_LOG(queryData, APP_ERR_INVALID_PARAM, "queryData cannot be nullptr");
    APPERR_RETURN_IF_NOT_LOG(dists, APP_ERR_INVALID_PARAM, "dists cannot be nullptr");
    APPERR_RETURN_IF_NOT_LOG(labels, APP_ERR_INVALID_PARAM, "labels cannot be nullptr");

    return this->pIndexCagraImpl->Search(n, queryData, topK, dists, labels);
}
}  // namespace ascend
}  // namespace faiss
