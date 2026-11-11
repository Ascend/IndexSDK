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

#include <cmath>
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
const std::vector<int> VALID_DIM = {64, 128, 256, 512, 1024, 1536, 3072};
const std::vector<int> VALID_GRAPH_DEGREE = {64, 128};
const std::vector<int> VALID_TOPK = {32, 64, 128, 256, 512, 1024};
constexpr int64_t MAX_DATA_NUM = 1000000000L;
constexpr int MAX_BATCH_SIZE = 4096;
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

APP_ERROR AscendIndexCagra::Init(int dim, const std::vector<int>& deviceList, const AscendIndexCagraConfig& config)
{
    std::lock_guard<std::mutex> lock(mtx);
    APPERR_RETURN_IF_NOT_LOG(pIndexCagraImpl == nullptr, APP_ERR_INVALID_PARAM, "CAGRA is already initialized");
    APPERR_RETURN_IF_NOT_LOG(std::find(VALID_DIM.begin(), VALID_DIM.end(), dim) != VALID_DIM.end(),
                             APP_ERR_INVALID_PARAM, "Unsupported CAGRA dimension");
    APPERR_RETURN_IF_NOT_LOG(deviceList.size() == 1 && deviceList[0] >= 0, APP_ERR_INVALID_PARAM,
                             "CAGRA requires one nonnegative device ID");
    APPERR_RETURN_IF_NOT_LOG((config.graphDegree == 64 || config.graphDegree == 128) &&
                                 config.graphDegree <= config.intermediateDegree && config.intermediateDegree <= 128 &&
                                 config.maxIterations > 0 && std::isfinite(config.terminationThreshold) &&
                                 config.terminationThreshold >= 0.0F && config.terminationThreshold <= 1.0F,
                             APP_ERR_INVALID_PARAM, "Invalid CAGRA build configuration");
    try
    {
        auto candidate = std::make_shared<AscendIndexCagraImpl>(dim, 0, deviceList);
        candidate->Configure(config);
        pIndexCagraImpl = std::move(candidate);
    }
    catch (const std::exception& e)
    {
        APP_LOG_ERROR("CAGRA Init failed: %s", e.what());
        return APP_ERR_INNER_ERROR;
    }
    return APP_ERR_OK;
}

APP_ERROR AscendIndexCagra::Train(int n, const float* data)
{
    std::lock_guard<std::mutex> lock(mtx);
    return TrainLocked(n, data, false);
}

APP_ERROR AscendIndexCagra::TrainLocked(int n, const float* data, bool rejectTrained)
{
    APPERR_RETURN_IF_NOT_LOG(pIndexCagraImpl != nullptr && pIndexCagraImpl->IsUnified(), APP_ERR_INVALID_PARAM,
                             "Train requires the unified Init overload");
    APPERR_RETURN_IF_NOT_LOG(data != nullptr && n > 1 && n <= MAX_DATA_NUM, APP_ERR_INVALID_PARAM,
                             "Invalid CAGRA training data");
    if (pIndexCagraImpl->IsTrained()) return rejectTrained ? APP_ERR_INVALID_PARAM : APP_ERR_OK;
    try
    {
        auto candidate = pIndexCagraImpl->CreateEmpty();
        const auto ret = candidate->Train(n, data);
        if (ret != APP_ERR_OK) return ret;
        pIndexCagraImpl.swap(candidate);
    }
    catch (const std::exception& e)
    {
        APP_LOG_ERROR("CAGRA Train failed: %s", e.what());
        return APP_ERR_INNER_ERROR;
    }
    return APP_ERR_OK;
}

APP_ERROR AscendIndexCagra::Add(int n, const float* data)
{
    std::lock_guard<std::mutex> lock(mtx);
    return TrainLocked(n, data, true);
}

APP_ERROR AscendIndexCagra::Reset()
{
    std::lock_guard<std::mutex> lock(mtx);
    try
    {
        auto empty = pIndexCagraImpl != nullptr && pIndexCagraImpl->IsUnified()
                         ? pIndexCagraImpl->CreateEmpty()
                         : std::shared_ptr<AscendIndexCagraImpl>();
        pIndexCagraImpl.swap(empty);
    }
    catch (const std::exception& e)
    {
        APP_LOG_ERROR("CAGRA Reset failed: %s", e.what());
        return APP_ERR_INNER_ERROR;
    }
    return APP_ERR_OK;
}

bool AscendIndexCagra::IsTrained() const
{
    std::lock_guard<std::mutex> lock(mtx);
    return pIndexCagraImpl != nullptr && pIndexCagraImpl->IsTrained();
}

int AscendIndexCagra::GetSize() const
{
    std::lock_guard<std::mutex> lock(mtx);
    return pIndexCagraImpl == nullptr ? 0 : pIndexCagraImpl->GetSize();
}

APP_ERROR AscendIndexCagra::GetKnnGraph(std::vector<uint32_t>& graph) const
{
    std::lock_guard<std::mutex> lock(mtx);
    APPERR_RETURN_IF_NOT_LOG(pIndexCagraImpl != nullptr && pIndexCagraImpl->IsTrained(), APP_ERR_INDEX_NOT_INIT,
                             "CAGRA is not trained");
    try
    {
        return pIndexCagraImpl->GetKnnGraph(graph);
    }
    catch (const std::exception& e)
    {
        APP_LOG_ERROR("CAGRA graph download failed: %s", e.what());
        return APP_ERR_INNER_ERROR;
    }
}

APP_ERROR AscendIndexCagra::SaveGraphToFile(const std::string& filePath) const
{
    std::lock_guard<std::mutex> lock(mtx);
    APPERR_RETURN_IF_NOT_LOG(!filePath.empty() && filePath.find('\0') == std::string::npos, APP_ERR_INVALID_PARAM,
                             "Invalid CAGRA graph file path");
    APPERR_RETURN_IF_NOT_LOG(pIndexCagraImpl != nullptr && pIndexCagraImpl->IsTrained(), APP_ERR_INDEX_NOT_INIT,
                             "CAGRA is not trained");
    try
    {
        return pIndexCagraImpl->SaveGraphToFile(filePath);
    }
    catch (const std::exception& e)
    {
        APP_LOG_ERROR("CAGRA graph export failed: %s", e.what());
        return APP_ERR_INNER_ERROR;
    }
}

APP_ERROR AscendIndexCagra::Init(int dim, int graphDegree, int dataNum, int topK, const std::vector<int>& deviceList)
{
    std::lock_guard<std::mutex> lock(mtx);
    APPERR_RETURN_IF_NOT_LOG(pIndexCagraImpl == nullptr, APP_ERR_INVALID_PARAM, "pIndexCagraImpl is not nullptr");
    APPERR_RETURN_IF_NOT_LOG(!deviceList.empty(), APP_ERR_INVALID_PARAM, "deviceList cannot be empty");
    APPERR_RETURN_IF_NOT_FMT(std::find(VALID_DIM.begin(), VALID_DIM.end(), dim) != VALID_DIM.end(),
                             APP_ERR_INVALID_PARAM,
                             "Invalid dimension %d, should be in {64, 128, 256, 512, 1024, 1536, 3072}", dim);
    APPERR_RETURN_IF_NOT_FMT(
        std::find(VALID_GRAPH_DEGREE.begin(), VALID_GRAPH_DEGREE.end(), graphDegree) != VALID_GRAPH_DEGREE.end(),
        APP_ERR_INVALID_PARAM, "Invalid graph degree %d, should be in {64, 128}", graphDegree);
    APPERR_RETURN_IF_NOT_FMT(std::find(VALID_TOPK.begin(), VALID_TOPK.end(), topK) != VALID_TOPK.end(),
                             APP_ERR_INVALID_PARAM, "Unsupported topK %d; expected {32,64,128,256,512,1024}", topK);
    APPERR_RETURN_IF_NOT_LOG(deviceList.size() == 1 && deviceList[0] >= 0, APP_ERR_INVALID_PARAM,
                             "Only 1 nonnegative chip ID supported for Ascend CAGRA");
    APPERR_RETURN_IF_NOT_FMT(dataNum > graphDegree && dataNum <= MAX_DATA_NUM && topK <= dataNum, APP_ERR_INVALID_PARAM,
                             "Invalid dataNum %d: require graphDegree (%d) < dataNum <= %lld and topK (%d) <= dataNum",
                             dataNum, graphDegree, static_cast<long long>(MAX_DATA_NUM), topK);

    try
    {
        auto candidate = std::make_shared<AscendIndexCagraImpl>(dim, topK, deviceList);
        const auto ret = candidate->Init(graphDegree, dataNum);
        if (ret != APP_ERR_OK) return ret;
        pIndexCagraImpl.swap(candidate);
    }
    catch (const std::exception& e)
    {
        APP_LOG_ERROR("Legacy CAGRA Init failed: %s", e.what());
        return APP_ERR_INNER_ERROR;
    }

    return APP_ERR_OK;
}

APP_ERROR AscendIndexCagra::Add(const uint32_t* graph, const uint32_t* hash, const float* data)
{
    std::lock_guard<std::mutex> lock(mtx);
    APPERR_RETURN_IF_NOT_LOG(pIndexCagraImpl != nullptr, APP_ERR_INVALID_PARAM, "pIndexCagraImpl is nullptr");
    APPERR_RETURN_IF_NOT_LOG(!pIndexCagraImpl->IsUnified(), APP_ERR_INVALID_PARAM,
                             "External graph Add requires legacy Init");
    APPERR_RETURN_IF_NOT_LOG(graph, APP_ERR_INVALID_PARAM, "graph cannot be nullptr");
    APPERR_RETURN_IF_NOT_LOG(hash, APP_ERR_INVALID_PARAM, "hash cannot be nullptr");
    APPERR_RETURN_IF_NOT_LOG(data, APP_ERR_INVALID_PARAM, "data cannot be nullptr");
    return this->pIndexCagraImpl->Add(graph, hash, data);
}

APP_ERROR AscendIndexCagra::QuantizeData(int n, const float* queryData, int ntotal, const float* baseData)
{
    std::lock_guard<std::mutex> lock(mtx);
    APPERR_RETURN_IF_NOT_LOG(pIndexCagraImpl != nullptr, APP_ERR_INVALID_PARAM, "pIndexCagraImpl is nullptr");
    APPERR_RETURN_IF_NOT_LOG(!pIndexCagraImpl->IsUnified(), APP_ERR_INVALID_PARAM,
                             "Manual quantization requires legacy Init");
    APPERR_RETURN_IF_NOT_FMT(n > 0 && n <= MAX_BATCH_SIZE, APP_ERR_INVALID_PARAM, "n %d, must be in range (0, %d]", n,
                             MAX_BATCH_SIZE);
    APPERR_RETURN_IF_NOT_LOG(queryData, APP_ERR_INVALID_PARAM, "queryData cannot be nullptr");
    APPERR_RETURN_IF_NOT_LOG(baseData, APP_ERR_INVALID_PARAM, "baseData cannot be nullptr");
    return this->pIndexCagraImpl->QuantizeData(n, queryData, ntotal, baseData);
}

APP_ERROR AscendIndexCagra::QuantizeQueries(int n, const float* queryData)
{
    std::lock_guard<std::mutex> lock(mtx);
    APPERR_RETURN_IF_NOT_LOG(pIndexCagraImpl != nullptr, APP_ERR_INVALID_PARAM, "pIndexCagraImpl is nullptr");
    APPERR_RETURN_IF_NOT_LOG(!pIndexCagraImpl->IsUnified(), APP_ERR_INVALID_PARAM,
                             "Manual quantization requires legacy Init");
    APPERR_RETURN_IF_NOT_FMT(n > 0 && n <= MAX_BATCH_SIZE, APP_ERR_INVALID_PARAM, "n %d, must be in range (0, %d]", n,
                             MAX_BATCH_SIZE);
    APPERR_RETURN_IF_NOT_LOG(queryData, APP_ERR_INVALID_PARAM, "queryData cannot be nullptr");
    return this->pIndexCagraImpl->QuantizeQueries(n, queryData);
}

APP_ERROR AscendIndexCagra::Search(int n, const float* queryData, int topK, float* dists, uint32_t* labels)
{
    std::lock_guard<std::mutex> lock(mtx);
    APPERR_RETURN_IF_NOT_LOG(pIndexCagraImpl != nullptr, APP_ERR_INVALID_PARAM, "pIndexCagraImpl is nullptr");
    APPERR_RETURN_IF_NOT_FMT(n > 0 && n <= MAX_BATCH_SIZE, APP_ERR_INVALID_PARAM, "n %d, must be in range (0, %d]", n,
                             MAX_BATCH_SIZE);
    APPERR_RETURN_IF_NOT_FMT(std::find(VALID_TOPK.begin(), VALID_TOPK.end(), topK) != VALID_TOPK.end(),
                             APP_ERR_INVALID_PARAM, "Unsupported topK %d; expected {32,64,128,256,512,1024}", topK);
    APPERR_RETURN_IF_NOT_LOG(queryData, APP_ERR_INVALID_PARAM, "queryData cannot be nullptr");
    APPERR_RETURN_IF_NOT_LOG(dists, APP_ERR_INVALID_PARAM, "dists cannot be nullptr");
    APPERR_RETURN_IF_NOT_LOG(labels, APP_ERR_INVALID_PARAM, "labels cannot be nullptr");

    try
    {
        return this->pIndexCagraImpl->Search(n, queryData, topK, dists, labels);
    }
    catch (const std::exception& e)
    {
        APP_LOG_ERROR("CAGRA Search failed: %s", e.what());
        return APP_ERR_INNER_ERROR;
    }
}
}  // namespace ascend
}  // namespace faiss
