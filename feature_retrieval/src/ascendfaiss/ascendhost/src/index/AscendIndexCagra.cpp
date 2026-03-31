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

#include "index/AscendIndexCagra.h"
#include "impl/AscendIndexCagraImpl.h"
#include "common/ErrorCode.h"
#include <mutex>
using namespace ascend;
namespace faiss {
namespace ascend {
const int64_t BIG_BATCH_MEM_SIZE_THRESHOLD = static_cast<int64_t>(4) * 1024 * 1024 * 1024;
// 构造函数实现
AscendIndexCagra::AscendIndexCagra()
{
}

AscendIndexCagra::~AscendIndexCagra()
{
}

APP_ERROR AscendIndexCagra::Init(const IndexCagraInitParams& params, const IndexCagraSearchParams& searchParams)
{
    APPERR_RETURN_IF_NOT_LOG(pIndexCagraImpl == nullptr, APP_ERR_INVALID_PARAM, "pIndexCagraImpl is not null");
    APPERR_RETURN_IF_NOT_LOG(!params.deviceList.empty(), APP_ERR_INVALID_PARAM, "deviceList cannot be empty");
    APPERR_RETURN_IF_NOT_FMT(params.dim >= 32 && params.dim <= 512, APP_ERR_INVALID_PARAM,
        "Invalid dimension %d, should be in range [32, 512]", params.dim);
    APPERR_RETURN_IF_NOT_FMT(params.graph_degree >= 32 && params.graph_degree <= 512,
        APP_ERR_INVALID_PARAM, "Invalid graph degree %d, should be in range [32, 512]", params.graph_degree);
    APPERR_RETURN_IF_NOT_LOG(params.deviceList.size() == 1, APP_ERR_INVALID_PARAM, 
        "Only 1 chip supported for Ascend CAGRA");
    APPERR_RETURN_IF_NOT_FMT((params.ascendResourceSize >= 0 && params.ascendResourceSize <= BIG_BATCH_MEM_SIZE_THRESHOLD),
        APP_ERR_INVALID_PARAM, "resources %lld, should be in range [0, 4GB]!", params.ascendResourceSize);
    APPERR_RETURN_IF_NOT_FMT(searchParams.dataNum >= 0 && searchParams.dataNum <= 1e9,
        APP_ERR_INVALID_PARAM, "Invalid dataNum %d, should be in range [0, 1e9]]", searchParams.dataNum);
    APPERR_RETURN_IF_NOT_FMT(searchParams.hashBitlen >= 16 && searchParams.hashBitlen <= 65536,
        APP_ERR_INVALID_PARAM, "Invalid hashBitlen %d, should be in range [16, 65536]", searchParams.hashBitlen);

    this->pIndexCagraImpl = std::make_shared<AscendIndexCagraImpl>(params);
    APPERR_RETURN_IF_NOT_LOG(this->pIndexCagraImpl != nullptr, APP_ERR_INNER_ERROR,
        "Failed to create AscendIndexCagraImpl");

    return this->pIndexCagraImpl->Init(params, searchParams);
}

// 添加图结构
APP_ERROR AscendIndexCagra::AddGraph(const std::vector<uint32_t>& graphData, const std::string& saveBinPath)
{
    APPERR_RETURN_IF_NOT_LOG(pIndexCagraImpl != nullptr, APP_ERR_INVALID_PARAM, "pIndexCagraImpl is null");
    APPERR_RETURN_IF_NOT_LOG(!graphData.empty(), APP_ERR_INVALID_PARAM, "graphData cannot be empty");
    
    return this->pIndexCagraImpl->AddGraph(graphData, saveBinPath);
}

// 检索接口
APP_ERROR AscendIndexCagra::Search(int n, const float* queryData, int topK, const uint32_t* graph, const uint32_t* hash,
    const float* data, float* dists, uint32_t* labels)
{
    APPERR_RETURN_IF_NOT_LOG(pIndexCagraImpl != nullptr, APP_ERR_INVALID_PARAM, "pIndexCagraImpl is null");
    APPERR_RETURN_IF_NOT_FMT(topK >= 16 && topK <= 128, APP_ERR_INVALID_PARAM,
        "topK %d, must be in range [16, 128]", topK);
    APPERR_RETURN_IF_NOT_LOG(queryData, APP_ERR_INVALID_PARAM, "queryData cannot be empty");
    APPERR_RETURN_IF_NOT_LOG(graph, APP_ERR_INVALID_PARAM, "graph cannot be empty");
    APPERR_RETURN_IF_NOT_LOG(hash, APP_ERR_INVALID_PARAM, "hash cannot be empty");
    APPERR_RETURN_IF_NOT_LOG(data, APP_ERR_INVALID_PARAM, "data cannot be empty");
    APPERR_RETURN_IF_NOT_LOG(dists, APP_ERR_INVALID_PARAM, "dists cannot be empty");
    APPERR_RETURN_IF_NOT_LOG(labels, APP_ERR_INVALID_PARAM, "labels cannot be empty");
    
    return this->pIndexCagraImpl->Search(n, queryData, topK, graph, hash, data, dists, labels);
}
} // namespace ascend
} // namespace faiss