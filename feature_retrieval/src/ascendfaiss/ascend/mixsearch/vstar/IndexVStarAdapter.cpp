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


#include "ErrorCode.h"
#include "utils/LogUtils.h"
#include "IndexVStarAdapter.h"

using namespace ascend;

namespace faiss {
namespace ascend {

std::shared_ptr<IndexVStar> IndexVStar::CreateIndex(const AscendIndexVstarInitParams& params)
{
    return std::make_shared<IndexVStarAdapter>(params);
}

std::shared_ptr<IndexVStar> IndexVStar::CreateIndex(const std::vector<int>& deviceList, bool verbose)
{
    return std::make_shared<IndexVStarAdapter>(deviceList, verbose);
}

IndexVStarAdapter::IndexVStarAdapter(const AscendIndexVstarInitParams& params)
{
    ascendSearchacc::IndexVstarInitParams vstarParam(params.dim, params.subSpaceDim,
        params.nlist, params.deviceList, params.verbose, params.resourceSize);
    instance = std::make_shared<ascendSearchacc::NpuIndexVStar>(vstarParam);
}

IndexVStarAdapter::IndexVStarAdapter(const std::vector<int>& deviceList, bool verbose)
    : instance(std::make_shared<ascendSearchacc::NpuIndexVStar>(deviceList, verbose)) {}

APP_ERROR IndexVStarAdapter::LoadIndex(const std::string& indexPath, IndexVStar* indexVStar)
{
    try {
        if (indexVStar == nullptr) {
            instance->LoadIndex(indexPath);
            return APP_ERR_OK;
        }
        auto adapter = dynamic_cast<IndexVStarAdapter*>(indexVStar);
        APPERR_RETURN_IF_NOT_LOG((adapter != nullptr), APP_ERR_INVALID_PARAM, "adapter index is nullptr.");

        instance->LoadIndex(indexPath, adapter->instance.get());
    } catch (std::exception &e) {
        APPERR_RETURN_IF_FMT(true, APP_ERR_INNER_ERROR, "%s", e.what());
    }
    return APP_ERR_OK;
}

APP_ERROR IndexVStarAdapter::WriteIndex(const std::string& indexPath)
{
    try {
        instance->WriteIndex(indexPath);
    } catch (std::exception &e) {
        APPERR_RETURN_IF_FMT(true, APP_ERR_INNER_ERROR, "%s", e.what());
    }
    return APP_ERR_OK;
}

APP_ERROR IndexVStarAdapter::AddCodeBooksByIndex(IndexVStar& indexVStar)
{
    auto adapter = dynamic_cast<IndexVStarAdapter*>(&indexVStar);
    APPERR_RETURN_IF_NOT_LOG((adapter != nullptr), APP_ERR_INVALID_PARAM, "adapter index is nullptr.");
 
    APP_ERROR ret = APP_ERR_OK;
    try {
        ret = instance->AddCodeBooks(adapter->instance.get());
    } catch (std::exception &e) {
        APPERR_RETURN_IF_FMT(true, APP_ERR_INNER_ERROR, "%s", e.what());
    }
    return ret;
}

APP_ERROR IndexVStarAdapter::AddCodeBooksByPath(const std::string& codebooksPath)
{
    APP_ERROR ret = APP_ERR_OK;
    try {
        ret = instance->AddCodeBooks(codebooksPath);
    } catch (std::exception &e) {
        APPERR_RETURN_IF_FMT(true, APP_ERR_INNER_ERROR, "%s", e.what());
    }
    return ret;
}

APP_ERROR IndexVStarAdapter::Add(const std::vector<float>& baseData)
{
    APP_ERROR ret = APP_ERR_OK;
    try {
        ret = instance->AddVectors(baseData);
    } catch (std::exception &e) {
        APPERR_RETURN_IF_FMT(true, APP_ERR_INNER_ERROR, "%s", e.what());
    }
    return ret;
}

APP_ERROR IndexVStarAdapter::AddWithIds(const std::vector<float>& baseData, const std::vector<int64_t>& ids)
{
    APP_ERROR ret = APP_ERR_OK;
    try {
        ret = instance->AddVectorsWithIds(baseData, ids);
    } catch (std::exception &e) {
        APPERR_RETURN_IF_FMT(true, APP_ERR_INNER_ERROR, "%s", e.what());
    }
    return ret;
}

APP_ERROR IndexVStarAdapter::DeleteByIds(const std::vector<int64_t>& ids)
{
    APP_ERROR ret = APP_ERR_OK;
    try {
        ret = instance->DeleteVectors(ids);
    } catch (std::exception &e) {
        APPERR_RETURN_IF_FMT(true, APP_ERR_INNER_ERROR, "%s", e.what());
    }
    return ret;
}

APP_ERROR IndexVStarAdapter::DeleteById(int64_t id)
{
    APP_ERROR ret = APP_ERR_OK;
    try {
        ret = instance->DeleteVectors(id);
    } catch (std::exception &e) {
        APPERR_RETURN_IF_FMT(true, APP_ERR_INNER_ERROR, "%s", e.what());
    }
    return ret;
}

APP_ERROR IndexVStarAdapter::DeleteByRange(int64_t startId, int64_t endId)
{
    APP_ERROR ret = APP_ERR_OK;
    try {
        ret = instance->DeleteVectors(startId, endId);
    } catch (std::exception &e) {
        APPERR_RETURN_IF_FMT(true, APP_ERR_INNER_ERROR, "%s", e.what());
    }
    return ret;
}

APP_ERROR IndexVStarAdapter::Search(const AscendIndexSearchParams& params) const
{
    ascendSearchacc::SearchImplParams searchParams(params.n, params.queryData,
        params.topK, params.dists, params.labels);
    APP_ERROR ret = APP_ERR_OK;
    try {
        ret = instance->Search(searchParams);
    } catch (std::exception &e) {
        APPERR_RETURN_IF_FMT(true, APP_ERR_INNER_ERROR, "%s", e.what());
    }
    return ret;
}

APP_ERROR IndexVStarAdapter::SearchWithMask(const AscendIndexSearchParams& params,
    const std::vector<uint8_t>& mask) const
{
    ascendSearchacc::SearchImplParams searchParams(params.n, params.queryData,
        params.topK, params.dists, params.labels);
    APP_ERROR ret = APP_ERR_OK;
    try {
        ret = instance->Search(searchParams, mask);
    } catch (std::exception &e) {
        APPERR_RETURN_IF_FMT(true, APP_ERR_INNER_ERROR, "%s", e.what());
    }
    return ret;
}

APP_ERROR IndexVStarAdapter::MultiSearch(std::vector<IndexVStar*>& indexes,
    const AscendIndexSearchParams& params, bool merge) const
{
    std::vector<ascendSearchacc::NpuIndexVStar*> vstarIndexes;
    for (size_t i = 0; i < indexes.size(); i++) {
        auto adapter = dynamic_cast<IndexVStarAdapter*>(indexes[i]);
        APPERR_RETURN_IF_NOT_LOG((adapter != nullptr), APP_ERR_INVALID_PARAM, "adapter index is nullptr.");
        vstarIndexes.emplace_back(adapter->instance.get());
    }
    ascendSearchacc::SearchImplParams searchParams(params.n, params.queryData,
        params.topK, params.dists, params.labels);
    APP_ERROR ret = APP_ERR_OK;
    try {
        ret = instance->MultiSearch(vstarIndexes, searchParams, merge);
    } catch (std::exception &e) {
        APPERR_RETURN_IF_FMT(true, APP_ERR_INNER_ERROR, "%s", e.what());
    }
    return ret;
}

APP_ERROR IndexVStarAdapter::MultiSearchWithMask(std::vector<IndexVStar*>& indexes,
    const AscendIndexSearchParams& params, const std::vector<uint8_t>& mask, bool merge) const
{
    std::vector<ascendSearchacc::NpuIndexVStar*> vstarIndexes;
    for (size_t i = 0; i < indexes.size(); i++) {
        auto indexWithMask = dynamic_cast<IndexVStarAdapter*>(indexes[i]);
        APPERR_RETURN_IF_NOT_LOG((indexWithMask != nullptr), APP_ERR_INVALID_PARAM, "indexWithMask is nullptr.");

        vstarIndexes.emplace_back(indexWithMask->instance.get());
    }
    ascendSearchacc::SearchImplParams searchParams(params.n, params.queryData,
        params.topK, params.dists, params.labels);
    APP_ERROR ret = APP_ERR_OK;
    try {
        ret = instance->MultiSearch(vstarIndexes, searchParams, mask, merge);
    } catch (std::exception &e) {
        APPERR_RETURN_IF_FMT(true, APP_ERR_INNER_ERROR, "%s", e.what());
    }
    return ret;
}

APP_ERROR IndexVStarAdapter::SetHyperSearchParams(const AscendIndexVstarHyperParams& params)
{
    ascendSearchacc::IndexVstarSearchParams vstarParam(params.nProbeL1,
        params.nProbeL2, params.l3SegmentNum);
    try {
        instance->SetSearchParams(vstarParam);
    } catch (std::exception &e) {
        APPERR_RETURN_IF_FMT(true, APP_ERR_INNER_ERROR, "%s", e.what());
    }
    return APP_ERR_OK;
}

APP_ERROR IndexVStarAdapter::GetHyperSearchParams(AscendIndexVstarHyperParams& params) const
{
    try {
        ascendSearchacc::IndexVstarSearchParams vstarParams = instance->GetSearchParams();
        params.nProbeL1 = vstarParams.params.nProbeL1;
        params.nProbeL2 = vstarParams.params.nProbeL2;
        params.l3SegmentNum = vstarParams.params.l3SegmentNum;
    } catch (std::exception &e) {
        APPERR_RETURN_IF_FMT(true, APP_ERR_INNER_ERROR, "%s", e.what());
    }
    return APP_ERR_OK;
}

APP_ERROR IndexVStarAdapter::GetDim(int& dim) const
{
    APP_ERROR ret = APP_ERR_OK;
    try {
        dim = instance->GetDim();
    } catch (std::exception &e) {
        APPERR_RETURN_IF_FMT(true, APP_ERR_INNER_ERROR, "%s", e.what());
    }
    return ret;
}

APP_ERROR IndexVStarAdapter::GetNTotal(uint64_t& ntotal) const
{
    APP_ERROR ret = APP_ERR_OK;
    try {
        ntotal = instance->GetNTotal();
    } catch (std::exception &e) {
        APPERR_RETURN_IF_FMT(true, APP_ERR_INNER_ERROR, "%s", e.what());
    }
    return ret;
}

APP_ERROR IndexVStarAdapter::Reset()
{
    APP_ERROR ret = APP_ERR_OK;
    try {
        ret = instance->Reset();
    } catch (std::exception &e) {
        APPERR_RETURN_IF_FMT(true, APP_ERR_INNER_ERROR, "%s", e.what());
    }
    return ret;
}

} // namespace faiss
} // namespace ascend