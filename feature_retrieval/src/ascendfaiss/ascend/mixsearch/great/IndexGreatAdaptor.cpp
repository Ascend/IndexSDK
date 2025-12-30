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


#include "IndexGreatAdaptor.h"
#include "common/ErrorCode.h"
using namespace ascend;

namespace faiss {
namespace ascend {
static ascendSearchacc::IndexVstarInitParams ConvertAModeParams(const AscendIndexVstarInitParams& aModeInitParams)
{
    ascendSearchacc::IndexVstarInitParams vStarParams {
        aModeInitParams.dim,
        aModeInitParams.subSpaceDim,
        aModeInitParams.nlist,
        aModeInitParams.deviceList,
        aModeInitParams.verbose,
        aModeInitParams.resourceSize
    };
    return vStarParams;
}

static ascendSearchacc::KModeInitParams ConvertKModeParams(const AscendIndexGreatInitParams& kModeInitParams)
{
    ascendSearchacc::KModeInitParams params {
        kModeInitParams.dim,
        kModeInitParams.degree,
        kModeInitParams.convPQM,
        kModeInitParams.evaluationType,
        kModeInitParams.expandingFactor
    };
    return params;
}

static ascendSearchacc::IndexGreatSearchParams ConvertHyperParams(const AscendIndexHyperParams& hyperParams)
{
    ascendSearchacc::IndexGreatSearchParams params {
        hyperParams.mode,
        hyperParams.vstarHyperParam.nProbeL1,
        hyperParams.vstarHyperParam.nProbeL2,
        hyperParams.vstarHyperParam.l3SegmentNum,
        hyperParams.expandingFactor
    };
    return params;
}

static AscendIndexHyperParams ConvertAscendHyperParams(const ascendSearchacc::IndexGreatSearchParams& params)
{
    AscendIndexVstarHyperParams vHyperParams { params.nProbeL1, params.nProbeL2, params.l3SegmentNum };
    AscendIndexHyperParams hyperParams { params.mode, vHyperParams, params.ef };
    return hyperParams;
}

static ascendSearchacc::SearchImplParams ConvertSearchParams(const AscendIndexSearchParams& searchParams)
{
    ascendSearchacc::SearchImplParams params {
        searchParams.n,
        searchParams.queryData,
        searchParams.topK,
        searchParams.dists,
        searchParams.labels
    };
    return params;
}

std::shared_ptr<IndexGreat> IndexGreat::CreateIndex(const std::string& mode,
                                                    const std::vector<int>& deviceList,
                                                    const bool verbose = false)
{
    return std::make_shared<IndexGreatAdaptor>(mode, deviceList, verbose);
}

std::shared_ptr<IndexGreat> IndexGreat::CreateIndex(const AscendIndexGreatInitParams& kModeInitParams)
{
    return std::make_shared<IndexGreatAdaptor>(kModeInitParams);
}

std::shared_ptr<IndexGreat> IndexGreat::CreateIndex(const AscendIndexVstarInitParams& aModeInitParams,
                                                    const AscendIndexGreatInitParams& kModeInitParams)
{
    return std::make_shared<IndexGreatAdaptor>(aModeInitParams, kModeInitParams);
}

IndexGreatAdaptor::IndexGreatAdaptor(const std::string& mode,
                                     const std::vector<int>& deviceList,
                                     const bool verbose = false)
{
    instance = std::make_shared<ascendSearchacc::IndexGreat>(mode, deviceList, verbose);
}

IndexGreatAdaptor::IndexGreatAdaptor(const AscendIndexGreatInitParams& kModeInitParams)
{
    ascendSearchacc::KModeInitParams kParams = ConvertKModeParams(kModeInitParams);
    ascendSearchacc::IndexGreatInitParams initParams("KMode", kParams);
    instance = std::make_shared<ascendSearchacc::IndexGreat>("KMode", initParams);
}

IndexGreatAdaptor::IndexGreatAdaptor(const AscendIndexVstarInitParams& aModeInitParams,
    const AscendIndexGreatInitParams& kModeInitParams)
{
    ascendSearchacc::IndexVstarInitParams aParams = ConvertAModeParams(aModeInitParams);
    ascendSearchacc::KModeInitParams kParams = ConvertKModeParams(kModeInitParams);
    ascendSearchacc::IndexGreatInitParams initParams("AKMode", aParams, kParams);
    instance = std::make_shared<ascendSearchacc::IndexGreat>("AKMode", initParams);
}

APP_ERROR IndexGreatAdaptor::Add(const std::vector<float>& baseRawData)
{
    APPERR_RETURN_IF_NOT_LOG(instance != nullptr, APP_ERR_INNER_ERROR,
        "IndexGreatAdaptor addVectors impl can not be nullptr");
    APP_ERROR ret = APP_ERR_OK;
    try {
        ret = instance->AddVectors(baseRawData);
    } catch (std::exception &e) {
        APPERR_RETURN_IF_FMT(true, APP_ERR_INNER_ERROR, "%s", e.what());
    }
    return ret;
}

APP_ERROR IndexGreatAdaptor::AddWithIds(const std::vector<float>& baseRawData, const std::vector<int64_t>& ids)
{
    APPERR_RETURN_IF_NOT_LOG(instance != nullptr, APP_ERR_INNER_ERROR, "IndexGreatAdaptor impl can not be nullptr");
    APP_ERROR ret = APP_ERR_OK;
    try {
        ret = instance->AddVectorsWithIds(baseRawData, ids);
    } catch (std::exception &e) {
        APPERR_RETURN_IF_FMT(true, APP_ERR_INNER_ERROR, "%s", e.what());
    }
    return ret;
}

APP_ERROR IndexGreatAdaptor::LoadIndex(const std::string& indexPath)
{
    APPERR_RETURN_IF_NOT_LOG(instance != nullptr, APP_ERR_INNER_ERROR,
        "IndexGreatAdaptor loadIndex impl can not be nullptr");
    try {
        instance->LoadIndex(indexPath);
    } catch (std::exception &e) {
        APPERR_RETURN_IF_FMT(true, APP_ERR_INNER_ERROR, "%s", e.what());
    }
    return APP_ERR_OK;
}

APP_ERROR IndexGreatAdaptor::LoadIndex(const std::string& aModeindexPath, const std::string& kModeindexPath)
{
    APPERR_RETURN_IF_NOT_LOG(instance != nullptr, APP_ERR_INNER_ERROR,
        "IndexGreatAdaptor loadIndex impl can not be nullptr");
    try {
        instance->LoadIndex(aModeindexPath, kModeindexPath);
    } catch (std::exception &e) {
        APPERR_RETURN_IF_FMT(true, APP_ERR_INNER_ERROR, "%s", e.what());
    }
    return APP_ERR_OK;
}

APP_ERROR IndexGreatAdaptor::WriteIndex(const std::string& indexPath)
{
    APPERR_RETURN_IF_NOT_LOG(instance != nullptr, APP_ERR_INNER_ERROR,
        "IndexGreatAdaptor writeIndex impl can not be nullptr");
    try {
        instance->WriteIndex(indexPath);
    } catch (std::exception &e) {
        APPERR_RETURN_IF_FMT(true, APP_ERR_INNER_ERROR, "%s", e.what());
    }
    return APP_ERR_OK;
}

APP_ERROR IndexGreatAdaptor::WriteIndex(const std::string& aModeindexPath, const std::string& kModeindexPath)
{
    APPERR_RETURN_IF_NOT_LOG(instance != nullptr, APP_ERR_INNER_ERROR,
        "IndexGreatAdaptor writeIndex impl can not be nullptr");
    try {
        instance->WriteIndex(aModeindexPath, kModeindexPath);
    } catch (std::exception &e) {
        APPERR_RETURN_IF_FMT(true, APP_ERR_INNER_ERROR, "%s", e.what());
    }
    return APP_ERR_OK;
}

APP_ERROR IndexGreatAdaptor::AddCodeBooks(const std::string& codeBooksPath)
{
    APPERR_RETURN_IF_NOT_LOG(instance != nullptr, APP_ERR_INNER_ERROR,
        "IndexGreatAdaptor addCodeBooks impl can not be nullptr");
    APP_ERROR ret = APP_ERR_OK;
    try {
        ret = instance->AddCodeBooks(codeBooksPath);
    } catch (std::exception &e) {
        APPERR_RETURN_IF_FMT(true, APP_ERR_INNER_ERROR, "%s", e.what());
    }
    return ret;
}

APP_ERROR IndexGreatAdaptor::Search(const AscendIndexSearchParams& searchParams)
{
    APPERR_RETURN_IF_NOT_LOG(instance != nullptr, APP_ERR_INNER_ERROR,
        "IndexGreatAdaptor search impl can not be nullptr");
    ascendSearchacc::SearchImplParams params = ConvertSearchParams(searchParams);
    APP_ERROR ret = APP_ERR_OK;
    try {
        ret = instance->Search(params);
    } catch (std::exception &e) {
        APPERR_RETURN_IF_FMT(true, APP_ERR_INNER_ERROR, "%s", e.what());
    }
    return ret;
}

APP_ERROR IndexGreatAdaptor::SearchWithMask(const AscendIndexSearchParams& searchParams,
                                            const std::vector<uint8_t>& mask)
{
    APPERR_RETURN_IF_NOT_LOG(instance != nullptr, APP_ERR_INNER_ERROR,
        "IndexGreatAdaptor searchWithMask impl can not be nullptr");
    ascendSearchacc::SearchImplParams params = ConvertSearchParams(searchParams);
    APP_ERROR ret = APP_ERR_OK;
    try {
        ret = instance->SearchWithMask(params, mask);
    } catch (std::exception &e) {
        APPERR_RETURN_IF_FMT(true, APP_ERR_INNER_ERROR, "%s", e.what());
    }
    return ret;
}

APP_ERROR IndexGreatAdaptor::SetHyperSearchParams(const AscendIndexHyperParams& hyperParams)
{
    APPERR_RETURN_IF_NOT_LOG(instance != nullptr, APP_ERR_INNER_ERROR,
        "IndexGreatAdaptor setHyperSearchParams impl can not be nullptr");
    ascendSearchacc::IndexGreatSearchParams params = ConvertHyperParams(hyperParams);
    try {
        instance->SetSearchParams(params);
    } catch (std::exception &e) {
        APPERR_RETURN_IF_FMT(true, APP_ERR_INNER_ERROR, "%s", e.what());
    }
    return APP_ERR_OK;
}

APP_ERROR IndexGreatAdaptor::GetHyperSearchParams(AscendIndexHyperParams& hyperParams) const
{
    APPERR_RETURN_IF_NOT_LOG(instance != nullptr, APP_ERR_INNER_ERROR,
        "IndexGreatAdaptor getHyperSearchParams impl can not be nullptr");
    try {
        ascendSearchacc::IndexGreatSearchParams params = instance->GetSearchParams();
        hyperParams = ConvertAscendHyperParams(params);
    } catch (std::exception &e) {
        APPERR_RETURN_IF_FMT(true, APP_ERR_INNER_ERROR, "%s", e.what());
    }
    return APP_ERR_OK;
}

APP_ERROR IndexGreatAdaptor::GetDim(int& dim) const
{
    APPERR_RETURN_IF_NOT_LOG(instance != nullptr, APP_ERR_INNER_ERROR,
        "IndexGreatAdaptor getDim impl can not be nullptr");
    try {
        dim = instance->GetDim();
    } catch (std::exception &e) {
        APPERR_RETURN_IF_FMT(true, APP_ERR_INNER_ERROR, "%s", e.what());
    }
    return APP_ERR_OK;
}

APP_ERROR IndexGreatAdaptor::GetNTotal(uint64_t& nTotal) const
{
    APPERR_RETURN_IF_NOT_LOG(instance != nullptr, APP_ERR_INNER_ERROR,
        "IndexGreatAdaptor getNTotal impl can not be nullptr");
    try {
        nTotal = instance->GetNTotal();
    } catch (std::exception &e) {
        APPERR_RETURN_IF_FMT(true, APP_ERR_INNER_ERROR, "%s", e.what());
    }
    return APP_ERR_OK;
}

APP_ERROR IndexGreatAdaptor::Reset()
{
    APPERR_RETURN_IF_NOT_LOG(instance != nullptr, APP_ERR_INNER_ERROR,
        "IndexGreatAdaptor reset impl can not be nullptr");
    try {
        instance->Reset();
    } catch (std::exception &e) {
        APPERR_RETURN_IF_FMT(true, APP_ERR_INNER_ERROR, "%s", e.what());
    }
    return APP_ERR_OK;
}
}
}