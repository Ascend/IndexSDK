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


#include "AscendIndexVStar.h"
#include "ErrorCode.h"
#include "IndexVStar.h"

namespace faiss {
namespace ascend {
namespace {
constexpr size_t MAX_INDEX_NUM = 150;  // 多index场景下index个数限制在150以内
} /* namespace */

AscendIndexVStar::AscendIndexVStar(const AscendIndexVstarInitParams& params)
    : impl(IndexVStar::CreateIndex(params)) {}

AscendIndexVStar::AscendIndexVStar(const std::vector<int>& deviceList, bool verbose)
    : impl(IndexVStar::CreateIndex(deviceList, verbose)) {}

APP_ERROR AscendIndexVStar::LoadIndex(const std::string& indexPath, AscendIndexVStar* indexVStar)
{
    APPERR_RETURN_IF_NOT_LOG((impl != nullptr), APP_ERR_INNER_ERROR, "impl is nullptr!");
    return (indexVStar == nullptr) ? impl->LoadIndex(indexPath) : impl->LoadIndex(indexPath, indexVStar->impl.get());
}

APP_ERROR AscendIndexVStar::WriteIndex(const std::string& indexPath)
{
    APPERR_RETURN_IF_NOT_LOG((impl != nullptr), APP_ERR_INNER_ERROR, "impl is nullptr!");
    return impl->WriteIndex(indexPath);
}

APP_ERROR AscendIndexVStar::AddCodeBooksByIndex(AscendIndexVStar& indexVStar)
{
    APPERR_RETURN_IF_NOT_LOG((impl != nullptr), APP_ERR_INNER_ERROR, "impl is nullptr!");
    return impl->AddCodeBooksByIndex(*indexVStar.impl);
}

APP_ERROR AscendIndexVStar::AddCodeBooksByPath(const std::string& codeBooksPath)
{
    APPERR_RETURN_IF_NOT_LOG((impl != nullptr), APP_ERR_INNER_ERROR, "impl is nullptr!");
    return impl->AddCodeBooksByPath(codeBooksPath);
}

APP_ERROR AscendIndexVStar::Add(const std::vector<float>& baseData)
{
    APPERR_RETURN_IF_NOT_LOG((impl != nullptr), APP_ERR_INNER_ERROR, "impl is nullptr!");
    return impl->Add(baseData);
}

APP_ERROR AscendIndexVStar::AddWithIds(const std::vector<float>& baseData, const std::vector<int64_t>& ids)
{
    APPERR_RETURN_IF_NOT_LOG((impl != nullptr), APP_ERR_INNER_ERROR, "impl is nullptr!");
    return impl->AddWithIds(baseData, ids);
}

APP_ERROR AscendIndexVStar::DeleteByIds(const std::vector<int64_t>& ids)
{
    APPERR_RETURN_IF_NOT_LOG((impl != nullptr), APP_ERR_INNER_ERROR, "impl is nullptr!");
    return impl->DeleteByIds(ids);
}

APP_ERROR AscendIndexVStar::DeleteById(int64_t id)
{
    APPERR_RETURN_IF_NOT_LOG((impl != nullptr), APP_ERR_INNER_ERROR, "impl is nullptr!");
    return impl->DeleteById(id);
}

APP_ERROR AscendIndexVStar::DeleteByRange(int64_t startId, int64_t endId)
{
    APPERR_RETURN_IF_NOT_LOG((impl != nullptr), APP_ERR_INNER_ERROR, "impl is nullptr!");
    return impl->DeleteByRange(startId, endId);
}

APP_ERROR AscendIndexVStar::Search(const AscendIndexSearchParams& params) const
{
    APPERR_RETURN_IF_NOT_LOG((impl != nullptr), APP_ERR_INNER_ERROR, "impl is nullptr!");
    return impl->Search(params);
}

APP_ERROR AscendIndexVStar::SearchWithMask(const AscendIndexSearchParams& params,
    const std::vector<uint8_t>& mask) const
{
    APPERR_RETURN_IF_NOT_LOG((impl != nullptr), APP_ERR_INNER_ERROR, "impl is nullptr!");
    return impl->SearchWithMask(params, mask);
}

APP_ERROR AscendIndexVStar::MultiSearch(std::vector<AscendIndexVStar*>& indexes,
    const AscendIndexSearchParams& params, bool merge) const
{
    APPERR_RETURN_IF_NOT_LOG((impl != nullptr), APP_ERR_INNER_ERROR, "impl is nullptr!");
    APPERR_RETURN_IF_NOT_FMT(indexes.size() > 0 && indexes.size() <= MAX_INDEX_NUM, APP_ERR_INVALID_PARAM,
        "size of indexes[%zu] must be > 0 and <= %zu!", indexes.size(), MAX_INDEX_NUM);
    std::vector<IndexVStar*> vstarIndexes;
    for (size_t i = 0; i < indexes.size(); i++) {
        APPERR_RETURN_IF_NOT_FMT(indexes[i] != nullptr, APP_ERR_INVALID_PARAM, "indexes[%zu] is nullptr!", i);
        vstarIndexes.emplace_back(indexes[i]->impl.get());
    }
    return impl->MultiSearch(vstarIndexes, params, merge);
}

APP_ERROR AscendIndexVStar::MultiSearchWithMask(std::vector<AscendIndexVStar*>& indexes,
    const AscendIndexSearchParams& params, const std::vector<uint8_t>& mask, bool merge)
{
    APPERR_RETURN_IF_NOT_LOG((impl != nullptr), APP_ERR_INNER_ERROR, "impl is nullptr!");
    APPERR_RETURN_IF_NOT_FMT((indexes.size() > 0 && indexes.size() <= MAX_INDEX_NUM), APP_ERR_INVALID_PARAM,
        "size of indexes[%zu] must be > 0 and <= %zu!", indexes.size(), MAX_INDEX_NUM);
    std::vector<IndexVStar*> vstarIndexes;
    for (size_t i = 0; i < indexes.size(); i++) {
        APPERR_RETURN_IF_NOT_FMT(indexes[i] != nullptr, APP_ERR_INVALID_PARAM, "indexes[%zu] is nullptr!", i);
        vstarIndexes.emplace_back(indexes[i]->impl.get());
    }
    return impl->MultiSearchWithMask(vstarIndexes, params, mask, merge);
}

APP_ERROR AscendIndexVStar::SetHyperSearchParams(const AscendIndexVstarHyperParams& params)
{
    APPERR_RETURN_IF_NOT_LOG((impl != nullptr), APP_ERR_INNER_ERROR, "impl is nullptr!");
    return impl->SetHyperSearchParams(params);
}

APP_ERROR AscendIndexVStar::GetHyperSearchParams(AscendIndexVstarHyperParams& params) const
{
    APPERR_RETURN_IF_NOT_LOG((impl != nullptr), APP_ERR_INNER_ERROR, "impl is nullptr!");
    return impl->GetHyperSearchParams(params);
}

APP_ERROR AscendIndexVStar::GetDim(int& dim) const
{
    APPERR_RETURN_IF_NOT_LOG((impl != nullptr), APP_ERR_INNER_ERROR, "impl is nullptr!");
    return impl->GetDim(dim);
}

APP_ERROR AscendIndexVStar::GetNTotal(uint64_t& ntotal) const
{
    APPERR_RETURN_IF_NOT_LOG((impl != nullptr), APP_ERR_INNER_ERROR, "impl is nullptr!");
    return impl->GetNTotal(ntotal);
}

APP_ERROR AscendIndexVStar::Reset()
{
    APPERR_RETURN_IF_NOT_LOG((impl != nullptr), APP_ERR_INNER_ERROR, "impl is nullptr!");
    return impl->Reset();
}

} // namespace faiss
} // namespace ascend