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


#include "include/AscendIndexGreat.h"
#include "common/ErrorCode.h"
#include "IndexGreat.h"
#include "ascenddaemon/utils/AscendUtils.h"

namespace faiss {
namespace ascend {
std::shared_mutex mtx;

AscendIndexGreat::AscendIndexGreat(const std::string& mode, const std::vector<int>& deviceList, const bool verbose)
    : impl(IndexGreat::CreateIndex(mode, deviceList, verbose)) {}

AscendIndexGreat::AscendIndexGreat(const AscendIndexGreatInitParams& kModeInitParams)
    : impl(IndexGreat::CreateIndex(kModeInitParams)) {}

AscendIndexGreat::AscendIndexGreat(const AscendIndexVstarInitParams& aModeInitParams,
                                   const AscendIndexGreatInitParams& kModeInitParams)
    : impl(IndexGreat::CreateIndex(aModeInitParams, kModeInitParams)) {}

APP_ERROR AscendIndexGreat::LoadIndex(const std::string& indexPath)
{
    auto lock = ::ascend::AscendMultiThreadManager::GetWriteLock(mtx);
    APPERR_RETURN_IF_NOT_LOG(impl != nullptr, APP_ERR_INNER_ERROR,
        "AscendIndexGreat loadIndex impl can not be nullptr");
    return impl->LoadIndex(indexPath);
}

APP_ERROR AscendIndexGreat::LoadIndex(const std::string& aModeIndexPath, const std::string& kModeIndexPath)
{
    auto lock = ::ascend::AscendMultiThreadManager::GetWriteLock(mtx);
    APPERR_RETURN_IF_NOT_LOG(impl != nullptr, APP_ERR_INNER_ERROR,
        "AscendIndexGreat loadIndex impl can not be nullptr");
    return impl->LoadIndex(aModeIndexPath, kModeIndexPath);
}

APP_ERROR AscendIndexGreat::WriteIndex(const std::string& indexPath)
{
    auto lock = ::ascend::AscendMultiThreadManager::GetWriteLock(mtx);
    APPERR_RETURN_IF_NOT_LOG(impl != nullptr, APP_ERR_INNER_ERROR,
        "AscendIndexGreat writeIndex impl can not be nullptr");
    return impl->WriteIndex(indexPath);
}

APP_ERROR AscendIndexGreat::WriteIndex(const std::string& aModeIndexPath, const std::string& kModeIndexPath)
{
    auto lock = ::ascend::AscendMultiThreadManager::GetWriteLock(mtx);
    APPERR_RETURN_IF_NOT_LOG(impl != nullptr, APP_ERR_INNER_ERROR,
        "AscendIndexGreat writeIndex impl can not be nullptr");
    return impl->WriteIndex(aModeIndexPath, kModeIndexPath);
}

APP_ERROR  AscendIndexGreat::AddCodeBooks(const std::string& codeBooksPath)
{
    auto lock = ::ascend::AscendMultiThreadManager::GetWriteLock(mtx);
    APPERR_RETURN_IF_NOT_LOG(impl != nullptr, APP_ERR_INNER_ERROR,
        "AscendIndexGreat addCodeBooks impl can not be nullptr");
    return impl->AddCodeBooks(codeBooksPath);
}

APP_ERROR AscendIndexGreat::Add(const std::vector<float>& baseRawData)
{
    auto lock = ::ascend::AscendMultiThreadManager::GetWriteLock(mtx);
    APPERR_RETURN_IF_NOT_LOG(impl != nullptr, APP_ERR_INNER_ERROR,
        "AscendIndexGreat addvector impl can not be nullptr");
    return impl->Add(baseRawData);
}

APP_ERROR AscendIndexGreat::AddWithIds(const std::vector<float>& baseRawData, const std::vector<int64_t>& ids)
{
    auto lock = ::ascend::AscendMultiThreadManager::GetWriteLock(mtx);
    APPERR_RETURN_IF_NOT_LOG(impl != nullptr, APP_ERR_INNER_ERROR,
        "AscendIndexGreat addvector with ids impl can not be nullptr");
    return impl->AddWithIds(baseRawData, ids);
}

APP_ERROR AscendIndexGreat::Reset()
{
    auto lock = ::ascend::AscendMultiThreadManager::GetWriteLock(mtx);
    APPERR_RETURN_IF_NOT_LOG(impl != nullptr, APP_ERR_INNER_ERROR,
        "AscendIndexGreat reset impl can not be nullptr");
    return impl->Reset();
}

APP_ERROR AscendIndexGreat::SetHyperSearchParams(const AscendIndexHyperParams& params)
{
    auto lock = ::ascend::AscendMultiThreadManager::GetWriteLock(mtx);
    APPERR_RETURN_IF_NOT_LOG(impl != nullptr, APP_ERR_INNER_ERROR,
        "AscendIndexGreat setHyperSearchParams impl can not be nullptr");
    return impl->SetHyperSearchParams(params);
}

APP_ERROR AscendIndexGreat::GetHyperSearchParams(AscendIndexHyperParams& params) const
{
    auto lock = ::ascend::AscendMultiThreadManager::GetWriteLock(mtx);
    APPERR_RETURN_IF_NOT_LOG(impl != nullptr, APP_ERR_INNER_ERROR,
        "AscendIndexGreat getHyperSearchParams impl can not be nullptr");
    return impl->GetHyperSearchParams(params);
}

APP_ERROR AscendIndexGreat::Search(const AscendIndexSearchParams& searchParams)
{
    auto lock = ::ascend::AscendMultiThreadManager::GetWriteLock(mtx);
    APPERR_RETURN_IF_NOT_LOG(impl != nullptr, APP_ERR_INNER_ERROR,
        "AscendIndexGreat search impl can not be nullptr");
    return impl->Search(searchParams);
}

APP_ERROR AscendIndexGreat::SearchWithMask(const AscendIndexSearchParams& searchParams,
                                           const std::vector<uint8_t>& mask)
{
    auto lock = ::ascend::AscendMultiThreadManager::GetWriteLock(mtx);
    APPERR_RETURN_IF_NOT_LOG(impl != nullptr, APP_ERR_INNER_ERROR,
        "AscendIndexGreat searchWithMask impl can not be nullptr");
    return impl->SearchWithMask(searchParams, mask);
}

APP_ERROR AscendIndexGreat::GetDim(int& dim) const
{
    auto lock = ::ascend::AscendMultiThreadManager::GetWriteLock(mtx);
    APPERR_RETURN_IF_NOT_LOG(impl != nullptr, APP_ERR_INNER_ERROR,
        "AscendIndexGreat getDim impl can not be nullptr");
    return impl->GetDim(dim);
}

APP_ERROR AscendIndexGreat::GetNTotal(uint64_t& nTotal) const
{
    auto lock = ::ascend::AscendMultiThreadManager::GetWriteLock(mtx);
    APPERR_RETURN_IF_NOT_LOG(impl != nullptr, APP_ERR_INNER_ERROR,
        "AscendIndexGreat getNTotal impl can not be nullptr");
    return impl->GetNTotal(nTotal);
}

}
}