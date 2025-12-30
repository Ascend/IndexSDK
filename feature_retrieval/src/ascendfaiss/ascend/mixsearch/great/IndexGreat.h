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

#ifndef INDEX_GREAT_INCLUDED
#define INDEX_GREAT_INCLUDED

#include <memory>
#include <string>
#include <vector>
#include "common/ErrorCode.h"
#include "include/AscendIndexMixSearchParams.h"
using namespace ascend;

namespace faiss {
namespace ascend {
class IndexGreat {
public:
    IndexGreat() = default;

    virtual ~IndexGreat() = default;

    virtual APP_ERROR Add(const std::vector<float>& baseRawData) = 0;

    virtual APP_ERROR AddWithIds(const std::vector<float>& baseRawData, const std::vector<int64_t>& ids) = 0;

    virtual APP_ERROR LoadIndex(const std::string& indexPath) = 0;

    virtual APP_ERROR LoadIndex(const std::string& aModeIndexPath, const std::string& kModeIndexPath) = 0;

    virtual APP_ERROR WriteIndex(const std::string& indexPath) = 0;
    
    virtual APP_ERROR WriteIndex(const std::string& aModeIndexPath, const std::string& kModeIndexPath) = 0;

    virtual APP_ERROR AddCodeBooks(const std::string& codeBooksPath) = 0;

    virtual APP_ERROR Search(const AscendIndexSearchParams& params) = 0;

    virtual APP_ERROR SearchWithMask(const AscendIndexSearchParams& params, const std::vector<uint8_t>& mask) = 0;

    virtual APP_ERROR SetHyperSearchParams(const AscendIndexHyperParams& params) = 0;

    virtual APP_ERROR GetHyperSearchParams(AscendIndexHyperParams& params) const = 0;

    virtual APP_ERROR GetDim(int& dim) const = 0;

    virtual APP_ERROR GetNTotal(uint64_t& nTotal) const = 0;

    virtual APP_ERROR Reset() = 0;

    static std::shared_ptr<IndexGreat> CreateIndex(const std::string& mode,
                                                   const std::vector<int>& deviceList,
                                                   const bool verbose);

    static std::shared_ptr<IndexGreat> CreateIndex(const AscendIndexGreatInitParams& kModeInitParams);

    static std::shared_ptr<IndexGreat> CreateIndex(const AscendIndexVstarInitParams& aModeInitParams,
                                                   const AscendIndexGreatInitParams& kModeInitParams);
};
}
}
#endif