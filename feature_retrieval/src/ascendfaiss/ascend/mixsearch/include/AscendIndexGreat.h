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


#ifndef ASCEND_INDEX_GREAT_INCLUDED
#define ASCEND_INDEX_GREAT_INCLUDED

#include <string>
#include <vector>
#include <memory>
#include "AscendIndexMixSearchParams.h"
using APP_ERROR = int;
namespace faiss {
namespace ascend {
class IndexGreat;
class AscendIndexGreat {
public:
    AscendIndexGreat(const std::string& mode, const std::vector<int>& deviceList, bool verbose = false);

    explicit AscendIndexGreat(const AscendIndexGreatInitParams& kModeInitParams);

    AscendIndexGreat(const AscendIndexVstarInitParams& aModeInitParams,
                     const AscendIndexGreatInitParams& kModeInitParams);

    virtual ~AscendIndexGreat() = default;

    APP_ERROR LoadIndex(const std::string& indexPath);

    APP_ERROR LoadIndex(const std::string& aModeIndexPath, const std::string& kModeIndexPath);

    APP_ERROR WriteIndex(const std::string& indexPath);

    APP_ERROR WriteIndex(const std::string& aModeIndexPath, const std::string& kModeIndexPath);

    APP_ERROR AddCodeBooks(const std::string& codeBooksPath);

    APP_ERROR Add(const std::vector<float>& baseRawData);

    APP_ERROR AddWithIds(const std::vector<float>& baseRawData, const std::vector<int64_t>& ids);

    APP_ERROR Search(const AscendIndexSearchParams& searchParams);

    APP_ERROR SearchWithMask(const AscendIndexSearchParams& searchParams,
                             const std::vector<uint8_t>& mask);

    AscendIndexGreat(const AscendIndexGreat &) = delete;
    AscendIndexGreat &operator=(const AscendIndexGreat &) = delete;

    APP_ERROR SetHyperSearchParams(const AscendIndexHyperParams& params);

    APP_ERROR GetHyperSearchParams(AscendIndexHyperParams& params) const;

    APP_ERROR GetDim(int& dim) const;

    APP_ERROR GetNTotal(uint64_t& nTotal) const;

    APP_ERROR Reset();
private:
    std::shared_ptr<faiss::ascend::IndexGreat> impl;
};
}
}
#endif // ASCEND_INDEX_GREAT_INCLUDED