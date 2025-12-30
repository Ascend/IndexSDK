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


#ifndef ASCEND_INDEX_VSTAR_H
#define ASCEND_INDEX_VSTAR_H

#include <string>
#include <vector>
#include <memory>
#include "AscendIndexMixSearchParams.h"

using APP_ERROR = int;
namespace faiss {
namespace ascend {
class IndexVStar;
class AscendIndexVStar {
public:
    explicit AscendIndexVStar(const AscendIndexVstarInitParams& params);

    AscendIndexVStar(const std::vector<int>& deviceList, bool verbose = false);

    APP_ERROR LoadIndex(const std::string& indexPath, AscendIndexVStar* indexVStar = nullptr);

    APP_ERROR WriteIndex(const std::string& indexPath);

    APP_ERROR AddCodeBooksByIndex(AscendIndexVStar& indexVStar);

    APP_ERROR AddCodeBooksByPath(const std::string& codeBooksPath);

    APP_ERROR Add(const std::vector<float>& baseData);

    APP_ERROR AddWithIds(const std::vector<float>& baseData, const std::vector<int64_t>& ids);

    APP_ERROR DeleteByIds(const std::vector<int64_t>& ids);

    APP_ERROR DeleteById(int64_t id);

    APP_ERROR DeleteByRange(int64_t startId, int64_t endId);

    APP_ERROR Search(const AscendIndexSearchParams& params) const;

    APP_ERROR SearchWithMask(const AscendIndexSearchParams& params, const std::vector<uint8_t>& mask) const;

    APP_ERROR MultiSearch(std::vector<AscendIndexVStar*>& indexes,
        const AscendIndexSearchParams& params, bool merge) const;

    APP_ERROR MultiSearchWithMask(std::vector<AscendIndexVStar*>& indexes, const AscendIndexSearchParams& params,
        const std::vector<uint8_t>& mask, bool merge);

    APP_ERROR SetHyperSearchParams(const AscendIndexVstarHyperParams& params);

    APP_ERROR GetHyperSearchParams(AscendIndexVstarHyperParams& params) const;

    APP_ERROR GetDim(int& dim) const;

    APP_ERROR GetNTotal(uint64_t& ntotal) const;

    APP_ERROR Reset();

    AscendIndexVStar(const AscendIndexVStar&) = delete;
    AscendIndexVStar& operator=(const AscendIndexVStar&) = delete;
private:
    std::shared_ptr<IndexVStar> impl { nullptr };
};
} // namespace faiss
} // namespace ascend
#endif // ASCEND_INDEX_VSTAR_H