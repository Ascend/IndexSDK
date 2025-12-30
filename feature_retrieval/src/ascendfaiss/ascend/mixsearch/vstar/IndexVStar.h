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


#ifndef INDEX_VSTAR_H
#define INDEX_VSTAR_H

#include <string>
#include <vector>
#include <memory>
#include "common/ErrorCode.h"
#include "common/utils/LogUtils.h"
#include "AscendIndexMixSearchParams.h"

using namespace ascend;

namespace faiss {
namespace ascend {
class IndexVStar {
public:
    IndexVStar() = default;

    virtual ~IndexVStar() = default;

    virtual APP_ERROR LoadIndex(const std::string& indexPath, IndexVStar* indexVStar = nullptr) = 0;

    virtual APP_ERROR WriteIndex(const std::string& indexPath) = 0;

    virtual APP_ERROR AddCodeBooksByIndex(IndexVStar& indexVStar) = 0;

    virtual APP_ERROR AddCodeBooksByPath(const std::string& codebooksPath) = 0;

    virtual APP_ERROR Add(const std::vector<float>& baseData) = 0;

    virtual APP_ERROR AddWithIds(const std::vector<float>& baseData, const std::vector<int64_t>& ids) = 0;

    virtual APP_ERROR DeleteByIds(const std::vector<int64_t>& ids) = 0;

    virtual APP_ERROR DeleteById(int64_t id) = 0;

    virtual APP_ERROR DeleteByRange(int64_t startId, int64_t endId) = 0;

    virtual APP_ERROR Search(const AscendIndexSearchParams& params) const = 0;

    virtual APP_ERROR SearchWithMask(const AscendIndexSearchParams& params, const std::vector<uint8_t>& mask) const = 0;

    virtual APP_ERROR MultiSearch(std::vector<IndexVStar*>& indexes,
        const AscendIndexSearchParams& params, bool merge) const = 0;

    virtual APP_ERROR MultiSearchWithMask(std::vector<IndexVStar*>& indexes, const AscendIndexSearchParams& params,
        const std::vector<uint8_t>& mask, bool merge) const = 0;

    virtual APP_ERROR SetHyperSearchParams(const AscendIndexVstarHyperParams& params) = 0;

    virtual APP_ERROR GetHyperSearchParams(AscendIndexVstarHyperParams& params) const = 0;

    virtual APP_ERROR GetDim(int& dim) const = 0;

    virtual APP_ERROR GetNTotal(uint64_t& ntotal) const = 0;

    virtual APP_ERROR Reset() = 0;

    static std::shared_ptr<IndexVStar> CreateIndex(const AscendIndexVstarInitParams& params);
    static std::shared_ptr<IndexVStar> CreateIndex(const std::vector<int>& deviceList, bool verbose = false);
};
} // namespace faiss
} // namespace ascend
#endif // INDEX_VSTAR_H