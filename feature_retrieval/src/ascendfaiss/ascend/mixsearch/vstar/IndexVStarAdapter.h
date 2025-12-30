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


#ifndef INDEX_VSTAR_ADAPTER_H
#define INDEX_VSTAR_ADAPTER_H

#include "IndexVStar.h"
#include "npu/NpuIndexVStar.h"

namespace faiss {
namespace ascend {

class IndexVStarAdapter : public IndexVStar {
public:
    explicit IndexVStarAdapter(const AscendIndexVstarInitParams& params);
    explicit IndexVStarAdapter(const std::vector<int>& deviceList, bool verbose = false);
    ~IndexVStarAdapter() override = default;

    APP_ERROR LoadIndex(const std::string& indexPath, IndexVStar* indexVStar = nullptr) override;

    APP_ERROR WriteIndex(const std::string& indexPath) override;

    APP_ERROR AddCodeBooksByIndex(IndexVStar& indexVStar) override;

    APP_ERROR AddCodeBooksByPath(const std::string& codebooksPath) override;

    APP_ERROR Add(const std::vector<float>& baseData) override;

    APP_ERROR AddWithIds(const std::vector<float>& baseData, const std::vector<int64_t>& ids) override;

    APP_ERROR DeleteByIds(const std::vector<int64_t>& ids) override;

    APP_ERROR DeleteById(int64_t id) override;

    APP_ERROR DeleteByRange(int64_t startId, int64_t endId) override;

    APP_ERROR Search(const AscendIndexSearchParams& params) const override;

    APP_ERROR SearchWithMask(const AscendIndexSearchParams& params, const std::vector<uint8_t>& mask) const override;

    APP_ERROR MultiSearch(std::vector<IndexVStar*>& indexes,
        const AscendIndexSearchParams& params, bool merge) const override;

    APP_ERROR MultiSearchWithMask(std::vector<IndexVStar*>& indexes, const AscendIndexSearchParams& params,
        const std::vector<uint8_t>& mask, bool merge) const override;

    APP_ERROR SetHyperSearchParams(const AscendIndexVstarHyperParams& params) override;

    APP_ERROR GetHyperSearchParams(AscendIndexVstarHyperParams& params) const override;

    APP_ERROR GetDim(int& dim) const override;

    APP_ERROR GetNTotal(uint64_t& ntotal) const override;

    APP_ERROR Reset() override;

    IndexVStarAdapter(const IndexVStarAdapter&) = delete;
    IndexVStarAdapter& operator=(const IndexVStarAdapter&) = delete;

private:
    std::shared_ptr<ascendSearchacc::NpuIndexVStar> instance { nullptr };
};
} // namespace faiss
} // namespace ascend
#endif // INDEX_VSTAR_ADAPTER_H