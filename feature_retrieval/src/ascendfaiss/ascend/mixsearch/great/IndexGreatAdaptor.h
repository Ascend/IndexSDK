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


#ifndef INDEX_GREAT_ADAPTOR_H
#define INDEX_GREAT_ADAPTOR_H
#include "IndexGreat.h"
#include "npu/IndexGreat.h"
namespace faiss {
namespace ascend {
class IndexGreatAdaptor : public IndexGreat {
public:
    IndexGreatAdaptor(const std::string& mode, const std::vector<int>& deviceList, const bool verbose);

    explicit IndexGreatAdaptor(const AscendIndexGreatInitParams& kModeInitParams);

    IndexGreatAdaptor(const AscendIndexVstarInitParams& aModeInitParams,
                      const AscendIndexGreatInitParams& kModeInitParams);

    ~IndexGreatAdaptor() override = default;

    APP_ERROR LoadIndex(const std::string& indexPath) override;

    APP_ERROR LoadIndex(const std::string& aModeIndexPath, const std::string& kModeIndexPath) override;

    APP_ERROR WriteIndex(const std::string& indexPath) override;
    
    APP_ERROR WriteIndex(const std::string& aModeIndexPath, const std::string& kModeIndexPath) override;

    APP_ERROR Add(const std::vector<float>& baseRawData) override;

    APP_ERROR AddWithIds(const std::vector<float>& baseRawData, const std::vector<int64_t>& ids) override;

    APP_ERROR AddCodeBooks(const std::string& codeBooksPath) override;

    APP_ERROR Search(const AscendIndexSearchParams& params) override;

    APP_ERROR SearchWithMask(const AscendIndexSearchParams& params, const std::vector<uint8_t>& mask) override;

    APP_ERROR SetHyperSearchParams(const AscendIndexHyperParams& params) override;

    APP_ERROR GetHyperSearchParams(AscendIndexHyperParams& params) const override;

    APP_ERROR GetDim(int& dim) const override;

    APP_ERROR GetNTotal(uint64_t& nTotal) const override;

    APP_ERROR Reset() override;

private:
    std::shared_ptr<ascendSearchacc::IndexGreat> instance { nullptr };
};

} // namespace ascend
} // namespace faiss
#endif