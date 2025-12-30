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


#ifndef VSTAR_MOCK_H
#define VSTAR_MOCK_H

#include <memory>
#include <gmock/gmock.h>
#include "IndexVStar.h"

namespace faiss {
namespace ascend {

class VstarMock : public IndexVStar {
public:
    MOCK_METHOD(APP_ERROR, LoadIndex, (const std::string&, IndexVStar*));
    MOCK_METHOD(APP_ERROR, WriteIndex, (const std::string&));
    MOCK_METHOD(APP_ERROR, AddCodeBooksByIndex, (IndexVStar&));
    MOCK_METHOD(APP_ERROR, AddCodeBooksByPath, (const std::string&));
    MOCK_METHOD(APP_ERROR, Add, (const std::vector<float>&));
    MOCK_METHOD(APP_ERROR, AddWithIds, (const std::vector<float>&, const std::vector<int64_t>&));
    MOCK_METHOD(APP_ERROR, DeleteByIds, (const std::vector<int64_t>&));
    MOCK_METHOD(APP_ERROR, DeleteById, (int64_t));
    MOCK_METHOD(APP_ERROR, DeleteByRange, (int64_t, int64_t));
    MOCK_METHOD(APP_ERROR, Search, (const AscendIndexSearchParams&), (const));
    MOCK_METHOD(APP_ERROR, SearchWithMask, (const AscendIndexSearchParams&, const std::vector<uint8_t>&), (const));
    MOCK_METHOD(APP_ERROR, MultiSearch, (std::vector<IndexVStar*>&, const AscendIndexSearchParams&, bool), (const));
    MOCK_METHOD(APP_ERROR, MultiSearchWithMask, (std::vector<IndexVStar*>&, const AscendIndexSearchParams&,
        const std::vector<uint8_t>&, bool), (const));
    MOCK_METHOD(APP_ERROR, SetHyperSearchParams, (const AscendIndexVstarHyperParams&));
    MOCK_METHOD(APP_ERROR, GetHyperSearchParams, (AscendIndexVstarHyperParams&), (const));
    MOCK_METHOD(APP_ERROR, GetDim, (int&), (const));
    MOCK_METHOD(APP_ERROR, GetNTotal, (uint64_t&), (const));
    MOCK_METHOD(APP_ERROR, Reset, ());
    MOCK_METHOD(std::shared_ptr<IndexVStar>, CreateIndex, (const AscendIndexVstarInitParams&));
    MOCK_METHOD(std::shared_ptr<IndexVStar>, CreateIndex, (const std::vector<int>&, bool verbose));

    static std::shared_ptr<VstarMock> defaultIndex;
};
} // namespace faiss
} // namespace ascend
#endif // VSTAR_MOCK_H