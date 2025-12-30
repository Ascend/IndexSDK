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


#ifndef GREAT_MOCK_H
#define GREAT_MOCK_H

#include <memory>
#include <gmock/gmock.h>
#include "AscendIndexMixSearchParams.h"
#include "IndexGreat.h"

using namespace faiss::ascend;

class GreatMock : public IndexGreat {
public:
    MOCK_METHOD(APP_ERROR, Add, (const std::vector<float>&));
    MOCK_METHOD(APP_ERROR, AddWithIds, (const std::vector<float>&, const std::vector<int64_t>&));
    MOCK_METHOD(APP_ERROR, LoadIndex, (const std::string&));
    MOCK_METHOD(APP_ERROR, WriteIndex, (const std::string&));
    MOCK_METHOD(APP_ERROR, LoadIndex, (const std::string&, const std::string&));
    MOCK_METHOD(APP_ERROR, WriteIndex, (const std::string&, const std::string&));
    MOCK_METHOD(APP_ERROR, AddCodeBooks, (const std::string&));
    MOCK_METHOD(APP_ERROR, Search, (const AscendIndexSearchParams&));
    MOCK_METHOD(APP_ERROR, SearchWithMask, (const AscendIndexSearchParams&, const std::vector<uint8_t>&));
    MOCK_METHOD(APP_ERROR, Reset, ());
    MOCK_METHOD(APP_ERROR, GetDim, (int&), (const));
    MOCK_METHOD(APP_ERROR, GetNTotal, (uint64_t&), (const));
    MOCK_METHOD(APP_ERROR, GetHyperSearchParams, (AscendIndexHyperParams&), (const));
    MOCK_METHOD(APP_ERROR, SetHyperSearchParams, (const AscendIndexHyperParams&));
    MOCK_METHOD(std::shared_ptr<IndexGreat>, CreateIndex, (const AscendIndexGreatInitParams&));
    MOCK_METHOD(std::shared_ptr<IndexGreat>, CreateIndex, (const AscendIndexVstarInitParams&,
        const AscendIndexGreatInitParams&));
    MOCK_METHOD(std::shared_ptr<IndexGreat>, CreateIndex, (const std::string&, const std::vector<int>&, bool));

    static std::shared_ptr<GreatMock> defaultGreat;
};

#endif