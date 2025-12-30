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


#include "GreatMock.h"

using namespace faiss::ascend;
using namespace std;

shared_ptr<GreatMock> GreatMock::defaultGreat;

shared_ptr<IndexGreat> IndexGreat::CreateIndex(const std::string& mode,
    const std::vector<int>& deviceList, bool verbose)
{
    return GreatMock::defaultGreat->CreateIndex(mode, deviceList, verbose);
}

shared_ptr<IndexGreat> IndexGreat::CreateIndex(const AscendIndexGreatInitParams& kModeInitParams)
{
    return GreatMock::defaultGreat->CreateIndex(kModeInitParams);
}

shared_ptr<IndexGreat> IndexGreat::CreateIndex(const AscendIndexVstarInitParams& aModeInitParams,
                                               const AscendIndexGreatInitParams& kModeInitParams)
{
    return GreatMock::defaultGreat->CreateIndex(aModeInitParams, kModeInitParams);
}