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


#include <iostream>
#include "IndexGreat.h"
namespace faiss {
namespace ascend {
std::shared_ptr<IndexGreat> IndexGreat::CreateIndex(const std::string&,
                                                    const std::vector<int>&,
                                                    const bool)
{
    std::cout << "create great index stub" << std::endl;
    return nullptr;
}

std::shared_ptr<IndexGreat> IndexGreat::CreateIndex(const AscendIndexGreatInitParams&)
{
    std::cout << "create great index stub" << std::endl;
    return nullptr;
}

std::shared_ptr<IndexGreat> IndexGreat::CreateIndex(const AscendIndexVstarInitParams&,
                                                    const AscendIndexGreatInitParams&)
{
    std::cout << "create great index stub" << std::endl;
    return nullptr;
}
}
}