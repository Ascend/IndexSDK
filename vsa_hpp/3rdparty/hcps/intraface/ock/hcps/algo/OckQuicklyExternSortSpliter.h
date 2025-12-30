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


#ifndef OCK_HCPS_ALGO_QUICKLY_EXTERN_SORT_SPLITER_H
#define OCK_HCPS_ALGO_QUICKLY_EXTERN_SORT_SPLITER_H
#include <cstdint>
#include <memory>
#include <vector>
#include "ock/utils/OckCompareUtils.h"
#include "ock/hcps/algo/OckSpliterSegement.h"
namespace ock {
namespace hcps {
namespace algo {
template <typename _Tp>
std::shared_ptr<spliter::VectorSortSpliterResult<_Tp>> CalcQuicklyExternSortSpliterSegments(
    std::vector<_Tp> &inputDatas, std::vector<_Tp> &outputDatas, uint64_t splitThreshold = 1024);
} // namespace algo
} // namespace hcps
} // namespace ock
#include "ock/hcps/algo/impl/OckQuicklyExternSortSpliterImpl.h"
#endif
