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


#ifndef OCK_HCPS_PIER_EXTERNAL_QUICKLY_SORT_OP_H
#define OCK_HCPS_PIER_EXTERNAL_QUICKLY_SORT_OP_H
#include <cstdint>
#include <memory>
#include <algorithm>
#include <vector>
#include "ock/hcps/stream/OckHeteroOperatorBase.h"
#include "ock/hcps/stream/OckHeteroStreamBase.h"
namespace ock {
namespace hcps {
namespace hop {

template <typename _Tp, typename _CompareT>
void ExternalQuicklySort(OckHeteroStreamBase &stream, std::vector<_Tp> &inputDatas, std::vector<_Tp> &outputDatas,
    const _CompareT &compare, uint64_t splitThreshold = 102400);

template <typename _Tp, typename _CompareT>
void ExternalQuicklyMerge(OckHeteroStreamBase &stream, std::vector<std::vector<_Tp>> &sortedGroups,
    std::vector<_Tp> &outSortedData, std::vector<_Tp> &tmpData, const _CompareT &compare,
    uint64_t splitThreshold = 102400);
}  // namespace hop
}  // namespace hcps
}  // namespace ock
#include "ock/hcps/hop/impl/OckExternalQuicklySortOpImpl.h"
#endif