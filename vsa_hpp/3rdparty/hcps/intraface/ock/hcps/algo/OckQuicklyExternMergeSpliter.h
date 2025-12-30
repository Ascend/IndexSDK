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


#ifndef OCK_HCPS_ALGO_QUICKLY_EXTERN_MERGE_SPLITER_H
#define OCK_HCPS_ALGO_QUICKLY_EXTERN_MERGE_SPLITER_H
#include <cstdint>
#include <memory>
#include <vector>
#include "ock/utils/OckCompareUtils.h"
#include "ock/hcps/algo/OckSpliterSegement.h"
namespace ock {
namespace hcps {
namespace algo {
/*
@brief 将sortedGroups中的数据排序， sortedGroups中每个std::vector<_Tp>都是排序好的数据。
@outSortedData 排序好的数据存放位置
@tmpData 排序过程中需要用到的临时数据的存放位置
*/
template <typename _Tp>
std::shared_ptr<spliter::VectorExternMergeSpliterResult<_Tp>> CalcQuicklyExternMergeSpliterSegments(
    std::vector<std::vector<_Tp>> &sortedGroups, std::vector<_Tp> &outSortedData, std::vector<_Tp> &tmpData);

}  // namespace algo
}  // namespace hcps
}  // namespace ock
#include "ock/hcps/algo/impl/OckQuicklyExternMergeSpliterImpl.h"
#endif
