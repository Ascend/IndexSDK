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


#ifndef OCK_VSA_HETERO_PIECEWISE_PROGRESSIVE_KERNEL_EXT_SELECT_RATE_H
#define OCK_VSA_HETERO_PIECEWISE_PROGRESSIVE_KERNEL_EXT_SELECT_RATE_H
#include <cstdint>
#include <algorithm>
#include "ock/utils/OckSafeUtils.h"
#include "ock/vsa/neighbor/hpp/kernel/impl/OckVsaNeighborRelation.h"
namespace ock {
namespace vsa {
namespace neighbor {
namespace hpp {
namespace impl {
bool UsingPureFilter(double keyCovRate);
uint32_t CalcTopNInSampleGroup(uint32_t topN, uint32_t curGroupCount, double keyCovRate, uint32_t groupRowCount,
    const std::deque<std::shared_ptr<relation::OckVsaNeighborRelationHmoGroup>> &relTable);
} // namespace impl
} // namespace hpp
} // namespace neighbor
} // namespace vsa
} // namespace ock
#endif