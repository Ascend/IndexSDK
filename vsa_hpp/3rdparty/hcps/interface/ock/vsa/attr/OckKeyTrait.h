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


#ifndef OCK_VSA_NEIGHBOR_ATTR_TRAITS_H
#define OCK_VSA_NEIGHBOR_ATTR_TRAITS_H
#include <ostream>
#include <type_traits>
#include <unordered_set>
#include <tuple>
#include "ock/utils/OckLimits.h"
#include "ock/utils/OckTypeTraits.h"
#include "ock/utils/OstreamUtils.h"
#include "ock/hcps/algo/OckElasticBitSet.h"
namespace ock {
namespace vsa {
namespace attr {
struct OckKeyTrait {};
} // namespace attr
} // namespace vsa
} // namespace ock
#endif