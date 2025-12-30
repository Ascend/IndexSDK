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


#ifndef OCK_VSA_HETERO_PIECEWISE_PROGRESSIVE_KERNEL_FILTER_TYPE_H
#define OCK_VSA_HETERO_PIECEWISE_PROGRESSIVE_KERNEL_FILTER_TYPE_H
#include <cstdint>
#include <utility>
#include <ostream>
namespace ock {
namespace vsa {
namespace neighbor {
namespace hpp {
enum class OckVsaHPPFilterType : int32_t {
    FULL_FILTER,
    WHOLE_SLICE,
    SLICE_FILTER
};

inline std::ostream &operator << (std::ostream &os, OckVsaHPPFilterType filterType)
{
    switch (filterType) {
        case ock::vsa::neighbor::hpp::OckVsaHPPFilterType::FULL_FILTER:
            os << "FULL_FILTER";
            break;
        case ock::vsa::neighbor::hpp::OckVsaHPPFilterType::WHOLE_SLICE:
            os << "WHOLE_SLICE";
            break;
        case ock::vsa::neighbor::hpp::OckVsaHPPFilterType::SLICE_FILTER:
            os << "SLICE_FILTER";
            break;
        default:
            os << "Unkown(" << static_cast<int32_t>(filterType) << ")";
    }
    return os;
}
} // namespace hpp
} // namespace neighbor
} // namespace vsa
} // namespace ock
#endif