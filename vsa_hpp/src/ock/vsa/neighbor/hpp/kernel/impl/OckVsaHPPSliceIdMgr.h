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


#ifndef OCK_VSA_HETERO_PIECEWISE_PROGRESSIVE_SLICE_INFO_MGR_H
#define OCK_VSA_HETERO_PIECEWISE_PROGRESSIVE_SLICE_INFO_MGR_H
#include <cstdint>
#include <memory>
#include <unordered_set>
namespace ock {
namespace vsa {
namespace neighbor {
namespace hpp {
namespace impl {
class OckVsaHPPSliceIdMgr {
public:
    virtual ~OckVsaHPPSliceIdMgr() noexcept = default;
    virtual const std::unordered_set<uint32_t> &SliceSet(uint32_t grpId) const = 0;
    virtual std::unordered_set<uint32_t> &SliceSet(uint32_t grpId) = 0;
    virtual bool AddSlice(uint32_t grpId, uint32_t sliceId) = 0;
    virtual uint32_t SliceCount(void) const = 0;
    virtual uint32_t GroupCount(void) const = 0;
    // 此处的group是deque下标概念， 并不是innerIdx中grpId
    static std::shared_ptr<OckVsaHPPSliceIdMgr> Create(uint32_t groupCount);
};

std::ostream &operator << (std::ostream &os, const OckVsaHPPSliceIdMgr &data);
} // namespace impl
} // namespace hpp
} // namespace neighbor
} // namespace vsa
} // namespace ock
#endif