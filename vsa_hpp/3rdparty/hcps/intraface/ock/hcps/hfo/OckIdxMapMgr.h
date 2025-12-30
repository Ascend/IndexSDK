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


#ifndef OCK_HCPS_HFO_HETERO_IDX_MAP_MGR_H
#define OCK_HCPS_HFO_HETERO_IDX_MAP_MGR_H
#include <cstdint>

namespace ock {
namespace hcps {
namespace hfo {
const uint64_t INVALID_IDX_VALUE = 0xFFFFFFFFFFFFFFFFULL;  // 不合法的idx

class OckIdx2OutterMapBase {
public:
    virtual ~OckIdx2OutterMapBase() noexcept = default;
    // INVALID_IDX_VALUE为非法值
    virtual uint64_t GetOutterIdx(uint64_t innerIdx) const = 0;
};
class OckIdx2InnerMapBase {
public:
    virtual ~OckIdx2InnerMapBase() noexcept = default;
    // INVALID_IDX_VALUE为非法值
    virtual uint64_t GetInnerIdx(uint64_t outterIdx) const = 0;
};
class OckIdxMapMgr : public OckIdx2OutterMapBase, public OckIdx2InnerMapBase {
public:
    virtual ~OckIdxMapMgr() noexcept = default;
};
}  // namespace hfo
}  // namespace hcps
}  // namespace ock
#endif
