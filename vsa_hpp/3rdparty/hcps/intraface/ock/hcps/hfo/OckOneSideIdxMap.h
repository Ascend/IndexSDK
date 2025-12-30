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


#ifndef OCK_HCPS_HFO_HETERO_ONE_SIDE_IDX_MAP_H
#define OCK_HCPS_HFO_HETERO_ONE_SIDE_IDX_MAP_H
#include <cstdint>
#include <memory>
#include <vector>
#include "ock/hmm/mgr/OckHmmMemoryPool.h"
#include "ock/hcps/hfo/OckIdxMapMgr.h"

namespace ock {
namespace hcps {
namespace hfo {
class OckOneSideIdxMap {
public:
    virtual ~OckOneSideIdxMap() noexcept = default;
    virtual void Add(uint64_t idx) = 0;
    virtual void BatchAdd(const uint64_t *idxAddr, uint64_t addCount) = 0;
    virtual void Reset(void) = 0;  // 清空当数据， 将curCount设置为0
    virtual uint64_t GetIdx(uint64_t pos) const = 0;
    virtual uint64_t Count(void) const = 0;
    virtual uint64_t MaxCount(void) const = 0;
    virtual const uint64_t* GetData() const = 0;
    virtual void AddFrom(const OckOneSideIdxMap &other, uint64_t otherStartIdx, uint64_t count) = 0;

    static std::shared_ptr<OckOneSideIdxMap> Create(uint64_t maxCount, hmm::OckHmmMemoryPool &memPool);
};
std::ostream &operator<<(std::ostream &os, const OckOneSideIdxMap &idxMap);
}  // namespace hfo
}  // namespace hcps
}  // namespace ock
#endif
