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


#ifndef OCK_HCPS_HFO_HETERO_TOKEN_IDX_MAP_H
#define OCK_HCPS_HFO_HETERO_TOKEN_IDX_MAP_H
#include <cstdint>
#include <memory>
#include <vector>
#include "ock/hmm/mgr/OckHmmMemoryPool.h"
#include "ock/hmm/helper/OckHmmAllocator.h"

namespace ock {
namespace hcps {
namespace hfo {

using OckHmmUint32Allocator = hmm::helper::OckHmmAllocator<uint32_t>;
using OckHmmUint32Vector = std::vector<uint32_t, OckHmmUint32Allocator>;
class OckTokenIdxMap {
public:
    virtual ~OckTokenIdxMap() noexcept = default;
    virtual uint32_t GroupId(void) const = 0;
    virtual void SetGroupId(uint32_t grpId) = 0;
    virtual void AddData(uint32_t tokenId, uint32_t rowId) = 0;
    virtual void InsertData(uint32_t tokenId, uint32_t rowId) = 0;
    virtual OckHmmUint32Vector &RowIds(uint32_t tokenId) = 0;
    virtual const OckHmmUint32Vector &RowIds(uint32_t tokenId) const = 0;
    virtual void DeleteToken(uint32_t tokenId) = 0;
    virtual void ClearAll(void) = 0;
    /*
    @brief 按照规格复制内存，分配好内存，但不复制数据
    */
    virtual std::shared_ptr<OckTokenIdxMap> CopySpec(hmm::OckHmmMemoryPool &memPool) const = 0;
    /*
    @brief DeleteData删除速度比较慢，当前认为这种场景比较少
    */
    virtual void DeleteData(uint32_t tokenId, uint32_t rowId) = 0;
    virtual bool InToken(uint32_t tokenId, uint32_t rowId) = 0;
    virtual uint32_t TokenNum(void) const = 0;
    static std::shared_ptr<OckTokenIdxMap> Create(uint32_t tokenNum, hmm::OckHmmMemoryPool &memPool, uint32_t grpId);
};
std::ostream &operator<<(std::ostream &os, const OckTokenIdxMap &idxMap);
}  // namespace hfo
}  // namespace hcps
}  // namespace ock
#endif
