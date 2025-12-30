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

#include <cstdint>
#include <algorithm>
#include "ock/log/OckHcpsLogger.h"
#include "ock/hcps/hfo/OckTokenIdxMap.h"

namespace ock {
namespace hcps {
namespace hfo {
class OckTokenIdxMapImpl : public OckTokenIdxMap {
public:
    virtual ~OckTokenIdxMapImpl() noexcept = default;
    OckTokenIdxMapImpl(uint32_t tokenCount, hmm::OckHmmMemoryPool &memPool, uint32_t groupId)
        : tokenNum(tokenCount), grpId(groupId)
    {
        for (uint32_t i = 0; i < tokenNum; ++i) {
            tokenIdxMap.push_back(std::make_shared<OckHmmUint32Vector>(OckHmmUint32Allocator(memPool)));
        }
    }
    OckTokenIdxMapImpl(const OckTokenIdxMapImpl &other, hmm::OckHmmMemoryPool &memPool)
        : tokenNum(other.tokenNum), grpId(other.grpId)
    {
        for (uint32_t i = 0; i < tokenNum; ++i) {
            tokenIdxMap.push_back(
                std::make_shared<OckHmmUint32Vector>(other.tokenIdxMap.at(i)->size(), OckHmmUint32Allocator(memPool)));
        }
    }
    uint32_t GroupId(void) const override
    {
        return grpId;
    }
    void SetGroupId(uint32_t idx) override
    {
        this->grpId = idx;
    }
    void AddData(uint32_t tokenId, uint32_t rowId) override
    {
        if (tokenId >= tokenNum) {
            return;
        }
        tokenIdxMap.at(tokenId)->push_back(rowId);
    }
    void InsertData(uint32_t tokenId, uint32_t rowId) override
    {
        if (tokenId >= tokenNum) {
            return;
        }
        auto &datas = *tokenIdxMap.at(tokenId);
        if (datas.empty()) {
            datas.push_back(rowId);
            return;
        }
        auto iter = std::lower_bound(datas.begin(), datas.end(), rowId);
        datas.insert(iter, rowId);
    }
    OckHmmUint32Vector &RowIds(uint32_t tokenId) override
    {
        return *tokenIdxMap.at(tokenId);
    }
    const OckHmmUint32Vector &RowIds(uint32_t tokenId) const override
    {
        return *tokenIdxMap.at(tokenId);
    }
    void DeleteToken(uint32_t tokenId) override
    {
        if (tokenId >= tokenNum) {
            return;
        }
        tokenIdxMap.at(tokenId)->clear();
    }
    void ClearAll(void) override
    {
        for (auto &data : tokenIdxMap) {
            data->clear();
        }
    }
    std::shared_ptr<OckTokenIdxMap> CopySpec(hmm::OckHmmMemoryPool &pool) const override
    {
        return std::make_shared<OckTokenIdxMapImpl>(*this, pool);
    }
    void DeleteData(uint32_t tokenId, uint32_t rowId) override
    {
        if (tokenId >= tokenNum) {
            return;
        }
        auto &datas = *tokenIdxMap.at(tokenId);
        auto iter = std::lower_bound(datas.begin(), datas.end(), rowId);
        if (iter == datas.end() || *iter != rowId) {
            return;
        }
        datas.erase(iter);
    }
    uint32_t TokenNum(void) const override
    {
        return tokenNum;
    }
    bool InToken(uint32_t tokenId, uint32_t rowId) override
    {
        if (tokenId >= tokenNum) {
            return false;
        }
        auto &datas = *tokenIdxMap.at(tokenId);
        auto iter = std::lower_bound(datas.begin(), datas.end(), rowId);
        if (iter == datas.end() || *iter != rowId) {
            return false;
        } else {
            return true;
        }
    }
    const uint32_t tokenNum{ 0 };
    uint32_t grpId{ 0 };
    std::vector<std::shared_ptr<OckHmmUint32Vector>> tokenIdxMap{};
};

std::shared_ptr<OckTokenIdxMap> OckTokenIdxMap::Create(
    uint32_t tokenNum, hmm::OckHmmMemoryPool &memPool, uint32_t grpId)
{
    return std::make_shared<OckTokenIdxMapImpl>(tokenNum, memPool, grpId);
}
std::ostream &operator<<(std::ostream &os, const OckTokenIdxMap &idxMap)
{
    return os << "{'tokenNum':" << idxMap.TokenNum() << ",'grpId':" << idxMap.GroupId() << "}";
}
}  // namespace hfo
}  // namespace hcps
}  // namespace ock
