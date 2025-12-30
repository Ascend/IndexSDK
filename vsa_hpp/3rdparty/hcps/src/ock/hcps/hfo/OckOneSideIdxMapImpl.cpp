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

#include <cstdlib>
#include <cstdint>
#include <securec.h>
#include <vector>
#include <memory>
#include <cstring>
#include "ock/hmm/mgr/OckHmmMemoryPool.h"
#include "ock/log/OckHcpsLogger.h"
#include "ock/hcps/hfo/OckOneSideIdxMap.h"
#include "ock/hmm/helper/OckHmmAllocator.h"
#include "ock/hcps/error/OckHcpsErrorCode.h"
#include "ock/utils/OckSafeUtils.h"

namespace ock {
namespace hcps {
namespace hfo {
class OckOneSideIdxMapImpl : public OckOneSideIdxMap {
public:
    using OutterAllocator = std::allocator<uint64_t>;
    using OutterVector = std::vector<uint64_t, OutterAllocator>;
    OckOneSideIdxMapImpl(uint64_t maximumCount, hmm::OckHmmMemoryPool &memPool)
        : maxCount(maximumCount), curCount(0), dataVec(maxCount, INVALID_IDX_VALUE, OutterAllocator())
    {}

    void Add(uint64_t idx) override
    {
        if (curCount >= maxCount) {
            OCK_HCPS_LOG_ERROR("Add(idx=" << idx << ") index exceed maxCount[" << maxCount << "]");
            return;
        }
        dataVec[curCount++] = idx;
    }

    void BatchAdd(const uint64_t *idxAddr, uint64_t addCount) override
    {
        if (idxAddr == nullptr) {
            return;
        }
        if (curCount + addCount > maxCount) {
            OCK_HCPS_LOG_ERROR("BatchAdd(addCount=" << addCount << ") index exceed maxCount[" << maxCount << "]");
            return;
        }
        if (addCount != 0) {
            auto ret = memcpy_s(&dataVec[curCount], addCount * sizeof(uint64_t), idxAddr, addCount * sizeof(uint64_t));
            if (ret != EOK) {
                OCK_HCPS_LOG_ERROR("memcpy_s failed, the errorCode is " << ret);
                return;
            }
        }
        curCount += addCount;
    }

    void Reset(void) override
    {
        curCount = 0;
    }

    uint64_t GetIdx(uint64_t pos) const override
    {
        if (pos >= curCount) {
            OCK_HCPS_LOG_ERROR("GetIdx(pos=" << pos << ") index exceed curCount[" << curCount << "]");
            return INVALID_IDX_VALUE;
        }
        return dataVec[pos];
    }

    uint64_t Count(void) const override
    {
        return curCount;
    }

    uint64_t MaxCount(void) const override
    {
        return maxCount;
    }

    const uint64_t *GetData() const override
    {
        return dataVec.data();
    }

    void AddFrom(const OckOneSideIdxMap &other, uint64_t otherStartIdx, uint64_t count) override
    {
        if (otherStartIdx + count > other.Count()) {
            OCK_HCPS_LOG_ERROR("AddFrom: invalid start index or copy count");
            return;
        }
        if (count != 0) {
            auto ret = memcpy_s(dataVec.data() + curCount, count * sizeof(uint64_t), other.GetData() + otherStartIdx,
                                count * sizeof(uint64_t));
            if (ret != EOK) {
                OCK_HCPS_LOG_ERROR("memcpy_s failed, copy count is " << (count * sizeof(uint64_t)));
                return;
            }
        }
        curCount += count;
    }

private:
    const uint64_t maxCount{ 0 };
    uint64_t curCount{ 0 };
    OutterVector dataVec{};
};
std::shared_ptr<OckOneSideIdxMap> OckOneSideIdxMap::Create(uint64_t maxCount, hmm::OckHmmMemoryPool &memPool)
{
    return std::make_shared<OckOneSideIdxMapImpl>(maxCount, memPool);
}
std::ostream &operator << (std::ostream &os, const OckOneSideIdxMap &idxMap)
{
    return os << "{'maxCount':" << idxMap.MaxCount() << ",'curCount':" << idxMap.Count() << "}";
}
} // namespace hfo
} // namespace hcps
} // namespace ock