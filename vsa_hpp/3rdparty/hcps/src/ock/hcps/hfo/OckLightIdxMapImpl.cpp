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
#include <cstring>
#include <algorithm>
#include "ock/log/OckHcpsLogger.h"
#include "ock/hcps/hfo/OckLightIdxMap.h"
#include "ock/hmm/helper/OckHmmAllocator.h"
#include "ock/hcps/error/OckHcpsErrorCode.h"
#include "ock/utils/OckSafeUtils.h"

namespace ock {
namespace hcps {
namespace hfo {
namespace {
struct Node {
    Node(void) : innerIdx(INVALID_IDX_VALUE), outterIdx(INVALID_IDX_VALUE) {}
    Node(uint64_t inner, uint64_t outter) : innerIdx(inner), outterIdx(outter) {}
    uint64_t innerIdx;
    uint64_t outterIdx;
};
struct CompareNode {
    bool operator () (const Node &lhs, const Node &rhs) const
    {
        return lhs.outterIdx < rhs.outterIdx;
    }
};
} // namespace
class OckLightIdxMapImpl : public OckLightIdxMap {
public:
    using NodeAllocator = std::allocator<Node>;
    using OutterAllocator = std::allocator<uint64_t>;
    using NodeVector = std::vector<Node, NodeAllocator>;
    using NodeVectorPtrAllocator = std::allocator<NodeVector *>;
    using OutterVector = std::vector<uint64_t, OutterAllocator>;
    OckLightIdxMapImpl(uint64_t maxNum, uint64_t bucketNum)
        : maxCount(maxNum),
          bucketCount(bucketNum),
          threadNumber(32UL),
          bucketCountPerThread(utils::SafeDivUp(bucketCount, threadNumber)),
          innerToOutterMap(maxCount, INVALID_IDX_VALUE, OutterAllocator()),
          outterToInnerMap(bucketCount, NodeVectorPtrAllocator())
    {
        for (size_t pos = 0; pos < outterToInnerMap.size(); ++pos) {
            outterToInnerMap[pos] = new NodeVector(NodeAllocator());
            if (outterToInnerMap[pos] != nullptr) {
                outterToInnerMap[pos]->reserve(utils::SafeDivUp(maxCount, bucketCount));
            }
        }
    }
    virtual ~OckLightIdxMapImpl() noexcept
    {
        for (size_t pos = 0; pos < outterToInnerMap.size(); ++pos) {
            if (outterToInnerMap[pos] != nullptr) {
                delete outterToInnerMap[pos];
            }
        }
    }
    uint64_t GetOutterIdx(uint64_t innerIdx) const override
    {
        if (innerIdx >= maxCount) {
            OCK_HCPS_LOG_ERROR("GetOutterIdx(innerIdx=" << innerIdx << ") innerIdx exceed maxCount[" << maxCount <<
                "]");
            return INVALID_IDX_VALUE;
        }
        return innerToOutterMap[innerIdx];
    }
    uint64_t GetInnerIdx(uint64_t outterIdx) const override
    {
        if (outterIdx >= std::numeric_limits<uint64_t>::max()) {
            OCK_HCPS_LOG_ERROR("GetInnerIdx(outterIdx=" << outterIdx << ", invalid outterIdx=" << outterIdx << ")");
            return INVALID_IDX_VALUE;
        }
        if (outterToInnerMap[utils::SafeMod(outterIdx, bucketCount)] == nullptr) {
            return INVALID_IDX_VALUE;
        }
        return GetIdxMap(*outterToInnerMap[utils::SafeMod(outterIdx, bucketCount)], outterIdx);
    }
    void SetIdxMap(uint64_t innerIdx, uint64_t outterIdx) override
    {
        if (innerIdx >= maxCount) {
            OCK_HCPS_LOG_ERROR("SetIdxMap(innerIdx=" << innerIdx << ",outterIdx=" << outterIdx <<
                ") innerIdx exceed maxCount[" << maxCount << "]");
            return;
        }
        if (outterIdx >= std::numeric_limits<uint64_t>::max()) {
            OCK_HCPS_LOG_ERROR("SetIdxMap(innerIdx=" << innerIdx << ", invalid outterIdx=" << outterIdx << ")");
            return;
        }
        if (outterToInnerMap[utils::SafeMod(outterIdx, bucketCount)] == nullptr) {
            return;
        }
        UpdateIdxMap(*outterToInnerMap[utils::SafeMod(outterIdx, bucketCount)], innerIdx, outterIdx);
        innerToOutterMap[innerIdx] = outterIdx;
    }
    std::shared_ptr<OckHeteroOperatorGroupQueue> CreateRemoveFrontByInnerOps(uint64_t count) override
    {
        auto opDeque = std::make_shared<OckHeteroOperatorGroupQueue>();
        if (count > maxCount) {
            OCK_HCPS_LOG_ERROR("CreateRemoveFrontByInnerOps(invalid count=" << count << ") count exceed maxCount[" <<
                maxCount << "]");
            return opDeque;
        }
        auto outerOps = std::make_shared<OckHeteroOperatorGroup>();
        for (uint64_t i = 0; i < threadNumber; ++i) {
            uint64_t startBucket = i * bucketCountPerThread;
            uint64_t endBucket = std::min(bucketCount, startBucket + bucketCountPerThread);
            outerOps->push_back(OckSimpleHeteroOperator<acladapter::OckTaskResourceType::HOST_CPU>::Create(
                [count, startBucket, endBucket, this](OckHeteroStreamContext &) {
                    SoftDelNodeByCount(count, startBucket, endBucket);
                    return hmm::HMM_SUCCESS;
                }));
        }
        opDeque->push(outerOps);
        auto innerOps = std::make_shared<OckHeteroOperatorGroup>();
        innerOps->push_back(OckSimpleHeteroOperator<acladapter::OckTaskResourceType::HOST_CPU>::Create(
            [count, this](OckHeteroStreamContext &) {
                // 将innerToOutterMap前count项设为INVALID_IDX_VALUE
                if (count == 0) {
                    return hmm::HMM_SUCCESS;
                }
                auto ret = memset_s(innerToOutterMap.data(), count * sizeof(uint64_t), 0xFFL, count * sizeof(uint64_t));
                if (ret != EOK) {
                    OCK_HCPS_LOG_ERROR("memset_s failed, the errorCode is " << ret);
                    return hmm::HMM_ERROR_UNKOWN_INNER_ERROR;
                }
                return hmm::HMM_SUCCESS;
            }));
        opDeque->push(innerOps);
        return opDeque;
    }
    std::shared_ptr<OckHeteroOperatorGroupQueue> CreateSetRemovedOps(uint64_t count, const uint64_t *outterIdx) override
    {
        auto opDeque = std::make_shared<OckHeteroOperatorGroupQueue>();
        if (count > maxCount) {
            OCK_HCPS_LOG_ERROR("CreateSetRemovedOps(invalid count=" << count << ") count exceed maxCount[" <<
                maxCount << "]");
            return opDeque;
        }
        if (outterIdx == nullptr) {
            return opDeque;
        }
        auto ops = std::make_shared<OckHeteroOperatorGroup>();
        for (uint64_t i = 0; i < threadNumber; ++i) {
            uint64_t startBucket = i * bucketCountPerThread;
            uint64_t endBucket = std::min(bucketCount, startBucket + bucketCountPerThread);
            ops->push_back(OckSimpleHeteroOperator<acladapter::OckTaskResourceType::HOST_CPU>::Create(
                [count, startBucket, endBucket, outterIdx, this](OckHeteroStreamContext &) {
                    SoftDelDataByOtterIdx(count, startBucket, endBucket, outterIdx);
                    return hmm::HMM_SUCCESS;
                }));
        }
        opDeque->push(ops);
        return opDeque;
    }
    std::shared_ptr<OckHeteroOperatorGroupQueue> CreateSetRemovedOps(uint64_t count, uint64_t innerStartIdx) override
    {
        auto opDeque = std::make_shared<OckHeteroOperatorGroupQueue>();
        if (innerStartIdx + count > maxCount) {
            OCK_HCPS_LOG_ERROR("CreateSetRemovedOps(innerStartIdx=" << innerStartIdx << " + count=" << count <<
                ") count exceed maxCount[" << maxCount << "]");
            return opDeque;
        }
        auto outerOps = std::make_shared<OckHeteroOperatorGroup>();
        for (uint64_t i = 0; i < threadNumber; ++i) {
            uint64_t startBucket = i * bucketCountPerThread;
            uint64_t endBucket = std::min(bucketCount, startBucket + bucketCountPerThread);
            outerOps->push_back(OckSimpleHeteroOperator<acladapter::OckTaskResourceType::HOST_CPU>::Create(
                [count, startBucket, endBucket, innerStartIdx, this](OckHeteroStreamContext &) {
                    SoftDelNodeByInnerIdx(count, startBucket, endBucket, innerStartIdx);
                    return hmm::HMM_SUCCESS;
                }));
        }
        opDeque->push(outerOps);
        auto innerOps = std::make_shared<OckHeteroOperatorGroup>();
        innerOps->push_back(OckSimpleHeteroOperator<acladapter::OckTaskResourceType::HOST_CPU>::Create(
            [count, innerStartIdx, this](OckHeteroStreamContext &) {
                // 将innerToOutterMap从innerStartIdx开始的count项设为INVALID_IDX_VALUE
                if (count == 0) {
                    return hmm::HMM_SUCCESS;
                }
                auto ret = memset_s(innerToOutterMap.data() + innerStartIdx, count * sizeof(uint64_t), 0xFFL,
                    count * sizeof(uint64_t));
                if (ret != EOK) {
                    OCK_HCPS_LOG_ERROR("memset_s failed, the errorCode is " << ret);
                    return hmm::HMM_ERROR_UNKOWN_INNER_ERROR;
                }
                return hmm::HMM_SUCCESS;
            }));
        opDeque->push(innerOps);
        return opDeque;
    }
    std::shared_ptr<OckHeteroOperatorGroup> CreateAddDatasOps(uint64_t count, const uint64_t *outterIdx,
        uint64_t innerStartIdx) override
    {
        auto ret = std::make_shared<OckHeteroOperatorGroup>();
        if (innerStartIdx + count > maxCount) {
            OCK_HCPS_LOG_ERROR("CreateAddDatasOps(innerStartIdx=" << innerStartIdx << " + count=" << count <<
                ") count exceed maxCount[" << maxCount << "]");
            return ret;
        }
        if (outterIdx == nullptr) {
            return ret;
        }
        for (uint64_t i = 0; i < threadNumber; ++i) {
            uint64_t startBucket = i * bucketCountPerThread;
            uint64_t endBucket = std::min(bucketCount, startBucket + bucketCountPerThread);
            ret->push_back(OckSimpleHeteroOperator<acladapter::OckTaskResourceType::HOST_CPU>::Create(
                [count, startBucket, endBucket, outterIdx, innerStartIdx, this](OckHeteroStreamContext &) {
                    BulkAddDatas(count, startBucket, endBucket, outterIdx, innerStartIdx);
                    return hmm::HMM_SUCCESS;
                }));
        }
        return ret;
    }
    void SetRemoved(uint64_t outterIdx) override
    {
        // outterToInnerMap 中的数据软删除
        uint64_t innerIdx = RemoveOutterToInnerData(outterIdx);
        // innerToOutter中数据打上删除标签
        if (innerIdx != INVALID_IDX_VALUE) {
            innerToOutterMap[innerIdx] = INVALID_IDX_VALUE;
        }
    }

    void SetRemovedByInnerId(uint64_t innerIdx) override
    {
        uint64_t outterIdx = innerToOutterMap[innerIdx];
        if (outterIdx != INVALID_IDX_VALUE) {
            RemoveOutterToInnerData(outterIdx);
        }
        innerToOutterMap[innerIdx] = INVALID_IDX_VALUE;
    }

    void Delete(uint64_t outterIdx) override
    {
        if (outterToInnerMap[utils::SafeMod(outterIdx, bucketCount)] == nullptr) {
            return;
        }
        NodeVector &datas = *outterToInnerMap[utils::SafeMod(outterIdx, bucketCount)];
        auto curIter = std::lower_bound(datas.begin(), datas.end(), Node(INVALID_IDX_VALUE, outterIdx), CompareNode());
        if (curIter == datas.end() || curIter->outterIdx != outterIdx) {
            return;
        }
        uint64_t innerIdx = curIter->innerIdx;

        // outter表删除对应条目
        datas.erase(curIter);
        // inner表软删除
        innerToOutterMap[innerIdx] = INVALID_IDX_VALUE;
        innerValidSize--;
    }

    void DeleteByInnerId(uint64_t innerIdx) override
    {
        if (innerIdx >= maxCount || innerToOutterMap[innerIdx] == INVALID_IDX_VALUE) {
            return;
        }
        uint64_t outterIdx = innerToOutterMap[innerIdx];
        if (outterToInnerMap[utils::SafeMod(outterIdx, bucketCount)] == nullptr) {
            return;
        }
        NodeVector &datas = *outterToInnerMap[utils::SafeMod(outterIdx, bucketCount)];
        auto curIter = std::lower_bound(datas.begin(), datas.end(), Node(INVALID_IDX_VALUE, outterIdx), CompareNode());

        datas.erase(curIter);
        innerToOutterMap[innerIdx] = INVALID_IDX_VALUE;
        innerValidSize--;
    }

    void BatchDeleteByInnerId(const std::vector<uint64_t> &innerIds, const std::vector<uint64_t> &copyIds,
        uint32_t count) override
    {
        if (innerIds.empty() || copyIds.empty()) {
            OCK_HCPS_LOG_ERROR("BatchDeleteByInnerId(innerIds or copyIds is empty)");
            return;
        }
        if (innerIds.size() != copyIds.size()) {
            OCK_HCPS_LOG_ERROR("BatchDeleteByInnerId(innerIds size is not equal to the copyIds size)");
            return;
        }
        if (count <= 0) {
            OCK_HCPS_LOG_ERROR("BatchDeleteByInnerId(count must be a positive integer!)");
            return;
        }
        uint64_t tmpInnerId;
        uint64_t tmpOutterId;
        uint64_t lastOutterId;

        for (uint32_t i = 0; i < count; i++) {
            tmpInnerId = innerIds[i];
            tmpOutterId = innerToOutterMap[tmpInnerId];
            lastOutterId = innerToOutterMap[copyIds[i]];
            if (outterToInnerMap[utils::SafeMod(tmpOutterId, bucketCount)] == nullptr) {
                continue;
            }
            NodeVector &curBucket = *outterToInnerMap[utils::SafeMod(tmpOutterId, bucketCount)];
            auto curIter = std::lower_bound(curBucket.begin(), curBucket.end(), Node(INVALID_IDX_VALUE, tmpOutterId),
                CompareNode());

            // 修改拷贝后的映射关系
            innerToOutterMap[tmpInnerId] = lastOutterId;
            if (outterToInnerMap[utils::SafeMod(lastOutterId, bucketCount)] == nullptr) {
                continue;
            }
            NodeVector &lastBucket = *outterToInnerMap[utils::SafeMod(lastOutterId, bucketCount)];
            auto lastIter = std::lower_bound(lastBucket.begin(), lastBucket.end(),
                Node(INVALID_IDX_VALUE, lastOutterId), CompareNode());
            lastIter->innerIdx = tmpInnerId;

            // 删除最后一条数据
            innerToOutterMap[copyIds[i]] = INVALID_IDX_VALUE;
            curBucket.erase(curIter);
        }
    }

    uint64_t InnerValidSize()
    {
        return innerValidSize;
    }

    bool InOutterMap(uint64_t outterIdx) override
    {
        if (outterToInnerMap[utils::SafeMod(outterIdx, bucketCount)] == nullptr) {
            return false;
        }
        NodeVector &datas = *outterToInnerMap[utils::SafeMod(outterIdx, bucketCount)];
        auto curIter = std::lower_bound(datas.begin(), datas.end(), Node(INVALID_IDX_VALUE, outterIdx), CompareNode());
        if (curIter == datas.end() || curIter->outterIdx != outterIdx || curIter->innerIdx == INVALID_IDX_VALUE) {
            return false;
        }
        return true;
    }

    void BatchRemoveByInner(uint64_t innerStartIdx, uint64_t count) override
    {
        if (innerStartIdx + count > maxCount) {
            OCK_HCPS_LOG_ERROR("BatchRemoveByInner(innerStartIdx=" << innerStartIdx << " + count=" << count <<
                ") count exceed maxCount[" << maxCount << "]");
            return;
        }
        for (uint64_t pos = 0; pos < count; ++pos) {
            // outterToInnerMap 中的数据打上删除标签
            RemoveOutterToInnerData(innerToOutterMap[innerStartIdx + pos]);
        }
        // 将innerToOutterMap从innerStartIdx位置开始的count项设为INVALID_IDX_VALUE
        if (count != 0) {
            auto ret = memset_s(innerToOutterMap.data() + innerStartIdx, count * sizeof(uint64_t), 0xFFL,
                count * sizeof(uint64_t));
            if (ret != EOK) {
                OCK_HCPS_LOG_ERROR("memset_s failed, the errorCode is " << ret);
            }
        }
    }

    void RemoveFrontByInner(uint64_t count) override
    {
        BatchRemoveByInner(0ULL, count);
    }
    std::vector<uint64_t> GetOutterIdxs(uint64_t innerStartIdx, uint64_t count) const override
    {
        std::vector<uint64_t> outterIdxs(count);
        if (count != 0) {
            auto ret = memcpy_s(outterIdxs.data(), outterIdxs.size() * sizeof(uint64_t),
                &innerToOutterMap[innerStartIdx], count * sizeof(uint64_t));
            if (ret != EOK) {
                OCK_HCPS_LOG_ERROR("memcpy_s failed, the errorCode is " << ret);
                return outterIdxs;
            }
        }
        return outterIdxs;
    }
    void GetOutterIdxs(uint64_t innerStartIdx, uint64_t count, uint64_t *pOutDataAddr) const override
    {
        if (pOutDataAddr == nullptr) {
            return;
        }
        if (count != 0) {
            auto ret = memcpy_s(pOutDataAddr, count * sizeof(uint64_t), &innerToOutterMap[innerStartIdx],
                count * sizeof(uint64_t));
            if (ret != EOK) {
                OCK_HCPS_LOG_ERROR("memcpy_s failed, the errorCode is " << ret);
                return;
            }
        }
    }
    void GetOutterIdxs(uint64_t innerStartIdx, uint64_t count, OckOneSideIdxMap &outData) const override
    {
        // 检查 innerStartIdx 是否超出内部idx的范围
        if (innerStartIdx >= maxCount) {
            OCK_HCPS_LOG_ERROR("GetOutterIdxs(innerStartIdx=" << innerStartIdx << ") innerStartIdx exceed maxCount[" <<
                maxCount << "]");
            return;
        }
        outData.BatchAdd(&innerToOutterMap[innerStartIdx], count);
    }
    void GetOutterIdxs(uint64_t *innerStartIdx, uint64_t count, uint64_t *pOutDataAddr) const override
    {
        if (innerStartIdx == nullptr || pOutDataAddr == nullptr) {
            return;
        }
        for (uint64_t i = 0; i < count; ++i) {
            if (innerStartIdx[i] >= maxCount) {
                pOutDataAddr[i] = INVALID_IDX_VALUE;
            } else {
                pOutDataAddr[i] = innerToOutterMap[innerStartIdx[i]];
            }
        }
    }

    std::ostream &Print(std::ostream &os) const override
    {
        return os << "{'maxCount':" << maxCount << ",'bucketCount':" << bucketCount << ",'threadNumber':" <<
            threadNumber << ",'bucketCountPerThread':" << bucketCountPerThread << "}";
    }

private:
    void BulkAddDatas(uint64_t count, uint64_t beginBucket, uint64_t endBucket, const uint64_t *outterIdx,
        uint64_t innerStartIdx)
    {
        for (uint64_t i = 0; i < count; i++) {
            uint64_t bucketId = utils::SafeMod(outterIdx[i], bucketCount);
            if (bucketId >= beginBucket && bucketId < endBucket) {
                SetIdxMap(innerStartIdx + i, outterIdx[i]);
            }
        }
    }

    void SoftDelNodeByInnerIdx(uint64_t count, uint64_t beginBucket, uint64_t endBucket, uint64_t innerStartIdx)
    {
        for (uint64_t j = 0; j < count; ++j) {
            uint64_t outter = innerToOutterMap[innerStartIdx + j];
            uint64_t bucketId = utils::SafeMod(outter, bucketCount);
            if (bucketId >= beginBucket && bucketId < endBucket) {
                RemoveOutterToInnerData(outter);
            }
        }
    }

    void SoftDelDataByOtterIdx(uint64_t count, uint64_t beginBucket, uint64_t endBucket, const uint64_t *outterIdx)
    {
        for (uint64_t j = 0; j < count; ++j) {
            uint64_t outter = outterIdx[j];
            uint64_t bucketId = utils::SafeMod(outter, bucketCount);
            if (bucketId >= beginBucket && bucketId < endBucket) {
                // 软删除outterToInnerMap和innerToOutterMap中的数据
                uint64_t innerIdx = RemoveOutterToInnerData(outter);
                if (innerIdx != INVALID_IDX_VALUE) {
                    innerToOutterMap[innerIdx] = INVALID_IDX_VALUE;
                }
            }
        }
    }

    void SoftDelNodeByCount(uint64_t count, uint64_t beginBucket, uint64_t endBucket)
    {
        for (uint64_t j = 0; j < count; ++j) {
            uint64_t outter = innerToOutterMap[j];
            uint64_t bucketId = utils::SafeMod(outter, bucketCount);
            if (bucketId >= beginBucket && bucketId < endBucket) {
                RemoveOutterToInnerData(outter);
            }
        }
    }

    uint64_t GetIdxMap(const NodeVector &datas, uint64_t outterIdx) const
    {
        if (datas.size() == 0) {
            return INVALID_IDX_VALUE;
        }
        auto iter = std::lower_bound(datas.begin(), datas.end(), Node(INVALID_IDX_VALUE, outterIdx), CompareNode());
        if (iter == datas.end() || iter->outterIdx != outterIdx) {
            return INVALID_IDX_VALUE;
        }
        return iter->innerIdx;
    }

    void UpdateIdxMap(NodeVector &datas, uint64_t innerIdx, uint64_t outterIdx)
    {
        if (datas.empty()) {
            datas.emplace_back(innerIdx, outterIdx);
            innerValidSize++;
            return;
        }
        Node newNode(innerIdx, outterIdx);
        auto iter = std::lower_bound(datas.begin(), datas.end(), newNode, CompareNode());
        if (iter == datas.end()) {
            datas.emplace_back(innerIdx, outterIdx);
            innerValidSize++;
        } else if (iter->outterIdx != outterIdx) {
            datas.insert(iter, newNode);
            innerValidSize++;
        } else {
            iter->innerIdx = innerIdx;
        }
    }

    /*
    @brief  outterToInnerMap打上删除标签， 并返回innerIdx, 方便对innerToOuttterMap的修改
    */
    uint64_t RemoveOutterToInnerData(uint64_t outterIdx)
    {
        if (outterIdx >= std::numeric_limits<uint64_t>::max()) {
            OCK_HCPS_LOG_WARN("RemoveOutterToInnerData(invalid outterIdx=" << outterIdx << ")");
            return INVALID_IDX_VALUE;
        }
        uint64_t offset = utils::SafeMod(outterIdx, bucketCount);
        if (outterToInnerMap[offset] == nullptr) {
            return INVALID_IDX_VALUE;
        }
        NodeVector &datas = *outterToInnerMap[offset];
        auto iter = std::lower_bound(datas.begin(), datas.end(), Node(INVALID_IDX_VALUE, outterIdx), CompareNode());
        if (iter == datas.end() || iter->outterIdx != outterIdx) {
            return INVALID_IDX_VALUE;
        }
        uint64_t innerIdx = iter->innerIdx;
        iter->innerIdx = INVALID_IDX_VALUE;
        innerValidSize--;
        return innerIdx;
    }
    const uint64_t maxCount;
    const uint64_t bucketCount;
    const uint64_t threadNumber;
    const uint64_t bucketCountPerThread;
    uint64_t innerValidSize = 0;
    OutterVector innerToOutterMap;
    std::vector<NodeVector *, NodeVectorPtrAllocator> outterToInnerMap;
};
std::shared_ptr<OckLightIdxMap> OckLightIdxMap::Create(uint64_t maxCount, hmm::OckHmmMemoryPool &memPool,
    uint64_t bucketCount)
{
    return std::make_shared<OckLightIdxMapImpl>(maxCount, bucketCount);
}
std::ostream &operator << (std::ostream &os, const OckLightIdxMap &idxMap)
{
    return idxMap.Print(os);
}
} // namespace hfo
} // namespace hcps
} // namespace ock
