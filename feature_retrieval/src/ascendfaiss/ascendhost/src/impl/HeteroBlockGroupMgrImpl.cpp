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


#include "ascenddaemon/utils/StaticUtils.h"
#include "ascendhost/include/impl/HeteroBlockGroupMgr.h"

namespace ascend {
DeviceStorgeInfo::DeviceStorgeInfo(size_t deviceDataNum, size_t deviceBlockNumPerGrp, size_t hostBlockNumPerGrp)
    : maxDeviceDataNum(deviceDataNum), maxDeviceBlockNumPerGrp(deviceBlockNumPerGrp),
      maxHostBlockNumPerGrp(hostBlockNumPerGrp)
{}

HeteroBlockInfo::HeteroBlockInfo(int pId, int block, uint32_t dataNum)
    : pageId(pId), blockId(block), dataNum(dataNum)
{}

HeteroBlockGroupMgr::~HeteroBlockGroupMgr() noexcept {}

size_t CalcGroupNum(int64_t ntotal, int blockSize, int64_t maxDeviceDataNum, size_t maxDeviceBlockNumPerGrp,
    size_t maxHostBlockNumPerGrp)
{
    int groupSize = static_cast<int>(maxDeviceBlockNumPerGrp) * blockSize;
    if (ntotal <= maxDeviceDataNum) {
        return utils::divUp(ntotal, groupSize);
    }
    size_t grpNumByDev = utils::divUp(utils::divUp(maxDeviceDataNum, blockSize), maxDeviceBlockNumPerGrp);
    int64_t maxMixDataSize = static_cast<int64_t>(grpNumByDev) * static_cast<int64_t>(blockSize) *
            static_cast<int64_t>(maxHostBlockNumPerGrp);
    if (ntotal <= maxMixDataSize + maxDeviceDataNum) {
        return grpNumByDev;
    }
    size_t grpNumPureHost = utils::divUp(utils::divUp(ntotal - maxDeviceDataNum - maxMixDataSize, blockSize),
        maxDeviceBlockNumPerGrp + maxHostBlockNumPerGrp);
    return grpNumPureHost + grpNumByDev;
}

class HeteroBlockGroupMgrImpl : public HeteroBlockGroupMgr {
public:
    ~HeteroBlockGroupMgrImpl() noexcept override {}

    template <typename T>
    HeteroBlockGroupMgrImpl(int64_t ntotal, int pageSize, int blockSize, DeviceStorgeInfo *deviceStorgeInfo,
        const std::vector<std::unique_ptr<DeviceVector<T>>> &baseShaped)
        : ntotal(ntotal),
          pageSize(pageSize),
          blockSize(blockSize),
          maxDeviceDataNum(deviceStorgeInfo->maxDeviceDataNum),
          maxDeviceBlockNumPerGrp(deviceStorgeInfo->maxDeviceBlockNumPerGrp),
          maxHostBlockNumPerGrp(deviceStorgeInfo->maxHostBlockNumPerGrp),
          addGrpIdx(0),
          addBlockCountOfGrp(0),
          groupSize((static_cast<int>(deviceStorgeInfo->maxDeviceBlockNumPerGrp) +
                     static_cast<int>(deviceStorgeInfo->maxHostBlockNumPerGrp)) * blockSize),

          groups(CalcGroupNum(ntotal, blockSize, deviceStorgeInfo->maxDeviceDataNum,
                              deviceStorgeInfo->maxDeviceBlockNumPerGrp, deviceStorgeInfo->maxHostBlockNumPerGrp))
    {
        APP_LOG_INFO("initGroups ntotal=%ld, pageSize=%d, blockSize=%d maxDeviceDataNum=%ld, "
                     "maxDeviceBlockNumPerGrp=%zu, maxHostBlockNumPerGrp=%zu",
                     ntotal, pageSize, blockSize, deviceStorgeInfo->maxDeviceDataNum,
                     deviceStorgeInfo->maxDeviceBlockNumPerGrp, deviceStorgeInfo->maxHostBlockNumPerGrp);
        InitGroups();
        APP_LOG_INFO("groups = %zu", groups.size());
        SplitHeteroBlockGroup(baseShaped);
    }

    size_t GetGroupSize() const override
    {
        return groupSize;
    }

    size_t Count() const override
    {
        return groups.size();
    }

    const HeteroBlockGroup &At(size_t pos) const override
    {
        return groups.at(pos);
    }

    ascend::idx_t OrderId2IndexId(ascend::idx_t searchOrderId) const override
    {
        size_t grpPos = utils::divDown(searchOrderId, groupSize);
        size_t blockPos = utils::divDown(utils::mod(searchOrderId, groupSize), blockSize);
        size_t idxInBlock = utils::mod(searchOrderId, blockSize);
        if (grpPos >= groups.size() || blockPos >= groups[grpPos].blocks.size()) {
            return searchOrderId;
        }
        auto &group = groups.at(grpPos);
        return group.blocks.at(blockPos).blockId * blockSize + idxInBlock;
    }

    void OrderIds2IndexIds(int n, int topN, ascend::idx_t *orderIds) const override
    {
        for (int i = 0; i < n; ++i) {
            int offset = i * topN;
            for (int j = 0; j < topN; ++j) {
                orderIds[j + offset] = OrderId2IndexId(orderIds[j + offset]);
            }
        }
    }

private:
    class MemoryLocationFilterBase {
    public:
        virtual ~MemoryLocationFilterBase() noexcept {}

        virtual bool Filter(bool isHostHmo) const = 0;
    };

    class HostMemoryLocationFilter : public MemoryLocationFilterBase {
    public:
        ~HostMemoryLocationFilter() noexcept override {}

        HostMemoryLocationFilter() {}

        bool Filter(bool isHostHmo) const override
        {
            return isHostHmo;
        }
    };

    class DeviceMemoryLocationFilter : public MemoryLocationFilterBase {
    public:
        ~DeviceMemoryLocationFilter() noexcept override {}

        DeviceMemoryLocationFilter() {}

        bool Filter(bool isHostHmo) const override
        {
            return !isHostHmo;
        }
    };

    void InitGroups()
    {
        const size_t blockNum = maxDeviceBlockNumPerGrp + maxHostBlockNumPerGrp;
        for (auto &grp : groups) {
            grp.blocks.reserve(blockNum);
        }
    }

    template <typename T>
    void SplitHeteroBlockGroup(const std::vector<std::unique_ptr<DeviceVector<T>>> &baseShaped)
    {
        if (ntotal > maxDeviceDataNum) {
            SplitHeteroBlockGroupImpl(0, maxDeviceDataNum, baseShaped, DeviceMemoryLocationFilter(),
                maxDeviceBlockNumPerGrp);
            SplitHeteroBlockGroupImpl(maxDeviceDataNum, ntotal, baseShaped, HostMemoryLocationFilter(),
                maxHostBlockNumPerGrp);
        } else {
            SplitHeteroBlockGroupImpl(0, ntotal, baseShaped, DeviceMemoryLocationFilter(), maxDeviceBlockNumPerGrp);
        }
    }

    template <typename T>
    void SplitHeteroBlockGroupImpl(int64_t startPos, int64_t endPos,
        const std::vector<std::unique_ptr<DeviceVector<T>>> &baseShaped,
        const MemoryLocationFilterBase &filter, size_t maxBlockNumPerGrp)
    {
        addGrpIdx = 0;
        addBlockCountOfGrp = 0;
        int64_t startPageId = utils::divDown(startPos, this->pageSize);
        int64_t endPageId = utils::divUp(endPos, this->pageSize);
        APP_LOG_INFO("SplitHeteroBlockGroupImpl startPageId=%d endPageId=%d", startPageId, endPageId);
        for (int64_t pageId = startPageId; pageId < endPageId; ++pageId) {
            int64_t pageOffset = pageId * this->pageSize; // 页偏移，即本页头位置-1
            int64_t blockOffset = utils::divDown(pageId * this->pageSize, blockSize); // block偏移，本页第一个block
             // 本页实际底库量
            int64_t curPageDataNum = std::min(this->ntotal - pageOffset, static_cast<int64_t>(this->pageSize));
            int64_t blockNum = utils::divUp(curPageDataNum, this->blockSize); // 本页实际block数
            for (int64_t i = 0; i < blockNum; i++) { // 逐个操作block
                auto hmo = baseShaped[blockOffset + i]->getHmo(); // 当前block获取hmo
                 // 当前block实际存放底库量
                int64_t computeNum = std::min(curPageDataNum - i * blockSize, static_cast<int64_t>(blockSize));
                if (hmo != nullptr && filter.Filter(hmo->IsHostHMO())) {
                    AddBlock(pageId, static_cast<int>(blockOffset + i), static_cast<uint32_t>(computeNum),
                        maxBlockNumPerGrp, hmo->IsHostHMO());
                }
            }
        }
    }

    void AddBlock(int pageId, int blockId, uint32_t dataNum, size_t maxBlockNumPerGrp, bool isHostHmo)
    {
        auto &grp = groups.at(addGrpIdx);
        grp.blocks.emplace_back(pageId, blockId, dataNum);
        grp.groupType = ToGroupType(grp.groupType, isHostHmo);
        APP_LOG_INFO("AddBlock pageId=%d blockId=%d dataNum=%u isHostHmo=%d grpIdx=%zu", pageId, blockId, dataNum,
            isHostHmo, addGrpIdx);
        addBlockCountOfGrp++;
        if (grp.groupType == HeteroBlockGroupType::BGT_PURE_HOST) {
            if (addBlockCountOfGrp < maxHostBlockNumPerGrp + maxDeviceBlockNumPerGrp) {
                return;
            }
        } else {
            if (addBlockCountOfGrp < maxBlockNumPerGrp) {
                return;
            }
        }
        addGrpIdx++;
        addBlockCountOfGrp = 0;
    }

    HeteroBlockGroupType ToGroupType(HeteroBlockGroupType curType, bool isHostHmo)
    {
        switch (curType) {
            case HeteroBlockGroupType::BGT_PURE_EMPTY:
                if (isHostHmo) {
                    return HeteroBlockGroupType::BGT_PURE_HOST;
                }
                return HeteroBlockGroupType::BGT_PURE_DEVICE;
            case HeteroBlockGroupType::BGT_PURE_DEVICE:
                if (isHostHmo) {
                    return HeteroBlockGroupType::BGT_MIX_HOST_DEVICE;
                }
                return HeteroBlockGroupType::BGT_PURE_DEVICE;
            case HeteroBlockGroupType::BGT_PURE_HOST:
                if (!isHostHmo) {
                    return HeteroBlockGroupType::BGT_MIX_HOST_DEVICE;
                }
                return HeteroBlockGroupType::BGT_PURE_HOST;
            case HeteroBlockGroupType::BGT_MIX_HOST_DEVICE:
                return HeteroBlockGroupType::BGT_MIX_HOST_DEVICE;
            default:
                return HeteroBlockGroupType::BGT_PURE_EMPTY;
        }
    }

    int64_t ntotal;
    int pageSize;
    int blockSize;
    int maxDeviceDataNum;
    const size_t maxDeviceBlockNumPerGrp;
    const size_t maxHostBlockNumPerGrp;
    size_t addGrpIdx;
    size_t addBlockCountOfGrp;
    int groupSize;
    std::vector<HeteroBlockGroup> groups;
};

std::unique_ptr<HeteroBlockGroupMgr> HeteroBlockGroupMgr::Create(int64_t ntotal, int pageSize, int blockSize,
    DeviceStorgeInfo *deviceStorgeInfo, const std::vector<std::unique_ptr<DeviceVector<int8_t>>> &baseShaped)
{
    return std::unique_ptr<HeteroBlockGroupMgr>(new HeteroBlockGroupMgrImpl(ntotal, pageSize, blockSize,
        deviceStorgeInfo, baseShaped));
}
}
