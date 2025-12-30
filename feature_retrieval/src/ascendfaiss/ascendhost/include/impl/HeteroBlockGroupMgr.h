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

#ifndef ASCEND_INDEX_HETERO_BLOCK_GROUP_INCLUDED
#define ASCEND_INDEX_HETERO_BLOCK_GROUP_INCLUDED

#include <vector>
#include <memory>
#include "ascenddaemon/utils/DeviceVector.h"
#include "ascenddaemon/impl/Index.h"


namespace ascend {
struct DeviceStorgeInfo {
    DeviceStorgeInfo(size_t deviceDataNum, size_t deviceBlockNumPerGrp, size_t hostBlockNumPerGrp);

    size_t maxDeviceDataNum;
    size_t maxDeviceBlockNumPerGrp;
    size_t maxHostBlockNumPerGrp;
};

struct HeteroBlockInfo {
    HeteroBlockInfo(int pageId, int blockId, uint32_t dataNum);

    int pageId;  // 页Id
    int blockId; // 在整个特征底库中的第几个block
    uint32_t dataNum;
};

enum class HeteroBlockGroupType {
    BGT_PURE_EMPTY,
    BGT_PURE_DEVICE,
    BGT_PURE_HOST,
    BGT_MIX_HOST_DEVICE,
};

struct HeteroBlockGroup {
    HeteroBlockGroup() : groupType(HeteroBlockGroupType::BGT_PURE_EMPTY) {}
    HeteroBlockGroupType groupType;
    std::vector<HeteroBlockInfo> blocks;
};

const size_t DFT_MAX_HOST_BLOCK_COUNT = 4;    // 每组最多x个Host Block
const size_t DFT_MAX_DEVICE_BLOCK_COUNT = 16; // 每组最多x个Device Block

class HeteroBlockGroupMgr {
public:
    virtual ~HeteroBlockGroupMgr() noexcept;

    virtual size_t Count() const = 0;

    virtual const HeteroBlockGroup &At(size_t pos) const = 0;

    virtual ascend::idx_t OrderId2IndexId(ascend::idx_t searchOrderId) const = 0;

    virtual void OrderIds2IndexIds(int n, int topN, ascend::idx_t *orderIds) const = 0;

    virtual size_t GetGroupSize() const = 0;

    static std::unique_ptr<HeteroBlockGroupMgr> Create(int64_t ntotal, int pageSize, int blockSize,
        DeviceStorgeInfo *deviceStorgeInfo, const std::vector<std::unique_ptr<DeviceVector<int8_t>>> &baseShaped);
};
}

#endif
