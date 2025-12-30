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


#ifndef HMM_INTF_H
#define HMM_INTF_H

#include <memory>
#include "AscendHMO.h"
#include "ErrorCode.h"

namespace ascend {

struct HmmMemoryInfo {
    HmmMemoryInfo(uint32_t deviceId, size_t deviceCapacity, size_t deviceBuffer, size_t hostCapacity, size_t hostBuffer)
        : deviceId(deviceId), deviceCapacity(deviceCapacity), deviceBuffer(deviceBuffer), hostCapacity(hostCapacity),
          hostBuffer(hostBuffer) {}
    uint32_t deviceId { 0 };
    size_t deviceCapacity { 0 };
    size_t deviceBuffer { 0 };
    size_t hostCapacity { 0 };
    size_t hostBuffer { 0 };
};

class HmmIntf {
public:
    virtual ~HmmIntf() = default;

    virtual APP_ERROR Init(const HmmMemoryInfo &memoryInfo) = 0;

    virtual std::pair<APP_ERROR, std::shared_ptr<AscendHMO>> CreateHmo(size_t size) = 0;

    static std::shared_ptr<HmmIntf> CreateHmm();
};

}

#endif // HMM_INTF_H