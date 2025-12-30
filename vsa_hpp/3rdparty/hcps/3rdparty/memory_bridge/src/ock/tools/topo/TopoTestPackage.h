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

#include <vector>
#include "ock/tools/topo/TopoDetectParam.h"
#include "ock/acladapter/executor/OckAsyncTaskExecuteService.h"

namespace ock {
namespace tools {
namespace topo {

class TopoTestPackage {
public:
    struct Package {
        explicit Package(uint32_t pkgSize, uint8_t *devAddress, uint8_t *hostAddress, uint8_t *devDstAddress,
            uint8_t *hostDstAddress);
        uint32_t packageSize;
        uint8_t *deviceAddr;
        uint8_t *hostAddr;
        uint8_t *deviceDstAddr;
        uint8_t *hostDstAddr;
    };
    virtual ~TopoTestPackage() noexcept = default;

    virtual Package GetPackage(uint32_t taskId) = 0;
    virtual hmm::OckHmmErrorCode GetError(void) const = 0;

    static std::shared_ptr<TopoTestPackage> Create(
        const TopoDetectParam &param, acladapter::OckAsyncTaskExecuteService &service);
};
}  // namespace topo
}  // namespace tools
}  // namespace ock