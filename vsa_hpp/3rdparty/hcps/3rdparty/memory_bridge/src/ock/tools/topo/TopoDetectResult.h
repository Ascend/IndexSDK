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

#ifndef OCK_MEMORY_BRIDGE_OCK_TOOL_TOPO_DECTECT_RESULT_H
#define OCK_MEMORY_BRIDGE_OCK_TOOL_TOPO_DECTECT_RESULT_H
#include <ostream>
#include <cstdint>
#include <string>
#include <vector>
#include "ock/acladapter/data/OckMemoryCopyKind.h"
#include "ock/tools/topo/TopoDetectParam.h"

namespace ock {
namespace tools {
namespace topo {

struct TopoDetectResult {
    explicit TopoDetectResult(void);
    uint64_t transferBytes;
    int64_t usedMicroseconds;
    hmm::OckHmmErrorCode errorCode;
    DeviceCpuSet deviceInfo{};
    acladapter::OckMemoryCopyKind copyKind;
};

std::string TopoDetectResultHeadStr(void);
std::ostream &operator<<(std::ostream &os, const TopoDetectResult &result);
std::ostream &operator<<(std::ostream &os, const std::vector<TopoDetectResult> &resultVec);

}  // namespace topo
}  // namespace tools
}  // namespace ock
#endif