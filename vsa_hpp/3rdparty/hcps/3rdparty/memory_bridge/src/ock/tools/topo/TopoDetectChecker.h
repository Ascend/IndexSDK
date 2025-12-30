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

#ifndef OCK_MEMORY_BRIDGE_OCK_TOOL_TOPO_DECTECT_CHECKER_H
#define OCK_MEMORY_BRIDGE_OCK_TOOL_TOPO_DECTECT_CHECKER_H
#include <vector>
#include <memory>
#include "ock/tools/topo/TopoDetectParam.h"
#include "ock/tools/topo/TopoDetectResult.h"

namespace ock {
namespace tools {
namespace topo {

class TopoDetectChecker {
public:
    virtual ~TopoDetectChecker() noexcept = default;

    virtual std::vector<TopoDetectResult> Check(void) = 0;

    static std::shared_ptr<TopoDetectChecker> Create(std::shared_ptr<TopoDetectParam> param);
};

}  // namespace topo
}  // namespace tools
}  // namespace ock
#endif