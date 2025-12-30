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

#ifndef OCK_MEMORY_BRIDGE_OCK_TOOL_TOPO_DECTECT_PARAM_VERIFY_H
#define OCK_MEMORY_BRIDGE_OCK_TOOL_TOPO_DECTECT_PARAM_VERIFY_H
#include <string>
#include <memory>
#include "ock/tools/topo/TopoDetectParam.h"

namespace ock {
namespace tools {
namespace topo {

struct ParamVerifyResult {
    ParamVerifyResult(hmm::OckHmmErrorCode ret, const std::string &message = "");
    void MergeOther(const ParamVerifyResult &other);
    hmm::OckHmmErrorCode retCode;
    std::string msg;
};
class TopoDetectParamVerify {
public:
    static std::shared_ptr<ParamVerifyResult> CheckAll(const TopoDetectParam &param);

private:
    static ParamVerifyResult CheckMode(const TopoDetectParam &param);
    static ParamVerifyResult CheckThreadNumPerDevice(const TopoDetectParam &param);
    static ParamVerifyResult CheckPackageNum(const TopoDetectParam &param);
    static ParamVerifyResult CheckPackageBytesPerTransfer(const TopoDetectParam &param);
    static ParamVerifyResult CheckDeviceCpuSet(const TopoDetectParam &param);
    static ParamVerifyResult CheckTransferType(const TopoDetectParam &param);
};
}  // namespace topo
}  // namespace tools
}  // namespace ock
#endif