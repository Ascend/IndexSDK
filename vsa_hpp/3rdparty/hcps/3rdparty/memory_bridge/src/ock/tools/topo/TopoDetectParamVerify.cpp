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

#include <thread>
#include <limits>
#include <sstream>
#include "ock/tools/topo/TopoDetectParamVerify.h"

namespace ock {
namespace tools {
namespace topo {

ParamVerifyResult::ParamVerifyResult(hmm::OckHmmErrorCode ret, const std::string &message) : retCode(ret), msg(message)
{}
void ParamVerifyResult::MergeOther(const ParamVerifyResult &other)
{
    if (retCode == hmm::HMM_SUCCESS) {
        retCode = other.retCode;
    }
    if (msg.empty()) {
        msg = other.msg;
    } else {
        if (!other.msg.empty()) {
            std::ostringstream os;
            os << msg << std::endl << other.msg;
            msg = os.str();
        }
    }
}
std::shared_ptr<ParamVerifyResult> TopoDetectParamVerify::CheckAll(const TopoDetectParam &param)
{
    auto ret = std::make_shared<ParamVerifyResult>(hmm::HMM_SUCCESS);
    ret->MergeOther(CheckMode(param));
    ret->MergeOther(CheckThreadNumPerDevice(param));
    ret->MergeOther(CheckPackageNum(param));
    ret->MergeOther(CheckPackageBytesPerTransfer(param));
    ret->MergeOther(CheckDeviceCpuSet(param));
    ret->MergeOther(CheckTransferType(param));
    return ret;
}
ParamVerifyResult TopoDetectParamVerify::CheckMode(const TopoDetectParam &param)
{
    if (param.GetModel() != DetectModel::PARALLEL && param.GetModel() != DetectModel::SERIAL) {
        return ParamVerifyResult(
            hmm::HMM_ERROR_INPUT_PARAM_NOT_SUPPORT_SUCH_OP,
            "Lack input with parameter -m, only the 'PARALLEL' or 'SERIAL' mode is supported.");
    }
    return ParamVerifyResult(hmm::HMM_SUCCESS);
}
ParamVerifyResult TopoDetectParamVerify::CheckThreadNumPerDevice(const TopoDetectParam &param)
{
    if (conf::OckSysConf::ToolConf().threadNumPerDevice.NotIn(param.ThreadNumPerDevice())) {
        std::ostringstream os;
        os << "The number of threads for each device must be in ["
           << conf::OckSysConf::ToolConf().threadNumPerDevice.minValue << ","
           << conf::OckSysConf::ToolConf().threadNumPerDevice.maxValue << "]";
        return ParamVerifyResult(hmm::HMM_ERROR_INPUT_PARAM_NOT_SUPPORT_SUCH_OP, os.str());
    }
    return ParamVerifyResult(hmm::HMM_SUCCESS);
}
ParamVerifyResult TopoDetectParamVerify::CheckPackageNum(const TopoDetectParam &param)
{
    if (conf::OckSysConf::ToolConf().testTime.NotIn(param.TestTime())) {
        std::ostringstream os;
        os << "The running time for each device must be in ["
           << conf::OckSysConf::ToolConf().testTime.minValue << ","
           << conf::OckSysConf::ToolConf().testTime.maxValue << "]";
        return ParamVerifyResult(hmm::HMM_ERROR_INPUT_PARAM_NOT_SUPPORT_SUCH_OP, os.str());
    }
    return ParamVerifyResult(hmm::HMM_SUCCESS);
}
ParamVerifyResult TopoDetectParamVerify::CheckPackageBytesPerTransfer(const TopoDetectParam &param)
{
    if (conf::OckSysConf::ToolConf().packagePerTransfer.NotIn(param.PackageBytesPerTransfer())) {
        std::ostringstream os;
        os << "The size of each data packet output by each device must be in ["
           << conf::OckSysConf::ToolConf().packagePerTransfer.minValue << ","
           << conf::OckSysConf::ToolConf().packagePerTransfer.maxValue << "](Bytes)";
        return ParamVerifyResult(hmm::HMM_ERROR_INPUT_PARAM_NOT_SUPPORT_SUCH_OP, os.str());
    }
    return ParamVerifyResult(hmm::HMM_SUCCESS);
}
ParamVerifyResult TopoDetectParamVerify::CheckDeviceCpuSet(const TopoDetectParam &param)
{
    uint32_t concurrency = std::thread::hardware_concurrency();
    if (param.GetDeviceInfo().empty()) {
        return ParamVerifyResult(
            hmm::HMM_ERROR_INPUT_PARAM_NOT_SUPPORT_SUCH_OP,
            "Lack input with parameter -d. Please type in deviceId and cpuId.");
    }
    for (auto &devInfo : param.GetDeviceInfo()) {
        for (auto &range : devInfo.cpuIds) {
            if (range.startId > range.endId) {
                std::ostringstream os;
                os << "The start value(" << range.startId << ") is greater than the end value(" << range.endId << ").";
                return ParamVerifyResult(hmm::HMM_ERROR_INPUT_PARAM_NOT_SUPPORT_SUCH_OP, os.str());
            }
            if (range.startId > std::numeric_limits<uint32_t>::max() || range.startId >= concurrency) {
                std::ostringstream os;
                os << "The CPU core id(" << range.startId << ") is too large.";
                return ParamVerifyResult(hmm::HMM_ERROR_INPUT_PARAM_NOT_SUPPORT_SUCH_OP, os.str());
            }
            if (range.endId > std::numeric_limits<uint32_t>::max() || range.endId >= concurrency) {
                std::ostringstream os;
                os << "The CPU core id(" << range.endId << ") is too large.";
                return ParamVerifyResult(hmm::HMM_ERROR_INPUT_PARAM_NOT_SUPPORT_SUCH_OP, os.str());
            }
        }
    }
    return ParamVerifyResult(hmm::HMM_SUCCESS);
}
ParamVerifyResult TopoDetectParamVerify::CheckTransferType(const TopoDetectParam &param)
{
    return ParamVerifyResult(hmm::HMM_SUCCESS);
}
}  // namespace topo
}  // namespace tools
}  // namespace ock