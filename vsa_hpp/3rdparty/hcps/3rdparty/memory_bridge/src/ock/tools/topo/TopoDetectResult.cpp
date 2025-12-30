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

#include <sstream>
#include <cstdint>
#include <iomanip>
#include "ock/utils/StrUtils.h"
#include "ock/tools/topo/TopoDetectResult.h"

namespace ock {
namespace tools {
namespace topo {
namespace {
template <typename DeviceT, typename CpuSetT, typename TransferT, typename TransTypeT, typename SpeedT>
std::string FormatResult(const DeviceT &dev, const CpuSetT &cpuSet, const TransferT &transfer, const SpeedT &speed,
    const TransTypeT &transType, const std::string &status)
{
    const int32_t devWidth = 10;
    const int32_t cpuWidth = 20;
    const int32_t transferWidth = 15;
    const int32_t speedWidth = 15;
    const int32_t statusWidth = 20;
    const int32_t transTypeWidth = 17;
    std::ostringstream os;
    os.precision(std::streamsize{ 2 });
    os << std::left;
    os << "|" << std::setw(devWidth) << dev << "|" << std::setw(cpuWidth) << cpuSet << "|" <<
        std::setw(transferWidth) << transfer << "|" << std::setw(speedWidth) << speed << "|" <<
        std::setw(transTypeWidth) << transType << "|" << std::setw(statusWidth) << status << "|";
    return os.str();
}
} // namespace

std::string TopoDetectResultHeadStr(void)
{
    return FormatResult("Device", "CpuIdSet", "Transfered(GB)", "Speed(GB/s)", "Transfer Type", "Result Message");
}
TopoDetectResult::TopoDetectResult(void)
    : transferBytes(0UL),
      usedMicroseconds(0UL),
      errorCode(hmm::HMM_SUCCESS),
      copyKind(acladapter::OckMemoryCopyKind::HOST_TO_DEVICE)
{}
std::ostream &operator << (std::ostream &os, const TopoDetectResult &result)
{
    const uint64_t gb2Bytes = 1024UL * 1024UL * 1024UL;
    const uint64_t second2MicroSecond = 1000000UL;
    const uint64_t cpuLineWidth = 20;
    std::string strCpuSet = utils::ToString(result.deviceInfo.cpuIds);
    std::string strCpuSetInLine = strCpuSet.substr(0, cpuLineWidth);
    if (result.errorCode != hmm::HMM_SUCCESS || result.usedMicroseconds == 0) {
        std::ostringstream osStatus;
        osStatus << "Error(" << utils::ToString(result.errorCode) << ")";
        os << FormatResult(result.deviceInfo.deviceId, strCpuSetInLine, "-", "-", result.copyKind, osStatus.str());
    } else {
        os << FormatResult(result.deviceInfo.deviceId, strCpuSetInLine,
            (double(result.transferBytes) / double(gb2Bytes)),
            (double(second2MicroSecond) * double(result.transferBytes) / double(gb2Bytes) /
            double(result.usedMicroseconds)),
            result.copyKind, "Normal");
    }
    os << std::endl;
    while (strCpuSet.size() > cpuLineWidth) {
        strCpuSet = strCpuSet.substr(cpuLineWidth);
        strCpuSetInLine = strCpuSet.substr(0, cpuLineWidth);
        os << FormatResult("", strCpuSetInLine, "", "", "", "");
        os << std::endl;
    }
    return os;
}

std::ostream &operator << (std::ostream &os, const std::vector<TopoDetectResult> &resultVec)
{
    os << "|----------+--------------------+---------------+---------------+-----------------+--------------------|" <<
        std::endl;
    os << TopoDetectResultHeadStr() << std::endl;
    os << "|----------+--------------------+---------------+---------------+-----------------+--------------------|" <<
        std::endl;
    for (auto &result : resultVec) {
        os << result;
    }
    os << "|----------+--------------------+---------------+---------------+-----------------+--------------------|" <<
        std::endl;
    return os;
}
} // namespace topo
} // namespace tools
} // namespace ock