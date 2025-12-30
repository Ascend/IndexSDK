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

#include <getopt.h>
#include <memory>
#include <sstream>
#include <string>
#include <iostream>
#include <iomanip>
#include <thread>
#include "ock/utils/StrUtils.h"
#include "ock/utils/OckSafeUtils.h"
#include "ock/tools/topo/TopoDetectParam.h"

namespace ock {
namespace tools {
namespace topo {

constexpr uint64_t MB_TO_BYTES = 1024U * 1024U;                                                         // 1M字节

TopoDetectParam::TopoDetectParam(void)
    : detectModel(DetectModel::UNKNOWN), threadNumPerDevice(conf::OckSysConf::ToolConf().defaultThreadNumPerDevice),
      testTimeSeconds(conf::OckSysConf::ToolConf().defaultTestTime),
      packageBytesPerTransfer(conf::OckSysConf::ToolConf().defaultPackageMBPerTransfer * MB_TO_BYTES),
      kind(acladapter::OckMemoryCopyKind::HOST_TO_DEVICE), transferTypeBitValue((uint32_t)kind)
{}

uint32_t TopoDetectParam::ThreadNumPerDevice(void) const
{
    return threadNumPerDevice;
}
uint32_t TopoDetectParam::TestTime(void) const
{
    return testTimeSeconds;
}
uint64_t TopoDetectParam::PackageBytesPerTransfer(void) const
{
    return packageBytesPerTransfer;
}
uint64_t TopoDetectParam::PackageMBytesPerTransfer(void) const
{
    return utils::SafeDivDown(packageBytesPerTransfer, MB_TO_BYTES);
}
void TopoDetectParam::ThreadNumPerDevice(uint32_t value)
{
    threadNumPerDevice = value;
}
void TopoDetectParam::TestTime(uint32_t value)
{
    testTimeSeconds = value;
}
void TopoDetectParam::PackageBytesPerTransfer(uint64_t value)
{
    packageBytesPerTransfer = value;
}
void TopoDetectParam::PackageMBytesPerTransfer(uint64_t value)
{
    packageBytesPerTransfer = value * MB_TO_BYTES;
}
DetectModel TopoDetectParam::GetModel(void) const
{
    return detectModel;
}
acladapter::OckMemoryCopyKind TopoDetectParam::Kind(void) const
{
    return kind;
}
void TopoDetectParam::SetModel(DetectModel data)
{
    detectModel = data;
}
bool TopoDetectParam::SetModel(const std::string &buff)
{
    DetectModel data = DetectModel::UNKNOWN;
    if (!FromString(data, buff)) {
        return false;
    }
    SetModel(data);
    return true;
}
bool TopoDetectParam::SetThreadNumPerDevice(const std::string &threadNumFromBuff)
{
    uint32_t threadNum = conf::OckSysConf::ToolConf().defaultThreadNumPerDevice;
    if (!ock::utils::FromString<uint32_t>(threadNum, threadNumFromBuff)) {
        return false;
    }
    if (conf::OckSysConf::ToolConf().threadNumPerDevice.NotIn(threadNum)) {
        return false;
    }
    ThreadNumPerDevice(threadNum);
    return true;
}
bool TopoDetectParam::SetTestTimePerDevice(const std::string &testTimeFromBuff)
{
    uint32_t testTime = conf::OckSysConf::ToolConf().defaultTestTime;
    if (!ock::utils::FromString<uint32_t>(testTime, testTimeFromBuff)) {
        return false;
    }
    if (conf::OckSysConf::ToolConf().testTime.NotIn(testTime)) {
        return false;
    }
    TestTime(testTime);
    return true;
}
bool TopoDetectParam::SetPacketSize(const std::string &packetSizeFromBuff)
{
    uint32_t packetSize = static_cast<uint32_t>(conf::OckSysConf::ToolConf().defaultPackageMBPerTransfer);
    if (!ock::utils::FromString<uint32_t>(packetSize, packetSizeFromBuff)) {
        return false;
    }
    if (conf::OckSysConf::ToolConf().packageMBPerTransfer.NotIn(packetSize)) {
        return false;
    }
    PackageMBytesPerTransfer(packetSize);
    return true;
}
void TopoDetectParam::Kind(acladapter::OckMemoryCopyKind value)
{
    kind = value;
}
uint32_t TopoDetectParam::TransferTypeBitValue(void) const
{
    return transferTypeBitValue;
}
void TopoDetectParam::SetTransferType(uint32_t transKind)
{
    transferTypeBitValue = transKind;
}
const std::unordered_set<DeviceCpuSet, DeviceCpuSetHash> &TopoDetectParam::GetDeviceInfo(void) const
{
    return deviceInfoSet;
}
bool TopoDetectParam::AppendDeviceInfo(const DeviceCpuSet &deviceInfo)
{
    auto ret = deviceInfoSet.insert(deviceInfo);
    return ret.second;
}
bool TopoDetectParam::AppendDeviceInfo(const std::string &buff)
{
    auto ret = DeviceCpuSet::ParseFrom(buff);
    if (!ret.first) {
        return false;
    }
    AppendDeviceInfo(*ret.second);
    return true;
}
void ShowOptionHelpMsg(const std::string &optionStr, const std::string &helpMsg)
{
    const uint32_t optionWidth = 40U;
    std::cout << std::left << std::setw(optionWidth) << optionStr;
    if (optionStr.length() > optionWidth) {
        std::cout << std::endl;
        std::cout << std::left << std::setw(optionWidth) << " ";
    }
    std::cout << helpMsg << std::endl;
}
void ShowHelpMessage(const std::string &programName)
{
    std::cout << std::left << "Usage: " << programName << " [OPTIONS]" << std::endl << std::endl;
    std::cout << "Options:" << std::endl;
    ShowOptionHelpMsg("-m <PARALLEL|SERIAL>", "Detection mode.");
    ShowOptionHelpMsg("-d [DeviceId:<CpuIdRange>,...]", "Device ID and CPU IDs.");
    ShowOptionHelpMsg("-t <DEVICE_TO_HOST|HOST_TO_DEVICE|HOST_TO_HOST|DEVICE_TO_DEVICE>",
        "Data Transfer Type, default(HOST_TO_DEVICE)");
    std::ostringstream osThread;
    osThread << "Number of threads for data transmission per device." << "The valid range of values is ["
             << conf::OckSysConf::ToolConf().threadNumPerDevice.minValue << ","
             << conf::OckSysConf::ToolConf().threadNumPerDevice.maxValue
             << "] default(" << conf::OckSysConf::ToolConf().defaultThreadNumPerDevice << ")";
    ShowOptionHelpMsg("-p [ThreadNum]", osThread.str());
    std::ostringstream osDevice;
    osDevice << "Test time per device(unit: seconds). The valid range of values["
             << conf::OckSysConf::ToolConf().testTime.minValue << ","
             << conf::OckSysConf::ToolConf().testTime.maxValue
             << "] default(" << conf::OckSysConf::ToolConf().defaultTestTime << ")";
    ShowOptionHelpMsg("-n [Seconds]", osDevice.str());
    std::ostringstream osCount;
    osCount << "Size of each packet(unit: MB). The valid range of values["
            << conf::OckSysConf::ToolConf().packageMBPerTransfer.minValue << ","
            << conf::OckSysConf::ToolConf().packageMBPerTransfer.maxValue
            << "] default(" << conf::OckSysConf::ToolConf().defaultPackageMBPerTransfer << ")";
    ShowOptionHelpMsg("-s [PackageSize]", osCount.str());
    ShowOptionHelpMsg("-h", "Show this message.");
    std::cout << std::endl;
    std::cout << "Example:" << std::endl;
    std::cout << "1. " << programName << " -m PARALLEL -t HOST_TO_DEVICE -d 0:4-5,7 -d 1:16,17-19 -d 2:32,33";
    std::cout << std::endl;
    std::cout << "2. " << programName << " -m SERIAL -d 0:4-5,7 -d 1:16,17-19 -p 4 -n 50 -s 128" << std::endl;
    std::cout << std::endl;
}
void TopoDetectParam::ParseDeviceOption(bool &parseSuccess, TopoDetectParam &ret, bool &noAnyGoodParam)
{
    if (optarg == nullptr) {
        parseSuccess = false;
    } else {
        if (!ret.AppendDeviceInfo(optarg)) {
            std::cerr << "Invalid device info[" << optarg << "]. with parameter -d" << std::endl;
            parseSuccess = false;
        }
        noAnyGoodParam = false;
    }
}
void TopoDetectParam::ParseModelOption(bool &parseSuccess, TopoDetectParam &ret, bool &noAnyGoodParam)
{
    if (optarg == nullptr) {
        parseSuccess = false;
    } else {
        if (!ret.SetModel(optarg)) {
            std::cerr << "Invalid mode[" << optarg << "]. with parameter -m, valid value is <PARALLEL|SERIAL>"
                      << std::endl;
            parseSuccess = false;
        }
        noAnyGoodParam = false;
    }
}
void TopoDetectParam::ParseTransferTypeOption(bool &parseSuccess, TopoDetectParam &param, bool &noAnyGoodParam)
{
    if (optarg == nullptr) {
        parseSuccess = false;
    } else {
        auto ret = acladapter::ParseMemoryCopyKind(optarg);
        if (!ret.first) {
            std::cerr << "Invalid transfer type[" << optarg << "]." << std::endl;
            parseSuccess = false;
        } else {
            param.SetTransferType(ret.second);
        }
        noAnyGoodParam = false;
    }
}
void TopoDetectParam::ParseThreadNumOption(bool &parseSuccess, TopoDetectParam &ret, bool &noAnyGoodParam)
{
    if (optarg == nullptr) {
        parseSuccess = false;
    } else {
        if (!ret.SetThreadNumPerDevice(optarg)) {
            std::cerr << "Invalid thread number[" << optarg << "]. with parameter -p, valid range is["
                      << conf::OckSysConf::ToolConf().threadNumPerDevice.minValue << ","
                      << conf::OckSysConf::ToolConf().threadNumPerDevice.maxValue
                      << "] default(" << conf::OckSysConf::ToolConf().defaultThreadNumPerDevice << ")" << std::endl;
            parseSuccess = false;
        }
        noAnyGoodParam = false;
    }
}
void TopoDetectParam::ParseTestTimeOption(bool &parseSuccess, TopoDetectParam &ret, bool &noAnyGoodParam)
{
    if (optarg == nullptr) {
        parseSuccess = false;
    } else {
        if (!ret.SetTestTimePerDevice(optarg)) {
            std::cerr << "Invalid test time[" << optarg << "]. with parameter -n, valid range is["
                      << conf::OckSysConf::ToolConf().testTime.minValue << ","
                      << conf::OckSysConf::ToolConf().testTime.maxValue
                      << "] default(" << conf::OckSysConf::ToolConf().defaultTestTime << ")" << std::endl;
            parseSuccess = false;
        }
        noAnyGoodParam = false;
    }
}
void TopoDetectParam::ParsePacketSizeOption(bool &parseSuccess, TopoDetectParam &ret, bool &noAnyGoodParam)
{
    if (optarg == nullptr) {
        parseSuccess = false;
    } else {
        if (!ret.SetPacketSize(optarg)) {
            std::cerr << "Invalid packet size[" << optarg << "]. with parameter -s, valid range is["
                      << conf::OckSysConf::ToolConf().packageMBPerTransfer.minValue << ","
                      << conf::OckSysConf::ToolConf().packageMBPerTransfer.maxValue
                      << "] default(" << conf::OckSysConf::ToolConf().defaultPackageMBPerTransfer << ")" << std::endl;
            parseSuccess = false;
        }
        noAnyGoodParam = false;
    }
}
void TopoDetectParam::ParseOption(
    int opt, bool &parseSuccess, TopoDetectParam &ret, bool &noAnyGoodParam, const std::string &programName)
{
    switch (opt) {
        case 'd':
            ParseDeviceOption(parseSuccess, ret, noAnyGoodParam);
            break;
        case 'm':
            ParseModelOption(parseSuccess, ret, noAnyGoodParam);
            break;
        case 't':
            ParseTransferTypeOption(parseSuccess, ret, noAnyGoodParam);
            break;
        case 'p':
            ParseThreadNumOption(parseSuccess, ret, noAnyGoodParam);
            break;
        case 'n':
            ParseTestTimeOption(parseSuccess, ret, noAnyGoodParam);
            break;
        case 's':
            ParsePacketSizeOption(parseSuccess, ret, noAnyGoodParam);
            break;
        case 'h':
            ShowHelpMessage(programName);
            break;
        default:
            std::cerr << "Invalid option" << std::endl;
            ShowHelpMessage(programName);
            parseSuccess = false;
            break;
    }
}
void TopoDetectParam::InitGetOptGlobalVar(int startParsePos)
{
    optarg = nullptr;
    optind = startParsePos;
    opterr = 0;
    optopt = 0;
}
std::shared_ptr<TopoDetectParam> TopoDetectParam::ParseArgs(int argc, char **argv)
{
    if (argc <= 1) {
        return std::shared_ptr<TopoDetectParam>();
    }
    const int startParsePos = 1;
    InitGetOptGlobalVar(startParsePos);
    bool parseSuccess = true;
    auto ret = std::make_shared<TopoDetectParam>();
    int opt = 0;
    const char *opString = "hm:t:d:p:n:s:";
    bool noAnyGoodParam = true;
    while ((opt = getopt(argc, argv, opString)) != -1) {
        ParseOption(opt, parseSuccess, *ret, noAnyGoodParam, argv[0]);
    }
    if (parseSuccess && !noAnyGoodParam) {
        return ret;
    }
    return std::shared_ptr<TopoDetectParam>();
}

std::string TopoDetectParam::ArgToString(int argc, char **argv)
{
    std::ostringstream os;
    for (int i = 0; i < argc; ++i) {
        if (i != 0) {
            os << " ";
        }
        os << argv[i];
    }
    return os.str();
}
}  // namespace topo
}  // namespace tools
}  // namespace ock