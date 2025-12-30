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

#ifndef OCK_MEMORY_BRIDGE_OCK_TOOL_TOPO_DECTECT_PARAM_H
#define OCK_MEMORY_BRIDGE_OCK_TOOL_TOPO_DECTECT_PARAM_H
#include <ostream>
#include <cstdint>
#include <unordered_set>
#include <vector>
#include "ock/conf/OckSysConf.h"
#include "ock/hmm/mgr/OckHmmHMObject.h"
#include "ock/acladapter/data/OckMemoryCopyKind.h"
#include "ock/tools/topo/DeviceCpuSet.h"
#include "ock/tools/topo/DetectModel.h"

namespace ock {
namespace tools {
namespace topo {

class TopoDetectParam {
public:
    explicit TopoDetectParam(void);

    DetectModel GetModel(void) const;
    uint32_t ThreadNumPerDevice(void) const;
    uint32_t TestTime(void) const;
    uint64_t PackageBytesPerTransfer(void) const;
    uint64_t PackageMBytesPerTransfer(void) const;
    acladapter::OckMemoryCopyKind Kind(void) const;
    void ThreadNumPerDevice(uint32_t value);
    void TestTime(uint32_t value);
    void PackageBytesPerTransfer(uint64_t value);
    void PackageMBytesPerTransfer(uint64_t value);
    void SetModel(DetectModel model);
    bool SetModel(const std::string &model);
    bool SetThreadNumPerDevice(const std::string &model);
    bool SetTestTimePerDevice(const std::string &model);
    bool SetPacketSize(const std::string &model);
    uint32_t TransferTypeBitValue(void) const;
    void Kind(acladapter::OckMemoryCopyKind value);
    void SetTransferType(uint32_t kind);
    const std::unordered_set<DeviceCpuSet, DeviceCpuSetHash> &GetDeviceInfo(void) const;
    bool AppendDeviceInfo(const DeviceCpuSet &deviceInfo);
    bool AppendDeviceInfo(const std::string &deviceInfo);
    static std::shared_ptr<TopoDetectParam> ParseArgs(int argc, char **argv);
    static std::string ArgToString(int argc, char **argv);

private:
    static void ParseDeviceOption(bool &parseSuccess, TopoDetectParam &ret, bool &noAnyGoodParam);
    static void ParseModelOption(bool &parseSuccess, TopoDetectParam &ret, bool &noAnyGoodParam);
    static void ParseTransferTypeOption(bool &parseSuccess, TopoDetectParam &ret, bool &noAnyGoodParam);
    static void ParseThreadNumOption(bool &parseSuccess, TopoDetectParam &param, bool &noAnyGoodParam);
    static void ParseTestTimeOption(bool &parseSuccess, TopoDetectParam &param, bool &noAnyGoodParam);
    static void ParsePacketSizeOption(bool &parseSuccess, TopoDetectParam &param, bool &noAnyGoodParam);
    static void ParseOption(
        int opt, bool &parseSuccess, TopoDetectParam &ret, bool &noAnyGoodParam, const std::string &programName);
    static void InitGetOptGlobalVar(int startParsePos);
    DetectModel detectModel;
    uint32_t threadNumPerDevice;
    uint32_t testTimeSeconds;
    uint64_t packageBytesPerTransfer;
    acladapter::OckMemoryCopyKind kind;
    uint32_t transferTypeBitValue;
    std::unordered_set<DeviceCpuSet, DeviceCpuSetHash> deviceInfoSet{};
};
}  // namespace topo
}  // namespace tools
}  // namespace ock
#endif