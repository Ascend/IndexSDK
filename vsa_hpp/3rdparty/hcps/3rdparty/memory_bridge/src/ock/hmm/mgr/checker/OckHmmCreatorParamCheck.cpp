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


#include <unistd.h>
#include "ock/acladapter/param/OckGetDeviceInfo.h"
#include "ock/hmm/mgr/checker/OckHmmCreatorParamCheck.h"

namespace ock {
namespace hmm {

void OckHmmCreatorParamCheck::CheckDeviceIdExists(OckHmmErrorCode &retCode, const OckHmmDeviceId deviceId)
{
    uint32_t deviceCount = 0;
    auto ret = acladapter::OckGetDeviceInfo::GetDeviceCount(&deviceCount);
    if (ret == ACL_SUCCESS && deviceId >= deviceCount) {
        OCK_HMM_LOG_ERROR("the deviceId(" << deviceId << ") not exists!");
        retCode = HMM_ERROR_INPUT_PARAM_DEVICE_NOT_EXISTS;
    }
}

void OckHmmCreatorParamCheck::CheckCpuIdExists(OckHmmErrorCode &retCode, const cpu_set_t &cpuSet)
{
    uint32_t cpuCount = static_cast<uint32_t>(sysconf(_SC_NPROCESSORS_CONF));
    for (uint32_t i = cpuCount; i <  __CPU_SETSIZE; ++i) {
        if (CPU_ISSET(i, &cpuSet) != 0) {
            OCK_HMM_LOG_ERROR("there's a cpuId(" << i << ") in cpuSet that does not exist in Physical environment!");
            retCode = HMM_ERROR_INPUT_PARAM_CPUID_NOT_EXISTS;
        }
    }
}

void OckHmmCreatorParamCheck::CheckDeviceCapacity(OckHmmErrorCode &retCode, const OckHmmDeviceId deviceId,
    const OckHmmMemoryCapacitySpecification devSpec)
{
    acladapter::OckGetDeviceInfo::Init(deviceId);

    size_t freeMem = 0ULL;
    size_t totalMem = 0ULL;
    auto ret = acladapter::OckGetDeviceInfo::GetMemInfo(aclrtMemAttr::ACL_DDR_MEM, &freeMem, &totalMem);
    if (ret != 0) {
        OCK_HMM_LOG_INFO("GetMemInfo failed. deviceId=" << deviceId << ", ret=" << ret);
        return;
    }
    OCK_HMM_LOG_INFO("Device " << deviceId << " free memory: " << freeMem << " Bytes, total memory " << totalMem <<
        " Bytes" << "reserve memory is " << RESERVE_MEMORY);
    if (devSpec.maxDataCapacity + devSpec.maxSwapCapacity + RESERVE_MEMORY >= freeMem) {
        OCK_HMM_LOG_ERROR("Device capacity is too big, devSpec.maxDataCapacity is " << devSpec.maxDataCapacity <<
            ", devSpec.maxSwapCapacity is " << devSpec.maxSwapCapacity << ", free memory is " << freeMem <<
            ", reserve memory is " << RESERVE_MEMORY);
        retCode = HMM_ERROR_INPUT_PARAM_OUT_OF_RANGE;
    }
    acladapter::OckGetDeviceInfo::FreeResources(deviceId);
}

OckHmmErrorCode OckHmmCreatorParamCheck::CheckParam(const OckHmmDeviceInfo &deviceInfo)
{
    auto retCode = HMM_SUCCESS;
    // 校验deviceId
    CheckParamRange(retCode, conf::OckSysConf::DeviceInfoConf().deviceId, deviceInfo.deviceId,
                    "deviceId", HMM_ERROR_INPUT_PARAM_OUT_OF_RANGE);
    CheckDeviceIdExists(retCode, deviceInfo.deviceId);

    // 校验device capacity信息
    if (retCode == HMM_SUCCESS) {
        CheckDeviceCapacity(retCode, deviceInfo.deviceId, deviceInfo.memorySpec.devSpec);
    }

    // 校验cpuSet
    CheckCpuIdExists(retCode, deviceInfo.cpuSet);

    // 校验transferThreadNum
    CheckParamRange(retCode, conf::OckSysConf::DeviceInfoConf().transferThreadNum, deviceInfo.transferThreadNum,
                    "transferThreadNum", HMM_ERROR_INPUT_PARAM_OUT_OF_RANGE);

    // 校验memorySpec
    CheckParamRange(retCode, conf::OckSysConf::DeviceInfoConf().deviceBaseCapacity,
                    deviceInfo.memorySpec.devSpec.maxDataCapacity, "devSpec.maxDataCapacity",
                    HMM_ERROR_INPUT_PARAM_OUT_OF_RANGE);
    CheckParamRange(retCode, conf::OckSysConf::DeviceInfoConf().deviceBufferCapacity,
                    deviceInfo.memorySpec.devSpec.maxSwapCapacity, "devSpec.maxSwapCapacity",
                    HMM_ERROR_INPUT_PARAM_OUT_OF_RANGE);
    CheckParamRange(retCode, conf::OckSysConf::DeviceInfoConf().hostBaseCapacity,
                    deviceInfo.memorySpec.hostSpec.maxDataCapacity, "hostSpec.maxDataCapacity",
                    HMM_ERROR_INPUT_PARAM_OUT_OF_RANGE);
    CheckParamRange(retCode, conf::OckSysConf::DeviceInfoConf().hostBufferCapacity,
                    deviceInfo.memorySpec.hostSpec.maxSwapCapacity, "hostSpec.maxSwapCapacity",
                    HMM_ERROR_INPUT_PARAM_OUT_OF_RANGE);

    return retCode;
}

OckHmmErrorCode OckHmmCreatorParamCheck::CheckParam(const OckHmmDeviceInfoVec &deviceInfoVec)
{
    if (deviceInfoVec.empty()) {
        OCK_HMM_LOG_ERROR("There is no element in deviceInfoVec!");
        return HMM_ERROR_INPUT_PARAM_EMPTY;
    }

    for (auto deviceInfo : deviceInfoVec) {
        auto retCode = CheckParam(deviceInfo);
        if (retCode != HMM_SUCCESS) {
            return retCode;
        }
    }

    return HMM_SUCCESS;
}

}  // namespace hmm
}  // namespace ock