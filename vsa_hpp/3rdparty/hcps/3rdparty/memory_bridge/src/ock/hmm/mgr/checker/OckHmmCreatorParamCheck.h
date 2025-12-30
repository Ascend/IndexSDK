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


#ifndef OCK_MEMORY_BRIDGE_HMM_HETERO_MGR_CREATOR_PARAM_CHECKER_H
#define OCK_MEMORY_BRIDGE_HMM_HETERO_MGR_CREATOR_PARAM_CHECKER_H
#include <string>
#include "ock/conf/OckSysConf.h"
#include "ock/log/OckLogger.h"
#include "ock/hmm/mgr/OckHmmHeteroMemoryMgr.h"

namespace ock {
namespace hmm {
const size_t RESERVE_MEMORY = 100ULL * 1024ULL * 1024ULL;
class OckHmmCreatorParamCheck {
public:
    template<typename T>
    static void CheckParamRange(OckHmmErrorCode &retCode, const ock::conf::ParamRange<T> &range,
                                T value, std::string paramName, OckHmmErrorCode errorCode)
    {
        if (range.NotIn(value)) {
            OCK_HMM_LOG_ERROR(paramName << " is out of range, which range is [" <<
                              range.minValue << ", " << range.maxValue << "].");
            retCode = errorCode;
        }
    }

    static void CheckDeviceIdExists(OckHmmErrorCode &retCode, const OckHmmDeviceId deviceId);
    static void CheckCpuIdExists(OckHmmErrorCode &retCode, const cpu_set_t &cpuSet);
    static void CheckDeviceCapacity(OckHmmErrorCode &retCode, const OckHmmDeviceId deviceId,
        const OckHmmMemoryCapacitySpecification devSpec);
    static OckHmmErrorCode CheckParam(const OckHmmDeviceInfo &deviceInfo);
    static OckHmmErrorCode CheckParam(const OckHmmDeviceInfoVec &deviceInfoVec);
};

}  // namespace hmm
}  // namespace ock
#endif