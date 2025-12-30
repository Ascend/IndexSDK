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
#include "ock/utils/OstreamUtils.h"
#include "ock/hmm/mgr/OckHmmHeteroMemoryMgr.h"

namespace ock {
namespace hmm {

std::ostream &operator<<(std::ostream &os, const cpu_set_t &cpuSet)
{
    int cpuCount = static_cast<int>(sysconf(_SC_NPROCESSORS_CONF));
    int beginPos = -1;
    int endPos = -1;
    os << "[";
    bool hasDataInOs = false;
    for (int i = 0; i < cpuCount; ++i) {
        if (CPU_ISSET(i, &cpuSet)) {
            if (beginPos == -1) {
                beginPos = i;
            }
            endPos = i;
        } else if (beginPos != -1) {
            if (hasDataInOs) {
                os << ", ";
            }
            if (endPos == beginPos) {
                os << endPos;
                hasDataInOs = true;
            } else {
                os << beginPos << "-" << endPos;
                hasDataInOs = true;
            }
            beginPos = -1;
            endPos = -1;
        }
    }
    if (endPos != -1) {
        if (hasDataInOs) {
            os << ", ";
        }
        if (endPos == beginPos) {
            os << endPos;
        } else {
            os << beginPos << "-" << endPos;
        }
    }
    os << "]";

    return os;
}

std::ostream &operator<<(std::ostream &os, const OckHmmDeviceInfo &data)
{
    return os << "{'deviceId':" << data.deviceId << ",'cpuSet':" << data.cpuSet
              << ",'transferThreadNum':" << data.transferThreadNum << ",'memorySpec':" << data.memorySpec << "}";
}

std::ostream &operator<<(std::ostream &os, const OckHmmDeviceInfoVec &data)
{
    return utils::PrintContainer(os, data);
}

std::ostream &operator<<(std::ostream &os, const OckHmmPureDeviceInfo &data)
{
    return os << "{'deviceId':" << data.deviceId << ",'cpuSet':" << data.cpuSet << ",'memorySpec':" << data.memorySpec
              << "}";
}

std::ostream &operator<<(std::ostream &os, const OckHmmPureDeviceInfoVec &data)
{
    return utils::PrintContainer(os, data);
}
}  // namespace hmm
}  // namespace ock