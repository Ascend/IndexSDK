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

#include <ostream>
#include <map>
#include "ock/acladapter/data/OckTaskResourceType.h"
namespace ock {
namespace acladapter {
namespace {
const std::map<OckTaskResourceType, const char *> g_ResourceTypeNameMap{
    {OckTaskResourceType::MEMORY_TRANSFER, "MEMORY_TRANSFER"},
    {OckTaskResourceType::DEVICE_MEMORY_OP, "DEVICE_MEMORY_OP"},
    {OckTaskResourceType::HOST_MEMORY_OP, "HOST_MEMORY_OP"},
    {OckTaskResourceType::DEVICE_AI_CUBE, "DEVICE_AI_CUBE"},
    {OckTaskResourceType::DEVICE_AI_VECTOR, "DEVICE_AI_VECTOR"},
    {OckTaskResourceType::DEVICE_AI_CPU, "DEVICE_AI_CPU"},
    {OckTaskResourceType::HOST_CPU, "HOST_CPU"},
    {OckTaskResourceType::HOST_STREAM, "HOST_STREAM"},
    {OckTaskResourceType::DEVICE_STREAM, "DEVICE_STREAM"},
};
}
std::ostream &operator<<(std::ostream &os, OckTaskResourceType resourceType)
{
    if (resourceType == OckTaskResourceType::ALL) {
        return os << "ALL";
    }
    bool firstData = true;
    for (auto iter = g_ResourceTypeNameMap.begin(); iter != g_ResourceTypeNameMap.end(); ++iter) {
        if (RhsInLhs(resourceType, iter->first)) {
            if (!firstData) {
                os << "|";
            }
            os << iter->second;
            firstData = false;
        }
    }
    if (firstData) {
        os << "UnknownResourceType(" << static_cast<uint32_t>(resourceType) << ")";
    }
    return os;
}
std::ostream &operator<<(std::ostream &os, const OckTaskThreadNumberMap &data)
{
    os << "[";
    for (auto iter = data.begin(); iter != data.end(); ++iter) {
        if (iter != data.begin()) {
            os << ",";
        }
        os << "{'resourceType':" << iter->first << ", 'threadCount':" << iter->second << "}";
    }
    os << "]";
    return os;
}
uint32_t CalcRelatedThreadNumber(const OckTaskThreadNumberMap &taskNumberMap, OckTaskResourceType resourceType)
{
    uint32_t streamThreadCount = 0;
    for (auto &value : taskNumberMap) {
        if (RhsInLhs(value.first, resourceType)) {
            streamThreadCount += value.second;
        }
    }
    return streamThreadCount;
}
}  // namespace acladapter
}  // namespace ock