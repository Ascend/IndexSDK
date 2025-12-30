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

#include "ock/hmm/mgr/OckHmmMemorySpecification.h"
#include "ock/utils/ReadalbeUtils.h"

namespace ock {
namespace hmm {
OckHmmMemoryUsedInfo::OckHmmMemoryUsedInfo(void)
    :usedBytes(0), unusedFragBytes(0), leftBytes(0), swapUsedBytes(0), swapLeftBytes(0)
{}

OckHmmMemoryUsedInfo &OckHmmMemoryUsedInfo::operator+=(const OckHmmMemoryUsedInfo &other)
{
    this->usedBytes += other.usedBytes;
    this->unusedFragBytes += other.unusedFragBytes;
    this->leftBytes += other.leftBytes;
    this->swapUsedBytes += other.swapUsedBytes;
    this->swapLeftBytes += other.swapLeftBytes;
    return *this;
}
bool operator==(const OckHmmMemoryCapacitySpecification &lhs, const OckHmmMemoryCapacitySpecification &rhs)
{
    return lhs.maxDataCapacity == rhs.maxDataCapacity && lhs.maxSwapCapacity == rhs.maxSwapCapacity;
}
bool operator==(const OckHmmMemorySpecification &lhs, const OckHmmMemorySpecification &rhs)
{
    return lhs.devSpec == rhs.devSpec && lhs.hostSpec == rhs.hostSpec;
}
bool operator==(const OckHmmMemoryUsedInfo &lhs, const OckHmmMemoryUsedInfo &rhs)
{
    return lhs.usedBytes == rhs.usedBytes && lhs.unusedFragBytes == rhs.unusedFragBytes &&
           lhs.leftBytes == rhs.leftBytes && lhs.swapUsedBytes == rhs.swapUsedBytes &&
           lhs.swapLeftBytes == rhs.swapLeftBytes;
}
bool operator==(const OckHmmResourceUsedInfo &lhs, const OckHmmResourceUsedInfo &rhs)
{
    return lhs.devUsedInfo == rhs.devUsedInfo && lhs.hostUsedInfo == rhs.hostUsedInfo;
}

bool operator!=(const OckHmmMemoryCapacitySpecification &lhs, const OckHmmMemoryCapacitySpecification &rhs)
{
    return !(lhs == rhs);
}
bool operator!=(const OckHmmMemorySpecification &lhs, const OckHmmMemorySpecification &rhs)
{
    return !(lhs == rhs);
}
bool operator!=(const OckHmmMemoryUsedInfo &lhs, const OckHmmMemoryUsedInfo &rhs)
{
    return !(lhs == rhs);
}
bool operator!=(const OckHmmResourceUsedInfo &lhs, const OckHmmResourceUsedInfo &rhs)
{
    return !(lhs == rhs);
}
std::ostream &operator<<(std::ostream &os, const OckHmmMemoryCapacitySpecification &memorySpec)
{
    return os << "{'maxDataCapacity':" << memorySpec.maxDataCapacity
              << ",'maxSwapCapacity':" << memorySpec.maxSwapCapacity << "}";
}
std::ostream &operator<<(std::ostream &os, const OckHmmMemoryUsedInfo &usedInfo)
{
    os << "{'usedBytes':";
    utils::ByteToReadable(os, usedInfo.usedBytes);
    os << ", 'unusedFragBytes':";
    utils::ByteToReadable(os, usedInfo.unusedFragBytes);
    os << ", 'leftBytes':";
    utils::ByteToReadable(os, usedInfo.leftBytes);
    os << ", 'swapUsedBytes':";
    utils::ByteToReadable(os, usedInfo.swapUsedBytes);
    os << ", 'swapLeftBytes':";
    utils::ByteToReadable(os, usedInfo.swapLeftBytes);
    os << "}";
    return os;
}
std::ostream &operator<<(std::ostream &os, const OckHmmMemorySpecification &memorySpec)
{
    return os << "{'devSpec':" << memorySpec.devSpec << ",'hostSpec':" << memorySpec.hostSpec << "}";
}
std::ostream &operator<<(std::ostream &os, const OckHmmResourceUsedInfo &usedInfo)
{
    return os << "{'devUsedInfo':" << usedInfo.devUsedInfo << ",'hostUsedInfo':" << usedInfo.hostUsedInfo << "}";
}
}  // namespace hmm
}  // namespace ock
