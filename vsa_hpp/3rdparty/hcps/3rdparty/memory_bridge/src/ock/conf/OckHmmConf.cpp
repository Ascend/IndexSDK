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

#include "ock/conf/OckHmmConf.h"
namespace ock {
namespace conf {
OckHmmConf::OckHmmConf(void)
    : maxHMOCountPerDevice(100000U),
      defaultFragThreshold(1024UL * 1024UL * 32UL),
      maxWaitTimeMilliSecond(3600000U),
      subMemoryBaseSize(256U),
      subMemoryLargerBaseSize(1ULL * 1024ULL),
      subMemoryLargestBaseSize(1ULL * 1024ULL * 1024ULL),
      maxAllocHmoBytes(128ULL * 1024ULL * 1024ULL * 1024ULL),
      minAllocHmoBytes(0ULL),
      maxMallocBytes(128ULL * 1024ULL * 1024ULL * 1024ULL),
      minMallocBytes(0ULL),
      maxFragThreshold(4ULL * 1024ULL * 1024ULL * 1024ULL),
      minFragThreshold(2ULL * 1024ULL * 1024ULL),
      maxMaxGapMilliSeconds(1000U),
      minMaxGapMilliSeconds(1U)
{}

bool operator==(const OckHmmConf &lhs, const OckHmmConf &rhs)
{
    return lhs.maxHMOCountPerDevice == rhs.maxHMOCountPerDevice &&
           lhs.defaultFragThreshold == rhs.defaultFragThreshold &&
           lhs.maxWaitTimeMilliSecond == rhs.maxWaitTimeMilliSecond &&
           lhs.subMemoryBaseSize == rhs.subMemoryBaseSize &&
           lhs.subMemoryLargerBaseSize == rhs.subMemoryLargerBaseSize &&
           lhs.subMemoryLargestBaseSize == rhs.subMemoryLargestBaseSize &&
           lhs.maxAllocHmoBytes == rhs.maxAllocHmoBytes &&
           lhs.minAllocHmoBytes == rhs.minAllocHmoBytes &&
           lhs.maxMallocBytes == rhs.maxMallocBytes &&
           lhs.minMallocBytes == rhs.minMallocBytes &&
           lhs.maxFragThreshold == rhs.maxFragThreshold &&
           lhs.maxMaxGapMilliSeconds == rhs.maxMaxGapMilliSeconds &&
           lhs.minMaxGapMilliSeconds == rhs.minMaxGapMilliSeconds;
}
bool operator!=(const OckHmmConf &lhs, const OckHmmConf &rhs)
{
    return !(lhs == rhs);
}
std::ostream &operator<<(std::ostream &os, const OckHmmConf &hmmConf)
{
    return os << "'maxHMOCountPerDevice'=" << hmmConf.maxHMOCountPerDevice
              << ",'defaultFragThreshold'=" << hmmConf.defaultFragThreshold
              << ",'maxWaitTimeMilliSecond'=" << hmmConf.maxWaitTimeMilliSecond
              << ",'subMemoryBaseSize'=" << hmmConf.subMemoryBaseSize
              << ",'subMemoryLargerBaseSize'=" << hmmConf.subMemoryLargerBaseSize
              << ",'subMemoryLargestBaseSize'=" << hmmConf.subMemoryLargestBaseSize
              << ",'maxAllocHmoBytes'=" << hmmConf.maxAllocHmoBytes
              << ",'minAllocHmoBytes'=" << hmmConf.minAllocHmoBytes
              << ",'maxMallocBytes'=" << hmmConf.maxMallocBytes
              << ",'minMallocBytes'=" << hmmConf.minMallocBytes
              << ",'maxFragThreshold'=" << hmmConf.maxFragThreshold
              << ",'minFragThreshold'=" << hmmConf.minFragThreshold
              << ",'maxMaxGapMilliSeconds'=" << hmmConf.maxMaxGapMilliSeconds
              << ",'minMaxGapMilliSeconds'=" << hmmConf.minMaxGapMilliSeconds;
}
}  // namespace conf
}  // namespace ock