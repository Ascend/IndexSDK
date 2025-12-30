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

#include "ock/conf/OckTopoDetectConf.h"
namespace ock {
namespace conf {

constexpr uint32_t MIN_THREAD_NUM_PER_DEVICE = 1;                                                       // 线程数
constexpr uint32_t MAX_THREAD_NUM_PER_DEVICE = 5;                                                       // 线程数
constexpr uint64_t MIN_PACKAGE_MB_PER_TRANSFER = 2U;                                                    // 2M
constexpr uint64_t MAX_PACKAGE_MB_PER_TRANSFER = 2048U;                                                 // 2048M
constexpr uint32_t MIN_TEST_TIME = 1U;                                                                  // 秒
constexpr uint32_t MAX_TEST_TIME = 60U;                                                                 // 秒
constexpr uint32_t DEFAULT_THREAD_NUM_PER_DEVICE = 2;                                                   // 线程数
constexpr uint64_t MB_TO_BYTES = 1024U * 1024U;                                                         // 1M字节
constexpr uint64_t DEFAULT_PACKAGE_MB_PER_TRANSFER = 64U;                                               // 64M
constexpr uint64_t MIN_PACKAGE_BYTES_PER_TRANSFER = MB_TO_BYTES * MIN_PACKAGE_MB_PER_TRANSFER;          // 2M
constexpr uint64_t MAX_PACKAGE_BYTES_PER_TRANSFER = MB_TO_BYTES * MAX_PACKAGE_MB_PER_TRANSFER;          // 2048M
constexpr uint32_t DEFAULT_TEST_TIME = 2U;                                                              // 秒

// 每10毫秒查询一次，查询8000次，这个时间超过MAX_TEST_TIME(60)秒，并为最后一个包可能导致的超时预留时间
OckTopoDetectConf::OckTopoDetectConf(void)
    : toolParrallQueryIntervalMicroSecond(10000U), toolParrallMaxQueryTimes(8000U),
    defaultThreadNumPerDevice(DEFAULT_THREAD_NUM_PER_DEVICE), defaultTestTime(DEFAULT_TEST_TIME),
    defaultPackageMBPerTransfer(DEFAULT_PACKAGE_MB_PER_TRANSFER),
    threadNumPerDevice(ParamRange<uint32_t>(MIN_THREAD_NUM_PER_DEVICE, MAX_THREAD_NUM_PER_DEVICE)),
    testTime(ParamRange<uint32_t>(MIN_TEST_TIME, MAX_TEST_TIME)),
    packageMBPerTransfer(ParamRange<uint64_t>(MIN_PACKAGE_MB_PER_TRANSFER, MAX_PACKAGE_MB_PER_TRANSFER)),
    packagePerTransfer(ParamRange<uint64_t>(MIN_PACKAGE_BYTES_PER_TRANSFER, MAX_PACKAGE_BYTES_PER_TRANSFER))
{}

bool operator==(const OckTopoDetectConf &lhs, const OckTopoDetectConf &rhs)
{
    return lhs.toolParrallQueryIntervalMicroSecond == rhs.toolParrallQueryIntervalMicroSecond &&
           lhs.toolParrallMaxQueryTimes == rhs.toolParrallMaxQueryTimes;
}
bool operator!=(const OckTopoDetectConf &lhs, const OckTopoDetectConf &rhs)
{
    return !(lhs == rhs);
}
std::ostream &operator<<(std::ostream &os, const OckTopoDetectConf &conf)
{
    return os << "'toolParrallQueryIntervalMicroSecond':" << conf.toolParrallQueryIntervalMicroSecond
              << ",'toolParrallMaxQueryTimes':" << conf.toolParrallMaxQueryTimes;
}
}  // namespace conf
}  // namespace ock