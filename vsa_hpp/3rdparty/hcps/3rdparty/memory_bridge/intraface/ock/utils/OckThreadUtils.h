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


#ifndef OCK_HCPS_PIER_THREAD_UTILS_H
#define OCK_HCPS_PIER_THREAD_UTILS_H
#include <cstdint>
#include <vector>
#include <thread>
namespace ock {
namespace utils {

void WaitThreadsExecComplete(std::vector<std::thread> &threads);
std::vector<cpu_set_t> DispatchCpuSet(const cpu_set_t &cpuSet, uint32_t threadCount);
cpu_set_t CombineCpuSets(std::vector<cpu_set_t> cpuSets);

}  // namespace utils
}  // namespace ock
#endif