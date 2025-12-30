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
#include "ock/utils/OckThreadUtils.h"
namespace ock {
namespace utils {
namespace {
void GetCpuIdSet(const cpu_set_t &cpuSet, std::vector<uint32_t> &outDatas)
{
    auto cpuCount = sysconf(_SC_NPROCESSORS_CONF);
    if (cpuCount >= 0) {
        for (uint32_t i = 0; i < static_cast<uint32_t>(cpuCount); ++i) {
            if (CPU_ISSET(i, &cpuSet)) {
                outDatas.push_back(i);
            }
        }
    }
}
void InitCpuSetVec(std::vector<cpu_set_t> &outData)
{
    for (auto &cpuSet : outData) {
        CPU_ZERO(&cpuSet);
    }
}
void SetCpuSetWithMoreCPUCores(
    const std::vector<uint32_t> &cpuIdSet, std::vector<cpu_set_t> &outData, uint32_t threadCount)
{
    for (uint32_t i = 0; i < cpuIdSet.size(); ++i) {
        CPU_SET(cpuIdSet[i], &outData[i % threadCount]);
    }
}
void SetCpuSetWithLessCPUCores(
    const std::vector<uint32_t> &cpuIdSet, std::vector<cpu_set_t> &outData, uint32_t threadCount)
{
    for (uint32_t i = 0; i < threadCount; ++i) {
        CPU_SET(cpuIdSet[i % cpuIdSet.size()], &outData[i]);
    }
}
}  // namespace

void WaitThreadsExecComplete(std::vector<std::thread> &threads)
{
    for (auto &thd : threads) {
        thd.join();
    }
}

std::vector<cpu_set_t> DispatchCpuSet(const cpu_set_t &cpuSet, uint32_t threadCount)
{
    if (threadCount == 0UL) {
        return std::vector<cpu_set_t>();
    }
    std::vector<uint32_t> cpuIdSet;
    GetCpuIdSet(cpuSet, cpuIdSet);

    std::vector<cpu_set_t> ret(threadCount);
    InitCpuSetVec(ret);
    if (cpuIdSet.empty()) {
        return ret;
    }
    if (threadCount <= cpuIdSet.size()) {
        SetCpuSetWithMoreCPUCores(cpuIdSet, ret, threadCount);
    } else {
        SetCpuSetWithLessCPUCores(cpuIdSet, ret, threadCount);
    }
    return ret;
}

cpu_set_t CombineCpuSets(std::vector<cpu_set_t> cpuSets)
{
    cpu_set_t ret;
    CPU_ZERO(&ret);
    for (uint32_t i = 0; i < cpuSets.size(); ++i) {
        CPU_OR(&ret, &ret, &cpuSets[i]);
    }
    return ret;
}
}  // namespace utils
}  // namespace ock