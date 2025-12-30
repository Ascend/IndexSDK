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


#include <npu/common/utils/AscendUtils.h>
#include <mutex>
#include <thread>
#include "npu/common/utils/CommonUtils.h"
#include "npu/common/utils/LogUtils.h"

namespace ascendSearchacc {
void AscendUtils::setCurrentDevice(int device)
{
    ACL_REQUIRE_OK(aclrtSetDevice(device));
}

void AscendUtils::resetCurrentDevice(int device)
{
    (void)aclrtResetDevice(device);
}

aclrtContext AscendUtils::getCurrentContext()
{
    aclrtContext ctx = 0;
    aclrtGetCurrentContext(&ctx);
    return ctx;
}

void AscendUtils::setCurrentContext(aclrtContext ctx)
{
    ACL_REQUIRE_OK(aclrtSetCurrentContext(ctx));
}

void AscendUtils::attachToCpu(int cpuId)
{
    size_t cpu = (size_t)cpuId % std::thread::hardware_concurrency();
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(cpu, &cpuset);
    (void)pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
}

void AscendUtils::attachToCpus(std::initializer_list<uint8_t> cpus)
{
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);

    for (auto cpuId : cpus) {
        size_t cpu = (size_t)cpuId % std::thread::hardware_concurrency();
        CPU_SET(cpu, &cpuset);
    }

    (void)pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
}

void AscendOperatorManager::init(std::string path)
{
    static bool isInited = false;
    static std::mutex mtx;

    std::lock_guard<std::mutex> lock(mtx);
    if (isInited) {
        return;
    }

#ifdef HOSTCPU
    char *modelpath = std::getenv("MX_INDEX_MODELPATH");
    if (modelpath != nullptr) {
        path = CommonUtils::RealPath(std::string(modelpath));
        ASCEND_THROW_IF_NOT_MSG(!path.empty(), "Modelpath from env is invalid");
        ASCEND_THROW_IF_NOT_FMT(CommonUtils::CheckPathValid(path),
                                "Modelpath from env: %s must be in home directory and readable", path.c_str());
        APP_LOG_INFO("Use env %s as modelpath", path.c_str());
    }
#endif
    ACL_REQUIRE_OK(aclopSetModelDir(path.c_str()));
    isInited = true;
}

AscendOperatorManager::~AscendOperatorManager()
{
}

DeviceScope::DeviceScope()
{
    AscendUtils::setCurrentDevice(0);
}

DeviceScope::~DeviceScope()
{
    AscendUtils::resetCurrentDevice(0);
}
}  // namespace ascendSearchacc
