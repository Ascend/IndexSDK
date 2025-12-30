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


#include <ascenddaemon/StandardAscendResources.h>

#include <ascenddaemon/utils/AscendUtils.h>
#include <ascenddaemon/utils/AscendStackMemory.h>
#include <common/utils/CommonUtils.h>
#include <common/utils/LogUtils.h>
#include <common/utils/SocUtils.h>

namespace ascendSearch {
// How many streams we allocate by default (for multi-streaming)
const int NUM_STREAMS = 2;

// Default temporary memory allocation
const size_t BIT_SHIFT_FOR_TEMP_MEM = 27;
const size_t DEFAULT_TEMP_MEM = static_cast<size_t>(1UL << BIT_SHIFT_FOR_TEMP_MEM);

// Max temporary memory allocation 1 << 33 means 8GB mem
const size_t MAX_TEMP_MEM = static_cast<size_t>(1UL << 33);
const size_t INVALID_TEMP_MEM = static_cast<size_t>(-1);
const size_t KB = static_cast<size_t>(1UL << 10);

namespace {
const int THREADS_CNT = faiss::ascendSearch::SocUtils::GetInstance().GetThreadsCnt();
const int MUL_SEARCH_THREADS_CNT = 4;
}

StandardAscendResources::StandardAscendResources(std::string modelPath)
    : defaultStream(nullptr),
      tempMemSize(INVALID_TEMP_MEM)
{
    APP_LOG_INFO("StandardAscendResources Construction");
    AscendOperatorManager::init(modelPath);
}

StandardAscendResources::~StandardAscendResources()
{
    APP_LOG_INFO("StandardAscendResources Destruction");
}

void StandardAscendResources::uninitialize()
{
    auto err = aclrtSetDevice(deviceId);
    if (err != ACL_SUCCESS) {
        (void)aclrtResetDevice(deviceId);
        APP_LOG_ERROR("failed to set device to %d in uninitialize", deviceId);
        return;
    }
    if (defaultStream) {
        err = aclrtSynchronizeStream(defaultStream);
        if (err != ACL_SUCCESS) {
            APP_LOG_ERROR("the result of aclrtSynchronizeStream() is err, err=%d.\n", err);
        }
        err = aclrtDestroyStream(defaultStream);
        if (err != ACL_SUCCESS) {
            APP_LOG_ERROR("the result of aclrtDestroyStream() is err, err=%d.\n", err);
        }
    }
    defaultStream = nullptr;

    for (auto& stream : alternateStreams) {
        err = aclrtSynchronizeStream(stream);
        if (err != ACL_SUCCESS) {
            APP_LOG_ERROR("the result of aclrtSynchronizeStream() is err, err=%d.\n", err);
        }
        err = aclrtDestroyStream(stream);
        if (err != ACL_SUCCESS) {
            APP_LOG_ERROR("the result of aclrtDestroyStream() is err, err=%d.\n", err);
        }
    }
    alternateStreams.clear();
    (void)aclrtResetDevice(deviceId);

    threadPool = nullptr;
    threadPoolMulSearch = nullptr;
    DistComputeOpsManager::getInstance().uninitialize();
    tempMemSize = INVALID_TEMP_MEM;
}

void StandardAscendResources::SetDevice(int deviceIdTmp)
{
    this->deviceId = deviceIdTmp;
}

void StandardAscendResources::noTempMemory()
{
    setTempMemory(0);
    setAscendMallocWarning(false);
}

void StandardAscendResources::setTempMemory(size_t size)
{
    if (tempMemSize != size) {
        tempMemSize = getDefaultTempMemSize(size);
        ascendStackMemory.allocMemory(tempMemSize);
    }
}

void StandardAscendResources::setDefaultTempMemory()
{
    setTempMemory(DEFAULT_TEMP_MEM);
}

void StandardAscendResources::setAscendMallocWarning(bool flag)
{
    ascendStackMemory.setMallocWarning(flag);
}

void StandardAscendResources::initialize()
{
    if (isInitialized()) {
        return;
    }

    // Create streams
    ACL_REQUIRE_OK(aclrtCreateStream(&defaultStream));

    for (int j = 0; j < NUM_STREAMS; ++j) {
        aclrtStream stream = nullptr;
        ACL_REQUIRE_OK(aclrtCreateStream(&stream));
        alternateStreams.push_back(stream);
    }

    if (tempMemSize == INVALID_TEMP_MEM) {
        setDefaultTempMemory();
    }

    threadPool = CREATE_UNIQUE_PTR(AscendThreadPool, THREADS_CNT);
    threadPoolMulSearch = CREATE_UNIQUE_PTR(AscendThreadPool, MUL_SEARCH_THREADS_CNT);
    DistComputeOpsManager::getInstance().initialize();
}

aclrtStream StandardAscendResources::getDefaultStream()
{
    initialize();
    return defaultStream;
}

void StandardAscendResources::syncDefaultStream()
{
    ACL_REQUIRE_OK(aclrtSynchronizeStream(getDefaultStream()));
}

std::vector<aclrtStream> StandardAscendResources::getAlternateStreams()
{
    initialize();
    return alternateStreams;
}

void StandardAscendResources::resetStack()
{
    ascendStackMemory.resetStack();
}

AscendMemory& StandardAscendResources::getMemoryManager()
{
    initialize();
    return ascendStackMemory;
}

bool StandardAscendResources::isInitialized() const
{
    return defaultStream != nullptr;
}

size_t StandardAscendResources::getDefaultTempMemSize(size_t requested) const
{
    return (requested > MAX_TEMP_MEM) ? MAX_TEMP_MEM : requested;
}
}  // namespace ascendSearch