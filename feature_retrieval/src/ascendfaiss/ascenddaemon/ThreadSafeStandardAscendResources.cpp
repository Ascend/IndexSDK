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


#include "ThreadSafeStandardAscendResources.h"

namespace ascend {
ThreadSafeStandardAscendResources::ThreadSafeStandardAscendResources(std::shared_ptr<DistComputeOpsManager> opsMng,
    std::string modelPath) : StandardAscendResources(opsMng, modelPath)
{
}

void ThreadSafeStandardAscendResources::setDevice(int deviceId)
{
    std::lock_guard<std::recursive_mutex> lock(mtx);
    StandardAscendResources::setDevice(deviceId);
}

void ThreadSafeStandardAscendResources::noTempMemory()
{
    std::lock_guard<std::recursive_mutex> lock(mtx);
    StandardAscendResources::noTempMemory();
}

void ThreadSafeStandardAscendResources::setTempMemory(size_t size)
{
    {
        std::lock_guard<std::recursive_mutex> lock(mtx);
        if (tempMemSize != size) {
            tempMemSize = getDefaultTempMemSize(size);
        }
    }
    // 这里不能锁住allocMemory，因为ascendStackMemory是线程安全的内部也有锁，会导致this阻塞；
    // 同时，this一旦阻塞，可能导致ascendStackMemory内部锁无法释放，而造成死锁
    ascendStackMemory.allocMemory(tempMemSize);
}

void ThreadSafeStandardAscendResources::setDefaultTempMemory()
{
    std::lock_guard<std::recursive_mutex> lock(mtx);
    StandardAscendResources::setDefaultTempMemory();
}

void ThreadSafeStandardAscendResources::setAscendMallocWarning(bool flag)
{
    std::lock_guard<std::recursive_mutex> lock(mtx);
    StandardAscendResources::setAscendMallocWarning(flag);
}

std::shared_ptr<AscendStreamIntf> ThreadSafeStandardAscendResources::createAscendStream(aclrtStream stream)
{
    return std::make_shared<ThreadSafeAscendStream>(stream, streamMtxMap[stream]);
}

void ThreadSafeStandardAscendResources::initialize()
{
    std::lock_guard<std::recursive_mutex> lock(mtx);
    StandardAscendResources::initialize();
}

void ThreadSafeStandardAscendResources::uninitialize()
{
    std::lock_guard<std::recursive_mutex> lock(mtx);
    StandardAscendResources::uninitialize();
}

size_t ThreadSafeStandardAscendResources::getResourceSize() const
{
    std::lock_guard<std::recursive_mutex> lock(mtx);
    return StandardAscendResources::getResourceSize();
}

[[nodiscard]] std::shared_ptr<AscendStreamIntf> ThreadSafeStandardAscendResources::getDefaultStream()
{
    std::lock_guard<std::recursive_mutex> lock(mtx);
    return StandardAscendResources::getDefaultStream();
}

[[nodiscard]] std::vector<std::shared_ptr<AscendStreamIntf>> ThreadSafeStandardAscendResources::getAlternateStreams()
{
    std::lock_guard<std::recursive_mutex> lock(mtx);
    return StandardAscendResources::getAlternateStreams();
}

AscendMemory& ThreadSafeStandardAscendResources::getMemoryManager()
{
    std::lock_guard<std::recursive_mutex> lock(mtx);
    return StandardAscendResources::getMemoryManager();
}

void ThreadSafeStandardAscendResources::addUseCount()
{
    std::lock_guard<std::recursive_mutex> lock(mtx);
    StandardAscendResources::addUseCount();
}

void ThreadSafeStandardAscendResources::reduceUseCount()
{
    std::lock_guard<std::recursive_mutex> lock(mtx);
    StandardAscendResources::reduceUseCount();
}
} // namespace ascend
