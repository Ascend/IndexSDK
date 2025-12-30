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


#include "ascenddaemon/StandardAscendResources.h"
#include "ascenddaemon/utils/AscendUtils.h"
#include "ascenddaemon/utils/AscendStackMemory.h"
#include "common/utils/CommonUtils.h"
#include "common/utils/LogUtils.h"
#include "common/utils/SocUtils.h"
#include "ThreadSafeStandardAscendResources.h"

namespace ascend {
// How many streams we allocate by default (for multi-streaming)
const int NUM_STREAMS = 2;

// Default temporary memory allocation
const size_t BIT_SHIFT_FOR_TEMP_MEM = 27;
const size_t DEFAULT_TEMP_MEM = static_cast<size_t>(1UL << BIT_SHIFT_FOR_TEMP_MEM);

// Max temporary memory allocation 1 << 35 means 32GB mem
const size_t MAX_TEMP_MEM = static_cast<size_t>(1UL << 35);
const size_t INVALID_TEMP_MEM = static_cast<size_t>(-1);
const size_t KB = static_cast<size_t>(1UL << 10);

namespace {
const int THREADS_CNT = faiss::ascend::SocUtils::GetInstance().GetThreadsCnt();
const int MUL_SEARCH_THREADS_CNT = 4;
}

std::shared_ptr<StandardAscendResources> StandardAscendResources::getInstance()
{
    static uint32_t deviceCnt = faiss::ascend::SocUtils::GetInstance().GetDeviceCount();
    ASCEND_THROW_IF_NOT_MSG(deviceCnt != 0, "not found davinci device. ");

    static std::mutex mtx;
    static std::map<int, std::shared_ptr<StandardAscendResources>> resources;
    int curDeviceId = 0;
    auto ret = aclrtGetDevice(&curDeviceId);
    if (ret != ACL_SUCCESS) {
        APP_LOG_ERROR("failed to rt get device, acl ret: %d", ret);
        curDeviceId = 0;
    }
    auto lock = AscendMultiThreadManager::GetWriteLock(mtx);
    if (resources.find(curDeviceId) == resources.end()) {
        std::shared_ptr<StandardAscendResources> res = nullptr;
        if (AscendMultiThreadManager::IsMultiThreadMode()) {
            res = std::make_shared<ThreadSafeStandardAscendResources>(DistComputeOpsManager::getShared());
        } else {
            res = std::make_shared<StandardAscendResources>(DistComputeOpsManager::getShared());
        }
        res->setDevice(curDeviceId);
        resources[curDeviceId] = res;
    }
    return resources[curDeviceId];
}

StandardAscendResources::StandardAscendResources(std::shared_ptr<DistComputeOpsManager> opsMng, std::string modelPath)
    : tempMemSize(INVALID_TEMP_MEM),
      opsMng(opsMng),
      defaultStream(0)
{
    APP_LOG_INFO("StandardAscendResources Construction");
    AscendOperatorManager::Init(modelPath);
    if (AscendOperatorManager::Init(modelPath)) {
        useOnlineOp = false;
    } else {
        useOnlineOp = true;
    }
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
        if (auto err = synchronizeStream(defaultStream)) {
            APP_LOG_ERROR("the result of synchronizeStream() is err, err=%d.\n", err);
        }
        if (auto err = aclrtDestroyStream(defaultStream)) {
            APP_LOG_ERROR("the result of aclrtDestroyStream() is err, err=%d.\n", err);
        }
    }
    defaultStream = nullptr;

    for (auto& stream : alternateStreams) {
        if (auto err = synchronizeStream(stream)) {
            APP_LOG_ERROR("the result of synchronizeStream() is err, err=%d.\n", err);
        }
        if (auto err = aclrtDestroyStream(stream)) {
            APP_LOG_ERROR("the result of aclrtDestroyStream() is err, err=%d.\n", err);
        }
    }
    alternateStreams.clear();
    (void)aclrtResetDevice(deviceId);

    if (opsMng != nullptr) {
        opsMng->uninitialize();
    }
    tempMemSize = INVALID_TEMP_MEM;
}

void StandardAscendResources::setDevice(int curDeviceId)
{
    this->deviceId = curDeviceId;
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
        aclrtStream stream = 0;
        ACL_REQUIRE_OK(aclrtCreateStream(&stream));
        alternateStreams.push_back(stream);
    }

    if (tempMemSize == INVALID_TEMP_MEM) {
        setDefaultTempMemory();
    }

    if (opsMng != nullptr) {
        opsMng->initialize();
    }
}

[[nodiscard]] std::shared_ptr<AscendStreamIntf> StandardAscendResources::getDefaultStream()
{
    initialize();
    return createAscendStream(defaultStream);
}

[[nodiscard]] std::vector<std::shared_ptr<AscendStreamIntf>> StandardAscendResources::getAlternateStreams()
{
    initialize();
    return { createAscendStream(alternateStreams.front()), createAscendStream(alternateStreams.back()) };
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

size_t StandardAscendResources::getDefaultTempMemSize(size_t requested)
{
    return (requested > MAX_TEMP_MEM) ? MAX_TEMP_MEM : requested;
}

std::shared_ptr<AscendStreamIntf> StandardAscendResources::createAscendStream(aclrtStream stream)
{
    return std::make_shared<AclrtStreamAdaptor>(stream);
}

void StandardAscendResources::addUseCount()
{
    useCount++;
}

void StandardAscendResources::reduceUseCount()
{
    /*
     * useCount为0，表示所有StandardAscendResources的引用已经析构，这里资源处理存在两种场景：
     * 1. AscendIndex析构，但进程仍未结束。此时需要释放资源。
     * 2. 进程结束，全局资源析构，此时不能释放资源（acl资源析构顺序不定）。
     * 判断是否处于进程结束这里使用了一个trick：即opsMng智能指针是在当前对象创建后再构造的，
     * 进程退出状态下，它一定会先析构，如果这里获取的opsMng.use_count为2，则表示其静态局部变量依然存在，
     * 即进程处于运行状态，需要释放资源。
     */
    useCount--;
    if (useCount == 0 && opsMng.use_count() == 2) { // 如果这里获取的opsMng.use_count为2，则表示其静态局部变量依然存在
        uninitialize();
        resetStack();
    }
}
} // namespace ascend
