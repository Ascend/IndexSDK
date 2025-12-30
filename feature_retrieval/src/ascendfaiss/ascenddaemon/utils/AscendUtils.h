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


#ifndef ASCEND_UTILS_INCLUDED
#define ASCEND_UTILS_INCLUDED

#include <initializer_list>
#include <string>
#include <mutex>
#include <shared_mutex>
#include <vector>
#include <optional>
#include <unordered_map>

#include "acl/acl.h"
#include "securec.h"

#include "ascenddaemon/utils/StaticUtils.h"
#include "common/utils/AscendAssert.h"

namespace ascend {

constexpr uint32_t MAX_DEVICEID = 1024; // 设备数量上限

aclError synchronizeStream(aclrtStream stream);

class AscendUtils {
public:
    // Sets the current thread-local Ascend device
    static void setCurrentDevice(int device);

    // Reset the current thread-local Ascend device
    static void resetCurrentDevice(int device);

    // Sets the current thread-local Ascend device
    static void setCurrentContext(aclrtContext ctx);

    // Returns the current thread-local Ascend context
    static aclrtContext getCurrentContext();
};

class AscendMultiThreadManager {
public:
    static bool IsMultiThreadMode();

    static void InitGetBaseMtx(const std::vector<int> &deviceList,
        std::unordered_map<int, std::mutex> &getBaseMtx);

    static std::optional<std::lock_guard<std::mutex>> LockGetBaseMtx(int deviceId,
        std::unordered_map<int, std::mutex> &getBaseMtx);

    template <typename T>
    [[nodiscard]] static std::optional<std::shared_lock<T>> GetReadLock(T &mutex)
    {
        if (!IsMultiThreadMode()) {
            return std::nullopt;
        }

        return { std::shared_lock<T>(mutex) };
    }

    template <typename T>
    [[nodiscard]] static std::optional<std::unique_lock<T>> GetWriteLock(T &mutex)
    {
        if (!IsMultiThreadMode()) {
            return std::nullopt;
        }

        return { std::unique_lock<T>(mutex) };
    }

private:
    static void GetMultiThreadMode(bool &multiThreadFlag);
};

class AscendOperatorManager {
public:
    static bool Init(std::string path);
    virtual ~AscendOperatorManager();

private:
    AscendOperatorManager() = delete;
};

class DeviceScope {
public:
    DeviceScope();
    ~DeviceScope();

    DeviceScope(const DeviceScope&) = delete;
    DeviceScope& operator=(const DeviceScope&) = delete;
};

class AscendGlobalLock {
public:
    static std::mutex &GetInstance(uint32_t device);
    AscendGlobalLock(const AscendGlobalLock&) = delete;
    AscendGlobalLock(AscendGlobalLock&&) = delete;
    AscendGlobalLock& operator=(const AscendGlobalLock&) = delete;
    AscendGlobalLock& operator=(AscendGlobalLock&&) = delete;
private:
    AscendGlobalLock() = default;
    ~AscendGlobalLock() = default;
};

#define WAITING_FLAG_READY(flag, checkTicks, timeout)                                                  \
    do {                                                                                               \
        int waitTicks_ = 0;                                                                            \
        double startWait_ = utils::getMillisecs();                                                     \
        while (!(flag)) {                                                                              \
            waitTicks_++;                                                                              \
            if (!(waitTicks_ % (checkTicks)) && ((utils::getMillisecs() - startWait_) >= (timeout))) { \
                APP_LOG_ERROR("wait timeout.");                                                        \
                return;                                                                                \
            }                                                                                          \
        }                                                                                              \
    } while (false)

#define WAITING_FLAG_READY_TIMEOUT(flag, timeout)                                                      \
    do {                                                                                               \
        double startWait = utils::getMillisecs();                                                      \
        while (!(flag)) {                                                                              \
            if ((utils::getMillisecs() - startWait) >= (timeout)) {                                    \
                APP_LOG_ERROR("wait timeout.");                                                        \
                return;                                                                                \
            }                                                                                          \
        }                                                                                              \
    } while (false)

// Wrapper to test return status of ACL functions
#define ACL_REQUIRE_OK(x)                                                           \
    do {                                                                            \
        auto err = (x);                                                             \
        ASCEND_THROW_IF_NOT_FMT(err == ACL_ERROR_NONE, "ACL error %d", static_cast<int>(err));   \
    } while (0)

// Wrapper to test return status of ACL functions and return
#define ACL_REQUIRE_OK_RET_CODE(x, errcode)                                                     \
    do {                                                                                        \
        auto err = (x);                                                                         \
        ASCEND_EXC_IF_NOT_FMT(err == ACL_ERROR_NONE, return (errcode), "ACL error %d", static_cast<int>(err)); \
    } while (0)

#define VALUE_UNUSED(x) (void)(x)
} // ascend

#define CALL_PARALLEL_FUNCTOR(devices, threadPool, functor)                         \
    do {                                                                            \
        if ((devices) > 1) {                                                        \
            std::vector<std::future<void>> functorRets;                             \
            for (size_t i = 0; i < (devices); i++) {                                \
                functorRets.emplace_back(threadPool->Enqueue(functor, i));          \
            }                                                                       \
                                                                                    \
            try {                                                                   \
                for (auto & ret : functorRets) {                                    \
                    ret.get();                                                      \
                }                                                                   \
            } catch (std::exception & e) {                                          \
                FAISS_THROW_FMT("wait for parallel future failed: %s", e.what());   \
            }                                                                       \
        } else {                                                                    \
            functor(0);                                                             \
        }                                                                           \
    } while (false)

#define CALL_PARALLEL_FUNCTOR_INDEXMAP(map, threadPool, functor)                         \
    do {                                                                                 \
        std::vector<std::future<void>> functorRets;                                      \
        for (auto & index : (map)) {                                                     \
            functorRets.emplace_back(                                                    \
                (threadPool)->Enqueue(functor, (index.first), (index.second)));          \
        }                                                                                \
                                                                                         \
        try {                                                                            \
            for (auto & ret : functorRets) {                                             \
                ret.get();                                                               \
            }                                                                            \
        } catch (std::exception & e) {                                                   \
            FAISS_THROW_FMT("wait for indexmap parallel future failed: %s", e.what());   \
        }                                                                                \
    } while (false)

#endif