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

#include "acl/acl.h"

#include <ascenddaemon/utils/StaticUtils.h>
#include <common/utils/AscendAssert.h>

namespace ascendSearch {
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

        static void attachToCpu(int cpuId);

        static void attachToCpus(std::initializer_list<uint8_t> cpus);
    };

    class AscendOperatorManager {
    public:
        static void init(std::string path);
        virtual ~AscendOperatorManager();

    private:
        AscendOperatorManager() = delete;
    };

    class DeviceScope {
    public:
        DeviceScope();
        ~DeviceScope();
    };

    class AscendGlobalLock {
    public:
        static std::mutex &GetInstance();
        AscendGlobalLock(const AscendGlobalLock&) = delete;
        AscendGlobalLock(AscendGlobalLock&&) = delete;
        AscendGlobalLock& operator=(const AscendGlobalLock&) = delete;
        AscendGlobalLock& operator=(AscendGlobalLock&&) = delete;
    private:
        AscendGlobalLock() = default;
        ~AscendGlobalLock() = default;
    };

#define CALL_PARALLEL_FUNCTOR(pcounter, pmin, forCounter, threadPool, functor) \
    do {                                                                       \
        if ((pcounter) >= (pmin)) {                                                \
            std::vector<std::future<void>> functorRets;                        \
            for (int i = 0; i < (forCounter); i++) {                             \
                functorRets.emplace_back((threadPool)->Enqueue(functor, i));     \
            }                                                                  \
                                                                               \
            try {                                                              \
                for (auto &ret : functorRets) {                                \
                    ret.get();                                                 \
                }                                                              \
            } catch (std::exception & e) {                                     \
                ASCEND_THROW_MSG("wait for parallel future failed.");          \
            }                                                                  \
        } else {                                                               \
            for (int i = 0; i < (forCounter); i++) {                             \
                functor(i);                                                    \
            }                                                                  \
        }                                                                      \
    } while (false)

#define CALL_PARALLEL_FUNCTOR_NOEXCEPTION(pcounter, pmin, forCounter, threadPool, functor) \
    do {                                                                                   \
        if ((pcounter) >= (pmin)) {                                                            \
            std::vector<std::future<void>> functorRets;                                    \
            for (int i = 0; i < (forCounter); i++) {                                         \
                functorRets.emplace_back((threadPool)->Enqueue(functor, i));                 \
            }                                                                              \
                                                                                           \
            for (auto &ret : functorRets) {                                                \
                ret.get();                                                                 \
            }                                                                              \
        } else {                                                                           \
            for (int i = 0; i < (forCounter); i++) {                                         \
                functor(i);                                                                \
            }                                                                              \
        }                                                                                  \
    } while (false)

#define WAITING_FLAG_READY(flag, checkTicks, timeout)                                                  \
    do {                                                                                               \
        int waitTicks_ = 0;                                                                            \
        double startWait_ = utils::getMillisecs();                                                     \
        while (!(flag)) {                                                                              \
            waitTicks_++;                                                                              \
            if (!(waitTicks_ % (checkTicks)) && ((utils::getMillisecs() - startWait_) >= (timeout))) { \
                APP_LOG_ERROR("wait timeout.");                                                        \
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

#endif