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


#ifndef OCK_MEMORY_BRIDGE_OCK_LOGGER_H
#define OCK_MEMORY_BRIDGE_OCK_LOGGER_H
#include <string.h>
#include <sstream>
#include <thread>
#include <mutex>
#include <atomic>
#include <stdexcept>
#include "ock/log/OckHmmLogHandler.h"
namespace ock {

class DftOckHmmLogHandler : public OckHmmLogHandler {
public:
    explicit DftOckHmmLogHandler() = default;
    ~DftOckHmmLogHandler() noexcept = default;
    void Write(int32_t level, const char *levelStr, const char *fileName, uint64_t lineNo, const char *msg) override;
    void Flush(void) override;
};
class NullOckHmmLogHandler : public OckHmmLogHandler {
public:
    explicit NullOckHmmLogHandler() = default;
    ~NullOckHmmLogHandler() noexcept = default;
    void Write(int32_t, const char *, const char *, uint64_t, const char *) override
    {}
    void Flush(void) override
    {}
};

class OckLogger {
public:
    virtual ~OckLogger() noexcept = default;
    explicit OckLogger(void);
    int32_t GetStartLevel(void) const;
    void Write(int32_t level, const char *levelStr, const char *fileName, uint64_t lineNo, const char *msg);
    void SetHandler(std::shared_ptr<OckHmmLogHandler> handler);
    void SetLevel(int32_t level);

    static OckLogger &Instance(void);

private:
    std::shared_ptr<OckHmmLogHandler> handler;
    std::atomic<int32_t> startLevel;
    std::mutex handlerMutex{};
};
#define OCK_HMM_LOG_FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)

#define INNER_OCK_LOG(level, levelStr, msg, moudleName)                                                              \
    do {                                                                                                             \
        if (level >= OckLogger::Instance().GetStartLevel()) {                                                        \
            std::ostringstream tmpLogOsStr;                                                                          \
            tmpLogOsStr << "[" << moudleName << "]" << msg;                                                          \
            OckLogger::Instance().Write(level, levelStr, OCK_HMM_LOG_FILENAME, __LINE__, tmpLogOsStr.str().c_str()); \
        }                                                                                                            \
    } while (0)
#define INNER_OCK_THROW(exceptClass, msg, moudleName)                                                     \
    do {                                                                                                  \
        if (OCK_LOG_LEVEL_FATAL >= OckLogger::Instance().GetStartLevel()) {                               \
            std::ostringstream tmpLogOsStr;                                                               \
            tmpLogOsStr << "[" << moudleName << "]" << msg;                                               \
            OckLogger::Instance().Write(                                                                  \
                OCK_LOG_LEVEL_FATAL, "FATAL", OCK_HMM_LOG_FILENAME, __LINE__, tmpLogOsStr.str().c_str()); \
            throw exceptClass(tmpLogOsStr.str());                                                         \
        }                                                                                                 \
    } while (0)

#define OCK_HMM_LOG_DEBUG(msg) INNER_OCK_LOG(OCK_LOG_LEVEL_DEBUG, "DEBUG", msg, "HMM")
#define OCK_HMM_LOG_INFO(msg) INNER_OCK_LOG(OCK_LOG_LEVEL_INFO, "INFO", msg, "HMM")
#define OCK_HMM_LOG_WARN(msg) INNER_OCK_LOG(OCK_LOG_LEVEL_WARN, "WARN", msg, "HMM")
#define OCK_HMM_LOG_ERROR(msg) INNER_OCK_LOG(OCK_LOG_LEVEL_ERROR, "ERROR", msg, "HMM")
#define OCK_HMM_LOG_FATAL(msg) INNER_OCK_LOG(OCK_LOG_LEVEL_FATAL, "FATAL", msg, "HMM")
#define OCK_HMM_LOG_THROW(exceptClass, msg) INNER_OCK_THROW(exceptClass, msg, "HMM")

}  // namespace ock
#endif