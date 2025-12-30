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
#include <chrono>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <sys/syscall.h>
#include "ock/log/OckLogger.h"

namespace ock {
void DftOckHmmLogHandler::Write(
    int32_t level, const char *levelStr, const char *fileName, uint64_t lineNo, const char *msg)
{
    auto now = std::chrono::system_clock::now();
    std::time_t nowTimeT = std::chrono::system_clock::to_time_t(now);
    std::tm nowTm;
    localtime_r(&nowTimeT, &nowTm);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) % 1000U;

    std::cout << "[" << getpid() << "][" << syscall(SYS_gettid) << "][" << (nowTm.tm_year + 1900U) << "-"
              << (nowTm.tm_mon + 1U) << "-" << nowTm.tm_mday << " " << nowTm.tm_hour << ":" << nowTm.tm_min << ":"
              << nowTm.tm_sec << "." << ms.count() << "][" << levelStr << "][" << fileName << ":" << lineNo << "]"
              << msg << std::endl;
}

void DftOckHmmLogHandler::Flush(void)
{
    std::cout.flush();
}
OckLogger::OckLogger(void) : handler(std::make_shared<DftOckHmmLogHandler>()), startLevel(OCK_LOG_LEVEL_ERROR)
{}

int32_t OckLogger::GetStartLevel(void) const
{
    return startLevel.load();
}

void OckLogger::Write(int32_t level, const char *levelStr, const char *fileName, uint64_t lineNo, const char *msg)
{
    if (level < startLevel.load()) {
        return;
    }
    std::lock_guard<std::mutex> guard(handlerMutex);
    handler->Write(level, levelStr, fileName, lineNo, msg);
}

void OckLogger::SetHandler(std::shared_ptr<OckHmmLogHandler> newHandler)
{
    std::lock_guard<std::mutex> guard(handlerMutex);
    handler = newHandler;
}

void OckLogger::SetLevel(int32_t level)
{
    startLevel.store(level);
}

OckLogger &OckLogger::Instance(void)
{
    static OckLogger ins;
    return ins;
}

OckLogger &g_instance_OckLogger = OckLogger::Instance();

void OckHmmSetLogHandler(std::shared_ptr<OckHmmLogHandler> handler)
{
    if (handler.get() == nullptr) {
        std::cout << "input handler is nullptr!" << std::endl;
        return;
    }
    OckLogger::Instance().SetHandler(handler);
}

void OckHmmSetLogLevel(int32_t level)
{
    if (level < OCK_LOG_LEVEL_DEBUG) {
        level = OCK_LOG_LEVEL_DEBUG;
    }
    if (level > OCK_LOG_LEVEL_FATAL) {
        level = OCK_LOG_LEVEL_FATAL;
    }
    OckLogger::Instance().SetLevel(level);
}
}  // namespace ock