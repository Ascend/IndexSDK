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


#ifndef OCK_HCPS_LOGGER_H
#define OCK_HCPS_LOGGER_H
#include <string.h>
#include <sstream>
#include <thread>
#include <mutex>
#include <atomic>
#include "ock/log/OckLogger.h"
namespace ock {
#define OCK_HCPS_LOG_DEBUG(msg) INNER_OCK_LOG(OCK_LOG_LEVEL_DEBUG, "DEBUG", msg, "HCPS")
#define OCK_HCPS_LOG_INFO(msg) INNER_OCK_LOG(OCK_LOG_LEVEL_INFO, "INFO", msg, "HCPS")
#define OCK_HCPS_LOG_WARN(msg) INNER_OCK_LOG(OCK_LOG_LEVEL_WARN, "WARN", msg, "HCPS")
#define OCK_HCPS_LOG_ERROR(msg) INNER_OCK_LOG(OCK_LOG_LEVEL_ERROR, "ERROR", msg, "HCPS")
#define OCK_HCPS_LOG_FATAL(msg) INNER_OCK_LOG(OCK_LOG_LEVEL_FATAL, "FATAL", msg, "HCPS")
#define OCK_HCPS_LOG_THROW(exceptClass, msg) INNER_OCK_THROW(exceptClass, msg, "HCPS")

#define OCK_CHECK_RETURN_ERRORCODE(errorCode)             \
    do {                                                  \
        auto innerTmpErrorCode = (errorCode);             \
        if (innerTmpErrorCode != ock::hmm::HMM_SUCCESS) { \
            return innerTmpErrorCode;                     \
        }                                                 \
    } while (0)

#define OCK_CHECK_PTR_NULL_RETURN(sharedPtr)     \
    do {                                    \
        auto innerTmpPtr = (sharedPtr);     \
        if (innerTmpPtr.get() == nullptr) { \
            return;             \
        }                                   \
    } while (0)
} // namespace ock
#endif