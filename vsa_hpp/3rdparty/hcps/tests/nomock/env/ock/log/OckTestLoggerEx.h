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

#ifndef HCPS_PIER_TESTS_NOMOCK_ENV_OCK_LOG_OCKTESTLOGGEREX_H
#define HCPS_PIER_TESTS_NOMOCK_ENV_OCK_LOG_OCKTESTLOGGEREX_H
#include <string.h>
#include <sstream>
#include <thread>
#include <mutex>
#include <atomic>
#include "ock/log/OckHcpsLogger.h"
namespace ock {
#define OCK_CHECK_ERRORCODE_MSG(errorCode, msg)             \
    do {                                                  \
        auto innerTmpErrorCode = (errorCode);             \
        if (innerTmpErrorCode != ock::hmm::HMM_SUCCESS) { \
            OCK_HCPS_LOG_ERROR("erroCode: " << errorCode << ", msg: " << msg);                     \
        }                                                 \
    } while (0)
} // namespace ock
#endif // HCPS_PIER_TESTS_NOMOCK_ENV_OCK_LOG_OCKTESTLOGGEREX_H
