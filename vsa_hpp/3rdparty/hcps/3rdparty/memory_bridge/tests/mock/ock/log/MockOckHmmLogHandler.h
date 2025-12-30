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


#ifndef OCK_MEMORY_BRIDGE_TEST_MOCK_OCK_HMM_LOG_HANDLER_H
#define OCK_MEMORY_BRIDGE_TEST_MOCK_OCK_HMM_LOG_HANDLER_H
#include <gmock/gmock.h>
#include "ock/log/OckHmmLogHandler.h"

namespace ock {

class MockOckHmmLogHandler : public OckHmmLogHandler {
public:
    MOCK_METHOD5(
        Write, void(int32_t level, const char *levelStr, const char *fileName, uint64_t lineNo, const char *msg));
    MOCK_METHOD0(Flush, void(void));
};

}  // namespace ock
#endif