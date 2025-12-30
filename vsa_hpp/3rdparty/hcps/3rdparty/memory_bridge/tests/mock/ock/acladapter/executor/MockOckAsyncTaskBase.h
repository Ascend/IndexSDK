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


#ifndef OCK_MEMORY_BRIDGE_OCK_ADAPTER_MOCK_OCK_ASYNC_TASK_BASE_H
#define OCK_MEMORY_BRIDGE_OCK_ADAPTER_MOCK_OCK_ASYNC_TASK_BASE_H
#include <gmock/gmock.h>
#include "ock/acladapter/executor/OckAsyncTaskBase.h"

namespace ock {
namespace acladapter {

class MockOckAsyncTaskBase : public OckAsyncTaskBase {
public:
    MOCK_CONST_METHOD0(ResourceType, OckTaskResourceType());
    MOCK_CONST_METHOD0(Name, std::string());
    MOCK_CONST_METHOD0(ParamInfo, std::string());
    MOCK_METHOD0(PreConditionMet, bool());
    MOCK_METHOD1(Run, void(OckAsyncTaskContext &context));
    MOCK_METHOD0(Cancel, void());
};

}  // namespace acladapter
}  // namespace ock
#endif