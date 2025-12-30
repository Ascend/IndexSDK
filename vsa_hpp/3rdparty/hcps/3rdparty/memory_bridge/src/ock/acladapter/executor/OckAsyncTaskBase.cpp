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

#include <ostream>
#include <map>
#include "ock/acladapter/executor/OckAsyncResultInnerBridge.h"
#include "ock/acladapter/executor/OckAsyncTaskBase.h"
namespace ock {
namespace acladapter {
OckDefaultResult::OckDefaultResult(hmm::OckHmmErrorCode retCode) : errorCode(retCode)
{}
hmm::OckHmmErrorCode OckDefaultResult::ErrorCode(void) const
{
    return errorCode;
}
std::ostream &operator<<(std::ostream &os, const OckAsyncTaskContext &context)
{
    return os << "'deviceId':" << context.GetDeviceId();
}
std::ostream &operator<<(std::ostream &os, const OckAsyncTaskBase &task)
{
    return os << "{'resourceType':" << task.ResourceType() << ", 'name':" << task.Name()
              << ", 'param':" << task.ParamInfo() << "}";
}
std::ostream &operator<<(std::ostream &os, const OckAsyncTaskParamBase &data)
{
    return os;
}
std::ostream &operator<<(std::ostream &os, const OckDftAsyncTaskParam &data)
{
    return os;
}
}  // namespace acladapter
}  // namespace ock