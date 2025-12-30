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

#include "ock/conf/OckAclAdapterConf.h"
namespace ock {
namespace conf {
OckAclAdapterConf::OckAclAdapterConf(void)
    : maxFreeWaitMilliSecondThreshold(750U * 1000U), taskThreadMaxCyclePickUpInterval(1U),
      taskThreadQueryStartInterval(10U), taskThreadMaxQueryStartTimes(500U), taskStatisticsMaxGapMicroseconds(100U),
      maxStreamOpTriggerMilliSecond(30U)
{}
bool operator==(const OckAclAdapterConf &lhs, const OckAclAdapterConf &rhs)
{
    return lhs.maxFreeWaitMilliSecondThreshold == rhs.maxFreeWaitMilliSecondThreshold &&
           lhs.taskThreadMaxCyclePickUpInterval == rhs.taskThreadMaxCyclePickUpInterval &&
           lhs.taskThreadQueryStartInterval == rhs.taskThreadQueryStartInterval &&
           lhs.taskThreadMaxQueryStartTimes == rhs.taskThreadMaxQueryStartTimes &&
           lhs.taskStatisticsMaxGapMicroseconds == rhs.taskStatisticsMaxGapMicroseconds &&
           lhs.maxStreamOpTriggerMilliSecond == rhs.maxStreamOpTriggerMilliSecond;
}
bool operator!=(const OckAclAdapterConf &lhs, const OckAclAdapterConf &rhs)
{
    return !(lhs == rhs);
}
std::ostream &operator<<(std::ostream &os, const OckAclAdapterConf &conf)
{
    return os << "'maxFreeWaitMilliSecondThreshold':" << conf.maxFreeWaitMilliSecondThreshold
              << ",'taskThreadMaxCyclePickUpInterval':" << conf.taskThreadMaxCyclePickUpInterval
              << ",'taskThreadQueryStartInterval':" << conf.taskThreadQueryStartInterval
              << ",'taskThreadMaxQueryStartTimes':" << conf.taskThreadMaxQueryStartTimes
              << ",'taskStatisticsMaxGapMicroseconds':" << conf.taskStatisticsMaxGapMicroseconds
              << ",'maxStreamOpTriggerMilliSecond':" << conf.maxStreamOpTriggerMilliSecond;
}
}  // namespace conf
}  // namespace ock