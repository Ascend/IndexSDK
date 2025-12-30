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

#ifndef OCK_MEMORY_BRIDGE_OCK_ACLADAPTER_CONF_H
#define OCK_MEMORY_BRIDGE_OCK_ACLADAPTER_CONF_H
#include <cstdint>
#include <ostream>

namespace ock {
namespace conf {
struct OckAclAdapterConf {
    OckAclAdapterConf(void);
    uint32_t maxFreeWaitMilliSecondThreshold;   // 内存释放的最大等待时间(单位: ms)
    uint32_t taskThreadMaxCyclePickUpInterval;  // 任务线程循环PickUp任务的最大间隔(单位: ms)
    uint32_t taskThreadQueryStartInterval;      // 等待线程初始化的sleep间隔(单位: ms)
    uint32_t taskThreadMaxQueryStartTimes;      // 等待线程初始化的最大查询次数(单位: 次)
    uint32_t taskStatisticsMaxGapMicroseconds;  // 一次连续数据中两次传输命令的最大间隔(单位：us)
    uint32_t maxStreamOpTriggerMilliSecond;     // 流操作的最大等待时间(单位: ms)
};
bool operator==(const OckAclAdapterConf &lhs, const OckAclAdapterConf &rhs);
bool operator!=(const OckAclAdapterConf &lhs, const OckAclAdapterConf &rhs);
std::ostream &operator<<(std::ostream &os, const OckAclAdapterConf &conf);

}  // namespace conf
}  // namespace ock
#endif