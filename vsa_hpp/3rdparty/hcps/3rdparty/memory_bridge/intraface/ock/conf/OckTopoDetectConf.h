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

#ifndef OCK_MEMORY_BRIDGE_OCK_TOOL_DETECT_CONF_H
#define OCK_MEMORY_BRIDGE_OCK_TOOL_DETECT_CONF_H
#include <cstdint>
#include <ostream>
#include "ock/conf/ParamRange.h"

namespace ock {
namespace conf {

struct OckTopoDetectConf {
    OckTopoDetectConf(void);
    uint32_t toolParrallQueryIntervalMicroSecond;  // 最大查询间隔(单位: 微秒)
    uint32_t toolParrallMaxQueryTimes;             // 最大查询次数(单位: 次)
    uint32_t defaultThreadNumPerDevice;            // 默认每设备线程数
    uint32_t defaultTestTime;                      // 默认每设备测试时间(单位: 秒)
    uint64_t defaultPackageMBPerTransfer;          // 默认包大小(单位: MB)

    ParamRange<uint32_t> threadNumPerDevice;
    ParamRange<uint32_t> testTime;
    ParamRange<uint64_t> packageMBPerTransfer;
    ParamRange<uint64_t> packagePerTransfer;
};
bool operator==(const OckTopoDetectConf &lhs, const OckTopoDetectConf &rhs);
bool operator!=(const OckTopoDetectConf &lhs, const OckTopoDetectConf &rhs);
std::ostream &operator<<(std::ostream &os, const OckTopoDetectConf &conf);
}  // namespace conf
}  // namespace ock
#endif