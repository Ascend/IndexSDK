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


#ifndef OCK_MEMORY_BRIDGE_OCK_ADAPTER_OCK_TASK_STATISTICS_MGR_H
#define OCK_MEMORY_BRIDGE_OCK_ADAPTER_OCK_TASK_STATISTICS_MGR_H
#include <memory>
#include <ostream>
#include <chrono>
#include "ock/hmm/mgr/OckHmmHMObject.h"
#include "ock/hmm/mgr/OckHmmTrafficStatisticsInfo.h"
#include "ock/acladapter/data/OckHmmTrafficInfo.h"
namespace ock {
namespace acladapter {

class OckTaskStatisticsMgr {
public:
    virtual ~OckTaskStatisticsMgr() noexcept = default;

    virtual void AddTrafficInfo(std::shared_ptr<OckHmmTrafficInfo> info) = 0;
    virtual std::shared_ptr<hmm::OckHmmTrafficStatisticsInfo> PickUp(uint32_t maxGapMilliSeconds) = 0;
    virtual std::shared_ptr<hmm::OckHmmTrafficStatisticsInfo> PickUp(
        hmm::OckHmmDeviceId deviceId, uint32_t maxGapMilliSeconds) = 0;

    static std::shared_ptr<OckTaskStatisticsMgr> Create(void);
};

}  // namespace acladapter
}  // namespace ock
#endif