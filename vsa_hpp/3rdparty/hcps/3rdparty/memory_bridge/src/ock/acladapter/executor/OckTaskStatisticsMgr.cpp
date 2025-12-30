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

#include <mutex>
#include <list>
#include <unordered_map>
#include <functional>
#include "ock/utils/OckSafeUtils.h"
#include "ock/conf/OckAclAdapterConf.h"
#include "ock/log/OckLogger.h"
#include "ock/acladapter/executor/OckTaskStatisticsMgr.h"

namespace ock {
namespace acladapter {

namespace {
struct Statistics {
    std::chrono::steady_clock::time_point startTime{};
    std::chrono::steady_clock::time_point endTime{};
    uint64_t movedBytes {0};
    void AddData(const std::shared_ptr<OckHmmTrafficInfo>& data)
    {
        if (movedBytes == 0) {
            endTime = data->endTime;
            startTime = data->startTime;
        } else {
            endTime = std::max(endTime, data->endTime);
            startTime = std::min(startTime, data->startTime);
        }
        movedBytes += data->movedBytes;
    }
};
}

class OckTaskStatisticsMgrImpl : public OckTaskStatisticsMgr {
public:
    virtual ~OckTaskStatisticsMgrImpl() noexcept = default;
    explicit OckTaskStatisticsMgrImpl(void) = default;

    void AddTrafficInfo(std::shared_ptr<OckHmmTrafficInfo> info) override
    {
        const uint32_t maxReserveDataSize = 300UL;  // 内存中最多保留300条数据
        std::lock_guard<std::mutex> guard(dataMutex);
        if (trafficMgr.size() >= maxReserveDataSize) {
            trafficMgr.pop_front();
        }
        trafficMgr.push_back(info);
    }

    std::shared_ptr<hmm::OckHmmTrafficStatisticsInfo> PickUp(uint32_t maxGapMilliSeconds) override
    {
        // 该函数追求高性能，需要做性能测试，耗时应该控制在0.01毫秒内
        std::lock_guard<std::mutex> guard(dataMutex);

        auto h2dStatistics = Collect(
            [](const std::shared_ptr<OckHmmTrafficInfo> &data) {
                    return data->copyKind != OckMemoryCopyKind::HOST_TO_DEVICE;
            }, maxGapMilliSeconds);
        auto d2hStatistics = Collect(
            [](const std::shared_ptr<OckHmmTrafficInfo> &data) {
                    return data->copyKind != OckMemoryCopyKind::DEVICE_TO_HOST;
            }, maxGapMilliSeconds);

        return MakeResult(h2dStatistics, d2hStatistics);
    }

    std::shared_ptr<hmm::OckHmmTrafficStatisticsInfo> PickUp(
        hmm::OckHmmDeviceId deviceId, uint32_t maxGapMilliSeconds) override
    {
        // 该函数追求高性能，需要做性能测试，耗时应该控制在0.01毫秒内
        std::lock_guard<std::mutex> guard(dataMutex);

        auto h2dStatistics = Collect(
            [deviceId](const std::shared_ptr<OckHmmTrafficInfo> &data) {
                    return data->copyKind != OckMemoryCopyKind::HOST_TO_DEVICE || data->deviceId != deviceId;
            }, maxGapMilliSeconds);
        auto d2hStatistics = Collect(
            [deviceId](const std::shared_ptr<OckHmmTrafficInfo> &data) {
                    return data->copyKind != OckMemoryCopyKind::DEVICE_TO_HOST || data->deviceId != deviceId;
            }, maxGapMilliSeconds);

        return MakeResult(h2dStatistics, d2hStatistics);
    }

private:
    static uint64_t DurationInMicroseconds(
            std::chrono::steady_clock::time_point start, std::chrono::steady_clock::time_point end)
    {
        return std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    }

    static bool IsStop(const Statistics &statistics, const OckHmmTrafficInfo &curData, uint32_t maxGapMilliSeconds)
    {
        if (statistics.movedBytes == 0) {
            return false;
        }

        if (curData.endTime + std::chrono::milliseconds(maxGapMilliSeconds) < statistics.startTime) {
            OCK_HMM_LOG_DEBUG("Statistics is interrupted, statistics.startTime = "
                << statistics.startTime.time_since_epoch().count() << ", curData: " << curData);
            return true;
        } else {
            return false;
        }
    }

    std::shared_ptr<Statistics> Collect(
            const std::function<bool(const std::shared_ptr<OckHmmTrafficInfo> &)>& excludeFun,
            uint32_t maxGapMilliSeconds)
    {
        auto ret = std::make_shared<Statistics>();
        for (auto iter = trafficMgr.rbegin(); iter != trafficMgr.rend(); ++iter) {
            if (excludeFun(*iter)) {
                continue;
            }
            if (IsStop(*ret, *(*iter), maxGapMilliSeconds)) {
                break;
            }
            ret->AddData(*iter);
        }
        return ret;
    }

    static std::shared_ptr<hmm::OckHmmTrafficStatisticsInfo> MakeResult(
            const std::shared_ptr<Statistics> &h2dStatistics, const std::shared_ptr<Statistics> &d2hStatistics)
    {
        auto statInfo = std::make_shared<hmm::OckHmmTrafficStatisticsInfo>();
        auto h2dDuration = DurationInMicroseconds(h2dStatistics->startTime, h2dStatistics->endTime);
        auto d2hDuration = DurationInMicroseconds(d2hStatistics->startTime, d2hStatistics->endTime);
        if (h2dStatistics->movedBytes != 0) {
            statInfo->host2DeviceMovedBytes = h2dStatistics->movedBytes;
            if (h2dDuration == 0ULL) {
                statInfo->host2DeviceSpeed = std::numeric_limits<double>::max();
            } else {
                statInfo->host2DeviceSpeed = static_cast<double>(h2dStatistics->movedBytes) /
                    static_cast<double>(h2dDuration);
            }
        }
        if (d2hStatistics->movedBytes != 0) {
            statInfo->device2hostMovedBytes = d2hStatistics->movedBytes;
            if (d2hDuration == 0ULL) {
                statInfo->device2hostSpeed = std::numeric_limits<double>::max();
            } else {
                statInfo->device2hostSpeed = static_cast<double>(d2hStatistics->movedBytes) /
                    static_cast<double>(d2hDuration);
            }
        }
        return statInfo;
    }

    std::mutex dataMutex{};
    std::list<std::shared_ptr<OckHmmTrafficInfo>> trafficMgr{};
};
std::shared_ptr<OckTaskStatisticsMgr> OckTaskStatisticsMgr::Create(void)
{
    return std::make_shared<OckTaskStatisticsMgrImpl>();
}

}  // namespace acladapter
}  // namespace ock