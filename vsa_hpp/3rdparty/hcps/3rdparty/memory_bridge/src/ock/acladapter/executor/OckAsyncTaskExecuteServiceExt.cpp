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


#include <memory>
#include <list>
#include <vector>
#include <mutex>
#include "acl/acl.h"
#include "ock/log/OckLogger.h"
#include "ock/utils/OckThreadUtils.h"
#include "ock/acladapter/executor/OckTaskRunThreadExt.h"
#include "ock/acladapter/executor/OckAsyncTaskExecuteService.h"
namespace ock {
namespace acladapter {
OckDeviceStreamInfo::OckDeviceStreamInfo(hmm::OckHmmDeviceId deviceIdx, OckDevRtStream devStream)
    : deviceId(deviceIdx), stream(devStream)
{}
class OckAsyncTaskExecuteServiceImpl : public OckAsyncTaskExecuteServiceExt {
public:
    using TaskQueueContainer = std::list<std::shared_ptr<OckAsyncTaskBase>>;
    virtual ~OckAsyncTaskExecuteServiceImpl() noexcept
    {
        Stop();
    }
    explicit OckAsyncTaskExecuteServiceImpl(
        hmm::OckHmmDeviceId deviceIdx, const cpu_set_t &cpuSets, const OckTaskThreadNumberMap &taskThreadMap)
        : deviceId(deviceIdx), startErrorCode(hmm::HMM_SUCCESS), cpuSet(cpuSets), taskThreadNumberMap(taskThreadMap),
          taskStatisticsMgr(OckTaskStatisticsMgr::Create())
    {
        auto cpuSetVec = utils::DispatchCpuSet(cpuSet, CalcThreadNumber(taskThreadMap));
        uint32_t curThdId = 0UL;
        for (auto &taskThreadNum : taskThreadMap) {
            AddThread(taskThreadNum.first, taskThreadNum.second, cpuSetVec, curThdId);
            curThdId += taskThreadNum.second;
        }
    }

    explicit OckAsyncTaskExecuteServiceImpl(hmm::OckHmmDeviceId deviceIdx, std::vector<cpu_set_t> &cpuSets,
        const OckTaskThreadNumberMap &taskThreadMap)
        : deviceId(deviceIdx),
          startErrorCode(hmm::HMM_SUCCESS),
          taskThreadNumberMap(taskThreadMap),
          taskStatisticsMgr(OckTaskStatisticsMgr::Create())
    {
        cpuSet = utils::CombineCpuSets(cpuSets);
        uint32_t curPos = 0UL;
        for (auto &taskThreadNum : taskThreadMap) {
            auto cpuSetVec = utils::DispatchCpuSet(cpuSets[curPos], taskThreadNum.second);
            AddThread(taskThreadNum.first, taskThreadNum.second, cpuSetVec, 0UL);
            curPos++;
        }
    }

    explicit OckAsyncTaskExecuteServiceImpl(std::vector<std::shared_ptr<OckTaskRunThreadBase>> threadList)
        : startErrorCode(hmm::HMM_SUCCESS), taskStatisticsMgr(OckTaskStatisticsMgr::Create())
    {
        if (threadList.size() != 0) {
            deviceId = threadList[0]->DeviceId();
        }
        for (uint32_t i = 0; i < threadPool.size(); ++i) {
            threadPool.push_back(threadList[i]);
            auto errorCode = threadPool.back()->Start();
            if (errorCode != hmm::HMM_SUCCESS) {
                startErrorCode = errorCode;
                break;
            }
        }
    }

    std::shared_ptr<OckAsyncTaskBase> PickUp(OckTaskResourceType resourceType) override
    {
        std::lock_guard<std::mutex> lock(queueMutex);
        for (TaskQueueContainer::iterator iter = taskQueue.begin(); iter != taskQueue.end(); ++iter) {
            if (RhsInLhs(resourceType, (*iter)->ResourceType()) && (*iter)->PreConditionMet()) {
                auto ret = *iter;
                iter = taskQueue.erase(iter);
                return ret;
            }
        }
        return std::shared_ptr<OckAsyncTaskBase>();
    }
    void AddTask(std::shared_ptr<OckAsyncTaskBase> task) override
    {
        OCK_HMM_LOG_DEBUG("AddTask resourceType=" << task->ResourceType());
        AddTaskIntoQueue(task);
    }
    void CancelAll(void) override
    {
        CancelExistsTask();
        NotifySubThreadStop();
    }
    void Stop(void) override
    {
        CancelAll();
        StopAll();
    }

    bool AllStarted(void) const override
    {
        for (auto &thr : threadPool) {
            if (!thr->Active()) {
                return false;
            }
        }
        return true;
    }
    uint32_t ThreadCount(void) const override
    {
        return static_cast<uint32_t>(threadPool.size());
    }
    uint32_t ActiveThreadCount(void) const override
    {
        uint32_t ret = 0;
        for (auto iter = threadPool.begin(); iter != threadPool.end(); ++iter) {
            if ((*iter)->Active()) {
                ret++;
            }
        }
        return ret;
    }
    uint32_t TaskCount(void) const override
    {
        std::lock_guard<std::mutex> lock(queueMutex);
        return static_cast<uint32_t>(taskQueue.size());
    }
    hmm::OckHmmErrorCode StartErrorCode(void) const override
    {
        return startErrorCode;
    }

    std::shared_ptr<OckTaskStatisticsMgr> TaskStatisticsMgr(void) override
    {
        return taskStatisticsMgr;
    }
    hmm::OckHmmDeviceId GetDeviceId(void) const override
    {
        return deviceId;
    }
    const cpu_set_t &GetCpuSet(void) const override
    {
        return cpuSet;
    }
    const OckTaskThreadNumberMap &TaskThreadNumberMap(void) const override
    {
        return taskThreadNumberMap;
    }

private:
    uint32_t CalcThreadNumber(const OckTaskThreadNumberMap &taskThreadMap) const
    {
        uint32_t ret = 0;
        for (auto &value : taskThreadMap) {
            ret += value.second;
        }
        return ret;
    }
    void NotifySubThreadAdded()
    {
        for (auto &thd : threadPool) {
            thd->NotifyAdded();
        }
    }
    void AddTaskIntoQueue(const std::shared_ptr<ock::acladapter::OckAsyncTaskBase> &task)
    {
        std::lock_guard<std::mutex> lock(queueMutex);
        taskQueue.push_back(task);
    }
    void NotifySubThreadStop()
    {
        for (auto iter = threadPool.begin(); iter != threadPool.end(); ++iter) {
            (*iter)->NotifyStop();
        }
    }
    void CancelExistsTask()
    {
        std::lock_guard<std::mutex> lock(queueMutex);
        for (auto &task : taskQueue) {
            task->Cancel();
        }
        taskQueue.clear();
    }
    void AddThread(
        OckTaskResourceType resourceType, uint32_t maxThreadNum, std::vector<cpu_set_t> &cpuSetVec, uint32_t curThdId)
    {
        const uint32_t maxThreadNumThreshold = 128U;
        if (maxThreadNum > maxThreadNumThreshold) {
            maxThreadNum = maxThreadNumThreshold;
        }
        for (uint32_t i = 0; i < maxThreadNum; ++i) {
            threadPool.push_back(CreateRunThread(deviceId, cpuSetVec[curThdId + i], *this, resourceType));
            auto errorCode = threadPool.back()->Start();
            if (errorCode != hmm::HMM_SUCCESS) {
                startErrorCode = errorCode;
                break;
            }
        }
    }
    void StopAll()
    {
        for (auto iter = threadPool.begin(); iter != threadPool.end(); ++iter) {
            (*iter)->Stop();
        }
    }

    mutable std::mutex queueMutex{};
    TaskQueueContainer taskQueue{};
    hmm::OckHmmDeviceId deviceId{ 0 };
    hmm::OckHmmErrorCode startErrorCode;
    cpu_set_t cpuSet{};
    OckTaskThreadNumberMap taskThreadNumberMap{};
    std::vector<std::shared_ptr<OckTaskRunThreadBase>> threadPool{};
    std::shared_ptr<OckTaskStatisticsMgr> taskStatisticsMgr{ nullptr };
};
std::shared_ptr<OckAsyncTaskExecuteService> OckAsyncTaskExecuteService::Create(
    hmm::OckHmmDeviceId deviceId, const cpu_set_t &cpuSet, const OckTaskThreadNumberMap &taskThreadMap)
{
    if (CalcRelatedThreadNumber(taskThreadMap, OckTaskResourceType::DEVICE_STREAM) > 1U) {
        OCK_HMM_LOG_ERROR("StreamTask too many stream thread." << taskThreadMap);
        return std::shared_ptr<OckAsyncTaskExecuteService>();
    }
    return std::make_shared<OckAsyncTaskExecuteServiceImpl>(deviceId, cpuSet, taskThreadMap);
}

std::shared_ptr<OckAsyncTaskExecuteService> OckAsyncTaskExecuteService::Create(hmm::OckHmmDeviceId deviceId,
    std::vector<cpu_set_t> &cpuSets, const OckTaskThreadNumberMap &taskThreadMap)
{
    if (CalcRelatedThreadNumber(taskThreadMap, OckTaskResourceType::DEVICE_STREAM) > 1U) {
        OCK_HMM_LOG_ERROR("StreamTask too many stream thread." << taskThreadMap);
        return std::shared_ptr<OckAsyncTaskExecuteService>();
    }
    if (cpuSets.size() < taskThreadMap.size()) {
        OCK_HMM_LOG_ERROR("the size of cpuSets(" << cpuSets.size() << ") cannot less than the size of taskThreadMap(" <<
            taskThreadMap.size() << ")");
        return std::shared_ptr<OckAsyncTaskExecuteService>();
    }
    return std::make_shared<OckAsyncTaskExecuteServiceImpl>(deviceId, cpuSets, taskThreadMap);
}

std::shared_ptr<OckAsyncTaskExecuteService> OckAsyncTaskExecuteService::Create(
    std::vector<std::shared_ptr<OckTaskRunThreadBase>> threadList)
{
    return std::make_shared<OckAsyncTaskExecuteServiceImpl>(threadList);
}
std::ostream &operator<<(std::ostream &os, const OckDeviceStreamInfo &streamInfo)
{
    return os << "{'deviceId':" << streamInfo.deviceId << "}";
}
std::ostream &operator<<(std::ostream &os, const OckAsyncTaskExecuteServiceExt &service)
{
    return os << "{'TaskCount':" << service.TaskCount() << ",'ActiveThreadCount':" << service.ActiveThreadCount()
              << ",'AllStarted':" << service.AllStarted() << ",'StartErrorCode':" << service.StartErrorCode()
              << ",'DeviceId':" << service.GetDeviceId() << ",'cpuSet':" << service.GetCpuSet() << "}";
}
std::ostream &operator<<(std::ostream &os, const OckAsyncTaskExecuteService &service)
{
    return os << "{'TaskCount':" << service.TaskCount() << ",'ActiveThreadCount':" << service.ActiveThreadCount()
              << ",'AllStarted':" << service.AllStarted() << ",'StartErrorCode':" << service.StartErrorCode() << "}";
}
}  // namespace acladapter
}  // namespace ock