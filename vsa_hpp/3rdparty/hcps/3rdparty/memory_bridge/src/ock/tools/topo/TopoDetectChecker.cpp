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

#include <vector>
#include <chrono>
#include <mutex>
#include <unordered_map>
#include "ock/conf/OckSysConf.h"
#include "ock/tools/topo/TopoTestPackage.h"
#include "ock/acladapter/data/OckMemoryCopyKind.h"
#include "ock/acladapter/executor/OckAsyncTaskExecuteService.h"
#include "ock/acladapter/task/OckMemoryCopyTask.h"
#include "ock/acladapter/executor/OckAsyncResultInnerBridge.h"
#include "ock/log/OckLogger.h"
#include "ock/tools/topo/OckMemoryCopyTestFunOp.h"
#include "ock/tools/topo/TopoDetectChecker.h"
namespace ock {
namespace tools {
namespace topo {

class TopoDetectCheckerImpl : public TopoDetectChecker, public OckMemoryCopyTestResultReceiver {
public:
    virtual ~TopoDetectCheckerImpl() noexcept = default;
    using BridgeT = acladapter::OckAsyncResultInnerBridge<acladapter::OckDefaultResult>;
    using TaskVecT = std::vector<std::shared_ptr<acladapter::OckAsyncTaskBase>>;
    explicit TopoDetectCheckerImpl(std::shared_ptr<TopoDetectParam> parameter) : param(parameter)
    {}
    std::vector<TopoDetectResult> Check(void)
    {
        std::vector<acladapter::OckMemoryCopyKind> copyKindList({acladapter::OckMemoryCopyKind::HOST_TO_DEVICE,
            acladapter::OckMemoryCopyKind::HOST_TO_HOST,
            acladapter::OckMemoryCopyKind::DEVICE_TO_DEVICE,
            acladapter::OckMemoryCopyKind::DEVICE_TO_HOST});
        std::vector<TopoDetectResult> ret;
        for (auto kind : copyKindList) {
            if (static_cast<uint32_t>(kind) & param->TransferTypeBitValue()) {
                param->Kind(kind);
                BuildResultVec();
                CheckImpl();
                MergeTopoDetectResultVec(ret);
            }
        }
        return ret;
    }

private:
    void Notify(hmm::OckHmmDeviceId deviceId, uint64_t movedBytes) override
    {
        std::lock_guard<std::mutex> guard(notifyMutex);
        OCK_HMM_LOG_DEBUG("Notify deviceId[" << deviceId << "]movedBytes[" << movedBytes << "]");
        deviceResult.at(deviceIdPosMap.at(deviceId)).transferBytes += movedBytes;
    }
    void MergeTopoDetectResultVec(std::vector<TopoDetectResult> &ret)
    {
        for (auto &topoRet : deviceResult) {
            ret.push_back(topoRet);
        }
    }
    void CheckImpl(void)
    {
        std::vector<std::shared_ptr<acladapter::OckAsyncTaskExecuteService>> serviceVec;
        std::vector<std::shared_ptr<TopoTestPackage>> packageVec;
        auto initRet = BuildPackageAndService(serviceVec, packageVec);
        if (initRet != hmm::HMM_SUCCESS) {
            SetResultRetCode(initRet);
        } else {
            if (param->GetModel() == DetectModel::SERIAL) {
                CheckSerial(serviceVec, packageVec);
            } else if (param->GetModel() == DetectModel::PARALLEL) {
                CheckParallel(serviceVec, packageVec);
            }
        }
    }
    void StopAll(TaskVecT &allTaskVec)
    {
        for (auto &task : allTaskVec) {
            task->Cancel();
        }
    }
    void SetResultRetCode(hmm::OckHmmErrorCode initRet)
    {
        for (auto &result : deviceResult) {
            result.errorCode = initRet;
        }
    }
    void BuildResultVec(void)
    {
        deviceResult.clear();
        deviceIdPosMap.clear();
        for (auto &devInfo : param->GetDeviceInfo()) {
            deviceIdPosMap.insert(std::make_pair(devInfo.deviceId, deviceResult.size()));
            deviceResult.push_back(TopoDetectResult());
            deviceResult.back().deviceInfo = devInfo;
            deviceResult.back().copyKind = param->Kind();
        }
    }
    hmm::OckHmmErrorCode BuildPackageAndService(
        std::vector<std::shared_ptr<acladapter::OckAsyncTaskExecuteService>> &serviceVec,
        std::vector<std::shared_ptr<TopoTestPackage>> &packageVec)
    {
        for (auto &devInfo : param->GetDeviceInfo()) {
            serviceVec.push_back(acladapter::OckAsyncTaskExecuteService::Create(devInfo.deviceId,
                devInfo.GetCpuSet(),
                {{acladapter::OckTaskResourceType::HMM, param->ThreadNumPerDevice()}}));
            OCK_HMM_LOG_INFO("devicdId=" << devInfo.deviceId << " retcode=" << serviceVec.back()->StartErrorCode());
            if (serviceVec.back()->StartErrorCode() != hmm::HMM_SUCCESS) {
                return serviceVec.back()->StartErrorCode();
            }
            packageVec.push_back(TopoTestPackage::Create(*param, *serviceVec.back()));
            if (packageVec.back()->GetError() != hmm::HMM_SUCCESS) {
                return packageVec.back()->GetError();
            }
        }
        return hmm::HMM_SUCCESS;
    }
    void CheckSerial(std::vector<std::shared_ptr<acladapter::OckAsyncTaskExecuteService>> &serviceVec,
        std::vector<std::shared_ptr<TopoTestPackage>> &packageVec)
    {
        for (size_t pos = 0; pos < param->GetDeviceInfo().size(); ++pos) {
            CheckTransferSerial(*serviceVec.at(pos), packageVec.at(pos), deviceResult.at(pos));
        }
    }

    std::shared_ptr<TaskVecT> BuildParallelTask(
        std::vector<std::shared_ptr<acladapter::OckAsyncTaskExecuteService>> &serviceVec,
        std::vector<std::shared_ptr<TopoTestPackage>> &packageVec,
        std::vector<std::vector<std::shared_ptr<BridgeT>>> &devBridgeVec)
    {
        auto allTaskVec = std::make_shared<TaskVecT>();
        std::shared_ptr<std::vector<std::shared_ptr<acladapter::OckAsyncTaskBase>>> ret;
        for (uint32_t taskId = 0; taskId < param->ThreadNumPerDevice(); ++taskId) {
            for (uint32_t devPos = 0; devPos < param->GetDeviceInfo().size(); ++devPos) {
                auto pkg = packageVec.at(devPos)->GetPackage(taskId);

                OCK_HMM_LOG_DEBUG("BuildParallelTask devPos=" << devPos << " taskId=" << taskId);
                allTaskVec->push_back(AddTask(devBridgeVec.at(devPos), *(serviceVec.at(devPos)), pkg));
                OCK_HMM_LOG_DEBUG("BuildParallelTask devPos=" << devPos << " taskId=" << taskId << " complete");
            }
        }
        return allTaskVec;
    }
    void InitDevBridgeVec(const std::vector<std::shared_ptr<acladapter::OckAsyncTaskExecuteService>> &serviceVec,
        std::vector<std::vector<std::shared_ptr<BridgeT>>> &devBridgeVec)
    {
        for (uint32_t devPos = 0; devPos < param->GetDeviceInfo().size(); ++devPos) {
            devBridgeVec.push_back(std::vector<std::shared_ptr<BridgeT>>());
            devBridgeVec.back().reserve(param->TestTime());
        }
    }
    void BuildDevPosSet(const std::vector<std::vector<std::shared_ptr<TopoDetectCheckerImpl::BridgeT>>> &devBridgeVec,
        std::unordered_set<uint32_t> &devPosSet)
    {
        for (uint32_t devPos = 0; devPos < devBridgeVec.size(); ++devPos) {
            devPosSet.insert(devPos);
        }
    }
    void ClearCompletedDevice(std::unordered_set<uint32_t> &devPosSet,
        std::vector<std::vector<std::shared_ptr<TopoDetectCheckerImpl::BridgeT>>> &devBridgeVec,
        const std::chrono::steady_clock::time_point &startTime)
    {
        for (auto iter = devPosSet.begin(); iter != devPosSet.end();) {
            auto devPos = *iter;
            if (IsAllTaskComplete(devBridgeVec.at(devPos), deviceResult.at(devPos))) {
                auto end = std::chrono::steady_clock::now();
                auto usedMicroseconds = std::chrono::duration_cast<std::chrono::microseconds>(end - startTime).count();
                deviceResult.at(devPos).usedMicroseconds = usedMicroseconds;
                iter = devPosSet.erase(iter);
            } else {
                iter++;
            }
        }
    }
    void SetDeviceTimeOut(std::unordered_set<uint32_t> &devPosSet,
        std::vector<std::vector<std::shared_ptr<TopoDetectCheckerImpl::BridgeT>>> &devBridgeVec)
    {
        for (auto devPos : devPosSet) {
            deviceResult.at(devPos).errorCode = hmm::HMM_ERROR_WAIT_TIME_OUT;
        }
    }
    void WaitAllDeviceComplete(std::vector<std::vector<std::shared_ptr<TopoDetectCheckerImpl::BridgeT>>> &devBridgeVec,
        std::vector<std::shared_ptr<acladapter::OckAsyncTaskExecuteService>> &serviceVec,
        const std::chrono::steady_clock::time_point &startTime)
    {
        std::unordered_set<uint32_t> devPosSet;
        BuildDevPosSet(devBridgeVec, devPosSet);
        int32_t maxWaitTime = conf::OckSysConf::ToolConf().toolParrallMaxQueryTimes;
        while (!devPosSet.empty() && maxWaitTime > 0) {
            ClearCompletedDevice(devPosSet, devBridgeVec, startTime);
            std::this_thread::sleep_for(
                std::chrono::microseconds(conf::OckSysConf::ToolConf().toolParrallQueryIntervalMicroSecond));
            maxWaitTime--;
        }
        if (maxWaitTime <= 0) {
            OCK_HMM_LOG_WARN("Wait time out(device count=" << devPosSet.size() << ")");
            SetDeviceTimeOut(devPosSet, devBridgeVec);
        }
    }
    void CheckParallel(std::vector<std::shared_ptr<acladapter::OckAsyncTaskExecuteService>> &serviceVec,
        std::vector<std::shared_ptr<TopoTestPackage>> &packageVec)
    {
        std::vector<std::vector<std::shared_ptr<BridgeT>>> devBridgeVec;
        InitDevBridgeVec(serviceVec, devBridgeVec);

        auto start = std::chrono::steady_clock::now();
        auto allTaskVec = BuildParallelTask(serviceVec, packageVec, devBridgeVec);
        WaitAllDeviceComplete(devBridgeVec, serviceVec, start);
        StopAll(*allTaskVec);
    }
    std::shared_ptr<acladapter::OckAsyncTaskBase> AddTask(std::vector<std::shared_ptr<BridgeT>> &bridgeVec,
        acladapter::OckAsyncTaskExecuteService &service, const TopoTestPackage::Package &pkg)
    {
        auto bridge = std::make_shared<BridgeT>();
        bridgeVec.push_back(bridge);
        uint8_t *pSrc = pkg.deviceAddr;
        uint8_t *pDst = pkg.hostAddr;
        if (param->Kind() == acladapter::OckMemoryCopyKind::HOST_TO_DEVICE) {
            pSrc = pkg.hostAddr;
            pDst = pkg.deviceAddr;
        } else if (param->Kind() == acladapter::OckMemoryCopyKind::DEVICE_TO_HOST) {
            pSrc = pkg.deviceAddr;
            pDst = pkg.hostAddr;
        } else if (param->Kind() == acladapter::OckMemoryCopyKind::HOST_TO_HOST) {
            pSrc = pkg.hostAddr;
            pDst = pkg.hostDstAddr;
        } else {
            pSrc = pkg.deviceAddr;
            pDst = pkg.deviceDstAddr;
        }
        auto taskParam =
            std::make_shared<OckMemoryCopyTestParam>(pDst, pkg.packageSize, pSrc, pkg.packageSize, param->Kind());
        taskParam->TestTime(param->TestTime());
        taskParam->Receiver(this);
        auto task = OckMemoryCopyTestTask::Create(taskParam, bridge);
        service.AddTask(task);
        return task;
    }
    std::shared_ptr<TaskVecT> BuildMoveTask(const std::shared_ptr<TopoTestPackage> &package,
        acladapter::OckAsyncTaskExecuteService &service, std::vector<std::shared_ptr<BridgeT>> &bridgeVec)
    {
        auto taskVec = std::make_shared<TaskVecT>();
        for (uint32_t i = 0; i < param->ThreadNumPerDevice(); ++i) {
            taskVec->push_back(AddTask(bridgeVec, service, package->GetPackage(i)));
        }
        return taskVec;
    }
    bool IsAllTaskComplete(const std::vector<std::shared_ptr<BridgeT>> &bridgeVec, TopoDetectResult &result)
    {
        for (auto bridge : bridgeVec) {
            if (!bridge->IsRunComplete()) {
                return false;
            }
        }
        return true;
    }
    void WaitTaskComplete(const std::vector<std::shared_ptr<BridgeT>> &bridgeVec, TopoDetectResult &result)
    {
        for (auto bridge : bridgeVec) {
            auto ret = bridge->WaitResult();
            if (ret.get() == nullptr) {
                result.errorCode = hmm::HMM_ERROR_WAIT_TIME_OUT;
            } else {
                if (ret->ErrorCode()) {
                    result.errorCode = ret->ErrorCode();
                }
            }
        }
    }
    void CheckTransferSerial(acladapter::OckAsyncTaskExecuteService &service, std::shared_ptr<TopoTestPackage> package,
        TopoDetectResult &result)
    {
        std::vector<std::shared_ptr<BridgeT>> bridgeVec;
        bridgeVec.reserve(param->TestTime());
        auto start = std::chrono::steady_clock::now();
        auto taskVec = BuildMoveTask(package, service, bridgeVec);
        WaitTaskComplete(bridgeVec, result);
        auto end = std::chrono::steady_clock::now();
        result.usedMicroseconds = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        StopAll(*taskVec);
    }

private:
    std::unordered_map<hmm::OckHmmDeviceId, uint32_t> deviceIdPosMap{};
    std::shared_ptr<TopoDetectParam> param;
    std::vector<TopoDetectResult> deviceResult{};
    std::mutex notifyMutex{};
};

std::shared_ptr<TopoDetectChecker> TopoDetectChecker::Create(std::shared_ptr<TopoDetectParam> param)
{
    return std::make_shared<TopoDetectCheckerImpl>(param);
}
}  // namespace topo
}  // namespace tools
}  // namespace ock