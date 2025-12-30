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

#include <list>
#include <mutex>
#include <unordered_map>
#include "ock/log/OckHcpsLogger.h"
#include "ock/hmm/mgr/OckHmmErrorCode.h"
#include "ock/acladapter/task/OckDftFunOpTask.h"
#include "ock/acladapter/utils/OckSyncUtils.h"
#include "ock/hcps/stream/OckHeteroStreamBase.h"
#include "ock/hcps/stream/OckDevStreamMgr.h"
namespace ock {
namespace hcps {

OckDevStreamInfo MakeOckDevStreamInfo(acladapter::OckDevRtStream stream, OckDevStreamType streamType)
{
    OckDevStreamInfo ret;
    ret.stream = stream;
    ret.streamType = streamType;
    return ret;
}
class OckDevStreamMgrImpl : public OckDevStreamMgr {
public:
    virtual ~OckDevStreamMgrImpl() noexcept
    {}
    OckDevStreamMgrImpl(void)
    {}

    OckDevStreamInfo CreateStream(acladapter::OckAsyncTaskExecuteService &service, OckDevStreamType streamType,
        OckHcpsErrorCode &errorCode) override
    {
        if (errorCode != hmm::HMM_SUCCESS || streamType == OckDevStreamType::AI_NULL) {
            return MakeOckDevStreamInfo(nullptr, streamType);
        }
        std::lock_guard<std::mutex> guard(mutex);
        auto stream = FindStreamImpl(service, streamType);
        if (stream != nullptr) {
            return MakeOckDevStreamInfo(stream, streamType);
        }
        return MakeOckDevStreamInfo(CreateStreamImpl(service, streamType, errorCode), streamType);
    }
    void DestroyStreams(acladapter::OckAsyncTaskExecuteService &service) override
    {
        std::lock_guard<std::mutex> guard(mutex);
        auto itService = streamContainter.find(&service);
        if (itService == streamContainter.end()) {
            return;
        }
        acladapter::OckSyncUtils syncUtils(service);
        for (auto &node : itService->second) {
            syncUtils.DestroyStream(node.second);
        }
        streamContainter.erase(itService);
    }

private:
    acladapter::OckDevRtStream CreateStreamImpl(
        acladapter::OckAsyncTaskExecuteService &service, OckDevStreamType streamType, OckHcpsErrorCode &errorCode)
    {
        if (errorCode != hmm::HMM_SUCCESS) {
            return nullptr;
        }
        acladapter::OckSyncUtils syncUtils(service);
        acladapter::OckDevRtStream stream;
        errorCode = syncUtils.CreateStream(stream);
        if (errorCode != hmm::HMM_SUCCESS) {
            return nullptr;
        }
        streamContainter[&service][streamType] = stream;
        return stream;
    }
    acladapter::OckDevRtStream FindStreamImpl(
        acladapter::OckAsyncTaskExecuteService &service, OckDevStreamType streamType)
    {
        auto itService = streamContainter.find(&service);
        if (itService == streamContainter.end()) {
            return nullptr;
        }
        auto itStream = itService->second.find(streamType);
        if (itStream == itService->second.end()) {
            return nullptr;
        }
        return itStream->second;
    }
    std::mutex mutex{};
    std::unordered_map<acladapter::OckAsyncTaskExecuteService *,
        std::unordered_map<OckDevStreamType, acladapter::OckDevRtStream>>
        streamContainter{};
};
OckDevStreamMgr &OckDevStreamMgr::Instance(void)
{
    static OckDevStreamMgrImpl sIns;
    return sIns;
}
OckDevStreamMgr &sGlobalOckDevStreamMgrIns = OckDevStreamMgr::Instance();

std::ostream &operator<<(std::ostream &os, const OckDevStreamInfo &data)
{
    return os << "' streamType:" << data.streamType;
}
}  // namespace hcps
}  // namespace ock