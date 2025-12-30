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


#include "HmmAdaptor.h"
#include <unordered_map>
#include <ock/hmm/OckHmmFactory.h>
#include <ock/log/OckHmmLogHandler.h>
#include "hmm/AscendHMOImpl.h"
#include "common/utils/LogUtils.h"

using namespace ascend;
using namespace ock::hmm;

namespace {
    constexpr uint64_t MIN_SWAP_CAPACITY = 64 * 1024 * 1024;

class OckHmmLogHandlerAdptor : public ock::OckHmmLogHandler {
public:
    virtual ~OckHmmLogHandlerAdptor() = default;

    void Write(int32_t level, const char *levelStr, const char *fileName, uint64_t lineNo, const char *msg) override
    {
        aclAppLog(static_cast<aclLogLevel>(level), levelStr, fileName, lineNo, msg);
    }

    // 刷写数据，当异常发生时，会先调用Flush接口，然后抛异常，用户根据Write函数自定义
    void Flush(void)
    {
        return;
    }
};

}

std::shared_ptr<HmmIntf> HmmIntf::CreateHmm()
{
    return std::make_shared<HmmAdaptor>();
}

void HmmAdaptor::SetHmmLog() const
{
    ock::OckHmmSetLogHandler(std::make_shared<OckHmmLogHandlerAdptor>());
    ock::OckHmmSetLogLevel(ock::OCK_LOG_LEVEL_DEBUG);
}

APP_ERROR HmmAdaptor::Init(const HmmMemoryInfo &memoryInfo)
{
    SetHmmLog();

    auto factory = OckHmmFactory::Create();
    if (factory == nullptr) {
        APP_LOG_ERROR("OckHmmFactory::Create error!");
        return APP_ERR_INNER_ERROR;
    }

    std::shared_ptr<OckHmmDeviceInfo> deviceInfo = std::make_shared<OckHmmDeviceInfo>();
    deviceInfo->deviceId = memoryInfo.deviceId;
    deviceInfo->transferThreadNum = 2; // hmm数据搬运线程数，该值影响性能，当前实测设置为2性能最优
    deviceInfo->memorySpec.devSpec.maxDataCapacity = memoryInfo.deviceCapacity;
    deviceInfo->memorySpec.devSpec.maxSwapCapacity = memoryInfo.deviceBuffer;
    deviceInfo->memorySpec.hostSpec.maxDataCapacity = memoryInfo.hostCapacity;
    deviceInfo->memorySpec.hostSpec.maxSwapCapacity = MIN_SWAP_CAPACITY;
    CPU_ZERO(&(deviceInfo->cpuSet)); // 默认不绑核，这里不确定cpuSet初始化内部是否有随机值，因此调用接口
    auto retPair = factory->CreateSingleDeviceMemoryMgr(deviceInfo);
    if (retPair.first != HMM_SUCCESS) {
        APP_LOG_ERROR("CreateSingleDeviceMemoryMgr failed, ret[%d]!", retPair.first);
        return APP_ERR_INVALID_PARAM;
    }

    hmm = retPair.second;

    return APP_ERR_OK;
}

std::pair<APP_ERROR, std::shared_ptr<AscendHMO>> HmmAdaptor::CreateHmo(size_t size)
{
    if (hmm == nullptr) {
        APP_LOG_ERROR("hmm is nullptr!");
        return { APP_ERR_INNER_ERROR, nullptr };
    }

    auto ret = hmm->Alloc(size);
    if (ret.first != HMM_SUCCESS) {
        APP_LOG_ERROR("hmm Alloc failed, ret[%d]!", ret.first);
        return { APP_ERR_INNER_ERROR, nullptr };
    }

    auto hmo = std::make_shared<AscendHMOImpl>(hmm, ret.second);
    return { APP_ERR_OK, hmo };
}
