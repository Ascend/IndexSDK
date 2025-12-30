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

#include <syslog.h>
#include <string>
#include "ock/hmm/mgr/OckHmmMgrCreator.h"
#include "ock/hmm/mgr/checker/OckHmmCreatorParamCheck.h"
#include "ock/hmm/OckHmmFactory.h"
#include "ock/log/OckLogger.h"
#include "ock/utils/StrUtils.h"

namespace ock {
namespace hmm {

class OckHmmFactoryImpl : public OckHmmFactory {
public:
    ~OckHmmFactoryImpl() noexcept override = default;
    explicit OckHmmFactoryImpl(void) = default;
    std::pair<OckHmmErrorCode, std::shared_ptr<OckHmmSingleDeviceMgr>> CreateSingleDeviceMemoryMgr(
        std::shared_ptr<OckHmmDeviceInfo> deviceInfo, uint32_t timeout) override
    {
        openlog("SingleDeviceMgr", LOG_CONS | LOG_PID, LOG_USER);
        if (deviceInfo.get() == nullptr) {
            OCK_HMM_LOG_ERROR("deviceInfo is nullptr!");
            syslog(LOG_INFO, "Failed to create OckHmmSingleDeviceMgr. deviceInfo is nullptr.\n");
            closelog();
            return std::make_pair(HMM_ERROR_INPUT_PARAM_EMPTY, std::shared_ptr<OckHmmSingleDeviceMgr>());
        }
        std::string info = utils::ToString(*deviceInfo);
        auto checkRet = OckHmmCreatorParamCheck::CheckParam(*deviceInfo);
        if (checkRet != HMM_SUCCESS) {
            syslog(LOG_INFO, "Failed to create OckHmmSingleDeviceMgr. deviceInfo = %s\n", info.c_str());
            closelog();
            return std::make_pair(checkRet, std::shared_ptr<OckHmmSingleDeviceMgr>());
        }
        auto ret = OckHmmMgrCreator::Create(*deviceInfo, timeout);
        if (ret.first == HMM_SUCCESS) {
            syslog(LOG_INFO, "Success to create OckHmmSingleDeviceMgr. deviceInfo = %s\n", info.c_str());
            closelog();
        } else {
            syslog(LOG_INFO, "Failed to create OckHmmSingleDeviceMgr. deviceInfo = %s\n", info.c_str());
            closelog();
        }
        return ret;
    }
    std::pair<OckHmmErrorCode, std::shared_ptr<OckHmmComposeDeviceMgr>> CreateComposeMemoryMgr(
        std::shared_ptr<OckHmmDeviceInfoVec> deviceInfoVec, uint32_t timeout) override
    {
        openlog("ComposeDeviceMgr", LOG_CONS | LOG_PID, LOG_USER);
        if (deviceInfoVec.get() == nullptr) {
            OCK_HMM_LOG_ERROR("deviceInfoVec is a nullptr!");
            syslog(LOG_INFO, "Failed to create OckHmmComposeDeviceMgr. deviceInfoVec is nullptr.\n");
            closelog();
            return std::make_pair(HMM_ERROR_INPUT_PARAM_EMPTY, std::shared_ptr<OckHmmComposeDeviceMgr>());
        }
        std::string info = utils::ToString(*deviceInfoVec);
        auto checkRet = OckHmmCreatorParamCheck::CheckParam(*deviceInfoVec);
        if (checkRet != HMM_SUCCESS) {
            syslog(LOG_INFO, "Failed to create OckHmmComposeDeviceMgr. deviceInfoVec = %s\n", info.c_str());
            closelog();
            return std::make_pair(checkRet, std::shared_ptr<OckHmmComposeDeviceMgr>());
        }
        auto ret = OckHmmMgrCreator::Create(deviceInfoVec, timeout);
        if (ret.first == HMM_SUCCESS) {
            syslog(LOG_INFO, "Success to create OckHmmComposeDeviceMgr. deviceInfo = %s\n", info.c_str());
            closelog();
        } else {
            syslog(LOG_INFO, "Failed to create OckHmmComposeDeviceMgr. deviceInfo = %s\n", info.c_str());
            closelog();
        }
        return ret;
    }
    std::pair<OckHmmErrorCode, std::shared_ptr<OckHmmShareDeviceMgr>> CreateShareMemoryMgr(
        std::shared_ptr<OckHmmPureDeviceInfoVec> deviceInfoVec,
        std::shared_ptr<OckHmmMemoryCapacitySpecification> hostSpec) override
    {
        return std::make_pair(HMM_ERROR_INPUT_PARAM_NOT_SUPPORT_SUCH_OP, std::shared_ptr<OckHmmShareDeviceMgr>());
    }
};
std::shared_ptr<OckHmmFactory> OckHmmFactory::Create(void)
{
    return std::make_shared<OckHmmFactoryImpl>();
}
}  // namespace hmm
}  // namespace ock