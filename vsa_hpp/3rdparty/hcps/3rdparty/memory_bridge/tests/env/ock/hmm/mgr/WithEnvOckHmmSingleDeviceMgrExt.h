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


#ifndef OCK_MEMORY_BRIDGE_OCK_ADAPTER_TASK_WITH_ENV_OCK_HMM_SINGLE_DEVICE_MGR_EXT_H
#define OCK_MEMORY_BRIDGE_OCK_ADAPTER_TASK_WITH_ENV_OCK_HMM_SINGLE_DEVICE_MGR_EXT_H
#include <memory>
#include <thread>
#include <chrono>
#include <map>
#include "gtest/gtest.h"
#include "securec.h"
#include "acl/acl.h"
#include "mockcpp/mockcpp.hpp"
#include "ock/conf/OckSysConf.h"
#include "ock/hmm/mgr/OckHmmHeteroMemoryLocation.h"
#include "ock/acladapter/WithEnvOckAsyncTaskExecuteService.h"
#include "ock/hmm/mgr/WithEnvOckHmmSubMemoryAllocSet.h"
#include "ock/hmm/mgr/data/OckHmmHMOObjectIDGenerator.h"
namespace ock {
namespace hmm {
template <typename _BaseT>
class WithEnvOckHmmSingleDeviceMgrExt
    : public acladapter::WithEnvOckAsyncTaskExecuteService<WithEnvOckHmmSubMemoryAllocSet<_BaseT>> {
public:
    using BaseT = acladapter::WithEnvOckAsyncTaskExecuteService<WithEnvOckHmmSubMemoryAllocSet<_BaseT>>;
    void SetUp(void) override
    {
        BaseT::SetUp();
        MOCKER(aclInit).stubs().will(returnValue(ACL_SUCCESS));
        MOCKER(aclrtSetDevice).stubs().will(returnValue(ACL_SUCCESS));
        this->taskService = acladapter::OckAsyncTaskExecuteService::Create(
            this->deviceId, this->cpuSet, {{acladapter::OckTaskResourceType::HMM, this->transferThreadNum}});
        this->deviceInfo = std::make_shared<OckHmmDeviceInfo>();
        this->deviceInfo->deviceId = this->deviceId;
        this->deviceInfo->cpuSet = this->cpuSet;
        this->deviceInfo->transferThreadNum = this->transferThreadNum;
        this->mgr = ext::CreateSingleDeviceMgr(deviceInfo, this->taskService, this->allocSet);
    }
    void TearDown(void) override
    {
        mgr.reset();
        deviceInfo.reset();
        BaseT::TearDown();
    }
    std::map<OckHmmMemoryAllocatePolicy, OckHmmHeteroMemoryLocation> BuildPolicyLocationMap()
    {
        std::map<OckHmmMemoryAllocatePolicy, OckHmmHeteroMemoryLocation> policyLocationMap;
        policyLocationMap[OckHmmMemoryAllocatePolicy::DEVICE_DDR_ONLY] = OckHmmHeteroMemoryLocation::DEVICE_DDR;
        policyLocationMap[OckHmmMemoryAllocatePolicy::LOCAL_HOST_ONLY] = OckHmmHeteroMemoryLocation::LOCAL_HOST_MEMORY;
        policyLocationMap[OckHmmMemoryAllocatePolicy::DEVICE_DDR_FIRST] = OckHmmHeteroMemoryLocation::DEVICE_DDR;
        return policyLocationMap;
    }
    std::shared_ptr<OckHmmHMObject> AllocHMO(
        MockOckHmmSubMemoryAlloc &alloc, const uint32_t hmoBytes, OckHmmMemoryAllocatePolicy policy)
    {
        this->MockAllocFreeWithNewDelete(alloc);
        auto hmoRet = this->mgr->Alloc(hmoBytes, policy);
        return hmoRet.second;
    }

    std::shared_ptr<OckHmmSingleDeviceMgr> mgr;
    std::shared_ptr<OckHmmDeviceInfo> deviceInfo;
};
}  // namespace hmm
}  // namespace ock
#endif