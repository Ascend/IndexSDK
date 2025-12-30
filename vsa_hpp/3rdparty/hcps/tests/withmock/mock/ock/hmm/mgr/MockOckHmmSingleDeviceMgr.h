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


#ifndef OCK_MEMORY_BRIDGE_HMM_MGR_MOCK_OCK_HMM_HETERO_SINGLE_DEVICE_MGR_H
#define OCK_MEMORY_BRIDGE_HMM_MGR_MOCK_OCK_HMM_HETERO_SINGLE_DEVICE_MGR_H
#include <gmock/gmock.h>
#include "ock/hmm/mgr/OckHmmHeteroMemoryMgr.h"

namespace ock {
namespace hmm {

class MockOckHmmSingleDeviceMgr : public OckHmmSingleDeviceMgr {
public:
    MOCK_METHOD2(Malloc, std::unique_ptr<OckHmmMemoryGuard>(uint64_t, OckHmmMemoryAllocatePolicy));
    MOCK_METHOD2(
        AllocImpl, std::pair<OckHmmErrorCode, std::shared_ptr<OckHmmHMObject>>(uint64_t, OckHmmMemoryAllocatePolicy));
    virtual std::pair<OckHmmErrorCode, std::shared_ptr<OckHmmHMObject>> Alloc(
        uint64_t hmoBytes, OckHmmMemoryAllocatePolicy policy = OckHmmMemoryAllocatePolicy::DEVICE_DDR_FIRST)
    {
        return this->AllocImpl(hmoBytes, policy);
    }
    MOCK_METHOD1(Free, void(std::shared_ptr<OckHmmHMObject>));
    MOCK_METHOD5(CopyHMO, OckHmmErrorCode(OckHmmHMObject &, uint64_t, OckHmmHMObject &, uint64_t, size_t));
    MOCK_CONST_METHOD1(GetUsedInfo, std::shared_ptr<OckHmmResourceUsedInfo>(uint64_t));
    MOCK_METHOD1(GetTrafficStatisticsInfoImpl, std::shared_ptr<OckHmmTrafficStatisticsInfo>(uint32_t));
    virtual std::shared_ptr<OckHmmTrafficStatisticsInfo> GetTrafficStatisticsInfo(uint32_t maxGapMilliSeconds = 10)
    {
        return this->GetTrafficStatisticsInfoImpl(maxGapMilliSeconds);
    }
    MOCK_CONST_METHOD0(GetSpecific, const OckHmmMemorySpecification &());
    MOCK_CONST_METHOD0(GetCpuSet, const cpu_set_t &());
    MOCK_CONST_METHOD0(GetDeviceId, OckHmmDeviceId());
    MOCK_METHOD1(AllocateHost, uint8_t *(size_t));
    MOCK_METHOD2(DeallocateHost, void(uint8_t *, size_t));
    MOCK_METHOD2(IncBindMemory, OckHmmErrorCode(OckHmmHeteroMemoryLocation allocType, uint64_t byteSize));
    MOCK_METHOD3(IncBindMemory,
        OckHmmErrorCode(OckHmmHeteroMemoryLocation allocType, uint64_t byteSize, uint32_t timeout));
};

}  // namespace hmm
}  // namespace ock
#endif