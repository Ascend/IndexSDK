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


#ifndef OCK_MEMORY_BRIDGE_HMM_MGR_MOCK_OCK_HMM_HMOBJECT_H
#define OCK_MEMORY_BRIDGE_HMM_MGR_MOCK_OCK_HMM_HMOBJECT_H
#include <gmock/gmock.h>
#include "ock/hmm/mgr/OckHmmHMObject.h"

namespace ock {
namespace hmm {

class MockOckHmmHMObject : public OckHmmHMObject {
public:
    MOCK_CONST_METHOD0(GetId, OckHmmHMOObjectID());
    MOCK_CONST_METHOD0(GetByteSize, uint64_t());
    MOCK_CONST_METHOD0(IntimateDeviceId, OckHmmDeviceId());
    MOCK_CONST_METHOD0(Addr, uintptr_t());
    MOCK_CONST_METHOD0(Location, OckHmmHeteroMemoryLocation());
    MOCK_METHOD4(
        GetBufferImpl, std::shared_ptr<OckHmmHMOBuffer>(OckHmmHeteroMemoryLocation, uint64_t, uint64_t, uint32_t));
    virtual std::shared_ptr<OckHmmHMOBuffer> GetBuffer(
        OckHmmHeteroMemoryLocation location, uint64_t offset = 0, uint64_t length = 0, uint32_t timeout = 0)
    {
        return this->GetBufferImpl(location, offset, length, timeout);
    }
    MOCK_METHOD3(GetBufferAsyncImpl,
        std::shared_ptr<OckHmmAsyncResult<OckHmmHMOBuffer>>(OckHmmHeteroMemoryLocation, uint64_t, uint64_t));
    virtual std::shared_ptr<OckHmmAsyncResult<OckHmmHMOBuffer>> GetBufferAsync(
        OckHmmHeteroMemoryLocation location, uint64_t offset = 0, uint64_t length = 0)
    {
        return this->GetBufferAsyncImpl(location, offset, length);
    }
    MOCK_METHOD1(ReleaseBuffer, void(std::shared_ptr<OckHmmHMOBuffer>));
};

}  // namespace hmm
}  // namespace ock
#endif