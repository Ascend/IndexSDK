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


#ifndef OCK_MEMORY_BRIDGE_HMM_HMOBJECT_EXT_H
#define OCK_MEMORY_BRIDGE_HMM_HMOBJECT_EXT_H
#include <cstdint>
#include <memory>
#include "ock/hmm/mgr/OckHmmHMObject.h"
#include "ock/conf/OckSysConf.h"
#include "ock/utils/OckIdGenerator.h"
#include "ock/acladapter/executor/OckAsyncTaskExecuteService.h"
#include "ock/hmm/mgr/OckHmmSubMemoryAlloc.h"
namespace ock {
namespace hmm {

using HmoCsnGenerator = utils::OckIdGenerator<conf::MaxHMOIDNumberPerDevice>;
class OckHmmHMObjectExt;
class OckHmmSubMemoryAllocDispatcher {
public:
    virtual ~OckHmmSubMemoryAllocDispatcher() = default;
    virtual std::shared_ptr<OckHmmSubMemoryAlloc> DevSwapAlloc(void) = 0;
    virtual std::shared_ptr<OckHmmSubMemoryAlloc> HostSwapAlloc(void) = 0;
    virtual std::shared_ptr<OckHmmSubMemoryAlloc> SwapAlloc(OckHmmHeteroMemoryLocation location) = 0;
    virtual void InnerFree(std::shared_ptr<OckHmmHMObjectExt> hmo) = 0;
};
class OckHmmHMObjectExt : public OckHmmHMObject {
public:
    virtual ~OckHmmHMObjectExt() noexcept = default;

    /*
    @brief 强制释放内存，供管理者调用使用，不直接面向用户
    */
    virtual void ForceReleaseMemory(void) = 0;

    virtual std::shared_ptr<acladapter::OckAsyncTaskExecuteService> Service(void) = 0;

    virtual bool Released(void) const = 0;

    static std::shared_ptr<OckHmmHMObjectExt> Create(OckHmmHMOObjectID objectId,
        OckHmmSubMemoryAllocDispatcher &allocDispatcher,
        std::shared_ptr<acladapter::OckAsyncTaskExecuteService> service, std::shared_ptr<HmoCsnGenerator> csnGenerator,
        std::unique_ptr<OckHmmMemoryGuard> &&memoryGuard);
};
class OckHmmHMObjectOutter : public OckHmmHMObject {
public:
    virtual ~OckHmmHMObjectOutter() noexcept;
    explicit OckHmmHMObjectOutter(
        std::shared_ptr<OckHmmHMObjectExt> hmoExt, OckHmmSubMemoryAllocDispatcher &memAllocDispatcher);

    OckHmmHMOObjectID GetId(void) const override;
    OckHmmDeviceId IntimateDeviceId(void) const override;
    uint64_t GetByteSize(void) const override;
    OckHmmHeteroMemoryLocation Location(void) const override;
    std::shared_ptr<OckHmmHMOBuffer> GetBuffer(
        OckHmmHeteroMemoryLocation location, uint64_t offset = 0, uint64_t length = 0, uint32_t timeout = 0) override;
    std::shared_ptr<OckHmmAsyncResult<OckHmmHMOBuffer>> GetBufferAsync(
        OckHmmHeteroMemoryLocation location, uint64_t offset = 0, uint64_t length = 0) override;
    void ReleaseBuffer(std::shared_ptr<OckHmmHMOBuffer> buffer) override;
    uintptr_t Addr(void) const override;

    OckHmmHMObjectExt *GetExtHmo(void);

    static std::shared_ptr<OckHmmHMObject> Create(
        std::shared_ptr<OckHmmHMObjectExt> hmo, OckHmmSubMemoryAllocDispatcher &allocDispatcher);

private:
    std::shared_ptr<OckHmmHMObjectExt> hmo;
    OckHmmSubMemoryAllocDispatcher &allocDispatcher;
};
}  // namespace hmm
}  // namespace ock
#endif