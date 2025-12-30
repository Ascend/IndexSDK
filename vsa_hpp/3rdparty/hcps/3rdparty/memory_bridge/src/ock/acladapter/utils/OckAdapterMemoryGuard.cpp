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


#include "ock/acladapter/utils/OckAdapterMemoryGuard.h"
#include "ock/acladapter/utils/OckSyncUtils.h"
#include "ock/conf/OckSysConf.h"
namespace ock {
namespace acladapter {

class OckAdapterMemoryGuardImpl : public OckAdapterMemoryGuard {
public:
    ~OckAdapterMemoryGuardImpl() noexcept override
    {
        if (useGuard && guardAddr != nullptr) {
            OckSyncUtils syncUtils(service);
            syncUtils.Free(guardAddr, location, conf::OckSysConf::AclAdapterConf().maxFreeWaitMilliSecondThreshold);
        }
    }
    OckAdapterMemoryGuardImpl(OckAsyncTaskExecuteService &taskService, uint8_t *guardAddress, uint64_t byteCount,
        hmm::OckHmmHeteroMemoryLocation memoryLocation)
        : useGuard(true), guardAddr(guardAddress), byteSize(byteCount), service(taskService), location(memoryLocation)
    {}

    OckAdapterMemoryGuardImpl(const OckAdapterMemoryGuardImpl &) = delete;
    OckAdapterMemoryGuardImpl &operator=(const OckAdapterMemoryGuardImpl &) = delete;

    uint8_t *ReleaseGuard(void) override
    {
        useGuard = false;
        return guardAddr;
    }
    uint8_t *GetAddr(void) override
    {
        return guardAddr;
    }
    uint64_t ByteSize(void) const override
    {
        return byteSize;
    }
    uintptr_t Addr(void) const override
    {
        return (uintptr_t)guardAddr;
    }
    hmm::OckHmmHeteroMemoryLocation Location(void) const override
    {
        return location;
    }

private:
    bool useGuard;
    uint8_t *guardAddr;
    uint64_t byteSize;
    OckAsyncTaskExecuteService &service;
    hmm::OckHmmHeteroMemoryLocation location;
};
std::unique_ptr<OckAdapterMemoryGuard> OckAdapterMemoryGuard::Create(OckAsyncTaskExecuteService &service,
    uint8_t *guardAddr, uint64_t byteSize, hmm::OckHmmHeteroMemoryLocation location)
{
    return std::make_unique<OckAdapterMemoryGuardImpl>(service, guardAddr, byteSize, location);
}
std::ostream &operator<<(std::ostream &os, const OckAdapterMemoryGuard &data)
{
    return os << "{'bytesize':" << data.ByteSize() << ",'location':" << data.Location() << "}";
}
}  // namespace acladapter
}  // namespace ock