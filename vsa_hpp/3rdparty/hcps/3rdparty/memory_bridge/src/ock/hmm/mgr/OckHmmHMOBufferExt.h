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


#ifndef OCK_MEMORY_BRIDGE_HMM_HMOBJECT_BUFFER_EXT_H
#define OCK_MEMORY_BRIDGE_HMM_HMOBJECT_BUFFER_EXT_H
#include <cstdint>
#include <memory>
#include "ock/hmm/mgr/OckHmmHMObjectExt.h"
namespace ock {
namespace hmm {

class OckHmmHMOBufferExt : public OckHmmHMOBuffer {
public:
    virtual ~OckHmmHMOBufferExt() noexcept = default;
    virtual void ReleaseData(void) = 0;
    virtual OckHmmHMObjectExt &GetHmo(void) = 0;
    virtual bool Released(void) const = 0;

    /*
    @brief buffer不复用HMO的地址的场景
    */
    static std::shared_ptr<OckHmmHMOBufferExt> Create(OckHmmErrorCode errorCode,
        std::unique_ptr<OckHmmMemoryGuard> &&memoryGuard, uint64_t offset, OckHmmHMObjectExt &hmo);
    /*
    @brief buffer属于对应HMO中的一部分数据的场景
    */
    static std::shared_ptr<OckHmmHMOBufferExt> Create(uint64_t offset, uint64_t byteSize, OckHmmHMObjectExt &hmo);
};

class OckHmmHMOBufferOutter : public OckHmmHMOBuffer {
public:
    ~OckHmmHMOBufferOutter() noexcept = default;
    OckHmmHMOBufferOutter(std::shared_ptr<OckHmmHMOBufferExt> buffer);

    uintptr_t Address(void) const override;
    uint64_t Size(void) const override;
    uint64_t Offset(void) const override;
    OckHmmHeteroMemoryLocation Location(void) const override;
    OckHmmHMOObjectID GetId(void) const override;
    OckHmmErrorCode FlushData(void) override;
    OckHmmErrorCode ErrorCode(void) const override;

    OckHmmHMOBufferExt *Inner(void);
    void ReleaseData(void);

    static std::shared_ptr<OckHmmHMOBuffer> Create(std::shared_ptr<OckHmmHMOBufferExt> buffer);

private:
    std::shared_ptr<OckHmmHMOBufferExt> buffer;
};
}  // namespace hmm
}  // namespace ock
#endif