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


#ifndef OCK_MEMORY_BRIDGE_HMOBJECT_H
#define OCK_MEMORY_BRIDGE_HMOBJECT_H
#include <cstdint>
#include <memory>
#include <vector>
#include "ock/hmm/mgr/OckHmmAsyncResult.h"
#include "ock/hmm/mgr/OckHmmHeteroMemoryLocation.h"
namespace ock {
namespace hmm {
using OckHmmHMOObjectID = uint64_t;
using OckHmmDeviceId = uint16_t;
class OckHmmHMOBufferBase {
public:
    virtual ~OckHmmHMOBufferBase() noexcept = default;
    /*
    @brief 该Buffer地址
    */
    virtual uintptr_t Address(void) const = 0;
    /*
    @brief 该Buffer大小
    */
    virtual uint64_t Size(void) const = 0;
};
std::ostream &operator<<(std::ostream &os, const OckHmmHMOBufferBase &buffer);
class OckHmmHMOBuffer : public OckHmmResultBase, public OckHmmHMOBufferBase {
public:
    virtual ~OckHmmHMOBuffer() noexcept = default;
    /*
    @brief 该Buffer相对对应HMO的Offset位置
    */
    virtual uint64_t Offset(void) const = 0;
    /*
    @brief 该Buffer的位置类型
    */
    virtual OckHmmHeteroMemoryLocation Location(void) const = 0;
    /*
    @brief 该Buffer所属的HMO对象ID
    */
    virtual OckHmmHMOObjectID GetId(void) const = 0;
    /*
    @brief 将buffer空间的数据刷写到HMO(前提是HMO没有被释放)
    */
    virtual OckHmmErrorCode FlushData(void) = 0;
};
class OckHmmSubHMObject {
public:
    virtual ~OckHmmSubHMObject() noexcept = default;
    virtual uint64_t GetByteSize(void) const = 0;
    virtual uintptr_t Addr(void) const = 0;
    /*
    @brief HMO的原始位置
    */
    virtual OckHmmHeteroMemoryLocation Location(void) const = 0;
    /*
    @brief 获取Buffer数据
    @offset 数据的起始位置
    @length 获取的数据长度
    @result 返回Buffer数据对象，当入参异常时，返回空， 当返回数据不为空时，可以根据HMOBuffer.ErrorCode看详细错误情况
    */
    virtual std::shared_ptr<OckHmmHMOBuffer> GetBuffer(
        OckHmmHeteroMemoryLocation location, uint64_t offset = 0, uint64_t length = 0, uint32_t timeout = 0) = 0;

    /*
    @brief 获取Buffer数据
    @offset 数据的起始位置
    @length 获取的数据长度
    @result 返回AsyncResult数据对象，当入参异常时，返回空
    */
    virtual std::shared_ptr<OckHmmAsyncResult<OckHmmHMOBuffer>> GetBufferAsync(
        OckHmmHeteroMemoryLocation location, uint64_t offset = 0, uint64_t length = 0) = 0;

    virtual void ReleaseBuffer(std::shared_ptr<OckHmmHMOBuffer> buffer) = 0;

    static std::shared_ptr<OckHmmSubHMObject> CreateSubHmo(
        std::shared_ptr<OckHmmSubHMObject> subHmoObject, uint64_t offset, uint64_t length);
};
class OckHmmHMObject : public OckHmmSubHMObject {
public:
    virtual ~OckHmmHMObject() noexcept = default;

    virtual OckHmmHMOObjectID GetId(void) const = 0;
    virtual OckHmmDeviceId IntimateDeviceId(void) const = 0;
    static std::shared_ptr<std::vector<std::shared_ptr<OckHmmSubHMObject>>> CreateSubHmoList(
        std::shared_ptr<OckHmmHMObject> hmo, uint64_t subHmoBytes);
    static std::shared_ptr<OckHmmSubHMObject> CreateSubHmo(
        std::shared_ptr<OckHmmHMObject> hmoObject, uint64_t offset, uint64_t length);
};

std::ostream &operator<<(std::ostream &os, const OckHmmSubHMObject &hmo);
std::ostream &operator<<(std::ostream &os, const OckHmmHMObject &hmo);
std::ostream &operator<<(std::ostream &os, const OckHmmHMOBuffer &buffer);
std::ostream &operator<<(std::ostream &os, const std::shared_ptr<OckHmmSubHMObject> &hmo);
std::ostream &operator<<(std::ostream &os, const std::shared_ptr<OckHmmHMObject> &hmo);
std::ostream &operator<<(std::ostream &os, const std::shared_ptr<OckHmmHMOBuffer> &buffer);
}  // namespace hmm
}  // namespace ock
#endif