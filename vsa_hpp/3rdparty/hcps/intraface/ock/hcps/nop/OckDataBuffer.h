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

#ifndef OCK_HCPS_OCK_DATA_BUFFER_H
#define OCK_HCPS_OCK_DATA_BUFFER_H
#include <vector>
#include <numeric>
#include "acl/acl.h"
#include "ock/log/OckHcpsLogger.h"
#include "ock/utils/OckSafeUtils.h"
#include "ock/hmm/mgr/OckHmmHMObject.h"
#include "ock/hcps/error/OckHcpsErrorCode.h"
namespace ock {
namespace hcps {
namespace nop {
class OckDataBuffer {
public:
    ~OckDataBuffer() noexcept = default;
    OckDataBuffer() = default;
    // 使用地址和长度构建OckDataBuffer，所用空间为hmoHolder的一部分，hmoHolder仅用于保证持有空间不被释放
    OckDataBuffer(uintptr_t address, uint64_t byteCount, std::shared_ptr<hmm::OckHmmHMObject> hmoObj,
        const std::vector<int64_t> &shapes = {})
        : addr(address), byteSize(byteCount), shape(shapes), hmoHolder(hmoObj){};
    // 使用hmo的buffer构建OckDataBuffer,持有空间的有效与否，由构建的buffer控制
    explicit OckDataBuffer(std::shared_ptr<hmm::OckHmmHMOBuffer> hmoBuffer, const std::vector<int64_t> &shapes = {})
        : addr(hmoBuffer->Address()), byteSize(hmoBuffer->Size()), shape(shapes), bufferHolder(hmoBuffer){};
    // 使用Hmo构建OckDataBuffer，hmoHolder保证了OckDataBuffer生存期间，hmo持有的空间不被释放
    explicit OckDataBuffer(std::shared_ptr<hmm::OckHmmHMObject> hmo, const std::vector<int64_t> &shapes = {})
        : addr(hmo->Addr()),
          byteSize(hmo->GetByteSize()),
          shape(shapes),
          hmoHolder(hmo),
          bufferHolder(hmo->GetBuffer(hmo->Location(), 0, hmo->GetByteSize())){};
    uint64_t GetByteSize() const;
    uintptr_t Addr() const;
    const std::vector<int64_t> &Shape() const;

    template <class T> std::shared_ptr<OckDataBuffer> SubBuffer(uint32_t index)
    {
        uint64_t newByteSize = utils::SafeDiv(byteSize, shape[0]);
        std::vector<int64_t> newShape(shape.cbegin() + 1, shape.cend());
        uintptr_t newAddr = addr + index * sizeof(T) * ElemCountOfShape(newShape);
        auto sliceDataBuffer = std::make_shared<OckDataBuffer>(newAddr, newByteSize, hmoHolder, newShape);
        return sliceDataBuffer;
    }
    static uint64_t ElemCountOfShape(const std::vector<int64_t> &shape)
    {
        return std::accumulate(shape.cbegin(), shape.cend(), 1LL, std::multiplies<int64_t>());
    }

private:
    uintptr_t addr{0};
    uint64_t byteSize{0};
    std::vector<int64_t> shape{};
    std::shared_ptr<hmm::OckHmmHMObject> hmoHolder{ nullptr };
    std::shared_ptr<hmm::OckHmmHMOBuffer> bufferHolder{ nullptr };
};

std::ostream &operator << (std::ostream &os, const OckDataBuffer &buffer);
bool operator == (const OckDataBuffer &lhs, const OckDataBuffer &rhs);
bool operator != (const OckDataBuffer &lhs, const OckDataBuffer &rhs);

template <typename T> static void FillBuffer(std::shared_ptr<OckDataBuffer> &buffer, const std::vector<T> &hostData)
{
    if (hostData.size() == 0) {
        return;
    }
    if (buffer->GetByteSize() != hostData.size() * sizeof(T)) {
        OCK_HCPS_LOG_ERROR("buffer and data size disagree!");
    }
    aclrtMemcpy(reinterpret_cast<void *>(buffer->Addr()), buffer->GetByteSize(), hostData.data(),
        hostData.size() * sizeof(T), ACL_MEMCPY_HOST_TO_DEVICE);
}
} // namespace nop
} // namespace hcps
} // namespace ock
#endif // OCK_HCPS_OCK_DATA_BUFFER_H