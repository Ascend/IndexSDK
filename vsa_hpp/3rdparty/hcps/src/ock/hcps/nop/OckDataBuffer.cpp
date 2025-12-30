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

#include "ock/utils/OstreamUtils.h"
#include "ock/hcps/nop/OckDataBuffer.h"
namespace ock {
namespace hcps {
namespace nop {
uint64_t OckDataBuffer::GetByteSize() const
{
    return byteSize;
}

uintptr_t OckDataBuffer::Addr() const
{
    return addr;
}

const std::vector<int64_t> &OckDataBuffer::Shape() const
{
    return shape;
}

std::ostream &operator << (std::ostream &os, const OckDataBuffer &data)
{
    os << "{'byteLength':" << data.GetByteSize() << ",'shape':";
    for (auto x : data.Shape()) {
        os << x << ",";
    }
    os << "}";
    return os;
}

bool operator == (const OckDataBuffer &lhs, const OckDataBuffer &rhs)
{
    return lhs.Addr() == rhs.Addr() && lhs.GetByteSize() == rhs.GetByteSize() && lhs.Shape() == rhs.Shape();
}

bool operator != (const OckDataBuffer &lhs, const OckDataBuffer &rhs)
{
    return !(lhs == rhs);
}
} // namespace nop
} // namespace hcps
} // namespace ock