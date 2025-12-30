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

#include <cstdlib>
#include <vector>
#include <gtest/gtest.h>
#include "ock/hcps/nop/OckDataBuffer.h"

namespace ock {
namespace hcps {
namespace nop {
namespace test {
class TestOckDataBuffer : public testing::Test {
public:
    TestOckDataBuffer() = default;
    uintptr_t originAddr{ 13U };
    uint64_t originByteSize{ 48U };
    std::vector<int64_t> originShape{ 4U, 2U, 3U };
    uintptr_t sliceAddr{ 25U };    // 13 + 2 * 3 * sizeof(int16_t) = 25
    uint64_t sliceByteSize{ 12U }; // 2 * 3 * sizeof(int16_t)
    std::vector<int64_t> sliceShape{ 2U, 3U };
    std::shared_ptr<hmm::OckHmmHMObject> holder;
};

TEST_F(TestOckDataBuffer, create_data_buffer)
{
    OckDataBuffer dataBuffer = OckDataBuffer(originAddr, originByteSize, holder, originShape);
    EXPECT_EQ(dataBuffer.Addr(), originAddr);
    EXPECT_EQ(dataBuffer.GetByteSize(), originByteSize);
    EXPECT_EQ(dataBuffer.Shape(), originShape);
}

TEST_F(TestOckDataBuffer, create_slice_buffer)
{
    auto originDataBuffer = std::make_shared<OckDataBuffer>(originAddr, originByteSize, holder, originShape);
    std::shared_ptr<OckDataBuffer> sliceDataBuffer;
    sliceDataBuffer = originDataBuffer->SubBuffer<int16_t>(1U);
    EXPECT_EQ(sliceDataBuffer->Addr(), sliceAddr);
    EXPECT_EQ(sliceDataBuffer->GetByteSize(), sliceByteSize);
    EXPECT_EQ(sliceDataBuffer->Shape(), sliceShape);
}

TEST_F(TestOckDataBuffer, print_data_buffer)
{
    OckDataBuffer dataBuffer = OckDataBuffer(originAddr, originByteSize, holder, originShape);
    std::ostringstream os;
    os << dataBuffer;
    std::string bufferString = os.str();
    EXPECT_EQ(bufferString, "{'byteLength':48,'shape':4,2,3,}");
}
} // namespace test
} // namespace nop
} // namespace hcps
} // namespace ock
