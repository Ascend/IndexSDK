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

#include <memory>
#include <vector>
#include <gtest/gtest.h>
#include "acl/acl.h"
#include "mockcpp/mockcpp.hpp"
#include "ock/hmm/OckHmmFactory.h"
#include "ock/utils/OckSafeUtils.h"
#include "ock/acladapter/utils/OckAscendFp16.h"
#include "ock/hcps/stream/MockOckHeteroStreamBase.h"
#include "ock/hcps/WithEnvOckHeteroHandler.h"
#include "ock/hcps/nop/dist_mask_with_extra_gen_op/OckDistMaskWithExtraGenOpDataBuffer.h"
namespace ock {
namespace hcps {
namespace nop {
namespace test {
class TestOckDistMaskWithExtraGenOpDataBuffer : public handler::WithEnvOckHeteroHandler<testing::Test> {
public:
    using BaseT = hcps::handler::WithEnvOckHeteroHandler<testing::Test>;
    void SetUp(void) override
    {
        BaseT::SetUp();
        hmm::OckHmmErrorCode errorCode = hmm::HMM_SUCCESS;
        handler = CreateSingleDeviceHandler(errorCode);
    }

    void TearDown(void) override
    {
        maskHmoGroup.reset();
        handler.reset();
        BaseT::TearDown();
    }

    void PrepareHmoGroup(void)
    {
        maskHmoGroup = std::make_shared<OckDistMaskWithExtraGenOpHmoGroup>();
        maskHmoGroup->queryTimes = AllocHmo(OPS_DATA_TYPE_ALIGN * sizeof(int32_t));
        maskHmoGroup->queryTokenIds = AllocHmo(utils::SafeDivUp(2500U, OPS_DATA_TYPE_ALIGN) * OPS_DATA_TYPE_TIMES);
        maskHmoGroup->attrTimes = AllocHmo(DEFAULT_CODE_BLOCK_SIZE * DEFAULT_GROUP_BLOCK_NUM * sizeof(int32_t));
        maskHmoGroup->attrTokenQuotients =
            AllocHmo(DEFAULT_CODE_BLOCK_SIZE * DEFAULT_GROUP_BLOCK_NUM * sizeof(int32_t));
        maskHmoGroup->attrTokenRemainders =
            AllocHmo(DEFAULT_CODE_BLOCK_SIZE * DEFAULT_GROUP_BLOCK_NUM * OPS_DATA_TYPE_TIMES);
        maskHmoGroup->mask = AllocHmo(DEFAULT_CODE_BLOCK_SIZE * DEFAULT_GROUP_BLOCK_NUM / OPS_DATA_TYPE_ALIGN);
        maskHmoGroup->extraMask = AllocHmo(DEFAULT_CODE_BLOCK_SIZE * DEFAULT_GROUP_BLOCK_NUM / OPS_DATA_TYPE_ALIGN);
    }

    std::shared_ptr<hcps::handler::OckHeteroHandler> handler;
    std::shared_ptr<OckDistMaskWithExtraGenOpHmoGroup> maskHmoGroup;

private:
    std::shared_ptr<hmm::OckHmmHMObject> AllocHmo(int64_t byteSize)
    {
        return handler->HmmMgrPtr()->Alloc(byteSize, hmm::OckHmmMemoryAllocatePolicy::DEVICE_DDR_ONLY).second;
    }
};

TEST_F(TestOckDistMaskWithExtraGenOpDataBuffer, alloc_buffers_from_hmo_group)
{
    OckDistMaskWithExtraGenOpMeta opSpec;
    auto buffer = ock::hcps::nop::OckDistMaskWithExtraGenOpDataBuffer::Create(opSpec);
    PrepareHmoGroup();
    buffer->AllocBuffersFromHmoGroup(maskHmoGroup, 0U,
        DEFAULT_CODE_BLOCK_SIZE * DEFAULT_GROUP_BLOCK_NUM / OPS_DATA_TYPE_ALIGN);
    EXPECT_EQ(*(buffer->InputQueryTime()), *(buffer->GetInputParams()[0U]));
    EXPECT_EQ(*(buffer->InputTokenBitSet()), *(buffer->GetInputParams()[1U]));
    EXPECT_EQ(*(buffer->InputAttrTimes()), *(buffer->GetInputParams()[2U]));
    EXPECT_EQ(*(buffer->InputAttrTokenQs()), *(buffer->GetInputParams()[3U]));
    EXPECT_EQ(*(buffer->InputAttrTokenRs()), *(buffer->GetInputParams()[4U]));
    EXPECT_EQ(*(buffer->InputExtraMask()), *(buffer->GetInputParams()[5U]));
    EXPECT_EQ(*(buffer->OutputMask()), *(buffer->GetOutputParams()[0U]));
}
} // namespace test
} // namespace nop
} // namespace hcps
} // namespace ock
