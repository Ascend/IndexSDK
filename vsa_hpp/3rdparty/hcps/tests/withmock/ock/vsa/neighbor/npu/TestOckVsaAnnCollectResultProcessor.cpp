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


#include <gtest/gtest.h>
#include "ock/vsa/neighbor/npu/OckVsaAnnCollectResultNpuProcessor.h"
#include "ock/hcps/WithEnvOckHeteroHandler.h"
namespace ock {
namespace vsa {
namespace neighbor {
namespace adapter {
namespace test {
using TYPE = uint64_t;
const uint64_t DIM = 256;
class TestOckVsaAnnCollectResultProcessor : public hcps::handler::WithEnvOckHeteroHandler<testing::Test> {
public:
    using BaseT = WithEnvOckHeteroHandler<testing::Test>;

    void SetUp(void) override
    {
        BaseT::SetUp();
    }

    void CreateNpuProcessor()
    {
        hmm::OckHmmErrorCode errorCode = hmm::HMM_SUCCESS;
        std::shared_ptr<hcps::handler::OckHeteroHandler> handler = CreateSingleDeviceHandler(errorCode);

        npuProcessor = OckVsaAnnCollectResultProcessor<TYPE, DIM>::CreateNPUProcessor(handler, queryCond, topN);
    }

    void TearDown(void) override
    {
        npuProcessor.reset();
        BaseT::TearDown();
    }

    std::vector<TYPE> queryCond{ 1 };
    std::shared_ptr<OckVsaAnnCollectResultProcessor<TYPE, DIM>> npuProcessor;
    uint32_t topN = 1024;
};

TEST_F(TestOckVsaAnnCollectResultProcessor, CreateNpuProcessor)
{
    CreateNpuProcessor();
    EXPECT_NE(npuProcessor, nullptr);
}
}
}
}
}
}