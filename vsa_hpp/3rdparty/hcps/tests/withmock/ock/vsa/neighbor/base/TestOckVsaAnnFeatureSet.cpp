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
#include "ock/vsa/neighbor/base/OckVsaAnnFeatureSet.h"
#include "ock/hmm/mgr/MockOckHmmHMObject.h"
#include "ock/hcps/WithEnvOckHeteroHandler.h"

namespace ock {
namespace vsa {
namespace neighbor {
namespace adapter {
namespace test {
class TestOckVsaAnnFeatureSet : public hcps::handler::WithEnvOckHeteroHandler<testing::Test> {
public:
    using BaseT = hcps::handler::WithEnvOckHeteroHandler<testing::Test>;
    std::shared_ptr<hmm::OckHmmHMObject> BuildHMO(void)
    {
        return std::shared_ptr<hmm::OckHmmHMObject>(new hmm::MockOckHmmHMObject());
    }
    std::deque<std::shared_ptr<hmm::OckHmmHMObject>> BuildHMOs(uint32_t count)
    {
        std::deque<std::shared_ptr<hmm::OckHmmHMObject>> ret;
        for (uint32_t i = 0; i < count; ++i) {
            ret.push_back(BuildHMO());
        }
        return ret;
    }
    uint32_t featureCount{3UL};
    uint32_t rowCountPerFeature{1024UL};
};

TEST_F(TestOckVsaAnnFeatureSet, Create)
{
    auto features = BuildHMOs(featureCount);
    auto norms = BuildHMOs(featureCount);
    bool usingMask = true;
    auto featureSet = OckVsaAnnFeatureSet::Create(usingMask, rowCountPerFeature);
    EXPECT_TRUE(featureSet->UsingMask());
    EXPECT_EQ(0UL, featureSet->FeatureCount());
    EXPECT_EQ(rowCountPerFeature, featureSet->RowCountPerFeature());

    featureSet->AddFeature(std::make_shared<OckVsaAnnFeature>(
        BuildHMO(), BuildHMO(), BuildHMO(), rowCountPerFeature - 1UL, rowCountPerFeature));
    EXPECT_FALSE(featureSet->GetFeature(0UL).Full());
    EXPECT_EQ(1UL, featureSet->FeatureCount());
}
TEST_F(TestOckVsaAnnFeatureSet, CreateByFeatures)
{
    auto features = BuildHMOs(featureCount);
    auto norms = BuildHMOs(featureCount);
    auto featureSet = OckVsaAnnFeatureSet::Create(features, norms, rowCountPerFeature);
    EXPECT_EQ(featureCount, featureSet->FeatureCount());
    EXPECT_FALSE(featureSet->UsingMask());
    EXPECT_EQ(rowCountPerFeature, featureSet->RowCountPerFeature());
    EXPECT_TRUE(featureSet->GetFeature(0UL).Full());
    EXPECT_EQ(rowCountPerFeature, featureSet->GetFeature(0ULL).validateRowCount);
    EXPECT_EQ(rowCountPerFeature, featureSet->GetFeature(0ULL).maxRowCount);
}
TEST_F(TestOckVsaAnnFeatureSet, MakeOckVsaAnnFeature)
{
    OckVsaErrorCode errorCode = hmm::HMM_SUCCESS;
    auto handler = CreateSingleDeviceHandler(errorCode);
    auto data = MakeOckVsaAnnFeature<int8_t, 256ULL, 2ULL>(*handler, 10UL, errorCode);
    EXPECT_TRUE(data.get() != nullptr);
    EXPECT_EQ(errorCode, hmm::HMM_SUCCESS);
}
}  // namespace test
}  // namespace adapter
}  // namespace neighbor
}  // namespace vsa
}  // namespace ock