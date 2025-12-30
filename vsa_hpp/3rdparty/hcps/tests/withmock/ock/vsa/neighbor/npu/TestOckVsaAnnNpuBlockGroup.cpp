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

#include <cstring>
#include <gtest/gtest.h>
#include "ock/log/OckHcpsLogger.h"
#include "ock/vsa/neighbor/npu/OckVsaAnnNpuBlockGroup.h"
#include "ock/hcps/WithEnvOckHeteroHandler.h"

namespace ock {
namespace vsa {
namespace neighbor {
namespace npu {
namespace test {
namespace {
const uint32_t DIMSIZE = 256UL;
}
class TestOckVsaAnnNpuBlockGroup : public hcps::handler::WithEnvOckHeteroHandler<testing::Test> {
public:
    using BaseT = WithEnvOckHeteroHandler<testing::Test>;
    using OckFloat16 = uint16_t;
    using DataT = int32_t;
    void SetUp(void) override
    {
        BaseT::SetUp();
        hmmSpec.devSpec.maxDataCapacity = 5ULL * 1024UL * 1024UL * 1024UL;  // 1024MB;
        hmmSpec.devSpec.maxSwapCapacity = 1024UL * 1024UL * 1024UL;         // 512MB;
        hmmSpec.hostSpec.maxDataCapacity = 5ULL * 1024UL * 1024UL * 1024UL; // 1024MB;
        hmmSpec.hostSpec.maxSwapCapacity = 1024UL * 1024UL * 1024UL;        // 512MB;
    }
    OckVsaErrorCode InitBlockGroup(void)
    {
        hmm::OckHmmErrorCode errorCode = hmm::HMM_SUCCESS;
        handler = CreateSingleDeviceHandler(errorCode);
        hostFeatures.reserve(blockGroupNum);
        for (uint32_t i = 0; i < blockGroupNum; ++i) {
            auto hostFeature =
                hcps::handler::helper::MakeHostHmo(*handler, blockRowCount * sizeof(DataT) * DIMSIZE, errorCode);
            hostFeatures.push_back(hostFeature);
            std::shared_ptr<hmm::OckHmmHMObject> featureHmo =
                hcps::handler::helper::MakeHostHmo(*handler, blockRowCount * sizeof(DataT) * DIMSIZE, errorCode);
            hcps::algo::OckShape<DataT, DIMSIZE> srcShape(featureHmo->Addr(), featureHmo->GetByteSize());
            srcShape.AddData(reinterpret_cast<DataT *>(hostFeature->Addr()), blockRowCount);

            auto featureDeviceHmo =
                hcps::handler::helper::MakeDeviceHmo(*handler, blockRowCount * sizeof(DataT) * DIMSIZE, errorCode);
            handler->HmmMgr().CopyHMO(*featureDeviceHmo, 0ULL, *featureHmo, 0ULL, featureHmo->GetByteSize());

            auto normHmo = hcps::handler::helper::MakeDeviceHmo(*handler, blockRowCount * 2, errorCode);
            auto attrTimeHmo =
                hcps::handler::helper::MakeDeviceHmo(*handler, blockRowCount * sizeof(int32_t), errorCode);
            auto attrTokenQuetHmo =
                hcps::handler::helper::MakeDeviceHmo(*handler, blockRowCount * sizeof(int32_t), errorCode);
            auto attrTokenReHmo =
                hcps::handler::helper::MakeDeviceHmo(*handler, blockRowCount * sizeof(uint8_t) * 2, errorCode);
            auto extKeyAttrsHmo =
                hcps::handler::helper::MakeDeviceHmo(*handler, blockRowCount * sizeof(uint8_t) * 2, errorCode);
            OCK_CHECK_RETURN_ERRORCODE(errorCode);

            blockGroup.features.push_back(featureDeviceHmo);
            blockGroup.norms.push_back(normHmo);
            blockGroup.keyAttrsTime.push_back(attrTimeHmo);
            blockGroup.keyAttrsQuotient.push_back(attrTokenQuetHmo);
            blockGroup.keyAttrsRemainder.push_back(attrTokenReHmo);
            blockGroup.extKeyAttrs.push_back(extKeyAttrsHmo);
            blockGroup.rowCount = blockRowCount;
            blockGroup.lastBlockRowCount = 0;
        }
        return errorCode;
    }
    bool CompareHMObjectByteSize(std::vector<std::shared_ptr<hmm::OckHmmHMObject>> hmObjectVector,
        std::shared_ptr<hmm::OckHmmHMObject> hmObject)
    {
        uint64_t hmObjectVectorByteSize = 0;
        for (size_t i = 0; i < hmObjectVector.size(); ++i) {
            hmObjectVectorByteSize += hmObjectVector.at(i)->GetByteSize();
        }
        return (hmObjectVectorByteSize == hmObject->GetByteSize());
    }
    void CompareBlockByteSize(void)
    {
        EXPECT_EQ(CompareHMObjectByteSize(blockGroup.features, blockInfo->feature), true);
        EXPECT_EQ(CompareHMObjectByteSize(blockGroup.norms, blockInfo->norm), true);
        EXPECT_EQ(CompareHMObjectByteSize(blockGroup.keyAttrsTime, blockInfo->keyAttrTime), true);
        EXPECT_EQ(CompareHMObjectByteSize(blockGroup.keyAttrsQuotient, blockInfo->keyAttrQuotient), true);
        EXPECT_EQ(CompareHMObjectByteSize(blockGroup.keyAttrsRemainder, blockInfo->keyAttrRemainder), true);
        EXPECT_EQ(CompareHMObjectByteSize(blockGroup.extKeyAttrs, blockInfo->extKeyAttr), true);
        EXPECT_EQ((blockGroup.rowCount == blockInfo->rowCount), true);
    }
    bool CompareHMObjectValue(std::vector<std::shared_ptr<hmm::OckHmmHMObject>> hmObjectVector,
        std::shared_ptr<hmm::OckHmmHMObject> hmObject)
    {
        uint8_t *hostAddre = reinterpret_cast<uint8_t *>(hmObject->Addr());
        for (size_t i = 0; i < hmObjectVector.size(); ++i) {
            auto hmObjectItemByteSize = hmObjectVector.at(i)->GetByteSize();
            auto compareResult = memcmp(reinterpret_cast<uint8_t *>(hmObjectVector.at(i)->Addr()),
                hostAddre + hmObjectItemByteSize * i, hmObjectItemByteSize);
            if (compareResult != 0) {
                return false;
            }
        }
        return true;
    }
    void CompareBlockValue(void)
    {
        EXPECT_EQ(CompareHMObjectValue(hostFeatures, blockInfo->feature), true);
        EXPECT_EQ(CompareHMObjectValue(blockGroup.norms, blockInfo->norm), true);
        EXPECT_EQ(CompareHMObjectValue(blockGroup.keyAttrsTime, blockInfo->keyAttrTime), true);
        EXPECT_EQ(CompareHMObjectValue(blockGroup.keyAttrsQuotient, blockInfo->keyAttrQuotient), true);
        EXPECT_EQ(CompareHMObjectValue(blockGroup.keyAttrsRemainder, blockInfo->keyAttrRemainder), true);
        EXPECT_EQ(CompareHMObjectValue(blockGroup.extKeyAttrs, blockInfo->extKeyAttr), true);
    }

    void TearDown(void) override
    {
        blockGroup.features.clear();
        blockGroup.norms.clear();
        blockGroup.keyAttrsTime.clear();
        blockGroup.keyAttrsQuotient.clear();
        blockGroup.keyAttrsRemainder.clear();
        blockGroup.extKeyAttrs.clear();
        blockInfo.reset();
        hostFeatures.clear();
        handler.reset();
        BaseT::TearDown();
    }

    uint32_t blockRowCount = 65536UL;; // 需要满足最小分形块的大小要求，CompareHMObjectValue 按照内存块比较
    uint32_t blockGroupNum = 64UL;
    std::shared_ptr<hcps::handler::OckHeteroHandler> handler;
    std::vector<std::shared_ptr<hmm::OckHmmHMObject>> hostFeatures;
    std::shared_ptr<OckVsaAnnRawBlockInfo> blockInfo;
    OckVsaAnnNpuBlockGroup blockGroup;
};
TEST_F(TestOckVsaAnnNpuBlockGroup, attrBytes)
{
    EXPECT_EQ(4ULL, OckVsaAnnKeyAttrInfo::KeyAttrTimeBytes());
    EXPECT_EQ(4ULL, OckVsaAnnKeyAttrInfo::KeyAttrQuotientBytes());
    EXPECT_EQ(2ULL, OckVsaAnnKeyAttrInfo::KeyAttrRemainderBytes());
    EXPECT_EQ(10ULL, OckVsaAnnKeyAttrInfo::KeyAttrAllBytes());
}
TEST_F(TestOckVsaAnnNpuBlockGroup, BlockCount)
{
    OckVsaAnnNpuBlockGroup grp;
    EXPECT_EQ(0ULL, grp.BlockCount());
    grp.features.push_back(std::shared_ptr<hmm::OckHmmHMObject>());
    EXPECT_EQ(1ULL, grp.BlockCount());
}
TEST_F(TestOckVsaAnnNpuBlockGroup, LoadGroupBlocksIntoHost)
{
    InitBlockGroup();
    hmm::OckHmmErrorCode errorCode = hmm::HMM_SUCCESS;
    blockInfo = LoadGroupBlocksIntoHost<DataT, DIMSIZE>(*handler, blockGroup, errorCode);
    CompareBlockByteSize();
    CompareBlockValue();
    EXPECT_EQ(errorCode, hmm::HMM_SUCCESS);
}
} // namespace test
} // namespace npu
} // namespace neighbor
} // namespace vsa
} // namespace ock