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


#ifndef OCK_VSA_NEIGHBOR_ADAPTER_FEATURE_SET_H
#define OCK_VSA_NEIGHBOR_ADAPTER_FEATURE_SET_H
#include <cstdint>
#include <memory>
#include <deque>
#include "ock/vsa/OckVsaErrorCode.h"
#include "ock/utils/OckSafeUtils.h"
#include "ock/hmm/mgr/OckHmmHMObject.h"
#include "ock/hcps/algo/OckElasticBitSet.h"
#include "ock/hcps/algo/OckShape.h"
#include "ock/hcps/handler/OckHeteroHandler.h"
#include "ock/hcps/handler/OckHandlerHmmHelper.h"

namespace ock {
namespace vsa {
namespace neighbor {
namespace adapter {
struct OckVsaAnnFeature {
    OckVsaAnnFeature(std::shared_ptr<hmm::OckHmmHMObject> features, std::shared_ptr<hmm::OckHmmHMObject> norms,
        std::shared_ptr<hmm::OckHmmHMObject> masks, uint32_t validateRowNum, uint32_t maxRowNum);
    bool Full(void) const;
    uint32_t validateRowCount;  // 有效行数
    uint32_t maxRowCount;
    std::shared_ptr<hmm::OckHmmHMObject> feature;  // 底库数据，分形好的
    std::shared_ptr<hmm::OckHmmHMObject> norm;     // Norm数据, 1/Len
    std::shared_ptr<hmm::OckHmmHMObject> mask;     // Mask数据
};
class OckVsaAnnFeatureSet {
public:
    virtual ~OckVsaAnnFeatureSet() noexcept = default;
    virtual uint32_t FeatureCount(void) const = 0;
    virtual uint32_t RowCountPerFeature(void) const = 0;
    virtual OckVsaAnnFeature &GetFeature(uint32_t featurePos) = 0;
    /*
    @brief 是否使用Mask， 如果不使用Mask，那么Mask数据不生效
    */
    virtual bool UsingMask(void) const = 0;
    virtual void AddFeature(std::shared_ptr<OckVsaAnnFeature> feature) = 0;

    static std::shared_ptr<OckVsaAnnFeatureSet> Create(bool usingMask, uint32_t rowCountPerFeature);
    static std::shared_ptr<OckVsaAnnFeatureSet> Create(std::deque<std::shared_ptr<hmm::OckHmmHMObject>> &features,
        std::deque<std::shared_ptr<hmm::OckHmmHMObject>> &norms, uint32_t rowCountPerFeature);
};
template <typename DataTemp, uint64_t DimSizeTemp, uint64_t NormTypeByteSize>
std::shared_ptr<OckVsaAnnFeature> MakeOckVsaAnnFeature(hcps::handler::OckHeteroHandler &handler, uint64_t maxRowCount,
    OckVsaErrorCode &errorCode)
{
    uint64_t featureByteSize = maxRowCount * sizeof(DataTemp) * DimSizeTemp;
    uint64_t normByteSize = maxRowCount * NormTypeByteSize;
    uint64_t maskByteSize = utils::SafeDivUp(maxRowCount, __CHAR_BIT__);
    uint64_t needIncHostByteSizes = featureByteSize + normByteSize + maskByteSize;
    errorCode = hcps::handler::helper::UseIncBindMemory(handler, needIncHostByteSizes, "MakeOckVsaAnnFeature");
    if (errorCode != hmm::HMM_SUCCESS) {
        return std::shared_ptr<adapter::OckVsaAnnFeature>();
    }
    return std::make_shared<OckVsaAnnFeature>(hcps::handler::helper::MakeHostHmo(handler, featureByteSize, errorCode),
        hcps::handler::helper::MakeHostHmo(handler, normByteSize, errorCode),
        hcps::handler::helper::MakeHostHmo(handler, maskByteSize, errorCode), 0UL, maxRowCount);
}
}  // namespace adapter
}  // namespace neighbor
}  // namespace vsa
}  // namespace ock
#endif