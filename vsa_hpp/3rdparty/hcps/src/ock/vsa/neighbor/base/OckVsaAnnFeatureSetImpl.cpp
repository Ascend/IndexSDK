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

#include "ock/vsa/neighbor/base/OckVsaAnnFeatureSet.h"

namespace ock {
namespace vsa {
namespace neighbor {
namespace adapter {
OckVsaAnnFeature::OckVsaAnnFeature(std::shared_ptr<hmm::OckHmmHMObject> features,
    std::shared_ptr<hmm::OckHmmHMObject> norms, std::shared_ptr<hmm::OckHmmHMObject> masks, uint32_t validateRowNum,
    uint32_t maxRowNum)
    : validateRowCount(validateRowNum), maxRowCount(maxRowNum), feature(features), norm(norms), mask(masks)
{}
bool OckVsaAnnFeature::Full(void) const
{
    return validateRowCount >= maxRowCount;
}
class OckVsaAnnFeatureSetImpl : public OckVsaAnnFeatureSet {
public:
    virtual ~OckVsaAnnFeatureSetImpl() noexcept = default;
    OckVsaAnnFeatureSetImpl(bool isUsingMask, uint32_t rowNumPerFeature)
        : usingMask(isUsingMask), rowCountPerFeature(rowNumPerFeature)
    {}
    uint32_t FeatureCount(void) const override
    {
        return static_cast<uint32_t>(features.size());
    }
    uint32_t RowCountPerFeature(void) const override
    {
        return rowCountPerFeature;
    }
    OckVsaAnnFeature &GetFeature(uint32_t featurePos) override
    {
        return *features.at(featurePos);
    }
    /*
    @brief 是否使用Mask， 如果不使用Mask，那么Mask数据不生效
    */
    bool UsingMask(void) const override
    {
        return usingMask;
    }
    void AddFeature(std::shared_ptr<OckVsaAnnFeature> feature) override
    {
        features.push_back(feature);
    }

private:
    bool usingMask{ false };
    uint32_t rowCountPerFeature{ 0 };
    std::vector<std::shared_ptr<OckVsaAnnFeature>> features{};
};

std::shared_ptr<OckVsaAnnFeatureSet> OckVsaAnnFeatureSet::Create(bool usingMask, uint32_t rowCountPerFeature)
{
    return std::make_shared<OckVsaAnnFeatureSetImpl>(usingMask, rowCountPerFeature);
}
std::shared_ptr<OckVsaAnnFeatureSet> OckVsaAnnFeatureSet::Create(
    std::deque<std::shared_ptr<hmm::OckHmmHMObject>> &features, std::deque<std::shared_ptr<hmm::OckHmmHMObject>> &norms,
    uint32_t rowCountPerFeature)
{
    auto ret = OckVsaAnnFeatureSet::Create(false, rowCountPerFeature);
    for (uint32_t i = 0; (i < features.size()) && (i < norms.size()); ++i) {
        ret->AddFeature(std::make_shared<OckVsaAnnFeature>(features.at(i),
            norms.at(i),
            std::shared_ptr<hmm::OckHmmHMObject>(),
            rowCountPerFeature,
            rowCountPerFeature));
    }
    return ret;
}
}  // namespace adapter
}  // namespace neighbor
}  // namespace vsa
}  // namespace ock