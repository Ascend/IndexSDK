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
#include <gtest/gtest.h>
#include <vector>
#include "ock/vsa/neighbor/npu/OckVsaAnnNpuIndex.h"
#include "ock/hcps/WithEnvOckHeteroHandler.h"
#include "ock/vsa/attr/OckTimeSpaceAttrTrait.h"

namespace ock {
namespace vsa {
namespace neighbor {
namespace npu {
namespace withenv {

struct OckVsaAnnAddFeatureParamHolder {
    OckVsaAnnAddFeatureParamHolder(void) = delete;
    OckVsaAnnAddFeatureParamHolder(const OckVsaAnnAddFeatureParamHolder &other) = delete;
    OckVsaAnnAddFeatureParamHolder(uint64_t count, uint32_t customAttrByteSize)
        : count(count), features(new int8_t[count]), attributes(new attr::OckTimeSpaceAttr[count]),
          labels(new int64_t[count])
    {
        if (customAttrByteSize > 0) {
            customAttr = new uint8_t[count * customAttrByteSize];
        } else {
            customAttr = nullptr;
        }
    }
    ~OckVsaAnnAddFeatureParamHolder() noexcept
    {
        if (features != nullptr) {
            delete[] features;
        }
        if (attributes != nullptr) {
            delete[] attributes;
        }
        if (labels != nullptr) {
            delete[] labels;
        }
        if (customAttr != nullptr) {
            delete[] customAttr;
        }
    }
    OckVsaAnnAddFeatureParam<int8_t, attr::OckTimeSpaceAttrTrait> ToParam(void)
    {
        return OckVsaAnnAddFeatureParam<int8_t, attr::OckTimeSpaceAttrTrait>(count,
            features,
            reinterpret_cast<attr::OckTimeSpaceAttrTrait::KeyTypeTuple *>(attributes),
            labels,
            customAttr);
    }
    uint64_t count;
    int8_t *features;
    attr::OckTimeSpaceAttr *attributes;
    int64_t *labels;
    uint8_t *customAttr;
};
}  // namespace withenv
template <typename _BaseT>
class WithEnvOckFeatureBuild : public _BaseT {
public:
    using BaseT = _BaseT;
    using OckVsaAnnAddFeatureParamT = OckVsaAnnAddFeatureParam<int8_t, attr::OckTimeSpaceAttrTrait>;
    void SetUp(void) override
    {
        BaseT::SetUp();
    }
    void TearDown(void) override
    {
        BaseT::TearDown();
    }
    OckVsaAnnAddFeatureParamT BuildFeature(uint64_t count, uint32_t customAttrByteSize)
    {
        paramHolderVec.push_back(std::make_shared<withenv::OckVsaAnnAddFeatureParamHolder>(count, customAttrByteSize));
        return paramHolderVec.back()->ToParam();
    }

    std::vector<std::shared_ptr<withenv::OckVsaAnnAddFeatureParamHolder>> paramHolderVec;
};
}  // namespace npu
}  // namespace neighbor
}  // namespace vsa
}  // namespace ock