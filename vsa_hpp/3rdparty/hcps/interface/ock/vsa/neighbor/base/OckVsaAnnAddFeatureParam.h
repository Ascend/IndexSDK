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


#ifndef OCK_VSA_ANN_INDEX_ADD_FEATURE_PARAM_H
#define OCK_VSA_ANN_INDEX_ADD_FEATURE_PARAM_H
#include <cstdint>
#include <memory>
#include "ock/vsa/OckVsaErrorCode.h"
#include "ock/hcps/algo/OckTopNQueue.h"

namespace ock {
namespace vsa {
namespace neighbor {

template <typename DataTemp, typename KeyTraitTemp>
struct OckVsaAnnAddFeatureParam {
    using DataT = DataTemp;
    using KeyTraitT = KeyTraitTemp;
    using KeyTypeTupleT = typename KeyTraitTemp::KeyTypeTuple;
    OckVsaAnnAddFeatureParam(uint64_t featureCount, const DataTemp *featureData, const KeyTypeTupleT *attrs,
        const int64_t *featureLabels, const uint8_t *customAttrs);
    /* @brief 指针数据总数减少，数据指针往前移动
     */
    void Shift(uint64_t offset, uint64_t dimSize, uint64_t customAttrByteSize);

    uint64_t count;
    const DataTemp *features;
    const KeyTypeTupleT *attributes;
    const int64_t *labels;
    const uint8_t *customAttr;
};
template <typename DataTemp, typename KeyTraitTemp>
struct AddFeatureParamMeta {
    using KeyTraitT = KeyTraitTemp;
    using KeyTypeTupleT = typename KeyTraitTemp::KeyTypeTuple;
    std::vector<DataTemp> validateFeatures{};
    std::vector<KeyTypeTupleT> attributes{};
    std::vector<uint8_t> customAttr{};
    std::vector<int64_t> validLabels{};
};
template <typename DataTemp, typename KeyTraitTemp>
OckVsaAnnAddFeatureParam<DataTemp, KeyTraitTemp>::OckVsaAnnAddFeatureParam(uint64_t featureCount,
    const DataTemp *featureData, const KeyTypeTupleT *attrs, const int64_t *featureLabels, const uint8_t *customAttrs)
    : count(featureCount), features(featureData), attributes(attrs), labels(featureLabels), customAttr(customAttrs)
{}
template <typename DataTemp, typename KeyTraitTemp>
void OckVsaAnnAddFeatureParam<DataTemp, KeyTraitTemp>::Shift(uint64_t offset, uint64_t dimSize,
    uint64_t customAttrByteSize)
{
    if (offset > count) {
        OCK_HMM_LOG_ERROR("Offset exceeds max count");
        return;
    }
    count -= offset;
    features += dimSize * offset;
    attributes += offset;
    labels += offset;
    customAttr += customAttrByteSize * offset;
}
}  // namespace neighbor
}  // namespace vsa
}  // namespace ock
#endif