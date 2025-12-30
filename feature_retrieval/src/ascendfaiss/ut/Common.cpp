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


#include "Common.h"
#include <random>
namespace ascend {
void FeatureAttrGenerator(std::vector<faiss::ascend::FeatureAttr> &attrs)
{
    size_t n = attrs.size();
    for (size_t i = 0; i < n; ++i) {
        attrs[i].time = static_cast<int32_t>(i % DATASPLIT); // mock data para
        attrs[i].tokenId = static_cast<int32_t>(i % DATASPLIT); // mock data para
    }
}

void ExtraValAttrGenerator(std::vector<faiss::ascend::ExtraValAttr> &attrs)
{
    size_t n = attrs.size();
    for (size_t i = 0; i < n; ++i) {
        attrs[i].val = static_cast<int32_t>(i % DATASPLIT); // mock data para
    }
}

void customAttrGenerator(std::vector<uint8_t> &customAttrs, size_t attrLen)
{
    size_t n = customAttrs.size() / attrLen;
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < attrLen; j++) {
            customAttrs[i * attrLen + j] = i + j;
        }
    }
}

template<>
void FeatureGenerator<int8_t>(std::vector<int8_t> &features)
{
    std::independent_bits_engine<std::mt19937, BITELN, uint8_t> engine(1);
    size_t n = features.size();
    for (size_t i = 0; i < n; ++i) {
        features[i] = engine() - UINT8_GAP;
    }
}
} // namespace ascend

