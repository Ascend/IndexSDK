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


#ifndef UT_COMMON_H
#define UT_COMMON_H

#include <random>
#include "AscendIndexTS.h"
 
namespace ascend {
constexpr int BITELN = 8;
constexpr int DATASPLIT = 4;
constexpr int UINT8_GAP = 128;
void FeatureAttrGenerator(std::vector<faiss::ascend::FeatureAttr> &attrs);
void ExtraValAttrGenerator(std::vector<faiss::ascend::ExtraValAttr> &attrs);
void customAttrGenerator(std::vector<uint8_t> &customAttrs, size_t attrLen);

template<typename T>
void FeatureGenerator(std::vector<T> &features)
{
    size_t n = features.size();
    for (size_t i = 0; i < n; ++i) {
        features[i] = drand48();
    }
}
}
#endif
