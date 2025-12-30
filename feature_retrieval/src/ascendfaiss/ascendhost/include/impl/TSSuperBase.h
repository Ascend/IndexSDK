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

#include <vector>
#include "common/ErrorCode.h"
#include "ascendhost/include/index/AscendIndexTS.h"
#ifndef FEATURERETRIEVAL_TSSUPERBASE_H
#define FEATURERETRIEVAL_TSSUPERBASE_H
namespace ascend {
class TSSuperBase {
public:
    virtual ~TSSuperBase() noexcept = default;
    virtual int64_t getAttrTotal() const = 0;
    virtual APP_ERROR addFeatureWithLabels(int64_t n, const void *features, const faiss::ascend::FeatureAttr *attrs,
        const int64_t *labels, const uint8_t *customAttr, const faiss::ascend::ExtraValAttr *extraVal) = 0;
    virtual APP_ERROR delFeatureWithLabels(int64_t n, const int64_t *labels) = 0;
    virtual APP_ERROR getFeatureByLabel(int64_t n, const int64_t *labels, void *features) const = 0;
    virtual APP_ERROR deleteFeatureByToken(int64_t count, const uint32_t *tokens) = 0;
    virtual APP_ERROR search(uint32_t count, const void *features, const faiss::ascend::AttrFilter *attrFilter,
        bool shareAttrFilter, uint32_t topk, int64_t *labels, float *distances, uint32_t *validNums,
        bool enableTimeFilter, const faiss::ascend::ExtraValFilter *extraValFilter) = 0;
    virtual APP_ERROR searchWithExtraMask(uint32_t count, const void *features,
        const faiss::ascend::AttrFilter *attrFilter, bool shareAttrFilter, uint32_t topk, const uint8_t *extraMask,
        uint64_t extraMaskLen, bool extraMaskIsAtDevice, int64_t *labels, float *distances, uint32_t *validNums,
        bool enableTimeFilter, const uint16_t *extraScore) = 0;
    virtual APP_ERROR getFeatureAttrsByLabel(int64_t n, const int64_t *labels,
        faiss::ascend::FeatureAttr *attrs) const = 0;

    virtual APP_ERROR getCustomAttrByBlockId(uint32_t blockId, uint8_t *&customAttr) const = 0;
    virtual APP_ERROR AddFeatureByIndice(int64_t n, const void *features, const faiss::ascend::FeatureAttr *attrs,
        const int64_t *indices, const uint8_t *customAttr, const faiss::ascend::ExtraValAttr *extraVal) = 0;
    virtual APP_ERROR GetFeatureByIndice(int64_t count, const int64_t *indices, int64_t *labels,
        void *features, faiss::ascend::FeatureAttr *attributes, faiss::ascend::ExtraValAttr *extraVal) const = 0;
    virtual APP_ERROR FastDeleteFeatureByIndice(int64_t n, const int64_t *indices) = 0;
    virtual APP_ERROR FastDeleteFeatureByRange(int64_t start, int n) = 0;
    virtual std::vector<uint8_t> GetBaseMask() const = 0;
};
}

#endif // FEATURERETRIEVAL_TSSUPERBASE_H