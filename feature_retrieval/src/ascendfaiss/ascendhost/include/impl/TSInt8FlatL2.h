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


#ifndef ASCENDHOST_SRC_ASCENDFAISS_ASCENDHOST_INCLUDE_IMPL_TSINT8FLATL2_H
#define ASCENDHOST_SRC_ASCENDFAISS_ASCENDHOST_INCLUDE_IMPL_TSINT8FLATL2_H
#include "IndexInt8FlatL2Aicpu.h"
#include "TSBase.h"
namespace ascend {
namespace {
} // namespace
class TSInt8FlatL2 : public TSBase, public IndexInt8FlatL2Aicpu {
public:
    TSInt8FlatL2(uint32_t deviceId, uint32_t dim, uint32_t tokenNum, uint64_t resources,
        uint32_t customAttrLen, uint32_t customAttrBlockSize);
    ~TSInt8FlatL2() = default;

    APP_ERROR addFeatureWithLabels(int64_t n, const void *features,
        const faiss::ascend::FeatureAttr *attrs, const int64_t *labels, const uint8_t *customAttr,
        const faiss::ascend::ExtraValAttr *val) override;
    APP_ERROR delFeatureWithLabels(int64_t n, const int64_t *labels) override;
    APP_ERROR getFeatureByLabel(int64_t n, const int64_t *labels, void *features) const override;
    APP_ERROR deleteFeatureByToken(int64_t count, const uint32_t *tokens) override;
    APP_ERROR search(uint32_t count, const void *features, const faiss::ascend::AttrFilter *attrFilter,
                     bool shareAttrFilter, uint32_t topk, int64_t *labels, float *distances,
                     uint32_t *validNums, bool enableTimeFilter,
                     const faiss::ascend::ExtraValFilter *extraValFilter) override;
    APP_ERROR searchWithExtraMask(uint32_t count, const void *features, const faiss::ascend::AttrFilter *attrFilter,
                                  bool shareAttrFilter, uint32_t topk, const uint8_t *extraMask,
                                  uint64_t extraMaskLen, bool extraMaskIsAtDevice, int64_t *labels, float *distances,
                                  uint32_t *validNums, bool enableTimeFilter, const float16_t *extraScore) override;
    APP_ERROR getBaseByRange(uint32_t offset, uint32_t num, int64_t *labels, void *features,
        faiss::ascend::FeatureAttr *attributes, faiss::ascend::ExtraValAttr *extraVal);

private:
    APP_ERROR searchPagedWithMasks(int pageIdx, int batch, const int8_t *features, int topK,
        AscendTensor<uint8_t, DIMS_3> &masks, AscendTensor<float16_t, DIMS_2> &outDistanceOnDevice,
        AscendTensor<int64_t, DIMS_2> &outIndicesOnDevice);
    APP_ERROR searchBatchWithShareMasks(int batch, const int8_t *features, int topK, float *distances,
        int64_t *labels, AscendTensor<uint8_t, DIMS_3> &masks);
    APP_ERROR searchBatchWithNonshareMasks(int batch, const int8_t *features, int topK, float *distances,
        int64_t *labels, AscendTensor<int32_t, DIMS_2> &queryTimes, AscendTensor<uint8_t, DIMS_2> &tokenIds);
    APP_ERROR searchBatchWithExtraNonshareMasks(int batch, const int8_t *features, int topK, float *distances,
        int64_t *labels, AscendTensor<int32_t, DIMS_2> &queryTimes, AscendTensor<uint8_t, DIMS_2> &tokenIds,
        const uint8_t *extraMask);
    APP_ERROR searchInPureDev(uint32_t count, const int8_t *features, const faiss::ascend::AttrFilter *attrFilter,
        uint32_t topk, int64_t *labels, float *distances);
    APP_ERROR searchInPureDevWithExtraMask(uint32_t count, const int8_t *features,
        const faiss::ascend::AttrFilter *attrFilter, uint32_t topk, const uint8_t *extraMask, int64_t *labels,
        float *distances);
    
    void addWithIds(idx_t n, const int8_t *features, const idx_t *featureIds);
    void addWithIdsImpl(int n, const int8_t *x);
    void queryVectorByIdx(int64_t idx, uint8_t *dis, int num) const;
    void removeLabels(const std::vector<int64_t> &removeIds);
    void removeIdsImpl(const std::vector<int64_t> &indices);
    void removeFeatureByIds(const std::vector<int64_t> &ids);
    void postProcess(uint64_t searchNum, int topK, AscendTensor<float16_t, DIMS_2> &outDistanceOnDevice,
        AscendTensor<int64_t, DIMS_2> &outIndicesOnDevice, float *distances, int64_t *labels);
    void getValidNum(uint64_t count, uint32_t topk, int64_t *labels, uint32_t *validNums) const;
    void runInt8L2DistCompute(int batch, uint32_t actualNum, bool shareMask,
                            const std::vector<const AscendTensorBase *> &input,
                            const std::vector<const AscendTensorBase *> &output,
                            aclrtStream stream) const;
    APP_ERROR resetInt8L2DistCompute(int codeNum, bool shareMask) const;
    std::vector<idx_t> ids;
    int code_size = 0;
    uint32_t deviceId = 0;
    bool shareAttrFilter = false;
};

} // namespace ascend

#endif // ASCENDHOST_SRC_ASCENDFAISS_ASCENDHOST_INCLUDE_IMPL_TSINT8FLATL2_H
