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


#ifndef ASCENDHOST_SRC_ASCENDFAISS_ASCENDHOST_INCLUDE_IMPL_TSFLATIP_H
#define ASCENDHOST_SRC_ASCENDFAISS_ASCENDHOST_INCLUDE_IMPL_TSFLATIP_H
#include "IndexFlatIPAicpu.h"
#include "TSBase.h"
namespace ascend {
namespace {
// The value range of dim
const std::vector<int> DIM_RANGE = { 64, 128, 256, 384, 512, 768, 1024 };
} // namespace
class TSFlatIP : public TSBase, public IndexFlatIPAicpu {
public:
    TSFlatIP(uint32_t deviceId, uint32_t dim, uint32_t tokenNum, uint64_t resources, uint32_t customAttrLen,
        uint32_t customAttrBlockSize, const std::vector<float> &scale = std::vector<float>());
    ~TSFlatIP() = default;
    APP_ERROR addFeatureWithLabels(int64_t n, const void *features,
        const faiss::ascend::FeatureAttr *attrs, const int64_t *labels, const uint8_t *customAttr,
        const faiss::ascend::ExtraValAttr *val) override;
    APP_ERROR AddFeatureByIndice(int64_t n, const void *features,
        const faiss::ascend::FeatureAttr *attrs, const int64_t *indices, const uint8_t *customAttr,
        const faiss::ascend::ExtraValAttr *extraVal) override;
    APP_ERROR GetFeatureByIndice(int64_t count, const int64_t *indices, int64_t *labels,
        void *features, faiss::ascend::FeatureAttr *attributes, faiss::ascend::ExtraValAttr *extraVal) const override;
    APP_ERROR AddFeatureWithIndice(int64_t n, int64_t replaceNum, const int64_t *indices,
        const std::vector<std::pair<int64_t, int64_t>> &segments, const float *features);
    APP_ERROR delFeatureWithLabels(int64_t n, const int64_t *labels) override;
    APP_ERROR getFeatureByLabel(int64_t n, const int64_t *labels, void *features) const override;
    APP_ERROR deleteFeatureByToken(int64_t count, const uint32_t *tokens) override;
    APP_ERROR search(uint32_t count, const void *features, const faiss::ascend::AttrFilter *attrFilter,
        bool shareAttrFilter, uint32_t topk, int64_t *labels, float *distances, uint32_t *validNums,
        bool enableTimeFilter, const faiss::ascend::ExtraValFilter *extraValFilter) override;
    APP_ERROR searchWithExtraMask(uint32_t count, const void *features, const faiss::ascend::AttrFilter *attrFilter,
        bool shareAttrFilter, uint32_t topk, const uint8_t *extraMask, uint64_t extraMaskLen, bool extraMaskIsAtDevice,
        int64_t *labels, float *distances, uint32_t *validNums, bool enableTimeFilter,
        const float16_t *extraScore) override;
    APP_ERROR getBaseByRange(uint32_t offset, uint32_t num, int64_t *labels, void *features,
        faiss::ascend::FeatureAttr *attributes, faiss::ascend::ExtraValAttr *extraVal);

protected:
    std::map<int, std::unique_ptr<AscendOperator>> distComputeShareMaskOps;
    std::map<int, std::unique_ptr<AscendOperator>> distComputeNonshareMaskOps;
    bool shareAttrFilter = false;

private:
    void resetDistMaskCompOp(int numLists);
    void resetAscendcDistMaskCompOp(int numLists);
    void runDistMaskCompute(int batch, bool shareMask, const std::vector<const AscendTensorBase *> &input,
        const std::vector<const AscendTensorBase *> &output, aclrtStream stream);
    void runAscendcDistMaskCompute(int batch, bool shareMask, const std::vector<const AscendTensorBase *> &input,
        const std::vector<const AscendTensorBase *> &output, aclrtStream stream);
    void runDistMaskComputeWithScale(int batch, bool isUsedExtraScore,
        const std::vector<const AscendTensorBase *> &input,
        const std::vector<const AscendTensorBase *> &output, aclrtStream stream);
    void resetDistMaskWithScaleCompOp(int numLists);
    void add_with_ids(idx_t n, const float *x, const idx_t *xids);
    void addWithIdsImpl(int n, uint16_t *x);
    APP_ERROR searchBatchWithNonshareMasks(int batch, const uint16_t *x, int topK, float *distances, int64_t *labels,
        AscendTensor<int32_t, DIMS_2> &queryTimes, AscendTensor<uint8_t, DIMS_2> &tokenIds);
    APP_ERROR searchBatchWithShareMasks(int batch, const uint16_t *x, int topK, float *distances, int64_t *labels,
        AscendTensor<uint8_t, DIMS_3> &masks);
    APP_ERROR searchBatchWithExtraNonshareMasks(int batch, const uint16_t *x, int topK, float *distances,
        int64_t *labels, AscendTensor<int32_t, DIMS_2> &queryTimes, AscendTensor<uint8_t, DIMS_2> &tokenIds,
        const uint8_t *extraMask, float16_t *extraScore = nullptr);
    APP_ERROR searchBatchWithExtraShareMasks(int batch, const uint16_t *x, int topK, float *distances, int64_t *labels,
        AscendTensor<uint8_t, DIMS_3> &masks, const uint8_t *extraMask);
    APP_ERROR searchPagedWithMasks(size_t pageId, size_t pageNum, int batch, const uint16_t *x,
        AscendTensor<uint8_t, DIMS_3> &masks, AscendTensor<float16_t, DIMS_2> &maxDistances,
        AscendTensor<int64_t, DIMS_2> &maxIndices, float16_t *extraScore = nullptr);
    void getValidNum(uint64_t count, uint32_t topk, int64_t *labels, uint32_t *validNums) const;
    void postProcess(int64_t searchNum, int topK, float16_t *outDistances, float *distances, int64_t *labels);
    void queryVectorByIdx(int64_t idx, float *dis) const;
    void queryInt8VectorByIdx(int64_t idx, float *dis) const;
    APP_ERROR SetScale(const std::vector<float> &scale);
    void removeLabels(const std::vector<int64_t> &removeIds);
    void removeIdsImpl(const std::vector<int64_t> &indices);
    // 新增附加相似度计算的算子
    void runDistMaskExtraScoreCompute(int batch, bool shareMask,
         const std::vector<const AscendTensorBase *> &input,
         const std::vector<const AscendTensorBase *> &output, aclrtStream stream);
    void resetDistMaskExtraScoreCompOp(int numLists);
    APP_ERROR runSharedAttrFilter(int32_t queryNum, uint32_t topk, const faiss::ascend::AttrFilter *attrFilter,
                                  const uint8_t *extraMask, const float *queryFeatures, int64_t *labels,
                                  float *distances);
    APP_ERROR runNonSharedAttrFilter(int32_t queryNum, uint32_t topk, uint64_t extraMaskLen,
                                     const faiss::ascend::AttrFilter *attrFilter, const uint8_t *extraMask,
                                     const float *queryFeatures, const float16_t *extraScore, int64_t *labels,
                                     float *distances);
    APP_ERROR CopyDataToHost(std::vector<float16_t>& outDistances, int64_t *labels,
        AscendTensor<float16_t, DIMS_2>& outDistanceOnDevice, AscendTensor<int64_t, DIMS_2>& outIndicesOnDevice);
    APP_ERROR AddFeatureImpl(int64_t singleAdd, const float *features, int64_t offset, int64_t startOffset);
    std::vector<idx_t> ids;
    int code_size = 0;
    uint32_t deviceId = 0;
    std::vector<float> scale;
    AscendTensor<float16_t, DIMS_1> scaleReciprocal;
};
} // namespace ascend

#endif // ASCENDHOST_SRC_ASCENDFAISS_ASCENDHOST_INCLUDE_IMPL_TSFLATIP_H
