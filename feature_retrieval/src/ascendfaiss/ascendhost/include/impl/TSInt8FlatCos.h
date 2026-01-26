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


#ifndef ASCENDHOST_SRC_ASCENDFAISS_ASCENDHOST_INCLUDE_IMPL_TSINT8FLATCOS_H
#define ASCENDHOST_SRC_ASCENDFAISS_ASCENDHOST_INCLUDE_IMPL_TSINT8FLATCOS_H
#include "IndexInt8FlatCosAicpu.h"
#include "TSBase.h"
namespace ascend {
namespace {
// The value range of dim
const std::vector<int> DIMS = {64, 128, 256, 384, 512, 768, 1024};
} // namespace
class TSInt8FlatCos : public TSBase, public IndexInt8FlatCosAicpu {
public:
    TSInt8FlatCos(uint32_t deviceId, uint32_t dim, uint32_t tokenNum, uint64_t resources, bool useHmm,
        uint32_t customAttrLen, uint32_t customAttrBlockSize);
    ~TSInt8FlatCos() = default;

    APP_ERROR delFeatureWithLabels(int64_t n, const int64_t *labels) override;
    APP_ERROR addFeatureWithLabels(int64_t n, const void *features,
        const faiss::ascend::FeatureAttr *attrs, const int64_t *labels, const uint8_t *customAttr,
        const faiss::ascend::ExtraValAttr *val) override;
    APP_ERROR deleteFeatureByToken(int64_t count, const uint32_t *tokens) override;
    APP_ERROR getFeatureByLabel(int64_t n, const int64_t *labels, void *features) const override;
    APP_ERROR search(uint32_t count, const void *features, const faiss::ascend::AttrFilter *attrFilter,
                     bool shareAttrFilter, uint32_t topk, int64_t *labels, float *distances,
                     uint32_t *validNums, bool enableTimeFilter,
                     const faiss::ascend::ExtraValFilter *extraValFilter) override;
    APP_ERROR searchWithExtraMask(uint32_t count, const void *features, const faiss::ascend::AttrFilter *attrFilter,
                                  bool shareAttrFilter, uint32_t topk, const uint8_t *extraMask, uint64_t extraMaskLen,
                                  bool extraMaskIsAtDevice, int64_t *labels, float *distances, uint32_t *validNums,
                                  bool enableTimeFilter, const float16_t *extraScore) override;
    APP_ERROR searchBatched(int n, const int8_t *x, const faiss::ascend::AttrFilter *attrFilter, int k,
                            float16_t *distance, idx_t *labels, const uint8_t *extraMask, uint64_t extraMaskLen,
                            bool extraMaskIsAtDevice);
    APP_ERROR getBaseByRange(uint32_t offset, uint32_t num, int64_t *labels, void *features,
        faiss::ascend::FeatureAttr *attributes, faiss::ascend::ExtraValAttr *extraVal);
    APP_ERROR getFeatureAttrsByLabel(int64_t n, const int64_t *labels,
        faiss::ascend::FeatureAttr *attrs) const override;
    APP_ERROR getExtraValAttrsByLabel(int64_t n, const int64_t *labels, faiss::ascend::ExtraValAttr *extraVal) const;
    APP_ERROR AddFeatureByIndice(int64_t n, const void *features,
        const faiss::ascend::FeatureAttr *attrs, const int64_t *indices, const uint8_t *customAttr,
        const faiss::ascend::ExtraValAttr *extraValAttr);
    APP_ERROR GetFeatureByIndice(int64_t count, const int64_t *indices, int64_t *labels,
        void *features, faiss::ascend::FeatureAttr *attributes, faiss::ascend::ExtraValAttr *extraVal) const;
    APP_ERROR AddFeatureWithIndice(int64_t n, int64_t replaceNum, const int64_t *indices,
    const std::vector<std::pair<int64_t, int64_t>> &segments, const int8_t *features);

private:
    APP_ERROR searchPagedWithMasks(int pageIdx, int batch, const int8_t *features, int topK,
        AscendTensor<uint8_t, DIMS_3> &masks, AscendTensor<float16_t, DIMS_2> &outDistanceOnDevice,
        AscendTensor<int64_t, DIMS_2> &outIndicesOnDevice, const float16_t *extraScore = nullptr);
    APP_ERROR searchBatchWithShareMasks(int batch, const int8_t *features, int topK, float *distances, int64_t *labels,
        AscendTensor<uint8_t, DIMS_3> &masks, const float16_t *extraScore = nullptr);
    APP_ERROR searchBatchWithNonshareMasks(int batch, const int8_t *features, int topK, float *distances,
        int64_t *labels, AscendTensor<int32_t, DIMS_2> &queryTimes, AscendTensor<uint8_t, DIMS_2> &tokenIds,
        AscendTensor<int16_t, DIMS_2> &valFilter, const float16_t *extraScore = nullptr);
    APP_ERROR searchBatchWithExtraNonshareMasks(int batch, const int8_t *features, int topK, float *distances,
        int64_t *labels, AscendTensor<int32_t, DIMS_2> &queryTimes, AscendTensor<uint8_t, DIMS_2> &tokenIds,
        const uint8_t *extraMask, const float16_t *extraScore = nullptr);
    APP_ERROR searchInPureDev(uint32_t count, const int8_t *features, const faiss::ascend::AttrFilter *attrFilter,
        uint32_t topk, int64_t *labels, float *distances, const faiss::ascend::ExtraValFilter *extraValFilter);
    APP_ERROR searchInPureDevWithExtraMask(uint32_t count, const int8_t *features,
        const faiss::ascend::AttrFilter *attrFilter, uint32_t topk, const uint8_t *extraMask, int64_t *labels,
        float *distances, const float16_t *extraScore = nullptr);
    APP_ERROR createMask(uint32_t count, const faiss::ascend::AttrFilter *attrFilter,
                         AscendTensor<uint8_t, DIMS_1> &genMasks);
    APP_ERROR createMask(uint32_t count, const faiss::ascend::AttrFilter *attrFilter, const uint8_t *extraMask,
                         uint64_t extraMaskLen, bool extraMaskIsAtDevice, AscendTensor<uint8_t, DIMS_1> &genMasks);
    APP_ERROR getFeatureByLabelInOrder(int64_t n, const int64_t *labels, void *features) const;
    void add_with_ids(idx_t n, const int8_t *x, const idx_t *xids);
    void addWithIdsImpl(int n, const int8_t *x);
    void queryVectorByIdx(int64_t idx, uint8_t *dis) const;
    void postProcess(int64_t searchNum, int topK, float16_t *inDistances, float *outDistances, idx_t *inLabels,
                     int64_t *outLabels, uint32_t *validNums);
    void removeLabels(const std::vector<int64_t> &removeIds);
    void removeIdsImpl(const std::vector<int64_t> &indices);
    void removeFeatureByIds(const std::vector<int64_t> &ids);
    void postProcess(uint64_t searchNum, int topK, AscendTensor<float16_t, DIMS_2> &outDistanceOnDevice,
        AscendTensor<int64_t, DIMS_2> &outIndicesOnDevice, float *distances, int64_t *labels);
    void getValidNum(uint64_t count, uint32_t topk, int64_t *labels, uint32_t *validNums) const;
    void runInt8CosDistCompute(int batch, bool shareMask,
                            const std::vector<const AscendTensorBase *> &input,
                            const std::vector<const AscendTensorBase *> &output,
                            aclrtStream stream) const;
    void runAscendcInt8CosDistCompute(int batch, bool shareMask,
                            const std::vector<const AscendTensorBase *> &input,
                            const std::vector<const AscendTensorBase *> &output,
                            aclrtStream stream) const;
    void runInt8CosExtraScore(int batch, bool shareMask, const std::vector<const AscendTensorBase *> &input,
                              const std::vector<const AscendTensorBase *> &output, aclrtStream stream) const;
    APP_ERROR resetInt8CosDistCompute(int codeNum, bool shareMask) const;
    APP_ERROR resetAscendcInt8CosDistCompute(int codeNum) const;
    APP_ERROR resetInt8CosExtraScore(int codeNum, bool shareMask) const;
    int64_t getInt8LabelsInIds(int64_t offset, const int64_t *labels) const;

    void buildAttrWithExtraVal(const faiss::ascend::AttrFilter *attrFilter,
        const faiss::ascend::ExtraValFilter *extraValFilter, int batch,
        AscendTensor<int32_t, DIMS_2> &queryTime, AscendTensor<uint8_t, DIMS_2> &tokenIds,
        AscendTensor<int16_t, DIMS_2> &valFilter);
    
    void generateMaskExtraVal(int batch, int blockOffset, int blockNum,
        AscendTensor<int32_t, DIMS_2> &queryTimes, AscendTensor<uint8_t, DIMS_2> &tokenIds,
        AscendTensor<int16_t, DIMS_2> &valFilter, AscendTensor<uint8_t, DIMS_3> &masks);

    APP_ERROR copyExtraScoreToDev(AscendTensor<float16_t, DIMS_2, idx_t> &extraScoreTensor,
                                  const float16_t *extraScore, int count);
    void calOpSize(std::vector<uint32_t> &opSizeHost, int computeNum, int blockNum);
    APP_ERROR callSearchWithShareMaskByBatch(int32_t queryNum, const int8_t *features, uint32_t topk,
        int64_t *labels, float *distances, AscendTensor<uint8_t, DIMS_3> &masks);
    std::vector<idx_t> ids;
    int code_size = 0;
    uint32_t deviceId = 0;
    bool shareAttrFilter = false;
    idx_t ntotalPad = 0;

    bool isInt8FirstUseExtraVal = false;
    std::once_flag int8FirstAddOnceFlag;
};

} // namespace ascend

#endif // ASCENDHOST_SRC_ASCENDFAISS_ASCENDHOST_INCLUDE_IMPL_TSINT8FLATCOS_H
