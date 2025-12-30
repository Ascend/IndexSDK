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

#ifndef TS_BINARY_FLAT_INCLUDED
#define TS_BINARY_FLAT_INCLUDED

#include "ascendhost/include/impl/AscendIndexBinaryFlatImpl.h"
#include "ascendhost/include/impl/TSBase.h"

namespace ascend {
namespace {
constexpr int64_t BINARY_FLAT_DEFAULT_MEM = 0x60000000; // 1.5G default min memory resource by batch=256
} // namespace
class TSBinaryFlat : public TSBase, public faiss::ascend::AscendIndexBinaryFlatImpl {
public:
    TSBinaryFlat(int deviceId, uint32_t dim, uint32_t tokenNum, uint64_t resources, uint32_t customAttrLen,
        uint32_t customAttrBlockSize);
    ~TSBinaryFlat() = default;

    APP_ERROR addFeatureWithLabels(int64_t n, const void *features,
        const faiss::ascend::FeatureAttr *attrs, const int64_t *labels, const uint8_t *customAttr,
        const faiss::ascend::ExtraValAttr *val) override;
    APP_ERROR delFeatureWithLabels(int64_t n, const int64_t *labels) override;
    APP_ERROR getFeatureByLabel(int64_t n, const int64_t *labels, void *features) const override;
    APP_ERROR getFeatureAttrsByLabel(int64_t n, const int64_t *labels,
        faiss::ascend::FeatureAttr *attrs) const override;
    APP_ERROR deleteFeatureByToken(int64_t count, const uint32_t *tokens) override;
    APP_ERROR search(uint32_t count,
                     const void *features,
                     const faiss::ascend::AttrFilter *attrFilter,
                     bool shareAttrFilter,
                     uint32_t topk,
                     int64_t *labels,
                     float *distances,
                     uint32_t *validNums,
                     bool enableTimeFilter,
                     const faiss::ascend::ExtraValFilter *extraValFilter) override;
    APP_ERROR searchWithExtraMask(uint32_t count,
                                  const void *features,
                                  const faiss::ascend::AttrFilter *attrFilter,
                                  bool shareAttrFilter,
                                  uint32_t topk,
                                  const uint8_t *extraMask,
                                  uint64_t extraMaskLen,
                                  bool extraMaskIsAtDevice,
                                  int64_t *labels,
                                  float *distances,
                                  uint32_t *validNums,
                                  bool enableTimeFilter,
                                  const float16_t *extraScore);
    APP_ERROR getBaseByRange(uint32_t offset, uint32_t num, int64_t *labels, void *features,
        faiss::ascend::FeatureAttr *attributes, faiss::ascend::ExtraValAttr *extraVal);
    
    APP_ERROR getExtraValAttrsByLabel(int64_t n, const int64_t *labels, faiss::ascend::ExtraValAttr *extraVal) const;
    
protected:
    std::map<int, std::unique_ptr<AscendOperator>> distComputeShareMaskOps;
    std::map<int, std::unique_ptr<AscendOperator>> distComputeNonshareMaskOps;
    bool shareAttrFilter = false;

private:
    inline void setFilter(bool shareAttrFilter, bool enableTimeFilter,
        const faiss::ascend::ExtraValFilter *extraValFilter)
    {
        this->shareAttrFilter = shareAttrFilter;
        this->enableTimeFilter = enableTimeFilter;
        this->enableValFilter = (extraValFilter != nullptr);
    }
    void queryVectorByIdx(int64_t idx, uint8_t *x) const;
    void resetMaskDistCompOp();
    void runDistMaskCompute(int batch, bool shareMask, const std::vector<const AscendTensorBase *> &input,
        const std::vector<const AscendTensorBase *> &output, aclrtStream stream);
    void searchPagedWithMasks(int pageIdx,
                            int batch,
                            const uint8_t *x,
                            int topK,
                            AscendTensor<uint8_t, DIMS_3> &masks,
                            AscendTensor<float16_t, DIMS_2> &outDistanceOnDevice,
                            AscendTensor<int64_t, DIMS_2> &outIndicesOnDevice);
    void searchBatchWithShareMasks(int batch, const uint8_t *x, int topK, float *distances, int64_t *labels,
                            AscendTensor<uint8_t, DIMS_3> &masks);
    void searchBatchWithNonshareMasks(int batch, const uint8_t *x, int topK, float *distances, int64_t *labels,
        AscendTensor<int32_t, DIMS_2> &queryTimes, AscendTensor<uint8_t, DIMS_2> &tokenIds,
        AscendTensor<int16_t, DIMS_2> &valFilter);
    void generateMaskExtraVal(int batch, int blockOffset, int blockNum,
        AscendTensor<int32_t, DIMS_2> &queryTimes, AscendTensor<uint8_t, DIMS_2> &tokenIds,
        AscendTensor<int16_t, DIMS_2> &valFilter, AscendTensor<uint8_t, DIMS_3> &masks);
    void searchBatchWithExtraNonshareMasks(int batch, const uint8_t *x, int topK, float *distances, int64_t *labels,
        AscendTensor<int32_t, DIMS_2> &queryTimes, AscendTensor<uint8_t, DIMS_2> &tokenIds, const uint8_t *extraMask);
    void postProcess(int64_t searchNum, int topK, AscendTensor<float16_t, DIMS_2> &outDistanceOnDevice,
        AscendTensor<int64_t, DIMS_2> &outIndicesOnDevice, float *distances, int64_t *labels);
    void getValidNum(uint64_t count, uint32_t topk, int64_t *labels, uint32_t *validNums) const;
    void removeLabels(const std::vector<int64_t> &removeIds);
    void setSearchWithExtraMaskAttr(bool shareAttrFilter, bool extraMaskIsAtDevice,
                                    uint64_t extraMaskLen, bool enableTimeFilter);
    void buildAttrWithExtraVal(const faiss::ascend::AttrFilter *attrFilter,
        const faiss::ascend::ExtraValFilter *extraValFilter, int batch,
        AscendTensor<int32_t, DIMS_2> &queryTime, AscendTensor<uint8_t, DIMS_2> &tokenIds,
        AscendTensor<int16_t, DIMS_2> &valFilter);
    
    int64_t getLabelsInIds(int64_t n, const int64_t *labels) const;

    bool isFirstUseExtraVal = false;
    std::once_flag firstAddOnceFlag;
};
} // namespace ascend

#endif