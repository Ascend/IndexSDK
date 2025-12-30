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


#ifndef ASCEND_INDEXSQ_INCLUDED
#define ASCEND_INDEXSQ_INCLUDED

#include <memory>

#include "ascenddaemon/impl/Index.h"
#include "ascenddaemon/utils/AscendTensor.h"
#include "ascenddaemon/utils/DeviceVector.h"
#include "ascenddaemon/utils/AscendOperator.h"

namespace ascend {
namespace {
const int ID_BLOCKS = 4;
const int MASK_SIZE = 64;
const int TS_SIZE = 8;
const int HELPER_SIZE = 128;
const int FILTER_SIZE = 6;
const int OFFSET_BASE = 1000;
const int OFFSET_DIV = 16384 * 8;
const int SQ_DEFAULT_DIST_COMPUTE_BATCH = 16384 * 16;
const int BURST_LEN = 64;
const int IDX_ACTUAL_NUM = 0;
const int IDX_COMP_OFFSET = 1; // block offset
const int IDX_MASK_LEN = 2;
const int PAGE_BLOCKS = 32;
}

class IndexSQ : public Index {
public:
    IndexSQ(int dim, bool filterable, int64_t resource = -1, int dBlockSize = SQ_DEFAULT_DIST_COMPUTE_BATCH);

    ~IndexSQ();

    APP_ERROR init() override;

    APP_ERROR reset() override;

    virtual APP_ERROR addVectors(size_t numVecs, const uint8_t *data, const float *preCompute);

    APP_ERROR addVectorsWithIds(size_t numVecs, const uint8_t *data, const Index::idx_t* ids, const float *preCompute);

    inline int getSize() const
    {
        return ntotal;
    }

    void getBaseEnd();

    APP_ERROR getVectors(uint32_t offset, uint32_t num, std::vector<uint8_t> &vectors);
    APP_ERROR getVectors(uint32_t offset, uint32_t num, uint8_t *vectors);

    inline int getDim() const
    {
        return dims;
    }

    inline uint32_t getSendBatch() const
    {
        return SEND_BATCH;
    }

    inline int getBlockSize() const
    {
        return codeBlockSize;
    }

    inline const std::vector<std::unique_ptr<DeviceVector<uint8_t>>> &getCodes() const
    {
        return codes;
    }

    inline const std::vector<std::unique_ptr<DeviceVector<float>>> &getPreCompute() const
    {
        return preCompute;
    }

    void updateTrainedValue(AscendTensor<float16_t, DIMS_1> &trainedMin,
                            AscendTensor<float16_t, DIMS_1> &trainedDiff);

    AscendTensor<float16_t, DIMS_1> vMin;
    AscendTensor<float16_t, DIMS_1> vDiff;

    // base data
    std::vector<std::unique_ptr<DeviceVector<uint8_t>>> codes;
    // precompute Data list
    std::vector<std::unique_ptr<DeviceVector<float>>> preCompute;

protected:
    APP_ERROR searchBatched(int64_t n, const float16_t* x, int64_t k, float16_t* distance, idx_t* labels,
        uint64_t filterSize, uint32_t* filters) override;

    APP_ERROR searchBatched(int64_t n, const float16_t *x, int64_t k, float16_t *distance, idx_t *labels,
        uint8_t* masks) override;

    APP_ERROR searchBatched(std::vector<Index *> indexes, int64_t n, const float16_t *x, int64_t k,
        float16_t *distances, idx_t *labels, uint32_t filterSize, std::vector<void *> &filters,
        bool isMultiFilterSearch) override;
 
    APP_ERROR computeMask(std::vector<Index *> &indexes, int n, const std::vector<void *> &filters,
        AscendTensor<uint8_t, DIMS_2, int64_t> &maskData, AscendTensor<int, DIMS_1> &maskOffset,
        bool isMultiFilterSearch);

    // get the capacity for DeviceVector, it is only used to the first DeviceVector of `codes`
    // 1. if size * dim <= 512k, capacity=512k
    // 2. if size * dim > `devVecCapacity`, capacity=`devVecCapacity`
    // 3. otherwise: capacity=2 * size * dim
    size_t getVecCapacity(size_t vecNum, size_t size) const;

    APP_ERROR addIds(size_t numVecs, const Index::idx_t* ids);

    APP_ERROR saveIds(int numVecs, const Index::idx_t *ids);

    void moveVectorForward(idx_t srcIdx, idx_t destIdx) override;
    void releaseUnusageSpace(int oldTotal, int remove) override;

    APP_ERROR computeMask(int n, uint32_t* filters, AscendTensor<uint8_t, DIMS_1, size_t>& masks);

    APP_ERROR resetCidFilterOperator() const;

    int getLastCidBlockSize(int cidNum) const;

    void runCidFilterOperator(int batch,
                              const std::vector<const AscendTensorBase *> &input,
                              const std::vector<const AscendTensorBase *> &output,
                              aclrtStream stream) const;

    APP_ERROR resetTopkCompOp() const;

    void runTopkCompute(int batch,
                        const std::vector<const AscendTensorBase *> &input,
                        const std::vector<const AscendTensorBase *> &output,
                        aclrtStream stream) const;
    
    APP_ERROR resetMultisearchTopkCompOp() const;

    void runMultisearchTopkCompute(int batch,
                                   const std::vector<const AscendTensorBase *> &input,
                                   const std::vector<const AscendTensorBase *> &output,
                                   aclrtStream stream) const;

    APP_ERROR computeMultisearchTopkParam(AscendTensor<uint32_t, DIMS_1> &indexOffsetInputs,
        AscendTensor<uint32_t, DIMS_1> &labelOffsetInputs, AscendTensor<uint16_t, DIMS_1> &reorderFlagInputs,
        std::vector<idx_t> &ntotals, std::vector<idx_t> &offsetBlocks) const;

    APP_ERROR resetSqDistOperator(std::string opTypeName, IndexTypeIdx indexType) const;
    APP_ERROR resetSqDistMaskOperator(std::string opTypeName, IndexTypeIdx indexMaskType) const;

    static const uint32_t SEND_BATCH = 16384;          // sq send batch for each channel

    int codeBlockSize;                                     // the size of codes block
    int computeBlockSize;
    int cidBlockSize;
    int devVecCapacity;
    int burstsOfBlock;
    bool filterable;

    // helper opertar compute data
    AscendTensor<uint16_t, DIMS_2> vand;
    AscendTensor<float16_t, DIMS_2> vmul;

    // cid data
    std::vector<std::unique_ptr<DeviceVector<uint8_t>>> cidIdx;

    std::vector<std::unique_ptr<DeviceVector<uint32_t>>> cidVal;

    std::vector<std::unique_ptr<DeviceVector<uint32_t>>> timestamps;

    int pageSize;   // As the base data is too large, the memory pool is not enough

private:
    APP_ERROR saveIdsHostCpu(int numVecs, const Index::idx_t *ids, int vecBefore, int dVecBefore);

    APP_ERROR getVectorsAiCpu(uint32_t offset, uint32_t num, uint8_t *vectors);
    DeviceVector<uint8_t, ExpandPolicySlim> dataVec;
    DeviceVector<int64_t, ExpandPolicySlim> attrsVec;

    APP_ERROR getMaskAndTsFilter(int n, size_t indexSize, const std::vector<void *> &filters,
        bool isMultiFilterSearch,
        std::vector<std::vector<std::vector<uint32_t>>> &maskFilters,
        std::vector<std::vector<std::vector<uint32_t>>> &tsFilters) const;

    void getSingleFilter(int n, uint32_t *filter,
        std::vector<std::vector<std::vector<uint32_t>>> &maskFilters,
        std::vector<std::vector<std::vector<uint32_t>>> &tsFilters) const;

    APP_ERROR getMultiFilter(int n, size_t indexSize, const std::vector<void *> &filters,
        std::vector<std::vector<std::vector<uint32_t>>> &maskFilters,
        std::vector<std::vector<std::vector<uint32_t>>> &tsFilters) const;

    void parseOneFilter(const AscendTensor<uint32_t, DIMS_2> &oneFilter, int offset,
        std::vector<uint32_t> &oneMaskFilter, std::vector<uint32_t> &oneTsFilter) const;

    void moveTSAttrForward(size_t srcIdx, size_t dstIdx);

    void releaseAttrUnusageSpace(size_t originNum, size_t removeNum);

    template <typename T>
    void moveAttrForward(size_t srcIdx, size_t dstIdx, const std::vector<std::unique_ptr<DeviceVector<T>>> &attr,
        size_t blockSize)
    {
        size_t srcBlockIdx = srcIdx / blockSize;
        size_t srcBlockOffset = srcIdx % blockSize;
        size_t dstBlockIdx = dstIdx / blockSize;
        size_t dstBlockOffset = dstIdx % blockSize;
        size_t attrSize = attr.size();
        bool blkIdxValid = (srcBlockIdx < attrSize) && (dstBlockIdx < attrSize);
        ASCEND_THROW_IF_NOT_FMT(blkIdxValid, "invalid idx, srcBlkIdx[%zu] dstBlkIdx[%zu] attrSize[%zu]",
            srcBlockIdx, dstBlockIdx, attrSize);

        size_t srcBlockSize = attr[srcBlockIdx]->size();
        size_t dstBlockSize = attr[dstBlockIdx]->size();
        bool blkOffValid = (srcBlockOffset < srcBlockSize) && (dstBlockOffset < dstBlockSize);
        ASCEND_THROW_IF_NOT_FMT(blkOffValid,
            "invalid offset, src:blkOff[%zu] blkIdx[%zu] blkSize[%zu] dst:blkOff[%zu] blkIdx[%zu] blkSize[%zu]",
            srcBlockOffset, srcBlockIdx, srcBlockSize, dstBlockOffset, dstBlockIdx, dstBlockSize);

        auto err = aclrtMemcpy(attr[dstBlockIdx]->data() + dstBlockOffset, (dstBlockSize - dstBlockOffset) * sizeof(T),
            attr[srcBlockIdx]->data() + srcBlockOffset, sizeof(T), ACL_MEMCPY_DEVICE_TO_DEVICE);
        ASCEND_THROW_IF_NOT_FMT(err == EOK, "Memcpy error %d", err);
    }
};
} // namespace ascend

#endif // ASCEND_INDEXSQ_INCLUDED
