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


#ifndef ASCEND_INDEX_IVF_SP_SQ_BASE_INCLUDED
#define ASCEND_INDEX_IVF_SP_SQ_BASE_INCLUDED

#include <vector>
#include <memory>

#include <ascenddaemon/impl/IndexIVF.h>
#include <ascenddaemon/utils/TopkOp.h>
#include <ascenddaemon/utils/AscendOperator.h>
#include <ascenddaemon/utils/DeviceVector.h>
#include <ascenddaemon/utils/DeviceVectorInl.h>
#include <ascenddaemon/utils/AscendTensor.h>
#include <common/threadpool/AscendThreadPool.h>

#include "ascenddaemon/utils/BatchQueueItem.h"
#include "ascenddaemon/IVFSPCodeBookTrainer.h"

namespace ascendSearch {
class IndexIVFSPSQ : public IndexIVF {
public:
    IndexIVFSPSQ(int dim, int dim2, int nlist, bool encodeResidual, int nprobes,
                 int searchListSize, int handleBatch, bool filterable,
                 int64_t resourceSize = -1);
    ~IndexIVFSPSQ();

    APP_ERROR trainCodeBook(IVFSPCodeBookTrainerInitParam &initParam, float *codebookPtr = nullptr) const;
    void addFinishMerge();
    APP_ERROR addDynamicUpdate(int listId);
    std::vector<size_t> bucketSize;
    std::vector<bool> isEmptyList;

    virtual APP_ERROR addVectors(int listId, size_t numVecs, const uint8_t *codes,
                                 const idx_t *indices, const float *preCompute, bool useNPU = false);

    APP_ERROR addVectorsAiCpu(int listId, AscendTensor<uint8_t, DIMS_2> &codesData);
    APP_ERROR addVectorsCtrlCpu(int listId, AscendTensor<uint8_t, DIMS_2> &codes);

    virtual void updateTrainedValue(AscendTensor<float16_t, DIMS_1> &trainedMin,
                                    AscendTensor<float16_t, DIMS_1> &trainedDiff);

    virtual APP_ERROR searchFilterImpl(int n, const float16_t *x, int k, float16_t *distances,
                                       idx_t *labels, uint32_t filterSize, uint32_t* filters);

    virtual APP_ERROR searchFilterImpl(std::vector<Index *> indexes, int n, int batchSize, const float16_t *x, int k,
                                       float16_t *distances, idx_t *labels, uint32_t filterSize, uint32_t* filters);

    virtual APP_ERROR searchFilterImpl(std::vector<Index *> indexes, int n, int batchSize, const float16_t *x, int k,
                                       float16_t *distances, idx_t *labels, uint32_t filterSize, uint32_t** filters);

    virtual APP_ERROR getCodeWord(int n, float *feature, float16_t *codeWord, idx_t *labels);

    APP_ERROR reset();
    APP_ERROR searchImpl(int n, const float16_t *x, int k, float16_t *distances, idx_t *labels) override;

    void updateCoarseCentroidsData(AscendTensor<float16_t, DIMS_2> &coarseCentroidsData) override;

    void updateCoarseCentroidsData(const IndexIVFSPSQ* loadedIndex);

    int  getShapedDataOffset(int idx) const;

    APP_ERROR loadDeviceAllData(const char *dataFile, float* codebookPtr, float* spsqPtr,
        const IndexIVFSPSQ* loadedIndex = nullptr);

    APP_ERROR loadDeviceAllData(const uint8_t* data, size_t dataLen, float* codebookPtr, float* spsqPtr,
        const IndexIVFSPSQ* loadedIndex = nullptr);

    APP_ERROR saveDeviceAllData(const char *dataFile, float* codebookPtr, float* spsqPtr);

    APP_ERROR saveDeviceAllData(uint8_t* &data, size_t &dataLen, float* codebookPtr, float* spsqPtr);

    APP_ERROR saveCodeBook(uint8_t*& data, size_t& dataLen, float* codebookPtr);

    APP_ERROR loadCodeBook(const uint8_t* data, size_t dataLen, float* codebookPtr);

    virtual int getCodeBookSize()
    {
        return 0;
    }

    inline int getSize() const
    {
        return ntotal;
    }

    inline int getDim() const
    {
        return dims;
    }

    inline int getBlockSize() const
    {
        return blockSize;
    }

    inline const std::vector<std::unique_ptr<DeviceVector<uint8_t>>> &getBaseShaped() const
    {
        return baseShaped;
    }

    size_t removeIds(const ::ascendSearch::IDSelector& sel);

    bool isAddFinish();
    
public:
    AscendTensor<float16_t, DIMS_1> vMin;
    AscendTensor<float16_t, DIMS_1> vDiff;
    AscendTensor<float16_t, DIMS_2> vDM;
    std::vector<int> addBatchSizes;

    int dim2;
    int blockSize;
    int devVecCapacity;
    int burstsOfBlock;
    int searchListSize;
    int handleBatch;
    bool filterable;
    std::vector<uint8_t> addFinishFlag;
    float* pListPreNorms;
    bool codebookFinished = false;

    AscendTensor<uint16_t, DIMS_2> vand;
    AscendTensor<float16_t, DIMS_2> vmul;

    std::vector<std::unique_ptr<DeviceVector<uint8_t>>> cidIdx;
    std::vector<std::unique_ptr<DeviceVector<uint32_t>>> cidVal;
    std::vector<std::unique_ptr<DeviceVector<uint32_t>>> timestamps;

    uint8_t* pCidIdxBase;
    uint32_t* pCidValBase;
    uint32_t* pTsBase;

    std::vector<uint64_t> idxOffset;
    std::vector<uint64_t> valOffset;
    std::vector<uint64_t> tsOffset;

    std::vector<std::unique_ptr<DeviceVector<float>>> preComputeData;
    std::vector<std::unique_ptr<DeviceVector<uint8_t>>> baseShaped;
    std::vector<std::unique_ptr<DeviceVector<uint8_t>>> deviceAllData;
    std::shared_ptr<AscendTensor<float16_t, DIMS_4>> coarseCentroidsShaped;
    std::shared_ptr<AscendTensor<float16_t, DIMS_1>> normCoarseCentroids;
    std::shared_ptr<AscendTensor<float16_t, DIMS_2>> coarseCentroidsIVFSPSQ;

    std::unique_ptr<AscendThreadPool> threadPool;
    std::unique_ptr<AscendOperator> distSqOp;
    std::unique_ptr<AscendOperator> cidFilterOp;
    std::unique_ptr<AscendOperator> matmulOp;
    std::map<int, std::unique_ptr<AscendOperator>> fpToFp16Ops;
    std::vector<std::vector<BatchQueueItem>> topkQueue;
    APP_ERROR resetFpToFp16Op();
    void runFpToFp16(AscendTensor<float, DIMS_2> &queryVecs,
        AscendTensor<float16_t, DIMS_2> &outQueryVecs,
        AscendTensor<uint16_t, DIMS_2> &flag, aclrtStream stream);
    void addCoarseCentroidsAiCpu(AscendTensor<float16_t, DIMS_2> &src,
        AscendTensor<float16_t, DIMS_4> &dst) override;
    void fvecNormsL2sqrAicpu(AscendTensor<float16_t, DIMS_1> &nr,
        AscendTensor<float16_t, DIMS_2> &x) override;

protected:
    APP_ERROR computeMask(int n, uint32_t* filters, AscendTensor<uint8_t, DIMS_1>& masks,
        AscendTensor<int, DIMS_1> &listId);

    APP_ERROR resetCidFilterOperator();

    APP_ERROR searchBatched(int n, const float16_t *x, int k, float16_t *distance, idx_t *labels,
        uint64_t filterSize, uint32_t* filters) override;

    virtual APP_ERROR searchBatched(std::vector<Index *> indexes, int n, const float16_t *x, int k,
                                    float16_t *distances, idx_t *labels, uint32_t filterSize, uint32_t *filters);
    virtual APP_ERROR searchBatched(std::vector<Index *> indexes, int n, const float16_t *x, int k,
                                    float16_t *distances, idx_t *labels, uint32_t filterSize, uint32_t **filters);
    virtual APP_ERROR searchImpl(AscendTensor<float16_t, DIMS_2> &queries, int k,
        AscendTensor<float16_t, DIMS_2> &outDistance, AscendTensor<idx_t, DIMS_2> &outIndices) = 0;

    void moveVectorForward(idx_t srcIdx, idx_t dstIdx) override;
    void releaseUnusageSpace(int oldTotal, int remove) override;

    void runCidFilterOperator(AscendTensor<uint8_t, DIMS_1>& idx,
                              AscendTensor<int32_t, DIMS_1>& val,
                              AscendTensor<int32_t, DIMS_1>& ts,
                              AscendTensor<uint64_t, DIMS_2>& offset,
                              AscendTensor<uint32_t, DIMS_1>& bucketSizes,
                              AscendTensor<int32_t, DIMS_2>& maskFilter,
                              AscendTensor<int32_t, DIMS_1>& timeFilter,
                              AscendTensor<uint16_t, DIMS_2>& andData,
                              AscendTensor<float16_t, DIMS_2>& mulData,
                              AscendTensor<uint16_t, DIMS_2>& result,
                              AscendTensor<uint16_t, DIMS_2>& flag,
                              aclrtStream stream);
};
} // namespace ascendSearch

#endif // ASCEND_INDEX_IVF_SP_SQ_BASE_INCLUDED
