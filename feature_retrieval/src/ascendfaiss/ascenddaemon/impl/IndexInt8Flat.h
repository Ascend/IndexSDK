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


#ifndef INDEX_INT8_FLAT_INCLUDED
#define INDEX_INT8_FLAT_INCLUDED

#include <vector>

#include "ascenddaemon/impl/IndexInt8.h"
#include "ascenddaemon/impl/Int8L2Norm.h"
#include "ascenddaemon/utils/DeviceMemMng.h"
#include "ascenddaemon/utils/DeviceVector.h"
#include "ascendhost/include/impl/HeteroBlockGroupMgr.h"

namespace ascend {
namespace {
const int FLAT_DEFAULT_DIST_COMPUTE_BATCH = 16384 * 16;

const int BINARY_BYTE_SIZE = 8;
const int DEFAULT_PAGE_BLOCK_NUM = 16; // default block number of one page for single index search
const int FLAT_BURST_LEN = 64;
const int IDX_ACTUAL_NUM = 0;
const int IDX_COMP_OFFSET = 1;
const int IDX_MASK_LEN = 2;
const int IDX_USE_MASK = 3;
}

enum class Int8FlatIndexType {
    INT8_FLAT_L2 = 0,
    INT8_FLAT_COS,
    INT8_FLAT_APPROXL2
};

class IndexSearchContext {
public:
    IndexSearchContext(AscendTensor<int8_t, DIMS_2> &queries, int topN, AscendTensor<float16_t, DIMS_2> &minDistances,
        AscendTensor<int64_t, DIMS_2> &minIndices)
        : queries(queries), topN(topN), minDistances(minDistances), minIndices(minIndices)
        {}
    
    AscendTensor<int8_t, DIMS_2> &queries;
    int topN;
    AscendTensor<float16_t, DIMS_2> &minDistances;
    AscendTensor<int64_t, DIMS_2> &minIndices;
};

struct IndexSearchTensorShare {
    explicit IndexSearchTensorShare(AscendResourcesProxy &resources, IndexSearchContext &ctx,
        const HeteroBlockGroupMgr &grpSpliter, const IndexSchemaBase &indexSchema);

    ~IndexSearchTensorShare();

    AscendTensor<float16_t, DIMS_3, size_t> &getDistResult(size_t grpId);

    AscendTensor<float16_t, DIMS_3, size_t> &getMinDistResult(size_t grpId);
    
    size_t groupCount;
    std::shared_ptr<AscendTensor<float16_t, DIMS_1>> queriesNorm;
    std::shared_ptr<AscendTensor<uint32_t, DIMS_2>> actualNum;
    std::shared_ptr<AscendTensor<float16_t, DIMS_3, size_t>> distResult;
    std::shared_ptr<AscendTensor<float16_t, DIMS_3, size_t>> minDistResult;
    std::shared_ptr<AscendTensor<float16_t, DIMS_3, size_t>> lastDistResult;
    std::shared_ptr<AscendTensor<float16_t, DIMS_3, size_t>> lastMinDistResult;
    std::vector<std::shared_ptr<AscendTensor<uint16_t, DIMS_3>>> opFlagVec;
    std::vector<std::shared_ptr<AscendTensor<uint32_t, DIMS_3>>> opSizeVec;
    std::vector<std::shared_ptr<AscendTensor<int64_t, DIMS_1>>> attrsInputVec;
};


template<typename P>
class IndexInt8Flat : public IndexInt8, public IndexSchemaBase {
public:
    IndexInt8Flat(int dim, MetricType metric = MetricType::METRIC_L2,
        int64_t resourceSize = -1, int blockSize = FLAT_DEFAULT_DIST_COMPUTE_BATCH);

    ~IndexInt8Flat();

    APP_ERROR setHeteroParam(uint32_t deviceId, size_t deviceCapacity, size_t deviceBuffer, size_t hostCapacity);

    APP_ERROR tryToSychResultAdvanced(int &hasCopiedCount, int &indexDoneCount, int indexId, int n, int batchSize,
        int k, float16_t *distances, idx_t *labels, AscendTensor<float16_t, DIMS_3, size_t> &srcDistances,
        AscendTensor<idx_t, DIMS_3, size_t> &srcIndices);

    APP_ERROR addVectors(AscendTensor<int8_t, DIMS_2> &rawData) override;

    void resizeBaseShaped(int n);

    APP_ERROR getVectors(uint32_t offset, uint32_t num, std::vector<int8_t> &vectors);

    void reset() override;

    inline idx_t getSize() const override
    {
        return ntotal;
    }

    inline int getBlockSize() const override
    {
        return codeBlockSize;
    }

    inline int getBurstsOfBlock(void) const override
    {
        return burstsOfBlock;
    }

    inline const std::vector<std::unique_ptr<DeviceVector<int8_t>>> &getBaseShaped() const
    {
        return baseShaped;
    }

    inline const std::vector<std::unique_ptr<DeviceVector<P>>> &getNormBase() const
    {
        return normBase;
    }

    void getBaseEnd() override;

    void setPageSize(uint16_t pageBlockNum) override;

    std::vector<std::unique_ptr<DeviceVector<int8_t>>> baseShaped;

    std::vector<std::unique_ptr<DeviceVector<P>>> normBase;

protected:

    APP_ERROR searchImpl(int n, const int8_t *x, int k, float16_t *distances, idx_t *labels) override;

    APP_ERROR searchImpl(std::vector<IndexInt8 *> indexes, int n, int batchSize, const int8_t *x, int k,
        float16_t *distances, idx_t *labels) override;

    virtual APP_ERROR searchPaged(int pageId, AscendTensor<int8_t, DIMS_2> &queries, int k,
                          AscendTensor<float16_t, DIMS_2> &outDistance, AscendTensor<int64_t, DIMS_2> &outIndices,
                          AscendTensor<uint8_t, DIMS_2> &mask);

    virtual void runDistCompute(int batch, const std::vector<const AscendTensorBase *> &input,
        const std::vector<const AscendTensorBase *> &output, aclrtStream stream, uint32_t actualNum = 0) const = 0;

    void computeNorm(AscendTensor<int8_t, DIMS_2> &rawData);
    P ivecNormL2sqr(const int8_t *x, size_t d);
    void ivecNormsL2sqr(P *nr, const int8_t *x, size_t d, size_t nx);

    void moveNormForward(idx_t srcIdx, idx_t dstIdx);
    void moveShapedForward(idx_t srcIdx, idx_t dstIdx);

    inline void moveVectorForward(idx_t srcIdx, idx_t dstIdx)
    {
        moveNormForward(srcIdx, dstIdx);
        moveShapedForward(srcIdx, dstIdx);
    }

    size_t removeIdsImpl(const IDSelector &sel);
    size_t removeIdsBatch(const std::vector<idx_t> &indices);
    size_t removeIdsRange(idx_t min, idx_t max);
    void removeInvalidData(int oldTotal, int remove);

    size_t calcShapedBaseSize(idx_t totalNum);
    size_t calcNormBaseSize(idx_t totalNum);

    void runTopkCompute(AscendTensor<float16_t, DIMS_3, size_t> &dists,
        AscendTensor<float16_t, DIMS_3, size_t> &mindists, AscendTensor<uint32_t, DIMS_3> &sizes,
        AscendTensor<uint16_t, DIMS_3> &flags, AscendTensor<int64_t, DIMS_1> &attrs,
        AscendTensor<float16_t, DIMS_2> &outdists, AscendTensor<int64_t, DIMS_2> &outlabel, aclrtStream stream);

    void runMultisearchTopkCompute(AscendTensor<float16_t, DIMS_3, size_t> &dists,
                                   AscendTensor<float16_t, DIMS_3, size_t> &maxDists,
                                   AscendTensor<uint32_t, DIMS_3> &sizes,
                                   AscendTensor<uint16_t, DIMS_3> &flags,
                                   AscendTensor<int64_t, DIMS_1> &attrs,
                                   AscendTensor<uint32_t, DIMS_1> &indexOffset,
                                   AscendTensor<uint32_t, DIMS_1> &pageOffset,
                                   AscendTensor<uint16_t, DIMS_1> &reorderFlag,
                                   AscendTensor<float16_t, DIMS_3, size_t> &outDists,
                                   AscendTensor<idx_t, DIMS_3, size_t> &outlabel,
                                   aclrtStream stream);

    APP_ERROR computeMultisearchTopkParam(AscendTensor<uint32_t, DIMS_1> &indexOffsetInputs,
        AscendTensor<uint32_t, DIMS_1> &labelOffsetInputs, AscendTensor<uint16_t, DIMS_1> &reorderFlagInputs,
        std::vector<idx_t> &ntotals, std::vector<idx_t> &offsetBlocks) const;

    APP_ERROR resetTopkCompOp();
    APP_ERROR resetMultisearchTopkCompOp();

    // get the capacity for DeviceVector, it is only used to the first DeviceVector of `codes`
    // 1. if size * dim <= 512k, capacity=512k
    // 2. if size * dim > `devVecCapacity`, capacity=`devVecCapacity`
    // 3. otherwise: capacity=2 * size * dim
    size_t getVecCapacity(size_t vecNum, size_t size) const;

    APP_ERROR getVectorsAiCpu(uint32_t offset, uint32_t num, std::vector<int8_t> &vectors);
    APP_ERROR copyAndSaveVectors(size_t startOffset, AscendTensor<int8_t, DIMS_2> &rawData);

protected:
    int codeBlockSize = FLAT_DEFAULT_DIST_COMPUTE_BATCH;                                     // the size of codes block
    int blockMaskSize = FLAT_DEFAULT_DIST_COMPUTE_BATCH / BINARY_BYTE_SIZE;
    int devVecCapacity = 0;
    int pageSize; // pageSize for single index search
    int burstsOfBlock;
    DeviceMemMng deviceMemMng;
    // aicpu op for topk computation
    std::map<int, std::unique_ptr<::ascend::AscendOperator>> topkComputeOps;
    Int8FlatIndexType int8FlatIndexType { Int8FlatIndexType::INT8_FLAT_L2 };
    std::unique_ptr<Int8L2Norm> int8L2Norm;
    int flagNum { 0 };
    bool isNeedCleanMinDist { false };
private:
    APP_ERROR addVectorsAicpu(size_t ntotal, AscendTensor<int8_t, DIMS_2> &rawData);
    APP_ERROR addVectors(AscendTensor<int8_t, DIMS_2> &rawData,
        int num, int dim, int vecSize, int addVecNum);
    APP_ERROR initResult(AscendTensor<float16_t, DIMS_3, size_t> &distances,
        AscendTensor<idx_t, DIMS_3, size_t> &indices) const;

    DeviceVector<int8_t, ExpandPolicySlim> dataVec;
    DeviceVector<int64_t, ExpandPolicySlim> attrsVec;
};
} // namespace ascend

#endif // INDEX_INT8_FLAT_INCLUDED
