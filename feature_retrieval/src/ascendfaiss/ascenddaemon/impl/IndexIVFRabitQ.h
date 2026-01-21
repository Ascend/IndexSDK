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

#ifndef ASCEND_INDEXIVFRABITQ_INCLUDED
#define ASCEND_INDEXIVFRABITQ_INCLUDED
#include "ascenddaemon/impl/IndexIVF.h"
#include "ascenddaemon/utils/AscendOperator.h"
#include "common/threadpool/AscendThreadPool.h"

namespace ascend {
namespace {
const int IVF_RABITQ_BURST_LEN = 64;
constexpr uint8_t IVF_RABITQ_BURST_LEN_LOW = 32;
constexpr uint8_t IVF_RABITQ_BURST_LEN_HIGH = 64;
constexpr uint8_t IVF_RABITQ_BURST_BLOCK_RATIO = 2;
constexpr uint8_t IVF_RABITQ_OPTIMIZE_BATCH_THRES = 48;
}

class IndexIVFRabitQ : public IndexIVF {
public:
    IndexIVFRabitQ(int numList, int dim, int nprobes, int64_t resourceSize = -1);

    ~IndexIVFRabitQ();

    APP_ERROR reset() override;

    APP_ERROR updateCoarseCenterImpl(std::vector<float> &centerData);

    APP_ERROR addVectors(int listId, size_t numVecs, const float *codes, const idx_t *indices);

    APP_ERROR searchImpl(int n, const float16_t* x, int k, float16_t* distances, idx_t* labels);

    APP_ERROR searchImpl(AscendTensor<float16_t, DIMS_2> &queries, int k,
        AscendTensor<float16_t, DIMS_2> &outDistance, AscendTensor<idx_t, DIMS_2> &outIndices);

    APP_ERROR searchPaged(size_t pageId, size_t pageNum, AscendTensor<float16_t, DIMS_2> &queries,
        AscendTensor<float16_t, DIMS_2> &maxDistances, AscendTensor<int64_t, DIMS_2> &maxIndices);
    
    APP_ERROR resize(int listId, size_t numVecs);

    size_t getListLength(int listId) const;
    std::unique_ptr<DeviceVector<float>> centroidsOnDevice;
    std::unique_ptr<DeviceVector<float>> OrthogonalMatrixOnDevice; // 旋转矩阵
    static int GetBurstsOfBlock(int nq, int blockSize, int &burstLen)
    {
        if (faiss::ascend::SocUtils::GetInstance().IsAscend910B()) {
            burstLen = IVF_RABITQ_BURST_LEN_HIGH;
        } else {
            burstLen = (nq > IVF_RABITQ_OPTIMIZE_BATCH_THRES) ? IVF_RABITQ_BURST_LEN_LOW : IVF_RABITQ_BURST_LEN_HIGH;
        }
        return utils::divUp(blockSize, burstLen) * IVF_RABITQ_BURST_BLOCK_RATIO;
    }
    APP_ERROR searchImpl(int n, const float* x, int k, float* distances, idx_t* labels, const float* srcIndexes);
protected:
    void uploadLUTMatrix();
    size_t addTiling(int listId, size_t numVecs,
                     std::vector<uint64_t>& offsetHost,
                     std::vector<uint64_t>& indexl2offsetHost,
                     std::vector<uint32_t>& baseSizeHost);
    APP_ERROR addCodes(int listId, AscendTensor<uint8_t, DIMS_2> &codesData);
    APP_ERROR resetCenterRotateL2Op();
    void runCenterRotateL2Op(AscendTensor<float, DIMS_2> &centroid,
                             AscendTensor<int32_t, DIMS_1> &vectorSize,
                             AscendTensor<float, DIMS_2> &matrix,
                             AscendTensor<float, DIMS_2> &rotateCentroid,
                             AscendTensor<float, DIMS_1> &centroidl2,
                             aclrtStream stream);
    APP_ERROR resetCenterLUTOp();
    void runCenterLUTOp(AscendTensor<float, DIMS_2> &centroid,
                        AscendTensor<float, DIMS_2> &matrix,
                        AscendTensor<float, DIMS_2> &centroidlut,
                        aclrtStream stream);
    APP_ERROR resetIndexRotateL2Op();
    void runIndexRotateL2Op(AscendTensor<float, DIMS_2> &index,
                            AscendTensor<int32_t, DIMS_1> &vectorSize,
                            AscendTensor<float, DIMS_2> &matrix,
                            AscendTensor<float, DIMS_2> &rotateIndex,
                            AscendTensor<float, DIMS_1> &indexl2,
                            aclrtStream stream);
    APP_ERROR resetIndexCodeAndPreComputeOp();
    void runIndexCodeAndPreComputeOp(AscendTensor<int32_t, DIMS_1> &vectorSize,
                                     AscendTensor<float, DIMS_2> &index,
                                     AscendTensor<float, DIMS_1> &indexl2,
                                     AscendTensor<float, DIMS_2> &centroid,
                                     AscendTensor<float, DIMS_1> &centroidl2,
                                     AscendTensor<uint8_t, DIMS_2> &indexCodes,
                                     AscendTensor<float, DIMS_1> &l2Result,
                                     AscendTensor<float, DIMS_1> &l1Result,
                                     aclrtStream stream);
    APP_ERROR resetQueryRotateOp();
    void runQueryRotateOp(int batch, AscendTensor<float, DIMS_2> &queries,
                          AscendTensor<float, DIMS_2> &matrix,
                          AscendTensor<float, DIMS_2> &rotateQueries,
                          aclrtStream stream);
    APP_ERROR resetQueryLUTOp();
    void runQueryLUTOp(int batch, AscendTensor<float, DIMS_2> &queries,
                       AscendTensor<float, DIMS_2> &matrix,
                       AscendTensor<float, DIMS_2> &querieslut,
                       aclrtStream stream);
    APP_ERROR resetL1TopkOp();
    void runL1TopkOp(AscendTensor<float, DIMS_2> &dists,
                     AscendTensor<float, DIMS_2> &vmdists,
                     AscendTensor<uint32_t, DIMS_2> &sizes,
                     AscendTensor<uint16_t, DIMS_2> &flags,
                     AscendTensor<int64_t, DIMS_1> &attrs,
                     AscendTensor<float, DIMS_2> &outdists,
                     AscendTensor<int64_t, DIMS_2> &outlabel,
                     aclrtStream stream);
    APP_ERROR resetL1DistOp();
    void runL1DistOp();
    APP_ERROR searchImplL1(AscendTensor<float, DIMS_2> &queries,
                           AscendTensor<float, DIMS_2> &rotateQueries,
                           AscendTensor<float, DIMS_2> &queriesLut,
                           AscendTensor<int64_t, DIMS_2> &l1TopNprobeIndicesHost,
                           AscendTensor<float, DIMS_2> &l1TopNprobeDistsHost);
    void runL1DistOp(int batch, AscendTensor<float, DIMS_2> &queries,
                     AscendTensor<float, DIMS_2> &centroidsDev, AscendTensor<float, DIMS_2> &dists,
                     AscendTensor<float, DIMS_2> &vmdists, AscendTensor<uint16_t, DIMS_2> &opFlag,
                     aclrtStream stream);
    bool l2DisOpReset(std::unique_ptr<AscendOperator> &op, int64_t batch);
    APP_ERROR resetL2DistOp();
    void resizeDistResult(size_t iterNum, size_t coreNum, size_t ivfRabitqBlockSize);
    APP_ERROR searchImplL2(AscendTensor<float, DIMS_2> &queries,
                           AscendTensor<float, DIMS_2> &queriesLut,
                           AscendTensor<int64_t, DIMS_2> &l1TopNprobeIndicesHost,
                           AscendTensor<float, DIMS_2> &l1TopNprobeDistsHost,
                           int k, float* distances, idx_t* labels);
    APP_ERROR searchWithBatch(int n, const float* x, int k, float* distances, idx_t* labels, const float* srcIndexes);
    void refine(int n, const float* x, int k, float* distances, idx_t* labels,
                float* topkdist, idx_t* topklabel, const float* srcIndexes);
    void runL2DistOp(AscendTensor<float, DIMS_2> &subQuery,
                     AscendTensor<float, DIMS_2> &subQuerylut,
                     AscendTensor<float, DIMS_2, size_t> &centroidslut,
                     AscendTensor<uint32_t, DIMS_1, size_t> &subQueryid,
                     AscendTensor<uint32_t, DIMS_1, size_t> &subCentroidsid,
                     AscendTensor<float, DIMS_1, size_t> &subCentroidsl2,
                     AscendTensor<uint8_t, DIMS_2, size_t> &codeVec,
                     AscendTensor<uint64_t, DIMS_1, size_t> &subOffset,
                     AscendTensor<uint32_t, DIMS_1, size_t> &subBaseSize,
                     AscendTensor<float, DIMS_1, size_t> &subIndexl2,
                     AscendTensor<float, DIMS_1, size_t> &subIndexl1,
                     AscendTensor<uint64_t, DIMS_1, size_t> &subIndexl2Offset,
                     AscendTensor<uint64_t, DIMS_1, size_t> &subIndexl1Offset,
                     AscendTensor<float, DIMS_2, size_t> &subDis,
                     AscendTensor<float, DIMS_2, size_t> &subVcMaxDis,
                     AscendTensor<uint16_t, DIMS_2, size_t> &subOpFlag,
                     aclrtStream stream);
    APP_ERROR resetL2TopkOp();
    void runL2TopkOp(int batch, AscendTensor<float, DIMS_2, size_t> &distResult,
                     AscendTensor<float, DIMS_2, size_t> &vmdistResult,
                     AscendTensor<int64_t, DIMS_2, size_t> &ids,
                     AscendTensor<uint32_t, DIMS_2, size_t> &sizes,
                     AscendTensor<uint16_t, DIMS_2, size_t> &flags,
                     AscendTensor<int64_t, DIMS_1> &attrs,
                     AscendTensor<float, DIMS_2, size_t> &outdists,
                     AscendTensor<uint64_t, DIMS_2, size_t> &outlabel,
                     aclrtStream stream);
    APP_ERROR fillDisOpInputData(int k, size_t batch, size_t segNum, size_t coreNum,
                                 AscendTensor<uint64_t, DIMS_2, size_t> &offset,
                                 AscendTensor<uint64_t, DIMS_2, size_t> &indexl2offset,
                                 AscendTensor<uint64_t, DIMS_2, size_t> &indexl1offset,
                                 AscendTensor<uint32_t, DIMS_2, size_t> &queryid,
                                 AscendTensor<uint32_t, DIMS_2, size_t> &centroidsid,
                                 AscendTensor<float, DIMS_2, size_t> &centroidsl2,
                                 AscendTensor<uint32_t, DIMS_2, size_t> &baseSize,
                                 AscendTensor<int64_t, DIMS_2, size_t> &ids,
                                 AscendTensor<int64_t, DIMS_1> &attrs,
                                 AscendTensor<int64_t, DIMS_2> &l1TopNprobeIndicesHost,
                                 AscendTensor<float, DIMS_2> &l1TopNprobeDistsHost);
    APP_ERROR fillL2TopkOpInputData(int k, size_t batch, size_t segNum, size_t coreNum,
                                  AscendTensor<int64_t, DIMS_1> &attrs);
    APP_ERROR fillL1TopkOpInputData(AscendTensor<int64_t, DIMS_1> &attrsInput);
    void fillDisOpInputDataByBlock(size_t batch, size_t segNum, size_t coreNum, size_t ivfFlatBlockSize,
                                   AscendTensor<uint32_t, DIMS_2, size_t> &queryidHostVec,
                                   AscendTensor<uint32_t, DIMS_2, size_t> &centroidsidHostVec,
                                   AscendTensor<float, DIMS_2, size_t> &centroidsl2HostVec,
                                   AscendTensor<uint32_t, DIMS_2, size_t> &baseSizeHostVec,
                                   AscendTensor<uint64_t, DIMS_2, size_t> &offsetHostVec,
                                   AscendTensor<uint64_t, DIMS_2, size_t> &indexl2OffsetHostVec,
                                   AscendTensor<uint64_t, DIMS_2, size_t> &indexl1OffsetHostVec,
                                   AscendTensor<int64_t, DIMS_2, size_t> &idsHostVec,
                                   AscendTensor<int64_t, DIMS_2> &l1TopNprobeIndicesHost,
                                   AscendTensor<float, DIMS_2> &l1TopNprobeDistsHost);
    void callL2DistanceOp(size_t batch, size_t segNum, size_t coreNum, size_t vcMaxLen,
                          AscendTensor<float, DIMS_2> &queryVec,
                          AscendTensor<float, DIMS_2> &queryLutVec,
                          AscendTensor<float, DIMS_2, size_t> &centroidsLutVec,
                          AscendTensor<uint32_t, DIMS_2, size_t> &queryid,
                          AscendTensor<uint32_t, DIMS_2, size_t> &centroidsid,
                          AscendTensor<float, DIMS_2, size_t> &centroidsl2,
                          AscendTensor<uint64_t, DIMS_2, size_t> &offset,
                          AscendTensor<uint32_t, DIMS_2, size_t> &baseSize,
                          AscendTensor<uint64_t, DIMS_2, size_t> &indexl2offset,
                          AscendTensor<uint64_t, DIMS_2, size_t> &indexl1offset,
                          AscendTensor<uint16_t, DIMS_2, size_t> &opFlag,
                          AscendTensor<float, DIMS_2, size_t> &disVec,
                          AscendTensor<float, DIMS_2, size_t> &vcMaxDisVec,
                          AscendTensor<uint8_t, DIMS_2, size_t> &codeVec,
                          AscendTensor<float, DIMS_1, size_t> &Indexl2,
                          AscendTensor<float, DIMS_1, size_t> &Indexl1,
                          aclrtStream &stream);
    size_t getMaxListNum(size_t batch, AscendTensor<int64_t, DIMS_2> &l1TopNprobeIndicesHost) const;
    size_t getMaxListNum(size_t batch, AscendTensor<int64_t, DIMS_2> &l1TopNprobeIndicesHost, int k,
                         float* distances, idx_t* labels) const;

    void moveVectorForward(int listId, idx_t srcIdx, idx_t dstIdx);
    void releaseUnusageSpace(int listId, size_t oldTotal, size_t remove);
    size_t removeIds(const ascend::IDSelector& sel);

protected:
    std::unique_ptr<DeviceVector<float>> LUTMatrixOnDevice;        // LUT 计算常值矩阵
    std::unique_ptr<DeviceVector<float>> CentroidLUTOnDevice;      // 聚类中心 LUT
    std::unique_ptr<DeviceVector<float>> CentroidL2OnDevice;       // 聚类中心 L2
    std::vector<std::unique_ptr<DeviceVector<float>>> IndexL2OnDevice;          // 索引 L2
    std::vector<std::unique_ptr<DeviceVector<float>>> IndexL1OnDevice;          // 索引 L1
    std::unique_ptr<AscendOperator> ivfRabitQRotateL2Ops;
    std::unique_ptr<AscendOperator> ivfCenterLUTOps;
    std::unique_ptr<AscendOperator> ivfRabitQIndexRotateL2Ops;
    std::unique_ptr<AscendOperator> ivfRabitQIndexCodeAndPreComputeOps;
    std::map<int, std::unique_ptr<AscendOperator>> ivfRabitqL2DistOps;
    std::vector<std::vector<std::unique_ptr<DeviceVector<uint8_t>>>> baseFp32;
    std::vector<size_t> listVecNum;
    std::map<int, std::unique_ptr<::ascend::AscendOperator>> queryRotateOps;
    std::map<int, std::unique_ptr<::ascend::AscendOperator>> queryLUTOps;
    std::map<int, std::unique_ptr<::ascend::AscendOperator>> topkFp32;
    std::map<int, std::unique_ptr<AscendOperator>> topkL2Fp32;
    std::map<int, std::unique_ptr<AscendOperator>> l1DistFp32Ops;
    int blockSize;
    uint8_t* pBaseFp32;
    float* pIndexL2;
    float* pIndexL1;
    std::unique_ptr<DeviceVector<float>> distResultOnDevice;
    size_t distResultSpace;
};
}  // namespace ascend

#endif  // ASCEND_INDEXIVFRABITQ_INCLUDED

