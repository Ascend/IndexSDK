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

#ifndef ASCEND_INDEXIVFPQ_INCLUDED
#define ASCEND_INDEXIVFPQ_INCLUDED
#include "ascenddaemon/impl/IndexIVF.h"
#include "ascenddaemon/utils/AscendOperator.h"
#include "common/threadpool/AscendThreadPool.h"

namespace ascend
{
namespace
{
constexpr int IVF_PQ_THREADS_CNT = 4;
constexpr int IVF_PQ_BURST_LEN = 64;
constexpr int MAX_BATCH_SIZE = 64;
constexpr uint8_t IVF_PQ_BURST_LEN_LOW = 32;
constexpr uint8_t IVF_PQ_BURST_LEN_HIGH = 64;
constexpr uint8_t IVF_PQ_BURST_BLOCK_RATIO = 2;
constexpr uint8_t IVF_PQ_OPTIMIZE_BATCH_THRES = 48;
}  // namespace

class IndexIVFPQ : public IndexIVF
{
   public:
    IndexIVFPQ(int numList, int dim, int M, int nbits, int nprobes, int64_t resourceSize = -1);

    ~IndexIVFPQ();

    APP_ERROR reset() override;

    APP_ERROR addPQCodes(int listId, size_t numVecs, const uint8_t *pqCodes, const idx_t *indices);

    APP_ERROR searchImpl(int n, const float16_t *x, int k, float16_t *distances, idx_t *labels);

    APP_ERROR searchImpl(AscendTensor<float16_t, DIMS_2> &queries, int k, AscendTensor<float16_t, DIMS_2> &outDistance,
                         AscendTensor<idx_t, DIMS_2> &outIndices);

    APP_ERROR searchPaged(size_t pageId, size_t pageNum, AscendTensor<float16_t, DIMS_2> &queries,
                          AscendTensor<float16_t, DIMS_2> &maxDistances, AscendTensor<int64_t, DIMS_2> &maxIndices);

    APP_ERROR resizeBasePQ(int listId, size_t numVecs);

    size_t getListLength(int listId) const;

    APP_ERROR getListVectorsReshaped(int listId, std::vector<unsigned char> &reshaped) const override;

    std::unique_ptr<DeviceVector<float>> centroidsOnDevice;
    std::unique_ptr<DeviceVector<float>> centroidsSqrSumOnDevice;
    std::unique_ptr<DeviceVector<float>> codeBookOnDevice;
    std::unique_ptr<DeviceVector<float>> clusteringOnDevice;

    // Mirrors faiss::Index::verbose; set by host before trainImpl.
    // Kept for ABI compatibility with existing AscendIndexIVFPQImpl callers.
    bool verbose = false;

    APP_ERROR trainImpl(int n, const float *x, int dim, int nlist, int niter, int seed, bool skipSample = false);

    void resetTrainSession();

    size_t getPQVecCapacity(size_t vecNum, size_t size, int M) const;

    APP_ERROR deletePQCodes(int listId, size_t numVecs, const idx_t *indices);

    static int GetBurstsOfBlock(int nq, int blockSize, int &burstLen)
    {
        if (faiss::ascend::SocUtils::GetInstance().IsAscend910B())
        {
            burstLen = IVF_PQ_BURST_LEN_HIGH;
        }
        else
        {
            burstLen = (nq > IVF_PQ_OPTIMIZE_BATCH_THRES) ? IVF_PQ_BURST_LEN_LOW : IVF_PQ_BURST_LEN_HIGH;
        }
        return utils::divUp(blockSize, burstLen) * IVF_PQ_BURST_BLOCK_RATIO;
    }
    APP_ERROR searchImpl(int n, const float *x, int k, float *distances, idx_t *labels);

    APP_ERROR assignCentroid(int nlist, int dim, int totalSize, std::vector<float> &centroids, float *trainDataHost,
                             std::vector<int64_t> &totalAssigns, bool useFastPath = false);

    APP_ERROR initClusterTrainOps(int nlist, int dim);

    APP_ERROR runKMeans(int nlist, int dim, int totalSize, int iter, float *centroidsHost,
                        AscendTensor<float, DIMS_2> &dataVector, std::vector<int64_t> &totalAssigns,
                        bool useDeviceCentroids = false, int64_t assignOffset = 0);

    APP_ERROR copyTrainCentroidsFromDevice(int nlist, int dim, const float *srcCentroidsDev,
                                           const float *srcCentroidsSqrSumDev);

    const float *getTrainCentroidsDevicePtr() const;
    const float *getTrainCentroidsSqrSumDevicePtr() const;

    void ensureTrainBatchWorkspace(int nlist, int batchSize);
    void ensureTrainAssignBuffer(int totalSize);
    void ensureTrainCentroidsDevice(int nlist, int dim);
    APP_ERROR syncTrainCentroidsToDevice(int nlist, int dim, const std::vector<float> &centroids);
    APP_ERROR syncTrainCentroidsFromHost(int nlist, int dim, const float *centroids);
    void ensureAssignQueryBuffer(size_t elemCount);
    void ensureL1Workspace(int batch, int nlist);
    APP_ERROR assignCentroidFast(AscendTensor<float, DIMS_2> &queries, std::vector<int64_t> &assignments);

   protected:
    void normL2(int dim, int nlist, float *data);
    APP_ERROR copyPQBlocks(int listId, size_t totalVecsInList, size_t codeSizePerVector,
                           std::vector<unsigned char> &reshaped, size_t &destOffset, size_t &processedVecs) const;
    APP_ERROR getListInitialize(int listId, std::vector<unsigned char> &reshaped, size_t &totalVecsInList,
                                size_t &codeSizePerVector, size_t &totalBytes) const;
    APP_ERROR prepareDelete(int listId, size_t numVecs, const idx_t *indices, size_t &currentVecNum,
                            std::vector<idx_t> &hostIds, std::vector<bool> &toDelete, std::vector<idx_t> &deletedIds,
                            size_t &deleteCount);
    APP_ERROR updateFilterPQCodes(int listId, size_t currentVecNum, std::vector<idx_t> &hostIds,
                                  std::vector<bool> &toDelete, size_t deleteCount, size_t bytesPerVector,
                                  std::vector<uint8_t> &newCodes, std::vector<idx_t> &newIds);
    APP_ERROR updateDeviceData(int listId, size_t newVecNum, std::vector<uint8_t> &newCodes,
                               const std::vector<idx_t> &newIds);
    APP_ERROR initTraining(int totalSize, int dim, int nlist, std::vector<float> &trainData,
                           std::vector<float> &centroids, int seed);
    void initKmeans(std::vector<float> &trainData, std::vector<float> &centroids, int totalSize, int dim, int nlist);
    APP_ERROR resetTrainOp(int nlist, int dim);
    APP_ERROR trainBatchImpl(int batchSize, int nlist, int dim, int processed, AscendTensor<float, DIMS_2> &dataVector,
                             AscendTensor<float, DIMS_2> &centroidsDev, AscendTensor<float, DIMS_1> &centroidsDoubleDev,
                             AscendTensor<uint32_t, DIMS_2> &sizesDev, AscendTensor<uint16_t, DIMS_2> &flagsDev,
                             AscendTensor<int64_t, DIMS_1> &attrsDev, int64_t *distMs, int64_t *topkMs);
    APP_ERROR execTrainBatch(AscendTensor<float, DIMS_2> &queries, AscendTensor<float, DIMS_2> &centroids,
                             AscendTensor<float, DIMS_1> &centroidsDouble, AscendTensor<uint32_t, DIMS_2> &sizes,
                             AscendTensor<uint16_t, DIMS_2> &flags, AscendTensor<int64_t, DIMS_1> &attrs,
                             int assignOffset, int64_t *distMs, int64_t *topkMs);
    APP_ERROR execTrainBatchDist(AscendTensor<float, DIMS_2> &queries, AscendTensor<float, DIMS_2> &centroids,
                                 AscendTensor<float, DIMS_1> &centroidsDouble, AscendTensor<uint16_t, DIMS_2> &flags,
                                 int bufIdx, int64_t *distMs);
    APP_ERROR execTrainBatchTopk(AscendTensor<uint32_t, DIMS_2> &sizes, AscendTensor<uint16_t, DIMS_2> &flags,
                                 AscendTensor<int64_t, DIMS_1> &attrs, int bufIdx, int batchSize, int nlist,
                                 int assignOffset);
    APP_ERROR updateCentroids(int nlist, int dim, int totalSize, const std::vector<int64_t> &totalAssigns,
                              const std::vector<float> &trainData, std::vector<float> &centroids);
    APP_ERROR updateCentroidsToDevice(int nlist, int dim, const std::vector<float> &centroids);
    APP_ERROR addCodes(int listId, AscendTensor<uint8_t, DIMS_2> &codesData);
    APP_ERROR resetTrainDistOp(int nlist, int dim);
    void runTrainDistOp(int batch, AscendTensor<float, DIMS_2> &queries, AscendTensor<float, DIMS_2> &centroids,
                        AscendTensor<float, DIMS_1> &codesDouble, AscendTensor<float, DIMS_2> &distances,
                        AscendTensor<float, DIMS_2> &vmdists, AscendTensor<uint16_t, DIMS_2> &opFlag,
                        aclrtStream stream);
    APP_ERROR resetTrainTopkOp(int nlist);
    void runTrainTopkOp(AscendTensor<float, DIMS_2> &dists, AscendTensor<float, DIMS_2> &vmdists,
                        AscendTensor<uint32_t, DIMS_2> &sizes, AscendTensor<uint16_t, DIMS_2> &flags,
                        AscendTensor<int64_t, DIMS_1> &attrs, AscendTensor<float, DIMS_2> &outdists,
                        AscendTensor<int64_t, DIMS_2> &outlabel, aclrtStream stream);
    APP_ERROR resetL1TopkOp();
    APP_ERROR resetL1DistOp();
    APP_ERROR resetL2DistOp();
    APP_ERROR resetL3TopkOp();
    APP_ERROR resetL3DistOp();
    APP_ERROR ensureL2L3SearchOps();

    APP_ERROR searchWithBatch(int n, const float *x, int k, float *distances, idx_t *labels);
    APP_ERROR searchImplL1(AscendTensor<float, DIMS_2> &queries, AscendTensor<int64_t, DIMS_2> &l1TopNprobeIndicesHost,
                           AscendTensor<float, DIMS_3, size_t> &l2SubspaceDistsDev);
    APP_ERROR searchImplL3(AscendTensor<int64_t, DIMS_2> &l1TopNprobeIndicesHost,
                           AscendTensor<float, DIMS_3, size_t> &l2SubspaceDistsDev, int k, float *distances,
                           idx_t *labels);

    void runL1TopkOp(AscendTensor<float, DIMS_2> &dists, AscendTensor<float, DIMS_2> &vmdists,
                     AscendTensor<uint32_t, DIMS_2> &sizes, AscendTensor<uint16_t, DIMS_2> &flags,
                     AscendTensor<int64_t, DIMS_1> &attrs, AscendTensor<float, DIMS_2> &outdists,
                     AscendTensor<int64_t, DIMS_2> &outlabel, aclrtStream stream);
    void runL1DistOp(int batch, AscendTensor<float, DIMS_2> &queries, AscendTensor<float, DIMS_2> &centroidsDev,
                     AscendTensor<float, DIMS_2> &dists, AscendTensor<float, DIMS_2> &vmdists,
                     AscendTensor<uint16_t, DIMS_2> &opFlag, aclrtStream stream);
    void runL2DistOp(int batch, AscendTensor<float, DIMS_2> &queries, AscendTensor<float, DIMS_3> &codeBook,
                     AscendTensor<float, DIMS_3, size_t> &dists, aclrtStream stream);
    void runL3TopkOp(AscendTensor<uint64_t, DIMS_3, size_t> &topkIndex, AscendTensor<float, DIMS_3, size_t> &topkValue,
                     AscendTensor<uint16_t, DIMS_2, size_t> &flags, AscendTensor<int64_t, DIMS_1> &attrs,
                     AscendTensor<float, DIMS_2, size_t> &outdists, AscendTensor<uint64_t, DIMS_2, size_t> &outlabel,
                     aclrtStream stream);
    void runL3DistOp(int batch, AscendTensor<float, DIMS_3, size_t> &queryPQ,
                     AscendTensor<uint8_t, DIMS_1, size_t> &codeBase, AscendTensor<int64_t, DIMS_2, size_t> &offset,
                     AscendTensor<int64_t, DIMS_2, size_t> &baseSize, AscendTensor<int32_t, DIMS_1, size_t> &topk,
                     AscendTensor<uint64_t, DIMS_1, size_t> &labelBase,
                     AscendTensor<int64_t, DIMS_2, size_t> &labelOffset, AscendTensor<float, DIMS_3, size_t> &dists,
                     AscendTensor<int32_t, DIMS_3, size_t> &topkIndex, AscendTensor<float, DIMS_3, size_t> &topkValue,
                     AscendTensor<uint64_t, DIMS_2, size_t> &topkIndexFinal,
                     AscendTensor<float, DIMS_2, size_t> &topkValueFinal,
                     AscendTensor<uint16_t, DIMS_1, size_t> &opFlag, aclrtStream stream);

    void callL3DistanceOp(
        size_t batch, size_t tileNum, size_t segNum, size_t coreNum, size_t kAligned,
        AscendTensor<float, DIMS_3, size_t> &l2SubspaceDists, AscendTensor<int64_t, DIMS_3, size_t> &codeOffset,
        AscendTensor<int64_t, DIMS_3, size_t> &codeSize, AscendTensor<int32_t, DIMS_1, size_t> &topk,
        AscendTensor<int64_t, DIMS_3, size_t> &labelOffset, AscendTensor<uint16_t, DIMS_2, size_t> &opFlag,
        AscendTensor<float, DIMS_3, size_t> &distResult, AscendTensor<int32_t, DIMS_3, size_t> &topkIndex,
        AscendTensor<float, DIMS_3, size_t> &topkValue, AscendTensor<uint64_t, DIMS_3, size_t> &topkIndexFinal,
        AscendTensor<float, DIMS_3, size_t> &topkValueFinal, AscendTensor<uint8_t, DIMS_1, size_t> &codeBase,
        AscendTensor<uint64_t, DIMS_1, size_t> &labelBase, aclrtStream &stream);
    int getL3SearchBatchCap() const;
    void fillDisOpInputDataByBlockPQ(size_t qIdx, size_t tIdx, size_t segIdx, size_t segNum, size_t coreNum,
                                     size_t ivfpqBlockSize, AscendTensor<int64_t, DIMS_3, size_t> &baseSizeHostVec,
                                     AscendTensor<int64_t, DIMS_3, size_t> &offsetHostVec,
                                     AscendTensor<int64_t, DIMS_3, size_t> &labeloffsetHostVec,
                                     AscendTensor<int64_t, DIMS_2> &l1TopNprobeIndicesHost);
    APP_ERROR fillDisOpInputDataPQ(int k, size_t batch, size_t tileNum, size_t segNum, size_t coreNum,
                                   AscendTensor<int64_t, DIMS_3, size_t> &offset,
                                   AscendTensor<int64_t, DIMS_3, size_t> &baseSize,
                                   AscendTensor<int64_t, DIMS_1> &attrs,
                                   AscendTensor<int64_t, DIMS_3, size_t> &labelOffset,
                                   AscendTensor<int64_t, DIMS_2> &l1TopNprobeIndicesHost);

    size_t getMaxListNum(size_t batch, AscendTensor<int64_t, DIMS_2> &l1TopNprobeIndicesHost) const;
    void initializeCodeBook(int M, int nbits, int dsubs);

    static uint64_t getActualRngSeed(const int seed);
    static void sampleTrainData(const float *x, int n, int dim, int totalSize, int seed, std::vector<float> &trainData);

   protected:
    std::map<int, std::unique_ptr<AscendOperator>> l3DistFp32Ops;
    std::vector<std::vector<std::unique_ptr<DeviceVector<uint8_t>>>> basePQCoder;
    std::vector<size_t> listVecNum;
    std::map<int, std::unique_ptr<::ascend::AscendOperator>> topkFp32;
    std::map<int, std::unique_ptr<AscendOperator>> topkL3Fp32;
    std::map<int, std::unique_ptr<AscendOperator>> l1DistFp32Ops;
    std::map<int, std::unique_ptr<AscendOperator>> l2DistFp32Ops;
    std::map<int, std::unique_ptr<AscendOperator>> trainDistOps;
    std::map<int, std::unique_ptr<AscendOperator>> trainTopkOps;
    int trainOpNlist_ = -1;
    int trainOpDim_ = -1;
    int trainWsNlist_ = -1;
    int trainWsMaxBatch_ = 0;
    std::unique_ptr<DeviceVector<float>> trainBatchDistsDev_;
    std::unique_ptr<DeviceVector<float>> trainBatchVmdistsDev_;
    std::unique_ptr<DeviceVector<float>> trainBatchDistsDevAlt_;
    std::unique_ptr<DeviceVector<float>> trainBatchVmdistsDevAlt_;
    std::unique_ptr<DeviceVector<float>> trainBatchOutdistsDev_;
    std::unique_ptr<DeviceVector<int64_t>> trainBatchOutlabelDev_;
    std::unique_ptr<DeviceVector<int64_t>> trainAssignsDev_;
    int trainAssignsCapacity_ = 0;
    std::unique_ptr<DeviceVector<float>> trainCentroidsDev_;
    std::unique_ptr<DeviceVector<float>> trainCentroidsSqrSumDev_;
    int trainCentroidsNlist_ = -1;
    int trainCentroidsDim_ = -1;
    std::unique_ptr<DeviceVector<float>> assignQueriesOnDevice_;
    size_t assignQueriesCapacityElems_ = 0;
    int l1WsNumLists_ = -1;
    int l1WsMaxBatch_ = 0;
    std::unique_ptr<DeviceVector<float>> l1DistsDev_;
    std::unique_ptr<DeviceVector<float>> l1VmdistsDev_;
    std::unique_ptr<DeviceVector<float>> l1TopNprobeDistsDev_;
    std::unique_ptr<DeviceVector<int64_t>> l1TopNprobeIndicesDev_;
    std::unique_ptr<DeviceVector<uint32_t>> l1OpSizeDev_;
    std::unique_ptr<DeviceVector<uint16_t>> l1OpFlagDev_;
    std::unique_ptr<DeviceVector<int64_t>> l1AttrsDev_;
    std::unique_ptr<DeviceVector<float>> assignDistsDevAlt_;
    std::unique_ptr<DeviceVector<float>> assignVmdistsDevAlt_;
    std::unique_ptr<DeviceVector<uint16_t>> assignOpFlagDevAlt_;
    std::unique_ptr<DeviceVector<float>> assignOutdistsDev_;
    int blockSize;
    int M;
    int nbits;
    int dsub;
    int ksub;
    int blockNum;
    uint8_t *pBasePQCoder;
    int devPQVecCapacity;
};
}  // namespace ascend

#endif  // ASCEND_INDEXIVFPQ_INCLUDED
