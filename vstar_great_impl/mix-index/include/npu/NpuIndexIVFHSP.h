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


#ifndef IVFSP_INDEXIVFHSP_H
#define IVFSP_INDEXIVFHSP_H

#include <vector>
#include <map>

#include "NpuIndexIVF.h"
#include "npu/common/DeviceVector.h"
#include "npu/common/AscendTensor.h"
#include "faiss/impl/ScalarQuantizer.h"
#include "npu/common/threadpool/AscendThreadPool.h"
#include "npu/common/threadpool/ThreadPool.h"
#include "impl/VisitedTable.h"
#include "utils/VstarIoUtil.h"
#ifdef USE_ACL_NN_INTERFACE
#include <aclnn/acl_meta.h>
#endif

namespace ascendSearchacc {
const int QUERY_BATCH_SIZE = 1024;
const int BASE_STORE_PAGE_SIZE = 8192;
const int PROGRESSBATCH = 64;
const int MAXTHREADNUM = 150;
#ifdef INDEX_SEG_SIZE
const int BASE_SEG_SIZE = INDEX_SEG_SIZE;
#else
const int BASE_SEG_SIZE = 64;
#endif
const int VCMIN_SEG_SIZE = 64;

// search pagesize must be less than 64M, becauseof rpc limitation
const size_t SEARCH_PAGE_SIZE = UNIT_PAGE_MB * KB * KB - RETAIN_SIZE;
// Or, maximum number 512K of vectors to consider per page of search
const size_t SEARCH_VEC_SIZE = 512 * KB;

struct SearchParams {
    SearchParams()
    {
    }
    SearchParams(int nProbeL1, int nProbeL2, int l3SegmentNum)
        : nProbeL1(nProbeL1), nProbeL2(nProbeL2), l3SegmentNum(l3SegmentNum)
    {
    }
    int nProbeL1 = 72;
    int nProbeL2 = 64;
    int l3SegmentNum = 512;
};

enum class ReadAndCheckType {
    dim,
    subdiml1,
    subdiml2,
    nlist1,
    nlist2,
    validRange,
    False,
    metricType
};

class NpuIndexIVFHSP : public NpuIndexIVF {
public:
    NpuIndexIVFHSP(int dim, int subSpaceDim1, int subSpaceDim2, int nList1, int nList2, MetricType metricType,
                   NpuIndexConfig config);

    explicit NpuIndexIVFHSP(NpuIndexConfig config);

    virtual ~NpuIndexIVFHSP();

    /**
     * Called By CPU Function, Try to Add Base Vectors to Dev Memory
     * @param baseRawData a flatten vector to store embeddings, size is numVec * Dim
     * @return a error flag
     */

    APP_ERROR AddVectorsVerbose(const std::vector<float> &baseRawData, bool verbose = true);

    virtual APP_ERROR Train(const std::vector<float> &data);

    /**
     * Called By CPU Function, Try to Add Code Books to Dev Memory
     * @param codeBooksL1 ivf_l1 centroids, the number of centroids is nList1,
     *                     every centroid represented as a metric [Dim, subSpaceDimL1],
     *                     All Data is stored as two dim Shape [Dim,  nListL1 * subSpaceDimL1]
     * @param codeBooksL2 ivf_l2 centroids, the number of centroids is nList1 * nList2,
     *                    every centroid represented as a metric [subSpaceDimL1, subSpaceDimL2],
     *                    All Data is stored as three dim is (nList1, subSpaceDimL1,  nListL2 * subSpaceDimL2)
     * @return
     */
    virtual APP_ERROR AddCodeBooks(const std::vector<float> &codeBooksL1, const std::vector<float> &codeBooksL2);

    APP_ERROR AddCodeBooks(const NpuIndexIVFHSP* loadedIndex);

    /**
     * Called By CPU Function
     * @param queryData [nq, Dim] flattened
     * @return
     */
    virtual APP_ERROR Search(std::vector<float> &queryData, int topK, std::vector<float> &dists,
                             std::vector<int64_t> &labels) const;
    virtual APP_ERROR Search(size_t n, float *queryData, int topK, float *dists, int64_t *labels) const;
    virtual APP_ERROR Search(const std::vector<NpuIndexIVFHSP *> &indexes, size_t n, float *queryData, int topK,
                             float *dists, int64_t *labels, bool merge); // MultiIndexSearch
    virtual APP_ERROR Search(const std::vector<NpuIndexIVFHSP *> &indexes, size_t n, uint8_t *mask, float *queryData,
                            int topK, float *dists, int64_t *labels, bool merge); // MultiIndexSearch With Mask
    virtual APP_ERROR Search(size_t n, uint8_t *mask, float *queryData, int topK, float *dists, int64_t *labels) const;

    virtual APP_ERROR DeleteVectors(const std::vector<int64_t> &ids);
    virtual APP_ERROR DeleteVectors(const int64_t &id);
    virtual APP_ERROR DeleteVectors(const int64_t &startId, const int64_t &endId);

    virtual APP_ERROR Reset();
    virtual APP_ERROR Init();

    void SetSearchParams(SearchParams params);
    SearchParams GetSearchParams() const;

    std::shared_ptr<const AscendTensor<float16_t, DIMS_4> > GetCodeBooksL1NPU() const;
    std::shared_ptr<const AscendTensor<float16_t, DIMS_4> > GetCodeBooksL2NPU() const;
    const float *GetCodeBooksL1CPU() const;
    const float *GetCodeBooksL2CPU() const;

    void WriteIndex(std::string indexPath);
    void LoadIndex(std::string indexPath, const NpuIndexIVFHSP *loadedIndex = nullptr);

    uint64_t GetNTotal() const;
    uint64_t GetUniqueBaseVecCounter() const;
    int GetDim() const;
    int GetNlistL1() const;
    int GetNlistL2() const;
    int GetSubDimL1() const;
    int GetSubDimL2() const;
    void forwardIdsMapReverse(std::map<int64_t, int64_t> idMapReverseForwarded);
    const std::map<int64_t, int64_t>& GetIdsMapReverse() const;
    
    void SetAddWithIds(bool isAddWithIds);
    bool GetAddWithIds() const;
    void SetIdMap(const std::map<int64_t, int64_t> &idMap);
    const std::map<int64_t, int64_t>& GetIdMap() const;

    void SetTopKDuringSearch(int topKSearch);

protected:
    APP_ERROR AddIntoNpuDataStore(size_t num, std::vector<std::vector<uint8_t> > &baseCodesByBucket,
                                  std::vector<std::vector<int64_t> > &idsByBucket,
                                  std::vector<std::vector<float> > &normL2ByBucket);

    /**
     * Data Type Transform, Require: n in {1024, 128, 64, 48, 36, 32, 30, 24, 18, 16, 12, 8, 6, 4, 2, 1}
     * @param floatDataNpu Shape is [n, Dim]
     * @param fp16DataNpu Shape is [n, Dim]
     */
    APP_ERROR DataTypeTransform(AscendTensor<float, DIMS_2> &floatDataNpu,
                                AscendTensor<float16_t, DIMS_2> &fp16DataNpu);

    /**
     * matrix reshape to Zz format
     * origin code for example (shape n X dim). n=16, dim = 128
     *       |  0_0  0_1  0_2  0_3 ...  0_125  0_126  0_127 |
     *       |  1_0  1_1  1_2  1_3 ...  1_125  1_126  1_127 |
     *       |        .                          .          |
     *       |        .                          .          |
     *       | 14_0 14_1 14_2 14_3 ... 14_125 14_126 14_127 |
     *       | 15_0 15_1 15_2 15_3 ... 15_125 15_126 15_127 |
     * shape dims 2: (dim/16 X n/16) X (16 X 16), after Zz format    dims4: (n/16) X (dim/16) X 16 X 16
     *       |   0_0   0_1 ...  0_14  0_15   1_0   1_1 ...  1_15 ...   15_15 |
     *       |  0_16  0_17 ...  0_30  0_31  1_16  1_17 ...  1_31 ...   15_31 |
     *       |        .                    .                  .         .    |
     *       |        .                    .                  .         .    |
     *       |  0_96  0_97 ... 0_110 0_111  1_96  1_97 ... 1_111 ...  15_111 |
     *       | 0_112 0_113 ... 0_126 0_127 1_112 1_113 ... 1_127 ...  15_127 |
     * n and dim must be 16 aligned, otherwise padding data is needed.
     * @param srcNpu Shape is [n, dim]
     * @param dstNpu Shape is [n // 16, dim // 16, 16, 16]
     * @return
     */
    template <typename T, aclDataType ACL_T>
    APP_ERROR ZzFormatReshape(AscendTensor<T, DIMS_2> &srcNpu, AscendTensor<T, DIMS_4> &dstNpu);

    /**
     * matrix reshape to Zz format
     * origin code for example (shape n X dim). n=16, dim = 128
     *       |  0_0  0_1  0_2  0_3 ...  0_125  0_126  0_127 |
     *       |  1_0  1_1  1_2  1_3 ...  1_125  1_126  1_127 |
     *       |        .                          .          |
     *       |        .                          .          |
     *       | 14_0 14_1 14_2 14_3 ... 14_125 14_126 14_127 |
     *       | 15_0 15_1 15_2 15_3 ... 15_125 15_126 15_127 |
     * shape dims 2: (dim/16 X n/16) X (16 X 16), after Zz format    dims4: (n/16) X (dim/16) X 16 X 16
     *       |   0_0   0_1 ...  0_14  0_15   1_0   1_1 ...  1_15 ...   15_15 |
     *       |  0_16  0_17 ...  0_30  0_31  1_16  1_17 ...  1_31 ...   15_31 |
     *       |        .                    .                  .         .    |
     *       |        .                    .                  .         .    |
     *       |  0_96  0_97 ... 0_110 0_111  1_96  1_97 ... 1_111 ...  15_111 |
     *       | 0_112 0_113 ... 0_126 0_127 1_112 1_113 ... 1_127 ...  15_127 |
     * n and dim must be 16 aligned, otherwise padding data is needed.
     * @tparam T
     * @param src Shape is [n, dim]
     * @param dst Shape is [n // nAlign, dim // dimAlign, nAlign, dimAlign]
     * @param n
     * @param dim
     * @param nAlign require nAlign % 16 == 0
     * @param dimAlign  require dimAlign % 16 == 0
     */
    template <typename T>
    void ZzFormatReshape(std::vector<T> &src, std::vector<T> &dst, size_t n, int dim, int nAlign = CUBE_ALIGN,
                         int dimAlign = CUBE_ALIGN);

protected:
    int GetAddPageSize(size_t byteSizePerTerm);

    template <typename T>
    void ReadAndCheck(T &existingParam, const ReadAndCheckType &checkParam, const VstarIOReader &fin);

    void GetVectorsAssignNPU(size_t nb, float *baseRawData, int64_t *assign, float *codeWords, float *precomputeNormL2,
                             int boostType = 1, bool verbose = true);

    void GetVectorsAssignNPUL1(size_t nb, float *baseRawData, int64_t *assign, float *codeWords,
                               float *precomputeNormL2, bool verbose = true);

    void ComputeSQCode(size_t n, const std::vector<float> &data, std::vector<uint8_t> &sqCodes);

    bool CalculateOffsetL3(const std::vector<NpuIndexIVFHSP *> &indexes, int n, int i,
                           std::vector<uint64_t> &labelL2Cpu, std::vector<uint64_t> &outOffset,
                           std::vector<uint64_t> &outIdsOffset);

    bool CalculateOffsetL3WithMask(const std::vector<NpuIndexIVFHSP *> &indexes, int n, const uint8_t *mask, int i,
                           std::vector<uint64_t> &labelL2Cpu, std::vector<uint64_t> &outOffset,
                           std::vector<uint64_t> &outIdsOffset);

    void UpdateBucketAddressInfo(size_t num, std::vector<uint64_t> &paddingPageOffset);

    APP_ERROR DeleteVectorsImpl(const std::vector<int64_t> &ids);

    size_t GetSearchPagedSize(size_t n, int k) const;

    void SearchImpl(int n, const uint8_t *mask, const float *x, int k, float *distances, int64_t *labels);
    void SearchImpl(int n, const float *x, int k, float *distances, int64_t *labels);
    void SearchImpl(const std::vector<NpuIndexIVFHSP *> &indexes, int n, const float *x, int k, float *distances,
                    int64_t *labels, bool merge);
    void SearchImpl(const std::vector<NpuIndexIVFHSP *> &indexes, int n, const uint8_t *mask, const float *x, int k,
                    float *distances, int64_t *labels, bool merge);

    APP_ERROR SearchBatchImpl(int n, AscendTensor<float, DIMS_2> &queryNpu, int k, float16_t *distances,
                              int64_t *labels);
    APP_ERROR SearchBatchImpl(const std::vector<NpuIndexIVFHSP *> &indexes, int n,
                              AscendTensor<float, DIMS_2> &queryNpu, int k, float16_t *distances, int64_t *labels,
                              bool merge);
    APP_ERROR SearchBatchImpl(int n, AscendTensor<uint8_t, DIMS_1> &maskBitNpu, AscendTensor<float, DIMS_2> &queryNpu,
                              int k, float16_t *distances, int64_t *labels);

    APP_ERROR SearchBatchImpl(const std::vector<NpuIndexIVFHSP *> &indexes, int n, const uint8_t *mask,
                              AscendTensor<float, DIMS_2> &queryNpu, int k, float16_t *distances, int64_t *labels,
                              bool merge);

    APP_ERROR SearchBatchImplL1(AscendTensor<float, DIMS_2> &queriesNpu, AscendTensor<float16_t, DIMS_2> &queryCodes,
                                AscendTensor<uint16_t, DIMS_2> &l1KIndiceNpu);

    APP_ERROR SearchBatchImplL2(AscendTensor<float16_t, DIMS_2> &queryCodesNpu,
                                AscendTensor<uint16_t, DIMS_2> &l1KIndicesNpu,
                                AscendTensor<uint64_t, DIMS_2> &addressOffsetOfBucketL3,
                                AscendTensor<uint64_t, DIMS_2> &idAdressL3);

    APP_ERROR SearchBatchImplL2(AscendTensor<uint8_t, DIMS_1> &maskBitNpu,
                                AscendTensor<float16_t, DIMS_2> &queryCodesNpu,
                                AscendTensor<uint16_t, DIMS_2> &l1KIndicesNpu,
                                AscendTensor<uint64_t, DIMS_2> &addressOffsetOfBucketL3,
                                AscendTensor<uint64_t, DIMS_2> &idAdressL3);

    APP_ERROR SearchBatchImplMultiL2(AscendTensor<float16_t, DIMS_2> &queryCodesNpu,
                                     AscendTensor<uint16_t, DIMS_2> &l1KIndicesNpu,
                                     AscendTensor<uint64_t, DIMS_2> &indicesL2);

    APP_ERROR SearchBatchImplL3(AscendTensor<float16_t, DIMS_2> &queryCodes,
                                AscendTensor<uint64_t, DIMS_2> &addressOffsetOfBucketL3,
                                AscendTensor<uint64_t, DIMS_2> &idAddressOfBucketL3,
                                AscendTensor<float16_t, DIMS_2> &outDists, AscendTensor<int64_t, DIMS_2> &outIndices);

    APP_ERROR SearchBatchImplMultiL3(const std::vector<NpuIndexIVFHSP *> &indexes, int i,
                                     AscendTensor<float16_t, DIMS_2> &queryCodes,
                                     AscendTensor<uint64_t, DIMS_3> &addressOffsetOfBucketL3,
                                     AscendTensor<uint64_t, DIMS_3> &idAddressOfBucketL3,
                                     AscendTensor<float16_t, DIMS_3> &distResult,
                                     AscendTensor<float16_t, DIMS_3> &vcMinDistResult,
                                     AscendTensor<uint16_t, DIMS_3> &opFlag);

    void UpdateOpAttrs();

    /******************************************************************
     * Ascend Operators
     ******************************************************************/
    void RunL1TopKOp(int batch, AscendTensor<float16_t, DIMS_2> &dists, AscendTensor<uint32_t, DIMS_2> &opSize,
                     AscendTensor<uint16_t, DIMS_2> &opFlag, AscendTensor<int64_t, DIMS_1> &attrs,
                     AscendTensor<float16_t, DIMS_2> &distResult, AscendTensor<uint16_t, DIMS_2> &labelResult);

    void RunL1DistOp(int batch, AscendTensor<float, DIMS_2> &query, AscendTensor<float16_t, DIMS_4> &dataShaped,
                     AscendTensor<float16_t, DIMS_2> &queryCode, AscendTensor<float16_t, DIMS_2> &dists,
                     AscendTensor<uint16_t, DIMS_2> &flag);

    void RunL2TopKWithMaskOp(AscendTensor<uint8_t, DIMS_1> &maskBitNpu, AscendTensor<float16_t, DIMS_2> &dists,
                             AscendTensor<uint16_t, DIMS_2> &l1Indices, AscendTensor<uint16_t, DIMS_2> &opFlag,
                             AscendTensor<int64_t, DIMS_1> &attr, AscendTensor<float16_t, DIMS_2> &distsRes,
                             AscendTensor<uint64_t, DIMS_2> &addressOffset, AscendTensor<uint64_t, DIMS_2> &idAddress);

    void RunL2TopKOp(AscendTensor<float16_t, DIMS_2> &dists, AscendTensor<uint16_t, DIMS_2> &l1Indices,
                     AscendTensor<uint16_t, DIMS_2> &opFlag, AscendTensor<int64_t, DIMS_1> &attr,
                     AscendTensor<float16_t, DIMS_2> &distsRes, AscendTensor<uint64_t, DIMS_2> &addressOffset,
                     AscendTensor<uint64_t, DIMS_2> &idAddress);

    void RunMultiL2TopKOp(AscendTensor<float16_t, DIMS_2> &dists, AscendTensor<uint16_t, DIMS_2> &l1Indices,
                          AscendTensor<uint16_t, DIMS_2> &opFlag, AscendTensor<int64_t, DIMS_1> &attr,
                          AscendTensor<float16_t, DIMS_2> &distsRes, AscendTensor<uint64_t, DIMS_2> &labelRes);

    void RunL3TopKOp(AscendTensor<float16_t, DIMS_2> &dists, AscendTensor<float16_t, DIMS_2> &distsVcMin,
                     AscendTensor<uint16_t, DIMS_2> &opFlag, AscendTensor<uint64_t, DIMS_2> &idAddressOffsetOfBucketL3,
                     AscendTensor<int64_t, DIMS_1> &attr,
                     //                         AscendTensor<uint64_t, DIMS_2> &addressOffsetOfBucketL3,
                     AscendTensor<float16_t, DIMS_2> &distsRes, AscendTensor<int64_t, DIMS_2> &labelsRes);

    void RunL3TopKOp(const std::vector<NpuIndexIVFHSP *> &indexes, int i, AscendTensor<float16_t, DIMS_3> &dists,
                     AscendTensor<float16_t, DIMS_3> &distsVcMin, AscendTensor<uint16_t, DIMS_3> &opFlag,
                     AscendTensor<uint64_t, DIMS_3> &idAddressOffsetOfBucketL3, AscendTensor<int64_t, DIMS_1> &attr,
                     //                         AscendTensor<uint64_t, DIMS_2> &addressOffsetOfBucketL3,
                     AscendTensor<float16_t, DIMS_3> &distsRes, AscendTensor<int64_t, DIMS_3> &labelsRes);

    void RunMultiL3TopKOp(AscendTensor<float16_t, DIMS_3> &dists, AscendTensor<float16_t, DIMS_3> &distsVcMin,
                          AscendTensor<uint16_t, DIMS_3> &opFlag,
                          AscendTensor<uint64_t, DIMS_3> &idAddressOffsetOfBucketL3,
                          AscendTensor<int64_t, DIMS_1> &attr, AscendTensor<uint64_t, DIMS_3> &addressOffsetOfBucketL3,
                          AscendTensor<float16_t, DIMS_3> &distsRes, AscendTensor<int64_t, DIMS_3> &labelsRes);

    void RunL2DistOp(AscendTensor<float16_t, DIMS_2> &queryCodes, AscendTensor<float16_t, DIMS_4> &codebookL2,
                     AscendTensor<uint16_t, DIMS_2> &offsets, AscendTensor<float16_t, DIMS_2> &outDists,
                     AscendTensor<uint16_t, DIMS_2> &flag);

    void RunL3DistOp(AscendTensor<float16_t, DIMS_2> &queryCodes,
                     AscendTensor<uint64_t, DIMS_2> &addressOffsetOfBucketL3, AscendTensor<float16_t, DIMS_2> &dists,
                     AscendTensor<float16_t, DIMS_2> &distsVcMin, AscendTensor<uint16_t, DIMS_2> &opFlag,
                     AscendTensor<int32_t, DIMS_2> &attr_nlistL1, AscendTensor<int32_t, DIMS_2> &attr_nlistL2,
                     AscendTensor<int32_t, DIMS_2> &attr_segmentL3);

    void RunMultiL3DistOp(const std::vector<NpuIndexIVFHSP *> &indexes, int i,
                          AscendTensor<float16_t, DIMS_2> &queryCodes,
                          AscendTensor<uint64_t, DIMS_3> &addressOffsetOfBucketL3,
                          AscendTensor<float16_t, DIMS_3> &dists, AscendTensor<float16_t, DIMS_3> &distsVcMin,
                          AscendTensor<uint16_t, DIMS_3> &opFlag, AscendTensor<int32_t, DIMS_2> &attr_nlistL1,
                          AscendTensor<int32_t, DIMS_2> &attr_nlistL2, AscendTensor<int32_t, DIMS_2> &attr_segmentL3);

    void RunMultiL3DistWithMaskOp(const std::vector<NpuIndexIVFHSP *> &indexes, int i,
                          AscendTensor<float16_t, DIMS_2> &queryCodes,
                          AscendTensor<uint64_t, DIMS_3> &addressOffsetOfBucketL3,
                          AscendTensor<float16_t, DIMS_3> &dists, AscendTensor<float16_t, DIMS_3> &distsVcMin,
                          AscendTensor<uint16_t, DIMS_3> &opFlag, AscendTensor<int32_t, DIMS_2> &attr_nlistL1,
                          AscendTensor<int32_t, DIMS_2> &attr_nlistL2, AscendTensor<int32_t, DIMS_2> &attr_segmentL3);

    void RunL3DistWithMaskOp(AscendTensor<float16_t, DIMS_2> &queryCodes,
                             AscendTensor<uint64_t, DIMS_2> &addressOffsetOfBucketL3,
                             AscendTensor<float16_t, DIMS_2> &dists, AscendTensor<float16_t, DIMS_2> &distsVcMin,
                             AscendTensor<uint16_t, DIMS_2> &opFlag, AscendTensor<int32_t, DIMS_2> &attr_nlistL1,
                             AscendTensor<int32_t, DIMS_2> &attr_nlistL2,
                             AscendTensor<int32_t, DIMS_2> &attr_segmentL3);

    void RunFpToFp16(ascendSearchacc::AscendOperator *op, AscendTensor<float, DIMS_2> &floatDataNpu,
                     AscendTensor<float16_t, DIMS_2> &fp16DataNpu, AscendTensor<uint16_t, DIMS_2> &flag);

    void RunMatMul(AscendTensor<float, DIMS_2> &LeftMatNpu, AscendTensor<float, DIMS_3> &RightMatNpu,
                   AscendTensor<float, DIMS_3> &OutMatNpu);

    APP_ERROR ResetL1DistOp();
    APP_ERROR ResetL2DistOp();
    APP_ERROR ResetL3DistOp();
    APP_ERROR ResetL3DistWithMaskOp();
    APP_ERROR ResetL1TopKOp();
    APP_ERROR ResetL2TopKWithMaskOp();
    APP_ERROR ResetL2TopKOp();
    APP_ERROR ResetL3TopKOp();
    APP_ERROR ResetFpToFp16();
    APP_ERROR ResetMatMul();
    APP_ERROR ResetMultiL2TopKOp();
    APP_ERROR ResetMultiL3TopKOp(int indexSize);

protected:
    std::unique_ptr<AscendThreadPool> threadPool;
    std::unique_ptr<faiss::ScalarQuantizer> sqHandler;
    static std::unique_ptr<ThreadPool> pool;
    std::shared_ptr<std::vector<VisitedTable> > vts;

    int nListL2 = 0;
    int subSpaceDimL1 = 0;
    int subSpaceDimL2 = 0;
    int topKHSP = 100;
    int dimStored = 0;
    bool maskFlag = false;
    bool isAddWithIds = false;
    uint64_t uniqueBaseVecCounter = 0;
    std::map<int64_t, int64_t> idMapReverse; // key is virtual id, value is real id
    std::map<int64_t, int64_t> idMap; // key is real id, value is virtual id

    unsigned char *minAddressOfBaseNpu = nullptr;

    /* *NPU Memory: Store base vectors By Pages, each page or each DeviceVector stores several bucketL2 base data */
    std::vector<std::unique_ptr<DeviceVector<unsigned char> > > baseShaped;
    /* *NPU Memory: store id information related with baseShaped */
    std::vector<std::unique_ptr<DeviceVector<int64_t> > > ids;
    std::vector<std::vector<int64_t> > idsCPU;
    std::vector<uint64_t> idsCpuAddressOfBucketVec;
    std::unique_ptr<DeviceVector<float16_t> > precomputeNormL2Npu = nullptr;
    std::unique_ptr<DeviceVector<uint8_t> > maskByteNpu = nullptr;
    std::vector<uint8_t> maskByteCpu;
    std::unique_ptr<DeviceVector<uint64_t> > isMaskOffset = nullptr;
    std::vector<uint64_t> isMaskOffsetCpu;

    /* *pos is the key of bucket id, key=bucketIdL1 * nListL2 + bucketIdL2， value is the device address of base data pos
     * offset of the key bucket */
    std::shared_ptr<AscendTensor<uint64_t, DIMS_1> > addressOffsetOfBucket = nullptr;  // baseAddressOffsetOfBucket,
                                                                                       // normL2AddressOffsetOfBucket,
                                                                                       // idsAddressOfBucket
    std::vector<uint64_t> addressOffsetOfBucketCPU;

    std::vector<size_t> bucketOffsetInPage;

    /* *Store L1 and L2 codeBooks */
    std::shared_ptr<AscendTensor<float16_t, DIMS_4> > codeBooksShapedL1Npu;  // (nlistL1 * subSpaceDimL1 // 16, dim //
                                                                             // 16, 16, 16)
    std::shared_ptr<AscendTensor<float16_t, DIMS_4> > codeBooksShapedL2Npu;  // (nlistL1 * nlistL2 * subSpaceDimL2 //
                                                                             // 16, subSpaceDimL1 // 16, 16, 16)

    std::vector<float> codeBooksL1Cpu;  // (nlistL1, subSpaceDimL1, dim) ==> (nlistL1 * subSpaceDimL1, dim) ==> (nlistL1
                                        // * subSpaceDimL1 * dim)
    std::vector<float> codeBooksL2Cpu;  // (nlistL1, nlistL2 * subSpaceDimL2, subSpaceDimL1)  ====> (nlistL1 * nlistL2 *
                                        // subSpaceDimL2 * subSpaceDimL1)

    /* *for SQ params */
    std::shared_ptr<AscendTensor<float16_t, DIMS_1> > vDiffNpu = nullptr;   // (subSpaceDimL2)
    std::shared_ptr<AscendTensor<float16_t, DIMS_1> > vDiff2Npu = nullptr;  // (subSpaceDimL2)

    /* *For Search */
    std::unique_ptr<SearchParams> searchParam;
    std::shared_ptr<AscendTensor<int64_t, DIMS_1> > searchL1OpAttrs;
    std::shared_ptr<AscendTensor<int64_t, DIMS_1> > searchL2OpAttrs;
    std::shared_ptr<AscendTensor<int64_t, DIMS_1> > searchMultiL2OpAttrs;
    std::shared_ptr<AscendTensor<int64_t, DIMS_1> > searchL3OpAttrs;

    std::vector<std::vector<uint8_t> > codeWordsByBucket;
    std::vector<std::vector<int64_t> > idxByBucket;
    std::vector<std::vector<float> > normL2ByBucket;

    /******************************************************************
     * Ascend Operators, 离线算子缓存
     ******************************************************************/
    // std::vector<int> ：= {1024, 128, 64, 48, 36, 32, 30, 24, 18, 16, 12, 8, 6, 4, 2, 1};
    std::vector<int> opAccessBatchList = {16, 8, 4, 2, 1};
    std::vector<int> opAccessAddBatchList = {PROGRESSBATCH};
    std::map<int, std::unique_ptr<AscendOperator> > l1TopKOps;
    std::map<int, std::unique_ptr<AscendOperator> > l2TopKOps;
    std::map<int, std::unique_ptr<AscendOperator> > l2MultiTopKOps;
    std::map<int, std::unique_ptr<AscendOperator> > l2TopKWithMaskOps;
    std::map<int, std::unique_ptr<AscendOperator> > l3TopKOps;
    std::map<int, std::unique_ptr<AscendOperator> > l3MultiTopKOps;

#ifdef USE_ACL_NN_INTERFACE
    std::map<int, aclOpExecutor *> l1DistOps;
    std::map<int, aclOpExecutor *> l3DistOps;
#else
    std::map<int, std::unique_ptr<AscendOperator> > fpToFp16Ops;
    std::map<int, std::unique_ptr<AscendOperator> > MatMulFp32L1Ops;
    std::map<int, std::unique_ptr<AscendOperator> > l1DistOps;
    std::map<int, std::unique_ptr<AscendOperator> > l2DistOps;
    std::map<int, std::unique_ptr<AscendOperator> > l3DistOps;
    std::map<int, std::unique_ptr<AscendOperator> > l3DistWithMaskOps;
#endif
    aclrtStream defaultStream = nullptr;
    aclrtStream aiCpuStream = nullptr;
    friend class NpuMultiIndexIVFHSP;
};
}  // namespace ascendSearchacc

#endif  // IVFSP_INDEXIVFHSP_H
