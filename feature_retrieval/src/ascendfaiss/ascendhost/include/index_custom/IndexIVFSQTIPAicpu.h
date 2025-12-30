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


#ifndef ASCEND_INDEXIVFSQT_IP_AICPU_INCLUDED
#define ASCEND_INDEXIVFSQT_IP_AICPU_INCLUDED

#include <unordered_map>

#include "common/threadpool/AscendThreadPool.h"
#include "index_custom/IndexIVFSQCIPAicpu.h"

namespace ascend {
class IndexIVFSQTIPAicpu : public IndexIVFSQCIPAicpu {
public:
    IndexIVFSQTIPAicpu(int numList, int dimIn, int dimOut, int nprobes, int64_t resourceSize = -1);

    ~IndexIVFSQTIPAicpu();

    APP_ERROR init() override;

    APP_ERROR reset();

    APP_ERROR addTmpDeviceData(int n, int dim, const uint8_t *data);

    size_t getTmpPageNums() const;

    DeviceVector<unsigned char>& getPageTmpData(int pageId);

    APP_ERROR updateSubCentroidsData(int total, float16_t* x);

    APP_ERROR addPageVectors(size_t numVecs, const uint8_t *codes, const idx_t *indices);

    void receiveBaseOffsetNum(int n, int listId, const int* offset, const int* num);

    void getBaseMaskSeg();

    void updateTParams(int l2NProbeRpc, int l3SegmentNumRpc);

    void setRatio(int kBufferRatio, int kHeapRatio);

    APP_ERROR searchImpl(int n, const float16_t *x, int k, float16_t *dists, idx_t *labels);

    void setPoppedDistAndIndex(float16_t* const distsRpc, idx_t* const indexRpc);

    void setNumProbes(int nprobes) override;

    void setSearchParams(int nprobe, int l2Probe, int l3SegmentNum);

    void setSortMode(int mode);

    size_t getPageLength(int pageId) const;

    APP_ERROR getPageVectorsReshaped(int pageId, uint8_t* reshaped) const;

    APP_ERROR getPageIndices(int pageId, idx_t* ids) const;

private:
    APP_ERROR resetSubcentersDistOp();

    APP_ERROR resetSqXDistOp();

    int getPageShapedDataOffset(int idx) const;

    void clearDeviceTmpData();

    APP_ERROR searchImplL1(const AscendTensor<float16_t, DIMS_2> &queries,
                           AscendTensor<uint16_t, DIMS_2> &l1KIndices,
                           AscendTensor<float16_t, DIMS_2> &queryOut);

    APP_ERROR searchImplL2(const AscendTensor<float16_t, DIMS_2> &queries,
                           const AscendTensor<uint16_t, DIMS_2> &l1KIndices,
                           AscendTensor<uint64_t, DIMS_2> &subListOffsetL3,
                           AscendTensor<int64_t, DIMS_2> &idResult,
                           AscendTensor<uint32_t, DIMS_2> &opSize);

    void runSubcentersDistOp(const AscendTensor<float16_t, DIMS_2>& queryVecs,
                             const AscendTensor<float16_t, DIMS_4>& shapedData,
                             const AscendTensor<float16_t, DIMS_2>& norms,
                             const AscendTensor<uint16_t, DIMS_2>& offsets,
                             AscendTensor<float16_t, DIMS_3>& outDists,
                             AscendTensor<uint16_t, DIMS_2>& flag,
                             aclrtStream stream);

    APP_ERROR searchImplL3X(const AscendTensor<float16_t, DIMS_2> &queries,
                            const AscendTensor<uint64_t, DIMS_2> &subListOffsetL3,
                            const AscendTensor<int64_t, DIMS_2> &idResult,
                            const AscendTensor<uint32_t, DIMS_2> &opSize,
                            AscendTensor<float16_t, DIMS_2> &outDists,
                            AscendTensor<int64_t, DIMS_2> &outIndices);

    void runSqXDistOp(const AscendTensor<float16_t, DIMS_2> &queries,
                      const AscendTensor<uint8_t, DIMS_1> &baseSegment,
                      const AscendTensor<uint64_t, DIMS_2> &segmentOffset,
                      AscendTensor<float16_t, DIMS_2> &result,
                      AscendTensor<float16_t, DIMS_2> &maxResult,
                      AscendTensor<uint16_t, DIMS_2> &flag,
                      aclrtStream stream);

    APP_ERROR initTopkIvfFuzzyAttrs(int asc, int k, int batch, AscendTensor<int64_t, DIMS_1> &attrs) const;

    APP_ERROR resetTopkIvfFuzzyOp();

    APP_ERROR runTopkIvfFuzzyOp(const AscendTensor<float16_t, DIMS_2> &dists,
                                const AscendTensor<float16_t, DIMS_2> &vmDists,
                                const AscendTensor<int64_t, DIMS_2> &ids,
                                const AscendTensor<uint32_t, DIMS_2> &opSize,
                                const AscendTensor<uint16_t, DIMS_3> &opFlag,
                                const AscendTensor<int64_t, DIMS_1> &attr,
                                AscendTensor<float16_t, DIMS_2> &outDists,
                                AscendTensor<int64_t, DIMS_2> &outLabels,
                                AscendTensor<float16_t, DIMS_2> &popDists,
                                AscendTensor<int64_t, DIMS_2> &popLabels,
                                aclrtStream stream);

    APP_ERROR initL1TopkAttrs();

    APP_ERROR resetTopkIvfsqtL1Op();

    APP_ERROR runTopkIvfsqtL1Op(const AscendTensor<float16_t, DIMS_2> &dists,
                                const AscendTensor<float16_t, DIMS_2> &vmDists,
                                const AscendTensor<uint16_t, DIMS_3> &opFlag,
                                const AscendTensor<int64_t, DIMS_1> &attr,
                                const AscendTensor<float16_t, DIMS_2> &queryIn,
                                const AscendTensor<int, DIMS_2> &compressIndex,
                                const AscendTensor<float, DIMS_2> &compressValue,
                                AscendTensor<float16_t, DIMS_2> &outDists,
                                AscendTensor<uint16_t, DIMS_2> &outLabels,
                                AscendTensor<float16_t, DIMS_2> &queryOut,
                                aclrtStream stream);

    APP_ERROR initL2TopkAttrs();

    APP_ERROR resetTopkIvfsqtL2Op();

    APP_ERROR runTopkIvfsqtL2Op(const AscendTensor<float16_t, DIMS_2> &dists,
                                const AscendTensor<uint16_t, DIMS_3> &opFlag,
                                const AscendTensor<int64_t, DIMS_1> &attr,
                                const AscendTensor<int, DIMS_1> &listSegNum,
                                const AscendTensor<uint64_t, DIMS_1> &listOffset,
                                const AscendTensor<idx_t *, DIMS_1> &listIndicesOffset,
                                const AscendTensor<uint32_t, DIMS_1> &listSizes,
                                const AscendTensor<uint16_t, DIMS_2> &l1KIndices,
                                AscendTensor<uint64_t, DIMS_2> &listOffsetL3,
                                AscendTensor<int64_t, DIMS_2> &idResult,
                                AscendTensor<uint32_t, DIMS_2> &opSize,
                                aclrtStream stream);
    APP_ERROR copyPopToHost(AscendTensor<float16_t, DIMS_2> &popDists, AscendTensor<int64_t, DIMS_2> &popLabels);

private:
    std::vector<size_t> pageIds;
    std::vector<int> offsetsInPage;

    // params for l2 stage
    AscendTensor<float16_t, DIMS_4> subcenters;
    AscendTensor<float16_t, DIMS_2> precomputed;

    // subcenter shaped params
    AscendTensor<float16_t, DIMS_2> subCentroids;
    AscendTensor<float16_t, DIMS_4> subCentroidsShaped;
    AscendTensor<float16_t, DIMS_1> normSubCentroids;

    // params for l3 stage
    std::unordered_map<int, std::vector<int>> baseOffset;
    std::unordered_map<int, std::vector<int>> baseNums;

    AscendTensor<int, DIMS_1> subListSegNumT;
    AscendTensor<uint64_t, DIMS_1> subListOffsetT;
    AscendTensor<idx_t *, DIMS_1> subListIndicesOffsetT;
    AscendTensor<uint32_t, DIMS_1> subListSizesT;

    int l2NProbe = 48;          // L2 nprobe
    int l3SegmentNum = 96;      // L3 ops Segment num

    float16_t* distancePopped;
    idx_t *indexPopped;

    int kBufferRatio;
    int kHeapRatio;
    int ivfFuzzyTopkMode = 0;

    std::unordered_map<int, std::unique_ptr<AscendOperator>> subcentersDistOps;

    std::unordered_map<int, std::unique_ptr<AscendOperator>> distSqXOps;

    std::unique_ptr<AscendThreadPool> threadPool;

    std::vector<std::unique_ptr<DeviceVector<unsigned char>>> deviceTmpData;

    // aicpu op for topk ivf fuzzy computation
    std::unordered_map<int, std::unique_ptr<AscendOperator>> topkIvfFuzzyOps;

    // aicpu op for topk ivfsqt l1 computation
    std::unordered_map<int, std::unique_ptr<AscendOperator>> topkIvfsqtL1Ops;
    // aicpu op for topk ivfsqt l2 computation
    std::unordered_map<int, std::unique_ptr<AscendOperator>> topkIvfsqtL2Ops;

    AscendTensor<int64_t, DIMS_1> l1Attrs;
    AscendTensor<int64_t, DIMS_1> l2Attrs;
    AscendTensor<int64_t, DIMS_1> transdataShapedAttr;
};
} // namespace ascend
#endif // ASCEND_INDEXIVFSQT_IP_AICPU_INCLUDED
