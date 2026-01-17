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


#ifndef ASCEND_INDEXIVFFLAT_INCLUDED
#define ASCEND_INDEXIVFFLAT_INCLUDED
#include "ascenddaemon/impl/IndexIVF.h"
#include "ascenddaemon/utils/AscendOperator.h"
#include "common/threadpool/AscendThreadPool.h"

namespace ascend {
namespace {
const int IVF_FLAT_BURST_LEN = 64;
constexpr uint8_t BURST_LEN_LOW = 32;
constexpr uint8_t BURST_LEN_HIGH = 64;
constexpr uint8_t BURST_BLOCK_RATIO = 2;
constexpr uint8_t OPTIMIZE_BATCH_THRES = 48;
}

class IndexIVFFlat : public IndexIVF {
public:
    IndexIVFFlat(int numList, int dim, int nprobes, int64_t resourceSize = -1);

    ~IndexIVFFlat();

    APP_ERROR reset() override;

    APP_ERROR addVectors(int listId, size_t numVecs, const float *codes, const idx_t *indices);

    APP_ERROR searchImpl(int n, const float16_t* x, int k, float16_t* distances, idx_t* labels);

    APP_ERROR searchImpl(AscendTensor<float16_t, DIMS_2> &queries, int k,
        AscendTensor<float16_t, DIMS_2> &outDistance, AscendTensor<idx_t, DIMS_2> &outIndices);

    APP_ERROR searchPaged(size_t pageId, size_t pageNum, AscendTensor<float16_t, DIMS_2> &queries,
        AscendTensor<float16_t, DIMS_2> &maxDistances, AscendTensor<int64_t, DIMS_2> &maxIndices);
    
    APP_ERROR resizeBaseFp32(int listId, size_t numVecs);

    APP_ERROR getListVectors(int listId, std::vector<float>& codes) const;

    size_t getListLength(int listId) const;
    std::unique_ptr<DeviceVector<float>> centroidsOnDevice;
    std::unique_ptr<DeviceVector<float>> centroidsSqrSumOnDevice;
    static int GetBurstsOfBlock(int nq, int blockSize, int &burstLen)
    {
        if (faiss::ascend::SocUtils::GetInstance().IsAscend910B()) {
            burstLen = BURST_LEN_HIGH;
        } else {
            burstLen = (nq > OPTIMIZE_BATCH_THRES) ? BURST_LEN_LOW : BURST_LEN_HIGH;
        }
        return utils::divUp(blockSize, burstLen) * BURST_BLOCK_RATIO;
    }
    APP_ERROR searchImpl(int n, const float* x, int k, float* distances, idx_t* labels);
    APP_ERROR updateCentroidsSqrSum(AscendTensor<float, DIMS_2> &centroidata);
    APP_ERROR searchImplL1(AscendTensor<float, DIMS_2> &queries,
                           AscendTensor<int64_t, DIMS_2> &l1TopNprobeIndicesHost);
    APP_ERROR addVectorsAsCentroid(AscendTensor<float, DIMS_2> &centroidata);
    APP_ERROR assign(AscendTensor<float, DIMS_2> &queries,
                     AscendTensor<int64_t, DIMS_2> &l1TopNprobeIndicesHost);

protected:
    APP_ERROR addCodes(int listId, AscendTensor<uint8_t, DIMS_2> &codesData);
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
    void runL1DistOp(int batch, AscendTensor<float, DIMS_2> &queries,
                     AscendTensor<float, DIMS_2> &centroidsDev, AscendTensor<float, DIMS_2> &dists,
                     AscendTensor<float, DIMS_2> &vmdists, AscendTensor<uint16_t, DIMS_2> &opFlag,
                     aclrtStream stream);
    APP_ERROR resetL2DistOp();
    APP_ERROR searchImplL2(AscendTensor<float, DIMS_2> &queries,
        AscendTensor<int64_t, DIMS_2> &l1TopNprobeIndicesHost, int k, float* distances, idx_t* labels);
    APP_ERROR searchWithBatch(int n, const float* x, int k, float* distances, idx_t* labels);
    void runL2DistOp(AscendTensor<float, DIMS_2, size_t> &subQuery,
                     AscendTensor<float, DIMS_2, size_t> &codeVec,
                     AscendTensor<uint64_t, DIMS_1, size_t> &subOffset,
                     AscendTensor<uint32_t, DIMS_1, size_t> &subBaseSize,
                     AscendTensor<float, DIMS_2, size_t> &subDis,
                     AscendTensor<float, DIMS_2, size_t> &subVcMaxDis,
                     AscendTensor<uint16_t, DIMS_2, size_t> &subOpFlag,
                     aclrtStream stream);
    APP_ERROR resetL2TopkOp();
    void runL2TopkOp(AscendTensor<float, DIMS_3, size_t> &distResult,
                     AscendTensor<float, DIMS_3, size_t> &vmdistResult,
                     AscendTensor<int64_t, DIMS_3, size_t> &ids,
                     AscendTensor<uint32_t, DIMS_3, size_t> &sizes,
                     AscendTensor<uint16_t, DIMS_3, size_t> &flags,
                     AscendTensor<int64_t, DIMS_1> &attrs,
                     AscendTensor<float, DIMS_2, size_t> &outdists,
                     AscendTensor<uint64_t, DIMS_2, size_t> &outlabel,
                     aclrtStream stream);
    APP_ERROR fillDisOpInputData(int k, size_t batch, size_t tileNum, size_t segNum, size_t coreNum,
                                 AscendTensor<uint64_t, DIMS_3, size_t> &offset,
                                 AscendTensor<uint32_t, DIMS_3, size_t> &baseSize,
                                 AscendTensor<int64_t, DIMS_3, size_t> &ids,
                                 AscendTensor<int64_t, DIMS_1> &attrs,
                                 AscendTensor<int64_t, DIMS_2> &l1TopNprobeIndicesHost);
    void fillDisOpInputDataByBlock(size_t qIdx, size_t tIdx, size_t segIdx,
                                   size_t segNum, size_t coreNum, size_t ivfFlatBlockSize,
                                   AscendTensor<uint32_t, DIMS_3, size_t> &baseSizeHostVec,
                                   AscendTensor<uint64_t, DIMS_3, size_t> &offsetHostVec,
                                   AscendTensor<int64_t, DIMS_3, size_t> &idsHostVec,
                                   AscendTensor<int64_t, DIMS_2> &l1TopNprobeIndicesHost);
    void callL2DistanceOp(size_t batch, size_t tileNum, size_t segNum, size_t coreNum, size_t vcMaxLen,
                          AscendTensor<float, DIMS_2, size_t> &queryVec,
                          AscendTensor<uint64_t, DIMS_3, size_t> &offset,
                          AscendTensor<uint32_t, DIMS_3, size_t> &baseSize,
                          AscendTensor<uint16_t, DIMS_3, size_t> &opFlag,
                          AscendTensor<float, DIMS_3, size_t> &disVec,
                          AscendTensor<float, DIMS_3, size_t> &vcMaxDisVec,
                          AscendTensor<float, DIMS_2, size_t> &codeVec,
                          aclrtStream &stream);
    size_t getMaxListNum(size_t batch, AscendTensor<int64_t, DIMS_2> &l1TopNprobeIndicesHost, int k,
                         float* distances, idx_t* labels) const;
    
    void moveVectorForward(int listId, idx_t srcIdx, idx_t dstIdx);
    void releaseUnusageSpace(int listId, size_t oldTotal, size_t remove);
    size_t removeIds(const ascend::IDSelector& sel);

protected:
    std::unique_ptr<AscendOperator> ivfFlatIPDistOp;
    std::map<int, std::unique_ptr<AscendOperator>> l2TopkOps;
    std::vector<std::vector<std::unique_ptr<DeviceVector<float>>>> baseFp32;
    std::vector<size_t> listVecNum;
    std::map<int, std::unique_ptr<::ascend::AscendOperator>> topkFp32;
    std::map<int, std::unique_ptr<AscendOperator>> topkL2Fp32;
    std::map<int, std::unique_ptr<AscendOperator>> l1DistFp32Ops;
    std::unique_ptr<AscendOperator> ivfFlatIPOps;
    int blockSize;
    int devFp32VecCapacity;
    float* pBaseFp32;
};
}  // namespace ascend

#endif  // ASCEND_INDEXIVFFLAT_INCLUDED

