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


#ifndef ASCEND_INDEXIVFSQ_INCLUDED
#define ASCEND_INDEXIVFSQ_INCLUDED

#include "ascenddaemon/impl/IndexIVF.h"
#include "ascenddaemon/utils/AscendOperator.h"
#include "common/threadpool/AscendThreadPool.h"

namespace ascend {
namespace {
const int THREADS_CNT = 4;
}

template<typename T>
class IndexIVFSQ : public IndexIVF {
public:
    IndexIVFSQ<T>(int numList, int dim, bool encodeResidual, int nprobes, int64_t resourceSize = -1);

    ~IndexIVFSQ<T>();

    APP_ERROR reset() override;

    // Before adding vectors, one can call this to reserve device memory
    // to improve the performance of adding
    APP_ERROR reserveMemory(size_t numVecs) override;

    APP_ERROR reserveMemory(int listId, size_t numVecs) override;

    // After adding vectors, one can call this to reclaim device memory
    // to exactly the amount needed. Returns space reclaimed in bytes
    size_t reclaimMemory() override;

    size_t reclaimMemory(int listId) override;

    virtual APP_ERROR addVectors(int listId, size_t numVecs, const uint8_t *codes,
                            const idx_t *indices, const float *preCompute);

    size_t removeIds(const ascend::IDSelector& sel) override;

    void updateCoarseCentroidsData(AscendTensor<float16_t, DIMS_2> &coarseCentroidsData);

    virtual void updateTrainedValue(AscendTensor<float16_t, DIMS_1> &trainedMin,
                                    AscendTensor<float16_t, DIMS_1> &trainedDiff);

    void calcResiduals(AscendTensor<float16_t, DIMS_1> &query,
                       AscendTensor<idx_t, DIMS_1> &nprobeIndices,
                       AscendTensor<float16_t, DIMS_2> &residulas);

    DeviceVector<float>& getListPrecompute(int listId) const;

    // whether the encoded vectors is shaped and need reconstruct when getListVectors
    bool listVectorsNeedReshaped() const override;

    // reconstruct the shaped code data to origin code when getListVectors
    APP_ERROR getListVectorsReshaped(int listId, std::vector<unsigned char>& reshaped) const override;

    APP_ERROR getListVectorsReshaped(int listId, unsigned char* reshaped) const override;

    APP_ERROR resetOp(const std::string &opTypeName,
                      std::unique_ptr<AscendOperator> &op,
                      const std::vector<std::pair<aclDataType, std::vector<int64_t>>> &input,
                      const std::vector<std::pair<aclDataType, std::vector<int64_t>>> &output);

    APP_ERROR runOp(AscendOperator *op,
                    const std::vector<const AscendTensorBase *> &input,
                    const std::vector<const AscendTensorBase *> &output,
                    aclrtStream stream);
    
    void runL2TopkOp(int batch,
                     const std::vector<const AscendTensorBase *> &input,
                     const std::vector<const AscendTensorBase *> &output,
                     aclrtStream stream);

protected:
    int getShapedDataOffset(int idx) const;

    APP_ERROR addCodes(int listId, AscendTensor<uint8_t, DIMS_2> &codesData);

    APP_ERROR addCodesAicpu(int listId, AscendTensor<uint8_t, DIMS_2> &codesData);

    APP_ERROR getListVectorsAicpu(int listId, int num, unsigned char *reshaped) const;

protected:
    AscendTensor<float16_t, DIMS_1> vMin;
    AscendTensor<float16_t, DIMS_1> vDiff;
    AscendTensor<float16_t, DIMS_2> vDM;

    std::unique_ptr<AscendOperator> distSqOp;

    std::map<int, std::unique_ptr<AscendOperator>> l2TopkOps;

    std::vector<std::unique_ptr<DeviceVector<T>>> preComputeData;

    std::unique_ptr<AscendThreadPool> threadPool;
};
}  // namespace ascend

#endif  // ASCEND_INDEXIVFSQ_INCLUDED

