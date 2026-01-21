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


#ifndef ASCEND_INDEXIVF_INCLUDED
#define ASCEND_INDEXIVF_INCLUDED

#include <vector>
#include <memory>

#include "ascenddaemon/utils/AscendTensor.h"
#include "ascenddaemon/utils/DeviceVector.h"
#include "ascenddaemon/impl/Index.h"

namespace ascend {
namespace {
const int BURST_LEN = 32; // SEARCH_LIST_SIZE must align to BURST_LEN
const int MIN_EXTREME_SIZE = 256; // the min size of extreme distance
}

class IndexIVF : public Index {
public:
    IndexIVF(int numList, int byteCntPerVector, int dim, int nprobes, int64_t resourceSize = -1);

    virtual ~IndexIVF();

    // Clear out all inverted lists
    APP_ERROR reset() override;

    // Return the number of dimension we are indexing
    int getDim() const;

    // Return the number of inverted lists
    size_t getNumLists() const;

    // Return the list length of a particular list
    size_t getListLength(int listId) const;

    // Return the list buffer length of code and id of a particular list
    size_t getMaxListDataIndicesBufferSize() const;

    // Return the list indices of a particular list in this device?
    DeviceVector<idx_t>& getListIndices(int listId) const;

    // Return the encoded vectors of a particular list in this device?
    DeviceVector<unsigned char>& getListVectors(int listId) const;

    // whether the encoded vectors is shaped and need reconstruct when getListVectors
    virtual bool listVectorsNeedReshaped() const;

    // reconstruct the shaped code data to origin code when getListVectors
    virtual APP_ERROR getListVectorsReshaped(int listId, std::vector<unsigned char>& reshaped) const;

    virtual APP_ERROR getListVectorsReshaped(int listId, unsigned char* reshaped) const;

    // Before adding vectors, one can call this to reserve device memory
    // to improve the performance of adding
    APP_ERROR reserveMemory(size_t numVecs) override;

    virtual APP_ERROR reserveMemory(int listId, size_t numVecs);

    // After adding vectors, one can call this to reclaim device memory
    // to exactly the amount needed. Returns space reclaimed in bytes
    size_t reclaimMemory() override;

    virtual size_t reclaimMemory(int listId);

    virtual void setNumProbes(int nprobes);

    virtual void updateCoarseCentroidsData(AscendTensor<float16_t, DIMS_2>& coarseCentroidsData);

    virtual int getbytesPerVector() const
    {
        return bytesPerVector;
    }

protected:
    APP_ERROR resetL1DistOp(int numLists);

    APP_ERROR resetL1TopkOp(); // aicpu op
    
    void runL1DistOp(AscendTensor<float16_t, DIMS_2>& queryVecs,
                     AscendTensor<float16_t, DIMS_4>& shapedData,
                     AscendTensor<float16_t, DIMS_1>& norms,
                     AscendTensor<float16_t, DIMS_2>& outDists,
                     AscendTensor<float16_t, DIMS_2>& outDistMins,
                     AscendTensor<uint16_t, DIMS_2>& flag,
                     aclrtStream stream);

    void runL1TopkOp(AscendTensor<float16_t, DIMS_2> &dists,
                     AscendTensor<float16_t, DIMS_2> &vmdists,
                     AscendTensor<uint32_t, DIMS_2> &sizes,
                     AscendTensor<uint16_t, DIMS_2> &flags,
                     AscendTensor<int64_t, DIMS_1> &attrs,
                     AscendTensor<float16_t, DIMS_2> &outdists,
                     AscendTensor<int64_t, DIMS_2> &outlabel,
                     aclrtStream stream);

    void addCoarseCentroidsAiCpu(AscendTensor<float16_t, DIMS_2> &src, AscendTensor<float16_t, DIMS_4> &dst);

    void addCoarseCentroidsCtrlCpu(AscendTensor<float16_t, DIMS_2> &src, AscendTensor<float16_t, DIMS_4> &dst);

protected:
    // Number of inverted files
    int numLists;

    // Number of bytes per vector in the list
    int bytesPerVector;

    // top nprobe for quantizer searching
    int nprobe;

    // Maximum list length seen
    int maxListLength;

    // the point of base for list address
    uint8_t* pListBase;

    // tensor store L1 coarse centroids
    AscendTensor<float16_t, DIMS_2> coarseCentroids;

    // tensor store L1 coarse centroids(Zz shaped data)
    AscendTensor<float16_t, DIMS_4> coarseCentroidsShaped;

    // tensor store L1 coarse centroids precomputed norms
    AscendTensor<float16_t, DIMS_1> normCoarseCentroids;

    // code data list
    std::vector<std::unique_ptr<DeviceVector<unsigned char>>> deviceListData;

    // indices data list
    std::vector<std::unique_ptr<DeviceVector<idx_t>>> deviceListIndices;
    std::vector<std::vector<idx_t>> listIndices;

    std::map<int, std::unique_ptr<AscendOperator>> l1TopkOps;
    std::map<int, std::unique_ptr<AscendOperator>> l1DistOps;
};
}  // namespace ascend

#endif  // ASCEND_INDEXIVF_INCLUDED
