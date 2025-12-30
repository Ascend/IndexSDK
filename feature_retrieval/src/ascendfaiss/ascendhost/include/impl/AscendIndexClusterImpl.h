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


#ifndef ASCEND_INDEX_CLUSTER_IMPL_INCLUDED
#define ASCEND_INDEX_CLUSTER_IMPL_INCLUDED

#include <algorithm>
#include <map>
#include <vector>
#include "ascenddaemon/AscendResourcesProxy.h"
#include "ascenddaemon/utils/AscendOperator.h"
#include "ascenddaemon/utils/AscendTensor.h"
#include "ascenddaemon/utils/DeviceVector.h"
#include "common/ErrorCode.h"
#include "common/AscendFp16.h"

using ascend::APP_ERROR;
using ascend::AscendOperator;
using ascend::AscendResourcesProxy;
using ascend::AscendTensor;
using ascend::DIMS_1;
using ascend::DIMS_2;
using ascend::DIMS_3;
using ascend::DIMS_4;

namespace faiss {
namespace ascend {

constexpr int SIZE_ALIGN = 8;
constexpr int CUBE_ALIGN = 16;
constexpr int BURST_LEN = 64;
constexpr int CLUSTER_BLOCK_SIZE = 16384 * 16;
constexpr int COMPUTE_BLOCK_SIZE = CLUSTER_BLOCK_SIZE * 16;
constexpr int MIN_RESOURCE = 0x8000000; // 0x8000000 mean 128MB
constexpr int64_t MAX_RESOURCE = 0x100000000; // 0x100000000 mean 4096MB
constexpr int FAKE_HUGE_BASE = 20000000; // Fake size setted for tik operators
constexpr int MAX_CAP = 12000000; // Upper limit for capacity
constexpr size_t MAX_BASE_SPACE = 12288000000; // max bytes to store base vectors.
constexpr uint32_t MAX_THREAD_NUM = 32;
constexpr uint8_t BURST_LEN_HIGH = 64;
class AscendIndexClusterImpl {
public:
    ~AscendIndexClusterImpl();

    AscendIndexClusterImpl(int dim, int capacity, int deviceId, int64_t resourceSize);

    APP_ERROR Init();

    void Finalize();

    APP_ERROR Add(int n, const float *featuresFloat, const uint32_t *indices);
    APP_ERROR Add(int n, const uint16_t *featuresFloat, const int64_t *indices);

    APP_ERROR ComputeDistanceByThreshold(const std::vector<uint32_t> &queryIdxArr, uint32_t codeStartIdx,
        uint32_t codeNum, float threshold, bool aboveFilter, std::vector<std::vector<float>> &resDistArr,
        std::vector<std::vector<uint32_t>> &resIdxArr);
    
    APP_ERROR SearchByThreshold(int n, const uint16_t *queries, float threshold, int topk, int *num,
        int64_t *labels, float *distances, unsigned int tableLen = 0, const float *table = nullptr);

    APP_ERROR Search(int n, const uint16_t *queries, int topk,
        float *distances, int64_t *labels, unsigned int tableLen, const float *table);
    
    APP_ERROR ComputeDistanceByIdx(int n, const uint16_t *queries, const int *num,
        const uint32_t *indices, float *distances, unsigned int tableLen = 0, const float *table = nullptr);
    
    void ComputeDistWholeBaseWithTable(AscendTensor<float16_t, DIMS_2> &queryTensor,
        AscendTensor<uint32_t, DIMS_1> &sizeTensor, AscendTensor<float16_t, DIMS_4> &shapedData,
        AscendTensor<uint32_t, DIMS_2> &idxTensor, AscendTensor<float, DIMS_2> &distanceTensor,
        AscendTensor<float, DIMS_1> &tableTensor, aclrtStream stream);
    
    void ComputeDistWholeBase(AscendTensor<float16_t, DIMS_2> &queryTensor,
        AscendTensor<uint32_t, DIMS_1> &sizeTensor, AscendTensor<float16_t, DIMS_4> &shapedData,
        AscendTensor<uint32_t, DIMS_2> &idxTensor, AscendTensor<float, DIMS_2> &distanceTensor, aclrtStream stream);
    
    APP_ERROR ComputeDistByIdxImpl(int n, const float16_t *queries, float *distances,
        std::tuple<int, const int *, const uint32_t *> idxInfo, std::tuple<unsigned int, const float *> tableInfo);
    
    APP_ERROR SearchImpl(int n, const uint16_t *queries, int topk, int64_t *indices, float *distances);

    void ComputeBlockDist(AscendTensor<float16_t, DIMS_2> &queryTensor,
        AscendTensor<uint8_t, DIMS_2> &mask, AscendTensor<float16_t, DIMS_4> &shapedData,
        AscendTensor<uint32_t, DIMS_2> &size, AscendTensor<float16_t, DIMS_2> &outDistances,
        AscendTensor<float16_t, DIMS_2> &maxDistances, AscendTensor<uint16_t, DIMS_2> &flag, aclrtStream stream);

    void runTopkCompute(AscendTensor<float16_t, DIMS_3> &dists, AscendTensor<float16_t, DIMS_3> &maxDists,
        AscendTensor<uint32_t, DIMS_3> &sizes, AscendTensor<uint16_t, DIMS_3> &flags,
        AscendTensor<int64_t, DIMS_1> &attrs, AscendTensor<float16_t, DIMS_2> &outDists,
        AscendTensor<int64_t, DIMS_2> &outLabel, aclrtStream stream);

    APP_ERROR Remove(int n, const int64_t *indices);
    APP_ERROR Get(int n, uint16_t *features, const int64_t *indices) const;
    void SetNTotal(int n);
    int GetNTotal() const;

protected:
    int dim;
    int pageSize{0};
    int capacity;
    int blockSize{0};
    int deviceId;
    int64_t resourceSize;
    int ntotal{0};
    bool isInitialized;
    int blockMaskSize{0};
    std::vector<int> computeBatchSizes;
    size_t fakeBaseSizeInBytes{0};
    std::unique_ptr<AscendResourcesProxy> pResources;
    std::unique_ptr<::ascend::DeviceVector<float16_t>> baseSpace;
    std::map<int, std::unique_ptr<AscendOperator>> queryCopyOps;
    std::map<int, std::unique_ptr<AscendOperator>> distComputeOps;
    std::map<int, std::unique_ptr<AscendOperator>> distanceFlatIpOps;
    std::map<int, std::unique_ptr<AscendOperator>> distanceFilterOps;
    std::map<int, std::unique_ptr<AscendOperator>> distComputeIdxOps;
    std::map<int, std::unique_ptr<AscendOperator>> distComputeIdxWithTableOps;
    std::map<int, std::unique_ptr<::ascend::AscendOperator>> topkComputeOps;

private:
    size_t GetMaxCapacity() const;
    APP_ERROR ComputeDistByThresholdBatched(const std::vector<uint32_t> &queryIdxArr, uint32_t codeStartIdx,
        uint32_t codeNum, float threshold, std::vector<std::vector<float>> &resDistArr,
        std::vector<std::vector<uint32_t>> &resIdxArr);

    APP_ERROR ComputeDistPaged(uint32_t n, AscendTensor<uint32_t, DIMS_1> &queryTensor, uint32_t codeStartIdxPad,
        uint32_t codeNumPad, float threshold, float *distances, uint32_t *indice, uint64_t *distOffset);

    APP_ERROR ComputeDistImpl(uint32_t n, AscendTensor<uint32_t, DIMS_1> &queryTensor,
    uint32_t codeStartIdx, uint32_t codeNumPad, float threshold, float *distances, uint32_t *indice,
    uint64_t *computedOffset);

    APP_ERROR DistanceFilter(AscendTensor<float, DIMS_1> &distanceTensor, AscendTensor<float, DIMS_1> &thresholdTensor,
        AscendTensor<uint32_t, DIMS_1> &baseSize, AscendTensor<uint32_t, DIMS_1> &initIdx,
        AscendTensor<float, DIMS_2> &filteredDist, AscendTensor<uint32_t, DIMS_2> &filteredIndice,
        AscendTensor<uint32_t, DIMS_1> &filteredCnt, aclrtStream stream);

    APP_ERROR DistanceFlatIpByIdx2(AscendTensor<uint32_t, DIMS_1> &queryOffset, AscendTensor<uint32_t, DIMS_1> &size,
        AscendTensor<float16_t, DIMS_4> &shape, AscendTensor<float, DIMS_1> &outTensor, aclrtStream stream);

    APP_ERROR AddPage(int num, const float *featuresFloat, uint32_t indice);
    APP_ERROR AddPage(int num, const uint16_t *featuresFloat, int64_t indice);
    
    APP_ERROR SearchPaged(int pageId, AscendTensor<float16_t, DIMS_2> &queries, int pageNum,
        AscendTensor<int64_t, DIMS_2> &idxs, AscendTensor<float16_t, DIMS_2> &distances);
    
    APP_ERROR TableMapping(int n, float *distances, size_t tableLen, const float *table, int topk);

    APP_ERROR ResetDistCompIdxWithTableOp();
    APP_ERROR ResetDistanceFilterOp();
    APP_ERROR ResetDistanceFlatIpByIdx2Op();
    APP_ERROR ResetDistCompIdxOp();
    APP_ERROR ResetTopkCompOp();
    APP_ERROR ResetDistCompOp(int numLists);

    APP_ERROR CheckComputeMaxNum(int n, const int *num, const uint32_t *indices, int& maxNum);
    APP_ERROR CheckParams(int n, const uint16_t *queries, const int *num, float *distances,
        unsigned int tableLen, const float *table);
    APP_ERROR InitAttr(int topk, int burstLen, int blockNum, int pageId, int pageNum,
        AscendTensor<int64_t, DIMS_1>& attrsInput);
    APP_ERROR CopyDisToHost(int maxNum, int idxCopyNum, float *distances, int n,
        AscendTensor<float, DIMS_3>& distResult);
    APP_ERROR MoveData(int& idxCopyNum, int maxNum, int n, const uint32_t *indice,
        AscendTensor<uint32_t, DIMS_3>& idxTensor);
};
}
}
#endif
