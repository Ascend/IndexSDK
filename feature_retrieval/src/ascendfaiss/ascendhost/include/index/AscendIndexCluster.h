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


#ifndef ASCEND_INDEX_CLUSTER_INCLUDED
#define ASCEND_INDEX_CLUSTER_INCLUDED

#include <cstdint>
#include <vector>
#include <mutex>
#include "faiss/MetaIndexes.h"
#include "ascend/AscendIndex.h"

using APP_ERROR = int;

namespace faiss {
namespace ascend {
class AscendIndexClusterImpl;

class AscendIndexCluster {
public:
    /* Construct an empty instance that can be added to */
    AscendIndexCluster();

    APP_ERROR Init(int dim, int capacity, faiss::MetricType metricType, const std::vector<int> &deviceList,
        int64_t resourceSize = -1);
    void Finalize();

    APP_ERROR AddFeatures(int n, const float *features, const uint32_t *indices);
    APP_ERROR AddFeatures(int n, const uint16_t *features, const int64_t *indices);
    AscendIndexCluster(const AscendIndexCluster&) = delete;

    AscendIndexCluster& operator=(const AscendIndexCluster&) = delete;

    /*
        distance compute interface with filter policy
        @param[in] queryIdxArr: the index of every query code in code base.
        @param[in] codeStartIdx: the start index of code base to be computed.
        @param[in] codeNum: the length of code base to be computed.
        @param[in] threshold: the value of threshold.
        @param[in] aboveFilter: true if to filter distance value is above threshold, false if to filter distance value
        is below threshold.
        @param[out] resDistArr: the distance value of code base of each query which meets threshold filter policy.
        @param[out] resIdxArr: the index of code base of each query which meets threshold filter policy.
        the shape[0] of queryIndexArr should be equal to the shape[0] of resDistArr, resIdxArr.
        for efficiency, caller can reserve max memory space(codeNum*sizeof(T)) for the second dim of resDistArr and
        resIdxArr to use repeatedly.
    */
    APP_ERROR ComputeDistanceByThreshold(const std::vector<uint32_t> &queryIdxArr, uint32_t codeStartIdx,
        uint32_t codeNum, float threshold, bool aboveFilter, std::vector<std::vector<float>> &resDistArr,
        std::vector<std::vector<uint32_t>> &resIdxArr);
    
    APP_ERROR SearchByThreshold(int n, const uint16_t *queries, float threshold, int topk, int *num,
        int64_t *indices, float *distances, unsigned int tableLen = 0, const float *table = nullptr);
    
    APP_ERROR ComputeDistanceByIdx(int n, const uint16_t *queries, const int *num, const uint32_t *indices,
        float *distances, unsigned int tableLen = 0, const float *table = nullptr);

    APP_ERROR RemoveFeatures(int n, const int64_t *indices);

    APP_ERROR GetFeatures(int n, uint16_t *features, const int64_t *indices) const;

    APP_ERROR SetNTotal(int n);

    int GetNTotal() const;

    virtual ~AscendIndexCluster() = default;

private:
    AscendIndexClusterImpl *pIndexClusterImpl; /* internal implementation */
    int capacity;
    mutable std::mutex mtx;
};
}
}
#endif
