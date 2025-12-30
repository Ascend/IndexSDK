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


#ifndef ASCEND_INDEX_ILFLAT_INCLUDED
#define ASCEND_INDEX_ILFLAT_INCLUDED

#include <cstdint>
#include <vector>
#include <mutex>
#include "faiss/MetaIndexes.h"
#include "ascend/AscendIndex.h"

using APP_ERROR = int;
namespace faiss {
namespace ascend {
using float16_t = uint16_t;
using idx_t = uint32_t;

enum class MEMORY_TYPE {
    INPUT_HOST_OUTPUT_HOST = 0,
    INPUT_DEVICE_OUTPUT_DEVICE,
    INPUT_DEVICE_OUTPUT_HOST,
    INPUT_HOST_OUTPUT_DEVICE,
};

class AscendIndexILFlatImpl;
class AscendIndexILFlat {
public:
    /* Construct an empty instance that can be added to */
    AscendIndexILFlat();

    APP_ERROR Init(int dim, int capacity, faiss::MetricType metricType, const std::vector<int> &deviceList,
        int64_t resourceSize = -1);  // 共享内存默认-1为最小值128M
    void Finalize();

    APP_ERROR AddFeatures(int n, const float16_t *features);

    APP_ERROR AddFeatures(int n, const float *features);

    APP_ERROR UpdateFeatures(int n, const float16_t *features, const idx_t *indices);

    APP_ERROR UpdateFeatures(int n, const float *features, const idx_t *indices);

    APP_ERROR SearchByThreshold(int n, const float16_t *queries, float threshold, int topk, int *num,
        idx_t *indices, float *distances, unsigned int tableLen = 0, const float *table = nullptr);

    APP_ERROR SearchByThreshold(int n, const float *queries, float threshold, int topk, int *num,
        idx_t *indices, float *distances, unsigned int tableLen = 0, const float *table = nullptr);

    APP_ERROR Search(int n, const float16_t *queries, int topk, idx_t *indices, float *distances,
        unsigned int tableLen = 0, const float *table = nullptr);

    APP_ERROR Search(int n, const float *queries, int topk, idx_t *indices, float *distances,
        unsigned int tableLen = 0, const float *table = nullptr);

    APP_ERROR ComputeDistance(int n, const float16_t *queries, float *distances, unsigned int tableLen = 0,
        const float *table = nullptr);
    
    APP_ERROR ComputeDistance(int n, const float *queries, float *distances, unsigned int tableLen = 0,
        const float *table = nullptr);

    APP_ERROR ComputeDistanceByIdx(int n, const float16_t *queries, const int *num, const idx_t *indices,
        float *distances, MEMORY_TYPE memoryType = MEMORY_TYPE::INPUT_HOST_OUTPUT_HOST,
        unsigned int tableLen = 0, const float *table = nullptr);
    
    APP_ERROR ComputeDistanceByIdx(int n, const float *queries, const int *num, const idx_t *indices,
        float *distances, MEMORY_TYPE memoryType = MEMORY_TYPE::INPUT_HOST_OUTPUT_HOST,
        unsigned int tableLen = 0, const float *table = nullptr);

    APP_ERROR RemoveFeatures(int n, const idx_t *indices);

    APP_ERROR GetFeatures(int n, float16_t *features, const idx_t *indices) const;

    APP_ERROR GetFeatures(int n, float *features, const idx_t *indices) const;

    APP_ERROR GetFeaturesOnDevice(int n, float16_t *features, const idx_t *indices) const;

    APP_ERROR GetFeaturesOnDevice(int n, float *features, const idx_t *indices) const;

    APP_ERROR SetNTotal(int n);

    int GetNTotal() const;

    AscendIndexILFlat(const AscendIndexILFlat&) = delete;

    AscendIndexILFlat& operator=(const AscendIndexILFlat&) = delete;

    virtual ~AscendIndexILFlat();

private:
    AscendIndexILFlatImpl *pIndexILFlatImpl;
    int capacity;
    mutable std::mutex mtx;
};
}
}
#endif
