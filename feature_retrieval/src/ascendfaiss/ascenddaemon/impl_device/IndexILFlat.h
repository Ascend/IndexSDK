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


#ifndef ASCEND_INDEXILFLAT_INCLUDED
#define ASCEND_INDEXILFLAT_INCLUDED

#include "IndexIL.h"
#include <cstdint>

namespace ascend {
class IndexILFlatImpl;

class IndexILFlat : public IndexIL {
public:
    IndexILFlat();
    virtual ~IndexILFlat();

    IndexILFlat(const IndexILFlat&) = delete;
    
    IndexILFlat& operator=(const IndexILFlat&) = delete;

    int GetNTotal() const override;

    // 按实际有效idx精确设置ntotal
    APP_ERROR SetNTotal(int n) override;

    // 申请/释放资源
    APP_ERROR Init(int dim, int capacity, AscendMetricType metricType, int64_t resourceSize = -1) override;

    APP_ERROR Finalize() override;

    // 特征管理接口
    APP_ERROR AddFeatures(int n, const float16_t *features, const idx_t *indices) override;

    APP_ERROR RemoveFeatures(int n, const idx_t *indices) override;

    APP_ERROR GetFeatures(int n, float16_t *features, const idx_t *indices) override;

    // 比对相关接口
    /* *
     * distances: [out] len=n*ntotal
     */
    APP_ERROR ComputeDistance(int n, const float16_t *queries, float *distances, unsigned int tableLen = 0,
        const float *table = nullptr);

    /* *
     * indices: [out] len=n*topk
     * distances: [out] len=n*topk
     */
    APP_ERROR Search(int n, const float16_t *queries, int topk, idx_t *indices, float *distances,
        unsigned int tableLen = 0, const float *table = nullptr);

    /* *
     * threshold: [in] 有table是拉伸后的threshold，无table是拉伸前的threshold
     * num: [out] len=n, result count by threshold for each query
     * distances: [out] len=n*ntotal, result of each query is offset by ntotal
     * indices: [out] len=n*ntotal, result of each query is offset by ntotal
     */
    APP_ERROR ComputeDistanceByThreshold(int n, const float16_t *queries, float threshold, int *num, idx_t *indices,
        float *distances, unsigned int tableLen = 0, const float *table = nullptr);

    /* *
     * threshold: [in] 有table是拉伸后的threshold，无table是拉伸前的threshold
     * num: [out] len=n, result count by threshold for each query
     * distances: [out] len=n*topk, result of each query is offset by topk
     * indices: [out] len=n*topk, result of each query is offset by topk
     */
    APP_ERROR SearchByThreshold(int n, const float16_t *queries, float threshold, int topk, int *num, idx_t *indices,
        float *distances, unsigned int tableLen = 0, const float *table = nullptr);

    /* *
     * num: [in] len=n, feature count by indices for each query
     * distances: [out] len=n * max(num)
     * indices:   [in] len=n * max(num)
     */
    APP_ERROR ComputeDistanceByIdx(int n, const float16_t *queries, const int *num, const idx_t *indices,
        float *distances, unsigned int tableLen = 0, const float *table = nullptr);

protected:
    IndexILFlatImpl *pIndexILFlatImpl; // internal implementation
};
}
#endif
