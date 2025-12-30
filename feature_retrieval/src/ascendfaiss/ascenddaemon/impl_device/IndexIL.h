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


#ifndef ASCEND_IndexIL_INCLUDED
#define ASCEND_IndexIL_INCLUDED

#ifdef HOSTCPU
#include <stdint.h>
using float16_t = uint16_t;
#else
#include <arm_fp16.h>
#endif
#include <ErrorCode.h>

namespace ascend {
using idx_t = uint32_t;

enum AscendMetricType {
    ASCEND_METRIC_INNER_PRODUCT,
    ASCEND_METRIC_L2,
    ASCEND_METRIC_COSINE
};

// IndexIL: indices as label, all indices ∈ [0, capacity)
class IndexIL {
public:
    IndexIL();
    virtual ~IndexIL();

    // 申请/释放资源
    virtual APP_ERROR Init(int dim, int capacity, AscendMetricType metricType, int64_t resourceSize) = 0;

    virtual APP_ERROR Finalize() = 0;

    virtual int GetNTotal() const = 0;

    virtual APP_ERROR SetNTotal(int n) = 0;

    // 特征管理接口
    virtual APP_ERROR AddFeatures(int n, const float16_t *features, const idx_t *indices) = 0;

    virtual APP_ERROR RemoveFeatures(int n, const idx_t *indices) = 0;

    virtual APP_ERROR GetFeatures(int n, float16_t *features, const idx_t *indices) = 0;

protected:
    int dim;
    int capacity;
    int metricType;
};
}

#endif // ASCEND_IndexIL_INCLUDED
