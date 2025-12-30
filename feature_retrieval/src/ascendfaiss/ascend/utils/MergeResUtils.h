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


#ifndef ASCEND_MERGE_RES_UTILS_INCLUDED
#define ASCEND_MERGE_RES_UTILS_INCLUDED

#include <faiss/Index.h>
#include "ascend/AscendIndex.h"

namespace faiss {
namespace ascend {
// Merge the result of search from different device
// `dist` and `label` is the result of search, the size is deviceCnt * n * k
// `distances` and `labels` is used to save merge result, the size is n * k
void MergeDeviceResult(std::vector<float>& dist, std::vector<ascend_idx_t>& label, int n, int k,
    float* distances, idx_t* labels, size_t deviceCnt, faiss::MetricType metricType);

// Merge the result of search from different index
// `dist` and `label` is the result of search, the size is indexCnt * n * k
// `distances` and `labels` is used to save merge result, the size is n * k
void MergeIndexResult(std::vector<std::vector<float>>& dist, std::vector<std::vector<idx_t>>& label,
    int n, int k, float* distances, idx_t* labels, size_t indexCnt, faiss::MetricType metricType);
} // ascend
} // faiss
#endif