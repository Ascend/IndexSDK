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


#include <algorithm>
#include <unordered_map>

#include <ascendsearch/ascend/utils/AscendUtils.h>
#include <ascendsearch/ascend/utils/MergeResUtils.h>

namespace faiss {
namespace ascendSearch {
std::function<bool(float, float)> GetCompareFunc(faiss::MetricType metricType)
{
    switch (metricType) {
        case faiss::METRIC_L2:
            std::less<float> lessComp;
            return lessComp;
        case faiss::METRIC_INNER_PRODUCT:
            std::greater<float> greaterComp;
            return greaterComp;
        default:
            FAISS_THROW_MSG("Unsupported metric type");
            break;
    }
}

void MergeDeviceResult(std::vector<float>& dist, std::vector<ascend_idx_t>& label, int n, int k,
    float* distances, idx_t* labels, size_t deviceCnt, faiss::MetricType metricType)
{
    auto compFunc = GetCompareFunc(metricType);

    // merge several topk results into one topk results
    // every topk result need to be reodered in ascending order
#pragma omp parallel for if (n > 10000)
    for (int i = 0; i < n; i++) {
        int num = 0;
        const int offset = i * k;
        std::vector<int> posit(deviceCnt, 0);
        while (num < k) {
            size_t id = 0;
            float disMerged = dist[offset + posit[0]];
            ascend_idx_t labelMerged = label[offset + posit[0]];
            for (size_t j = 1; j < deviceCnt; j++) {
                int pos = offset + posit[j];
                if (compFunc(dist[j * k + pos], disMerged)) {
                    disMerged = dist[j * k + pos];
                    labelMerged = label[j * k + pos];
                    id = j;
                }
            }

            *(distances + offset + num) = disMerged;
            *(labels + offset + num) = static_cast<idx_t>(labelMerged);
            posit[id]++;
            num++;
        }
    }
}

void MergeIndexResult(std::vector<std::vector<float>>& dist, std::vector<std::vector<idx_t>>& label,
    int n, int k, float* distances, idx_t* labels, size_t indexCnt, faiss::MetricType metricType)
{
    auto compFunc = GetCompareFunc(metricType);
    
#pragma omp parallel for if (n > 100)
    for (int i = 0; i < n; i++) {
        int num = 0;
        const int offset = i * k;
        std::vector<int> posit(indexCnt, 0);
        while (num < k) {
            size_t id = 0;
            float disMerged = dist[0][offset + posit[0]];
            idx_t labelMerged = label[0][offset + posit[0]];
            for (size_t j = 1; j < indexCnt; j++) {
                int pos = offset + posit[j];
                if (compFunc(dist[j][pos], disMerged)) {
                    disMerged = dist[j][pos];
                    labelMerged = label[j][pos];
                    id = j;
                }
            }

            *(distances + offset + num) = disMerged;
            *(labels + offset + num) = labelMerged;
            posit[id]++;
            num++;
        }
    }
}
}
}