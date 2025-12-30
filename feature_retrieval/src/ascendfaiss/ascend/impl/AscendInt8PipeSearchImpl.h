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


#ifndef ASCEND_INT8_PIPE_SEARCH_IMPL_INCLUDED
#define ASCEND_INT8_PIPE_SEARCH_IMPL_INCLUDED

#include <algorithm>
#include "ascend/AscendIndexInt8.h"
#include "ascend/utils/fp16.h"

namespace faiss {
namespace ascend {
class AscendInt8PipeSearchImpl {
public:
    AscendInt8PipeSearchImpl(AscendIndexInt8 *index, const AscendIndexInt8FlatConfig &config)
        : indexProxy(index), int8FlatConfig(config)
    {
        InitSearchPipeline();
    }
    ~AscendInt8PipeSearchImpl() {}

    void InitSearchPipeline()
    {
        searchPipelineQuery = std::unique_ptr<std::vector<int8_t>>(new std::vector<int8_t>());
        searchPipelineQueryPrev = std::unique_ptr<std::vector<int8_t>>(new std::vector<int8_t>());
        searchPipelineQueryNorm = std::unique_ptr<std::vector<float>>(new std::vector<float>());
        searchPipelineQueryNormPrev = std::unique_ptr<std::vector<float>>(new std::vector<float>());
        searchPipelineDist = std::unique_ptr<std::vector<std::vector<float>>>(new std::vector<std::vector<float>>());
        searchPipelineDistPrev =
            std::unique_ptr<std::vector<std::vector<float>>>(new std::vector<std::vector<float>>());
        searchPipelineDistHalf =
            std::unique_ptr<std::vector<std::vector<uint16_t>>>(new std::vector<std::vector<uint16_t>>());
        searchPipelineDistHalfPrev =
            std::unique_ptr<std::vector<std::vector<uint16_t>>>(new std::vector<std::vector<uint16_t>>());
        searchPipelineLabel =
            std::unique_ptr<std::vector<std::vector<ascend_idx_t>>>(new std::vector<std::vector<ascend_idx_t>>());
        searchPipelineLabelPrev =
            std::unique_ptr<std::vector<std::vector<ascend_idx_t>>>(new std::vector<std::vector<ascend_idx_t>>());
    }

    void SearchPipelinePrepare(int n, int k, const int8_t *x, size_t offset) const
    {
        size_t deviceCnt = int8FlatConfig.deviceList.size();

        // convert query data from float to fp16, device use fp16 data to search
        searchPipelineQuery->resize(n * indexProxy->getDim(), 0);
        transform(x + offset * indexProxy->getDim(), x + (offset + n) * indexProxy->getDim(),
                  searchPipelineQuery->begin(), [](int8_t temp) { return temp; });
        searchPipelineQueryNorm->resize(n, 0);
        searchPipelineDist->resize(deviceCnt, std::vector<float>(n * k, 0));
        searchPipelineDistHalf->resize(deviceCnt, std::vector<uint16_t>(n * k, 0));
        searchPipelineLabel->resize(deviceCnt, std::vector<ascend_idx_t>(n * k, 0));
    }

    void SearchPipelineFinish(int n, int k) const
    {
        // scale for int32 convert to fp16
        float scale = 0.01 / std::min(indexProxy->getDim() / 64, std::max(indexProxy->getDim() / 128 + 1, 4));
        size_t deviceCnt = int8FlatConfig.deviceList.size();
        for (size_t idx = 0; idx < deviceCnt; ++idx) {
            // convert result data from fp16 to float
            transform(searchPipelineDistHalfPrev->at(idx).begin(), searchPipelineDistHalfPrev->at(idx).end(),
                searchPipelineDistPrev->at(idx).begin(), [](uint16_t temp) { return (float)fp16(temp); });
            if (indexProxy->getMetricType() != faiss::METRIC_L2) {
                continue;
            }

            for (int i = 0; i < n; ++i) {
                for (int j = 0; j < k; ++j) {
                    float distSum = searchPipelineDistPrev->at(idx)[i * k + j] + searchPipelineQueryNormPrev->data()[i];
                    searchPipelineDistPrev->at(idx)[i * k + j] = (distSum < 0.0) ? 0.0 :
                        std::sqrt((searchPipelineDistPrev->at(idx)[i * k + j] +
                                   searchPipelineQueryNormPrev->data()[i]) / scale);
                }
            }
        }
    }

    void ClearPrevData()
    {
        searchPipelineQueryPrev->clear();
        searchPipelineQueryNormPrev->clear();
        searchPipelineDistPrev->clear();
        searchPipelineDistHalfPrev->clear();
        searchPipelineLabelPrev->clear();
    }

    void MoveDataForNext()
    {
        // move data to prepare the next page
        *searchPipelineQueryPrev = std::move(*searchPipelineQuery);
        *searchPipelineQueryNormPrev = std::move(*searchPipelineQueryNorm);
        *searchPipelineDistPrev = std::move(*searchPipelineDist);
        *searchPipelineDistHalfPrev = std::move(*searchPipelineDistHalf);
        *searchPipelineLabelPrev = std::move(*searchPipelineLabel);
    }

    // for pipelining search (we use pointer because search has a const interface)
    std::unique_ptr<std::vector<int8_t>> searchPipelineQuery = nullptr;
    std::unique_ptr<std::vector<int8_t>> searchPipelineQueryPrev = nullptr;
    std::unique_ptr<std::vector<float>> searchPipelineQueryNorm = nullptr;
    std::unique_ptr<std::vector<float>> searchPipelineQueryNormPrev = nullptr;
    std::unique_ptr<std::vector<std::vector<float>>> searchPipelineDist = nullptr;
    std::unique_ptr<std::vector<std::vector<float>>> searchPipelineDistPrev = nullptr;
    std::unique_ptr<std::vector<std::vector<uint16_t>>> searchPipelineDistHalf = nullptr;
    std::unique_ptr<std::vector<std::vector<uint16_t>>> searchPipelineDistHalfPrev = nullptr;
    std::unique_ptr<std::vector<std::vector<ascend_idx_t>>> searchPipelineLabel = nullptr;
    std::unique_ptr<std::vector<std::vector<ascend_idx_t>>> searchPipelineLabelPrev = nullptr;

    AscendIndexInt8 *indexProxy;
    const AscendIndexInt8FlatConfig &int8FlatConfig;
};
}  // namespace ascend
}  // namespace faiss
#endif  // ASCEND_INT8_PIPE_SEARCH_IMPL_INCLUDED