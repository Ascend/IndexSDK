/*
 * -------------------------------------------------------------------------
 * This file is part of the IndexSDK project.
 * Copyright (c) 2026 Huawei Technologies Co.,Ltd.
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
#ifndef ASCEND_INDEX_CAGRA_H
#define ASCEND_INDEX_CAGRA_H

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace faiss
{
namespace ascend
{
using APP_ERROR = int;
class AscendIndexCagraImpl;

struct AscendIndexCagraConfig
{
    uint32_t graphDegree{64};
    uint32_t intermediateDegree{128};
    uint32_t maxIterations{20};
    float terminationThreshold{0.0001F};
    bool verbose{false};
    bool guaranteeConnectivity{true};
};

class AscendIndexCagra
{
   public:
    AscendIndexCagra();
    AscendIndexCagra(const AscendIndexCagra&) = delete;
    AscendIndexCagra& operator=(const AscendIndexCagra&) = delete;
    virtual ~AscendIndexCagra();
    // End-to-end FP32/L2 index: seven dimensions, graphDegree 64/128, G <= M <= 128.
    APP_ERROR Init(int dim, const std::vector<int>& deviceList, const AscendIndexCagraConfig& config = {});
    // Train builds the full graph and quantizes the base; repeated Train is a no-op.
    // Add builds an empty index and rejects an already-trained index; Reset before adding another dataset.
    APP_ERROR Train(int n, const float* data);
    APP_ERROR Add(int n, const float* data);
    APP_ERROR Reset();
    bool IsTrained() const;
    int GetSize() const;
    APP_ERROR GetKnnGraph(std::vector<uint32_t>& graph) const;
    // Write the trained graph as N*graphDegree native uint32_t values, row-major, without a header.
    // Existing files are overwritten; the caller selects the destination path.
    APP_ERROR SaveGraphToFile(const std::string& filePath) const;

    // Legacy external-graph API; dim: seven dimensions; graphDegree: 64/128.
    // topK: {32,64,128,256,512,1024}, not greater than dataNum.
    // The dataset size is also limited by available host and device memory.
    APP_ERROR Init(int dim, int graphDegree, int dataNum, int topK, const std::vector<int>& deviceList);
    // hash is retained for ABI compatibility and must be non-null; its contents are not read.
    // Per-query visited state is allocated internally and cleared by the search kernel.
    APP_ERROR Add(const uint32_t* graph, const uint32_t* hash, const float* data);
    APP_ERROR QuantizeData(int n, const float* queryData, int ntotal, const float* baseData);
    APP_ERROR QuantizeQueries(int n, const float* queryData);
    // Unified mode automatically quantizes queries; k: {32,64,128,256,512,1024}, k <= N on each call.
    // Legacy mode requires prepared queries and topK matching the legacy Init value.
    APP_ERROR Search(int n, const float* queryData, int topK, float* dists, uint32_t* labels);

   private:
    APP_ERROR TrainLocked(int n, const float* data, bool rejectTrained);
    std::shared_ptr<AscendIndexCagraImpl> pIndexCagraImpl;
    mutable std::mutex mtx;
};
}  // namespace ascend
}  // namespace faiss
#endif  // ASCEND_INDEX_CAGRA_H
