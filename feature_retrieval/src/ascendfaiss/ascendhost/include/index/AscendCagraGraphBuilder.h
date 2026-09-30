/*
 * -------------------------------------------------------------------------
 * This file is part of the IndexSDK project.
 * Copyright (c) 2026 Huawei Technologies Co.,Ltd.
 *
 * IndexSDK is licensed under Mulan PSL v2.
 * -------------------------------------------------------------------------
 */

#ifndef ASCEND_CAGRA_GRAPH_BUILDER_H
#define ASCEND_CAGRA_GRAPH_BUILDER_H

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

class AscendCagraGraphBuilderImpl;

struct AscendCagraGraphBuildConfig
{
    uint32_t intermediateDegree{128};
    uint32_t maxIterations{20};
    float terminationThreshold{0.0001F};
    bool verbose{false};
    bool guaranteeConnectivity{true};
};

class AscendCagraGraphBuilder
{
   public:
    AscendCagraGraphBuilder();
    AscendCagraGraphBuilder(const AscendCagraGraphBuilder &) = delete;
    AscendCagraGraphBuilder &operator=(const AscendCagraGraphBuilder &) = delete;
    virtual ~AscendCagraGraphBuilder();

    // The default SIMT graph builder supports FP32 dimensions in [1, 3072].
    APP_ERROR Init(int dim, int graphDegree, int dataNum, const std::vector<int> &deviceList);

    // Builds a row-major uint32 adjacency matrix with dataNum * graphDegree entries.
    // Every input element must be a finite FP32 value; NaN and infinity are unsupported.
    APP_ERROR Build(const float *data, uint32_t *graph, const AscendCagraGraphBuildConfig &config = {});

    // Convenience API that writes the same row-major matrix to a binary file. The same finite FP32 input requirement
    // applies.
    APP_ERROR BuildToFile(const float *data, const std::string &graphFilePath,
                          const AscendCagraGraphBuildConfig &config = {});

   private:
    std::unique_ptr<AscendCagraGraphBuilderImpl> impl;
    std::mutex mutex;
};
}  // namespace ascend
}  // namespace faiss

#endif  // ASCEND_CAGRA_GRAPH_BUILDER_H
