/*
 * -------------------------------------------------------------------------
 * This file is part of the IndexSDK project.
 * Copyright (c) 2026 Huawei Technologies Co.,Ltd.
 *
 * IndexSDK is licensed under Mulan PSL v2.
 * -------------------------------------------------------------------------
 */

#ifndef ASCEND_CAGRA_GRAPH_BUILDER_IMPL_INCLUDED
#define ASCEND_CAGRA_GRAPH_BUILDER_IMPL_INCLUDED

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

#include "ascenddaemon/AscendResourcesProxy.h"
#include "ascenddaemon/utils/AscendOperator.h"
#include "common/ErrorCode.h"

namespace faiss
{
namespace ascend
{

using namespace ::ascend;

constexpr uint32_t CAGRA_MAX_INTERMEDIATE_DEGREE = 128;
constexpr uint32_t CAGRA_SAMPLE_DEGREE_CAP = 32;
constexpr uint32_t CAGRA_SIZES_CHANNEL_COUNT = 4;
constexpr uint32_t CAGRA_MIN_BACKBONE_DEGREE = 2;

class AscendCagraGraphBuilderImpl
{
   public:
    struct Config
    {
        int deviceId = 0;
        uint32_t graphDegree = 64;
        int64_t dataSize = 0;
    };

    struct BuildConfig
    {
        uint32_t intermediateDegree = CAGRA_MAX_INTERMEDIATE_DEGREE;
        uint32_t maxIterations = 20;
        float terminationThreshold = 0.0001F;
        bool verbose = false;
        bool guaranteeConnectivity = true;
    };

    AscendCagraGraphBuilderImpl(int dim, const Config &config);
    ~AscendCagraGraphBuilderImpl();

    APP_ERROR Initialize();

    /**
     * Build a row-major uint32 adjacency matrix into caller-owned memory.
     */
    APP_ERROR BuildGraph(const float *data, uint32_t *graph, const BuildConfig &buildConfig);
    APP_ERROR BuildGraphToFile(const float *data, const std::string &graphFilePath, const BuildConfig &buildConfig);

    void Reset();
    int64_t GetGraphSize() const { return graphSize; }
    uint32_t GetGraphDegree() const { return config.graphDegree; }

   private:
    APP_ERROR ResetGraphBuildOp(const BuildConfig &buildConfig);
    APP_ERROR CallGraphBuildKernel(const float *data, uint32_t *graph, const BuildConfig &buildConfig);
    APP_ERROR SetDevice();
    APP_ERROR SaveGraphToFile(const std::string &filePath, const uint32_t *graph, size_t numElements);
    APP_ERROR ValidateGraph(const uint32_t *graph, uint32_t degree) const;

   private:
    int dim;
    Config config;
    BuildConfig buildConfig;
    int64_t graphSize = 0;
    bool initialized = false;
    std::unique_ptr<AscendOperator> nndInitOp;
    std::unique_ptr<AscendOperator> nndSampleReverseOp;
    std::unique_ptr<AscendOperator> nndLocalJoinOp;
    std::unique_ptr<AscendOperator> nndUpdateOp;
    std::unique_ptr<AscendOperator> cagraPruneReverseOp;
    std::unique_ptr<AscendOperator> cagraMergeOp;
    std::unique_ptr<AscendResourcesProxy> resources;
};

}  // namespace ascend
}  // namespace faiss

#endif  // ASCEND_CAGRA_GRAPH_BUILDER_IMPL_INCLUDED
