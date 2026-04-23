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

#ifndef ASCEND_INDEX_CAGRA_IMPL_HOST
#define ASCEND_INDEX_CAGRA_IMPL_HOST

#include <vector>
#include <string>
#include <memory>
#include <map>
#include "ascenddaemon/AscendResourcesProxy.h"
#include "ascenddaemon/utils/AscendOperator.h"
#include "ascenddaemon/utils/AscendTensor.h"
#include "ascenddaemon/utils/DeviceVector.h"
#include "ascend/utils/fp16.h"
#include "common/ErrorCode.h"
#include "common/AscendFp16.h"

#include "index/AscendIndexCagra.h"
using ascend::APP_ERROR;
using ascend::AscendOperator;
using ascend::AscendResourcesProxy;
using ascend::AscendTensor;
using ascend::DeviceVector;
using ascend::DIMS_1;
using ascend::DIMS_2;
using ascend::DIMS_3;
using ascend::DIMS_4;

class AscendThreadPool;

namespace faiss {
namespace ascend {

class AscendIndexCagraImpl {
public:

    struct BuildConfig {
        uint32_t graphDegree = 32;           // 图的度数（K值）
        int64_t dataSize = 0;                // 数据大小（数据点数量）
        
        BuildConfig() = default;
        
        BuildConfig(uint32_t graph_degree, int64_t data_size)
            : graphDegree(graph_degree)
            , dataSize(data_size)
        {}
    };

    AscendIndexCagraImpl(const IndexCagraInitParams& params);
    virtual ~AscendIndexCagraImpl();

    APP_ERROR Init(const IndexCagraInitParams& params, const IndexCagraSearchParams& searchParams);

    APP_ERROR AddGraph(const std::vector<uint32_t>& graphData, const std::string& saveBinPath);

    APP_ERROR BuildGraph(int64_t n, const float* data, const std::string& graphFilePath, const BuildConfig& buildConfig);

    APP_ERROR Search(int n, const float* queryData, int topK, const uint32_t* graph, const uint32_t* hash,
        const float* data, float* dists, uint32_t* labels);

    AscendIndexCagraImpl(const AscendIndexCagraImpl&) = delete;
    AscendIndexCagraImpl& operator=(const AscendIndexCagraImpl&) = delete;

protected:
    int ntotal{0};
    int degree{0};
    int hashLen{0};
    int dim{0};
    int topK{0};
    size_t dataNum{0};
    int hashBitlen{0};
    int pageSize{0};
    std::vector<int> deviceList;
    int64_t ascendResourceSize;
    std::map<int, std::unique_ptr<AscendOperator>> cagraSearchOp;
    std::vector<int> searchBatchSizes;
private:

    APP_ERROR SearchImpl(int n, const float* queryData, int topK, const uint32_t* graph, const uint32_t* hash,
        const float* data, uint32_t *labels, float *dists);

    APP_ERROR SearchPaged(AscendTensor<float, DIMS_2> &queries, AscendTensor<uint32_t, DIMS_2>& graphDevice,
        AscendTensor<uint32_t, DIMS_2>& hashDevice, AscendTensor<float, DIMS_2> &data,
        AscendTensor<uint32_t, DIMS_2> &outIndices, AscendTensor<float, DIMS_2> &outDistances);
    
    void ComputeBlockDist(AscendTensor<float, DIMS_2> &queryTensor,
        AscendTensor<uint32_t, DIMS_2>& graphDevice,
        AscendTensor<uint32_t, DIMS_2>& hashDevice, AscendTensor<float, DIMS_2> &data,
        AscendTensor<uint32_t, DIMS_2> &outIndices,
        AscendTensor<float, DIMS_2> &outDistances, aclrtStream stream);

    APP_ERROR ResetCagraSearchOp();
    std::unique_ptr<AscendResourcesProxy> pResources;
    
    IndexCagraInitParams initParams;
    IndexCagraSearchParams searchParams;
    
    // Graph data
    std::vector<uint32_t> graphData;
    std::string graphBinPath;
    
    // Device information
    int deviceId;
    bool isInitialized;

    // 构图算子
    std::unique_ptr<AscendOperator> preprocessDataOp;      // 数据预处理算子
    std::unique_ptr<AscendOperator> addReverseEdgesOp;     // 添加反向边算子
    std::unique_ptr<AscendOperator> localJoinOp;           // 局部连接算子
    std::unique_ptr<AscendOperator> pruneOp;               // 剪枝算子
    std::unique_ptr<AscendOperator> makeRevGraphOp;        // 生成反向图算子
    
    BuildConfig buildConfig;
    bool graphBuildInitialized = false;
    
    std::unique_ptr<DeviceVector<ascend::fp16>> preprocessedDataDevice; // 预处理后的数据 (float16)
    std::unique_ptr<DeviceVector<float>> l2NormDataDevice;              // L2范数数据 (float32)
    
    std::unique_ptr<uint32_t[]> currentGraph;              // 当前图 (host侧)
    std::unique_ptr<uint32_t[]> newCandidates;             // 新候选邻居 (host侧)
    
    APP_ERROR buildGraphImpl(int64_t n, const float* data, uint32_t* graph);
    
    APP_ERROR prepareGraphBuild(int64_t n, const float* data, uint32_t* graph);
    
    APP_ERROR executeNNdescentIteration(int64_t n,
                                        const std::vector<int>& oldForwardEdgeCounts,
                                        const std::vector<int>& newForwardEdgeCounts);
    
    APP_ERROR executePruneAndFilter(int64_t n,
                                    std::vector<uint64_t>& destNodes);
    
    APP_ERROR executeReverseGraphGeneration(int64_t n,
                                            const std::vector<uint64_t>& destNodes,
                                            uint32_t* graph);
    
    APP_ERROR randomInitializeGraph(int64_t n, uint32_t* graph);
    
    APP_ERROR sampleOldNewNeighbors(int64_t n, const uint32_t* currentGraph, uint32_t* newCandidates,
                                    std::vector<int>& oldForwardEdgeCounts, std::vector<int>& newForwardEdgeCounts);
    
    struct SampleCounts {
        int oldCount = 0;
        int newCount = 0;
        int totalCount() const { return oldCount + newCount; }
    };
    
    void sampleNodeNeighbors(int64_t nodeId, int64_t n, const uint32_t* currentGraph,
                             uint32_t* candidates, SampleCounts& counts);
    
    APP_ERROR resetAllGraphOps();
    
    APP_ERROR resetPreprocessDataOp();
    
    APP_ERROR resetAddReverseEdgesOp();
    
    APP_ERROR resetLocalJoinOp();
    
    APP_ERROR resetPruneOp();
    
    APP_ERROR resetMakeRevGraphOp();
    
    APP_ERROR saveGraphToFile(const std::string& filePath, const uint32_t* graph, size_t numElements);

    APP_ERROR runPreprocessData(int64_t n, const float* data);
    
    APP_ERROR runAddReverseEdges(int64_t n, uint32_t actualGraphDegree,
                                 const AscendTensor<uint64_t, DIMS_1>& numSamplesTensor,
                                 const AscendTensor<uint32_t, DIMS_2>& graphDevice,
                                 const AscendTensor<int, DIMS_1>& forwardEdgeCountsDevice,
                                 AscendTensor<uint32_t, DIMS_2>& reverseGraphDevice,
                                 AscendTensor<int, DIMS_1>& reverseEdgeCountsDevice);
    
    APP_ERROR runLocalJoinKernel(int64_t n,
                                 const AscendTensor<uint32_t, DIMS_2>& newGraphDevice,
                                 const AscendTensor<uint32_t, DIMS_2>& newReverseGraphDevice,
                                 const AscendTensor<int, DIMS_1>& newForwardEdgeCountsDevice,
                                 const AscendTensor<int, DIMS_1>& newBackwardEdgeCountsDevice,
                                 const AscendTensor<uint32_t, DIMS_2>& oldGraphDevice,
                                 const AscendTensor<uint32_t, DIMS_2>& oldReverseGraphDevice,
                                 const AscendTensor<int, DIMS_1>& oldForwardEdgeCountsDevice,
                                 const AscendTensor<int, DIMS_1>& oldBackwardEdgeCountsDevice,
                                 const ascend::fp16* preprocessedData,
                                 const float* l2NormData,
                                 AscendTensor<uint32_t, DIMS_2>& outputGraphDevice,
                                 AscendTensor<float, DIMS_2>& outputDistancesDevice);
    
    APP_ERROR runPrune(int64_t n,
                       const AscendTensor<uint64_t, DIMS_2>& knnGraphDevice,
                       AscendTensor<uint8_t, DIMS_2>& detourCountDevice,
                       AscendTensor<uint32_t, DIMS_1>& numNoDetourEdgesDevice,
                       AscendTensor<uint64_t, DIMS_1>& statsDevice);
    
    APP_ERROR runMakeRevGraph(int64_t n,
                              const AscendTensor<uint64_t, DIMS_1>& destNodesDevice,
                              AscendTensor<uint64_t, DIMS_2>& revGraphDevice,
                              AscendTensor<uint32_t, DIMS_1>& revGraphCountDevice);
    
    APP_ERROR runPruneBatch(uint64_t batchId,
                            uint64_t batchSize,
                            uint64_t currentBatchSize,
                            const AscendTensor<uint64_t, DIMS_2>& knnGraphDevice,
                            const AscendTensor<uint64_t, DIMS_1>& nodeCountTensor,
                            const AscendTensor<uint64_t, DIMS_1>& inputDegreeTensor,
                            const AscendTensor<uint64_t, DIMS_1>& outputDegreeTensor,
                            AscendTensor<uint64_t, DIMS_1>& batchSizeTensor,
                            AscendTensor<uint8_t, DIMS_2>& detourCountDevice,
                            AscendTensor<uint32_t, DIMS_1>& numNoDetourEdgesDevice,
                            AscendTensor<uint64_t, DIMS_1>& statsDevice,
                            aclrtStream stream);
};

} // namespace ascend
} // namespace faiss

#endif // ASCEND_INDEX_CAGRA_IMPL_HOST