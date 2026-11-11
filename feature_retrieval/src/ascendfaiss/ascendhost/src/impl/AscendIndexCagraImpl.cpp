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

#include "impl/AscendIndexCagraImpl.h"

#include <acl/acl.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <initializer_list>
#include <limits>
#include <memory>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "ascenddaemon/utils/AscendOpDesc.h"
#include "ascenddaemon/utils/AscendTensor.h"
#include "ascenddaemon/utils/AscendUtils.h"
#include "ascenddaemon/utils/Limits.h"
#include "ascenddaemon/utils/Random.h"
#include "common/ErrorCode.h"
#include "common/utils/CommonUtils.h"
#include "common/utils/DataType.h"
#include "common/utils/LogUtils.h"
#include "common/utils/SocUtils.h"

using namespace ascend;
namespace faiss
{
namespace ascend
{
namespace
{

// Construction resources are scoped to Train and released before search data is allocated.
using InputBuffers = std::shared_ptr<std::vector<const aclDataBuffer *>>;
using OutputBuffers = std::shared_ptr<std::vector<aclDataBuffer *>>;
using GraphBuildClock = std::chrono::steady_clock;
constexpr uint32_t CAGRA_MAX_INTERMEDIATE_DEGREE = 128;
constexpr uint32_t CAGRA_SAMPLE_DEGREE_CAP = 32;
constexpr uint32_t CAGRA_SIZES_CHANNEL_COUNT = 4;
constexpr uint32_t CAGRA_MIN_BACKBONE_DEGREE = 2;
constexpr uint32_t CAGRA_NODE_MASK = 0x7fffffffU;
constexpr int MAX_CAGRA_BUILD_DIM = 3072;
constexpr int CAGRA_SIMT_CACHED_DIM = 128;
constexpr int CAGRA_SIMT_BATCHED_MAX_DIM = 512;

void MoveNearestNeighborFirst(const float *data, uint32_t *graph, int64_t nodeNum, int dim, uint32_t degree)
{
    // BuildGraph validates the graph before this pass. Reorder only existing edges, after
    // pruning, reverse-edge merging and optional connectivity repair have all finished.
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int64_t row = 0; row < nodeNum; ++row)
    {
        const float *query = data + static_cast<size_t>(row) * dim;
        uint32_t *neighbors = graph + static_cast<size_t>(row) * degree;
        uint32_t bestRank = 0;
        float bestDistance = std::numeric_limits<float>::infinity();
        for (uint32_t rank = 0; rank < degree; ++rank)
        {
            const float *candidate = data + static_cast<size_t>(neighbors[rank]) * dim;
            float distance = 0.0F;
            for (int d = 0; d < dim; ++d)
            {
                const float diff = query[d] - candidate[d];
                distance += diff * diff;
            }
            // Equal distances keep their existing priority.
            if (distance < bestDistance)
            {
                bestDistance = distance;
                bestRank = rank;
            }
        }
        std::rotate(neighbors, neighbors + bestRank, neighbors + bestRank + 1);
    }
}

uint32_t FindGraphRoot(std::vector<uint32_t> &parents, uint32_t node)
{
    uint32_t root = node;
    while (parents[root] != root)
    {
        root = parents[root];
    }
    while (parents[node] != node)
    {
        const uint32_t parent = parents[node];
        parents[node] = root;
        node = parent;
    }
    return root;
}

bool AddBackboneEdge(uint32_t lhs, uint32_t rhs, uint32_t maximumDegree, std::vector<std::vector<uint32_t>> &backbone)
{
    if (lhs == rhs || backbone[lhs].size() >= maximumDegree || backbone[rhs].size() >= maximumDegree)
    {
        return false;
    }
    if (std::find(backbone[lhs].begin(), backbone[lhs].end(), rhs) != backbone[lhs].end())
    {
        return false;
    }
    backbone[lhs].push_back(rhs);
    backbone[rhs].push_back(lhs);
    return true;
}

APP_ERROR GuaranteeGraphConnectivity(const std::vector<uint32_t> &intermediateGraph, uint32_t *graph, uint32_t nodeNum,
                                     uint32_t intermediateDegree, uint32_t outputDegree, bool verbose)
{
    APPERR_RETURN_IF_NOT(graph != nullptr, APP_ERR_INVALID_PARAM);
    if (nodeNum <= 1 || outputDegree < CAGRA_MIN_BACKBONE_DEGREE)
    {
        return APP_ERR_OK;
    }

    const uint32_t maximumBackboneDegree = std::max(CAGRA_MIN_BACKBONE_DEGREE, outputDegree / 2);
    std::vector<uint32_t> parents(nodeNum);
    std::iota(parents.begin(), parents.end(), 0U);
    std::vector<std::vector<uint32_t>> backbone(nodeNum);
    uint32_t componentCount = nodeNum;

    // Consume the sorted NN-Descent graph rank by rank, as cuVS approximate MST optimization does.
    for (uint32_t rank = 0; rank < intermediateDegree && componentCount > 1; ++rank)
    {
        for (uint32_t row = 0; row < nodeNum; ++row)
        {
            const uint32_t neighbor =
                intermediateGraph[static_cast<uint64_t>(row) * intermediateDegree + rank] & CAGRA_NODE_MASK;
            if (neighbor >= nodeNum || neighbor == row)
            {
                continue;
            }
            uint32_t rowRoot = FindGraphRoot(parents, row);
            uint32_t neighborRoot = FindGraphRoot(parents, neighbor);
            if (rowRoot == neighborRoot || !AddBackboneEdge(row, neighbor, maximumBackboneDegree, backbone))
            {
                continue;
            }
            parents[neighborRoot] = rowRoot;
            --componentCount;
        }
    }

    std::vector<std::vector<uint32_t>> components(nodeNum);
    for (uint32_t row = 0; row < nodeNum; ++row)
    {
        components[FindGraphRoot(parents, row)].push_back(row);
    }
    uint32_t mainRoot = 0;
    size_t mainSize = 0;
    for (uint32_t root = 0; root < nodeNum; ++root)
    {
        if (components[root].size() > mainSize)
        {
            mainRoot = root;
            mainSize = components[root].size();
        }
    }
    const uint32_t disconnectedComponents = componentCount;

    // cuVS connects a disconnected KNN forest to its largest component. Distribute the endpoints so
    // that no single node consumes the protected degree budget.
    size_t mainCursor = 0;
    for (uint32_t root = 0; root < nodeNum; ++root)
    {
        if (root == mainRoot || components[root].empty())
        {
            continue;
        }
        uint32_t source = nodeNum;
        for (uint32_t node : components[root])
        {
            if (backbone[node].size() < maximumBackboneDegree)
            {
                source = node;
                break;
            }
        }
        uint32_t target = nodeNum;
        for (size_t attempt = 0; attempt < components[mainRoot].size(); ++attempt)
        {
            const uint32_t node = components[mainRoot][mainCursor % components[mainRoot].size()];
            ++mainCursor;
            if (backbone[node].size() < maximumBackboneDegree)
            {
                target = node;
                break;
            }
        }
        APPERR_RETURN_IF_NOT_LOG(source < nodeNum && target < nodeNum, APP_ERR_INNER_ERROR,
                                 "Unable to allocate a degree-constrained CAGRA connectivity edge");
        APPERR_RETURN_IF_NOT_LOG(AddBackboneEdge(source, target, maximumBackboneDegree, backbone), APP_ERR_INNER_ERROR,
                                 "Failed to add a CAGRA connectivity edge");
    }

    std::vector<uint32_t> mergedRow(outputDegree);
    uint64_t protectedEdges = 0;
    uint32_t maximumProtectedDegree = 0;
    for (uint32_t row = 0; row < nodeNum; ++row)
    {
        uint32_t size = 0;
        for (uint32_t neighbor : backbone[row])
        {
            mergedRow[size++] = neighbor;
        }
        protectedEdges += size;
        maximumProtectedDegree = std::max(maximumProtectedDegree, size);
        const uint64_t rowOffset = static_cast<uint64_t>(row) * outputDegree;
        const auto appendGraphRange = [&](uint32_t begin, uint32_t end, uint32_t targetSize)
        {
            for (uint32_t rank = begin; rank < end && size < targetSize; ++rank)
            {
                const uint32_t candidate = graph[rowOffset + rank];
                if (candidate >= nodeNum || candidate == row ||
                    std::find(mergedRow.begin(), mergedRow.begin() + size, candidate) != mergedRow.begin() + size)
                {
                    continue;
                }
                mergedRow[size++] = candidate;
            }
        };

        // CagraMerge protects the first half of the pruned row and places reverse edges in the
        // second half. Insert the connectivity backbone into that protected budget instead of
        // prepending it to the whole row and truncating reverse edges from the tail. This follows
        // the cuVS merge order: connectivity edges, protected pruned edges, then reverse edges.
        const uint32_t protectedDegree = outputDegree / 2;
        const uint32_t protectedTarget = std::max(protectedDegree, size);
        appendGraphRange(0, protectedDegree, protectedTarget);
        appendGraphRange(protectedDegree, outputDegree, protectedTarget);
        appendGraphRange(protectedDegree, outputDegree, outputDegree);
        appendGraphRange(0, protectedDegree, outputDegree);
        APPERR_RETURN_IF_NOT_LOG(size == outputDegree, APP_ERR_INNER_ERROR,
                                 "Connectivity merge produced an incomplete CAGRA graph row");
        std::copy(mergedRow.begin(), mergedRow.end(), graph + rowOffset);
    }

    if (verbose)
    {
        std::fprintf(stderr,
                     "CAGRA connectivity backbone input_components=%u protected_directed_edges=%llu "
                     "max_protected_degree=%u\n",
                     disconnectedComponents, static_cast<unsigned long long>(protectedEdges), maximumProtectedDegree);
        std::fflush(stderr);
    }
    return APP_ERR_OK;
}

InputBuffers MakeInputs(std::initializer_list<const AscendTensorBase *> tensors)
{
    InputBuffers buffers(new std::vector<const aclDataBuffer *>(), CommonUtils::AclInputBufferDelete);
    for (const auto *tensor : tensors)
    {
        aclDataBuffer *buffer = aclCreateDataBuffer(tensor->getVoidData(), tensor->getSizeInBytes());
        ASCEND_THROW_IF_NOT_MSG(buffer != nullptr, "aclCreateDataBuffer failed for CAGRA graph-build input");
        buffers->emplace_back(buffer);
    }
    return buffers;
}

OutputBuffers MakeOutputs(std::initializer_list<AscendTensorBase *> tensors)
{
    OutputBuffers buffers(new std::vector<aclDataBuffer *>(), CommonUtils::AclOutputBufferDelete);
    for (auto *tensor : tensors)
    {
        aclDataBuffer *buffer = aclCreateDataBuffer(tensor->getVoidData(), tensor->getSizeInBytes());
        ASCEND_THROW_IF_NOT_MSG(buffer != nullptr, "aclCreateDataBuffer failed for CAGRA graph-build output");
        buffers->emplace_back(buffer);
    }
    return buffers;
}

void InitGraphBuildOperator(std::unique_ptr<AscendOperator> &op, const char *opName)
{
    if (op != nullptr && op->init())
    {
        return;
    }

    const char *recentError = aclGetRecentErrMsg();
    op.reset();
    ASCEND_THROW_FMT("%s aclopCreateHandle failed, recentError=%s", opName,
                     recentError == nullptr ? "<empty>" : recentError);
}

void SynchronizeGraphBuildStep(aclrtStream stream, const char *step, bool verbose,
                               GraphBuildClock::time_point startTime)
{
    const aclError ret = synchronizeStream(stream);
    if (ret != ACL_ERROR_NONE)
    {
        const char *recentError = aclGetRecentErrMsg();
        ASCEND_THROW_FMT("CAGRA graph-build step %s failed, aclError=%d, recentError=%s", step, static_cast<int>(ret),
                         recentError == nullptr ? "<empty>" : recentError);
    }
    if (verbose)
    {
        const double elapsedMs = std::chrono::duration<double, std::milli>(GraphBuildClock::now() - startTime).count();
        std::fprintf(stderr, "CAGRA graph-build step %s done elapsed_ms=%.4f\n", step, elapsedMs);
        std::fflush(stderr);
    }
}
class CagraGraphBuildContext
{
   public:
    struct Config
    {
        int deviceId = 0;
        uint32_t graphDegree = 64;
        int64_t dataSize = 0;
    };

    using BuildConfig = AscendIndexCagraConfig;

    CagraGraphBuildContext(int dim, const Config &config);
    ~CagraGraphBuildContext();

    CagraGraphBuildContext(const CagraGraphBuildContext &) = delete;
    CagraGraphBuildContext &operator=(const CagraGraphBuildContext &) = delete;
    CagraGraphBuildContext(CagraGraphBuildContext &&) = delete;
    CagraGraphBuildContext &operator=(CagraGraphBuildContext &&) = delete;

    APP_ERROR Initialize();

    /**
     * Build a row-major uint32 adjacency matrix into caller-owned memory.
     */
    APP_ERROR BuildGraph(const float *data, uint32_t *graph, const BuildConfig &buildConfig);

    void Reset();

   private:
    APP_ERROR ResetGraphBuildOp(const BuildConfig &buildConfig);
    APP_ERROR CallGraphBuildKernel(const float *data, uint32_t *graph, const BuildConfig &buildConfig);
    APP_ERROR SetDevice();
    APP_ERROR ValidateGraph(const uint32_t *graph, uint32_t degree) const;

   private:
    int dim;
    Config config;
    BuildConfig buildConfig;
    bool initialized = false;
    std::unique_ptr<AscendOperator> nndInitOp;
    std::unique_ptr<AscendOperator> nndSampleReverseOp;
    std::unique_ptr<AscendOperator> nndLocalJoinOp;
    std::unique_ptr<AscendOperator> nndUpdateOp;
    std::unique_ptr<AscendOperator> cagraPruneReverseOp;
    std::unique_ptr<AscendOperator> cagraMergeOp;
    std::unique_ptr<AscendResourcesProxy> resources;
};

CagraGraphBuildContext::CagraGraphBuildContext(int dim, const Config &config) : dim(dim), config(config) {}

CagraGraphBuildContext::~CagraGraphBuildContext() { Reset(); }

APP_ERROR CagraGraphBuildContext::Initialize()
{
    if (initialized)
    {
        return APP_ERR_OK;
    }
    APPERR_RETURN_IF_NOT(dim > 0 && dim <= MAX_CAGRA_BUILD_DIM, APP_ERR_INVALID_PARAM);
    APPERR_RETURN_IF_NOT(config.dataSize > 1 && config.dataSize <= std::numeric_limits<int32_t>::max(),
                         APP_ERR_INVALID_PARAM);
    APPERR_RETURN_IF_NOT((config.graphDegree == 64 || config.graphDegree == 128) &&
                             config.graphDegree < static_cast<uint64_t>(config.dataSize),
                         APP_ERR_INVALID_PARAM);
    APPERR_RETURN_IF_NOT(config.deviceId >= 0, APP_ERR_INVALID_PARAM);
    const uint32_t deviceCount = SocUtils::GetInstance().GetDeviceCount();
    APPERR_RETURN_IF_NOT(static_cast<uint32_t>(config.deviceId) < deviceCount, APP_ERR_INVALID_PARAM);
    auto ret = SetDevice();
    ASCEND_THROW_IF_NOT_MSG(ret == APP_ERR_OK, "SetDevice failed");
    resources = CREATE_UNIQUE_PTR(AscendResourcesProxy);
    initialized = true;
    return APP_ERR_OK;
}

APP_ERROR CagraGraphBuildContext::SetDevice()
{
    aclError ret = aclrtSetDevice(config.deviceId);
    ASCEND_THROW_IF_NOT_MSG(ret == ACL_ERROR_NONE, "Set device failed");
    return APP_ERR_OK;
}

APP_ERROR CagraGraphBuildContext::BuildGraph(const float *data, uint32_t *graph, const BuildConfig &newBuildConfig)
{
    APPERR_RETURN_IF_NOT(initialized, APP_ERR_INDEX_NOT_INIT);
    APPERR_RETURN_IF_NOT(dim > 0, APP_ERR_INVALID_PARAM);
    APPERR_RETURN_IF_NOT(dim <= MAX_CAGRA_BUILD_DIM, APP_ERR_INVALID_PARAM);
    APPERR_RETURN_IF_NOT(config.dataSize > 1 && config.dataSize <= std::numeric_limits<int32_t>::max(),
                         APP_ERR_INVALID_PARAM);
    APPERR_RETURN_IF_NOT(data != nullptr, APP_ERR_INVALID_PARAM);
    APPERR_RETURN_IF_NOT(graph != nullptr, APP_ERR_INVALID_PARAM);
    APPERR_RETURN_IF_NOT(newBuildConfig.intermediateDegree >= config.graphDegree, APP_ERR_INVALID_PARAM);
    APPERR_RETURN_IF_NOT(newBuildConfig.intermediateDegree <= CAGRA_MAX_INTERMEDIATE_DEGREE, APP_ERR_INVALID_PARAM);
    APPERR_RETURN_IF_NOT(newBuildConfig.intermediateDegree < static_cast<uint64_t>(config.dataSize),
                         APP_ERR_INVALID_PARAM);
    APPERR_RETURN_IF_NOT(newBuildConfig.maxIterations > 0, APP_ERR_INVALID_PARAM);
    APPERR_RETURN_IF_NOT(newBuildConfig.terminationThreshold >= 0.0F && newBuildConfig.terminationThreshold <= 1.0F,
                         APP_ERR_INVALID_PARAM);

    const bool shapeChanged = nndInitOp == nullptr || nndSampleReverseOp == nullptr || nndLocalJoinOp == nullptr ||
                              nndUpdateOp == nullptr || cagraPruneReverseOp == nullptr || cagraMergeOp == nullptr ||
                              buildConfig.intermediateDegree != newBuildConfig.intermediateDegree;
    buildConfig = newBuildConfig;

    auto ret = SetDevice();
    ASCEND_THROW_IF_NOT_MSG(ret == APP_ERR_OK, "SetDevice failed");
    if (shapeChanged)
    {
        ret = ResetGraphBuildOp(buildConfig);
        ASCEND_THROW_IF_NOT_MSG(ret == APP_ERR_OK, "ResetGraphBuildOp failed");
    }

    ret = CallGraphBuildKernel(data, graph, buildConfig);
    ASCEND_THROW_IF_NOT_MSG(ret == APP_ERR_OK, "CallGraphBuildKernel failed");
    ret = ValidateGraph(graph, config.graphDegree);
    ASCEND_THROW_IF_NOT_MSG(ret == APP_ERR_OK, "CAGRA graph validation failed");
    const auto reorderStart = GraphBuildClock::now();
    MoveNearestNeighborFirst(data, graph, config.dataSize, dim, config.graphDegree);
    if (buildConfig.verbose)
    {
        const double elapsedMs =
            std::chrono::duration<double, std::milli>(GraphBuildClock::now() - reorderStart).count();
        std::fprintf(stderr, "CAGRA graph-build step NearestNeighborFirst done elapsed_ms=%.4f\n", elapsedMs);
        std::fflush(stderr);
    }
    return APP_ERR_OK;
}

APP_ERROR
CagraGraphBuildContext::ResetGraphBuildOp(const BuildConfig &cfg)
{
    const int64_t sampleDegree = std::min<int64_t>(CAGRA_SAMPLE_DEGREE_CAP, cfg.intermediateDegree);
    const std::vector<int64_t> dataShape({config.dataSize, static_cast<int64_t>(dim)});
    const std::vector<int64_t> intermediateShape({config.dataSize, cfg.intermediateDegree});
    const std::vector<int64_t> sampleShape({config.dataSize, sampleDegree});
    const std::vector<int64_t> sizesShape({config.dataSize, CAGRA_SIZES_CHANNEL_COUNT});
    const std::vector<int64_t> countShape({config.dataSize});
    const std::vector<int64_t> scalarShape({1});
    const std::vector<int64_t> graphShape({config.dataSize, config.graphDegree});

    nndInitOp.reset();
    nndSampleReverseOp.reset();
    nndLocalJoinOp.reset();
    nndUpdateOp.reset();
    cagraPruneReverseOp.reset();
    cagraMergeOp.reset();

    {
        AscendOpDesc desc("CagraNndInit");
        desc.addInputTensorDesc(ACL_FLOAT, dataShape.size(), dataShape.data(), ACL_FORMAT_ND);
        desc.addOutputTensorDesc(ACL_UINT32, intermediateShape.size(), intermediateShape.data(), ACL_FORMAT_ND);
        desc.addOutputTensorDesc(ACL_FLOAT, intermediateShape.size(), intermediateShape.data(), ACL_FORMAT_ND);
        ACL_REQUIRE_OK(aclopSetAttrInt(desc.opAttr, "intermediate_degree", cfg.intermediateDegree));
        nndInitOp = CREATE_UNIQUE_PTR(AscendOperator, desc);
        InitGraphBuildOperator(nndInitOp, "CagraNndInit");
    }
    {
        AscendOpDesc desc("CagraNndSampleReverse");
        desc.addInputTensorDesc(ACL_UINT32, intermediateShape.size(), intermediateShape.data(), ACL_FORMAT_ND);
        for (uint32_t i = 0; i < CAGRA_SIZES_CHANNEL_COUNT; ++i)
        {
            desc.addOutputTensorDesc(ACL_UINT32, sampleShape.size(), sampleShape.data(), ACL_FORMAT_ND);
        }
        desc.addOutputTensorDesc(ACL_UINT32, sizesShape.size(), sizesShape.data(), ACL_FORMAT_ND);
        nndSampleReverseOp = CREATE_UNIQUE_PTR(AscendOperator, desc);
        InitGraphBuildOperator(nndSampleReverseOp, "CagraNndSampleReverse");
    }
    {
        AscendOpDesc desc("CagraNndLocalJoin");
        desc.addInputTensorDesc(ACL_FLOAT, dataShape.size(), dataShape.data(), ACL_FORMAT_ND);
        for (uint32_t i = 0; i < CAGRA_SIZES_CHANNEL_COUNT; ++i)
        {
            desc.addInputTensorDesc(ACL_UINT32, sampleShape.size(), sampleShape.data(), ACL_FORMAT_ND);
        }
        desc.addInputTensorDesc(ACL_UINT32, sizesShape.size(), sizesShape.data(), ACL_FORMAT_ND);
        desc.addOutputTensorDesc(ACL_UINT32, intermediateShape.size(), intermediateShape.data(), ACL_FORMAT_ND);
        desc.addOutputTensorDesc(ACL_FLOAT, intermediateShape.size(), intermediateShape.data(), ACL_FORMAT_ND);
        desc.addOutputTensorDesc(ACL_UINT32, countShape.size(), countShape.data(), ACL_FORMAT_ND);
        ACL_REQUIRE_OK(aclopSetAttrInt(desc.opAttr, "candidate_degree", cfg.intermediateDegree));
        nndLocalJoinOp = CREATE_UNIQUE_PTR(AscendOperator, desc);
        InitGraphBuildOperator(nndLocalJoinOp, "CagraNndLocalJoin");
    }
    {
        AscendOpDesc desc("CagraNndUpdate");
        desc.addInputTensorDesc(ACL_UINT32, intermediateShape.size(), intermediateShape.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_FLOAT, intermediateShape.size(), intermediateShape.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_UINT32, sampleShape.size(), sampleShape.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_UINT32, sizesShape.size(), sizesShape.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_UINT32, intermediateShape.size(), intermediateShape.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_FLOAT, intermediateShape.size(), intermediateShape.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_UINT32, countShape.size(), countShape.data(), ACL_FORMAT_ND);
        desc.addOutputTensorDesc(ACL_UINT32, intermediateShape.size(), intermediateShape.data(), ACL_FORMAT_ND);
        desc.addOutputTensorDesc(ACL_FLOAT, intermediateShape.size(), intermediateShape.data(), ACL_FORMAT_ND);
        desc.addOutputTensorDesc(ACL_UINT32, scalarShape.size(), scalarShape.data(), ACL_FORMAT_ND);
        nndUpdateOp = CREATE_UNIQUE_PTR(AscendOperator, desc);
        InitGraphBuildOperator(nndUpdateOp, "CagraNndUpdate");
    }
    {
        AscendOpDesc desc("CagraPruneReverse");
        desc.addInputTensorDesc(ACL_UINT32, intermediateShape.size(), intermediateShape.data(), ACL_FORMAT_ND);
        desc.addOutputTensorDesc(ACL_UINT32, graphShape.size(), graphShape.data(), ACL_FORMAT_ND);
        desc.addOutputTensorDesc(ACL_UINT32, graphShape.size(), graphShape.data(), ACL_FORMAT_ND);
        desc.addOutputTensorDesc(ACL_UINT32, countShape.size(), countShape.data(), ACL_FORMAT_ND);
        ACL_REQUIRE_OK(aclopSetAttrInt(desc.opAttr, "output_degree", config.graphDegree));
        cagraPruneReverseOp = CREATE_UNIQUE_PTR(AscendOperator, desc);
        InitGraphBuildOperator(cagraPruneReverseOp, "CagraPruneReverse");
    }
    {
        AscendOpDesc desc("CagraMerge");
        desc.addInputTensorDesc(ACL_UINT32, graphShape.size(), graphShape.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_UINT32, graphShape.size(), graphShape.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_UINT32, countShape.size(), countShape.data(), ACL_FORMAT_ND);
        desc.addOutputTensorDesc(ACL_UINT32, graphShape.size(), graphShape.data(), ACL_FORMAT_ND);
        cagraMergeOp = CREATE_UNIQUE_PTR(AscendOperator, desc);
        InitGraphBuildOperator(cagraMergeOp, "CagraMerge");
    }
    return APP_ERR_OK;
}

APP_ERROR CagraGraphBuildContext::CallGraphBuildKernel(const float *data, uint32_t *graph, const BuildConfig &cfg)
{
    APPERR_RETURN_IF_NOT(nndInitOp != nullptr && nndSampleReverseOp != nullptr && nndLocalJoinOp != nullptr &&
                             nndUpdateOp != nullptr && cagraPruneReverseOp != nullptr && cagraMergeOp != nullptr,
                         APP_ERR_ACL_OP_NOT_FOUND);
    APPERR_RETURN_IF_NOT(resources != nullptr, APP_ERR_INNER_ERROR);

    auto streamPtr = resources->getDefaultStream();
    auto stream = streamPtr->GetStream();
    auto &mem = resources->getMemoryManager();
    const int n32 = static_cast<int>(config.dataSize);
    const int intermediateDegree = static_cast<int>(cfg.intermediateDegree);
    const int graphDegree = static_cast<int>(config.graphDegree);
    const int sampleDegree = static_cast<int>(std::min<uint32_t>(CAGRA_SAMPLE_DEGREE_CAP, cfg.intermediateDegree));

    AscendTensor<float, DIMS_2> dataDevice(mem, {n32, dim}, stream);
    AscendTensor<uint32_t, DIMS_2> graphA(mem, {n32, intermediateDegree}, stream);
    AscendTensor<uint32_t, DIMS_2> graphB(mem, {n32, intermediateDegree}, stream);
    AscendTensor<float, DIMS_2> distanceA(mem, {n32, intermediateDegree}, stream);
    AscendTensor<float, DIMS_2> distanceB(mem, {n32, intermediateDegree}, stream);
    AscendTensor<uint32_t, DIMS_2> forwardNew(mem, {n32, sampleDegree}, stream);
    AscendTensor<uint32_t, DIMS_2> forwardOld(mem, {n32, sampleDegree}, stream);
    AscendTensor<uint32_t, DIMS_2> reverseNew(mem, {n32, sampleDegree}, stream);
    AscendTensor<uint32_t, DIMS_2> reverseOld(mem, {n32, sampleDegree}, stream);
    AscendTensor<uint32_t, DIMS_2> sizes(mem, {n32, static_cast<int>(CAGRA_SIZES_CHANNEL_COUNT)}, stream);
    AscendTensor<uint32_t, DIMS_2> candidateIds(mem, {n32, intermediateDegree}, stream);
    AscendTensor<float, DIMS_2> candidateDistances(mem, {n32, intermediateDegree}, stream);
    AscendTensor<uint32_t, DIMS_1> candidateCounts(mem, {n32}, stream);
    AscendTensor<uint32_t, DIMS_1> updateCountDevice(mem, {1}, stream);
    AscendTensor<uint32_t, DIMS_2> prunedGraph(mem, {n32, graphDegree}, stream);
    AscendTensor<uint32_t, DIMS_2> reverseGraph(mem, {n32, graphDegree}, stream);
    AscendTensor<uint32_t, DIMS_1> reverseCounts(mem, {n32}, stream);
    AscendTensor<uint32_t, DIMS_2> finalGraph(mem, {n32, graphDegree}, stream);

    aclError aclRet =
        aclrtMemcpy(dataDevice.data(), dataDevice.getSizeInBytes(), data,
                    static_cast<size_t>(config.dataSize) * dim * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE);
    APPERR_RETURN_IF_NOT_LOG(aclRet == ACL_ERROR_NONE, APP_ERR_INNER_ERROR, "Failed to copy CAGRA data to device");

    auto stepStart = GraphBuildClock::now();
    {
        auto inputs = MakeInputs({&dataDevice});
        auto outputs = MakeOutputs({&graphA, &distanceA});
        nndInitOp->exec(*inputs, *outputs, stream);
    }
    if (cfg.verbose)
    {
        SynchronizeGraphBuildStep(stream, "CagraNndInit", true, stepStart);
        const char *distancePath =
            dim <= CAGRA_SIMT_CACHED_DIM
                ? "cached"
                : (dim <= CAGRA_SIMT_BATCHED_MAX_DIM ? "streaming_batched4" : "streaming_paired2");
        std::fprintf(stderr, "CAGRA graph-build local_join_mode=simt distance_path=%s dim=%d\n", distancePath, dim);
        std::fflush(stderr);
    }

    AscendTensor<uint32_t, DIMS_2> *currentGraph = &graphA;
    AscendTensor<uint32_t, DIMS_2> *nextGraph = &graphB;
    AscendTensor<float, DIMS_2> *currentDistances = &distanceA;
    AscendTensor<float, DIMS_2> *nextDistances = &distanceB;
    const uint64_t edgeCount = static_cast<uint64_t>(config.dataSize) * cfg.intermediateDegree;
    const uint64_t stopUpdates =
        static_cast<uint64_t>(static_cast<double>(cfg.terminationThreshold) * static_cast<double>(edgeCount));
    std::vector<uint32_t> candidateCountsHost;
    if (cfg.verbose)
    {
        candidateCountsHost.resize(config.dataSize);
    }

    uint32_t iterationsExecuted = 0;
    uint32_t finalUpdateCount = 0;
    bool converged = false;
    for (uint32_t iteration = 0; iteration < cfg.maxIterations; ++iteration)
    {
        sizes.zero();
        candidateCounts.zero();
        updateCountDevice.zero();
        stepStart = GraphBuildClock::now();
        {
            auto inputs = MakeInputs({currentGraph});
            auto outputs = MakeOutputs({&forwardNew, &forwardOld, &reverseNew, &reverseOld, &sizes});
            nndSampleReverseOp->exec(*inputs, *outputs, stream);
        }
        if (cfg.verbose)
        {
            SynchronizeGraphBuildStep(stream, "CagraNndSampleReverse", true, stepStart);
        }
        stepStart = GraphBuildClock::now();
        {
            auto inputs = MakeInputs({&dataDevice, &forwardNew, &forwardOld, &reverseNew, &reverseOld, &sizes});
            auto outputs = MakeOutputs({&candidateIds, &candidateDistances, &candidateCounts});
            nndLocalJoinOp->exec(*inputs, *outputs, stream);
        }
        if (cfg.verbose)
        {
            SynchronizeGraphBuildStep(stream, "CagraNndLocalJoin", true, stepStart);
            aclRet = aclrtMemcpy(candidateCountsHost.data(), candidateCounts.getSizeInBytes(), candidateCounts.data(),
                                 candidateCounts.getSizeInBytes(), ACL_MEMCPY_DEVICE_TO_HOST);
            APPERR_RETURN_IF_NOT_LOG(aclRet == ACL_ERROR_NONE, APP_ERR_INNER_ERROR,
                                     "Failed to read NN-Descent candidate counts");
            uint64_t candidateReservations = 0;
            uint64_t droppedCandidates = 0;
            uint32_t rowsAtCapacity = 0;
            uint32_t maximumReservations = 0;
            for (uint32_t count : candidateCountsHost)
            {
                candidateReservations += count;
                maximumReservations = std::max(maximumReservations, count);
                if (count >= cfg.intermediateDegree)
                {
                    ++rowsAtCapacity;
                }
                if (count > cfg.intermediateDegree)
                {
                    droppedCandidates += count - cfg.intermediateDegree;
                }
            }
            std::fprintf(stderr,
                         "CAGRA NN-Descent iteration=%u proposals=%llu dropped_after_capacity=%llu "
                         "full_rows=%u max_proposals_per_row=%u capacity=%u count_mode=exact\n",
                         iteration, static_cast<unsigned long long>(candidateReservations),
                         static_cast<unsigned long long>(droppedCandidates), rowsAtCapacity, maximumReservations,
                         cfg.intermediateDegree);
            std::fflush(stderr);
        }
        stepStart = GraphBuildClock::now();
        {
            auto inputs = MakeInputs({currentGraph, currentDistances, &forwardNew, &sizes, &candidateIds,
                                      &candidateDistances, &candidateCounts});
            auto outputs = MakeOutputs({nextGraph, nextDistances, &updateCountDevice});
            nndUpdateOp->exec(*inputs, *outputs, stream);
        }
        SynchronizeGraphBuildStep(stream, "CagraNndUpdate", cfg.verbose, stepStart);
        uint32_t updateCount = 0;
        aclRet = aclrtMemcpy(&updateCount, sizeof(updateCount), updateCountDevice.data(), sizeof(updateCount),
                             ACL_MEMCPY_DEVICE_TO_HOST);
        APPERR_RETURN_IF_NOT_LOG(aclRet == ACL_ERROR_NONE, APP_ERR_INNER_ERROR,
                                 "Failed to read NN-Descent update count");
        std::swap(currentGraph, nextGraph);
        std::swap(currentDistances, nextDistances);
        iterationsExecuted = iteration + 1;
        finalUpdateCount = updateCount;
        if (cfg.verbose)
        {
            std::fprintf(stderr, "CAGRA NN-Descent iteration=%u updates=%u threshold=%llu\n", iteration, updateCount,
                         static_cast<unsigned long long>(stopUpdates));
            std::fflush(stderr);
        }
        if (updateCount <= stopUpdates)
        {
            converged = true;
            break;
        }
    }
    if (cfg.verbose)
    {
        std::fprintf(stderr, "CAGRA NN-Descent summary iterations=%u final_updates=%u threshold=%llu converged=%s\n",
                     iterationsExecuted, finalUpdateCount, static_cast<unsigned long long>(stopUpdates),
                     converged ? "true" : "false");
        std::fflush(stderr);
    }

    std::vector<uint32_t> intermediateGraphHost;
    if (cfg.guaranteeConnectivity)
    {
        stepStart = GraphBuildClock::now();
        intermediateGraphHost.resize(edgeCount);
        aclRet = aclrtMemcpy(intermediateGraphHost.data(), currentGraph->getSizeInBytes(), currentGraph->data(),
                             currentGraph->getSizeInBytes(), ACL_MEMCPY_DEVICE_TO_HOST);
        APPERR_RETURN_IF_NOT_LOG(aclRet == ACL_ERROR_NONE, APP_ERR_INNER_ERROR,
                                 "Failed to copy the NN-Descent graph for connectivity optimization");
        if (cfg.verbose)
        {
            const double elapsedMs =
                std::chrono::duration<double, std::milli>(GraphBuildClock::now() - stepStart).count();
            std::fprintf(stderr, "CAGRA graph-build step CopyIntermediateGraph done elapsed_ms=%.4f\n", elapsedMs);
            std::fflush(stderr);
        }
    }

    reverseCounts.zero();
    stepStart = GraphBuildClock::now();
    {
        auto inputs = MakeInputs({currentGraph});
        auto outputs = MakeOutputs({&prunedGraph, &reverseGraph, &reverseCounts});
        cagraPruneReverseOp->exec(*inputs, *outputs, stream);
    }
    if (cfg.verbose)
    {
        SynchronizeGraphBuildStep(stream, "CagraPruneReverse", true, stepStart);
    }
    stepStart = GraphBuildClock::now();
    {
        auto inputs = MakeInputs({&prunedGraph, &reverseGraph, &reverseCounts});
        auto outputs = MakeOutputs({&finalGraph});
        cagraMergeOp->exec(*inputs, *outputs, stream);
    }

    SynchronizeGraphBuildStep(stream, "CagraMerge", cfg.verbose, stepStart);
    aclRet = aclrtMemcpy(graph, finalGraph.getSizeInBytes(), finalGraph.data(), finalGraph.getSizeInBytes(),
                         ACL_MEMCPY_DEVICE_TO_HOST);
    APPERR_RETURN_IF_NOT_LOG(aclRet == ACL_ERROR_NONE, APP_ERR_INNER_ERROR, "Failed to copy CAGRA graph from device");
    if (cfg.guaranteeConnectivity)
    {
        stepStart = GraphBuildClock::now();
        const APP_ERROR connectivityRet =
            GuaranteeGraphConnectivity(intermediateGraphHost, graph, static_cast<uint32_t>(config.dataSize),
                                       cfg.intermediateDegree, config.graphDegree, cfg.verbose);
        APPERR_RETURN_IF_NOT(connectivityRet == APP_ERR_OK, connectivityRet);
        if (cfg.verbose)
        {
            const double elapsedMs =
                std::chrono::duration<double, std::milli>(GraphBuildClock::now() - stepStart).count();
            std::fprintf(stderr, "CAGRA graph-build step GuaranteeConnectivity done elapsed_ms=%.4f\n", elapsedMs);
            std::fflush(stderr);
        }
    }
    return APP_ERR_OK;
}

void CagraGraphBuildContext::Reset()
{
    nndInitOp.reset();
    nndSampleReverseOp.reset();
    nndLocalJoinOp.reset();
    nndUpdateOp.reset();
    cagraPruneReverseOp.reset();
    cagraMergeOp.reset();
    resources.reset();
    initialized = false;
}

APP_ERROR CagraGraphBuildContext::ValidateGraph(const uint32_t *graph, uint32_t degree) const
{
    for (int64_t row = 0; row < config.dataSize; ++row)
    {
        const size_t rowOffset = static_cast<size_t>(row) * degree;
        for (uint32_t rank = 0; rank < degree; ++rank)
        {
            const uint32_t node = graph[rowOffset + rank];
            if (node >= static_cast<uint64_t>(config.dataSize) || node == static_cast<uint64_t>(row))
            {
                APP_LOG_ERROR("Invalid CAGRA edge: row=%ld rank=%u node=%u\n", row, rank, node);
                return APP_ERR_INNER_ERROR;
            }
            for (uint32_t previous = 0; previous < rank; ++previous)
            {
                if (graph[rowOffset + previous] == node)
                {
                    APP_LOG_ERROR("Duplicate CAGRA edge: row=%ld node=%u\n", row, node);
                    return APP_ERR_INNER_ERROR;
                }
            }
        }
    }
    return APP_ERR_OK;
}

const uint32_t RANDOM_SEED = 1234;
const int LOG_INTERVAL = 10;
const int MAX_SEARCH_BATCH = 64;
const int PRE_COMPUTE_SIZE = 4;
const int MAX_QUERY_BATCH_SIZE = 4096;
std::vector<int> ASCEND_CAGRA_SEARCH_BATCHES = {64, 32, 16, 8, 4, 2, 1};
}  // namespace

AscendIndexCagraImpl::AscendIndexCagraImpl(int dim, int topK, const std::vector<int> &deviceList)
    : dim(dim), topK(topK), deviceList(deviceList)
{
}

AscendIndexCagraImpl::~AscendIndexCagraImpl()
{
    if (pResources != nullptr) (void)aclrtSetDevice(deviceId);
}

APP_ERROR AscendIndexCagraImpl::Init(int graphDegree, int dataNum) { return Init(graphDegree, dataNum, true); }

APP_ERROR AscendIndexCagraImpl::Init(int graphDegree, int dataNum, bool eagerSearchModels)
{
    APP_LOG_INFO("AscendIndexCagraImpl::Init start");
    APPERR_RETURN_IF_NOT_LOG(graphDegree == 64 || graphDegree == 128, APP_ERR_INVALID_PARAM,
                             "CAGRA graph degree must be 64 or 128");

    deviceId = deviceList[0];
    ASCEND_THROW_IF_NOT(deviceId >= 0);
    this->searchBatchSizes = ASCEND_CAGRA_SEARCH_BATCHES;
    auto ret = aclrtSetDevice(deviceId);
    APP_LOG_INFO("AscendIndexCagraImpl::Init on device(%d).\n", deviceId);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_ACL_SET_DEVICE_FAILED, "failed to set device(%d)", ret);
    cagraSearchOp.clear();
    pResources = CREATE_UNIQUE_PTR(AscendResourcesProxy);
    pResources->setTempMemory(ascendResourceSize);
    pResources->initialize();
    int queryBit = 6;
    int baseBit = 6;
    int offset = (dim + 7) / 8;
    this->rotatedSize = offset * queryBit;
    this->codeSize = offset * baseBit + 16;
    this->dataNum = dataNum;
    this->degree = graphDegree;

    ret = eagerSearchModels ? ResetCagraSearchOp(this->topK) : APP_ERR_OK;
    APPERR_RETURN_IF_NOT_LOG(ret == APP_ERR_OK, APP_ERR_ACL_OP_LOAD_MODEL_FAILED, "ResetCagraSearchOp init failed !!!");

    APP_LOG_INFO("AscendIndexCagraImpl::Init finished");
    return APP_ERR_OK;
}

APP_ERROR AscendIndexCagraImpl::Add(const uint32_t *graph, const uint32_t *hash, const float *data)
{
    // Init receives the row count as int; tensor extents use the same type.
    const int rowCount = static_cast<int>(this->dataNum);
    AscendTensor<uint32_t, DIMS_2> graphDevice({rowCount, this->degree});
    auto ret = aclrtMemcpy(graphDevice.data(), graphDevice.getSizeInBytes(), graph,
                           this->dataNum * this->degree * sizeof(uint32_t), ACL_MEMCPY_HOST_TO_DEVICE);
    APPERR_RETURN_IF_NOT_FMT((ret == ACL_SUCCESS), APP_ERR_INNER_ERROR, "copy graphDevice to device fail(%d)!", ret);

    // The public hash argument is retained for API compatibility; visited state is per query.
    (void)hash;
    const int hashWords = static_cast<int>((this->dataNum + 31) / 32);
    AscendTensor<uint32_t, DIMS_2> hashDevice({MAX_SEARCH_BATCH, hashWords});
    hashDevice.zero();

    AscendTensor<float, DIMS_2> dataDevice({rowCount, this->dim});
    ret = aclrtMemcpy(dataDevice.data(), dataDevice.getSizeInBytes(), data, this->dataNum * this->dim * sizeof(float),
                      ACL_MEMCPY_HOST_TO_DEVICE);
    APPERR_RETURN_IF_NOT_FMT((ret == ACL_SUCCESS), APP_ERR_INNER_ERROR, "copy dataDevice to device fail(%d)!", ret);
    g_graphDevice = std::move(graphDevice);
    g_hashDevice = std::move(hashDevice);
    g_dataDevice = std::move(dataDevice);
    g_code = AscendTensor<uint8_t, DIMS_1, int64_t>();
    g_centroid.clear();
    g_inverseRotation.clear();
    g_precompute.clear();
    g_rotated.clear();
    trained = false;
    return APP_ERR_OK;
}

void AscendIndexCagraImpl::Configure(const AscendIndexCagraConfig &config)
{
    unifiedMode = true;
    buildConfig = config;
}

std::shared_ptr<AscendIndexCagraImpl> AscendIndexCagraImpl::CreateEmpty() const
{
    auto empty = std::make_shared<AscendIndexCagraImpl>(dim, 0, deviceList);
    empty->Configure(buildConfig);
    return empty;
}

APP_ERROR AscendIndexCagraImpl::Train(int n, const float *data)
{
    APPERR_RETURN_IF_NOT_LOG(unifiedMode && data != nullptr && n > static_cast<int>(buildConfig.intermediateDegree),
                             APP_ERR_INVALID_PARAM, "Train requires N > intermediateDegree");
    const size_t elements = static_cast<size_t>(n) * dim;
    APPERR_RETURN_IF_NOT_LOG(std::all_of(data, data + elements, [](float v) { return std::isfinite(v); }),
                             APP_ERR_INVALID_PARAM, "CAGRA input must contain finite FP32 values");
    std::vector<uint32_t> graph(static_cast<size_t>(n) * buildConfig.graphDegree);
    {
        CagraGraphBuildContext::Config config;
        config.deviceId = deviceList[0];
        config.graphDegree = buildConfig.graphDegree;
        config.dataSize = n;
        CagraGraphBuildContext context(dim, config);
        auto ret = context.Initialize();
        if (ret != APP_ERR_OK) return ret;
        ret = context.BuildGraph(data, graph.data(), buildConfig);
        if (ret != APP_ERR_OK) return ret;
    }  // Release construction buffers before allocating the persistent search data.
    auto ret = Init(buildConfig.graphDegree, n, false);
    if (ret != APP_ERR_OK) return ret;
    ret = Add(graph.data(), nullptr, data);
    if (ret != APP_ERR_OK) return ret;
    return QuantizeBase(n, data);
}

APP_ERROR AscendIndexCagraImpl::GetKnnGraph(std::vector<uint32_t> &graph) const
{
    APPERR_RETURN_IF_NOT_LOG(trained, APP_ERR_INDEX_NOT_INIT, "CAGRA is not trained");
    auto ret = aclrtSetDevice(deviceId);
    APPERR_RETURN_IF_NOT_LOG(ret == ACL_SUCCESS, APP_ERR_ACL_SET_DEVICE_FAILED, "Failed to select CAGRA device");
    std::vector<uint32_t> copy(dataNum * degree);
    ret = aclrtMemcpy(copy.data(), copy.size() * sizeof(uint32_t), g_graphDevice.data(), g_graphDevice.getSizeInBytes(),
                      ACL_MEMCPY_DEVICE_TO_HOST);
    APPERR_RETURN_IF_NOT_LOG(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "Failed to download CAGRA graph");
    graph.swap(copy);
    return APP_ERR_OK;
}

APP_ERROR AscendIndexCagraImpl::SaveGraphToFile(const std::string &filePath) const
{
    std::vector<uint32_t> graph;
    const auto ret = GetKnnGraph(graph);
    APPERR_RETURN_IF_NOT(ret == APP_ERR_OK, ret);
    FILE *file = std::fopen(filePath.c_str(), "wb");
    APPERR_RETURN_IF_NOT_LOG(file != nullptr, APP_ERR_INVALID_PARAM, "Failed to open CAGRA graph file");
    const size_t written = std::fwrite(graph.data(), sizeof(uint32_t), graph.size(), file);
    const int closeResult = std::fclose(file);
    APPERR_RETURN_IF_NOT_LOG(written == graph.size() && closeResult == 0, APP_ERR_INNER_ERROR,
                             "Failed to write complete CAGRA graph file");
    return APP_ERR_OK;
}

APP_ERROR FloatToBytes(float val, uint8_t *bytes)
{
    uint32_t bits;
    int ret = memcpy_s(&bits, sizeof(bits), &val, sizeof(float));
    APPERR_RETURN_IF_NOT_FMT((ret == 0), APP_ERR_INNER_ERROR, "memcpy_s failed in FloatToBytes(%d)!", ret);
    bytes[0] = static_cast<uint8_t>(bits & 0xFFu);
    bytes[1] = static_cast<uint8_t>((bits >> 8) & 0xFFu);
    bytes[2] = static_cast<uint8_t>((bits >> 16) & 0xFFu);
    bytes[3] = static_cast<uint8_t>((bits >> 24) & 0xFFu);
    return APP_ERR_OK;
}

APP_ERROR PackbitsLittle(const uint8_t *bits, int n, uint8_t *packed)
{
    int nBytes = (n + 7) / 8;
    int ret = memset_s(packed, nBytes, 0, nBytes);
    APPERR_RETURN_IF_NOT_FMT((ret == 0), APP_ERR_INNER_ERROR, "memset_s failed in PackbitsLittle(%d)!", ret);
    for (int i = 0; i < n; ++i)
        if (bits[i] & 1u) packed[i / 8] |= static_cast<uint8_t>(1u << (i % 8));
    return APP_ERR_OK;
}

std::vector<float> AscendIndexCagraImpl::GenerateOrthogonalMatrix(int D, unsigned int seed)
{
    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> mat(D * D);
    for (int i = 0; i < D * D; ++i) mat[i] = dist(rng);

    std::vector<float> Q(D * D);
    for (int i = 0; i < D; ++i)
    {
        // 复制当前列
        for (int j = 0; j < D; ++j) Q[i * D + j] = mat[i * D + j];
        // 减去之前所有投影
        for (int k = 0; k < i; ++k)
        {
            float proj = 0.0f;
            for (int j = 0; j < D; ++j) proj += Q[i * D + j] * Q[k * D + j];
            for (int j = 0; j < D; ++j) Q[i * D + j] -= proj * Q[k * D + j];
        }
        // 归一化
        float norm = 0.0f;
        for (int j = 0; j < D; ++j) norm += Q[i * D + j] * Q[i * D + j];
        norm = std::sqrt(norm);
        if (norm < 1e-10f) continue;
        for (int j = 0; j < D; ++j) Q[i * D + j] /= norm;
    }
    return Q;
}

APP_ERROR AscendIndexCagraImpl::QuantizeData(int nq, const float *queryData, int ntotal, const float *baseData)
{
    APP_LOG_INFO("AscendIndexCagraImpl::QuantizeData start");
    APPERR_RETURN_IF_NOT_LOG(nq > 0 && nq <= MAX_QUERY_BATCH_SIZE && queryData != nullptr && baseData != nullptr,
                             APP_ERR_INVALID_PARAM, "Invalid quantization input");
    const auto ret = QuantizeBase(ntotal, baseData);
    if (ret != APP_ERR_OK) return ret;
    return QuantizeQueries(nq, queryData);
}

APP_ERROR AscendIndexCagraImpl::QuantizeBase(int ntotal, const float *baseData)
{
    APPERR_RETURN_IF_NOT_LOG(baseData != nullptr && ntotal > 0 && static_cast<size_t>(ntotal) == dataNum,
                             APP_ERR_INVALID_PARAM, "ntotal must match Init dataNum");
    APPERR_RETURN_IF_NOT_LOG(g_graphDevice.data() != nullptr && g_dataDevice.data() != nullptr, APP_ERR_INVALID_PARAM,
                             "Add must precede base quantization");
    int baseBit = 6;  // 量化到6bit
    int seed = 42;
    const int offset = (dim + 7) / 8;
    // 1) 随机正交矩阵
    std::vector<float> P = GenerateOrthogonalMatrix(dim, seed);
    std::vector<float> invP = MatTranspose(P, dim, dim);

    // 2) 质心：沿 ntotal 方向求每维均值
    std::vector<float> centroid(dim, 0.0f);
    for (int j = 0; j < dim; ++j)
    {
        float sum = 0.0f;
        for (int i = 0; i < ntotal; ++i) sum += baseData[static_cast<size_t>(i) * dim + j];
        centroid[j] = sum / static_cast<float>(ntotal);
    }

    // 3) 量化 base
    std::vector<float> baseCentered(static_cast<size_t>(ntotal) * dim);
    for (int i = 0; i < ntotal; ++i)
        for (int j = 0; j < dim; ++j)
            baseCentered[static_cast<size_t>(i) * dim + j] = baseData[static_cast<size_t>(i) * dim + j] - centroid[j];

    std::vector<float> oPrime = MatMul(baseCentered, invP, ntotal, dim, dim);

    std::vector<float> maxAbsBase(ntotal);
    for (int i = 0; i < ntotal; ++i)
    {
        float m = 0.0f;
        for (int j = 0; j < dim; ++j) m = std::fmax(m, std::fabs(oPrime[static_cast<size_t>(i) * dim + j]));
        maxAbsBase[i] = m;
    }

    const float halfRangeBase = static_cast<float>((1 << (baseBit - 1)) - 1);
    const float shiftBase = static_cast<float>(1 << (baseBit - 1));
    const float maxValBase = static_cast<float>((1 << baseBit) - 1);

    std::vector<float> scaleX(ntotal);
    for (int i = 0; i < ntotal; ++i) scaleX[i] = (maxAbsBase[i] == 0.0f) ? 1.0f : maxAbsBase[i] / halfRangeBase;

    std::vector<uint8_t> xQuant(static_cast<size_t>(ntotal) * dim);
    std::vector<float> sumXQuant(ntotal, 0.0f);
    for (int i = 0; i < ntotal; ++i)
    {
        for (int j = 0; j < dim; ++j)
        {
            float v = std::round(oPrime[static_cast<size_t>(i) * dim + j] / scaleX[i] + shiftBase);
            if (v < 0.0f) v = 0.0f;
            if (v > maxValBase) v = maxValBase;
            xQuant[static_cast<size_t>(i) * dim + j] = static_cast<uint8_t>(v);
            sumXQuant[i] += v;
        }
    }

    std::vector<uint8_t> codes(static_cast<size_t>(ntotal) * this->codeSize, 0);
    std::vector<uint8_t> bitBuf(dim);
    int ret = 0;
    for (int i = 0; i < ntotal; ++i)
    {
        for (int p = 0; p < baseBit; ++p)
        {
            for (int j = 0; j < dim; ++j)
                bitBuf[j] = static_cast<uint8_t>((xQuant[static_cast<size_t>(i) * dim + j] >> p) & 1);
            ret =
                PackbitsLittle(bitBuf.data(), dim, codes.data() + static_cast<size_t>(i) * this->codeSize + p * offset);
            APPERR_RETURN_IF_NOT_LOG(ret == APP_ERR_OK, APP_ERR_INNER_ERROR, "PackbitsLittle failed");
        }
        const int pos = offset * baseBit;
        float orCL2sqr = 0.0f;
        for (int j = 0; j < dim; ++j)
        {
            float diff = baseData[static_cast<size_t>(i) * dim + j] - centroid[j];
            orCL2sqr += diff * diff;
        }
        ret = FloatToBytes(orCL2sqr, codes.data() + static_cast<size_t>(i) * this->codeSize + pos);
        APPERR_RETURN_IF_NOT_LOG(ret == APP_ERR_OK, APP_ERR_INNER_ERROR, "orCL2sqr FloatToBytes failed");
        ret = FloatToBytes(shiftBase, codes.data() + static_cast<size_t>(i) * this->codeSize + pos + 4);
        APPERR_RETURN_IF_NOT_LOG(ret == APP_ERR_OK, APP_ERR_INNER_ERROR, "shiftBase FloatToBytes failed");
        ret = FloatToBytes(scaleX[i], codes.data() + static_cast<size_t>(i) * this->codeSize + pos + 8);
        APPERR_RETURN_IF_NOT_LOG(ret == APP_ERR_OK, APP_ERR_INNER_ERROR, "scaleX FloatToBytes failed");
        ret = FloatToBytes(sumXQuant[i], codes.data() + static_cast<size_t>(i) * this->codeSize + pos + 12);
        APPERR_RETURN_IF_NOT_LOG(ret == APP_ERR_OK, APP_ERR_INNER_ERROR, "sumXQuant FloatToBytes failed");
    }

    AscendTensor<uint8_t, DIMS_1, int64_t> codeDev({static_cast<int64_t>(this->dataNum) * this->codeSize});
    ret = aclrtMemcpy(codeDev.data(), codeDev.getSizeInBytes(), codes.data(),
                      this->dataNum * this->codeSize * sizeof(uint8_t), ACL_MEMCPY_HOST_TO_DEVICE);
    APPERR_RETURN_IF_NOT_FMT((ret == ACL_SUCCESS), APP_ERR_INNER_ERROR, "copy codeDev to device fail(%d)!", ret);

    g_code = std::move(codeDev);
    g_centroid = std::move(centroid);
    g_inverseRotation = std::move(invP);
    g_precompute.clear();
    g_rotated.clear();

    trained = true;
    return APP_ERR_OK;
}

APP_ERROR AscendIndexCagraImpl::QuantizeQueries(int nq, const float *queryData)
{
    APPERR_RETURN_IF_NOT_FMT(nq > 0 && nq <= MAX_QUERY_BATCH_SIZE && queryData != nullptr, APP_ERR_INVALID_PARAM,
                             "QuantizeQueries requires a query batch in [1, %d]", MAX_QUERY_BATCH_SIZE);
    APPERR_RETURN_IF_NOT_LOG(
        g_centroid.size() == static_cast<size_t>(dim) && g_inverseRotation.size() == static_cast<size_t>(dim) * dim,
        APP_ERR_INVALID_PARAM, "QuantizeData must be called before QuantizeQueries");
    APPERR_RETURN_IF_NOT_LOG(
        std::all_of(queryData, queryData + static_cast<size_t>(nq) * dim, [](float v) { return std::isfinite(v); }),
        APP_ERR_INVALID_PARAM, "CAGRA queries must contain finite FP32 values");
    // Prepare only this request's query codes; base quantization remains a one-time cost.
    const int queryBit = 6;
    const int offset = (dim + 7) / 8;
    std::vector<uint8_t> bitBuf(dim);
    std::vector<float> queryCentered(static_cast<size_t>(nq) * dim);
    for (int i = 0; i < nq; ++i)
        for (int j = 0; j < dim; ++j)
            queryCentered[static_cast<size_t>(i) * dim + j] =
                queryData[static_cast<size_t>(i) * dim + j] - g_centroid[j];

    std::vector<float> qPrime = MatMul(queryCentered, g_inverseRotation, nq, dim, dim);

    std::vector<float> maxAbsQ(nq);
    for (int i = 0; i < nq; ++i)
    {
        float m = 0.0f;
        for (int j = 0; j < dim; ++j) m = std::fmax(m, std::fabs(qPrime[static_cast<size_t>(i) * dim + j]));
        maxAbsQ[i] = m;
    }

    const float halfRangeQ = static_cast<float>((1 << (queryBit - 1)) - 1);
    const float shiftQuery = static_cast<float>(1 << (queryBit - 1));
    const float maxValQ = static_cast<float>((1 << queryBit) - 1);

    std::vector<float> scaleQ(nq);
    for (int i = 0; i < nq; ++i) scaleQ[i] = (maxAbsQ[i] == 0.0f) ? 1.0f : maxAbsQ[i] / halfRangeQ;

    std::vector<uint8_t> qQuant(static_cast<size_t>(nq) * dim);
    std::vector<float> sumQQuant(nq, 0.0f);
    for (int i = 0; i < nq; ++i)
    {
        for (int j = 0; j < dim; ++j)
        {
            float v = std::round(qPrime[static_cast<size_t>(i) * dim + j] / scaleQ[i] + shiftQuery);
            if (v < 0.0f) v = 0.0f;
            if (v > maxValQ) v = maxValQ;
            qQuant[static_cast<size_t>(i) * dim + j] = static_cast<uint8_t>(v);
            sumQQuant[i] += v;
        }
    }

    std::vector<uint8_t> rotatedQq(static_cast<size_t>(nq) * this->rotatedSize, 0);
    std::vector<float> precomputeAll(nq * PRE_COMPUTE_SIZE, 0.0f);
    for (int n = 0; n < nq; ++n)
    {
        for (int p = 0; p < queryBit; ++p)
        {
            for (int j = 0; j < dim; ++j)
                bitBuf[j] = static_cast<uint8_t>((qQuant[static_cast<size_t>(n) * dim + j] >> p) & 1);
            const auto ret = PackbitsLittle(bitBuf.data(), dim, rotatedQq.data() + n * this->rotatedSize + p * offset);
            APPERR_RETURN_IF_NOT_LOG(ret == APP_ERR_OK, APP_ERR_INNER_ERROR, "PackbitsLittle failed");
        }
        float qrToCL2sqr = 0.0f;
        for (int j = 0; j < dim; ++j)
        {
            float diff = queryData[static_cast<size_t>(n) * dim + j] - g_centroid[j];
            qrToCL2sqr += diff * diff;
        }
        precomputeAll[n * PRE_COMPUTE_SIZE + 0] = qrToCL2sqr;
        precomputeAll[n * PRE_COMPUTE_SIZE + 1] = shiftQuery;
        precomputeAll[n * PRE_COMPUTE_SIZE + 2] = scaleQ[n];
        precomputeAll[n * PRE_COMPUTE_SIZE + 3] = sumQQuant[n];
    }

    g_precompute = std::move(precomputeAll);
    g_rotated = std::move(rotatedQq);
    return APP_ERR_OK;
}

APP_ERROR AscendIndexCagraImpl::Search(int n, const float *queryData, int topK, float *dists, uint32_t *labels)
{
    APP_LOG_INFO("AscendIndexCagraImpl::Search start");
    APPERR_RETURN_IF_NOT_LOG(!unifiedMode || trained, APP_ERR_INDEX_NOT_INIT, "CAGRA is not trained");
    APPERR_RETURN_IF_NOT_LOG((topK == 32 || topK == 64 || topK == 128 || topK == 256 || topK == 512 || topK == 1024) &&
                                 static_cast<size_t>(topK) <= dataNum,
                             APP_ERR_INVALID_PARAM, "CAGRA topK must be in {32,64,128,256,512,1024} and at most N");
    const auto deviceRet = aclrtSetDevice(deviceId);
    APPERR_RETURN_IF_NOT_LOG(deviceRet == ACL_SUCCESS, APP_ERR_ACL_SET_DEVICE_FAILED, "Failed to select CAGRA device");
    if (unifiedMode)
    {
        APPERR_RETURN_IF_NOT_LOG(trained, APP_ERR_INDEX_NOT_INIT, "CAGRA is not trained");
        auto ret = QuantizeQueries(n, queryData);
        if (ret != APP_ERR_OK) return ret;
        ret = ResetCagraSearchOp(topK);
        if (ret != APP_ERR_OK) return ret;
    }
    APPERR_RETURN_IF_NOT_LOG(n > 0 && static_cast<size_t>(n) * PRE_COMPUTE_SIZE <= g_precompute.size() &&
                                 static_cast<size_t>(n) * rotatedSize <= g_rotated.size(),
                             APP_ERR_INVALID_PARAM, "Search queries must be quantized before use");
    APPERR_RETURN_IF_NOT_FMT(unifiedMode || topK == this->topK, APP_ERR_INVALID_PARAM,
                             "Search topK %d must match Init topK %d", topK, this->topK);

    size_t size = this->searchBatchSizes.size();
    int searched = 0;
    for (size_t i = 0; i < size; i++)
    {
        int batchSize = this->searchBatchSizes[i];
        if ((n - searched) >= batchSize)
        {
            int page = (n - searched) / batchSize;
            for (int j = 0; j < page; j++)
            {
                auto ret = SearchImpl(batchSize, queryData + searched * this->dim, topK,
                                      g_precompute.data() + searched * PRE_COMPUTE_SIZE,
                                      g_rotated.data() + searched * this->rotatedSize, labels + searched * topK,
                                      dists + searched * topK);
                APPERR_RETURN_IF_NOT_FMT(ret == APP_ERR_OK, APP_ERR_INNER_ERROR,
                                         "AscendIndexCagraImpl SearchImpl failed(%d)", ret);
                searched += batchSize;
            }
        }
    }

    APP_LOG_INFO("AscendIndexCagraImpl::Search finished");
    return APP_ERR_OK;
}

APP_ERROR AscendIndexCagraImpl::SearchImpl(int n, const float *queryData, int topK, float *precompute, uint8_t *rotated,
                                           uint32_t *labels, float *dists)
{
    APP_LOG_INFO("AscendIndexCagraImpl::SearchImpl operation start.\n");

    AscendTensor<float, DIMS_2> queries({n, this->dim});
    auto ret = aclrtMemcpy(queries.data(), queries.getSizeInBytes(), queryData, n * this->dim * sizeof(float),
                           ACL_MEMCPY_HOST_TO_DEVICE);
    APPERR_RETURN_IF_NOT_FMT((ret == ACL_SUCCESS), APP_ERR_INNER_ERROR, "copy queries to device fail(%d)!", ret);

    AscendTensor<float, DIMS_1> preCodeDev({n * PRE_COMPUTE_SIZE});
    ret = aclrtMemcpy(preCodeDev.data(), preCodeDev.getSizeInBytes(), precompute, n * PRE_COMPUTE_SIZE * sizeof(float),
                      ACL_MEMCPY_HOST_TO_DEVICE);
    APPERR_RETURN_IF_NOT_FMT((ret == ACL_SUCCESS), APP_ERR_INNER_ERROR, "copy preCodeDev to device fail(%d)!", ret);

    AscendTensor<uint8_t, DIMS_1> rotatedDev({n * this->rotatedSize});
    ret = aclrtMemcpy(rotatedDev.data(), rotatedDev.getSizeInBytes(), rotated, n * this->rotatedSize * sizeof(uint8_t),
                      ACL_MEMCPY_HOST_TO_DEVICE);
    APPERR_RETURN_IF_NOT_FMT((ret == ACL_SUCCESS), APP_ERR_INNER_ERROR, "copy rotatedDev to device fail(%d)!", ret);

    AscendTensor<uint32_t, DIMS_2> outIndices({n, topK});
    AscendTensor<float, DIMS_2> outDistances({n, topK});

    ret = SearchPaged(queries, preCodeDev, rotatedDev, outIndices, outDistances);
    APPERR_RETURN_IF_NOT_FMT(ret == APP_ERR_OK, APP_ERR_INNER_ERROR, "AscendIndexCagraImpl SearchPaged failed(%d)",
                             ret);

    ret = aclrtMemcpy(dists, n * topK * sizeof(float), outDistances.data(), outDistances.getSizeInBytes(),
                      ACL_MEMCPY_DEVICE_TO_HOST);
    APPERR_RETURN_IF_NOT_FMT((ret == ACL_SUCCESS), APP_ERR_INNER_ERROR, "copy outDistances back to host(%d)", ret);

    ret = aclrtMemcpy(labels, n * topK * sizeof(uint32_t), outIndices.data(), outIndices.getSizeInBytes(),
                      ACL_MEMCPY_DEVICE_TO_HOST);
    APPERR_RETURN_IF_NOT_FMT((ret == ACL_SUCCESS), APP_ERR_INNER_ERROR, "copy outIndices back to host(%d)", ret);
    APP_LOG_INFO("AscendIndexCagraImpl::SearchImpl operation end.\n");
    return APP_ERR_OK;
}

APP_ERROR AscendIndexCagraImpl::SearchPaged(AscendTensor<float, DIMS_2> &queries,
                                            AscendTensor<float, DIMS_1> &preCompute,
                                            AscendTensor<uint8_t, DIMS_1> &rotated,
                                            AscendTensor<uint32_t, DIMS_2> &outIndices,
                                            AscendTensor<float, DIMS_2> &outDistances)
{
    APP_LOG_INFO("AscendIndexCagraImpl::SearchPaged operation start.\n");
    auto streamPtr = this->pResources->getDefaultStream();
    auto stream = streamPtr->GetStream();
    ComputeBlockDist(queries, g_graphDevice, g_hashDevice, g_dataDevice, preCompute, g_code, rotated, outIndices,
                     outDistances, stream);

    auto ret = synchronizeStream(stream);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "synchronizeStream failed: %i\n", ret);

    APP_LOG_INFO("AscendIndexCagraImpl::SearchPaged operation end.\n");
    return APP_ERR_OK;
}

void AscendIndexCagraImpl::ComputeBlockDist(AscendTensor<float, DIMS_2> &queryTensor,
                                            AscendTensor<uint32_t, DIMS_2> &graphDevice,
                                            AscendTensor<uint32_t, DIMS_2> &hashDevice,
                                            AscendTensor<float, DIMS_2> &data, AscendTensor<float, DIMS_1> &preCompute,
                                            AscendTensor<uint8_t, DIMS_1, int64_t> &preCode,
                                            AscendTensor<uint8_t, DIMS_1> &rotated,
                                            AscendTensor<uint32_t, DIMS_2> &outIndices,
                                            AscendTensor<float, DIMS_2> &outDistances, aclrtStream stream)
{
    AscendOperator *op = nullptr;
    int batchSize = queryTensor.getSize(0);
    const auto group = cagraSearchOp.find(outIndices.getSize(1));
    if (group != cagraSearchOp.end())
    {
        const auto found = group->second.find(batchSize);
        if (found != group->second.end()) op = found->second.get();
    }
    ASCEND_THROW_IF_NOT(op);

    std::shared_ptr<std::vector<const aclDataBuffer *>> distOpInput(new std::vector<const aclDataBuffer *>(),
                                                                    CommonUtils::AclInputBufferDelete);
    distOpInput->emplace_back(aclCreateDataBuffer(queryTensor.data(), queryTensor.getSizeInBytes()));
    distOpInput->emplace_back(aclCreateDataBuffer(graphDevice.data(), graphDevice.getSizeInBytes()));
    distOpInput->emplace_back(aclCreateDataBuffer(hashDevice.data(), hashDevice.getSizeInBytes()));
    distOpInput->emplace_back(aclCreateDataBuffer(data.data(), data.getSizeInBytes()));
    distOpInput->emplace_back(aclCreateDataBuffer(preCompute.data(), preCompute.getSizeInBytes()));
    distOpInput->emplace_back(aclCreateDataBuffer(preCode.data(), preCode.getSizeInBytes()));
    distOpInput->emplace_back(aclCreateDataBuffer(rotated.data(), rotated.getSizeInBytes()));

    std::shared_ptr<std::vector<aclDataBuffer *>> distOpOutput(new std::vector<aclDataBuffer *>(),
                                                               CommonUtils::AclOutputBufferDelete);
    distOpOutput->emplace_back(aclCreateDataBuffer(outDistances.data(), outDistances.getSizeInBytes()));
    distOpOutput->emplace_back(aclCreateDataBuffer(outIndices.data(), outIndices.getSizeInBytes()));

    op->exec(*distOpInput, *distOpOutput, stream);
}

APP_ERROR AscendIndexCagraImpl::ResetCagraSearchOp(int searchTopK)
{
    if (cagraSearchOp.count(searchTopK) != 0) return APP_ERR_OK;
    APP_LOG_INFO("AscendIndexCagraImpl::ResetCagraSearchOp operation started.\n");
    auto cagraSearchOpReset = [&](std::unique_ptr<AscendOperator> &op, int64_t batch) -> APP_ERROR
    {
        const int64_t rowCount = static_cast<int64_t>(this->dataNum);
        AscendOpDesc desc("CagraRabitq");
        std::vector<int64_t> queryShape({batch, this->dim});
        std::vector<int64_t> knnShape({rowCount, this->degree});
        std::vector<int64_t> hashShape({MAX_SEARCH_BATCH, static_cast<int64_t>((this->dataNum + 31) / 32)});
        std::vector<int64_t> ptrShape({rowCount, this->dim});
        std::vector<int64_t> preCompute({batch * PRE_COMPUTE_SIZE});
        std::vector<int64_t> preCode({rowCount * this->codeSize});
        std::vector<int64_t> rotated({batch * this->rotatedSize});

        std::vector<int64_t> DistShape({batch, searchTopK});
        std::vector<int64_t> IndiceShape({batch, searchTopK});

        desc.addInputTensorDesc(ACL_FLOAT, queryShape.size(), queryShape.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_UINT32, knnShape.size(), knnShape.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_UINT32, hashShape.size(), hashShape.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_FLOAT, ptrShape.size(), ptrShape.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_FLOAT, preCompute.size(), preCompute.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_UINT8, preCode.size(), preCode.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_UINT8, rotated.size(), rotated.data(), ACL_FORMAT_ND);

        desc.addOutputTensorDesc(ACL_FLOAT, DistShape.size(), DistShape.data(), ACL_FORMAT_ND);
        desc.addOutputTensorDesc(ACL_UINT32, IndiceShape.size(), IndiceShape.data(), ACL_FORMAT_ND);
        const auto attrRet = aclopSetAttrInt(desc.opAttr, "top_k", searchTopK);
        APPERR_RETURN_IF_NOT_FMT(attrRet == ACL_SUCCESS, APP_ERR_INVALID_PARAM, "Failed to set CagraRabitq top_k: %d",
                                 attrRet);

        op.reset();
        op = CREATE_UNIQUE_PTR(AscendOperator, desc);
        return op->init() ? APP_ERR_OK : APP_ERR_ACL_OP_LOAD_MODEL_FAILED;
    };

    SearchOpMap prepared;
    for (auto batch : this->searchBatchSizes)
    {
        const auto ret = cagraSearchOpReset(prepared[batch], batch);
        APPERR_RETURN_IF_NOT_LOG(ret == APP_ERR_OK, ret, "CAGRA search model initialization failed");
    }
    cagraSearchOp.emplace(searchTopK, std::move(prepared));

    APP_LOG_INFO("AscendIndexCagraImpl::ResetCagraSearchOp operation end.\n");
    return APP_ERR_OK;
}

}  // namespace ascend
}  // namespace faiss
