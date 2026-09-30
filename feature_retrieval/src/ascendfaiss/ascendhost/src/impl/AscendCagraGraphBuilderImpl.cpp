/*
 * -------------------------------------------------------------------------
 * This file is part of the IndexSDK project.
 * Copyright (c) 2026 Huawei Technologies Co.,Ltd.
 *
 * IndexSDK is licensed under Mulan PSL v2.
 * -------------------------------------------------------------------------
 */

#include "impl/AscendCagraGraphBuilderImpl.h"

#include <acl/acl.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <initializer_list>
#include <limits>
#include <memory>
#include <new>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "ascenddaemon/utils/AscendOpDesc.h"
#include "ascenddaemon/utils/AscendTensor.h"
#include "ascenddaemon/utils/AscendUtils.h"
#include "common/utils/CommonUtils.h"
#include "common/utils/LogUtils.h"
#include "common/utils/SocUtils.h"

namespace faiss
{
namespace ascend
{
namespace
{
using InputBuffers = std::shared_ptr<std::vector<const aclDataBuffer *>>;
using OutputBuffers = std::shared_ptr<std::vector<aclDataBuffer *>>;
using GraphBuildClock = std::chrono::steady_clock;
constexpr uint32_t CAGRA_NODE_MASK = 0x7fffffffU;
constexpr int MAX_CAGRA_BUILD_DIM = 3072;
constexpr int CAGRA_SIMT_CACHED_DIM = 128;
constexpr int CAGRA_SIMT_BATCHED_MAX_DIM = 512;

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
}  // namespace

AscendCagraGraphBuilderImpl::AscendCagraGraphBuilderImpl(int dim, const Config &config) : dim(dim), config(config) {}

AscendCagraGraphBuilderImpl::~AscendCagraGraphBuilderImpl() { Reset(); }

APP_ERROR AscendCagraGraphBuilderImpl::Initialize()
{
    if (initialized)
    {
        return APP_ERR_OK;
    }
    APPERR_RETURN_IF_NOT(dim > 0 && dim <= MAX_CAGRA_BUILD_DIM, APP_ERR_INVALID_PARAM);
    APPERR_RETURN_IF_NOT(config.dataSize > 1 && config.dataSize <= std::numeric_limits<int32_t>::max(),
                         APP_ERR_INVALID_PARAM);
    APPERR_RETURN_IF_NOT(config.graphDegree > 0 && config.graphDegree < static_cast<uint64_t>(config.dataSize),
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

APP_ERROR AscendCagraGraphBuilderImpl::SetDevice()
{
    aclError ret = aclrtSetDevice(config.deviceId);
    ASCEND_THROW_IF_NOT_MSG(ret == ACL_ERROR_NONE, "Set device failed");
    return APP_ERR_OK;
}

APP_ERROR AscendCagraGraphBuilderImpl::BuildGraph(const float *data, uint32_t *graph, const BuildConfig &newBuildConfig)
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

    graphSize = config.dataSize;
    ret = CallGraphBuildKernel(data, graph, buildConfig);
    ASCEND_THROW_IF_NOT_MSG(ret == APP_ERR_OK, "CallGraphBuildKernel failed");
    ret = ValidateGraph(graph, config.graphDegree);
    ASCEND_THROW_IF_NOT_MSG(ret == APP_ERR_OK, "CAGRA graph validation failed");
    return APP_ERR_OK;
}

APP_ERROR AscendCagraGraphBuilderImpl::BuildGraphToFile(const float *data, const std::string &graphFilePath,
                                                        const BuildConfig &newBuildConfig)
{
    APPERR_RETURN_IF_NOT(initialized, APP_ERR_INDEX_NOT_INIT);
    APPERR_RETURN_IF_NOT(data != nullptr, APP_ERR_INVALID_PARAM);
    APPERR_RETURN_IF_NOT(!graphFilePath.empty(), APP_ERR_INVALID_PARAM);
    const size_t dataSize = static_cast<size_t>(config.dataSize);
    const size_t maxGraphElements = std::numeric_limits<size_t>::max() / sizeof(uint32_t);
    APPERR_RETURN_IF_NOT(config.graphDegree > 0 && dataSize <= maxGraphElements / config.graphDegree,
                         APP_ERR_ACL_BAD_ALLOC);
    const size_t graphElements = dataSize * config.graphDegree;
    std::unique_ptr<uint32_t[]> graph(new (std::nothrow) uint32_t[graphElements]);
    APPERR_RETURN_IF_NOT(graph != nullptr, APP_ERR_ACL_BAD_ALLOC);
    APP_ERROR ret = BuildGraph(data, graph.get(), newBuildConfig);
    APPERR_RETURN_IF_NOT(ret == APP_ERR_OK, ret);
    return SaveGraphToFile(graphFilePath, graph.get(), graphElements);
}

APP_ERROR
AscendCagraGraphBuilderImpl::ResetGraphBuildOp(const BuildConfig &cfg)
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

APP_ERROR AscendCagraGraphBuilderImpl::CallGraphBuildKernel(const float *data, uint32_t *graph, const BuildConfig &cfg)
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

void AscendCagraGraphBuilderImpl::Reset()
{
    nndInitOp.reset();
    nndSampleReverseOp.reset();
    nndLocalJoinOp.reset();
    nndUpdateOp.reset();
    cagraPruneReverseOp.reset();
    cagraMergeOp.reset();
    resources.reset();
    graphSize = 0;
    initialized = false;
}

APP_ERROR AscendCagraGraphBuilderImpl::SaveGraphToFile(const std::string &filePath, const uint32_t *graph,
                                                       size_t numElements)
{
    FILE *file = fopen(filePath.c_str(), "wb");
    APPERR_RETURN_IF_NOT_LOG(file != nullptr, APP_ERR_INNER_ERROR, "Failed to open graph output file");
    const size_t written = fwrite(graph, sizeof(uint32_t), numElements, file);
    const int closeRet = fclose(file);
    APPERR_RETURN_IF_NOT_LOG(written == numElements && closeRet == 0, APP_ERR_INNER_ERROR,
                             "Failed to write complete CAGRA graph");
    return APP_ERR_OK;
}

APP_ERROR AscendCagraGraphBuilderImpl::ValidateGraph(const uint32_t *graph, uint32_t degree) const
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

}  // namespace ascend
}  // namespace faiss
