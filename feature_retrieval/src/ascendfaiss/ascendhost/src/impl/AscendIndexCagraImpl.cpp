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

#include "impl/AscendIndexCagraImpl.h"

#include <algorithm>
#include <fstream>
#include <cstdio>
#include <string>
#include <faiss/impl/FaissAssert.h>

#include "ascend/utils/fp16.h"
#include "ascenddaemon/utils/Limits.h"
#include "ascenddaemon/utils/Random.h"
#include "ascenddaemon/utils/AscendOpDesc.h"
#include "ascenddaemon/utils/AscendUtils.h"
#include "common/threadpool/AscendThreadPool.h"
#include "common/utils/CommonUtils.h"
#include "common/utils/LogUtils.h"
#include "common/utils/SocUtils.h"
#include "common/utils/DataType.h"
#include "common/ErrorCode.h"
#include "index_custom/IndexFlatATAicpu.h"
#include "ops/cpukernel/impl/utils/kernel_shared_def.h"
#include <acl/acl.h>
using namespace ascend;
namespace faiss {
namespace ascend {
namespace {
const int64_t CAGRA_MAX_MEM = 0x100000000; // 0x100000000 mean 4096MB
const int MAX_GRAPH_DEGREE = 256;
const uint32_t RANDOM_SEED = 1234;
const int LOG_INTERVAL = 10;
std::vector<int> ASCEND_CAGRA_SEARCH_BATCHES = {32, 16, 8, 4, 2, 1};
}

AscendIndexCagraImpl::AscendIndexCagraImpl(const IndexCagraInitParams& params)
    : dim(params.dim), degree(params.graph_degree), deviceList(params.deviceList),
    ascendResourceSize(params.ascendResourceSize)
{
}

AscendIndexCagraImpl::~AscendIndexCagraImpl()
{
    int deviceId = deviceList[0];
    (void)aclrtResetDevice(deviceId);
}

APP_ERROR AscendIndexCagraImpl::Init(const IndexCagraInitParams& params, const IndexCagraSearchParams& searchParams)
{
    APP_LOG_INFO("AscendIndexCagraImpl::Init start");
    
    // Set device
    deviceId = params.deviceList[0];
    ASCEND_THROW_IF_NOT(deviceId >= 0);
    this->searchBatchSizes = ASCEND_CAGRA_SEARCH_BATCHES;
    auto ret = aclrtSetDevice(deviceId);
    APP_LOG_INFO("AscendIndexCagraImpl Init on device(%d).\n", deviceId);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_ACL_SET_DEVICE_FAILED, "failed to set device(%d)", ret);
    pResources = CREATE_UNIQUE_PTR(AscendResourcesProxy);
    pResources->setTempMemory(ascendResourceSize);
    pResources->initialize();

    topK = searchParams.topk;
    hashBitlen = searchParams.hashBitlen;
    dataNum = searchParams.dataNum;

    // Reset operators
    ret = ResetCagraSearchOp();
    APPERR_RETURN_IF_NOT_LOG(ret == APP_ERR_OK, APP_ERR_ACL_OP_LOAD_MODEL_FAILED,
        "ResetCagraSearchOp init failed !!!");
    
    isInitialized = true;
    APP_LOG_INFO("AscendIndexCagraImpl::Init finished");
    return APP_ERR_OK;
}


APP_ERROR AscendIndexCagraImpl::AddGraph(const std::vector<uint32_t>& graphData, const std::string& saveBinPath)
{
    APP_LOG_INFO("AscendIndexCagraImpl::AddGraph start");
    
    FAISS_THROW_IF_NOT_MSG(isInitialized, "AscendIndexCagraImpl is not initialized");
    FAISS_THROW_IF_NOT_MSG(!graphData.empty(), "Graph data can not be empty");
    
    // Save graph data
    this->graphData = graphData;
    this->graphBinPath = saveBinPath;
    
    APP_LOG_INFO("AscendIndexCagraImpl::AddGraph finished");
    return APP_ERR_OK;
}

APP_ERROR AscendIndexCagraImpl::BuildGraph(int64_t n, const float* data, const std::string& graphFilePath, const BuildConfig& buildConfig)
{
    APP_LOG_INFO("AscendIndexCagraImpl::BuildGraph start");
    
    FAISS_THROW_IF_NOT_MSG(isInitialized, "AscendIndexCagraImpl is not initialized");
    FAISS_THROW_IF_NOT_MSG(n > 0, "Data size must be positive");
    FAISS_THROW_IF_NOT_MSG(data != nullptr, "Data cannot be null");
    FAISS_THROW_IF_NOT_MSG(buildConfig.graphDegree > 0, "Graph degree must be positive");
    
    this->buildConfig = buildConfig;
    this->buildConfig.dataSize = n;
    
    auto ret = resetAllGraphOps();
    ASCEND_THROW_IF_NOT_MSG(ret == APP_ERR_OK, "Failed to reset graph operators!");
    
    size_t graphElements = static_cast<size_t>(n) * static_cast<size_t>(buildConfig.graphDegree);
    auto graph = std::make_unique<uint32_t[]>(graphElements);
    
    ret = buildGraphImpl(n, data, graph.get());
    ASCEND_THROW_IF_NOT_MSG(ret == APP_ERR_OK, "buildGraphImpl failed!");
    
    if (!graphFilePath.empty()) {
        ret = saveGraphToFile(graphFilePath, graph.get(), graphElements);
        ASCEND_THROW_IF_NOT_MSG(ret == APP_ERR_OK, "saveGraphToFile failed!");
        APP_LOG_INFO("Graph saved to file: %s\n", graphFilePath.c_str());
    }
    
    std::vector<uint32_t> graphVector(graph.get(), graph.get() + graphElements);
    this->graphData = graphVector;
    
    APP_LOG_INFO("AscendIndexCagraImpl::BuildGraph finished, graph size: %zu elements", graphElements);
    return APP_ERR_OK;
}

APP_ERROR AscendIndexCagraImpl::Search(int n, const float* queryData, int topK, const uint32_t* graph,
    const uint32_t* hash, const float* data, float* dists, uint32_t* labels)
{
    APP_LOG_INFO("AscendIndexCagraImpl::Search start");

    size_t size = this->searchBatchSizes.size();
    int searched = 0;
    for (size_t i = 0; i < size; i++) {
        int batchSize = this->searchBatchSizes[i];
        if ((n - searched) >= batchSize) {
            int page = (n - searched) / batchSize;
            for (int j = 0; j < page; j++) {
                auto ret = SearchImpl(batchSize, queryData + searched * this->dim, topK,
                    graph, hash, data, labels + searched * topK, dists + searched * topK);
                APPERR_RETURN_IF_NOT_FMT(ret == APP_ERR_OK, APP_ERR_INNER_ERROR,
                    "AscendIndexCagraImpl SearchImpl failed(%d)", ret);
                searched += batchSize;
            }
        }
    }

    APP_LOG_INFO("AscendIndexCagraImpl::Search finished");
    return APP_ERR_OK;
}

APP_ERROR AscendIndexCagraImpl::SearchImpl(int n, const float* queryData, int topK, const uint32_t* graph, const uint32_t* hash,
    const float* data, uint32_t *labels, float *dists)
{
    APP_LOG_INFO("AscendIndexCagraImpl SearchImpl operation start.\n");
    auto streamPtr = this->pResources->getDefaultStream();
    auto stream = streamPtr->GetStream();
    auto &mem = this->pResources->getMemoryManager();

    AscendTensor<float, DIMS_2> queries(mem, {n, this->dim}, stream);
    auto ret = aclrtMemcpy(queries.data(), queries.getSizeInBytes(), queryData, n * this->dim * sizeof(float),
        ACL_MEMCPY_HOST_TO_DEVICE);
    APPERR_RETURN_IF_NOT_FMT((ret == ACL_SUCCESS), APP_ERR_INNER_ERROR, "copy queries to device fail(%d)!", ret);

    AscendTensor<uint32_t, DIMS_2> graphDevice(mem, {this->dataNum, this->degree}, stream);
    ret = aclrtMemcpy(graphDevice.data(), graphDevice.getSizeInBytes(), graph,
        this->dataNum * this->degree * sizeof(uint32_t), ACL_MEMCPY_HOST_TO_DEVICE);
    APPERR_RETURN_IF_NOT_FMT((ret == ACL_SUCCESS), APP_ERR_INNER_ERROR, "copy graphDevice to device fail(%d)!", ret);

    AscendTensor<uint32_t, DIMS_2> hashDevice(mem, {n, this->hashBitlen}, stream);
    ret = aclrtMemcpy(hashDevice.data(), hashDevice.getSizeInBytes(), hash,
        n * this->hashBitlen * sizeof(uint32_t), ACL_MEMCPY_HOST_TO_DEVICE);
    APPERR_RETURN_IF_NOT_FMT((ret == ACL_SUCCESS), APP_ERR_INNER_ERROR, "copy hashDevice to device fail(%d)!", ret);

    AscendTensor<float, DIMS_2> dataDevice(mem, {this->dataNum, this->dim}, stream);
    ret = aclrtMemcpy(dataDevice.data(), dataDevice.getSizeInBytes(), data,
        this->dataNum * this->dim * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE);
    APPERR_RETURN_IF_NOT_FMT((ret == ACL_SUCCESS), APP_ERR_INNER_ERROR, "copy dataDevice to device fail(%d)!", ret);

    AscendTensor<uint32_t, DIMS_2> outIndices(mem, { n, topK }, stream);
    AscendTensor<float, DIMS_2> outDistances(mem, { n, topK }, stream);

    ret = SearchPaged(queries, graphDevice, hashDevice, dataDevice, outIndices, outDistances);
    APPERR_RETURN_IF_NOT_FMT(ret == APP_ERR_OK, APP_ERR_INNER_ERROR,
        "AscendIndexCagraImpl SearchPaged failed(%d)", ret);

    ret = aclrtMemcpy(dists, n * topK * sizeof(float), outDistances.data(),
        outDistances.getSizeInBytes(), ACL_MEMCPY_DEVICE_TO_HOST);
    APPERR_RETURN_IF_NOT_FMT((ret == ACL_SUCCESS), APP_ERR_INNER_ERROR, "copy outDistances back to host(%d)", ret);

    ret = aclrtMemcpy(labels, n * topK * sizeof(uint32_t), outIndices.data(), outIndices.getSizeInBytes(),
        ACL_MEMCPY_DEVICE_TO_HOST);
    APPERR_RETURN_IF_NOT_FMT((ret == ACL_SUCCESS), APP_ERR_INNER_ERROR, "copy outIndices back to host(%d)", ret);
    APP_LOG_INFO("AscendIndexCagraImpl SearchImpl operation end.\n");
    return APP_ERR_OK;
}

APP_ERROR AscendIndexCagraImpl::SearchPaged(AscendTensor<float, DIMS_2> &queries,
    AscendTensor<uint32_t, DIMS_2>& graphDevice,
    AscendTensor<uint32_t, DIMS_2>& hashDevice, AscendTensor<float, DIMS_2> &data,
    AscendTensor<uint32_t, DIMS_2> &outIndices, AscendTensor<float, DIMS_2> &outDistances)
{
    APP_LOG_INFO("AscendIndexCagraImpl SearchPaged operation start.\n");
    auto streamPtr = this->pResources->getDefaultStream();
    auto stream = streamPtr->GetStream();
    ComputeBlockDist(queries, graphDevice, hashDevice, data, outIndices, outDistances, stream);

    auto ret = synchronizeStream(stream);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "synchronizeStream failed: %i\n", ret);

    APP_LOG_INFO("AscendIndexCagraImpl SearchPaged operation end.\n");
    return APP_ERR_OK;
}

void AscendIndexCagraImpl::ComputeBlockDist(AscendTensor<float, DIMS_2> &queryTensor,
    AscendTensor<uint32_t, DIMS_2>& graphDevice, AscendTensor<uint32_t, DIMS_2>& hashDevice,
    AscendTensor<float, DIMS_2> &data, AscendTensor<uint32_t, DIMS_2> &outIndices,
    AscendTensor<float, DIMS_2> &outDistances, aclrtStream stream)
{
    AscendOperator *op = nullptr;
    int batchSize = queryTensor.getSize(0);
    if (this->cagraSearchOp.find(batchSize) != this->cagraSearchOp.end()) {
        op = this->cagraSearchOp[batchSize].get();
    }
    ASCEND_THROW_IF_NOT(op);

    std::shared_ptr<std::vector<const aclDataBuffer *>> distOpInput(
        new std::vector<const aclDataBuffer *>(), CommonUtils::AclInputBufferDelete);
    distOpInput->emplace_back(aclCreateDataBuffer(queryTensor.data(), queryTensor.getSizeInBytes()));
    distOpInput->emplace_back(aclCreateDataBuffer(graphDevice.data(), graphDevice.getSizeInBytes()));
    distOpInput->emplace_back(aclCreateDataBuffer(hashDevice.data(), hashDevice.getSizeInBytes()));
    distOpInput->emplace_back(aclCreateDataBuffer(data.data(), data.getSizeInBytes()));

    std::shared_ptr<std::vector<aclDataBuffer *>> distOpOutput(
        new std::vector<aclDataBuffer *>(), CommonUtils::AclOutputBufferDelete);
    distOpOutput->emplace_back(aclCreateDataBuffer(outDistances.data(), outDistances.getSizeInBytes()));
    distOpOutput->emplace_back(aclCreateDataBuffer(outIndices.data(), outIndices.getSizeInBytes()));

    op->exec(*distOpInput, *distOpOutput, stream);
}

APP_ERROR AscendIndexCagraImpl::ResetCagraSearchOp()
{
    APP_LOG_INFO("AscendIndexCagraImpl ResetCagraSearchOp operation started.\n");
    auto cagraSearchOpReset = [&](std::unique_ptr<AscendOperator> &op, int64_t batch) {
        AscendOpDesc desc("Cagra");
        std::vector<int64_t> queryShape({ batch, this->dim });
        std::vector<int64_t> knnShape({ this->dataNum, this->degree });
        std::vector<int64_t> hashShape({ batch, this->hashBitlen });
        std::vector<int64_t> ptrShape({ this->dim, this->dataNum });
        
        std::vector<int64_t> DistShape({ batch, this->topK });
        std::vector<int64_t> IndiceShape({ batch, this->topK });

        desc.addInputTensorDesc(ACL_FLOAT, queryShape.size(), queryShape.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_UINT32, knnShape.size(), knnShape.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_UINT32, hashShape.size(), hashShape.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_FLOAT, ptrShape.size(), ptrShape.data(), ACL_FORMAT_ND);

        desc.addOutputTensorDesc(ACL_FLOAT, DistShape.size(), DistShape.data(), ACL_FORMAT_ND);
        desc.addOutputTensorDesc(ACL_UINT32, IndiceShape.size(), IndiceShape.data(), ACL_FORMAT_ND);

        op.reset();
        op = CREATE_UNIQUE_PTR(AscendOperator, desc);
        return op->init();
    };

    for (auto batch : this->searchBatchSizes) {
        cagraSearchOp[batch] = std::unique_ptr<AscendOperator>(nullptr);
        APPERR_RETURN_IF_NOT_LOG(cagraSearchOpReset(cagraSearchOp[batch], batch),
            APP_ERR_ACL_OP_LOAD_MODEL_FAILED, "op init failed !!!");
    }
    
    APP_LOG_INFO("AscendIndexCagraImpl ResetCagraSearchOp operation end.\n");
    return APP_ERR_OK;
}

// ==================== Graph Building Functions ====================

APP_ERROR AscendIndexCagraImpl::resetAllGraphOps()
{
    APP_LOG_INFO("Resetting all graph building operators\n");
    
    auto ret = resetPreprocessDataOp();
    ASCEND_THROW_IF_NOT_MSG(ret == APP_ERR_OK, "resetPreprocessDataOp failed!");
    
    ret = resetAddReverseEdgesOp();
    ASCEND_THROW_IF_NOT_MSG(ret == APP_ERR_OK, "resetAddReverseEdgesOp failed!");
    
    ret = resetLocalJoinOp();
    ASCEND_THROW_IF_NOT_MSG(ret == APP_ERR_OK, "resetLocalJoinOp failed!");
    
    ret = resetPruneOp();
    ASCEND_THROW_IF_NOT_MSG(ret == APP_ERR_OK, "resetPruneOp failed!");
    
    ret = resetMakeRevGraphOp();
    ASCEND_THROW_IF_NOT_MSG(ret == APP_ERR_OK, "resetMakeRevGraphOp failed!");
    
    APP_LOG_INFO("All graph building operators reset successfully\n");
    
    return APP_ERR_OK;
}


APP_ERROR AscendIndexCagraImpl::resetPreprocessDataOp()
{
    auto opReset = [&](std::unique_ptr<AscendOperator> &op) {
        AscendOpDesc desc("PreprocessData");
        
        std::vector<int64_t> dataShape({ buildConfig.dataSize, static_cast<int64_t>(dim) });
        std::vector<int64_t> dimShape({ 1 });
        std::vector<int64_t> offsetShape({ 1 });
        std::vector<int64_t> outputDataShape({ buildConfig.dataSize, static_cast<int64_t>(dim) });
        std::vector<int64_t> l2NormShape({ buildConfig.dataSize });

        desc.addInputTensorDesc(ACL_FLOAT, dataShape.size(), dataShape.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_INT32, dimShape.size(), dimShape.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_UINT64, offsetShape.size(), offsetShape.data(), ACL_FORMAT_ND);
        
        desc.addOutputTensorDesc(ACL_FLOAT16, outputDataShape.size(), outputDataShape.data(), ACL_FORMAT_ND);
        desc.addOutputTensorDesc(ACL_FLOAT, l2NormShape.size(), l2NormShape.data(), ACL_FORMAT_ND);
        
        op.reset();
        op = CREATE_UNIQUE_PTR(AscendOperator, desc);
        return op->init();
    };
    
    preprocessDataOp = std::unique_ptr<AscendOperator>(nullptr);
    APPERR_RETURN_IF_NOT_LOG(opReset(preprocessDataOp), APP_ERR_ACL_OP_LOAD_MODEL_FAILED, "PreprocessDataKernel operator init failed");
    
    return APP_ERR_OK;
}

APP_ERROR AscendIndexCagraImpl::resetAddReverseEdgesOp()
{
    auto opReset = [&](std::unique_ptr<AscendOperator> &op) {
        AscendOpDesc desc("AddReverseEdges");
        
        std::vector<int64_t> graphShape({buildConfig.dataSize, static_cast<int64_t>(buildConfig.graphDegree)});
        std::vector<int64_t> numSamplesShape({1});
        std::vector<int64_t> edgeCountShape({buildConfig.dataSize});

        desc.addInputTensorDesc(ACL_INT32, graphShape.size(), graphShape.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_UINT64, numSamplesShape.size(), numSamplesShape.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_INT32, edgeCountShape.size(), edgeCountShape.data(), ACL_FORMAT_ND);
        
        desc.addOutputTensorDesc(ACL_INT32, graphShape.size(), graphShape.data(), ACL_FORMAT_ND);
        desc.addOutputTensorDesc(ACL_INT32, edgeCountShape.size(), edgeCountShape.data(), ACL_FORMAT_ND);
        
        op.reset();
        op = CREATE_UNIQUE_PTR(AscendOperator, desc);
        return op->init();
    };
    
    addReverseEdgesOp = std::unique_ptr<AscendOperator>(nullptr);
    APPERR_RETURN_IF_NOT_LOG(opReset(addReverseEdgesOp), APP_ERR_ACL_OP_LOAD_MODEL_FAILED, "AddReverseEdges operator init failed");
    
    return APP_ERR_OK;
}

APP_ERROR AscendIndexCagraImpl::resetLocalJoinOp()
{
    auto opReset = [&](std::unique_ptr<AscendOperator> &op) {
        AscendOpDesc desc("LocalJoin");
        
        std::vector<int64_t> graphShape({buildConfig.dataSize, static_cast<int64_t>(buildConfig.graphDegree)});
        std::vector<int64_t> edgeCountShape({buildConfig.dataSize});
        std::vector<int64_t> dataShape({buildConfig.dataSize, static_cast<int64_t>(dim)});
        std::vector<int64_t> l2NormShape({buildConfig.dataSize});
        std::vector<int64_t> dimShape({1});
        std::vector<int64_t> numSamplesShape({1});
        std::vector<int64_t> degreeOnDeviceShape({1});
        std::vector<int64_t> locksShape({buildConfig.dataSize * ((buildConfig.graphDegree + 31) / 32)});
        std::vector<int64_t> outputGraphShape({buildConfig.dataSize, static_cast<int64_t>(buildConfig.graphDegree)});
        std::vector<int64_t> distanceShape({buildConfig.dataSize, static_cast<int64_t>(buildConfig.graphDegree)});

        desc.addInputTensorDesc(ACL_INT32, graphShape.size(), graphShape.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_INT32, graphShape.size(), graphShape.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_INT32, edgeCountShape.size(), edgeCountShape.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_INT32, edgeCountShape.size(), edgeCountShape.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_INT32, graphShape.size(), graphShape.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_INT32, graphShape.size(), graphShape.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_INT32, edgeCountShape.size(), edgeCountShape.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_INT32, edgeCountShape.size(), edgeCountShape.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_UINT64, numSamplesShape.size(), numSamplesShape.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_FLOAT16, dataShape.size(), dataShape.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_INT32, dimShape.size(), dimShape.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_UINT64, degreeOnDeviceShape.size(), degreeOnDeviceShape.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_INT32, locksShape.size(), locksShape.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_FLOAT, l2NormShape.size(), l2NormShape.data(), ACL_FORMAT_ND);
        
        desc.addOutputTensorDesc(ACL_INT32, outputGraphShape.size(), outputGraphShape.data(), ACL_FORMAT_ND);
        desc.addOutputTensorDesc(ACL_FLOAT, distanceShape.size(), distanceShape.data(), ACL_FORMAT_ND);
        
        op.reset();
        op = CREATE_UNIQUE_PTR(AscendOperator, desc);
        return op->init();
    };
    
    localJoinOp = std::unique_ptr<AscendOperator>(nullptr);
    APPERR_RETURN_IF_NOT_LOG(opReset(localJoinOp), APP_ERR_ACL_OP_LOAD_MODEL_FAILED, "localJoinKernel operator init failed");
    
    return APP_ERR_OK;
}

APP_ERROR AscendIndexCagraImpl::resetPruneOp()
{
    auto opReset = [&](std::unique_ptr<AscendOperator> &op) {
        AscendOpDesc desc("KernPrune");
        
        std::vector<int64_t> graphShape({static_cast<int64_t>(buildConfig.dataSize), static_cast<int64_t>(buildConfig.graphDegree)});
        std::vector<int64_t> scalarShape({1});
        std::vector<int64_t> detourCountShape({static_cast<int64_t>(buildConfig.dataSize), static_cast<int64_t>(buildConfig.graphDegree)});
        std::vector<int64_t> numNoDetourEdgesShape({static_cast<int64_t>(buildConfig.dataSize)});
        std::vector<int64_t> statsShape({2});
        
        desc.addInputTensorDesc(ACL_UINT64, graphShape.size(), graphShape.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_UINT64, scalarShape.size(), scalarShape.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_UINT64, scalarShape.size(), scalarShape.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_UINT64, scalarShape.size(), scalarShape.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_UINT64, scalarShape.size(), scalarShape.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_UINT64, scalarShape.size(), scalarShape.data(), ACL_FORMAT_ND);
        
        desc.addOutputTensorDesc(ACL_UINT8, detourCountShape.size(), detourCountShape.data(), ACL_FORMAT_ND);
        desc.addOutputTensorDesc(ACL_UINT32, numNoDetourEdgesShape.size(), numNoDetourEdgesShape.data(), ACL_FORMAT_ND);
        desc.addOutputTensorDesc(ACL_UINT64, statsShape.size(), statsShape.data(), ACL_FORMAT_ND);
        
        op.reset();
        op = CREATE_UNIQUE_PTR(AscendOperator, desc);
        return op->init();
    };
    
    pruneOp = std::unique_ptr<AscendOperator>(nullptr);
    APPERR_RETURN_IF_NOT_LOG(opReset(pruneOp), APP_ERR_ACL_OP_LOAD_MODEL_FAILED, "KernPrune operator init failed");
    
    return APP_ERR_OK;
}

APP_ERROR AscendIndexCagraImpl::resetMakeRevGraphOp()
{
    auto opReset = [&](std::unique_ptr<AscendOperator> &op) {
        AscendOpDesc desc("KernMakeRevGraph");
        
        std::vector<int64_t> destNodesShape({static_cast<int64_t>(buildConfig.dataSize)});
        std::vector<int64_t> scalarShape({1});
        std::vector<int64_t> revGraphShape({static_cast<int64_t>(buildConfig.dataSize), static_cast<int64_t>(buildConfig.graphDegree)});
        std::vector<int64_t> revGraphCountShape({static_cast<int64_t>(buildConfig.dataSize)});
        
        desc.addInputTensorDesc(ACL_UINT64, destNodesShape.size(), destNodesShape.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_UINT64, scalarShape.size(), scalarShape.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_UINT64, scalarShape.size(), scalarShape.data(), ACL_FORMAT_ND);
        
        desc.addOutputTensorDesc(ACL_UINT64, revGraphShape.size(), revGraphShape.data(), ACL_FORMAT_ND);
        desc.addOutputTensorDesc(ACL_UINT32, revGraphCountShape.size(), revGraphCountShape.data(), ACL_FORMAT_ND);
        
        op.reset();
        op = CREATE_UNIQUE_PTR(AscendOperator, desc);
        return op->init();
    };
    
    makeRevGraphOp = std::unique_ptr<AscendOperator>(nullptr);
    APPERR_RETURN_IF_NOT_LOG(opReset(makeRevGraphOp), APP_ERR_ACL_OP_LOAD_MODEL_FAILED, "KernMakeRevGraph operator init failed");
    
    return APP_ERR_OK;
}

APP_ERROR AscendIndexCagraImpl::runPreprocessData(int64_t n, const float* data)
{
    APPERR_RETURN_IF_NOT(preprocessDataOp != nullptr, APP_ERR_ACL_OP_NOT_FOUND);
    APPERR_RETURN_IF_NOT(pResources != nullptr, APP_ERR_INNER_ERROR);
    
    auto streamPtr = pResources->getDefaultStream();
    auto stream = streamPtr->GetStream();
    auto &mem = pResources->getMemoryManager();
    
    preprocessedDataDevice = std::make_unique<DeviceVector<ascend::fp16>>();
    l2NormDataDevice = std::make_unique<DeviceVector<float>>();
    
    AscendTensor<float, DIMS_2> inputData(mem, {static_cast<int>(n), dim}, stream);
    AscendTensor<int32_t, DIMS_1> dimTensor(mem, {1}, stream);
    AscendTensor<uint64_t, DIMS_1> offsetTensor(mem, {1}, stream);
    
    AscendTensor<ascend::fp16, DIMS_2> preprocessedDataTemp(mem, {static_cast<int>(n), dim}, stream);
    AscendTensor<float, DIMS_1> l2NormDataTemp(mem, {static_cast<int>(n)}, stream);
    
    auto ret = aclrtMemcpy(inputData.data(), inputData.getSizeInBytes(),
                           data, n * dim * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE);
    APPERR_RETURN_IF_NOT_FMT(ret == APP_ERR_OK, APP_ERR_INNER_ERROR,
                             "Failed to copy input data to device, size: %zu", inputData.getSizeInBytes());
    
    int32_t dimValue = dim;
    ret = aclrtMemcpy(dimTensor.data(), dimTensor.getSizeInBytes(),
                      &dimValue, sizeof(int32_t), ACL_MEMCPY_HOST_TO_DEVICE);
    APPERR_RETURN_IF_NOT_FMT(ret == APP_ERR_OK, APP_ERR_INNER_ERROR,
                             "Failed to copy dim to device, size: %zu", dimTensor.getSizeInBytes());
    
    uint64_t offsetValue = 0;
    ret = aclrtMemcpy(offsetTensor.data(), offsetTensor.getSizeInBytes(),
                      &offsetValue, sizeof(uint64_t), ACL_MEMCPY_HOST_TO_DEVICE);
    APPERR_RETURN_IF_NOT_FMT(ret == APP_ERR_OK, APP_ERR_INNER_ERROR,
                             "Failed to copy offset to device, size: %zu", offsetTensor.getSizeInBytes());
    
    std::shared_ptr<std::vector<const aclDataBuffer *>> preprocessOpInput(
        new std::vector<const aclDataBuffer *>(), CommonUtils::AclInputBufferDelete);
    preprocessOpInput->emplace_back(aclCreateDataBuffer(inputData.data(), inputData.getSizeInBytes()));
    preprocessOpInput->emplace_back(aclCreateDataBuffer(dimTensor.data(), dimTensor.getSizeInBytes()));
    preprocessOpInput->emplace_back(aclCreateDataBuffer(offsetTensor.data(), offsetTensor.getSizeInBytes()));
    
    std::shared_ptr<std::vector<aclDataBuffer *>> preprocessOpOutput(
        new std::vector<aclDataBuffer *>(), CommonUtils::AclOutputBufferDelete);
    preprocessOpOutput->emplace_back(aclCreateDataBuffer(preprocessedDataTemp.data(), preprocessedDataTemp.getSizeInBytes()));
    preprocessOpOutput->emplace_back(aclCreateDataBuffer(l2NormDataTemp.data(), l2NormDataTemp.getSizeInBytes()));
    
    preprocessDataOp->exec(*preprocessOpInput, *preprocessOpOutput, stream);
    
    ret = synchronizeStream(stream);
    APPERR_RETURN_IF_NOT_FMT(ret == APP_ERR_OK, APP_ERR_INNER_ERROR,
                             "Failed to synchronize stream: %d", ret);
    
    size_t preprocessedDataSize = n * dim;
    size_t l2NormDataSize = n;
    
    preprocessedDataDevice->resize(preprocessedDataSize);
    l2NormDataDevice->resize(l2NormDataSize);
    
    ret = aclrtMemcpy(preprocessedDataDevice->data(), preprocessedDataDevice->size() * sizeof(ascend::fp16),
                      preprocessedDataTemp.data(), preprocessedDataTemp.getSizeInBytes(),
                      ACL_MEMCPY_DEVICE_TO_DEVICE);
    APPERR_RETURN_IF_NOT_FMT(ret == APP_ERR_OK, APP_ERR_INNER_ERROR,
                             "Failed to copy preprocessed data to DeviceVector: %d", ret);
    
    ret = aclrtMemcpy(l2NormDataDevice->data(), l2NormDataDevice->size() * sizeof(float),
                      l2NormDataTemp.data(), l2NormDataTemp.getSizeInBytes(),
                      ACL_MEMCPY_DEVICE_TO_DEVICE);
    APPERR_RETURN_IF_NOT_FMT(ret == APP_ERR_OK, APP_ERR_INNER_ERROR,
                             "Failed to copy L2 norm data to DeviceVector: %d", ret);

    return APP_ERR_OK;
}

APP_ERROR AscendIndexCagraImpl::runAddReverseEdges(int64_t n, uint32_t actualGraphDegree,
                                                   const AscendTensor<uint64_t, DIMS_1>& numSamplesTensor,
                                                   const AscendTensor<uint32_t, DIMS_2>& graphDevice,
                                                   const AscendTensor<int, DIMS_1>& forwardEdgeCountsDevice,
                                                   AscendTensor<uint32_t, DIMS_2>& reverseGraphDevice,
                                                   AscendTensor<int, DIMS_1>& reverseEdgeCountsDevice)
{
    APPERR_RETURN_IF_NOT(addReverseEdgesOp != nullptr, APP_ERR_ACL_OP_NOT_FOUND);
    APPERR_RETURN_IF_NOT(pResources != nullptr, APP_ERR_INNER_ERROR);
    
    auto streamPtr = pResources->getDefaultStream();
    auto stream = streamPtr->GetStream();
    auto &mem = pResources->getMemoryManager();
    
    AscendTensor<int, DIMS_2> graphInt32Device(mem, {static_cast<int>(n), static_cast<int>(actualGraphDegree)}, stream);
    auto ret = aclrtMemcpy(graphInt32Device.data(), graphInt32Device.getSizeInBytes(),
                           graphDevice.data(), graphDevice.getSizeInBytes(), ACL_MEMCPY_DEVICE_TO_DEVICE);
    APPERR_RETURN_IF_NOT_LOG(ret == APP_ERR_OK, APP_ERR_INNER_ERROR, "Failed to copy graph to int32 tensor!");
    
    AscendTensor<int, DIMS_2> reverseGraphInt32Device(mem, {static_cast<int>(n), static_cast<int>(actualGraphDegree)}, stream);
    
    std::shared_ptr<std::vector<const aclDataBuffer *>> addReverseEdgesOpInput(
        new std::vector<const aclDataBuffer *>(), CommonUtils::AclInputBufferDelete);
    
    addReverseEdgesOpInput->emplace_back(aclCreateDataBuffer(graphInt32Device.data(), graphInt32Device.getSizeInBytes()));
    addReverseEdgesOpInput->emplace_back(aclCreateDataBuffer(numSamplesTensor.data(), numSamplesTensor.getSizeInBytes()));
    addReverseEdgesOpInput->emplace_back(aclCreateDataBuffer(forwardEdgeCountsDevice.data(), forwardEdgeCountsDevice.getSizeInBytes()));
    
    std::shared_ptr<std::vector<aclDataBuffer *>> addReverseEdgesOpOutput(
        new std::vector<aclDataBuffer *>(), CommonUtils::AclOutputBufferDelete);
    addReverseEdgesOpOutput->emplace_back(aclCreateDataBuffer(reverseGraphInt32Device.data(), reverseGraphInt32Device.getSizeInBytes()));
    addReverseEdgesOpOutput->emplace_back(aclCreateDataBuffer(reverseEdgeCountsDevice.data(), reverseEdgeCountsDevice.getSizeInBytes()));
    
    addReverseEdgesOp->exec(*addReverseEdgesOpInput, *addReverseEdgesOpOutput, stream);
    
    ret = synchronizeStream(stream);
    APPERR_RETURN_IF_NOT_FMT(ret == APP_ERR_OK, APP_ERR_INNER_ERROR,
                             "Failed to synchronize stream: %d", ret);
    
    ret = aclrtMemcpy(reverseGraphDevice.data(), reverseGraphDevice.getSizeInBytes(),
                      reverseGraphInt32Device.data(), reverseGraphInt32Device.getSizeInBytes(),
                      ACL_MEMCPY_DEVICE_TO_DEVICE);
    APPERR_RETURN_IF_NOT_LOG(ret == APP_ERR_OK, APP_ERR_INNER_ERROR, "Failed to copy reverse graph from int32 to uint32 tensor!");
    
    return APP_ERR_OK;
}

APP_ERROR AscendIndexCagraImpl::runLocalJoinKernel(int64_t n,
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
                                                   AscendTensor<float, DIMS_2>& outputDistancesDevice)
{
    APPERR_RETURN_IF_NOT(localJoinOp != nullptr, APP_ERR_ACL_OP_NOT_FOUND);
    APPERR_RETURN_IF_NOT(pResources != nullptr, APP_ERR_INNER_ERROR);
    
    auto streamPtr = pResources->getDefaultStream();
    auto stream = streamPtr->GetStream();
    auto &mem = pResources->getMemoryManager();
    
    AscendTensor<int, DIMS_1> locksDevice(mem, {static_cast<int>(n) * ((buildConfig.graphDegree + 31) / 32)}, stream);
    auto ret = aclrtMemset(locksDevice.data(), locksDevice.getSizeInBytes(), 0, locksDevice.getSizeInBytes());
    APPERR_RETURN_IF_NOT_FMT(ret == APP_ERR_OK, APP_ERR_INNER_ERROR,
                             "Failed to initialize locks array to 0, size: %zu", locksDevice.getSizeInBytes());
    
    AscendTensor<int32_t, DIMS_1> dimTensor(mem, {1}, stream);
    int32_t dimValue = static_cast<int32_t>(dim);
    ret = aclrtMemcpy(dimTensor.data(), dimTensor.getSizeInBytes(),
                      &dimValue, sizeof(int32_t), ACL_MEMCPY_HOST_TO_DEVICE);
    APPERR_RETURN_IF_NOT_FMT(ret == APP_ERR_OK, APP_ERR_INNER_ERROR,
                             "Failed to copy dim to device, size: %zu", dimTensor.getSizeInBytes());
    
    AscendTensor<uint64_t, DIMS_1> numSamplesTensor(mem, {1}, stream);
    uint64_t numSamplesValue = static_cast<uint64_t>(buildConfig.graphDegree);
    ret = aclrtMemcpy(numSamplesTensor.data(), numSamplesTensor.getSizeInBytes(),
                      &numSamplesValue, sizeof(uint64_t), ACL_MEMCPY_HOST_TO_DEVICE);
    APPERR_RETURN_IF_NOT_FMT(ret == APP_ERR_OK, APP_ERR_INNER_ERROR,
                             "Failed to copy numSamples to device, size: %zu", numSamplesTensor.getSizeInBytes());
    
    AscendTensor<uint64_t, DIMS_1> degreeOnDeviceTensor(mem, {1}, stream);
    uint64_t degreeOnDeviceValue = static_cast<uint64_t>(buildConfig.graphDegree);
    ret = aclrtMemcpy(degreeOnDeviceTensor.data(), degreeOnDeviceTensor.getSizeInBytes(),
                      &degreeOnDeviceValue, sizeof(uint64_t), ACL_MEMCPY_HOST_TO_DEVICE);
    APPERR_RETURN_IF_NOT_FMT(ret == APP_ERR_OK, APP_ERR_INNER_ERROR,
                             "Failed to copy degreeOnDevice to device, size: %zu", degreeOnDeviceTensor.getSizeInBytes());
    
    AscendTensor<ascend::fp16, DIMS_2> preprocessedDataTensor(mem, {static_cast<int>(n), dim}, stream);
    AscendTensor<float, DIMS_1> l2NormDataTensor(mem, {static_cast<int>(n)}, stream);
    
    ret = aclrtMemcpy(preprocessedDataTensor.data(), preprocessedDataTensor.getSizeInBytes(),
                      preprocessedData, n * dim * sizeof(ascend::fp16), ACL_MEMCPY_DEVICE_TO_DEVICE);
    APPERR_RETURN_IF_NOT_FMT(ret == APP_ERR_OK, APP_ERR_INNER_ERROR,
                             "Failed to copy preprocessed data to tensor, size: %zu", preprocessedDataTensor.getSizeInBytes());
    
    ret = aclrtMemcpy(l2NormDataTensor.data(), l2NormDataTensor.getSizeInBytes(),
                      l2NormData, n * sizeof(float), ACL_MEMCPY_DEVICE_TO_DEVICE);
    APPERR_RETURN_IF_NOT_FMT(ret == APP_ERR_OK, APP_ERR_INNER_ERROR,
                             "Failed to copy L2 norm data to tensor, size: %zu", l2NormDataTensor.getSizeInBytes());
    
    AscendTensor<int, DIMS_2> newGraphInt32Device(mem, {static_cast<int>(n), static_cast<int>(buildConfig.graphDegree)}, stream);
    AscendTensor<int, DIMS_2> newReverseGraphInt32Device(mem, {static_cast<int>(n), static_cast<int>(buildConfig.graphDegree)}, stream);
    AscendTensor<int, DIMS_2> oldGraphInt32Device(mem, {static_cast<int>(n), static_cast<int>(buildConfig.graphDegree)}, stream);
    AscendTensor<int, DIMS_2> oldReverseGraphInt32Device(mem, {static_cast<int>(n), static_cast<int>(buildConfig.graphDegree)}, stream);
    
    ret = aclrtMemcpy(newGraphInt32Device.data(), newGraphInt32Device.getSizeInBytes(),
                      newGraphDevice.data(), newGraphDevice.getSizeInBytes(), ACL_MEMCPY_DEVICE_TO_DEVICE);
    APPERR_RETURN_IF_NOT_LOG(ret == APP_ERR_OK, APP_ERR_INNER_ERROR, "Failed to copy newGraph to int32 tensor!");
    
    ret = aclrtMemcpy(newReverseGraphInt32Device.data(), newReverseGraphInt32Device.getSizeInBytes(),
                      newReverseGraphDevice.data(), newReverseGraphDevice.getSizeInBytes(), ACL_MEMCPY_DEVICE_TO_DEVICE);
    APPERR_RETURN_IF_NOT_LOG(ret == APP_ERR_OK, APP_ERR_INNER_ERROR, "Failed to copy newReverseGraph to int32 tensor!");
    
    ret = aclrtMemcpy(oldGraphInt32Device.data(), oldGraphInt32Device.getSizeInBytes(),
                      oldGraphDevice.data(), oldGraphDevice.getSizeInBytes(), ACL_MEMCPY_DEVICE_TO_DEVICE);
    APPERR_RETURN_IF_NOT_LOG(ret == APP_ERR_OK, APP_ERR_INNER_ERROR, "Failed to copy oldGraph to int32 tensor!");
    
    ret = aclrtMemcpy(oldReverseGraphInt32Device.data(), oldReverseGraphInt32Device.getSizeInBytes(),
                      oldReverseGraphDevice.data(), oldReverseGraphDevice.getSizeInBytes(), ACL_MEMCPY_DEVICE_TO_DEVICE);
    APPERR_RETURN_IF_NOT_LOG(ret == APP_ERR_OK, APP_ERR_INNER_ERROR, "Failed to copy oldReverseGraph to int32 tensor!");
    
    AscendTensor<int, DIMS_2> outputGraphInt32Device(mem, {static_cast<int>(n), static_cast<int>(buildConfig.graphDegree)}, stream);
    
    std::shared_ptr<std::vector<const aclDataBuffer *>> localJoinOpInput(
        new std::vector<const aclDataBuffer *>(), CommonUtils::AclInputBufferDelete);
    localJoinOpInput->emplace_back(aclCreateDataBuffer(newGraphInt32Device.data(), newGraphInt32Device.getSizeInBytes()));
    localJoinOpInput->emplace_back(aclCreateDataBuffer(newReverseGraphInt32Device.data(), newReverseGraphInt32Device.getSizeInBytes()));
    localJoinOpInput->emplace_back(aclCreateDataBuffer(newForwardEdgeCountsDevice.data(), newForwardEdgeCountsDevice.getSizeInBytes()));
    localJoinOpInput->emplace_back(aclCreateDataBuffer(newBackwardEdgeCountsDevice.data(), newBackwardEdgeCountsDevice.getSizeInBytes()));
    localJoinOpInput->emplace_back(aclCreateDataBuffer(oldGraphInt32Device.data(), oldGraphInt32Device.getSizeInBytes()));
    localJoinOpInput->emplace_back(aclCreateDataBuffer(oldReverseGraphInt32Device.data(), oldReverseGraphInt32Device.getSizeInBytes()));
    localJoinOpInput->emplace_back(aclCreateDataBuffer(oldForwardEdgeCountsDevice.data(), oldForwardEdgeCountsDevice.getSizeInBytes()));
    localJoinOpInput->emplace_back(aclCreateDataBuffer(oldBackwardEdgeCountsDevice.data(), oldBackwardEdgeCountsDevice.getSizeInBytes()));
    localJoinOpInput->emplace_back(aclCreateDataBuffer(numSamplesTensor.data(), numSamplesTensor.getSizeInBytes()));
    localJoinOpInput->emplace_back(aclCreateDataBuffer(preprocessedDataTensor.data(), preprocessedDataTensor.getSizeInBytes()));
    localJoinOpInput->emplace_back(aclCreateDataBuffer(dimTensor.data(), dimTensor.getSizeInBytes()));
    localJoinOpInput->emplace_back(aclCreateDataBuffer(degreeOnDeviceTensor.data(), degreeOnDeviceTensor.getSizeInBytes()));
    localJoinOpInput->emplace_back(aclCreateDataBuffer(locksDevice.data(), locksDevice.getSizeInBytes()));
    localJoinOpInput->emplace_back(aclCreateDataBuffer(l2NormDataTensor.data(), l2NormDataTensor.getSizeInBytes()));
    
    std::shared_ptr<std::vector<aclDataBuffer *>> localJoinOpOutput(
        new std::vector<aclDataBuffer *>(), CommonUtils::AclOutputBufferDelete);
    localJoinOpOutput->emplace_back(aclCreateDataBuffer(outputGraphInt32Device.data(), outputGraphInt32Device.getSizeInBytes()));
    localJoinOpOutput->emplace_back(aclCreateDataBuffer(outputDistancesDevice.data(), outputDistancesDevice.getSizeInBytes()));
    
    localJoinOp->exec(*localJoinOpInput, *localJoinOpOutput, stream);
    
    ret = synchronizeStream(stream);
    APPERR_RETURN_IF_NOT_FMT(ret == APP_ERR_OK, APP_ERR_INNER_ERROR,
                             "Failed to synchronize stream: %d", ret);
    
    ret = aclrtMemcpy(outputGraphDevice.data(), outputGraphDevice.getSizeInBytes(),
                      outputGraphInt32Device.data(), outputGraphInt32Device.getSizeInBytes(),
                      ACL_MEMCPY_DEVICE_TO_DEVICE);
    APPERR_RETURN_IF_NOT_LOG(ret == APP_ERR_OK, APP_ERR_INNER_ERROR, "Failed to copy output graph from int32 to uint32 tensor!");
    
    return APP_ERR_OK;
}

APP_ERROR AscendIndexCagraImpl::runPrune(int64_t n,
                                         const AscendTensor<uint64_t, DIMS_2>& knnGraphDevice,
                                         AscendTensor<uint8_t, DIMS_2>& detourCountDevice,
                                         AscendTensor<uint32_t, DIMS_1>& numNoDetourEdgesDevice,
                                         AscendTensor<uint64_t, DIMS_1>& statsDevice)
{
    APPERR_RETURN_IF_NOT(pruneOp != nullptr, APP_ERR_ACL_OP_NOT_FOUND);
    APPERR_RETURN_IF_NOT(pResources != nullptr, APP_ERR_INNER_ERROR);
    
    auto streamPtr = pResources->getDefaultStream();
    auto stream = streamPtr->GetStream();
    auto &mem = pResources->getMemoryManager();
    
    uint64_t graphSize = static_cast<uint64_t>(n);
    uint64_t batchSize = std::min(static_cast<uint32_t>(graphSize), 256 * 1024u);
    uint64_t numBatch = (graphSize + batchSize - 1) / batchSize;
    
    APP_LOG_INFO("Prune batch processing: graphSize=%lu, batchSize=%lu, numBatch=%lu\n", 
                 graphSize, batchSize, numBatch);
    
    AscendTensor<uint64_t, DIMS_1> nodeCountTensor(mem, {1}, stream);
    uint64_t nodeCountValue = static_cast<uint64_t>(n);
    auto ret = aclrtMemcpy(nodeCountTensor.data(), nodeCountTensor.getSizeInBytes(),
                           &nodeCountValue, sizeof(uint64_t), ACL_MEMCPY_HOST_TO_DEVICE);
    APPERR_RETURN_IF_NOT_FMT(ret == APP_ERR_OK, APP_ERR_INNER_ERROR,
                             "Failed to copy node count to device, size: %zu", nodeCountTensor.getSizeInBytes());

    AscendTensor<uint64_t, DIMS_1> inputDegreeTensor(mem, {1}, stream);
    uint64_t inputDegreeValue = static_cast<uint64_t>(buildConfig.graphDegree);
    ret = aclrtMemcpy(inputDegreeTensor.data(), inputDegreeTensor.getSizeInBytes(),
                      &inputDegreeValue, sizeof(uint64_t), ACL_MEMCPY_HOST_TO_DEVICE);
    APPERR_RETURN_IF_NOT_FMT(ret == APP_ERR_OK, APP_ERR_INNER_ERROR,
                             "Failed to copy input degree to device, size: %zu", inputDegreeTensor.getSizeInBytes());
    
    AscendTensor<uint64_t, DIMS_1> outputDegreeTensor(mem, {1}, stream);
    uint64_t outputDegreeValue = static_cast<uint64_t>(buildConfig.graphDegree);
    ret = aclrtMemcpy(outputDegreeTensor.data(), outputDegreeTensor.getSizeInBytes(),
                      &outputDegreeValue, sizeof(uint64_t), ACL_MEMCPY_HOST_TO_DEVICE);
    APPERR_RETURN_IF_NOT_FMT(ret == APP_ERR_OK, APP_ERR_INNER_ERROR,
                             "Failed to copy output degree to device, size: %zu", outputDegreeTensor.getSizeInBytes());
    
    AscendTensor<uint64_t, DIMS_1> batchSizeTensor(mem, {1}, stream);
    
    for (uint64_t batchId = 0; batchId < numBatch; ++batchId) {
        uint64_t startIdx = batchId * batchSize;
        uint64_t currentBatchSize = std::min(batchSize, graphSize - startIdx);
        
        ret = runPruneBatch(batchId, batchSize, currentBatchSize, knnGraphDevice,
                            nodeCountTensor, inputDegreeTensor, outputDegreeTensor,
                            batchSizeTensor, detourCountDevice, numNoDetourEdgesDevice,
                            statsDevice, stream);
        APPERR_RETURN_IF_NOT_FMT(ret == APP_ERR_OK, APP_ERR_INNER_ERROR,
                                 "runPruneBatch failed for batch %lu!", batchId);
    }
    
    return APP_ERR_OK;
}

APP_ERROR AscendIndexCagraImpl::runPruneBatch(uint64_t batchId,
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
                                              aclrtStream stream)
{
    auto ret = aclrtMemcpy(batchSizeTensor.data(), batchSizeTensor.getSizeInBytes(),
                          &currentBatchSize, sizeof(uint64_t), ACL_MEMCPY_HOST_TO_DEVICE);
    APPERR_RETURN_IF_NOT_FMT(ret == APP_ERR_OK, APP_ERR_INNER_ERROR,
                             "Failed to copy batch size to device, size: %zu", batchSizeTensor.getSizeInBytes());
    
    auto streamPtr = pResources->getDefaultStream();
    auto &mem = pResources->getMemoryManager();
    
    AscendTensor<uint64_t, DIMS_1> batchIdTensor(mem, {1}, stream);
    ret = aclrtMemcpy(batchIdTensor.data(), batchIdTensor.getSizeInBytes(),
                      &batchId, sizeof(uint64_t), ACL_MEMCPY_HOST_TO_DEVICE);
    APPERR_RETURN_IF_NOT_FMT(ret == APP_ERR_OK, APP_ERR_INNER_ERROR,
                             "Failed to copy batch id to device, size: %zu", batchIdTensor.getSizeInBytes());
    
    std::shared_ptr<std::vector<const aclDataBuffer *>> pruneOpInput(
        new std::vector<const aclDataBuffer *>(), CommonUtils::AclInputBufferDelete);
    pruneOpInput->emplace_back(aclCreateDataBuffer(knnGraphDevice.data(), knnGraphDevice.getSizeInBytes()));
    pruneOpInput->emplace_back(aclCreateDataBuffer(nodeCountTensor.data(), nodeCountTensor.getSizeInBytes()));
    pruneOpInput->emplace_back(aclCreateDataBuffer(inputDegreeTensor.data(), inputDegreeTensor.getSizeInBytes()));
    pruneOpInput->emplace_back(aclCreateDataBuffer(outputDegreeTensor.data(), outputDegreeTensor.getSizeInBytes()));
    pruneOpInput->emplace_back(aclCreateDataBuffer(batchSizeTensor.data(), batchSizeTensor.getSizeInBytes()));
    pruneOpInput->emplace_back(aclCreateDataBuffer(batchIdTensor.data(), batchIdTensor.getSizeInBytes()));
    
    std::shared_ptr<std::vector<aclDataBuffer *>> pruneOpOutput(
        new std::vector<aclDataBuffer *>(), CommonUtils::AclOutputBufferDelete);
    pruneOpOutput->emplace_back(aclCreateDataBuffer(detourCountDevice.data(), detourCountDevice.getSizeInBytes()));
    pruneOpOutput->emplace_back(aclCreateDataBuffer(numNoDetourEdgesDevice.data(), numNoDetourEdgesDevice.getSizeInBytes()));
    pruneOpOutput->emplace_back(aclCreateDataBuffer(statsDevice.data(), statsDevice.getSizeInBytes()));

    pruneOp->exec(*pruneOpInput, *pruneOpOutput, stream);
    
    ret = synchronizeStream(stream);
    APPERR_RETURN_IF_NOT_FMT(ret == APP_ERR_OK, APP_ERR_INNER_ERROR,
                             "Failed to synchronize stream: %d", ret);
    
    return APP_ERR_OK;
}

APP_ERROR AscendIndexCagraImpl::runMakeRevGraph(int64_t n,
                                                const AscendTensor<uint64_t, DIMS_1>& destNodesDevice,
                                                AscendTensor<uint64_t, DIMS_2>& revGraphDevice,
                                                AscendTensor<uint32_t, DIMS_1>& revGraphCountDevice)
{
    APPERR_RETURN_IF_NOT(makeRevGraphOp != nullptr, APP_ERR_ACL_OP_NOT_FOUND);
    APPERR_RETURN_IF_NOT(pResources != nullptr, APP_ERR_INNER_ERROR);
    
    auto streamPtr = pResources->getDefaultStream();
    auto stream = streamPtr->GetStream();
    auto &mem = pResources->getMemoryManager();
    
    // 初始化输出tensor为0xFF（表示无效邻居）
    auto ret = aclrtMemset(revGraphDevice.data(), revGraphDevice.getSizeInBytes(),
                           0xFF, revGraphDevice.getSizeInBytes());
    APPERR_RETURN_IF_NOT_FMT(ret == APP_ERR_OK, APP_ERR_INNER_ERROR,
                             "Failed to initialize revGraphDevice to 0xFF, size: %zu", revGraphDevice.getSizeInBytes());
    
    ret = aclrtMemset(revGraphCountDevice.data(), revGraphCountDevice.getSizeInBytes(),
                      0, revGraphCountDevice.getSizeInBytes());
    APPERR_RETURN_IF_NOT_FMT(ret == APP_ERR_OK, APP_ERR_INNER_ERROR,
                             "Failed to initialize revGraphCountDevice to 0, size: %zu", revGraphCountDevice.getSizeInBytes());
    
    AscendTensor<uint64_t, DIMS_1> graphSizeTensor(mem, {1}, stream);
    uint64_t graphSizeValue = static_cast<uint64_t>(n);
    ret = aclrtMemcpy(graphSizeTensor.data(), graphSizeTensor.getSizeInBytes(),
                      &graphSizeValue, sizeof(uint64_t), ACL_MEMCPY_HOST_TO_DEVICE);
    APPERR_RETURN_IF_NOT_FMT(ret == APP_ERR_OK, APP_ERR_INNER_ERROR,
                             "Failed to copy graph size to device, size: %zu", graphSizeTensor.getSizeInBytes());
    
    AscendTensor<uint64_t, DIMS_1> degreeTensor(mem, {1}, stream);
    uint64_t degreeValue = static_cast<uint64_t>(buildConfig.graphDegree);
    ret = aclrtMemcpy(degreeTensor.data(), degreeTensor.getSizeInBytes(),
                      &degreeValue, sizeof(uint64_t), ACL_MEMCPY_HOST_TO_DEVICE);
    APPERR_RETURN_IF_NOT_FMT(ret == APP_ERR_OK, APP_ERR_INNER_ERROR,
                             "Failed to copy degree to device, size: %zu", degreeTensor.getSizeInBytes());
    
    // 按列遍历：每轮处理所有节点的第k个邻居
    for (uint64_t k = 0; k < buildConfig.graphDegree; ++k) {
        if (k % LOG_INTERVAL == 0) {
            APP_LOG_INFO("KernMakeRevGraph processing column %lu/%u\n", k + 1, buildConfig.graphDegree);
        }
        
        // 提取第k列：所有节点的第k个邻居
        std::vector<uint64_t> destNodesHost(n);
        #pragma omp parallel for
        for (int64_t i = 0; i < n; ++i) {
            // destNodesDevice是扁平化的，第i个节点第k个邻居的位置
            size_t idx = static_cast<size_t>(i) * buildConfig.graphDegree + k;
            destNodesHost[i] = destNodesDevice.data()[idx];
        }
        
        // 拷贝到设备
        AscendTensor<uint64_t, DIMS_1> currentColumnDevice(mem, {static_cast<int>(n)}, stream);
        ret = aclrtMemcpy(currentColumnDevice.data(), currentColumnDevice.getSizeInBytes(),
                          destNodesHost.data(), n * sizeof(uint64_t), ACL_MEMCPY_HOST_TO_DEVICE);
        APPERR_RETURN_IF_NOT_FMT(ret == APP_ERR_OK, APP_ERR_INNER_ERROR,
                                 "Failed to copy column %lu to device", k);
        
        std::shared_ptr<std::vector<const aclDataBuffer *>> revGraphOpInput(
            new std::vector<const aclDataBuffer *>(), CommonUtils::AclInputBufferDelete);
        revGraphOpInput->emplace_back(aclCreateDataBuffer(currentColumnDevice.data(), currentColumnDevice.getSizeInBytes()));
        revGraphOpInput->emplace_back(aclCreateDataBuffer(graphSizeTensor.data(), graphSizeTensor.getSizeInBytes()));
        revGraphOpInput->emplace_back(aclCreateDataBuffer(degreeTensor.data(), degreeTensor.getSizeInBytes()));
        
        std::shared_ptr<std::vector<aclDataBuffer *>> revGraphOpOutput(
            new std::vector<aclDataBuffer *>(), CommonUtils::AclOutputBufferDelete);
        revGraphOpOutput->emplace_back(aclCreateDataBuffer(revGraphDevice.data(), revGraphDevice.getSizeInBytes()));
        revGraphOpOutput->emplace_back(aclCreateDataBuffer(revGraphCountDevice.data(), revGraphCountDevice.getSizeInBytes()));
        
        makeRevGraphOp->exec(*revGraphOpInput, *revGraphOpOutput, stream);
        
        ret = synchronizeStream(stream);
        APPERR_RETURN_IF_NOT_FMT(ret == APP_ERR_OK, APP_ERR_INNER_ERROR,
                                 "Failed to synchronize stream for column %lu: %d", k, ret);
    }
    
    return APP_ERR_OK;
}

APP_ERROR AscendIndexCagraImpl::randomInitializeGraph(int64_t n, uint32_t* graph)
{
    APP_LOG_INFO("Randomly initializing graph for %ld points with degree %u\n", n, buildConfig.graphDegree);
    
    APPERR_RETURN_IF_NOT(n > 0, APP_ERR_INVALID_PARAM);
    APPERR_RETURN_IF_NOT(graph != nullptr, APP_ERR_INVALID_PARAM);
    APPERR_RETURN_IF_NOT(buildConfig.graphDegree > 0, APP_ERR_INVALID_PARAM);
    APPERR_RETURN_IF_NOT(buildConfig.graphDegree <= static_cast<uint32_t>(n) - 1, APP_ERR_INVALID_PARAM);
    
    ::ascend::RandomGenerator rng(RANDOM_SEED);
    
    size_t totalElements = static_cast<size_t>(n) * static_cast<size_t>(buildConfig.graphDegree);
    
    for (int64_t i = 0; i < n; ++i) {
        std::vector<bool> selected(n, false);
        selected[i] = true;
        
        for (uint32_t j = 0; j < buildConfig.graphDegree; ++j) {
            uint32_t neighbor;
            int attempts = 0;
            const int MAX_ATTEMPTS = 1000;
            
            do {
                neighbor = static_cast<uint32_t>(rng.RandUnsignedInt(n));
                attempts++;
                
                if (attempts > MAX_ATTEMPTS) {
                    APP_LOG_ERROR("Failed to find unique neighbor for point %ld after %d attempts\n", i, MAX_ATTEMPTS);
                    return APP_ERR_INNER_ERROR;
                }
            } while (selected[neighbor]);
            
            selected[neighbor] = true;
            graph[i * buildConfig.graphDegree + j] = neighbor;
        }
    }
    
    APP_LOG_INFO("Graph randomly initialized successfully with seed %u\n", RANDOM_SEED);
    
    return APP_ERR_OK;
}

void AscendIndexCagraImpl::sampleNodeNeighbors(int64_t nodeId, int64_t n, const uint32_t* currentGraph,
                                               uint32_t* candidates, SampleCounts& counts)
{
    const uint32_t* currentNeighbors = &currentGraph[nodeId * buildConfig.graphDegree];
    counts.oldCount = 0;
    counts.newCount = 0;
    
    // 前 graphDegree/2 个：从当前邻居中直接采样
    for (uint32_t j = 0; j < buildConfig.graphDegree && counts.oldCount < buildConfig.graphDegree / 2; ++j) {
        uint32_t neighbor = currentNeighbors[j];
        if (neighbor >= static_cast<uint32_t>(n)) continue;
        if (neighbor == static_cast<uint32_t>(nodeId)) continue;
        candidates[counts.oldCount] = neighbor;
        counts.oldCount++;
    }
    
    // 后 graphDegree/2 个：从邻居的邻居中采样
    for (uint32_t j = 0; j < buildConfig.graphDegree && counts.newCount < buildConfig.graphDegree - counts.oldCount; ++j) {
        uint32_t neighbor = currentNeighbors[j];
        if (neighbor >= static_cast<uint32_t>(n)) continue;
        const uint32_t* neighborNeighbors = &currentGraph[neighbor * buildConfig.graphDegree];
        
        for (uint32_t k = 0; k < buildConfig.graphDegree && counts.newCount < buildConfig.graphDegree - counts.oldCount; ++k) {
            uint32_t candidate = neighborNeighbors[k];
            if (candidate >= static_cast<uint32_t>(n)) continue;
            if (candidate == static_cast<uint32_t>(nodeId)) continue;
            
            bool alreadySampled = false;
            for (int m = 0; m < counts.totalCount(); ++m) {
                if (candidates[m] == candidate) {
                    alreadySampled = true;
                    break;
                }
            }
            
            if (!alreadySampled) {
                candidates[counts.totalCount()] = candidate;
                counts.newCount++;
            }
        }
    }
    
    // 如果数据量不足，随机填充
    ::ascend::RandomGenerator rng(RANDOM_SEED + static_cast<int64_t>(nodeId));
    while (counts.totalCount() < buildConfig.graphDegree) {
        uint32_t candidate = static_cast<uint32_t>(rng.RandUnsignedInt(n));
        if (candidate == static_cast<uint32_t>(nodeId)) continue;
        
        // 检查是否已采样
        bool alreadySampled = false;
        for (int m = 0; m < counts.totalCount(); ++m) {
            if (candidates[m] == candidate) {
                alreadySampled = true;
                break;
            }
        }
        if (alreadySampled) continue;
        
        candidates[counts.totalCount()] = candidate;
        counts.newCount++;
    }
}

APP_ERROR AscendIndexCagraImpl::sampleOldNewNeighbors(int64_t n, const uint32_t* currentGraph, uint32_t* newCandidates,
                                                      std::vector<int>& oldForwardEdgeCounts, std::vector<int>& newForwardEdgeCounts)
{
    APPERR_RETURN_IF_NOT(n > 0, APP_ERR_INVALID_PARAM);
    APPERR_RETURN_IF_NOT(currentGraph != nullptr, APP_ERR_INVALID_PARAM);
    APPERR_RETURN_IF_NOT(newCandidates != nullptr, APP_ERR_INVALID_PARAM);
    APPERR_RETURN_IF_NOT(buildConfig.graphDegree > 0, APP_ERR_INVALID_PARAM);
    
    oldForwardEdgeCounts.resize(n, 0);
    newForwardEdgeCounts.resize(n, 0);
    
    #pragma omp parallel for
    for (int64_t i = 0; i < n; ++i) {
        uint32_t* candidates = &newCandidates[i * buildConfig.graphDegree];
        SampleCounts counts;
        
        sampleNodeNeighbors(i, n, currentGraph, candidates, counts);
        
        // forwardEdgeCounts 表示有效邻居的总数
        // oldForwardEdgeCounts 用于 currentGraph，表示当前图中每个节点的邻居数量（graphDegree）
        // newForwardEdgeCounts 用于 newCandidates，表示新候选邻居的总数（counts.totalCount()）
        oldForwardEdgeCounts[i] = buildConfig.graphDegree;
        newForwardEdgeCounts[i] = counts.totalCount();
    }

    return APP_ERR_OK;
}

APP_ERROR AscendIndexCagraImpl::prepareGraphBuild(int64_t n, const float* data, uint32_t* graph)
{
    auto ret = runPreprocessData(n, data);
    ASCEND_THROW_IF_NOT_MSG(ret == APP_ERR_OK, "runPreprocessData failed!");
    
    ret = randomInitializeGraph(n, graph);
    ASCEND_THROW_IF_NOT_MSG(ret == APP_ERR_OK, "randomInitializeGraph failed!");
    
    size_t graphElements = static_cast<size_t>(n) * static_cast<size_t>(buildConfig.graphDegree);
    size_t candidateElements = static_cast<size_t>(n) * static_cast<size_t>(buildConfig.graphDegree);
    
    currentGraph = std::make_unique<uint32_t[]>(graphElements);
    newCandidates = std::make_unique<uint32_t[]>(candidateElements);
    
    ret = memcpy_s(currentGraph.get(), graphElements * sizeof(uint32_t), graph, graphElements * sizeof(uint32_t));
    APPERR_RETURN_IF_NOT_LOG(ret == 0, APP_ERR_INNER_ERROR, "memcpy_s failed in prepareGraphBuild");
    
    return APP_ERR_OK;
}

APP_ERROR AscendIndexCagraImpl::executeNNdescentIteration(int64_t n,
                                                          const std::vector<int>& oldForwardEdgeCounts,
                                                          const std::vector<int>& newForwardEdgeCounts)
{
    auto streamPtr = pResources->getDefaultStream();
    auto stream = streamPtr->GetStream();
    auto &mem = pResources->getMemoryManager();
    
    AscendTensor<uint32_t, DIMS_2> currentGraphDevice(mem, {static_cast<int>(n), static_cast<int>(buildConfig.graphDegree)}, stream);
    size_t graphElements = static_cast<size_t>(n) * static_cast<size_t>(buildConfig.graphDegree);
    auto ret = aclrtMemcpy(currentGraphDevice.data(), currentGraphDevice.getSizeInBytes(),
                           currentGraph.get(), graphElements * sizeof(uint32_t), ACL_MEMCPY_HOST_TO_DEVICE);
    ASCEND_THROW_IF_NOT_MSG(ret == APP_ERR_OK, "Failed to copy current graph to device!");
    
    AscendTensor<uint32_t, DIMS_2> newCandidatesDevice(mem, {static_cast<int>(n), static_cast<int>(buildConfig.graphDegree)}, stream);
    size_t candidateElements = static_cast<size_t>(n) * static_cast<size_t>(buildConfig.graphDegree);
    ret = aclrtMemcpy(newCandidatesDevice.data(), newCandidatesDevice.getSizeInBytes(),
                      newCandidates.get(), candidateElements * sizeof(uint32_t), ACL_MEMCPY_HOST_TO_DEVICE);
    ASCEND_THROW_IF_NOT_MSG(ret == APP_ERR_OK, "Failed to copy new candidates to device!");
    
    AscendTensor<int, DIMS_1> oldForwardEdgeCountsDevice(mem, {static_cast<int>(n)}, stream);
    ret = aclrtMemcpy(oldForwardEdgeCountsDevice.data(), oldForwardEdgeCountsDevice.getSizeInBytes(),
                      oldForwardEdgeCounts.data(), n * sizeof(int), ACL_MEMCPY_HOST_TO_DEVICE);
    ASCEND_THROW_IF_NOT_MSG(ret == APP_ERR_OK, "Failed to copy old forward edge counts to device!");
    
    AscendTensor<int, DIMS_1> newForwardEdgeCountsDevice(mem, {static_cast<int>(n)}, stream);
    ret = aclrtMemcpy(newForwardEdgeCountsDevice.data(), newForwardEdgeCountsDevice.getSizeInBytes(),
                      newForwardEdgeCounts.data(), n * sizeof(int), ACL_MEMCPY_HOST_TO_DEVICE);
    ASCEND_THROW_IF_NOT_MSG(ret == APP_ERR_OK, "Failed to copy new forward edge counts to device!");
    
    ret = synchronizeStream(stream);
    ASCEND_THROW_IF_NOT_MSG(ret == APP_ERR_OK, "Failed to synchronize stream!");
    
    AscendTensor<uint64_t, DIMS_1> numSamplesTensor(mem, {1}, stream);
    uint64_t numSamplesValue = static_cast<uint64_t>(buildConfig.graphDegree);
    auto aclRet = aclrtMemcpy(numSamplesTensor.data(), numSamplesTensor.getSizeInBytes(),
                              &numSamplesValue, sizeof(uint64_t), ACL_MEMCPY_HOST_TO_DEVICE);
    ASCEND_THROW_IF_NOT_MSG(aclRet == ACL_ERROR_NONE, "Failed to copy numSamples to device!");
    
    AscendTensor<uint32_t, DIMS_2> newReverseGraphDevice(mem, {static_cast<int>(n), static_cast<int>(buildConfig.graphDegree)}, stream);
    AscendTensor<int, DIMS_1> newBackwardEdgeCountsDevice(mem, {static_cast<int>(n)}, stream);
    // 为新邻居候选图构建反向图
    ret = runAddReverseEdges(n, buildConfig.graphDegree, numSamplesTensor, newCandidatesDevice, newForwardEdgeCountsDevice,
                             newReverseGraphDevice, newBackwardEdgeCountsDevice);
    ASCEND_THROW_IF_NOT_MSG(ret == APP_ERR_OK, "runAddReverseEdges for new graph failed!");
    
    AscendTensor<uint32_t, DIMS_2> oldReverseGraphDevice(mem, {static_cast<int>(n), static_cast<int>(buildConfig.graphDegree)}, stream);
    AscendTensor<int, DIMS_1> oldBackwardEdgeCountsDevice(mem, {static_cast<int>(n)}, stream);
    // 为当前邻居图构建反向图
    ret = runAddReverseEdges(n, buildConfig.graphDegree, numSamplesTensor, currentGraphDevice, oldForwardEdgeCountsDevice,
                             oldReverseGraphDevice, oldBackwardEdgeCountsDevice);
    ASCEND_THROW_IF_NOT_MSG(ret == APP_ERR_OK, "runAddReverseEdges for old graph failed!");
    
    AscendTensor<uint32_t, DIMS_2> outputGraphDevice(mem, {static_cast<int>(n), static_cast<int>(buildConfig.graphDegree)}, stream);
    AscendTensor<float, DIMS_2> outputDistancesDevice(mem, {static_cast<int>(n), static_cast<int>(buildConfig.graphDegree)}, stream);
    
    // 执行本地连接算子，将新候选邻居与当前邻居图合并
    ret = runLocalJoinKernel(n, newCandidatesDevice, newReverseGraphDevice,
                             newForwardEdgeCountsDevice, newBackwardEdgeCountsDevice,
                             currentGraphDevice, oldReverseGraphDevice,
                             oldForwardEdgeCountsDevice, oldBackwardEdgeCountsDevice,
                             preprocessedDataDevice->data(), l2NormDataDevice->data(),
                             outputGraphDevice, outputDistancesDevice);
    ASCEND_THROW_IF_NOT_MSG(ret == APP_ERR_OK, "runLocalJoinKernel failed!");
    
    auto outputGraphHost = std::make_unique<uint32_t[]>(graphElements);
    aclRet = aclrtMemcpy(outputGraphHost.get(), graphElements * sizeof(uint32_t),
                         outputGraphDevice.data(), outputGraphDevice.getSizeInBytes(),
                         ACL_MEMCPY_DEVICE_TO_HOST);
    ASCEND_THROW_IF_NOT_MSG(aclRet == ACL_ERROR_NONE, "Failed to copy output graph from device!");
    
    auto outputDistancesHost = std::make_unique<float[]>(graphElements);
    aclRet = aclrtMemcpy(outputDistancesHost.get(), graphElements * sizeof(float),
                         outputDistancesDevice.data(), outputDistancesDevice.getSizeInBytes(),
                         ACL_MEMCPY_DEVICE_TO_HOST);
    ASCEND_THROW_IF_NOT_MSG(aclRet == ACL_ERROR_NONE, "Failed to copy distances from device!");
    // 对每个节点，根据距离排序并更新当前邻居图
    for (int64_t i = 0; i < n; ++i) {
        uint32_t* candidates = &outputGraphHost[i * buildConfig.graphDegree];
        float* distances = &outputDistancesHost[i * buildConfig.graphDegree];
        
        std::vector<int> indices(buildConfig.graphDegree);
        for (uint32_t j = 0; j < buildConfig.graphDegree; ++j) {
            indices[j] = j;
        }
        
        std::sort(indices.begin(), indices.end(),
                    [&](int a, int b) { return distances[a] < distances[b]; });
        
        for (uint32_t j = 0; j < buildConfig.graphDegree; ++j) {
            currentGraph[i * buildConfig.graphDegree + j] = candidates[indices[j]];
        }
    }

    return APP_ERR_OK;
}

APP_ERROR AscendIndexCagraImpl::executePruneAndFilter(int64_t n,
                                                      std::vector<uint64_t>& destNodes)
{
    auto streamPtr = pResources->getDefaultStream();
    auto stream = streamPtr->GetStream();
    auto &mem = pResources->getMemoryManager();
    
    size_t graphElements = static_cast<size_t>(n) * static_cast<size_t>(buildConfig.graphDegree);
    
    AscendTensor<uint64_t, DIMS_2> knnGraphDevice(mem, {static_cast<int>(n), static_cast<int>(buildConfig.graphDegree)}, stream);
    
    auto knnGraphHost = std::make_unique<uint64_t[]>(graphElements);
    for (size_t i = 0; i < graphElements; ++i) {
        knnGraphHost[i] = static_cast<uint64_t>(currentGraph.get()[i]);
    }
    
    auto aclRet = aclrtMemcpy(knnGraphDevice.data(), knnGraphDevice.getSizeInBytes(),
                              knnGraphHost.get(), graphElements * sizeof(uint64_t),
                              ACL_MEMCPY_HOST_TO_DEVICE);
    ASCEND_THROW_IF_NOT_MSG(aclRet == ACL_ERROR_NONE, "Failed to copy knn graph to device!");
    
    AscendTensor<uint8_t, DIMS_2> detourCountDevice(mem, {static_cast<int>(n), static_cast<int>(buildConfig.graphDegree)}, stream);
    AscendTensor<uint32_t, DIMS_1> numNoDetourEdgesDevice(mem, {static_cast<int>(n)}, stream);
    AscendTensor<uint64_t, DIMS_1> statsDevice(mem, {2}, stream);
    
    auto ret = runPrune(n, knnGraphDevice, detourCountDevice, numNoDetourEdgesDevice, statsDevice);
    ASCEND_THROW_IF_NOT_MSG(ret == APP_ERR_OK, "runPrune failed!");
    
    auto detourCountHost = std::make_unique<uint8_t[]>(graphElements);
    aclRet = aclrtMemcpy(detourCountHost.get(), graphElements * sizeof(uint8_t),
                         detourCountDevice.data(), detourCountDevice.getSizeInBytes(),
                         ACL_MEMCPY_DEVICE_TO_HOST);
    ASCEND_THROW_IF_NOT_MSG(aclRet == ACL_ERROR_NONE, "Failed to copy detour count from device!");
    
    auto numNoDetourEdgesHost = std::make_unique<uint32_t[]>(n);
    aclRet = aclrtMemcpy(numNoDetourEdgesHost.get(), n * sizeof(uint32_t),
                         numNoDetourEdgesDevice.data(), numNoDetourEdgesDevice.getSizeInBytes(),
                         ACL_MEMCPY_DEVICE_TO_HOST);
    ASCEND_THROW_IF_NOT_MSG(aclRet == ACL_ERROR_NONE, "Failed to copy numNoDetourEdges from device!");
    
    auto filteredGraphHost = std::make_unique<uint64_t[]>(graphElements);
    auto filteredGraphCountHost = std::make_unique<uint32_t[]>(n);
    
    for (int64_t i = 0; i < n; ++i) {
        filteredGraphCountHost[i] = 0;
    }
    
    size_t totalFilteredEdges = 0;
    for (int64_t i = 0; i < n; ++i) {
        uint32_t filteredCount = 0;
        for (uint32_t j = 0; j < buildConfig.graphDegree; ++j) {
            size_t idx = i * buildConfig.graphDegree + j;
            if (detourCountHost[idx] == 0) {
                uint64_t neighbor = knnGraphHost[idx];
                filteredGraphHost[i * buildConfig.graphDegree + filteredCount] = neighbor;
                filteredCount++;
                destNodes.push_back(neighbor);
            }
        }
        filteredGraphCountHost[i] = filteredCount;
        totalFilteredEdges += filteredCount;
    }
    
    return APP_ERR_OK;
}

APP_ERROR AscendIndexCagraImpl::executeReverseGraphGeneration(int64_t n,
                                                              const std::vector<uint64_t>& destNodes,
                                                              uint32_t* graph)
{
    auto streamPtr = pResources->getDefaultStream();
    auto stream = streamPtr->GetStream();
    auto &mem = pResources->getMemoryManager();
    
    size_t graphElements = static_cast<size_t>(n) * static_cast<size_t>(buildConfig.graphDegree);
    
    AscendTensor<uint64_t, DIMS_1> destNodesDevice(mem, {static_cast<int>(destNodes.size())}, stream);
    
    if (destNodes.size() > 0) {
        auto aclRet = aclrtMemcpy(destNodesDevice.data(), destNodesDevice.getSizeInBytes(),
                                  destNodes.data(), destNodes.size() * sizeof(uint64_t),
                                  ACL_MEMCPY_HOST_TO_DEVICE);
        ASCEND_THROW_IF_NOT_MSG(aclRet == ACL_ERROR_NONE, "Failed to copy dest nodes to device!");
    }
    
    AscendTensor<uint64_t, DIMS_2> revGraphDevice(mem, {static_cast<int>(n), static_cast<int>(buildConfig.graphDegree)}, stream);
    AscendTensor<uint32_t, DIMS_1> revGraphCountDevice(mem, {static_cast<int>(n)}, stream);
    
    auto ret = runMakeRevGraph(n, destNodesDevice, revGraphDevice, revGraphCountDevice);
    ASCEND_THROW_IF_NOT_MSG(ret == APP_ERR_OK, "runMakeRevGraph failed!");
    
    if (destNodes.size() > 0) {
        auto revGraphHost = std::make_unique<uint64_t[]>(graphElements);
        auto aclRet = aclrtMemcpy(revGraphHost.get(), graphElements * sizeof(uint64_t),
                                  revGraphDevice.data(), revGraphDevice.getSizeInBytes(),
                                  ACL_MEMCPY_DEVICE_TO_HOST);
        if (aclRet == ACL_ERROR_NONE) {
            for (size_t i = 0; i < graphElements; ++i) {
                graph[i] = static_cast<uint32_t>(revGraphHost[i]);
            }
        }
    }
    
    return APP_ERR_OK;
}

APP_ERROR AscendIndexCagraImpl::buildGraphImpl(int64_t n, const float* data, uint32_t* graph)
{
    APP_LOG_INFO("Executing complete graph build pipeline\n");
    
    auto ret = prepareGraphBuild(n, data, graph);
    ASCEND_THROW_IF_NOT_MSG(ret == APP_ERR_OK, "prepareGraphBuild failed!");
    
    std::vector<int> oldForwardEdgeCounts(n);
    std::vector<int> newForwardEdgeCounts(n);
    
    ret = sampleOldNewNeighbors(n, currentGraph.get(), newCandidates.get(),
                                oldForwardEdgeCounts, newForwardEdgeCounts);
    ASCEND_THROW_IF_NOT_MSG(ret == APP_ERR_OK, "sampleOldNewNeighbors failed!");
    
    const int MAX_ITERATIONS = 10;
    for (int iteration = 0; iteration < MAX_ITERATIONS; ++iteration) {

        APP_LOG_INFO("[TIMING] === NN-Descent Iteration %d/%d START ===\n", iteration + 1, MAX_ITERATIONS);
        ret = executeNNdescentIteration(n, oldForwardEdgeCounts, newForwardEdgeCounts);
        ASCEND_THROW_IF_NOT_MSG(ret == APP_ERR_OK, "executeNNdescentIteration failed!");
        APP_LOG_INFO("[TIMING] === NN-Descent Iteration %d/%d END ===\n", iteration + 1, MAX_ITERATIONS);
        
        ret = sampleOldNewNeighbors(n, currentGraph.get(), newCandidates.get(),
                                    oldForwardEdgeCounts, newForwardEdgeCounts);
        ASCEND_THROW_IF_NOT_MSG(ret == APP_ERR_OK, "sampleOldNewNeighbors for next iteration failed!");
    }
    
    size_t graphElements = static_cast<size_t>(n) * static_cast<size_t>(buildConfig.graphDegree);
    ret = memcpy_s(graph, graphElements * sizeof(uint32_t), currentGraph.get(), graphElements * sizeof(uint32_t));
    APPERR_RETURN_IF_NOT_LOG(ret == 0, APP_ERR_INNER_ERROR, "memcpy_s failed in buildGraphImpl");
    
    std::vector<uint64_t> destNodes;
    
    ret = executePruneAndFilter(n, destNodes);
    ASCEND_THROW_IF_NOT_MSG(ret == APP_ERR_OK, "executePruneAndFilter failed!");
    
    ret = executeReverseGraphGeneration(n, destNodes, graph);
    ASCEND_THROW_IF_NOT_MSG(ret == APP_ERR_OK, "executeReverseGraphGeneration failed!");
    
    currentGraph.reset();
    newCandidates.reset();
    
    APP_LOG_INFO("NN-Descent iteration completed, %d iterations, graph size: %zu\n", MAX_ITERATIONS, graphElements);
    
    return APP_ERR_OK;
}

APP_ERROR AscendIndexCagraImpl::saveGraphToFile(const std::string& filePath, const uint32_t* graph, size_t numElements)
{
    APP_LOG_INFO("Saving graph to file: %s, elements: %zu\n", filePath.c_str(), numElements);
    
    FILE* fp = fopen(filePath.c_str(), "wb");
    APPERR_RETURN_IF_FMT(fp == nullptr, APP_ERR_INNER_ERROR,
                        "Failed to open file for writing: %s", filePath.c_str());
    
    size_t bytesToWrite = numElements * sizeof(uint32_t);
    size_t written = fwrite(graph, sizeof(uint32_t), numElements, fp);
    
    if (written != numElements) {
        fclose(fp);
        APPERR_RETURN_IF_FMT(true, APP_ERR_INNER_ERROR,
                             "Failed to write complete graph data to file: %s, expected %zu elements, wrote %zu",
                             filePath.c_str(), numElements, written);
    }
    
    int ret = fclose(fp);
    ASCEND_THROW_IF_NOT_MSG(ret == 0, "Failed to close file!");
    
    APP_LOG_INFO("Graph saved successfully to %s, size: %zu bytes\n", filePath.c_str(), bytesToWrite);
    
    return APP_ERR_OK;
}

} // namespace ascend
} // namespace faiss