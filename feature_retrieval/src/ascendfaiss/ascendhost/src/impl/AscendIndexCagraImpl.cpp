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
#include <faiss/impl/FaissAssert.h>

#include "ascend/utils/fp16.h"
#include "ascenddaemon/utils/Limits.h"
#include "ascenddaemon/utils/Random.h"
#include "common/threadpool/AscendThreadPool.h"
#include "common/utils/CommonUtils.h"
#include "common/utils/LogUtils.h"
#include "common/utils/SocUtils.h"
using namespace ascend;
namespace faiss {
namespace ascend {
namespace {
const int64_t CAGRA_MAX_MEM = 0x100000000; // 0x100000000 mean 4096MB
const int MAX_GRAPH_DEGREE = 256;
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

} // namespace ascend
} // namespace faiss