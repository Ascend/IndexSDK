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


#ifndef ASCEND_FAISS_RPC_INDEX_IVF_SP_SQ_H
#define ASCEND_FAISS_RPC_INDEX_IVF_SP_SQ_H

#include <vector>

#include <faiss/IndexScalarQuantizer.h>
#include <faiss/MetaIndexes.h>
#include <ascendsearch/ascend/rpc/AscendRpcCommon.h>

namespace faiss {
namespace ascendSearch {
struct IndexIVFSPSQParameter {
    IndexIVFSPSQParameter(int d, int dim2, int k, faiss::ScalarQuantizer::QuantizerType quantizerType,
                     faiss::MetricType metricType, int64_t resource, bool slim, bool filterable,
                     int nlist, bool encodeResidual, int handleBatch, int nprobe, int searchListSize)
        : dim(d), dim2(dim2), k(k), qtype(quantizerType), metric(metricType),
        resourceSize(resource), slim(slim), filterable(filterable),
        nlist(nlist), encodeResidual(encodeResidual), handleBatch(handleBatch),
        nprobe(nprobe), searchListSize(searchListSize)
    {}

    int dim;
    int dim2;
    int k;
    faiss::ScalarQuantizer::QuantizerType qtype;
    faiss::MetricType metric;
    int64_t resourceSize;
    bool slim;
    bool filterable;
    int nlist;
    bool encodeResidual;
    int handleBatch;
    int nprobe;
    int searchListSize;
};

struct RpcIndexCodeBookTrainerConfig {
    RpcIndexCodeBookTrainerConfig() {}

    int numIter = 1;
    int device = 0;
    float ratio = 1.0;
    int batchSize = 32768;
    int codeNum = 32768;
    bool verbose = true;
    std::string codeBookOutputDir = "";
    std::string learnDataPath = "";
    const float *memLearnData = nullptr;
    size_t memLearnDataSize = 0;
};

// create spsq index
RpcError RpcCreateIndexIVFSPSQ(rpcContext ctx, int &indexId, const IndexIVFSPSQParameter &parameter);

// update trained value to device
RpcError RpcIndexIVFSPSQUpdateTrainedValue(rpcContext ctx, int indexId, int dim, uint16_t *vmin,
    uint16_t *vdiff, bool isIvfSQ = false);

RpcError RpcIndexIVFSPSQAddWithIds(rpcContext ctx, int indexId, int n,
    int dim, int listId, const uint8_t *data, const ascend_idx_t *ids, const float *precomputedVal,
    bool useNPU);

RpcError RpcIndexIVFSPSQAddCodeBook(rpcContext ctx, int indexId, int n, int dim,
    int dim2, const uint16_t *data, idx_t *offset);

RpcError RpcIndexIVFSPSQAddCodeBook(rpcContext ctx, int indexId, rpcContext ctxLoaded, int indexIdLoaded);

RpcError RpcIndexIVFSPSQUpdateNprobe(rpcContext ctx, int indexId, int nprobe);
// get dataset size
RpcError RpcIndexIVFSPSQGetBaseSize(rpcContext ctx, int indexId, uint32_t &size);

RpcError RpcIndexIVFSPSQAddFinish(rpcContext ctx, int indexId);

RpcError RpcIndexIVFSPSQTrainCodeBook(rpcContext ctx, int indexId,
                                      const RpcIndexCodeBookTrainerConfig &codeBookTrainerConfig,
                                      float *codebookPtr = nullptr);

RpcError RpcIndexIVFSPSQGetCodeWord(rpcContext ctx, int indexId, int n, int dim, const float *feature,
    uint16_t *codeWord, idx_t* labels);

RpcError RpcIndexIVFSPSQLoadAllData(rpcContext ctx, int indexId,
    const char *dataPath, float* codebookPtr, float* spsqPtr);

RpcError RpcIndexIVFSPSQLoadAllData(rpcContext ctx, int indexId,
    rpcContext ctxLoaded, int indexIdLoaded, const char *dataPath);

RpcError RpcIndexIVFSPSQLoadAllData(rpcContext ctx, int indexId,
    const uint8_t* data, size_t dataLen, float* codebookPtr, float* spsqPtr);

RpcError RpcIndexIVFSPSQLoadAllData(rpcContext ctx, int indexId,
    rpcContext ctxLoaded, int indexIdLoaded, const uint8_t* data, size_t dataLen);

RpcError RpcIndexIVFSPSQLoadCodeBook(rpcContext ctx, int indexId,
    const uint8_t* data, size_t dataLen, float *codebookPtr);

RpcError RpcIndexIVFSPSQSaveAllData(rpcContext ctx, int indexId,
    const char *dataPath, float* codebookPtr, float* spsqPtr);

RpcError RpcIndexIVFSPSQSaveAllData(rpcContext ctx, int indexId,
    uint8_t* &data, size_t &dataLen, float* codebookPtr, float* spsqPtr);

RpcError RpcIndexIVFSPSQSaveCodeBook(rpcContext ctx, int indexId,
    uint8_t* &data, size_t &dataLen, float *codebookPtr);

RpcError RpcIndexIVFSPSQRemoveIds(rpcContext ctx, int indexId, int n,
    ascend_idx_t *ids, uint32_t *numRemoved);

RpcError RpcIndexIVFSPSQRemoveRangeIds(rpcContext ctx, int indexId,
    ascend_idx_t min, ascend_idx_t max, uint32_t *numRemoved);

RpcError RpcIndexIVFSPSQGetListLength(rpcContext ctx, int indexId, int listId, uint32_t &len);

RpcError RpcIndexGetAddFinish(rpcContext ctx, int indexId, bool &addFinish);

} // namespace ascendSearch
} // namespace faiss
#endif
