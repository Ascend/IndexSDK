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


#ifndef ASCEND_FAISS_RPC_CONTEXT_H
#define ASCEND_FAISS_RPC_CONTEXT_H

#include <ascendsearch/ascend/rpc/AscendRpcCommon.h>
#include <vector>
namespace faiss {
namespace ascendSearch {
// create/destroy rpc context
RpcError RpcCreateContext(int deviceId, rpcContext *ctx);
RpcError RpcDestroyContext(rpcContext ctx);

// index search
RpcError RpcIndexSearch(rpcContext ctx, int indexId, int n, int dim, int k,
    const uint16_t *query, uint16_t *distance, ascend_idx_t *label);
RpcError RpcIndexSearchFilter(rpcContext ctx, int indexId, int n, int dim, int k,
    const uint16_t *query, uint16_t *distance, ascend_idx_t *label, int maskSize, const uint8_t* mask);
RpcError RpcIndexSearchFilter(rpcContext ctx, int indexId, int64_t n, int dim, int64_t k,
    const uint16_t *query, uint16_t *distance, ascend_idx_t *label, uint64_t filtersSize, const uint32_t* filters);

// multi index search
RpcError RpcMultiIndexSearch(rpcContext ctx, int n, int dim, int k, const uint16_t *query,
    uint16_t *distance, ascend_idx_t *label, std::vector<int> indexIds);

RpcError RpcMultiIndexSearchFilter(rpcContext ctx, int n, int dim, int k, const uint16_t *query,
    uint16_t *distance, ascend_idx_t *label, uint32_t filtersSize, const uint32_t* filters, std::vector<int> indexIds);

RpcError RpcMultiIndexSearchFilter(rpcContext ctx, int n, int dim, int k, const uint16_t *query,
    uint16_t *distance, ascend_idx_t *label, uint32_t filtersSize, const uint32_t** filters, std::vector<int> indexIds);

// reset index
RpcError RpcIndexReset(rpcContext ctx, int indexId);

// reserve or reclaim device memory for database
RpcError RpcIndexReserveMemory(rpcContext ctx, int &indexId, uint32_t numVec);
RpcError RpcIndexReclaimMemory(rpcContext ctx, int &indexId, uint32_t &sizeMem);
} // namespace rpc
} // namespace ascendSearch
#endif
