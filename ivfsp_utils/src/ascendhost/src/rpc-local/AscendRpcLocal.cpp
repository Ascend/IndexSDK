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


#include <ascendsearch/ascend/rpc/AscendRpc.h>

#include "ascenddaemon/impl/AuxIndexStructures.h"
#include "common/utils/SocUtils.h"
#include "rpc-local/RpcLocalSession.h"
#include "ascenddaemon/impl/Index.h"

using namespace ::ascendSearch;

namespace faiss {
namespace ascendSearch {
RpcError RpcCreateContext(int deviceId, rpcContext *ctx)
{
    ASCEND_THROW_IF_NOT(ctx);
    ASCEND_THROW_IF_NOT(deviceId >= 0);
    uint32_t devCount = SocUtils::GetInstance().GetDeviceCount();
    APPERR_RETURN_IF_NOT_FMT(static_cast<uint32_t>(deviceId) < devCount, RPC_ERROR_ERROR,
                             "Device %d is invalid, total device %u", deviceId, devCount);

    auto *session = new (std::nothrow) RpcLocalSession(deviceId);
    APPERR_RETURN_IF_NOT_FMT(
        session, RPC_ERROR_ERROR, "Create RpcLocalSession failed, deviceId=%d\n", deviceId);
    *ctx = static_cast<rpcContext>(session);
    return RPC_ERROR_NONE;
}

std::unordered_map<int, ::ascendSearch::Index *> RpcLocalSession::indices;

RpcError RpcDestroyContext(rpcContext ctx)
{
    auto *session = static_cast<RpcLocalSession *>(ctx);
    delete session;

    return RPC_ERROR_NONE;
}

RpcError RpcDestroyIndex(rpcContext ctx, int indexId)
{
    auto *session = static_cast<RpcLocalSession *>(ctx);
    auto *index = session->GetIndex(indexId);
    APPERR_RETURN_IF_NOT_FMT(index, RPC_ERROR_ERROR,
                             "Invalid index id: %d\n", indexId);
    session->indices.erase(indexId);
    delete index;

    return RPC_ERROR_NONE;
}

RpcError RpcIndexSearch(rpcContext ctx, int indexId, int n, int dim, int k, const uint16_t *query, uint16_t *distance,
    ascend_idx_t *label)
{
    VALUE_UNUSED(dim);
    auto *session = static_cast<RpcLocalSession *>(ctx);
    ACL_REQUIRE_OK_RET_CODE(aclrtSetDevice(session->deviceId), RPC_ERROR_ERROR);

    auto *index = session->GetIndex(indexId);
    APPERR_RETURN_IF_NOT_FMT(index, RPC_ERROR_ERROR,
                             "Invalid index id: %d\n", indexId);

    AscendTensor<uint16_t, DIMS_2> tensorDevQueries({ n, dim });
    auto ret = aclrtMemcpy(tensorDevQueries.data(), tensorDevQueries.getSizeInBytes(),
                           query, n * dim * sizeof(uint16_t), ACL_MEMCPY_HOST_TO_DEVICE);
    APPERR_RETURN_IF_NOT_FMT(
        ret == ACL_SUCCESS, RPC_ERROR_ERROR, "aclrtMemcpy error %d", static_cast<int>(ret));

    ret = index->search(n, tensorDevQueries.data(), k, distance, static_cast<::ascendSearch::idx_t *>(label));
    APPERR_RETURN_IF_NOT_FMT(ret == APP_ERR_OK, RPC_ERROR_ERROR,
                             "Failed to search index id: %d\n", indexId);

    return RPC_ERROR_NONE;
}

RpcError RpcIndexSearchFilter(rpcContext ctx, int indexId, int n, int dim, int k,
    const uint16_t *query, uint16_t *distance, ascend_idx_t *label, int maskSize, const uint8_t* mask)
{
    VALUE_UNUSED(dim);
    auto *session = static_cast<RpcLocalSession *>(ctx);
    ACL_REQUIRE_OK_RET_CODE(aclrtSetDevice(session->deviceId), RPC_ERROR_ERROR);

    auto *index = session->GetIndex(indexId);
    APPERR_RETURN_IF_NOT_FMT(index, RPC_ERROR_ERROR,
                             "Invalid index id: %d\n", indexId);

    AscendTensor<uint8_t, DIMS_2> maskDevice({ n, maskSize });
    auto ret = aclrtMemcpy(maskDevice.data(), maskDevice.getSizeInBytes(),
                           mask, maskSize * sizeof(uint8_t), ACL_MEMCPY_HOST_TO_DEVICE);
    APPERR_RETURN_IF_NOT_FMT(
        ret == ACL_SUCCESS, RPC_ERROR_ERROR, "Mem operator error %d", static_cast<int>(ret));
    AscendTensor<uint16_t, DIMS_2> tensorDevQueries({ n, dim });
    ret = aclrtMemcpy(tensorDevQueries.data(), tensorDevQueries.getSizeInBytes(),
                      query, n * dim * sizeof(uint16_t), ACL_MEMCPY_HOST_TO_DEVICE);
    APPERR_RETURN_IF_NOT_FMT(
        ret == ACL_SUCCESS, RPC_ERROR_ERROR, "aclrtMemcpy error %d", static_cast<int>(ret));
    ret = index->searchFilter(n, tensorDevQueries.data(), k, distance,
        static_cast<::ascendSearch::idx_t *>(label), maskDevice.data());
    APPERR_RETURN_IF_NOT_FMT(ret == APP_ERR_OK, RPC_ERROR_ERROR,
                             "Failed to searchfilter index id: %d\n", indexId);

    return RPC_ERROR_NONE;
}

RpcError RpcIndexSearchFilter(rpcContext ctx, int indexId, int64_t n, int dim, int64_t k,
    const uint16_t *query, uint16_t *distance, ascend_idx_t *label, uint64_t filtersSize, const uint32_t* filters)
{
    VALUE_UNUSED(dim);
    auto *session = static_cast<RpcLocalSession *>(ctx);
    ACL_REQUIRE_OK_RET_CODE(aclrtSetDevice(session->deviceId), RPC_ERROR_ERROR);

    auto *index = session->GetIndex(indexId);
    APPERR_RETURN_IF_NOT_FMT(index, RPC_ERROR_ERROR,
                             "Invalid index id: %d\n", indexId);
    AscendTensor<uint16_t, DIMS_2> tensorDevQueries({ static_cast<int>(n), dim });
    auto ret = aclrtMemcpy(tensorDevQueries.data(), tensorDevQueries.getSizeInBytes(),
                           query, n * dim * sizeof(uint16_t), ACL_MEMCPY_HOST_TO_DEVICE);
    APPERR_RETURN_IF_NOT_FMT(
        ret == ACL_SUCCESS, RPC_ERROR_ERROR, "aclrtMemcpy error %d", static_cast<int>(ret));
    ret = index->searchFilter(n, tensorDevQueries.data(), k, distance,
        static_cast<::ascendSearch::idx_t *>(label), filtersSize, const_cast<uint32_t *>(filters));
    APPERR_RETURN_IF_NOT_FMT(ret == APP_ERR_OK, RPC_ERROR_ERROR,
                             "Failed to searchfilter index id: %d\n", indexId);

    return RPC_ERROR_NONE;
}

RpcError RpcMultiIndexSearch(rpcContext ctx, int n, int dim, int k, const uint16_t *query, uint16_t *distance,
    ascend_idx_t *label, std::vector<int> indexIds)
{
    auto *session = static_cast<RpcLocalSession *>(ctx);
    ACL_REQUIRE_OK_RET_CODE(aclrtSetDevice(session->deviceId), RPC_ERROR_ERROR);
    int32_t indexNum = static_cast<int64_t>(indexIds.size());
    std::vector<::ascendSearch::Index *> indexes(indexNum);
    for (int32_t i = 0; i < indexNum; ++i) {
        indexes[i] = session->GetIndex(indexIds[i]);
        APPERR_RETURN_IF_NOT_FMT(indexes[i], RPC_ERROR_ERROR, "Invalid index id: %d\n", indexIds[i]);
    }

    AscendTensor<uint16_t, DIMS_2> tensorDevQueries({ n, dim });
    auto ret = aclrtMemcpy(tensorDevQueries.data(), tensorDevQueries.getSizeInBytes(), query,
        n * dim * sizeof(uint16_t), ACL_MEMCPY_HOST_TO_DEVICE);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, RPC_ERROR_ERROR, "aclrtMemcpy error %d", static_cast<int>(ret));

    ret = indexes[0]->search(indexes, n, tensorDevQueries.data(), k, distance,
        static_cast<::ascendSearch::idx_t *>(label));
    APPERR_RETURN_IF_NOT_FMT(
        ret == APP_ERR_OK, RPC_ERROR_ERROR, "Failed to MultiSearch index id: %d\n", indexIds[0]);

    return RPC_ERROR_NONE;
}

// To ensure this method works properly, make sure all the IDs in indexIds are from the same NPU device.

RpcError RpcMultiIndexSearchFilter(rpcContext ctx, int n, int dim, int k, const uint16_t *query,
    uint16_t *distance, ascend_idx_t *label, uint32_t filtersSize, const uint32_t* filters, std::vector<int> indexIds)
{
    auto *currsession = static_cast<RpcLocalSession *>(ctx);
    ACL_REQUIRE_OK_RET_CODE(aclrtSetDevice(currsession->deviceId), RPC_ERROR_ERROR);
    int32_t indexNum = static_cast<int32_t>(indexIds.size());
    std::vector<::ascendSearch::Index *> indexes(indexNum);
    for (int32_t i = 0; i < indexNum; ++i) {
        indexes[i] = currsession->GetIndex(indexIds[i]);
        APPERR_RETURN_IF_NOT_FMT(indexes[i], RPC_ERROR_ERROR, "Invalid index id: %d\n", indexIds[i]);
    }

    AscendTensor<uint16_t, DIMS_2> deviceQuery({ n, dim });
    auto ret = aclrtMemcpy(deviceQuery.data(), deviceQuery.getSizeInBytes(), query,
        n * dim * sizeof(uint16_t), ACL_MEMCPY_HOST_TO_DEVICE);
    APPERR_RETURN_IF_NOT_FMT(
        ret == ACL_SUCCESS, RPC_ERROR_ERROR, "aclrtMemcpy error %d", static_cast<int>(ret));

    ret = indexes[0]->searchFilter(indexes, n, deviceQuery.data(), k, distance,
        static_cast<::ascendSearch::idx_t *>(label), filtersSize, const_cast<uint32_t *>(filters));
    APPERR_RETURN_IF_NOT_FMT(
        ret == APP_ERR_OK, RPC_ERROR_ERROR, "Failed to MultiSearch index id: %d\n", indexIds[0]);

    return RPC_ERROR_NONE;
}

RpcError RpcMultiIndexSearchFilter(rpcContext ctx, int n, int dim, int k, const uint16_t *query,
    uint16_t *distance, ascend_idx_t *label, uint32_t filtersSize, const uint32_t** filters, std::vector<int> indexIds)
{
    auto *session = static_cast<RpcLocalSession *>(ctx);
    ACL_REQUIRE_OK_RET_CODE(aclrtSetDevice(session->deviceId), RPC_ERROR_ERROR);
    int32_t indexNum = static_cast<int32_t>(indexIds.size());
    std::vector<::ascendSearch::Index *> indexes(indexNum);
    for (int32_t i = 0; i < indexNum; ++i) {
        indexes[i] = session->GetIndex(indexIds[i]);
        APPERR_RETURN_IF_NOT_FMT(indexes[i], RPC_ERROR_ERROR, "Invalid index id: %d\n", indexIds[i]);
    }

    AscendTensor<uint16_t, DIMS_2> tensorDevQueries({ n, dim });
    auto ret = aclrtMemcpy(tensorDevQueries.data(), tensorDevQueries.getSizeInBytes(), query,
        n * dim * sizeof(uint16_t), ACL_MEMCPY_HOST_TO_DEVICE);
    APPERR_RETURN_IF_NOT_FMT(
        ret == ACL_SUCCESS, RPC_ERROR_ERROR, "aclrtMemcpy error %d", static_cast<int>(ret));

    ret = indexes[0]->searchFilter(indexes, n, tensorDevQueries.data(), k, distance,
        static_cast<::ascendSearch::idx_t *>(label), filtersSize, const_cast<uint32_t **>(filters));
    APPERR_RETURN_IF_NOT_FMT(
        ret == APP_ERR_OK, RPC_ERROR_ERROR, "Failed to MultiSearch index id: %d\n", indexIds[0]);

    return RPC_ERROR_NONE;
}

RpcError RpcIndexReset(rpcContext ctx, int indexId)
{
    APP_LOG_INFO("Reset index %d\n", indexId);
    auto *session = static_cast<RpcLocalSession *>(ctx);
    ACL_REQUIRE_OK_RET_CODE(aclrtSetDevice(session->deviceId), RPC_ERROR_ERROR);

    auto *index = session->GetIndex(indexId);
    APPERR_RETURN_IF_NOT_FMT(index, RPC_ERROR_ERROR,
                             "Invalid index id: %d\n", indexId);
    auto ret = index->reset();
    APPERR_RETURN_IF_NOT_FMT(ret == APP_ERR_OK, RPC_ERROR_ERROR,
                             "Failed to reset index id: %d\n", indexId);

    return RPC_ERROR_NONE;
}

RpcError RpcIndexReserveMemory(rpcContext ctx, int &indexId, uint32_t numVec)
{
    APP_LOG_INFO("index %d reset\n", indexId);
    auto *session = static_cast<RpcLocalSession *>(ctx);
    ACL_REQUIRE_OK_RET_CODE(aclrtSetDevice(session->deviceId), RPC_ERROR_ERROR);

    auto *index = session->GetIndex(indexId);
    APPERR_RETURN_IF_NOT_FMT(index, RPC_ERROR_ERROR,
                             "Invalid index id: %d\n", indexId);
    auto ret = index->reserveMemory(numVec);
    APPERR_RETURN_IF_NOT_FMT(ret == APP_ERR_OK, RPC_ERROR_ERROR,
                             "Failed to reserveMemory index id: %d\n", indexId);

    return RPC_ERROR_NONE;
}

RpcError RpcIndexReclaimMemory(rpcContext ctx, int &indexId, uint32_t &sizeMem)
{
    APP_LOG_INFO("index %d reset\n", indexId);
    auto *session = static_cast<RpcLocalSession *>(ctx);
    ACL_REQUIRE_OK_RET_CODE(aclrtSetDevice(session->deviceId), RPC_ERROR_ERROR);

    auto *index = session->GetIndex(indexId);
    APPERR_RETURN_IF_NOT_FMT(index, RPC_ERROR_ERROR,
                             "Invalid index id: %d\n", indexId);
    sizeMem = index->reclaimMemory();

    return RPC_ERROR_NONE;
}
} // namespace ascendSearch
} // namespace faiss
