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


#ifndef ASCEND_FAISS_RPC_COMMON_H
#define ASCEND_FAISS_RPC_COMMON_H

#include <cstdint>

#include <faiss/Index.h>
#include <common/ErrorCode.h>
#include <common/utils/AscendAssert.h>
#include <common/utils/LogUtils.h>

namespace faiss {
namespace ascendSearch {
using rpcContext = void *;
using ascend_idx_t = uint64_t;

enum RpcError {
    RPC_ERROR_NONE = 0,
    RPC_ERROR_ERROR = 1,
};

RpcError RpcDestroyIndex(rpcContext ctx, int indexId);
} // namespace ascendSearch
} // namespace faiss
#endif
