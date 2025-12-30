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


#ifndef ASCENDHOST_RPC_LOCAL_SESSION_H
#define ASCENDHOST_RPC_LOCAL_SESSION_H

#include <unordered_map>

#include <ascenddaemon/impl/Index.h>

namespace faiss {
namespace ascendSearch {
struct RpcLocalSession {
    explicit RpcLocalSession(int devId = 0): deviceId(devId) {}

    ::ascendSearch::Index *GetIndex(int indexId) const
    {
        if (indices.find(indexId) == indices.end()) {
            return nullptr;
        }
        return indices[indexId];
    }

    int AddIndex(::ascendSearch::Index *index) const
    {
        int indexId = -1;
        for (auto it = indices.begin(); it != indices.end(); ++it) {
            if (it->first > indexId) {
                indexId = it->first;
            }
        }
        indexId += 1;
        indices[indexId] = index;
        return indexId;
    }

    int deviceId;
    static std::unordered_map<int, ::ascendSearch::Index *> indices;
};
} // namespace ascendSearch
} // namespace faiss
#endif
