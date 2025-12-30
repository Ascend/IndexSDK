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


//
// Created by mxIndex team on 2023/8/2.
//

#ifndef FEATURERETRIEVAL_SRC_ASCENDFAISS_ASCEND_INDEXIMPLBASE_H
#define FEATURERETRIEVAL_SRC_ASCENDFAISS_ASCEND_INDEXIMPLBASE_H


#include <faiss/Index.h>

#include "AscendIndex.h"
#include "common/threadpool/AscendThreadPool.h"


namespace ascend {
class Index;
}
namespace faiss {
namespace ascendSearch {
class AscendIndexIVFSPSQ;
}
namespace ascend {
class IndexImplBase {
public:
    IndexImplBase() = default;
    virtual ~IndexImplBase() = default;
    // query index label from idxDeviceMap
    virtual faiss::idx_t GetIdxFromDeviceMap(int deviceId, int idxId) const = 0;
    // query AscendThreadPool from impl
    virtual const std::shared_ptr<AscendThreadPool> GetPool() const = 0;
    // check the config parameters of index with another index
    virtual void CheckIndexParams(IndexImplBase &index, bool checkFilterable = false) const = 0;
    // query IVFSPSQPtr from IVFSP impl
    virtual faiss::ascendSearch::AscendIndexIVFSPSQ *GetIVFSPSQPtr() const = 0;
    // only return Index* or IndexInt8*
    virtual void* GetActualIndex(int deviceId, bool isNeedSetDevice = true) const = 0;
};

} // namespace ascend
} // namespace faiss
#endif // FEATURERETRIEVAL_SRC_ASCENDFAISS_ASCEND_INDEXIMPLBASE_H
