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


#ifndef ASCEND_TASK_QUEUE_ITEMS_H
#define ASCEND_TASK_QUEUE_ITEMS_H

#include <vector>
#include <ascenddaemon/utils/AscendTensor.h>

namespace ascendSearch {
using idx_t = uint64_t;

namespace {
    const int EXTREME_LIST_SIZE = faiss::ascendSearch::SocUtils::GetInstance().GetExtremeListSize();
    const int HANDLE_BATCH = faiss::ascendSearch::SocUtils::GetInstance().GetHandleBatch();
    const int MAX_HANDLE_BATCH = 256;
    const int BURST_LENS = 64;
    // must be CUBE_ALIGN aligned
    const int SEARCH_LIST_SIZES = faiss::ascendSearch::SocUtils::GetInstance().GetSearchListSize();
}

// Batch distribution for AscendOp: Now only implmented for IVFSQIP
struct BatchQueueItem {
    BatchQueueItem() : dist(nullptr), ids(nullptr), executing(false),
        handleBatch(MAX_HANDLE_BATCH), searchListSize(SEARCH_LIST_SIZES) {}

    void BatchQueueItemCopyImpl(const BatchQueueItem &item)
    {
        dist = item.dist;
        handleBatch = item.handleBatch;
        for (int i = 0; i < handleBatch; ++i) {
            extremes[i] = item.extremes[i];
        }
        ids = item.ids;

        if (item.flags.data() != nullptr) {
            flags = std::move(item.flags);
        }

        if (item.sizes.data() != nullptr) {
            sizes = std::move(item.sizes);
        }

        executing = item.executing ? true : false;
        searchListSize = item.searchListSize;
    }

    BatchQueueItem(const BatchQueueItem &item)
    {
        BatchQueueItemCopyImpl(item);
    }

    BatchQueueItem &operator=(const BatchQueueItem &item)
    {
        BatchQueueItemCopyImpl(item);
        return *this;
    }

    void SetExecuting(float16_t *dist, float16_t *extreme, idx_t **id, uint16_t *flag, uint32_t *s)
    {
        ASCEND_THROW_IF_NOT(dist != nullptr);
        ASCEND_THROW_IF_NOT(extreme != nullptr);
        ASCEND_THROW_IF_NOT(id != nullptr);
        ASCEND_THROW_IF_NOT(flag != nullptr);
        ASCEND_THROW_IF_NOT(s != nullptr);

        int maxesLen = searchListSize / BURST_LENS * 2;
        for (int i = 0; i < handleBatch; ++i) {
            extremes[i] = extreme + i * maxesLen;
        }

        this->dist = dist;
        this->ids = id;
        AscendTensor<uint32_t, DIMS_1> tmpSizes(s, { handleBatch });
        sizes = std::move(tmpSizes);
        AscendTensor<uint16_t, DIMS_2> tmpFlags(flag, { handleBatch, FLAG_SIZE });

        flags = std::move(tmpFlags);
        executing = true;
    }

    inline bool IsExecuting()
    {
        return executing;
    }

    float16_t *dist;                                // distance result mem pointer

    float16_t *extremes[MAX_HANDLE_BATCH] = {};              // extreme distance result mem pointer

    idx_t **ids;                                    // ids mem pointer

    AscendTensor<uint32_t, DIMS_1> sizes;      // size to idicate how many code to calc

    AscendTensor<uint16_t, DIMS_2> flags;      // flag mem pointer for aicore

    std::atomic<bool> executing;                    // whether the item has beed added to stream for executing

    int handleBatch;

    int searchListSize;
};
}
#endif // ASCEND_TASK_QUEUE_ITEMS_H
