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

 
#ifndef HEAP_SORT_INCLUDED
#define HEAP_SORT_INCLUDED

#include <tuple>
#include "ascenddaemon/utils/AscendTensor.h"

namespace ascend {
    template<typename D, typename I, typename __Compare>
    void pushHeap(const size_t size, D* heapDist, I* heapId,
                  const D pushDist, const I pushId, __Compare compare)
    {
        size_t i = 1;
        size_t leftChild;
        size_t rightChild;

        heapDist--;
        heapId--;
        while (1) {
            leftChild = i << 1;
            rightChild = leftChild + 1;
            if (leftChild > size) {
                break;
            }

            if (rightChild == (size + 1) || compare(heapDist[leftChild], heapDist[rightChild])) {
                if (compare(pushDist, heapDist[leftChild])) {
                    break;
                }

                heapDist[i] = heapDist[leftChild];
                heapId[i] = heapId[leftChild];
                i = leftChild;
            } else {
                if (compare(pushDist, heapDist[rightChild])) {
                    break;
                }

                heapDist[i] = heapDist[rightChild];
                heapId[i] = heapId[rightChild];
                i = rightChild;
            }
        }

        heapDist[i] = pushDist;
        heapId[i] = pushId;
    }

    template<typename DP, typename __Compare, typename IDX>
    void popHeap(size_t k, DP heapDist, IDX* heapId, __Compare compare)
    {
        heapDist--; /* Use 1-based indexing for easier node->child translation */
        heapId--;
        auto val = heapDist[k];
        size_t i = 1;
        size_t leftChild;
        size_t rightChild;
        while (1) {
            leftChild = i << 1;
            rightChild = leftChild + 1;
            if (leftChild > k) {
                break;
            }

            if (rightChild == k + 1 || compare(heapDist[leftChild], heapDist[rightChild])) {
                if (compare(val, heapDist[leftChild])) {
                    break;
                }
                heapDist[i] = heapDist[leftChild];
                heapId[i] = heapId[leftChild];
                i = leftChild;
            } else {
                if (compare(val, heapDist[rightChild])) {
                    break;
                }
                heapDist[i] = heapDist[rightChild];
                heapId[i] = heapId[rightChild];
                i = rightChild;
            }
        }
        heapDist[i] = heapDist[k];
        heapId[i] = heapId[k];
    }
}  // namespace ascend

#endif
