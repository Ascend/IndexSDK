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


#ifndef HEAP_SORT_FUZZY_INCLUDED
#define HEAP_SORT_FUZZY_INCLUDED

#include <tuple>
#include <vector>
#include <arm_fp16.h>
#include "npu/common/AscendTensor.h"
#include "CircularQueue.h"

namespace ascendSearchacc {
using idx_t = uint64_t;

template <typename D, typename I, typename __Compare>
inline bool compareDistAndId(const D d1, const I i1, const D d2, const I i2, __Compare compare)
{
    return compare(d1, d2) || (d1 == d2 && compare(i1, i2));
}

template <typename D, typename I, typename __Compare>
void pushHeapFuzzy(const size_t size, D *heapDist, I *heapId, const D pushDist, const I pushId, __Compare compare,
                   CircularQueue<D, I> &bufferPopped)
{
    size_t i = 1;
    size_t leftChild;
    size_t rightChild;

    bufferPopped.push(heapDist[0], heapId[0]);

    heapDist--;
    heapId--;
    while (1) {
        leftChild = i << 1;
        rightChild = leftChild + 1;
        if (leftChild > size) {
            break;
        }

        if (rightChild == (size + 1) || compareDistAndId(heapDist[leftChild], heapId[leftChild], heapDist[rightChild],
                                                         heapId[rightChild], compare)) {
            if (compareDistAndId(pushDist, pushId, heapDist[leftChild], heapId[leftChild], compare)) {
                break;
            }

            heapDist[i] = heapDist[leftChild];
            heapId[i] = heapId[leftChild];
            i = leftChild;
        } else {
            if (compareDistAndId(pushDist, pushId, heapDist[rightChild], heapId[rightChild], compare)) {
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

template <typename D, typename __Compare>
void popHeapFuzzy(size_t k, D *heapDist, idx_t *heapId, __Compare compare)
{
    heapDist--; /* Use 1-based indexing for easier node->child translation */
    heapId--;
    D val = heapDist[k];
    idx_t id = heapId[k];
    size_t i = 1;
    size_t leftChild;
    size_t rightChild;
    while (1) {
        leftChild = i << 1;
        rightChild = leftChild + 1;
        if (leftChild > k) {
            break;
        }

        if (rightChild == k + 1 || compareDistAndId(heapDist[leftChild], heapId[leftChild], heapDist[rightChild],
                                                    heapId[rightChild], compare)) {
            if (compareDistAndId(val, id, heapDist[leftChild], heapId[leftChild], compare)) {
                break;
            }
            heapDist[i] = heapDist[leftChild];
            heapId[i] = heapId[leftChild];
            i = leftChild;
        } else {
            if (compareDistAndId(val, id, heapDist[rightChild], heapId[rightChild], compare)) {
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
}  // namespace ascendSearchacc

#endif
