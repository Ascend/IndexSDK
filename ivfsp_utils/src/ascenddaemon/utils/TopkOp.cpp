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

#include <cstdint>
#include <ascenddaemon/utils/TopkOp.h>
#include "common/AscendFp16.h"

namespace ascendSearch {
    template<typename T, typename E, typename D, bool ASC, typename IDX>
    TopkOp<T, E, D, ASC, IDX>::TopkOp() {}

    template<typename T, typename E, typename D, bool ASC, typename IDX>
    TopkOp<T, E, D, ASC, IDX>::~TopkOp() {}

    template<typename T, typename E, typename D, bool ASC, typename IDX>
    void TopkOp<T, E, D, ASC, IDX>::Reorder(AscendTensor<D, DIMS_2> &topkDistance,
        AscendTensor<IDX, DIMS_2> &topkIndices) const
    {
        VALUE_UNUSED(topkDistance);
        VALUE_UNUSED(topkIndices);
    }

    template<typename T, typename E, typename D, bool ASC, typename IDX>
    bool TopkOp<T, E, D, ASC, IDX>::Exec(AscendTensor<D, DIMS_2> &distance, AscendTensor<IDX, DIMS_2> &indices,
        AscendTensor<D, DIMS_2> &topkDistance, AscendTensor<IDX, DIMS_2> &topkIndices, const uint32_t indexOffset) const
    {
        VALUE_UNUSED(distance);
        VALUE_UNUSED(indices);
        VALUE_UNUSED(topkDistance);
        VALUE_UNUSED(topkIndices);
        VALUE_UNUSED(indexOffset);
        return false;
    }

    template<typename T, typename E, typename D, bool ASC, typename IDX>
    bool TopkOp<T, E, D, ASC, IDX>::Exec(AscendTensor<D, DIMS_1> &distance, AscendTensor<IDX, DIMS_1> &indices,
        AscendTensor<D, DIMS_1> &topkDistance, AscendTensor<IDX, DIMS_1> &topkIndices, const uint32_t realSize,
        const uint32_t indexOffset) const
    {
        VALUE_UNUSED(distance);
        VALUE_UNUSED(indices);
        VALUE_UNUSED(topkDistance);
        VALUE_UNUSED(topkIndices);
        VALUE_UNUSED(realSize);
        VALUE_UNUSED(indexOffset);
        return false;
    }

    template<typename T, typename E, typename D, bool ASC, typename IDX>
    bool TopkOp<T, E, D, ASC, IDX>::Exec(std::tuple<D *, D *, IDX *> &opOutTp, std::tuple<D *, IDX *, int> &topKHeapTp,
        std::tuple<D *, uint32_t *> &tmpHeapTp, const uint32_t realSize, const int burstLen,
        const uint32_t indexOffset) const
    {
        VALUE_UNUSED(opOutTp);
        VALUE_UNUSED(topKHeapTp);
        VALUE_UNUSED(tmpHeapTp);
        VALUE_UNUSED(realSize);
        VALUE_UNUSED(burstLen);
        VALUE_UNUSED(indexOffset);
        return false;
    }

    template<typename T, typename E, typename D, bool ASC, typename IDX>
    bool TopkOp<T, E, D, ASC, IDX>::Exec(std::tuple<D *, D *, IDX *> &opOutTp, std::tuple<D *, IDX *> &topKHeapTp,
        std::tuple<D *, uint32_t *> &tmpHeapTp, const std::tuple<int, int, int, int> &paramTp) const
    {
        VALUE_UNUSED(opOutTp);
        VALUE_UNUSED(topKHeapTp);
        VALUE_UNUSED(tmpHeapTp);
        VALUE_UNUSED(paramTp);
        return false;
    }

    template<typename T, typename E, typename D, bool ASC, typename IDX>
    bool TopkOp<T, E, D, ASC, IDX>::Exec(std::tuple<D *, D *, IDX *> &opOutTp, std::tuple<D *, IDX *, int> &topKHeapTp,
        std::tuple<D *, uint32_t *> &tmpHeapTp, IDX *extremeIdxBak, const uint32_t realSize, const int blockSize,
        const int burstLen, const uint32_t indexOffset) const
    {
        VALUE_UNUSED(opOutTp);
        VALUE_UNUSED(topKHeapTp);
        VALUE_UNUSED(tmpHeapTp);
        VALUE_UNUSED(extremeIdxBak);
        VALUE_UNUSED(realSize);
        VALUE_UNUSED(blockSize);
        VALUE_UNUSED(burstLen);
        VALUE_UNUSED(indexOffset);
        return false;
    }

    template<typename T, typename E, typename D, bool ASC, typename IDX>
    bool TopkOp<T, E, D, ASC, IDX>::Exec(std::tuple<D *, D *, IDX *> &opOutTp,
        std::tuple<D *, uint16_t *, int> &topKHeapTp,
        std::tuple<D *, uint32_t *> &tmpHeapTp, const uint32_t realSize, const int burstLen) const
    {
        VALUE_UNUSED(opOutTp);
        VALUE_UNUSED(topKHeapTp);
        VALUE_UNUSED(tmpHeapTp);
        VALUE_UNUSED(realSize);
        VALUE_UNUSED(burstLen);
        return false;
    }

    template<typename T, typename E, typename D, bool ASC, typename IDX>
    bool TopkOp<T, E, D, ASC, IDX>::Exec(std::tuple<D *, D **, IDX **, int> &opOutTp,
        std::tuple<D *, IDX *, int> &topKHeapTp, std::tuple<D *, uint32_t *> &tmpHeapTp,
        const std::tuple<int, int> &lenTp, const AscendTensor<uint32_t, DIMS_1> &listSize) const
    {
        VALUE_UNUSED(opOutTp);
        VALUE_UNUSED(topKHeapTp);
        VALUE_UNUSED(tmpHeapTp);
        VALUE_UNUSED(lenTp);
        VALUE_UNUSED(listSize);
        return false;
    }

    template<typename T, typename E, typename D, bool ASC, typename IDX>
    bool TopkOp<T, E, D, ASC, IDX>::Exec(std::tuple<D *, D *, IDX *> &opOutTp, std::tuple<D *, IDX *, int> &topKHeapTp,
        const uint32_t realSize, const uint32_t indexOffset, const int burstLen) const
    {
        VALUE_UNUSED(opOutTp);
        VALUE_UNUSED(topKHeapTp);
        VALUE_UNUSED(realSize);
        VALUE_UNUSED(indexOffset);
        VALUE_UNUSED(burstLen);
        return false;
    }

    template<typename T, typename E, typename D, bool ASC, typename IDX>
    bool TopkOp<T, E, D, ASC, IDX>::Exec(AscendTensor<D, 1> &distance, AscendTensor<IDX, 1> &indices,
        AscendTensor<D, 1> &topkDistance, AscendTensor<IDX, 1> &topkIndices) const
    {
        VALUE_UNUSED(distance);
        VALUE_UNUSED(indices);
        VALUE_UNUSED(topkDistance);
        VALUE_UNUSED(topkIndices);
        return false;
    }

    template<typename T, typename E, typename D, bool ASC, typename IDX>
    int partitionv3(D* nums, IDX* idxs, int left, int right)
    {
        VALUE_UNUSED(nums);
        VALUE_UNUSED(idxs);
        VALUE_UNUSED(left);
        VALUE_UNUSED(right);
        return 0;
    }

    template<typename D, typename IDX = uint64_t>
    int partitionv3(D* nums, IDX* idxs, bool* flags, int left, int right)
    {
        VALUE_UNUSED(nums);
        VALUE_UNUSED(idxs);
        VALUE_UNUSED(flags);
        VALUE_UNUSED(left);
        VALUE_UNUSED(right);
        return 0;
    }

    // 前半段保存比top k小的元素，后半段保存比top k大的元素，后半段的元素个数为top k
    template<typename D, typename IDX = uint64_t>
    int quickTopKv2(D* nums, IDX * idxs, bool* flags, int low, int high, int k)
    {
        VALUE_UNUSED(nums);
        VALUE_UNUSED(idxs);
        VALUE_UNUSED(flags);
        VALUE_UNUSED(low);
        VALUE_UNUSED(high);
        VALUE_UNUSED(k);
        return 0;
    }

    template<typename T, typename E, typename D, bool ASC, typename IDX>
    void TopkOp<T, E, D, ASC, IDX>::QuickSort(D* nums, IDX * idxs, bool* flags, int low, int high) const
    {
        VALUE_UNUSED(nums);
        VALUE_UNUSED(idxs);
        VALUE_UNUSED(flags);
        VALUE_UNUSED(low);
        VALUE_UNUSED(high);
    }

// flags 默认初始化为false
// flags为flase表示global id，为true表示local id
    template<typename T, typename E, typename D, bool ASC, typename IDX>
    bool TopkOp<T, E, D, ASC, IDX>::Exec(std::tuple<D *, D **, IDX **, int> &opOutTp,
                                         std::tuple<D *, IDX *, int> &topKHeapTp,
                                         std::tuple<D *, uint32_t *> &tmpHeapTp,
                                         const std::tuple<int, int> &lenTp,
                                         const AscendTensor<uint32_t, DIMS_1> &listSize,
                                         IDX *extremeIdxBak,
                                         bool* flags,
                                         bool* tmpFlags) const

    {
        VALUE_UNUSED(opOutTp);
        VALUE_UNUSED(topKHeapTp);
        VALUE_UNUSED(tmpHeapTp);
        VALUE_UNUSED(lenTp);
        VALUE_UNUSED(listSize);
        VALUE_UNUSED(extremeIdxBak);
        VALUE_UNUSED(flags);
        VALUE_UNUSED(tmpFlags);
        return false;
    }

    template<typename T, typename E, typename D, bool ASC, typename IDX>
    bool TopkOp<T, E, D, ASC, IDX>::checkParams(AscendTensor<D, 1> &distance, AscendTensor<IDX, 1> &indices,
        AscendTensor<D, 1> &topkDistance, AscendTensor<IDX, 1> &topkIndices) const
    {
        VALUE_UNUSED(distance);
        VALUE_UNUSED(indices);
        VALUE_UNUSED(topkDistance);
        VALUE_UNUSED(topkIndices);
        return false;
    }

    template<typename T, typename E, typename D, bool ASC, typename IDX>
    bool TopkOp<T, E, D, ASC, IDX>::checkParams(AscendTensor<D, DIMS_2> &distance, AscendTensor<IDX, DIMS_2> &indices,
        AscendTensor<D, DIMS_2> &topkDistance, AscendTensor<IDX, DIMS_2> &topkIndices) const
    {
        VALUE_UNUSED(distance);
        VALUE_UNUSED(indices);
        VALUE_UNUSED(topkDistance);
        VALUE_UNUSED(topkIndices);
        return false;
    }

    template class TopkOp<std::greater<float16_t>, std::greater_equal<float16_t>, float16_t>;
    template class TopkOp<std::less<float16_t>, std::less_equal<float16_t>, float16_t>;
    template class TopkOp<std::less<float16_t>, std::less_equal<float16_t>, float16_t, false>;
    template class TopkOp<std::less<float16_t>, std::less_equal<float16_t>, float16_t, false, uint32_t>;
    template class TopkOp<std::greater<int32_t>, std::greater_equal<int32_t>, int32_t>;
} // namespace ascendSearch
