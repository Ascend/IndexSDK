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


#include <algorithm>
#include <npu/common/utils/TopkOp.h>
#include <npu/common/utils/Limits.h>
#include <npu/common/utils/HeapSort.h>

namespace ascendSearchacc {
namespace {
int32_t GetDefaultVal(int32_t val, bool asc)
{
    static_cast<void>(val);  // unused variable val
    if (asc) {
        return std::numeric_limits<int32_t>::max();
    }
    return std::numeric_limits<int32_t>::min();
}

float16_t GetDefaultVal(float16_t val, bool asc)
{
    static_cast<void>(val);  // unused variable val
    if (asc) {
        return Limits<float16_t>::getMax();
    }
    return Limits<float16_t>::getMin();
}
}  // namespace

template <typename T, typename E, typename D, bool ASC, typename IDX>
TopkOp<T, E, D, ASC, IDX>::TopkOp()
{
}

template <typename T, typename E, typename D, bool ASC, typename IDX>
TopkOp<T, E, D, ASC, IDX>::~TopkOp()
{
}

template <typename T, typename E, typename D, bool ASC, typename IDX>
void TopkOp<T, E, D, ASC, IDX>::reorder(AscendTensor<D, DIMS_2> &topkDistance, AscendTensor<IDX, DIMS_2> &topkIndices)
{
    int n = topkIndices.getSize(0);

#pragma omp parallel for if (n > 1) num_threads(n < 6 ? n : 6)
    for (int i = 0; i < n; i++) {
        auto dist = topkDistance[i].view();
        auto id = topkIndices[i].view();
        reorder(dist, id);
    }
}

template <typename T, typename E, typename D, bool ASC, typename IDX>
void TopkOp<T, E, D, ASC, IDX>::reorder(AscendTensor<D, 1> &topkDistance, AscendTensor<IDX, 1> &topkIndices)
{
    D *dist = topkDistance.data();
    IDX *ids = topkIndices.data();
    size_t k = (size_t)topkIndices.getSize(0);

    size_t j = 0;

    for (size_t i = 0; i < k; i++) {
        /* top element should be put at the end of the list */
        D val = dist[0];
        IDX id = ids[0];

        popHeap(k - i, dist, ids, compare);
        dist[k - j - 1] = val;
        ids[k - j - 1] = id;
        if (id != std::numeric_limits<IDX>::max()) {
            j++;
        }
    }

    auto err = memmove_s(dist, k * sizeof(*dist), dist + k - j, j * sizeof(*dist));
    ASCEND_THROW_IF_NOT_FMT(err == EOK, "An error occured when memmove in TopK reorder, %d!\n", err);
    err = memmove_s(ids, k * sizeof(*ids), ids + k - j, j * sizeof(*ids));
    ASCEND_THROW_IF_NOT_FMT(err == EOK, "An error occured when memmove in TopK reorder, %d!\n", err);

    for (; j < k; j++) {
        dist[j] = GetDefaultVal(dist[j], ASC);
        // invalid id should be -1 or max unsigned value
        ids[j] = std::numeric_limits<IDX>::max();
    }
}

template <typename T, typename E, typename D, bool ASC, typename IDX>
bool TopkOp<T, E, D, ASC, IDX>::exec(AscendTensor<D, DIMS_2> &distance, AscendTensor<IDX, DIMS_2> &indices,
                                     AscendTensor<D, DIMS_2> &topkDistance, AscendTensor<IDX, DIMS_2> &topkIndices,
                                     const uint32_t indexOffset)
{
    if (!checkParams(distance, indices, topkDistance, topkIndices)) {
        return false;
    }

    D *disArray = distance.data();
    IDX *idArray = indices.data();
    D *disResult = topkDistance.data();
    IDX *idResult = topkIndices.data();
    const int batch = distance.getSize(0);
    const int numDist = distance.getSize(1);
    const int kVal = topkDistance.getSize(1);
    bool idFlag = idArray != nullptr;

#pragma omp parallel for if (batch > 1) num_threads(batch < 6 ? batch : 6)
    for (int i = 0; i < batch; i++) {
        int batchOffset = numDist * i;
        int heapOffset = kVal * i;
        for (int j = 0; j < numDist; j++) {
            if (compareEqual(disArray[batchOffset + j], disResult[heapOffset])) {
                continue;
            }
            IDX index = idFlag ? *(idArray + batchOffset + j) : (indexOffset + j);
            pushHeap(kVal, disResult + heapOffset, idResult + heapOffset, disArray[batchOffset + j], index, compare);
        }
    }

    return true;
}

template <typename T, typename E, typename D, bool ASC, typename IDX>
bool TopkOp<T, E, D, ASC, IDX>::exec(AscendTensor<D, DIMS_1> &distance, AscendTensor<IDX, DIMS_1> &indices,
                                     AscendTensor<D, DIMS_1> &topkDistance, AscendTensor<IDX, DIMS_1> &topkIndices,
                                     const uint32_t realSize, const uint32_t indexOffset)
{
    if (!checkParams(distance, indices, topkDistance, topkIndices)) {
        return false;
    }

    if (distance.getSize(0) < static_cast<int>(realSize)) {
        return false;
    }

    D *disArray = distance.data();
    IDX *idArray = indices.data();
    D *disResult = topkDistance.data();
    IDX *idResult = topkIndices.data();
    const int kVal = topkDistance.getSize(0);
    bool idFlag = idArray != nullptr;

    for (uint32_t j = 0; j < realSize; j++) {
        if (compareEqual(disArray[j], disResult[0])) {
            continue;
        }
        IDX index = idFlag ? *(idArray + j) : (indexOffset + j);
        pushHeap(kVal, disResult, idResult, *(disArray + j), index, compare);
    }

    return true;
}

template <typename T, typename E, typename D, bool ASC, typename IDX>
bool TopkOp<T, E, D, ASC, IDX>::exec(std::tuple<D *, D *, IDX *> &opOutTp, std::tuple<D *, IDX *, int> &topKHeapTp,
                                     std::tuple<D *, uint32_t *> &tmpHeapTp, const uint32_t realSize,
                                     const int burstLen, const uint32_t indexOffset)
{
    D *extremeDisArray = std::get<1>(opOutTp);
    D *tmpDisResult = std::get<0>(tmpHeapTp);
    uint32_t *tmpIdResult = std::get<1>(tmpHeapTp);

    const int kVal = std::get<2>(topKHeapTp);
    const int extremeSize = (realSize + burstLen - 1) / burstLen * 2;
    const int halfMinBatch = burstLen / 2;

    // 2 reps skip index, only handle distance
    for (int i = 0; i < extremeSize; i += 2) {
        if (compareEqual(extremeDisArray[i], tmpDisResult[0])) {
            continue;
        }

        pushHeap(kVal, tmpDisResult, tmpIdResult, extremeDisArray[i], static_cast<uint32_t>(i * halfMinBatch), compare);
    }
    std::sort(tmpIdResult, tmpIdResult + kVal);

    D *disArray = std::get<0>(opOutTp);
    IDX *idArray = std::get<2>(opOutTp);  // 2 reps second element
    D *disResult = std::get<0>(topKHeapTp);
    IDX *idResult = std::get<1>(topKHeapTp);

    const bool idFlag = idArray != nullptr;
    const uint32_t *tmpIdEnd = tmpIdResult + std::min(kVal, extremeSize / 2);

    for (uint32_t *pid = tmpIdResult; pid != tmpIdEnd; ++pid) {
        const uint32_t endIdx = std::min(pid[0] + burstLen, realSize);
        for (uint32_t j = pid[0]; j < endIdx; ++j) {
            if (compareEqual(disArray[j], disResult[0])) {
                continue;
            }

            IDX index = idFlag ? *(idArray + j) : (indexOffset + j);
            pushHeap(kVal, disResult, idResult, disArray[j], index, compare);
        }
    }

    return true;
}

template <typename T, typename E, typename D, bool ASC, typename IDX>
bool TopkOp<T, E, D, ASC, IDX>::exec(std::tuple<D *, D *, IDX *> &opOutTp, std::tuple<D *, IDX *> &topKHeapTp,
                                     std::tuple<D *, uint32_t *> &tmpHeapTp, std::tuple<int, int, int, int> &paramTp)
{
    const int queryNum = std::get<0>(paramTp);
    const int baseSize = std::get<1>(paramTp);
    const int kVal = std::get<2>(paramTp);
    const int burstLen = std::get<3>(paramTp);
    const int extremeSize = (baseSize + burstLen - 1) / burstLen * 2;
    const int halfBatch = burstLen / 2;

    D *extremeDisArray = std::get<1>(opOutTp);
    D *tmpDisResult = std::get<0>(tmpHeapTp);
    uint32_t *tmpIdResult = std::get<1>(tmpHeapTp);

    for (int qid = 0; qid < queryNum; ++qid) {
        D *extremes = extremeDisArray + qid * extremeSize;
        D *tmpDis = tmpDisResult + qid * kVal;
        uint32_t *tmpId = tmpIdResult + qid * kVal;
        // 2 reps skip index, only handle distance
        for (int j = 0; j < extremeSize; j += 2) {
            if (compareEqual(extremes[j], tmpDis[0])) {
                continue;
            }
            pushHeap(kVal, tmpDis, tmpId, extremes[j], static_cast<uint32_t>(j * halfBatch), compare);
        }
    }

    D *disArray = std::get<0>(opOutTp);
    IDX *idArray = std::get<2>(opOutTp);  // 2 reps second element
    D *disResult = std::get<0>(topKHeapTp);
    IDX *idResult = std::get<1>(topKHeapTp);
    const bool idFlag = idArray != nullptr;

    for (int qid = 0; qid < queryNum; ++qid) {
        D *disArr = disArray + qid * baseSize;
        D *disRes = disResult + qid * kVal;
        IDX *idRes = idResult + qid * kVal;
        uint32_t *tmpId = tmpIdResult + qid * kVal;
        const uint32_t *tmpIdEnd = tmpId + kVal;

        for (uint32_t *pid = tmpId; pid != tmpIdEnd; ++pid) {
            const uint32_t endIdx = std::min(pid[0] + burstLen, (uint32_t)baseSize);
            for (uint32_t j = pid[0]; j < endIdx; ++j) {
                if (compareEqual(disArr[j], disRes[0])) {
                    continue;
                }

                IDX index = idFlag ? *(idArray + j) : j;
                pushHeap(kVal, disRes, idRes, disArr[j], index, compare);
            }
        }
    }

    return true;
}

template <typename T, typename E, typename D, bool ASC, typename IDX>
bool TopkOp<T, E, D, ASC, IDX>::exec(std::tuple<D *, D *, IDX *> &opOutTp, std::tuple<D *, IDX *, int> &topKHeapTp,
                                     std::tuple<D *, uint32_t *> &, IDX *extremeIdxBak,
                                     const uint32_t realSize, const int blockSize, const int burstLen,
                                     const uint32_t indexOffset)
{
    D *extremeDisArray = std::get<1>(opOutTp);

    const int kVal = std::get<2>(topKHeapTp);
    const int extremeSize = (realSize + burstLen - 1) / burstLen * 2;
    const int halfMinBatch = burstLen / 2;

    D *disArray = std::get<0>(opOutTp);
    IDX *idArray = std::get<2>(opOutTp);  // 2 reps second element
    D *disResult = std::get<0>(topKHeapTp);
    IDX *idResult = std::get<1>(topKHeapTp);

    const bool idFlag = idArray != nullptr;
    // 2 reps skip index, only handle distance
    for (int i = 0; i < extremeSize; i += 2) {
        if (compareEqual(extremeDisArray[i], disResult[0])) {
            continue;
        }

        auto *extremeIdx = &extremeDisArray[i + 1];
        uint16_t *extremeIdxInt = reinterpret_cast<uint16_t *>(extremeIdx);
        IDX realIdx = IDX(i * halfMinBatch + *extremeIdxInt);
        if (realIdx >= realSize) {
            pushHeap(kVal, disResult, idResult, disArray[i * halfMinBatch], IDX(i * halfMinBatch + indexOffset),
                     compare);
            continue;
        }
        realIdx = idFlag ? *(idArray + realIdx) : (indexOffset + realIdx);
        pushHeap(kVal, disResult, idResult, extremeDisArray[i], realIdx, compare);
    }

    for (int i = 0; i < kVal; ++i) {
        extremeIdxBak[i] = idResult[i];
    }

    const IDX *tmpIdEnd = extremeIdxBak + kVal;

    for (IDX *pid = extremeIdxBak; pid != tmpIdEnd; ++pid) {
        if ((pid[0] < indexOffset) || (pid[0] == std::numeric_limits<IDX>::max())) {
            continue;
        }
        IDX curStart = pid[0] / burstLen * burstLen;
        curStart -= pid[0] / blockSize * blockSize;
        const IDX endIdx = std::min(curStart + burstLen, (IDX)realSize);

        for (IDX j = curStart; j < endIdx; ++j) {
            if (compareEqual(disArray[j], disResult[0])) {
                continue;
            }
            IDX index = idFlag ? *(idArray + j) : (indexOffset + j);
            if (index == pid[0]) {
                continue;
            }

            pushHeap(kVal, disResult, idResult, disArray[j], index, compare);
        }
    }
    return true;
}

template <typename T, typename E, typename D, bool ASC, typename IDX>
bool TopkOp<T, E, D, ASC, IDX>::exec(std::tuple<D *, D *, IDX *> &opOutTp, std::tuple<D *, uint16_t *, int> &topKHeapTp,
                                     std::tuple<D *, uint32_t *> &tmpHeapTp, const uint32_t realSize,
                                     const int burstLen)
{
    D *extremeDisArray = std::get<1>(opOutTp);
    D *tmpDisResult = std::get<0>(tmpHeapTp);
    uint32_t *tmpIdResult = std::get<1>(tmpHeapTp);

    const int kVal = std::get<2>(topKHeapTp);
    const int extremeSize = (realSize + burstLen - 1) / burstLen * 2;
    const int halfMinBatch = burstLen / 2;

    // 2 reps skip index, only handle distance
    for (int i = 0; i < extremeSize; i += 2) {
        if (compareEqual(extremeDisArray[i], tmpDisResult[0])) {
            continue;
        }
        pushHeap(kVal, tmpDisResult, tmpIdResult, extremeDisArray[i], static_cast<uint32_t>(i * halfMinBatch), compare);
    }

    D *disArray = std::get<0>(opOutTp);
    IDX *idArray = std::get<2>(opOutTp);  // 2 reps second element
    D *disResult = std::get<0>(topKHeapTp);
    uint16_t *idResult = std::get<1>(topKHeapTp);

    const bool idFlag = idArray != nullptr;
    const uint32_t *tmpIdEnd = tmpIdResult + std::min(kVal, extremeSize / 2);

    for (uint32_t *pid = tmpIdResult; pid != tmpIdEnd; ++pid) {
        const uint32_t endIdx = std::min(pid[0] + burstLen, realSize);
        for (uint32_t j = pid[0]; j < endIdx; ++j) {
            if (compareEqual(disArray[j], disResult[0])) {
                continue;
            }

            uint16_t index = idFlag ? *(idArray + j) : j;
            pushHeap(kVal, disResult, idResult, disArray[j], index, compare);
        }
    }

    return true;
}

template <typename T, typename E, typename D, bool ASC, typename IDX>
bool TopkOp<T, E, D, ASC, IDX>::exec(std::tuple<D *, D **, IDX **, int> &opOutTp,
                                     std::tuple<D *, IDX *, int> &topKHeapTp, std::tuple<D *, uint32_t *> &tmpHeapTp,
                                     std::tuple<int, int> &lenTp, AscendTensor<uint32_t, DIMS_1> &listSize)
{
    // the list num is index 3
    int listNum = std::get<3>(opOutTp);
    D **extremeDisArray = std::get<1>(opOutTp);
    D *tmpDisResult = std::get<0>(tmpHeapTp);
    uint32_t *tmpIdResult = std::get<1>(tmpHeapTp);

    const int burstLen = std::get<0>(lenTp);
    const int listLen = std::get<1>(lenTp);
    const int kVal = std::get<2>(topKHeapTp);

    int tmpHeapSize = 0;
    for (int i = 0; i < listNum; ++i) {
        const uint32_t extremeSize = (listSize[i].value() + burstLen - 1) / (uint32_t)burstLen;

        for (uint32_t j = 0; j < extremeSize; ++j) {
            // extremeDisArray contains extreme and index, the multi 2
            if (compareEqual(extremeDisArray[i][2 * j], tmpDisResult[0])) {
                continue;
            }
            // extremeDisArray contains extreme and index, the multi 2
            pushHeap(kVal, tmpDisResult, tmpIdResult, extremeDisArray[i][2 * j],
                     static_cast<uint32_t>(i * listLen + j * (uint32_t)burstLen), compare);
            ++tmpHeapSize;
        }
    }
    std::sort(tmpIdResult, tmpIdResult + kVal);

    D *disArray = std::get<0>(opOutTp);
    IDX **idArray = std::get<2>(opOutTp);  // 2 reps second element
    D *disResult = std::get<0>(topKHeapTp);
    IDX *idResult = std::get<1>(topKHeapTp);

    const uint32_t *tmpIdEnd = tmpIdResult + std::min(kVal, tmpHeapSize);

    for (uint32_t *pid = tmpIdResult; pid != tmpIdEnd; ++pid) {
        const uint32_t i = pid[0] / listLen;
        const uint32_t endIdx = std::min(pid[0] + burstLen, i * listLen + listSize[i].value());
        for (uint32_t j = pid[0]; j < endIdx; ++j) {
            if (compareEqual(disArray[j], disResult[0])) {
                continue;
            }

            IDX index = idArray[i][j % listLen];
            pushHeap(kVal, disResult, idResult, disArray[j], index, compare);
        }
    }

    return true;
}

template <typename T, typename E, typename D, bool ASC, typename IDX>
bool TopkOp<T, E, D, ASC, IDX>::exec(std::tuple<D *, D *, IDX *> &opOutTp, std::tuple<D *, IDX *, int> &topKHeapTp,
                                     const uint32_t realSize, const uint32_t indexOffset, const int burstLen)
{
    D *disArray = std::get<0>(opOutTp);
    D *extremeDisArray = std::get<1>(opOutTp);
    IDX *idArray = std::get<2>(opOutTp);  // 2 reps second element
    D *disResult = std::get<0>(topKHeapTp);
    IDX *idResult = std::get<1>(topKHeapTp);
    const int kVal = std::get<2>(topKHeapTp);
    bool idFlag = idArray != nullptr;

    // 2 reps skip index, only handle distance
    for (uint32_t i = 0, k = 0; i < realSize; i += burstLen, k += 2) {
        const uint32_t endIdx = std::min(i + (uint32_t)burstLen, realSize);
        for (uint32_t j = i; j < endIdx; ++j) {
            if (compareEqual(extremeDisArray[k], disResult[0])) {
                break;
            }

            if (compareEqual(disResult[0], disArray[j])) {
                IDX index = idFlag ? *(idArray + j) : (indexOffset + j);
                pushHeap(kVal, disResult, idResult, disArray[j], index, compare);
            }
        }
    }

    return true;
}

template <typename T, typename E, typename D, bool ASC, typename IDX>
bool TopkOp<T, E, D, ASC, IDX>::exec(AscendTensor<D, 1> &distance, AscendTensor<IDX, 1> &indices,
                                     AscendTensor<D, 1> &topkDistance, AscendTensor<IDX, 1> &topkIndices)
{
    if (!checkParams(distance, indices, topkDistance, topkIndices)) {
        return false;
    }

    D *disArray = distance.data();
    IDX *idArray = indices.data();
    D *disResult = topkDistance.data();
    IDX *idResult = topkIndices.data();
    const uint32_t numDist = (uint32_t)distance.getSize(0);
    const int kVal = topkDistance.getSize(0);
    bool idFlag = idArray != nullptr;

    for (uint32_t j = 0; j < numDist; j++) {
        if (compareEqual(disArray[j], disResult[0])) {
            continue;
        }
        IDX index = idFlag ? *(idArray + j) : j;
        pushHeap(kVal, disResult, idResult, *(disArray + j), index, compare);
    }

    return true;
}

template <typename T, typename E, typename D, bool ASC, typename IDX>
bool TopkOp<T, E, D, ASC, IDX>::checkParams(AscendTensor<D, 1> &distance, AscendTensor<IDX, 1> &,
                                            AscendTensor<D, 1> &topkDistance, AscendTensor<IDX, 1> &topkIndices)
{
    if (distance.data() == nullptr || topkDistance.data() == nullptr || topkIndices.data() == nullptr) {
        return false;
    }

    int outDistSize = topkDistance.getSize(0);
    int outIdSize = topkIndices.getSize(0);
    return outIdSize == outDistSize;
}

template <typename T, typename E, typename D, bool ASC, typename IDX>
bool TopkOp<T, E, D, ASC, IDX>::checkParams(AscendTensor<D, DIMS_2> &distance, AscendTensor<IDX, DIMS_2> &,
                                            AscendTensor<D, DIMS_2> &topkDistance,
                                            AscendTensor<IDX, DIMS_2> &topkIndices)
{
    if (distance.data() == nullptr || topkDistance.data() == nullptr || topkIndices.data() == nullptr) {
        return false;
    }

    int inDistSize0 = distance.getSize(0);
    int outDistSize0 = topkDistance.getSize(0);
    int outDistSize1 = topkDistance.getSize(1);
    int outIdSize0 = topkIndices.getSize(0);
    int outIdSize1 = topkIndices.getSize(1);
    return !(inDistSize0 != outDistSize0 || inDistSize0 != outIdSize0 || outIdSize1 != outDistSize1);
}

template class TopkOp<std::greater<float16_t>, std::greater_equal<float16_t>, float16_t>;

template class TopkOp<std::less<float16_t>, std::less_equal<float16_t>, float16_t>;

template class TopkOp<std::less<float16_t>, std::less_equal<float16_t>, float16_t, false>;

template class TopkOp<std::less<float16_t>, std::less_equal<float16_t>, float16_t, false, uint32_t>;

template class TopkOp<std::greater<int32_t>, std::greater_equal<int32_t>, int32_t>;
}  // namespace ascendSearchacc