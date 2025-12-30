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

 
#ifndef TOPK_OP_INCLUDED
#define TOPK_OP_INCLUDED

#include <tuple>

#include "ascenddaemon/utils/AscendTensor.h"

namespace ascend {
template<typename T, typename E, typename D, bool ASC = true, typename IDX = uint64_t>
class TopkOp {
public:
    TopkOp<T, E, D, ASC, IDX>();
    virtual ~TopkOp<T, E, D, ASC, IDX>();

    bool exec(AscendTensor<D, DIMS_2>& distance,
              AscendTensor<IDX, DIMS_2>& indices,
              AscendTensor<D, DIMS_2>& topkDistance,
              AscendTensor<IDX, DIMS_2>& topkIndices,
              const uint32_t indexOffset = 0);

    bool exec(AscendTensor<D, DIMS_1> &distance,
                  AscendTensor<IDX, DIMS_1> &indices,
                  AscendTensor<D, DIMS_1> &topkDistance,
                  AscendTensor<IDX, DIMS_1> &topkIndices,
                  const uint32_t realSize,
                  const uint32_t indexOffset);

    bool exec(AscendTensor<D, DIMS_1>& distance,
              AscendTensor<IDX, DIMS_1>& indices,
              AscendTensor<D, DIMS_1>& topkDistance,
              AscendTensor<IDX, DIMS_1>& topkIndices);

    bool exec(std::tuple<D*, D*, IDX*> &opOutTp,
              std::tuple<D*, IDX*, int> &topKHeapTp,
              std::tuple<D*, uint32_t*> &tmpHeapTp,
              const uint32_t realSize,
              const int burstLen,
              const uint32_t indexOffset = 0);

    bool exec(std::tuple<D*, D*, IDX*> &opOutTp,
              std::tuple<D*, IDX*, int> &topKHeapTp,
              std::tuple<D*, uint32_t*> &tmpHeapTp,
              IDX *extremeIdxBak,
              const uint32_t realSize,
              const int blockSize,
              const int burstLen,
              const uint32_t indexOffset = 0);

    bool exec(std::tuple<D *, D *, IDX *> &opOutTp,
             std::tuple<D *, IDX *> &topKHeapTp,
             std::tuple<D *, uint32_t *> &tmpHeapTp,
             std::tuple<int, int, int, int> &paramTp);

    bool exec(std::tuple<D*, D*, IDX*> &opOutTp,
              std::tuple<D*, uint16_t*, int> &topKHeapTp,
              std::tuple<D*, uint32_t*> &tmpHeapTp,
              const uint32_t realSize,
              const int burstLen);

    bool exec(std::tuple<D *, D **, IDX **, int> &opOutTp,
              std::tuple<D *, IDX *, int> &topKHeapTp,
              std::tuple<D *, uint32_t *> &tmpHeapTp,
              std::tuple<int, int> &lenTp,
              AscendTensor<uint32_t, DIMS_1> &listSize);

    bool exec(std::tuple<D*, D*, IDX*> &opOutTp,
              std::tuple<D*, IDX*, int> &topKHeapTp,
              const uint32_t realSize,
              const uint32_t indexOffset,
              const int burstLen);

    void reorder(AscendTensor<D, DIMS_2>& topkDistance,
                 AscendTensor<IDX, DIMS_2>& topkIndices);

    void reorder(AscendTensor<D, DIMS_1>& topkDistance,
                 AscendTensor<IDX, DIMS_1>& topkIndices);

private:
    bool checkParams(AscendTensor<D, DIMS_2>& distance,
                     AscendTensor<IDX, DIMS_2>& indices,
                     AscendTensor<D, DIMS_2>& topkDistance,
                     AscendTensor<IDX, DIMS_2>& topkIndices);

    bool checkParams(AscendTensor<D, DIMS_1>& distance,
                     AscendTensor<IDX, DIMS_1>& indices,
                     AscendTensor<D, DIMS_1>& topkDistance,
                     AscendTensor<IDX, DIMS_1>& topkIndices);

private:
    T compare;
    E compareEqual;
};
}  // namespace ascend

#endif
