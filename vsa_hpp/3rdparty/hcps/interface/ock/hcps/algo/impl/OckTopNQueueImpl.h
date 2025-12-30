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


#ifndef OCK_HCPS_HFO_HETERO_TOPN_QUEUE_IMPL_H
#define OCK_HCPS_HFO_HETERO_TOPN_QUEUE_IMPL_H
#include <cstdint>
#include <vector>
#include <ostream>
#include <memory>
#include <queue>
#include "ock/acladapter/utils/OckAscendFp16.h"

namespace ock {
namespace hcps {
namespace algo {
template <typename _DistanceT, typename _IdxTypeT>
OckTopNNode<_DistanceT, _IdxTypeT>::OckTopNNode(_IdxTypeT id, _DistanceT dist) : idx(id), distance(dist)
{}
template <typename _DistanceT, typename _IdxTypeT>
bool OckTopNNode<_DistanceT, _IdxTypeT>::operator == (const OckTopNNode &other) const
{
    return idx == other.idx;
}
template <typename _DistanceT, typename _IdxTypeT>
bool OckTopNNode<_DistanceT, _IdxTypeT>::operator == (_IdxTypeT otherIdx) const
{
    return idx == otherIdx;
}
template <typename _DistanceT, typename _IdxTypeT>
bool OckCompareAscAdapter<_DistanceT, _IdxTypeT>::operator () (const OckTopNNode<_DistanceT, _IdxTypeT> &lhs,
    const OckTopNNode<_DistanceT, _IdxTypeT> &rhs)
{
    return lhs.distance < rhs.distance;
}
template <typename _DistanceT, typename _IdxTypeT>
bool OckCompareAscAdapter<_DistanceT, _IdxTypeT>::operator () (const OckTopNNode<_DistanceT, _IdxTypeT> &lhs,
    _DistanceT distance)
{
    return lhs.distance < distance;
}
template <typename _DistanceT, typename _IdxTypeT>
bool OckCompareAscAdapter<_DistanceT, _IdxTypeT>::operator () (_DistanceT distance,
    const OckTopNNode<_DistanceT, _IdxTypeT> &rhs)
{
    return distance < rhs.distance;
}
template <typename _DistanceT, typename _IdxTypeT>
bool OckCompareDescAdapter<_DistanceT, _IdxTypeT>::operator () (const OckTopNNode<_DistanceT, _IdxTypeT> &lhs,
    const OckTopNNode<_DistanceT, _IdxTypeT> &rhs)
{
    return lhs.distance > rhs.distance;
}
template <typename _DistanceT, typename _IdxTypeT>
bool OckCompareDescAdapter<_DistanceT, _IdxTypeT>::operator () (const OckTopNNode<_DistanceT, _IdxTypeT> &lhs,
    _DistanceT distance)
{
    return lhs.distance > distance;
}
template <typename _DistanceT, typename _IdxTypeT>
bool OckCompareDescAdapter<_DistanceT, _IdxTypeT>::operator () (_DistanceT distance,
    const OckTopNNode<_DistanceT, _IdxTypeT> &rhs)
{
    return distance > rhs.distance;
}
template <typename _DistanceT, typename _IdxTypeT, typename _CompareAdapterT>
OckTopNQueue<_DistanceT, _IdxTypeT, _CompareAdapterT>::OckTopNQueue(uint32_t topK) : topN(topK),
    cmpFun(_CompareAdapterT()), pq(_CompareAdapterT())
{}
template <typename _DistanceT, typename _IdxTypeT, typename _CompareAdapterT>
bool OckTopNQueue<_DistanceT, _IdxTypeT, _CompareAdapterT>::Empty(void) const
{
    return pq.empty();
}
template <typename _DistanceT, typename _IdxTypeT, typename _CompareAdapterT>
OckTopNNode<_DistanceT, _IdxTypeT> OckTopNQueue<_DistanceT, _IdxTypeT, _CompareAdapterT>::Pop(void)
{
    NodeT ret = pq.top();
    pq.pop();
    return ret;
}
template <typename _DistanceT, typename _IdxTypeT, typename _CompareAdapterT>
std::shared_ptr<std::vector<OckTopNNode<_DistanceT, _IdxTypeT>>> OckTopNQueue<_DistanceT, _IdxTypeT, _CompareAdapterT>::PopAll(void)
{
    std::shared_ptr<std::vector<OckTopNNode<_DistanceT, _IdxTypeT>>> ret =
        std::make_shared<std::vector<OckTopNNode<_DistanceT, _IdxTypeT>>>();
    ret->reserve(pq.size());
    while (!pq.empty()) {
        ret->push_back(pq.top());
        pq.pop();
    }
    return ret;
}
template <typename _DistanceT, typename _IdxTypeT, typename _CompareAdapterT>
const std::vector<OckTopNNode<_DistanceT, _IdxTypeT>> &OckTopNQueue<_DistanceT, _IdxTypeT, _CompareAdapterT>::GetAll(
    void)
{
    // 获取priority_queue内部的vector
    const std::vector<OckTopNNode<_DistanceT, _IdxTypeT>> &vec = pq.getContainer();
    return vec;
}
template <typename _DistanceT, typename _IdxTypeT, typename _CompareAdapterT>
bool OckTopNQueue<_DistanceT, _IdxTypeT, _CompareAdapterT>::AddData(_IdxTypeT idx, DistanceT distance)
{
    if (pq.size() < topN) {
        pq.push(NodeT(idx, distance));
        return true;
    } else {
        if (cmpFun(distance, pq.top())) {
            pq.pop();
            pq.push(NodeT(idx, distance));
            return true;
        }
    }
    return false;
}

template <typename _DistanceT, typename _IdxTypeT, typename _CompareAdapterT>
void OckTopNQueue<_DistanceT, _IdxTypeT, _CompareAdapterT>::AddNodes(const std::vector<NodeT> &nodes)
{
    for (auto &node : nodes) {
        this->AddData(node.idx, node.distance);
    }
}

template <typename _DistanceT, typename _IdxTypeT, typename _CompareAdapterT>
std::shared_ptr<OckTopNQueue<_DistanceT, _IdxTypeT, _CompareAdapterT>> OckTopNQueue<_DistanceT, _IdxTypeT, _CompareAdapterT>::Create(uint32_t topN)
{
    return std::make_shared<OckTopNQueue<_DistanceT, _IdxTypeT, _CompareAdapterT>>(topN);
}

template <typename _DistanceT, typename _IdxTypeT, typename _CompareAdapterT>
std::vector<std::shared_ptr<OckTopNQueue<_DistanceT, _IdxTypeT, _CompareAdapterT>>> OckTopNQueue<_DistanceT, _IdxTypeT, _CompareAdapterT>::CreateMany(
    uint32_t batchSize, uint32_t topN)
{
    std::vector<std::shared_ptr<OckTopNQueue<_DistanceT, _IdxTypeT, _CompareAdapterT>>> ret;
    for (uint32_t i = 0; i < batchSize; ++i) {
        ret.push_back(OckTopNQueue<_DistanceT, _IdxTypeT, _CompareAdapterT>::Create(topN));
    }
    return ret;
}
} // namespace algo
} // namespace hcps
} // namespace ock
#endif
