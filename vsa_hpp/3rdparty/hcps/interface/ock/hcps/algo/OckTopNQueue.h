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


#ifndef OCK_HCPS_HFO_HETERO_TOPN_QUEUE_H
#define OCK_HCPS_HFO_HETERO_TOPN_QUEUE_H
#include <cstdint>
#include <vector>
#include <ostream>
#include <memory>
#include <queue>

namespace ock {
namespace hcps {
namespace algo {
template <typename T, typename Container = std::vector<T>, typename Compare = std::less<typename Container::value_type>>
class MyPriorityQueue : public std::priority_queue<T, Container, Compare> {
public:
    // 构造函数
    MyPriorityQueue(const Compare &compFunc = Compare()) : std::priority_queue<T, Container, Compare>(compFunc)
    {}
    // 获取底层容器的方法
    const Container &getContainer() const
    {
        return this->c;
    }
};
template <typename _DistanceT, typename _IdxTypeT> struct OckTopNNode {
    OckTopNNode(void) = default;
    OckTopNNode(_IdxTypeT id, _DistanceT dist);
    bool operator == (const OckTopNNode &other) const;
    bool operator == (_IdxTypeT otherIdx) const;
    friend std::ostream &operator << (std::ostream &os, const OckTopNNode &obj)
    {
        return os << "{'idx':" << obj.idx << ",'dis':" << obj.distance << "}";
    }
    _IdxTypeT idx;
    _DistanceT distance;
};
template <typename _DistanceT, typename _IdxTypeT> struct OckCompareAscAdapter {
    bool operator () (const OckTopNNode<_DistanceT, _IdxTypeT> &lhs, const OckTopNNode<_DistanceT, _IdxTypeT> &rhs);
    bool operator () (const OckTopNNode<_DistanceT, _IdxTypeT> &lhs, _DistanceT distance);
    bool operator () (_DistanceT distance, const OckTopNNode<_DistanceT, _IdxTypeT> &rhs);
};
template <typename _DistanceT, typename _IdxTypeT> struct OckCompareDescAdapter {
    bool operator () (const OckTopNNode<_DistanceT, _IdxTypeT> &lhs, const OckTopNNode<_DistanceT, _IdxTypeT> &rhs);
    bool operator () (const OckTopNNode<_DistanceT, _IdxTypeT> &lhs, _DistanceT distance);
    bool operator () (_DistanceT distance, const OckTopNNode<_DistanceT, _IdxTypeT> &rhs);
};
template <typename _DistanceT, typename _IdxTypeT, typename _CompareAdapterT> class OckTopNQueue {
public:
    using DistanceT = _DistanceT;
    using NodeT = OckTopNNode<_DistanceT, _IdxTypeT>;
    using PriorityQueue = MyPriorityQueue<NodeT, std::vector<NodeT>, _CompareAdapterT>;

    ~OckTopNQueue() noexcept = default;
    OckTopNQueue(uint32_t topK);

    bool Empty(void) const;
    NodeT Pop(void);
    std::shared_ptr<std::vector<NodeT>> PopAll(void);
    const std::vector<NodeT>& GetAll(void);
    bool AddData(_IdxTypeT idx, _DistanceT distance);
    void AddNodes(const std::vector<NodeT> &nodes);

    static std::shared_ptr<OckTopNQueue> Create(uint32_t topN = 100UL);
    static std::vector<std::shared_ptr<OckTopNQueue>> CreateMany(uint32_t batchSize, uint32_t topN);

private:
    const uint32_t topN;
    _CompareAdapterT cmpFun;
    PriorityQueue pq;
};

using DoubleNode = OckTopNNode<double, uint64_t>;
using FloatNode = OckTopNNode<float, uint64_t>;
} // namespace algo
} // namespace hcps
} // namespace ock
#include "ock/hcps/algo/impl/OckTopNQueueImpl.h"
#endif
