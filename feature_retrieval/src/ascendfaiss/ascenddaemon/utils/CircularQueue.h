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


#ifndef CIRCULAR_QUEUE_INCLUDED
#define CIRCULAR_QUEUE_INCLUDED

#include <tuple>
#include <vector>

namespace ascend {
// A circular queue that assumes pushed items are in
//   non-decreasing order if heap is min heap, or
//   non-increasing order if heap is max heap.
// ("item" is defined as a <dist, id> pair)
template<typename D, typename I> class CircularQueue {
public:
    CircularQueue(int k) : dists(k + 1), ids(k + 1), size(k), begin(0), end(0) {}

    void push(D pushDist, I pushId)
    {
        // do not push if pushId is the same as the last pushed id
        if (!this->empty() && ids[this->lastPos()] == pushId) {
            return;
        }

        dists[end] = pushDist;
        ids[end] = pushId;
        end = (end == size) ? 0 : (end + 1);
        if (end == begin) {
            begin = (begin == size) ? 0 : (begin + 1);
        }
    }

    // pop in LIFO order !!
    // return true if the queue is non-empty, or false if empty
    bool pop(D &poppedDist, I &poppedId)
    {
        if (this->empty()) {
            return false;
        }

        int pos = this->lastPos();
        poppedDist = dists[pos];
        poppedId = ids[pos];
        end = (end == 0) ? size : (end - 1);
        return true;
    }

    inline bool empty()
    {
        return begin == end;
    }

    void reset()
    {
        begin = 0;
        end = 0;
    }

    inline int getBegin() const
    {
        return begin;
    }

    inline int getEnd() const
    {
        return end;
    }

    inline int getSize() const
    {
        return size;
    }

    inline D *getDists()
    {
        return dists.data();
    }

    inline I *getIds()
    {
        return ids.data();
    }

    ~CircularQueue() {}

private:
    inline int lastPos()
    {
        return (end == 0) ? size : (end - 1);
    }

    inline bool full()
    {
        return (begin == 0) ? (end == size) : (begin == end + 1);
    }

    std::vector<D> dists;
    std::vector<I> ids;
    int size;
    int begin, end;
};
} // namespace ascend

#endif
