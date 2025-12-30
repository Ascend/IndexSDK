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


#ifndef RETRIEVAL_ALGO_ACC_LIB_THREADPOOL_H
#define RETRIEVAL_ALGO_ACC_LIB_THREADPOOL_H

#include <vector>
#include <queue>
#include <future>
#include <functional>

// A simple C++11 Thread Pool implementation from progschj/ThreadPool
namespace ascendsearch {
class ThreadPool {
public:
    explicit ThreadPool(size_t);
    template <class F, class... Args>
    auto Enqueue(F &&f, Args &&... args) -> std::future<typename std::result_of<F(Args...)>::type>;
    ~ThreadPool();

private:
    std::vector<std::thread> workers;
    std::queue<std::function<void()> > tasks;

    std::mutex queueMutex;
    std::condition_variable condition;
    bool stop;
};

template <class F, class... Args>
auto ThreadPool::Enqueue(F &&f, Args &&... args) -> std::future<typename std::result_of<F(Args...)>::type>
{
    using return_type = typename std::result_of<F(Args...)>::type;

    auto task = std::make_shared<std::packaged_task<return_type()> >(
        std::bind(std::forward<F>(f), std::forward<Args>(args)...));

    std::future<return_type> res = task->get_future();
    {
        std::unique_lock<std::mutex> lock(queueMutex);

        tasks.emplace([task]() { (*task)(); });
    }
    condition.notify_one();
    return res;
}

}  // namespace ascendsearch

#endif  // RETRIEVAL_ALGO_ACC_LIB_THREADPOOL_H