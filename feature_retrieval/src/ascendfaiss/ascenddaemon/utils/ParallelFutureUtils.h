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

#ifndef ASCEND_PARALLEL_FUTURE_UTILS_INCLUDED
#define ASCEND_PARALLEL_FUTURE_UTILS_INCLUDED

#include <cstddef>
#include <exception>
#include <future>
#include <vector>

namespace ascend
{
namespace utils
{

// submit must append at most taskCount futures, immediately after each successful enqueue.
// Reserve before submission so recording an accepted task cannot fail to allocate.
template <typename Submit>
void RunParallelAndWait(std::size_t taskCount, Submit &&submit)
{
    std::vector<std::future<void>> futures;
    futures.reserve(taskCount);
    std::exception_ptr firstException;
    try
    {
        submit(futures);
    }
    catch (...)
    {
        firstException = std::current_exception();
    }

    // Packaged-task future destruction does not wait. Consume every accepted task,
    // even after a submission or worker failure, before caller-owned state can unwind.
    for (auto &future : futures)
    {
        try
        {
            future.get();
        }
        catch (...)
        {
            if (!firstException)
            {
                firstException = std::current_exception();
            }
        }
    }
    if (firstException)
    {
        std::rethrow_exception(firstException);
    }
}

}  // namespace utils
}  // namespace ascend

#endif  // ASCEND_PARALLEL_FUTURE_UTILS_INCLUDED
