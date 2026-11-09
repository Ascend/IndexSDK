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

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <stdexcept>
#include <thread>
#include <utility>

#include "ascenddaemon/utils/ParallelFutureUtils.h"

namespace
{

// The first task has already failed before dispatch. A sibling cannot finish until
// the test releases it, so packaged-task future destruction cannot mask an early exit.
std::exception_ptr RunWithBlockedSibling(std::exception_ptr workerFailure,
                                         std::exception_ptr submissionFailure = nullptr)
{
    std::packaged_task<void()> first(
        [&]
        {
            if (workerFailure)
            {
                std::rethrow_exception(workerFailure);
            }
        });
    auto firstFuture = first.get_future();
    first();

    std::promise<void> releaseSibling;
    auto releaseFuture = releaseSibling.get_future();
    std::atomic<bool> siblingCompleted{false};
    std::packaged_task<void()> sibling(
        [&]
        {
            releaseFuture.wait();
            siblingCompleted.store(true);
        });
    auto siblingFuture = sibling.get_future();
    std::thread siblingThread(std::move(sibling));

    std::promise<void> submissionObserved;
    auto submissionFuture = submissionObserved.get_future();
    std::promise<void> returned;
    auto returnedFuture = returned.get_future();
    std::exception_ptr result;
    bool completedAtExit = false;
    bool reservedBeforeSubmit = false;
    std::thread caller(
        [&]
        {
            bool submissionSignaled = false;
            try
            {
                const std::size_t taskCount = submissionFailure ? 3 : 2;
                ascend::utils::RunParallelAndWait(taskCount,
                                                  [&](auto &futures)
                                                  {
                                                      reservedBeforeSubmit =
                                                          futures.empty() && futures.capacity() >= taskCount;
                                                      futures.emplace_back(std::move(firstFuture));
                                                      futures.emplace_back(std::move(siblingFuture));
                                                      submissionObserved.set_value();
                                                      submissionSignaled = true;
                                                      if (submissionFailure)
                                                      {
                                                          // Two tasks accepted; the third enqueue fails before
                                                          // acceptance.
                                                          std::rethrow_exception(submissionFailure);
                                                      }
                                                  });
            }
            catch (...)
            {
                result = std::current_exception();
            }
            completedAtExit = siblingCompleted.load();
            if (!submissionSignaled)
            {
                // Also signal when submission failed before entering the callback.
                submissionObserved.set_value();
            }
            returned.set_value();
        });

    submissionFuture.wait();
    EXPECT_EQ(returnedFuture.wait_for(std::chrono::milliseconds(100)), std::future_status::timeout);
    releaseSibling.set_value();
    caller.join();
    siblingThread.join();
    EXPECT_TRUE(completedAtExit);
    EXPECT_TRUE(reservedBeforeSubmit);
    return result;
}

TEST(TestParallelFutureUtils, WorkerFailureDrainsDelayedSibling)
{
    auto failure = RunWithBlockedSibling(std::make_exception_ptr(std::runtime_error("first worker")));
    ASSERT_TRUE(failure);
    try
    {
        std::rethrow_exception(failure);
    }
    catch (const std::runtime_error &error)
    {
        EXPECT_STREQ(error.what(), "first worker");
    }
    catch (...)
    {
        FAIL() << "Unexpected exception type";
    }
}

TEST(TestParallelFutureUtils, PartialSubmissionFailureDrainsAcceptedTasksAndTakesPrecedence)
{
    auto failure = RunWithBlockedSibling(std::make_exception_ptr(std::runtime_error("worker")),
                                         std::make_exception_ptr(std::runtime_error("enqueue")));
    ASSERT_TRUE(failure);
    try
    {
        std::rethrow_exception(failure);
    }
    catch (const std::runtime_error &error)
    {
        EXPECT_STREQ(error.what(), "enqueue");
    }
    catch (...)
    {
        FAIL() << "Unexpected exception type";
    }
}

TEST(TestParallelFutureUtils, NonStandardWorkerFailureDrainsDelayedSibling)
{
    auto failure = RunWithBlockedSibling(std::make_exception_ptr(17));
    ASSERT_TRUE(failure);
    try
    {
        std::rethrow_exception(failure);
    }
    catch (int value)
    {
        EXPECT_EQ(value, 17);
    }
    catch (...)
    {
        FAIL() << "Non-standard exception was not preserved";
    }
}

TEST(TestParallelFutureUtils, NonStandardSubmissionFailureDrainsAcceptedTasks)
{
    auto failure = RunWithBlockedSibling(nullptr, std::make_exception_ptr(23));
    ASSERT_TRUE(failure);
    try
    {
        std::rethrow_exception(failure);
    }
    catch (int value)
    {
        EXPECT_EQ(value, 23);
    }
    catch (...)
    {
        FAIL() << "Non-standard exception was not preserved";
    }
}

TEST(TestParallelFutureUtils, SuccessDrainsDelayedSibling) { EXPECT_FALSE(RunWithBlockedSibling(nullptr)); }

TEST(TestParallelFutureUtils, LaterWorkerFailureDoesNotReplaceFirstFailure)
{
    EXPECT_THROW(ascend::utils::RunParallelAndWait(2,
                                                   [](auto &futures)
                                                   {
                                                       std::packaged_task<void()> first([] { throw 31; });
                                                       std::packaged_task<void()> second(
                                                           [] { throw std::runtime_error("later worker"); });
                                                       futures.emplace_back(first.get_future());
                                                       futures.emplace_back(second.get_future());
                                                       first();
                                                       second();
                                                   }),
                 int);
}

TEST(TestParallelFutureUtils, EmptySubmissionSucceeds)
{
    bool submitted = false;
    EXPECT_NO_THROW(ascend::utils::RunParallelAndWait(0,
                                                      [&](auto &futures)
                                                      {
                                                          EXPECT_TRUE(futures.empty());
                                                          submitted = true;
                                                      }));
    EXPECT_TRUE(submitted);
}

TEST(TestParallelFutureUtils, FailureBeforeAnyTaskIsAcceptedIsPreserved)
{
    EXPECT_THROW(ascend::utils::RunParallelAndWait(2, [](auto &) { throw std::runtime_error("first enqueue"); }),
                 std::runtime_error);
}

}  // namespace

int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
