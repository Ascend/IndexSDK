/*
 * -------------------------------------------------------------------------
 * This file is part of the IndexSDK project.
 * Copyright (c) 2026 Huawei Technologies Co.,Ltd.
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

#ifndef ASCEND_UT_TEST_ENV_GUARD_INCLUDED
#define ASCEND_UT_TEST_ENV_GUARD_INCLUDED

#include <cstdlib>
#include <string>

namespace ascend
{

// Restores an environment variable when it goes out of scope.
//
// A bare setenv() in SetUp() leaks into every later test in the same binary, and the
// memory pool freezes settings such as its chunk size when it is first created, so a
// leaked value lets test order decide what unrelated suites observe. It also clobbers
// whatever the person running the suite configured.
class ScopedEnv
{
   public:
    ScopedEnv(const char *name, const char *value) : name_(name)
    {
        const char *previous = std::getenv(name);
        if (previous != nullptr)
        {
            had_ = true;
            previous_ = previous;
        }
        (void)setenv(name, value, 1);
    }

    ~ScopedEnv()
    {
        if (had_)
        {
            (void)setenv(name_, previous_.c_str(), 1);
        }
        else
        {
            (void)unsetenv(name_);
        }
    }

    ScopedEnv(const ScopedEnv &) = delete;
    ScopedEnv &operator=(const ScopedEnv &) = delete;

   private:
    const char *name_;
    std::string previous_;
    bool had_ = false;
};
}  // namespace ascend

#endif  // ASCEND_UT_TEST_ENV_GUARD_INCLUDED
