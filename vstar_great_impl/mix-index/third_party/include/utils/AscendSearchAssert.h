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


#ifndef ASCENDSEARCH_ASCENDSEARCHASSERT_H
#define ASCENDSEARCH_ASCENDSEARCHASSERT_H

#include <cstdio>
#include <cstdlib>
#include <string>

#include "securec.h"
#include <cstring>

#include "AscendSearchException.h"


#define ASCENDSEARCH_ASSERT(X) do { \
    if (!(X)) {                                               \
        fprintf(stderr,                                       \
                "AscendSearch assertion '%s' failed in %s "   \
                "at %s:%d\n",                                 \
                #X, __PRETTY_FUNCTION__, __FILE__, __LINE__); \
        abort();                                              \
    }                                                         \
} while (false)

// /
// / Exceptions for returning user errors
// /

#define ASCENDSEARCH_THROW_MSG(MSG) do { \
    throw ascendsearch::AscendSearchException(MSG, __PRETTY_FUNCTION__, __FILE__, __LINE__); \
} while (false)

#define ASCENDSEARCH_THROW_FMT(FMT, ...) do { \
    int size = 1024;                                                                        \
    std::string __s(size, '\0');                                                            \
    while (snprintf_s(&__s[0], __s.size(), __s.size(), FMT, __VA_ARGS__) < 0) {             \
        size = size * 2;                                                                    \
        __s.resize(size, '\0');                                                             \
    }                                                                                       \
    throw ascendsearch::AscendSearchException(__s, __PRETTY_FUNCTION__, __FILE__, __LINE__); \
} while (false)

// /
// / Exceptions thrown upon a conditional failure
// /

#define ASCENDSEARCH_THROW_IF_NOT(X) do { \
    if (!(X)) {                                           \
        ASCENDSEARCH_THROW_FMT("Error: '%s' failed", #X); \
    }                                                     \
} while (false)

#define ASCENDSEARCH_THROW_IF_NOT_MSG(X, MSG) do { \
    if (!(X)) {                                                 \
        ASCENDSEARCH_THROW_FMT("Error: '%s' failed: " MSG, #X); \
    }                                                           \
} while (false)

#define ASCENDSEARCH_THROW_IF_NOT_FMT(X, FMT, ...) do { \
    if (!(X)) {                                                              \
        ASCENDSEARCH_THROW_FMT("Error: '%s' failed: " FMT, #X, __VA_ARGS__); \
    }                                                                        \
} while (false)

#endif  // ASCENDSEARCH_ASCENDSEARCHASSERT_H
