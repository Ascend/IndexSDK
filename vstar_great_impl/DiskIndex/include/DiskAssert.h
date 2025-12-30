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


#pragma once

#include <string>
#include "DiskException.h"

/* to avoid using snprint_f, we use c++17's features to format a string with indefinite number of arguments */
template <typename... Args>
std::string FormatMessage(Args&&... args)
{
    std::ostringstream oss;
    (oss << ... << std::forward<Args>(args));  // C++17 fold expression
    return oss.str();
}

namespace diskann_pro {

#define DISK_THROW_MSG(MSG)                                                                                   \
    do {                                                                                                        \
        throw diskann_pro::DiskException(MSG, __PRETTY_FUNCTION__, __FILE__, __LINE__);                          \
    } while (false)

#define DISK_THROW_FMT(FMT, ...)                                                                              \
    do {                                                                                                        \
        std::string sTmp = FormatMessage(FMT, __VA_ARGS__);                                                     \
        throw diskann_pro::DiskException(sTmp, __PRETTY_FUNCTION__, __FILE__, __LINE__);                          \
    } while (false)


// Exceptions thrown upon a conditional failure
#define DISK_THROW_IF_NOT(X)                                                                                  \
    do                                                                                                          \
    {                                                                                                           \
        if (!(X)) {                                                                                             \
            DISK_THROW_FMT("Error %s failed: ", #X);                                                             \
        }                                                                                                       \
    } while (false)

#define DISK_THROW_IF_NOT_MSG(X, MSG)                                                                         \
    do                                                                                                          \
    {                                                                                                           \
        if (!(X)) {                                                                                             \
            DISK_THROW_MSG("Error " #X " failed: " MSG);                                                        \
        }                                                                                                       \
    } while (false)

#define DISK_THROW_IF_NOT_FMT(X, FMT, ...)                                                                     \
    do                                                                                                          \
    {                                                                                                           \
        if (!(X)) {                                                                                             \
            DISK_THROW_FMT("Error %s failed: " FMT, #X, __VA_ARGS__);                                           \
        }                                                                                                       \
    } while (false)
}