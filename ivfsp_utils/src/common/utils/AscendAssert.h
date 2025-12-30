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


#ifndef ASCEND_ASSERT_H_INCLUDED
#define ASCEND_ASSERT_H_INCLUDED

#include <common/utils/AscendException.h>
#include <string>
#include <securec.h>

#include "acl/acl.h"


// Exceptions for returning user errors
#define ASCEND_THROW_MSG(MSG)                                                                                   \
    do {                                                                                                        \
        throw ::ascendSearch::AscendException(MSG, __PRETTY_FUNCTION__, __FILE__, __LINE__);                          \
    } while (false)

#define ASCEND_THROW_FMT(FMT, ...)                                                                              \
    do {                                                                                                        \
        aclAppLog(ACL_ERROR, __FUNCTION__, __FILE__, __LINE__, FMT, __VA_ARGS__);                               \
        int size = 1024;                                                                                        \
        std::string sTmp(size, '\0');                                                                            \
        while (snprintf_s(&sTmp[0], sTmp.size(), sTmp.size(), FMT, __VA_ARGS__) < 0) {                             \
            size = size * 2;                                                                                    \
            sTmp.resize(size, '\0');                                                                             \
        }                                                                                                       \
        throw ::ascendSearch::AscendException(sTmp, __PRETTY_FUNCTION__, __FILE__, __LINE__);                          \
    } while (false)


// Exceptions thrown upon a conditional failure
#define ASCEND_THROW_IF_NOT(X)                                                                                  \
    do                                                                                                          \
    {                                                                                                           \
        if (!(X)) {                                                                                             \
            aclAppLog(ACL_ERROR, __FUNCTION__, __FILE__, __LINE__, "Error: '%s' failed", #X);                   \
            ASCEND_THROW_FMT("Error: '%s' failed", #X);                                                         \
        }                                                                                                       \
    } while (false)

#define ASCEND_THROW_IF_NOT_CODE(X, CODE)                                                                       \
    do                                                                                                          \
    {                                                                                                           \
        if (!(X)) {                                                                                             \
            CODE;                                                                                               \
            aclAppLog(ACL_ERROR, __FUNCTION__, __FILE__, __LINE__, "Error: '%s' failed", #X);                   \
            ASCEND_THROW_FMT("Error: '%s' failed", #X);                                                         \
        }                                                                                                       \
    } while (false)

#define ASCEND_THROW_IF_NOT_MSG(X, MSG)                                                                         \
    do                                                                                                          \
    {                                                                                                           \
        if (!(X)) {                                                                                             \
            aclAppLog(ACL_ERROR, __FUNCTION__, __FILE__, __LINE__, "Error: '%s' failed", #X);                   \
            ASCEND_THROW_FMT("Error: '%s' failed: " MSG, #X);                                                   \
        }                                                                                                       \
    } while (false)

#define ASCEND_THROW_IF_NOT_FMT(X, FMT, ...)                                                                    \
    do                                                                                                          \
    {                                                                                                           \
        if (!(X)) {                                                                                             \
            aclAppLog(ACL_ERROR, __FUNCTION__, __FILE__, __LINE__, "Error: '%s' failed", #X, __VA_ARGS__);      \
            ASCEND_THROW_FMT("Error: '%s' failed: " FMT, #X, __VA_ARGS__);                                      \
        }                                                                                                       \
    } while (false)

#define ASCEND_EXC_IF_NOT(X, CODE)                                                                              \
    do                                                                                                          \
    {                                                                                                           \
        if (!(X)) {                                                                                             \
            CODE;                                                                                               \
        }                                                                                                       \
    } while (false)

#define ASCEND_EXC_IF_NOT_LOG(X, CODE, MSG)                                                                     \
    do                                                                                                          \
    {                                                                                                           \
        if (!(X)) {                                                                                             \
            aclAppLog(ACL_ERROR, __FUNCTION__, __FILE__, __LINE__, MSG);                                        \
            CODE;                                                                                               \
        }                                                                                                       \
    } while (false)

#define ASCEND_EXC_IF_NOT_FMT(X, CODE, FMT, ...)                                                                \
    do                                                                                                          \
    {                                                                                                           \
        if (!(X)) {                                                                                             \
            aclAppLog(ACL_ERROR, __FUNCTION__, __FILE__, __LINE__, "Err: '%s' failed: " FMT, #X, __VA_ARGS__);  \
            CODE;                                                                                               \
        }                                                                                                       \
    } while (false)

#endif
