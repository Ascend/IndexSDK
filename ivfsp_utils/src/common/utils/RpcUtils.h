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


#ifndef ASCEND_RPC_UTILS_H
#define ASCEND_RPC_UTILS_H

#include <string>
#include <cstdio>
#include <cstdint>
#include <syslog.h>
#include <sys/time.h>

#include <common/ErrorCode.h>
#include <common/utils/AscendAssert.h>

namespace faiss {
namespace ascendSearch {

/* -------------------------- Shared types -------------------------- */
using client_id_t = int32_t;
using index_id_t = int32_t;

/* ------------------------------ Utils ------------------------------ */
inline uint64_t getTimeUs()
{
    struct timeval tv;
    gettimeofday(&tv, nullptr);
    const int secConverter = 1000;
    return tv.tv_sec * secConverter * secConverter + tv.tv_usec;
}

inline double getTimeSecondElapsed()
{
    static double startTimeUs = getTimeUs();
    const int secConverter = 1000;
    return (getTimeUs() - startTimeUs) / static_cast<double>(secConverter * secConverter);
}

#define RPC_RETURN_IF_NOT_OK(X)                                                                                     \
    do                                                                                                              \
    {                                                                                                               \
        auto ret = X;                                                                                               \
        if (ret) {                                                                                                  \
            resp.mutable_result()->set_errorcode(ret);                                                              \
            return 0;                                                                                               \
        }                                                                                                           \
    } while (false)

#define RPC_RETURN_IF_NOT(X, ERRCODE)                                                                               \
    do                                                                                                              \
    {                                                                                                               \
        if (!(X)) {                                                                                                 \
            resp.mutable_result()->set_errorcode(ERRCODE);                                                          \
            return 0;                                                                                               \
        }                                                                                                           \
    } while (false)

#define RPC_RETURN_IF_NOT_LOG(X, ERRCODE, MSG)                                                                      \
    do                                                                                                              \
    {                                                                                                               \
        if (!(X)) {                                                                                                 \
            aclAppLog(ACL_ERROR, __FUNCTION__, __FILE__, __LINE__, MSG);                                            \
            resp.mutable_result()->set_errorcode(ERRCODE);                                                          \
            return 0;                                                                                               \
        }                                                                                                           \
    } while (false)

#define RPC_RETURN_IF_NOT_FMT(X, ERRCODE, FMT, ...)                                                                 \
    do                                                                                                              \
    {                                                                                                               \
        if (!(X)) {                                                                                                 \
            aclAppLog(ACL_ERROR, __FUNCTION__, __FILE__, __LINE__, "Err: '%s' failed: " FMT, #X, __VA_ARGS__);      \
            resp.mutable_result()->set_errorcode(ERRCODE);                                                          \
            return 0;                                                                                               \
        }                                                                                                           \
    } while (false)

#define RPC_RETURN_FMT(ERRCODE, FMT, ...)                                                                           \
    do                                                                                                              \
    {                                                                                                               \
        aclAppLog(ACL_ERROR, __FUNCTION__, __FILE__, __LINE__, "Err: '%s' failed: " FMT, __VA_ARGS__);              \
        resp.mutable_result()->set_errorcode(ERRCODE);                                                              \
        return 0;                                                                                                   \
    } while (false)

#define HDC_RETURN_IF_NOT(X)                                                                                        \
    do                                                                                                              \
    {                                                                                                               \
        if (!(X)) {                                                                                                 \
            return HDC_RPC_ERROR_ERROR;                                                                             \
        }                                                                                                           \
    } while (false)

#define HDC_RETURN_IF_NOT_LOG(X, MSG)                                                                               \
    do                                                                                                              \
    {                                                                                                               \
        if (!(X)) {                                                                                                 \
            aclAppLog(ACL_ERROR, __FUNCTION__, __FILE__, __LINE__, MSG);                                            \
            return HDC_RPC_ERROR_ERROR;                                                                             \
        }                                                                                                           \
    } while (false)

#define HDC_RETURN_IF_NOT_FMT(X, FMT, ...)                                                                          \
    do                                                                                                              \
    {                                                                                                               \
        if (!(X)) {                                                                                                 \
            aclAppLog(ACL_ERROR, __FUNCTION__, __FILE__, __LINE__, "Err: '%s' failed: " FMT, #X, __VA_ARGS__);      \
            return HDC_RPC_ERROR_ERROR;                                                                             \
        }                                                                                                           \
    } while (false)
}  // namespace ascendSearch
}  // namespace faiss

#endif  // ASCEND_RPC_UTILS_H
