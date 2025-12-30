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


#ifndef ASCEND_ERROR_CODE_H
#define ASCEND_ERROR_CODE_H

#include "acl/acl.h"

namespace ascend {
using APP_ERROR = int;

enum {
    APP_ERR_OK = 0,

    // define the error code of ACL, this is same with the aclError which is error code of ACL API
    // Error codes 1001~1999 are reserved for the ACL. Do not add other error codes.
    APP_ACL_BASE = 1000,
    APP_ERR_ACL_BAD_ALLOC,                                  // ACL: memory allocation failed
    APP_ERR_ACL_LOAD_MODEL_FAILED,                          // ACL: model load failed
    APP_ERR_ACL_UNLOAD_MODEL_FAILED,                        // ACL: model unload failed
    APP_ERR_ACL_GET_DIMS_FAILURE,                           // ACL: failed to get dimension information
    APP_ERR_ACL_CREATE_MODEL_DESC_FAILED,                   // ACL: failed to create model information
    APP_ERR_ACL_DESTORY_MODEL_DESC_FAILED,                  // ACL: failed to destory model information
    APP_ERR_ACL_GET_MODEL_DESC_FAILED,                      // ACL: failed to get model information
    APP_ERR_ACL_MODEL_EXEC_FAILURE,                         // ACL: model inference failed
    APP_ERR_ACL_CREATE_DATA_SET_FAILED,                     // ACL: failed to create dataset
    APP_ERR_ACL_DESTORY_DATA_SET_FAILED,                    // ACL: failed to destory dataset
    APP_ERR_ACL_CREATE_DATA_BUF_FAILED,                     // ACL: failed to create databuffer
    APP_ERR_ACL_DESTORY_DATA_BUF_FAILED,                    // ACL: failed to destory databuffer
    APP_ERR_ACL_ADD_DATA_BUF_FAILED,                        // ACL: failed to add databuffer to dataset
    APP_ERR_ACL_GET_DATA_BUF_ADDR_NULL,                     // ACL: failed to get the address of databuffer
    APP_ERR_ACL_OP_NOT_FOUND,                               // ACL: op model not found
    APP_ERR_ACL_OP_LOAD_MODEL_FAILED,                       // ACL: op model load failed
    APP_ERR_ACL_OP_EXEC_FAILED,                             // ACL: op model execute failed, not running in aicore
    APP_ERR_ACL_SET_DEVICE_FAILED,                          // ACL: failed to set device
    APP_ERR_ACL_END,                                        // ACL: end of ACL ERR

    // define the APP error code, range: 2001~2999
    APP_BASE = 2000,
    APP_ERR_INVALID_PARAM,                                  // invalid parameter
    APP_ERR_IVALID_HDC_DATA,                                // invalid HDC transport data
    APP_ERR_INDEX_NOT_FOUND,                                // index not found
    APP_ERR_TRANSFORMER_NOT_FOUND,                          // transformer not found
    APP_ERR_CLUSTERING_NOT_FOUND,                           // clustering not found
    APP_ERR_INFERENCE_NOT_FOUND,                            // inference not found
    APP_ERR_REQUEST_ERROR,                                  // request parameter error
    APP_ERR_NOT_IMPLEMENT,                                  // method not implement
    APP_ERR_ILLEGAL_OPERATION,                             // illegal operation
    APP_ERR_INNER_ERROR,                                    // internal error
    APP_ERR_TIMEOUT,                                        // timout
    APP_CREATE_INDEX_FAILED,                                // create index failed
    APP_CREATE_TRANSFORM_FAILED,                            // create index failed
    APP_CREATE_INFERENCE_FAILED,                            // create index failed
    APP_ERR_INVALID_TABLE_INDEX,                            // invalid index for mapping table
    APP_ERR_INDEX_NOT_INIT,                                 // index not initialize
    APP_ERR_END                                             // end of APP ERR
};

#define APPERR_RETURN_IF(X, ERRCODE)                                                                                \
    do                                                                                                              \
    {                                                                                                               \
        if (X) {                                                                                                    \
            return ERRCODE;                                                                                         \
        }                                                                                                           \
    } while (false)

#define APPERR_RETURN_IF_FMT(X, ERRCODE, FMT, ...)                                                                  \
    do                                                                                                              \
    {                                                                                                               \
        if (X) {                                                                                                    \
            aclAppLog(ACL_ERROR, __FUNCTION__, __FILE__, __LINE__, "Err: '%s' failed: " FMT, #X, __VA_ARGS__);      \
            return ERRCODE;                                                                                         \
        }                                                                                                           \
    } while (false)

#define APPERR_RETURN_IF_NOT(X, ERRCODE)                                                                            \
    do                                                                                                              \
    {                                                                                                               \
        if (!(X)) {                                                                                                 \
            return ERRCODE;                                                                                         \
        }                                                                                                           \
    } while (false)

#define APPERR_RETURN_IF_NOT_OK(X)                                                                                  \
    do                                                                                                              \
    {                                                                                                               \
        auto ret = X;                                                                                               \
        if (ret) {                                                                                                  \
            return ret;                                                                                             \
        }                                                                                                           \
    } while (false)

#define APPERR_RETURN_IF_NOT_LOG(X, ERRCODE, MSG)                                                                   \
    do                                                                                                              \
    {                                                                                                               \
        if (!(X)) {                                                                                                 \
            aclAppLog(ACL_ERROR, __FUNCTION__, __FILE__, __LINE__, MSG);                                            \
            return ERRCODE;                                                                                         \
        }                                                                                                           \
    } while (false)

#define APPERR_RETURN_IF_NOT_FMT(X, ERRCODE, FMT, ...)                                                              \
    do                                                                                                              \
    {                                                                                                               \
        if (!(X)) {                                                                                                 \
            aclAppLog(ACL_ERROR, __FUNCTION__, __FILE__, __LINE__, "Err: '%s' failed: " FMT, #X, __VA_ARGS__);      \
            return ERRCODE;                                                                                         \
        }                                                                                                           \
    } while (false)
} // namespace ascend

#endif // ASCEND_ERROR_CODE_H
