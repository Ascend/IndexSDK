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

#ifndef LIBASCENDCL_ACL_BASE_H
#define LIBASCENDCL_ACL_BASE_H

#include <cstdint>
#include <cstddef>
#include <vector>

#ifdef __cplusplus
extern "C" {
#endif

typedef int aclError;
typedef uint16_t aclFloat16;
static const int ACL_ERROR_NONE = 0;
static const int ACL_SUCCESS = 0;

static const int ACL_ERROR_UNINITIALIZE = 100001;
static const int ACL_ERROR_REPEAT_INITIALIZE = 100002;
static const int ACL_ERROR_BAD_ALLOC = 200000;
static const int ACL_ERROR_STORAGE_OVER_LIMIT = 300000;
static const int ACL_ERROR_INTERNAL_ERROR = 500000;
static const int ACL_ERROR_FAILURE = 500001;

typedef enum {
    ACL_DT_UNDEFINED = -1,
    ACL_FLOAT = 0,
    ACL_FLOAT16 = 1,
    ACL_INT8 = 2,
    ACL_INT32 = 3,
    ACL_UINT8 = 4,
    ACL_INT16 = 6,
    ACL_UINT16 = 7,
    ACL_UINT32 = 8,
    ACL_INT64 = 9,
    ACL_UINT64 = 10,
    ACL_DOUBLE = 11,
    ACL_BOOL = 12,
    ACL_STRING = 13,
    ACL_COMPLEX64 = 16,
    ACL_COMPLEX128 = 17,
    ACL_BF16 = 27
} aclDataType;

typedef enum {
    ACL_FORMAT_UNDEFINED = -1,
    ACL_FORMAT_NCHW = 0,
    ACL_FORMAT_NHWC = 1,
    ACL_FORMAT_ND = 2,
    ACL_FORMAT_NC1HWC0 = 3,
    ACL_FORMAT_FRACTAL_Z = 4,
    ACL_FORMAT_NC1HWC0_C04 = 12,
    ACL_FORMAT_HWCN = 16,
    ACL_FORMAT_NDHWC = 27,
    ACL_FORMAT_FRACTAL_NZ = 29,
    ACL_FORMAT_NCDHW = 30,
    ACL_FORMAT_NDC1HWC0 = 32,
    ACL_FRACTAL_Z_3D = 33
} aclFormat;

typedef enum {
    ACL_DEBUG = 0,
    ACL_INFO = 1,
    ACL_WARNING = 2,
    ACL_ERROR = 3,
} aclLogLevel;

typedef enum {
    ACL_MEMTYPE_DEVICE = 0,
    ACL_MEMTYPE_HOST = 1,
    ACL_MEMTYPE_HOST_COMPILE_INDEPENDENT = 2
} aclMemType;

struct aclDataBuffer {
    void *data{nullptr};
    size_t size{0};
};

struct aclTensorDesc {
    aclDataType dataType{ACL_FLOAT};
    int numDims{0};
    const int64_t *dims{nullptr};
    aclFormat format{ACL_FORMAT_ND};
};

struct tagAclrtContext;
using aclrtContext = tagAclrtContext *;

struct tagAclrtStream {
    int32_t deviceId{0};
    aclrtContext ctxt {nullptr};
};
using aclrtStream = tagAclrtStream *;

struct tagAclrtContext {
    int32_t deviceId{0};
    aclrtStream m_defaultStream{};
    std::vector<aclrtStream> streams{};
};

aclDataBuffer *aclCreateDataBuffer(void *data, size_t size);
aclError aclDestroyDataBuffer(const aclDataBuffer *dataBuffer);

aclTensorDesc *aclCreateTensorDesc(aclDataType dataType,
                                   int numDims,
                                   const int64_t *dims,
                                   aclFormat format);
void aclDestroyTensorDesc(const aclTensorDesc *desc);
aclError aclGetTensorDescDimV2(const aclTensorDesc *desc, size_t index, int64_t *dimSize);
size_t aclGetTensorDescNumDims(const aclTensorDesc *desc);
size_t aclGetTensorDescElementCount(const aclTensorDesc *desc);
size_t aclGetTensorDescSize(const aclTensorDesc *desc);

const char *aclrtGetSocName();

aclFloat16 aclFloatToFloat16(float value);
float aclFloat16ToFloat(aclFloat16 value);

void aclAppLog(aclLogLevel logLevel, const char *func, const char *file, uint32_t line,
               const char *fmt, ...);

#define ACL_APP_LOG(level, fmt, ...) \
    aclAppLog(level, __FUNCTION__, __FILE__, __LINE__, fmt, ##__VA_ARGS__)

#ifdef __cplusplus
}
#endif

#endif // LIBASCENDCL_ACL_BASE_H
