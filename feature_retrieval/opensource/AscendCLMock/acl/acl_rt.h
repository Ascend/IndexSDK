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

#ifndef LIBASCENDCL_ACL_RT_H
#define LIBASCENDCL_ACL_RT_H

#include <cstdint>
#include <cstddef>
#include "acl_base.h"

#ifdef __cplusplus
extern "C" {
#endif

enum aclrtRunMode {
    ACL_DEVICE,
    ACL_HOST,
};

enum aclrtMemMallocPolicy {
    ACL_MEM_MALLOC_HUGE_FIRST,
    ACL_MEM_MALLOC_HUGE_ONLY,
    ACL_MEM_MALLOC_NORMAL_ONLY,
    ACL_MEM_MALLOC_HUGE_FIRST_P2P,
    ACL_MEM_MALLOC_HUGE_ONLY_P2P,
    ACL_MEM_MALLOC_NORMAL_ONLY_P2P,
};

enum aclrtMemcpyKind {
    ACL_MEMCPY_HOST_TO_HOST,
    ACL_MEMCPY_HOST_TO_DEVICE,
    ACL_MEMCPY_DEVICE_TO_HOST,
    ACL_MEMCPY_DEVICE_TO_DEVICE,
};

typedef enum aclrtCallbackBlockType {
    ACL_CALLBACK_NO_BLOCK,
    ACL_CALLBACK_BLOCK,
} aclrtCallbackBlockType;

typedef void (*aclrtCallback)(void *userData);

aclError aclrtMalloc(void **devPtr, size_t size, aclrtMemMallocPolicy policy);
aclError aclrtMallocHost(void **hostPtr, size_t size);
aclError aclrtMemcpy(void *dst, size_t destMax, const void *src, size_t count, aclrtMemcpyKind kind);
aclError aclrtMemset(void *devPtr, size_t maxCount, int32_t value, size_t count);
aclError aclrtFree(void *devPtr);
aclError aclrtFreeHost(void *hostPtr);
aclError aclrtMemcpyAsync(void *dst,
                          size_t destMax,
                          const void *src,
                          size_t count,
                          aclrtMemcpyKind kind,
                          aclrtStream stream);

aclError aclrtSetDevice(int32_t deviceId);
aclError aclrtGetDevice(int32_t *deviceId);
aclError aclrtResetDevice(int32_t deviceId);
aclError aclrtGetDeviceCount(uint32_t *count);

aclError aclrtCreateStream(aclrtStream *stream);
aclError aclrtDestroyStream(aclrtStream stream);
aclError aclrtSynchronizeStream(aclrtStream stream);

aclError aclrtCreateContext(aclrtContext *context, int32_t deviceId);
aclError aclrtDestroyContext(aclrtContext context);
aclError aclrtGetCurrentContext(aclrtContext *context);
aclError aclrtSetCurrentContext(aclrtContext context);

aclError aclrtGetRunMode(aclrtRunMode *runMode);

aclError aclrtLaunchCallback(aclrtCallback fn, void *userData, aclrtCallbackBlockType blockType, void *stream);

aclError aclrtProcessReport(int32_t timeout);
aclError aclrtCreateStreamWithConfig(aclrtStream *stream, uint32_t priority, uint32_t flag);
aclError aclrtSubscribeReport(uint64_t threadId, aclrtStream stream);
aclError aclrtUnSubscribeReport(uint64_t threadId, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // LIBASCENDCL_ACL_RT_H
