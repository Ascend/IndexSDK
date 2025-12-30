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

#ifndef LIBASCENDCL_ACL_OP_H
#define LIBASCENDCL_ACL_OP_H

#include <map>
#include "acl_base.h"
#include "acl_rt.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct aclopHandle aclopHandle;
typedef struct aclopAttr aclopAttr;
typedef struct aclopKernelDesc aclopKernelDesc;

typedef void (*aclDataDeallocator)(void *data, size_t length);

struct aclopAttr {
    int attr;
    std::map<std::string, float> attrs;
};

struct aclopHandle {
    const char *opName{nullptr};
    int numInputs{0};
    int numOutPuts{0};
    aclTensorDesc *inputDesc{nullptr};
    aclDataBuffer *inputData{nullptr};
    aclTensorDesc *outputDesc{nullptr};
    aclDataBuffer *outputData{nullptr};
    aclopAttr *opAttr{nullptr};
};

typedef enum aclEngineType {
    ACL_ENGINE_SYS,
    ACL_ENGINE_AICORE,
    ACL_ENGINE_VECTOR,
} aclopEngineType;

typedef aclError (*aclopCompileFunc)(int numInputs, const aclTensorDesc * const inputDesc[], int numOutputs,
    const aclTensorDesc * const outputDesc[], const aclopAttr *opAttr, aclopKernelDesc *aclopKernelDesc);

aclopAttr *aclopCreateAttr();
void aclopDestroyAttr(const aclopAttr *attr);
aclError aclopSetModelDir(const char *modelDir);

aclError aclopCreateHandle(const char *opType,
                           int numInputs,
                           const aclTensorDesc *const inputDesc[],
                           int numOutputs,
                           const aclTensorDesc *const outputDesc[],
                           const aclopAttr *opAttr,
                           aclopHandle **handle);

aclError aclopExecWithHandle(aclopHandle *handle,
                             int numInputs,
                             const aclDataBuffer *const inputs[],
                             int numOutputs,
                             aclDataBuffer *const outputs[],
                             aclrtStream stream);

void aclopDestroyHandle(aclopHandle *handle);

aclError aclopExecuteV2(const char *opType, int numInputs, aclTensorDesc *inputDesc[], aclDataBuffer *inputs[],
    int numOutputs, aclTensorDesc *outputDesc[], aclDataBuffer *outputs[], aclopAttr *attr, void *stream);
aclError aclopSetAttrFloat(aclopAttr *attr, const char *attrName, float attrValue);
aclError aclopSetAttrInt(aclopAttr *attr, const char *attrName, int64_t attrValue);
aclError aclopSetKernelArgs(aclopKernelDesc *kernelDesc, const char *kernelId, uint32_t blockDim, const void *args,
    uint32_t argSize);
#ifdef __cplusplus
}
#endif

#endif // LIBASCENDCL_ACL_OP_H
