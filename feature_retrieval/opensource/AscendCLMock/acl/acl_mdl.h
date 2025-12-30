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

// acl_mdl 还需等待实现
#ifndef LIBASCENDCL_ACL_MDL_H
#define LIBASCENDCL_ACL_MDL_H

#include <stddef.h>
#include <stdint.h>

#include "acl_base.h"
#include "acl_rt.h"
#include "acl_op.h"

#ifdef __cplusplus
extern "C" {
#endif

#define ACL_MAX_DIM_CNT          128
#define ACL_MAX_TENSOR_NAME_LEN  128
#define ACL_MAX_HW_NUM           128
#define ACL_MAX_BATCH_NUM        128

typedef struct aclmdlDataset {
    int tmp;
} aclmdlDataset;

typedef struct aclmdlDesc {
    int tmp;
} aclmdlDesc;

typedef struct aclmdlAIPP {
    int tmp;
} aclmdlAIPP;

typedef struct aclAippExtendInfo {
    int tmp;
} aclAippExtendInfo;

typedef struct aclmdlConfigHandle {
    int tmp;
} aclmdlConfigHandle;

typedef struct aclmdlIODims {
    char name[ACL_MAX_TENSOR_NAME_LEN]; /**< tensor name */
    size_t dimCount;  /**< dim array count */
    int64_t dims[ACL_MAX_DIM_CNT]; /**< dim data array */
} aclmdlIODims;

typedef struct aclmdlHW {
    size_t hwCount;                 /* *< height&width array count */
    uint64_t hw[ACL_MAX_HW_NUM][2]; /* *< height&width data array */
} aclmdlHW;

typedef enum {
    ACL_MDL_PRIORITY_INT32 = 0,
    ACL_MDL_LOAD_TYPE_SIZET,
    ACL_MDL_PATH_PTR,     /* *< pointer to model load path with deep copy */
    ACL_MDL_MEM_ADDR_PTR, /* *< pointer to model memory with shallow copy */
    ACL_MDL_MEM_SIZET,
    ACL_MDL_WEIGHT_ADDR_PTR, /* *< pointer to weight memory of model with shallow copy */
    ACL_MDL_WEIGHT_SIZET,
    ACL_MDL_WORKSPACE_ADDR_PTR, /* *< pointer to worksapce memory of model with shallow copy */
    ACL_MDL_WORKSPACE_SIZET,
    ACL_MDL_INPUTQ_NUM_SIZET,
    ACL_MDL_INPUTQ_ADDR_PTR, /* *< pointer to inputQ with shallow copy */
    ACL_MDL_OUTPUTQ_NUM_SIZET,
    ACL_MDL_OUTPUTQ_ADDR_PTR, /* *< pointer to outputQ with shallow copy */
    ACL_MDL_WORKSPACE_MEM_OPTIMIZE
} aclmdlConfigAttr;

typedef struct aclmdlBatch {
    size_t batchCount;                 /* *< batch array count */
    uint64_t batch[ACL_MAX_BATCH_NUM]; /* *< batch data array */
} aclmdlBatch;

aclDataType aclmdlGetInputDataType(const aclmdlDesc *modelDesc, size_t index);
aclDataType aclmdlGetOutputDataType(const aclmdlDesc *modelDesc, size_t index);
size_t aclmdlGetNumInputs(aclmdlDesc *modelDesc);
size_t aclmdlGetNumOutputs(aclmdlDesc *modelDesc);
size_t aclmdlGetInputSizeByIndex(aclmdlDesc *modelDesc, size_t index);
size_t aclmdlGetOutputSizeByIndex(aclmdlDesc *modelDesc, size_t index);
aclError aclmdlGetInputDims(const aclmdlDesc *modelDesc, size_t index, aclmdlIODims *dims);
aclError aclmdlGetOutputDims(const aclmdlDesc *modelDesc, size_t index, aclmdlIODims *dims);
aclmdlDataset *aclmdlCreateDataset();
aclError aclmdlDestroyDataset(const aclmdlDataset *dataset);
aclError aclmdlAddDatasetBuffer(aclmdlDataset *dataset, aclDataBuffer *dataBuffer);
aclError aclmdlExecute(uint32_t modelId, const aclmdlDataset *input, aclmdlDataset *output);
aclError aclmdlExecuteAsync(uint32_t modelId, const aclmdlDataset *input, aclmdlDataset *output, aclrtStream stream);
aclError aclmdlUnload(uint32_t modelId);
aclError aclmdlLoadFromMem(const void *model,  size_t modelSize,
                           uint32_t *modelId);
aclmdlDesc *aclmdlCreateDesc();
aclError aclmdlDestroyDesc(aclmdlDesc *modelDesc);
aclError aclmdlGetDesc(aclmdlDesc *modelDesc, uint32_t modelId);

aclError aclmdlSetDynamicBatchSize(uint32_t modelId, aclmdlDataset *dataset, size_t index, uint64_t batchSize);

aclError aclmdlSetInputDynamicDims(uint32_t modelId, aclmdlDataset *dataset, size_t index, const aclmdlIODims *dims);
aclError aclmdlGetDynamicHW(const aclmdlDesc *modelDesc, size_t index, aclmdlHW *hw);
aclError aclmdlDestroyConfigHandle(aclmdlConfigHandle *handle);
aclFormat aclmdlGetOutputFormat(const aclmdlDesc *modelDesc, size_t index);
aclmdlConfigHandle *aclmdlCreateConfigHandle();
aclError aclmdlGetDynamicBatch(const aclmdlDesc *modelDesc, aclmdlBatch *batch);
aclError aclmdlSetDatasetTensorDesc(aclmdlDataset *dataset, aclTensorDesc *tensorDesc, size_t index);

aclError aclmdlGetCurOutputDims(const aclmdlDesc *modelDesc, size_t index, aclmdlIODims *dims);
aclError aclmdlSetConfigOpt(aclmdlConfigHandle *handle, aclmdlConfigAttr attr, const void *attrValue, size_t valueSize);

size_t aclmdlGetDatasetNumBuffers(const aclmdlDataset *dataset);
aclError aclmdlGetInputDynamicGearCount(const aclmdlDesc *modelDesc, size_t index, size_t *gearCount);
aclError aclmdlLoadWithConfig(const aclmdlConfigHandle *handle, uint32_t *modelId);
aclDataBuffer *aclmdlGetDatasetBuffer(const aclmdlDataset *dataset, size_t index);
aclError aclmdlSetDynamicHWSize(uint32_t modelId, aclmdlDataset *dataset, size_t index, uint64_t height,
    uint64_t width);
aclTensorDesc *aclmdlGetDatasetTensorDesc(const aclmdlDataset *dataset, size_t index);
aclError aclmdlQuerySize(const char *fileName, size_t *workSize, size_t *weightSize);
aclError aclmdlLoadFromFile(const char *modelPath, uint32_t *modelId);
aclFormat aclmdlGetInputFormat(const aclmdlDesc *modelDesc, size_t index);
aclError aclmdlQuerySizeFromMem(const void *model, size_t modelSize, size_t *workSize, size_t *weightSize);
aclError aclmdlGetInputDynamicDims(const aclmdlDesc *modelDesc, size_t index, aclmdlIODims *dims, size_t gearCount);
aclError aclmdlGetInputIndexByName(const aclmdlDesc *modelDesc, const char *name, size_t *index);
aclError aclmdlGetInputDimsV2(const aclmdlDesc *modelDesc, size_t index, aclmdlIODims *dims);

#ifdef __cplusplus
}
#endif

#endif // LIBASCENDCL_ACL_MDL_H
