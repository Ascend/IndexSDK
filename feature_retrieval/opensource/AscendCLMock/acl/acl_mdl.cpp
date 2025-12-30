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

#include <cstdlib>
#include <string>
#include <cstring>
#include <cstdarg>
#include "acl.h"
#include "simu/AscendSimu.h"
#include "acl_mdl.h"

// acl_mdl 还需等待实现
#ifdef __cplusplus
extern "C" {
#endif

aclDataType aclmdlGetInputDataType(const aclmdlDesc *modelDesc, size_t index)
{
    return ACL_FLOAT;
}

aclDataType aclmdlGetOutputDataType(const aclmdlDesc *modelDesc, size_t index)
{
    return ACL_FLOAT;
}

size_t aclmdlGetNumInputs(aclmdlDesc *modelDesc)
{
    return 0;
}

size_t aclmdlGetNumOutputs(aclmdlDesc *modelDesc)
{
    return 0;
}

size_t aclmdlGetInputSizeByIndex(aclmdlDesc *modelDesc, size_t index)
{
    return 0;
}

size_t aclmdlGetOutputSizeByIndex(aclmdlDesc *modelDesc, size_t index)
{
    return 0;
}

aclError aclmdlGetInputDims(const aclmdlDesc *modelDesc, size_t index, aclmdlIODims *dims)
{
    return ACL_SUCCESS;
}

aclError aclmdlGetOutputDims(const aclmdlDesc *modelDesc, size_t index, aclmdlIODims *dims)
{
    return ACL_SUCCESS;
}

aclmdlDataset *aclmdlCreateDataset()
{
    return nullptr;
}

aclError aclmdlDestroyDataset(const aclmdlDataset *dataset)
{
    return ACL_SUCCESS;
}

aclError aclmdlAddDatasetBuffer(aclmdlDataset *dataset, aclDataBuffer *dataBuffer)
{
    return ACL_SUCCESS;
}

aclError aclmdlExecute(uint32_t modelId, const aclmdlDataset *input, aclmdlDataset *output)
{
    return ACL_SUCCESS;
}

aclError aclmdlExecuteAsync(uint32_t modelId, const aclmdlDataset *input, aclmdlDataset *output, aclrtStream stream)
{
    return ACL_SUCCESS;
}

aclError aclmdlUnload(uint32_t modelId)
{
    return ACL_SUCCESS;
}

aclError aclmdlLoadFromMem(const void *model, size_t modelSize, uint32_t *modelId)
{
    return ACL_SUCCESS;
}

aclmdlDesc *aclmdlCreateDesc()
{
    aclmdlDesc *mdl = new aclmdlDesc();
    return mdl;
}

aclError aclmdlDestroyDesc(aclmdlDesc *modelDesc)
{
    delete modelDesc;
    return ACL_SUCCESS;
}

aclError aclmdlGetDesc(aclmdlDesc *modelDesc, uint32_t modelId)
{
    return ACL_SUCCESS;
}

aclError aclmdlSetDynamicBatchSize(uint32_t modelId, aclmdlDataset *dataset, size_t index, uint64_t batchSize)
{
    return 0;
}

aclError aclmdlSetInputDynamicDims(uint32_t modelId, aclmdlDataset *dataset, size_t index, const aclmdlIODims *dims)
{
    return 0;
}

aclError aclmdlGetDynamicHW(const aclmdlDesc *modelDesc, size_t index, aclmdlHW *hw)
{
    return 0;
}

aclError aclmdlDestroyConfigHandle(aclmdlConfigHandle *handle)
{
    return 0;
}

aclFormat aclmdlGetOutputFormat(const aclmdlDesc *modelDesc, size_t index)
{
    return ACL_FORMAT_NCHW;
}

aclmdlConfigHandle *aclmdlCreateConfigHandle()
{
    aclmdlConfigHandle *handle = new aclmdlConfigHandle{ 1 };
    return handle;
}

aclError aclmdlGetDynamicBatch(const aclmdlDesc *modelDesc, aclmdlBatch *batch)
{
    batch->batchCount = 1;
    return 0;
}

aclError aclmdlSetDatasetTensorDesc(aclmdlDataset *dataset, aclTensorDesc *tensorDesc, size_t index)
{
    return 0;
}

aclError aclmdlGetCurOutputDims(const aclmdlDesc *modelDesc, size_t index, aclmdlIODims *dims)
{
    return 0;
}

aclError aclmdlSetConfigOpt(aclmdlConfigHandle *handle, aclmdlConfigAttr attr, const void *attrValue, size_t valueSize)
{
    return 0;
}

size_t aclmdlGetDatasetNumBuffers(const aclmdlDataset *dataset)
{
    return 0;
}

aclError aclmdlGetInputDynamicGearCount(const aclmdlDesc *modelDesc, size_t index, size_t *gearCount)
{
    return 0;
}

aclError aclmdlLoadWithConfig(const aclmdlConfigHandle *handle, uint32_t *modelId)
{
    return 0;
}

aclDataBuffer *aclmdlGetDatasetBuffer(const aclmdlDataset *dataset, size_t index)
{
    return 0;
}

aclError aclmdlSetDynamicHWSize(uint32_t modelId, aclmdlDataset *dataset, size_t index, uint64_t height, uint64_t width)
{
    return 0;
}

aclTensorDesc *aclmdlGetDatasetTensorDesc(const aclmdlDataset *dataset, size_t index)
{
    return nullptr;
}

aclError aclmdlQuerySize(const char *fileName, size_t *workSize, size_t *weightSize)
{
    return 0;
}

aclError aclmdlLoadFromFile(const char *modelPath, uint32_t *modelId)
{
    return 0;
}

aclError aclmdlGetInputIndexByName(const aclmdlDesc *modelDesc, const char *name, size_t *index)
{
    return 0;
}

aclFormat aclmdlGetInputFormat(const aclmdlDesc *modelDesc, size_t index)
{
    return ACL_FORMAT_NCHW;
}

aclError aclmdlQuerySizeFromMem(const void *model, size_t modelSize, size_t *workSize, size_t *weightSize)
{
    return 0;
}
aclError aclmdlGetInputDynamicDims(const aclmdlDesc *modelDesc, size_t index, aclmdlIODims *dims, size_t gearCount)
{
    return 0;
}

aclError aclmdlGetInputDimsV2(const aclmdlDesc *modelDesc, size_t index, aclmdlIODims *dims)
{
    dims->dimCount = 3;
    dims->dims[0] = 3;
    dims->dims[1] = 640;
    dims->dims[2] = 640;
    return 0;
}

#ifdef __cplusplus
}
#endif