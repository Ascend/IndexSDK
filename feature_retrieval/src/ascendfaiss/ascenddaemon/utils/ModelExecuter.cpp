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


#include "ascenddaemon/utils/ModelExecuter.h"
#include "common/utils/AscendAssert.h"
#include "common/utils/LogUtils.h"

namespace ascend {
ModelExecuter::ModelExecuter(const void* model, size_t modelSize)
{
    // 1. load model
    aclError ret = aclmdlLoadFromMem(model, modelSize, &modelId);
    ASCEND_THROW_IF_NOT_MSG(ret == ACL_ERROR_NONE, "load model from memory failed");

    // 2. get model desc
    modelDesc = aclmdlCreateDesc();
    ASCEND_THROW_IF_NOT_MSG(modelDesc != nullptr, "create model description failed");
    ret = aclmdlGetDesc(modelDesc, modelId);
    ASCEND_THROW_IF_NOT_MSG(ret == ACL_ERROR_NONE, "get model description failed");

    // 3. get inputSizes and outputSize
    size_t inputNum = aclmdlGetNumInputs(modelDesc);
    for (size_t i = 0; i < inputNum; ++i) {
        inputSizes.push_back(aclmdlGetInputSizeByIndex(modelDesc, i));
    }
    size_t outputNum = aclmdlGetNumOutputs(modelDesc);
    for (size_t i = 0; i < outputNum; ++i) {
        outputSizes.push_back(aclmdlGetOutputSizeByIndex(modelDesc, i));
    }
}

ModelExecuter::~ModelExecuter()
{
    aclError ret = aclmdlUnload(modelId);
    if (ret != ACL_ERROR_NONE) {
        APP_LOG_ERROR("unload model failed, modelId is %u\n", modelId);
    }

    if (modelDesc != nullptr) {
        (void)aclmdlDestroyDesc(modelDesc);
        modelDesc = nullptr;
    }
}

size_t ModelExecuter::getInputNumDims(int index)
{
    aclmdlIODims dims;
    auto ret = aclmdlGetInputDims(modelDesc, index, &dims);
    ASCEND_THROW_IF_NOT_FMT(ret == ACL_ERROR_NONE, "get input dims failed, index is %d", index);

    return dims.dimCount;
}

size_t ModelExecuter::getOutputNumDims(int index)
{
    aclmdlIODims dims;
    aclError ret = aclmdlGetOutputDims(modelDesc, index, &dims);
    ASCEND_THROW_IF_NOT_FMT(ret == ACL_ERROR_NONE, "get output dims failed, index is %d", index);

    return dims.dimCount;
}

int64_t ModelExecuter::getInputDim(size_t index, size_t dimIndex)
{
    aclmdlIODims dims;
    auto ret = aclmdlGetInputDims(modelDesc, index, &dims);
    ASCEND_THROW_IF_NOT_FMT(ret == ACL_ERROR_NONE, "get input dims failed, index is %zu", index);
    ASCEND_THROW_IF_NOT_FMT(dimIndex < dims.dimCount, "get input dim failed, the dimCount=%d", dims.dimCount);

    return dims.dims[dimIndex];
}

int64_t ModelExecuter::getOutputDim(size_t index, size_t dimIndex)
{
    aclmdlIODims dims;
    aclError ret = aclmdlGetOutputDims(modelDesc, index, &dims);
    ASCEND_THROW_IF_NOT_FMT(ret == ACL_ERROR_NONE, "get output dims failed, index is %zu", index);
    ASCEND_THROW_IF_NOT_FMT(dimIndex < dims.dimCount, "get output dim failed, the dimCount=%d", dims.dimCount);

    return dims.dims[dimIndex];
}

void ModelExecuter::execute(void *inputData, void* outputData)
{
    // 1. create dataset and data buffer
    aclmdlDataset *input = aclmdlCreateDataset();
    ASCEND_THROW_IF_NOT_MSG(input != nullptr, "create intput dataset failed");

    aclmdlDataset *output = aclmdlCreateDataset();
    if (output == nullptr) {
        releaseResource(input);
        ASCEND_THROW_MSG("create output dataset failed");
    }

    aclDataBuffer *inputDb = aclCreateDataBuffer(inputData, inputSizes[0]);
    if (inputDb == nullptr) {
        releaseResource(input, output);
        ASCEND_THROW_MSG("create input data buffer failed");
    }

    aclDataBuffer *outputDb = aclCreateDataBuffer(outputData, outputSizes[0]);
    if (outputDb == nullptr) {
        releaseResource(input, output, inputDb);
        ASCEND_THROW_MSG("create output data buffer failed");
    }

    // 2. add data buffer to dataset
    aclError ret = aclmdlAddDatasetBuffer(input, inputDb);
    if (ret != ACL_ERROR_NONE) {
        releaseResource(input, output, inputDb, outputDb);
        ASCEND_THROW_MSG("add input dataset buffer failed");
    }

    ret = aclmdlAddDatasetBuffer(output, outputDb);
    if (ret != ACL_ERROR_NONE) {
        releaseResource(input, output, inputDb, outputDb);
        ASCEND_THROW_MSG("add output dataset buffer failed");
    }

    // 3. execute
    ret = aclmdlExecute(modelId, input, output);
    if (ret != ACL_ERROR_NONE) {
        releaseResource(input, output, inputDb, outputDb);
        ASCEND_THROW_FMT("execute model failed, modelId is %u", modelId);
    }

    // 4. destroy data buffer and dataset
    releaseResource(input, output, inputDb, outputDb);
}

void ModelExecuter::releaseResource(aclmdlDataset *input, aclmdlDataset *output,
                                    aclDataBuffer *inputDb, aclDataBuffer *outputDb)
{
    if (inputDb != nullptr) {
        aclDestroyDataBuffer(inputDb);
    }
    if (outputDb != nullptr) {
        aclDestroyDataBuffer(outputDb);
    }
    if (input != nullptr) {
        aclmdlDestroyDataset(input);
    }
    if (output != nullptr) {
        aclmdlDestroyDataset(output);
    }
}
}
