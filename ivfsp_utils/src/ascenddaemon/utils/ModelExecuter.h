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


#ifndef ASCEND_MODEL_EXECUTER_INCLUDED
#define ASCEND_MODEL_EXECUTER_INCLUDED

#include <string>
#include <vector>

#include "acl/acl.h"

namespace ascendSearch {
class ModelExecuter {
public:
    // Load model from memory
    ModelExecuter(const void* model, size_t modelSize);

    ~ModelExecuter();

    aclDataType getInputDataType(size_t index) const
    {
        return aclmdlGetInputDataType(modelDesc, index);
    }

    aclDataType getOutputDataType(size_t index) const
    {
        return aclmdlGetOutputDataType(modelDesc, index);
    }

    inline size_t getNumInputs() const
    {
        return aclmdlGetNumInputs(modelDesc);
    }

    inline size_t getNumOutputs() const
    {
        return aclmdlGetNumOutputs(modelDesc);
    }

    size_t getInputNumDims(int index) const;

    size_t getOutputNumDims(int index) const;

    int64_t getInputDim(size_t index, size_t dimIndex) const;

    int64_t getOutputDim(size_t index, size_t dimIndex) const;

    // Model infer
    void execute(void *inputData, void* outputData);

private:
    void releaseResource(aclmdlDataset *input = nullptr, aclmdlDataset *output = nullptr,
                         aclDataBuffer *inputDb = nullptr, aclDataBuffer *outputDb = nullptr) const;

private:
    uint32_t modelId;
    aclmdlDesc *modelDesc = nullptr;

    std::vector<size_t> inputSizes;
    std::vector<size_t> outputSizes;
};
}
#endif // ASCEND_MODEL_EXECUTER_INCLUDED
