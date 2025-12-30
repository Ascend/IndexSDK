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


#ifndef ASCEND_MODEL_INFERENCE_INCLUDED
#define ASCEND_MODEL_INFERENCE_INCLUDED

#include <vector>
#include <memory>

#include "common/ErrorCode.h"
#include "ascenddaemon/utils/ModelExecuter.h"

namespace ascend {
class ModelInference {
public:
    ModelInference(const void* model, size_t modelSize);

    ~ModelInference();

    // inference data without inputType and outputType, leave it to the model to identify
    // the model handle `batch` vectors each time, the inputData and outputData must be multiples of `batch` vectors
    APP_ERROR Infer(size_t n, char* inputData, char* outputData);

public:
    // Input data type
    int inputType;

    // Output data type
    int outputType;

    // Input data dimension
    int dimIn;

    // Output data dimension
    int dimOut;

    // Batch size for inference
    int batch;

private:
    std::unique_ptr<ModelExecuter> modelExecuter;

    size_t inputLen;

    size_t outputLen;
};
} // namespace ascend

#endif // ASCEND_MODEL_INFERENCE_INCLUDED
