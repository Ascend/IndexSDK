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


#include "ascenddaemon/utils/ModelInference.h"
#include "ascenddaemon/utils/StaticUtils.h"
#include "common/utils/AscendAssert.h"
#include "common/utils/DataType.h"
#include "common/utils/CommonUtils.h"

#include "acl/acl.h"

using namespace ::faiss::ascend;

namespace ascend {
ModelInference::ModelInference(const void* model, size_t modelSize)
{
    modelExecuter = CREATE_UNIQUE_PTR(ModelExecuter, model, modelSize);
    this->inputType = modelExecuter->getInputDataType(0);
    this->outputType = modelExecuter->getOutputDataType(0);
    this->batch = modelExecuter->getInputDim(0, 0);
    this->dimIn = modelExecuter->getInputDim(0, 1);
    this->dimOut = modelExecuter->getOutputDim(0, 1);

    this->inputLen = static_cast<size_t>(batch * dimIn * getTypeSize(inputType));
    this->outputLen = static_cast<size_t>(batch * dimOut * getTypeSize(outputType));
}

ModelInference::~ModelInference() {}

APP_ERROR ModelInference::Infer(size_t n, char* inputData, char* outputData)
{
    // 1. check inputData and outputData
    APPERR_RETURN_IF_NOT_LOG(inputData != nullptr, APP_ERR_INVALID_PARAM, "the input data is nullptr");
    APPERR_RETURN_IF_NOT_LOG(outputData != nullptr, APP_ERR_INVALID_PARAM, "the output data is nullptr");

    // 2. execute with batch
    int batchNum = static_cast<int>(utils::divUp(n, static_cast<size_t>(batch)));
    for (int b = 0; b < batchNum; ++b) {
        modelExecuter->execute(inputData + b * inputLen, outputData + b * outputLen);
    }

    return APP_ERR_OK;
}
} // namespace ascend
