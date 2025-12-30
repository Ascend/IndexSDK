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


#ifndef ASCEND_NN_INFERENCE_IMPL_INCLUDED
#define ASCEND_NN_INFERENCE_IMPL_INCLUDED

#include "ModelInference.h"

#include <faiss/impl/FaissAssert.h>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

class AscendThreadPool;

namespace faiss {
namespace ascend {
using rpcContext = void *;

class AscendNNInferenceImpl {
public:
    AscendNNInferenceImpl(std::vector<int> deviceList, const char *model, uint64_t modelSize);

    ~AscendNNInferenceImpl() = default;

    void infer(size_t n, const char *inputData, char *outputData);

    // AscendNNInference object is NON-copyable
    AscendNNInferenceImpl(const AscendNNInferenceImpl &) = delete;
    AscendNNInferenceImpl &operator = (const AscendNNInferenceImpl &) = delete;

    int getInputType() const;

    int getOutputType() const;

    int getDimIn() const;

    int getDimOut() const;

    int getDimBatch() const;

private:
    void initInference(const char *model, uint64_t modelSize);

    // Handles paged infer if the infer set is too large, passes to
    // inferImpl to actually perform the infer for the current page
    void inferPaged(int n, const char *x, char *outputData);

    // Actually perform the infer
    void inferImpl(int n, const char *x, char *outputData);

    // Get the size of memory every database vector needed to store
    size_t getElementSize() const;

    void inferenceInfer(int deviceId, int n, const char* data, uint64_t dataLen, std::vector<char> &output);

private:
    // Whether to print infer log
    bool verbose;

    // the data type before inference dimension
    int inputType;

    // the data type after inference dimension
    int outputType;

    // Vector dimension before inference dimension
    int dimIn;

    // Vector dimension after inference dimension
    int dimOut;

    // The number of samples selected by the model for one inference
    int batch;

private:
    // The infer model path
    std::string modelPath;

    // The chip ID for inferring
    std::vector<int> deviceList;

    std::unordered_map<int, std::shared_ptr<::ascend::ModelInference>> inferences;

    // Thread pool for multithread processing
    std::shared_ptr<AscendThreadPool> pool;
};
} // ascend
} // faiss
#endif // ASCEND_NN_INFERENCE_IMPL_INCLUDED