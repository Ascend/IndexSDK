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


#ifndef ASCEND_NN_INFERENCE_INCLUDED
#define ASCEND_NN_INFERENCE_INCLUDED

#include <memory>
#include <vector>
#include <string>
#include <unordered_map>
#include <faiss/impl/FaissAssert.h>

class AscendThreadPool;

namespace faiss {
namespace ascend {
using rpcContext = void *;
class AscendNNInferenceImpl;
class AscendNNInference {
public:
    AscendNNInference(std::vector<int> deviceList, const char *model, uint64_t modelSize);

    ~AscendNNInference();

    void infer(size_t n, const char *inputData, char *outputData) const;

    // AscendNNInference object is NON-copyable
    AscendNNInference(const AscendNNInference &) = delete;
    AscendNNInference &operator = (const AscendNNInference &) = delete;

    int getInputType() const;

    int getOutputType() const;

    int getDimIn() const;

    int getDimOut() const;

    int getDimBatch() const;

protected:
    std::shared_ptr<AscendNNInferenceImpl> impl_;
};
} // ascend
} // faiss
#endif // ASCEND_NN_INFERENCE_INCLUDED