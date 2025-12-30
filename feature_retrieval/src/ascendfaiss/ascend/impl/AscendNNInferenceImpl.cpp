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


#include "AscendNNInferenceImpl.h"

#include <algorithm>
#include <string>
#include <set>

#include <faiss/impl/AuxIndexStructures.h>
#include <securec.h>

#include "AscendTensor.h"
#include "ascend/AscendIndex.h"
#include "ascend/utils/fp16.h"
#include "common/threadpool/AscendThreadPool.h"
#include "common/utils/DataType.h"
#include "common/utils/LogUtils.h"


namespace faiss {
namespace ascend {
namespace {
const size_t KB = 1024;
const size_t RETAIN_SIZE = 2048;
// get pagesize must be less than 32M, becauseof rpc limitation
const size_t PAGE_SIZE = 32U * KB * KB - RETAIN_SIZE;

// Or, maximum number 512K of vectors to consider per page of infer
const size_t VEC_SIZE = 512U * KB;

const size_t MAX_NN_MODEL_SIZE = 128 * 1024 * 1024;

const int MAX_INPUT_TYPE = 10;

const int MAX_DEVICELIST_SIZE = 32;

bool CheckParametersNN(int inputType, int outputType, int batch, int dimIn, int dimOut)
{
    const std::vector<int> dimsIn = { 64, 128, 256, 384, 512, 768, 1024 };
    const std::vector<int> dimsOut = { 32, 64, 96, 128, 256 };
    const std::vector<int> batches = { 1, 2, 4, 8, 16, 32, 64, 128 };

    if (inputType < 0 || inputType > MAX_INPUT_TYPE) {
        APP_LOG_ERROR("Unsupported inputType, should be in [0, %d]", MAX_INPUT_TYPE);
        return false;
    }

    if (outputType < 0 || outputType > MAX_INPUT_TYPE) {
        APP_LOG_ERROR("Unsupported outputType, should be in [0, %d]", MAX_INPUT_TYPE);
        return false;
    }

    if (std::find(batches.begin(), batches.end(), batch) == batches.end()) {
        APP_LOG_ERROR("Unsupported batch:%d", batch);
        return false;
    }

    if (std::find(dimsIn.begin(), dimsIn.end(), dimIn) == dimsIn.end()) {
        APP_LOG_ERROR("Unsupported dimIn:%d", dimIn);
        return false;
    }

    if (std::find(dimsOut.begin(), dimsOut.end(), dimOut) == dimsOut.end()) {
        APP_LOG_ERROR("Unsupported dimOut:%d", dimOut);
        return false;
    }

    return true;
}
} // namespace

AscendNNInferenceImpl::AscendNNInferenceImpl(std::vector<int> deviceList, const char *model, uint64_t modelSize)
    : verbose(false)
{
    APP_LOG_INFO("AscendNNInference construction start");
    FAISS_THROW_IF_NOT_MSG(deviceList.size() > 0 && deviceList.size() <= MAX_DEVICELIST_SIZE,
        "device list should be > 0 and <= 32");
    FAISS_THROW_IF_NOT_MSG(model != nullptr, "model can not be nullptr");
    FAISS_THROW_IF_NOT_MSG(modelSize > 0 && modelSize <= MAX_NN_MODEL_SIZE, "modelSize should be in (0, 128MB].");

    std::set<int> uniqueDeviceList(deviceList.begin(), deviceList.end());
    if (uniqueDeviceList.size() != deviceList.size()) {
        std::string deviceListStr;
        for (auto id : deviceList) {
            deviceListStr += std::to_string(id) + ",";
        }
        FAISS_THROW_FMT("some device IDs are the same, please check it {%s}", deviceListStr.c_str());
    }

    this->deviceList = deviceList;
    if (deviceList.size() > 1) {
        this->pool = std::make_shared<AscendThreadPool>(deviceList.size());
    }
    initInference(model, modelSize);
    APP_LOG_INFO("AscendNNInference construction finished");
}

void AscendNNInferenceImpl::initInference(const char *model, uint64_t modelSize)
{
    for (size_t i = 0; i < deviceList.size(); i++) {
        int deviceId = deviceList[i];
        FAISS_THROW_IF_NOT_MSG(aclrtSetDevice(deviceId) == ACL_SUCCESS, "set device failed");

        auto modelData = static_cast<const void *>(model);
        FAISS_THROW_IF_NOT_MSG(modelData != nullptr,  "model can not be nullptr");
        FAISS_THROW_IF_NOT_MSG(modelSize > 0 && modelSize <= MAX_NN_MODEL_SIZE, "modelSize must be > 0 and <= 128MB");
        std::shared_ptr<::ascend::ModelInference> inference =
            std::make_shared<::ascend::ModelInference>(modelData, modelSize);
        FAISS_THROW_IF_NOT_MSG(inference, "Failed to create inference");

        auto paramCheckStatus = CheckParametersNN(inference->inputType, inference->outputType, inference->batch,
                                                  inference->dimIn, inference->dimOut);
        if (!paramCheckStatus) {
            FAISS_THROW_IF_NOT_MSG(paramCheckStatus, "Invalid param  for MModelInference(), please check input param");
        }
        inputType = inference->inputType;
        outputType = inference->outputType;
        batch = inference->batch;
        dimIn = inference->dimIn;
        dimOut = inference->dimOut;
        inferences[deviceId] = inference;
    }
}

void AscendNNInferenceImpl::infer(size_t n, const char *inputData, char *outputData)
{
    FAISS_THROW_IF_NOT_FMT((n > 0) && (n < MAX_N), "n must be > 0 and < %ld", MAX_N);
    FAISS_THROW_IF_NOT_MSG(inputData, "inputData can not be nullptr.");
    FAISS_THROW_IF_NOT_MSG(outputData, "outputData can not be nullptr.");

    return inferPaged(n, inputData, outputData);
}


int AscendNNInferenceImpl::getInputType() const
{
    return inputType;
}

int AscendNNInferenceImpl::getOutputType() const
{
    return outputType;
}

int AscendNNInferenceImpl::getDimIn() const
{
    return dimIn;
}

int AscendNNInferenceImpl::getDimOut() const
{
    return dimOut;
}

int AscendNNInferenceImpl::getDimBatch() const
{
    return batch;
}


void AscendNNInferenceImpl::inferPaged(int n, const char *x, char *outputData)
{
    APP_LOG_INFO("AscendNNInference start to inferPaged with %d vector(s).\n", n);
    size_t totalSize = static_cast<size_t>(n) * getElementSize();
    if (totalSize > PAGE_SIZE || static_cast<size_t>(n) > VEC_SIZE) {
        // How many vectors fit into kInferPageSize?
        size_t maxNumVecsForPageSize = PAGE_SIZE / getElementSize();
        // Always add at least 1 vector, if we have huge vectors
        maxNumVecsForPageSize = std::max(maxNumVecsForPageSize, static_cast<size_t>(1));

        // Since the model infers batch vectors each time, so tileSize is bet align by batch
        size_t tileSize = std::min(static_cast<size_t>(n), maxNumVecsForPageSize / batch * batch);

        for (size_t i = 0; i < static_cast<size_t>(n); i += tileSize) {
            size_t curNum = std::min(tileSize, n - i);
            if (this->verbose) {
                printf("AscendNNInference::infer: inferring %zu:%zu / %d\n", i, i + curNum, n);
            }
            inferImpl(curNum, x + i * static_cast<size_t>(this->dimIn) * static_cast<size_t>(getTypeSize(inputType)),
                outputData + i * static_cast<size_t>(this->dimOut) * static_cast<size_t>(getTypeSize(outputType)));
        }
    } else {
        if (this->verbose) {
            printf("AscendNNInference::infer: inferring 0:%d / %d\n", n, n);
        }
        inferImpl(n, x, outputData);
    }
    APP_LOG_INFO("AscendNNInference inferPaged operation finished.\n");
}

void AscendNNInferenceImpl::inferImpl(int n, const char *x, char *outputData)
{
    APP_LOG_INFO("AscendNNInference inferImpl operation started: n=%d.\n", n);
    int batchNum = (n + batch - 1) / batch;
    char *xi = const_cast<char *>(x);

    // if n is not a multiple of batch
    std::vector<char> xAlign;
    if (n % batch != 0) {
        xAlign.resize(static_cast<size_t>(batchNum) * static_cast<size_t>(batch) * getElementSize());
        size_t inferLen =
            static_cast<size_t>(n) * static_cast<size_t>(dimIn) * static_cast<size_t>(getTypeSize(inputType));
        auto ret = memcpy_s(xAlign.data(), inferLen, x, inferLen);
        FAISS_THROW_IF_NOT_FMT(ret == EOK, "memcpy_s faild, error code is %d\n", ret);
        xi = xAlign.data();
    }

    // Allocate the total to each chip
    int deviceCnt = static_cast<int>(deviceList.size());
    FAISS_THROW_IF_NOT_MSG(deviceCnt > 0, "Invalid deviceList size, should have at least one device\n");
    std::vector<int> inferMap(deviceCnt, 0);

    for (int i = 0; i < deviceCnt; i++) {
        inferMap[i] = batchNum / deviceCnt * batch;
    }

    // if n is not a multiple of batch, then index of `deviceCnt - i` is not a full batch
    // we must avoid memory copy out of bounds
    for (int i = 0; i < batchNum % deviceCnt; i++) {
        inferMap[deviceCnt - 1 - i] += batch;
    }

    int offsum = 0;
    std::vector<int> offsetMap(deviceCnt, 0);
    for (int i = 0; i < deviceCnt; i++) {
        offsetMap[i] = offsum;
        offsum += inferMap.at(i);
    }

    std::vector<std::vector<char>> inferResult(deviceCnt, std::vector<char>());
    auto inferFunctor = [&](int idx) {
        int num = inferMap.at(idx);
        if (num != 0) {
            inferenceInfer(deviceList[idx], num, xi + offsetMap[idx] * this->dimIn,
                           num * dimIn * getTypeSize(inputType), inferResult[idx]);
        }
    };
    // Call multi-thread and multi-chip parallel inference
    CALL_PARALLEL_FUNCTOR(static_cast<size_t>(deviceCnt), pool, inferFunctor);

    // to avoid memory modify copy out of bounds, modify the index of `deviceCnt - i` to the real num to infer
    inferMap[deviceCnt - 1] = inferMap[deviceCnt - 1] + (n % batch - batch) % batch;

    char *dest = outputData;
    for (int i = 0; i < deviceCnt; i++) {
        int num = inferMap.at(i);
        if (num == 0) {
            continue;
        }

        size_t resultLen =
            static_cast<size_t>(num) * static_cast<size_t>(dimOut) * static_cast<size_t>(getTypeSize(outputType));

        FAISS_THROW_IF_NOT_FMT(inferResult[i].size() >= resultLen,
            "Invalid inferResult[i].size() is %zu, expected: %zu.\n",
            inferResult[i].size(), resultLen);
        auto ret = memcpy_s(dest, resultLen, inferResult[i].data(), resultLen);
        FAISS_THROW_IF_NOT_FMT(ret == EOK, "memcpy_s faild, error code is %d\n", ret);

        dest += resultLen;
    }
    APP_LOG_INFO("AscendNNInference inferImpl operation finished.\n");
}

size_t AscendNNInferenceImpl::getElementSize() const
{
    // Guaranteed that the data transfer size does not exceed the PAGE_SIZE
    return std::max(this->dimIn * getTypeSize(inputType), this->dimOut * getTypeSize(outputType));
}

int GetMaxN(int inputType, int outputType, int dimIn, int dimOut)
{
    // Guaranteed that the data transfer size does not exceed the PAGE_SIZE
    return static_cast<int>(PAGE_SIZE) / std::max(dimIn * getTypeSize(inputType), dimOut * getTypeSize(outputType));
}

void AscendNNInferenceImpl::inferenceInfer(int deviceId, int n, const char* data, uint64_t dataLen,
                                           std::vector<char> &output)
{
    FAISS_THROW_IF_NOT_FMT(inferences.find(deviceId) != inferences.end(),
                           "deviceId is out of range, deviceId=%d.", deviceId);
    FAISS_THROW_IF_NOT_MSG(aclrtSetDevice(deviceId) == ACL_SUCCESS, "set device failed");
    auto inference = inferences[deviceId];

    int maxN = GetMaxN(inference->inputType, inference->outputType, inference->dimIn, inference->dimOut);
    FAISS_THROW_IF_NOT_FMT((n > 0) && (n <= maxN), "n must be > 0 and <= %d", maxN);

    FAISS_THROW_IF_NOT_FMT(dataLen == static_cast<size_t>(n) * static_cast<size_t>(inference->dimIn) *
                                     static_cast<size_t>(getTypeSize(inference->inputType)),
                           "Invalid data size %zu, expected: %zu", dataLen,
                           static_cast<size_t>(n) * static_cast<size_t>(inference->dimIn) *
                               static_cast<size_t>(getTypeSize(inference->inputType)));
    using namespace ::ascend;
    auto typeSize = getTypeSize(inference->outputType);
    size_t outSize = static_cast<size_t>(utils::roundUp(n, inference->batch)) *
        static_cast<size_t>(inference->dimOut) * static_cast<size_t>(typeSize);

    AscendTensor<char, DIMS_1> tmpInput({ static_cast<int>(dataLen) });
    AscendTensor<char, DIMS_1> tmpOutput({ static_cast<int>(outSize) });

    auto err = aclrtMemcpy(tmpInput.data(), tmpInput.getSizeInBytes(),
                           data, dataLen, ACL_MEMCPY_HOST_TO_DEVICE);
    FAISS_THROW_IF_NOT_MSG(err == ACL_SUCCESS,  "Failed to copy input to device");
    APP_LOG_INFO("Model inference %d actual start\n", deviceId);
    FAISS_THROW_IF_NOT_MSG(inference->Infer(n, tmpInput.data(), tmpOutput.data()) == APP_ERR_OK,
                           "model inference failed");
    APP_LOG_INFO("Model inference %d end\n", deviceId);
    std::vector<char> backOutput(outSize);
    auto ret = aclrtMemcpy(backOutput.data(), backOutput.size() * sizeof(char),
                           tmpOutput.data(), outSize, ACL_MEMCPY_DEVICE_TO_HOST);
    FAISS_THROW_IF_NOT_MSG(ret == ACL_SUCCESS, "Failed to copy output to host");
    output.insert(output.end(), backOutput.begin(), backOutput.end());
}
} // ascend
} // faiss