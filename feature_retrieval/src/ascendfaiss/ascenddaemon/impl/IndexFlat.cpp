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


#include "acl/acl_op_compiler.h"
#include "ascenddaemon/impl/IndexFlat.h"

#include "ascenddaemon/impl/AuxIndexStructures.h"
#include "common/utils/CommonUtils.h"
#include "common/utils/OpLauncher.h"
#include "ops/cpukernel/impl/utils/kernel_shared_def.h"

namespace ascend {
namespace {
const int KB = 1024;
const int FLAT_BLOCK_SIZE = 16384 * 16;
const int FLAT_COMPUTE_PAGE = FLAT_BLOCK_SIZE * 16;
const int THREADS_CNT = faiss::ascend::SocUtils::GetInstance().GetThreadsCnt();
}

std::mutex IndexFlat::multiSearchTopkMtx;

IndexFlat::IndexFlat(int dim, int64_t resourceSize)
    : Index(dim, resourceSize), blockSize(FLAT_BLOCK_SIZE), pageSize(FLAT_COMPUTE_PAGE), maskData(nullptr), maskLen(0)
{
    ASCEND_THROW_IF_NOT(dims % CUBE_ALIGN == 0);

    // IndexFlat does not need training
    isTrained = true;

    int dim1 = utils::divUp(blockSize, CUBE_ALIGN);
    int dim2 = utils::divUp(dims, CUBE_ALIGN);
    this->devVecCapacity = dim1 * dim2 * CUBE_ALIGN * CUBE_ALIGN;
    // int8类型的特征的dim维度需要按照32分段
    int dim2Int8 = utils::divUp(dims, CUBE_ALIGN_INT8);
    this->devInt8VecCapacity = dim1 * dim2Int8 * CUBE_ALIGN * CUBE_ALIGN_INT8;

    if (dim == 2048) { // dim 2048, 客户要求bs= 20
        searchBatchSizes = {48, 36, 32, 30, 24, 20, 18, 16, 12, 8, 6, 4, 2, 1};
    } else {
        searchBatchSizes = {48, 36, 32, 30, 24, 18, 16, 12, 8, 6, 4, 2, 1};
    }
    // the result constain min value and index, the multi 2
    this->burstsOfBlock = utils::divUp(this->blockSize, static_cast<int>(BURST_LEN)) * 2;
    this->blockMaskSize = utils::divUp(this->blockSize, BIT_OF_UINT8);
    if (isUseOnlineOp()) {
        resetTransdataShapeOnlineOp();
        resetTransdataShaperawOnlineOp();
    }
}

IndexFlat::~IndexFlat() { }

size_t IndexFlat::calcShapedBaseSize(idx_t totalNum)
{
    size_t numBatch = utils::divUp(totalNum, blockSize);
    int dim1 = utils::divUp(blockSize, CUBE_ALIGN);
    int dim2 = utils::divUp(dims, CUBE_ALIGN);
    return numBatch * (size_t)(dim1 * dim2 * CUBE_ALIGN * CUBE_ALIGN);
}

void IndexFlat::resizeBaseShaped(size_t n)
{
    int newVecSize = static_cast<int>(utils::divUp(this->ntotal + n, this->blockSize));
    int vecSize = static_cast<int>(utils::divUp(this->ntotal, this->blockSize));
    int addVecNum = newVecSize - vecSize;

    // 1. adapt old block size
    if (vecSize > 0) {
        int vecId = vecSize - 1;
        if (addVecNum > 0) {
            // there is a new block needed, then the old last one block must be fulled
            this->baseShaped.at(vecId)->resize(this->devVecCapacity, true);
        } else {
            auto capacity = getVecCapacity(
                this->ntotal - static_cast<size_t>(vecId * blockSize) + static_cast<size_t>(n),
                this->baseShaped.at(vecId)->size());
            this->baseShaped.at(vecId)->resize(capacity, true);
        }
    }

    // 2. add new new block and set size
    for (int i = 0; i < addVecNum; ++i) {
        this->baseShaped.emplace_back(CREATE_UNIQUE_PTR(DeviceVector<float16_t>, MemorySpace::DEVICE_HUGEPAGE));
        int64_t newVecId = static_cast<int64_t>(vecSize) + static_cast<int64_t>(i);
        if (newVecId == newVecSize - 1) {
            // compute the last block size and resize
            auto capacity = getVecCapacity(
                this->ntotal - static_cast<size_t>(newVecId * blockSize) + static_cast<size_t>(n),
                this->baseShaped.at(newVecId)->size());
            this->baseShaped.at(newVecId)->resize(capacity, true);
        } else {
            // the blocks in the middle must be full.
            this->baseShaped.at(newVecId)->resize(this->devVecCapacity, true);
        }
    }
}

// 保存量化后的int8的特征数据，需要保存到baseShapedInt8中，同时获取需要扩充的空间需要调用getInt8VecCapacity获取
void IndexFlat::resizeBaseShapedInt8(size_t n)
{
    int newVecSize = static_cast<int>(utils::divUp(this->ntotal + n, this->blockSize));
    int vecSize = static_cast<int>(utils::divUp(this->ntotal, this->blockSize));
    int addVecNum = newVecSize - vecSize;

    // 1. adapt old block size
    if (vecSize > 0) {
        int vecId = vecSize - 1;
        if (addVecNum > 0) {
            // there is a new block needed, then the old last one block must be fulled
            baseShapedInt8.at(vecId)->resize(this->devInt8VecCapacity, true);
        } else {
            auto capacity = getInt8VecCapacity(
                this->ntotal - static_cast<size_t>(vecId * blockSize) + static_cast<size_t>(n),
                baseShapedInt8.at(vecId)->size());
            baseShapedInt8.at(vecId)->resize(capacity, true);
        }
    }

    // 2. add new new block and set size
    for (int i = 0; i < addVecNum; ++i) {
        baseShapedInt8.emplace_back(CREATE_UNIQUE_PTR(DeviceVector<int8_t>, MemorySpace::DEVICE_HUGEPAGE));
        int64_t newVecId = static_cast<int64_t>(vecSize) + static_cast<int64_t>(i);
        if (newVecId == newVecSize - 1) {
            // compute the last block size and resize
            auto capacity = getInt8VecCapacity(
                this->ntotal - static_cast<size_t>(newVecId * blockSize) + static_cast<size_t>(n),
                baseShapedInt8.at(newVecId)->size());
            baseShapedInt8.at(newVecId)->resize(capacity, true);
        } else {
            // the blocks in the middle must be full.
            baseShapedInt8.at(newVecId)->resize(this->devInt8VecCapacity, true);
        }
    }
}

APP_ERROR IndexFlat::copyAndSaveVectors(size_t startOffset, AscendTensor<float16_t, DIMS_2> &rawData)
{
    if (faiss::ascend::SocUtils::GetInstance().IsZZCodeFormat()) {
        return addVectorsAiCpu(startOffset, rawData);
    } else {
        return AddCodeNDFormat(rawData, startOffset, blockSize, baseShaped);
    }
}

APP_ERROR IndexFlat::copyAndSaveVectors(size_t startOffset, AscendTensor<int8_t, DIMS_2> &rawData)
{
    if (faiss::ascend::SocUtils::GetInstance().IsZZCodeFormat()) {
        return addInt8VectorsAicpu(startOffset, rawData);
    } else {
        return AddCodeNDFormat(rawData, startOffset, blockSize, baseShapedInt8);
    }
}

APP_ERROR IndexFlat::addVectors(AscendTensor<float16_t, DIMS_2> &rawData)
{
    int n = rawData.getSize(0);
    int d = rawData.getSize(1);
    APPERR_RETURN_IF_NOT_FMT(n >= 0, APP_ERR_INVALID_PARAM, "the number of vectors added is %d", n);
    APPERR_RETURN_IF_NOT_FMT(d == this->dims, APP_ERR_INVALID_PARAM,
                             "the dim of add vectors is %d, not equal to base", d);

    resizeBaseShaped(static_cast<size_t>(n));
    return copyAndSaveVectors(ntotal, rawData);
}

void IndexFlat::resetTransdataShapeOnlineOp()
{
    std::vector<int64_t> onlineShape0 { -1, -1 };
    std::vector<int64_t> onlineShape1 { -1 };
    std::vector<int64_t> onlineShape2 { -1, -1, -1, -1 };
    std::vector<aclTensorDesc *> inputDesc;
    int64_t dimRange0[2][2] = {{1, -1}, {1, -1}};
    appendTensorDesc(inputDesc, onlineShape0, ACL_FLOAT16, dimRange0, ACL_FORMAT_ND);

    int64_t dimRange1[1][2] = {{1, -1}};
    appendTensorDesc(inputDesc, onlineShape1, ACL_INT64, dimRange1, ACL_FORMAT_ND);

    std::vector<aclTensorDesc *> outputDesc;
    int64_t dimRange2[4][2] = {{1, -1}, {1, -1}, {1, -1}, {1, -1}};
    appendTensorDesc(outputDesc, onlineShape2, ACL_FLOAT16, dimRange2, ACL_FORMAT_ND);
    aclopAttr *opAttr = aclopCreateAttr();
    auto ret = aclSetCompileopt(ACL_OP_JIT_COMPILE, "enable");
    if (ret != APP_ERR_OK) {
        destroyOpResource(opAttr, inputDesc, outputDesc);
        ASCEND_THROW_IF_NOT_FMT(ret == APP_ERR_OK, "enable jit compile fail opName:TransdataShaped: %i\n", ret);
    }

    auto compRet = aclopCompile("TransdataShaped", inputDesc.size(), inputDesc.data(), outputDesc.size(),
                                outputDesc.data(), opAttr, ACL_ENGINE_SYS, ACL_COMPILE_SYS, nullptr);
    destroyOpResource(opAttr, inputDesc, outputDesc);
    ASCEND_THROW_IF_NOT_FMT(compRet == APP_ERR_OK, "compile TransdataShaped op failed: %i", compRet);
}

void IndexFlat::resetTransdataShaperawOnlineOp()
{
    std::vector<int64_t> onlineShape0 { -1, -1, -1, -1 };
    std::vector<int64_t> onlineShape1 { -1 };
    std::vector<int64_t> onlineShape2 { -1, -1 };
    std::vector<aclTensorDesc *> inputDesc;
    int64_t dimRange0[4][2] = {{1, -1}, {1, -1}, {1, -1}, {1, -1}};
    appendTensorDesc(inputDesc, onlineShape0, ACL_FLOAT16, dimRange0, ACL_FORMAT_ND);

    int64_t dimRange1[1][2] = {{1, -1}};
    appendTensorDesc(inputDesc, onlineShape1, ACL_INT64, dimRange1, ACL_FORMAT_ND);

    std::vector<aclTensorDesc *> outputDesc;
    int64_t dimRange2[2][2] = {{1, -1}, {1, -1}};
    appendTensorDesc(outputDesc, onlineShape2, ACL_FLOAT16, dimRange2, ACL_FORMAT_ND);
    aclopAttr *opAttr = aclopCreateAttr();
    auto ret = aclSetCompileopt(ACL_OP_JIT_COMPILE, "enable");
    if (ret != APP_ERR_OK) {
        destroyOpResource(opAttr, inputDesc, outputDesc);
        ASCEND_THROW_IF_NOT_FMT(ret == APP_ERR_OK, "enable jit compile fail opName:TransdataRaw: %i\n", ret);
    }

    auto compRet = aclopCompile("TransdataRaw", inputDesc.size(), inputDesc.data(), outputDesc.size(),
                                outputDesc.data(), opAttr, ACL_ENGINE_SYS, ACL_COMPILE_SYS, nullptr);
    destroyOpResource(opAttr, inputDesc, outputDesc);
    ASCEND_THROW_IF_NOT_FMT(compRet == APP_ERR_OK, "compile TransdataRaw op failed: %i", compRet);
}

void IndexFlat::execTransdataShapedOp(std::string &opName, aclrtStream stream,
                                      AscendTensor<float16_t, DIMS_2> &src1,
                                      AscendTensor<int64_t, DIMS_1> &src2,
                                      AscendTensor<float16_t, DIMS_4> &dst)
{
    if (isUseOnlineOp()) {
        std::vector<int64_t> shape0 { static_cast<int64_t>(src1.getSize(0)), static_cast<int64_t>(src1.getSize(1)) };
        std::vector<int64_t> shape1 { static_cast<int64_t>(src2.getSize(0)) };
        std::vector<int64_t> shape2 { static_cast<int64_t>(dst.getSize(0)), static_cast<int64_t>(dst.getSize(1)),
                                      static_cast<int64_t>(dst.getSize(2)), static_cast<int64_t>(dst.getSize(3))};
        std::vector<aclTensorDesc *> inputDesc;
        inputDesc.emplace_back(aclCreateTensorDesc(ACL_FLOAT16, shape0.size(), shape0.data(), ACL_FORMAT_ND));
        inputDesc.emplace_back(aclCreateTensorDesc(ACL_INT64, shape1.size(), shape1.data(), ACL_FORMAT_ND));
        std::vector<aclTensorDesc *> outputDesc;
        outputDesc.emplace_back(aclCreateTensorDesc(ACL_FLOAT16, shape2.size(), shape2.data(), ACL_FORMAT_ND));
        std::vector<aclDataBuffer *> input;
        input.emplace_back(aclCreateDataBuffer(src1.data(), src1.getSizeInBytes()));
        input.emplace_back(aclCreateDataBuffer(src2.data(), src2.getSizeInBytes()));
        std::vector<aclDataBuffer *> output;
        output.emplace_back(aclCreateDataBuffer(dst.data(), dst.getSizeInBytes()));
        aclopAttr *opAttr = aclopCreateAttr();
        auto ret = aclopExecuteV2(opName.c_str(), inputDesc.size(), inputDesc.data(), input.data(),
                                  outputDesc.size(), outputDesc.data(), output.data(), opAttr, stream);
        destroyOpResource(opAttr, inputDesc, outputDesc, input, output);
        ASCEND_THROW_IF_NOT_FMT(ret == APP_ERR_OK, "exec TransdataShaped op failed: %i", ret);
    } else {
        LaunchOpTwoInOneOut<float16_t, DIMS_2, ACL_FLOAT16,
                            int64_t, DIMS_1, ACL_INT64,
                            float16_t, DIMS_4, ACL_FLOAT16>(opName, stream, src1, src2, dst);
    }
}

void IndexFlat::execTransdataShapedrawOp(std::string &opName, aclrtStream stream,
                                         AscendTensor<float16_t, DIMS_4> &src1,
                                         AscendTensor<int64_t, DIMS_1> &src2,
                                         AscendTensor<float16_t, DIMS_2> &dst)
{
    if (isUseOnlineOp()) {
        std::vector<int64_t> shape0 { static_cast<int64_t>(src1.getSize(0)), static_cast<int64_t>(src1.getSize(1)),
                                      static_cast<int64_t>(src1.getSize(2)), static_cast<int64_t>(src1.getSize(3))};
        std::vector<int64_t> shape1 { static_cast<int64_t>(src2.getSize(0)) };
        std::vector<int64_t> shape2 { static_cast<int64_t>(dst.getSize(0)), static_cast<int64_t>(dst.getSize(1)) };
        std::vector<aclTensorDesc *> inputDescRaw;
        inputDescRaw.emplace_back(aclCreateTensorDesc(ACL_FLOAT16, shape0.size(), shape0.data(), ACL_FORMAT_ND));
        inputDescRaw.emplace_back(aclCreateTensorDesc(ACL_INT64, shape1.size(), shape1.data(), ACL_FORMAT_ND));
        std::vector<aclTensorDesc *> outputDescRaw;
        outputDescRaw.emplace_back(aclCreateTensorDesc(ACL_FLOAT16, shape2.size(), shape2.data(), ACL_FORMAT_ND));
        std::vector<aclDataBuffer *> input;
        input.emplace_back(aclCreateDataBuffer(src1.data(), src1.getSizeInBytes()));
        input.emplace_back(aclCreateDataBuffer(src2.data(), src2.getSizeInBytes()));
        std::vector<aclDataBuffer *> output;
        output.emplace_back(aclCreateDataBuffer(dst.data(), dst.getSizeInBytes()));
        aclopAttr *opAttr = aclopCreateAttr();
        auto ret = aclopExecuteV2(opName.c_str(), inputDescRaw.size(), inputDescRaw.data(), input.data(),
                                  outputDescRaw.size(), outputDescRaw.data(), output.data(), opAttr, stream);
        destroyOpResource(opAttr, inputDescRaw, outputDescRaw, input, output);
        ASCEND_THROW_IF_NOT_FMT(ret == APP_ERR_OK, "exec TransdataShaped op failed: %i", ret);
    } else {
        LaunchOpTwoInOneOut<float16_t, DIMS_4, ACL_FLOAT16,
                            int64_t, DIMS_1, ACL_INT64,
                            float16_t, DIMS_2, ACL_FLOAT16>(opName, stream, src1, src2, dst);
    }
}

APP_ERROR IndexFlat::addVectorsAiCpu(size_t startOffset, AscendTensor<float16_t, DIMS_2> &rawData)
{
    int n = rawData.getSize(0);
    std::string opName = "TransdataShaped";
    auto &mem = resources.getMemoryManager();
    auto streamPtr = resources.getDefaultStream();
    auto stream = streamPtr->GetStream();
    AscendTensor<float16_t, DIMS_2> data(mem, {n, dims}, stream);
    auto ret = aclrtMemcpy(data.data(), data.getSizeInBytes(),
                           rawData.data(), rawData.getSizeInBytes(), ACL_MEMCPY_HOST_TO_DEVICE);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "Mem operator error %d", (int)ret);

    int blockNum = utils::divUp(static_cast<int>(startOffset) + n, blockSize);
    AscendTensor<int64_t, DIMS_2> attrs(mem, {blockNum, aicpu::TRANSDATA_SHAPED_ATTR_IDX_COUNT}, stream);

    size_t blockSize = static_cast<size_t>(this->blockSize);
    for (size_t i = 0; i < static_cast<size_t>(n);) {
        size_t total = startOffset + i;
        size_t offsetInBlock = total % blockSize;
        size_t leftInBlock = blockSize - offsetInBlock;
        size_t leftInData = static_cast<size_t>(n) - i;
        size_t copyCount = std::min(leftInBlock, leftInData);
        size_t blockIdx = total / blockSize;

        int copy = static_cast<int>(copyCount);
        AscendTensor<float16_t, DIMS_2> src(data[i].data(), {copy, dims});
        AscendTensor<float16_t, DIMS_4> dst(baseShaped[blockIdx]->data(),
            {utils::divUp(this->blockSize, CUBE_ALIGN), utils::divUp(dims, CUBE_ALIGN), CUBE_ALIGN, CUBE_ALIGN});
        AscendTensor<int64_t, DIMS_1> attr = attrs[blockIdx].view();
        attr[aicpu::TRANSDATA_SHAPED_ATTR_NTOTAL_IDX] = offsetInBlock;
        execTransdataShapedOp(opName, stream, src, attr, dst);
        i += copyCount;
    }

    ret = synchronizeStream(stream);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR,
        "synchronizeStream addVector stream failed: %i\n", ret);

    return APP_ERR_OK;
}

// 将int8类型特征分形保存到baseShapedInt8中，注意dst的shape中第1个维度需要按照CUBE_ALIGN_INT8对齐
APP_ERROR IndexFlat::addInt8VectorsAicpu(size_t startOffset, AscendTensor<int8_t, DIMS_2> &rawData)
{
    int n = rawData.getSize(0);
    std::string opName = "TransdataShaped";
    auto &mem = resources.getMemoryManager();
    auto streamPtr = resources.getDefaultStream();
    auto stream = streamPtr->GetStream();
    AscendTensor<int8_t, DIMS_2> data(mem, {n, dims}, stream);
    auto ret = aclrtMemcpy(data.data(), data.getSizeInBytes(),
        rawData.data(), rawData.getSizeInBytes(), ACL_MEMCPY_HOST_TO_DEVICE);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "Mem operator error %d", ret);

    int blockNum = utils::divUp(static_cast<int>(startOffset) + n, blockSize);
    AscendTensor<int64_t, DIMS_2> attrs(mem, {blockNum, aicpu::TRANSDATA_SHAPED_ATTR_IDX_COUNT}, stream);

    size_t blockSize = static_cast<size_t>(this->blockSize);
    for (size_t i = 0; i < static_cast<size_t>(n);) {
        size_t total = startOffset + i;
        size_t offsetInBlock = total % blockSize;
        size_t leftInBlock = blockSize - offsetInBlock;
        size_t leftInData = static_cast<size_t>(n) - i;
        size_t copyCount = std::min(leftInBlock, leftInData);
        size_t blockIdx = total / blockSize;

        int copy = static_cast<int>(copyCount);
        AscendTensor<int8_t, DIMS_2> src(data[i].data(), {copy, dims});
        AscendTensor<int8_t, DIMS_4> dst(baseShapedInt8[blockIdx]->data(),
            {utils::divUp(this->blockSize, CUBE_ALIGN), utils::divUp(dims, CUBE_ALIGN_INT8),
            CUBE_ALIGN, CUBE_ALIGN_INT8});
        AscendTensor<int64_t, DIMS_1> attr = attrs[blockIdx].view();
        attr[aicpu::TRANSDATA_SHAPED_ATTR_NTOTAL_IDX] = offsetInBlock;

        LaunchOpTwoInOneOut<int8_t, DIMS_2, ACL_INT8,
                            int64_t, DIMS_1, ACL_INT64,
                            int8_t, DIMS_4, ACL_INT8>(opName, stream, src, attr, dst);

        i += copyCount;
    }

    ret = synchronizeStream(stream);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR,
        "synchronizeStream addVector stream failed: %i\n", ret);

    return APP_ERR_OK;
}

APP_ERROR IndexFlat::getVectors(uint32_t offset, uint32_t num, std::vector<float16_t> &vectors)
{
    uint32_t actualNum;
    if (offset >= ntotal) {
        actualNum = 0;
    } else if (offset + num >= ntotal) {
        actualNum = ntotal - offset;
    } else {
        actualNum = num;
    }

    vectors.resize(actualNum * dims);

    if (faiss::ascend::SocUtils::GetInstance().IsZZCodeFormat()) {
        return getVectorsAiCpu(offset, actualNum, vectors);
    } else {
        return GetVectorsNDFormat(offset, actualNum, blockSize, baseShaped, vectors);
    }
}

void IndexFlat::getBaseEnd()
{
    // 释放用于getBase申请的device内存
    dataVec.clear();
    dataInt8Vec.clear();
    attrsVec.clear();
}

APP_ERROR IndexFlat::getVectorsAiCpu(uint32_t offset, uint32_t num, std::vector<float16_t> &vectors)
{
    if (faiss::ascend::SocUtils::GetInstance().IsAscend910B()) {
        size_t blockSize = static_cast<size_t>(this->blockSize);
        size_t total = offset;
        size_t blockIdx = total / blockSize;
        size_t offsetInBlock = total % blockSize;
        int srcOffset = offsetInBlock * dims;
        auto ret = aclrtMemcpy(vectors.data(), vectors.size() * sizeof(float16_t),
        baseShaped[blockIdx]->data() + srcOffset, vectors.size() * sizeof(float16_t), ACL_MEMCPY_DEVICE_TO_HOST);
        APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "Mem operator error %d", (int)ret);
    } else {
        std::string opName = "TransdataRaw";
        auto streamPtr = resources.getAlternateStreams().back();
        auto stream = streamPtr->GetStream();
        int blockNum = utils::divUp(static_cast<int>(ntotal), blockSize);
        // dataVec、attrsVec需要使用完后清理
        dataVec.resize(static_cast<size_t>(num) * static_cast<size_t>(dims), true);
        attrsVec.resize(blockNum * aicpu::TRANSDATA_RAW_ATTR_IDX_COUNT, true);

        AscendTensor<float16_t, DIMS_2> data(dataVec.data(), {static_cast<int>(num), dims});
        AscendTensor<int64_t, DIMS_2> attrs(attrsVec.data(), {blockNum, aicpu::TRANSDATA_RAW_ATTR_IDX_COUNT});

        size_t blockSize = static_cast<size_t>(this->blockSize);
        for (size_t i = 0; i < num;) {
            size_t total = offset + i;
            size_t offsetInBlock = total % blockSize;
            size_t leftInBlock = blockSize - offsetInBlock;
            size_t leftInData = num - i;
            size_t copyCount = std::min(leftInBlock, leftInData);
            size_t blockIdx = total / blockSize;

            int copy = static_cast<int>(copyCount);
            AscendTensor<float16_t, DIMS_2> dst(data[i].data(), {copy, dims});
            AscendTensor<float16_t, DIMS_4> src(baseShaped[blockIdx]->data(),
                {utils::divUp(this->blockSize, CUBE_ALIGN), utils::divUp(dims, CUBE_ALIGN), CUBE_ALIGN, CUBE_ALIGN});
            AscendTensor<int64_t, DIMS_1> attr = attrs[blockIdx].view();
            attr[aicpu::TRANSDATA_RAW_ATTR_OFFSET_IDX] = offsetInBlock;
            execTransdataShapedrawOp(opName, stream, src, attr, dst);

            i += copyCount;
        }

        auto ret = synchronizeStream(stream);
        APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR,
            "synchronizeStream getVector stream failed: %i\n", ret);

        ret = aclrtMemcpy(vectors.data(), vectors.size() * sizeof(float16_t),
        data.data(), data.getSizeInBytes(), ACL_MEMCPY_DEVICE_TO_HOST);
        APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "Mem operator error %d", (int)ret);
    }
    return APP_ERR_OK;
}

// 将int8类型特征从baseShapedInt8中反分形获取出来，注意dst的shape中第1个维度需要按照CUBE_ALIGN_INT8对齐
APP_ERROR IndexFlat::getInt8VectorsAiCpu(uint32_t offset, uint32_t num, std::vector<int8_t> &vectors)
{
    std::string opName = "TransdataRaw";
    auto streamPtr = resources.getAlternateStreams().back();
    auto stream = streamPtr->GetStream();
    int blockNum = utils::divUp(static_cast<int>(ntotal), blockSize);
    // dataVec、attrsVec需要使用完后清理
    dataInt8Vec.resize(static_cast<size_t>(num) * static_cast<size_t>(dims), true);
    attrsVec.resize(blockNum * aicpu::TRANSDATA_RAW_ATTR_IDX_COUNT, true);

    AscendTensor<int8_t, DIMS_2> data(dataInt8Vec.data(), {static_cast<int>(num), dims});
    AscendTensor<int64_t, DIMS_2> attrs(attrsVec.data(), {blockNum, aicpu::TRANSDATA_RAW_ATTR_IDX_COUNT});

    size_t blockSize = static_cast<size_t>(this->blockSize);
    for (size_t i = 0; i < num;) {
        size_t total = offset + i;
        size_t offsetInBlock = total % blockSize;
        size_t leftInBlock = blockSize - offsetInBlock;
        size_t leftInData = num - i;
        size_t copyCount = std::min(leftInBlock, leftInData);
        size_t blockIdx = total / blockSize;

        int copy = static_cast<int>(copyCount);
        AscendTensor<int8_t, DIMS_2> dst(data[i].data(), {copy, dims});
        AscendTensor<int8_t, DIMS_4> src(baseShapedInt8[blockIdx]->data(),
            {utils::divUp(this->blockSize, CUBE_ALIGN), utils::divUp(dims, CUBE_ALIGN_INT8),
            CUBE_ALIGN, CUBE_ALIGN_INT8});
        AscendTensor<int64_t, DIMS_1> attr = attrs[blockIdx].view();
        attr[aicpu::TRANSDATA_RAW_ATTR_OFFSET_IDX] = offsetInBlock;

        LaunchOpTwoInOneOut<int8_t, DIMS_4, ACL_INT8,
                            int64_t, DIMS_1, ACL_INT64,
                            int8_t, DIMS_2, ACL_INT8>(opName, stream, src, attr, dst);

        i += copyCount;
    }

    auto ret = synchronizeStream(stream);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR,
        "synchronizeStream addVector stream failed: %i\n", ret);

    ret = aclrtMemcpy(vectors.data(), vectors.size() * sizeof(int8_t),
                      data.data(), data.getSizeInBytes(), ACL_MEMCPY_DEVICE_TO_HOST);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "Mem operator error %d", (int)ret);

    return APP_ERR_OK;
}

APP_ERROR IndexFlat::searchImpl(int n, const float16_t *x, int k, float16_t *distances, idx_t *labels)
{
    AscendTensor<float16_t, DIMS_2> queries(const_cast<float16_t *>(x), { n, dims });
    AscendTensor<float16_t, DIMS_2> outDistances(distances, { n, k });
    AscendTensor<idx_t, DIMS_2> outIndices(labels, { n, k });
    outIndices.initValue(std::numeric_limits<idx_t>::max());

    // 避免searchFilterImpl异常未设置maskData=nullptr影响普通search
    maskData = nullptr;

    return searchImpl(queries, k, outDistances, outIndices);
}

size_t IndexFlat::getVecCapacity(size_t vecNum, size_t size) const
{
    size_t minCapacity = 512 * KB;

    // 计算needSize的逻辑可以参考TransdataShaped算子的拷贝逻辑
    // needSize并不是vecNum * dims，因为在分形时，目的地址有可能超过这个范围的（不能被CUBE_ALIGN整除时）
    int dimAlign = dims / CUBE_ALIGN;
    size_t divisor = vecNum / static_cast<size_t>(CUBE_ALIGN);
    size_t remainder = vecNum % static_cast<size_t>(CUBE_ALIGN);
    size_t offset = divisor * static_cast<size_t>(dimAlign * CUBE_ALIGN * CUBE_ALIGN) + \
        remainder * static_cast<size_t>(CUBE_ALIGN);
    size_t needSize = offset + static_cast<size_t>(dimAlign * CUBE_ALIGN * CUBE_ALIGN);
    needSize = std::max(needSize, vecNum * static_cast<size_t>(dims));

    const size_t align = 2 * KB * KB;

    // 1. needSize is smaller than minCapacity
    if (needSize < minCapacity) {
        return minCapacity;
    }

    // 2. needSize is smaller than current code.size, no need to grow
    if (needSize <= size) {
        return size;
    }

    // 3. 2048 is the max code_each_loop, 2048 * dims for operator aligned
    size_t retMemory = utils::roundUp((needSize + 2048 * dims), align);

    // 2048 * dims for operator aligned
    return std::min(retMemory - 2048 * dims, static_cast<size_t>(this->devVecCapacity));
}

// 保存int8特征类型需要新增空间的计算方式，和fp16有较大区别
size_t IndexFlat::getInt8VecCapacity(size_t vecNum, size_t size) const
{
    // 定义最小容量为512KB
    const size_t minCapacity = 512 * 1024;
    // 计算需要的大小，向上取整到1024的倍数，并乘以维度
    const size_t needSize = utils::roundUp(vecNum, 1024) * dims;
    // 如果需要的大小小于最小容量，则返回最小容量
    if (needSize < minCapacity) {
        return minCapacity;
    }

    // 如果需要的大小小于等于给定的大小，则返回给定的大小
    if (needSize <= size) {
        return size;
    }

    // 计算维度1，向上取整到CUBE_ALIGN的倍数
    int dim1 = utils::divUp(this->blockSize, CUBE_ALIGN);
    // 计算维度2，向上取整到CUBE_ALIGN_INT8的倍数
    int dim2 = utils::divUp(this->dims, CUBE_ALIGN_INT8);
    auto devInt8VecCapacity = dim1 * dim2 * CUBE_ALIGN * CUBE_ALIGN_INT8;
    // 返回需要的大小和设备Int8向量容量中的较小值
    return std::min(needSize, static_cast<size_t>(devInt8VecCapacity));
}

void IndexFlat::moveVectorForward(idx_t srcIdx, idx_t dstIdx)
{
    ASCEND_THROW_IF_NOT(srcIdx >= dstIdx);
    size_t blockSizeL = (size_t)this->blockSize;
    size_t cubeAlignL = (size_t)CUBE_ALIGN;
    size_t srcIdx1 = srcIdx / blockSizeL;
    size_t srcIdx2 = srcIdx % blockSizeL;

    size_t dstIdx1 = dstIdx / blockSizeL;
    size_t dstIdx2 = dstIdx % blockSizeL;

    if (!faiss::ascend::SocUtils::GetInstance().IsZZCodeFormat()) {
        RemoveForwardParam param = {
            static_cast<size_t>(srcIdx1), static_cast<size_t>(srcIdx2),
            static_cast<size_t>(dstIdx1), static_cast<size_t>(dstIdx2)
        };
        auto ret = RemoveForwardNDFormat(param, dims, baseShaped);
        ASCEND_THROW_IF_NOT_FMT(ret == ACL_SUCCESS, "RemoveForwardNDFormat error %d", ret);
        return;
    }

    uint32_t dim2 = static_cast<uint32_t>(utils::divUp(dims, CUBE_ALIGN));

    float16_t *srcDataPtr = baseShaped[srcIdx1]->data() + (srcIdx2 / cubeAlignL) * (dim2 * cubeAlignL * cubeAlignL) +
        (srcIdx2 % cubeAlignL) * (cubeAlignL);
    float16_t *dstDataPtr = baseShaped[dstIdx1]->data() + (dstIdx2 / cubeAlignL) * (dim2 * cubeAlignL * cubeAlignL) +
        (dstIdx2 % cubeAlignL) * (cubeAlignL);

    for (uint32_t i = 0; i < dim2; i++) {
        auto ret = aclrtMemcpy(dstDataPtr, cubeAlignL * sizeof(float16_t),
            srcDataPtr, cubeAlignL * sizeof(float16_t), ACL_MEMCPY_DEVICE_TO_DEVICE);
        ASCEND_THROW_IF_NOT_FMT(ret == ACL_SUCCESS, "Mem operator error %d", (int)ret);
        dstDataPtr += cubeAlignL * cubeAlignL;
        srcDataPtr += cubeAlignL * cubeAlignL;
    }
}

void IndexFlat::releaseUnusageSpace(int oldTotal, int remove)
{
    int oldVecSize = utils::divUp(oldTotal, this->blockSize);
    int vecSize = utils::divUp(oldTotal - remove, this->blockSize);

    for (int i = oldVecSize - 1; i >= vecSize; --i) {
        this->baseShaped.at(i)->clear();
    }
}

APP_ERROR IndexFlat::reset()
{
    size_t dvSize = utils::divUp(this->ntotal, this->blockSize);
    for (size_t i = 0; i < dvSize; ++i) {
        baseShaped.at(i)->clear();
    }
    ntotal = 0;

    return APP_ERR_OK;
}

void IndexFlat::runMultisearchTopkCompute(int batch, const std::vector<const AscendTensorBase *> &input,
                                          const std::vector<const AscendTensorBase *> &output, aclrtStream stream)
{
    auto lock = ::ascend::AscendMultiThreadManager::GetWriteLock(multiSearchTopkMtx);
    std::vector<int> keys({batch});
    OpsMngKey opsKey(keys);
    auto ret = DistComputeOpsManager::getInstance().runOp(multiSearchTopkType, opsKey, input, output, stream);
    ASCEND_THROW_IF_NOT_FMT(ret == APP_ERR_OK, "indexFlat run operator failed: %i\n", ret);
}

APP_ERROR IndexFlat::resetOnlineMultisearchTopk()
{
    std::vector<int64_t> shape0_3 { -1, -1, -1 };
    std::vector<int64_t> shape4_7 { -1 };
    std::vector<int64_t> shape8_9 { -1, -1, -1 };
    std::vector<aclTensorDesc *> inputDesc;
    int64_t dimRange0_3[3][2] = {{1, -1}, {1, -1}, {1, -1}};
    int64_t dimRange4_7[1][2] = {{1, -1}};
    int64_t dimRange8_9[3][2] = {{1, -1}, {1, -1}, {1, -1}};
    appendTensorDesc(inputDesc, shape0_3, ACL_FLOAT16, dimRange0_3, ACL_FORMAT_ND);
    appendTensorDesc(inputDesc, shape0_3, ACL_FLOAT16, dimRange0_3, ACL_FORMAT_ND);
    appendTensorDesc(inputDesc, shape0_3, ACL_UINT32, dimRange0_3, ACL_FORMAT_ND);
    appendTensorDesc(inputDesc, shape0_3, ACL_UINT16, dimRange0_3, ACL_FORMAT_ND);
    appendTensorDesc(inputDesc, shape4_7, ACL_INT64, dimRange4_7, ACL_FORMAT_ND);
    appendTensorDesc(inputDesc, shape4_7, ACL_UINT32, dimRange4_7, ACL_FORMAT_ND);
    appendTensorDesc(inputDesc, shape4_7, ACL_UINT32, dimRange4_7, ACL_FORMAT_ND);
    appendTensorDesc(inputDesc, shape4_7, ACL_UINT16, dimRange4_7, ACL_FORMAT_ND);
    std::vector<aclTensorDesc *> outputDesc;
    appendTensorDesc(outputDesc, shape8_9, ACL_FLOAT16, dimRange8_9, ACL_FORMAT_ND);
    appendTensorDesc(outputDesc, shape8_9, ACL_INT64, dimRange8_9, ACL_FORMAT_ND);
    aclopAttr *opAttr = aclopCreateAttr();
    auto ret = aclSetCompileopt(ACL_OP_JIT_COMPILE, "enable");
    if (ret != APP_ERR_OK) {
        destroyOpResource(opAttr, inputDesc, outputDesc);
        APPERR_RETURN_IF_NOT_FMT(ret == APP_ERR_OK, APP_ERR_INNER_ERROR,
            "enable jit compile fail opName:TopkMultisearch : %i\n", ret);
    }

    ret = aclopCompile("TopkMultisearch", inputDesc.size(), inputDesc.data(), outputDesc.size(),
                       outputDesc.data(), opAttr, ACL_ENGINE_SYS, ACL_COMPILE_SYS, nullptr);
    destroyOpResource(opAttr, inputDesc, outputDesc);
    APPERR_RETURN_IF_NOT_FMT(ret == APP_ERR_OK, APP_ERR_INNER_ERROR, "compile TopkMultisearch op failed: %i", ret);
    return APP_ERR_OK;
}

APP_ERROR IndexFlat::resetTopkOnline()
{
    std::vector<aclTensorDesc *> inputDesc;
    std::vector<int64_t> shape0_3 { -1, -1, -1 };
    std::vector<int64_t> shape4 { -1 };
    std::vector<int64_t> shape5_6 { -1, -1 };
    int64_t dimRange0_3[3][2] = {{1, -1}, {1, -1}, {1, -1}};
    appendTensorDesc(inputDesc, shape0_3, ACL_FLOAT16, dimRange0_3, ACL_FORMAT_ND);
    appendTensorDesc(inputDesc, shape0_3, ACL_FLOAT16, dimRange0_3, ACL_FORMAT_ND);
    appendTensorDesc(inputDesc, shape0_3, ACL_UINT32, dimRange0_3, ACL_FORMAT_ND);
    appendTensorDesc(inputDesc, shape0_3, ACL_UINT16, dimRange0_3, ACL_FORMAT_ND);
    int64_t dimRange4[1][2] = {{1, -1}};
    appendTensorDesc(inputDesc, shape4, ACL_INT64, dimRange4, ACL_FORMAT_ND);
    
    std::vector<aclTensorDesc *> outputDesc;
    int64_t dimRange5_6[2][2] = {{1, -1}, {1, -1}};
    appendTensorDesc(outputDesc, shape5_6, ACL_FLOAT16, dimRange5_6, ACL_FORMAT_ND);
    appendTensorDesc(outputDesc, shape5_6, ACL_INT64, dimRange5_6, ACL_FORMAT_ND);
    aclopAttr *opAttr = aclopCreateAttr();
    auto ret = aclSetCompileopt(ACL_OP_JIT_COMPILE, "enable");
    if (ret != APP_ERR_OK) {
        destroyOpResource(opAttr, inputDesc, outputDesc);
        APPERR_RETURN_IF_NOT_FMT(ret == APP_ERR_OK, APP_ERR_INNER_ERROR,
            "IndexFlat aclSetCompileopt enable jit compile fail opName:topkFlat : %i\n", ret);
    }
    ret = aclopCompile("TopkFlat", inputDesc.size(), inputDesc.data(), outputDesc.size(),
                       outputDesc.data(), opAttr, ACL_ENGINE_SYS, ACL_COMPILE_SYS, nullptr);
    destroyOpResource(opAttr, inputDesc, outputDesc);
    
    APPERR_RETURN_IF_NOT_FMT(ret == APP_ERR_OK, APP_ERR_INNER_ERROR, "compile topkFlat op failed: %i", ret);
    return APP_ERR_OK;
}

APP_ERROR IndexFlat::resetOfflineMultisearchTopk(IndexTypeIdx topkType, int flagNum)
{
    auto lock = ::ascend::AscendMultiThreadManager::GetWriteLock(multiSearchTopkMtx);
    multiSearchTopkType = topkType;
    auto topkCompOpReset = [&](std::unique_ptr<AscendOperator> &op, int64_t batch) {
        AscendOpDesc desc("TopkMultisearch");
        int burstLen = BURST_LEN_HIGH;
        auto curBurstsOfBlock = GetBurstsOfBlock(batch, this->blockSize, burstLen);
        std::vector<int64_t> shape0 { 0, batch, this->blockSize };
        std::vector<int64_t> shape1 { 0, batch, curBurstsOfBlock };
        std::vector<int64_t> shape2 { 0, CORE_NUM, SIZE_ALIGN };
        std::vector<int64_t> shape3 { 0, flagNum, FLAG_SIZE };
        std::vector<int64_t> shape4 { aicpu::TOPK_MULTISEARCH_ATTR_IDX_COUNT };
        std::vector<int64_t> shape5 { 0 };
        std::vector<int64_t> shape6 { 0, batch, 0 };

        desc.addInputTensorDesc(ACL_FLOAT16, shape0.size(), shape0.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_FLOAT16, shape1.size(), shape1.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_UINT32, shape2.size(), shape2.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_UINT16, shape3.size(), shape3.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_INT64, shape4.size(), shape4.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_UINT32, shape5.size(), shape5.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_UINT32, shape5.size(), shape5.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_UINT16, shape5.size(), shape5.data(), ACL_FORMAT_ND);

        desc.addOutputTensorDesc(ACL_FLOAT16, shape6.size(), shape6.data(), ACL_FORMAT_ND);
        desc.addOutputTensorDesc(ACL_INT64, shape6.size(), shape6.data(), ACL_FORMAT_ND);

        op.reset();
        op = CREATE_UNIQUE_PTR(AscendOperator, desc);
        return op->init();
    };
    std::map<OpsMngKey, std::unique_ptr<AscendOperator>>& topkComputeOps =
        DistComputeOpsManager::getInstance().getDistComputeOps(topkType);
    for (auto batch : searchBatchSizes) {
        std::vector<int> keys({batch});
        OpsMngKey opsKey(keys);
        topkComputeOps[opsKey] = std::unique_ptr<AscendOperator>(nullptr);
        APPERR_RETURN_IF_NOT_LOG(topkCompOpReset(topkComputeOps[opsKey], batch), APP_ERR_ACL_OP_LOAD_MODEL_FAILED,
            "topk op init failed");
    }
    return APP_ERR_OK;
}

APP_ERROR IndexFlat::computeMultisearchTopkParam(AscendTensor<uint32_t, DIMS_1> &indexOffsetInputs,
    AscendTensor<uint32_t, DIMS_1> &labelOffsetInputs, AscendTensor<uint16_t, DIMS_1> &reorderFlagInputs,
    std::vector<idx_t> &ntotals, std::vector<idx_t> &offsetBlocks) const
{
    size_t indexSize = ntotals.size();
    idx_t blockNum = offsetBlocks[indexSize];
    std::vector<uint32_t> indexOffset(blockNum);
    std::vector<uint32_t> labelOffset(blockNum);
    std::vector<uint16_t> reorderFlag(blockNum);

    for (size_t indexId = 0; indexId < indexSize; ++indexId) {
        int blocks =  static_cast<int>(utils::divUp(ntotals[indexId], blockSize));
        for (int i = 0; i < blocks; ++i) {
            int blockIdx = (static_cast<int>(offsetBlocks[indexId]) + i);
            indexOffset[blockIdx] = static_cast<uint32_t>(indexId);
            labelOffset[blockIdx] = static_cast<uint32_t>(i);
            reorderFlag[blockIdx] = (i == blocks - 1) ? 1 : 0;
        }
    }

    auto ret = aclrtMemcpy(indexOffsetInputs.data(), indexOffsetInputs.getSizeInBytes(), indexOffset.data(),
        indexOffset.size() * sizeof(uint32_t), ACL_MEMCPY_HOST_TO_DEVICE);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "Mem operator error %d", (int)ret);
    ret = aclrtMemcpy(labelOffsetInputs.data(), labelOffsetInputs.getSizeInBytes(), labelOffset.data(),
        labelOffset.size() * sizeof(uint32_t), ACL_MEMCPY_HOST_TO_DEVICE);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "Mem operator error %d", (int)ret);
    ret = aclrtMemcpy(reorderFlagInputs.data(), reorderFlagInputs.getSizeInBytes(), reorderFlag.data(),
        reorderFlag.size() * sizeof(uint16_t), ACL_MEMCPY_HOST_TO_DEVICE);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "Mem operator error %d", (int)ret);

    return APP_ERR_OK;
}

APP_ERROR IndexFlat::searchFilterImpl(int n, const float16_t *x, int k, float16_t *distances, idx_t *labels,
    uint8_t *masks, uint32_t maskRealLen)
{
    auto streamPtr = resources.getDefaultStream();
    auto stream = streamPtr->GetStream();
    auto &mem = resources.getMemoryManager();

    AscendTensor<float16_t, DIMS_2> queries(const_cast<float16_t *>(x), { n, dims });
    AscendTensor<float16_t, DIMS_2> outDistances(mem, { n, k }, stream);
    AscendTensor<int64_t, DIMS_2> outIndices(mem, { n, k }, stream);

    // 通过成员变量保存mask，后面可以复用searchPaged接口
    // 预留空间至少一个block长度
    maskLen = maskRealLen;
    // maskRealLen是1个query对应实际的mask长度，算子搬运需要按照32B对齐
    int32_t alignLen = utils::roundUp(static_cast<int32_t>(maskRealLen), CUBE_ALIGN_INT8);
    // 当ntotal较小时，mask内存长度为blockMaskSize，不小于一个block的底库的对应的长度
    int32_t memoryLen = std::max(alignLen, this->blockMaskSize);
    AscendTensor<uint8_t, DIMS_2> maskTensor(mem, { n, memoryLen }, stream);
    auto ret = aclrtMemcpy(maskTensor.data(), maskTensor.getSizeInBytes(),
                           masks, static_cast<size_t>(n) * static_cast<size_t>(maskRealLen),
                           ACL_MEMCPY_HOST_TO_DEVICE);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "copy mask to device failed: %i\n", ret);

    // 为了和普通的search共用接口，通过成员变量将mask数据传递到searchPaged
    maskData = maskTensor.data();

    int pageNum = static_cast<int>(utils::divUp(this->ntotal, pageSize));
    for (int pageId = 0; pageId < pageNum; ++pageId) {
        APP_ERROR ret = searchPaged(pageId, pageNum, queries, outDistances, outIndices);
        APPERR_RETURN_IF(ret, ret);
    }
    // 调用完成后设置mask失效，避免影响普通search
    maskData = nullptr;

    // memcpy data back from dev to host
    ret = aclrtMemcpy(distances, n * k * sizeof(float16_t),
                      outDistances.data(), outDistances.getSizeInBytes(),
                      ACL_MEMCPY_DEVICE_TO_HOST);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "aclrtMemcpy dist to host failed: %d", ret);

    ret = aclrtMemcpy(labels, n * k * sizeof(idx_t),
                      outIndices.data(), outIndices.getSizeInBytes(),
                      ACL_MEMCPY_DEVICE_TO_HOST);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "aclrtMemcpy indices to host failed: %d", ret);

    return APP_ERR_OK;
}

APP_ERROR IndexFlat::runTopkOnlineOp(int batch, int flagNum, topkOpParams &params, aclrtStream stream)
{
    AscendOpDesc desc("TopkFlat");
    int burstLen = BURST_LEN_HIGH;
    auto curBurstsOfBlock = GetBurstsOfBlock(batch, this->blockSize, burstLen);
    std::vector<int64_t> shape0 { static_cast<int64_t>(params.dists.getSize(0)), batch, this->blockSize };
    std::vector<int64_t> shape1 { static_cast<int64_t>(params.maxdists.getSize(0)), batch, curBurstsOfBlock };
    std::vector<int64_t> shape2 { static_cast<int64_t>(params.sizes.getSize(0)), CORE_NUM, SIZE_ALIGN };
    std::vector<int64_t> shape3 { static_cast<int64_t>(params.flags.getSize(0)), flagNum, FLAG_SIZE };
    std::vector<int64_t> shape4 { aicpu::TOPK_FLAT_ATTR_IDX_COUNT };
    std::vector<int64_t> shape5 { batch, static_cast<int64_t>(params.outdists.getSize(1)) };

    std::vector<aclTensorDesc *> inputDesc;

    appendTensorDesc(inputDesc, shape0, ACL_FLOAT16, ACL_FORMAT_ND);
    appendTensorDesc(inputDesc, shape1, ACL_FLOAT16, ACL_FORMAT_ND);
    appendTensorDesc(inputDesc, shape2, ACL_UINT32, ACL_FORMAT_ND);
    appendTensorDesc(inputDesc, shape3, ACL_UINT16, ACL_FORMAT_ND);
    appendTensorDesc(inputDesc, shape4, ACL_INT64, ACL_FORMAT_ND);

    std::vector<aclTensorDesc *> outputDesc;
    appendTensorDesc(outputDesc, shape5, ACL_FLOAT16, ACL_FORMAT_ND);
    appendTensorDesc(outputDesc, shape5, ACL_INT64, ACL_FORMAT_ND);

    std::vector<aclDataBuffer *> topkOpInput;
    topkOpInput.emplace_back(aclCreateDataBuffer(params.dists.data(), params.dists.getSizeInBytes()));
    topkOpInput.emplace_back(aclCreateDataBuffer(params.maxdists.data(), params.maxdists.getSizeInBytes()));
    topkOpInput.emplace_back(aclCreateDataBuffer(params.sizes.data(), params.sizes.getSizeInBytes()));
    topkOpInput.emplace_back(aclCreateDataBuffer(params.flags.data(), params.flags.getSizeInBytes()));
    topkOpInput.emplace_back(aclCreateDataBuffer(params.attrs.data(), params.attrs.getSizeInBytes()));

    std::vector<aclDataBuffer *> topkOpOutput;
    topkOpOutput.emplace_back(aclCreateDataBuffer(params.outdists.data(), params.outdists.getSizeInBytes()));
    topkOpOutput.emplace_back(aclCreateDataBuffer(params.outlabel.data(), params.outlabel.getSizeInBytes()));

    aclopAttr *opAttr = aclopCreateAttr();
    
    auto ret = aclopExecuteV2("TopkFlat", inputDesc.size(),
                              inputDesc.data(),
                              topkOpInput.data(),
                              outputDesc.size(),
                              outputDesc.data(),
                              topkOpOutput.data(),
                              opAttr, stream);
    destroyOpResource(opAttr, inputDesc, outputDesc, topkOpInput, topkOpOutput);
    return ret;
}

} // namespace ascend
