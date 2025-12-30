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


#include "ascenddaemon/impl/IndexSQ.h"

#include <algorithm>
#include <unordered_map>

#include "ascenddaemon/impl/AuxIndexStructures.h"
#include "ascenddaemon/utils/Limits.h"
#include "ascenddaemon/utils/StaticUtils.h"
#include "ascenddaemon/utils/DistComputeOpsManager.h"
#include "common/utils/LogUtils.h"
#include "common/utils/CommonUtils.h"
#include "common/utils/OpLauncher.h"
#include "ascend/utils/fp16.h"
#include "ops/cpukernel/impl/utils/kernel_shared_def.h"

namespace ascend {
namespace {
const int KB = 1024;
const int BIT_OF_UINT16 = 16;
const int RETAIN_BLOCK_SIZE = 2048;
const std::vector<int> CID_BLOCKS = {32 * KB, 64 * KB, 128 * KB, 256 * KB, 512 * KB, 1024 * KB};
const int DIM_384 = 384;
const int DIM_256 = 256;
const int DIM_512 = 512;
const int BLOCK_COUNT = 16;
}

IndexSQ::IndexSQ(int dim, bool filterable, int64_t resourceSize, int dBlockSize)
    : Index(dim, resourceSize),
      codeBlockSize(dBlockSize),
      devVecCapacity(0),
      filterable(filterable)
{
    ASCEND_THROW_IF_NOT(dims % CUBE_ALIGN == 0);

    isTrained = false;

    // supported batch size
    if (dim == 768) { // dim 768场景受限算子内存，最大仅支持batch32
        searchBatchSizes = { 32, 16, 8, 4, 2, 1 };
    } else {
        searchBatchSizes = { 48, 32, 16, 8, 4, 2, 1 };
    }
    // Double the BURST_LEN after round up, hence here we multiply 2
    this->burstsOfBlock = (this->codeBlockSize + BURST_LEN - 1) / BURST_LEN * 2;
    this->computeBlockSize = this->codeBlockSize - RETAIN_BLOCK_SIZE;
    this->cidBlockSize = CID_BLOCKS.back();
    this->pageSize = this->computeBlockSize * BLOCK_COUNT;

    reset();

    int dim1 = utils::divUp(this->computeBlockSize, CUBE_ALIGN);
    int dim2 = utils::divUp(this->dims, CUBE_ALIGN);
    this->devVecCapacity = dim1 * dim2 * CUBE_ALIGN * CUBE_ALIGN;

    // init vand and vmul
    AscendTensor<uint16_t, DIMS_2> andData({ ID_BLOCKS, HELPER_SIZE });
    AscendTensor<float16_t, DIMS_2> mulData({ ID_BLOCKS, HELPER_SIZE });

    for (unsigned int i = 0; i < ID_BLOCKS; ++i) {
        std::vector<uint16_t> andDataTemp(HELPER_SIZE);
        std::vector<float> mulDataFp32Temp(HELPER_SIZE);
        std::vector<uint16_t> mulDataFp16Temp(HELPER_SIZE);
        std::fill_n(andDataTemp.data(), HELPER_SIZE, (1UL << i) + (1UL << (BIT_OF_UINT8 + i)));
        std::fill_n(mulDataFp32Temp.data(), HELPER_SIZE, 1.0 / (1UL << i));

        std::transform(std::begin(mulDataFp32Temp), std::end(mulDataFp32Temp), std::begin(mulDataFp16Temp),
            [](float temp) { return faiss::ascend::fp16(temp).data; });

        auto error = aclrtMemcpy(andData[i].data(), HELPER_SIZE * sizeof(uint16_t),
                                 andDataTemp.data(), HELPER_SIZE * sizeof(uint16_t),
                                 ACL_MEMCPY_HOST_TO_DEVICE);
        ASCEND_THROW_IF_NOT_FMT(error == ACL_SUCCESS, "failed to aclrtMemcpy (error %d)", (int)error);
        error = aclrtMemcpy(mulData[i].data(), HELPER_SIZE * sizeof(float16_t),
                            mulDataFp16Temp.data(), HELPER_SIZE * sizeof(uint16_t),
                            ACL_MEMCPY_HOST_TO_DEVICE);
        ASCEND_THROW_IF_NOT_FMT(error == ACL_SUCCESS, "failed to aclrtMemcpy (error %d)", (int)error);
    }

    this->vand = std::move(andData);
    this->vmul = std::move(mulData);
}

IndexSQ::~IndexSQ() {}

APP_ERROR IndexSQ::init()
{
    auto ret = resetTopkCompOp();
    APPERR_RETURN_IF_NOT_FMT(ret == APP_ERR_OK, ret, "failed to reset topk op: %d", ret);
    ret = resetMultisearchTopkCompOp();
    APPERR_RETURN_IF_NOT_FMT(ret == APP_ERR_OK, ret, "failed to reset multisearch topk op: %d", ret);
    return resetCidFilterOperator();
}

APP_ERROR IndexSQ::reset()
{
    this->codes.clear();
    this->preCompute.clear();
    this->ntotal = 0;
    cidIdx.clear();
    cidVal.clear();
    timestamps.clear();

    return APP_ERR_OK;
}

APP_ERROR IndexSQ::addVectors(size_t numVecs, const uint8_t *data, const float *preCompute)
{
    APPERR_RETURN_IF(numVecs == 0, APP_ERR_OK);
    VALUE_UNUSED(preCompute);
    size_t blockSize = static_cast<size_t>(this->computeBlockSize);
    int newVecSize = static_cast<int>(utils::divUp(this->ntotal + numVecs, blockSize));
    int vecSize = static_cast<int>(utils::divUp(this->ntotal, blockSize));
    ASCEND_THROW_IF_NOT(vecSize == static_cast<int>(codes.size()));

    for (int i = 0; i < newVecSize - vecSize; ++i) {
        this->codes.emplace_back(CREATE_UNIQUE_PTR(DeviceVector<uint8_t>, MemorySpace::DEVICE_HUGEPAGE));
        if (vecSize + i > 0) {
            this->codes.at(vecSize + i)->resize(this->devVecCapacity, true);
        }
    }

    // resize little to avoid waste for idx 0
    if (ntotal < static_cast<idx_t>(computeBlockSize)) {
        auto capacity = getVecCapacity(this->ntotal + numVecs, this->codes.at(0)->size());
        this->codes.at(0)->resize(capacity, true);
    }

    std::string opName = "TransdataShaped";
    auto &mem = resources.getMemoryManager();
    auto streamPtr = resources.getDefaultStream();
    auto stream = streamPtr->GetStream();
    AscendTensor<uint8_t, DIMS_2> dataTensor(mem, {static_cast<int>(numVecs), dims}, stream);
    auto ret = aclrtMemcpy(dataTensor.data(), dataTensor.getSizeInBytes(),
                           data, static_cast<size_t>(numVecs) * static_cast<size_t>(dims) * sizeof(uint8_t),
                           ACL_MEMCPY_HOST_TO_DEVICE);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "Mem operator error %d", ret);

    int blockNum = static_cast<int>(utils::divUp(static_cast<int>(ntotal) + numVecs, blockSize));
    AscendTensor<int64_t, DIMS_2> attrs(mem, {blockNum, aicpu::TRANSDATA_SHAPED_ATTR_IDX_COUNT}, stream);

    for (size_t i = 0; i < static_cast<size_t>(numVecs);) {
        size_t total = ntotal + i;
        size_t offsetInBlock = total % blockSize;
        size_t leftInBlock = blockSize - offsetInBlock;
        size_t leftInData = static_cast<size_t>(numVecs) - i;
        size_t copyCount = std::min(leftInBlock, leftInData);
        size_t blockIdx = total / blockSize;

        int copy = static_cast<int>(copyCount);
        AscendTensor<uint8_t, DIMS_2> src(dataTensor[i].data(), {copy, dims});
        AscendTensor<uint8_t, DIMS_4> dst((this->codes[blockIdx]->data()),
            {utils::divUp(this->computeBlockSize, CUBE_ALIGN), utils::divUp(dims, CUBE_ALIGN), CUBE_ALIGN, CUBE_ALIGN});
        AscendTensor<int64_t, DIMS_1> attr = attrs[blockIdx].view();
        attr[aicpu::TRANSDATA_SHAPED_ATTR_NTOTAL_IDX] = offsetInBlock;

        LaunchOpTwoInOneOut<uint8_t, DIMS_2, ACL_UINT8,
                            int64_t, DIMS_1, ACL_INT64,
                            uint8_t, DIMS_4, ACL_UINT8>(opName, stream, src, attr, dst);

        i += copyCount;
    }

    ret = synchronizeStream(stream);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "synchronize addVector stream failed: %i\n", ret);

    return APP_ERR_OK;
}

size_t IndexSQ::getVecCapacity(size_t vecNum, size_t size) const
{
    // alloc 512 KB memory at least, 1024 for K
    // 首先dist算子都是要求容量条数1024对齐的
    // 其次算子要求的容量条数大小必须满足 capacity >= 7 * (capacity + 2048) // 8 // 64 // 16 * 64 * 16
    // 如果这里的256维和512维不进行minCapcity限制的话，后续计算出的容量条数可能为6144，会不符合上述公式导致算子报错
    const std::unordered_map<int, size_t> minCapcityMap {
        { DIM_256, DIM_256 * KB * 8 },
        { DIM_384, DIM_384 * KB * 2 },
        { DIM_512, DIM_512 * KB * 8 }
    };
    auto it = minCapcityMap.find(dims);
    const size_t minCapacity = (it != minCapcityMap.end()) ? it->second : 512 * KB;

    const size_t needSize = vecNum * dims;
    // the align memory is 2 MB, 1024 for K
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
    const size_t reserveMemory = 2048 * dims;
    size_t retMemory = utils::roundUp((needSize + reserveMemory), align);

    // 2048 * dims for operator aligned
    return std::min(retMemory - reserveMemory, static_cast<size_t>(this->devVecCapacity));
}

APP_ERROR IndexSQ::addVectorsWithIds(size_t numVecs, const uint8_t *data, const Index::idx_t* ids,
    const float *preCompute)
{
    if (numVecs == 0) {
        return APP_ERR_OK;
    }

    // 1. add ids
    auto ret = addIds(numVecs, ids);
    APPERR_RETURN_IF_NOT_FMT(ret == APP_ERR_OK, ret, "addIds failed: %d", ret);

    // 2. add vector
    return addVectors(numVecs, data, preCompute);
}

APP_ERROR IndexSQ::addIds(size_t numVecs, const Index::idx_t* ids)
{
    APPERR_RETURN_IF(numVecs == 0 || ids == nullptr, APP_ERR_OK);

    int newVecSize = static_cast<int>(utils::divUp(this->ntotal + numVecs, this->cidBlockSize));
    APPERR_RETURN_IF(newVecSize <= 0, APP_ERR_OK);
    size_t vecSize = cidIdx.size();

    // 1. emplace DeviceVector
    size_t addNum = static_cast<size_t>(newVecSize) > vecSize ? static_cast<size_t>(newVecSize) - vecSize : 0;
    for (size_t i = 0; i < addNum; ++i) {
        this->cidIdx.emplace_back(CREATE_UNIQUE_PTR(DeviceVector<uint8_t>));
        this->cidVal.emplace_back(CREATE_UNIQUE_PTR(DeviceVector<uint32_t>));
        this->timestamps.emplace_back(CREATE_UNIQUE_PTR(DeviceVector<uint32_t>));
    }

    // 2. resize vector except the new last
    for (int i = 0; i < newVecSize - 1; ++i) {
        this->cidIdx.at(i)->resize(this->cidBlockSize, true);
        this->cidVal.at(i)->resize(this->cidBlockSize, true);
        this->timestamps.at(i)->resize(this->cidBlockSize, true);
    }

    // 3. resize the new last size
    int lastNum = static_cast<int>(this->ntotal) + static_cast<int>(numVecs) - (newVecSize - 1) * this->cidBlockSize;
    int lastBlockSize = getLastCidBlockSize(lastNum);
    this->cidIdx.at(newVecSize - 1)->resize(lastBlockSize, true);
    this->cidVal.at(newVecSize - 1)->resize(lastBlockSize, true);
    this->timestamps.at(newVecSize - 1)->resize(lastBlockSize, true);

    // 4. save ids
    
    return saveIds(numVecs, ids);
}

int IndexSQ::getLastCidBlockSize(int cidNum) const
{
    for (const auto size : CID_BLOCKS) {
        if (size >= cidNum) {
            return size;
        }
    }
    return CID_BLOCKS.back();
}

APP_ERROR IndexSQ::saveIds(int numVecs, const Index::idx_t *ids)
{
    int vecIndex = static_cast<int>(this->ntotal / static_cast<idx_t>(this->cidBlockSize));
    int dVecIndex = static_cast<int>(this->ntotal % static_cast<idx_t>(this->cidBlockSize));

    ASCEND_THROW_IF_NOT(static_cast<int>(this->cidIdx.size()) > vecIndex);
    ASCEND_THROW_IF_NOT(static_cast<int>(this->cidVal.size()) > vecIndex);
    ASCEND_THROW_IF_NOT(static_cast<int>(this->timestamps.size()) > vecIndex);
    ASCEND_THROW_IF_NOT(this->cidBlockSize > dVecIndex);

    return saveIdsHostCpu(numVecs, ids, vecIndex, dVecIndex);
}

APP_ERROR IndexSQ::saveIdsHostCpu(int numVecs, const Index::idx_t *ids, int vecBefore, int dVecBefore)
{
    std::vector<uint8_t> cidIdxTemp(numVecs);
    std::vector<uint32_t> cidValTemp(numVecs);
    std::vector<uint32_t> timesTemp(numVecs);

    for (int i = 0; i < numVecs; ++i) {
        const int blockSize = 32;
        auto cid = static_cast<uint8_t>((ids[i] >> 42) & 0x7F); // the 42 bits on the right is others,
                                                                // 0x7F means 1111111, for get 7 bits
        cidIdxTemp[i] = 1UL << (cid / blockSize);
        cidValTemp[i] = 1UL << (cid % blockSize);
        timesTemp[i] = static_cast<uint32_t>((ids[i] >> 10) & 0xFFFFFFFF); // the 10 bits on the right is useless;
    }

    if ((dVecBefore + numVecs) > this->cidBlockSize) {
        int vecIndex = utils::divUp((numVecs + dVecBefore), this->cidBlockSize);
        int dVecLast = (numVecs + dVecBefore) % this->cidBlockSize;
        for (int i = 0; i < vecIndex; i++) {
            int srcPtrOffset = 0;
            int dstPtrOffset = dVecBefore;
            int sizeLen = this->cidBlockSize - dVecBefore;

            if (i > 0) {
                srcPtrOffset = (cidBlockSize - dVecBefore) + (i - 1) * cidBlockSize;
                dstPtrOffset = 0;
                if (i == vecIndex - 1) {
                    sizeLen = (dVecLast == 0)? this->cidBlockSize : dVecLast;
                } else {
                    sizeLen = this->cidBlockSize;
                }
            }

            auto ret = aclrtMemcpy(this->cidIdx[vecBefore + i]->data() + dstPtrOffset, sizeLen * sizeof(uint8_t),
                cidIdxTemp.data() + srcPtrOffset, sizeLen * sizeof(uint8_t), ACL_MEMCPY_HOST_TO_DEVICE);
            APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "Mem operator error %d", (int)ret);

            ret = aclrtMemcpy(this->cidVal[vecBefore + i]->data() + dstPtrOffset, sizeLen * sizeof(uint32_t),
                cidValTemp.data() + srcPtrOffset, sizeLen * sizeof(uint32_t), ACL_MEMCPY_HOST_TO_DEVICE);
            APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "Mem operator error %d", (int)ret);

            ret = aclrtMemcpy(this->timestamps[vecBefore + i]->data() + dstPtrOffset, sizeLen * sizeof(uint32_t),
                timesTemp.data() + srcPtrOffset, sizeLen * sizeof(uint32_t), ACL_MEMCPY_HOST_TO_DEVICE);
            APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "Mem operator error %d", (int)ret);
        }
    } else {
        auto ret = aclrtMemcpy(this->cidIdx[vecBefore]->data() + dVecBefore, numVecs * sizeof(uint8_t),
            cidIdxTemp.data(), numVecs * sizeof(uint8_t), ACL_MEMCPY_HOST_TO_DEVICE);
        APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "Mem operator error %d", (int)ret);

        ret = aclrtMemcpy(this->cidVal[vecBefore]->data() + dVecBefore, numVecs * sizeof(uint32_t),
            cidValTemp.data(), numVecs * sizeof(uint32_t), ACL_MEMCPY_HOST_TO_DEVICE);
        APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "Mem operator error %d", (int)ret);

        ret = aclrtMemcpy(this->timestamps[vecBefore]->data() + dVecBefore, numVecs * sizeof(uint32_t),
            timesTemp.data(), numVecs * sizeof(uint32_t), ACL_MEMCPY_HOST_TO_DEVICE);
        APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "Mem operator error %d", (int)ret);
    }
    return APP_ERR_OK;
}

APP_ERROR IndexSQ::getVectors(uint32_t offset, uint32_t num, std::vector<uint8_t> &vectors)
{
    if (this->ntotal == 0 || offset >= this->ntotal) {
        return APP_ERR_INVALID_PARAM;
    }
    size_t actualNum = (offset + num >= this->ntotal) ? (this->ntotal - offset) : num;
    size_t actualSize = actualNum * static_cast<size_t>(this->dims);
    vectors.resize(actualSize);

    return getVectors(offset, num, vectors.data());
}

APP_ERROR IndexSQ::getVectors(uint32_t offset, uint32_t num, uint8_t *vectors)
{
    if (this->ntotal == 0 || offset >= this->ntotal) {
        return APP_ERR_INVALID_PARAM;
    }

    uint32_t actualNum = (offset + num >= this->ntotal) ? (this->ntotal - offset) : num;
    return getVectorsAiCpu(offset, actualNum, vectors);
}

APP_ERROR IndexSQ::getVectorsAiCpu(uint32_t offset, uint32_t num, uint8_t *vectors)
{
    std::string opName = "TransdataRaw";

    auto streamPtr = resources.getAlternateStreams().back();
    auto stream = streamPtr->GetStream();
    int blockNum = utils::divUp(static_cast<int>(ntotal), computeBlockSize);
    // dataVec、attrsVec需要使用完后清理
    dataVec.resize(static_cast<size_t>(num) * static_cast<size_t>(dims), true);
    attrsVec.resize(blockNum * aicpu::TRANSDATA_RAW_ATTR_IDX_COUNT, true);
    AscendTensor<uint8_t, DIMS_2> data(dataVec.data(), {static_cast<int>(num), dims});
    AscendTensor<int64_t, DIMS_2> attrs(attrsVec.data(), {blockNum, aicpu::TRANSDATA_RAW_ATTR_IDX_COUNT});

    size_t blockSize = static_cast<size_t>(this->computeBlockSize);
    for (size_t i = 0; i < num;) {
        size_t total = offset + i;
        size_t offsetInBlock = total % blockSize;
        size_t leftInBlock = blockSize - offsetInBlock;
        size_t leftInData = num - i;
        size_t copyCount = std::min(leftInBlock, leftInData);
        size_t blockIdx = total / blockSize;

        int copy = static_cast<int>(copyCount);
        AscendTensor<uint8_t, DIMS_2> dst(data[i].data(), {copy, dims});
        AscendTensor<uint8_t, DIMS_4> src(codes[blockIdx]->data(),
            {utils::divUp(this->codeBlockSize, CUBE_ALIGN), utils::divUp(dims, CUBE_ALIGN),
            CUBE_ALIGN, CUBE_ALIGN});
        AscendTensor<int64_t, DIMS_1> attr = attrs[blockIdx].view();
        attr[aicpu::TRANSDATA_RAW_ATTR_OFFSET_IDX] = offsetInBlock;

        LaunchOpTwoInOneOut<uint8_t, DIMS_4, ACL_UINT8,
                            int64_t, DIMS_1, ACL_INT64,
                            uint8_t, DIMS_2, ACL_UINT8>(opName, stream, src, attr, dst);

        i += copyCount;
    }

    auto ret = synchronizeStream(stream);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR,
        "synchronizeStream addVector stream failed: %i\n", ret);

    size_t vectorSize = static_cast<size_t>(num) * static_cast<size_t>(this->dims) * sizeof(uint8_t);
    ret = aclrtMemcpy(vectors, vectorSize, data.data(), data.getSizeInBytes(), ACL_MEMCPY_DEVICE_TO_HOST);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "Mem operator error %d", (int)ret);

    return APP_ERR_OK;
}

void IndexSQ::getBaseEnd()
{
    // 释放用于getBase申请的device内存
    dataVec.clear();
    attrsVec.clear();
}

void IndexSQ::updateTrainedValue(AscendTensor<float16_t, DIMS_1> &trainedMin,
    AscendTensor<float16_t, DIMS_1> &trainedDiff)
{
    int dimMin = trainedMin.getSize(0);
    int dimDiff = trainedDiff.getSize(0);
    ASCEND_THROW_IF_NOT_FMT(dimMin == dimDiff && dimMin == this->dims,
        "sq trained data's shape invalid.(%d, %d) vs (dim:%d)", dimMin, dimDiff, this->dims);

    AscendTensor<float16_t, DIMS_1> minTensor({ dimMin });
    AscendTensor<float16_t, DIMS_1> diffTensor({ dimDiff });
    minTensor.copyFromSync(trainedMin);
    diffTensor.copyFromSync(trainedDiff);
    vMin = std::move(minTensor);
    vDiff = std::move(diffTensor);

    this->isTrained = true;
}

APP_ERROR IndexSQ::searchBatched(int64_t n, const float16_t *x, int64_t k, float16_t *distance, idx_t *labels,
    uint8_t* masks)
{
    // 因内存优化改动算子，mask长度需按照1024对齐
    constexpr idx_t alignSize = 1024;
    int64_t ntotalAlign = static_cast<int64_t>(utils::roundUp(this->ntotal, alignSize));
    uint32_t ntotalAlignMaskSize = static_cast<uint32_t>(ntotalAlign / BIT_OF_UINT8);

    auto streamPtr = resources.getDefaultStream();
    auto stream = streamPtr->GetStream();
    auto &mem = resources.getMemoryManager();
    AscendTensor<uint8_t, DIMS_1, size_t> newMaskData(
        mem, { static_cast<size_t>(n) * ntotalAlignMaskSize }, stream);
    newMaskData.zero();

    uint32_t originMaskSize = utils::divUp(ntotal, BIT_OF_UINT8);
    for (uint64_t i = 0; i < static_cast<uint64_t>(n); i++) {
        uint64_t newMaskOffset = i * ntotalAlignMaskSize;
        uint64_t originMaskOffset = i * originMaskSize;
        auto ret = aclrtMemcpy(newMaskData.data() + newMaskOffset, newMaskData.getSizeInBytes() - newMaskOffset,
            masks + originMaskOffset, originMaskSize, ACL_MEMCPY_DEVICE_TO_DEVICE);
        APPERR_RETURN_IF(ret, ret);
    }

    uint8_t* newMasks = newMaskData.data();
    uint64_t maskOffset = 0;
    size_t size = searchBatchSizes.size();
    if (n > 1 && size > 0) {
        int64_t searched = 0;
        for (size_t i = 0; i < size; i++) {
            int64_t batchSize = searchBatchSizes[i];
            if ((n - searched) >= batchSize) {
                int64_t page = (n - searched) / batchSize;
                for (int64_t j = 0; j < page; j++) {
                    maskOffset = static_cast<uint64_t>(searched) * ntotalAlignMaskSize;
                    APP_ERROR ret = searchFilterImpl(batchSize, x + searched * this->dims, k, distance + searched * k,
                        labels + searched * k, newMasks + maskOffset, ntotalAlignMaskSize);
                    APPERR_RETURN_IF(ret, ret);
 
                    searched += batchSize;
                }
            }
        }

        for (int64_t i = searched; i < n; i++) {
            maskOffset = static_cast<uint64_t>(i) * ntotalAlignMaskSize;
            APP_ERROR ret = searchFilterImpl(1, x + i * this->dims, k, distance + i * k, labels + i * k,
                newMasks + maskOffset, ntotalAlignMaskSize);
            APPERR_RETURN_IF(ret, ret);
        }
        return APP_ERR_OK;
    } else {
        return searchFilterImpl(n, x, k, distance, labels, newMasks, ntotalAlignMaskSize);
    }
}

APP_ERROR IndexSQ::searchBatched(int64_t n, const float16_t *x, int64_t k, float16_t *distance, idx_t *labels,
    uint64_t, uint32_t* filters)
{
    auto streamPtr = resources.getDefaultStream();
    auto stream = streamPtr->GetStream();
    auto &mem = resources.getMemoryManager();
    int64_t ntotalAlign = static_cast<int64_t>(utils::roundUp(this->ntotal, this->cidBlockSize));

    // 1. compute mask
    AscendTensor<uint8_t, DIMS_1, size_t> maskData(
        mem, { static_cast<size_t>(n) * ntotalAlign / BIT_OF_UINT8 }, stream);
    maskData.zero();
    computeMask(n, filters, maskData);

    // 2. search by page, this is different from Index::searchBatched
    uint8_t* masks = maskData.data();
    uint64_t maskOffset = 0;
    uint32_t maskLen = static_cast<uint32_t>(ntotalAlign / BIT_OF_UINT8);
    size_t size = searchBatchSizes.size();
    if (n > 1 && size > 0) {
        int64_t searched = 0;
        for (size_t i = 0; i < size; i++) {
            int64_t batchSize = searchBatchSizes[i];
            if ((n - searched) >= batchSize) {
                int64_t page = (n - searched) / batchSize;
                for (int64_t j = 0; j < page; j++) {
                    maskOffset =  static_cast<uint64_t>(searched) * maskLen;
                    APP_ERROR ret = searchFilterImpl(batchSize, x + searched * this->dims, k, distance + searched * k,
                        labels + searched * k, masks + maskOffset, maskLen);
                    APPERR_RETURN_IF(ret, ret);

                    searched += batchSize;
                }
            }
        }

        for (int64_t i = searched; i < n; i++) {
            maskOffset = static_cast<uint64_t>(i) * maskLen;
            APP_ERROR ret = searchFilterImpl(1, x + i * this->dims, k, distance + i * k, labels + i * k,
                masks + maskOffset, maskLen);
            APPERR_RETURN_IF(ret, ret);
        }
        return APP_ERR_OK;
    } else {
        return searchFilterImpl(n, x, k, distance, labels, masks, maskLen);
    }

    return Index::searchBatched(n, x, k, distance, labels, maskData.data());
}

APP_ERROR IndexSQ::computeMask(int n, uint32_t* filters, AscendTensor<uint8_t, DIMS_1, size_t>& masks)
{
    auto streamPtr = resources.getDefaultStream();
    auto stream = streamPtr->GetStream();
    auto &mem = resources.getMemoryManager();
    int batchNum = static_cast<int>(utils::divUp(this->ntotal, this->cidBlockSize));

    // 1. costruct the id operator param
    AscendTensor<uint32_t, DIMS_2> filterData(filters, { n, FILTER_SIZE });
    AscendTensor<uint16_t, DIMS_3> opResult(
        reinterpret_cast<uint16_t *>(masks.data()), { n, batchNum, cidBlockSize / BIT_OF_UINT16 });
    AscendTensor<uint32_t, DIMS_2> opTsFilter(mem, { n, TS_SIZE }, stream);
    AscendTensor<uint16_t, DIMS_4> opFlag(mem, { n, batchNum, CORE_NUM, FLAG_SIZE }, stream);
    std::vector<std::vector<uint32_t>> opMaskFilter(n, std::vector<uint32_t>(ID_BLOCKS * MASK_SIZE));

    AscendTensor<int32_t, DIMS_3> maskFilter(mem, { n, ID_BLOCKS, MASK_SIZE }, stream);
    AscendTensor<int32_t, DIMS_2> timeFilter({ n, TS_SIZE });
    // 2. run the disance operator to compute the mask
    for (int nIdx = 0; nIdx < n; ++nIdx) {
        std::vector<int32_t> maskFilterTemp(MASK_SIZE);
        for (int bIdx = 0; bIdx < ID_BLOCKS; ++bIdx) {
            std::fill_n(maskFilterTemp.data(), MASK_SIZE, filterData[nIdx][bIdx].value());
            auto ret = aclrtMemcpy(maskFilter[nIdx][bIdx].data(), MASK_SIZE * sizeof(int32_t),
                                   maskFilterTemp.data(), MASK_SIZE * sizeof(int32_t), ACL_MEMCPY_HOST_TO_DEVICE);
            APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "Mem operator error %d", (int)ret);
        }

        std::vector<int32_t> timeFilterTemp(TS_SIZE);
        timeFilterTemp[0] = -static_cast<int32_t>(filterData[nIdx][ID_BLOCKS].value());
        timeFilterTemp[1] = -static_cast<int32_t>(filterData[nIdx][ID_BLOCKS + 1].value());

        auto ret = aclrtMemcpy(timeFilter[nIdx].data(), TS_SIZE * sizeof(int32_t),
                               timeFilterTemp.data(), TS_SIZE * sizeof(int32_t), ACL_MEMCPY_HOST_TO_DEVICE);
        APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "Mem operator error %d", (int)ret);

        for (int i = 0; i < batchNum; ++i) {
            const int blockSize = static_cast<int>(cidIdx[i]->capacity());
            AscendTensor<uint8_t, DIMS_1> idx(cidIdx[i]->data(), { blockSize });
            AscendTensor<int32_t, DIMS_1> val(reinterpret_cast<int32_t *>(cidVal[i]->data()), { blockSize });
            AscendTensor<int32_t, DIMS_1> ts(reinterpret_cast<int32_t *>(timestamps[i]->data()), {  blockSize });
            AscendTensor<uint16_t, DIMS_1> result(opResult[nIdx][i].data(), { blockSize / BIT_OF_UINT16 });
            AscendTensor<uint16_t, DIMS_2> flag(opFlag[nIdx][i].data(), { CORE_NUM, FLAG_SIZE });
            AscendTensor<int32_t, DIMS_2> oneQueryMaskFilter(maskFilter[nIdx].data(), { ID_BLOCKS, MASK_SIZE });
            AscendTensor<int32_t, DIMS_1> oneQueryTimeFilter(timeFilter[nIdx].data(), { TS_SIZE });

            std::vector<const AscendTensorBase *> input {&idx, &val, &ts, &oneQueryMaskFilter,
                &oneQueryTimeFilter, &vand, &vmul};
            std::vector<const AscendTensorBase *> output {&result, &flag};
            // output的flag在算子内部并未真正使用，后续如需使用请修改算子
            runCidFilterOperator(blockSize, input, output, stream);
        }
    }

    // 3. wait all the op task compute
    auto ret = synchronizeStream(stream);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "synchronizeStream failed: %i\n", ret);

    return APP_ERR_OK;
}

void IndexSQ::parseOneFilter(const AscendTensor<uint32_t, DIMS_2> &oneFilter, int offset,
    std::vector<uint32_t> &oneMaskFilter, std::vector<uint32_t> &oneTsFilter) const
{
    for (int i = 0; i < ID_BLOCKS; ++i) {
        std::fill_n(oneMaskFilter.data() + i * MASK_SIZE, MASK_SIZE, oneFilter[offset][i].value());
    }
    oneTsFilter[0] = -oneFilter[offset][ID_BLOCKS].value();
    oneTsFilter[1] = -oneFilter[offset][ID_BLOCKS + 1].value();
}

void IndexSQ::getSingleFilter(int n, uint32_t *filter,
    std::vector<std::vector<std::vector<uint32_t>>> &maskFilters,
    std::vector<std::vector<std::vector<uint32_t>>> &tsFilters) const
{
    AscendTensor<uint32_t, DIMS_2> filterData(filter, { n, FILTER_SIZE });
    for (int i = 0; i < n; ++i) {
        std::vector<std::vector<uint32_t>> oneQueryMaskFilter;
        std::vector<std::vector<uint32_t>> oneQueryTsFilter;
        std::vector<uint32_t> oneMaskFilter(ID_BLOCKS * MASK_SIZE);  // 算子需要maskFilter数据每个数值*64
        std::vector<uint32_t> oneTsFilter(TS_SIZE);  // 算子需要tsFilter长度为8，实际只使用到前两位
        parseOneFilter(filterData, i, oneMaskFilter, oneTsFilter);

        oneQueryMaskFilter.emplace_back(oneMaskFilter);
        oneQueryTsFilter.emplace_back(oneTsFilter);

        maskFilters.emplace_back(oneQueryMaskFilter);
        tsFilters.emplace_back(oneQueryTsFilter);
    }

    return;
}

APP_ERROR IndexSQ::getMultiFilter(int n, size_t indexSize, const std::vector<void *> &filters,
    std::vector<std::vector<std::vector<uint32_t>>> &maskFilters,
    std::vector<std::vector<std::vector<uint32_t>>> &tsFilters) const
{
    APPERR_RETURN_IF_NOT_FMT((filters.size() == static_cast<size_t>(n)), APP_ERR_INNER_ERROR,
        "filters size:%zu is error, n:%d", filters.size(), n);
    for (int i = 0; i < n; i++) {
        AscendTensor<uint32_t, DIMS_2> oneQueryFilter(reinterpret_cast<uint32_t *>(filters[i]),
            { static_cast<int>(indexSize), FILTER_SIZE });
        std::vector<std::vector<uint32_t>> oneQueryMaskFilter;
        std::vector<std::vector<uint32_t>> oneQueryTsFilter;
        for (size_t j = 0; j < indexSize; j++) {
            std::vector<uint32_t> oneMaskFilter(ID_BLOCKS * MASK_SIZE);  // 算子需要maskFilter数据每个数值*64
            std::vector<uint32_t> oneTsFilter(TS_SIZE);  // 算子需要tsFilter长度为8，实际只使用到前两位
            parseOneFilter(oneQueryFilter, j, oneMaskFilter, oneTsFilter);
            oneQueryMaskFilter.emplace_back(oneMaskFilter);
            oneQueryTsFilter.emplace_back(oneTsFilter);
        }
        maskFilters.emplace_back(oneQueryMaskFilter);
        tsFilters.emplace_back(oneQueryTsFilter);
    }
    return APP_ERR_OK;
}

APP_ERROR IndexSQ::getMaskAndTsFilter(int n, size_t indexSize, const std::vector<void *> &filters,
    bool isMultiFilterSearch,
    std::vector<std::vector<std::vector<uint32_t>>> &maskFilters,
    std::vector<std::vector<std::vector<uint32_t>>> &tsFilters) const
{
    if (isMultiFilterSearch) {
        return getMultiFilter(n, indexSize, filters, maskFilters, tsFilters);
    }
    getSingleFilter(n, reinterpret_cast<uint32_t *>(filters[0]), maskFilters, tsFilters);
    return APP_ERR_OK;
}

APP_ERROR IndexSQ::computeMask(std::vector<Index *> &indexes, int n, const std::vector<void *> &filters,
    AscendTensor<uint8_t, DIMS_2, int64_t> &maskData, AscendTensor<int, DIMS_1> &maskOffset, bool isMultiFilterSearch)
{
    auto streamPtr = resources.getDefaultStream();
    auto stream = streamPtr->GetStream();
    auto &memManager = resources.getMemoryManager();
    // 1. costruct the id operator param
    // 这里使用三维数组获取解析后的filter，一维为n个指针数组，二维为indexes.size个filter，三维为解析后的filter
    std::vector<std::vector<std::vector<uint32_t>>> maskFilters;
    std::vector<std::vector<std::vector<uint32_t>>> tsFilters;
    auto ret = getMaskAndTsFilter(n, indexes.size(), filters, isMultiFilterSearch, maskFilters, tsFilters);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "getMaskAndTsFilter fail,ret = %d", ret);

    // 2. run the disance operator to compute the mask
    AscendTensor<uint16_t, DIMS_2> opFlag(memManager, { CORE_NUM, FLAG_SIZE }, stream);
    AscendTensor<uint16_t, DIMS_2, int64_t> opResult(reinterpret_cast<uint16_t *>(maskData.data()),
        { static_cast<int64_t>(n), maskData.getSize(1) * BIT_OF_UINT8 / BIT_OF_UINT16 });

    AscendTensor<int32_t, DIMS_4> maskFilterTensor({n, static_cast<int>(indexes.size()), ID_BLOCKS, MASK_SIZE});
    AscendTensor<int32_t, DIMS_3> timeFilterTensor(memManager, {n, static_cast<int>(indexes.size()), TS_SIZE}, stream);

    std::vector<uint32_t> allMaskFilter(static_cast<size_t>(n) * indexes.size() * ID_BLOCKS * MASK_SIZE, 0);
    std::vector<uint32_t> allTsFilter(static_cast<size_t>(n) * indexes.size() * TS_SIZE, 0);
    for (int i = 0; i < n; i++) {
        for (size_t j = 0; j < indexes.size(); j++) {
            auto filterIdx = isMultiFilterSearch ? j : 0;  // 单filter时，所有index使用同一个filter，index维度仅有一个元素，下标固定为0
            auto maskOffset = (static_cast<size_t>(i) * indexes.size() + j) * ID_BLOCKS * MASK_SIZE;
            auto memRet = memcpy_s(allMaskFilter.data() + maskOffset,
                (allMaskFilter.size() - maskOffset) * sizeof(uint32_t), maskFilters[i][filterIdx].data(),
                maskFilters[i][filterIdx].size() * sizeof(uint32_t));
            APPERR_RETURN_IF_NOT_FMT(memRet == EOK, APP_ERR_INNER_ERROR, "memcpy_s error %d", memRet);
            auto tsOffset = (static_cast<size_t>(i) * indexes.size() + j) * TS_SIZE;
            memRet = memcpy_s(allTsFilter.data() + tsOffset, (allTsFilter.size() - tsOffset)  * sizeof(uint32_t),
                tsFilters[i][filterIdx].data(), tsFilters[i][filterIdx].size() * sizeof(uint32_t));
            APPERR_RETURN_IF_NOT_FMT(memRet == EOK, APP_ERR_INNER_ERROR, "memcpy_s error %d", memRet);
        }
    }

    ret = aclrtMemcpy(maskFilterTensor.data(), maskFilterTensor.getSizeInBytes(),
        allMaskFilter.data(), allMaskFilter.size() * sizeof(uint32_t), ACL_MEMCPY_HOST_TO_DEVICE);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "Mem operator error %d", ret);
    ret = aclrtMemcpy(timeFilterTensor.data(), timeFilterTensor.getSizeInBytes(),
        allTsFilter.data(), allTsFilter.size() * sizeof(uint32_t), ACL_MEMCPY_HOST_TO_DEVICE);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "Mem operator error %d", ret);

    for (int nIdx = 0; nIdx < n; ++nIdx) {
        for (size_t idxId = 0; idxId < indexes.size(); ++idxId) {
            int offset = maskOffset[idxId];
            auto index = dynamic_cast<IndexSQ *>(indexes[idxId]);
            
            for (size_t bIdx = 0; bIdx < index->cidIdx.size(); ++bIdx) {
                const int blockSize = static_cast<int>(index->cidIdx[bIdx]->capacity());
                AscendTensor<uint8_t, DIMS_1> idx(index->cidIdx[bIdx]->data(), { blockSize });
                AscendTensor<int32_t, DIMS_1> val(reinterpret_cast<int32_t *>(index->cidVal[bIdx]->data()),
                    { blockSize });
                AscendTensor<int32_t, DIMS_1> ts(reinterpret_cast<int32_t *>(index->timestamps[bIdx]->data()),
                    { blockSize });
                AscendTensor<uint16_t, DIMS_1> result(opResult[nIdx].data() + offset / BIT_OF_UINT16,
                    { blockSize / BIT_OF_UINT16 });
                AscendTensor<int32_t, DIMS_2> maskFilter(maskFilterTensor[nIdx][idxId].data(), {ID_BLOCKS, MASK_SIZE});
                AscendTensor<int32_t, DIMS_1> timeFilter(timeFilterTensor[nIdx][idxId].data(), { TS_SIZE });

                std::vector<const AscendTensorBase *> input {&idx, &val, &ts, &maskFilter, &timeFilter, &vand, &vmul};
                std::vector<const AscendTensorBase *> output {&result, &opFlag};
                // output的opFlag在算子内部并未真正使用，后续如需使用请修改算子
                runCidFilterOperator(blockSize, input, output, stream);
                offset += blockSize;
            }
        }
    }
    // 3. wait all the op task compute
    auto err = synchronizeStream(stream);
    APPERR_RETURN_IF_NOT_FMT(err == ACL_SUCCESS, APP_ERR_INNER_ERROR, "synchronizeStream failed: %i\n", err);
 
    return APP_ERR_OK;
}

APP_ERROR IndexSQ::searchBatched(std::vector<Index*> indexes, int64_t n, const float16_t *x, int64_t k,
    float16_t *distances, idx_t *labels, uint32_t, std::vector<void *> &filters, bool isMultiFilterSearch)
{
    APP_ERROR ret = APP_ERR_OK;

    // check param
    size_t size = searchBatchSizes.size();
    if (size == 0 || n <= 0) {
        return APP_ERR_INVALID_PARAM;
    }

    auto streamPtr = resources.getDefaultStream();
    auto stream = streamPtr->GetStream();
    auto &mem = resources.getMemoryManager();

    // 1. compute mask
    // compute meta data
    int64_t maskBitLen = 0;

    std::vector<int> maskOffsetTemp(indexes.size());
    AscendTensor<int, DIMS_1> maskOffset(maskOffsetTemp.data(), { static_cast<int>(indexes.size()) });

    for (size_t i = 0; i < indexes.size(); ++i) {
        int64_t tmpLen = 0;
        auto index = dynamic_cast<IndexSQ*>(indexes[i]);
        size_t cidSize = index->cidIdx.size();
        for (size_t j = 0; j < cidSize; ++j) {
            tmpLen += static_cast<int64_t>(index->cidIdx[j]->capacity());
        }
 
        maskOffset[i] = maskBitLen;
        maskBitLen += tmpLen;
    }

    // compute mask
    const int64_t maskLen = maskBitLen / BIT_OF_UINT8;
    AscendTensor<uint8_t, DIMS_2, int64_t> maskData(mem, { n, maskLen }, stream);
    maskData.zero();
    ret = computeMask(indexes, n, filters, maskData, maskOffset, isMultiFilterSearch);
    APPERR_RETURN_IF(ret, ret);

    // 2. search filter by page
    // init result
    initSearchResult(indexes.size(), n, k, distances, labels);

    int64_t searched = 0;
    for (size_t i = 0; i < size; i++) {
        int64_t batchSize = searchBatchSizes[i];
        if ((n - searched) >= batchSize) {
            int64_t page = (n - searched) / batchSize;
            for (int64_t j = 0; j < page; j++) {
                size_t offset = static_cast<size_t>(searched) * maskLen;
                AscendTensor<uint8_t, DIMS_2, int64_t> masks(maskData.data() + offset, {batchSize, maskLen});

                ret = searchFilterImpl(indexes, n, batchSize, x + searched * this->dims, k,
                    distances + searched * k, labels + searched * k, masks, maskOffset);
                APPERR_RETURN_IF(ret, ret);
                searched += batchSize;
            }
        }
    }

    for (int64_t i = searched; i < n; i++) {
        size_t offset = static_cast<size_t>(i) * maskLen;
        AscendTensor<uint8_t, DIMS_2, int64_t> masks(maskData.data() + offset, { 1, maskLen });
 
        ret = searchFilterImpl(indexes, n, 1, x + i * this->dims, k, distances + i * k, labels + i * k, masks,
            maskOffset);
        APPERR_RETURN_IF(ret, ret);
    }

    reorder(indexes.size(), n, k, distances, labels);
    return APP_ERR_OK;
}

void IndexSQ::moveTSAttrForward(size_t srcIdx, size_t dstIdx)
{
    if (!filterable) {
        return;
    }

    moveAttrForward(srcIdx, dstIdx, cidIdx, cidBlockSize);
    moveAttrForward(srcIdx, dstIdx, cidVal, cidBlockSize);
    moveAttrForward(srcIdx, dstIdx, timestamps, cidBlockSize);
}

void IndexSQ::moveVectorForward(idx_t srcIdx, idx_t dstIdx)
{
    ASCEND_THROW_IF_NOT(srcIdx >= dstIdx);
    if (srcIdx == dstIdx) {
        return;
    }

    moveTSAttrForward(static_cast<size_t>(srcIdx), static_cast<size_t>(dstIdx));

    int srcIdx1 = static_cast<int>(srcIdx) / this->computeBlockSize;
    int srcIdx2 = static_cast<int>(srcIdx) % this->computeBlockSize;
    
    int dstIdx1 = static_cast<int>(dstIdx) / this->computeBlockSize;
    int dstIdx2 = static_cast<int>(dstIdx) % this->computeBlockSize;
    int dim2 = utils::divUp(this->dims, CUBE_ALIGN);

    uint8_t *srcDataPtr = this->codes[srcIdx1]->data() +
        (srcIdx2 / CUBE_ALIGN) * (dim2 * CUBE_ALIGN * CUBE_ALIGN) + (srcIdx2 % CUBE_ALIGN) * (CUBE_ALIGN);
    uint8_t *dstDataPtr = this->codes[dstIdx1]->data() +
        (dstIdx2 / CUBE_ALIGN) * (dim2 * CUBE_ALIGN * CUBE_ALIGN) + (dstIdx2 % CUBE_ALIGN) * (CUBE_ALIGN);

    for (int i = 0; i < dim2; i++) {
        auto err = aclrtMemcpy(dstDataPtr, CUBE_ALIGN * sizeof(uint8_t),
                               srcDataPtr, CUBE_ALIGN * sizeof(uint8_t), ACL_MEMCPY_DEVICE_TO_DEVICE);
        ASCEND_THROW_IF_NOT_FMT(err == EOK, "Memcpy error %d", (int)err);
        dstDataPtr += CUBE_ALIGN * CUBE_ALIGN;
        srcDataPtr += CUBE_ALIGN * CUBE_ALIGN;
    }
}

void IndexSQ::releaseAttrUnusageSpace(size_t originNum, size_t removeNum)
{
    if (!filterable) {
        return;
    }

    ASCEND_THROW_IF_NOT_FMT(originNum >= removeNum, "num error, originNum[%zu] removeNum[%zu]", originNum, removeNum);
    size_t newSize = utils::divUp(originNum - removeNum, static_cast<size_t>(cidBlockSize));
    cidIdx.resize(newSize);
    cidVal.resize(newSize);
    timestamps.resize(newSize);
}

void IndexSQ::releaseUnusageSpace(int oldTotal, int remove)
{
    releaseAttrUnusageSpace(oldTotal, remove);

    int oldVecSize = utils::divUp(oldTotal, this->computeBlockSize);
    int vecSize = utils::divUp(oldTotal - remove, this->computeBlockSize);

    for (int i = 0; i < oldVecSize - vecSize; ++i) {
        this->codes.pop_back();
    }
}

// 该接口的output中flag在算子内部中并未实际使用
void IndexSQ::runCidFilterOperator(int batch,
                                   const std::vector<const AscendTensorBase *> &input,
                                   const std::vector<const AscendTensorBase *> &output,
                                   aclrtStream stream) const
{
    IndexTypeIdx indexType = IndexTypeIdx::ITI_SQ_CID_FILTER;
    std::vector<int> keys({batch, dims});
    OpsMngKey opsKey(keys);
    auto ret = DistComputeOpsManager::getInstance().runOp(indexType, opsKey, input, output, stream);
    ASCEND_THROW_IF_NOT_FMT(ret == APP_ERR_OK, "run operator failed: %i\n", ret);
}

APP_ERROR IndexSQ::resetCidFilterOperator() const
{
    std::string opTypeName = "CidFilter";
    IndexTypeIdx indexType = IndexTypeIdx::ITI_SQ_CID_FILTER;
    for (auto blockSize : CID_BLOCKS) {
        std::vector<int64_t> idxShape({ blockSize });
        std::vector<int64_t> valShape({ blockSize });
        std::vector<int64_t> tsShape({ blockSize });
        std::vector<int64_t> maskFilterShape({ ID_BLOCKS, MASK_SIZE });
        std::vector<int64_t> tsFilterShape({ TS_SIZE });
        std::vector<int64_t> andDataShape({ ID_BLOCKS, HELPER_SIZE });
        std::vector<int64_t> mulDataShape({ ID_BLOCKS, HELPER_SIZE });
        std::vector<int64_t> resultShape({ blockSize / BIT_OF_UINT16 });
        std::vector<int64_t> flagShape({ CORE_NUM, FLAG_SIZE });

        std::vector<std::pair<aclDataType, std::vector<int64_t>>> input {
            { ACL_UINT8, idxShape },
            { ACL_INT32, valShape },
            { ACL_INT32, tsShape },
            { ACL_INT32, maskFilterShape },
            { ACL_INT32, tsFilterShape },
            { ACL_UINT16, andDataShape },
            { ACL_FLOAT16, mulDataShape }
        };
        std::vector<std::pair<aclDataType, std::vector<int64_t>>> output {
            { ACL_UINT16, resultShape },
            { ACL_UINT16, flagShape }
        };
        std::vector<int> keys({blockSize, dims});
        OpsMngKey opsKey(keys);
        auto ret = DistComputeOpsManager::getInstance().resetOp(opTypeName, indexType, opsKey, input, output);
        APPERR_RETURN_IF_NOT_FMT(ret == APP_ERR_OK, ret, "op init failed: %i", ret);
    }
    return APP_ERR_OK;
}

void IndexSQ::runTopkCompute(int batch,
                             const std::vector<const AscendTensorBase *> &input,
                             const std::vector<const AscendTensorBase *> &output,
                             aclrtStream stream) const
{
    IndexTypeIdx indexType = IndexTypeIdx::ITI_TOPK_FLAT;
    std::vector<int> keys({batch, dims, this->codeBlockSize});
    OpsMngKey opsKey(keys);
    auto ret = DistComputeOpsManager::getInstance().runOp(indexType, opsKey, input, output, stream);
    ASCEND_THROW_IF_NOT_FMT(ret == APP_ERR_OK, "run operator failed: %i\n", ret);
}

APP_ERROR IndexSQ::resetTopkCompOp() const
{
    std::string opTypeName = "TopkFlat";
    IndexTypeIdx indexType = IndexTypeIdx::ITI_TOPK_FLAT;
    for (auto batch : searchBatchSizes) {
        std::vector<int64_t> shape0 { 0, batch, this->codeBlockSize };
        std::vector<int64_t> shape1 { 0, batch, this->burstsOfBlock };
        std::vector<int64_t> shape2 { 0, CORE_NUM, SIZE_ALIGN };
        std::vector<int64_t> shape3 { 0, FLAG_NUM, FLAG_SIZE };
        std::vector<int64_t> shape4 { aicpu::TOPK_FLAT_ATTR_IDX_COUNT };
        std::vector<int64_t> shape5 { batch, 0 };

        std::vector<std::pair<aclDataType, std::vector<int64_t>>> input {
            { ACL_FLOAT16, shape0 },
            { ACL_FLOAT16, shape1 },
            { ACL_UINT32, shape2 },
            { ACL_UINT16, shape3 },
            { ACL_INT64, shape4 }
        };
        std::vector<std::pair<aclDataType, std::vector<int64_t>>> output {
            { ACL_FLOAT16, shape5 },
            { ACL_INT64, shape5 }
        };
        std::vector<int> keys({batch, dims, this->codeBlockSize});
        OpsMngKey opsKey(keys);
        auto ret = DistComputeOpsManager::getInstance().resetOp(opTypeName, indexType, opsKey, input, output);
        APPERR_RETURN_IF_NOT_FMT(ret == APP_ERR_OK, ret, "op init failed: %i", ret);
    }
    return APP_ERR_OK;
}

void IndexSQ::runMultisearchTopkCompute(int batch,
                                        const std::vector<const AscendTensorBase *> &input,
                                        const std::vector<const AscendTensorBase *> &output,
                                        aclrtStream stream) const
{
    IndexTypeIdx indexType = IndexTypeIdx::ITI_TOPK_MULTISEARCH;
    std::vector<int> keys({batch, codeBlockSize});
    OpsMngKey opsKey(keys);
    auto ret = DistComputeOpsManager::getInstance().runOp(indexType, opsKey, input, output, stream);
    ASCEND_THROW_IF_NOT_FMT(ret == APP_ERR_OK, "run operator failed: %i\n", ret);
}

APP_ERROR IndexSQ::resetMultisearchTopkCompOp() const
{
    std::string opTypeName = "TopkMultisearch";
    IndexTypeIdx indexType = IndexTypeIdx::ITI_TOPK_MULTISEARCH;
    for (auto batch : searchBatchSizes) {
        std::vector<int64_t> shape0 { 0, batch, this->codeBlockSize };
        std::vector<int64_t> shape1 { 0, batch, this->burstsOfBlock };
        std::vector<int64_t> shape2 { 0, CORE_NUM, SIZE_ALIGN };
        std::vector<int64_t> shape3 { 0, FLAG_NUM, FLAG_SIZE };
        std::vector<int64_t> shape4 { aicpu::TOPK_MULTISEARCH_ATTR_IDX_COUNT };
        std::vector<int64_t> shape5 { 0 };
        std::vector<int64_t> shape6 { 0, batch, 0 };

        std::vector<std::pair<aclDataType, std::vector<int64_t>>> input {
            { ACL_FLOAT16, shape0 },
            { ACL_FLOAT16, shape1 },
            { ACL_UINT32, shape2 },
            { ACL_UINT16, shape3 },
            { ACL_INT64, shape4 },
            { ACL_UINT32, shape5 },
            { ACL_UINT32, shape5 },
            { ACL_UINT16, shape5 }
        };
        std::vector<std::pair<aclDataType, std::vector<int64_t>>> output {
            { ACL_FLOAT16, shape6 },
            { ACL_INT64, shape6 }
        };
        std::vector<int> keys({batch, codeBlockSize});
        OpsMngKey opsKey(keys);
        auto ret = DistComputeOpsManager::getInstance().resetOp(opTypeName, indexType, opsKey, input, output);
        APPERR_RETURN_IF_NOT_FMT(ret == APP_ERR_OK, ret, "op init failed: %i", ret);
    }
    return APP_ERR_OK;
}

APP_ERROR IndexSQ::computeMultisearchTopkParam(AscendTensor<uint32_t, DIMS_1> &indexOffsetInputs,
    AscendTensor<uint32_t, DIMS_1> &labelOffsetInputs, AscendTensor<uint16_t, DIMS_1> &reorderFlagInputs,
    std::vector<idx_t> &ntotals, std::vector<idx_t> &offsetBlocks) const
{
    size_t indexSize = ntotals.size();
    idx_t blockNum = offsetBlocks[indexSize];
    std::vector<uint32_t> indexOffset(blockNum);
    std::vector<uint32_t> labelOffset(blockNum);
    std::vector<uint16_t> reorderFlag(blockNum);

    for (size_t indexId = 0; indexId < indexSize; ++indexId) {
        int blocks = static_cast<int>(utils::divUp(ntotals[indexId], static_cast<idx_t>(computeBlockSize)));
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

APP_ERROR IndexSQ::resetSqDistOperator(std::string opTypeName, IndexTypeIdx indexType) const
{
    for (auto batch : searchBatchSizes) {
        std::vector<int64_t> queryShape({ batch, dims });
        std::vector<int64_t> codeShape({ this->codeBlockSize / CUBE_ALIGN, this->dims / CUBE_ALIGN,
            CUBE_ALIGN, CUBE_ALIGN });
        std::vector<int64_t> vdiffShape({ this->dims });
        std::vector<int64_t> vminShape({ this->dims });
        std::vector<int64_t> sizeShape({ CORE_NUM, SIZE_ALIGN });
        std::vector<int64_t> resultShape({ batch, this->codeBlockSize });
        std::vector<int64_t> precompShape;
        // L2是最小值 IP是最大值
        std::vector<int64_t> extremumResultShape({ batch, this->burstsOfBlock });
        std::vector<int64_t> flagShape({ FLAG_NUM, FLAG_SIZE });

        std::vector<std::pair<aclDataType, std::vector<int64_t>>> input {
            { ACL_FLOAT16, queryShape },
            { ACL_UINT8, codeShape },
            { ACL_FLOAT16, vdiffShape },
            { ACL_FLOAT16, vminShape },
            { ACL_UINT32, sizeShape }
        };

        if (opTypeName == "DistanceSQ8L2Mins") {
            precompShape = { this->codeBlockSize };
            input = {
                { ACL_FLOAT16, queryShape },
                { ACL_UINT8, codeShape },
                { ACL_FLOAT, precompShape },
                { ACL_FLOAT16, vdiffShape },
                { ACL_FLOAT16, vminShape },
                { ACL_UINT32, sizeShape }
            };
        }

        std::vector<std::pair<aclDataType, std::vector<int64_t>>> output {
            { ACL_FLOAT16, resultShape },
            { ACL_FLOAT16, extremumResultShape },
            { ACL_UINT16, flagShape }
        };
        std::vector<int> keys({batch, dims, codeBlockSize});
        OpsMngKey opsKey(keys);
        auto ret = DistComputeOpsManager::getInstance().resetOp(opTypeName, indexType, opsKey, input, output);
        APPERR_RETURN_IF_NOT_FMT(ret == APP_ERR_OK, ret, "op init failed: %i", ret);
    }

    return APP_ERR_OK;
}

APP_ERROR IndexSQ::resetSqDistMaskOperator(std::string opTypeName, IndexTypeIdx indexMaskType) const
{
    for (auto batch : searchBatchSizes) {
        std::vector<int64_t> queryShape({ batch, dims });
        std::vector<int64_t> maskShape({ batch, utils::divUp(this->codeBlockSize, 8) }); // divUp to 8
        std::vector<int64_t> codeShape({ this->codeBlockSize / CUBE_ALIGN, this->dims / CUBE_ALIGN,
            CUBE_ALIGN, CUBE_ALIGN });
        std::vector<int64_t> vdiffShape({ this->dims });
        std::vector<int64_t> sizeShape({ CORE_NUM, SIZE_ALIGN });
        std::vector<int64_t> vminShape({ this->dims });
        std::vector<int64_t> resultShape({ batch, this->codeBlockSize });
        std::vector<int64_t> precompShape;
        // L2是最小值 IP是最大值
        std::vector<int64_t> extremumResultShape({ batch, this->burstsOfBlock });
        std::vector<int64_t> flagShape({ FLAG_NUM, FLAG_SIZE });

        std::vector<std::pair<aclDataType, std::vector<int64_t>>> input {
            { ACL_FLOAT16, queryShape },
            { ACL_UINT8, maskShape },
            { ACL_UINT8, codeShape },
            { ACL_FLOAT16, vdiffShape },
            { ACL_FLOAT16, vminShape },
            { ACL_UINT32, sizeShape }
        };

        if (opTypeName == "DistanceMaskedSQ8L2Mins") {
            precompShape = { this->codeBlockSize };
            input = {
                { ACL_FLOAT16, queryShape },
                { ACL_UINT8, maskShape },
                { ACL_UINT8, codeShape },
                { ACL_FLOAT, precompShape },
                { ACL_FLOAT16, vdiffShape },
                { ACL_FLOAT16, vminShape },
                { ACL_UINT32, sizeShape }
            };
        }

        std::vector<std::pair<aclDataType, std::vector<int64_t>>> output {
            { ACL_FLOAT16, resultShape },
            { ACL_FLOAT16, extremumResultShape },
            { ACL_UINT16, flagShape }
        };
    std::vector<int> keys({batch, dims, codeBlockSize});
        OpsMngKey opsKey(keys);
        auto ret = DistComputeOpsManager::getInstance().resetOp(opTypeName, indexMaskType, opsKey, input, output);
        APPERR_RETURN_IF_NOT_FMT(ret == APP_ERR_OK, ret, "op init failed: %i", ret);
    }

    return APP_ERR_OK;
}

} // ascend