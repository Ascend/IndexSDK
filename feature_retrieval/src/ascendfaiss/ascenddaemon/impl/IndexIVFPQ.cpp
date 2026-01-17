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

#include "ascenddaemon/impl/IndexIVFPQ.h"
#include <algorithm>
#include <atomic>
#include <cmath>
#include <random>
#include <iostream>
#include <fstream>
#include <vector>
#include <cstring>
#include <chrono>
#include <omp.h>
#include <sys/time.h>
#include <faiss/utils/random.h>

#include "ascenddaemon/impl/AuxIndexStructures.h"
#include "ascenddaemon/utils/Limits.h"
#include "common/utils/CommonUtils.h"
#include "common/utils/LogUtils.h"
#include "common/utils/OpLauncher.h"
#include "ops/cpukernel/impl/utils/kernel_shared_def.h"

namespace ascend {
const int KB = 1024;
const int MAX_ITER_NUM = 25;
const int IVF_PQ_BLOCK_SIZE = 16384 * 16;
const int BYTES_PER_FLOAT = 4;
const int MAX_TOPK = 320;
IndexIVFPQ::IndexIVFPQ(int numList, int dim, int M, int nbits, int nprobes, int64_t resourceSize)
    : IndexIVF(numList, dim * BYTES_PER_FLOAT, dim, nprobes, resourceSize), blockSize(IVF_PQ_BLOCK_SIZE), M(M)
{
    ASCEND_THROW_IF_NOT(dim % CUBE_ALIGN == 0);
    int dim1 = utils::divUp(blockSize, CUBE_ALIGN);
    int dim2PQ = utils::divUp(M, CUBE_ALIGN);
    devPQVecCapacity = dim1 * dim2PQ * CUBE_ALIGN * CUBE_ALIGN;
    isTrained = false;
    // supported batch size
    searchBatchSizes = {64, 32, 16, 8, 4, 2, 1};
    listVecNum = std::vector<size_t>(numList, 0);
    basePQCoder.resize(numList);
    pBasePQCoder = reinterpret_cast<uint8_t*>(0xffffffffffffffff);
    blockNum = 32;
    this->nbits = nbits;
    ksub = (1 << nbits);
    dsub = utils::divUp(dim, M);
    centroidsOnDevice = CREATE_UNIQUE_PTR(DeviceVector<float>, MemorySpace::DEVICE_HUGEPAGE);
    centroidsOnDevice->resize(numList * dim);
    centroidsSqrSumOnDevice = CREATE_UNIQUE_PTR(DeviceVector<float>, MemorySpace::DEVICE_HUGEPAGE);
    centroidsSqrSumOnDevice->resize(numList);
    initializeCodeBook(M, nbits, dim/M);
    auto ret = resetL1TopkOp();
    ASCEND_THROW_IF_NOT_MSG(ret == APP_ERR_OK, "resetL1TopkOp failed!");
    ret = resetL1DistOp();
    ASCEND_THROW_IF_NOT_MSG(ret == APP_ERR_OK, "resetL1DistOp failed!");
    ret = resetL2DistOp();
    ASCEND_THROW_IF_NOT_MSG(ret == APP_ERR_OK, "resetL2DistOp failed!");
    ret = resetL3TopkOp();
    ASCEND_THROW_IF_NOT_MSG(ret == APP_ERR_OK, "resetL3TopkOp failed!");
    ret = resetL3DistOp();
    ASCEND_THROW_IF_NOT_MSG(ret == APP_ERR_OK, "resetL3DistOp failed!");
}

IndexIVFPQ::~IndexIVFPQ() { }

APP_ERROR IndexIVFPQ::reset()
{
    IndexIVF::reset();
    for (size_t i = 0; i < basePQCoder.size(); i++) {
        basePQCoder[i].clear();
    }
    basePQCoder.clear();
    basePQCoder.resize(numLists);

    listVecNum.clear();
    listVecNum.resize(numLists, 0);

    if (centroidsOnDevice) {
        centroidsOnDevice->clear();
        centroidsOnDevice->resize(numLists * dims);
    }

    isTrained = false;

    return APP_ERR_OK;
}

void IndexIVFPQ::initializeCodeBook(int M, int nbits, int dsubs)
{
    int ksubs = 1 << nbits;
    codeBookOnDevice = CREATE_UNIQUE_PTR(DeviceVector<float>, MemorySpace::DEVICE_HUGEPAGE);
    codeBookOnDevice->resize(M * dsubs * ksubs, true);
}

uint64_t IndexIVFPQ::getActualRngSeed(const int seed)
{
    return (seed >= 0) ? seed : static_cast<uint64_t>(std::chrono::high_resolution_clock::now()
            .time_since_epoch()
            .count());
}

size_t IndexIVFPQ::getPQVecCapacity(size_t vecNum, size_t size, int M) const
{
    size_t minCapacity = 512 * static_cast<size_t>(KB);

    int vecAlign = M / CUBE_ALIGN;
    size_t divisor = vecNum / static_cast<size_t>(CUBE_ALIGN);
    size_t remainder = vecNum % static_cast<size_t>(CUBE_ALIGN);

    size_t offset = divisor * static_cast<size_t>(vecAlign * CUBE_ALIGN * CUBE_ALIGN) + \
    remainder * static_cast<size_t>(CUBE_ALIGN);
    size_t needSize = offset + static_cast<size_t>(vecAlign * CUBE_ALIGN * CUBE_ALIGN);

    needSize = std::max(needSize, vecNum * static_cast<size_t>(M));

    const size_t align = 2 * KB * KB;

    if (needSize < minCapacity) {
        return minCapacity;
    }
    if (needSize <= size) {
        return size;
    }
    size_t retMemory = utils::roundUp((needSize + 2048 * M), align);
    return std::min(retMemory - 2048 * M, static_cast<size_t>(devPQVecCapacity));
}

APP_ERROR IndexIVFPQ::resizeBasePQ(int listId, size_t numVecs)
{
    int newVecSize = static_cast<int>(utils::divUp(listVecNum[listId] + numVecs, this->blockSize));
    int vecSize = static_cast<int>(utils::divUp(listVecNum[listId], this->blockSize));
    int addVecNum = newVecSize - vecSize;

    if (vecSize > 0) {
        int vecId = vecSize - 1;
        if (addVecNum > 0) {
            basePQCoder[listId].at(vecId)->resize(this->blockSize * M, true);
        } else {
            size_t vecNumInLastBlock = listVecNum[listId] -
                                     static_cast<size_t>(vecId * blockSize) + static_cast<size_t>(numVecs);
            auto oldCapacity = basePQCoder[listId].at(vecId)->size();
            auto capacity = getPQVecCapacity(vecNumInLastBlock, oldCapacity, M);
            basePQCoder[listId].at(vecId)->resize(capacity, true);
        }
    }

    for (int i = 0; i < addVecNum; ++i) {
        basePQCoder.at(listId).emplace_back(CREATE_UNIQUE_PTR(DeviceVector<uint8_t>, MemorySpace::DEVICE_HUGEPAGE));
        size_t newVecId = static_cast<size_t>(vecSize + i);
        size_t tailVecId = newVecSize <= 1 ? 0 : static_cast<size_t>(newVecSize - 1);

        if (newVecId == tailVecId) {
            size_t vecNumInLastBlock = listVecNum[listId] - newVecId * static_cast<size_t>(blockSize) +
                                     static_cast<size_t>(numVecs);
            auto capacity = getPQVecCapacity(vecNumInLastBlock, 0, M);
            basePQCoder[listId].at(newVecId)->resize(capacity, true);
        } else {
            basePQCoder[listId].at(newVecId)->resize(this->blockSize * M, true);
        }

        pBasePQCoder = basePQCoder[listId].at(newVecId)->data() < pBasePQCoder ?
                       basePQCoder[listId].at(newVecId)->data() : pBasePQCoder;
    }
    return APP_ERR_OK;
}

APP_ERROR IndexIVFPQ::addPQCodes(int listId, size_t numVecs, const uint8_t *pqCodes, const idx_t *indices)
{
    APPERR_RETURN_IF_NOT_FMT(listId >= 0 && listId < numLists, APP_ERR_INVALID_PARAM,
                             "the listId is %d, out of numLists(%d)", listId, numLists);
    APPERR_RETURN_IF(numVecs == 0, APP_ERR_OK);

    int bytes_per_vector = M;

    AscendTensor<uint8_t, DIMS_2> pqCodesData(const_cast<uint8_t *>(pqCodes),
                                              { static_cast<int32_t>(numVecs), bytes_per_vector });

    APPERR_RETURN_IF_NOT_LOG(APP_ERR_OK == resizeBasePQ(listId, numVecs),
                             APP_ERR_INNER_ERROR, "resize PQ base failed!");

    auto ret = AddCodeNDFormat<uint8_t>(pqCodesData, listVecNum[listId], blockSize, basePQCoder[listId]);
    APPERR_RETURN_IF_NOT_FMT(ret == APP_ERR_OK, APP_ERR_INNER_ERROR,
                             "add PQ codes to device failed %d", ret);

    maxListLength = std::max(maxListLength, static_cast<int>(getListLength(listId)));
    maxListLength = utils::roundUp(maxListLength, CUBE_ALIGN);
    listVecNum[listId] += numVecs;
    IndexIVF::ntotal += numVecs;

    deviceListIndices[listId]->append(indices, numVecs, true);

    return APP_ERR_OK;
}

size_t IndexIVFPQ::getListLength(int listId) const
{
    ASCEND_THROW_IF_NOT((listId < numLists) && (listId >= 0));

    return listVecNum[listId];
}

APP_ERROR IndexIVFPQ::getListInitialize(int listId, std::vector<unsigned char>& reshaped,
                                        size_t& totalVecsInList, size_t& codeSizePerVector,
                                        size_t& totalBytes) const
{
    APPERR_RETURN_IF_NOT_FMT(listId >= 0 && listId < numLists, APP_ERR_INVALID_PARAM,
                             "Invalid listId %d, numLists=%d", listId, numLists);

    totalVecsInList = listVecNum[listId];
    if (totalVecsInList == 0) {
        reshaped.clear();
        return APP_ERR_OK;
    }

    codeSizePerVector = static_cast<size_t>(M);
    totalBytes = totalVecsInList * codeSizePerVector;
    reshaped.resize(totalBytes);

    return APP_ERR_OK;
}

APP_ERROR IndexIVFPQ::copyPQBlocks(int listId, size_t totalVecsInList, size_t codeSizePerVector,
                                   std::vector<unsigned char>& reshaped, size_t& destOffset,
                                   size_t& processedVecs) const
{
    const auto& blocks = basePQCoder[listId];
    size_t numBlocks = blocks.size();
    if (numBlocks == 0) {
        APPERR_RETURN_IF_NOT_FMT(totalVecsInList == 0, APP_ERR_INNER_ERROR,
                                 "List %d has %zu vectors but no blocks",
                                 listId, totalVecsInList);
        return APP_ERR_OK;
    }

    destOffset = 0;
    processedVecs = 0;

    for (size_t blockIdx = 0; blockIdx < numBlocks; blockIdx++) {
        const auto& block = blocks[blockIdx];
        if (!block || block->size() == 0) {
            APP_LOG_WARNING("Block %zu in list %d is null or empty", blockIdx, listId);
            continue;
        }

        size_t vecsInBlock = (blockIdx == numBlocks - 1) ? totalVecsInList - processedVecs
                                                         : std::min(static_cast<size_t>(blockSize),
                                                                    totalVecsInList - processedVecs);

        size_t bytesToCopy = vecsInBlock * codeSizePerVector;

        if (block->size() < bytesToCopy) {
            size_t actualVecsInBlock = block->size() / codeSizePerVector;
            bytesToCopy = actualVecsInBlock * codeSizePerVector;
            vecsInBlock = actualVecsInBlock;
        }

        if (bytesToCopy == 0) {
            continue;
        }

        auto ret = aclrtMemcpy(reshaped.data() + destOffset, bytesToCopy,
                               block->data(), bytesToCopy,
                               ACL_MEMCPY_DEVICE_TO_HOST);
        APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, ACL_ERROR_FAILURE,
                                 "Failed to copy block %zu: %d", blockIdx, ret);

        destOffset += bytesToCopy;
        processedVecs += vecsInBlock;

        APP_LOG_DEBUG("Block %zu: copied %zu vectors (%zu bytes), total processed %zu/%zu\n",
                      blockIdx, vecsInBlock, bytesToCopy, processedVecs, totalVecsInList);
    }

    return APP_ERR_OK;
}

APP_ERROR IndexIVFPQ::getListVectorsReshaped(int listId, std::vector<unsigned char>& reshaped) const
{
    size_t totalVecsInList;
    size_t codeSizePerVector;
    size_t totalBytes;
    size_t destOffset;
    size_t processedVecs;
    auto ret = getListInitialize(listId, reshaped, totalVecsInList, codeSizePerVector, totalBytes);
    APPERR_RETURN_IF_NOT_FMT(ret == APP_ERR_OK, ret,
                             "Failed to get list initialize for list %d: %d", listId, ret);

    ret = copyPQBlocks(listId, totalVecsInList, codeSizePerVector, reshaped, destOffset, processedVecs);
    APPERR_RETURN_IF_NOT_FMT(ret == APP_ERR_OK, ret,
                             "Failed to process Blocks for list %d: %d", listId, ret);

    APPERR_RETURN_IF_NOT_FMT(processedVecs == totalVecsInList, APP_ERR_INNER_ERROR,
                             "Vector count mismatch: expected %zu, processed %zu",
                             totalVecsInList, processedVecs);

    APPERR_RETURN_IF_NOT_FMT(destOffset == totalBytes, APP_ERR_INNER_ERROR,
                             "Byte count mismatch: expected %zu, copied %zu",
                             totalBytes, destOffset);

    APP_LOG_INFO("Successfully reshaped list %d: %zu vectors, %zu bytes, %zu blocks\n",
                 listId, totalVecsInList, totalBytes, basePQCoder[listId].size());

    return APP_ERR_OK;
}

APP_ERROR IndexIVFPQ::prepareDelete(int listId, size_t numVecs, const idx_t *indices,
                                    size_t& currentVecNum, std::vector<idx_t>& hostIds,
                                    std::vector<bool>& toDelete, std::vector<idx_t>& deletedIds,
                                    size_t& deleteCount)
{
    APPERR_RETURN_IF_NOT_FMT(listId >= 0 && listId < numLists, APP_ERR_INVALID_PARAM,
                             "the listId is %d, out of numLists(%d)", listId, numLists);
    APPERR_RETURN_IF(numVecs == 0, APP_ERR_OK);

    currentVecNum = listVecNum[listId];
    if (currentVecNum == 0) {
        APP_LOG_WARNING("List %d is empty, nothing to delete\n", listId);
        return APP_ERR_OK;
    }

    hostIds.resize(currentVecNum);
    if (currentVecNum > 0) {
        auto ret = aclrtMemcpy(hostIds.data(),
                               currentVecNum * sizeof(idx_t),
                               deviceListIndices[listId]->data(),
                               currentVecNum * sizeof(idx_t),
                               ACL_MEMCPY_DEVICE_TO_HOST);

        APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, ACL_ERROR_FAILURE,
                                 "Failed to copy IDs from device: %d", ret);
    }

    std::unordered_set<idx_t> deleteSet;
    for (size_t i = 0; i < numVecs; i++) {
        deleteSet.insert(indices[i]);
    }

    toDelete.resize(currentVecNum, false);
    deletedIds.clear();
    deleteCount = 0;

    for (size_t i = 0; i < currentVecNum; i++) {
        if (deleteSet.find(hostIds[i]) != deleteSet.end()) {
            toDelete[i] = true;
            deletedIds.push_back(hostIds[i]);
            deleteCount++;
        }
    }

    if (deleteCount == 0) {
        APP_LOG_WARNING("No matching IDs found to delete in list %d\n", listId);
        return APP_ERR_OK;
    }

    APP_LOG_DEBUG("Found %zu IDs to delete from list %d\n", deleteCount, listId);

    return APP_ERR_OK;
}

APP_ERROR IndexIVFPQ::updateFilterPQCodes(int listId, size_t currentVecNum, std::vector<idx_t>& hostIds,
                                          std::vector<bool>& toDelete, size_t deleteCount,
                                          size_t bytesPerVector, std::vector<uint8_t>& newCodes,
                                          std::vector<idx_t>& newIds)
{
    std::vector<uint8_t> reshapedCodes;
    APP_ERROR ret = getListVectorsReshaped(listId, reshapedCodes);
    APPERR_RETURN_IF_NOT_FMT(ret == APP_ERR_OK, ret,
                             "Failed to get PQ codes for list %d: %d", listId, ret);

    size_t newVecNum = currentVecNum - deleteCount;
    newCodes.resize(newVecNum * bytesPerVector);
    newIds.resize(newVecNum);

    size_t srcOffset = 0;
    size_t dstOffset = 0;
    size_t idIndex = 0;

    for (size_t i = 0; i < currentVecNum; i++) {
        if (!toDelete[i]) {
            std::copy(reshapedCodes.begin() + srcOffset,
                      reshapedCodes.begin() + srcOffset + bytesPerVector,
                      newCodes.begin() + dstOffset);

            newIds[idIndex++] = hostIds[i];

            dstOffset += bytesPerVector;
        }
        srcOffset += bytesPerVector;
    }

    return APP_ERR_OK;
}

APP_ERROR IndexIVFPQ::updateDeviceData(int listId, size_t newVecNum, std::vector<uint8_t>& newCodes,
                                       const std::vector<idx_t>& newIds)
{
    basePQCoder[listId].clear();
    deviceListIndices[listId]->resize(0, true);

    if (newVecNum > 0) {
        listVecNum[listId] = 0;

        AscendTensor<uint8_t, DIMS_2> newPQCodesTensor(newCodes.data(),
                                                       { static_cast<int32_t>(newVecNum),
                                                        static_cast<int32_t>(bytesPerVector) });

        APP_ERROR ret = resizeBasePQ(listId, newVecNum);
        APPERR_RETURN_IF_NOT_FMT(ret == APP_ERR_OK, ret,
                                 "Failed to resize base PQ for list %d: %d", listId, ret);

        ret = AddCodeNDFormat<uint8_t>(newPQCodesTensor, 0, blockSize, basePQCoder[listId]);
        APPERR_RETURN_IF_NOT_FMT(ret == APP_ERR_OK, ret,
                                 "Failed to add PQ codes to device: %d", ret);

        deviceListIndices[listId]->append(newIds.data(), newVecNum, true);

        listVecNum[listId] = newVecNum;
    } else {
        listVecNum[listId] = 0;
    }

    return APP_ERR_OK;
}

APP_ERROR IndexIVFPQ::deletePQCodes(int listId, size_t numVecs, const idx_t *indices)
{
    size_t currentVecNum;
    std::vector<idx_t> hostIds;
    std::vector<bool> toDelete;
    std::vector<idx_t> deletedIds;
    size_t deleteCount;

    APP_ERROR ret = prepareDelete(listId, numVecs, indices, currentVecNum, hostIds, toDelete, deletedIds, deleteCount);
    APPERR_RETURN_IF_NOT_FMT(ret == APP_ERR_OK, ret,
                             "Failed to prepare delete for list %d: %d", listId, ret);

    size_t bytesPerVector = static_cast<size_t>(M);
    std::vector<uint8_t> newCodes;
    std::vector<idx_t> newIds;
    ret = updateFilterPQCodes(listId, currentVecNum, hostIds, toDelete, deleteCount, bytesPerVector, newCodes, newIds);
    APPERR_RETURN_IF_NOT_FMT(ret == APP_ERR_OK, ret,
                             "Failed to process PQ Codes for list %d: %d", listId, ret);

    size_t newVecNum = currentVecNum - deleteCount;

    ret = updateDeviceData(listId, newVecNum, newCodes, newIds);
    APPERR_RETURN_IF_NOT_FMT(ret == APP_ERR_OK, ret,
                             "Failed to update to device for list %d: %d", listId, ret);

    IndexIVF::ntotal -= deleteCount;

    maxListLength = 0;
    for (int i = 0; i < numLists; i++) {
        if (listVecNum[i] > static_cast<size_t>(maxListLength)) {
            maxListLength = static_cast<int>(listVecNum[i]);
        }
    }

    if (maxListLength > 0) {
        maxListLength = utils::roundUp(maxListLength, CUBE_ALIGN);
    }

    APP_LOG_INFO("Successfully deleted %zu vectors from list %d, remaining: %zu\n",
                 deleteCount, listId, listVecNum[listId]);

    return APP_ERR_OK;
}

void IndexIVFPQ::normL2(int dim, int nlist, float* data)
{
#pragma omp parallel for
    for (int i = 0; i < nlist; ++i) {
        float* vec = data + i * dim;

        float norm2 = 0.0f;
        for (int d = 0; d < dim; ++d) {
            norm2 += vec[d] * vec[d];
        }

        if (norm2 > 1e-30f) {
            float inv_norm = 1.0f / sqrtf(norm2);
            for (int d = 0; d < dim; ++d) {
                vec[d] *= inv_norm;
            }
        }
    }
}

APP_ERROR IndexIVFPQ::initTraining(int totalSize, int dim, int nlist, const float* x,
                                   std::unique_ptr<float[], aclFloatDeleter>& data_dev,
                                   std::vector<float>& trainData, std::vector<float>& centroids)
{
    trainData.resize(totalSize * dim);
    auto ret = memcpy_s(trainData.data(), totalSize * dim * sizeof(float), x, totalSize * dim * sizeof(float));
    ASCEND_THROW_IF_NOT_FMT(ret == EOK, "trainData memcpy_s failed %d", ret);

    size_t data_bytes = totalSize * dim * sizeof(float);
    ret = aclrtMalloc((void**)&data_dev, data_bytes, ACL_MEM_MALLOC_HUGE_FIRST);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR,
                             "allocate data mem failed %d", ret);
    ret = aclrtMemcpy(data_dev.get(), data_bytes, trainData.data(),
                      data_bytes, ACL_MEMCPY_HOST_TO_DEVICE);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR,
                             "update data failed %d", ret);
    std::vector<int> perm(totalSize, 0);
    const uint64_t actSeed = getActualRngSeed(1234);
    faiss::rand_perm(perm.data(), totalSize, actSeed + 1);
    for (int i = 0; i < nlist; i++) {
        int offset = i * dim;
        auto ret = memcpy_s(centroids.data() + offset, (centroids.size() - offset) * sizeof(float),
                            trainData.data() + perm[i] * dim, dim * sizeof(float));
        ASCEND_THROW_IF_NOT_FMT(ret == EOK, "Failed to copy to centroids (error %d)", ret);
    }
    normL2(dim, nlist, centroids.data());
    return APP_ERR_OK;
}

APP_ERROR IndexIVFPQ::resetTrainOp(int nlist, int dim)
{
    APP_ERROR ret = resetTrainDistOp(nlist, dim);
    APPERR_RETURN_IF_NOT_FMT(ret == APP_ERR_OK, APP_ERR_INNER_ERROR,
                             "reset dist op failed %d", ret);

    ret = resetTrainTopkOp(nlist);
    APPERR_RETURN_IF_NOT_FMT(ret == APP_ERR_OK, APP_ERR_INNER_ERROR,
                             "reset topk op failed %d", ret);

    return APP_ERR_OK;
}

APP_ERROR IndexIVFPQ::runKMeans(int nlist, int dim, int totalSize, int iter,
                                std::vector<float>& centroids, std::unique_ptr<float[], aclFloatDeleter>& data_dev,
                                std::vector<int64_t>& totalAssigns)
{
    std::vector<float> centroid_double(nlist);
    for (int i = 0; i < nlist; i++) {
        float normSQ = 0.0f;
        for (int j = 0; j < dim; j++) {
            float val = centroids[i * dim + j];
            normSQ += val * val;
        }
        centroid_double[i] = normSQ;
    }

    std::vector<int64_t> batchAssigns;

    int processed = 0;
    while (processed < totalSize) {
        int remaining = totalSize - processed;
        int batchSize = 64;
        for (int bs : searchBatchSizes) {
            if (bs <= remaining) {
                batchSize = bs;
                break;
            }
        }
        batchAssigns.resize(batchSize);

        auto ret = trainBatchImpl(batchSize, nlist, dim, processed, data_dev, centroids, centroid_double,
                                  batchAssigns);
        APPERR_RETURN_IF_NOT_FMT(ret == APP_ERR_OK, ret,
                                 "Failed to initialize and execute train batch in %d %d %d", processed, iter, ret);

        ret = aclrtMemcpy(totalAssigns.data() + processed, batchSize * sizeof(int64_t),
                          batchAssigns.data(), batchSize * sizeof(int64_t), ACL_MEMCPY_HOST_TO_HOST);
        APPERR_RETURN_IF_NOT_FMT(ret == APP_ERR_OK, ret, "Failed update assign in %d %d %d", processed, iter, ret);
        processed += batchSize;
    }

    return APP_ERR_OK;
}

APP_ERROR IndexIVFPQ::trainBatchImpl(int batchSize, int nlist, int dim, int processed,
                                     std::unique_ptr<float[], aclFloatDeleter>& data_dev,
                                     std::vector<float>& centroids, std::vector<float>& centroid_double,
                                     std::vector<int64_t>& batchAssigns)
{
    auto &mem = resources.getMemoryManager();
    auto streamPtr = resources.getDefaultStream();
    auto stream = streamPtr->GetStream();
    AscendTensor<float, DIMS_2> queries_tensor(data_dev.get() + processed * dim, {batchSize, dim});
    AscendTensor<float, DIMS_2> centroids_tensor(mem, {nlist, dim}, stream);
    AscendTensor<float, DIMS_1> codes_double_tensor(mem, {nlist}, stream);
    AscendTensor<uint32_t, DIMS_2> sizes_tensor(mem, {CORE_NUM, SIZE_ALIGN}, stream);
    AscendTensor<uint16_t, DIMS_2> flags_tensor(mem, {CORE_NUM, FLAG_SIZE}, stream);
    AscendTensor<int64_t, DIMS_1> attrs_tensor(mem, { aicpu::TOPK_FLAT_ATTR_IDX_COUNT }, stream);

    auto ret = aclrtMemcpy(centroids_tensor.data(), centroids_tensor.getSizeInBytes(),
                           centroids.data(), centroids_tensor.getSizeInBytes(), ACL_MEMCPY_HOST_TO_DEVICE);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR,
                             "update centroids failed in %d : %d", processed, ret);
    ret = aclrtMemcpy(codes_double_tensor.data(), codes_double_tensor.getSizeInBytes(),
                      centroid_double.data(), codes_double_tensor.getSizeInBytes(), ACL_MEMCPY_HOST_TO_DEVICE);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR,
                             "update centroids L2 failed in %d : %d", processed, ret);
    sizes_tensor[0][0] = nlist;
    flags_tensor.zero();
    std::vector<int64_t> attrs_host(aicpu::TOPK_FLAT_ATTR_IDX_COUNT);
    attrs_host[aicpu::TOPK_FLAT_ATTR_ASC_IDX] = 1;
    attrs_host[aicpu::TOPK_FLAT_ATTR_K_IDX] = nprobe;
    attrs_host[aicpu::TOPK_FLAT_ATTR_BURST_LEN_IDX] = IVF_PQ_BURST_LEN;
    attrs_host[aicpu::TOPK_FLAT_ATTR_BLOCK_NUM_IDX] = 1;
    attrs_host[aicpu::TOPK_FLAT_ATTR_PAGE_IDX] = 0;
    attrs_host[aicpu::TOPK_FLAT_ATTR_PAGE_NUM_IDX] = 1;
    attrs_host[aicpu::TOPK_FLAT_ATTR_PAGE_SIZE_IDX] = 0;
    attrs_host[aicpu::TOPK_FLAT_ATTR_QUICK_HEAP] = 1;
    attrs_host[aicpu::TOPK_FLAT_ATTR_BLOCK_SIZE] = nlist;
    ret = aclrtMemcpy(attrs_tensor.data(), attrs_tensor.getSizeInBytes(),
                      attrs_host.data(), attrs_tensor.getSizeInBytes(), ACL_MEMCPY_HOST_TO_DEVICE);
    APPERR_RETURN_IF_NOT_LOG(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "Failed to copy attr to device");

    ret = execTrainBatch(queries_tensor, centroids_tensor, codes_double_tensor,
                         sizes_tensor, flags_tensor, attrs_tensor, batchAssigns);
    APPERR_RETURN_IF_NOT_FMT(ret == APP_ERR_OK, ret,
                             "Failed to execute train batch, batchSize=%d, nlist=%d, dim=%d, ret=%d",
                             batchSize, nlist, dim, ret);
    return APP_ERR_OK;
}

APP_ERROR IndexIVFPQ::execTrainBatch(AscendTensor<float, DIMS_2> &queries_tensor,
                                     AscendTensor<float, DIMS_2> &centroids_tensor,
                                     AscendTensor<float, DIMS_1> &codes_double_tensor,
                                     AscendTensor<uint32_t, DIMS_2> &sizes_tensor,
                                     AscendTensor<uint16_t, DIMS_2> &flags_tensor,
                                     AscendTensor<int64_t, DIMS_1> &attrs_tensor,
                                     std::vector<int64_t>& batchAssigns)
{
    auto &mem = resources.getMemoryManager();
    auto streamPtr = resources.getDefaultStream();
    auto stream = streamPtr->GetStream();
    auto streamAicpuPtr = resources.getAlternateStreams()[0];
    auto streamAicpu = streamAicpuPtr->GetStream();
    int batchSize = queries_tensor.getSize(0);
    int nlist = centroids_tensor.getSize(0);
    AscendTensor<float, DIMS_2> dists_tensor(mem, {batchSize, nlist}, stream);
    AscendTensor<float, DIMS_2> vmdists_tensor(mem, {batchSize, std::min(nlist / IVF_PQ_BURST_LEN * 2,
                                                                         MIN_EXTREME_SIZE)}, stream);
    AscendTensor<float, DIMS_2> outdists_tensor(mem, {batchSize, nprobe}, stream);
    AscendTensor<int64_t, DIMS_2> outlabel_tensor(mem, {batchSize, nprobe}, stream);

    runTrainDistOp(batchSize, queries_tensor, centroids_tensor, codes_double_tensor,
                   dists_tensor, vmdists_tensor, flags_tensor, stream);
    auto ret = synchronizeStream(stream);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR,
                             "synchronizeStream default stream: %i\n", ret);
    runTrainTopkOp(dists_tensor, vmdists_tensor, sizes_tensor, flags_tensor,
                   attrs_tensor, outdists_tensor, outlabel_tensor, streamAicpu);
    ret = synchronizeStream(streamAicpu);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR,
                             "synchronizeStream aicpu stream failed: %i\n", ret);

    for (int i = 0; i < batchSize; ++i) {
        ret = aclrtMemcpy(batchAssigns.data() + i, sizeof(int64_t),
                          outlabel_tensor.data() + i * nprobe, sizeof(int64_t),
                          ACL_MEMCPY_DEVICE_TO_HOST);
        APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR,
                                 "copy topk result to host failed: %d", ret);
    }
    return APP_ERR_OK;
}

APP_ERROR IndexIVFPQ::updateCentroids(int nlist, int dim, int totalSize, const std::vector<int64_t>& totalAssigns,
                                      const std::vector<float>& trainData, std::vector<float>& centroids)
{
    auto ret = memset_s(centroids.data(), dim * nlist * sizeof(float), 0, dim * nlist * sizeof(float));
    ASCEND_THROW_IF_NOT_FMT(ret == EOK, "set centroids to 0 failed %d", ret);
    std::vector<int32_t> hassign(nlist);
#pragma omp parallel
    {
        int nt = omp_get_num_threads(); // total num threads
        int rank = omp_get_thread_num(); // get thread ID
        int64_t c0 = (nlist * rank) / nt;
        int64_t c1 = (nlist * (rank + 1)) / nt;
        for (int i = 0; i < totalSize; i++) {
            ASCEND_THROW_IF_NOT_FMT(totalAssigns[i] >= 0, "assign[%d] %jd < 0 is invalid", i, totalAssigns[i]);
            int ci = totalAssigns[i];
            ASCEND_THROW_IF_NOT_FMT(ci < nlist, "assign %d >= nlist is invalid.", ci);
            if (ci >= c0 && ci < c1) {
                float* c = centroids.data() + ci * dim;
                const float* xi = trainData.data() + i * dim;
                hassign[ci] += 1;
                for (int j = 0; j < dim; j++) {
                    c[j] += xi[j];
                }
            }
        }
    }
#pragma omp parallel for
    for (int ci = 0; ci < nlist; ci++) {
        if (hassign[ci] == 0) {
            continue;
        }
        float norm = 1 / static_cast<float>(hassign[ci]);
        float* c = centroids.data() + ci * dim;
        for (int j = 0; j < dim; j++) {
            c[j] *= norm;
        }
    }

    return APP_ERR_OK;
}

APP_ERROR IndexIVFPQ::updateCentroidsToDevice(int nlist, int dim, const std::vector<float>& centroids)
{
    if (!clusteringOnDevice) {
        clusteringOnDevice = std::make_unique<DeviceVector<float>>();
    }

    if (clusteringOnDevice->size() != static_cast<size_t>(nlist * dim)) {
        clusteringOnDevice->resize(nlist * dim);
    }

    size_t centroids_bytes = nlist * dim * sizeof(float);
    auto ret = aclrtMemcpy(clusteringOnDevice->data(), centroids_bytes, centroids.data(),
                           centroids_bytes, ACL_MEMCPY_HOST_TO_DEVICE);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR,
                             "update centroids result to Device failed, size=%zu, ret=%d",
                             centroids_bytes, ret);

    return APP_ERR_OK;
}

APP_ERROR IndexIVFPQ::trainImpl(int n, const float* x, int dim, int nlist)
{
    int totalSize = std::min(nlist * 256, n);

    std::vector<float> trainData(totalSize * dim);
    std::unique_ptr<float[], aclFloatDeleter> data_dev = nullptr;
    std::vector<float> centroids(nlist * dim);
    auto ret = initTraining(totalSize, dim, nlist, x, data_dev, trainData, centroids);
    APPERR_RETURN_IF_NOT_FMT(ret == APP_ERR_OK, ret,
                             "Failed to initialize training, n=%d, dim=%d, nlist=%d, ret=%d",
                             n, dim, nlist, ret);

    ret = resetTrainOp(nlist, dim);
    APPERR_RETURN_IF_NOT_FMT(ret == APP_ERR_OK, ret,
                             "Failed to reset training operations, nlist=%d, dim=%d, ret=%d",
                             nlist, dim, ret);

    std::vector<int64_t> totalAssigns(totalSize);
    for (int iter = 0; iter < MAX_ITER_NUM; iter++) {
        ret = runKMeans(nlist, dim, totalSize, iter, centroids, data_dev, totalAssigns);
        APPERR_RETURN_IF_NOT_FMT(ret == APP_ERR_OK, ret,
                                 "Failed to run KMeans, nlist=%d, dim=%d, totalSize=%d, iter=%d, ret=%d",
                                 nlist, dim, totalSize, iter, ret);

        ret = updateCentroids(nlist, dim, totalSize, totalAssigns, trainData, centroids);
        APPERR_RETURN_IF_NOT_FMT(ret == APP_ERR_OK, ret,
                                 "Failed to update centroids, nlist=%d, dim=%d, totalSize=%d, ret=%d",
                                 nlist, dim, totalSize, ret);
        normL2(dim, nlist, centroids.data());
    }

    ret = updateCentroidsToDevice(nlist, dim, centroids);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR,
                             "update centroids result to Device failed, ret=%d", ret);

    return APP_ERR_OK;
}

APP_ERROR IndexIVFPQ::resetTrainDistOp(int nlist, int dim)
{
    auto trainDisOpReset = [&](std::unique_ptr<AscendOperator> &op, int batch) {
        AscendOpDesc desc("DistanceFlatL2MinsAtFP32");

        std::vector<int64_t> queryShape({ batch, dim });
        std::vector<int64_t> codesShape({ nlist, dim });
        std::vector<int64_t> codesDoubleShape({ nlist });
        std::vector<int64_t> distResultShape({ batch, nlist });
        std::vector<int64_t> minResultShape({ batch, std::min(nlist / IVF_PQ_BURST_LEN * 2, MIN_EXTREME_SIZE) });
        std::vector<int64_t> flagShape({ CORE_NUM, 16 });

        desc.addInputTensorDesc(ACL_FLOAT, queryShape.size(), queryShape.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_FLOAT, codesShape.size(), codesShape.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_FLOAT, codesDoubleShape.size(), codesDoubleShape.data(), ACL_FORMAT_ND);
        desc.addOutputTensorDesc(ACL_FLOAT, distResultShape.size(), distResultShape.data(), ACL_FORMAT_ND);
        desc.addOutputTensorDesc(ACL_FLOAT, minResultShape.size(), minResultShape.data(), ACL_FORMAT_ND);
        desc.addOutputTensorDesc(ACL_UINT16, flagShape.size(), flagShape.data(), ACL_FORMAT_ND);

        op.reset();
        op = CREATE_UNIQUE_PTR(AscendOperator, desc);
        return op->init();
    };

    for (auto batch: searchBatchSizes) {
        trainDistOps[batch] = std::unique_ptr<AscendOperator>(nullptr);
        APPERR_RETURN_IF_NOT_LOG(trainDisOpReset(trainDistOps[batch], batch),
                                 APP_ERR_ACL_OP_LOAD_MODEL_FAILED, "train dist op init failed");
    }
    return APP_ERR_OK;
}

void IndexIVFPQ::runTrainDistOp(int batch,
                                AscendTensor<float, DIMS_2> &queries,
                                AscendTensor<float, DIMS_2> &centroids,
                                AscendTensor<float, DIMS_1> &codesDouble,
                                AscendTensor<float, DIMS_2> &distances,
                                AscendTensor<float, DIMS_2> &vmdists,
                                AscendTensor<uint16_t, DIMS_2> &opFlag,
                                aclrtStream stream)
{
    AscendOperator *op = nullptr;
    if (trainDistOps.find(batch) != trainDistOps.end()) {
        op = trainDistOps[batch].get();
    }
    ASCEND_THROW_IF_NOT(op);

    std::shared_ptr<std::vector<const aclDataBuffer *>> inputs(
            new std::vector<const aclDataBuffer *>(),
            CommonUtils::AclInputBufferDelete);

    inputs->emplace_back(aclCreateDataBuffer(queries.data(), queries.getSizeInBytes()));
    inputs->emplace_back(aclCreateDataBuffer(centroids.data(), centroids.getSizeInBytes()));
    inputs->emplace_back(aclCreateDataBuffer(codesDouble.data(), codesDouble.getSizeInBytes()));

    std::shared_ptr<std::vector<aclDataBuffer *>> outputs(
            new std::vector<aclDataBuffer *>(),
            CommonUtils::AclOutputBufferDelete);

    outputs->emplace_back(aclCreateDataBuffer(distances.data(), distances.getSizeInBytes()));
    outputs->emplace_back(aclCreateDataBuffer(vmdists.data(), vmdists.getSizeInBytes()));
    outputs->emplace_back(aclCreateDataBuffer(opFlag.data(), opFlag.getSizeInBytes()));

    op->exec(*inputs, *outputs, stream);
}

APP_ERROR IndexIVFPQ::resetTrainTopkOp(int nlist)
{
    auto topkTrainOpReset = [&](std::unique_ptr<AscendOperator> &op, int64_t batch) {
        AscendOpDesc desc("TopkFlatFp32");
        std::vector<int64_t> shape0 { 1, batch, nlist };
        std::vector<int64_t> shape1 { 1, batch, std::min(nlist / IVF_PQ_BURST_LEN * 2, MIN_EXTREME_SIZE) };
        std::vector<int64_t> shape2 { 1, CORE_NUM, SIZE_ALIGN };
        std::vector<int64_t> shape3 { 1, CORE_NUM, FLAG_SIZE };
        std::vector<int64_t> shape4 { aicpu::TOPK_FLAT_ATTR_IDX_COUNT };
        std::vector<int64_t> shape5 { batch, 0 };

        desc.addInputTensorDesc(ACL_FLOAT, shape0.size(), shape0.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_FLOAT, shape1.size(), shape1.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_UINT32, shape2.size(), shape2.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_UINT16, shape3.size(), shape3.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_INT64, shape4.size(), shape4.data(), ACL_FORMAT_ND);

        desc.addOutputTensorDesc(ACL_FLOAT, shape5.size(), shape5.data(), ACL_FORMAT_ND);
        desc.addOutputTensorDesc(ACL_INT64, shape5.size(), shape5.data(), ACL_FORMAT_ND);
        op.reset();
        op = CREATE_UNIQUE_PTR(AscendOperator, desc);
        return op->init();
    };

    for (auto batch : searchBatchSizes) {
        trainTopkOps[batch] = std::unique_ptr<AscendOperator>(nullptr);
        APPERR_RETURN_IF_NOT_LOG(topkTrainOpReset(trainTopkOps[batch], batch),
                                 APP_ERR_ACL_OP_LOAD_MODEL_FAILED, "train topk op init failed");
    }
    return APP_ERR_OK;
}

void IndexIVFPQ::runTrainTopkOp(AscendTensor<float, DIMS_2> &dists,
                                AscendTensor<float, DIMS_2> &vmdists,
                                AscendTensor<uint32_t, DIMS_2> &sizes,
                                AscendTensor<uint16_t, DIMS_2> &flags,
                                AscendTensor<int64_t, DIMS_1> &attrs,
                                AscendTensor<float, DIMS_2> &outdists,
                                AscendTensor<int64_t, DIMS_2> &outlabel,
                                aclrtStream stream)
{
    AscendOperator *op = nullptr;
    int batch = dists.getSize(0);
    int nlist = dists.getSize(1);
    int vmdists_cols = vmdists.getSize(1);
    if (trainTopkOps.find(batch) != trainTopkOps.end()) {
        op = trainTopkOps[batch].get();
    }
    ASCEND_THROW_IF_NOT(op);

    AscendTensor<float, DIMS_3> distsTopk(dists.data(), {1, batch, nlist});
    AscendTensor<float, DIMS_3> vmdistsTopk(vmdists.data(), {1, batch, vmdists_cols});
    AscendTensor<uint32_t, DIMS_3> sizesTopk(sizes.data(), {1, CORE_NUM, SIZE_ALIGN});
    AscendTensor<uint16_t, DIMS_3> flagsTopk(flags.data(), {1, CORE_NUM, FLAG_SIZE});

    std::shared_ptr<std::vector<const aclDataBuffer *>> inputs(
            new std::vector<const aclDataBuffer *>(),
            CommonUtils::AclInputBufferDelete);

    inputs->emplace_back(aclCreateDataBuffer(distsTopk.data(), distsTopk.getSizeInBytes()));
    inputs->emplace_back(aclCreateDataBuffer(vmdistsTopk.data(), vmdistsTopk.getSizeInBytes()));
    inputs->emplace_back(aclCreateDataBuffer(sizesTopk.data(), sizesTopk.getSizeInBytes()));
    inputs->emplace_back(aclCreateDataBuffer(flagsTopk.data(), flagsTopk.getSizeInBytes()));
    inputs->emplace_back(aclCreateDataBuffer(attrs.data(), attrs.getSizeInBytes()));

    std::shared_ptr<std::vector<aclDataBuffer *>> outputs(
            new std::vector<aclDataBuffer *>(),
            CommonUtils::AclOutputBufferDelete);

    outputs->emplace_back(aclCreateDataBuffer(outdists.data(), outdists.getSizeInBytes()));
    outputs->emplace_back(aclCreateDataBuffer(outlabel.data(), outlabel.getSizeInBytes()));

    op->exec(*inputs, *outputs, stream);
}

APP_ERROR IndexIVFPQ::resetL1TopkOp()
{
    auto topkCompOpReset = [&](std::unique_ptr<AscendOperator> &op, int64_t batch) {
        AscendOpDesc desc("TopkFlatFp32");
        std::vector<int64_t> shape0 { 1, batch, numLists };
        std::vector<int64_t> shape1 { 1, batch,  std::min(numLists / IVF_PQ_BURST_LEN * 2, MIN_EXTREME_SIZE) };
        std::vector<int64_t> shape2 { 1, CORE_NUM, SIZE_ALIGN };
        std::vector<int64_t> shape3 { 1, CORE_NUM, FLAG_SIZE };
        std::vector<int64_t> shape4 { aicpu::TOPK_FLAT_ATTR_IDX_COUNT };
        std::vector<int64_t> shape5 { batch, 0 };

        desc.addInputTensorDesc(ACL_FLOAT, shape0.size(), shape0.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_FLOAT, shape1.size(), shape1.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_UINT32, shape2.size(), shape2.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_UINT16, shape3.size(), shape3.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_INT64, shape4.size(), shape4.data(), ACL_FORMAT_ND);

        desc.addOutputTensorDesc(ACL_FLOAT, shape5.size(), shape5.data(), ACL_FORMAT_ND);
        desc.addOutputTensorDesc(ACL_INT64, shape5.size(), shape5.data(), ACL_FORMAT_ND);
        op.reset();
        op = CREATE_UNIQUE_PTR(AscendOperator, desc);
        return op->init();
    };

    for (auto batch : searchBatchSizes) {
        topkFp32[batch] = std::unique_ptr<AscendOperator>(nullptr);
        APPERR_RETURN_IF_NOT_LOG(topkCompOpReset(topkFp32[batch], batch),
                                 APP_ERR_ACL_OP_LOAD_MODEL_FAILED, "topk op init failed");
    }
    return APP_ERR_OK;
}

void IndexIVFPQ::runL1TopkOp(AscendTensor<float, DIMS_2> &dists,
                             AscendTensor<float, DIMS_2> &vmdists,
                             AscendTensor<uint32_t, DIMS_2> &sizes,
                             AscendTensor<uint16_t, DIMS_2> &flags,
                             AscendTensor<int64_t, DIMS_1> &attrs,
                             AscendTensor<float, DIMS_2> &outdists,
                             AscendTensor<int64_t, DIMS_2> &outlabel,
                             aclrtStream stream)
{
    AscendOperator *op = nullptr;
    int batch = dists.getSize(0);
    if (topkFp32.find(batch) != topkFp32.end()) {
        op = topkFp32[batch].get();
    }
    ASCEND_THROW_IF_NOT(op);
    AscendTensor<float, DIMS_3> distsTopk(dists.data(), {1, batch, numLists});
    AscendTensor<float, DIMS_3> vmdistsTopk(vmdists.data(),
                                            {1, batch, std::min(numLists / IVF_PQ_BURST_LEN * 2, MIN_EXTREME_SIZE)});
    AscendTensor<uint32_t, DIMS_3> sizesTopk(sizes.data(), {1, CORE_NUM, SIZE_ALIGN});
    AscendTensor<uint16_t, DIMS_3> flagsTopk(flags.data(), {1, CORE_NUM, FLAG_SIZE});
    std::shared_ptr<std::vector<const aclDataBuffer *>> topkOpInput(
            new std::vector<const aclDataBuffer *>(), CommonUtils::AclInputBufferDelete);
    topkOpInput->emplace_back(aclCreateDataBuffer(distsTopk.data(), distsTopk.getSizeInBytes()));
    topkOpInput->emplace_back(aclCreateDataBuffer(vmdistsTopk.data(), vmdistsTopk.getSizeInBytes()));
    topkOpInput->emplace_back(aclCreateDataBuffer(sizesTopk.data(), sizesTopk.getSizeInBytes()));
    topkOpInput->emplace_back(aclCreateDataBuffer(flagsTopk.data(), flagsTopk.getSizeInBytes()));
    topkOpInput->emplace_back(aclCreateDataBuffer(attrs.data(), attrs.getSizeInBytes()));

    std::shared_ptr<std::vector<aclDataBuffer *>> topkOpOutput(
            new std::vector<aclDataBuffer *>(), CommonUtils::AclOutputBufferDelete);
    topkOpOutput->emplace_back(aclCreateDataBuffer(outdists.data(), outdists.getSizeInBytes()));
    topkOpOutput->emplace_back(aclCreateDataBuffer(outlabel.data(), outlabel.getSizeInBytes()));

    op->exec(*topkOpInput, *topkOpOutput, stream);
}


APP_ERROR IndexIVFPQ::resetL1DistOp()
{
    auto l1DisOpReset = [&](std::unique_ptr<AscendOperator> &op, int batch) {
        AscendOpDesc desc("DistanceFlatL2MinsAtFP32");
        std::vector<int64_t> queryShape({ batch, dims });
        std::vector<int64_t> codesShape({ numLists, dims });
        std::vector<int64_t> codesSqrSum({ numLists });

        std::vector<int64_t> distResultShape({ batch, numLists });
        std::vector<int64_t> minResultShape({ batch, numLists / IVF_PQ_BURST_LEN * 2});
        std::vector<int64_t> flagShapeShape({ CORE_NUM, 16 });

        desc.addInputTensorDesc(ACL_FLOAT, queryShape.size(), queryShape.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_FLOAT, codesShape.size(), codesShape.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_FLOAT, codesSqrSum.size(), codesSqrSum.data(), ACL_FORMAT_ND);

        desc.addOutputTensorDesc(ACL_FLOAT, distResultShape.size(), distResultShape.data(), ACL_FORMAT_ND);
        desc.addOutputTensorDesc(ACL_FLOAT, minResultShape.size(), minResultShape.data(), ACL_FORMAT_ND);
        desc.addOutputTensorDesc(ACL_UINT16, flagShapeShape.size(), flagShapeShape.data(), ACL_FORMAT_ND);
        op.reset();
        op = CREATE_UNIQUE_PTR(AscendOperator, desc);
        return op->init();
    };
    for (auto batch: searchBatchSizes) {
        l1DistFp32Ops[batch] = std::unique_ptr<AscendOperator>(nullptr);
        APPERR_RETURN_IF_NOT_LOG(l1DisOpReset(l1DistFp32Ops[batch], batch),
                                 APP_ERR_ACL_OP_LOAD_MODEL_FAILED, "L1 distance op init failed");
    }
    return APP_ERR_OK;
}

APP_ERROR IndexIVFPQ::resetL2DistOp()
{
    auto l2DisOpReset = [&](std::unique_ptr<AscendOperator> &op, int batch) {
        AscendOpDesc desc("AscendcIvfpqSubspaceDistance");
        std::vector<int64_t> queryShape({batch, dims});
        std::vector<int64_t> codeBookShape({M, ksub, dsub});
        std::vector<int64_t> distResultShape({batch, M, ksub});
        std::vector<int64_t> minResultShape({batch, M, ksub / IVF_PQ_BURST_LEN * 2});
        std::vector<int64_t> flagShapeShape({batch, M, 16});

        desc.addInputTensorDesc(ACL_FLOAT, queryShape.size(), queryShape.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_FLOAT, codeBookShape.size(), codeBookShape.data(), ACL_FORMAT_ND);
        desc.addOutputTensorDesc(ACL_FLOAT, distResultShape.size(), distResultShape.data(), ACL_FORMAT_ND);
        desc.addOutputTensorDesc(ACL_FLOAT, minResultShape.size(), minResultShape.data(), ACL_FORMAT_ND);
        desc.addOutputTensorDesc(ACL_UINT16, flagShapeShape.size(), flagShapeShape.data(), ACL_FORMAT_ND);
        op.reset();
        op = CREATE_UNIQUE_PTR(AscendOperator, desc);
        return op->init();
    };
    for (auto batch: searchBatchSizes) {
        l2DistFp32Ops[batch] = std::unique_ptr<AscendOperator>(nullptr);
        APPERR_RETURN_IF_NOT_LOG(l2DisOpReset(l2DistFp32Ops[batch], batch),
                                 APP_ERR_ACL_OP_LOAD_MODEL_FAILED, "L2 distance op init failed");
    }
    return APP_ERR_OK;
}


APP_ERROR IndexIVFPQ::resetL3DistOp()
{
    auto l3DistOpReset = [&](std::unique_ptr<AscendOperator> &op) {
        AscendOpDesc desc("AscendcIvfpqSearchDistanceL2");

        std::vector<int64_t> queryPQShape({ M, ksub });
        std::vector<int64_t> codeBaseShape({ M });
        std::vector<int64_t> codeOffsetShape({ blockNum });
        std::vector<int64_t> codeSizeShape({ blockNum });
        std::vector<int64_t> topk({ MAX_TOPK, 16384 });
        std::vector<int64_t> distResultShape({ blockNum, blockSize });
        std::vector<int64_t> topkIndex({ blockNum, MAX_TOPK });
        std::vector<int64_t> topkValue({ blockNum, MAX_TOPK });
        std::vector<int64_t> flagShape({ blockNum, 16 });

        desc.addInputTensorDesc(ACL_FLOAT, queryPQShape.size(), queryPQShape.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_UINT8, codeBaseShape.size(), codeBaseShape.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_INT64, codeOffsetShape.size(), codeOffsetShape.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_INT64, codeSizeShape.size(), codeSizeShape.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_INT32, topk.size(), topk.data(), ACL_FORMAT_ND);

        desc.addOutputTensorDesc(ACL_FLOAT, distResultShape.size(), distResultShape.data(), ACL_FORMAT_ND);
        desc.addOutputTensorDesc(ACL_INT32, topkIndex.size(), topkIndex.data(), ACL_FORMAT_ND);
        desc.addOutputTensorDesc(ACL_FLOAT, topkValue.size(), topkValue.data(), ACL_FORMAT_ND);
        desc.addOutputTensorDesc(ACL_UINT16, flagShape.size(), flagShape.data(), ACL_FORMAT_ND);

        op.reset();
        op = CREATE_UNIQUE_PTR(AscendOperator, desc);
        return op->init();
    };

    APPERR_RETURN_IF_NOT_LOG(l3DistOpReset(ivfpqSearchDistOp),
                             APP_ERR_ACL_OP_LOAD_MODEL_FAILED,
                             "AscendcIvfpqSearchDistanceL2 op init failed");

    return APP_ERR_OK;
}

void IndexIVFPQ::runL1DistOp(int batch, AscendTensor<float, DIMS_2> &queries,
                             AscendTensor<float, DIMS_2> &centroidsDev, AscendTensor<float, DIMS_2> &dists,
                             AscendTensor<float, DIMS_2> &vmdists, AscendTensor<uint16_t, DIMS_2> &opFlag,
                             aclrtStream stream)
{
    AscendOperator *op = nullptr;
    if (l1DistFp32Ops.find(batch) != l1DistFp32Ops.end()) {
        op = l1DistFp32Ops[batch].get();
    }
    ASCEND_THROW_IF_NOT(op);

    AscendTensor<float, DIMS_1> centroidsSqrSum(centroidsSqrSumOnDevice->data(), {numLists});
    std::shared_ptr<std::vector<const aclDataBuffer *>> topkOpInput(
            new std::vector<const aclDataBuffer *>(), CommonUtils::AclInputBufferDelete);
    topkOpInput->emplace_back(aclCreateDataBuffer(queries.data(), queries.getSizeInBytes()));
    topkOpInput->emplace_back(aclCreateDataBuffer(centroidsDev.data(), centroidsDev.getSizeInBytes()));
    topkOpInput->emplace_back(aclCreateDataBuffer(centroidsSqrSum.data(), centroidsSqrSum.getSizeInBytes()));

    std::shared_ptr<std::vector<aclDataBuffer *>> topkOpOutput(
            new std::vector<aclDataBuffer *>(), CommonUtils::AclOutputBufferDelete);
    topkOpOutput->emplace_back(aclCreateDataBuffer(dists.data(), dists.getSizeInBytes()));
    topkOpOutput->emplace_back(aclCreateDataBuffer(vmdists.data(), vmdists.getSizeInBytes()));
    topkOpOutput->emplace_back(aclCreateDataBuffer(opFlag.data(), opFlag.getSizeInBytes()));
    op->exec(*topkOpInput, *topkOpOutput, stream);
    return;
}

void IndexIVFPQ::runL2DistOp(int batch, AscendTensor<float, DIMS_2> &queries,
                             AscendTensor<float, DIMS_3> &codeBook, AscendTensor<float, DIMS_3, size_t> &dists,
                             AscendTensor<float, DIMS_3> &vmdists, AscendTensor<uint16_t, DIMS_3> &opFlag,
                             aclrtStream stream)
{
    AscendOperator *op = nullptr;
    if (l2DistFp32Ops.find(batch) != l2DistFp32Ops.end()) {
        op = l2DistFp32Ops[batch].get();
    }
    ASCEND_THROW_IF_NOT(op);
    std::shared_ptr<std::vector<const aclDataBuffer *>> topkOpInput(
            new std::vector<const aclDataBuffer *>(), CommonUtils::AclInputBufferDelete);
    topkOpInput->emplace_back(aclCreateDataBuffer(queries.data(), queries.getSizeInBytes()));
    topkOpInput->emplace_back(aclCreateDataBuffer(codeBook.data(), codeBook.getSizeInBytes()));

    std::shared_ptr<std::vector<aclDataBuffer *>> topkOpOutput(
            new std::vector<aclDataBuffer *>(), CommonUtils::AclOutputBufferDelete);
    topkOpOutput->emplace_back(aclCreateDataBuffer(dists.data(), dists.getSizeInBytes()));
    topkOpOutput->emplace_back(aclCreateDataBuffer(vmdists.data(), vmdists.getSizeInBytes()));
    topkOpOutput->emplace_back(aclCreateDataBuffer(opFlag.data(), opFlag.getSizeInBytes()));

    op->exec(*topkOpInput, *topkOpOutput, stream);
    return;
}

void IndexIVFPQ::runL3DistOp(
    AscendTensor<float, DIMS_2, size_t> &queryPQ,
    AscendTensor<uint8_t, DIMS_1, size_t> &codeBase,
    AscendTensor<int64_t, DIMS_1, size_t> &offset,
    AscendTensor<int64_t, DIMS_1, size_t> &baseSize,
    AscendTensor<int32_t, DIMS_2, size_t> &topk,
    AscendTensor<float, DIMS_2, size_t> &dists,
    AscendTensor<int32_t, DIMS_2, size_t> &topkIndex,
    AscendTensor<float, DIMS_2, size_t> &topkValue,
    AscendTensor<uint16_t, DIMS_2, size_t> &opFlag,
    aclrtStream stream)
{
    AscendOperator *op = ivfpqSearchDistOp.get();
    ASCEND_THROW_IF_NOT(op);

    std::shared_ptr<std::vector<const aclDataBuffer *>> distOpInput(
            new std::vector<const aclDataBuffer *>(), CommonUtils::AclInputBufferDelete);
    distOpInput->emplace_back(aclCreateDataBuffer(queryPQ.data(), queryPQ.getSizeInBytes()));
    distOpInput->emplace_back(aclCreateDataBuffer(codeBase.data(), codeBase.getSizeInBytes()));
    distOpInput->emplace_back(aclCreateDataBuffer(offset.data(), offset.getSizeInBytes()));
    distOpInput->emplace_back(aclCreateDataBuffer(baseSize.data(), baseSize.getSizeInBytes()));
    distOpInput->emplace_back(aclCreateDataBuffer(topk.data(), topk.getSizeInBytes()));

    std::shared_ptr<std::vector<aclDataBuffer *>> distOpOutput(
            new std::vector<aclDataBuffer *>(), CommonUtils::AclOutputBufferDelete);
    distOpOutput->emplace_back(aclCreateDataBuffer(dists.data(), dists.getSizeInBytes()));
    distOpOutput->emplace_back(aclCreateDataBuffer(topkIndex.data(), topkIndex.getSizeInBytes()));
    distOpOutput->emplace_back(aclCreateDataBuffer(topkValue.data(), topkValue.getSizeInBytes()));
    distOpOutput->emplace_back(aclCreateDataBuffer(opFlag.data(), opFlag.getSizeInBytes()));
    op->exec(*distOpInput, *distOpOutput, stream);
    return;
}

APP_ERROR IndexIVFPQ::resetL3TopkOp()
{
    auto topkCompOpReset = [&](std::unique_ptr<AscendOperator> &op, int64_t batch) {
        AscendOpDesc desc("TopkIvfpqL3");
        std::vector<int64_t> shape0 { batch, 0, blockNum, MAX_TOPK };
        std::vector<int64_t> shape1 { batch, 0, blockNum, MAX_TOPK};
        std::vector<int64_t> shape2 { batch, 0, blockNum };
        std::vector<int64_t> shape3 { batch, 0, blockNum };
        std::vector<int64_t> shape4 { batch, 0, blockNum * 16 };
        std::vector<int64_t> shape5 { aicpu::TOPK_IVFPQ_L3_ATTR_IDX_COUNT };

        std::vector<int64_t> shape6 { batch, 0 };

        desc.addInputTensorDesc(ACL_INT32, shape0.size(), shape0.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_FLOAT, shape1.size(), shape1.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_INT64, shape2.size(), shape2.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_INT64, shape3.size(), shape3.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_UINT16, shape4.size(), shape4.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_INT64, shape5.size(), shape5.data(), ACL_FORMAT_ND);

        desc.addOutputTensorDesc(ACL_FLOAT, shape6.size(), shape6.data(), ACL_FORMAT_ND);
        desc.addOutputTensorDesc(ACL_UINT64, shape6.size(), shape6.data(), ACL_FORMAT_ND);

        op.reset();
        op = CREATE_UNIQUE_PTR(AscendOperator, desc);
        return op->init();
    };

    for (auto batch : searchBatchSizes) {
        topkL3Fp32[batch] = std::unique_ptr<AscendOperator>(nullptr);
        APPERR_RETURN_IF_NOT_LOG(topkCompOpReset(topkL3Fp32[batch], batch),
                                 APP_ERR_ACL_OP_LOAD_MODEL_FAILED, "l3 topk op init failed");
    }

    return APP_ERR_OK;
}

void IndexIVFPQ::runL3TopkOp(AscendTensor<int32_t, DIMS_3, size_t> &topkIndex,
                             AscendTensor<float, DIMS_3, size_t> &topkValue,
                             AscendTensor<int64_t, DIMS_3, size_t> &ids,
                             AscendTensor<int64_t, DIMS_3, size_t> &sizes,
                             AscendTensor<uint16_t, DIMS_3, size_t> &flags,
                             AscendTensor<int64_t, DIMS_1> &attrs,
                             AscendTensor<float, DIMS_2, size_t> &outdists,
                             AscendTensor<uint64_t, DIMS_2, size_t> &outlabel,
                             aclrtStream stream)
{
    int batch = static_cast<int>(topkIndex.getSize(0));
    std::vector<const AscendTensorBase *> input{&topkIndex, &topkValue, &ids, &sizes, &flags, &attrs};
    std::vector<const AscendTensorBase *> output{&outdists, &outlabel};
    AscendOperator *op = nullptr;
    if (topkL3Fp32.find(batch) != topkL3Fp32.end()) {
        op = topkL3Fp32[batch].get();
    }
    ASCEND_THROW_IF_NOT(op);
    std::shared_ptr<std::vector<const aclDataBuffer *>> topkOpInput(
            new std::vector<const aclDataBuffer *>(), CommonUtils::AclInputBufferDelete);
    for (auto tensor : input) {
        topkOpInput->emplace_back(aclCreateDataBuffer(tensor->getVoidData(), tensor->getSizeInBytes()));
    }
    std::shared_ptr<std::vector<aclDataBuffer *>> topkOpOutput(
            new std::vector<aclDataBuffer *>(), CommonUtils::AclOutputBufferDelete);
    for (auto tensor : output) {
        topkOpOutput->emplace_back(aclCreateDataBuffer(tensor->getVoidData(), tensor->getSizeInBytes()));
    }
    op->exec(*topkOpInput, *topkOpOutput, stream);
}


void IndexIVFPQ::callL3DistanceOp(size_t batch, size_t tileNum, size_t segNum, size_t coreNum, size_t kAligned,
                                  AscendTensor<float, DIMS_3, size_t> &l2SubspaceDists,
                                  AscendTensor<int64_t, DIMS_3, size_t> &codeOffset,
                                  AscendTensor<int64_t, DIMS_3, size_t> &codeSize,
                                  AscendTensor<int32_t, DIMS_2, size_t> &topk,
                                  AscendTensor<uint16_t, DIMS_3, size_t> &opFlag,
                                  AscendTensor<float, DIMS_3, size_t> &distResult,
                                  AscendTensor<int32_t, DIMS_3, size_t> &topkIndex,
                                  AscendTensor<float, DIMS_3, size_t> &topkValue,
                                  AscendTensor<uint8_t, DIMS_1, size_t> &codeBase,
                                  aclrtStream &stream)
{
    size_t ivfpqBlockSize = static_cast<size_t>(IVF_PQ_BLOCK_SIZE);
    for (size_t qIdx = 0; qIdx < batch; qIdx++) {
        AscendTensor<float, DIMS_2, size_t> queryPQ(l2SubspaceDists[qIdx][0].data(),
                                                    {static_cast<size_t>(M), static_cast<size_t>(ksub)});
        for (size_t tIdx = 0; tIdx < tileNum; tIdx++) {
            for (size_t segIdx = 0; segIdx < segNum; segIdx++) {
                AscendTensor <int64_t, DIMS_1, size_t> subcodeOffset(codeOffset[qIdx][tIdx][segIdx * coreNum].data(),
                                                                     {coreNum});
                AscendTensor <int64_t, DIMS_1, size_t> subcodeSize(codeSize[qIdx][tIdx][segIdx * coreNum].data(),
                                                                   {coreNum});
                AscendTensor <uint16_t, DIMS_2, size_t> subOpFlag(opFlag[qIdx][tIdx][segIdx * coreNum * 16].data(),
                                                                  {coreNum, 16});
                AscendTensor<float, DIMS_2, size_t> subDist(distResult[qIdx][tIdx]
                                                                      [segIdx * coreNum * ivfpqBlockSize].data(),
                                                            {coreNum, ivfpqBlockSize});
                AscendTensor <int32_t, DIMS_2, size_t> subTopkIndex(topkIndex[qIdx][tIdx]
                                                                            [segIdx * coreNum * kAligned].data(),
                                                                    {coreNum, kAligned});
                AscendTensor<float, DIMS_2, size_t> subTopkValue(topkValue[qIdx][tIdx]
                                                                          [segIdx * coreNum * kAligned].data(),
                                                                 {coreNum, kAligned});

                runL3DistOp(queryPQ, codeBase, subcodeOffset, subcodeSize, topk,
                            subDist, subTopkIndex, subTopkValue, subOpFlag, stream);
            }
        }
    }
}

APP_ERROR IndexIVFPQ::fillDisOpInputDataPQ(int k, size_t batch, size_t tileNum, size_t segNum, size_t coreNum,
                                           AscendTensor<int64_t, DIMS_3, size_t> &offset,
                                           AscendTensor<int64_t, DIMS_3, size_t> &baseSize,
                                           AscendTensor<int64_t, DIMS_3, size_t> &ids,
                                           AscendTensor<int64_t, DIMS_1> &attrs,
                                           AscendTensor<int64_t, DIMS_2> &l1TopNprobeIndicesHost)
{
    size_t ivfpqBlockSize = static_cast<size_t>(IVF_PQ_BLOCK_SIZE);
    std::vector<int64_t> offsetHost(batch * tileNum * segNum * coreNum, 0);
    AscendTensor<int64_t, DIMS_3, size_t> offsetHostVec(offsetHost.data(), {batch, tileNum, segNum * coreNum});
    std::vector<int64_t> baseSizeHost(batch * tileNum * segNum * coreNum, 0);
    AscendTensor<int64_t, DIMS_3, size_t> baseSizeHostVec(baseSizeHost.data(), {batch, tileNum, segNum * coreNum});
    std::vector<int64_t> idsHost(batch * tileNum * segNum * coreNum);
    AscendTensor<int64_t, DIMS_3, size_t> idsHostVec(idsHost.data(), {batch, tileNum, segNum * coreNum});
    for (size_t qIdx = 0; qIdx < batch; qIdx++) {
        for (size_t tIdx = 0; tIdx < tileNum; tIdx++) {
            for (size_t segIdx = 0; segIdx < segNum; segIdx++) {
                fillDisOpInputDataByBlockPQ(qIdx, tIdx, segIdx, segNum, coreNum, ivfpqBlockSize,
                                            baseSizeHostVec, offsetHostVec, idsHostVec, l1TopNprobeIndicesHost);
            }
        }
    }

    auto ret = aclrtMemcpy(offset.data(), offset.getSizeInBytes(), offsetHostVec.data(),
                           offsetHostVec.getSizeInBytes(), ACL_MEMCPY_HOST_TO_DEVICE);
    APPERR_RETURN_IF_NOT_FMT(ret == APP_ERR_OK, APP_ERR_INNER_ERROR, "copy offset to device failed %d", ret);
    ret = aclrtMemcpy(baseSize.data(), baseSize.getSizeInBytes(), baseSizeHostVec.data(),
                      baseSizeHostVec.getSizeInBytes(), ACL_MEMCPY_HOST_TO_DEVICE);
    APPERR_RETURN_IF_NOT_FMT(ret == APP_ERR_OK, APP_ERR_INNER_ERROR, "copy basesize to device failed %d", ret);
    ret = aclrtMemcpy(ids.data(), ids.getSizeInBytes(), idsHostVec.data(),
                      idsHostVec.getSizeInBytes(), ACL_MEMCPY_HOST_TO_DEVICE);
    APPERR_RETURN_IF_NOT_FMT(ret == APP_ERR_OK, APP_ERR_INNER_ERROR, "copy ids to device failed %d", ret);

    std::vector<int64_t> attrsVec(aicpu::TOPK_IVFPQ_L3_ATTR_IDX_COUNT);
    attrsVec[aicpu::TOPK_IVFPQ_L3_ATTR_ASC_IDX] = 1;
    attrsVec[aicpu::TOPK_IVFPQ_L3_ATTR_K_IDX] = k;
    attrsVec[aicpu::TOPK_IVFPQ_L3_ATTR_BLOCK_NUM_IDX] = static_cast<int64_t>(tileNum * segNum);
    attrsVec[aicpu::TOPK_IVFPQ_L3_ATTR_FLAG_NUM_IDX] = static_cast<int64_t>(coreNum);
    ret = aclrtMemcpy(attrs.data(), attrs.getSizeInBytes(),
                      attrsVec.data(), attrsVec.size() * sizeof(int64_t), ACL_MEMCPY_HOST_TO_DEVICE);
    APPERR_RETURN_IF_NOT_FMT(ret == APP_ERR_OK, APP_ERR_INNER_ERROR, "copy attrs to device failed %d", ret);
    return APP_ERR_OK;
}

void IndexIVFPQ::fillDisOpInputDataByBlockPQ(size_t qIdx, size_t tIdx, size_t segIdx,
                                             size_t segNum, size_t coreNum, size_t ivfpqBlockSize,
                                             AscendTensor<int64_t, DIMS_3, size_t> &baseSizeHostVec,
                                             AscendTensor<int64_t, DIMS_3, size_t> &offsetHostVec,
                                             AscendTensor<int64_t, DIMS_3, size_t> &idsHostVec,
                                             AscendTensor<int64_t, DIMS_2> &l1TopNprobeIndicesHost)
{
    for (size_t cIdx = 0; cIdx < coreNum; cIdx++) {
        if (cIdx + tIdx * coreNum < static_cast<size_t>(nprobe)) {
            int64_t listId = l1TopNprobeIndicesHost[qIdx][tIdx * coreNum + (segIdx * coreNum + cIdx) / segNum].value();
            size_t listNum = deviceListIndices[listId]->size();
            size_t proccessLen = std::min(listNum - segIdx * ivfpqBlockSize, ivfpqBlockSize);
            baseSizeHostVec[qIdx][tIdx][segIdx * coreNum + cIdx].value(proccessLen);
            int64_t idAddr = reinterpret_cast<int64_t>(deviceListIndices[listId]->data() +
                                                      ((segIdx * coreNum + cIdx) % segNum) * ivfpqBlockSize);
            idsHostVec[qIdx][tIdx][cIdx + segIdx * coreNum].value(idAddr);
            size_t lastBlock = (segIdx * coreNum + cIdx) % segNum;
            if (lastBlock * ivfpqBlockSize < basePQCoder[listId].size()) {
                uint64_t offsetSeg = reinterpret_cast<uint64_t>(basePQCoder[listId].at(segIdx)->data()) -
                                     reinterpret_cast<uint64_t>(pBasePQCoder);
                offsetHostVec[qIdx][tIdx][segIdx * coreNum + cIdx].value(offsetSeg);
            }
        }
    }
}

size_t IndexIVFPQ::getMaxListNum(size_t batch, AscendTensor<int64_t, DIMS_2> &l1TopNprobeIndicesHost) const
{
    size_t maxLen = 0;
    for (size_t qIdx = 0; qIdx < batch; qIdx++) {
        for (size_t probId = 0; probId < static_cast<size_t>(nprobe); probId++) {
            int64_t listId = l1TopNprobeIndicesHost[qIdx][probId].value();
            size_t listNum = deviceListIndices[listId]->size();
            maxLen = maxLen > listNum ? maxLen : listNum;
        }
    }
    return maxLen;
}

APP_ERROR IndexIVFPQ::searchImplL1(AscendTensor<float, DIMS_2> &queries,
                                   AscendTensor<int64_t, DIMS_2> &l1TopNprobeIndicesHost,
                                   AscendTensor<float, DIMS_3, size_t> &l2SubspaceDistsDev)
{
    auto &mem = resources.getMemoryManager();
    auto streamPtr = resources.getDefaultStream();
    auto stream = streamPtr->GetStream();
    auto streamAicpuPtr = resources.getAlternateStreams()[0];
    auto streamAicpu = streamAicpuPtr->GetStream();

    int n = queries.getSize(0);
    AscendTensor<float, DIMS_2> dists(mem, {n, numLists}, stream);
    int minDistSize = std::min(numLists / IVF_PQ_BURST_LEN * 2, MIN_EXTREME_SIZE);
    AscendTensor<float, DIMS_2> vmdists(mem, {n, minDistSize}, stream);
    AscendTensor<uint32_t, DIMS_2> opSize(mem, {CORE_NUM, SIZE_ALIGN}, stream);
    opSize[0][0] = numLists;
    AscendTensor<uint16_t, DIMS_2> opFlag(mem, {CORE_NUM, FLAG_SIZE}, stream);
    opFlag.zero();

    AscendTensor<int64_t, DIMS_1> attrsInput(mem, { aicpu::TOPK_FLAT_ATTR_IDX_COUNT }, stream);
    std::vector<int64_t> attrs(aicpu::TOPK_FLAT_ATTR_IDX_COUNT);
    attrs[aicpu::TOPK_FLAT_ATTR_ASC_IDX] = 1;
    attrs[aicpu::TOPK_FLAT_ATTR_K_IDX] = nprobe;
    attrs[aicpu::TOPK_FLAT_ATTR_BURST_LEN_IDX] = IVF_PQ_BURST_LEN;
    attrs[aicpu::TOPK_FLAT_ATTR_BLOCK_NUM_IDX] = 1;
    attrs[aicpu::TOPK_FLAT_ATTR_PAGE_IDX] = 0;
    attrs[aicpu::TOPK_FLAT_ATTR_PAGE_NUM_IDX] = 1;
    attrs[aicpu::TOPK_FLAT_ATTR_PAGE_SIZE_IDX] = 0;
    attrs[aicpu::TOPK_FLAT_ATTR_QUICK_HEAP] = 1;
    attrs[aicpu::TOPK_FLAT_ATTR_BLOCK_SIZE] = numLists;

    auto ret = aclrtMemcpy(attrsInput.data(), attrsInput.getSizeInBytes(),
                           attrs.data(), attrs.size() * sizeof(int64_t), ACL_MEMCPY_HOST_TO_DEVICE);
    APPERR_RETURN_IF_NOT_LOG(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "Failed to copy attr to device");

    AscendTensor<float, DIMS_2> l1TopNprobeDists(mem, {n, nprobe}, stream);
    AscendTensor<int64_t, DIMS_2> l1TopNprobeIndices(mem, {n, nprobe}, stream);

    AscendTensor<float, DIMS_3> codeBookDev(codeBookOnDevice->data(), {M, ksub, dsub});
    AscendTensor<float, DIMS_3> l2MinSubspaceDistsDev(mem, {n, M, ksub / IVF_PQ_BURST_LEN * 2}, stream);
    AscendTensor<uint16_t, DIMS_3> opFlagL2(mem, {n, M, 16}, stream);

    // run l1&l2 distance calculation
    AscendTensor<float, DIMS_2> centroidsDev(centroidsOnDevice->data(), {numLists, dims});
    runL1TopkOp(dists, vmdists, opSize, opFlag, attrsInput, l1TopNprobeDists, l1TopNprobeIndices, streamAicpu);
    runL1DistOp(n, queries, centroidsDev, dists, vmdists, opFlag, stream);
    runL2DistOp(n, queries, codeBookDev, l2SubspaceDistsDev, l2MinSubspaceDistsDev, opFlagL2, stream);

    ret = synchronizeStream(stream);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR,
                             "synchronizeStream default stream: %i\n", ret);
    ret = synchronizeStream(streamAicpu);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR,
                             "synchronizeStream aicpu stream failed: %i\n", ret);

    ret = aclrtMemcpy(l1TopNprobeIndicesHost.data(), l1TopNprobeIndicesHost.getSizeInBytes(),
                      l1TopNprobeIndices.data(), l1TopNprobeIndices.getSizeInBytes(),
                      ACL_MEMCPY_DEVICE_TO_HOST);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "Mem operator error %d", ret);

    return APP_ERR_OK;
}

APP_ERROR IndexIVFPQ::searchImplL3(AscendTensor<int64_t, DIMS_2> &l1TopNprobeIndicesHost,
                                   AscendTensor<float, DIMS_3, size_t> &l2SubspaceDistsDev,
                                   int k, float* distances, idx_t* labels)
{
    auto &mem = resources.getMemoryManager();
    auto streamPtr = resources.getDefaultStream();
    auto stream = streamPtr->GetStream();
    auto streamAicpuPtr = resources.getAlternateStreams()[0];
    auto streamAicpu = streamAicpuPtr->GetStream();

    size_t batch = static_cast<size_t>(l2SubspaceDistsDev.getSize(0));
    size_t tileNum = static_cast<size_t>(utils::divUp(nprobe, blockNum));
    size_t coreNum = static_cast<size_t>(blockNum);
    size_t maxLen = getMaxListNum(batch, l1TopNprobeIndicesHost);
    size_t ivfpqBlockSize = static_cast<size_t>(IVF_PQ_BLOCK_SIZE);
    size_t segNum = utils::divUp(maxLen, ivfpqBlockSize);
    size_t kAligned = static_cast<size_t>(MAX_TOPK);

    AscendTensor<uint16_t, DIMS_3, size_t> opFlag(mem, {batch, tileNum, segNum * coreNum * 16}, stream);
    (void)opFlag.zero();
    AscendTensor<int64_t, DIMS_3, size_t> offset(mem, {batch, tileNum, segNum * coreNum}, stream);
    AscendTensor<int64_t, DIMS_3, size_t> baseSize(mem, {batch, tileNum, segNum * coreNum}, stream);
    AscendTensor<uint8_t, DIMS_1, size_t> codeBase(pBasePQCoder, {static_cast<size_t>(M)});

    AscendTensor<int32_t, DIMS_3, size_t> topkIndex(mem, {batch, tileNum, coreNum * segNum * kAligned}, stream);
    AscendTensor<float, DIMS_3, size_t> topkValue(mem, {batch, tileNum, coreNum * segNum * kAligned}, stream);
    AscendTensor<float, DIMS_3, size_t> distResult(mem, {batch, tileNum, coreNum * segNum * ivfpqBlockSize}, stream);
    AscendTensor<int64_t, DIMS_3, size_t> ids(mem, {batch, tileNum, segNum * coreNum}, stream);

    AscendTensor<int64_t, DIMS_1> attrs(mem, {aicpu::TOPK_IVFPQ_L3_ATTR_IDX_COUNT}, stream);
    AscendTensor<int32_t, DIMS_2, size_t> topk(mem, {kAligned, 16384}, stream);

    fillDisOpInputDataPQ(k, batch, tileNum, segNum, coreNum, offset, baseSize, ids, attrs, l1TopNprobeIndicesHost);

    AscendTensor<float, DIMS_2, size_t> outDist(mem, {batch, static_cast<size_t>(k)}, stream);
    AscendTensor<idx_t, DIMS_2, size_t> outLabel(mem, {batch, static_cast<size_t>(k)}, stream);

    runL3TopkOp(topkIndex, topkValue, ids, baseSize, opFlag, attrs, outDist, outLabel, streamAicpu);
    callL3DistanceOp(batch, tileNum, segNum, coreNum, kAligned, l2SubspaceDistsDev, offset, baseSize, topk,
                     opFlag, distResult, topkIndex, topkValue, codeBase, stream);
    auto ret = synchronizeStream(stream);
    APPERR_RETURN_IF_NOT_FMT(ret == APP_ERR_OK, APP_ERR_INNER_ERROR, "sync aicore stream failed %d", ret);

    ret = synchronizeStream(streamAicpu);
    APPERR_RETURN_IF_NOT_FMT(ret == APP_ERR_OK, APP_ERR_INNER_ERROR, "sync aicpu stream failed %d", ret);

    ret = aclrtMemcpy(distances, batch * k * sizeof(float), outDist.data(),
                      outDist.getSizeInBytes(), ACL_MEMCPY_DEVICE_TO_HOST);
    APPERR_RETURN_IF_NOT_FMT(ret == APP_ERR_OK, APP_ERR_INNER_ERROR, "copy distances to host failed %d", ret);
    ret = aclrtMemcpy(labels, batch * k * sizeof(idx_t), outLabel.data(),
                      outLabel.getSizeInBytes(), ACL_MEMCPY_DEVICE_TO_HOST);
    APPERR_RETURN_IF_NOT_FMT(ret == APP_ERR_OK, APP_ERR_INNER_ERROR, "copy outLabel to host failed %d", ret);
    return APP_ERR_OK;
}

APP_ERROR IndexIVFPQ::searchWithBatch(int n, const float * x, int k, float* distances, idx_t* labels)
{
    auto &mem = resources.getMemoryManager();
    auto streamPtr = resources.getDefaultStream();
    auto stream = streamPtr->GetStream();
    AscendTensor<float, DIMS_2> queries(mem, {n, dims}, stream);
    auto ret = aclrtMemcpy(queries.data(), queries.getSizeInBytes(),
                           x, n * dims * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "copy query error %d", ret);

    std::vector<int64_t> l1TopNprobeIndicesVec(n * nprobe, 0);
    AscendTensor<int64_t, DIMS_2> l1TopNprobeIndicesHost(l1TopNprobeIndicesVec.data(), { n, nprobe });
    AscendTensor<float, DIMS_3, size_t> l2SubspaceDistsDev(mem,
                                                           {static_cast<size_t>(n), static_cast<size_t>(M),
                                                            static_cast<size_t>(ksub)},
                                                           stream);
    ret = searchImplL1(queries, l1TopNprobeIndicesHost, l2SubspaceDistsDev);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, ret, "IVFPQ L1 search failed! %d", ret);

    ret = searchImplL3(l1TopNprobeIndicesHost, l2SubspaceDistsDev, k, distances, labels);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, ret, "IVFPQ L3 search failed! %d", ret);
    return ACL_SUCCESS;
}

APP_ERROR IndexIVFPQ::searchImpl(int n, const float * x, int k, float* distances, idx_t* labels)
{
    APPERR_RETURN_IF_NOT_FMT(k >= 1 && k <= MAX_TOPK, APP_ERR_INVALID_PARAM,
                             "the k is %d, out of MAX_TOPK(%d)", k, MAX_TOPK);

    APP_ERROR ret = APP_ERR_OK;
    if (n == 1 || searchBatchSizes.empty()) {
        return searchWithBatch(n, x, k, distances, labels);
    }
    size_t size = searchBatchSizes.size();
    int64_t searched = 0;
    for (size_t i = 0; i < size; i++) {
        int64_t batchSize = searchBatchSizes[i];
        if ((n - searched) >= batchSize) {
            int64_t page = (n - searched) / batchSize;
            for (int64_t j = 0; j < page; j++) {
                ret = searchWithBatch(batchSize, x + searched * dims, k, distances + searched * k,
                                      labels + searched * k);
                APPERR_RETURN_IF(ret, ret);
                searched += batchSize;
            }
        }
    }
    for (int64_t i = searched; i < n; i++) {
        ret = searchWithBatch(1, x + i * dims, k, distances + i * k, labels + i * k);
        APPERR_RETURN_IF(ret, ret);
    }
    return APP_ERR_OK;
}

APP_ERROR IndexIVFPQ::searchImpl(int n, const float16_t* x, int k, float16_t* distances, idx_t* labels)
{
    VALUE_UNUSED(n);
    VALUE_UNUSED(x);
    VALUE_UNUSED(k);
    VALUE_UNUSED(distances);
    VALUE_UNUSED(labels);
    return APP_ERR_OK;
}

APP_ERROR IndexIVFPQ::searchImpl(AscendTensor<float16_t, DIMS_2> &queries, int k,
                                 AscendTensor<float16_t, DIMS_2> &outDistance, AscendTensor<idx_t, DIMS_2> &outIndices)
{
    VALUE_UNUSED(queries);
    VALUE_UNUSED(k);
    VALUE_UNUSED(outDistance);
    VALUE_UNUSED(outIndices);
    return APP_ERR_OK;
}

APP_ERROR IndexIVFPQ::searchPaged(size_t pageId, size_t pageNum, AscendTensor<float16_t, DIMS_2> &queries,
                                  AscendTensor<float16_t, DIMS_2> &maxDistances,
                                  AscendTensor<int64_t, DIMS_2> &maxIndices)
{
    VALUE_UNUSED(pageId);
    VALUE_UNUSED(pageNum);
    VALUE_UNUSED(queries);
    VALUE_UNUSED(maxDistances);
    VALUE_UNUSED(maxIndices);
    return APP_ERR_OK;
}

} // ascend