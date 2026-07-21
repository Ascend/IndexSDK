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

#include <faiss/impl/FaissAssert.h>
#include <faiss/utils/random.h>
#include <omp.h>
#include <sys/time.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <vector>

#include "ascenddaemon/impl/AuxIndexStructures.h"
#include "ascenddaemon/utils/DevVecMemStrategyIntf.h"
#include "ascenddaemon/utils/Limits.h"
#include "common/utils/CommonUtils.h"
#include "common/utils/LogUtils.h"
#include "common/utils/OpLauncher.h"
#include "ops/cpukernel/impl/utils/kernel_shared_def.h"

namespace ascend
{
const int KB = 1024;
const int MAX_ITER_NUM = 25;
const int TRAIN_SAMPLES_PER_LIST = 40;
const int MAX_TRAIN_SAMPLES = 10000000;
const int IVF_PQ_BLOCK_SIZE = 16384 * 16;

static int trainMinDistCols(int nlist) { return nlist / IVF_PQ_BURST_LEN * 2; }

static int computeTrainTotalSize(int nlist, int n)
{
    int capped = std::min(nlist * (nlist <= 16384 ? 256 : TRAIN_SAMPLES_PER_LIST), n);
    return std::min(capped, MAX_TRAIN_SAMPLES);
}
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
    pBasePQCoder = reinterpret_cast<uint8_t *>(0xffffffffffffffff);
    blockNum = 128;
    this->nbits = nbits;
    ksub = (1 << nbits);
    dsub = utils::divUp(dim, M);
    centroidsOnDevice = CREATE_UNIQUE_PTR(DeviceVector<float>, MemorySpace::DEVICE_HUGEPAGE);
    centroidsOnDevice->resize(numList * dim);
    centroidsSqrSumOnDevice = CREATE_UNIQUE_PTR(DeviceVector<float>, MemorySpace::DEVICE_HUGEPAGE);
    centroidsSqrSumOnDevice->resize(numList);

    initializeCodeBook(M, nbits, dim / M);
    auto ret = resetL1TopkOp();
    ASCEND_THROW_IF_NOT_MSG(ret == APP_ERR_OK, "resetL1TopkOp failed!");
    ret = resetL1DistOp();
    ASCEND_THROW_IF_NOT_MSG(ret == APP_ERR_OK, "resetL1DistOp failed!");
}

IndexIVFPQ::~IndexIVFPQ() {}

APP_ERROR IndexIVFPQ::reset()
{
    IndexIVF::reset();
    for (size_t i = 0; i < basePQCoder.size(); i++)
    {
        basePQCoder[i].clear();
    }
    basePQCoder.clear();
    basePQCoder.resize(numLists);

    listVecNum.clear();
    listVecNum.resize(numLists, 0);

    if (centroidsOnDevice)
    {
        centroidsOnDevice->clear();
        centroidsOnDevice->resize(numLists * dims);
    }

    isTrained = false;

    resetTrainSession();
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
    return (seed >= 0) ? seed
                       : static_cast<uint64_t>(std::chrono::high_resolution_clock::now().time_since_epoch().count());
}

size_t IndexIVFPQ::getPQVecCapacity(size_t vecNum, size_t size, int M) const
{
    const size_t totalMinBudget = 512UL * 1024 * 1024;
    size_t minCapacity = std::min(static_cast<size_t>(512 * KB), totalMinBudget / static_cast<size_t>(numLists));
    minCapacity = std::max(minCapacity, static_cast<size_t>(4 * KB));

    int vecAlign = M / CUBE_ALIGN;
    size_t divisor = vecNum / static_cast<size_t>(CUBE_ALIGN);
    size_t remainder = vecNum % static_cast<size_t>(CUBE_ALIGN);

    size_t offset =
        divisor * static_cast<size_t>(vecAlign * CUBE_ALIGN * CUBE_ALIGN) + remainder * static_cast<size_t>(CUBE_ALIGN);
    size_t needSize = offset + static_cast<size_t>(vecAlign * CUBE_ALIGN * CUBE_ALIGN);

    needSize = std::max(needSize, vecNum * static_cast<size_t>(M));

    const size_t align = 2 * KB * KB;

    if (needSize < minCapacity)
    {
        return minCapacity;
    }
    if (needSize <= size)
    {
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

    if (vecSize > 0)
    {
        int vecId = vecSize - 1;
        if (addVecNum > 0)
        {
            basePQCoder[listId].at(vecId)->resize(this->blockSize * M, true);
        }
        else
        {
            size_t vecNumInLastBlock =
                listVecNum[listId] - static_cast<size_t>(vecId * blockSize) + static_cast<size_t>(numVecs);
            auto oldCapacity = basePQCoder[listId].at(vecId)->size();
            auto capacity = getPQVecCapacity(vecNumInLastBlock, oldCapacity, M);
            basePQCoder[listId].at(vecId)->resize(capacity, true);
        }
    }

    for (int i = 0; i < addVecNum; ++i)
    {
        basePQCoder.at(listId).emplace_back(CREATE_UNIQUE_PTR(DeviceVector<uint8_t>, MemorySpace::DEVICE_HUGEPAGE));
        size_t newVecId = static_cast<size_t>(vecSize + i);
        size_t tailVecId = newVecSize <= 1 ? 0 : static_cast<size_t>(newVecSize - 1);

        if (newVecId == tailVecId)
        {
            size_t vecNumInLastBlock =
                listVecNum[listId] - newVecId * static_cast<size_t>(blockSize) + static_cast<size_t>(numVecs);
            auto capacity = getPQVecCapacity(vecNumInLastBlock, 0, M);
            basePQCoder[listId].at(newVecId)->resize(capacity, true);
        }
        else
        {
            basePQCoder[listId].at(newVecId)->resize(this->blockSize * M, true);
        }

        pBasePQCoder = basePQCoder[listId].at(newVecId)->data() < pBasePQCoder
                           ? basePQCoder[listId].at(newVecId)->data()
                           : pBasePQCoder;
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
                                              {static_cast<int32_t>(numVecs), bytes_per_vector});

    APPERR_RETURN_IF_NOT_LOG(APP_ERR_OK == resizeBasePQ(listId, numVecs), APP_ERR_INNER_ERROR,
                             "resize PQ base failed!");

    auto ret = AddCodeNDFormat<uint8_t>(pqCodesData, listVecNum[listId], blockSize, basePQCoder[listId]);
    APPERR_RETURN_IF_NOT_FMT(ret == APP_ERR_OK, APP_ERR_INNER_ERROR, "add PQ codes to device failed %d", ret);

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

APP_ERROR IndexIVFPQ::getListInitialize(int listId, std::vector<unsigned char> &reshaped, size_t &totalVecsInList,
                                        size_t &codeSizePerVector, size_t &totalBytes) const
{
    APPERR_RETURN_IF_NOT_FMT(listId >= 0 && listId < numLists, APP_ERR_INVALID_PARAM, "Invalid listId %d, numLists=%d",
                             listId, numLists);

    totalVecsInList = listVecNum[listId];
    if (totalVecsInList == 0)
    {
        reshaped.clear();
        return APP_ERR_OK;
    }

    codeSizePerVector = static_cast<size_t>(M);
    totalBytes = totalVecsInList * codeSizePerVector;
    reshaped.resize(totalBytes);

    return APP_ERR_OK;
}

APP_ERROR IndexIVFPQ::copyPQBlocks(int listId, size_t totalVecsInList, size_t codeSizePerVector,
                                   std::vector<unsigned char> &reshaped, size_t &destOffset,
                                   size_t &processedVecs) const
{
    const auto &blocks = basePQCoder[listId];
    size_t numBlocks = blocks.size();
    if (numBlocks == 0)
    {
        APPERR_RETURN_IF_NOT_FMT(totalVecsInList == 0, APP_ERR_INNER_ERROR, "List %d has %zu vectors but no blocks",
                                 listId, totalVecsInList);
        return APP_ERR_OK;
    }

    destOffset = 0;
    processedVecs = 0;

    for (size_t blockIdx = 0; blockIdx < numBlocks; blockIdx++)
    {
        const auto &block = blocks[blockIdx];
        if (!block || block->size() == 0)
        {
            APP_LOG_WARNING("Block %zu in list %d is null or empty", blockIdx, listId);
            continue;
        }

        size_t vecsInBlock = (blockIdx == numBlocks - 1)
                                 ? totalVecsInList - processedVecs
                                 : std::min(static_cast<size_t>(blockSize), totalVecsInList - processedVecs);

        size_t bytesToCopy = vecsInBlock * codeSizePerVector;

        if (block->size() < bytesToCopy)
        {
            size_t actualVecsInBlock = block->size() / codeSizePerVector;
            bytesToCopy = actualVecsInBlock * codeSizePerVector;
            vecsInBlock = actualVecsInBlock;
        }

        if (bytesToCopy == 0)
        {
            continue;
        }

        auto ret = aclrtMemcpy(reshaped.data() + destOffset, bytesToCopy, block->data(), bytesToCopy,
                               ACL_MEMCPY_DEVICE_TO_HOST);
        APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, ACL_ERROR_FAILURE, "Failed to copy block %zu: %d", blockIdx, ret);

        destOffset += bytesToCopy;
        processedVecs += vecsInBlock;

        APP_LOG_DEBUG("Block %zu: copied %zu vectors (%zu bytes), total processed %zu/%zu\n", blockIdx, vecsInBlock,
                      bytesToCopy, processedVecs, totalVecsInList);
    }

    return APP_ERR_OK;
}

APP_ERROR IndexIVFPQ::getListVectorsReshaped(int listId, std::vector<unsigned char> &reshaped) const
{
    size_t totalVecsInList;
    size_t codeSizePerVector;
    size_t totalBytes;
    size_t destOffset;
    size_t processedVecs;
    auto ret = getListInitialize(listId, reshaped, totalVecsInList, codeSizePerVector, totalBytes);
    APPERR_RETURN_IF_NOT_FMT(ret == APP_ERR_OK, ret, "Failed to get list initialize for list %d: %d", listId, ret);

    ret = copyPQBlocks(listId, totalVecsInList, codeSizePerVector, reshaped, destOffset, processedVecs);
    APPERR_RETURN_IF_NOT_FMT(ret == APP_ERR_OK, ret, "Failed to process Blocks for list %d: %d", listId, ret);

    APPERR_RETURN_IF_NOT_FMT(processedVecs == totalVecsInList, APP_ERR_INNER_ERROR,
                             "Vector count mismatch: expected %zu, processed %zu", totalVecsInList, processedVecs);

    APPERR_RETURN_IF_NOT_FMT(destOffset == totalBytes, APP_ERR_INNER_ERROR,
                             "Byte count mismatch: expected %zu, copied %zu", totalBytes, destOffset);

    APP_LOG_INFO("Successfully reshaped list %d: %zu vectors, %zu bytes, %zu blocks\n", listId, totalVecsInList,
                 totalBytes, basePQCoder[listId].size());

    return APP_ERR_OK;
}

APP_ERROR IndexIVFPQ::prepareDelete(int listId, size_t numVecs, const idx_t *indices, size_t &currentVecNum,
                                    std::vector<idx_t> &hostIds, std::vector<bool> &toDelete,
                                    std::vector<idx_t> &deletedIds, size_t &deleteCount)
{
    APPERR_RETURN_IF_NOT_FMT(listId >= 0 && listId < numLists, APP_ERR_INVALID_PARAM,
                             "the listId is %d, out of numLists(%d)", listId, numLists);
    APPERR_RETURN_IF(numVecs == 0, APP_ERR_OK);

    currentVecNum = listVecNum[listId];
    if (currentVecNum == 0)
    {
        APP_LOG_WARNING("List %d is empty, nothing to delete\n", listId);
        return APP_ERR_OK;
    }

    hostIds.resize(currentVecNum);
    if (currentVecNum > 0)
    {
        auto ret = aclrtMemcpy(hostIds.data(), currentVecNum * sizeof(idx_t), deviceListIndices[listId]->data(),
                               currentVecNum * sizeof(idx_t), ACL_MEMCPY_DEVICE_TO_HOST);

        APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, ACL_ERROR_FAILURE, "Failed to copy IDs from device: %d", ret);
    }

    std::unordered_set<idx_t> deleteSet;
    for (size_t i = 0; i < numVecs; i++)
    {
        deleteSet.insert(indices[i]);
    }

    toDelete.resize(currentVecNum, false);
    deletedIds.clear();
    deleteCount = 0;

    for (size_t i = 0; i < currentVecNum; i++)
    {
        if (deleteSet.find(hostIds[i]) != deleteSet.end())
        {
            toDelete[i] = true;
            deletedIds.push_back(hostIds[i]);
            deleteCount++;
        }
    }

    if (deleteCount == 0)
    {
        APP_LOG_WARNING("No matching IDs found to delete in list %d\n", listId);
        return APP_ERR_OK;
    }

    APP_LOG_DEBUG("Found %zu IDs to delete from list %d\n", deleteCount, listId);

    return APP_ERR_OK;
}

APP_ERROR IndexIVFPQ::updateFilterPQCodes(int listId, size_t currentVecNum, std::vector<idx_t> &hostIds,
                                          std::vector<bool> &toDelete, size_t deleteCount, size_t bytesPerVector,
                                          std::vector<uint8_t> &newCodes, std::vector<idx_t> &newIds)
{
    std::vector<uint8_t> reshapedCodes;
    APP_ERROR ret = getListVectorsReshaped(listId, reshapedCodes);
    APPERR_RETURN_IF_NOT_FMT(ret == APP_ERR_OK, ret, "Failed to get PQ codes for list %d: %d", listId, ret);

    size_t newVecNum = currentVecNum - deleteCount;
    newCodes.resize(newVecNum * bytesPerVector);
    newIds.resize(newVecNum);

    size_t srcOffset = 0;
    size_t dstOffset = 0;
    size_t idIndex = 0;

    for (size_t i = 0; i < currentVecNum; i++)
    {
        if (!toDelete[i])
        {
            std::copy(reshapedCodes.begin() + srcOffset, reshapedCodes.begin() + srcOffset + bytesPerVector,
                      newCodes.begin() + dstOffset);

            newIds[idIndex++] = hostIds[i];

            dstOffset += bytesPerVector;
        }
        srcOffset += bytesPerVector;
    }

    return APP_ERR_OK;
}

APP_ERROR IndexIVFPQ::updateDeviceData(int listId, size_t newVecNum, std::vector<uint8_t> &newCodes,
                                       const std::vector<idx_t> &newIds)
{
    basePQCoder[listId].clear();
    deviceListIndices[listId]->resize(0, true);

    if (newVecNum > 0)
    {
        listVecNum[listId] = 0;

        AscendTensor<uint8_t, DIMS_2> newPQCodesTensor(
            newCodes.data(), {static_cast<int32_t>(newVecNum), static_cast<int32_t>(bytesPerVector)});

        APP_ERROR ret = resizeBasePQ(listId, newVecNum);
        APPERR_RETURN_IF_NOT_FMT(ret == APP_ERR_OK, ret, "Failed to resize base PQ for list %d: %d", listId, ret);

        ret = AddCodeNDFormat<uint8_t>(newPQCodesTensor, 0, blockSize, basePQCoder[listId]);
        APPERR_RETURN_IF_NOT_FMT(ret == APP_ERR_OK, ret, "Failed to add PQ codes to device: %d", ret);

        deviceListIndices[listId]->append(newIds.data(), newVecNum, true);
        listVecNum[listId] = newVecNum;
    }
    else
    {
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
    APPERR_RETURN_IF_NOT_FMT(ret == APP_ERR_OK, ret, "Failed to prepare delete for list %d: %d", listId, ret);

    size_t bytesPerVector = static_cast<size_t>(M);
    std::vector<uint8_t> newCodes;
    std::vector<idx_t> newIds;
    ret = updateFilterPQCodes(listId, currentVecNum, hostIds, toDelete, deleteCount, bytesPerVector, newCodes, newIds);
    APPERR_RETURN_IF_NOT_FMT(ret == APP_ERR_OK, ret, "Failed to process PQ Codes for list %d: %d", listId, ret);

    size_t newVecNum = currentVecNum - deleteCount;

    ret = updateDeviceData(listId, newVecNum, newCodes, newIds);
    APPERR_RETURN_IF_NOT_FMT(ret == APP_ERR_OK, ret, "Failed to update to device for list %d: %d", listId, ret);

    IndexIVF::ntotal -= deleteCount;

    maxListLength = 0;
    for (int i = 0; i < numLists; i++)
    {
        if (listVecNum[i] > static_cast<size_t>(maxListLength))
        {
            maxListLength = static_cast<int>(listVecNum[i]);
        }
    }

    if (maxListLength > 0)
    {
        maxListLength = utils::roundUp(maxListLength, CUBE_ALIGN);
    }

    APP_LOG_INFO("Successfully deleted %zu vectors from list %d, remaining: %zu\n", deleteCount, listId,
                 listVecNum[listId]);

    return APP_ERR_OK;
}

void IndexIVFPQ::normL2(int dim, int nlist, float *data)
{
#pragma omp parallel for
    for (int i = 0; i < nlist; ++i)
    {
        float *vec = data + i * dim;

        float norm2 = 0.0f;
        for (int d = 0; d < dim; ++d)
        {
            norm2 += vec[d] * vec[d];
        }

        if (norm2 > 1e-30f)
        {
            float inv_norm = 1.0f / sqrtf(norm2);
            for (int d = 0; d < dim; ++d)
            {
                vec[d] *= inv_norm;
            }
        }
    }
}

APP_ERROR IndexIVFPQ::initTraining(int totalSize, int dim, int nlist, std::vector<float> &trainData,
                                   std::vector<float> &centroids, int seed)
{
    const size_t totalSizeSz = static_cast<size_t>(totalSize);
    const size_t dimSz = static_cast<size_t>(dim);
    const size_t numElems = totalSizeSz * dimSz;
    const size_t numBytes = numElems * sizeof(float);
    if (verbose)
    {
        APP_LOG_INFO("IVFPQ initTraining: totalSize=%d, dim=%d, nlist=%d, trainDataBytes=%zu\n", totalSize, dim, nlist,
                     numBytes);
    }
    constexpr size_t kLargeTrainDataBytes = 1024UL * 1024 * 1024;
    if (verbose && numBytes > kLargeTrainDataBytes)
    {
        APP_LOG_WARNING("IVFPQ initTraining: large trainData allocation %zu bytes (>1GB), may cause memory pressure\n",
                        numBytes);
    }

    std::vector<int> perm(totalSize, 0);
    const uint64_t actSeed = getActualRngSeed(seed);
    faiss::rand_perm(perm.data(), totalSize, actSeed + 1);
    for (int i = 0; i < nlist; i++)
    {
        const size_t dstOffset = static_cast<size_t>(i) * dimSz;
        const size_t srcOffset = static_cast<size_t>(perm[i]) * dimSz;
        const size_t rowBytes = dimSz * sizeof(float);
        auto ret = memcpy_s(centroids.data() + dstOffset, (centroids.size() - dstOffset) * sizeof(float),
                            trainData.data() + srcOffset, rowBytes);
        ASCEND_THROW_IF_NOT_FMT(ret == EOK, "Failed to copy to centroids (error %d)", ret);
    }
    normL2(dim, nlist, centroids.data());
    return APP_ERR_OK;
}

void IndexIVFPQ::sampleTrainData(const float *x, int n, int dim, int totalSize, int seed, std::vector<float> &trainData)
{
    const size_t dimSz = static_cast<size_t>(dim);
    const size_t totalSizeSz = static_cast<size_t>(totalSize);
    trainData.resize(totalSizeSz * dimSz);
    const size_t numBytes = totalSizeSz * dimSz * sizeof(float);

    if (totalSize >= n)
    {
        auto ret = memcpySChunked(trainData.data(), numBytes, x, numBytes);
        ASCEND_THROW_IF_NOT_FMT(ret == EOK, "trainData memcpy_s failed %d", ret);
        return;
    }

    std::vector<size_t> indices(static_cast<size_t>(n));
    std::iota(indices.begin(), indices.end(), 0);
    std::mt19937 gen(static_cast<uint32_t>(getActualRngSeed(seed)));
    std::shuffle(indices.begin(), indices.end(), gen);
    for (int i = 0; i < totalSize; i++)
    {
        const size_t srcIdx = indices[static_cast<size_t>(i)];
        const float *src = x + srcIdx * dimSz;
        float *dst = trainData.data() + static_cast<size_t>(i) * dimSz;
        auto ret = memcpy_s(dst, dimSz * sizeof(float), src, dimSz * sizeof(float));
        ASCEND_THROW_IF_NOT_FMT(ret == EOK, "trainData sample memcpy_s failed %d", ret);
    }
    APP_LOG_DEBUG("IVFPQ sampleTrainData: shuffled %d vectors from input n=%d\n", totalSize, n);
}

APP_ERROR IndexIVFPQ::resetTrainOp(int nlist, int dim)
{
    trainDistOps.clear();
    trainTopkOps.clear();
    APP_ERROR ret = resetTrainDistOp(nlist, dim);
    APPERR_RETURN_IF_NOT_FMT(ret == APP_ERR_OK, APP_ERR_INNER_ERROR, "reset dist op failed %d", ret);

    ret = resetTrainTopkOp(nlist);
    APPERR_RETURN_IF_NOT_FMT(ret == APP_ERR_OK, APP_ERR_INNER_ERROR, "reset topk op failed %d", ret);

    return APP_ERR_OK;
}

void IndexIVFPQ::ensureTrainBatchWorkspace(int nlist, int batchSize)
{
    const int maxBatch = std::max(batchSize, MAX_BATCH_SIZE);
    if (trainWsNlist_ == nlist && trainWsMaxBatch_ >= maxBatch && trainBatchDistsDev_)
    {
        return;
    }

    if (!trainBatchDistsDev_)
    {
        trainBatchDistsDev_ = CREATE_UNIQUE_PTR(DeviceVector<float>, MemorySpace::DEVICE_HUGEPAGE);
        trainBatchVmdistsDev_ = CREATE_UNIQUE_PTR(DeviceVector<float>, MemorySpace::DEVICE_HUGEPAGE);
        trainBatchDistsDevAlt_ = CREATE_UNIQUE_PTR(DeviceVector<float>, MemorySpace::DEVICE_HUGEPAGE);
        trainBatchVmdistsDevAlt_ = CREATE_UNIQUE_PTR(DeviceVector<float>, MemorySpace::DEVICE_HUGEPAGE);
        trainBatchOutdistsDev_ = CREATE_UNIQUE_PTR(DeviceVector<float>, MemorySpace::DEVICE_HUGEPAGE);
        trainBatchOutlabelDev_ = CREATE_UNIQUE_PTR(DeviceVector<int64_t>, MemorySpace::DEVICE_HUGEPAGE);
    }

    trainBatchDistsDev_->resize(static_cast<size_t>(maxBatch) * static_cast<size_t>(nlist));
    trainBatchVmdistsDev_->resize(static_cast<size_t>(maxBatch) * static_cast<size_t>(trainMinDistCols(nlist)));
    trainBatchDistsDevAlt_->resize(static_cast<size_t>(maxBatch) * static_cast<size_t>(nlist));
    trainBatchVmdistsDevAlt_->resize(static_cast<size_t>(maxBatch) * static_cast<size_t>(trainMinDistCols(nlist)));
    trainBatchOutdistsDev_->resize(static_cast<size_t>(maxBatch));
    trainBatchOutlabelDev_->resize(static_cast<size_t>(maxBatch));
    trainWsNlist_ = nlist;
    trainWsMaxBatch_ = maxBatch;
}

void IndexIVFPQ::ensureTrainAssignBuffer(int totalSize)
{
    if (trainAssignsDev_ && trainAssignsCapacity_ >= totalSize)
    {
        return;
    }
    if (!trainAssignsDev_)
    {
        trainAssignsDev_ = CREATE_UNIQUE_PTR(DeviceVector<int64_t>, MemorySpace::DEVICE_HUGEPAGE);
    }
    trainAssignsDev_->resize(static_cast<size_t>(totalSize));
    trainAssignsCapacity_ = totalSize;
}

void IndexIVFPQ::ensureTrainCentroidsDevice(int nlist, int dim)
{
    if (trainCentroidsDev_ && trainCentroidsNlist_ == nlist && trainCentroidsDim_ == dim)
    {
        return;
    }
    if (!trainCentroidsDev_)
    {
        trainCentroidsDev_ = CREATE_UNIQUE_PTR(DeviceVector<float>, MemorySpace::DEVICE_HUGEPAGE);
        trainCentroidsSqrSumDev_ = CREATE_UNIQUE_PTR(DeviceVector<float>, MemorySpace::DEVICE_HUGEPAGE);
    }
    trainCentroidsDev_->resize(static_cast<size_t>(nlist) * static_cast<size_t>(dim));
    trainCentroidsSqrSumDev_->resize(static_cast<size_t>(nlist));
    trainCentroidsNlist_ = nlist;
    trainCentroidsDim_ = dim;
}

APP_ERROR IndexIVFPQ::syncTrainCentroidsToDevice(int nlist, int dim, const std::vector<float> &centroids)
{
    ensureTrainCentroidsDevice(nlist, dim);
    const size_t centroidsBytes = static_cast<size_t>(nlist) * static_cast<size_t>(dim) * sizeof(float);
    auto ret = aclrtMemcpy(trainCentroidsDev_->data(), centroidsBytes, centroids.data(), centroidsBytes,
                           ACL_MEMCPY_HOST_TO_DEVICE);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "upload train centroids failed, size=%zu, ret=%d",
                             centroidsBytes, ret);

    std::vector<float> centroidsSqrSum(static_cast<size_t>(nlist));
    for (int i = 0; i < nlist; i++)
    {
        float normSQ = 0.0f;
        for (int j = 0; j < dim; j++)
        {
            float val = centroids[static_cast<size_t>(i) * static_cast<size_t>(dim) + static_cast<size_t>(j)];
            normSQ += val * val;
        }
        centroidsSqrSum[static_cast<size_t>(i)] = normSQ;
    }
    ret = aclrtMemcpy(trainCentroidsSqrSumDev_->data(), static_cast<size_t>(nlist) * sizeof(float),
                      centroidsSqrSum.data(), static_cast<size_t>(nlist) * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "upload train centroids sqr sum failed, ret=%d",
                             ret);
    return APP_ERR_OK;
}

APP_ERROR IndexIVFPQ::syncTrainCentroidsFromHost(int nlist, int dim, const float *centroids)
{
    APPERR_RETURN_IF_NOT_LOG(centroids != nullptr, APP_ERR_INVALID_PARAM, "centroids must not be null");
    ensureTrainCentroidsDevice(nlist, dim);
    const size_t centroidsBytes = static_cast<size_t>(nlist) * static_cast<size_t>(dim) * sizeof(float);
    auto ret =
        aclrtMemcpy(trainCentroidsDev_->data(), centroidsBytes, centroids, centroidsBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "upload train centroids failed, size=%zu, ret=%d",
                             centroidsBytes, ret);

    std::vector<float> centroidsSqrSum(static_cast<size_t>(nlist));
    for (int i = 0; i < nlist; i++)
    {
        float normSQ = 0.0f;
        for (int j = 0; j < dim; j++)
        {
            float val = centroids[static_cast<size_t>(i) * static_cast<size_t>(dim) + static_cast<size_t>(j)];
            normSQ += val * val;
        }
        centroidsSqrSum[static_cast<size_t>(i)] = normSQ;
    }
    ret = aclrtMemcpy(trainCentroidsSqrSumDev_->data(), static_cast<size_t>(nlist) * sizeof(float),
                      centroidsSqrSum.data(), static_cast<size_t>(nlist) * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "upload train centroids sqr sum failed, ret=%d",
                             ret);
    return APP_ERR_OK;
}

void IndexIVFPQ::ensureAssignQueryBuffer(size_t elemCount)
{
    if (assignQueriesOnDevice_ && assignQueriesCapacityElems_ >= elemCount)
    {
        return;
    }
    if (!assignQueriesOnDevice_)
    {
        assignQueriesOnDevice_ = CREATE_UNIQUE_PTR(DeviceVector<float>, MemorySpace::DEVICE_HUGEPAGE);
    }
    assignQueriesOnDevice_->resize(elemCount);
    assignQueriesCapacityElems_ = elemCount;
}

void IndexIVFPQ::ensureL1Workspace(int batch, int nlist)
{
    const int maxBatch = std::max(batch, MAX_BATCH_SIZE);
    if (l1WsNumLists_ == nlist && l1WsMaxBatch_ >= maxBatch && l1DistsDev_)
    {
        return;
    }

    if (!l1DistsDev_)
    {
        l1DistsDev_ = CREATE_UNIQUE_PTR(DeviceVector<float>, MemorySpace::DEVICE_HUGEPAGE);
        l1VmdistsDev_ = CREATE_UNIQUE_PTR(DeviceVector<float>, MemorySpace::DEVICE_HUGEPAGE);
        l1TopNprobeDistsDev_ = CREATE_UNIQUE_PTR(DeviceVector<float>, MemorySpace::DEVICE_HUGEPAGE);
        l1TopNprobeIndicesDev_ = CREATE_UNIQUE_PTR(DeviceVector<int64_t>, MemorySpace::DEVICE_HUGEPAGE);
        l1OpSizeDev_ = CREATE_UNIQUE_PTR(DeviceVector<uint32_t>, MemorySpace::DEVICE_HUGEPAGE);
        l1OpFlagDev_ = CREATE_UNIQUE_PTR(DeviceVector<uint16_t>, MemorySpace::DEVICE_HUGEPAGE);
        l1AttrsDev_ = CREATE_UNIQUE_PTR(DeviceVector<int64_t>, MemorySpace::DEVICE_HUGEPAGE);
    }

    const int minDistSize = nlist / IVF_PQ_BURST_LEN * 2;
    l1DistsDev_->resize(static_cast<size_t>(maxBatch) * static_cast<size_t>(nlist));
    l1VmdistsDev_->resize(static_cast<size_t>(maxBatch) * static_cast<size_t>(minDistSize));
    l1TopNprobeDistsDev_->resize(static_cast<size_t>(maxBatch) * static_cast<size_t>(nprobe));
    l1TopNprobeIndicesDev_->resize(static_cast<size_t>(maxBatch) * static_cast<size_t>(nprobe));
    l1OpSizeDev_->resize(static_cast<size_t>(CORE_NUM) * static_cast<size_t>(SIZE_ALIGN));
    l1OpFlagDev_->resize(static_cast<size_t>(CORE_NUM) * static_cast<size_t>(FLAG_SIZE));
    l1AttrsDev_->resize(aicpu::TOPK_FLAT_ATTR_IDX_COUNT);
    l1WsNumLists_ = nlist;
    l1WsMaxBatch_ = maxBatch;
}

APP_ERROR IndexIVFPQ::initClusterTrainOps(int nlist, int dim)
{
    if (trainOpNlist_ == nlist && trainOpDim_ == dim)
    {
        return APP_ERR_OK;
    }
    auto ret = resetTrainOp(nlist, dim);
    APPERR_RETURN_IF_NOT_FMT(ret == APP_ERR_OK, ret, "Failed to reset cluster train ops, nlist=%d, dim=%d, ret=%d",
                             nlist, dim, ret);
    trainOpNlist_ = nlist;
    trainOpDim_ = dim;
    return APP_ERR_OK;
}

const float *IndexIVFPQ::getTrainCentroidsDevicePtr() const
{
    return trainCentroidsDev_ ? trainCentroidsDev_->data() : nullptr;
}

const float *IndexIVFPQ::getTrainCentroidsSqrSumDevicePtr() const
{
    return trainCentroidsSqrSumDev_ ? trainCentroidsSqrSumDev_->data() : nullptr;
}

APP_ERROR IndexIVFPQ::copyTrainCentroidsFromDevice(int nlist, int dim, const float *srcCentroidsDev,
                                                   const float *srcCentroidsSqrSumDev)
{
    APPERR_RETURN_IF_NOT_LOG(srcCentroidsDev != nullptr && srcCentroidsSqrSumDev != nullptr, APP_ERR_INVALID_PARAM,
                             "source centroid device buffers must not be null");
    ensureTrainCentroidsDevice(nlist, dim);
    const size_t centroidsBytes = static_cast<size_t>(nlist) * static_cast<size_t>(dim) * sizeof(float);
    auto ret = aclrtMemcpy(trainCentroidsDev_->data(), centroidsBytes, srcCentroidsDev, centroidsBytes,
                           ACL_MEMCPY_DEVICE_TO_DEVICE);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR,
                             "copy train centroids D2D failed, size=%zu, ret=%d", centroidsBytes, ret);
    ret = aclrtMemcpy(trainCentroidsSqrSumDev_->data(), static_cast<size_t>(nlist) * sizeof(float),
                      srcCentroidsSqrSumDev, static_cast<size_t>(nlist) * sizeof(float), ACL_MEMCPY_DEVICE_TO_DEVICE);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "copy train centroids sqrSum D2D failed, ret=%d",
                             ret);
    return APP_ERR_OK;
}

APP_ERROR IndexIVFPQ::runKMeans(int nlist, int dim, int totalSize, int iter, float *centroidsHost,
                                AscendTensor<float, DIMS_2> &dataVector, std::vector<int64_t> &totalAssigns,
                                bool useDeviceCentroids, int64_t assignOffset)
{
    FAISS_THROW_IF_NOT_MSG(useDeviceCentroids || centroidsHost, "centroidsHost can not be nullptr.");
    auto &mem = resources.getMemoryManager();
    auto streamPtr = resources.getDefaultStream();
    auto stream = streamPtr->GetStream();

    // NOTE: centroidsDev/centroidsDoubleDev are declared AFTER the unconditional
    // stack tensors below. When the coarse centroids are not resident on device
    // they get allocated from the stack memory here, so they must be the last
    // stack allocations in order to satisfy the LIFO free contract enforced by
    // AscendStackMemory::Stack::returnAlloc (destroyed first at scope exit).
    AscendTensor<uint32_t, DIMS_2> sizesDev(mem, {CORE_NUM, SIZE_ALIGN}, stream);
    AscendTensor<uint16_t, DIMS_2> flagsDev(mem, {CORE_NUM, FLAG_SIZE}, stream);
    AscendTensor<int64_t, DIMS_1> attrsDev(mem, {aicpu::TOPK_FLAT_ATTR_IDX_COUNT}, stream);
    AscendTensor<float, DIMS_2> centroidsDev;
    AscendTensor<float, DIMS_1> centroidsDoubleDev;

    auto h2dStart = std::chrono::high_resolution_clock::now();
    const bool useTrainCentroids = useDeviceCentroids && trainCentroidsDev_ && trainCentroidsSqrSumDev_ &&
                                   trainCentroidsNlist_ == nlist && trainCentroidsDim_ == dim;
    if (useTrainCentroids)
    {
        centroidsDev = AscendTensor<float, DIMS_2>(trainCentroidsDev_->data(), {nlist, dim});
        centroidsDoubleDev = AscendTensor<float, DIMS_1>(trainCentroidsSqrSumDev_->data(), {nlist});
    }
    else if (useDeviceCentroids && centroidsOnDevice && centroidsSqrSumOnDevice &&
             centroidsOnDevice->size() == static_cast<size_t>(nlist * dim) &&
             centroidsSqrSumOnDevice->size() == static_cast<size_t>(nlist))
    {
        centroidsDev = AscendTensor<float, DIMS_2>(centroidsOnDevice->data(), {nlist, dim});
        centroidsDoubleDev = AscendTensor<float, DIMS_1>(centroidsSqrSumOnDevice->data(), {nlist});
    }
    else
    {
        centroidsDev = AscendTensor<float, DIMS_2>(mem, {nlist, dim}, stream);
        centroidsDoubleDev = AscendTensor<float, DIMS_1>(mem, {nlist}, stream);
        std::vector<float> centroidsDoubleHost(nlist);
        for (int i = 0; i < nlist; i++)
        {
            float normSQ = 0.0f;
            for (int j = 0; j < dim; j++)
            {
                float val = centroidsHost[i * dim + j];
                normSQ += val * val;
            }
            centroidsDoubleHost[i] = normSQ;
        }

        auto ret = aclrtMemcpy(centroidsDev.data(), centroidsDev.getSizeInBytes(), centroidsHost,
                               centroidsDev.getSizeInBytes(), ACL_MEMCPY_HOST_TO_DEVICE);
        APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "upload centroids failed iter=%d ret=%d",
                                 iter, ret);
        ret = aclrtMemcpy(centroidsDoubleDev.data(), centroidsDoubleDev.getSizeInBytes(), centroidsDoubleHost.data(),
                          centroidsDoubleDev.getSizeInBytes(), ACL_MEMCPY_HOST_TO_DEVICE);
        APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR,
                                 "upload centroidsDouble failed iter=%d ret=%d", iter, ret);
    }

    sizesDev[0][0] = nlist;
    flagsDev.zero();
    std::vector<int64_t> attrsHost(aicpu::TOPK_FLAT_ATTR_IDX_COUNT);
    attrsHost[aicpu::TOPK_FLAT_ATTR_ASC_IDX] = 1;
    attrsHost[aicpu::TOPK_FLAT_ATTR_K_IDX] = 1;
    attrsHost[aicpu::TOPK_FLAT_ATTR_BURST_LEN_IDX] = IVF_PQ_BURST_LEN;
    attrsHost[aicpu::TOPK_FLAT_ATTR_BLOCK_NUM_IDX] = 1;
    attrsHost[aicpu::TOPK_FLAT_ATTR_PAGE_IDX] = 0;
    attrsHost[aicpu::TOPK_FLAT_ATTR_PAGE_NUM_IDX] = 1;
    attrsHost[aicpu::TOPK_FLAT_ATTR_PAGE_SIZE_IDX] = 0;
    attrsHost[aicpu::TOPK_FLAT_ATTR_QUICK_HEAP] = 1;
    attrsHost[aicpu::TOPK_FLAT_ATTR_BLOCK_SIZE] = nlist;
    auto ret = aclrtMemcpy(attrsDev.data(), attrsDev.getSizeInBytes(), attrsHost.data(),
                           attrsHost.size() * sizeof(int64_t), ACL_MEMCPY_HOST_TO_DEVICE);
    APPERR_RETURN_IF_NOT_LOG(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "Failed to copy attr to device");
    auto centroidH2dMs =
        std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - h2dStart)
            .count();

    int64_t distMs = 0;
    int64_t topkMs = 0;
    int batchCount = 0;

    ensureTrainAssignBuffer(totalSize);
    auto streamAicpuPtr = resources.getAlternateStreams()[0];
    auto streamAicpu = streamAicpuPtr->GetStream();

    int maxBatch = 1;
    for (int bs : searchBatchSizes)
    {
        maxBatch = std::max(maxBatch, bs);
    }
    ensureTrainBatchWorkspace(nlist, maxBatch);

    int processed = 0;
    int batchIdx = 0;
    int pendingTopkBuf = -1;
    while (processed < totalSize)
    {
        int remaining = totalSize - processed;
        int batchSize = 64;
        for (int bs : searchBatchSizes)
        {
            if (bs <= remaining)
            {
                batchSize = bs;
                break;
            }
        }

        const int buf = batchIdx % 2;
        if (pendingTopkBuf == buf)
        {
            auto topkSyncStart = std::chrono::high_resolution_clock::now();
            ret = synchronizeStream(streamAicpu);
            APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR,
                                     "synchronizeStream aicpu stream failed: %i\n", ret);
            topkMs += std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() -
                                                                            topkSyncStart)
                          .count();
            pendingTopkBuf = -1;
        }

        AscendTensor<float, DIMS_2> queries(dataVector.data() + processed * dim, {batchSize, dim});
        ret = execTrainBatchDist(queries, centroidsDev, centroidsDoubleDev, flagsDev, buf, &distMs);
        APPERR_RETURN_IF_NOT_FMT(ret == APP_ERR_OK, ret,
                                 "Failed to execute train batch dist, batchSize=%d, nlist=%d, dim=%d, ret=%d",
                                 batchSize, nlist, dim, ret);

        ret = execTrainBatchTopk(sizesDev, flagsDev, attrsDev, buf, batchSize, nlist, processed);
        APPERR_RETURN_IF_NOT_FMT(ret == APP_ERR_OK, ret,
                                 "Failed to execute train batch topk, batchSize=%d, nlist=%d, ret=%d", batchSize, nlist,
                                 ret);
        pendingTopkBuf = buf;

        processed += batchSize;
        batchIdx++;
        batchCount++;
    }

    if (pendingTopkBuf >= 0)
    {
        auto topkSyncStart = std::chrono::high_resolution_clock::now();
        ret = synchronizeStream(streamAicpu);
        APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "synchronizeStream aicpu stream failed: %i\n",
                                 ret);
        topkMs += std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() -
                                                                        topkSyncStart)
                      .count();
    }

    const size_t assignBytes = static_cast<size_t>(totalSize) * sizeof(int64_t);
    ret = aclrtMemcpy(totalAssigns.data() + assignOffset, assignBytes, trainAssignsDev_->data(), assignBytes,
                      ACL_MEMCPY_DEVICE_TO_HOST);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "copy train assignments to host failed: %d", ret);

    if (verbose)
    {
        APP_LOG_INFO(
            "IVFPQ runKMeans iter=%d profile: batches=%d, centroidH2d=%lld ms, dist=%lld ms, topk=%lld ms, "
            "useDeviceCentroids=%d\n",
            iter, batchCount, static_cast<long long>(centroidH2dMs), static_cast<long long>(distMs),
            static_cast<long long>(topkMs), static_cast<int>(useDeviceCentroids));
    }
    return APP_ERR_OK;
}

APP_ERROR IndexIVFPQ::trainBatchImpl(int batchSize, int nlist, int dim, int processed,
                                     AscendTensor<float, DIMS_2> &dataVector, AscendTensor<float, DIMS_2> &centroidsDev,
                                     AscendTensor<float, DIMS_1> &centroidsDoubleDev,
                                     AscendTensor<uint32_t, DIMS_2> &sizesDev, AscendTensor<uint16_t, DIMS_2> &flagsDev,
                                     AscendTensor<int64_t, DIMS_1> &attrsDev, int64_t *distMs, int64_t *topkMs)
{
    AscendTensor<float, DIMS_2> queries(dataVector.data() + processed * dim, {batchSize, dim});
    auto ret = execTrainBatch(queries, centroidsDev, centroidsDoubleDev, sizesDev, flagsDev, attrsDev, processed,
                              distMs, topkMs);
    APPERR_RETURN_IF_NOT_FMT(ret == APP_ERR_OK, ret,
                             "Failed to execute train batch, batchSize=%d, nlist=%d, dim=%d, ret=%d", batchSize, nlist,
                             dim, ret);
    return APP_ERR_OK;
}

APP_ERROR IndexIVFPQ::execTrainBatchDist(AscendTensor<float, DIMS_2> &queries, AscendTensor<float, DIMS_2> &centroids,
                                         AscendTensor<float, DIMS_1> &centroidsDouble,
                                         AscendTensor<uint16_t, DIMS_2> &flags, int bufIdx, int64_t *distMs)
{
    auto streamPtr = resources.getDefaultStream();
    auto stream = streamPtr->GetStream();
    int batchSize = queries.getSize(0);
    int nlist = centroids.getSize(0);
    ensureTrainBatchWorkspace(nlist, batchSize);
    float *distsPtr = (bufIdx == 0) ? trainBatchDistsDev_->data() : trainBatchDistsDevAlt_->data();
    float *vmdistsPtr = (bufIdx == 0) ? trainBatchVmdistsDev_->data() : trainBatchVmdistsDevAlt_->data();
    AscendTensor<float, DIMS_2> dists(distsPtr, {batchSize, nlist});
    AscendTensor<float, DIMS_2> vmdists(vmdistsPtr, {batchSize, trainMinDistCols(nlist)});

    auto distStart = std::chrono::high_resolution_clock::now();
    runTrainDistOp(batchSize, queries, centroids, centroidsDouble, dists, vmdists, flags, stream);
    auto ret = synchronizeStream(stream);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "synchronizeStream default stream: %i\n", ret);
    if (distMs != nullptr)
    {
        *distMs +=
            std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - distStart)
                .count();
    }
    return APP_ERR_OK;
}

APP_ERROR IndexIVFPQ::execTrainBatchTopk(AscendTensor<uint32_t, DIMS_2> &sizes, AscendTensor<uint16_t, DIMS_2> &flags,
                                         AscendTensor<int64_t, DIMS_1> &attrs, int bufIdx, int batchSize, int nlist,
                                         int assignOffset)
{
    auto streamAicpuPtr = resources.getAlternateStreams()[0];
    auto streamAicpu = streamAicpuPtr->GetStream();
    float *distsPtr = (bufIdx == 0) ? trainBatchDistsDev_->data() : trainBatchDistsDevAlt_->data();
    float *vmdistsPtr = (bufIdx == 0) ? trainBatchVmdistsDev_->data() : trainBatchVmdistsDevAlt_->data();
    AscendTensor<float, DIMS_2> dists(distsPtr, {batchSize, nlist});
    AscendTensor<float, DIMS_2> vmdists(vmdistsPtr, {batchSize, trainMinDistCols(nlist)});
    AscendTensor<float, DIMS_2> outdists(trainBatchOutdistsDev_->data(), {batchSize, 1});
    AscendTensor<int64_t, DIMS_2> outlabel(trainAssignsDev_->data() + assignOffset, {batchSize, 1});

    runTrainTopkOp(dists, vmdists, sizes, flags, attrs, outdists, outlabel, streamAicpu);
    return APP_ERR_OK;
}

APP_ERROR IndexIVFPQ::execTrainBatch(AscendTensor<float, DIMS_2> &queries, AscendTensor<float, DIMS_2> &centroids,
                                     AscendTensor<float, DIMS_1> &centroidsDouble,
                                     AscendTensor<uint32_t, DIMS_2> &sizes, AscendTensor<uint16_t, DIMS_2> &flags,
                                     AscendTensor<int64_t, DIMS_1> &attrs, int assignOffset, int64_t *distMs,
                                     int64_t *topkMs)
{
    auto streamAicpuPtr = resources.getAlternateStreams()[0];
    auto streamAicpu = streamAicpuPtr->GetStream();
    int batchSize = queries.getSize(0);
    int nlist = centroids.getSize(0);

    auto topkStart = std::chrono::high_resolution_clock::now();
    auto ret = execTrainBatchDist(queries, centroids, centroidsDouble, flags, 0, distMs);
    APPERR_RETURN_IF_NOT_FMT(ret == APP_ERR_OK, ret, "execTrainBatchDist failed, ret=%d", ret);
    ret = execTrainBatchTopk(sizes, flags, attrs, 0, batchSize, nlist, assignOffset);
    APPERR_RETURN_IF_NOT_FMT(ret == APP_ERR_OK, ret, "execTrainBatchTopk failed, ret=%d", ret);
    ret = synchronizeStream(streamAicpu);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "synchronizeStream aicpu stream failed: %i\n",
                             ret);
    if (topkMs != nullptr)
    {
        *topkMs +=
            std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - topkStart)
                .count();
    }
    return APP_ERR_OK;
}

APP_ERROR IndexIVFPQ::updateCentroids(int nlist, int dim, int totalSize, const std::vector<int64_t> &totalAssigns,
                                      const std::vector<float> &trainData, std::vector<float> &centroids)
{
    auto ret = memset_s(centroids.data(), dim * nlist * sizeof(float), 0, dim * nlist * sizeof(float));
    ASCEND_THROW_IF_NOT_FMT(ret == EOK, "set centroids to 0 failed %d", ret);
    std::vector<int32_t> hassign(nlist);
#pragma omp parallel
    {
        int nt = omp_get_num_threads();   // total num threads
        int rank = omp_get_thread_num();  // get thread ID
        int64_t c0 = (nlist * rank) / nt;
        int64_t c1 = (nlist * (rank + 1)) / nt;
        for (int i = 0; i < totalSize; i++)
        {
            ASCEND_THROW_IF_NOT_FMT(totalAssigns[i] >= 0, "assign[%d] %jd < 0 is invalid", i, totalAssigns[i]);
            int ci = totalAssigns[i];
            ASCEND_THROW_IF_NOT_FMT(ci < nlist, "assign %d >= nlist is invalid.", ci);
            if (ci >= c0 && ci < c1)
            {
                float *c = centroids.data() + ci * dim;
                const float *xi = trainData.data() + i * dim;
                hassign[ci] += 1;
                for (int j = 0; j < dim; j++)
                {
                    c[j] += xi[j];
                }
            }
        }
    }
#pragma omp parallel for
    for (int ci = 0; ci < nlist; ci++)
    {
        if (hassign[ci] == 0)
        {
            continue;
        }
        float norm = 1 / static_cast<float>(hassign[ci]);
        float *c = centroids.data() + ci * dim;
        for (int j = 0; j < dim; j++)
        {
            c[j] *= norm;
        }
    }

    return APP_ERR_OK;
}

APP_ERROR IndexIVFPQ::updateCentroidsToDevice(int nlist, int dim, const std::vector<float> &centroids)
{
    if (!clusteringOnDevice)
    {
        clusteringOnDevice = std::make_unique<DeviceVector<float>>();
    }

    if (clusteringOnDevice->size() != static_cast<size_t>(nlist * dim))
    {
        clusteringOnDevice->resize(nlist * dim);
    }

    size_t centroidsBytes = nlist * dim * sizeof(float);
    auto ret = aclrtMemcpy(clusteringOnDevice->data(), centroidsBytes, centroids.data(), centroidsBytes,
                           ACL_MEMCPY_HOST_TO_DEVICE);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR,
                             "update centroids result to Device failed, size=%zu, ret=%d", centroidsBytes, ret);

    return APP_ERR_OK;
}

APP_ERROR IndexIVFPQ::assignCentroidFast(AscendTensor<float, DIMS_2> &queries, std::vector<int64_t> &assignments)
{
    const int totalSize = queries.getSize(0);
    APPERR_RETURN_IF_NOT_FMT(totalSize > 0 && assignments.size() >= static_cast<size_t>(totalSize),
                             APP_ERR_INVALID_PARAM, "invalid fast assignment output size=%zu, totalSize=%d",
                             assignments.size(), totalSize);

    auto streamPtr = resources.getDefaultStream();
    auto stream = streamPtr->GetStream();
    auto streamAicpuPtr = resources.getAlternateStreams()[0];
    auto streamAicpu = streamAicpuPtr->GetStream();

    const int maxBatch = searchBatchSizes.empty() ? 1 : searchBatchSizes.front();
    ensureL1Workspace(maxBatch, numLists);
    ensureTrainAssignBuffer(totalSize);
    if (!assignDistsDevAlt_)
    {
        assignDistsDevAlt_ = CREATE_UNIQUE_PTR(DeviceVector<float>, MemorySpace::DEVICE_HUGEPAGE);
        assignVmdistsDevAlt_ = CREATE_UNIQUE_PTR(DeviceVector<float>, MemorySpace::DEVICE_HUGEPAGE);
        assignOpFlagDevAlt_ = CREATE_UNIQUE_PTR(DeviceVector<uint16_t>, MemorySpace::DEVICE_HUGEPAGE);
        assignOutdistsDev_ = CREATE_UNIQUE_PTR(DeviceVector<float>, MemorySpace::DEVICE_HUGEPAGE);
    }
    const int minDistSize = numLists / IVF_PQ_BURST_LEN * 2;
    assignDistsDevAlt_->resize(static_cast<size_t>(maxBatch) * numLists);
    assignVmdistsDevAlt_->resize(static_cast<size_t>(maxBatch) * minDistSize);
    assignOpFlagDevAlt_->resize(static_cast<size_t>(CORE_NUM) * FLAG_SIZE);
    // Double-buffer k=1 topk distance outputs independently of search nprobe.
    assignOutdistsDev_->resize(static_cast<size_t>(maxBatch) * 2);

    AscendTensor<uint32_t, DIMS_2> opSize(l1OpSizeDev_->data(), {CORE_NUM, SIZE_ALIGN});
    opSize[0][0] = numLists;
    AscendTensor<int64_t, DIMS_1> attrsInput(l1AttrsDev_->data(), {aicpu::TOPK_FLAT_ATTR_IDX_COUNT});
    std::vector<int64_t> attrs(aicpu::TOPK_FLAT_ATTR_IDX_COUNT);
    attrs[aicpu::TOPK_FLAT_ATTR_ASC_IDX] = 1;
    attrs[aicpu::TOPK_FLAT_ATTR_K_IDX] = 1;
    attrs[aicpu::TOPK_FLAT_ATTR_BURST_LEN_IDX] = IVF_PQ_BURST_LEN;
    attrs[aicpu::TOPK_FLAT_ATTR_BLOCK_NUM_IDX] = 1;
    attrs[aicpu::TOPK_FLAT_ATTR_PAGE_IDX] = 0;
    attrs[aicpu::TOPK_FLAT_ATTR_PAGE_NUM_IDX] = 1;
    attrs[aicpu::TOPK_FLAT_ATTR_PAGE_SIZE_IDX] = 0;
    attrs[aicpu::TOPK_FLAT_ATTR_QUICK_HEAP] = 1;
    attrs[aicpu::TOPK_FLAT_ATTR_BLOCK_SIZE] = numLists;
    auto ret = aclrtMemcpy(attrsInput.data(), attrsInput.getSizeInBytes(), attrs.data(), attrs.size() * sizeof(int64_t),
                           ACL_MEMCPY_HOST_TO_DEVICE);
    APPERR_RETURN_IF_NOT_LOG(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "Failed to copy fast assignment attrs to device");

    AscendTensor<float, DIMS_2> centroidsDev(centroidsOnDevice->data(), {numLists, dims});
    int processed = 0;
    int batchCount = 0;
    bool pendingTopk[2] = {false, false};
    const auto computeStart = std::chrono::high_resolution_clock::now();
    while (processed < totalSize)
    {
        const int remaining = totalSize - processed;
        int batchSize = 1;
        for (int candidate : searchBatchSizes)
        {
            if (candidate <= remaining)
            {
                batchSize = candidate;
                break;
            }
        }

        const int bufferIndex = batchCount % 2;
        if (pendingTopk[bufferIndex])
        {
            ret = synchronizeStream(streamAicpu);
            APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR,
                                     "fast assignment pipelined aicpu sync failed: %d", ret);
            pendingTopk[0] = false;
            pendingTopk[1] = false;
        }

        AscendTensor<float, DIMS_2> batchQueries(queries.data() + static_cast<size_t>(processed) * dims,
                                                 {batchSize, dims});
        float *distsPtr = bufferIndex == 0 ? l1DistsDev_->data() : assignDistsDevAlt_->data();
        float *vmdistsPtr = bufferIndex == 0 ? l1VmdistsDev_->data() : assignVmdistsDevAlt_->data();
        uint16_t *flagPtr = bufferIndex == 0 ? l1OpFlagDev_->data() : assignOpFlagDevAlt_->data();
        AscendTensor<float, DIMS_2> dists(distsPtr, {batchSize, numLists});
        AscendTensor<float, DIMS_2> vmdists(vmdistsPtr, {batchSize, minDistSize});
        AscendTensor<uint16_t, DIMS_2> opFlag(flagPtr, {CORE_NUM, FLAG_SIZE});
        AscendTensor<float, DIMS_2> outdists(assignOutdistsDev_->data() + bufferIndex * maxBatch, {batchSize, 1});
        AscendTensor<int64_t, DIMS_2> outlabels(trainAssignsDev_->data() + processed, {batchSize, 1});

        // Launch TopK first (waits on op flags), then Dist (produces flags), matching
        // searchImplL1. Dist ops are serialized on the default stream, so we only sync
        // when reclaiming a double-buffer slot / at the end. This overlaps Dist(N+1)
        // with TopK(N).
        opFlag.zero();
        runL1TopkOp(dists, vmdists, opSize, opFlag, attrsInput, outdists, outlabels, streamAicpu);
        runL1DistOp(batchSize, batchQueries, centroidsDev, dists, vmdists, opFlag, stream);
        pendingTopk[bufferIndex] = true;
        processed += batchSize;
        batchCount++;
    }
    ret = synchronizeStream(stream);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR,
                             "fast assignment final default stream sync failed: %d", ret);
    if (pendingTopk[0] || pendingTopk[1])
    {
        ret = synchronizeStream(streamAicpu);
        APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR,
                                 "fast assignment final aicpu stream sync failed: %d", ret);
    }
    const auto computeMs =
        std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - computeStart)
            .count();

    const auto d2hStart = std::chrono::high_resolution_clock::now();
    const size_t assignBytes = static_cast<size_t>(totalSize) * sizeof(int64_t);
    ret =
        aclrtMemcpy(assignments.data(), assignBytes, trainAssignsDev_->data(), assignBytes, ACL_MEMCPY_DEVICE_TO_HOST);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "copy fast assignments to host failed: %d", ret);
    const auto d2hMs =
        std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - d2hStart)
            .count();
    if (verbose)
    {
        APP_LOG_INFO("IVFPQ fast assign profile: vectors=%d, batches=%d, compute=%lld ms, d2h=%lld ms\n", totalSize,
                     batchCount, static_cast<long long>(computeMs), static_cast<long long>(d2hMs));
    }
    return APP_ERR_OK;
}

APP_ERROR IndexIVFPQ::assignCentroid(int nlist, int dim, int totalSize, std::vector<float> &centroids,
                                     float *trainDataHost, std::vector<int64_t> &totalAssigns, bool useFastPath)
{
    const size_t elemCount = static_cast<size_t>(totalSize) * static_cast<size_t>(dim);
    ensureAssignQueryBuffer(elemCount);
    const size_t dataBytes = elemCount * sizeof(float);
    const auto h2dStart = std::chrono::high_resolution_clock::now();
    auto ret = aclrtMemcpyChunked(assignQueriesOnDevice_->data(), dataBytes, trainDataHost, dataBytes,
                                  ACL_MEMCPY_HOST_TO_DEVICE);
    APPERR_RETURN_IF_NOT_LOG(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "Failed to copy data to device");
    const auto h2dMs =
        std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - h2dStart)
            .count();

    AscendTensor<float, DIMS_2> dataTensor(assignQueriesOnDevice_->data(), {totalSize, dim});
    const bool useDeviceCentroids = centroidsOnDevice && centroidsSqrSumOnDevice && nlist == numLists && dim == dims &&
                                    centroidsOnDevice->size() == static_cast<size_t>(nlist * dim) &&
                                    centroidsSqrSumOnDevice->size() == static_cast<size_t>(nlist);
    if (useFastPath && useDeviceCentroids)
    {
        ret = assignCentroidFast(dataTensor, totalAssigns);
        APPERR_RETURN_IF_NOT_FMT(ret == APP_ERR_OK, ret,
                                 "Failed fast centroid assignment, nlist=%d, dim=%d, totalSize=%d, ret=%d", nlist, dim,
                                 totalSize, ret);
        if (verbose)
        {
            APP_LOG_INFO("IVFPQ assignCentroid: path=fast, vectors=%d, h2d=%lld ms\n", totalSize,
                         static_cast<long long>(h2dMs));
        }
        return APP_ERR_OK;
    }

    if (trainOpNlist_ != nlist || trainOpDim_ != dim)
    {
        ret = initClusterTrainOps(nlist, dim);
        APPERR_RETURN_IF_NOT_FMT(ret == APP_ERR_OK, ret,
                                 "Failed to reset assignCentroid operations, nlist=%d, dim=%d, ret=%d", nlist, dim,
                                 ret);
    }
    ret = runKMeans(nlist, dim, totalSize, 0, centroids.data(), dataTensor, totalAssigns, useDeviceCentroids);
    APPERR_RETURN_IF_NOT_FMT(ret == APP_ERR_OK, ret, "Failed to run KMeans, nlist=%d, dim=%d, totalSize=%d, ret=%d",
                             nlist, dim, totalSize, ret);
    if (verbose)
    {
        APP_LOG_INFO("IVFPQ assignCentroid: path=legacy, vectors=%d, h2d=%lld ms\n", totalSize,
                     static_cast<long long>(h2dMs));
    }
    return APP_ERR_OK;
}

void IndexIVFPQ::resetTrainSession()
{
    trainOpNlist_ = -1;
    trainOpDim_ = -1;
    trainWsNlist_ = -1;
    trainWsMaxBatch_ = 0;
    trainAssignsCapacity_ = 0;
    trainCentroidsNlist_ = -1;
    trainCentroidsDim_ = -1;
    assignQueriesCapacityElems_ = 0;
    trainDistOps.clear();
    trainTopkOps.clear();
    trainBatchDistsDev_.reset();
    trainBatchVmdistsDev_.reset();
    trainBatchDistsDevAlt_.reset();
    trainBatchVmdistsDevAlt_.reset();
    trainBatchOutdistsDev_.reset();
    trainBatchOutlabelDev_.reset();
    trainAssignsDev_.reset();
    trainCentroidsDev_.reset();
    trainCentroidsSqrSumDev_.reset();
    assignQueriesOnDevice_.reset();
    assignDistsDevAlt_.reset();
    assignVmdistsDevAlt_.reset();
    assignOpFlagDevAlt_.reset();
    assignOutdistsDev_.reset();
}

APP_ERROR IndexIVFPQ::trainImpl(int n, const float *x, int dim, int nlist, int niter, int seed, bool skipSample)
{
    auto train_start = std::chrono::high_resolution_clock::now();
    auto &mem = resources.getMemoryManager();
    auto streamPtr = resources.getDefaultStream();
    auto stream = streamPtr->GetStream();
    int totalSize = skipSample ? n : computeTrainTotalSize(nlist, n);
    if (niter <= 0)
    {
        niter = MAX_ITER_NUM;
    }
    if (verbose)
    {
        APP_LOG_INFO("IVFPQ trainImpl start: n=%d, dim=%d, nlist=%d, totalSize=%d, niter=%d, skipSample=%d\n", n, dim,
                     nlist, totalSize, niter, static_cast<int>(skipSample));
    }
    if (verbose && !skipSample && totalSize < n)
    {
        APP_LOG_INFO("IVFPQ Training data capped: input n=%d -> totalSize=%d (per_list=%d, max=%d)\n", n, totalSize,
                     TRAIN_SAMPLES_PER_LIST, MAX_TRAIN_SAMPLES);
    }
    const size_t totalSizeSz = static_cast<size_t>(totalSize);
    const size_t dimSz = static_cast<size_t>(dim);
    const size_t nlistSz = static_cast<size_t>(nlist);
    std::vector<float> trainData;
    if (skipSample)
    {
        trainData.assign(x, x + totalSizeSz * dimSz);
    }
    else
    {
        sampleTrainData(x, n, dim, totalSize, seed, trainData);
    }
    std::vector<float> centroids(nlistSz * dimSz);
    auto ret = initTraining(totalSize, dim, nlist, trainData, centroids, seed);
    APPERR_RETURN_IF_NOT_FMT(ret == APP_ERR_OK, ret, "Failed to initialize training, n=%d, dim=%d, nlist=%d, ret=%d", n,
                             dim, nlist, ret);

    ret = initClusterTrainOps(nlist, dim);
    APPERR_RETURN_IF_NOT_FMT(ret == APP_ERR_OK, ret, "Failed to reset training operations, nlist=%d, dim=%d, ret=%d",
                             nlist, dim, ret);
    const size_t dataBytes = totalSizeSz * dimSz * sizeof(float);
    AscendTensor<float, DIMS_2> dataVector(mem, {totalSize, dim}, stream);
    ret = aclrtMemcpyChunked(dataVector.data(), dataBytes, trainData.data(), dataBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    APPERR_RETURN_IF_NOT_LOG(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "Failed to upload data to device");

    std::vector<int64_t> totalAssigns(totalSize);
    bool useDeviceCentroids = false;
    for (int iter = 0; iter < niter; iter++)
    {
        if (verbose)
        {
            APP_LOG_INFO("IVFPQ trainImpl iter %d/%d: runKMeans start\n", iter + 1, niter);
        }
        auto iter_start = std::chrono::high_resolution_clock::now();
        ret = runKMeans(nlist, dim, totalSize, iter, centroids.data(), dataVector, totalAssigns, useDeviceCentroids);
        APPERR_RETURN_IF_NOT_FMT(ret == APP_ERR_OK, ret,
                                 "Failed to run KMeans, nlist=%d, dim=%d, totalSize=%d, iter=%d, ret=%d", nlist, dim,
                                 totalSize, iter, ret);

        ret = updateCentroids(nlist, dim, totalSize, totalAssigns, trainData, centroids);
        APPERR_RETURN_IF_NOT_FMT(ret == APP_ERR_OK, ret,
                                 "Failed to update centroids, nlist=%d, dim=%d, totalSize=%d, ret=%d", nlist, dim,
                                 totalSize, ret);
        normL2(dim, nlist, centroids.data());
        ret = syncTrainCentroidsToDevice(nlist, dim, centroids);
        APPERR_RETURN_IF_NOT_FMT(ret == APP_ERR_OK, ret,
                                 "Failed to sync train centroids to device, nlist=%d, dim=%d, ret=%d", nlist, dim, ret);
        useDeviceCentroids = true;
        auto iter_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                                std::chrono::high_resolution_clock::now() - iter_start)
                                .count();
        if (verbose)
        {
            APP_LOG_INFO("IVFPQ trainImpl iter %d/%d: runKMeans done, updateCentroids done, elapsed=%lld ms\n",
                         iter + 1, niter, static_cast<long long>(iter_elapsed));
        }
    }
    ret = updateCentroidsToDevice(nlist, dim, centroids);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR,
                             "update centroids result to Device failed, ret=%d", ret);
    auto total_elapsed =
        std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - train_start)
            .count();
    if (verbose)
    {
        APP_LOG_INFO("IVFPQ trainImpl finished: nlist=%d, dim=%d, totalSize=%d, totalElapsed=%lld ms\n", nlist, dim,
                     totalSize, static_cast<long long>(total_elapsed));
    }
    return APP_ERR_OK;
}

APP_ERROR IndexIVFPQ::resetTrainDistOp(int nlist, int dim)
{
    auto trainDisOpReset = [&](std::unique_ptr<AscendOperator> &op, int batch)
    {
        AscendOpDesc desc("DistanceFlatL2MinsAtFP32");

        std::vector<int64_t> queryShape({batch, dim});
        std::vector<int64_t> codesShape({nlist, dim});
        std::vector<int64_t> codesDoubleShape({nlist});
        std::vector<int64_t> distResultShape({batch, nlist});
        std::vector<int64_t> minResultShape({batch, trainMinDistCols(nlist)});
        std::vector<int64_t> flagShape({CORE_NUM, 16});

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

    for (auto batch : searchBatchSizes)
    {
        trainDistOps[batch] = std::unique_ptr<AscendOperator>(nullptr);
        APPERR_RETURN_IF_NOT_LOG(trainDisOpReset(trainDistOps[batch], batch), APP_ERR_ACL_OP_LOAD_MODEL_FAILED,
                                 "train dist op init failed");
    }
    return APP_ERR_OK;
}

void IndexIVFPQ::runTrainDistOp(int batch, AscendTensor<float, DIMS_2> &queries, AscendTensor<float, DIMS_2> &centroids,
                                AscendTensor<float, DIMS_1> &codesDouble, AscendTensor<float, DIMS_2> &distances,
                                AscendTensor<float, DIMS_2> &vmdists, AscendTensor<uint16_t, DIMS_2> &opFlag,
                                aclrtStream stream)
{
    AscendOperator *op = nullptr;
    if (trainDistOps.find(batch) != trainDistOps.end())
    {
        op = trainDistOps[batch].get();
    }
    ASCEND_THROW_IF_NOT(op);

    std::shared_ptr<std::vector<const aclDataBuffer *>> inputs(new std::vector<const aclDataBuffer *>(),
                                                               CommonUtils::AclInputBufferDelete);

    inputs->emplace_back(aclCreateDataBuffer(queries.data(), queries.getSizeInBytes()));
    inputs->emplace_back(aclCreateDataBuffer(centroids.data(), centroids.getSizeInBytes()));
    inputs->emplace_back(aclCreateDataBuffer(codesDouble.data(), codesDouble.getSizeInBytes()));

    std::shared_ptr<std::vector<aclDataBuffer *>> outputs(new std::vector<aclDataBuffer *>(),
                                                          CommonUtils::AclOutputBufferDelete);

    outputs->emplace_back(aclCreateDataBuffer(distances.data(), distances.getSizeInBytes()));
    outputs->emplace_back(aclCreateDataBuffer(vmdists.data(), vmdists.getSizeInBytes()));
    outputs->emplace_back(aclCreateDataBuffer(opFlag.data(), opFlag.getSizeInBytes()));

    op->exec(*inputs, *outputs, stream);
}

APP_ERROR IndexIVFPQ::resetTrainTopkOp(int nlist)
{
    auto topkTrainOpReset = [&](std::unique_ptr<AscendOperator> &op, int64_t batch)
    {
        AscendOpDesc desc("TopkFlatFp32");
        std::vector<int64_t> shape0{1, batch, nlist};
        std::vector<int64_t> shape1{1, batch, trainMinDistCols(nlist)};
        std::vector<int64_t> shape2{1, CORE_NUM, SIZE_ALIGN};
        std::vector<int64_t> shape3{1, CORE_NUM, FLAG_SIZE};
        std::vector<int64_t> shape4{aicpu::TOPK_FLAT_ATTR_IDX_COUNT};
        std::vector<int64_t> shape5{batch, 0};

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

    for (auto batch : searchBatchSizes)
    {
        trainTopkOps[batch] = std::unique_ptr<AscendOperator>(nullptr);
        APPERR_RETURN_IF_NOT_LOG(topkTrainOpReset(trainTopkOps[batch], batch), APP_ERR_ACL_OP_LOAD_MODEL_FAILED,
                                 "train topk op init failed");
    }
    return APP_ERR_OK;
}

void IndexIVFPQ::runTrainTopkOp(AscendTensor<float, DIMS_2> &dists, AscendTensor<float, DIMS_2> &vmdists,
                                AscendTensor<uint32_t, DIMS_2> &sizes, AscendTensor<uint16_t, DIMS_2> &flags,
                                AscendTensor<int64_t, DIMS_1> &attrs, AscendTensor<float, DIMS_2> &outdists,
                                AscendTensor<int64_t, DIMS_2> &outlabel, aclrtStream stream)
{
    AscendOperator *op = nullptr;
    int batch = dists.getSize(0);
    int nlist = dists.getSize(1);
    int vmdists_cols = vmdists.getSize(1);
    if (trainTopkOps.find(batch) != trainTopkOps.end())
    {
        op = trainTopkOps[batch].get();
    }
    ASCEND_THROW_IF_NOT(op);

    AscendTensor<float, DIMS_3> distsTopk(dists.data(), {1, batch, nlist});
    AscendTensor<float, DIMS_3> vmdistsTopk(vmdists.data(), {1, batch, vmdists_cols});
    AscendTensor<uint32_t, DIMS_3> sizesTopk(sizes.data(), {1, CORE_NUM, SIZE_ALIGN});
    AscendTensor<uint16_t, DIMS_3> flagsTopk(flags.data(), {1, CORE_NUM, FLAG_SIZE});

    std::shared_ptr<std::vector<const aclDataBuffer *>> inputs(new std::vector<const aclDataBuffer *>(),
                                                               CommonUtils::AclInputBufferDelete);

    inputs->emplace_back(aclCreateDataBuffer(distsTopk.data(), distsTopk.getSizeInBytes()));
    inputs->emplace_back(aclCreateDataBuffer(vmdistsTopk.data(), vmdistsTopk.getSizeInBytes()));
    inputs->emplace_back(aclCreateDataBuffer(sizesTopk.data(), sizesTopk.getSizeInBytes()));
    inputs->emplace_back(aclCreateDataBuffer(flagsTopk.data(), flagsTopk.getSizeInBytes()));
    inputs->emplace_back(aclCreateDataBuffer(attrs.data(), attrs.getSizeInBytes()));

    std::shared_ptr<std::vector<aclDataBuffer *>> outputs(new std::vector<aclDataBuffer *>(),
                                                          CommonUtils::AclOutputBufferDelete);

    outputs->emplace_back(aclCreateDataBuffer(outdists.data(), outdists.getSizeInBytes()));
    outputs->emplace_back(aclCreateDataBuffer(outlabel.data(), outlabel.getSizeInBytes()));

    op->exec(*inputs, *outputs, stream);
}

APP_ERROR IndexIVFPQ::resetL1TopkOp()
{
    auto topkCompOpReset = [&](std::unique_ptr<AscendOperator> &op, int64_t batch)
    {
        AscendOpDesc desc("TopkFlatFp32");
        std::vector<int64_t> shape0{1, batch, numLists};
        std::vector<int64_t> shape1{1, batch, numLists / IVF_PQ_BURST_LEN * 2};
        std::vector<int64_t> shape2{1, CORE_NUM, SIZE_ALIGN};
        std::vector<int64_t> shape3{1, CORE_NUM, FLAG_SIZE};
        std::vector<int64_t> shape4{aicpu::TOPK_FLAT_ATTR_IDX_COUNT};
        std::vector<int64_t> shape5{batch, 0};

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

    for (auto batch : searchBatchSizes)
    {
        topkFp32[batch] = std::unique_ptr<AscendOperator>(nullptr);
        APPERR_RETURN_IF_NOT_LOG(topkCompOpReset(topkFp32[batch], batch), APP_ERR_ACL_OP_LOAD_MODEL_FAILED,
                                 "topk op init failed");
    }
    return APP_ERR_OK;
}

void IndexIVFPQ::runL1TopkOp(AscendTensor<float, DIMS_2> &dists, AscendTensor<float, DIMS_2> &vmdists,
                             AscendTensor<uint32_t, DIMS_2> &sizes, AscendTensor<uint16_t, DIMS_2> &flags,
                             AscendTensor<int64_t, DIMS_1> &attrs, AscendTensor<float, DIMS_2> &outdists,
                             AscendTensor<int64_t, DIMS_2> &outlabel, aclrtStream stream)
{
    AscendOperator *op = nullptr;
    int batch = dists.getSize(0);
    if (topkFp32.find(batch) != topkFp32.end())
    {
        op = topkFp32[batch].get();
    }
    ASCEND_THROW_IF_NOT(op);
    AscendTensor<float, DIMS_3> distsTopk(dists.data(), {1, batch, numLists});
    AscendTensor<float, DIMS_3> vmdistsTopk(vmdists.data(), {1, batch, numLists / IVF_PQ_BURST_LEN * 2});
    AscendTensor<uint32_t, DIMS_3> sizesTopk(sizes.data(), {1, CORE_NUM, SIZE_ALIGN});
    AscendTensor<uint16_t, DIMS_3> flagsTopk(flags.data(), {1, CORE_NUM, FLAG_SIZE});
    std::shared_ptr<std::vector<const aclDataBuffer *>> topkOpInput(new std::vector<const aclDataBuffer *>(),
                                                                    CommonUtils::AclInputBufferDelete);
    topkOpInput->emplace_back(aclCreateDataBuffer(distsTopk.data(), distsTopk.getSizeInBytes()));
    topkOpInput->emplace_back(aclCreateDataBuffer(vmdistsTopk.data(), vmdistsTopk.getSizeInBytes()));
    topkOpInput->emplace_back(aclCreateDataBuffer(sizesTopk.data(), sizesTopk.getSizeInBytes()));
    topkOpInput->emplace_back(aclCreateDataBuffer(flagsTopk.data(), flagsTopk.getSizeInBytes()));
    topkOpInput->emplace_back(aclCreateDataBuffer(attrs.data(), attrs.getSizeInBytes()));

    std::shared_ptr<std::vector<aclDataBuffer *>> topkOpOutput(new std::vector<aclDataBuffer *>(),
                                                               CommonUtils::AclOutputBufferDelete);
    topkOpOutput->emplace_back(aclCreateDataBuffer(outdists.data(), outdists.getSizeInBytes()));
    topkOpOutput->emplace_back(aclCreateDataBuffer(outlabel.data(), outlabel.getSizeInBytes()));

    op->exec(*topkOpInput, *topkOpOutput, stream);
}

APP_ERROR IndexIVFPQ::resetL1DistOp()
{
    auto l1DisOpReset = [&](std::unique_ptr<AscendOperator> &op, int batch)
    {
        AscendOpDesc desc("DistanceFlatL2MinsAtFP32");
        std::vector<int64_t> queryShape({batch, dims});
        std::vector<int64_t> codesShape({numLists, dims});
        std::vector<int64_t> codesSqrSum({numLists});

        std::vector<int64_t> distResultShape({batch, numLists});
        std::vector<int64_t> minResultShape({batch, numLists / IVF_PQ_BURST_LEN * 2});
        std::vector<int64_t> flagShapeShape({CORE_NUM, 16});

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
    for (auto batch : searchBatchSizes)
    {
        l1DistFp32Ops[batch] = std::unique_ptr<AscendOperator>(nullptr);
        APPERR_RETURN_IF_NOT_LOG(l1DisOpReset(l1DistFp32Ops[batch], batch), APP_ERR_ACL_OP_LOAD_MODEL_FAILED,
                                 "L1 distance op init failed");
    }
    return APP_ERR_OK;
}

APP_ERROR IndexIVFPQ::ensureL2L3SearchOps()
{
    if (!l2DistFp32Ops.empty())
    {
        return APP_ERR_OK;
    }
    auto ret = resetL2DistOp();
    APPERR_RETURN_IF_NOT_FMT(ret == APP_ERR_OK, ret, "resetL2DistOp failed %d", ret);
    ret = resetL3TopkOp();
    APPERR_RETURN_IF_NOT_FMT(ret == APP_ERR_OK, ret, "resetL3TopkOp failed %d", ret);
    ret = resetL3DistOp();
    APPERR_RETURN_IF_NOT_FMT(ret == APP_ERR_OK, ret, "resetL3DistOp failed %d", ret);
    return APP_ERR_OK;
}

APP_ERROR IndexIVFPQ::resetL2DistOp()
{
    auto l2DisOpReset = [&](std::unique_ptr<AscendOperator> &op, int batch)
    {
        AscendOpDesc desc("AscendcIvfpqSubspaceDistance");
        std::vector<int64_t> queryShape({batch, dims});
        std::vector<int64_t> codeBookShape({M, ksub, dsub});
        std::vector<int64_t> distResultShape({batch, M, ksub});

        desc.addInputTensorDesc(ACL_FLOAT, queryShape.size(), queryShape.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_FLOAT, codeBookShape.size(), codeBookShape.data(), ACL_FORMAT_ND);
        desc.addOutputTensorDesc(ACL_FLOAT, distResultShape.size(), distResultShape.data(), ACL_FORMAT_ND);
        op.reset();
        op = CREATE_UNIQUE_PTR(AscendOperator, desc);
        return op->init();
    };
    for (auto batch : searchBatchSizes)
    {
        l2DistFp32Ops[batch] = std::unique_ptr<AscendOperator>(nullptr);
        APPERR_RETURN_IF_NOT_LOG(l2DisOpReset(l2DistFp32Ops[batch], batch), APP_ERR_ACL_OP_LOAD_MODEL_FAILED,
                                 "L2 distance op init failed");
    }
    return APP_ERR_OK;
}

APP_ERROR IndexIVFPQ::resetL3DistOp()
{
    auto l3DistOpReset = [&](std::unique_ptr<AscendOperator> &op, int batch)
    {
        const char *opName = faiss::ascend::SocUtils::GetInstance().IsAscendA5() ? "AscendcIvfpqSearchDistanceL2Simt"
                                                                                 : "AscendcIvfpqSearchDistanceL2";
        AscendOpDesc desc(opName);
        std::vector<int64_t> queryPQShape({batch, M, ksub});
        std::vector<int64_t> codeBaseShape({M});
        std::vector<int64_t> codeOffsetShape({batch, blockNum});
        std::vector<int64_t> codeSizeShape({batch, blockNum});
        std::vector<int64_t> topk({MAX_TOPK});
        std::vector<int64_t> labelBaseShape({M});
        std::vector<int64_t> labelOffsetShape({batch, blockNum});

        std::vector<int64_t> distResultShape({batch, blockNum, blockSize});
        std::vector<int64_t> topkIndex({batch, blockNum, MAX_TOPK});
        std::vector<int64_t> topkValue({batch, blockNum, MAX_TOPK});
        std::vector<int64_t> topkIndexFinalShape({batch, MAX_TOPK});
        std::vector<int64_t> topkValueFinalShape({batch, MAX_TOPK});
        std::vector<int64_t> flagShape({16});

        desc.addInputTensorDesc(ACL_FLOAT, queryPQShape.size(), queryPQShape.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_UINT8, codeBaseShape.size(), codeBaseShape.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_INT64, codeOffsetShape.size(), codeOffsetShape.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_INT64, codeSizeShape.size(), codeSizeShape.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_INT32, topk.size(), topk.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_UINT64, labelBaseShape.size(), labelBaseShape.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_INT64, labelOffsetShape.size(), labelOffsetShape.data(), ACL_FORMAT_ND);

        desc.addOutputTensorDesc(ACL_FLOAT, distResultShape.size(), distResultShape.data(), ACL_FORMAT_ND);
        desc.addOutputTensorDesc(ACL_INT32, topkIndex.size(), topkIndex.data(), ACL_FORMAT_ND);
        desc.addOutputTensorDesc(ACL_FLOAT, topkValue.size(), topkValue.data(), ACL_FORMAT_ND);
        desc.addOutputTensorDesc(ACL_UINT64, topkIndexFinalShape.size(), topkIndexFinalShape.data(), ACL_FORMAT_ND);
        desc.addOutputTensorDesc(ACL_FLOAT, topkValueFinalShape.size(), topkValueFinalShape.data(), ACL_FORMAT_ND);
        desc.addOutputTensorDesc(ACL_UINT16, flagShape.size(), flagShape.data(), ACL_FORMAT_ND);

        op.reset();
        op = CREATE_UNIQUE_PTR(AscendOperator, desc);
        return op->init();
    };

    for (auto batch : searchBatchSizes)
    {
        l3DistFp32Ops[batch] = std::unique_ptr<AscendOperator>(nullptr);
        APPERR_RETURN_IF_NOT_LOG(l3DistOpReset(l3DistFp32Ops[batch], batch), APP_ERR_ACL_OP_LOAD_MODEL_FAILED,
                                 "L3 distance op init failed");
    }
    return APP_ERR_OK;
}

void IndexIVFPQ::runL1DistOp(int batch, AscendTensor<float, DIMS_2> &queries, AscendTensor<float, DIMS_2> &centroidsDev,
                             AscendTensor<float, DIMS_2> &dists, AscendTensor<float, DIMS_2> &vmdists,
                             AscendTensor<uint16_t, DIMS_2> &opFlag, aclrtStream stream)
{
    AscendOperator *op = nullptr;
    if (l1DistFp32Ops.find(batch) != l1DistFp32Ops.end())
    {
        op = l1DistFp32Ops[batch].get();
    }
    ASCEND_THROW_IF_NOT(op);

    AscendTensor<float, DIMS_1> centroidsSqrSum(centroidsSqrSumOnDevice->data(), {numLists});
    std::shared_ptr<std::vector<const aclDataBuffer *>> topkOpInput(new std::vector<const aclDataBuffer *>(),
                                                                    CommonUtils::AclInputBufferDelete);
    topkOpInput->emplace_back(aclCreateDataBuffer(queries.data(), queries.getSizeInBytes()));
    topkOpInput->emplace_back(aclCreateDataBuffer(centroidsDev.data(), centroidsDev.getSizeInBytes()));
    topkOpInput->emplace_back(aclCreateDataBuffer(centroidsSqrSum.data(), centroidsSqrSum.getSizeInBytes()));

    std::shared_ptr<std::vector<aclDataBuffer *>> topkOpOutput(new std::vector<aclDataBuffer *>(),
                                                               CommonUtils::AclOutputBufferDelete);
    topkOpOutput->emplace_back(aclCreateDataBuffer(dists.data(), dists.getSizeInBytes()));
    topkOpOutput->emplace_back(aclCreateDataBuffer(vmdists.data(), vmdists.getSizeInBytes()));
    topkOpOutput->emplace_back(aclCreateDataBuffer(opFlag.data(), opFlag.getSizeInBytes()));
    op->exec(*topkOpInput, *topkOpOutput, stream);
    return;
}

void IndexIVFPQ::runL2DistOp(int batch, AscendTensor<float, DIMS_2> &queries, AscendTensor<float, DIMS_3> &codeBook,
                             AscendTensor<float, DIMS_3, size_t> &dists, aclrtStream stream)
{
    AscendOperator *op = nullptr;
    if (l2DistFp32Ops.find(batch) != l2DistFp32Ops.end())
    {
        op = l2DistFp32Ops[batch].get();
    }
    ASCEND_THROW_IF_NOT(op);
    std::shared_ptr<std::vector<const aclDataBuffer *>> topkOpInput(new std::vector<const aclDataBuffer *>(),
                                                                    CommonUtils::AclInputBufferDelete);
    topkOpInput->emplace_back(aclCreateDataBuffer(queries.data(), queries.getSizeInBytes()));
    topkOpInput->emplace_back(aclCreateDataBuffer(codeBook.data(), codeBook.getSizeInBytes()));

    std::shared_ptr<std::vector<aclDataBuffer *>> topkOpOutput(new std::vector<aclDataBuffer *>(),
                                                               CommonUtils::AclOutputBufferDelete);
    topkOpOutput->emplace_back(aclCreateDataBuffer(dists.data(), dists.getSizeInBytes()));

    op->exec(*topkOpInput, *topkOpOutput, stream);
    return;
}

void IndexIVFPQ::runL3DistOp(
    int batch, AscendTensor<float, DIMS_3, size_t> &queryPQ, AscendTensor<uint8_t, DIMS_1, size_t> &codeBase,
    AscendTensor<int64_t, DIMS_2, size_t> &offset, AscendTensor<int64_t, DIMS_2, size_t> &baseSize,
    AscendTensor<int32_t, DIMS_1, size_t> &topk, AscendTensor<uint64_t, DIMS_1, size_t> &labelBase,
    AscendTensor<int64_t, DIMS_2, size_t> &labelOffset, AscendTensor<float, DIMS_3, size_t> &dists,
    AscendTensor<int32_t, DIMS_3, size_t> &topkIndex, AscendTensor<float, DIMS_3, size_t> &topkValue,
    AscendTensor<uint64_t, DIMS_2, size_t> &topkIndexFinal, AscendTensor<float, DIMS_2, size_t> &topkValueFinal,
    AscendTensor<uint16_t, DIMS_1, size_t> &opFlag, aclrtStream stream)
{
    AscendOperator *op = nullptr;
    if (l3DistFp32Ops.find(batch) != l3DistFp32Ops.end())
    {
        op = l3DistFp32Ops[batch].get();
    }
    ASCEND_THROW_IF_NOT(op);

    std::shared_ptr<std::vector<const aclDataBuffer *>> distOpInput(new std::vector<const aclDataBuffer *>(),
                                                                    CommonUtils::AclInputBufferDelete);
    distOpInput->emplace_back(aclCreateDataBuffer(queryPQ.data(), queryPQ.getSizeInBytes()));
    distOpInput->emplace_back(aclCreateDataBuffer(codeBase.data(), codeBase.getSizeInBytes()));
    distOpInput->emplace_back(aclCreateDataBuffer(offset.data(), offset.getSizeInBytes()));
    distOpInput->emplace_back(aclCreateDataBuffer(baseSize.data(), baseSize.getSizeInBytes()));
    distOpInput->emplace_back(aclCreateDataBuffer(topk.data(), topk.getSizeInBytes()));
    distOpInput->emplace_back(aclCreateDataBuffer(labelBase.data(), labelBase.getSizeInBytes()));
    distOpInput->emplace_back(aclCreateDataBuffer(labelOffset.data(), labelOffset.getSizeInBytes()));

    std::shared_ptr<std::vector<aclDataBuffer *>> distOpOutput(new std::vector<aclDataBuffer *>(),
                                                               CommonUtils::AclOutputBufferDelete);
    distOpOutput->emplace_back(aclCreateDataBuffer(dists.data(), dists.getSizeInBytes()));
    distOpOutput->emplace_back(aclCreateDataBuffer(topkIndex.data(), topkIndex.getSizeInBytes()));
    distOpOutput->emplace_back(aclCreateDataBuffer(topkValue.data(), topkValue.getSizeInBytes()));
    distOpOutput->emplace_back(aclCreateDataBuffer(topkIndexFinal.data(), topkIndexFinal.getSizeInBytes()));
    distOpOutput->emplace_back(aclCreateDataBuffer(topkValueFinal.data(), topkValueFinal.getSizeInBytes()));
    distOpOutput->emplace_back(aclCreateDataBuffer(opFlag.data(), opFlag.getSizeInBytes()));
    op->exec(*distOpInput, *distOpOutput, stream);
    return;
}

APP_ERROR IndexIVFPQ::resetL3TopkOp()
{
    auto topkCompOpReset = [&](std::unique_ptr<AscendOperator> &op, int64_t batch)
    {
        AscendOpDesc desc("TopkIvfpqL3");
        std::vector<int64_t> shape0{0, batch, MAX_TOPK};
        std::vector<int64_t> shape1{0, batch, MAX_TOPK};
        std::vector<int64_t> shape2{0, 16};
        std::vector<int64_t> shape3{aicpu::TOPK_IVFPQ_L3_ATTR_IDX_COUNT};

        std::vector<int64_t> shape4{batch, 0};

        desc.addInputTensorDesc(ACL_UINT64, shape0.size(), shape0.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_FLOAT, shape1.size(), shape1.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_UINT16, shape2.size(), shape2.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_INT64, shape3.size(), shape3.data(), ACL_FORMAT_ND);

        desc.addOutputTensorDesc(ACL_FLOAT, shape4.size(), shape4.data(), ACL_FORMAT_ND);
        desc.addOutputTensorDesc(ACL_UINT64, shape4.size(), shape4.data(), ACL_FORMAT_ND);

        op.reset();
        op = CREATE_UNIQUE_PTR(AscendOperator, desc);
        return op->init();
    };

    for (auto batch : searchBatchSizes)
    {
        topkL3Fp32[batch] = std::unique_ptr<AscendOperator>(nullptr);
        APPERR_RETURN_IF_NOT_LOG(topkCompOpReset(topkL3Fp32[batch], batch), APP_ERR_ACL_OP_LOAD_MODEL_FAILED,
                                 "l3 topk op init failed");
    }

    return APP_ERR_OK;
}

void IndexIVFPQ::runL3TopkOp(AscendTensor<uint64_t, DIMS_3, size_t> &topkIndex,
                             AscendTensor<float, DIMS_3, size_t> &topkValue,
                             AscendTensor<uint16_t, DIMS_2, size_t> &flags, AscendTensor<int64_t, DIMS_1> &attrs,
                             AscendTensor<float, DIMS_2, size_t> &outdists,
                             AscendTensor<uint64_t, DIMS_2, size_t> &outlabel, aclrtStream stream)
{
    int batch = static_cast<int>(topkIndex.getSize(2) / MAX_TOPK);
    std::vector<const AscendTensorBase *> input{&topkIndex, &topkValue, &flags, &attrs};
    std::vector<const AscendTensorBase *> output{&outdists, &outlabel};
    AscendOperator *op = nullptr;
    if (topkL3Fp32.find(batch) != topkL3Fp32.end())
    {
        op = topkL3Fp32[batch].get();
    }
    ASCEND_THROW_IF_NOT(op);
    std::shared_ptr<std::vector<const aclDataBuffer *>> topkOpInput(new std::vector<const aclDataBuffer *>(),
                                                                    CommonUtils::AclInputBufferDelete);
    for (auto tensor : input)
    {
        topkOpInput->emplace_back(aclCreateDataBuffer(tensor->getVoidData(), tensor->getSizeInBytes()));
    }
    std::shared_ptr<std::vector<aclDataBuffer *>> topkOpOutput(new std::vector<aclDataBuffer *>(),
                                                               CommonUtils::AclOutputBufferDelete);
    for (auto tensor : output)
    {
        topkOpOutput->emplace_back(aclCreateDataBuffer(tensor->getVoidData(), tensor->getSizeInBytes()));
    }
    op->exec(*topkOpInput, *topkOpOutput, stream);
}

void IndexIVFPQ::callL3DistanceOp(
    size_t batch, size_t tileNum, size_t segNum, size_t coreNum, size_t kAligned,
    AscendTensor<float, DIMS_3, size_t> &l2SubspaceDists, AscendTensor<int64_t, DIMS_3, size_t> &codeOffset,
    AscendTensor<int64_t, DIMS_3, size_t> &codeSize, AscendTensor<int32_t, DIMS_1, size_t> &topk,
    AscendTensor<int64_t, DIMS_3, size_t> &labelOffset, AscendTensor<uint16_t, DIMS_2, size_t> &opFlag,
    AscendTensor<float, DIMS_3, size_t> &distResult, AscendTensor<int32_t, DIMS_3, size_t> &topkIndex,
    AscendTensor<float, DIMS_3, size_t> &topkValue, AscendTensor<uint64_t, DIMS_3, size_t> &topkIndexFinal,
    AscendTensor<float, DIMS_3, size_t> &topkValueFinal, AscendTensor<uint8_t, DIMS_1, size_t> &codeBase,
    AscendTensor<uint64_t, DIMS_1, size_t> &labelBase, aclrtStream &stream)
{
    size_t ivfpqBlockSize = static_cast<size_t>(IVF_PQ_BLOCK_SIZE);
    // distResult / topkIndex / topkValue are single-tile workspaces reused across (tIdx, segIdx).
    AscendTensor<float, DIMS_3, size_t> subDist(distResult.data(), {batch, coreNum, ivfpqBlockSize});
    AscendTensor<int32_t, DIMS_3, size_t> subTopkIndex(topkIndex.data(), {batch, coreNum, kAligned});
    AscendTensor<float, DIMS_3, size_t> subTopkValue(topkValue.data(), {batch, coreNum, kAligned});
    for (size_t tIdx = 0; tIdx < tileNum; tIdx++)
    {
        for (size_t segIdx = 0; segIdx < segNum; segIdx++)
        {
            AscendTensor<int64_t, DIMS_2, size_t> subcodeOffset(codeOffset[tIdx][segIdx * batch * coreNum].data(),
                                                                {batch, coreNum});
            AscendTensor<int64_t, DIMS_2, size_t> subcodeSize(codeSize[tIdx][segIdx * batch * coreNum].data(),
                                                              {batch, coreNum});
            AscendTensor<uint16_t, DIMS_1, size_t> subOpFlag(opFlag[tIdx][segIdx * 16].data(), {16});

            AscendTensor<int64_t, DIMS_2, size_t> sublabelOffset(labelOffset[tIdx][segIdx * batch * coreNum].data(),
                                                                 {batch, coreNum});

            AscendTensor<uint64_t, DIMS_2, size_t> subTopkIndexFinal(
                topkIndexFinal[tIdx][segIdx * batch * coreNum * kAligned].data(), {batch, kAligned});
            AscendTensor<float, DIMS_2, size_t> subTopkValueFinal(
                topkValueFinal[tIdx][segIdx * batch * coreNum * kAligned].data(), {batch, kAligned});

            runL3DistOp(batch, l2SubspaceDists, codeBase, subcodeOffset, subcodeSize, topk, labelBase, sublabelOffset,
                        subDist, subTopkIndex, subTopkValue, subTopkIndexFinal, subTopkValueFinal, subOpFlag, stream);
        }
    }
}

int IndexIVFPQ::getL3SearchBatchCap() const
{
    // Cap so one-tile distResult fits in ~1.5GB: batch * blockNum * IVF_PQ_BLOCK_SIZE * 4.
    constexpr size_t kBudgetBytes = 1536ULL * 1024ULL * 1024ULL;
    const size_t perQuery = static_cast<size_t>(blockNum) * static_cast<size_t>(IVF_PQ_BLOCK_SIZE) * sizeof(float);
    int cap = perQuery == 0 ? 1 : static_cast<int>(kBudgetBytes / perQuery);
    if (cap < 1)
    {
        cap = 1;
    }
    int best = 1;
    for (int bs : searchBatchSizes)
    {
        if (bs <= cap && bs > best)
        {
            best = bs;
        }
    }
    return best;
}

APP_ERROR IndexIVFPQ::fillDisOpInputDataPQ(int k, size_t batch, size_t tileNum, size_t segNum, size_t coreNum,
                                           AscendTensor<int64_t, DIMS_3, size_t> &offset,
                                           AscendTensor<int64_t, DIMS_3, size_t> &baseSize,
                                           AscendTensor<int64_t, DIMS_1> &attrs,
                                           AscendTensor<int64_t, DIMS_3, size_t> &labelOffset,
                                           AscendTensor<int64_t, DIMS_2> &l1TopNprobeIndicesHost)
{
    size_t ivfpqBlockSize = static_cast<size_t>(IVF_PQ_BLOCK_SIZE);
    std::vector<int64_t> offsetHost(tileNum * segNum * batch * coreNum, 0);
    AscendTensor<int64_t, DIMS_3, size_t> offsetHostVec(offsetHost.data(), {tileNum, segNum, batch * coreNum});
    std::vector<int64_t> baseSizeHost(tileNum * segNum * batch * coreNum, 0);
    AscendTensor<int64_t, DIMS_3, size_t> baseSizeHostVec(baseSizeHost.data(), {tileNum, segNum, batch * coreNum});
    std::vector<int64_t> labeloffsetHost(tileNum * segNum * batch * coreNum, 0);
    AscendTensor<int64_t, DIMS_3, size_t> labeloffsetHostVec(labeloffsetHost.data(),
                                                             {tileNum, segNum, batch * coreNum});

    for (size_t tIdx = 0; tIdx < tileNum; tIdx++)
    {
        for (size_t segIdx = 0; segIdx < segNum; segIdx++)
        {
            for (size_t qIdx = 0; qIdx < batch; qIdx++)
            {
                fillDisOpInputDataByBlockPQ(qIdx, tIdx, segIdx, segNum, coreNum, ivfpqBlockSize, baseSizeHostVec,
                                            offsetHostVec, labeloffsetHostVec, l1TopNprobeIndicesHost);
            }
        }
    }

    auto ret = aclrtMemcpy(offset.data(), offset.getSizeInBytes(), offsetHostVec.data(), offsetHostVec.getSizeInBytes(),
                           ACL_MEMCPY_HOST_TO_DEVICE);
    APPERR_RETURN_IF_NOT_FMT(ret == APP_ERR_OK, APP_ERR_INNER_ERROR, "copy offset to device failed %d", ret);

    ret = aclrtMemcpy(baseSize.data(), baseSize.getSizeInBytes(), baseSizeHostVec.data(),
                      baseSizeHostVec.getSizeInBytes(), ACL_MEMCPY_HOST_TO_DEVICE);
    APPERR_RETURN_IF_NOT_FMT(ret == APP_ERR_OK, APP_ERR_INNER_ERROR, "copy basesize to device failed %d", ret);

    ret = aclrtMemcpy(labelOffset.data(), labelOffset.getSizeInBytes(), labeloffsetHostVec.data(),
                      labeloffsetHostVec.getSizeInBytes(), ACL_MEMCPY_HOST_TO_DEVICE);
    APPERR_RETURN_IF_NOT_FMT(ret == APP_ERR_OK, APP_ERR_INNER_ERROR, "copy label offset to device failed %d", ret);

    std::vector<int64_t> attrsVec(aicpu::TOPK_IVFPQ_L3_ATTR_IDX_COUNT);
    attrsVec[aicpu::TOPK_IVFPQ_L3_ATTR_ASC_IDX] = 1;
    attrsVec[aicpu::TOPK_IVFPQ_L3_ATTR_K_IDX] = k;
    attrsVec[aicpu::TOPK_IVFPQ_L3_ATTR_BLOCK_NUM_IDX] = static_cast<int64_t>(tileNum * segNum);
    attrsVec[aicpu::TOPK_IVFPQ_L3_ATTR_BATCH_NUM_IDX] = static_cast<int64_t>(batch);
    ret = aclrtMemcpy(attrs.data(), attrs.getSizeInBytes(), attrsVec.data(), attrsVec.size() * sizeof(int64_t),
                      ACL_MEMCPY_HOST_TO_DEVICE);
    APPERR_RETURN_IF_NOT_FMT(ret == APP_ERR_OK, APP_ERR_INNER_ERROR, "copy attrs to device failed %d", ret);
    return APP_ERR_OK;
}

void IndexIVFPQ::fillDisOpInputDataByBlockPQ(size_t qIdx, size_t tIdx, size_t segIdx, size_t segNum, size_t coreNum,
                                             size_t ivfpqBlockSize,
                                             AscendTensor<int64_t, DIMS_3, size_t> &baseSizeHostVec,
                                             AscendTensor<int64_t, DIMS_3, size_t> &offsetHostVec,
                                             AscendTensor<int64_t, DIMS_3, size_t> &labeloffsetHostVec,
                                             AscendTensor<int64_t, DIMS_2> &l1TopNprobeIndicesHost)
{
    // Prefer a non-empty list as the label address base (list 0 is often empty under multi-device sharding).
    idx_t *labelBasePtr = nullptr;
    for (size_t i = 0; i < deviceListIndices.size(); ++i)
    {
        if (deviceListIndices[i] && deviceListIndices[i]->size() > 0 && deviceListIndices[i]->data() != nullptr)
        {
            labelBasePtr = deviceListIndices[i]->data();
            break;
        }
    }

    for (size_t cIdx = 0; cIdx < coreNum; cIdx++)
    {
        if (cIdx + tIdx * coreNum >= static_cast<size_t>(nprobe))
        {
            continue;
        }
        int64_t listId = l1TopNprobeIndicesHost[qIdx][tIdx * coreNum + (segIdx * coreNum + cIdx) / segNum].value();
        if (listId < 0 || static_cast<size_t>(listId) >= deviceListIndices.size() || !deviceListIndices[listId])
        {
            baseSizeHostVec[tIdx][segIdx][qIdx * coreNum + cIdx].value(0);
            continue;
        }
        size_t listNum = deviceListIndices[listId]->size();
        size_t proccessLen =
            (segIdx * ivfpqBlockSize >= listNum) ? 0 : std::min(listNum - segIdx * ivfpqBlockSize, ivfpqBlockSize);
        baseSizeHostVec[tIdx][segIdx][qIdx * coreNum + cIdx].value(static_cast<int64_t>(proccessLen));
        if (proccessLen == 0 || segIdx >= basePQCoder[listId].size() || labelBasePtr == nullptr ||
            deviceListIndices[listId]->data() == nullptr)
        {
            continue;
        }

        int64_t idAddrDevice = reinterpret_cast<int64_t>(deviceListIndices[listId]->data() +
                                                         ((segIdx * coreNum + cIdx) % segNum) * ivfpqBlockSize);
        labeloffsetHostVec[tIdx][segIdx][qIdx * coreNum + cIdx].value(
            (idAddrDevice - reinterpret_cast<int64_t>(labelBasePtr)) / static_cast<int64_t>(sizeof(idx_t)));

        uint64_t offsetSeg = reinterpret_cast<uint64_t>(basePQCoder[listId].at(segIdx)->data()) -
                             reinterpret_cast<uint64_t>(pBasePQCoder);
        offsetHostVec[tIdx][segIdx][qIdx * coreNum + cIdx].value(static_cast<int64_t>(offsetSeg));
    }
}

size_t IndexIVFPQ::getMaxListNum(size_t batch, AscendTensor<int64_t, DIMS_2> &l1TopNprobeIndicesHost) const
{
    size_t maxLen = 0;
    for (size_t qIdx = 0; qIdx < batch; qIdx++)
    {
        for (size_t probId = 0; probId < static_cast<size_t>(nprobe); probId++)
        {
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
    auto ret = ensureL2L3SearchOps();
    APPERR_RETURN_IF_NOT_FMT(ret == APP_ERR_OK, ret, "ensureL2L3SearchOps failed %d", ret);

    auto streamPtr = resources.getDefaultStream();
    auto stream = streamPtr->GetStream();
    auto streamAicpuPtr = resources.getAlternateStreams()[0];
    auto streamAicpu = streamAicpuPtr->GetStream();

    int n = queries.getSize(0);
    ensureL1Workspace(n, numLists);
    const int minDistSize = numLists / IVF_PQ_BURST_LEN * 2;
    AscendTensor<float, DIMS_2> dists(l1DistsDev_->data(), {n, numLists});
    AscendTensor<float, DIMS_2> vmdists(l1VmdistsDev_->data(), {n, minDistSize});
    AscendTensor<uint32_t, DIMS_2> opSize(l1OpSizeDev_->data(), {CORE_NUM, SIZE_ALIGN});
    opSize[0][0] = numLists;
    AscendTensor<uint16_t, DIMS_2> opFlag(l1OpFlagDev_->data(), {CORE_NUM, FLAG_SIZE});
    opFlag.zero();

    AscendTensor<int64_t, DIMS_1> attrsInput(l1AttrsDev_->data(), {aicpu::TOPK_FLAT_ATTR_IDX_COUNT});
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

    ret = aclrtMemcpy(attrsInput.data(), attrsInput.getSizeInBytes(), attrs.data(), attrs.size() * sizeof(int64_t),
                      ACL_MEMCPY_HOST_TO_DEVICE);
    APPERR_RETURN_IF_NOT_LOG(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "Failed to copy attr to device");

    AscendTensor<float, DIMS_2> l1TopNprobeDists(l1TopNprobeDistsDev_->data(), {n, nprobe});
    AscendTensor<int64_t, DIMS_2> l1TopNprobeIndices(l1TopNprobeIndicesDev_->data(), {n, nprobe});

    AscendTensor<float, DIMS_3> codeBookDev(codeBookOnDevice->data(), {M, ksub, dsub});

    // run l1&l2 distance calculation
    AscendTensor<float, DIMS_2> centroidsDev(centroidsOnDevice->data(), {numLists, dims});
    runL1TopkOp(dists, vmdists, opSize, opFlag, attrsInput, l1TopNprobeDists, l1TopNprobeIndices, streamAicpu);
    runL1DistOp(n, queries, centroidsDev, dists, vmdists, opFlag, stream);
    runL2DistOp(n, queries, codeBookDev, l2SubspaceDistsDev, stream);

    ret = synchronizeStream(stream);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "synchronizeStream default stream: %i\n", ret);
    ret = synchronizeStream(streamAicpu);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "synchronizeStream aicpu stream failed: %i\n",
                             ret);

    ret = aclrtMemcpy(l1TopNprobeIndicesHost.data(), l1TopNprobeIndicesHost.getSizeInBytes(), l1TopNprobeIndices.data(),
                      l1TopNprobeIndices.getSizeInBytes(), ACL_MEMCPY_DEVICE_TO_HOST);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "Mem operator error %d", ret);

    return APP_ERR_OK;
}

APP_ERROR IndexIVFPQ::searchImplL3(AscendTensor<int64_t, DIMS_2> &l1TopNprobeIndicesHost,
                                   AscendTensor<float, DIMS_3, size_t> &l2SubspaceDistsDev, int k, float *distances,
                                   idx_t *labels)
{
    auto ret = ensureL2L3SearchOps();
    APPERR_RETURN_IF_NOT_FMT(ret == APP_ERR_OK, ret, "ensureL2L3SearchOps failed %d", ret);

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

    // No vectors in any probed list on this device (common with large nlist + sharding).
    if (maxLen == 0 || segNum == 0)
    {
        for (size_t i = 0; i < batch; i++)
        {
            for (int j = 0; j < k; j++)
            {
                distances[i * static_cast<size_t>(k) + static_cast<size_t>(j)] = std::numeric_limits<float>::max();
                labels[i * static_cast<size_t>(k) + static_cast<size_t>(j)] = -1;
            }
        }
        return APP_ERR_OK;
    }

    idx_t *labelBasePtr = nullptr;
    for (size_t i = 0; i < deviceListIndices.size(); ++i)
    {
        if (deviceListIndices[i] && deviceListIndices[i]->size() > 0 && deviceListIndices[i]->data() != nullptr)
        {
            labelBasePtr = deviceListIndices[i]->data();
            break;
        }
    }
    APPERR_RETURN_IF_NOT_LOG(labelBasePtr != nullptr, APP_ERR_INNER_ERROR,
                             "all nprobe cluster is empty by L1 search !!!");

    AscendTensor<uint16_t, DIMS_2, size_t> opFlag(mem, {tileNum, segNum * 16}, stream);
    (void)opFlag.zero();
    AscendTensor<int64_t, DIMS_3, size_t> offset(mem, {tileNum, segNum, batch * coreNum}, stream);
    AscendTensor<int64_t, DIMS_3, size_t> baseSize(mem, {tileNum, segNum, batch * coreNum}, stream);
    AscendTensor<uint8_t, DIMS_1, size_t> codeBase(pBasePQCoder, {static_cast<size_t>(M)});
    AscendTensor<uint64_t, DIMS_1, size_t> labelBase(reinterpret_cast<uint64_t *>(labelBasePtr),
                                                     {static_cast<size_t>(M)});
    AscendTensor<int64_t, DIMS_3, size_t> labelOffset(mem, {tileNum, segNum, batch * coreNum}, stream);

    AscendTensor<int32_t, DIMS_3, size_t> topkIndex(mem, {1, 1, batch * coreNum * kAligned}, stream);
    AscendTensor<float, DIMS_3, size_t> topkValue(mem, {1, 1, batch * coreNum * kAligned}, stream);
    AscendTensor<uint64_t, DIMS_3, size_t> topkIndexFinal(mem, {tileNum, segNum, batch * kAligned}, stream);
    AscendTensor<float, DIMS_3, size_t> topkValueFinal(mem, {tileNum, segNum, batch * kAligned}, stream);
    // Single-tile dist workspace reused across (tile, seg) iterations in callL3DistanceOp.
    AscendTensor<float, DIMS_3, size_t> distResult(mem, {1, 1, batch * coreNum * ivfpqBlockSize}, stream);

    AscendTensor<int64_t, DIMS_1> attrs(mem, {aicpu::TOPK_IVFPQ_L3_ATTR_IDX_COUNT}, stream);
    AscendTensor<int32_t, DIMS_1, size_t> topk(mem, {kAligned}, stream);

    fillDisOpInputDataPQ(k, batch, tileNum, segNum, coreNum, offset, baseSize, attrs, labelOffset,
                         l1TopNprobeIndicesHost);

    AscendTensor<float, DIMS_2, size_t> outDist(mem, {batch, static_cast<size_t>(k)}, stream);
    AscendTensor<idx_t, DIMS_2, size_t> outLabel(mem, {batch, static_cast<size_t>(k)}, stream);

    // no need aicpu merge
    if (tileNum == 1 && segNum == 1)
    {
        callL3DistanceOp(batch, tileNum, segNum, coreNum, kAligned, l2SubspaceDistsDev, offset, baseSize, topk,
                         labelOffset, opFlag, distResult, topkIndex, topkValue, topkIndexFinal, topkValueFinal,
                         codeBase, labelBase, stream);
        ret = synchronizeStream(stream);
        APPERR_RETURN_IF_NOT_FMT(ret == APP_ERR_OK, APP_ERR_INNER_ERROR, "sync aicore stream failed %d", ret);

        std::vector<idx_t> topkIndexFinalHost(batch * kAligned, 0);
        std::vector<float> topkValueFinalHost(batch * kAligned, 0);
        ret = aclrtMemcpy(topkValueFinalHost.data(), batch * kAligned * sizeof(float), topkValueFinal.data(),
                          batch * kAligned * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
        APPERR_RETURN_IF_NOT_FMT(ret == APP_ERR_OK, APP_ERR_INNER_ERROR, "copy topkValueFinal to host failed %d", ret);
        ret = aclrtMemcpy(topkIndexFinalHost.data(), batch * kAligned * sizeof(idx_t), topkIndexFinal.data(),
                          batch * kAligned * sizeof(idx_t), ACL_MEMCPY_DEVICE_TO_HOST);
        APPERR_RETURN_IF_NOT_FMT(ret == APP_ERR_OK, APP_ERR_INNER_ERROR, "copy topkIndexFinal to host failed %d", ret);

        for (int i = 0; i < batch; i++)
        {
            for (int j = 0; j < k; j++)
            {
                labels[i * k + j] = topkIndexFinalHost[i * kAligned + j];
                distances[i * k + j] = topkValueFinalHost[i * kAligned + j];
            }
        }
        return APP_ERR_OK;
    }

    runL3TopkOp(topkIndexFinal, topkValueFinal, opFlag, attrs, outDist, outLabel, streamAicpu);
    callL3DistanceOp(batch, tileNum, segNum, coreNum, kAligned, l2SubspaceDistsDev, offset, baseSize, topk, labelOffset,
                     opFlag, distResult, topkIndex, topkValue, topkIndexFinal, topkValueFinal, codeBase, labelBase,
                     stream);
    ret = synchronizeStream(stream);
    APPERR_RETURN_IF_NOT_FMT(ret == APP_ERR_OK, APP_ERR_INNER_ERROR, "sync aicore stream failed %d", ret);
    ret = synchronizeStream(streamAicpu);
    APPERR_RETURN_IF_NOT_FMT(ret == APP_ERR_OK, APP_ERR_INNER_ERROR, "sync aicpu stream failed %d", ret);

    ret = aclrtMemcpy(distances, batch * k * sizeof(float), outDist.data(), outDist.getSizeInBytes(),
                      ACL_MEMCPY_DEVICE_TO_HOST);
    APPERR_RETURN_IF_NOT_FMT(ret == APP_ERR_OK, APP_ERR_INNER_ERROR, "copy distances to host failed %d", ret);
    ret = aclrtMemcpy(labels, batch * k * sizeof(idx_t), outLabel.data(), outLabel.getSizeInBytes(),
                      ACL_MEMCPY_DEVICE_TO_HOST);
    APPERR_RETURN_IF_NOT_FMT(ret == APP_ERR_OK, APP_ERR_INNER_ERROR, "copy outLabel to host failed %d", ret);
    return APP_ERR_OK;
}

APP_ERROR IndexIVFPQ::searchWithBatch(int n, const float *x, int k, float *distances, idx_t *labels)
{
    auto &mem = resources.getMemoryManager();
    auto streamPtr = resources.getDefaultStream();
    auto stream = streamPtr->GetStream();
    AscendTensor<float, DIMS_2> queries(mem, {n, dims}, stream);
    auto ret =
        aclrtMemcpy(queries.data(), queries.getSizeInBytes(), x, n * dims * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "copy query error %d", ret);

    std::vector<int64_t> l1TopNprobeIndicesVec(n * nprobe, 0);
    AscendTensor<int64_t, DIMS_2> l1TopNprobeIndicesHost(l1TopNprobeIndicesVec.data(), {n, nprobe});
    AscendTensor<float, DIMS_3, size_t> l2SubspaceDistsDev(
        mem, {static_cast<size_t>(n), static_cast<size_t>(M), static_cast<size_t>(ksub)}, stream);
    ret = searchImplL1(queries, l1TopNprobeIndicesHost, l2SubspaceDistsDev);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, ret, "IVFPQ L1 search failed! %d", ret);

    ret = searchImplL3(l1TopNprobeIndicesHost, l2SubspaceDistsDev, k, distances, labels);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, ret, "IVFPQ L3 search failed! %d", ret);
    return ACL_SUCCESS;
}

APP_ERROR IndexIVFPQ::searchImpl(int n, const float *x, int k, float *distances, idx_t *labels)
{
    APPERR_RETURN_IF_NOT_FMT(k >= 1 && k <= MAX_TOPK, APP_ERR_INVALID_PARAM, "the k is %d, out of MAX_TOPK(%d)", k,
                             MAX_TOPK);

    APP_ERROR ret = APP_ERR_OK;
    if (n == 1 || searchBatchSizes.empty())
    {
        return searchWithBatch(n, x, k, distances, labels);
    }
    const int l3BatchCap = getL3SearchBatchCap();
    size_t size = searchBatchSizes.size();
    int64_t searched = 0;
    for (size_t i = 0; i < size; i++)
    {
        int64_t batchSize = searchBatchSizes[i];
        if (batchSize > l3BatchCap)
        {
            continue;
        }
        if ((n - searched) >= batchSize)
        {
            int64_t page = (n - searched) / batchSize;
            for (int64_t j = 0; j < page; j++)
            {
                ret =
                    searchWithBatch(batchSize, x + searched * dims, k, distances + searched * k, labels + searched * k);
                APPERR_RETURN_IF(ret, ret);
                searched += batchSize;
            }
        }
    }
    for (int64_t i = searched; i < n; i++)
    {
        ret = searchWithBatch(1, x + i * dims, k, distances + i * k, labels + i * k);
        APPERR_RETURN_IF(ret, ret);
    }
    return APP_ERR_OK;
}

APP_ERROR IndexIVFPQ::searchImpl(int n, const float16_t *x, int k, float16_t *distances, idx_t *labels)
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

}  // namespace ascend
