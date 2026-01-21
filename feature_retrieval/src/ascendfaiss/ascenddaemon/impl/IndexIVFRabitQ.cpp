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

#include "ascenddaemon/impl/IndexIVFRabitQ.h"
#include <iostream>
#include <algorithm>
#include <cmath>
#include <atomic>

#include "ascenddaemon/impl/AuxIndexStructures.h"
#include "ascenddaemon/utils/Limits.h"
#include "common/utils/CommonUtils.h"
#include "common/utils/LogUtils.h"
#include "common/utils/OpLauncher.h"
#include "ops/cpukernel/impl/utils/kernel_shared_def.h"

namespace ascend {
const int KB = 1024;
const int IVF_RABITQ_BLOCK_SIZE = 16384 * 2;
const int SCAN_BIT = 8;
const int LUT_NUM = std::pow(2, SCAN_BIT);
IndexIVFRabitQ::IndexIVFRabitQ(int numList, int dim, int nprobes, int64_t resourceSize)
    : IndexIVF(numList, dim * 4, dim, nprobes, resourceSize), blockSize(IVF_RABITQ_BLOCK_SIZE)
{
    ASCEND_THROW_IF_NOT(dim % CUBE_ALIGN == 0);
    isTrained = false;
    searchBatchSizes = {64, 32, 16, 8, 4, 2, 1}; // supported batch size
    listVecNum = std::vector<size_t>(numList, 0);
    baseFp32.resize(numList);
    for (size_t i = 0; i < numList; i++) {
        IndexL1OnDevice.emplace_back(CREATE_UNIQUE_PTR(DeviceVector<float>, MemorySpace::DEVICE_HUGEPAGE));
        IndexL2OnDevice.emplace_back(CREATE_UNIQUE_PTR(DeviceVector<float>, MemorySpace::DEVICE_HUGEPAGE));
    }
    pBaseFp32 = reinterpret_cast<uint8_t*>(0xffffffffffffffff);  // 基地址初始化为最大的无效值 0xffffffffffffffff
    pIndexL2 = reinterpret_cast<float*>(0xffffffffffffffff);  // 基地址初始化为最大的无效值 0xffffffffffffffff
    pIndexL1 = reinterpret_cast<float*>(0xffffffffffffffff);  // 基地址初始化为最大的无效值 0xffffffffffffffff
    centroidsOnDevice = CREATE_UNIQUE_PTR(DeviceVector<float>, MemorySpace::DEVICE_HUGEPAGE);
    centroidsOnDevice->resize(numList * dims);
    OrthogonalMatrixOnDevice = CREATE_UNIQUE_PTR(DeviceVector<float>, MemorySpace::DEVICE_HUGEPAGE);
    OrthogonalMatrixOnDevice->resize(dims * dims);
    CentroidL2OnDevice = CREATE_UNIQUE_PTR(DeviceVector<float>, MemorySpace::DEVICE_HUGEPAGE);
    CentroidL2OnDevice->resize(numList);
    if (faiss::ascend::SocUtils::GetInstance().IsAscendA5()) {
        LUTMatrixOnDevice = CREATE_UNIQUE_PTR(DeviceVector<float>, MemorySpace::DEVICE_HUGEPAGE);
        LUTMatrixOnDevice->resize(SCAN_BIT * LUT_NUM);
        CentroidLUTOnDevice = CREATE_UNIQUE_PTR(DeviceVector<float>, MemorySpace::DEVICE_HUGEPAGE);
        CentroidLUTOnDevice->resize(numList * dims / SCAN_BIT * LUT_NUM);
        uploadLUTMatrix();
    }
    distResultOnDevice = CREATE_UNIQUE_PTR(DeviceVector<float>, MemorySpace::DEVICE_HUGEPAGE);
    distResultSpace = CORE_NUM * static_cast<size_t>(IVF_RABITQ_BLOCK_SIZE);
    distResultOnDevice->resize(distResultSpace);
    auto ret = resetL1TopkOp();
    ASCEND_THROW_IF_NOT_MSG(ret == APP_ERR_OK, "resetL1TopkOp failed!");
    ret = resetL1DistOp();
    ASCEND_THROW_IF_NOT_MSG(ret == APP_ERR_OK, "resetL1DistOp failed!");
    ret = resetL2DistOp();
    ASCEND_THROW_IF_NOT_MSG(ret == APP_ERR_OK, "resetL2DistOp failed!");
    ret = resetL2TopkOp();
    ASCEND_THROW_IF_NOT_MSG(ret == APP_ERR_OK, "resetL2TopkOp failed!");
    ret = resetCenterRotateL2Op();
    ASCEND_THROW_IF_NOT_MSG(ret == APP_ERR_OK, "resetCenterRotateL2Op failed!");
    ret = resetCenterLUTOp();
    ASCEND_THROW_IF_NOT_MSG(ret == APP_ERR_OK, "resetCenterLUTOp failed!");
    ret = resetIndexRotateL2Op();
    ASCEND_THROW_IF_NOT_MSG(ret == APP_ERR_OK, "resetIndexRotateL2Op failed!");
    ret = resetIndexCodeAndPreComputeOp();
    ASCEND_THROW_IF_NOT_MSG(ret == APP_ERR_OK, "resetIndexCodeAndPreComputeOp failed!");
    ret = resetQueryRotateOp();
    ASCEND_THROW_IF_NOT_MSG(ret == APP_ERR_OK, "resetQueryRotateOp failed!");
    ret = resetQueryLUTOp();
    ASCEND_THROW_IF_NOT_MSG(ret == APP_ERR_OK, "resetQueryLUTOp failed!");
}

IndexIVFRabitQ::~IndexIVFRabitQ() {}

APP_ERROR IndexIVFRabitQ::reset()
{
    IndexIVF::reset();
    for (size_t i = 0; i < baseFp32.size(); i++) {
        baseFp32[i].clear();
    }
    baseFp32.clear();
    IndexL1OnDevice.clear();
    IndexL2OnDevice.clear();
    centroidsOnDevice->clear();
    OrthogonalMatrixOnDevice->clear();
    LUTMatrixOnDevice->clear();
    CentroidLUTOnDevice->clear();
    CentroidL2OnDevice->clear();
    distResultOnDevice->clear();
    return APP_ERR_OK;
}

APP_ERROR IndexIVFRabitQ::resize(int listId, size_t numVecs)
{
    if (numVecs == 0) {
        return APP_ERR_OK;
    }
    size_t currentVecNum = listVecNum[listId];
    IndexL2OnDevice[listId]->resize(currentVecNum + numVecs, 0);
    pIndexL2 = IndexL2OnDevice[listId]->data() < pIndexL2 ? IndexL2OnDevice[listId]->data() : pIndexL2;
    IndexL1OnDevice[listId]->resize(currentVecNum + numVecs, 0);
    pIndexL1 = IndexL1OnDevice[listId]->data() < pIndexL1 ? IndexL1OnDevice[listId]->data() : pIndexL1;

    auto& blockList = baseFp32[listId];
    
    size_t remainingVecs = numVecs;
    while (remainingVecs > 0) {
        size_t tailBlkId = blockList.empty() ? 0 : blockList.size() - 1;
        if (blockList.empty()) {
            blockList.emplace_back(CREATE_UNIQUE_PTR(DeviceVector<uint8_t>, MemorySpace::DEVICE_HUGEPAGE));
            tailBlkId = 0;
        }

        size_t currentBlockUsed = currentVecNum % static_cast<size_t>(blockSize);
        size_t currentBlockRemaining;
        if (currentVecNum > 0 && currentBlockUsed == 0) {
            currentBlockRemaining = 0;
        } else {
            currentBlockRemaining = static_cast<size_t>(blockSize) - currentBlockUsed;
        }
        if (currentBlockRemaining == 0) {
            blockList.emplace_back(CREATE_UNIQUE_PTR(DeviceVector<uint8_t>, MemorySpace::DEVICE_HUGEPAGE));
            tailBlkId = blockList.size() - 1;
            currentBlockRemaining = static_cast<size_t>(blockSize);
        }
        size_t vecsToAdd = std::min(remainingVecs, currentBlockRemaining);
        size_t newSize = (currentVecNum % static_cast<size_t>(blockSize)) + vecsToAdd;
        blockList[tailBlkId]->resize(newSize * ((dims + 7) / 8), true);
        currentVecNum += vecsToAdd;
        remainingVecs -= vecsToAdd;
        pBaseFp32 =
            blockList.at(tailBlkId)->data() < pBaseFp32 ? blockList.at(tailBlkId)->data() : pBaseFp32;
    }
    return APP_ERR_OK;
}

size_t IndexIVFRabitQ::getListLength(int listId) const
{
    ASCEND_THROW_IF_NOT((listId < numLists) && (listId >= 0));

    return listVecNum[listId];
}

void IndexIVFRabitQ::uploadLUTMatrix()
{
    APP_LOG_INFO("IndexIVFRabitQ  create LUTMatrix started\n");
    std::vector<float> LUTMatrix(SCAN_BIT * LUT_NUM, 0);
    for (int i = 0; i < SCAN_BIT; i++) {
        int mask = (1 << i);
        for (int j = 0; j < LUT_NUM; j++) {
            if (j & mask) {
                LUTMatrix[i * LUT_NUM + j] = 1;
            }
        }
    }

    auto ret = aclrtMemcpy(LUTMatrixOnDevice->data(), LUTMatrixOnDevice->size() * sizeof(float),
                           LUTMatrix.data(), SCAN_BIT * LUT_NUM * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE);
    ASCEND_THROW_IF_NOT_FMT(ret == ACL_SUCCESS, "Failed to aclrtMemcpy LUTMatrix, result is: %d\n", ret);
}

APP_ERROR IndexIVFRabitQ::updateCoarseCenterImpl(std::vector<float> &centerData)
{
    APPERR_RETURN_IF_NOT_FMT(centerData.size() == dims * numLists, APP_ERR_INVALID_PARAM,
        "the centerData size is %d, not dims(%d) * numLists(%d) %d",
        centerData.size(), dims, numLists, dims * numLists);

    auto &mem = resources.getMemoryManager();
    auto streamPtr = resources.getDefaultStream();
    auto stream = streamPtr->GetStream();

    AscendTensor<float, DIMS_2> originCentroids(mem, {numLists, dims}, stream);
    // 聚类中心传到npu
    auto ret = aclrtMemcpy(originCentroids.data(), originCentroids.getSizeInBytes(),
                           centerData.data(), dims * numLists * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE);
    ASCEND_THROW_IF_NOT_FMT(ret == ACL_SUCCESS, "Failed to aclrtMemcpy CoarseCenter, result is: %d\n", ret);

    // c = Pcr, ||cr||^2 预计算
    AscendTensor<int32_t, DIMS_1> vectorSizeVec(mem, {1}, stream);
    AscendTensor<float, DIMS_2> centroidsDev(centroidsOnDevice->data(), {numLists, dims});
    AscendTensor<float, DIMS_2> rotatematrixDev(OrthogonalMatrixOnDevice->data(), {dims, dims});
    AscendTensor<float, DIMS_1> centroidsl2(CentroidL2OnDevice->data(), {numLists});
    std::vector<int32_t> vectorSize(1, 0);
    vectorSize[0] = numLists;
    ret = aclrtMemcpy(vectorSizeVec.data(), sizeof(int32_t), vectorSize.data(),
                      sizeof(int32_t), ACL_MEMCPY_HOST_TO_DEVICE);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "copy vectorSize to device failed %d", ret);
    runCenterRotateL2Op(originCentroids, vectorSizeVec, rotatematrixDev, centroidsDev, centroidsl2, stream);
    ret = synchronizeStream(stream);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR,
        "synchronizeStream default stream: %i\n", ret);
    if (faiss::ascend::SocUtils::GetInstance().IsAscendA5()) {
        // <x, c> LUT预计算
        AscendTensor<float, DIMS_2> centroidsrotate(centroidsOnDevice->data(), {numLists * dims / SCAN_BIT, SCAN_BIT});
        AscendTensor<float, DIMS_2> lutmatrixDev(LUTMatrixOnDevice->data(), {SCAN_BIT, LUT_NUM});
        AscendTensor<float, DIMS_2> centroidslut(CentroidLUTOnDevice->data(), {numLists * dims / SCAN_BIT, LUT_NUM});
        runCenterLUTOp(centroidsrotate, lutmatrixDev, centroidslut, stream);
        ret = synchronizeStream(stream);
        APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR,
            "synchronizeStream default stream: %i\n", ret);
    }
    return APP_ERR_OK;
}

APP_ERROR IndexIVFRabitQ::resetCenterRotateL2Op()
{
    auto CenterRotateL2OpReset = [&](std::unique_ptr<AscendOperator> &op) {
        AscendOpDesc desc("RotateAndL2AtFP32");
        std::vector<int64_t> codesShape({ numLists, dims });
        std::vector<int64_t> sizeShape({ 1 });
        std::vector<int64_t> matrixShape({ dims, dims });
        
        std::vector<int64_t> rotateResultShape({ numLists, dims });
        std::vector<int64_t> l2ResultShape({ numLists });
        desc.addInputTensorDesc(ACL_FLOAT, codesShape.size(), codesShape.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_INT32, sizeShape.size(), sizeShape.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_FLOAT, matrixShape.size(), matrixShape.data(), ACL_FORMAT_ND);
        desc.addOutputTensorDesc(ACL_FLOAT, rotateResultShape.size(), rotateResultShape.data(), ACL_FORMAT_ND);
        desc.addOutputTensorDesc(ACL_FLOAT, l2ResultShape.size(), l2ResultShape.data(), ACL_FORMAT_ND);
        op.reset();
        op = CREATE_UNIQUE_PTR(AscendOperator, desc);
        return op->init();
    };

    ivfRabitQRotateL2Ops = std::unique_ptr<AscendOperator>(nullptr);
    APPERR_RETURN_IF_NOT_LOG(CenterRotateL2OpReset(ivfRabitQRotateL2Ops),
                             APP_ERR_ACL_OP_LOAD_MODEL_FAILED, "IVFRabitQ Rotate L2 FP32 distance op init failed");
    return APP_ERR_OK;
}

void IndexIVFRabitQ::runCenterRotateL2Op(AscendTensor<float, DIMS_2> &centroid,
                                         AscendTensor<int32_t, DIMS_1> &vectorSize,
                                         AscendTensor<float, DIMS_2> &matrix,
                                         AscendTensor<float, DIMS_2> &rotateCentroid,
                                         AscendTensor<float, DIMS_1> &centroidl2,
                                         aclrtStream stream)
{
    AscendOperator *op = ivfRabitQRotateL2Ops.get();
    ASCEND_THROW_IF_NOT(op);
    std::shared_ptr<std::vector<const aclDataBuffer *>> topkOpInput(
        new std::vector<const aclDataBuffer *>(), CommonUtils::AclInputBufferDelete);
    topkOpInput->emplace_back(aclCreateDataBuffer(centroid.data(), centroid.getSizeInBytes()));
    topkOpInput->emplace_back(aclCreateDataBuffer(vectorSize.data(), vectorSize.getSizeInBytes()));
    topkOpInput->emplace_back(aclCreateDataBuffer(matrix.data(), matrix.getSizeInBytes()));

    std::shared_ptr<std::vector<aclDataBuffer *>> topkOpOutput(
        new std::vector<aclDataBuffer *>(), CommonUtils::AclOutputBufferDelete);
    topkOpOutput->emplace_back(aclCreateDataBuffer(rotateCentroid.data(), rotateCentroid.getSizeInBytes()));
    topkOpOutput->emplace_back(aclCreateDataBuffer(centroidl2.data(), centroidl2.getSizeInBytes()));
    op->exec(*topkOpInput, *topkOpOutput, stream);
    return;
}

APP_ERROR IndexIVFRabitQ::resetCenterLUTOp()
{
    auto CenterLUTOpReset = [&](std::unique_ptr<AscendOperator> &op) {
        AscendOpDesc desc("MatmulAtFP32");
        std::vector<int64_t> codesShape({ numLists * dims / SCAN_BIT, SCAN_BIT });
        std::vector<int64_t> matrixShape({ SCAN_BIT, LUT_NUM });
        
        std::vector<int64_t> lutResultShape({ numLists * dims / SCAN_BIT, LUT_NUM });
        desc.addInputTensorDesc(ACL_FLOAT, codesShape.size(), codesShape.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_FLOAT, matrixShape.size(), matrixShape.data(), ACL_FORMAT_ND);
        desc.addOutputTensorDesc(ACL_FLOAT, lutResultShape.size(), lutResultShape.data(), ACL_FORMAT_ND);
        op.reset();
        op = CREATE_UNIQUE_PTR(AscendOperator, desc);
        return op->init();
    };

    ivfCenterLUTOps = std::unique_ptr<AscendOperator>(nullptr);
    APPERR_RETURN_IF_NOT_LOG(CenterLUTOpReset(ivfCenterLUTOps),
                             APP_ERR_ACL_OP_LOAD_MODEL_FAILED, "IVFRabitQ Center LUT FP32 op init failed");
    return APP_ERR_OK;
}

void IndexIVFRabitQ::runCenterLUTOp(AscendTensor<float, DIMS_2> &centroid,
                                    AscendTensor<float, DIMS_2> &matrix,
                                    AscendTensor<float, DIMS_2> &centroidlut,
                                    aclrtStream stream)
{
    AscendOperator *op = ivfCenterLUTOps.get();
    ASCEND_THROW_IF_NOT(op);
    std::shared_ptr<std::vector<const aclDataBuffer *>> topkOpInput(
        new std::vector<const aclDataBuffer *>(), CommonUtils::AclInputBufferDelete);
    topkOpInput->emplace_back(aclCreateDataBuffer(centroid.data(), centroid.getSizeInBytes()));
    topkOpInput->emplace_back(aclCreateDataBuffer(matrix.data(), matrix.getSizeInBytes()));

    std::shared_ptr<std::vector<aclDataBuffer *>> topkOpOutput(
        new std::vector<aclDataBuffer *>(), CommonUtils::AclOutputBufferDelete);
    topkOpOutput->emplace_back(aclCreateDataBuffer(centroidlut.data(), centroidlut.getSizeInBytes()));
    op->exec(*topkOpInput, *topkOpOutput, stream);
    return;
}

size_t IndexIVFRabitQ::addTiling(int listId, size_t numVecs,
                                 std::vector<uint64_t>& offsetHost,
                                 std::vector<uint64_t>& indexl2offsetHost,
                                 std::vector<uint32_t>& baseSizeHost)
{
    size_t ntotal = listVecNum[listId];
    size_t blockIdx = ntotal / blockSize;
    size_t offsetInBlock = ntotal % blockSize;
    size_t leftInBlock = blockSize - offsetInBlock;
    size_t tileNum = 1;
    if (leftInBlock < numVecs) {
        tileNum += (numVecs - leftInBlock + blockSize - 1) / blockSize;
    }
    offsetHost.resize(tileNum);
    indexl2offsetHost.resize(tileNum);
    baseSizeHost.resize(tileNum);

    baseSizeHost[0] = std::min(leftInBlock, numVecs);
    indexl2offsetHost[0] = ntotal;
    offsetHost[0] = (reinterpret_cast<uint64_t>(baseFp32[listId].at(blockIdx)->data() +
                                                offsetInBlock * ((dims + 7) / 8)) -
                     reinterpret_cast<uint64_t>(pBaseFp32));
    for (int i = 1; i < tileNum - 1; i++) {
        baseSizeHost[i] = blockSize;
        indexl2offsetHost[i] = indexl2offsetHost[i - 1] + baseSizeHost[i - 1];
        blockIdx++;
        offsetHost[i] = (reinterpret_cast<uint64_t>(baseFp32[listId].at(blockIdx)->data()) -
                                        reinterpret_cast<uint64_t>(pBaseFp32));
    }
    if (tileNum > 1) {
        baseSizeHost[tileNum - 1] = (numVecs - leftInBlock) % blockSize;
        if (baseSizeHost[tileNum - 1] == 0) {
            baseSizeHost[tileNum - 1] = blockSize;
        }
        indexl2offsetHost[tileNum - 1] = indexl2offsetHost[tileNum - 2] + baseSizeHost[tileNum - 2];
        blockIdx++;
        offsetHost[tileNum - 1] = (reinterpret_cast<uint64_t>(baseFp32[listId].at(blockIdx)->data()) -
                                        reinterpret_cast<uint64_t>(pBaseFp32));
    }
    return tileNum;
}

APP_ERROR IndexIVFRabitQ::addVectors(int listId, size_t numVecs, const float *codes, const idx_t *indices)
{
    APPERR_RETURN_IF_NOT_FMT(listId >= 0 && listId < numLists, APP_ERR_INVALID_PARAM,
        "the listId is %d, out of numLists(%d)", listId, numLists);
    APPERR_RETURN_IF(numVecs == 0, APP_ERR_OK);
    APPERR_RETURN_IF_NOT_LOG(APP_ERR_OK == resize(listId, numVecs), APP_ERR_INNER_ERROR, "resize base failed!");
    std::vector<uint64_t> offsetHost;
    std::vector<uint64_t> indexl2offsetHost;
    std::vector<uint32_t> baseSizeHost;
    size_t tileNum = addTiling(listId, numVecs, offsetHost, indexl2offsetHost, baseSizeHost);

    auto &mem = resources.getMemoryManager();
    auto streamPtr = resources.getDefaultStream();
    auto stream = streamPtr->GetStream();
    AscendTensor<float, DIMS_2> rotatematrixDev(OrthogonalMatrixOnDevice->data(), {dims, dims});
    AscendTensor<float, DIMS_2> centroidDev(centroidsOnDevice->data() + listId * dims, {1, dims});
    AscendTensor<float, DIMS_1> centroidl2Dev(CentroidL2OnDevice->data() + listId, {1});
    AscendTensor<int32_t, DIMS_1> vectorSizeVec(mem, {1}, stream);
    AscendTensor<float, DIMS_2> indexOriginVec(mem, {blockSize, dims}, stream);
    AscendTensor<float, DIMS_2> indexRotateVec(mem, {blockSize, dims}, stream);
    AscendTensor<float, DIMS_1> indexL2Vec(mem, {blockSize}, stream);
    std::vector<int32_t> vectorSize(1, 0);
    uint64_t codesOffset = 0;
    for (int i = 0; i < tileNum; i++) {
        // 原始索引向量传到npu
        auto ret = aclrtMemcpy(indexOriginVec.data(), baseSizeHost[i] * dims * sizeof(float), codes + codesOffset,
                               baseSizeHost[i] * dims * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE);
        APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "copy index to device failed %d", ret);
        codesOffset += baseSizeHost[i] * dims;
        vectorSize[0] = baseSizeHost[i];
        ret = aclrtMemcpy(vectorSizeVec.data(), sizeof(int32_t),
                          vectorSize.data(), sizeof(int32_t), ACL_MEMCPY_HOST_TO_DEVICE);
        APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "copy vectorSize to device failed %d", ret);
        AscendTensor<float, DIMS_1> indexesl2(IndexL2OnDevice[listId]->data() + indexl2offsetHost[i], {blockSize});
        AscendTensor<float, DIMS_1> indexesl1(IndexL1OnDevice[listId]->data() + indexl2offsetHost[i], {blockSize});
        AscendTensor<uint8_t, DIMS_2> indexCodes((uint8_t *)(reinterpret_cast<uint64_t>(pBaseFp32) +
                                                             offsetHost[i]), {blockSize, dims/8});
        // o = Por, ||or||^2 预计算
        runIndexRotateL2Op(indexOriginVec, vectorSizeVec, rotatematrixDev, indexRotateVec, indexL2Vec, stream);
        ret = synchronizeStream(stream);
        APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR,
            "synchronizeStream default stream: %i\n", ret);
        runIndexCodeAndPreComputeOp(vectorSizeVec, indexRotateVec, indexL2Vec, centroidDev,
                                    centroidl2Dev, indexCodes, indexesl2, indexesl1, stream);
        ret = synchronizeStream(stream);
        APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR,
            "synchronizeStream default stream: %i\n", ret);
    }
    maxListLength = std::max(maxListLength, static_cast<int>(getListLength(listId)));
    maxListLength = utils::roundUp(maxListLength, CUBE_ALIGN);
    listVecNum[listId] += numVecs;
    IndexIVF::ntotal += numVecs;
    deviceListIndices[listId]->append(indices, numVecs, true);
    return APP_ERR_OK;
}

APP_ERROR IndexIVFRabitQ::resetIndexRotateL2Op()
{
    auto IndexRotateL2OpReset = [&](std::unique_ptr<AscendOperator> &op) {
        AscendOpDesc desc("RotateAndL2AtFP32");
        std::vector<int64_t> codesShape({ blockSize, dims });
        std::vector<int64_t> sizeShape({ 1 });
        std::vector<int64_t> matrixShape({ dims, dims });
        
        std::vector<int64_t> rotateResultShape({ blockSize, dims });
        std::vector<int64_t> l2ResultShape({ blockSize });
        desc.addInputTensorDesc(ACL_FLOAT, codesShape.size(), codesShape.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_INT32, sizeShape.size(), sizeShape.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_FLOAT, matrixShape.size(), matrixShape.data(), ACL_FORMAT_ND);
        desc.addOutputTensorDesc(ACL_FLOAT, rotateResultShape.size(), rotateResultShape.data(), ACL_FORMAT_ND);
        desc.addOutputTensorDesc(ACL_FLOAT, l2ResultShape.size(), l2ResultShape.data(), ACL_FORMAT_ND);
        op.reset();
        op = CREATE_UNIQUE_PTR(AscendOperator, desc);
        return op->init();
    };

    ivfRabitQIndexRotateL2Ops = std::unique_ptr<AscendOperator>(nullptr);
    APPERR_RETURN_IF_NOT_LOG(IndexRotateL2OpReset(ivfRabitQIndexRotateL2Ops),
                             APP_ERR_ACL_OP_LOAD_MODEL_FAILED, "IVFRabitQ Rotate L2 FP32 distance op init failed");
    return APP_ERR_OK;
}

void IndexIVFRabitQ::runIndexRotateL2Op(AscendTensor<float, DIMS_2> &index,
                                        AscendTensor<int32_t, DIMS_1> &vectorSize,
                                        AscendTensor<float, DIMS_2> &matrix,
                                        AscendTensor<float, DIMS_2> &rotateIndex,
                                        AscendTensor<float, DIMS_1> &indexl2,
                                        aclrtStream stream)
{
    AscendOperator *op = ivfRabitQIndexRotateL2Ops.get();
    ASCEND_THROW_IF_NOT(op);
    std::shared_ptr<std::vector<const aclDataBuffer *>> topkOpInput(
        new std::vector<const aclDataBuffer *>(), CommonUtils::AclInputBufferDelete);
    topkOpInput->emplace_back(aclCreateDataBuffer(index.data(), index.getSizeInBytes()));
    topkOpInput->emplace_back(aclCreateDataBuffer(vectorSize.data(), vectorSize.getSizeInBytes()));
    topkOpInput->emplace_back(aclCreateDataBuffer(matrix.data(), matrix.getSizeInBytes()));

    std::shared_ptr<std::vector<aclDataBuffer *>> topkOpOutput(
        new std::vector<aclDataBuffer *>(), CommonUtils::AclOutputBufferDelete);
    topkOpOutput->emplace_back(aclCreateDataBuffer(rotateIndex.data(), rotateIndex.getSizeInBytes()));
    topkOpOutput->emplace_back(aclCreateDataBuffer(indexl2.data(), indexl2.getSizeInBytes()));
    op->exec(*topkOpInput, *topkOpOutput, stream);
    return;
}

APP_ERROR IndexIVFRabitQ::resetIndexCodeAndPreComputeOp()
{
    auto IndexCodeAndPreComputeOpReset = [&](std::unique_ptr<AscendOperator> &op) {
        AscendOpDesc desc("IndexCodeAndPrecompute");
        std::vector<int64_t> sizeShape({ 1 });
        std::vector<int64_t> indexShape({ blockSize, dims });
        std::vector<int64_t> indexl2Shape({ blockSize });
        std::vector<int64_t> centroidShape({ 1, dims });
        std::vector<int64_t> centroidl2Shape({ 1 });
        
        std::vector<int64_t> indexCodeShape({ blockSize, dims/8 });
        std::vector<int64_t> l2ResultShape({ blockSize });
        std::vector<int64_t> l1ResultShape({ blockSize });
        desc.addInputTensorDesc(ACL_INT32, sizeShape.size(), sizeShape.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_FLOAT, indexShape.size(), indexShape.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_FLOAT, indexl2Shape.size(), indexl2Shape.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_FLOAT, centroidShape.size(), centroidShape.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_FLOAT, centroidl2Shape.size(), centroidl2Shape.data(), ACL_FORMAT_ND);
        desc.addOutputTensorDesc(ACL_UINT8, indexCodeShape.size(), indexCodeShape.data(), ACL_FORMAT_ND);
        desc.addOutputTensorDesc(ACL_FLOAT, l2ResultShape.size(), l2ResultShape.data(), ACL_FORMAT_ND);
        desc.addOutputTensorDesc(ACL_FLOAT, l1ResultShape.size(), l1ResultShape.data(), ACL_FORMAT_ND);
        op.reset();
        op = CREATE_UNIQUE_PTR(AscendOperator, desc);
        return op->init();
    };

    ivfRabitQIndexCodeAndPreComputeOps = std::unique_ptr<AscendOperator>(nullptr);
    APPERR_RETURN_IF_NOT_LOG(IndexCodeAndPreComputeOpReset(ivfRabitQIndexCodeAndPreComputeOps),
                             APP_ERR_ACL_OP_LOAD_MODEL_FAILED, "IVFRabitQ Index PreCompute FP32 op init failed");
    return APP_ERR_OK;
}

void IndexIVFRabitQ::runIndexCodeAndPreComputeOp(AscendTensor<int32_t, DIMS_1> &vectorSize,
                                                 AscendTensor<float, DIMS_2> &index,
                                                 AscendTensor<float, DIMS_1> &indexl2,
                                                 AscendTensor<float, DIMS_2> &centroid,
                                                 AscendTensor<float, DIMS_1> &centroidl2,
                                                 AscendTensor<uint8_t, DIMS_2> &indexCodes,
                                                 AscendTensor<float, DIMS_1> &l2Result,
                                                 AscendTensor<float, DIMS_1> &l1Result,
                                                 aclrtStream stream)
{
    AscendOperator *op = ivfRabitQIndexCodeAndPreComputeOps.get();
    ASCEND_THROW_IF_NOT(op);
    std::shared_ptr<std::vector<const aclDataBuffer *>> topkOpInput(
        new std::vector<const aclDataBuffer *>(), CommonUtils::AclInputBufferDelete);
    topkOpInput->emplace_back(aclCreateDataBuffer(vectorSize.data(), vectorSize.getSizeInBytes()));
    topkOpInput->emplace_back(aclCreateDataBuffer(index.data(), index.getSizeInBytes()));
    topkOpInput->emplace_back(aclCreateDataBuffer(indexl2.data(), indexl2.getSizeInBytes()));
    topkOpInput->emplace_back(aclCreateDataBuffer(centroid.data(), centroid.getSizeInBytes()));
    topkOpInput->emplace_back(aclCreateDataBuffer(centroidl2.data(), centroidl2.getSizeInBytes()));

    std::shared_ptr<std::vector<aclDataBuffer *>> topkOpOutput(
        new std::vector<aclDataBuffer *>(), CommonUtils::AclOutputBufferDelete);
    topkOpOutput->emplace_back(aclCreateDataBuffer(indexCodes.data(), indexCodes.getSizeInBytes()));
    topkOpOutput->emplace_back(aclCreateDataBuffer(l2Result.data(), l2Result.getSizeInBytes()));
    topkOpOutput->emplace_back(aclCreateDataBuffer(l1Result.data(), l1Result.getSizeInBytes()));
    op->exec(*topkOpInput, *topkOpOutput, stream);
    return;
}

APP_ERROR IndexIVFRabitQ::resetQueryRotateOp()
{
    auto queryRotateOpReset = [&](std::unique_ptr<AscendOperator> &op, int batch) {
        AscendOpDesc desc("MatmulAtFP32");
        std::vector<int64_t> queryShape({ batch, dims });
        std::vector<int64_t> matrixShape({ dims, dims });
        
        std::vector<int64_t> rotateResultShape({ batch, dims });
        desc.addInputTensorDesc(ACL_FLOAT, queryShape.size(), queryShape.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_FLOAT, matrixShape.size(), matrixShape.data(), ACL_FORMAT_ND);
        desc.addOutputTensorDesc(ACL_FLOAT, rotateResultShape.size(), rotateResultShape.data(), ACL_FORMAT_ND);
        op.reset();
        op = CREATE_UNIQUE_PTR(AscendOperator, desc);
        return op->init();
    };
    for (auto batch: searchBatchSizes) {
        queryRotateOps[batch] = std::unique_ptr<AscendOperator>(nullptr);
        APPERR_RETURN_IF_NOT_LOG(queryRotateOpReset(queryRotateOps[batch], batch),
                                 APP_ERR_ACL_OP_LOAD_MODEL_FAILED, "L1 distance op init failed");
    }
    return APP_ERR_OK;
}

void IndexIVFRabitQ::runQueryRotateOp(int batch, AscendTensor<float, DIMS_2> &queries,
                                      AscendTensor<float, DIMS_2> &matrix,
                                      AscendTensor<float, DIMS_2> &rotateQueries,
                                      aclrtStream stream)
{
    AscendOperator *op = nullptr;
    if (queryRotateOps.find(batch) != queryRotateOps.end()) {
        op = queryRotateOps[batch].get();
    }
    ASCEND_THROW_IF_NOT(op);
    std::shared_ptr<std::vector<const aclDataBuffer *>> topkOpInput(
        new std::vector<const aclDataBuffer *>(), CommonUtils::AclInputBufferDelete);
    topkOpInput->emplace_back(aclCreateDataBuffer(queries.data(), queries.getSizeInBytes()));
    topkOpInput->emplace_back(aclCreateDataBuffer(matrix.data(), matrix.getSizeInBytes()));

    std::shared_ptr<std::vector<aclDataBuffer *>> topkOpOutput(
        new std::vector<aclDataBuffer *>(), CommonUtils::AclOutputBufferDelete);
    topkOpOutput->emplace_back(aclCreateDataBuffer(rotateQueries.data(), rotateQueries.getSizeInBytes()));
    op->exec(*topkOpInput, *topkOpOutput, stream);
    return;
}

APP_ERROR IndexIVFRabitQ::resetQueryLUTOp()
{
    auto queryLUTOpReset = [&](std::unique_ptr<AscendOperator> &op, int batch) {
        AscendOpDesc desc("MatmulAtFP32");
        std::vector<int64_t> queryShape({ batch * dims / SCAN_BIT, SCAN_BIT });
        std::vector<int64_t> matrixShape({ SCAN_BIT, LUT_NUM });
        
        std::vector<int64_t> lutResultShape({ batch * dims / SCAN_BIT, LUT_NUM });
        desc.addInputTensorDesc(ACL_FLOAT, queryShape.size(), queryShape.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_FLOAT, matrixShape.size(), matrixShape.data(), ACL_FORMAT_ND);
        desc.addOutputTensorDesc(ACL_FLOAT, lutResultShape.size(), lutResultShape.data(), ACL_FORMAT_ND);
        op.reset();
        op = CREATE_UNIQUE_PTR(AscendOperator, desc);
        return op->init();
    };
    for (auto batch: searchBatchSizes) {
        queryLUTOps[batch] = std::unique_ptr<AscendOperator>(nullptr);
        APPERR_RETURN_IF_NOT_LOG(queryLUTOpReset(queryLUTOps[batch], batch),
                                 APP_ERR_ACL_OP_LOAD_MODEL_FAILED, "IVFRabitQ Query LUT FP32 op init failed");
    }
    return APP_ERR_OK;
}

void IndexIVFRabitQ::runQueryLUTOp(int batch, AscendTensor<float, DIMS_2> &queries,
                                   AscendTensor<float, DIMS_2> &matrix,
                                   AscendTensor<float, DIMS_2> &querieslut,
                                   aclrtStream stream)
{
    AscendOperator *op = nullptr;
    if (queryLUTOps.find(batch) != queryLUTOps.end()) {
        op = queryLUTOps[batch].get();
    }
    ASCEND_THROW_IF_NOT(op);
    std::shared_ptr<std::vector<const aclDataBuffer *>> topkOpInput(
        new std::vector<const aclDataBuffer *>(), CommonUtils::AclInputBufferDelete);
    topkOpInput->emplace_back(aclCreateDataBuffer(queries.data(), queries.getSizeInBytes()));
    topkOpInput->emplace_back(aclCreateDataBuffer(matrix.data(), matrix.getSizeInBytes()));

    std::shared_ptr<std::vector<aclDataBuffer *>> topkOpOutput(
        new std::vector<aclDataBuffer *>(), CommonUtils::AclOutputBufferDelete);
    topkOpOutput->emplace_back(aclCreateDataBuffer(querieslut.data(), querieslut.getSizeInBytes()));
    op->exec(*topkOpInput, *topkOpOutput, stream);
    return;
}

APP_ERROR IndexIVFRabitQ::resetL1TopkOp()
{
    auto topkCompOpReset = [&](std::unique_ptr<AscendOperator> &op, int64_t batch) {
        AscendOpDesc desc("TopkFlatFp32");
        std::vector<int64_t> shape0 { 1, batch, numLists };
        std::vector<int64_t> shape1 { 1, batch,  numLists / IVF_RABITQ_BURST_LEN * 2 };
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

void IndexIVFRabitQ::runL1TopkOp(AscendTensor<float, DIMS_2> &dists,
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
    AscendTensor<float, DIMS_3> vmdistsTopk(vmdists.data(), {1, batch,
                                            std::min(numLists / IVF_RABITQ_BURST_LEN * 2, MIN_EXTREME_SIZE)});
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

APP_ERROR IndexIVFRabitQ::resetL1DistOp()
{
    auto l1DisOpReset = [&](std::unique_ptr<AscendOperator> &op, int batch) {
        AscendOpDesc desc("DistanceFlatL2MinsAtFP32");
        std::vector<int64_t> queryShape({ batch, dims });
        std::vector<int64_t> codesShape({ numLists, dims });
        std::vector<int64_t> codesSqrSum({ numLists });
        std::vector<int64_t> distResultShape({ batch, numLists });
        std::vector<int64_t> minResultShape({ batch, numLists / IVF_RABITQ_BURST_LEN * 2});
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

void IndexIVFRabitQ::runL1DistOp(int batch, AscendTensor<float, DIMS_2> &queries,
                                 AscendTensor<float, DIMS_2> &centroidsDev, AscendTensor<float, DIMS_2> &dists,
                                 AscendTensor<float, DIMS_2> &vmdists, AscendTensor<uint16_t, DIMS_2> &opFlag,
                                 aclrtStream stream)
{
    AscendOperator *op = nullptr;
    if (l1DistFp32Ops.find(batch) != l1DistFp32Ops.end()) {
        op = l1DistFp32Ops[batch].get();
    }
    ASCEND_THROW_IF_NOT(op);
    AscendTensor<float, DIMS_1> centroidsSqrSum(CentroidL2OnDevice->data(), {numLists});
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

APP_ERROR IndexIVFRabitQ::resetL2TopkOp()
{
    auto topkCompOpReset = [&](std::unique_ptr<AscendOperator> &op, int64_t batch) {
        AscendOpDesc desc("TopkIvfRabitqFp32");
        std::vector<int64_t> shape0 { 0, IVF_RABITQ_BLOCK_SIZE };
        std::vector<int64_t> shape1 { 0, (IVF_RABITQ_BLOCK_SIZE + IVF_RABITQ_BURST_LEN -1) / IVF_RABITQ_BURST_LEN * 2};
        std::vector<int64_t> shape2 { 0, 1 };
        std::vector<int64_t> shape3 { 0, 1 };
        std::vector<int64_t> shape4 { 0, 16 };
        std::vector<int64_t> shape5 { aicpu::TOPK_IVF_RABITQ_ATTR_IDX_COUNT };

        std::vector<int64_t> shape6 { batch, 0 };

        desc.addInputTensorDesc(ACL_FLOAT, shape0.size(), shape0.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_FLOAT, shape1.size(), shape1.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_INT64, shape2.size(), shape2.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_UINT32, shape3.size(), shape3.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_UINT16, shape4.size(), shape4.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_INT64, shape5.size(), shape5.data(), ACL_FORMAT_ND);

        desc.addOutputTensorDesc(ACL_FLOAT, shape6.size(), shape6.data(), ACL_FORMAT_ND);
        desc.addOutputTensorDesc(ACL_UINT64, shape6.size(), shape6.data(), ACL_FORMAT_ND);

        op.reset();
        op = CREATE_UNIQUE_PTR(AscendOperator, desc);
        return op->init();
    };

    for (auto batch : searchBatchSizes) {
        topkL2Fp32[batch] = std::unique_ptr<AscendOperator>(nullptr);
        APPERR_RETURN_IF_NOT_LOG(topkCompOpReset(topkL2Fp32[batch], batch),
                                 APP_ERR_ACL_OP_LOAD_MODEL_FAILED, "l2 topk op init failed");
    }

    return APP_ERR_OK;
}

void IndexIVFRabitQ::runL2TopkOp(int batch, AscendTensor<float, DIMS_2, size_t> &distResult,
                                 AscendTensor<float, DIMS_2, size_t> &vmdistResult,
                                 AscendTensor<int64_t, DIMS_2, size_t> &ids,
                                 AscendTensor<uint32_t, DIMS_2, size_t> &sizes,
                                 AscendTensor<uint16_t, DIMS_2, size_t> &flags,
                                 AscendTensor<int64_t, DIMS_1> &attrs,
                                 AscendTensor<float, DIMS_2, size_t> &outdists,
                                 AscendTensor<uint64_t, DIMS_2, size_t> &outlabel,
                                 aclrtStream stream)
{
    std::vector<const AscendTensorBase *> input{&distResult, &vmdistResult, &ids, &sizes, &flags, &attrs};
    std::vector<const AscendTensorBase *> output{&outdists, &outlabel};
    AscendOperator *op = nullptr;
    if (topkL2Fp32.find(batch) != topkL2Fp32.end()) {
        op = topkL2Fp32[batch].get();
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

bool IndexIVFRabitQ::l2DisOpReset(std::unique_ptr<AscendOperator> &op, int64_t batch)
{
    AscendOpDesc desc;
    std::vector<int64_t> queryShape({ 1, dims });
    std::vector<int64_t> querylutShape;
    std::vector<int64_t> centroidslutShape;
    std::vector<int64_t> queryidShape({ CORE_NUM });
    std::vector<int64_t> centroidsidShape({ CORE_NUM });
    std::vector<int64_t> centroidsl2Shape({ CORE_NUM });
    std::vector<int64_t> codesShape({ IVF_RABITQ_BLOCK_SIZE, dims/8 });
    std::vector<int64_t> offsetShape({CORE_NUM});
    std::vector<int64_t> sizeShape({CORE_NUM});
    std::vector<int64_t> indexl2Shape({IVF_RABITQ_BLOCK_SIZE});
    std::vector<int64_t> indexl1Shape({IVF_RABITQ_BLOCK_SIZE});
    std::vector<int64_t> indexesl2offsetShape({CORE_NUM});
    std::vector<int64_t> indexesl1offsetShape({CORE_NUM});
    std::vector<int64_t> distResultShape({ CORE_NUM, IVF_RABITQ_BLOCK_SIZE });
    std::vector<int64_t> mixResultShape({ CORE_NUM,
        (IVF_RABITQ_BLOCK_SIZE + IVF_RABITQ_BURST_LEN -1) / IVF_RABITQ_BURST_LEN * 2});
    std::vector<int64_t> flagShapeShape({ CORE_NUM, 16 });
    if (faiss::ascend::SocUtils::GetInstance().IsAscendA5()) {
        desc.setOpName("DistanceIVFRabitqL2FP32Simt");
        querylutShape = std::vector<int64_t>({ batch * dims / SCAN_BIT, LUT_NUM });
        centroidslutShape = std::vector<int64_t>({ numLists * dims / SCAN_BIT, LUT_NUM });
    } else {
        desc.setOpName("DistanceIVFRabitqL2FP32");
        querylutShape = std::vector<int64_t>({ batch, dims });
        centroidslutShape = std::vector<int64_t>({ numLists, dims }); // YCY: select
    }
    desc.addInputTensorDesc(ACL_FLOAT, queryShape.size(), queryShape.data(), ACL_FORMAT_ND);
    desc.addInputTensorDesc(ACL_FLOAT, querylutShape.size(), querylutShape.data(), ACL_FORMAT_ND);
    desc.addInputTensorDesc(ACL_FLOAT, centroidslutShape.size(), centroidslutShape.data(), ACL_FORMAT_ND);
    desc.addInputTensorDesc(ACL_UINT32, queryidShape.size(), queryidShape.data(), ACL_FORMAT_ND);
    desc.addInputTensorDesc(ACL_UINT32, centroidsidShape.size(), centroidsidShape.data(), ACL_FORMAT_ND);
    desc.addInputTensorDesc(ACL_FLOAT, centroidsl2Shape.size(), centroidsl2Shape.data(), ACL_FORMAT_ND);
    desc.addInputTensorDesc(ACL_UINT8, codesShape.size(), codesShape.data(), ACL_FORMAT_ND);
    desc.addInputTensorDesc(ACL_UINT64, offsetShape.size(), offsetShape.data(), ACL_FORMAT_ND);
    desc.addInputTensorDesc(ACL_UINT32, sizeShape.size(), sizeShape.data(), ACL_FORMAT_ND);
    desc.addInputTensorDesc(ACL_FLOAT, indexl2Shape.size(), indexl2Shape.data(), ACL_FORMAT_ND);
    desc.addInputTensorDesc(ACL_FLOAT, indexl1Shape.size(), indexl1Shape.data(), ACL_FORMAT_ND);
    desc.addInputTensorDesc(ACL_UINT64, indexesl2offsetShape.size(), indexesl2offsetShape.data(), ACL_FORMAT_ND);
    desc.addInputTensorDesc(ACL_UINT64, indexesl1offsetShape.size(), indexesl1offsetShape.data(), ACL_FORMAT_ND);
    desc.addOutputTensorDesc(ACL_FLOAT, distResultShape.size(), distResultShape.data(), ACL_FORMAT_ND);
    desc.addOutputTensorDesc(ACL_FLOAT, mixResultShape.size(), mixResultShape.data(), ACL_FORMAT_ND);
    desc.addOutputTensorDesc(ACL_UINT16, flagShapeShape.size(), flagShapeShape.data(), ACL_FORMAT_ND);
    op.reset();
    op = CREATE_UNIQUE_PTR(AscendOperator, desc);
    return op->init();
}

APP_ERROR IndexIVFRabitQ::resetL2DistOp()
{
    for (auto batch : searchBatchSizes) {
        ivfRabitqL2DistOps[batch] = std::unique_ptr<AscendOperator>(nullptr);
        APPERR_RETURN_IF_NOT_LOG(l2DisOpReset(ivfRabitqL2DistOps[batch], batch),
                                 APP_ERR_ACL_OP_LOAD_MODEL_FAILED, "l2 dist op init failed");
    }
    return APP_ERR_OK;
}

void IndexIVFRabitQ::runL2DistOp(AscendTensor<float, DIMS_2> &subQuery,
                                 AscendTensor<float, DIMS_2> &subQuerylut,
                                 AscendTensor<float, DIMS_2, size_t> &centroidslut,
                                 AscendTensor<uint32_t, DIMS_1, size_t> &subQueryid,
                                 AscendTensor<uint32_t, DIMS_1, size_t> &subCentroidsid,
                                 AscendTensor<float, DIMS_1, size_t> &subCentroidsl2,
                                 AscendTensor<uint8_t, DIMS_2, size_t> &codeVec,
                                 AscendTensor<uint64_t, DIMS_1, size_t> &subOffset,
                                 AscendTensor<uint32_t, DIMS_1, size_t> &subBaseSize,
                                 AscendTensor<float, DIMS_1, size_t> &indexl2,
                                 AscendTensor<float, DIMS_1, size_t> &indexl1,
                                 AscendTensor<uint64_t, DIMS_1, size_t> &subIndexl2Offset,
                                 AscendTensor<uint64_t, DIMS_1, size_t> &subIndexl1Offset,
                                 AscendTensor<float, DIMS_2, size_t> &subDis,
                                 AscendTensor<float, DIMS_2, size_t> &subVcMaxDis,
                                 AscendTensor<uint16_t, DIMS_2, size_t> &subOpFlag,
                                 aclrtStream stream)
{
    int batch;
    if (faiss::ascend::SocUtils::GetInstance().IsAscendA5()) {
        batch = static_cast<int>(subQuerylut.getSize(0) / (dims / SCAN_BIT));
    } else {
        batch = static_cast<int>(subQuerylut.getSize(0));
    }
    AscendOperator *op = nullptr;
    if (ivfRabitqL2DistOps.find(batch) != ivfRabitqL2DistOps.end()) {
        op = ivfRabitqL2DistOps[batch].get();
    }
    ASCEND_THROW_IF_NOT(op);
    std::shared_ptr<std::vector<const aclDataBuffer *>> topkOpInput(
        new std::vector<const aclDataBuffer *>(), CommonUtils::AclInputBufferDelete);
    topkOpInput->emplace_back(aclCreateDataBuffer(subQuery.data(), subQuery.getSizeInBytes()));
    topkOpInput->emplace_back(aclCreateDataBuffer(subQuerylut.data(), subQuerylut.getSizeInBytes()));
    topkOpInput->emplace_back(aclCreateDataBuffer(centroidslut.data(), centroidslut.getSizeInBytes()));
    topkOpInput->emplace_back(aclCreateDataBuffer(subQueryid.data(), subQueryid.getSizeInBytes()));
    topkOpInput->emplace_back(aclCreateDataBuffer(subCentroidsid.data(), subCentroidsid.getSizeInBytes()));
    topkOpInput->emplace_back(aclCreateDataBuffer(subCentroidsl2.data(), subCentroidsl2.getSizeInBytes()));
    topkOpInput->emplace_back(aclCreateDataBuffer(codeVec.data(), codeVec.getSizeInBytes()));
    topkOpInput->emplace_back(aclCreateDataBuffer(subOffset.data(), subOffset.getSizeInBytes()));
    topkOpInput->emplace_back(aclCreateDataBuffer(subBaseSize.data(), subBaseSize.getSizeInBytes()));
    topkOpInput->emplace_back(aclCreateDataBuffer(indexl2.data(), indexl2.getSizeInBytes()));
    topkOpInput->emplace_back(aclCreateDataBuffer(indexl1.data(), indexl1.getSizeInBytes()));
    topkOpInput->emplace_back(aclCreateDataBuffer(subIndexl2Offset.data(), subIndexl2Offset.getSizeInBytes()));
    topkOpInput->emplace_back(aclCreateDataBuffer(subIndexl1Offset.data(), subIndexl1Offset.getSizeInBytes()));

    std::shared_ptr<std::vector<aclDataBuffer *>> topkOpOutput(
        new std::vector<aclDataBuffer *>(), CommonUtils::AclOutputBufferDelete);
    topkOpOutput->emplace_back(aclCreateDataBuffer(subDis.data(), subDis.getSizeInBytes()));
    topkOpOutput->emplace_back(aclCreateDataBuffer(subVcMaxDis.data(), subVcMaxDis.getSizeInBytes()));
    topkOpOutput->emplace_back(aclCreateDataBuffer(subOpFlag.data(), subOpFlag.getSizeInBytes()));
    op->exec(*topkOpInput, *topkOpOutput, stream);
    return;
}

void IndexIVFRabitQ::fillDisOpInputDataByBlock(size_t batch, size_t segNum, size_t coreNum, size_t ivfRabitqBlockSize,
                                               AscendTensor<uint32_t, DIMS_2, size_t> &queryidHostVec,
                                               AscendTensor<uint32_t, DIMS_2, size_t> &centroidsidHostVec,
                                               AscendTensor<float, DIMS_2, size_t> &centroidsl2HostVec,
                                               AscendTensor<uint32_t, DIMS_2, size_t> &baseSizeHostVec,
                                               AscendTensor<uint64_t, DIMS_2, size_t> &offsetHostVec,
                                               AscendTensor<uint64_t, DIMS_2, size_t> &indexl2OffsetHostVec,
                                               AscendTensor<uint64_t, DIMS_2, size_t> &indexl1OffsetHostVec,
                                               AscendTensor<int64_t, DIMS_2, size_t> &idsHostVec,
                                               AscendTensor<int64_t, DIMS_2> &l1TopNprobeIndicesHost,
                                               AscendTensor<float, DIMS_2> &l1TopNprobeDistsHost)
{
    size_t cIdx = 0;
    size_t iter = 0;
    for (size_t qIdx = 0; qIdx < batch; qIdx++) {
        for (size_t tIdx = 0; tIdx < nprobe; tIdx++) {
            int64_t listId = l1TopNprobeIndicesHost[qIdx][tIdx].value();
            float centerl2 = l1TopNprobeDistsHost[qIdx][tIdx].value();
            size_t blockNum = baseFp32[listId].size();
            for (size_t segIdx = 0; segIdx < blockNum; segIdx++) {
                queryidHostVec[iter][cIdx].value(qIdx);
                centroidsidHostVec[iter][cIdx].value(listId);
                centroidsl2HostVec[iter][cIdx].value(centerl2);
                size_t listNum = deviceListIndices[listId]->size();
                size_t proccessLen = std::min(listNum - segIdx * ivfRabitqBlockSize, ivfRabitqBlockSize);
                baseSizeHostVec[iter][cIdx].value(proccessLen);
                int64_t idAddr = reinterpret_cast<int64_t>(deviceListIndices[listId]->data()
                                                           + segIdx * ivfRabitqBlockSize);
                idsHostVec[iter][cIdx].value(idAddr);

                uint64_t offsetSeg = reinterpret_cast<uint64_t>(baseFp32[listId].at(segIdx)->data()) -
                                        reinterpret_cast<uint64_t>(pBaseFp32);
                offsetHostVec[iter][cIdx].value(offsetSeg);
                uint64_t indexl2OffsetSeg = reinterpret_cast<uint64_t>(IndexL2OnDevice[listId]->data() +
                                        segIdx * ivfRabitqBlockSize) - reinterpret_cast<uint64_t>(pIndexL2);
                indexl2OffsetHostVec[iter][cIdx].value(indexl2OffsetSeg);
                uint64_t indexl1OffsetSeg = reinterpret_cast<uint64_t>(IndexL1OnDevice[listId]->data() +
                                        segIdx * ivfRabitqBlockSize) - reinterpret_cast<uint64_t>(pIndexL1);
                indexl1OffsetHostVec[iter][cIdx].value(indexl1OffsetSeg);

                cIdx++;
                iter += cIdx / coreNum;
                cIdx %= coreNum;
            }
            for (size_t segIdx = blockNum; segIdx < segNum; segIdx++) {
                cIdx++;
                iter += cIdx / coreNum;
                cIdx %= coreNum;
            }
        }
    }
}

APP_ERROR IndexIVFRabitQ::fillDisOpInputData(int k, size_t batch, size_t segNum, size_t coreNum,
                                             AscendTensor<uint64_t, DIMS_2, size_t> &offset,
                                             AscendTensor<uint64_t, DIMS_2, size_t> &indexl2offset,
                                             AscendTensor<uint64_t, DIMS_2, size_t> &indexl1offset,
                                             AscendTensor<uint32_t, DIMS_2, size_t> &queryid,
                                             AscendTensor<uint32_t, DIMS_2, size_t> &centroidsid,
                                             AscendTensor<float, DIMS_2, size_t> &centroidsl2,
                                             AscendTensor<uint32_t, DIMS_2, size_t> &baseSize,
                                             AscendTensor<int64_t, DIMS_2, size_t> &ids,
                                             AscendTensor<int64_t, DIMS_1> &attrs,
                                             AscendTensor<int64_t, DIMS_2> &l1TopNprobeIndicesHost,
                                             AscendTensor<float, DIMS_2> &l1TopNprobeDistsHost)
{
    size_t iterNum = (batch * nprobe * segNum + coreNum - 1)/ coreNum;
    size_t ivfRabitqBlockSize = static_cast<size_t>(IVF_RABITQ_BLOCK_SIZE);
    std::vector<uint64_t> offsetHost(iterNum * coreNum, 0);
    AscendTensor<uint64_t, DIMS_2, size_t> offsetHostVec(offsetHost.data(), {iterNum, coreNum});
    std::vector<uint64_t> indexl2OffsetHost(iterNum * coreNum, 0);
    AscendTensor<uint64_t, DIMS_2, size_t> indexl2OffsetHostVec(indexl2OffsetHost.data(), {iterNum, coreNum});
    std::vector<uint64_t> indexl1OffsetHost(iterNum * coreNum, 0);
    AscendTensor<uint64_t, DIMS_2, size_t> indexl1OffsetHostVec(indexl1OffsetHost.data(), {iterNum, coreNum});
    std::vector<uint32_t> queryidHost(iterNum * coreNum, 0);
    AscendTensor<uint32_t, DIMS_2, size_t> queryidHostVec(queryidHost.data(), {iterNum, coreNum});
    std::vector<uint32_t> centroidsidHost(iterNum * coreNum, 0);
    AscendTensor<uint32_t, DIMS_2, size_t> centroidsidHostVec(centroidsidHost.data(), {iterNum, coreNum});
    std::vector<float> centroidsl2Host(iterNum * coreNum, 0);
    AscendTensor<float, DIMS_2, size_t> centroidsl2HostVec(centroidsl2Host.data(), {iterNum, coreNum});
    std::vector<uint32_t> baseSizeHost(iterNum * coreNum, 0);
    AscendTensor<uint32_t, DIMS_2, size_t> baseSizeHostVec(baseSizeHost.data(), {iterNum, coreNum});
    std::vector<int64_t> idsHost(iterNum * coreNum, 0);
    AscendTensor<int64_t, DIMS_2, size_t> idsHostVec(idsHost.data(), {iterNum, coreNum});
    fillDisOpInputDataByBlock(batch, segNum, coreNum, ivfRabitqBlockSize, queryidHostVec, centroidsidHostVec,
                              centroidsl2HostVec, baseSizeHostVec, offsetHostVec, indexl2OffsetHostVec,
                              indexl1OffsetHostVec, idsHostVec, l1TopNprobeIndicesHost, l1TopNprobeDistsHost);
    auto ret = aclrtMemcpy(offset.data(), offset.getSizeInBytes(), offsetHostVec.data(),
                           offsetHostVec.getSizeInBytes(), ACL_MEMCPY_HOST_TO_DEVICE);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "copy offset to device failed %d", ret);
    ret = aclrtMemcpy(indexl2offset.data(), indexl2offset.getSizeInBytes(), indexl2OffsetHostVec.data(),
                      indexl2OffsetHostVec.getSizeInBytes(), ACL_MEMCPY_HOST_TO_DEVICE);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "copy indexl2offset to device failed %d", ret);
    ret = aclrtMemcpy(indexl1offset.data(), indexl1offset.getSizeInBytes(), indexl1OffsetHostVec.data(),
                      indexl1OffsetHostVec.getSizeInBytes(), ACL_MEMCPY_HOST_TO_DEVICE);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "copy indexl1offset to device failed %d", ret);
    ret = aclrtMemcpy(queryid.data(), queryid.getSizeInBytes(), queryidHostVec.data(),
                      queryidHostVec.getSizeInBytes(), ACL_MEMCPY_HOST_TO_DEVICE);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "copy queryid to device failed %d", ret);
    ret = aclrtMemcpy(centroidsid.data(), centroidsid.getSizeInBytes(), centroidsidHostVec.data(),
                      centroidsidHostVec.getSizeInBytes(), ACL_MEMCPY_HOST_TO_DEVICE);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "copy centroidsid to device failed %d", ret);
    ret = aclrtMemcpy(centroidsl2.data(), centroidsl2.getSizeInBytes(), centroidsl2HostVec.data(),
                      centroidsl2HostVec.getSizeInBytes(), ACL_MEMCPY_HOST_TO_DEVICE);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "copy centroidsl2 to device failed %d", ret);
    ret = aclrtMemcpy(baseSize.data(), baseSize.getSizeInBytes(), baseSizeHostVec.data(),
                      baseSizeHostVec.getSizeInBytes(), ACL_MEMCPY_HOST_TO_DEVICE);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "copy basesize to device failed %d", ret);
    ret = aclrtMemcpy(ids.data(), ids.getSizeInBytes(), idsHostVec.data(),
                      idsHostVec.getSizeInBytes(), ACL_MEMCPY_HOST_TO_DEVICE);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "copy ids to device failed %d", ret);

    fillL2TopkOpInputData(k, batch, segNum, coreNum, attrs);
    return APP_ERR_OK;
}

APP_ERROR IndexIVFRabitQ::fillL2TopkOpInputData(int k, size_t batch, size_t segNum, size_t coreNum,
                                                AscendTensor<int64_t, DIMS_1> &attrs)
{
    std::vector<int64_t> attrsVec(aicpu::TOPK_IVF_RABITQ_ATTR_IDX_COUNT);
    attrsVec[aicpu::TOPK_IVF_RABITQ_ATTR_ASC_IDX] = 1;
    attrsVec[aicpu::TOPK_IVF_RABITQ_ATTR_K_IDX] = k;
    attrsVec[aicpu::TOPK_IVF_RABITQ_ATTR_BURST_LEN_IDX] = IVF_RABITQ_BURST_LEN;
    attrsVec[aicpu::TOPK_IVF_RABITQ_ATTR_BLOCK_NUM_IDX] = static_cast<int64_t>(nprobe * segNum);
    attrsVec[aicpu::TOPK_IVF_RABITQ_ATTR_QUERY_NUM_IDX] = static_cast<int64_t>(batch);
    attrsVec[aicpu::TOPK_IVF_RABITQ_ATTR_CORE_NUM_IDX] = static_cast<int64_t>(coreNum);
    auto ret = aclrtMemcpy(attrs.data(), attrs.getSizeInBytes(),
                           attrsVec.data(), attrsVec.size() * sizeof(int64_t), ACL_MEMCPY_HOST_TO_DEVICE);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "copy attrs to device failed %d", ret);
    return APP_ERR_OK;
}

APP_ERROR IndexIVFRabitQ::fillL1TopkOpInputData(AscendTensor<int64_t, DIMS_1> &attrsInput)
{
    std::vector<int64_t> attrs(aicpu::TOPK_FLAT_ATTR_IDX_COUNT);
    attrs[aicpu::TOPK_FLAT_ATTR_ASC_IDX] = 1;
    attrs[aicpu::TOPK_FLAT_ATTR_K_IDX] = nprobe;
    attrs[aicpu::TOPK_FLAT_ATTR_BURST_LEN_IDX] = IVF_RABITQ_BURST_LEN;
    attrs[aicpu::TOPK_FLAT_ATTR_BLOCK_NUM_IDX] = 1;
    attrs[aicpu::TOPK_FLAT_ATTR_PAGE_IDX] = 0;
    attrs[aicpu::TOPK_FLAT_ATTR_PAGE_NUM_IDX] = 1;
    attrs[aicpu::TOPK_FLAT_ATTR_PAGE_SIZE_IDX] = 0;
    attrs[aicpu::TOPK_FLAT_ATTR_QUICK_HEAP] = 1;
    attrs[aicpu::TOPK_FLAT_ATTR_BLOCK_SIZE] = numLists;
    auto ret = aclrtMemcpy(attrsInput.data(), attrsInput.getSizeInBytes(),
                           attrs.data(), attrs.size() * sizeof(int64_t), ACL_MEMCPY_HOST_TO_DEVICE);
    APPERR_RETURN_IF_NOT_LOG(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "Failed to copy attr to device");
    return APP_ERR_OK;
}

void IndexIVFRabitQ::callL2DistanceOp(size_t batch, size_t segNum, size_t coreNum, size_t vcMaxLen,
                                      AscendTensor<float, DIMS_2> &queryVec,
                                      AscendTensor<float, DIMS_2> &queryLutVec,
                                      AscendTensor<float, DIMS_2, size_t> &centroidsLutVec,
                                      AscendTensor<uint32_t, DIMS_2, size_t> &queryid,
                                      AscendTensor<uint32_t, DIMS_2, size_t> &centroidsid,
                                      AscendTensor<float, DIMS_2, size_t> &centroidsl2,
                                      AscendTensor<uint64_t, DIMS_2, size_t> &offset,
                                      AscendTensor<uint32_t, DIMS_2, size_t> &baseSize,
                                      AscendTensor<uint64_t, DIMS_2, size_t> &indexl2offset,
                                      AscendTensor<uint64_t, DIMS_2, size_t> &indexl1offset,
                                      AscendTensor<uint16_t, DIMS_2, size_t> &opFlag,
                                      AscendTensor<float, DIMS_2, size_t> &disVec,
                                      AscendTensor<float, DIMS_2, size_t> &vcMaxDisVec,
                                      AscendTensor<uint8_t, DIMS_2, size_t> &codeVec,
                                      AscendTensor<float, DIMS_1, size_t> &Indexl2,
                                      AscendTensor<float, DIMS_1, size_t> &Indexl1,
                                      aclrtStream &stream)
{
    size_t ivfRabitqBlockSize = static_cast<size_t>(IVF_RABITQ_BLOCK_SIZE);
    size_t iterNum = (batch * nprobe * segNum + coreNum - 1)/ coreNum;
    for (size_t iter = 0; iter < iterNum; iter++) {
        AscendTensor<float, DIMS_2> subQuery(queryVec[0][0].data(), {1, dims});
        AscendTensor<uint32_t, DIMS_1, size_t> subQueryid(queryid[iter].data(), {coreNum});
        AscendTensor<uint32_t, DIMS_1, size_t> subCentroidsid(centroidsid[iter].data(), {coreNum});
        AscendTensor<float, DIMS_1, size_t> subCentroidsl2(centroidsl2[iter].data(), {coreNum});
        AscendTensor<uint64_t, DIMS_1, size_t> subOffset(offset[iter].data(), {coreNum});
        AscendTensor<uint32_t, DIMS_1, size_t> subBaseSize(baseSize[iter].data(), {coreNum});
        AscendTensor<uint64_t, DIMS_1, size_t> subIndexl2Offset(indexl2offset[iter].data(), {coreNum});
        AscendTensor<uint64_t, DIMS_1, size_t> subIndexl1Offset(indexl1offset[iter].data(), {coreNum});
        AscendTensor<uint16_t, DIMS_2, size_t> subOpFlag(opFlag[iter].data(), {coreNum, 16});
        AscendTensor<float, DIMS_2, size_t> subDis(disVec[iter].data(), {coreNum, ivfRabitqBlockSize});
        AscendTensor<float, DIMS_2, size_t> subVcMaxDis(vcMaxDisVec[iter].data(), {coreNum, vcMaxLen});
        if (faiss::ascend::SocUtils::GetInstance().IsAscendA5()) {
            runL2DistOp(subQuery, queryLutVec, centroidsLutVec, subQueryid, subCentroidsid,
                      subCentroidsl2, codeVec, subOffset, subBaseSize, Indexl2,
                      Indexl1, subIndexl2Offset, subIndexl1Offset, subDis, subVcMaxDis, subOpFlag, stream);
        } else {
            runL2DistOp(subQuery, queryVec, centroidsLutVec, subQueryid, subCentroidsid,
                      subCentroidsl2, codeVec, subOffset, subBaseSize, Indexl2,
                      Indexl1, subIndexl2Offset, subIndexl1Offset, subDis, subVcMaxDis, subOpFlag, stream);
        }
    }
}

size_t IndexIVFRabitQ::getMaxListNum(size_t batch, AscendTensor<int64_t, DIMS_2> &l1TopNprobeIndicesHost,
                                     int k, float* distances, idx_t* labels) const
{
    size_t maxLen = 0;
    for (size_t qIdx = 0; qIdx < batch; qIdx++) {
        for (size_t probId = 0; probId < static_cast<size_t>(nprobe); probId++) {
            int64_t listId = l1TopNprobeIndicesHost[qIdx][probId].value();
            size_t listNum = deviceListIndices[listId]->size();
            maxLen = maxLen > listNum ? maxLen : listNum;
        }
    }
    size_t topk = static_cast<size_t>(k);
    if (maxLen == 0) {
        for (size_t i = 0; i < batch; i++) {
            for (size_t j = 0; j < topk; j++) {
                distances[i * topk + j] = std::numeric_limits<float>::min();
                labels[i * topk + j] = -1;  // 使用-1表示无效ID
            }
        }
    }
    return maxLen;
}

void IndexIVFRabitQ::resizeDistResult(size_t iterNum, size_t coreNum, size_t ivfRabitqBlockSize)
{
    size_t disSpace = iterNum * coreNum * ivfRabitqBlockSize;
    if (disSpace > this->distResultSpace) {
        distResultOnDevice->resize(disSpace);
        this->distResultSpace = disSpace;
    }
}

APP_ERROR IndexIVFRabitQ::searchImplL2(AscendTensor<float, DIMS_2> &queries,
                                       AscendTensor<float, DIMS_2> &queriesLut,
                                       AscendTensor<int64_t, DIMS_2> &l1TopNprobeIndicesHost,
                                       AscendTensor<float, DIMS_2> &l1TopNprobeDistsHost,
                                       int k, float* distances, idx_t* labels)
{
    auto &mem = resources.getMemoryManager();
    auto streamPtr = resources.getDefaultStream();
    auto stream = streamPtr->GetStream();
    auto streamAicpuPtr = resources.getAlternateStreams()[0];
    auto streamAicpu = streamAicpuPtr->GetStream();
    size_t batch = static_cast<size_t>(queries.getSize(0));
    size_t coreNum = static_cast<size_t>(CORE_NUM);
    size_t ivfRabitqBlockSize = static_cast<size_t>(IVF_RABITQ_BLOCK_SIZE);
    size_t segNum = utils::divUp(getMaxListNum(batch, l1TopNprobeIndicesHost, k, distances, labels),
                                 ivfRabitqBlockSize);
    APPERR_RETURN_IF_NOT_LOG(segNum != 0, APP_ERR_OK, "all nprobe cluster is empty by L1 search !!!");
    size_t iterNum = (batch * nprobe * segNum + coreNum - 1)/ coreNum;
    AscendTensor<uint16_t, DIMS_2, size_t> opFlag(mem, {iterNum, coreNum * 16}, stream);
    (void)opFlag.zero();
    AscendTensor<uint64_t, DIMS_2, size_t> offset(mem, {iterNum, coreNum}, stream);
    AscendTensor<uint64_t, DIMS_2, size_t> indexl2offset(mem, {iterNum, coreNum}, stream);
    AscendTensor<uint64_t, DIMS_2, size_t> indexl1offset(mem, {iterNum, coreNum}, stream);
    AscendTensor<uint32_t, DIMS_2, size_t> queryid(mem, {iterNum, coreNum}, stream);
    AscendTensor<uint32_t, DIMS_2, size_t> centroidsid(mem, {iterNum, coreNum}, stream);
    AscendTensor<float, DIMS_2, size_t> centroidsl2(mem, {iterNum, coreNum}, stream);
    AscendTensor<uint32_t, DIMS_2, size_t> baseSize(mem, {iterNum, coreNum}, stream);
    AscendTensor<float, DIMS_2, size_t> centroidsLutVec;
    if (faiss::ascend::SocUtils::GetInstance().IsAscendA5()) {
        centroidsLutVec = AscendTensor<float, DIMS_2, size_t>
          (CentroidLUTOnDevice->data(), {static_cast<size_t>(numLists * dims / SCAN_BIT), LUT_NUM});
    } else {
        centroidsLutVec = AscendTensor<float, DIMS_2, size_t>
          (centroidsOnDevice->data(), {static_cast<size_t>(numLists), static_cast<size_t>(dims)});
    }
    AscendTensor<uint8_t, DIMS_2, size_t> codeVec(pBaseFp32, {ivfRabitqBlockSize, static_cast<size_t>(dims / 8)});
    AscendTensor<float, DIMS_1, size_t> Indexl2(pIndexL2, {ivfRabitqBlockSize});
    AscendTensor<float, DIMS_1, size_t> Indexl1(pIndexL1, {ivfRabitqBlockSize});
    size_t vcMaxLen = static_cast<size_t>((ivfRabitqBlockSize  + IVF_RABITQ_BURST_LEN -1)/ IVF_RABITQ_BURST_LEN * 2);
    resizeDistResult(iterNum, coreNum, ivfRabitqBlockSize);
    AscendTensor<float, DIMS_2, size_t> disVec(distResultOnDevice->data(), {iterNum, coreNum * ivfRabitqBlockSize});
    AscendTensor<int64_t, DIMS_2, size_t> ids(mem, {iterNum, coreNum}, stream);
    AscendTensor<float, DIMS_2, size_t> vcMaxDisVec(mem, {iterNum, coreNum * vcMaxLen}, stream);
    AscendTensor<int64_t, DIMS_1> attrs(mem, {aicpu::TOPK_IVF_RABITQ_ATTR_IDX_COUNT}, stream);
    fillDisOpInputData(k, batch, segNum, coreNum, offset, indexl2offset, indexl1offset, queryid, centroidsid,
                       centroidsl2, baseSize, ids, attrs, l1TopNprobeIndicesHost, l1TopNprobeDistsHost);
    AscendTensor<float, DIMS_2, size_t> outDist(mem, {batch, static_cast<size_t>(k)}, stream);
    AscendTensor<idx_t, DIMS_2, size_t> outLabel(mem, {batch, static_cast<size_t>(k)}, stream);
    runL2TopkOp(batch, disVec, vcMaxDisVec, ids, baseSize, opFlag, attrs, outDist, outLabel, streamAicpu);
    callL2DistanceOp(batch, segNum, coreNum, vcMaxLen, queries, queriesLut, centroidsLutVec,
                     queryid, centroidsid, centroidsl2, offset, baseSize, indexl2offset, indexl1offset,
                     opFlag, disVec, vcMaxDisVec, codeVec, Indexl2, Indexl1, stream);
    auto ret = synchronizeStream(stream);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "synchronizeStream default stream: %i\n", ret);
    ret = synchronizeStream(streamAicpu);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "sync aicpu stream failed %d", ret);
    ret = aclrtMemcpy(distances, batch * k * sizeof(float), outDist.data(),
                      outDist.getSizeInBytes(), ACL_MEMCPY_DEVICE_TO_HOST);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "copy distances to host failed %d", ret);
    ret = aclrtMemcpy(labels, batch * k * sizeof(idx_t), outLabel.data(),
                      outLabel.getSizeInBytes(), ACL_MEMCPY_DEVICE_TO_HOST);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "copy outLabel to host failed %d", ret);
    return APP_ERR_OK;
}

APP_ERROR IndexIVFRabitQ::searchImplL1(AscendTensor<float, DIMS_2> &queries,
                                       AscendTensor<float, DIMS_2> &rotateQueries,
                                       AscendTensor<float, DIMS_2> &queriesLut,
                                       AscendTensor<int64_t, DIMS_2> &l1TopNprobeIndicesHost,
                                       AscendTensor<float, DIMS_2> &l1TopNprobeDistsHost)
{
    auto &mem = resources.getMemoryManager();
    auto streamPtr = resources.getDefaultStream();
    auto stream = streamPtr->GetStream();
    int n = queries.getSize(0);
    AscendTensor<float, DIMS_2> rotatematrixDev(OrthogonalMatrixOnDevice->data(), {dims, dims});
    runQueryRotateOp(n, queries, rotatematrixDev, rotateQueries, stream);
    auto streamAicpuPtr = resources.getAlternateStreams()[0];
    auto streamAicpu = streamAicpuPtr->GetStream();
    // L1 dist op output: dists / vmdists / opFlag
    AscendTensor<float, DIMS_2> dists(mem, {n, numLists}, stream);
    int minDistSize = numLists / IVF_RABITQ_BURST_LEN * 2;
    AscendTensor<float, DIMS_2> vmdists(mem, {n, minDistSize}, stream);
    AscendTensor<uint32_t, DIMS_2> opSize(mem, {CORE_NUM, SIZE_ALIGN}, stream);
    opSize[0][0] = numLists;
    AscendTensor<uint16_t, DIMS_2> opFlag(mem, {CORE_NUM, FLAG_SIZE}, stream);
    opFlag.zero();
    AscendTensor<int64_t, DIMS_1> attrsInput(mem, { aicpu::TOPK_FLAT_ATTR_IDX_COUNT }, stream);
    fillL1TopkOpInputData(attrsInput);
    AscendTensor<float, DIMS_2> l1TopNprobeDists(mem, {n, nprobe}, stream);
    AscendTensor<int64_t, DIMS_2> l1TopNprobeIndices(mem, {n, nprobe}, stream);
    auto ret = synchronizeStream(stream);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "synchronizeStream default stream: %i\n", ret);
    // run l1 distance calculation
    AscendTensor<float, DIMS_2> centroidsDev(centroidsOnDevice->data(), {numLists, dims});
    runL1TopkOp(dists, vmdists, opSize, opFlag, attrsInput, l1TopNprobeDists, l1TopNprobeIndices, streamAicpu);
    runL1DistOp(n, rotateQueries, centroidsDev, dists, vmdists, opFlag, stream);
    ret = synchronizeStream(stream);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "synchronizeStream default stream: %i\n", ret);
    ret = synchronizeStream(streamAicpu);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR,
                             "synchronizeStream aicpu stream failed: %i\n", ret);
    ret = aclrtMemcpy(l1TopNprobeIndicesHost.data(), l1TopNprobeIndicesHost.getSizeInBytes(),
                      l1TopNprobeIndices.data(), l1TopNprobeIndices.getSizeInBytes(), ACL_MEMCPY_DEVICE_TO_HOST);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "Mem operator error %d", ret);
    ret = aclrtMemcpy(l1TopNprobeDistsHost.data(), l1TopNprobeDistsHost.getSizeInBytes(),
                      l1TopNprobeDists.data(), l1TopNprobeDists.getSizeInBytes(), ACL_MEMCPY_DEVICE_TO_HOST);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "Mem operator error %d", ret);
    if (faiss::ascend::SocUtils::GetInstance().IsAscendA5()) {
        // <x, q> LUT预计算
        AscendTensor<float, DIMS_2> queriesfastscan(rotateQueries.data(), {n * dims / SCAN_BIT, SCAN_BIT});
        AscendTensor<float, DIMS_2> lutmatrixDev(LUTMatrixOnDevice->data(), {SCAN_BIT, LUT_NUM});
        runQueryLUTOp(n, queriesfastscan, lutmatrixDev, queriesLut, stream);
        ret = synchronizeStream(stream);
        APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR,
                                 "synchronizeStream default stream: %i\n", ret);
    }
    return APP_ERR_OK;
}

APP_ERROR IndexIVFRabitQ::searchWithBatch(int n, const float * x, int k,
                                          float* distances, idx_t* labels, const float* srcIndexes)
{
    auto &mem = resources.getMemoryManager();
    auto streamPtr = resources.getDefaultStream();
    auto stream = streamPtr->GetStream();
    AscendTensor<float, DIMS_2> queries(mem, { n, dims }, stream);
    AscendTensor<float, DIMS_2> rotateQueries(mem, {n, dims}, stream);
    AscendTensor<float, DIMS_2> queriesLut(mem, {n * dims / SCAN_BIT, LUT_NUM}, stream);
    auto ret = aclrtMemcpy(queries.data(), queries.getSizeInBytes(),
                           x, n * dims * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "copy query error %d", ret);
    std::vector<int64_t> l1TopNprobeIndicesVec(n * nprobe, 0);
    AscendTensor<int64_t, DIMS_2> l1TopNprobeIndicesHost(l1TopNprobeIndicesVec.data(), { n, nprobe });
    std::vector<float> l1TopNprobeDistsVec(n * nprobe, 0);
    AscendTensor<float, DIMS_2> l1TopNprobeDistsHost(l1TopNprobeDistsVec.data(), { n, nprobe });

    ret = searchImplL1(queries, rotateQueries, queriesLut, l1TopNprobeIndicesHost, l1TopNprobeDistsHost);
    APPERR_RETURN_IF_NOT_FMT(ret == APP_ERR_OK, ret, "ivfflat L1 search failed! %d", ret);

    if (srcIndexes == nullptr) {
        ret = searchImplL2(rotateQueries, queriesLut, l1TopNprobeIndicesHost,
                           l1TopNprobeDistsHost, k, distances, labels);
        APPERR_RETURN_IF_NOT_FMT(ret == APP_ERR_OK, ret, "ivfflat L2 search failed! %d", ret);
        return APP_ERR_OK;
    }
    std::vector<float> topkdist(n * k, 0);
    std::vector<idx_t> topklabel(n * k, 0);
    ret = searchImplL2(rotateQueries, queriesLut, l1TopNprobeIndicesHost,
                       l1TopNprobeDistsHost, k, topkdist.data(), topklabel.data());
    APPERR_RETURN_IF_NOT_FMT(ret == APP_ERR_OK, ret, "ivfflat L2 search failed! %d", ret);
    refine(n, x, k, distances, labels, topkdist.data(), topklabel.data(), srcIndexes);
    return APP_ERR_OK;
}

template <typename T, typename C>
void UpdateHeap(float *dists, T *label, int64_t len, int64_t index, C &&cmp)
{
    int64_t l = 0;
    int64_t r = 0;
    int64_t m = 0;
    while (true) {
        l = 2 * index + 1; // 2 * index + 1 to find left subnode
        r = 2 * index + 2; // 2 * index + 2 to find right subnode
        m = index;
        if (l < len && cmp(dists[l], dists[m])) {
            m = l;
        }
        if (r < len && cmp(dists[r], dists[m])) {
            m = r;
        }
        if (m != index) {
            std::swap(dists[m], dists[index]);
            std::swap(label[m], label[index]);
            index = m;
        } else {
            break;
        }
    }
}

float fvec_L2sqr(const float* x, const float* y, size_t d)
{
    size_t i;
    float res = 0;
    for (i = 0; i < d; i++) {
        const float tmp = x[i] - y[i];
        res += tmp * tmp;
    }
    return res;
}

void fvec_L2sqr_by_idx(float* __restrict dis, const float* x, const float* y,
                       const idx_t* __restrict ids, size_t d, size_t nx, size_t ny)
{
#pragma omp parallel for
    for (int64_t j = 0; j < nx; j++) {
        const idx_t* __restrict idsj = ids + j * ny;
        const float* xj = x + j * d;
        float* __restrict disj = dis + j * ny;
        for (size_t i = 0; i < ny; i++) {
            if (idsj[i] < 0) {
                disj[i] = INFINITY;
            } else {
                disj[i] = fvec_L2sqr(xj, y + d * idsj[i], d);
            }
        }
    }
}

void IndexIVFRabitQ::refine(int n, const float * x, int k, float* distances,
                            idx_t* labels, float* topkdist, idx_t* topklabel, const float* srcIndexes)
{
    fvec_L2sqr_by_idx(topkdist, x, srcIndexes, topklabel, dims, n, k);

    std::fill_n(labels, n * k, 0xffffffffffffffff); // 0xffffffffffffffff为无效label
    std::fill_n(distances, n * k, 0x7f7fffff); // 小端排序模式初始化为0x7f7fffff，即最大值
    auto cmp = [](float a, float b) -> bool { return a > b; };
#pragma omp parallel for
    for (int64_t i = 0; i < n; ++i) {
        const idx_t offset = i * k;
        float* outdists = distances + offset;
        float* srcdist = topkdist + offset;
        idx_t* outlabels = labels + offset;
        idx_t* srclabel = topklabel + offset;
        for (int64_t j = 0; j < k; ++j) {
            if (cmp(outdists[0], srcdist[j])) {
                outdists[0] = srcdist[j];
                outlabels[0] = srclabel[j];
                UpdateHeap(outdists, outlabels, k, 0, cmp);
            }
        }
        for (int64_t j = k - 1; j >= 1; --j) {
            std::swap(outdists[0], outdists[j]);
            std::swap(outlabels[0], outlabels[j]);
            UpdateHeap(outdists, outlabels, j, 0, cmp);
        }
    }
}

APP_ERROR IndexIVFRabitQ::searchImpl(int n, const float * x, int k,
                                     float* distances, idx_t* labels, const float* srcIndexes)
{
    APP_ERROR ret = APP_ERR_OK;
    if (n == 1 || searchBatchSizes.empty()) {
        return searchWithBatch(n, x, k, distances, labels, srcIndexes);
    }
    size_t size = searchBatchSizes.size();
    int64_t searched = 0;
    for (size_t i = 0; i < size; i++) {
        int64_t batchSize = searchBatchSizes[i];
        if ((n - searched) >= batchSize) {
            int64_t page = (n - searched) / batchSize;
            for (int64_t j = 0; j < page; j++) {
                ret = searchWithBatch(batchSize, x + searched * dims, k, distances + searched * k,
                    labels + searched * k, srcIndexes);
                APPERR_RETURN_IF(ret, ret);
                searched += batchSize;
            }
        }
    }
    for (int64_t i = searched; i < n; i++) {
        ret = searchWithBatch(1, x + i * dims, k, distances + i * k, labels + i * k, srcIndexes);
        APPERR_RETURN_IF(ret, ret);
    }
    return APP_ERR_OK;
}

void IndexIVFRabitQ::moveVectorForward(int listId, idx_t srcIdx, idx_t dstIdx)
{
    ASCEND_THROW_IF_NOT(srcIdx >= dstIdx);
    size_t blockSizeL = static_cast<size_t>(blockSize);
    size_t srcIdx1 = srcIdx / blockSizeL;
    size_t srcIdx2 = srcIdx % blockSizeL;

    size_t dstIdx1 = dstIdx / blockSizeL;
    size_t dstIdx2 = dstIdx % blockSizeL;

    RemoveForwardParam param = {
        static_cast<size_t>(srcIdx1), static_cast<size_t>(srcIdx2),
        static_cast<size_t>(dstIdx1), static_cast<size_t>(dstIdx2)
    };

    auto ret = RemoveForwardNDFormat(param, dims, baseFp32[listId]);
    ASCEND_THROW_IF_NOT_FMT(ret == ACL_SUCCESS, "RemoveForwardNDFormat error %d", ret);
    listVecNum[listId]--;
    return;
}

void IndexIVFRabitQ::releaseUnusageSpace(int listId, size_t oldTotal, size_t remove)
{
    size_t oldVecSize = utils::divUp(oldTotal, static_cast<size_t>(blockSize));
    size_t vecSize = utils::divUp(oldTotal - remove,  static_cast<size_t>(blockSize));

    for (size_t i = oldVecSize - 1; i >= vecSize; --i) {
        baseFp32[listId].at(i)->clear();
    }
}

size_t IndexIVFRabitQ::removeIds(const ascend::IDSelector& sel)
{
    size_t removeCntAll = 0;
#pragma omp parallel for reduction(+ : removeCntAll) num_threads(CommonUtils::GetThreadMaxNums())
    for (int id = 0; id < numLists; id++) {
        size_t removeCnt = 0;
        size_t oldCnt = deviceListIndices[id]->size();
        auto &indicesList = deviceListIndices[id];
        if (indicesList->size() == 0) {
            continue;
        }
        std::vector<idx_t> indicesVec(indicesList->size());
        auto ret = aclrtMemcpy(indicesVec.data(), indicesList->size() * sizeof(idx_t),
                               indicesList->data(), indicesList->size() * sizeof(idx_t), ACL_MEMCPY_DEVICE_TO_HOST);
        ASCEND_THROW_IF_NOT_FMT(ret == ACL_SUCCESS, "Memcpy error %d", ret);
        idx_t *indicesCheckerPtr = indicesVec.data();
        idx_t *indicesPtr = indicesList->data();
        bool hasMoved = false;
        size_t j = indicesList->size() - 1;
        std::vector<size_t> delIndices;
        for (size_t i = 0; i <= j;) {
            if (!sel.is_member(indicesCheckerPtr[i])) {
                i++;
                continue;
            }
            delIndices.push_back(i);
            auto err = aclrtMemcpy(indicesPtr + i, sizeof(idx_t),
                                   indicesPtr + j, sizeof(idx_t), ACL_MEMCPY_DEVICE_TO_DEVICE);
            ASCEND_THROW_IF_NOT_FMT(err == ACL_SUCCESS, "Memcpy error %d", err);
            indicesCheckerPtr[i] = indicesCheckerPtr[j];
            j--;
            hasMoved = true;
        }
        if (!delIndices.empty()) {
            std::vector<idx_t> sortData(delIndices.begin(), delIndices.end());
            std::sort(sortData.begin(), sortData.end(), std::greater<idx_t>());
            for (const auto index : sortData) {
                moveVectorForward(id, listVecNum[id] - 1, index);
                removeCnt++;
                --this->ntotal;
            }
            releaseUnusageSpace(id, oldCnt, removeCnt);
        }
        if (hasMoved) {
            indicesList->resize(j + 1);
            indicesList->reclaim(false);
        }
        removeCntAll += removeCnt;
    }
    return removeCntAll;
}

APP_ERROR IndexIVFRabitQ::searchImpl(int n, const float16_t* x, int k, float16_t* distances, idx_t* labels)
{
    VALUE_UNUSED(n);
    VALUE_UNUSED(x);
    VALUE_UNUSED(k);
    VALUE_UNUSED(distances);
    VALUE_UNUSED(labels);
    return APP_ERR_OK;
}

APP_ERROR IndexIVFRabitQ::searchImpl(AscendTensor<float16_t, DIMS_2> &queries, int k,
    AscendTensor<float16_t, DIMS_2> &outDistance, AscendTensor<idx_t, DIMS_2> &outIndices)
{
    VALUE_UNUSED(queries);
    VALUE_UNUSED(k);
    VALUE_UNUSED(outDistance);
    VALUE_UNUSED(outIndices);
    return APP_ERR_OK;
}

APP_ERROR IndexIVFRabitQ::searchPaged(size_t pageId, size_t pageNum, AscendTensor<float16_t, DIMS_2> &queries,
    AscendTensor<float16_t, DIMS_2> &maxDistances, AscendTensor<int64_t, DIMS_2> &maxIndices)
{
    VALUE_UNUSED(pageId);
    VALUE_UNUSED(pageNum);
    VALUE_UNUSED(queries);
    VALUE_UNUSED(maxDistances);
    VALUE_UNUSED(maxIndices);
    return APP_ERR_OK;
}

} // ascend
