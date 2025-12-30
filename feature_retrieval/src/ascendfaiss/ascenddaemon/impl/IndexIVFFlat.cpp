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


#include "ascenddaemon/impl/IndexIVFFlat.h"
#include <algorithm>
#include <atomic>

#include "ascenddaemon/impl/AuxIndexStructures.h"
#include "ascenddaemon/utils/Limits.h"
#include "common/utils/CommonUtils.h"
#include "common/utils/LogUtils.h"
#include "common/utils/OpLauncher.h"
#include "ops/cpukernel/impl/utils/kernel_shared_def.h"


namespace ascend {
const int KB = 1024;
const int IVF_FLAT_BLOCK_SIZE = 16384 * 16;
const int IVF_FLAT_COMPUTE_PAGE = IVF_FLAT_BLOCK_SIZE * 2;
IndexIVFFlat::IndexIVFFlat(int numList, int dim, int nprobes, int64_t resourceSize)
    : IndexIVF(numList, dim * 4, dim, nprobes, resourceSize), blockSize(IVF_FLAT_BLOCK_SIZE)
{
    ASCEND_THROW_IF_NOT(dim % CUBE_ALIGN == 0);
    int dim1 = utils::divUp(blockSize, CUBE_ALIGN);
    int dim2Fp32 = utils::divUp(dims, CUBE_ALIGN_FP32);
    devFp32VecCapacity = dim1 * dim2Fp32 * CUBE_ALIGN * CUBE_ALIGN_FP32;
    isTrained = false;
    // supported batch size
    searchBatchSizes = {64, 32, 16, 8, 4, 2, 1};
    listVecNum = std::vector<size_t>(numList, 0);
    baseFp32.resize(numList);
    pBaseFp32 = reinterpret_cast<float*>(0xffffffffffffffff);  // 基地址初始化为最大的无效值 0xffffffffffffffff
    centroidsOnDevice = CREATE_UNIQUE_PTR(DeviceVector<float>, MemorySpace::DEVICE_HUGEPAGE);
    centroidsOnDevice->resize(numList * dims);
    auto ret = resetL1TopkOp();
    ASCEND_THROW_IF_NOT_MSG(ret == APP_ERR_OK, "resetL1TopkOp failed!");
    ret = resetL1DistOp();
    ASCEND_THROW_IF_NOT_MSG(ret == APP_ERR_OK, "resetL1DistOp failed!");
    ret = resetL2DistOp();
    ASCEND_THROW_IF_NOT_MSG(ret == APP_ERR_OK, "resetL2DistOp failed!");
    ret = resetL2TopkOp();
    ASCEND_THROW_IF_NOT_MSG(ret == APP_ERR_OK, "resetL2TopkOp failed!");
}

IndexIVFFlat::~IndexIVFFlat() {}

APP_ERROR IndexIVFFlat::reset()
{
    IndexIVF::reset();
    for (size_t i = 0; i < baseFp32.size(); i++) {
        baseFp32[i].clear();
    }
    baseFp32.clear();
    centroidsOnDevice->clear();
    return APP_ERR_OK;
}

APP_ERROR IndexIVFFlat::resizeBaseFp32(int listId, size_t numVecs)
{
    if (numVecs == 0) {
        return APP_ERR_OK;
    }
    size_t currentVecNum = listVecNum[listId];
    auto& blockList = baseFp32[listId];
    
    size_t remainingVecs = numVecs;
    while (remainingVecs > 0) {
        size_t tailBlkId = blockList.empty() ? 0 : blockList.size() - 1;
        if (blockList.empty()) {
            blockList.emplace_back(CREATE_UNIQUE_PTR(DeviceVector<float>, MemorySpace::DEVICE_HUGEPAGE));
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
            blockList.emplace_back(CREATE_UNIQUE_PTR(DeviceVector<float>, MemorySpace::DEVICE_HUGEPAGE));
            tailBlkId = blockList.size() - 1;
            currentBlockRemaining = static_cast<size_t>(blockSize);
        }
        size_t vecsToAdd = std::min(remainingVecs, currentBlockRemaining);
        size_t newSize = (currentVecNum % static_cast<size_t>(blockSize)) + vecsToAdd;
        blockList[tailBlkId]->resize(newSize * dims, true);
        currentVecNum += vecsToAdd;
        remainingVecs -= vecsToAdd;
        pBaseFp32 =
            blockList.at(tailBlkId)->data() < pBaseFp32 ? blockList.at(tailBlkId)->data() : pBaseFp32;
    }
    return APP_ERR_OK;
}

size_t IndexIVFFlat::getListLength(int listId) const
{
    ASCEND_THROW_IF_NOT((listId < numLists) && (listId >= 0));

    return listVecNum[listId];
}

APP_ERROR IndexIVFFlat::addVectors(int listId, size_t numVecs, const float *codes, const idx_t *indices)
{
    APPERR_RETURN_IF_NOT_FMT(listId >= 0 && listId < numLists, APP_ERR_INVALID_PARAM,
        "the listId is %d, out of numLists(%d)", listId, numLists);
    APPERR_RETURN_IF(numVecs == 0, APP_ERR_OK);
    AscendTensor<float, DIMS_2> codesData(const_cast<float *>(codes), { static_cast<int>(numVecs), dims });
    APPERR_RETURN_IF_NOT_LOG(APP_ERR_OK == resizeBaseFp32(listId, numVecs), APP_ERR_INNER_ERROR, "resize base failed!");
    auto ret = AddCodeNDFormat(codesData, listVecNum[listId], blockSize, baseFp32[listId]);
    APPERR_RETURN_IF_NOT_FMT(ret == APP_ERR_OK, APP_ERR_INNER_ERROR, "add vector to device failed %d", ret);

    maxListLength = std::max(maxListLength, static_cast<int>(getListLength(listId)));
    maxListLength = utils::roundUp(maxListLength, CUBE_ALIGN);
    listVecNum[listId] += numVecs;
    IndexIVF::ntotal += numVecs;
    deviceListIndices[listId]->append(indices, numVecs, true);
    return APP_ERR_OK;
}

APP_ERROR IndexIVFFlat::resetL1TopkOp()
{
    auto topkCompOpReset = [&](std::unique_ptr<AscendOperator> &op, int64_t batch) {
        AscendOpDesc desc("TopkFlatFp32");
        std::vector<int64_t> shape0 { 1, batch, numLists };
        std::vector<int64_t> shape1 { 1, batch,  numLists / IVF_FLAT_BURST_LEN * 2 };
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

void IndexIVFFlat::runL1TopkOp(AscendTensor<float, DIMS_2> &dists,
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
                                            std::min(numLists / IVF_FLAT_BURST_LEN * 2, MIN_EXTREME_SIZE)});
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

APP_ERROR IndexIVFFlat::resetL1DistOp()
{
    auto l1DisOpReset = [&](std::unique_ptr<AscendOperator> &op, int batch) {
        AscendOpDesc desc("DistanceFlatL2MinsAtFP32");
        std::vector<int64_t> queryShape({ batch, dims });
        std::vector<int64_t> codesShape({ numLists, dims });
        
        std::vector<int64_t> distResultShape({ batch, numLists });
        std::vector<int64_t> minResultShape({ batch, numLists / IVF_FLAT_BURST_LEN * 2});
        std::vector<int64_t> flagShapeShape({ CORE_NUM, 16 });
        desc.addInputTensorDesc(ACL_FLOAT, queryShape.size(), queryShape.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_FLOAT, codesShape.size(), codesShape.data(), ACL_FORMAT_ND);
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

void IndexIVFFlat::runL1DistOp(int batch, AscendTensor<float, DIMS_2> &queries,
                               AscendTensor<float, DIMS_2> &centroidsDev, AscendTensor<float, DIMS_2> &dists,
                               AscendTensor<float, DIMS_2> &vmdists, AscendTensor<uint16_t, DIMS_2> &opFlag,
                               aclrtStream stream)
{
    AscendOperator *op = nullptr;
    if (l1DistFp32Ops.find(batch) != l1DistFp32Ops.end()) {
        op = l1DistFp32Ops[batch].get();
    }
    ASCEND_THROW_IF_NOT(op);
    std::shared_ptr<std::vector<const aclDataBuffer *>> topkOpInput(
        new std::vector<const aclDataBuffer *>(), CommonUtils::AclInputBufferDelete);
    topkOpInput->emplace_back(aclCreateDataBuffer(queries.data(), queries.getSizeInBytes()));
    topkOpInput->emplace_back(aclCreateDataBuffer(centroidsDev.data(), centroidsDev.getSizeInBytes()));

    std::shared_ptr<std::vector<aclDataBuffer *>> topkOpOutput(
        new std::vector<aclDataBuffer *>(), CommonUtils::AclOutputBufferDelete);
    topkOpOutput->emplace_back(aclCreateDataBuffer(dists.data(), dists.getSizeInBytes()));
    topkOpOutput->emplace_back(aclCreateDataBuffer(vmdists.data(), vmdists.getSizeInBytes()));
    topkOpOutput->emplace_back(aclCreateDataBuffer(opFlag.data(), opFlag.getSizeInBytes()));
    op->exec(*topkOpInput, *topkOpOutput, stream);
    return;
}

APP_ERROR IndexIVFFlat::resetL2TopkOp()
{
    auto topkCompOpReset = [&](std::unique_ptr<AscendOperator> &op, int64_t batch) {
        AscendOpDesc desc("TopkIvfFp32");
        std::vector<int64_t> shape0 { batch, 0, CORE_NUM, IVF_FLAT_BLOCK_SIZE };
        std::vector<int64_t> shape1 { batch, 0, CORE_NUM,
                                      (IVF_FLAT_BLOCK_SIZE + IVF_FLAT_BURST_LEN -1) / IVF_FLAT_BURST_LEN * 2};
        std::vector<int64_t> shape2 { batch, 0, CORE_NUM };
        std::vector<int64_t> shape3 { batch, 0, CORE_NUM };
        std::vector<int64_t> shape4 { batch, 0, CORE_NUM * 16 };
        std::vector<int64_t> shape5 { aicpu::TOPK_IVF_ATTR_IDX_COUNT };

        std::vector<int64_t> shape6 { batch, 0 };

        desc.addInputTensorDesc(ACL_FLOAT, shape0.size(), shape0.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_FLOAT, shape1.size(), shape1.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_INT64, shape2.size(), shape2.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_UINT32, shape3.size(), shape3.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_UINT16, shape4.size(), shape4.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_INT64, shape5.size(), shape5.data(), ACL_FORMAT_ND);

        desc.addOutputTensorDesc(ACL_FLOAT, shape6.size(), shape5.data(), ACL_FORMAT_ND);
        desc.addOutputTensorDesc(ACL_UINT64, shape6.size(), shape5.data(), ACL_FORMAT_ND);

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

void IndexIVFFlat::runL2TopkOp(AscendTensor<float, DIMS_3, size_t> &distResult,
                               AscendTensor<float, DIMS_3, size_t> &vmdistResult,
                               AscendTensor<int64_t, DIMS_3, size_t> &ids,
                               AscendTensor<uint32_t, DIMS_3, size_t> &sizes,
                               AscendTensor<uint16_t, DIMS_3, size_t> &flags,
                               AscendTensor<int64_t, DIMS_1> &attrs,
                               AscendTensor<float, DIMS_2, size_t> &outdists,
                               AscendTensor<uint64_t, DIMS_2, size_t> &outlabel,
                               aclrtStream stream)
{
    int batch = static_cast<int>(distResult.getSize(0));
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

APP_ERROR IndexIVFFlat::resetL2DistOp()
{
    auto l2DisOpReset = [&](std::unique_ptr<AscendOperator> &op) {
        AscendOpDesc desc("DistanceIVFFlatIpFP32");
        std::vector<int64_t> queryShape({ 1, dims });
        std::vector<int64_t> codesShape({ IVF_FLAT_BLOCK_SIZE, dims });
        std::vector<int64_t> offsetShape({CORE_NUM});
        std::vector<int64_t> sizeShape({CORE_NUM});
        std::vector<int64_t> distResultShape({ CORE_NUM, IVF_FLAT_BLOCK_SIZE });
        std::vector<int64_t> mixResultShape({ CORE_NUM,
            (IVF_FLAT_BLOCK_SIZE + IVF_FLAT_BURST_LEN -1) / IVF_FLAT_BURST_LEN * 2});
        std::vector<int64_t> flagShapeShape({ CORE_NUM, 16 });
        desc.addInputTensorDesc(ACL_FLOAT, queryShape.size(), queryShape.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_FLOAT, codesShape.size(), codesShape.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_UINT64, offsetShape.size(), offsetShape.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_UINT32, sizeShape.size(), sizeShape.data(), ACL_FORMAT_ND);
        desc.addOutputTensorDesc(ACL_FLOAT, distResultShape.size(), distResultShape.data(), ACL_FORMAT_ND);
        desc.addOutputTensorDesc(ACL_FLOAT, mixResultShape.size(), mixResultShape.data(), ACL_FORMAT_ND);
        desc.addOutputTensorDesc(ACL_UINT16, flagShapeShape.size(), flagShapeShape.data(), ACL_FORMAT_ND);
        op.reset();
        op = CREATE_UNIQUE_PTR(AscendOperator, desc);
        return op->init();
    };

    ivfFlatIPOps = std::unique_ptr<AscendOperator>(nullptr);
    APPERR_RETURN_IF_NOT_LOG(l2DisOpReset(ivfFlatIPOps),
                             APP_ERR_ACL_OP_LOAD_MODEL_FAILED, "IVFFLAT IP FP32 distance op init failed");
    return APP_ERR_OK;
}

void IndexIVFFlat::runL2DistOp(AscendTensor<float, DIMS_2, size_t> &subQuery,
                               AscendTensor<float, DIMS_2, size_t> &codeVec,
                               AscendTensor<uint64_t, DIMS_1, size_t> &subOffset,
                               AscendTensor<uint32_t, DIMS_1, size_t> &subBaseSize,
                               AscendTensor<float, DIMS_2, size_t> &subDis,
                               AscendTensor<float, DIMS_2, size_t> &subVcMaxDis,
                               AscendTensor<uint16_t, DIMS_2, size_t> &subOpFlag,
                               aclrtStream stream)
{
    AscendOperator *op = ivfFlatIPOps.get();
    ASCEND_THROW_IF_NOT(op);
    std::shared_ptr<std::vector<const aclDataBuffer *>> topkOpInput(
        new std::vector<const aclDataBuffer *>(), CommonUtils::AclInputBufferDelete);
    topkOpInput->emplace_back(aclCreateDataBuffer(subQuery.data(), subQuery.getSizeInBytes()));
    topkOpInput->emplace_back(aclCreateDataBuffer(codeVec.data(), codeVec.getSizeInBytes()));
    topkOpInput->emplace_back(aclCreateDataBuffer(subOffset.data(), subOffset.getSizeInBytes()));
    topkOpInput->emplace_back(aclCreateDataBuffer(subBaseSize.data(), subBaseSize.getSizeInBytes()));

    std::shared_ptr<std::vector<aclDataBuffer *>> topkOpOutput(
        new std::vector<aclDataBuffer *>(), CommonUtils::AclOutputBufferDelete);
    topkOpOutput->emplace_back(aclCreateDataBuffer(subDis.data(), subDis.getSizeInBytes()));
    topkOpOutput->emplace_back(aclCreateDataBuffer(subVcMaxDis.data(), subVcMaxDis.getSizeInBytes()));
    topkOpOutput->emplace_back(aclCreateDataBuffer(subOpFlag.data(), subOpFlag.getSizeInBytes()));
    op->exec(*topkOpInput, *topkOpOutput, stream);
    return;
}

void IndexIVFFlat::fillDisOpInputDataByBlock(size_t qIdx, size_t tIdx, size_t segIdx,
                                             size_t segNum, size_t coreNum, size_t ivfFlatBlockSize,
                                             AscendTensor<uint32_t, DIMS_3, size_t> &baseSizeHostVec,
                                             AscendTensor<uint64_t, DIMS_3, size_t> &offsetHostVec,
                                             AscendTensor<int64_t, DIMS_3, size_t> &idsHostVec,
                                             AscendTensor<int64_t, DIMS_2> &l1TopNprobeIndicesHost)
{
    for (size_t cIdx = 0; cIdx < coreNum; cIdx++) {
        if (tIdx * coreNum + (segIdx * coreNum + cIdx) / segNum < static_cast<size_t>(nprobe)) {
            int64_t listId =
                l1TopNprobeIndicesHost[qIdx][tIdx * coreNum + (segIdx * coreNum + cIdx) / segNum].value();
            size_t listNum = deviceListIndices[listId]->size();
            size_t proccessLen = (segIdx * ivfFlatBlockSize >= listNum) ? \
                                 0 : std::min(listNum - segIdx * ivfFlatBlockSize, ivfFlatBlockSize);
            baseSizeHostVec[qIdx][tIdx][segIdx * coreNum + cIdx].value(proccessLen);
            if (proccessLen == 0) {
                continue;
            }
            int64_t idAddr =
                reinterpret_cast<int64_t>(deviceListIndices[listId]->data() +
                                            ((segIdx * coreNum + cIdx) % segNum) * ivfFlatBlockSize);
            idsHostVec[qIdx][tIdx][cIdx + segIdx * coreNum].value(idAddr);
            size_t lastBlock = (segIdx * coreNum + cIdx) % segNum;
            if (lastBlock * ivfFlatBlockSize < baseFp32[listId].size()) {
                uint64_t offsetSeg = reinterpret_cast<uint64_t>(baseFp32[listId].at(segIdx)->data()) -
                                        reinterpret_cast<uint64_t>(pBaseFp32);
                offsetHostVec[qIdx][tIdx][segIdx * coreNum + cIdx].value(offsetSeg);
            }
        }
    }
}

APP_ERROR IndexIVFFlat::fillDisOpInputData(int k, size_t batch, size_t tileNum, size_t segNum, size_t coreNum,
                                           AscendTensor<uint64_t, DIMS_3, size_t> &offset,
                                           AscendTensor<uint32_t, DIMS_3, size_t> &baseSize,
                                           AscendTensor<int64_t, DIMS_3, size_t> &ids,
                                           AscendTensor<int64_t, DIMS_1> &attrs,
                                           AscendTensor<int64_t, DIMS_2> &l1TopNprobeIndicesHost)
{
    size_t ivfFlatBlockSize = static_cast<size_t>(IVF_FLAT_BLOCK_SIZE);
    std::vector<uint64_t> offsetHost(batch * tileNum * segNum * coreNum, 0);
    AscendTensor<uint64_t, DIMS_3, size_t> offsetHostVec(offsetHost.data(), {batch, tileNum, segNum * coreNum});
    std::vector<uint32_t> baseSizeHost(batch * tileNum * segNum * coreNum, 0);
    AscendTensor<uint32_t, DIMS_3, size_t> baseSizeHostVec(baseSizeHost.data(), {batch, tileNum, segNum * coreNum});
    std::vector<int64_t> idsHost(batch * tileNum * segNum * coreNum);
    AscendTensor<int64_t, DIMS_3, size_t> idsHostVec(idsHost.data(), {batch, tileNum, segNum * coreNum});
    for (size_t qIdx = 0; qIdx < batch; qIdx++) {
        for (size_t tIdx = 0; tIdx < tileNum; tIdx++) {
            for (size_t segIdx = 0; segIdx < segNum; segIdx++) {
                fillDisOpInputDataByBlock(qIdx, tIdx, segIdx, segNum, coreNum, ivfFlatBlockSize,
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
    std::vector<int64_t> attrsVec(aicpu::TOPK_IVF_ATTR_IDX_COUNT);
    attrsVec[aicpu::TOPK_IVF_ATTR_ASC_IDX] = 0;
    attrsVec[aicpu::TOPK_IVF_ATTR_K_IDX] = k;
    attrsVec[aicpu::TOPK_IVF_ATTR_BURST_LEN_IDX] = IVF_FLAT_BURST_LEN;
    attrsVec[aicpu::TOPK_IVF_ATTR_BLOCK_NUM_IDX] = static_cast<int64_t>(tileNum * segNum);
    attrsVec[aicpu::TOPK_IVF_ATTR_FLAG_NUM_IDX] = static_cast<int64_t>(coreNum);
    ret = aclrtMemcpy(attrs.data(), attrs.getSizeInBytes(),
                      attrsVec.data(), attrsVec.size() * sizeof(int64_t), ACL_MEMCPY_HOST_TO_DEVICE);
    APPERR_RETURN_IF_NOT_FMT(ret == APP_ERR_OK, APP_ERR_INNER_ERROR, "copy attrs to device failed %d", ret);
    return APP_ERR_OK;
}

void IndexIVFFlat::callL2DistanceOp(size_t batch, size_t tileNum, size_t segNum, size_t coreNum, size_t vcMaxLen,
                                    AscendTensor<float, DIMS_2, size_t> &queryVec,
                                    AscendTensor<uint64_t, DIMS_3, size_t> &offset,
                                    AscendTensor<uint32_t, DIMS_3, size_t> &baseSize,
                                    AscendTensor<uint16_t, DIMS_3, size_t> &opFlag,
                                    AscendTensor<float, DIMS_3, size_t> &disVec,
                                    AscendTensor<float, DIMS_3, size_t> &vcMaxDisVec,
                                    AscendTensor<float, DIMS_2, size_t> &codeVec,
                                    aclrtStream &stream)
{
    size_t ivfFlatBlockSize = static_cast<size_t>(IVF_FLAT_BLOCK_SIZE);
    for (size_t qIdx = 0; qIdx < batch; qIdx++) {
        AscendTensor<float, DIMS_2, size_t> subQuery(queryVec[qIdx][0].data(), {1, static_cast<size_t>(dims)});
        for (size_t tIdx = 0; tIdx < tileNum; tIdx++) {
            for (size_t segIdx = 0; segIdx < segNum; segIdx++) {
                AscendTensor<uint64_t, DIMS_1, size_t> subOffset(offset[qIdx][tIdx][segIdx * coreNum].data(), {coreNum});
                AscendTensor<uint32_t, DIMS_1, size_t> subBaseSize(baseSize[qIdx][tIdx][segIdx * coreNum].data(), {coreNum});
                AscendTensor<uint16_t, DIMS_2, size_t> subOpFlag(opFlag[qIdx][tIdx][segIdx * coreNum * 16].data(), {coreNum, 16});
                AscendTensor<float, DIMS_2, size_t> subDis(disVec[qIdx][tIdx][segIdx * coreNum * ivfFlatBlockSize].data(),
                                                   {coreNum, ivfFlatBlockSize});
                AscendTensor<float, DIMS_2, size_t> subVcMaxDis(vcMaxDisVec[qIdx][tIdx][segIdx * vcMaxLen].data(),
                    {coreNum, vcMaxLen});
                runL2DistOp(subQuery, codeVec, subOffset, subBaseSize, subDis, subVcMaxDis, subOpFlag, stream);
            }
        }
    }
}

size_t IndexIVFFlat::getMaxListNum(size_t batch, AscendTensor<int64_t, DIMS_2> &l1TopNprobeIndicesHost, int k,
                                   float* distances, idx_t* labels) const
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

APP_ERROR IndexIVFFlat::searchImplL2(AscendTensor<float, DIMS_2> &queries,
    AscendTensor<int64_t, DIMS_2> &l1TopNprobeIndicesHost, int k, float* distances, idx_t* labels)
{
    auto &mem = resources.getMemoryManager();
    auto streamPtr = resources.getDefaultStream();
    auto stream = streamPtr->GetStream();
    auto streamAicpuPtr = resources.getAlternateStreams()[0];
    auto streamAicpu = streamAicpuPtr->GetStream();
    size_t batch = static_cast<size_t>(queries.getSize(0));
    size_t tileNum = static_cast<size_t>(utils::divUp(nprobe, CORE_NUM));
    size_t coreNum = static_cast<size_t>(CORE_NUM);
    size_t maxLen = getMaxListNum(batch, l1TopNprobeIndicesHost, k, distances, labels);
    APPERR_RETURN_IF_NOT_LOG(maxLen != 0, APP_ERR_OK, "all nprobe cluster is empty by L1 search !!!");
    size_t ivfFlatBlockSize = static_cast<size_t>(IVF_FLAT_BLOCK_SIZE);
    size_t segNum = utils::divUp(maxLen, ivfFlatBlockSize);
    AscendTensor<float, DIMS_2, size_t> queryVec(mem, {batch, static_cast<size_t>(dims)}, stream);
    AscendTensor<uint16_t, DIMS_3, size_t> opFlag(mem, {batch, tileNum, segNum * coreNum * 16}, stream);
    (void)opFlag.zero();
    AscendTensor<uint64_t, DIMS_3, size_t> offset(mem, {batch, tileNum, segNum * coreNum}, stream);
    AscendTensor<uint32_t, DIMS_3, size_t> baseSize(mem, {batch, tileNum, segNum * coreNum}, stream);
    AscendTensor<float, DIMS_2, size_t> codeVec(pBaseFp32, {ivfFlatBlockSize, static_cast<size_t>(dims)});
    size_t vcMaxLen = static_cast<size_t>(ivfFlatBlockSize / IVF_FLAT_BURST_LEN * 2);
    AscendTensor<float, DIMS_3, size_t> disVec(mem, {batch, tileNum, coreNum * segNum * ivfFlatBlockSize}, stream);
    (void)disVec.zero();
    AscendTensor<int64_t, DIMS_3, size_t> ids(mem, {batch, tileNum, segNum * coreNum}, stream);
    AscendTensor<float, DIMS_3, size_t> vcMaxDisVec(mem, {batch, tileNum,
                                            segNum * coreNum * vcMaxLen}, stream);
    AscendTensor<int64_t, DIMS_1> attrs(mem, {aicpu::TOPK_IVF_ATTR_IDX_COUNT}, stream);
    auto ret = aclrtMemcpy(queryVec.data(), queryVec.getSizeInBytes(), queries.data(),
                           queries.getSizeInBytes(), ACL_MEMCPY_HOST_TO_DEVICE);
    APPERR_RETURN_IF_NOT_FMT(ret == APP_ERR_OK, APP_ERR_INNER_ERROR, "copy queries to device failed %d", ret);
    fillDisOpInputData(k, batch, tileNum, segNum, coreNum, offset, baseSize, ids, attrs, l1TopNprobeIndicesHost);
    AscendTensor<float, DIMS_2, size_t> outDist(mem, {batch, static_cast<size_t>(k)}, stream);
    AscendTensor<idx_t, DIMS_2, size_t> outLabel(mem, {batch, static_cast<size_t>(k)}, stream);
    runL2TopkOp(disVec, vcMaxDisVec, ids, baseSize, opFlag, attrs, outDist, outLabel, streamAicpu);
    callL2DistanceOp(batch, tileNum, segNum, coreNum, vcMaxLen, queryVec, offset, baseSize,
                     opFlag, disVec, vcMaxDisVec, codeVec, stream);
    ret = synchronizeStream(stream);
    ret = synchronizeStream(streamAicpu);
    APPERR_RETURN_IF_NOT_FMT(ret == APP_ERR_OK, APP_ERR_INNER_ERROR, "sync aicpu stream failed %d", ret);
    ret = aclrtMemcpy(distances, batch * k * sizeof(float), outDist.data(),
                      outDist.getSizeInBytes(), ACL_MEMCPY_HOST_TO_DEVICE);
    APPERR_RETURN_IF_NOT_FMT(ret == APP_ERR_OK, APP_ERR_INNER_ERROR, "copy distances to host failed %d", ret);
    ret = aclrtMemcpy(labels, batch * k * sizeof(idx_t), outLabel.data(),
                      outLabel.getSizeInBytes(), ACL_MEMCPY_HOST_TO_DEVICE);
    APPERR_RETURN_IF_NOT_FMT(ret == APP_ERR_OK, APP_ERR_INNER_ERROR, "copy outLabel to host failed %d", ret);
    return APP_ERR_OK;
}

APP_ERROR IndexIVFFlat::searchImplL1(AscendTensor<float, DIMS_2> &queries,
                                     AscendTensor<int64_t, DIMS_2> &l1TopNprobeIndicesHost)
{
    auto &mem = resources.getMemoryManager();
    auto streamPtr = resources.getDefaultStream();
    auto stream = streamPtr->GetStream();
    auto streamAicpuPtr = resources.getAlternateStreams()[0];
    auto streamAicpu = streamAicpuPtr->GetStream();
    // L1 dist op output: dists / vmdists / opFlag
    int n = queries.getSize(0);
    AscendTensor<float, DIMS_2> dists(mem, {n, numLists}, stream);
    int minDistSize = numLists / IVF_FLAT_BURST_LEN * 2;
    AscendTensor<float, DIMS_2> vmdists(mem, {n, minDistSize}, stream);
    AscendTensor<uint32_t, DIMS_2> opSize(mem, {CORE_NUM, SIZE_ALIGN}, stream);
    opSize[0][0] = numLists;
    AscendTensor<uint16_t, DIMS_2> opFlag(mem, {CORE_NUM, FLAG_SIZE}, stream);
    opFlag.zero();

    AscendTensor<int64_t, DIMS_1> attrsInput(mem, { aicpu::TOPK_FLAT_ATTR_IDX_COUNT }, stream);
    std::vector<int64_t> attrs(aicpu::TOPK_FLAT_ATTR_IDX_COUNT);
    attrs[aicpu::TOPK_FLAT_ATTR_ASC_IDX] = 1;
    attrs[aicpu::TOPK_FLAT_ATTR_K_IDX] = nprobe;
    attrs[aicpu::TOPK_FLAT_ATTR_BURST_LEN_IDX] = IVF_FLAT_BURST_LEN;
    attrs[aicpu::TOPK_FLAT_ATTR_BLOCK_NUM_IDX] = 1;
    attrs[aicpu::TOPK_FLAT_ATTR_PAGE_IDX] = 0;
    attrs[aicpu::TOPK_FLAT_ATTR_PAGE_NUM_IDX] = 1;
    attrs[aicpu::TOPK_FLAT_ATTR_PAGE_SIZE_IDX] = 0;
    attrs[aicpu::TOPK_FLAT_ATTR_QUICK_HEAP] = 0;
    attrs[aicpu::TOPK_FLAT_ATTR_BLOCK_SIZE] = numLists;

    auto ret = aclrtMemcpy(attrsInput.data(), attrsInput.getSizeInBytes(),
                           attrs.data(), attrs.size() * sizeof(int64_t), ACL_MEMCPY_HOST_TO_DEVICE);
    APPERR_RETURN_IF_NOT_LOG(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "Failed to copy attr to device");

    AscendTensor<float, DIMS_2> l1TopNprobeDists(mem, {n, nprobe}, stream);
    AscendTensor<int64_t, DIMS_2> l1TopNprobeIndices(mem, {n, nprobe}, stream);

    // run l1 distance calculation
    AscendTensor<float, DIMS_2> centroidsDev(centroidsOnDevice->data(), {numLists, dims});
    runL1DistOp(n, queries, centroidsDev, dists, vmdists, opFlag, stream);
    ret = synchronizeStream(stream);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR,
        "synchronizeStream default stream: %i\n", ret);
    runL1TopkOp(dists, vmdists, opSize, opFlag, attrsInput, l1TopNprobeDists, l1TopNprobeIndices, streamAicpu);

    ret = synchronizeStream(streamAicpu);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR,
        "synchronizeStream aicpu stream failed: %i\n", ret);

    ret = aclrtMemcpy(l1TopNprobeIndicesHost.data(), l1TopNprobeIndicesHost.getSizeInBytes(),
                      l1TopNprobeIndices.data(), l1TopNprobeIndices.getSizeInBytes(),
                      ACL_MEMCPY_DEVICE_TO_HOST);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "Mem operator error %d", ret);

    return APP_ERR_OK;
}

APP_ERROR IndexIVFFlat::searchWithBatch(int n, const float * x, int k, float* distances, idx_t* labels)
{
    auto &mem = resources.getMemoryManager();
    auto streamPtr = resources.getDefaultStream();
    auto stream = streamPtr->GetStream();
    AscendTensor<float, DIMS_2> queries(mem, { n, dims }, stream);
    auto ret = aclrtMemcpy(queries.data(), queries.getSizeInBytes(),
                           x, n * dims * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "copy query error %d", ret);
    std::vector<int64_t> l1TopNprobeIndicesVec(n * nprobe, 0);
    AscendTensor<int64_t, DIMS_2> l1TopNprobeIndicesHost(l1TopNprobeIndicesVec.data(), { n, nprobe });
    ret = searchImplL1(queries, l1TopNprobeIndicesHost);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, ret, "ivfflat L1 search failed! %d", ret);
    ret = searchImplL2(queries, l1TopNprobeIndicesHost, k, distances, labels);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, ret, "ivfflat L2 search failed! %d", ret);
    return ACL_SUCCESS;
}

APP_ERROR IndexIVFFlat::searchImpl(int n, const float * x, int k, float* distances, idx_t* labels)
{
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

void IndexIVFFlat::moveVectorForward(int listId, idx_t srcIdx, idx_t dstIdx)
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

void IndexIVFFlat::releaseUnusageSpace(int listId, size_t oldTotal, size_t remove)
{
    size_t oldVecSize = utils::divUp(oldTotal, static_cast<size_t>(blockSize));
    size_t vecSize = utils::divUp(oldTotal - remove,  static_cast<size_t>(blockSize));

    for (size_t i = oldVecSize - 1; i >= vecSize; --i) {
        baseFp32[listId].at(i)->clear();
    }
}

size_t IndexIVFFlat::removeIds(const ascend::IDSelector& sel)
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

APP_ERROR IndexIVFFlat::searchImpl(int n, const float16_t* x, int k, float16_t* distances, idx_t* labels)
{
    VALUE_UNUSED(n);
    VALUE_UNUSED(x);
    VALUE_UNUSED(k);
    VALUE_UNUSED(distances);
    VALUE_UNUSED(labels);
    return APP_ERR_OK;
}

APP_ERROR IndexIVFFlat::searchImpl(AscendTensor<float16_t, DIMS_2> &queries, int k,
    AscendTensor<float16_t, DIMS_2> &outDistance, AscendTensor<idx_t, DIMS_2> &outIndices)
{
    VALUE_UNUSED(queries);
    VALUE_UNUSED(k);
    VALUE_UNUSED(outDistance);
    VALUE_UNUSED(outIndices);
    return APP_ERR_OK;
}

APP_ERROR IndexIVFFlat::searchPaged(size_t pageId, size_t pageNum, AscendTensor<float16_t, DIMS_2> &queries,
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
