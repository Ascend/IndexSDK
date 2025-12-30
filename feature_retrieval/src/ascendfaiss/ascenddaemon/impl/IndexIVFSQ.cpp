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


#include "ascenddaemon/impl/IndexIVFSQ.h"

#include <algorithm>
#include <atomic>

#include "ascenddaemon/impl/AuxIndexStructures.h"
#include "ascenddaemon/utils/Limits.h"
#include "common/utils/CommonUtils.h"
#include "common/utils/LogUtils.h"
#include "common/utils/OpLauncher.h"
#include "ops/cpukernel/impl/utils/kernel_shared_def.h"

namespace ascend {
template<typename T>
IndexIVFSQ<T>::IndexIVFSQ(int numList, int dim, bool, int nprobes, int64_t resourceSize)
    : IndexIVF(numList, dim, dim, nprobes, resourceSize),
      threadPool(CREATE_UNIQUE_PTR(AscendThreadPool, THREADS_CNT))
{
    ASCEND_THROW_IF_NOT(dim % CUBE_ALIGN == 0);

    isTrained = false;

    // supported batch size
    searchBatchSizes = {64, 32, 16, 8, 4, 2, 1};

    for (int i = 0; i < numLists; ++i) {
        preComputeData.push_back(CREATE_UNIQUE_PTR(DeviceVector<float>, MemorySpace::DEVICE_HUGEPAGE));
    }
}

template<typename T>
IndexIVFSQ<T>::~IndexIVFSQ() {}

template<typename T>
APP_ERROR IndexIVFSQ<T>::reset()
{
    // reset the database and precomputed, but trained values is maintained.
    IndexIVF::reset();

    preComputeData.clear();
    for (int i = 0; i < numLists; ++i) {
        preComputeData.push_back(CREATE_UNIQUE_PTR(DeviceVector<float>, MemorySpace::DEVICE_HUGEPAGE));
    }

    return APP_ERR_OK;
}

template<typename T>
APP_ERROR IndexIVFSQ<T>::reserveMemory(size_t numVecs)
{
    size_t numVecsPerList = utils::divUp(numVecs, static_cast<size_t>(numLists));
    if (numVecsPerList < 1) {
        return APP_ERR_OK;
    }

    numVecsPerList = utils::roundUp(numVecsPerList, static_cast<size_t>(CUBE_ALIGN));
    size_t tmpLen = numVecsPerList * static_cast<size_t>(numLists);
    IndexIVF::reserveMemory(tmpLen);

    for (auto &list : preComputeData) {
        list->reserve(numVecsPerList);
    }
    return APP_ERR_OK;
}

template<typename T>
APP_ERROR IndexIVFSQ<T>::reserveMemory(int listId, size_t numVecs)
{
    APPERR_RETURN_IF_NOT_FMT((listId < numLists) && (listId >= 0), APP_ERR_INVALID_PARAM,
        "the listId is out of numLists, listId=%d, numLists=%d", listId, numLists);

    if (numVecs < 1) {
        return APP_ERR_OK;
    }

    numVecs = utils::roundUp(numVecs, static_cast<size_t>(CUBE_ALIGN));
    IndexIVF::reserveMemory(listId, numVecs);

    preComputeData[listId]->reserve(numVecs);
    return APP_ERR_OK;
}

template<typename T>
size_t IndexIVFSQ<T>::reclaimMemory()
{
    size_t totalReclaimed = IndexIVF::reclaimMemory();

    for (auto &list : preComputeData) {
        totalReclaimed += list->reclaim(true);
    }

    return totalReclaimed;
}

template<typename T>
size_t IndexIVFSQ<T>::reclaimMemory(int listId)
{
    ASCEND_THROW_IF_NOT((listId < numLists) && (listId >= 0));

    size_t totalReclaimed = IndexIVF::reclaimMemory(listId);

    totalReclaimed += preComputeData[listId]->reclaim(true);

    return totalReclaimed;
}

template<typename T>
APP_ERROR IndexIVFSQ<T>::addVectors(int listId, size_t numVecs, const uint8_t *codes,
    const idx_t *indices, const float *preCompute)
{
    APPERR_RETURN_IF_NOT_LOG(this->isTrained, APP_ERR_ILLEGAL_OPERATION, "the index is not trained");
    APPERR_RETURN_IF_NOT_FMT(listId >= 0 && listId < numLists, APP_ERR_INVALID_PARAM,
        "the listId is %d, out of numLists(%d)", listId, numLists);
    APPERR_RETURN_IF(numVecs == 0, APP_ERR_OK);

    // code need to be Zz format because of DistanceComputeSQ8 operator's limitation.
    //       origin code for example (shape n X dim). n=16, dim = 128
    //       |  0_0  0_1  0_2  0_3 ...  0_125  0_126  0_127 |
    //       |  1_0  1_1  1_2  1_3 ...  1_125  1_126  1_127 |
    //       |        .                          .          |
    //       |        .                          .          |
    //       | 14_0 14_1 14_2 14_3 ... 14_125 14_126 14_127 |
    //       | 15_0 15_1 15_2 15_3 ... 15_125 15_126 15_127 |
    //                              | shape dims 2: (dim/16 X n/16) X (16 X 16),
    //             after Zz format    dims4: (n/16) X (dim/16) X 16 X 16
    //       |   0_0   0_1 ...  0_14  0_15   1_0   1_1 ...  1_15 ...   15_15 |
    //       |  0_16  0_17 ...  0_30  0_31  1_16  1_17 ...  1_31 ...   15_31 |
    //       |        .                    .                  .         .    |
    //       |        .                    .                  .         .    |
    //       |  0_96  0_97 ... 0_110 0_111  1_96  1_97 ... 1_111 ...  15_111 |
    //       | 0_112 0_113 ... 0_126 0_127 1_112 1_113 ... 1_127 ...  15_127 |
    // n and dim must be 16 aligned, otherwise padding data is needed.

    // 1. save codes data
    AscendTensor<uint8_t, DIMS_2> codesData(const_cast<uint8_t *>(codes), { static_cast<int>(numVecs), this->dims });
    size_t originLen = getListLength(listId);
    size_t tmpLen = utils::roundUp((originLen + numVecs), static_cast<size_t>(CUBE_ALIGN));
    deviceListData[listId]->resize(tmpLen * this->dims, true);

    APPERR_RETURN_IF(addCodesAicpu(listId, codesData) != APP_ERR_OK, APP_ERR_INNER_ERROR);

    // 2. save pre compute data if not null
    if (preCompute != nullptr) {
        preComputeData[listId]->resize(tmpLen);
        float *precompData = preComputeData[listId]->data() + originLen;
        auto err = aclrtMemcpy(precompData, numVecs * sizeof(float),
            preCompute, numVecs * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE);
        APPERR_RETURN_IF_NOT_FMT(err == ACL_SUCCESS, APP_ERR_INNER_ERROR, "Memcpy error %d", (int)err);
    }
    deviceListIndices[listId]->append(indices, numVecs, true);

    maxListLength = std::max(maxListLength, static_cast<int>(getListLength(listId)));
    maxListLength = utils::roundUp(maxListLength, CUBE_ALIGN);
    this->ntotal += numVecs;

    return APP_ERR_OK;
}

template<typename T>
void IndexIVFSQ<T>::updateCoarseCentroidsData(AscendTensor<float16_t, DIMS_2> &coarseCentroidsData)
{
    // update coarse centroids for L1 search.
    IndexIVF::updateCoarseCentroidsData(coarseCentroidsData);

    // isTrained need to be set when all trained values are updated.
    // if vMin has been updated, set isTrained = true
    if (this->vMin.data()) {
        this->isTrained = true;
    }
}

template<typename T>
void IndexIVFSQ<T>::updateTrainedValue(AscendTensor<float16_t, DIMS_1> &trainedMin,
                                       AscendTensor<float16_t, DIMS_1> &trainedDiff)
{
    int dimMin = trainedMin.getSize(0);
    int dimDiff = trainedDiff.getSize(0);
    ASCEND_THROW_IF_NOT_FMT(dimMin == dimDiff && dimMin == this->dims,
                            "sq trained data's shape invalid.(%d, %d) vs (dim:%d)", dimMin, dimDiff, this->dims);

    AscendTensor<float16_t, DIMS_1> minTensor({ dimMin });
    AscendTensor<float16_t, DIMS_1> diffTensor({ dimDiff });
    minTensor.copyFromSync(trainedMin, ACL_MEMCPY_HOST_TO_DEVICE);
    diffTensor.copyFromSync(trainedDiff, ACL_MEMCPY_HOST_TO_DEVICE);
    vMin = std::move(minTensor);
    vDiff = std::move(diffTensor);

    AscendTensor<float16_t, DIMS_2> dmTensor({ 2, this->dims });
    auto ret = aclrtMemcpy(dmTensor[0].data(), static_cast<size_t>(this->dims) * sizeof(float16_t),
        vDiff.data(), static_cast<size_t>(this->dims) * sizeof(float16_t), ACL_MEMCPY_DEVICE_TO_DEVICE);
    ASCEND_THROW_IF_NOT_FMT(ret == ACL_SUCCESS, "Memcpy error %d", (int)ret);
    ret = aclrtMemcpy(dmTensor[1].data(), static_cast<size_t>(this->dims) * sizeof(float16_t),
        vMin.data(), static_cast<size_t>(this->dims) * sizeof(float16_t), ACL_MEMCPY_DEVICE_TO_DEVICE);
    ASCEND_THROW_IF_NOT_FMT(ret == ACL_SUCCESS, "Memcpy error %d", (int)ret);
    vDM = std::move(dmTensor);

    // isTrained need to be set when all trained values are updated.
    // if coarseCentroids has been updated, set isTrained = true
    if (this->coarseCentroids.data()) {
        this->isTrained = true;
    }
}

template<typename T>
void IndexIVFSQ<T>::calcResiduals(AscendTensor<float16_t, DIMS_1> &query,
                                  AscendTensor<idx_t, DIMS_1> &nprobeIndices,
                                  AscendTensor<float16_t, DIMS_2> &residulas)
{
    int dim = query.getSize(0);
    int probes = nprobeIndices.getSize(0);
    ASCEND_THROW_IF_NOT(probes == this->nprobe);
    ASCEND_THROW_IF_NOT(dim == this->dims);

    // query - L1 coarse centroids
    for (int probeIdx = 0; probeIdx < probes; ++probeIdx) {
        int list = static_cast<int>(nprobeIndices[probeIdx].value());
        for (int k = 0; k < dim; k++) {
            residulas[probeIdx][k] = query[k].value() - coarseCentroids[list][k].value();
        }
    };
}

template<typename T>
size_t IndexIVFSQ<T>::removeIds(const ascend::IDSelector& sel)
{
    //
    // | id0 id1 id2 ... id98 id99 id100 |
    //            ^                  |
    //            |__________________|
    // if id2 is to be removed, copy the last id(id100) to
    // the mem place of id2, and decrease the list size to size-1.
    size_t removeCnt = 0;

    // dims is alignd with CUBE_ALIGN, no padding data in horizontal direction
    int dimShaped = utils::divUp(this->dims, CUBE_ALIGN);

#pragma omp parallel for reduction(+ : removeCnt) num_threads(CommonUtils::GetThreadMaxNums())
    for (int id = 0; id < numLists; id++) {
        auto &indicesList = deviceListIndices[id];
        auto &codeList = deviceListData[id];
        auto &precompList = preComputeData[id];
        if (indicesList->size() == 0) {
            continue;
        }

        // host need copy indices from device to host to check is_member
        std::vector<idx_t> indicesVec(indicesList->size());
        auto ret = aclrtMemcpy(indicesVec.data(), indicesList->size() * sizeof(idx_t),
                               indicesList->data(), indicesList->size() * sizeof(idx_t), ACL_MEMCPY_DEVICE_TO_HOST);
        ASCEND_THROW_IF_NOT_FMT(ret == ACL_SUCCESS, "Memcpy error %d", (int)ret);
        idx_t *indicesCheckerPtr = indicesVec.data();

        idx_t *indicesPtr = indicesList->data();
        float *precompPtr = precompList->data();
        uint8_t *codePtr = static_cast<uint8_t *>(codeList->data());
        bool hasMoved = false;
        int j = (int)indicesList->size() - 1;
        for (int i = 0; i <= j;) {
            if (!sel.is_member(indicesCheckerPtr[i])) {
                i++;
                continue;
            }

            auto err = aclrtMemcpy(indicesPtr + i, sizeof(idx_t),
                                   indicesPtr + j, sizeof(idx_t), ACL_MEMCPY_DEVICE_TO_DEVICE);
            ASCEND_THROW_IF_NOT_FMT(err == ACL_SUCCESS, "Memcpy error %d", (int)err);
            indicesCheckerPtr[i] = indicesCheckerPtr[j];

            if (precompPtr != nullptr) {
                err = aclrtMemcpy(precompPtr + i, sizeof(float),
                                  precompPtr + j, sizeof(float), ACL_MEMCPY_DEVICE_TO_DEVICE);
                ASCEND_THROW_IF_NOT_FMT(err == ACL_SUCCESS, "Memcpy error %d", (int)err);
            }

            uint8_t *src = codePtr + getShapedDataOffset(j);
            uint8_t *dst = codePtr + getShapedDataOffset(i);
            for (int k = 0; k < dimShaped; k++) {
                err = aclrtMemcpy(dst, static_cast<size_t>(CUBE_ALIGN) * sizeof(uint8_t),
                                  src, static_cast<size_t>(CUBE_ALIGN) * sizeof(uint8_t), ACL_MEMCPY_DEVICE_TO_DEVICE);
                ASCEND_THROW_IF_NOT_FMT(err == ACL_SUCCESS, "Memcpy error %d", (int)err);
                src += (CUBE_ALIGN * CUBE_ALIGN);
                dst += (CUBE_ALIGN * CUBE_ALIGN);
            }

            j--;
            removeCnt++;
            hasMoved = true;
        }

        // if some code has been removed, list need to be resize and reclaim memory
        if (hasMoved) {
            size_t tmpLen = utils::roundUp((j + 1), CUBE_ALIGN);
            indicesList->resize(j + 1);
            codeList->resize(tmpLen * this->dims);
            precompList->resize(tmpLen);
            indicesList->reclaim(false);
            codeList->reclaim(false);
            precompList->reclaim(false);
        }
    }

    this->ntotal -= removeCnt;
    return removeCnt;
}

template<typename T>
int IndexIVFSQ<T>::getShapedDataOffset(int idx) const
{
    int offset = this->dims * utils::roundDown(idx, CUBE_ALIGN);
    offset += (idx % CUBE_ALIGN) * CUBE_ALIGN;
    return offset;
}

template<typename T>
APP_ERROR IndexIVFSQ<T>::addCodes(int listId, AscendTensor<uint8_t, DIMS_2> &codesData)
{
    size_t numVecs = static_cast<size_t>(codesData.getSize(0));
    size_t originLen = getListLength(listId);

    // dims is alignd with CUBE_ALIGN, no padding data in horizontal direction
    size_t dimShaped = static_cast<size_t>(utils::divUp(this->dims, CUBE_ALIGN));
    auto memErr = EOK;

// input codes are contigous(numVecs X dims), reconstruct the codes into Zz format.
#pragma omp parallel for if (numVecs >= 100) num_threads(CommonUtils::GetThreadMaxNums())
    for (size_t i = 0; i < numVecs; i++) {
        int seq = static_cast<int>(originLen + i);
        uint8_t *tmpData = static_cast<uint8_t *>(deviceListData[listId]->data()) + getShapedDataOffset(seq);

        for (size_t j = 0; j < dimShaped; j++) {
            auto err = memcpy_s(tmpData, static_cast<size_t>(CUBE_ALIGN) * sizeof(uint8_t),
                codesData[i][j * CUBE_ALIGN].data(), static_cast<size_t>(CUBE_ALIGN) * sizeof(uint8_t));
            ASCEND_EXC_IF_NOT_FMT(err == EOK, memErr = err,
                "memcpy codesData err, i=%d, j=%d, err=%d", i, j, err);

            tmpData += (CUBE_ALIGN * CUBE_ALIGN);
        }
    }

    APPERR_RETURN_IF_NOT_LOG(memErr == EOK, APP_ERR_INNER_ERROR, "Memcpy error");

    return APP_ERR_OK;
}

template<typename T>
APP_ERROR IndexIVFSQ<T>::addCodesAicpu(int listId, AscendTensor<uint8_t, DIMS_2> &codesData)
{
    std::string opName = "TransdataShaped";
    int n = codesData.getSize(0);
    auto &mem = resources.getMemoryManager();
    auto streamPtr = resources.getDefaultStream();
    auto stream = streamPtr->GetStream();
    AscendTensor<int8_t, DIMS_2> src(mem, {n, dims}, stream);
    auto ret = aclrtMemcpy(src.data(), src.getSizeInBytes(),
                           codesData.data(), codesData.getSizeInBytes(), ACL_MEMCPY_HOST_TO_DEVICE);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "Mem operator error %d", (int)ret);

    size_t originLen = getListLength(listId);
    AscendTensor<int64_t, DIMS_1> attr(mem, {aicpu::TRANSDATA_SHAPED_ATTR_IDX_COUNT}, stream);
    attr[0] = originLen;

    int total = static_cast<int>(originLen) + n;
    AscendTensor<int8_t, DIMS_4> dst(reinterpret_cast<int8_t *>(deviceListData[listId]->data()),
        {utils::divUp(total, CUBE_ALIGN), utils::divUp(dims, CUBE_ALIGN), CUBE_ALIGN, CUBE_ALIGN});

    LaunchOpTwoInOneOut<int8_t, DIMS_2, ACL_INT8,
                        int64_t, DIMS_1, ACL_INT64,
                        int8_t, DIMS_4, ACL_INT8>(opName, stream, src, attr, dst);

    ret = synchronizeStream(stream);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR,
        "synchronizeStream addVectors stream failed: %i\n", ret);

    return APP_ERR_OK;
}

template<typename T>
APP_ERROR IndexIVFSQ<T>::getListVectorsAicpu(int listId, int num, unsigned char *reshaped) const
{
    std::string opName = "TransdataRaw";
    auto &mem = resources.getMemoryManager();
    auto streamPtr = resources.getDefaultStream();
    auto stream = streamPtr->GetStream();
    AscendTensor<int8_t, DIMS_2> dst(mem, {num, dims}, stream);

    AscendTensor<int64_t, DIMS_1> attr(mem, {aicpu::TRANSDATA_RAW_ATTR_IDX_COUNT}, stream);
    attr[aicpu::TRANSDATA_RAW_ATTR_OFFSET_IDX] = 0;

    AscendTensor<int8_t, DIMS_4> src(reinterpret_cast<int8_t *>(getListVectors(listId).data()),
        {utils::divUp(num, CUBE_ALIGN), utils::divUp(dims, CUBE_ALIGN), CUBE_ALIGN, CUBE_ALIGN});

    LaunchOpTwoInOneOut<int8_t, DIMS_4, ACL_INT8,
                        int64_t, DIMS_1, ACL_INT64,
                        int8_t, DIMS_2, ACL_INT8>(opName, stream, src, attr, dst);

    auto ret = synchronizeStream(stream);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR,
        "synchronizeStream getVector stream failed: %i\n", ret);

    ret = aclrtMemcpy(reshaped, dst.getSizeInBytes(),
                      dst.data(), dst.getSizeInBytes(), ACL_MEMCPY_DEVICE_TO_HOST);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "Mem operator error %d", (int)ret);

    return APP_ERR_OK;
}

template<typename T>
DeviceVector<float> &IndexIVFSQ<T>::getListPrecompute(int listId) const
{
    ASCEND_THROW_IF_NOT((listId < numLists) && (listId >= 0));
    return *preComputeData[listId];
}

template<typename T>
bool IndexIVFSQ<T>::listVectorsNeedReshaped() const
{
    return true;
}

template<typename T>
APP_ERROR IndexIVFSQ<T>::getListVectorsReshaped(int listId, std::vector<unsigned char> &reshaped) const
{
    size_t size = getListLength(listId);
    reshaped.resize(size * this->dims);
    return getListVectorsReshaped(listId, reshaped.data());
}

template<typename T>
APP_ERROR IndexIVFSQ<T>::getListVectorsReshaped(int listId, unsigned char* reshaped) const
{
    APPERR_RETURN_IF_NOT_FMT((listId < numLists) && (listId >= 0), APP_ERR_INVALID_PARAM,
        "listId is out of numLists, listId=%d, numLists=%d", listId, numLists);

    size_t size = getListLength(listId);
    return getListVectorsAicpu(listId, static_cast<int>(size), reshaped);
}

template<typename T>
APP_ERROR IndexIVFSQ<T>::resetOp(const std::string &opTypeName,
                                 std::unique_ptr<AscendOperator> &op,
                                 const std::vector<std::pair<aclDataType, std::vector<int64_t>>> &input,
                                 const std::vector<std::pair<aclDataType, std::vector<int64_t>>> &output)
{
    AscendOpDesc desc(opTypeName);
    for (auto &data : input) {
        desc.addInputTensorDesc(data.first, data.second.size(), data.second.data(), ACL_FORMAT_ND);
    }
    
    for (auto &data : output) {
        desc.addOutputTensorDesc(data.first, data.second.size(), data.second.data(), ACL_FORMAT_ND);
    }
    op = CREATE_UNIQUE_PTR(AscendOperator, desc);
    bool ret = op->init();
    APPERR_RETURN_IF_NOT_FMT(ret, APP_ERR_ACL_OP_LOAD_MODEL_FAILED,
                             "op init failed, op name: %s\n", opTypeName.c_str());
    return APP_ERR_OK;
}

template<typename T>
APP_ERROR IndexIVFSQ<T>::runOp(AscendOperator *op,
                               const std::vector<const AscendTensorBase *> &input,
                               const std::vector<const AscendTensorBase *> &output,
                               aclrtStream stream)
{
    std::shared_ptr<std::vector<const aclDataBuffer *>> distSqOpInput(
        new std::vector<const aclDataBuffer *>(), CommonUtils::AclInputBufferDelete);
    for (auto &data : input) {
        distSqOpInput->emplace_back(aclCreateDataBuffer(data->getVoidData(), data->getSizeInBytes()));
    }
    std::shared_ptr<std::vector<aclDataBuffer *>> distSqOpOutput(
        new std::vector<aclDataBuffer *>(), CommonUtils::AclOutputBufferDelete);
    for (auto &data : output) {
        distSqOpOutput->emplace_back(aclCreateDataBuffer(data->getVoidData(), data->getSizeInBytes()));
    }

    op->exec(*distSqOpInput, *distSqOpOutput, stream);
    return APP_ERR_OK;
}

template<typename T>
void IndexIVFSQ<T>::runL2TopkOp(int batch,
                                const std::vector<const AscendTensorBase *> &input,
                                const std::vector<const AscendTensorBase *> &output,
                                aclrtStream stream)
{
    AscendOperator *op = nullptr;
    if (l2TopkOps.find(batch) != l2TopkOps.end()) {
        op = l2TopkOps[batch].get();
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

template class IndexIVFSQ<float>;
} // ascend
