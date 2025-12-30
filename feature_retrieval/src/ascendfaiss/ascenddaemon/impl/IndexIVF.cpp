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


#include "ascenddaemon/impl/IndexIVF.h"

#include "ascenddaemon/utils/Limits.h"
#include "common/utils/CommonUtils.h"
#include "common/utils/LogUtils.h"
#include "common/utils/OpLauncher.h"
#include "ops/cpukernel/impl/utils/kernel_shared_def.h"

namespace ascend {
IndexIVF::IndexIVF(int numList, int byteCntPerVector, int dim, int nprobes, int64_t resourceSize)
    : Index(dim, resourceSize),
      numLists(numList),
      bytesPerVector(byteCntPerVector),
      nprobe(nprobes),
      maxListLength(0)
{
    ASCEND_THROW_IF_NOT(numList > 0);
    for (int i = 0; i < numLists; ++i) {
        deviceListData.emplace_back(CREATE_UNIQUE_PTR(DeviceVector<unsigned char>, MemorySpace::DEVICE_HUGEPAGE));
        deviceListIndices.emplace_back(CREATE_UNIQUE_PTR(DeviceVector<idx_t>, MemorySpace::DEVICE_HUGEPAGE));
    }

    pListBase = deviceListData[0]->data();
    for (int i = 1; i < numLists; ++i) {
        pListBase = std::min(pListBase, deviceListData[i]->data());
    }
}

IndexIVF::~IndexIVF() {}

APP_ERROR IndexIVF::reset()
{
    deviceListData.clear();
    deviceListIndices.clear();
    for (int i = 0; i < numLists; ++i) {
        deviceListData.emplace_back(CREATE_UNIQUE_PTR(DeviceVector<unsigned char>, MemorySpace::DEVICE_HUGEPAGE));
        deviceListIndices.emplace_back(CREATE_UNIQUE_PTR(DeviceVector<idx_t>, MemorySpace::DEVICE_HUGEPAGE));
    }

    maxListLength = 0;
    this->ntotal = 0;

    return APP_ERR_OK;
}

int IndexIVF::getDim() const
{
    return dims;
}

APP_ERROR IndexIVF::reserveMemory(size_t numVecs)
{
    size_t numVecsPerList = utils::divUp(numVecs, static_cast<size_t>(numLists));
    if (numVecsPerList < 1) {
        return APP_ERR_OK;
    }

    size_t bytesPerDataList = numVecsPerList * static_cast<size_t>(bytesPerVector);
    for (auto& list : deviceListData) {
        list->reserve(bytesPerDataList);
    }

    for (auto& list : deviceListIndices) {
        list->reserve(numVecsPerList);
    }

    return APP_ERR_OK;
}

APP_ERROR IndexIVF::reserveMemory(int listId, size_t numVecs)
{
    if (numVecs < 1) {
        return APP_ERR_OK;
    }

    ASCEND_THROW_IF_NOT((listId < numLists) && (listId >= 0));
    size_t bytesDataList = numVecs * static_cast<size_t>(bytesPerVector);
    deviceListData[listId]->reserve(bytesDataList);
    deviceListIndices[listId]->reserve(numVecs);

    return APP_ERR_OK;
}

size_t IndexIVF::reclaimMemory()
{
    size_t totalReclaimed = 0;

    for (auto& list : deviceListData) {
        totalReclaimed += list->reclaim(true);
    }

    for (auto& list : deviceListIndices) {
        totalReclaimed += list->reclaim(true);
    }

    return totalReclaimed;
}

size_t IndexIVF::reclaimMemory(int listId)
{
    ASCEND_THROW_IF_NOT((listId < numLists) && (listId >= 0));

    size_t totalReclaimed = 0;
    totalReclaimed += deviceListData[listId]->reclaim(true);
    totalReclaimed += deviceListIndices[listId]->reclaim(true);

    return totalReclaimed;
}

void IndexIVF::setNumProbes(int nprobes)
{
    this->nprobe = nprobes;
}

size_t IndexIVF::getNumLists() const
{
    return numLists;
}

size_t IndexIVF::getListLength(int listId) const
{
    ASCEND_THROW_IF_NOT((listId < numLists) && (listId >= 0));
    return deviceListIndices[listId]->size();
}

size_t IndexIVF::getMaxListDataIndicesBufferSize() const
{
    return maxListLength * (this->dims * sizeof(unsigned char) + sizeof(idx_t));
}

DeviceVector<idx_t>& IndexIVF::getListIndices(int listId) const
{
    ASCEND_THROW_IF_NOT((listId < numLists) && (listId >= 0));
    return *deviceListIndices[listId];
}

DeviceVector<unsigned char>& IndexIVF::getListVectors(int listId) const
{
    ASCEND_THROW_IF_NOT((listId < numLists) && (listId >= 0));
    return *deviceListData[listId];
}

bool IndexIVF::listVectorsNeedReshaped() const
{
    return false;
}

APP_ERROR IndexIVF::getListVectorsReshaped(int listId, std::vector<unsigned char>& reshaped) const
{
    APP_LOG_ERROR("getListVectorsReshaped not implemented for this type of index.\n");
    VALUE_UNUSED(listId);
    VALUE_UNUSED(reshaped);

    return APP_ERR_NOT_IMPLEMENT;
}

APP_ERROR IndexIVF::getListVectorsReshaped(int listId, unsigned char* reshaped) const
{
    APP_LOG_ERROR("getListVectorsReshaped not implemented for this type of index.\n");
    VALUE_UNUSED(listId);
    VALUE_UNUSED(reshaped);

    return APP_ERR_NOT_IMPLEMENT;
}

void IndexIVF::updateCoarseCentroidsData(AscendTensor<float16_t, DIMS_2>& coarseCentroidsData)
{
    int numCoarseCents = coarseCentroidsData.getSize(0);
    int dimCoarseCents = coarseCentroidsData.getSize(1);
    ASCEND_THROW_IF_NOT_FMT(numCoarseCents == numLists && dimCoarseCents == dims,
                            "coarse centroids data's shape invalid.(%d X %d) vs (%d X %d)",
                            numCoarseCents, dimCoarseCents, numLists, dims);

    AscendTensor<float16_t, DIMS_2> deviceCoarseCentroids(
        { numCoarseCents, dimCoarseCents });
    deviceCoarseCentroids.copyFromSync(coarseCentroidsData, ACL_MEMCPY_HOST_TO_DEVICE);
    coarseCentroids = std::move(deviceCoarseCentroids);

    // coarse centroids need to be Zz format because of DistanceCompute operator's limitation.
    //       origin code for example (shape n X dim). n=15, dim = 127. n and dim need to be 16 aligned,
    //         n aligned = 16, dim aligned = 128, the space for aligned need to been padded to 0x00
    //       |  0_0  0_1  0_2  0_3 ...  0_125  0_126 0x00 |
    //       |  1_0  1_1  1_2  1_3 ...  1_125  1_126 0x00 |
    //       |        .                          .        |
    //       |        .                          .        |
    //       | 14_0 14_1 14_2 14_3 ... 14_125 14_126 0x00 |
    //       | 0x00 0x00 0x00 0x00 ...   0x00   0x00 0x00 |
    //                              |
    //             after Zz format  (shape dims 2: n X dim, dims4: (n/16) X (dim/16) X 16 X 16)
    //       |   0_0   0_1 ...  0_14  0_15   1_0   1_1 ...  1_15 ...   7_15 |
    //       |  0_16  0_17 ...  0_30  0_31  1_16  1_17 ...  1_31 ...   7_31 |
    //       |        .                    .                  .         .   |
    //       |        .                    .                  .         .   |
    //       |  0_96  0_97 ... 0_110 0_111  1_96  1_97 ... 1_111 ...  7_111 |
    //       | 0_112 0_113 ... 0_126  0x00 1_112 1_113 ...  0x00 ...   0x00 |
    //       |   8_0   8_1 ...  8_14  8_15   9_0   9_1 ...  9_15 ...   0x00 |
    //       |  8_16  8_17 ...  8_30  8_31  9_16  9_17 ...  9_31 ...   0x00 |
    //       |        .                    .                  .         .   |
    //       |        .                    .                  .         .   |
    //       |  8_96  8_97 ... 8_110 8_111  9_96  9_97 ... 9_111 ...   0x00 |
    //       | 8_112 8_113 ... 8_126  0x00 9_112 9_113 ...  0x00 ...   0x00 |
    int dim1 = utils::divUp(numCoarseCents, CUBE_ALIGN);
    int dim2 = utils::divUp(dimCoarseCents, CUBE_ALIGN);

    AscendTensor<float16_t, DIMS_4> tmpShapedCentroids({ dim1, dim2, CUBE_ALIGN, CUBE_ALIGN });
    AscendTensor<float16_t, DIMS_1> tmpNormTensor({ numCoarseCents });

    addCoarseCentroidsAiCpu(coarseCentroids, tmpShapedCentroids);
    fvecNormsL2sqrAicpu(tmpNormTensor, coarseCentroids);

    coarseCentroidsShaped = std::move(tmpShapedCentroids);
    normCoarseCentroids = std::move(tmpNormTensor);
}

APP_ERROR IndexIVF::resetL1DistOp(int numLists)
{
    auto l1DistOpReset = [&](std::unique_ptr<AscendOperator> &op, int64_t batch) {
        AscendOpDesc desc("DistanceFlatL2Mins");
        std::vector<int64_t> queryShape({ batch, dims });
        std::vector<int64_t> coarseCentroidsShape({ utils::divUp(numLists, CUBE_ALIGN),
            utils::divUp(dims, CUBE_ALIGN), CUBE_ALIGN, CUBE_ALIGN });
        std::vector<int64_t> preNormsShape({ numLists });
        std::vector<int64_t> distResultShape({ batch, numLists });
        // the result constain min value and index, the multi 2
        std::vector<int64_t> distMinsShape({ batch, std::max(numLists / BURST_LEN * 2, MIN_EXTREME_SIZE) });
        std::vector<int64_t> flagShape({ CORE_NUM, FLAG_SIZE });

        desc.addInputTensorDesc(ACL_FLOAT16, queryShape.size(), queryShape.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_FLOAT16, coarseCentroidsShape.size(), coarseCentroidsShape.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_FLOAT16, preNormsShape.size(), preNormsShape.data(), ACL_FORMAT_ND);
        desc.addOutputTensorDesc(ACL_FLOAT16, distResultShape.size(), distResultShape.data(), ACL_FORMAT_ND);
        desc.addOutputTensorDesc(ACL_FLOAT16, distMinsShape.size(), distMinsShape.data(), ACL_FORMAT_ND);
        desc.addOutputTensorDesc(ACL_UINT16, flagShape.size(), flagShape.data(), ACL_FORMAT_ND);

        op.reset();
        op = CREATE_UNIQUE_PTR(AscendOperator, desc);
        return op->init();
    };

    for (auto batch : searchBatchSizes) {
        l1DistOps[batch] = std::unique_ptr<AscendOperator>(nullptr);
        APPERR_RETURN_IF_NOT_LOG(l1DistOpReset(l1DistOps[batch], batch),
                                 APP_ERR_ACL_OP_LOAD_MODEL_FAILED, "op init failed");
    }

    return APP_ERR_OK;
}

APP_ERROR IndexIVF::resetL1TopkOp()
{
    auto topkCompOpReset = [&](std::unique_ptr<AscendOperator> &op, int64_t batch) {
        AscendOpDesc desc("TopkFlat");
        std::vector<int64_t> shape0 { 0, batch, numLists };
        std::vector<int64_t> shape1 { 0, batch, std::max(numLists / BURST_LEN * 2, MIN_EXTREME_SIZE) };
        std::vector<int64_t> shape2 { 0, CORE_NUM, SIZE_ALIGN };
        std::vector<int64_t> shape3 { 0, CORE_NUM, FLAG_SIZE };
        std::vector<int64_t> shape4 { aicpu::TOPK_FLAT_ATTR_IDX_COUNT };
        std::vector<int64_t> shape5 { batch, 0 };

        desc.addInputTensorDesc(ACL_FLOAT16, shape0.size(), shape0.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_FLOAT16, shape1.size(), shape1.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_UINT32, shape2.size(), shape2.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_UINT16, shape3.size(), shape3.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_INT64, shape4.size(), shape4.data(), ACL_FORMAT_ND);

        desc.addOutputTensorDesc(ACL_FLOAT16, shape5.size(), shape5.data(), ACL_FORMAT_ND);
        desc.addOutputTensorDesc(ACL_INT64, shape5.size(), shape5.data(), ACL_FORMAT_ND);

        op.reset();
        op = CREATE_UNIQUE_PTR(AscendOperator, desc);
        return op->init();
    };
    for (auto batch : searchBatchSizes) {
        l1TopkOps[batch] = std::unique_ptr<AscendOperator>(nullptr);
        APPERR_RETURN_IF_NOT_LOG(topkCompOpReset(l1TopkOps[batch], batch),
                                 APP_ERR_ACL_OP_LOAD_MODEL_FAILED, "l1 topk op init failed");
    }

    return APP_ERR_OK;
}

void IndexIVF::runL1DistOp(AscendTensor<float16_t, DIMS_2>& queryVecs,
                           AscendTensor<float16_t, DIMS_4>& shapedData,
                           AscendTensor<float16_t, DIMS_1>& norms,
                           AscendTensor<float16_t, DIMS_2>& outDists,
                           AscendTensor<float16_t, DIMS_2>& outDistMins,
                           AscendTensor<uint16_t, DIMS_2>& flag,
                           aclrtStream stream)
{
    AscendOperator *op = nullptr;
    int batch = queryVecs.getSize(0);
    if (l1DistOps.find(batch) != l1DistOps.end()) {
        op = l1DistOps[batch].get();
    }
    ASCEND_THROW_IF_NOT(op);

    std::shared_ptr<std::vector<const aclDataBuffer *>> distOpInput(
        new std::vector<const aclDataBuffer *>(), CommonUtils::AclInputBufferDelete);
    distOpInput->emplace_back(aclCreateDataBuffer(queryVecs.data(), queryVecs.getSizeInBytes()));
    distOpInput->emplace_back(aclCreateDataBuffer(shapedData.data(), shapedData.getSizeInBytes()));
    distOpInput->emplace_back(aclCreateDataBuffer(norms.data(), norms.getSizeInBytes()));

    std::shared_ptr<std::vector<aclDataBuffer *>> distOpOutput(
        new std::vector<aclDataBuffer *>(), CommonUtils::AclOutputBufferDelete);
    distOpOutput->emplace_back(aclCreateDataBuffer(outDists.data(), outDists.getSizeInBytes()));
    distOpOutput->emplace_back(aclCreateDataBuffer(outDistMins.data(), outDistMins.getSizeInBytes()));
    distOpOutput->emplace_back(aclCreateDataBuffer(flag.data(), flag.getSizeInBytes()));

    op->exec(*distOpInput, *distOpOutput, stream);
}

void IndexIVF::runL1TopkOp(AscendTensor<float16_t, DIMS_2> &dists,
                           AscendTensor<float16_t, DIMS_2> &vmdists,
                           AscendTensor<uint32_t, DIMS_2> &sizes,
                           AscendTensor<uint16_t, DIMS_2> &flags,
                           AscendTensor<int64_t, DIMS_1> &attrs,
                           AscendTensor<float16_t, DIMS_2> &outdists,
                           AscendTensor<int64_t, DIMS_2> &outlabel,
                           aclrtStream stream)
{
    AscendOperator *op = nullptr;
    int batch = dists.getSize(0);
    if (l1TopkOps.find(batch) != l1TopkOps.end()) {
        op = l1TopkOps[batch].get();
    }
    ASCEND_THROW_IF_NOT(op);

    std::shared_ptr<std::vector<const aclDataBuffer *>> topkOpInput(
        new std::vector<const aclDataBuffer *>(), CommonUtils::AclInputBufferDelete);
    topkOpInput->emplace_back(aclCreateDataBuffer(dists.data(), dists.getSizeInBytes()));
    topkOpInput->emplace_back(aclCreateDataBuffer(vmdists.data(), vmdists.getSizeInBytes()));
    topkOpInput->emplace_back(aclCreateDataBuffer(sizes.data(), sizes.getSizeInBytes()));
    topkOpInput->emplace_back(aclCreateDataBuffer(flags.data(), flags.getSizeInBytes()));
    topkOpInput->emplace_back(aclCreateDataBuffer(attrs.data(), attrs.getSizeInBytes()));

    std::shared_ptr<std::vector<aclDataBuffer *>> topkOpOutput(
        new std::vector<aclDataBuffer *>(), CommonUtils::AclOutputBufferDelete);
    topkOpOutput->emplace_back(aclCreateDataBuffer(outdists.data(), outdists.getSizeInBytes()));
    topkOpOutput->emplace_back(aclCreateDataBuffer(outlabel.data(), outlabel.getSizeInBytes()));

    op->exec(*topkOpInput, *topkOpOutput, stream);
}

void IndexIVF::addCoarseCentroidsCtrlCpu(AscendTensor<float16_t, DIMS_2> &src,
                                         AscendTensor<float16_t, DIMS_4> &dst)
{
    int numCoarseCents = src.getSize(0);
    int dimCoarseCents = src.getSize(1);
    int dim1 = dst.getSize(0);
    int dim2 = dst.getSize(1);
    for (int i = 0; i < dim1; i++) {
        for (int j = 0; j < dim2; j++) {
            float16_t *tmpData = dst[i][j].data();
            int hpadding = (j == (dim2 - 1)) ?
                ((j + 1) * CUBE_ALIGN - dimCoarseCents) : 0;
            int vpadding = (i == (dim1 - 1)) ?
                ((i + 1) * CUBE_ALIGN - numCoarseCents) : 0;
            for (int v = 0; v < (CUBE_ALIGN - vpadding); v++) {
                auto err = memcpy_s(tmpData,
                                    static_cast<size_t>(CUBE_ALIGN - hpadding) * sizeof(float16_t),
                                    src[i * CUBE_ALIGN + v][j * CUBE_ALIGN].data(),
                                    static_cast<size_t>(CUBE_ALIGN - hpadding) * sizeof(float16_t));
                ASCEND_THROW_IF_NOT_FMT(err == EOK, "memcpy err, i=%d, j=%d, err=%d", i, j, err);
                tmpData += (CUBE_ALIGN - hpadding);

                if (hpadding) {
                    auto err = memset_s(tmpData, sizeof(float16_t) * static_cast<size_t>(hpadding),
                                        0x0, sizeof(float16_t) * static_cast<size_t>(hpadding));
                    ASCEND_THROW_IF_NOT_FMT(err == EOK, "memset err, i=%d, j=%d, err=%d", i, j, err);
                    tmpData += hpadding;
                }
            }

            for (int vp = 0; vp < vpadding; vp++) {
                auto err = memset_s(tmpData, sizeof(float16_t) * static_cast<size_t>(CUBE_ALIGN),
                                    0x0, sizeof(float16_t) * static_cast<size_t>(CUBE_ALIGN));
                ASCEND_THROW_IF_NOT_FMT(err == EOK, "memset err, i=%d, j=%d, err=%d", i, j, err);
                tmpData += CUBE_ALIGN;
            }
        }
    }
}

void IndexIVF::addCoarseCentroidsAiCpu(AscendTensor<float16_t, DIMS_2> &src,
                                       AscendTensor<float16_t, DIMS_4> &dst)
{
    std::string opName = "TransdataShaped";
    auto &mem = resources.getMemoryManager();
    auto streamPtr = resources.getDefaultStream();
    auto stream = streamPtr->GetStream();
    AscendTensor<int64_t, DIMS_1> attr(mem, {aicpu::TRANSDATA_SHAPED_ATTR_IDX_COUNT}, stream);
    attr[aicpu::TRANSDATA_SHAPED_ATTR_NTOTAL_IDX] = 0;
    LaunchOpTwoInOneOut<float16_t, DIMS_2, ACL_FLOAT16,
                        int64_t, DIMS_1, ACL_INT64,
                        float16_t, DIMS_4, ACL_FLOAT16>(opName, stream, src, attr, dst);
    auto ret = synchronizeStream(stream);
    ASCEND_THROW_IF_NOT_FMT(ret == ACL_SUCCESS, "synchronizeStream addCoarseCentroids stream failed: %i\n", ret);
}
}  // namespace ascend
