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


#include "index_custom/IndexIVFSQTIPAicpu.h"

#include "ascenddaemon/utils/Limits.h"
#include "common/utils/CommonUtils.h"
#include "common/utils/OpLauncher.h"
#include "ops/cpukernel/impl/utils/kernel_shared_def.h"

namespace ascend {
namespace {
const int L3_PAGE_SIZE = 8192;        // page size for base in huge_page
const int BASE_SIZE = 16384;           // base num for l3 operator to gen baseSegmentOffset
const int SUBCENTER_NUM = 64;          // Subcenter num for each nlist
const int L3_SEGMENT_SIZE = 64;        // L3 ops segment size
const int MASK_UNIT = 8;               // L3 ops mask unit
const int L3_BURST_LEN = 16;           // L3 ops burst len to get max
const int MASK_MAX_VALUE = 255;        // L3 MASK_MAX_VALUE
const int THREADS_CNT_T = faiss::ascend::SocUtils::GetInstance().GetThreadsCnt();
const int L1_QUERY_BATCH = 32;
const int L2_QUERY_BATCH = faiss::ascend::SocUtils::GetInstance().GetQueryBatch();
const int L3_QUERY_BATCH = faiss::ascend::SocUtils::GetInstance().GetQueryBatch();
}

IndexIVFSQTIPAicpu::IndexIVFSQTIPAicpu(int numList, int dimIn, int dimOut, int nprobes, int64_t resourceSize)
    : IndexIVFSQCIPAicpu(numList, dimIn, dimOut, nprobes, resourceSize),
    threadPool(CREATE_UNIQUE_PTR(AscendThreadPool, THREADS_CNT_T)) {}

IndexIVFSQTIPAicpu::~IndexIVFSQTIPAicpu() {}

APP_ERROR IndexIVFSQTIPAicpu::init()
{
    APP_LOG_INFO("IndexIVFSQTIPAicpu Initialize operation started.\n");
    APPERR_RETURN_IF_NOT_OK(resetL1DistOp(numLists));
    APPERR_RETURN_IF_NOT_OK(resetSubcentersDistOp());
    APPERR_RETURN_IF_NOT_OK(resetSqXDistOp());
    searchBatchSizes = {4096, 2048, 1024, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1};
    APPERR_RETURN_IF_NOT_OK(resetTopkIvfsqtL1Op());
    APPERR_RETURN_IF_NOT_OK(resetTopkIvfsqtL2Op());
    APPERR_RETURN_IF_NOT_OK(resetTopkIvfFuzzyOp());
    APPERR_RETURN_IF_NOT_OK(initL1TopkAttrs());
    APPERR_RETURN_IF_NOT_OK(initL2TopkAttrs());

    transdataShapedAttr = AscendTensor<int64_t, DIMS_1>({ aicpu::TRANSDATA_SHAPED_ATTR_IDX_COUNT });
    transdataShapedAttr[aicpu::TRANSDATA_SHAPED_ATTR_NTOTAL_IDX] = 0;

    APP_LOG_INFO("IndexIVFSQTIPAicpu Initialize operation end.\n");
    return APP_ERR_OK;
}

APP_ERROR IndexIVFSQTIPAicpu::reset()
{
    APP_LOG_INFO("IndexIVFSQTIPAicpu reset operation started.\n");
    deviceListData.clear();
    deviceListIndices.clear();

    clearDeviceTmpData();

    pageIds.resize(numLists * SUBCENTER_NUM);
    offsetsInPage.resize(numLists * SUBCENTER_NUM);

    maxListLength = 0;
    this->ntotal = 0;
    APP_LOG_INFO("IndexIVFSQTIPAicpu reset operation end.\n");
    return APP_ERR_OK;
}

void IndexIVFSQTIPAicpu::clearDeviceTmpData()
{
    APP_LOG_INFO("IndexIVFSQTIPAicpu clearDeviceTmpData operation started.\n");
    for (size_t pageId = 0; pageId < deviceTmpData.size(); pageId++) {
        deviceTmpData[pageId]->clear();
    }
    deviceTmpData.clear();
    APP_LOG_INFO("IndexIVFSQTIPAicpu clearDeviceTmpData operation end.\n");
}

APP_ERROR IndexIVFSQTIPAicpu::resetSubcentersDistOp()
{
    APP_LOG_INFO("IndexIVFSQTIPAicpu resetSubcentersDistOp operation start.\n");
    auto subcentersDistOpReset = [&](std::unique_ptr<AscendOperator> &op, int64_t batch) {
        AscendOpDesc desc("DistanceFlatSubcenters");
        std::vector<int64_t> queryShape({ batch, dims });
        std::vector<int64_t> subcentersShape({ numLists, utils::divUp(dims, CUBE_ALIGN), SUBCENTER_NUM, CUBE_ALIGN });
        std::vector<int64_t> preNormsShape({ numLists, SUBCENTER_NUM });
        std::vector<int64_t> offsetsShape({ batch, nprobe });
        std::vector<int64_t> distResultShape({ batch, nprobe, SUBCENTER_NUM });
        // the result constain min value and index, the multi 2
        std::vector<int64_t> flagShape({ CORE_NUM, FLAG_SIZE });

        desc.addInputTensorDesc(ACL_FLOAT16, queryShape.size(), queryShape.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_FLOAT16, subcentersShape.size(), subcentersShape.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_FLOAT16, preNormsShape.size(), preNormsShape.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_UINT16, offsetsShape.size(), offsetsShape.data(), ACL_FORMAT_ND);
        desc.addOutputTensorDesc(ACL_FLOAT16, distResultShape.size(), distResultShape.data(), ACL_FORMAT_ND);
        desc.addOutputTensorDesc(ACL_UINT16, flagShape.size(), flagShape.data(), ACL_FORMAT_ND);

        op.reset();
        op = CREATE_UNIQUE_PTR(AscendOperator, desc);
        return op->init();
    };

    for (auto batch : {1, 2, 4, 8, 16}) {
        subcentersDistOps[batch] = std::unique_ptr<AscendOperator>(nullptr);
        APPERR_RETURN_IF_NOT_LOG(subcentersDistOpReset(subcentersDistOps[batch], batch),
            APP_ERR_ACL_OP_LOAD_MODEL_FAILED, "subcentersDist op init failed");
    }
    APP_LOG_INFO("IndexIVFSQTIPAicpu resetSubcentersDistOp operation end.\n");
    return APP_ERR_OK;
}

APP_ERROR IndexIVFSQTIPAicpu::resetSqXDistOp()
{
    APP_LOG_INFO("IndexIVFSQTIPAicpu resetSqXDistOp operation started.\n");
    for (auto batch : {1, 2, 4, 8, 16}) {
        AscendOpDesc desc("DistanceIVFSQ8IPX");
        std::vector<int64_t> queryShape({ batch, dimOut });
        std::vector<int64_t> baseSegmentShape({ BASE_SIZE * dimOut});
        std::vector<int64_t> segmentOffsetShape({ batch, l3SegmentNum });
        std::vector<int64_t> vdmShape({ 2, dimOut });
        std::vector<int64_t> resultShape({ batch, l3SegmentNum * L3_SEGMENT_SIZE });
        std::vector<int64_t> maxResultShape({ batch,
            l3SegmentNum * 2 * (L3_SEGMENT_SIZE / L3_BURST_LEN) }); // each maximum contains 2 values
        std::vector<int64_t> flagShape({ CORE_NUM, FLAG_SIZE });

        desc.addInputTensorDesc(ACL_FLOAT16, queryShape.size(), queryShape.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_UINT8, baseSegmentShape.size(), baseSegmentShape.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_UINT64, segmentOffsetShape.size(), segmentOffsetShape.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_FLOAT16, vdmShape.size(), vdmShape.data(), ACL_FORMAT_ND);

        desc.addOutputTensorDesc(ACL_FLOAT16, resultShape.size(), resultShape.data(), ACL_FORMAT_ND);
        desc.addOutputTensorDesc(ACL_FLOAT16, maxResultShape.size(), maxResultShape.data(), ACL_FORMAT_ND);
        desc.addOutputTensorDesc(ACL_UINT16, flagShape.size(), flagShape.data(), ACL_FORMAT_ND);

        distSqXOps[batch] = CREATE_UNIQUE_PTR(AscendOperator, desc);
        APPERR_RETURN_IF_NOT_LOG(distSqXOps[batch]->init(), APP_ERR_ACL_OP_LOAD_MODEL_FAILED,
            "DistanceIVFSQ8IPX op init failed");
    }
    APP_LOG_INFO("IndexIVFSQTIPAicpu resetSqXDistOp operation end.\n");
    return APP_ERR_OK;
}

APP_ERROR IndexIVFSQTIPAicpu::addTmpDeviceData(int n, int dim, const uint8_t *data)
{
    APP_LOG_INFO("IndexIVFSQTIPAicpu addTmpDeviceData operation started.\n");
    int curPageId = static_cast<int64_t>(deviceTmpData.size());
    deviceTmpData.emplace_back(CREATE_UNIQUE_PTR(DeviceVector<unsigned char>, MemorySpace::DEVICE));
    deviceTmpData[curPageId]->append(data, n * dim);
    APP_LOG_INFO("IndexIVFSQTIPAicpu addTmpDeviceData operation end.\n");
    return APP_ERR_OK;
}

size_t IndexIVFSQTIPAicpu::getTmpPageNums() const
{
    APP_LOG_INFO("IndexIVFSQTIPAicpu getTmpPageNums operation called.\n");
    return deviceTmpData.size();
}

DeviceVector<unsigned char>& IndexIVFSQTIPAicpu::getPageTmpData(int pageId)
{
    APP_LOG_INFO("IndexIVFSQTIPAicpu getPageTmpData operation called.\n");
    ASCEND_THROW_IF_NOT(pageId >= 0 && static_cast<size_t>(pageId) < deviceTmpData.size());
    return *(deviceTmpData[pageId]);
}

size_t IndexIVFSQTIPAicpu::getPageLength(int pageId) const
{
    ASCEND_THROW_IF_NOT(((size_t)pageId < deviceListIndices.size()) && (pageId >= 0));
    return deviceListIndices[pageId]->size();
}

APP_ERROR IndexIVFSQTIPAicpu::getPageIndices(int pageId, idx_t* ids) const
{
    APPERR_RETURN_IF_NOT_FMT(((size_t)pageId < deviceListIndices.size()) && (pageId >= 0),
        APP_ERR_INNER_ERROR, "pageId must be [0, %ld)", deviceListIndices.size());
    auto &data = *deviceListIndices[pageId];
    size_t listIndicesSize = deviceListIndices[pageId]->size();
    int ret = aclrtMemcpy(ids, listIndicesSize * sizeof(idx_t),
        data.data(), listIndicesSize * sizeof(idx_t), ACL_MEMCPY_DEVICE_TO_HOST);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "cp indices to host error %d", (int)ret);

    return APP_ERR_OK;
}

APP_ERROR IndexIVFSQTIPAicpu::getPageVectorsReshaped(int pageId, uint8_t* reshaped) const
{
    APPERR_RETURN_IF_NOT_FMT(((size_t)pageId < deviceListIndices.size()) && (pageId >= 0), APP_ERR_INVALID_PARAM,
        "pageId is out of deviceList size, pageId=%d, listSize=%ld", pageId, deviceListIndices.size());

    size_t size = getPageLength(pageId);
    auto &data = *deviceListData[pageId];
    int dimShaped = utils::divUp(this->dimOut, CUBE_ALIGN);
    auto memErr = EOK;

    size_t listDataSize = deviceListData[pageId]->size();
    std::vector<uint8_t> listDataHost(listDataSize);
    int ret = aclrtMemcpy(listDataHost.data(), listDataSize, data.data(), listDataSize, ACL_MEMCPY_DEVICE_TO_HOST);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "cp list to host error %d", (int)ret);

// reshape code from Zz format data to contigous format.
#pragma omp parallel for if (size >= 100) num_threads(CommonUtils::GetThreadMaxNums())
    for (size_t i = 0; i < size; i++) {
        int offset = getPageShapedDataOffset(i);
        auto srcPtr = listDataHost.data() + offset;
        auto dstPtr = reshaped + i * this->dimOut * sizeof(unsigned char);
        for (int j = 0; j < dimShaped; j++) {
            auto err = memcpy_s(dstPtr + j * CUBE_ALIGN,
                                CUBE_ALIGN * sizeof(unsigned char),
                                srcPtr + j * L3_PAGE_SIZE * CUBE_ALIGN,
                                CUBE_ALIGN * sizeof(unsigned char));
            ASCEND_EXC_IF_NOT_FMT(err == EOK, memErr = err, "memcpy src error, i=%d, j=%d, err=%d", i, j, err);
        }
    }
    APPERR_RETURN_IF_NOT_LOG(memErr == EOK, APP_ERR_INNER_ERROR, "memcpy error");

    return APP_ERR_OK;
}

APP_ERROR IndexIVFSQTIPAicpu::updateSubCentroidsData(int total, float16_t* x)
{
    APP_LOG_INFO("IndexIVFSQTIPAicpu updateSubCentroidsData operation started.\n");
    auto streamPtr = resources.getDefaultStream();
    auto stream = streamPtr->GetStream();
    auto &mem = resources.getMemoryManager();
    int numSubCents = total / numLists;
    AscendTensor<float16_t, DIMS_3> subCentroidsData(x, { numLists, numSubCents, dimIn });

    int dim2 = utils::divUp(dimIn, CUBE_ALIGN);
    if (AscendMultiThreadManager::IsMultiThreadMode()) {
        AscendTensor<float16_t, DIMS_4> tmpShapedCentroids({ numLists, dim2, numSubCents, CUBE_ALIGN });
        subCentroidsShaped = std::move(tmpShapedCentroids);
    } else {
        AscendTensor<float16_t, DIMS_4> tmpShapedCentroids(
            mem, { numLists, dim2, numSubCents, CUBE_ALIGN }, stream);
        subCentroidsShaped = std::move(tmpShapedCentroids);
    }
    std::vector<float16_t> tmpShapedCentroidsVecHost(subCentroidsShaped.numElements());
    AscendTensor<float16_t, DIMS_4> tmpShapedCentroidsHost(
        tmpShapedCentroidsVecHost.data(), { numLists, dim2, numSubCents, CUBE_ALIGN });

    auto memErr = EOK;

    for (int h = 0; h < numLists; h++) {
        for (int j = 0; j < dim2; j++) {
            float16_t *tmpData = tmpShapedCentroidsHost[h][j].data();
            int hpadding = (j == (dim2 - 1)) ? ((j + 1) * CUBE_ALIGN - dimIn) : 0;
            for (int v = 0; v < numSubCents; v++) {
                auto err = memcpy_s(tmpData, (CUBE_ALIGN - hpadding) * sizeof(float16_t),
                    subCentroidsData[h][v][j * CUBE_ALIGN].data(),
                    (CUBE_ALIGN - hpadding) * sizeof(float16_t));
                ASCEND_EXC_IF_NOT_FMT(err == EOK, memErr = err, "memcpy subCentroids error, h=%d, err=%d", h, err);
                tmpData += (CUBE_ALIGN - hpadding);

                if (hpadding) {
                    auto err = memset_s(tmpData, sizeof(float16_t) * hpadding, 0x0, sizeof(float16_t) * hpadding);
                    ASCEND_EXC_IF_NOT_FMT(err == EOK, memErr = err, "set subCentroids error, h=%d, err=%d", h, err);
                    tmpData += hpadding;
                }
            }
        }
    }
    APPERR_RETURN_IF_NOT_LOG(memErr == EOK, APP_ERR_INNER_ERROR, "memcpy error");
    int ret = aclrtMemcpy(subCentroidsShaped.data(), subCentroidsShaped.getSizeInBytes(),
        tmpShapedCentroidsHost.data(), tmpShapedCentroidsHost.getSizeInBytes(), ACL_MEMCPY_HOST_TO_DEVICE);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR,
        "aclrtMemcpy subCentroidsShaped error %d", (int)ret);

    // update L2 norm of subCoarseCentroids
    AscendTensor<float16_t, DIMS_1> tmpNormTensor({ total });
    AscendTensor<float16_t, DIMS_2> rawData(mem, { total, dimIn }, stream);
    ret = aclrtMemcpy(rawData.data(), rawData.getSizeInBytes(),
        x, total * dimIn * sizeof(float16_t), ACL_MEMCPY_HOST_TO_DEVICE);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR,
        "aclrtMemcpy adapSubCents to deivce error %d", ret);
    fvecNormsL2sqrAicpu(tmpNormTensor, rawData);
    normSubCentroids = std::move(tmpNormTensor);

    AscendTensor<float16_t, DIMS_4> subcentersTmp(subCentroidsShaped.data(),
        { numLists, dimIn / CUBE_ALIGN, SUBCENTER_NUM, CUBE_ALIGN });
    AscendTensor<float16_t, DIMS_2> precomputedTmp(normSubCentroids.data(), { numLists, SUBCENTER_NUM });

    subcenters = std::move(subcentersTmp);
    precomputed = std::move(precomputedTmp);
    APP_LOG_INFO("IndexIVFSQTIPAicpu updateSubCentroidsData operation end.\n");
    return APP_ERR_OK;
}

int IndexIVFSQTIPAicpu::getPageShapedDataOffset(int idx) const
{
    return idx * CUBE_ALIGN;
}

APP_ERROR IndexIVFSQTIPAicpu::addPageVectors(size_t numVecs, const uint8_t *codes, const idx_t *indices)
{
    APP_LOG_INFO("IndexIVFSQTIPAicpu addPageVectors operation started.\n");
    APPERR_RETURN_IF(numVecs == 0 || numVecs > L3_PAGE_SIZE, APP_ERR_OK);
    APPERR_RETURN_IF_NOT_LOG(this->isTrained, APP_ERR_ILLEGAL_OPERATION, "the index is not trained");

    this->dims = dimOut;

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
    auto streamPtr = resources.getDefaultStream();
    auto stream = streamPtr->GetStream();
    auto &mem = resources.getMemoryManager();
    AscendTensor<uint8_t, DIMS_2> codesData(mem, { static_cast<int>(numVecs), this->dims }, stream);
    auto ret = aclrtMemcpy(codesData.data(), codesData.getSizeInBytes(),
        codes, codesData.getSizeInBytes(), ACL_MEMCPY_HOST_TO_DEVICE);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "Failed to copy codes to device, ret:%d", ret);

    int curPageId = static_cast<int64_t>(deviceListIndices.size());
    deviceListData.emplace_back(CREATE_UNIQUE_PTR(DeviceVector<unsigned char>, MemorySpace::DEVICE));
    deviceListIndices.emplace_back(CREATE_UNIQUE_PTR(DeviceVector<idx_t>, MemorySpace::DEVICE));
    deviceListData[curPageId]->resize(L3_PAGE_SIZE * this->dims, true);
    deviceListIndices[curPageId]->append(indices, numVecs, true);

    AscendTensor<uint8_t, DIMS_4> dst(static_cast<uint8_t *>(deviceListData[curPageId]->data()),
        { utils::divUp(static_cast<int>(numVecs), L3_PAGE_SIZE),
          utils::divUp(this->dims, CUBE_ALIGN), L3_PAGE_SIZE, CUBE_ALIGN });

    std::string opName = "TransdataShaped";
    LaunchOpTwoInOneOut<uint8_t, DIMS_2, ACL_UINT8, int64_t, DIMS_1, ACL_INT64, uint8_t, DIMS_4, ACL_UINT8>(
        opName, stream, codesData, transdataShapedAttr, dst);

    ret = synchronizeStream(stream);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR,
        "synchronizeStream failed: %i\n", ret);

    this->ntotal += numVecs;
    this->dims = dimIn;
    APP_LOG_INFO("IndexIVFSQTIPAicpu addPageVectors operation end.\n");
    return APP_ERR_OK;
}

void IndexIVFSQTIPAicpu::receiveBaseOffsetNum(int n, int listId, const int* offset, const int* num)
{
    APP_LOG_INFO("IndexIVFSQTIPAicpu receiveBaseOffsetNum operation started.\n");
    if (baseOffset.find(listId) != baseOffset.end()) {
        baseOffset.erase(listId);
    }
    baseOffset.emplace(listId, std::vector<int>(offset, offset + n));

    if (baseNums.find(listId) != baseNums.end()) {
        baseNums.erase(listId);
    }
    baseNums.emplace(listId, std::vector<int>(num, num + n));
    APP_LOG_INFO("IndexIVFSQTIPAicpu receiveBaseOffsetNum operation end.\n");
}

void IndexIVFSQTIPAicpu::getBaseMaskSeg()
{
    APP_LOG_INFO("IndexIVFSQTIPAicpu getBaseMaskSeg operation started.\n");
    ASCEND_THROW_IF_NOT_MSG(deviceListData.size() > 0,
        "deviceListData is empty, please check the database maybe too similar to classify");
    pListBase = deviceListData[0]->data();
    for (size_t i = 1; i < deviceListData.size(); ++i) {
        pListBase = std::min(pListBase, deviceListData[i]->data());
    }

    // Determine the offset of a sub-bucket based on the sub-bucket information.
    maxListLength = L3_PAGE_SIZE;
    size_t curPageId = 0;
    size_t pageOffset = 0;
    for (int listId = 0; listId < numLists; listId++) {
        for (int subListId = 0; subListId < SUBCENTER_NUM; subListId++) {
            int sublistGlobalId = listId * SUBCENTER_NUM + subListId;
            int curNum = baseNums[listId][subListId];
            pageIds[sublistGlobalId] = curPageId;
            offsetsInPage[sublistGlobalId] = static_cast<int>(pageOffset);
            pageOffset += static_cast<size_t>(curNum);
            if (curNum == 0) {
                continue;
            }
            if (pageOffset == deviceListIndices[curPageId]->size()) {
                pageOffset = 0;
                curPageId += 1;
                if (curPageId >= deviceListIndices.size()) {
                    break;
                }
            }
        }
    }
    std::vector<int> subListSegNum(numLists * SUBCENTER_NUM);
    std::vector<uint64_t> subListOffset(numLists * SUBCENTER_NUM);
    std::vector<idx_t *> subListIndicesOffset(numLists * SUBCENTER_NUM);
    std::vector<uint32_t> subListSizes(numLists * SUBCENTER_NUM);

#pragma omp parallel for num_threads(CommonUtils::GetThreadMaxNums())
    for (int listId = 0; listId < numLists; listId++) {
        for (int subListId = 0; subListId < SUBCENTER_NUM; subListId++) {
            int curOffset = baseOffset[listId][subListId];
            int curNum = baseNums[listId][subListId];

            if (curOffset != -1 && curNum != 0) {
                int sublistGlobalId = listId * SUBCENTER_NUM + subListId;
                int segs = (curNum + L3_SEGMENT_SIZE - 1) / L3_SEGMENT_SIZE;
                subListSegNum[sublistGlobalId] = segs;
                int offsetInPage = offsetsInPage[sublistGlobalId];
                subListOffset[sublistGlobalId] = deviceListData[pageIds[sublistGlobalId]]->data() - pListBase +
                    getPageShapedDataOffset(offsetInPage);
                subListIndicesOffset[sublistGlobalId] = deviceListIndices[pageIds[sublistGlobalId]]->data() +
                    offsetInPage;

                int lastSegNum = curNum % L3_SEGMENT_SIZE;
                subListSizes[sublistGlobalId] = lastSegNum;
                if (lastSegNum == 0) {
                    subListSizes[sublistGlobalId] = L3_SEGMENT_SIZE;
                }
            }
        }
    }

    AscendTensor<int, DIMS_1> subListSegNumTmp({ numLists * SUBCENTER_NUM });
    int ret = aclrtMemcpy(subListSegNumTmp.data(), subListSegNumTmp.getSizeInBytes(),
        subListSegNum.data(), subListSegNum.size() * sizeof(int), ACL_MEMCPY_HOST_TO_DEVICE);
    ASCEND_THROW_IF_NOT_FMT(ret == ACL_SUCCESS, "cpoy subListSegNum to device failed %d", (int)ret);
    subListSegNumT = std::move(subListSegNumTmp);

    AscendTensor<uint64_t, DIMS_1> subListOffsetTmp({ numLists * SUBCENTER_NUM });
    ret = aclrtMemcpy(subListOffsetTmp.data(), subListOffsetTmp.getSizeInBytes(),
        subListOffset.data(), subListOffset.size() * sizeof(uint64_t), ACL_MEMCPY_HOST_TO_DEVICE);
    ASCEND_THROW_IF_NOT_FMT(ret == ACL_SUCCESS, "cpoy subListOffset to device failed %d", (int)ret);
    subListOffsetT = std::move(subListOffsetTmp);

    AscendTensor<idx_t*, DIMS_1> subListIndicesOffsetTmp({ numLists * SUBCENTER_NUM });
    ret = aclrtMemcpy(subListIndicesOffsetTmp.data(), subListIndicesOffsetTmp.getSizeInBytes(),
        subListIndicesOffset.data(), subListIndicesOffset.size() * sizeof(int64_t*), ACL_MEMCPY_HOST_TO_DEVICE);
    ASCEND_THROW_IF_NOT_FMT(ret == ACL_SUCCESS, "cpoy subListIndicesOffset to device failed %d", (int)ret);
    subListIndicesOffsetT = std::move(subListIndicesOffsetTmp);

    AscendTensor<uint32_t, DIMS_1> subListSizesTmp({ numLists * SUBCENTER_NUM });
    ret = aclrtMemcpy(subListSizesTmp.data(), subListSizesTmp.getSizeInBytes(),
        subListSizes.data(), subListSizes.size() * sizeof(uint32_t), ACL_MEMCPY_HOST_TO_DEVICE);
    ASCEND_THROW_IF_NOT_FMT(ret == ACL_SUCCESS, "cpoy subListSizes to device failed %d", (int)ret);
    subListSizesT = std::move(subListSizesTmp);

    // clear sub-bucket data
    baseOffset.clear();
    baseOffset.reserve(numLists);
    baseNums.clear();
    baseNums.reserve(numLists);
    APP_LOG_INFO("IndexIVFSQTIPAicpu getBaseMaskSeg operation end.\n");
}

void IndexIVFSQTIPAicpu::setSearchParams(int nprobe, int l2Probe, int l3SegmentNum)
{
    APP_LOG_INFO("IndexIVFSQTIPAicpu setSearchParams operation started.\n");
    this->nprobe = nprobe;
    this->l2NProbe = l2Probe;
    this->l3SegmentNum = l3SegmentNum;

    int ret = resetTopkIvfsqtL2Op();
    ASCEND_THROW_IF_NOT_FMT(ret == APP_ERR_OK, "reset l2 topk op failed, error :%d", ret);
    ret = resetSubcentersDistOp();
    ASCEND_THROW_IF_NOT_FMT(ret == APP_ERR_OK, "reset subcenters dist op failed, error :%d", ret);
    ret = resetSqXDistOp();
    ASCEND_THROW_IF_NOT_FMT(ret == APP_ERR_OK, "reset SqX op failed, error :%d", ret);
    ret = initL1TopkAttrs();
    ASCEND_THROW_IF_NOT_FMT(ret == APP_ERR_OK, "init l1 attrs failed, error :%d", ret);
    ret = initL2TopkAttrs();
    ASCEND_THROW_IF_NOT_FMT(ret == APP_ERR_OK, "init l2 attrs failed, error :%d", ret);
    APP_LOG_INFO("IndexIVFSQTIPAicpu setSearchParams operation end.\n");
}

void IndexIVFSQTIPAicpu::setSortMode(int mode)
{
    APP_LOG_INFO("IndexIVFSQTIPAicpu setSortMode operation started.\n");
    this->ivfFuzzyTopkMode = mode;
    APP_LOG_INFO("IndexIVFSQTIPAicpu setSortMode operation end.\n");
}

void IndexIVFSQTIPAicpu::updateTParams(int l2NProbeRpc, int l3SegmentNumRpc)
{
    APP_LOG_INFO("IndexIVFSQTIPAicpu updateTParams operation started.\n");
    this->l2NProbe = l2NProbeRpc;
    this->l3SegmentNum = l3SegmentNumRpc;
    int ret = resetSqXDistOp();
    ASCEND_THROW_IF_NOT_FMT(ret == APP_ERR_OK, "reset SqX op failed, error :%d", ret);
    ret = initL2TopkAttrs();
    ASCEND_THROW_IF_NOT_FMT(ret == APP_ERR_OK, "init l2 attrs failed, error :%d", ret);
    APP_LOG_INFO("IndexIVFSQTIPAicpu updateTParams operation end.\n");
}

void IndexIVFSQTIPAicpu::setNumProbes(int nprobes)
{
    APP_LOG_INFO("IndexIVFSQTIPAicpu setNumProbes operation started.\n");
    ASCEND_THROW_IF_NOT_MSG(nprobes > 0, "nprobe must be greater than 0");
    ASCEND_THROW_IF_NOT_MSG(nprobes <= numLists, "nprobe must be less than or equal to nlist");
    this->nprobe = nprobes;
    int ret = resetTopkIvfsqtL2Op();
    ASCEND_THROW_IF_NOT_FMT(ret == APP_ERR_OK, "reset l2 topk op failed, error :%d", ret);
    ret = resetSubcentersDistOp();
    ASCEND_THROW_IF_NOT_FMT(ret == APP_ERR_OK, "reset subcenters dist op failed, error :%d", ret);
    ret = initL1TopkAttrs();
    ASCEND_THROW_IF_NOT_FMT(ret == APP_ERR_OK, "init l1 attrs failed, error :%d", ret);
    ret = initL2TopkAttrs();
    ASCEND_THROW_IF_NOT_FMT(ret == APP_ERR_OK, "init l2 attrs failed, error :%d", ret);
    APP_LOG_INFO("IndexIVFSQTIPAicpu setNumProbes operation end.\n");
}

void IndexIVFSQTIPAicpu::setRatio(int kBufferRatio, int kHeapRatio)
{
    this->kBufferRatio = kBufferRatio;
    this->kHeapRatio = kHeapRatio;
}

APP_ERROR IndexIVFSQTIPAicpu::searchImpl(int n, const float16_t *x, int k, float16_t *dists, idx_t *labels)
{
    APP_LOG_INFO("IndexIVFSQTIPAicpu searchImpl operation started.\n");
    auto streamPtr = resources.getDefaultStream();
    auto stream = streamPtr->GetStream();
    auto &mem = resources.getMemoryManager();
    AscendTensor<float16_t, DIMS_2> queries(mem, { n, dimIn }, stream);
    auto ret = aclrtMemcpy(queries.data(), queries.getSizeInBytes(),
                           x, queries.getSizeInBytes(), ACL_MEMCPY_HOST_TO_DEVICE);
    APPERR_RETURN_IF_NOT_LOG(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "Failed to copy query to device");

    // 1. L1 search nprobe IVF list
    AscendTensor<uint16_t, DIMS_2> l1KIndices(mem, { n, nprobe }, stream);
    AscendTensor<float16_t, DIMS_2> compressQueries(mem, { n, dimOut }, stream);  // need zero, do in aicpu
    ret = searchImplL1(queries, l1KIndices, compressQueries);
    APPERR_RETURN_IF_NOT_FMT(ret == APP_ERR_OK, APP_ERR_INNER_ERROR, "ivfsqt search l1 failed: %i\n", ret);

    // 2. L2 search, search codes in nprobe IVF list to find topk results
    // L2 output, if SUBCENTER_NUM is dynymic need to cat tiles
    AscendTensor<uint64_t, DIMS_2> subListOffsetL3(mem, { n, l3SegmentNum }, stream);
    AscendTensor<int64_t, DIMS_2> idResult(mem, { n, l3SegmentNum }, stream);
    AscendTensor<uint32_t, DIMS_2> opSize(mem, { n, l3SegmentNum }, stream);
    ret = searchImplL2(queries, l1KIndices, subListOffsetL3, idResult, opSize);
    APPERR_RETURN_IF_NOT_FMT(ret == APP_ERR_OK, APP_ERR_INNER_ERROR, "ivfsqt search l2 failed: %i\n", ret);

    // 3. L3 search, search topk in l2nprobe list
    AscendTensor<float16_t, DIMS_2> outDists(mem, { n, k }, stream);
    AscendTensor<int64_t, DIMS_2> outIndices(mem, { n, k }, stream);
    ret = searchImplL3X(compressQueries, subListOffsetL3, idResult, opSize, outDists, outIndices);
    APPERR_RETURN_IF_NOT_FMT(ret == APP_ERR_OK, APP_ERR_INNER_ERROR, "ivfsqt search l3 failed: %i\n", ret);
    ret = aclrtMemcpy(dists, n * k * sizeof(float16_t),
                      outDists.data(), outDists.getSizeInBytes(), ACL_MEMCPY_DEVICE_TO_HOST);
    APPERR_RETURN_IF_NOT_LOG(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "Failed to copy outDists to host");
    ret = aclrtMemcpy(labels, n * k * sizeof(idx_t),
                      outIndices.data(), outIndices.getSizeInBytes(), ACL_MEMCPY_DEVICE_TO_HOST);
    APPERR_RETURN_IF_NOT_LOG(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "Failed to copy labels to host");
    APP_LOG_INFO("IndexIVFSQTIPAicpu searchImpl operation end.\n");
    return APP_ERR_OK;
}

APP_ERROR IndexIVFSQTIPAicpu::searchImplL1(const AscendTensor<float16_t, DIMS_2> &queries,
                                           AscendTensor<uint16_t, DIMS_2> &l1KIndices,
                                           AscendTensor<float16_t, DIMS_2> &queryOut)
{
    APP_LOG_INFO("IndexIVFSQTIPAicpu searchImplL1 operation started.\n");
    auto &mem = resources.getMemoryManager();
    auto streamPtr = resources.getDefaultStream();
    auto stream = streamPtr->GetStream();
    auto streamAicpuPtr = resources.getAlternateStreams();
    auto streamAicpu = streamAicpuPtr[0]->GetStream();
    int n = queries.getSize(0);

    AscendTensor<float16_t, DIMS_2> l1Dists(mem, { n, numLists }, stream);
    // the result constain min value and index, the multi 2
    int minDistSize = std::max(numLists / BURST_LEN * 2, MIN_EXTREME_SIZE);
    AscendTensor<float16_t, DIMS_2> l1DistMins(mem, { n, minDistSize }, stream);

    AscendTensor<uint16_t, DIMS_3> opFlag(mem, { n, CORE_NUM, FLAG_SIZE }, stream);
    std::vector<uint16_t> opFlagVec(n * CORE_NUM * FLAG_SIZE);
    int ret = aclrtMemcpy(opFlag.data(), opFlag.getSizeInBytes(),
        opFlagVec.data(), opFlagVec.size() * sizeof(uint16_t), ACL_MEMCPY_HOST_TO_DEVICE);
    ASCEND_THROW_IF_NOT_FMT(ret == ACL_SUCCESS, "cpoy opFlag to device failed %d", (int)ret);

    AscendTensor<float16_t, DIMS_2> outDistances(mem, { n, nprobe }, streamAicpu);

    AscendTensor<int64_t, DIMS_1> attrsInput(mem, { aicpu::TOPK_IVFSQT_L1_ATTR_IDX_COUNT }, streamAicpu);
    std::vector<int64_t> attrsHost(aicpu::TOPK_IVFSQT_L1_ATTR_IDX_COUNT);
    attrsHost[aicpu::TOPK_IVFSQT_L1_ATTR_ASC_IDX] = 1;
    attrsHost[aicpu::TOPK_IVFSQT_L1_ATTR_K_IDX] = nprobe;
    attrsHost[aicpu::TOPK_IVFSQT_L1_ATTR_BURST_LEN_IDX] = BURST_LEN;
    attrsHost[aicpu::TOPK_IVFSQT_L1_ATTR_OP_SIZE_IDX] = numLists;
    attrsHost[aicpu::TOPK_IVFSQT_L1_ATTR_Q_BATCH_SIZE_IDX] = L1_QUERY_BATCH;
    attrsHost[aicpu::TOPK_IVFSQT_L1_ATTR_QUICK_HEAP] = 1;
    ret = aclrtMemcpy(attrsInput.data(), attrsInput.getSizeInBytes(),
        attrsHost.data(), attrsHost.size() * sizeof(int64_t), ACL_MEMCPY_HOST_TO_DEVICE);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "init l1 topk attrs faile %d", (int)ret);

    for (int nIdx = 0; nIdx < n; nIdx += L1_QUERY_BATCH) {
        int batch = std::min(n - nIdx, L1_QUERY_BATCH);
        int nIdxBase = nIdx / L1_QUERY_BATCH * L1_QUERY_BATCH;

        AscendTensor<float16_t, DIMS_2> queryL1(queries[nIdxBase].data(), { batch, dimIn });
        AscendTensor<float16_t, DIMS_2> l1DistsTmp(l1Dists[nIdxBase].data(), { batch, numLists });
        AscendTensor<float16_t, DIMS_2> l1DistMinsTmp(l1DistMins[nIdxBase].data(), { batch, minDistSize });
        AscendTensor<uint16_t, DIMS_2> opFlag2D(opFlag[nIdxBase].data(), { CORE_NUM, FLAG_SIZE });
        runL1DistOp(queryL1, coarseCentroidsShaped, normCoarseCentroids, l1DistsTmp, l1DistMinsTmp, opFlag2D, stream);

        if (nIdx == 0) {
            runTopkIvfsqtL1Op(l1Dists, l1DistMins, opFlag, attrsInput, queries, vcompressIndex, vcompressValue,
                outDistances, l1KIndices, queryOut, streamAicpu);
        }
    }

    ret = synchronizeStream(stream);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR,
        "synchronizeStream aicore stream failed: %i\n", ret);

    ret = synchronizeStream(streamAicpu);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR,
        "synchronizeStream aicpu stream failed: %i\n", ret);
    APP_LOG_INFO("IndexIVFSQTIPAicpu searchImplL1 operation end.\n");
    return APP_ERR_OK;
}

APP_ERROR IndexIVFSQTIPAicpu::resetTopkIvfsqtL1Op()
{
    APP_LOG_INFO("IndexIVFSQTIPAicpu resetTopkIvfsqtL1Op operation started.\n");
    auto opReset = [&](std::unique_ptr<AscendOperator> &op, int batch) {
        AscendOpDesc desc("TopkIvfsqtL1");
        int ratio = dimIn / dimOut;
        int minDistSize = std::max(numLists / BURST_LEN * 2, MIN_EXTREME_SIZE);
        std::vector<int64_t> distsShape({ batch, numLists });
        std::vector<int64_t> vmDistsShape({ batch, minDistSize });
        std::vector<int64_t> opFlagShape({ batch, CORE_NUM, FLAG_SIZE });
        std::vector<int64_t> attrShape { aicpu::TOPK_IVFSQT_L1_ATTR_IDX_COUNT };
        std::vector<int64_t> shapeQueryIn { batch, dimIn };
        std::vector<int64_t> shapeCompressIndex { dimOut, ratio };
        std::vector<int64_t> shapeCompressValue { ratio, dimOut };

        desc.addInputTensorDesc(ACL_FLOAT16, distsShape.size(), distsShape.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_FLOAT16, vmDistsShape.size(), vmDistsShape.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_UINT16, opFlagShape.size(), opFlagShape.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_INT64, attrShape.size(), attrShape.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_FLOAT16, shapeQueryIn.size(), shapeQueryIn.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_INT32, shapeCompressIndex.size(), shapeCompressIndex.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_FLOAT, shapeCompressValue.size(), shapeCompressValue.data(), ACL_FORMAT_ND);

        std::vector<int64_t> outDistsShape { batch, 0 };
        std::vector<int64_t> outLabelsShape { batch, 0 };
        std::vector<int64_t> shapeQueryOut { batch, dimOut };
        desc.addOutputTensorDesc(ACL_FLOAT16, outDistsShape.size(), outDistsShape.data(), ACL_FORMAT_ND);
        desc.addOutputTensorDesc(ACL_UINT16, outLabelsShape.size(), outLabelsShape.data(), ACL_FORMAT_ND);
        desc.addOutputTensorDesc(ACL_FLOAT16, shapeQueryOut.size(), shapeQueryOut.data(), ACL_FORMAT_ND);

        op.reset();
        op = CREATE_UNIQUE_PTR(AscendOperator, desc);
        return op->init();
    };

    for (auto batch : searchBatchSizes) {
        topkIvfsqtL1Ops[batch] = std::unique_ptr<AscendOperator>(nullptr);
        APPERR_RETURN_IF_NOT_LOG(opReset(topkIvfsqtL1Ops[batch], batch),
            APP_ERR_ACL_OP_LOAD_MODEL_FAILED, "Failed to init topkIvfsqtL1Ops");
    }
    APP_LOG_INFO("IndexIVFSQTIPAicpu resetTopkIvfsqtL1Op operation end.\n");
    return APP_ERR_OK;
}

APP_ERROR IndexIVFSQTIPAicpu::runTopkIvfsqtL1Op(const AscendTensor<float16_t, DIMS_2> &dists,
                                                const AscendTensor<float16_t, DIMS_2> &vmDists,
                                                const AscendTensor<uint16_t, DIMS_3> &opFlag,
                                                const AscendTensor<int64_t, DIMS_1> &attr,
                                                const AscendTensor<float16_t, DIMS_2> &queryIn,
                                                const AscendTensor<int, DIMS_2> &compressIndex,
                                                const AscendTensor<float, DIMS_2> &compressValue,
                                                AscendTensor<float16_t, DIMS_2> &outDists,
                                                AscendTensor<uint16_t, DIMS_2> &outLabels,
                                                AscendTensor<float16_t, DIMS_2> &queryOut,
                                                aclrtStream stream)
{
    APP_LOG_INFO("IndexIVFSQTIPAicpu runTopkIvfsqtL1Op operation started.\n");
    AscendOperator *topkIvfsqtL1Op = nullptr;
    int batch = dists.getSize(0);
    if (topkIvfsqtL1Ops.count(batch) != 0) {
        topkIvfsqtL1Op = topkIvfsqtL1Ops[batch].get();
    }
    ASCEND_THROW_IF_NOT(topkIvfsqtL1Op);

    // prepare for input data's buffer
    std::shared_ptr<std::vector<const aclDataBuffer *>> topkIvfsqtL1OpInput(
        new std::vector<const aclDataBuffer *>(), CommonUtils::AclInputBufferDelete);
    topkIvfsqtL1OpInput->emplace_back(aclCreateDataBuffer(dists.data(), dists.getSizeInBytes()));
    topkIvfsqtL1OpInput->emplace_back(aclCreateDataBuffer(vmDists.data(), vmDists.getSizeInBytes()));
    topkIvfsqtL1OpInput->emplace_back(aclCreateDataBuffer(opFlag.data(), opFlag.getSizeInBytes()));
    topkIvfsqtL1OpInput->emplace_back(aclCreateDataBuffer(attr.data(), attr.getSizeInBytes()));
    topkIvfsqtL1OpInput->emplace_back(aclCreateDataBuffer(queryIn.data(), queryIn.getSizeInBytes()));
    topkIvfsqtL1OpInput->emplace_back(aclCreateDataBuffer(compressIndex.data(), compressIndex.getSizeInBytes()));
    topkIvfsqtL1OpInput->emplace_back(aclCreateDataBuffer(compressValue.data(), compressValue.getSizeInBytes()));

    // prepare for output data's buffer
    std::shared_ptr<std::vector<aclDataBuffer *>> topkIvfsqtL1OpOutput(
        new std::vector<aclDataBuffer *>(), CommonUtils::AclOutputBufferDelete);
    topkIvfsqtL1OpOutput->emplace_back(aclCreateDataBuffer(outDists.data(), outDists.getSizeInBytes()));
    topkIvfsqtL1OpOutput->emplace_back(aclCreateDataBuffer(outLabels.data(), outLabels.getSizeInBytes()));
    topkIvfsqtL1OpOutput->emplace_back(aclCreateDataBuffer(queryOut.data(), queryOut.getSizeInBytes()));

    // async executing operator
    topkIvfsqtL1Op->exec(*topkIvfsqtL1OpInput, *topkIvfsqtL1OpOutput, stream);
    APP_LOG_INFO("IndexIVFSQTIPAicpu runTopkIvfsqtL1Op operation end.\n");
    return APP_ERR_OK;
}

APP_ERROR IndexIVFSQTIPAicpu::resetTopkIvfsqtL2Op()
{
    APP_LOG_INFO("IndexIVFSQTIPAicpu resetTopkIvfsqtL2Op operation started.\n");
    auto opReset = [&](std::unique_ptr<AscendOperator> &op, int batch) {
        AscendOpDesc desc("TopkIvfsqtL2");
        int centroisNum = nprobe * SUBCENTER_NUM;
        std::vector<int64_t> distsShape({ batch, centroisNum });
        std::vector<int64_t> opFlagShape({ batch, CORE_NUM, FLAG_SIZE });
        std::vector<int64_t> attrShape { aicpu::TOPK_IVFSQT_L2_ATTR_IDX_COUNT };
        std::vector<int64_t> subListSegNumShape { numLists * SUBCENTER_NUM };
        std::vector<int64_t> subListOffsetShape { numLists * SUBCENTER_NUM };
        std::vector<int64_t> subListIndicesOffsetShape { numLists * SUBCENTER_NUM };
        std::vector<int64_t> subListSizesShape { numLists * SUBCENTER_NUM };
        std::vector<int64_t> l1KIndicesShape { batch, 0 };

        desc.addInputTensorDesc(ACL_FLOAT16, distsShape.size(), distsShape.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_UINT16, opFlagShape.size(), opFlagShape.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_INT64, attrShape.size(), attrShape.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_INT32, subListSegNumShape.size(), subListSegNumShape.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_UINT64, subListOffsetShape.size(), subListOffsetShape.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_INT64, subListIndicesOffsetShape.size(),
            subListIndicesOffsetShape.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_UINT32, subListSizesShape.size(), subListSizesShape.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_UINT16, l1KIndicesShape.size(), l1KIndicesShape.data(), ACL_FORMAT_ND);

        std::vector<int64_t> subListOffsetL3Shape { batch, 0 };
        std::vector<int64_t> idResultShape { batch, 0 };
        std::vector<int64_t> opSizeShape { batch, 0 };
        desc.addOutputTensorDesc(ACL_UINT64, subListOffsetL3Shape.size(), subListOffsetL3Shape.data(), ACL_FORMAT_ND);
        desc.addOutputTensorDesc(ACL_INT64, idResultShape.size(), idResultShape.data(), ACL_FORMAT_ND);
        desc.addOutputTensorDesc(ACL_UINT32, opSizeShape.size(), opSizeShape.data(), ACL_FORMAT_ND);

        op.reset();
        op = CREATE_UNIQUE_PTR(AscendOperator, desc);
        return op->init();
    };

    for (auto batch : searchBatchSizes) {
        topkIvfsqtL2Ops[batch] = std::unique_ptr<AscendOperator>(nullptr);
        APPERR_RETURN_IF_NOT_LOG(opReset(topkIvfsqtL2Ops[batch], batch),
            APP_ERR_ACL_OP_LOAD_MODEL_FAILED, "Failed to init topkIvfsqtL2Ops");
    }
    APP_LOG_INFO("IndexIVFSQTIPAicpu resetTopkIvfsqtL2Op operation end.\n");
    return APP_ERR_OK;
}

APP_ERROR IndexIVFSQTIPAicpu::runTopkIvfsqtL2Op(const AscendTensor<float16_t, DIMS_2> &dists,
                                                const AscendTensor<uint16_t, DIMS_3> &opFlag,
                                                const AscendTensor<int64_t, DIMS_1> &attr,
                                                const AscendTensor<int, DIMS_1> &listSegNum,
                                                const AscendTensor<uint64_t, DIMS_1> &listOffset,
                                                const AscendTensor<idx_t *, DIMS_1> &listIndicesOffset,
                                                const AscendTensor<uint32_t, DIMS_1> &listSizes,
                                                const AscendTensor<uint16_t, DIMS_2> &l1KIndices,
                                                AscendTensor<uint64_t, DIMS_2> &listOffsetL3,
                                                AscendTensor<int64_t, DIMS_2> &idResult,
                                                AscendTensor<uint32_t, DIMS_2> &opSize,
                                                aclrtStream stream)
{
    APP_LOG_INFO("IndexIVFSQTIPAicpu runTopkIvfsqtL2Op operation started.\n");
    AscendOperator *topkIvfsqtL2Op = nullptr;
    int batch = dists.getSize(0);
    if (topkIvfsqtL2Ops.count(batch) != 0) {
        topkIvfsqtL2Op = topkIvfsqtL2Ops[batch].get();
    }
    ASCEND_THROW_IF_NOT(topkIvfsqtL2Op);

    // prepare for input data's buffer
    std::shared_ptr<std::vector<const aclDataBuffer *>> topkIvfsqtL2OpInput(
        new std::vector<const aclDataBuffer *>(), CommonUtils::AclInputBufferDelete);
    topkIvfsqtL2OpInput->emplace_back(aclCreateDataBuffer(dists.data(), dists.getSizeInBytes()));
    topkIvfsqtL2OpInput->emplace_back(aclCreateDataBuffer(opFlag.data(), opFlag.getSizeInBytes()));
    topkIvfsqtL2OpInput->emplace_back(aclCreateDataBuffer(attr.data(), attr.getSizeInBytes()));
    topkIvfsqtL2OpInput->emplace_back(aclCreateDataBuffer(listSegNum.data(), listSegNum.getSizeInBytes()));
    topkIvfsqtL2OpInput->emplace_back(aclCreateDataBuffer(listOffset.data(), listOffset.getSizeInBytes()));
    topkIvfsqtL2OpInput->emplace_back(
        aclCreateDataBuffer(listIndicesOffset.data(), listIndicesOffset.getSizeInBytes()));
    topkIvfsqtL2OpInput->emplace_back(aclCreateDataBuffer(listSizes.data(), listSizes.getSizeInBytes()));
    topkIvfsqtL2OpInput->emplace_back(aclCreateDataBuffer(l1KIndices.data(), l1KIndices.getSizeInBytes()));

    // prepare for output data's buffer
    std::shared_ptr<std::vector<aclDataBuffer *>> topkIvfsqtL2OpOutput(
        new std::vector<aclDataBuffer *>(), CommonUtils::AclOutputBufferDelete);
    topkIvfsqtL2OpOutput->emplace_back(aclCreateDataBuffer(listOffsetL3.data(), listOffsetL3.getSizeInBytes()));
    topkIvfsqtL2OpOutput->emplace_back(aclCreateDataBuffer(idResult.data(), idResult.getSizeInBytes()));
    topkIvfsqtL2OpOutput->emplace_back(aclCreateDataBuffer(opSize.data(), opSize.getSizeInBytes()));

    // async executing operator
    topkIvfsqtL2Op->exec(*topkIvfsqtL2OpInput, *topkIvfsqtL2OpOutput, stream);
    APP_LOG_INFO("IndexIVFSQTIPAicpu runTopkIvfsqtL2Op operation end.\n");
    return APP_ERR_OK;
}

APP_ERROR IndexIVFSQTIPAicpu::initL1TopkAttrs()
{
    AscendTensor<int64_t, DIMS_1> attrsInput({ aicpu::TOPK_IVFSQT_L1_ATTR_IDX_COUNT });
    std::vector<int64_t> attrsHost(aicpu::TOPK_IVFSQT_L1_ATTR_IDX_COUNT);
    attrsHost[aicpu::TOPK_IVFSQT_L1_ATTR_ASC_IDX] = 1;
    attrsHost[aicpu::TOPK_IVFSQT_L1_ATTR_K_IDX] = nprobe;
    attrsHost[aicpu::TOPK_IVFSQT_L1_ATTR_BURST_LEN_IDX] = BURST_LEN;
    attrsHost[aicpu::TOPK_IVFSQT_L1_ATTR_OP_SIZE_IDX] = numLists;
    attrsHost[aicpu::TOPK_IVFSQT_L1_ATTR_Q_BATCH_SIZE_IDX] = L1_QUERY_BATCH;
    attrsHost[aicpu::TOPK_IVFSQT_L1_ATTR_QUICK_HEAP] = 1;
    int ret = aclrtMemcpy(attrsInput.data(), attrsInput.getSizeInBytes(),
        attrsHost.data(), attrsHost.size() * sizeof(int64_t), ACL_MEMCPY_HOST_TO_DEVICE);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "init l1 topk attrs faile %d", (int)ret);

    l1Attrs = std::move(attrsInput);
    return APP_ERR_OK;
}

APP_ERROR IndexIVFSQTIPAicpu::initL2TopkAttrs()
{
    AscendTensor<int64_t, DIMS_1> attrsInput({ aicpu::TOPK_IVFSQT_L2_ATTR_IDX_COUNT });
    std::vector<int64_t> attrsHost(aicpu::TOPK_IVFSQT_L2_ATTR_IDX_COUNT);
    attrsHost[aicpu::TOPK_IVFSQT_L2_ATTR_K_IDX] = l2NProbe;
    attrsHost[aicpu::TOPK_IVFSQT_L2_ATTR_SUBCENTER_NUM_IDX] = SUBCENTER_NUM;
    attrsHost[aicpu::TOPK_IVFSQT_L2_ATTR_L3_SEG_NUM_IDX] = l3SegmentNum;
    attrsHost[aicpu::TOPK_IVFSQT_L2_ATTR_L3_SEG_SIZE_IDX] = L3_SEGMENT_SIZE;
    attrsHost[aicpu::TOPK_IVFSQT_L2_ATTR_PAGE_SHAPED_DATA_OFFSET_STEP_IDX] = getPageShapedDataOffset(L3_SEGMENT_SIZE);
    attrsHost[aicpu::TOPK_IVFSQT_L2_ATTR_L1_NPROBE_IDX] = nprobe;
    attrsHost[aicpu::TOPK_IVFSQT_L2_ATTR_Q_BATCH_SIZE_IDX] = L2_QUERY_BATCH;
    int ret = aclrtMemcpy(attrsInput.data(), attrsInput.getSizeInBytes(),
        attrsHost.data(), attrsHost.size() * sizeof(int64_t), ACL_MEMCPY_HOST_TO_DEVICE);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "init l2 topk attrs failed %d", (int)ret);

    l2Attrs = std::move(attrsInput);
    return APP_ERR_OK;
}

APP_ERROR IndexIVFSQTIPAicpu::searchImplL2(const AscendTensor<float16_t, DIMS_2> &queries,
                                           const AscendTensor<uint16_t, DIMS_2> &l1KIndices,
                                           AscendTensor<uint64_t, DIMS_2> &subListOffsetL3,
                                           AscendTensor<int64_t, DIMS_2> &idResult,
                                           AscendTensor<uint32_t, DIMS_2> &opSize)
{
    APP_LOG_INFO("IndexIVFSQTIPAicpu searchImplL2 operation started.\n");
    auto streamPtr = resources.getDefaultStream();
    auto stream = streamPtr->GetStream();
    auto streamAicpuPtr = resources.getAlternateStreams()[0];
    auto streamAicpu = streamAicpuPtr->GetStream();
    auto &mem = resources.getMemoryManager();
    int n = queries.getSize(0);

    // subcentersDistOp Output
    AscendTensor<float16_t, DIMS_3> subDists(mem, { n, nprobe, SUBCENTER_NUM }, stream);
    AscendTensor<uint16_t, DIMS_3> opFlagSub(mem, { n, CORE_NUM, FLAG_SIZE }, stream);
    std::vector<uint16_t> opFlagSubVec(n * CORE_NUM * FLAG_SIZE);
    int ret = aclrtMemcpy(opFlagSub.data(), opFlagSub.getSizeInBytes(),
        opFlagSubVec.data(), opFlagSubVec.size() * sizeof(uint16_t), ACL_MEMCPY_HOST_TO_DEVICE);
    ASCEND_THROW_IF_NOT_FMT(ret == ACL_SUCCESS, "cpoy opFlagSub to device failed %d", (int)ret);

    int subCentroisNum = nprobe * SUBCENTER_NUM;
    AscendTensor<float16_t, DIMS_2> subTopkDistances(mem, { n, l2NProbe }, streamAicpu);
    AscendTensor<int64_t, DIMS_2> subTopkIndices(mem, { n, l2NProbe }, streamAicpu);

    AscendTensor<float16_t, DIMS_2> subDistsShaped(subDists.data(), { n, subCentroisNum });

    for (int nIdx = 0; nIdx < n; nIdx += L2_QUERY_BATCH) {
        int batch = std::min(n - nIdx, L2_QUERY_BATCH);
        int nIdxBase = nIdx / L2_QUERY_BATCH * L2_QUERY_BATCH;

        AscendTensor<float16_t, DIMS_2> queryL1(queries[nIdxBase].data(), { batch, dimIn });
        AscendTensor<uint16_t, DIMS_2> offsetsTemp(l1KIndices[nIdxBase].data(), { batch, nprobe });
        AscendTensor<float16_t, DIMS_3> distsTemp(subDists[nIdxBase].data(), { batch, nprobe, SUBCENTER_NUM });
        AscendTensor<uint16_t, DIMS_2> opFlag2D(opFlagSub[nIdxBase].data(), { CORE_NUM, FLAG_SIZE });
        runSubcentersDistOp(queryL1, subcenters, precomputed, offsetsTemp, distsTemp, opFlag2D, stream);

        if (nIdx == 0) {
            runTopkIvfsqtL2Op(subDistsShaped, opFlagSub, l2Attrs, subListSegNumT, subListOffsetT,
                subListIndicesOffsetT, subListSizesT, l1KIndices, subListOffsetL3, idResult, opSize, streamAicpu);
        }
    }

    ret = synchronizeStream(stream);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR,
        "synchronizeStream aicore stream failed: %i\n", ret);

    ret = synchronizeStream(streamAicpu);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR,
        "synchronizeStream aicpu stream failed: %i\n", ret);
    APP_LOG_INFO("IndexIVFSQTIPAicpu searchImplL2 operation end.\n");
    return APP_ERR_OK;
}

void IndexIVFSQTIPAicpu::runSubcentersDistOp(const AscendTensor<float16_t, DIMS_2>& queryVecs,
                                             const AscendTensor<float16_t, DIMS_4>& shapedData,
                                             const AscendTensor<float16_t, DIMS_2>& norms,
                                             const AscendTensor<uint16_t, DIMS_2>& offsets,
                                             AscendTensor<float16_t, DIMS_3>& outDists,
                                             AscendTensor<uint16_t, DIMS_2>& flag,
                                             aclrtStream stream)
{
    APP_LOG_INFO("IndexIVFSQTIPAicpu runSubcentersDistOp operation started.\n");
    AscendOperator *op = nullptr;
    int batch = queryVecs.getSize(0);
    if (subcentersDistOps.find(batch) != subcentersDistOps.end()) {
        op = subcentersDistOps[batch].get();
    }
    ASCEND_THROW_IF_NOT(op);

    std::shared_ptr<std::vector<const aclDataBuffer *>> distOpInput(
        new std::vector<const aclDataBuffer *>(), CommonUtils::AclInputBufferDelete);
    distOpInput->emplace_back(aclCreateDataBuffer(queryVecs.data(), queryVecs.getSizeInBytes()));
    distOpInput->emplace_back(aclCreateDataBuffer(shapedData.data(), shapedData.getSizeInBytes()));
    distOpInput->emplace_back(aclCreateDataBuffer(norms.data(), norms.getSizeInBytes()));
    distOpInput->emplace_back(aclCreateDataBuffer(offsets.data(), offsets.getSizeInBytes()));

    std::shared_ptr<std::vector<aclDataBuffer *>> distOpOutput(
        new std::vector<aclDataBuffer *>(), CommonUtils::AclOutputBufferDelete);
    distOpOutput->emplace_back(aclCreateDataBuffer(outDists.data(), outDists.getSizeInBytes()));
    distOpOutput->emplace_back(aclCreateDataBuffer(flag.data(), flag.getSizeInBytes()));

    op->exec(*distOpInput, *distOpOutput, stream);
    APP_LOG_INFO("IndexIVFSQTIPAicpu runSubcentersDistOp operation end.\n");
}

void IndexIVFSQTIPAicpu::setPoppedDistAndIndex(float16_t* const distsRpc, idx_t* const indexRpc)
{
    this->distancePopped = distsRpc;
    this->indexPopped = indexRpc;
}

APP_ERROR IndexIVFSQTIPAicpu::searchImplL3X(const AscendTensor<float16_t, DIMS_2> &queries,
                                            const AscendTensor<uint64_t, DIMS_2> &subListOffsetL3,
                                            const AscendTensor<int64_t, DIMS_2> &idResult,
                                            const AscendTensor<uint32_t, DIMS_2> &opSize,
                                            AscendTensor<float16_t, DIMS_2> &outDists,
                                            AscendTensor<int64_t, DIMS_2> &outIndices)
{
    APP_LOG_INFO("IndexIVFSQTIPAicpu searchImplL3X operation started.\n");
    auto streamPtr = resources.getDefaultStream();
    auto stream = streamPtr->GetStream();
    auto &mem = resources.getMemoryManager();
    auto streamAicpuPtr = resources.getAlternateStreams()[0];
    auto streamAicpu = streamAicpuPtr->GetStream();
    int n = queries.getSize(0);
    int k = outDists.getSize(1);

    int extremeLen = 2 * (L3_SEGMENT_SIZE / L3_BURST_LEN); // each maximum contains 2 values
    // tensor for operator flags
    AscendTensor<uint16_t, DIMS_3> opFlag(mem, { n, CORE_NUM, FLAG_SIZE}, stream);
    std::vector<uint16_t> opFlagVec(n * CORE_NUM * FLAG_SIZE);
    int ret = aclrtMemcpy(opFlag.data(), opFlag.getSizeInBytes(),
        opFlagVec.data(), opFlagVec.size() * sizeof(uint16_t), ACL_MEMCPY_HOST_TO_DEVICE);
    ASCEND_THROW_IF_NOT_FMT(ret == ACL_SUCCESS, "cpoy opFlag to device failed %d", (int)ret);

    AscendTensor<float16_t, DIMS_2> distResult(mem, { n, l3SegmentNum * L3_SEGMENT_SIZE }, stream);
    AscendTensor<float16_t, DIMS_2> maxDistResult(mem, { n, l3SegmentNum * extremeLen }, stream);

    AscendTensor<int64_t, DIMS_1> attrs(mem, { aicpu::TOPK_IVF_FUZZY_ATTR_IDX_COUNT }, streamAicpu);
    ret = initTopkIvfFuzzyAttrs(0, k, L3_QUERY_BATCH, attrs);  // ip distance
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "init topk attrs failed %d", (int)ret);

    int kBuffer = k / kBufferRatio;
    AscendTensor<float16_t, DIMS_2> popDists(mem, { n, kBuffer }, streamAicpu);
    AscendTensor<int64_t, DIMS_2> popLabels(mem, { n, kBuffer }, streamAicpu);

    ret = aclrtMemset(popDists.data(), popDists.getSizeInBytes(), 0, popDists.getSizeInBytes());
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "init popDists to 0s failed: %d", ret);

    for (int nIdx = 0; nIdx < n; nIdx += L3_QUERY_BATCH) {
        int batch = std::min(L3_QUERY_BATCH, n - nIdx);

        AscendTensor<float16_t, DIMS_2> query(queries[nIdx].data(), { batch, dimOut });
        AscendTensor<uint8_t, DIMS_1> base(pListBase, { BASE_SIZE * dimOut });
        AscendTensor<uint64_t, DIMS_2> offset(subListOffsetL3[nIdx].data(), { batch, l3SegmentNum });
        AscendTensor<float16_t, DIMS_2> result(distResult[nIdx].data(), { batch, l3SegmentNum * L3_SEGMENT_SIZE });
        AscendTensor<float16_t, DIMS_2> maxResult(maxDistResult[nIdx].data(), { batch, l3SegmentNum * extremeLen});
        AscendTensor<uint16_t, DIMS_2> flag(opFlag[nIdx].data(), { CORE_NUM, FLAG_SIZE });

        runSqXDistOp(query, base, offset, result, maxResult, flag, stream);

        if (nIdx == 0) {
            runTopkIvfFuzzyOp(distResult, maxDistResult, idResult, opSize, opFlag, attrs,
                outDists, outIndices, popDists, popLabels, streamAicpu);
        }
    }

    ret = synchronizeStream(stream);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "synchronize aicore stream failed: %i\n", ret);

    ret = synchronizeStream(streamAicpu);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "synchronize aicpu stream failed: %i\n", ret);

    ret = copyPopToHost(popDists, popLabels);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "copyPopToHost error %d", ret);
    APP_LOG_INFO("IndexIVFSQTIPAicpu searchImplL3X operation end.\n");
    return APP_ERR_OK;
}

APP_ERROR IndexIVFSQTIPAicpu::copyPopToHost(AscendTensor<float16_t, DIMS_2> &popDists,
    AscendTensor<int64_t, DIMS_2> &popLabels)
{
    auto ret = aclrtMemcpy(this->distancePopped, popDists.getSizeInBytes(),
        popDists.data(), popDists.getSizeInBytes(), ACL_MEMCPY_DEVICE_TO_HOST);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "copy popDists to host error %d", ret);

    ret = aclrtMemcpy(this->indexPopped, popLabels.getSizeInBytes(),
        popLabels.data(), popLabels.getSizeInBytes(), ACL_MEMCPY_DEVICE_TO_HOST);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "copy popLabels to host error %d", ret);
    return ret;
}

void IndexIVFSQTIPAicpu::runSqXDistOp(const AscendTensor<float16_t, DIMS_2> &queries,
                                      const AscendTensor<uint8_t, DIMS_1> &baseSegment,
                                      const AscendTensor<uint64_t, DIMS_2> &segmentOffset,
                                      AscendTensor<float16_t, DIMS_2> &result,
                                      AscendTensor<float16_t, DIMS_2> &maxResult,
                                      AscendTensor<uint16_t, DIMS_2> &flag,
                                      aclrtStream stream)
{
    APP_LOG_INFO("IndexIVFSQTIPAicpu runSqXDistOp operation started.\n");
    AscendOperator *distSqXOp = nullptr;
    int batch = queries.getSize(0);
    if (distSqXOps.find(batch) != distSqXOps.end()) {
        distSqXOp = distSqXOps[batch].get();
    }
    ASCEND_THROW_IF_NOT(distSqXOp);

    // prepare for input data's buffer
    std::shared_ptr<std::vector<const aclDataBuffer *>> distSqXOpInput(
        new std::vector<const aclDataBuffer *>(), CommonUtils::AclInputBufferDelete);
    distSqXOpInput->emplace_back(aclCreateDataBuffer(queries.data(), queries.getSizeInBytes()));
    distSqXOpInput->emplace_back(aclCreateDataBuffer(baseSegment.data(), baseSegment.getSizeInBytes()));
    distSqXOpInput->emplace_back(aclCreateDataBuffer(segmentOffset.data(), segmentOffset.getSizeInBytes()));
    distSqXOpInput->emplace_back(aclCreateDataBuffer(vDM.data(), vDM.getSizeInBytes()));

    // prepare for output data's buffer
    std::shared_ptr<std::vector<aclDataBuffer *>> distSqXOpOutput(
        new std::vector<aclDataBuffer *>(), CommonUtils::AclOutputBufferDelete);
    distSqXOpOutput->emplace_back(aclCreateDataBuffer(result.data(), result.getSizeInBytes()));
    distSqXOpOutput->emplace_back(aclCreateDataBuffer(maxResult.data(), maxResult.getSizeInBytes()));
    distSqXOpOutput->emplace_back(aclCreateDataBuffer(flag.data(), flag.getSizeInBytes()));

    // async executing operator
    distSqXOp->exec(*distSqXOpInput, *distSqXOpOutput, stream);
    APP_LOG_INFO("IndexIVFSQTIPAicpu runSqXDistOp operation end.\n");
}

APP_ERROR IndexIVFSQTIPAicpu::runTopkIvfFuzzyOp(const AscendTensor<float16_t, DIMS_2> &dists,
                                                const AscendTensor<float16_t, DIMS_2> &vmDists,
                                                const AscendTensor<int64_t, DIMS_2> &ids,
                                                const AscendTensor<uint32_t, DIMS_2> &opSize,
                                                const AscendTensor<uint16_t, DIMS_3> &opFlag,
                                                const AscendTensor<int64_t, DIMS_1> &attr,
                                                AscendTensor<float16_t, DIMS_2> &outDists,
                                                AscendTensor<int64_t, DIMS_2> &outLabels,
                                                AscendTensor<float16_t, DIMS_2> &popDists,
                                                AscendTensor<int64_t, DIMS_2> &popLabels,
                                                aclrtStream stream)
{
    APP_LOG_INFO("IndexIVFSQTIPAicpu runTopkIvfFuzzyOp operation started.\n");
    AscendOperator *topkIvfFuzzyOp = nullptr;
    int batch = dists.getSize(0);
    if (topkIvfFuzzyOps.count(batch) != 0) {
        topkIvfFuzzyOp = topkIvfFuzzyOps[batch].get();
    }
    ASCEND_THROW_IF_NOT(topkIvfFuzzyOp);

    // prepare for input data's buffer
    std::shared_ptr<std::vector<const aclDataBuffer *>> topkIvfFuzzyOpInput(
        new std::vector<const aclDataBuffer *>(), CommonUtils::AclInputBufferDelete);
    topkIvfFuzzyOpInput->emplace_back(aclCreateDataBuffer(dists.data(), dists.getSizeInBytes()));
    topkIvfFuzzyOpInput->emplace_back(aclCreateDataBuffer(vmDists.data(), vmDists.getSizeInBytes()));
    topkIvfFuzzyOpInput->emplace_back(aclCreateDataBuffer(ids.data(), ids.getSizeInBytes()));
    topkIvfFuzzyOpInput->emplace_back(aclCreateDataBuffer(opSize.data(), opSize.getSizeInBytes()));
    topkIvfFuzzyOpInput->emplace_back(aclCreateDataBuffer(opFlag.data(), opFlag.getSizeInBytes()));
    topkIvfFuzzyOpInput->emplace_back(aclCreateDataBuffer(attr.data(), attr.getSizeInBytes()));

    // prepare for output data's buffer
    std::shared_ptr<std::vector<aclDataBuffer *>> topkIvfFuzzyOpOutput(
        new std::vector<aclDataBuffer *>(), CommonUtils::AclOutputBufferDelete);
    topkIvfFuzzyOpOutput->emplace_back(aclCreateDataBuffer(outDists.data(), outDists.getSizeInBytes()));
    topkIvfFuzzyOpOutput->emplace_back(aclCreateDataBuffer(outLabels.data(), outLabels.getSizeInBytes()));
    topkIvfFuzzyOpOutput->emplace_back(aclCreateDataBuffer(popDists.data(), popDists.getSizeInBytes()));
    topkIvfFuzzyOpOutput->emplace_back(aclCreateDataBuffer(popLabels.data(), popLabels.getSizeInBytes()));

    // async executing operator
    topkIvfFuzzyOp->exec(*topkIvfFuzzyOpInput, *topkIvfFuzzyOpOutput, stream);
    APP_LOG_INFO("IndexIVFSQTIPAicpu runTopkIvfFuzzyOp operation end.\n");
    return APP_ERR_OK;
}

APP_ERROR IndexIVFSQTIPAicpu::resetTopkIvfFuzzyOp()
{
    APP_LOG_INFO("IndexIVFSQTIPAicpu resetTopkIvfFuzzyOp operation started.\n");
    auto opReset = [&](std::unique_ptr<AscendOperator> &op, int batch) {
        AscendOpDesc desc("TopkIvfFuzzy");
        std::vector<int64_t> distsShape({ batch, 0 });
        std::vector<int64_t> vmDistsShape({ batch, 0 });
        std::vector<int64_t> idsShape({ batch, 0 });
        std::vector<int64_t> opSizeShape({ batch, 0 });
        std::vector<int64_t> opFlagShape({ batch, CORE_NUM, FLAG_SIZE });
        std::vector<int64_t> attrShape { aicpu::TOPK_IVF_FUZZY_ATTR_IDX_COUNT };

        desc.addInputTensorDesc(ACL_FLOAT16, distsShape.size(), distsShape.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_FLOAT16, vmDistsShape.size(), vmDistsShape.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_INT64, idsShape.size(), idsShape.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_UINT32, opSizeShape.size(), opSizeShape.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_UINT16, opFlagShape.size(), opFlagShape.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_INT64, attrShape.size(), attrShape.data(), ACL_FORMAT_ND);

        std::vector<int64_t> outDistsShape { batch, 0 };
        std::vector<int64_t> outLabelsShape { batch, 0 };
        std::vector<int64_t> popDistsShape { batch, 0 };
        std::vector<int64_t> popLabelsShape { batch, 0 };
        desc.addOutputTensorDesc(ACL_FLOAT16, outDistsShape.size(), outDistsShape.data(), ACL_FORMAT_ND);
        desc.addOutputTensorDesc(ACL_INT64, outLabelsShape.size(), outLabelsShape.data(), ACL_FORMAT_ND);
        desc.addOutputTensorDesc(ACL_FLOAT16, popDistsShape.size(), popDistsShape.data(), ACL_FORMAT_ND);
        desc.addOutputTensorDesc(ACL_INT64, popLabelsShape.size(), popLabelsShape.data(), ACL_FORMAT_ND);

        op.reset();
        op = CREATE_UNIQUE_PTR(AscendOperator, desc);
        return op->init();
    };

    for (auto batch : searchBatchSizes) {
        topkIvfFuzzyOps[batch] = std::unique_ptr<AscendOperator>(nullptr);
        APPERR_RETURN_IF_NOT_LOG(opReset(topkIvfFuzzyOps[batch], batch),
            APP_ERR_ACL_OP_LOAD_MODEL_FAILED, "Failed to init topkIvfFuzzyOps");
    }
    APP_LOG_INFO("IndexIVFSQTIPAicpu resetTopkIvfFuzzyOp operation end.\n");
    return APP_ERR_OK;
}

APP_ERROR IndexIVFSQTIPAicpu::initTopkIvfFuzzyAttrs(int asc,
                                                    int k,
                                                    int batch,
                                                    AscendTensor<int64_t, DIMS_1> &attrs) const
{
    std::vector<int64_t> attrsHost(aicpu::TOPK_IVF_FUZZY_ATTR_IDX_COUNT);
    // attrs: [0]asc, [1]k, [2]burst_len, [3]l3 seg num, [4]l3 seg size, [5]k heap ratio, [6]k buf ratio, [7]batch size
    attrsHost[aicpu::TOPK_IVF_FUZZY_ATTR_ASC_IDX] = asc;
    attrsHost[aicpu::TOPK_IVF_FUZZY_ATTR_K_IDX] = k;
    attrsHost[aicpu::TOPK_IVF_FUZZY_ATTR_BURST_LEN_IDX] = L3_BURST_LEN;
    attrsHost[aicpu::TOPK_IVF_FUZZY_ATTR_L3_SEG_NUM_IDX] = l3SegmentNum;
    attrsHost[aicpu::TOPK_IVF_FUZZY_ATTR_L3_SEG_SIZE_IDX] = L3_SEGMENT_SIZE;
    attrsHost[aicpu::TOPK_IVF_FUZZY_ATTR_K_HEAP_RATIO_IDX] = kHeapRatio;
    attrsHost[aicpu::TOPK_IVF_FUZZY_ATTR_K_BUF_RATIO_IDX] = kBufferRatio;
    attrsHost[aicpu::TOPK_IVF_FUZZY_ATTR_Q_BATCH_SIZE_IDX] = batch;
    attrsHost[aicpu::TOPK_IVF_FUZZY_ATTR_SORT_MODE] = ivfFuzzyTopkMode;
    int ret = aclrtMemcpy(attrs.data(), attrs.getSizeInBytes(),
        attrsHost.data(), attrsHost.size() * sizeof(int64_t), ACL_MEMCPY_HOST_TO_DEVICE);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "Mem operator error %d", (int)ret);
    return APP_ERR_OK;
}
} // ascend
