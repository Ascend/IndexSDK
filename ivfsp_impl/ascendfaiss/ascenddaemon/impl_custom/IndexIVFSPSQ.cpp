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

#include "ascenddaemon/impl_custom/IndexIVFSPSQ.h"

#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <fcntl.h>
#include <cstdio>
#include <fstream>
#include <algorithm>
#include <omp.h>
#include <cerrno>
#include <cstring>
#include <memory>
#include <initializer_list>

#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/io.h>
#include <faiss/impl/io_macros.h>

#include "ascenddaemon/impl/AuxIndexStructures.h"
#include "common/utils/CommonUtils.h"
#include "ascenddaemon/utils/Limits.h"
#include "common/utils/OpLauncher.h"
#include "ops/cpukernel/impl/utils/kernel_shared_def.h"
#include "ascendsearch/ascend/utils/fp16.h"
#include "ascenddaemon/utils/IoUtil.h"

namespace ascendSearch {
namespace {
const int FLAT_BLOCK_SIZE = 16384 * 16;
const int THREADS_CNT = 6;
const int SP_SQ_BURST_LEN = 64;
const int SP_CORE_NUM = 8;
const int MASK_SIZE = 64;
const int FILTER_SIZE = 6;
const int ID_BLOCKS = 4;
const int TS_SIZE = 8;
const int BIT_OF_UINT16 = 16;
const int HELPER_SIZE = 128;
const int QUERY_NUM = 1024;
const auto MAX_DATAFILE_SIZE = 56L * 1024L * 1024L * 1024L;
const size_t ONE_BUCKET_MAX_VECTOR_NUM = 400000000;
const size_t LOCAL_SECUREC_MEM_MAX_LEN = 0x7fffffffUL; // max buffer size secure_c supports (2GB)
constexpr int MAGIC_NUMBER_LEN = 4; // 序列化头魔术字长度为4
}

template<typename T>
void CopyDataForLoad(T* dest, const uint8_t* src, size_t sizeBytes, size_t &offset, size_t dataLen)
{
    ASCEND_THROW_IF_NOT_MSG(dataLen >= offset + sizeBytes, "memcpy error: insufficient data length");
    int err = 0;
    size_t copyCounts = sizeBytes / LOCAL_SECUREC_MEM_MAX_LEN;
    for (size_t i = 0; i < copyCounts; ++i) {
        err = memcpy_s(dest + i * LOCAL_SECUREC_MEM_MAX_LEN,
                       std::min(LOCAL_SECUREC_MEM_MAX_LEN, dataLen - offset - i * LOCAL_SECUREC_MEM_MAX_LEN),
                       src + offset + i * LOCAL_SECUREC_MEM_MAX_LEN,
                       LOCAL_SECUREC_MEM_MAX_LEN);
        ASCEND_THROW_IF_NOT_FMT(err == ACL_SUCCESS, "memcpy (error %d)", err);
    }
    size_t remainBytes = sizeBytes - (copyCounts * LOCAL_SECUREC_MEM_MAX_LEN);
    err = memcpy_s(dest + copyCounts * LOCAL_SECUREC_MEM_MAX_LEN,
                   std::min(LOCAL_SECUREC_MEM_MAX_LEN, dataLen - offset - copyCounts * LOCAL_SECUREC_MEM_MAX_LEN),
                   src + offset + copyCounts * LOCAL_SECUREC_MEM_MAX_LEN,
                   remainBytes);
    ASCEND_THROW_IF_NOT_FMT(err == ACL_SUCCESS, "memcpy (error %d)", err);
    offset += sizeBytes;
}

template<typename T>
void CopyDataForSave(uint8_t* dest, T* src, size_t sizeBytes, size_t &offset, size_t dataLen)
{
    ASCEND_THROW_IF_NOT_MSG(dataLen >= offset + sizeBytes, "memcpy error: insufficient data length");
    int err = 0;
    size_t copyCounts = sizeBytes / LOCAL_SECUREC_MEM_MAX_LEN;
    for (size_t i = 0; i < copyCounts; ++i) {
        err = memcpy_s(dest + offset + i * LOCAL_SECUREC_MEM_MAX_LEN,
                       std::min(LOCAL_SECUREC_MEM_MAX_LEN, dataLen - offset - i * LOCAL_SECUREC_MEM_MAX_LEN),
                       src + i * LOCAL_SECUREC_MEM_MAX_LEN,
                       LOCAL_SECUREC_MEM_MAX_LEN);
        ASCEND_THROW_IF_NOT_FMT(err == ACL_SUCCESS, "memcpy (error %d)", err);
    }
    size_t remainBytes = sizeBytes - (copyCounts * LOCAL_SECUREC_MEM_MAX_LEN);
    err = memcpy_s(dest + offset + copyCounts * LOCAL_SECUREC_MEM_MAX_LEN,
                   std::min(LOCAL_SECUREC_MEM_MAX_LEN, dataLen - offset - copyCounts * LOCAL_SECUREC_MEM_MAX_LEN),
                   src + copyCounts * LOCAL_SECUREC_MEM_MAX_LEN,
                   remainBytes);
    ASCEND_THROW_IF_NOT_FMT(err == ACL_SUCCESS, "memcpy (error %d)", err);
    offset += sizeBytes;
}

IndexIVFSPSQ::IndexIVFSPSQ(int dim, int dim2, int nlist, bool encodeResidual, int nprobes,
                           int searchListSize, int handleBatch, bool filterable, int64_t resourceSize)
    : IndexIVF(nlist, dim, dim, nprobes, resourceSize),
      dim2(dim2),
      blockSize(FLAT_BLOCK_SIZE),
      searchListSize(searchListSize),
      handleBatch(handleBatch),
      filterable(filterable),
      pCidIdxBase(nullptr),
      pCidValBase(nullptr),
      pTsBase(nullptr)
{
    VALUE_UNUSED(encodeResidual);
    ASCEND_THROW_IF_NOT(dims % CUBE_ALIGN == 0);

    // resize addFinish to the size of nlist
    addFinishFlag.resize(numLists, 0);

    // IndexIVFSPSQ does not need training
    isTrained = false;

    // supported batch size
    searchBatchSizes = { 128, 64, 48, 36, 32, 30, 24, 18, 16, 12, 8, 6, 4, 2, 1 };
    // Double the SP_SQ_BURST_LEN after round up, hence here we multiply 2
    this->burstsOfBlock = (this->blockSize + SP_SQ_BURST_LEN - 1) / SP_SQ_BURST_LEN * 2;

    int dims1 = utils::divUp(blockSize, CUBE_ALIGN);
    int dims2 = utils::divUp(dims, CUBE_ALIGN);
    this->devVecCapacity = dims1 * dims2 * CUBE_ALIGN * CUBE_ALIGN;

    // init vand and vmul
    AscendTensor<uint16_t, DIMS_2> andData({ ID_BLOCKS, HELPER_SIZE });
    AscendTensor<float16_t, DIMS_2> mulData({ ID_BLOCKS, HELPER_SIZE });

#ifdef HOSTCPU
    for (unsigned int i = 0; i < ID_BLOCKS; ++i) {
        std::vector<uint16_t> andDataTemp(HELPER_SIZE);
        std::vector<float> mulDataFp32Temp(HELPER_SIZE);
        std::vector<uint16_t> mulDataFp16Temp(HELPER_SIZE);
        std::fill_n(andDataTemp.data(), HELPER_SIZE, (1UL << i) + (1UL << (BIT_OF_UINT8 + i)));
        std::fill_n(mulDataFp32Temp.data(), HELPER_SIZE, 1.0 / (1UL << i));

        std::transform(std::begin(mulDataFp32Temp), std::end(mulDataFp32Temp), std::begin(mulDataFp16Temp),
            [](float temp) { return faiss::ascendSearch::fp16(temp).data; });

        auto error = aclrtMemcpy(andData[i].data(), HELPER_SIZE * sizeof(uint16_t),
                                 andDataTemp.data(), HELPER_SIZE * sizeof(uint16_t),
                                 ACL_MEMCPY_HOST_TO_DEVICE);
        ASCEND_THROW_IF_NOT_FMT(error == ACL_SUCCESS, "failed to aclrtMemcpy (error %d)", (int)error);
        error = aclrtMemcpy(mulData[i].data(), HELPER_SIZE * sizeof(float16_t),
                            mulDataFp16Temp.data(), HELPER_SIZE * sizeof(uint16_t),
                            ACL_MEMCPY_HOST_TO_DEVICE);
        ASCEND_THROW_IF_NOT_FMT(error == ACL_SUCCESS, "failed to aclrtMemcpy (error %d)", (int)error);
    }
#else
    for (unsigned int i = 0; i < ID_BLOCKS; ++i) {
        std::fill_n(andData[i].data(), HELPER_SIZE, (1UL << i) + (1UL << (BIT_OF_UINT8 + i)));
        std::fill_n(mulData[i].data(), HELPER_SIZE, 1.0 / (1UL << i));
    }
#endif
    this->vand = std::move(andData);
    this->vmul = std::move(mulData);

    preComputeData.clear();
    for (int i = 0; i < numLists; ++i) {
        preComputeData.push_back(CREATE_UNIQUE_PTR(DeviceVector<float>, MemorySpace::DEVICE));
    }

    pListPreNorms = preComputeData[0]->data();
    for (int i = 1; i < numLists; ++i) {
        pListPreNorms = std::min(pListPreNorms, preComputeData[i]->data());
    }

    for (int i = 0; i < numLists; ++i) {
        deviceAllData.emplace_back(CREATE_UNIQUE_PTR(DeviceVector<unsigned char>, MemorySpace::DEVICE));
    }
    if (filterable) {
        ASCEND_THROW_IF_NOT_MSG(resetCidFilterOperator() == APP_ERR_OK, "reset CidFilter failed");
    }

    addBatchSizes = {1024};
    ASCEND_THROW_IF_NOT_MSG(resetFpToFp16Op() == APP_ERR_OK, "reset FpToFp16 failed");
}

IndexIVFSPSQ::~IndexIVFSPSQ() {}

int IndexIVFSPSQ::getShapedDataOffset(int idx) const
{
    int offset = this->dim2 * utils::roundDown(idx, CUBE_ALIGN);
    offset += (idx % CUBE_ALIGN) * CUBE_ALIGN;
    return offset;
}

APP_ERROR IndexIVFSPSQ::getCodeWord(int n, float *feature, float16_t *codeWord, idx_t *labels)
{
    VALUE_UNUSED(n);
    VALUE_UNUSED(feature);
    VALUE_UNUSED(codeWord);
    VALUE_UNUSED(labels);
    APP_LOG_INFO("IndexIVFSPSQ::getCodeWord\n");
    return APP_ERR_OK;
}

APP_ERROR IndexIVFSPSQ::trainCodeBook(IVFSPCodeBookTrainerInitParam &initParam, float *codebookPtr) const
{
    initParam.nlist = numLists;
    initParam.dim = dims;
    initParam.nonzeroNum = dim2;
    if (codebookPtr != nullptr) {
        initParam.trainAndAdd = true;
    }
    IVFSPCodeBookTrainer codebookTrainer(initParam);
    if (initParam.learnDataPath.empty()) {
        codebookTrainer.ReadMemLearnData(initParam.memLearnData, initParam.memLearnDataSize, initParam.ratio);
    } else {
        codebookTrainer.ReadFile(initParam.learnDataPath, initParam.ratio);
    }
    codebookTrainer.Train(initParam.numIter);
    if (codebookPtr != nullptr) {
        size_t zeroOffset = 0;
        auto codeBookFp32Output = codebookTrainer.GetCodeBook();
        uint8_t *codebookTrainerPtr = reinterpret_cast<uint8_t *>(codeBookFp32Output.data());
        size_t codebookTrainerByteSize = codeBookFp32Output.size() * sizeof(float);
        CopyDataForLoad(codebookPtr, codebookTrainerPtr, codebookTrainerByteSize, zeroOffset, codebookTrainerByteSize);
    }
    return APP_ERR_OK;
}

void IndexIVFSPSQ::updateCoarseCentroidsData(AscendTensor<float16_t, DIMS_2>& coarseCentroidsData)
{
    if (!codebookFinished) {
        int numCoarseCents = coarseCentroidsData.getSize(0);
        int dimCoarseCents = coarseCentroidsData.getSize(1);

        std::initializer_list<int> deviceCoarseCentroidsInit = { numCoarseCents, dimCoarseCents };
        auto deviceCoarseCentroids = std::make_shared<AscendTensor<float16_t, DIMS_2>>(deviceCoarseCentroidsInit);
        deviceCoarseCentroids->copyFromSync(coarseCentroidsData, ACL_MEMCPY_HOST_TO_DEVICE);
        coarseCentroidsIVFSPSQ = deviceCoarseCentroids;

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

        std::initializer_list<int> tmpShapedCentroidsInit = { dim1, dim2, CUBE_ALIGN, CUBE_ALIGN };
        auto tmpShapedCentroids = std::make_shared<AscendTensor<float16_t, DIMS_4>>(tmpShapedCentroidsInit);

        std::initializer_list<int> tmpNormTensorInit = { numCoarseCents };
        auto tmpNormTensor = std::make_shared<AscendTensor<float16_t, DIMS_1>>(tmpNormTensorInit);
#ifdef HOSTCPU
        addCoarseCentroidsAiCpu(*coarseCentroidsIVFSPSQ, *tmpShapedCentroids);
        fvecNormsL2sqrAicpu(*tmpNormTensor, *coarseCentroidsIVFSPSQ);
#else
        addCoarseCentroidsCtrlCpu(*coarseCentroidsData, *tmpShapedCentroids);
        fvecNormsL2sqr(tmpNormTensor->data(), coarseCentroidsData->data(), dimCoarseCents, numCoarseCents);
#endif

        coarseCentroidsShaped = tmpShapedCentroids;
        normCoarseCentroids = tmpNormTensor;
        codebookFinished = true;
    }

    if (this->vMin.data()) {
        this->isTrained = true;
    }
}

void IndexIVFSPSQ::updateCoarseCentroidsData(const IndexIVFSPSQ* loadedIndex)
{
    if (!codebookFinished) {
        coarseCentroidsShaped = loadedIndex->coarseCentroidsShaped;
        normCoarseCentroids = loadedIndex->normCoarseCentroids;
        coarseCentroidsIVFSPSQ = loadedIndex->coarseCentroidsIVFSPSQ;
        codebookFinished = true;
    }

    if (this->vMin.data()) {
        this->isTrained = true;
    }
}

APP_ERROR IndexIVFSPSQ::addDynamicUpdate(int listId)
{
    size_t listLen = bucketSize[listId];
    size_t tmpLen = utils::roundUp(listLen, static_cast<size_t>(CUBE_ALIGN));
    deviceListData[listId]->resize(tmpLen * this->dim2, true);

    uint8_t *tmpData = static_cast<uint8_t *>(deviceAllData[listId]->data());
    auto err = aclrtMemcpy(deviceListData[listId]->data(), tmpLen * this->dim2,
                           tmpData, tmpLen * this->dim2, ACL_MEMCPY_DEVICE_TO_DEVICE);
    APPERR_RETURN_IF_NOT_FMT(err == ACL_SUCCESS, APP_ERR_INNER_ERROR, "Memcpy error %d", (int)err);
    preComputeData[listId]->resize(tmpLen, true);
    float *precompData = preComputeData[listId]->data();
    tmpData = reinterpret_cast<uint8_t *>(deviceAllData[listId]->data()) + tmpLen * this->dim2;
    err = aclrtMemcpy(precompData, tmpLen * sizeof(float),
                      tmpData, tmpLen * sizeof(float), ACL_MEMCPY_DEVICE_TO_DEVICE);
    APPERR_RETURN_IF_NOT_FMT(err == ACL_SUCCESS, APP_ERR_INNER_ERROR, "Memcpy error %d", (int)err);

    deviceListIndices[listId]->resize(listLen, true);
    idx_t *indices = deviceListIndices[listId]->data();
    tmpData = reinterpret_cast<uint8_t *>(deviceAllData[listId]->data()) + tmpLen * (this->dim2 + sizeof(float));
    err = aclrtMemcpy(indices, listLen * sizeof(idx_t),
                      tmpData, listLen * sizeof(idx_t), ACL_MEMCPY_DEVICE_TO_DEVICE);
    APPERR_RETURN_IF_NOT_FMT(err == ACL_SUCCESS, APP_ERR_INNER_ERROR, "Memcpy error %d", (int)err);

    return APP_ERR_OK;
}

void IndexIVFSPSQ::addFinishMerge()
{
    bucketSize.resize(numLists);
    isEmptyList.resize(numLists);
    if (filterable) {
        idxOffset.resize(numLists);
        valOffset.resize(numLists);
        tsOffset.resize(numLists);
    }
    float PRECOMPUTEMAX = 65504.0;

    for (size_t listId = 0; listId < static_cast<size_t>(numLists); listId++) {
        size_t listLen = getListLength(listId);
        if (listLen == 0) {
            deviceListData[listId]->resize(CUBE_ALIGN*this->dim2);
            std::vector<uint8_t> codesZero(CUBE_ALIGN*this->dim2, 0);
            auto err = aclrtMemcpy(static_cast<uint8_t *>(deviceListData[listId]->data()), CUBE_ALIGN*this->dim2,
                static_cast<uint8_t *>(codesZero.data()), CUBE_ALIGN*this->dim2, ACL_MEMCPY_HOST_TO_DEVICE);
            ASCEND_THROW_IF_NOT_FMT(err == ACL_SUCCESS, "Memcpy error %d", err);
            std::vector<float> preComputeMax(CUBE_ALIGN, PRECOMPUTEMAX);
            preComputeData[listId]->resize(CUBE_ALIGN);
            err = aclrtMemcpy(preComputeData[listId]->data(), CUBE_ALIGN*sizeof(float),
                preComputeMax.data(), CUBE_ALIGN*sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE);
            ASCEND_THROW_IF_NOT_FMT(err == ACL_SUCCESS, "Memcpy error %d", err);
            std::vector<idx_t> IndicesPad(CUBE_ALIGN, -1);
            deviceListIndices[listId]->resize(CUBE_ALIGN);
            err = aclrtMemcpy(deviceListIndices[listId]->data(), CUBE_ALIGN * sizeof(idx_t),
                IndicesPad.data(), CUBE_ALIGN * sizeof(idx_t), ACL_MEMCPY_HOST_TO_DEVICE);
            ASCEND_THROW_IF_NOT_FMT(err == ACL_SUCCESS, "Memcpy error %d", err);
            listLen = CUBE_ALIGN;
            isEmptyList[listId] = true;
        } else {
            isEmptyList[listId] = false;
        }
        size_t tmpLen = utils::roundUp(listLen, static_cast<size_t>(CUBE_ALIGN));
        size_t tmpLenAlignFilter = utils::roundUp(listLen, static_cast<size_t>(FILTER_ALIGN));
        if (filterable) {
            deviceAllData[listId]->resize(tmpLen * (this->dim2 + sizeof(idx_t)
                + sizeof(float)) + tmpLenAlignFilter*(sizeof(uint32_t) + sizeof(uint32_t) + sizeof(uint8_t)), true);
        } else {
            deviceAllData[listId]->resize(tmpLen * (this->dim2 + sizeof(idx_t) + sizeof(float)), true);
        }

        uint8_t *dstData = static_cast<uint8_t *>(deviceAllData[listId]->data());
        uint8_t *tmpData = static_cast<uint8_t *>(deviceListData[listId]->data());
        
        auto err = aclrtMemcpy(dstData, tmpLen * this->dim2,
            tmpData, tmpLen * this->dim2, ACL_MEMCPY_DEVICE_TO_DEVICE);
        ASCEND_THROW_IF_NOT_FMT(err == ACL_SUCCESS, "Memcpy error %d", err);

        dstData = reinterpret_cast<uint8_t *>(deviceAllData[listId]->data()) + tmpLen * this->dim2;
        tmpData = reinterpret_cast<uint8_t *>(preComputeData[listId]->data());
        err = aclrtMemcpy(dstData, tmpLen * sizeof(float),
            tmpData, tmpLen * sizeof(float), ACL_MEMCPY_DEVICE_TO_DEVICE);
        ASCEND_THROW_IF_NOT_FMT(err == ACL_SUCCESS, "Memcpy error %d", err);

        dstData = reinterpret_cast<uint8_t *>(deviceAllData[listId]->data()) + tmpLen * (this->dim2 + sizeof(float));
        tmpData = reinterpret_cast<uint8_t *>(deviceListIndices[listId]->data());
        err = aclrtMemcpy(dstData, listLen * sizeof(idx_t),
            tmpData, listLen * sizeof(idx_t), ACL_MEMCPY_DEVICE_TO_DEVICE);
        ASCEND_THROW_IF_NOT_FMT(err == ACL_SUCCESS, "Memcpy error %d", err);

        if (filterable) {
            std::vector<uint8_t> idxData(tmpLenAlignFilter, 0);
            std::vector<uint32_t> valData(tmpLenAlignFilter, 0);
            std::vector<uint32_t> timeData(tmpLenAlignFilter, 0);
            std::vector<idx_t> indicesData(tmpLen, 0);
            err = aclrtMemcpy(indicesData.data(), listLen * sizeof(idx_t),
                deviceListIndices[listId]->data(), listLen * sizeof(idx_t), ACL_MEMCPY_DEVICE_TO_HOST);
            ASCEND_THROW_IF_NOT_FMT(err == ACL_SUCCESS, "Memcpy error %d", err);
            int rightShiftBit = 10;
            for (size_t i = 0; i < listLen; ++i) {
                const int blockSize_ = 32;
                auto cid = static_cast<uint8_t>((indicesData[i] >> 42) & 0x7F); // the 42 bits on the right is others,
                // 0x7F means 1111111, for get 7 bits
                idxData[i] = 1UL << (cid / blockSize_);
                valData[i] = 1UL << (cid % blockSize_);
                timeData[i] = static_cast<uint32_t>((indicesData[i] >> rightShiftBit) & 0xFFFFFFFF);
            }
            dstData = reinterpret_cast<uint8_t *>(deviceAllData[listId]->data())
                + tmpLen * (this->dim2 + sizeof(float) + sizeof(idx_t));
            err = aclrtMemcpy(dstData, tmpLen * sizeof(uint8_t),
                              idxData.data(), tmpLen * sizeof(uint8_t), ACL_MEMCPY_HOST_TO_DEVICE);
            ASCEND_THROW_IF_NOT_FMT(err == ACL_SUCCESS, "Memcpy error %d", err);

            err = aclrtMemcpy(dstData + tmpLenAlignFilter * sizeof(uint8_t), tmpLen * sizeof(uint32_t),
                              valData.data(), tmpLen * sizeof(uint32_t), ACL_MEMCPY_HOST_TO_DEVICE);
            ASCEND_THROW_IF_NOT_FMT(err == ACL_SUCCESS, "Memcpy error %d", err);

            err = aclrtMemcpy(dstData + tmpLenAlignFilter * (sizeof(uint8_t)
                + sizeof(uint32_t)), tmpLen *  sizeof(uint32_t),
                timeData.data(), tmpLen * sizeof(uint32_t), ACL_MEMCPY_HOST_TO_DEVICE);
            ASCEND_THROW_IF_NOT_FMT(err == ACL_SUCCESS, "Memcpy error %d", err);
        }

        bucketSize[listId] = listLen;
        deviceListData[listId]->clear();
        preComputeData[listId]->clear();
        deviceListIndices[listId]->clear();
        addFinishFlag[listId] = 1;
    }

    pListBase = deviceAllData[0]->data();
    for (int i = 1; i < numLists; ++i) {
        pListBase = std::min(pListBase, deviceAllData[i]->data());
    }

    if (filterable) {
        for (int listId = 0; listId < numLists; ++listId) {
            size_t listLen = bucketSize[listId];
            size_t tmpLen = utils::roundUp(listLen, static_cast<size_t>(CUBE_ALIGN));
            size_t tmpLenAlignFilter = utils::roundUp(listLen, static_cast<size_t>(FILTER_ALIGN));
            uint8_t *cidIdxData = reinterpret_cast<uint8_t *>(deviceAllData[listId]->data()) +
                                  tmpLen * (this->dim2 + sizeof(float) + sizeof(idx_t));
            idxOffset[listId] = cidIdxData - reinterpret_cast<uint8_t *>(pListBase);

            int32_t *cidValData = reinterpret_cast<int32_t *>(
                    reinterpret_cast<uint8_t *>(deviceAllData[listId]->data()) +
                    tmpLen * (this->dim2 + sizeof(float) + sizeof(idx_t)) + tmpLenAlignFilter*sizeof(uint8_t));
            valOffset[listId] = cidValData - reinterpret_cast<int32_t *>(pListBase);

            int32_t *timestampsData = reinterpret_cast<int32_t *>(
                reinterpret_cast<uint8_t *>(deviceAllData[listId]->data()) +
                tmpLen * (this->dim2 + sizeof(float) + sizeof(idx_t))
                + tmpLenAlignFilter*(sizeof(uint8_t) + sizeof(uint32_t)));
            tsOffset[listId] = timestampsData - reinterpret_cast<int32_t *>(pListBase);
        }
    }

#ifdef HOSTCPU
#else
    topkQueue.resize(searchBatchSizes[0]);
    int maxScanSeg = utils::divUp(maxListLength, searchListSize);
    for (int i = 0; i < searchBatchSizes[0]; i++) {
        topkQueue[i].resize(maxScanSeg * nprobe);
    }
#endif
}

APP_ERROR IndexIVFSPSQ::addVectors(int listId, size_t numVecs, const uint8_t *codes,
    const idx_t *indices, const float *preCompute, bool useNPU)
{
    APP_LOG_INFO("IndexIVFSPSQ::addVectors start\n");
    APPERR_RETURN_IF_NOT_LOG(this->isTrained, APP_ERR_ILLEGAL_OPERATRION, "the index is not trained");
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

    if (addFinishFlag[listId]) {
        addDynamicUpdate(listId);
        deviceAllData[listId]->clear();
        addFinishFlag[listId] = 0;
    }

    // 1. save codes data
    AscendTensor<uint8_t, DIMS_2> codesData(const_cast<uint8_t *>(codes), { static_cast<int>(numVecs), this->dim2 });
    size_t originLen = getListLength(listId);
    size_t tmpLen = utils::roundUp((originLen + numVecs), static_cast<size_t>(CUBE_ALIGN));
    deviceListData[listId]->resize(tmpLen * this->dim2);

    if (useNPU) { // use NPU (aiCPU) to add vectors for larger ntotal; controlled by caller
        APPERR_RETURN_IF(addVectorsAiCpu(listId, codesData) != APP_ERR_OK, APP_ERR_INNER_ERROR);
    } else {
        APPERR_RETURN_IF(addVectorsCtrlCpu(listId, codesData) != APP_ERR_OK, APP_ERR_INNER_ERROR);
    }

    APP_LOG_INFO("IndexIVFSPSQ::addVectors preCompute\n");
    // 2. save pre compute data if not null

    if (preCompute != nullptr) {
        preComputeData[listId]->resize(tmpLen, true);
        float *precompData = preComputeData[listId]->data() + originLen;
        auto err = aclrtMemcpy(precompData, numVecs * sizeof(float),
                               preCompute, numVecs * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE);
        APPERR_RETURN_IF_NOT_FMT(err == ACL_SUCCESS, APP_ERR_INNER_ERROR, "Memcpy error %d", (int)err);
    }
    deviceListIndices[listId]->append(indices, numVecs, true);

#pragma omp critical
    {
        maxListLength = std::max(maxListLength, static_cast<int>(getListLength(listId)));
        maxListLength = utils::roundUp(maxListLength, CUBE_ALIGN);
        this->ntotal += numVecs;
    }

    return APP_ERR_OK;
}

APP_ERROR IndexIVFSPSQ::addVectorsCtrlCpu(int listId, AscendTensor<uint8_t, DIMS_2> &codesData)
{
    size_t numVecs = static_cast<size_t>(codesData.getSize(0));
    size_t originLen = getListLength(listId);

    // dims is alignd with CUBE_ALIGN, no padding data in horizontal direction
    int dimShaped = utils::divUp(this->dim2, CUBE_ALIGN);
    auto memErr = ACL_SUCCESS;

    // input codes are contigous(numVecs X dims), reconstruct the codes into Zz format.
    for (size_t i = 0; i < numVecs; i++) {
        int seq = static_cast<int>(originLen + i);
        uint8_t *tmpData = static_cast<uint8_t *>(deviceListData[listId]->data()) + getShapedDataOffset(seq);

        for (int j = 0; j < dimShaped; j++) {
            auto err = aclrtMemcpy(tmpData, CUBE_ALIGN * sizeof(uint8_t),
                codesData[i][j * CUBE_ALIGN].data(),
                CUBE_ALIGN * sizeof(uint8_t), ACL_MEMCPY_HOST_TO_DEVICE);
            ASCEND_EXC_IF_NOT_FMT(err == ACL_SUCCESS, memErr = err,
                                  "memcpy codesData err, i=%d, j=%d, err=%d", i, j, err);

            tmpData += (CUBE_ALIGN * CUBE_ALIGN);
        }
    }

    APPERR_RETURN_IF_NOT_LOG(memErr == ACL_SUCCESS, APP_ERR_INNER_ERROR, "Memcpy error");
    return APP_ERR_OK;
}

APP_ERROR IndexIVFSPSQ::addVectorsAiCpu(int listId, AscendTensor<uint8_t, DIMS_2> &codesData)
{
    std::string opName = "TransdataShapedSp";
    int n = codesData.getSize(0);
    auto &mem = resources.getMemoryManager();
    auto stream = resources.getDefaultStream();
    AscendTensor<int8_t, DIMS_2> src(mem, {n, this->dim2}, stream);
    auto ret = aclrtMemcpy(src.data(), src.getSizeInBytes(),
                           codesData.data(), codesData.getSizeInBytes(), ACL_MEMCPY_HOST_TO_DEVICE);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "Mem operator error %d", ret);

    size_t originLen = getListLength(listId);
    AscendTensor<int64_t, DIMS_1> attr(mem, {aicpu::TRANSDATA_SHAPED_SP_ATTR_IDX_COUNT}, stream);
    attr[0] = originLen;

    int total = static_cast<int>(originLen) + n;
    AscendTensor<int8_t, DIMS_4> dst(reinterpret_cast<int8_t *>(deviceListData[listId]->data()),
        {utils::divUp(total, CUBE_ALIGN),
        utils::divUp(this->dim2, CUBE_ALIGN), CUBE_ALIGN, CUBE_ALIGN});

    LaunchOpTwoInOneOut<int8_t, DIMS_2, ACL_INT8, int64_t, DIMS_1, ACL_INT64,
        int8_t, DIMS_4, ACL_INT8>(opName, stream, src, attr, dst);

    ret = aclrtSynchronizeStream(stream);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR,
                             "aclrtSynchronizeStream addVectors stream failed: %i\n", ret);

    return APP_ERR_OK;
}

APP_ERROR IndexIVFSPSQ::searchImpl(int n, const float16_t *x, int k, float16_t *distances, idx_t *labels)
{
    AscendTensor<float16_t, DIMS_2> queries(const_cast<float16_t *>(x), { n, dims });
    AscendTensor<float16_t, DIMS_2> outDistances(distances, { n, k });
    AscendTensor<idx_t, DIMS_2> outIndices(labels, { n, k });
    outIndices.initValue(std::numeric_limits<idx_t>::max());

    return searchImpl(queries, k, outDistances, outIndices);
}

void IndexIVFSPSQ::updateTrainedValue(AscendTensor<float16_t, DIMS_1> &trainedMin,
    AscendTensor<float16_t, DIMS_1> &trainedDiff)
{
    int dimMin = trainedMin.getSize(0);
    int dimDiff = trainedDiff.getSize(0);
    ASCEND_THROW_IF_NOT_FMT(dimMin == dimDiff && dimMin == this->dim2,
                            "sq trained data's shape invalid.(%d, %d) vs (dim:%d)", dimMin, dimDiff, this->dims);

    AscendTensor<float16_t, DIMS_1> minTensor({ dimMin });
    AscendTensor<float16_t, DIMS_1> diffTensor({ dimDiff });
    minTensor.copyFromSync(trainedMin);
    diffTensor.copyFromSync(trainedDiff);
    vMin = std::move(minTensor);
    vDiff = std::move(diffTensor);

    AscendTensor<float16_t, DIMS_2> dmTensor({ 2, this->dim2 });
    auto ret = aclrtMemcpy(dmTensor[0].data(), this->dim2 * sizeof(float16_t),
        vDiff.data(), this->dim2 * sizeof(float16_t), ACL_MEMCPY_DEVICE_TO_DEVICE);
    ASCEND_THROW_IF_NOT_FMT(ret == ACL_SUCCESS, "Mem operator error %d", static_cast<int>(ret));
    ret = aclrtMemcpy(dmTensor[1].data(), this->dim2 * sizeof(float16_t),
        vMin.data(), this->dim2 * sizeof(float16_t), ACL_MEMCPY_DEVICE_TO_DEVICE);
    ASCEND_THROW_IF_NOT_FMT(ret == ACL_SUCCESS, "Mem operator error %d", static_cast<int>(ret));
    vDM = std::move(dmTensor);

    // isTrained need to be set when all trained values are updated.
    // if coarseCentroidsIVFSPSQ has been updated, set isTrained = true
    if (this->coarseCentroidsIVFSPSQ == nullptr) {
        // 如果coarseCentriods是nullptr，唯一的可能性是其在共享码本时传入的索引实例时一个Load进来的码本实例，
        // 此时其他需要的码本数据 (coarseCentroidsShaped和NormTensor)已经被填充好，可以继续
        this->isTrained = true;
    } else if (this->coarseCentroidsIVFSPSQ->data()) {
        // 若不为nullptr，一定时由正常码本添加而来，同样可以继续
        this->isTrained = true;
    }
}

APP_ERROR IndexIVFSPSQ::searchBatched(int n, const float16_t *x, int k, float16_t *distance, idx_t *labels,
    uint64_t filterSize, uint32_t* filters)
{
    size_t size = searchBatchSizes.size();
    if (n > 1 && size > 0) {
        int64_t searched = 0;
        for (size_t i = 0; i < size; i++) {
            int batchSize = searchBatchSizes[i];
            if ((n - searched) >= batchSize) {
                int page = (n - searched) / batchSize;
                for (int j = 0; j < page; j++) {
                    APP_ERROR ret = searchFilterImpl(batchSize,
                    x + searched * this->dims, k, distance + searched * k,
                    labels + searched * k, filterSize/n*batchSize, filters+searched*filterSize/n);
                    APPERR_RETURN_IF(ret, ret);

                    searched += batchSize;
                }
            }
        }

        for (int64_t i = searched; i < n; i++) {
            APP_ERROR ret = searchFilterImpl(1, x + i * this->dims, k, distance + i * k, labels + i * k,
                                             filterSize/n, filters+filterSize/n*i);
            APPERR_RETURN_IF(ret, ret);
        }
        return APP_ERR_OK;
    } else {
        return searchFilterImpl(n, x, k, distance, labels, filterSize, filters);
    }

    return APP_ERR_OK;
}

APP_ERROR IndexIVFSPSQ::searchFilterImpl(int n, const float16_t *x, int k, float16_t *distances,
    idx_t *labels, uint32_t filterSize, uint32_t* filters)
{
    VALUE_UNUSED(n);
    VALUE_UNUSED(k);
    VALUE_UNUSED(filterSize);
    APPERR_RETURN_IF_NOT_LOG(x, APP_ERR_INVALID_PARAM, "x can not be nullptr.");
    APPERR_RETURN_IF_NOT_LOG(distances, APP_ERR_INVALID_PARAM, "distances can not be nullptr.");
    APPERR_RETURN_IF_NOT_LOG(labels, APP_ERR_INVALID_PARAM, "labels can not be nullptr.");
    APPERR_RETURN_IF_NOT_LOG(filters, APP_ERR_INVALID_PARAM, "filters can not be nullptr.");
    APP_LOG_INFO("IndexIVFSPSQ searchFilterImpl\n");
    return APP_ERR_OK;
}

APP_ERROR IndexIVFSPSQ::searchFilterImpl(std::vector<Index *> indexes, int n, int batchSize, const float16_t *x, int k,
    float16_t *distances, idx_t *labels, uint32_t filterSize, uint32_t* filters)
{
    VALUE_UNUSED(n);
    VALUE_UNUSED(batchSize);
    VALUE_UNUSED(k);
    VALUE_UNUSED(filterSize);
    APPERR_RETURN_IF_NOT_LOG(indexes.data(), APP_ERR_INVALID_PARAM, "indexes can not be nullptr.");
    APPERR_RETURN_IF_NOT_LOG(x, APP_ERR_INVALID_PARAM, "x can not be nullptr.");
    APPERR_RETURN_IF_NOT_LOG(distances, APP_ERR_INVALID_PARAM, "distances can not be nullptr.");
    APPERR_RETURN_IF_NOT_LOG(labels, APP_ERR_INVALID_PARAM, "labels can not be nullptr.");
    APPERR_RETURN_IF_NOT_LOG(filters, APP_ERR_INVALID_PARAM, "filters can not be nullptr.");
    APP_LOG_INFO("IndexIVFSPSQ searchFilterImpl\n");
    return APP_ERR_OK;
}

APP_ERROR IndexIVFSPSQ::searchFilterImpl(std::vector<Index *> indexes, int n, int batchSize, const float16_t *x, int k,
    float16_t *distances, idx_t *labels, uint32_t filterSize, uint32_t** filters)
{
    VALUE_UNUSED(n);
    VALUE_UNUSED(batchSize);
    VALUE_UNUSED(k);
    VALUE_UNUSED(filterSize);
    APPERR_RETURN_IF_NOT_LOG(indexes.data(), APP_ERR_INVALID_PARAM, "indexes can not be nullptr.");
    APPERR_RETURN_IF_NOT_LOG(x, APP_ERR_INVALID_PARAM, "x can not be nullptr.");
    APPERR_RETURN_IF_NOT_LOG(distances, APP_ERR_INVALID_PARAM, "distances can not be nullptr.");
    APPERR_RETURN_IF_NOT_LOG(labels, APP_ERR_INVALID_PARAM, "labels can not be nullptr.");
    APPERR_RETURN_IF_NOT_LOG(filters, APP_ERR_INVALID_PARAM, "filters can not be nullptr.");
    APP_LOG_INFO("IndexIVFSPSQ searchFilterImpl\n");
    return APP_ERR_OK;
}

APP_ERROR IndexIVFSPSQ::searchBatched(std::vector<Index*> indexes, int n, const float16_t *x, int k,
    float16_t *distances, idx_t *labels, uint32_t filterSize, uint32_t* filters)
{
    APP_ERROR ret = APP_ERR_OK;

    // check param
    size_t size = searchBatchSizes.size();
    if (size == 0 || n <= 0) {
        return APP_ERR_INVALID_PARAM;
    }

    // init result
    AscendTensor<float16_t, DIMS_3> outDistances(distances, { (int)indexes.size(), n, k });
    AscendTensor<idx_t, DIMS_3> outIndices(labels, { (int)indexes.size(), n, k });
    outDistances.initValue(Limits<float16_t>::getMin());
    outIndices.initValue(std::numeric_limits<idx_t>::max());

    int searched = 0;
    for (size_t i = 0; i < size; i++) {
        int batchSize = searchBatchSizes[i];
        if ((n - searched) >= batchSize) {
            int page = (n - searched) / batchSize;
            for (int j = 0; j < page; j++) {
                ret = searchFilterImpl(indexes, n, batchSize, x + searched * this->dims, k,
                    distances + searched * k, labels + searched * k,
                    filterSize/n*batchSize, filters + searched * filterSize / n);
                APPERR_RETURN_IF(ret, ret);
                searched += batchSize;
            }
        }
    }

    for (int i = searched; i < n; i++) {
        ret = searchFilterImpl(indexes, n, 1, x + i * this->dims, k,
            distances + i * k, labels + i * k, filterSize/n, filters+i*filterSize/n);
        APPERR_RETURN_IF(ret, ret);
    }

    reorder(indexes.size(), n, k, distances, labels);
    return APP_ERR_OK;
}

APP_ERROR IndexIVFSPSQ::searchBatched(std::vector<Index*> indexes, int n, const float16_t *x, int k,
    float16_t *distances, idx_t *labels, uint32_t filterSize, uint32_t** filters)
{
    APP_ERROR ret = APP_ERR_OK;

    // check param
    size_t size = searchBatchSizes.size();
    if (size == 0 || n <= 0) {
        return APP_ERR_INVALID_PARAM;
    }

    // init result
    AscendTensor<float16_t, DIMS_3> outDistances(distances, { (int)indexes.size(), n, k });
    AscendTensor<idx_t, DIMS_3> outIndices(labels, { (int)indexes.size(), n, k });
    outDistances.initValue(Limits<float16_t>::getMin());
    outIndices.initValue(std::numeric_limits<idx_t>::max());

    int searched = 0;
    for (size_t i = 0; i < size; i++) {
        int batchSize = searchBatchSizes[i];
        if ((n - searched) >= batchSize) {
            int page = (n - searched) / batchSize;
            for (int j = 0; j < page; j++) {
                ret = searchFilterImpl(indexes, n, batchSize, x + searched * this->dims, k,
                    distances + searched * k, labels + searched * k,
                    filterSize / n * batchSize, filters + searched);
                APPERR_RETURN_IF(ret, ret);
                searched += batchSize;
            }
        }
    }

    for (int i = searched; i < n; i++) {
        ret = searchFilterImpl(indexes, n, 1, x + i * this->dims, k,
            distances + i * k, labels + i * k, filterSize/n, filters+i);
        APPERR_RETURN_IF(ret, ret);
    }

    reorder(indexes.size(), n, k, distances, labels);
    return APP_ERR_OK;
}


APP_ERROR IndexIVFSPSQ::computeMask(int n, uint32_t* filters, AscendTensor<uint8_t, DIMS_1>& masks,
                                    AscendTensor<int, DIMS_1> &listId)
{
    VALUE_UNUSED(n);
    VALUE_UNUSED(filters);
    VALUE_UNUSED(masks);
    VALUE_UNUSED(listId);
    return APP_ERR_OK;
}

void IndexIVFSPSQ::moveVectorForward(idx_t srcIdx, idx_t dstIdx)
{
    ASCEND_THROW_IF_NOT(srcIdx >= dstIdx);
    int srcIdx1 = srcIdx / this->blockSize;
    int srcIdx2 = srcIdx % this->blockSize;

    int dstIdx1 = dstIdx / this->blockSize;
    int dstIdx2 = dstIdx % this->blockSize;

    int dim2 = utils::divUp(dims, CUBE_ALIGN);

    uint8_t *srcDataPtr = baseShaped[srcIdx1]->data() + (srcIdx2 / CUBE_ALIGN) * (dim2 * CUBE_ALIGN * CUBE_ALIGN) +
        (srcIdx2 % CUBE_ALIGN) * (CUBE_ALIGN);
    uint8_t *dstDataPtr = baseShaped[dstIdx1]->data() + (dstIdx2 / CUBE_ALIGN) * (dim2 * CUBE_ALIGN * CUBE_ALIGN) +
        (dstIdx2 % CUBE_ALIGN) * (CUBE_ALIGN);

    for (int i = 0; i < dim2; i++) {
        auto ret = aclrtMemcpy(dstDataPtr, CUBE_ALIGN * sizeof(uint8_t),
            srcDataPtr, CUBE_ALIGN * sizeof(uint8_t), ACL_MEMCPY_DEVICE_TO_DEVICE);
        ASCEND_THROW_IF_NOT_FMT(ret == ACL_SUCCESS, "Mem operator error %d",
            static_cast<int>(ret));
        dstDataPtr += CUBE_ALIGN * CUBE_ALIGN;
        srcDataPtr += CUBE_ALIGN * CUBE_ALIGN;
    }
}

void IndexIVFSPSQ::releaseUnusageSpace(int oldTotal, int remove)
{
    int oldVecSize = utils::divUp(oldTotal, this->blockSize);
    int vecSize = utils::divUp(oldTotal - remove, this->blockSize);

    for (int i = oldVecSize - 1; i >= vecSize; --i) {
        this->baseShaped.at(i)->clear();
    }
}

APP_ERROR IndexIVFSPSQ::reset()
{
    int dvSize = utils::divUp(this->ntotal, this->blockSize);
    for (int i = 0; i < dvSize; ++i) {
        baseShaped.at(i)->clear();
    }
    ntotal = 0;
    codebookFinished = false;
    return APP_ERR_OK;
}

void IndexIVFSPSQ::runFpToFp16(AscendTensor<float, DIMS_2> &queryVecs,
    AscendTensor<float16_t, DIMS_2> &outQueryVecs,
    AscendTensor<uint16_t, DIMS_2> &flag, aclrtStream stream)
{
    AscendOperator *op = nullptr;
    int batch = queryVecs.getSize(0);
    if (fpToFp16Ops.find(batch) != fpToFp16Ops.end()) {
        op = fpToFp16Ops[batch].get();
    }
    ASCEND_THROW_IF_NOT(op);

    std::shared_ptr<std::vector<const aclDataBuffer *>> distOpInput(
        new std::vector<const aclDataBuffer *>(), CommonUtils::AclInputBufferDelete);
    distOpInput->emplace_back(aclCreateDataBuffer(queryVecs.data(), queryVecs.getSizeInBytes()));

    std::shared_ptr<std::vector<aclDataBuffer *>> distOpOutput(
         new std::vector<aclDataBuffer *>(), CommonUtils::AclOutputBufferDelete);
    distOpOutput->emplace_back(aclCreateDataBuffer(outQueryVecs.data(), outQueryVecs.getSizeInBytes()));
    distOpOutput->emplace_back(aclCreateDataBuffer(flag.data(), flag.getSizeInBytes()));

    op->exec(*distOpInput, *distOpOutput, stream);
}


APP_ERROR IndexIVFSPSQ::resetFpToFp16Op()
{
    auto distCompOpReset = [&](std::unique_ptr<AscendOperator> &op, int64_t batch) {
        AscendOpDesc desc("FpToFp16");
        std::vector<int64_t> queryShape({ batch, dims });
        std::vector<int64_t> queryFp16Shape({ batch, dims });
        std::vector<int64_t> flagShape({ SP_CORE_NUM, FLAG_SIZE });

        desc.addInputTensorDesc(ACL_FLOAT, queryShape.size(), queryShape.data(), ACL_FORMAT_ND);
        desc.addOutputTensorDesc(ACL_FLOAT16, queryFp16Shape.size(), queryFp16Shape.data(), ACL_FORMAT_ND);
        desc.addOutputTensorDesc(ACL_UINT16, flagShape.size(), flagShape.data(), ACL_FORMAT_ND);

        op.reset();
        op = CREATE_UNIQUE_PTR(AscendOperator, desc);
        return op->init();
    };

    for (auto batch : addBatchSizes) {
        fpToFp16Ops[batch] = std::unique_ptr<AscendOperator>(nullptr);
        APPERR_RETURN_IF_NOT_LOG(distCompOpReset(fpToFp16Ops[batch], batch),
                                 APP_ERR_ACL_OP_LOAD_MODEL_FAILED, "op init failed");
    }

    for (auto batch : searchBatchSizes) {
        fpToFp16Ops[batch] = std::unique_ptr<AscendOperator>(nullptr);
        APPERR_RETURN_IF_NOT_LOG(distCompOpReset(fpToFp16Ops[batch], batch),
                                 APP_ERR_ACL_OP_LOAD_MODEL_FAILED, "op init failed");
    }

    return APP_ERR_OK;
}


void IndexIVFSPSQ::runCidFilterOperator(AscendTensor<uint8_t, DIMS_1>& idx,
    AscendTensor<int32_t, DIMS_1>& val,
    AscendTensor<int32_t, DIMS_1>& ts,
    AscendTensor<uint64_t, DIMS_2>& offset,
    AscendTensor<uint32_t, DIMS_1>& bucketSizes,
    AscendTensor<int32_t, DIMS_2>& maskFilter,
    AscendTensor<int32_t, DIMS_1>& timeFilter,
    AscendTensor<uint16_t, DIMS_2>& andData,
    AscendTensor<float16_t, DIMS_2>& mulData,
    AscendTensor<uint16_t, DIMS_2>& result,
    AscendTensor<uint16_t, DIMS_2>& flag,
    aclrtStream stream)
{
    std::shared_ptr<std::vector<const aclDataBuffer *>> opInput(
        new std::vector<const aclDataBuffer *>, CommonUtils::AclInputBufferDelete);
    opInput->emplace_back(aclCreateDataBuffer(idx.data(), idx.getSizeInBytes()));
    opInput->emplace_back(aclCreateDataBuffer(val.data(), val.getSizeInBytes()));
    opInput->emplace_back(aclCreateDataBuffer(ts.data(), ts.getSizeInBytes()));
    opInput->emplace_back(aclCreateDataBuffer(offset.data(), offset.getSizeInBytes()));
    opInput->emplace_back(aclCreateDataBuffer(bucketSizes.data(), bucketSizes.getSizeInBytes()));
    opInput->emplace_back(aclCreateDataBuffer(maskFilter.data(), maskFilter.getSizeInBytes()));
    opInput->emplace_back(aclCreateDataBuffer(timeFilter.data(), timeFilter.getSizeInBytes()));
    opInput->emplace_back(aclCreateDataBuffer(andData.data(), andData.getSizeInBytes()));
    opInput->emplace_back(aclCreateDataBuffer(mulData.data(), mulData.getSizeInBytes()));

    std::shared_ptr<std::vector<aclDataBuffer *>> opOutput(
         new std::vector<aclDataBuffer *>, CommonUtils::AclOutputBufferDelete);
    opOutput->emplace_back(aclCreateDataBuffer(result.data(), result.getSizeInBytes()));
    opOutput->emplace_back(aclCreateDataBuffer(flag.data(), flag.getSizeInBytes()));

    cidFilterOp->exec(*opInput, *opOutput, stream);
}

APP_ERROR IndexIVFSPSQ::resetCidFilterOperator()
{
    AscendOpDesc desc("IvfCidFilter3");
    APP_LOG_INFO("resetCidFilterOperator handleBatch=%d\n", handleBatch);
    std::vector<int64_t> idxShape({ searchListSize });
    std::vector<int64_t> valShape({ searchListSize });
    std::vector<int64_t> tsShape({ searchListSize });
    std::vector<int64_t> offsetShape({ 3, handleBatch });
    std::vector<int64_t> bucketSizesShape({ handleBatch });
    std::vector<int64_t> maskFilterShape({ ID_BLOCKS, MASK_SIZE });
    std::vector<int64_t> tsFilterShape({ TS_SIZE });
    std::vector<int64_t> andDataShape({ ID_BLOCKS, HELPER_SIZE });
    std::vector<int64_t> mulDataShape({ ID_BLOCKS, HELPER_SIZE });
    std::vector<int64_t> resultShape({ handleBatch, searchListSize / BIT_OF_UINT16 });
    std::vector<int64_t> flagShape({ CORE_NUM, FLAG_SIZE });

    desc.addInputTensorDesc(ACL_UINT8, idxShape.size(), idxShape.data(), ACL_FORMAT_ND);
    desc.addInputTensorDesc(ACL_INT32, valShape.size(), valShape.data(), ACL_FORMAT_ND);
    desc.addInputTensorDesc(ACL_INT32, tsShape.size(), tsShape.data(), ACL_FORMAT_ND);
    desc.addInputTensorDesc(ACL_UINT64, offsetShape.size(), offsetShape.data(), ACL_FORMAT_ND);
    desc.addInputTensorDesc(ACL_UINT32, bucketSizesShape.size(), bucketSizesShape.data(), ACL_FORMAT_ND);
    desc.addInputTensorDesc(ACL_INT32, maskFilterShape.size(), maskFilterShape.data(), ACL_FORMAT_ND);
    desc.addInputTensorDesc(ACL_INT32, tsFilterShape.size(), tsFilterShape.data(), ACL_FORMAT_ND);
    desc.addInputTensorDesc(ACL_UINT16, andDataShape.size(), andDataShape.data(), ACL_FORMAT_ND);
    desc.addInputTensorDesc(ACL_FLOAT16, mulDataShape.size(), mulDataShape.data(), ACL_FORMAT_ND);

    desc.addOutputTensorDesc(ACL_UINT16, resultShape.size(), resultShape.data(), ACL_FORMAT_ND);
    desc.addOutputTensorDesc(ACL_UINT16, flagShape.size(), flagShape.data(), ACL_FORMAT_ND);

    cidFilterOp.reset();
    cidFilterOp = CREATE_UNIQUE_PTR(AscendOperator, desc);
    APPERR_RETURN_IF_NOT_LOG(cidFilterOp->init(), APP_ERR_ACL_OP_LOAD_MODEL_FAILED, "op init failed");

    return APP_ERR_OK;
}
/**
 * @brief 对输入文件路径进行校验
 *
 * @param dataFile 文件路径
 */
void LoadCheck(const char *dataFile)
{
    std::ifstream allDataFin(dataFile, std::ios::binary);
    struct stat st;

    ASCEND_THROW_IF_NOT_FMT(lstat(dataFile, &st) == 0, "ERROR: %s", strerror(errno));
    ASCEND_THROW_IF_NOT_FMT(allDataFin.is_open(),
        "ERROR: Data file input[%s] can not be read!", dataFile);
    ASCEND_THROW_IF_NOT_FMT((st.st_mode & S_IFMT) == S_IFREG,
        "ERROR: Data file input[%s] is not a Regular File!", dataFile);
    ASCEND_THROW_IF_NOT_FMT(st.st_size < MAX_DATAFILE_SIZE,
        "ERROR: Data file input[%s] is oversize(< 56 GB required)", dataFile);

    allDataFin.seekg(0, std::ios_base::end);
    size_t dataLen = allDataFin.tellg();
    // 16 is the size of parameter used to check codebook
    ASCEND_THROW_IF_NOT_MSG(dataLen >= 16, "index size is not correct!");
    allDataFin.close();
}

APP_ERROR IndexIVFSPSQ::loadDeviceAllData(const char *dataFile, float* codebookPtr, float* spsqPtr,
    const IndexIVFSPSQ* loadedIndex)
{
    size_t bucketLen = 0;
    if (filterable) {
        bucketLen = this->dim2 + sizeof(idx_t) + sizeof(float) + sizeof(uint32_t) + sizeof(uint32_t) + sizeof(uint8_t);
    } else {
        bucketLen = this->dim2 + sizeof(idx_t) + sizeof(float);
    }

    // 对读取文件内容进行校验
    LoadCheck(dataFile);

    bucketSize.resize(numLists);
    isEmptyList.resize(numLists);
    if (filterable) {
        idxOffset.resize(numLists);
        valOffset.resize(numLists);
        tsOffset.resize(numLists);
    }

    std::string dataFileString(dataFile);
    FSPIOReader indexReader(dataFileString);

    // add check of the index
    char fourcc[MAGIC_NUMBER_LEN] = {'I', 'W', 'S', 'P'};
    char foureccBuffer[MAGIC_NUMBER_LEN];
    // check 4 IWSP
    indexReader.ReadAndCheck(foureccBuffer, MAGIC_NUMBER_LEN); // 4 checking flags in total
    for (int i = 0; i < MAGIC_NUMBER_LEN; i++) { // 4 checking flags in total
        if (foureccBuffer[i] != fourcc[i]) {
            ASCEND_THROW_MSG("index format is not correct.");
        }
    }

    int dim_ = 0;
    int dim2_ = 0;
    int ncentroids_ = 0;
    bool filterable_ = false;
    int handleBatch_ = 0;
    int nprobe_ = 0;
    int searchListSize_ = 0;
    indexReader.ReadAndCheck(&dim_, sizeof(int));
    indexReader.ReadAndCheck(&dim2_, sizeof(int));
    indexReader.ReadAndCheck(&ncentroids_, sizeof(int));
    indexReader.ReadAndCheck(&filterable_, sizeof(bool));
    indexReader.ReadAndCheck(&handleBatch_, sizeof(int));
    indexReader.ReadAndCheck(&nprobe_, sizeof(int));
    indexReader.ReadAndCheck(&searchListSize_, sizeof(int));
    if ((dim_ != dims) || (dim2_ != dim2) || (ncentroids_ != numLists)
        || (filterable_ != filterable) || (handleBatch_ != handleBatch)
        || (nprobe_ != nprobe) || (searchListSize_ != searchListSize)) {
        ASCEND_THROW_MSG("index shape is not correct."); // raise error
    }

    // begin of real data
    for (size_t listId = 0; listId < (size_t)numLists; listId++) {
        int idVal = -1;
        int listLenVal = 0;
        indexReader.ReadAndCheck(&idVal, sizeof(int));
        indexReader.ReadAndCheck(&listLenVal, sizeof(int));
        size_t listLen = (size_t) listLenVal;
        if ((idVal < 0) || (idVal >= numLists)) {
            ASCEND_THROW_MSG("index file is not correct!");
        }
        if (listLen == 0) {
            isEmptyList[listId] = true;
            listLen = static_cast<size_t>(CUBE_ALIGN);
        } else if (listLen > ONE_BUCKET_MAX_VECTOR_NUM) {
            ASCEND_THROW_MSG("index file is not correct, bucket is too huge!");
        } else {
            isEmptyList[listId] = false;
        }

        size_t tmpLen = utils::roundUp(listLen, static_cast<size_t>(CUBE_ALIGN));
        size_t tmpLenAlignFilter = utils::roundUp(listLen, static_cast<size_t>(FILTER_ALIGN));

        if (listId != (size_t) idVal) { // 跳过bucketLen * tmpLen，目的是往后挪动文件流
            std::vector<char> tmpForSkip(bucketLen * tmpLen);
            indexReader.ReadAndCheck(tmpForSkip.data(), bucketLen * tmpLen);
            continue;
        }

        if (filterable) {
            deviceAllData[listId]->resize(tmpLen * bucketLen +
                (tmpLenAlignFilter - tmpLen) * (sizeof(uint32_t) + sizeof(uint32_t) + sizeof(uint8_t)), true);
        } else {
            deviceAllData[listId]->resize(tmpLen * bucketLen, true);
        }

        std::vector<char> dataShrPtr(tmpLen * bucketLen);
        indexReader.ReadAndCheck(dataShrPtr.data(), tmpLen * bucketLen);

        uint8_t *dataPtr = reinterpret_cast<uint8_t *>(dataShrPtr.data());
        uint8_t *dstDataOffSet = static_cast<uint8_t *>(deviceAllData[listId]->data());

        uint8_t *tmpData = dataPtr;
        uint8_t *dstData = dstDataOffSet;
        auto ret = aclrtMemcpy(dstData, tmpLen * this->dim2,
            tmpData, tmpLen * this->dim2, ACL_MEMCPY_HOST_TO_DEVICE);
        APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "Mem operator error %d", ret);

        dstData = dstDataOffSet + tmpLen * this->dim2;
        tmpData = dataPtr + tmpLen * this->dim2;
        ret = aclrtMemcpy(dstData, tmpLen * sizeof(float),
            tmpData, tmpLen * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE);
        APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "Mem operator error %d", ret);

        dstData = dstDataOffSet + tmpLen * (this->dim2 + sizeof(float));
        tmpData = dataPtr + tmpLen * (this->dim2 + sizeof(float));
        ret = aclrtMemcpy(dstData, tmpLen * sizeof(idx_t),
            tmpData, tmpLen * sizeof(idx_t), ACL_MEMCPY_HOST_TO_DEVICE);
        APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "Mem operator error %d", ret);

        if (filterable) {
            dataPtr = dataPtr + tmpLen * (this->dim2 + sizeof(float) + sizeof(idx_t));
            dstDataOffSet = dstDataOffSet + tmpLen * (this->dim2 + sizeof(float) + sizeof(idx_t));

            dstData = dstDataOffSet;
            tmpData = dataPtr;
            ret = aclrtMemcpy(dstData, tmpLenAlignFilter * sizeof(uint8_t),
                tmpData, tmpLen * sizeof(uint8_t), ACL_MEMCPY_HOST_TO_DEVICE);
            APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "Mem operator error %d", ret);

            dstData = dstDataOffSet + tmpLenAlignFilter * sizeof(uint8_t);
            tmpData = dataPtr + tmpLen * sizeof(uint8_t);
            ret = aclrtMemcpy(dstData, tmpLenAlignFilter * sizeof(uint32_t),
                tmpData, tmpLen * sizeof(uint32_t), ACL_MEMCPY_HOST_TO_DEVICE);
            APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "Mem operator error %d", ret);

            dstData = dstDataOffSet + tmpLenAlignFilter * (sizeof(uint8_t) + sizeof(uint32_t));
            tmpData = dataPtr + tmpLen * (sizeof(uint8_t) + sizeof(uint32_t));
            ret = aclrtMemcpy(dstData, tmpLenAlignFilter * sizeof(uint32_t),
                tmpData, tmpLen * sizeof(uint32_t), ACL_MEMCPY_HOST_TO_DEVICE);
            APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "Mem operator error %d", ret);
        }

        bucketSize[listId] = listLen;
        this->ntotal += listLen;
        if (isEmptyList[listId]) {
            this->ntotal -= static_cast<size_t>(CUBE_ALIGN);
        }
        maxListLength = std::max(maxListLength, static_cast<int>(listLen));
        maxListLength = utils::roundUp(maxListLength, CUBE_ALIGN);
        addFinishFlag[listId] = 1;
    }

    pListBase = deviceAllData[0]->data();
    for (int i = 1; i < numLists; ++i) {
        pListBase = std::min(pListBase, deviceAllData[i]->data());
    }

    std::vector<float16_t> tmpDataPtr(numLists * dim2 * this->dims);
    float16_t *tmpData = tmpDataPtr.data();
    indexReader.ReadAndCheck(tmpData, (size_t)numLists * dim2 * this->dims * sizeof(float16_t));
    if (loadedIndex == nullptr) {
        std::initializer_list<int> tmpShapedCentroidsInit = {utils::divUp(dim2*numLists, CUBE_ALIGN),
                                                             utils::divUp(dims, CUBE_ALIGN), CUBE_ALIGN, CUBE_ALIGN };
        auto tmpShapedCentroids = std::make_shared<AscendTensor<float16_t, DIMS_4>>(tmpShapedCentroidsInit);
        auto ret = aclrtMemcpy(tmpShapedCentroids->data(), numLists * dim2 * this->dims * sizeof(float16_t),
            tmpData, numLists * dim2 * this->dims * sizeof(float16_t), ACL_MEMCPY_HOST_TO_DEVICE);
        APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "Mem operator error %d", ret);
        coarseCentroidsShaped = tmpShapedCentroids;
    }

    std::vector<float16_t> tmpDataPtr2(numLists * dim2);
    tmpData = tmpDataPtr2.data();
    indexReader.ReadAndCheck(tmpData, (size_t)numLists * dim2 * sizeof(float16_t));
    if (loadedIndex == nullptr) {
        std::initializer_list<int> tmpNormTensorInit = { numLists * dim2 };
        auto tmpNormTensor = std::make_shared<AscendTensor<float16_t, DIMS_1>>(tmpNormTensorInit);
        auto ret = aclrtMemcpy(tmpNormTensor->data(), numLists * dim2 * sizeof(float16_t),
            tmpData, numLists * dim2 * sizeof(float16_t), ACL_MEMCPY_HOST_TO_DEVICE);
        APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "Mem operator error %d", ret);
        normCoarseCentroids = tmpNormTensor;
    }
    
    if (loadedIndex != nullptr) {
        // 此处使当前索引共享传入的索引的码本
        this->updateCoarseCentroidsData(loadedIndex);
    }

    AscendTensor<float16_t, DIMS_1> minTensor({ this->dim2 });
    AscendTensor<float16_t, DIMS_1> diffTensor({ this->dim2 });
    std::vector<float16_t> tmpDataPtr3(this->dim2);
    indexReader.ReadAndCheck(tmpDataPtr3.data(), (size_t)this->dim2 * sizeof(float16_t));
    auto ret = aclrtMemcpy(minTensor.data(), this->dim2 * sizeof(float16_t),
        tmpDataPtr3.data(), this->dim2 * sizeof(float16_t), ACL_MEMCPY_HOST_TO_DEVICE);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "Mem operator error %d", ret);

    std::vector<float16_t> tmpDataPtr4(this->dim2);
    indexReader.ReadAndCheck(tmpDataPtr4.data(), (size_t)this->dim2 * sizeof(float16_t));
    ret = aclrtMemcpy(diffTensor.data(), this->dim2 * sizeof(float16_t),
                      tmpDataPtr4.data(), this->dim2 * sizeof(float16_t), ACL_MEMCPY_HOST_TO_DEVICE);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "Mem operator error %d", ret);
    updateTrainedValue(minTensor, diffTensor);

    std::vector<float> tmpDataPtr5(numLists * dim2 * this->dims);
    indexReader.ReadAndCheck(tmpDataPtr5.data(), (size_t)numLists * dim2 * this->dims * sizeof(float));
    std::vector<float> tmpDataPtr6(this->dim2 * 2); // 2 means size of vec
    indexReader.ReadAndCheck(tmpDataPtr6.data(), (size_t) this->dim2 * 2 * sizeof(float)); // 2 means size of vec

    if (loadedIndex == nullptr) { // 如果索引共享，则这些数据在Impl层已完成共享，因此此处仅处理非共享情况
        auto retNum = memcpy_s(codebookPtr, numLists * dim2 * this->dims * sizeof(float),
            tmpDataPtr5.data(), numLists * dim2 * this->dims * sizeof(float));
        APPERR_RETURN_IF_NOT_FMT(retNum == 0, APP_ERR_INNER_ERROR, "Mem operator error %d", retNum);
        retNum = memcpy_s(spsqPtr, this->dim2 * 2 * sizeof(float), // 2 means size of vec
            tmpDataPtr6.data(), this->dim2 * 2 * sizeof(float)); // 2 means size of vec
        APPERR_RETURN_IF_NOT_FMT(retNum == 0, APP_ERR_INNER_ERROR, "Mem operator error %d", retNum);
    }

    if (filterable) {
        for (int listId = 0; listId < numLists; ++listId) {
            size_t listLen = bucketSize[listId];
            size_t tmpLen = utils::roundUp(listLen, static_cast<size_t>(CUBE_ALIGN));
            size_t tmpLenAlignFilter = utils::roundUp(listLen, static_cast<size_t>(FILTER_ALIGN));
            uint8_t *cidIdxData = reinterpret_cast<uint8_t *>(deviceAllData[listId]->data()) +
                                  tmpLen * (this->dim2 + sizeof(float) + sizeof(idx_t));
            idxOffset[listId] = cidIdxData - reinterpret_cast<uint8_t *>(pListBase);

            int32_t *cidValData = reinterpret_cast<int32_t *>(
                reinterpret_cast<uint8_t *>(deviceAllData[listId]->data()) +
                tmpLen * (this->dim2 + sizeof(float) + sizeof(idx_t)) + tmpLenAlignFilter*sizeof(uint8_t));
            valOffset[listId] = cidValData - reinterpret_cast<int32_t *>(pListBase);

            int32_t *timestampsData = reinterpret_cast<int32_t *>(
                reinterpret_cast<uint8_t *>(deviceAllData[listId]->data()) +
                tmpLen * (this->dim2 + sizeof(float) + sizeof(idx_t))
                + tmpLenAlignFilter*(sizeof(uint8_t) + sizeof(uint32_t)));
            tsOffset[listId] = timestampsData - reinterpret_cast<int32_t *>(pListBase);
        }
    }

#ifdef HOSTCPU
#else
    topkQueue.resize(searchBatchSizes[0]);
    int maxScanSeg = utils::divUp(maxListLength, searchListSize);
    for (int i = 0; i < searchBatchSizes[0]; i++) {
        topkQueue[i].resize(maxScanSeg * nprobe);
    }
#endif
    isTrained = true;
    return APP_ERR_OK;
}

APP_ERROR IndexIVFSPSQ::saveDeviceAllData(const char *dataFile, float *codebookPtr, float *spsqPtr)
{
    // Before we save, if data has not been moved to deviceAllData, call allFinishMerge() to do so
    if (!isAddFinish()) {
        addFinishMerge();
    }

    size_t bucketLen = 0;
    if (filterable) {
        bucketLen =
            this->dim2 + sizeof(idx_t) + sizeof(float) + sizeof(uint32_t) + sizeof(uint32_t) + sizeof(uint8_t);
    } else {
        bucketLen = this->dim2 + sizeof(idx_t) + sizeof(float);
    }

    std::string dataFileString(dataFile);
    FSPIOWriter indexWriter(dataFileString);

    char fourcc[MAGIC_NUMBER_LEN] = {'I', 'W', 'S', 'P'};
    for (int i = 0; i < MAGIC_NUMBER_LEN; i++) {
        indexWriter.WriteAndCheck((&(fourcc[i])), sizeof(char));
    }
    indexWriter.WriteAndCheck((&(this->dims)), sizeof(int));
    indexWriter.WriteAndCheck((&(this->dim2)), sizeof(int));
    indexWriter.WriteAndCheck((&(numLists)), sizeof(int));
    indexWriter.WriteAndCheck(&(filterable), sizeof(filterable));
    indexWriter.WriteAndCheck(&(handleBatch), sizeof(handleBatch));
    indexWriter.WriteAndCheck(&(nprobe), sizeof(nprobe));
    indexWriter.WriteAndCheck(&(searchListSize), sizeof(searchListSize));

    for (int i = 0; i < numLists; i++) {
        indexWriter.WriteAndCheck((&i), sizeof(int));
        if (isEmptyList[i]) {
            int zeroLen = 0;
            indexWriter.WriteAndCheck((&zeroLen), sizeof(int));
        } else {
            int currBucketSize = static_cast<int>(bucketSize[i]);
            indexWriter.WriteAndCheck((&currBucketSize), sizeof(int));
        }
        size_t tmpLen = utils::roundUp(bucketSize[i], static_cast<size_t>(CUBE_ALIGN));
        size_t tmpLenAlignFilter = utils::roundUp(bucketSize[i], static_cast<size_t>(FILTER_ALIGN));
        std::vector <uint8_t> data(bucketLen * tmpLen);
        if (filterable) {
            auto ret = aclrtMemcpy(data.data(),
                (this->dim2 + sizeof(idx_t) + sizeof(float) + sizeof(uint8_t)) * tmpLen,
                deviceAllData[i]->data(),
                (this->dim2 + sizeof(idx_t) + sizeof(float) + sizeof(uint8_t)) * tmpLen,
                ACL_MEMCPY_DEVICE_TO_HOST);
            ASCEND_THROW_IF_NOT_FMT(ret == ACL_SUCCESS, "memcpy (error %d)", static_cast<int>(ret));
            ret = aclrtMemcpy(data.data() + (this->dim2 + sizeof(idx_t) + sizeof(float) + sizeof(uint8_t)) * tmpLen,
                tmpLen * sizeof(uint32_t),
                deviceAllData[i]->data() + (this->dim2 + sizeof(idx_t) +
                sizeof(float)) * tmpLen + tmpLenAlignFilter * sizeof(uint8_t),
                tmpLen * sizeof(uint32_t), ACL_MEMCPY_DEVICE_TO_HOST);
            ASCEND_THROW_IF_NOT_FMT(ret == ACL_SUCCESS, "memcpy (error %d)", static_cast<int>(ret));
            ret = aclrtMemcpy(data.data() + (this->dim2 + sizeof(idx_t) + sizeof(float) +
                sizeof(uint8_t) + sizeof(uint32_t)) * tmpLen,
                tmpLen * sizeof(uint32_t),
                deviceAllData[i]->data() + (this->dim2 + sizeof(idx_t) + sizeof(float)) *
                tmpLen + tmpLenAlignFilter * (sizeof(uint8_t) + sizeof(uint32_t)),
                tmpLen * sizeof(uint32_t),
                ACL_MEMCPY_DEVICE_TO_HOST);
            ASCEND_THROW_IF_NOT_FMT(ret == ACL_SUCCESS, "memcpy (error %d)", static_cast<int>(ret));
        } else {
            auto ret = aclrtMemcpy(data.data(), bucketLen * tmpLen,
                deviceAllData[i]->data(), bucketLen * tmpLen, ACL_MEMCPY_DEVICE_TO_HOST);
            ASCEND_THROW_IF_NOT_FMT(ret == ACL_SUCCESS, "memcpy (error %d)", static_cast<int>(ret));
        }

        indexWriter.WriteAndCheck(data.data(), bucketLen * tmpLen);
    }

    std::vector<float16_t> tmpShapedCentroids(numLists * dim2 * this->dims);
    auto ret = aclrtMemcpy(tmpShapedCentroids.data(), numLists * dim2 * this->dims * sizeof(float16_t),
        coarseCentroidsShaped->data(), numLists * dim2 * this->dims * sizeof(float16_t),
        ACL_MEMCPY_DEVICE_TO_HOST);
    ASCEND_THROW_IF_NOT_FMT(ret == ACL_SUCCESS, "memcpy (error %d)", static_cast<int>(ret));
    indexWriter.WriteAndCheck(tmpShapedCentroids.data(), numLists * dim2 * this->dims * sizeof(float16_t));

    std::vector<float16_t> tmpNormTensor(numLists * dim2);
    ret = aclrtMemcpy(tmpNormTensor.data(), numLists * dim2 * sizeof(float16_t),
        normCoarseCentroids->data(), numLists * dim2 * sizeof(float16_t), ACL_MEMCPY_DEVICE_TO_HOST);
    ASCEND_THROW_IF_NOT_FMT(ret == ACL_SUCCESS, "memcpy (error %d)", static_cast<int>(ret));
    indexWriter.WriteAndCheck(tmpNormTensor.data(), numLists * dim2 * sizeof(float16_t));
    
    std::vector<float16_t> minTensor(this->dim2);
    ret = aclrtMemcpy(minTensor.data(), this->dim2 * sizeof(float16_t),
        vMin.data(), this->dim2 * sizeof(float16_t), ACL_MEMCPY_DEVICE_TO_HOST);
    ASCEND_THROW_IF_NOT_FMT(ret == ACL_SUCCESS, "memcpy (error %d)", static_cast<int>(ret));
    indexWriter.WriteAndCheck(minTensor.data(), this->dim2 * sizeof(float16_t));

    std::vector<float16_t> diffTensor(this->dim2);
    ret = aclrtMemcpy(diffTensor.data(), this->dim2 * sizeof(float16_t),
        vDiff.data(), this->dim2 * sizeof(float16_t), ACL_MEMCPY_DEVICE_TO_HOST);
    ASCEND_THROW_IF_NOT_FMT(ret == ACL_SUCCESS, "memcpy (error %d)", static_cast<int>(ret));
    indexWriter.WriteAndCheck(diffTensor.data(), this->dim2 * sizeof(float16_t));

    indexWriter.WriteAndCheck(codebookPtr, this->dim2*this->dims*numLists * sizeof(float));
    indexWriter.WriteAndCheck(spsqPtr, this->dim2*(sizeof(float)+sizeof(float)));
    return APP_ERR_OK;
}

APP_ERROR IndexIVFSPSQ::saveDeviceAllData(uint8_t*& data, size_t& dataLen, float* codebookPtr, float* spsqPtr)
{
    if (!isAddFinish()) {
        addFinishMerge();
    }

    size_t dataLenTmp = 0; // 申请临时变量存储数据长度

    size_t bucketLen = 0;
    if (filterable) {
        bucketLen =
            this->dim2 + sizeof(idx_t) + sizeof(float) + sizeof(uint32_t) + sizeof(uint32_t) + sizeof(uint8_t);
    } else {
        bucketLen = this->dim2 + sizeof(idx_t) + sizeof(float);
    }

    char fourcc[MAGIC_NUMBER_LEN] = {'I', 'W', 'S', 'P'};
    for (int i = 0; i < numLists; i++) {
        size_t tmpLen = utils::roundUp(bucketSize[i], static_cast<size_t>(CUBE_ALIGN));
        dataLenTmp += 2 * sizeof(int); // save both i and bucketSize, thus 2
        dataLenTmp += (tmpLen * bucketLen) * sizeof(uint8_t);
    }
    dataLenTmp += sizeof(fourcc) + // 校验符
        6 * sizeof(int) + // 存储 dims, dim2, numLists, handleBatch, nprobe, searchListSize 6个参数
        sizeof(bool) + // filterable布尔值
        numLists * dim2 * this->dims * sizeof(float16_t) + // tmpShapedCentriods
        numLists * dim2 * sizeof(float16_t) + // tmpNormTensors
        this->dim2 * sizeof(float16_t) + // MinTensor
        this->dim2 * sizeof(float16_t) + // DiffTensor
        this->dim2 * this->dims * numLists * sizeof(float) + // codebookPtr
        this->dim2 * (sizeof(float) + sizeof(float)); // spsqPtr
    
    size_t dataOffset = 0; // 记录每次拷贝进入data指针需要的偏移量
    uint8_t* dataTmp = new uint8_t[dataLenTmp]; // 申请一个临时指针并用智能指针管理
    std::unique_ptr<uint8_t[]> del(dataTmp);

    CopyDataForSave(dataTmp, fourcc, sizeof(fourcc), dataOffset, dataLenTmp);
    CopyDataForSave(dataTmp, &(this->dims), sizeof(int), dataOffset, dataLenTmp);
    CopyDataForSave(dataTmp, &(this->dim2), sizeof(int), dataOffset, dataLenTmp);
    CopyDataForSave(dataTmp, &(numLists), sizeof(int), dataOffset, dataLenTmp);
    CopyDataForSave(dataTmp, &(filterable), sizeof(filterable), dataOffset, dataLenTmp);
    CopyDataForSave(dataTmp, &(handleBatch), sizeof(handleBatch), dataOffset, dataLenTmp);
    CopyDataForSave(dataTmp, &(nprobe), sizeof(nprobe), dataOffset, dataLenTmp);
    CopyDataForSave(dataTmp, &(searchListSize), sizeof(searchListSize), dataOffset, dataLenTmp);
    
    for (int i = 0; i < numLists; i++) {
        CopyDataForSave(dataTmp, &i, sizeof(int), dataOffset, dataLenTmp);
        if (isEmptyList[i]) {
            int zeroLen = 0;
            CopyDataForSave(dataTmp, &zeroLen, sizeof(int), dataOffset, dataLenTmp);
        } else {
            int currBucketSize = static_cast<int>(bucketSize[i]);
            CopyDataForSave(dataTmp, &currBucketSize, sizeof(int), dataOffset, dataLenTmp);
        }
        size_t tmpLen = utils::roundUp(bucketSize[i], static_cast<size_t>(CUBE_ALIGN));
        size_t tmpLenAlignFilter = utils::roundUp(bucketSize[i], static_cast<size_t>(FILTER_ALIGN));
        std::vector <uint8_t> tmpData(bucketLen * tmpLen);
        if (filterable) {
            auto ret = aclrtMemcpy(tmpData.data(),
                (this->dim2 + sizeof(idx_t) + sizeof(float) + sizeof(uint8_t)) * tmpLen,
                deviceAllData[i]->data(),
                (this->dim2 + sizeof(idx_t) + sizeof(float) + sizeof(uint8_t)) * tmpLen,
                ACL_MEMCPY_DEVICE_TO_HOST);
            ASCEND_THROW_IF_NOT_FMT(ret == ACL_SUCCESS, "memcpy (error %d)", static_cast<int>(ret));
            ret = aclrtMemcpy(tmpData.data() + (this->dim2 + sizeof(idx_t) + sizeof(float) + sizeof(uint8_t)) * tmpLen,
                tmpLen * sizeof(uint32_t),
                deviceAllData[i]->data() + (this->dim2 + sizeof(idx_t) +
                sizeof(float)) * tmpLen + tmpLenAlignFilter * sizeof(uint8_t),
                tmpLen * sizeof(uint32_t), ACL_MEMCPY_DEVICE_TO_HOST);
            ASCEND_THROW_IF_NOT_FMT(ret == ACL_SUCCESS, "memcpy (error %d)", static_cast<int>(ret));
            ret = aclrtMemcpy(tmpData.data() + (this->dim2 + sizeof(idx_t) + sizeof(float) +
                sizeof(uint8_t) + sizeof(uint32_t)) * tmpLen,
                tmpLen * sizeof(uint32_t),
                deviceAllData[i]->data() + (this->dim2 + sizeof(idx_t) + sizeof(float)) *
                tmpLen + tmpLenAlignFilter * (sizeof(uint8_t) + sizeof(uint32_t)),
                tmpLen * sizeof(uint32_t),
                ACL_MEMCPY_DEVICE_TO_HOST);
            ASCEND_THROW_IF_NOT_FMT(ret == ACL_SUCCESS, "memcpy (error %d)", static_cast<int>(ret));
        } else {
            auto ret = aclrtMemcpy(tmpData.data(), bucketLen * tmpLen,
                deviceAllData[i]->data(), bucketLen * tmpLen, ACL_MEMCPY_DEVICE_TO_HOST);
            ASCEND_THROW_IF_NOT_FMT(ret == ACL_SUCCESS, "memcpy (error %d)", static_cast<int>(ret));
        }
        CopyDataForSave(dataTmp, tmpData.data(), bucketLen * tmpLen * sizeof(uint8_t), dataOffset, dataLenTmp);
    }

    std::vector<float16_t> tmpShapedCentroids(numLists * dim2 * this->dims);
    auto ret = aclrtMemcpy(tmpShapedCentroids.data(), numLists * dim2 * this->dims * sizeof(float16_t),
        coarseCentroidsShaped->data(), numLists * dim2 * this->dims * sizeof(float16_t),
        ACL_MEMCPY_DEVICE_TO_HOST);
    ASCEND_THROW_IF_NOT_FMT(ret == ACL_SUCCESS, "memcpy (error %d)", static_cast<int>(ret));

    CopyDataForSave(dataTmp, tmpShapedCentroids.data(), numLists * dim2 * this->dims * sizeof(float16_t), dataOffset,
                    dataLenTmp);

    std::vector<float16_t> tmpNormTensor(numLists * dim2);
    ret = aclrtMemcpy(tmpNormTensor.data(), numLists * dim2 * sizeof(float16_t),
        normCoarseCentroids->data(), numLists * dim2 * sizeof(float16_t), ACL_MEMCPY_DEVICE_TO_HOST);
    ASCEND_THROW_IF_NOT_FMT(ret == ACL_SUCCESS, "memcpy (error %d)", static_cast<int>(ret));
        
    CopyDataForSave(dataTmp, tmpNormTensor.data(), numLists * dim2 * sizeof(float16_t), dataOffset, dataLenTmp);
    
    std::vector<float16_t> minTensor(this->dim2);
    ret = aclrtMemcpy(minTensor.data(), this->dim2 * sizeof(float16_t),
        vMin.data(), this->dim2 * sizeof(float16_t), ACL_MEMCPY_DEVICE_TO_HOST);
    ASCEND_THROW_IF_NOT_FMT(ret == ACL_SUCCESS, "memcpy (error %d)", static_cast<int>(ret));
    
    CopyDataForSave(dataTmp, minTensor.data(), this->dim2 * sizeof(float16_t), dataOffset, dataLenTmp);

    std::vector<float16_t> diffTensor(this->dim2);
    ret = aclrtMemcpy(diffTensor.data(), this->dim2 * sizeof(float16_t),
        vDiff.data(), this->dim2 * sizeof(float16_t), ACL_MEMCPY_DEVICE_TO_HOST);
    ASCEND_THROW_IF_NOT_FMT(ret == ACL_SUCCESS, "memcpy (error %d)", static_cast<int>(ret));

    CopyDataForSave(dataTmp, diffTensor.data(), this->dim2 * sizeof(float16_t), dataOffset, dataLenTmp);
    CopyDataForSave(dataTmp, codebookPtr, this->dim2 * this->dims * numLists * sizeof(float), dataOffset, dataLenTmp);
    CopyDataForSave(dataTmp, spsqPtr, this->dim2 * (sizeof(float) + sizeof(float)), dataOffset, dataLenTmp);

    del.release(); // del对被管理的临时指针放权，并将其赋值给用户输入data; 将dataLenTmp赋值给dataLen
    data = dataTmp;
    dataLen = dataLenTmp;
    return APP_ERR_OK;
}

APP_ERROR IndexIVFSPSQ::loadDeviceAllData(const uint8_t* data, size_t dataLen, float* codebookPtr, float* spsqPtr,
    const IndexIVFSPSQ* loadedIndex)
{
    size_t bucketLen = 0;
    if (filterable) {
        bucketLen = this->dim2 + sizeof(idx_t) + sizeof(float) + sizeof(uint32_t) + sizeof(uint32_t) + sizeof(uint8_t);
    } else {
        bucketLen = this->dim2 + sizeof(idx_t) + sizeof(float);
    }

    bucketSize.resize(numLists);
    isEmptyList.resize(numLists);
    if (filterable) {
        idxOffset.resize(numLists);
        valOffset.resize(numLists);
        tsOffset.resize(numLists);
    }
    
    // 计算每个参数的偏移量
    size_t offset = 0;

    char fourcc[MAGIC_NUMBER_LEN] = {'I', 'W', 'S', 'P'};
    char foureccBuffer[MAGIC_NUMBER_LEN];
    CopyDataForLoad(foureccBuffer, data, sizeof(foureccBuffer), offset, dataLen);
    for (int i = 0; i < MAGIC_NUMBER_LEN; i++) {
        if (foureccBuffer[i] != fourcc[i]) {
            ASCEND_THROW_MSG("index format is not correct."); // raise error
        }
    }

    int dim_ = 0;
    int dim2_ = 0;
    int ncentroids_ = 0;
    bool filterable_ = false;
    int handleBatch_ = 0;
    int nprobe_ = 0;
    int searchListSize_ = 0;
    CopyDataForLoad(&dim_, data, sizeof(int), offset, dataLen);
    CopyDataForLoad(&dim2_, data, sizeof(int), offset, dataLen);
    CopyDataForLoad(&ncentroids_, data, sizeof(int), offset, dataLen);
    CopyDataForLoad(&filterable_, data, sizeof(bool), offset, dataLen);
    CopyDataForLoad(&handleBatch_, data, sizeof(int), offset, dataLen);
    CopyDataForLoad(&nprobe_, data, sizeof(int), offset, dataLen);
    CopyDataForLoad(&searchListSize_, data, sizeof(int), offset, dataLen);

    if ((dim_ != dims) || (dim2_ != dim2) || (ncentroids_ != numLists)
        || (filterable_ != filterable) || (handleBatch_ != handleBatch)
        || (nprobe_ != nprobe) || (searchListSize_ != searchListSize)) {
        ASCEND_THROW_MSG("index shape is not correct."); // raise error
    }

    // begin of real data
    for (size_t listId = 0; listId < (size_t)numLists; listId++) {
        int idVal = -1;
        int listLenVal = 0;
        
        CopyDataForLoad(&idVal, data, sizeof(int), offset, dataLen);
        CopyDataForLoad(&listLenVal, data, sizeof(int), offset, dataLen);

        size_t listLen = (size_t) listLenVal;
        if ((idVal < 0) || (idVal >= numLists)) {
            ASCEND_THROW_MSG("index file is not correct!");
        }
        if (listLen == 0) {
            isEmptyList[listId] = true;
            listLen = static_cast<size_t>(CUBE_ALIGN);
        } else if (listLen > ONE_BUCKET_MAX_VECTOR_NUM) {
            ASCEND_THROW_MSG("index file is not correct, bucket is too huge!");
        } else {
            isEmptyList[listId] = false;
        }

        size_t tmpLen = utils::roundUp(listLen, static_cast<size_t>(CUBE_ALIGN));
        size_t tmpLenAlignFilter = utils::roundUp(listLen, static_cast<size_t>(FILTER_ALIGN));

        if (listId != (size_t) idVal) { // 跳过bucketLen * tmpLen，目的是往后挪动文件流
            std::vector<char> tmpForSkip(bucketLen * tmpLen);
            CopyDataForLoad(tmpForSkip.data(), data, bucketLen * tmpLen * sizeof(char), offset, dataLen);
            continue;
        }

        if (filterable) {
            deviceAllData[listId]->resize(tmpLen * bucketLen +
                (tmpLenAlignFilter - tmpLen) * (sizeof(uint32_t) + sizeof(uint32_t) + sizeof(uint8_t)), true);
        } else {
            deviceAllData[listId]->resize(tmpLen * bucketLen, true);
        }

        std::vector<char> dataShrPtr(tmpLen * bucketLen);
        CopyDataForLoad(dataShrPtr.data(), data, bucketLen * tmpLen * sizeof(char), offset, dataLen);

        uint8_t *dataPtr = reinterpret_cast<uint8_t *>(dataShrPtr.data());
        uint8_t *dstDataOffSet = static_cast<uint8_t *>(deviceAllData[listId]->data());

        uint8_t *tmpData = dataPtr;
        uint8_t *dstData = dstDataOffSet;
        auto ret = aclrtMemcpy(dstData, tmpLen * this->dim2,
            tmpData, tmpLen * this->dim2, ACL_MEMCPY_HOST_TO_DEVICE);
        APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "Mem operator error %d", ret);

        dstData = dstDataOffSet + tmpLen * this->dim2;
        tmpData = dataPtr + tmpLen * this->dim2;
        ret = aclrtMemcpy(dstData, tmpLen * sizeof(float),
            tmpData, tmpLen * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE);
        APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "Mem operator error %d", ret);

        dstData = dstDataOffSet + tmpLen * (this->dim2 + sizeof(float));
        tmpData = dataPtr + tmpLen * (this->dim2 + sizeof(float));
        ret = aclrtMemcpy(dstData, tmpLen * sizeof(idx_t),
            tmpData, tmpLen * sizeof(idx_t), ACL_MEMCPY_HOST_TO_DEVICE);
        APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "Mem operator error %d", ret);

        if (filterable) {
            dataPtr = dataPtr + tmpLen * (this->dim2 + sizeof(float) + sizeof(idx_t));
            dstDataOffSet = dstDataOffSet + tmpLen * (this->dim2 + sizeof(float) + sizeof(idx_t));

            dstData = dstDataOffSet;
            tmpData = dataPtr;
            ret = aclrtMemcpy(dstData, tmpLenAlignFilter * sizeof(uint8_t),
                tmpData, tmpLen * sizeof(uint8_t), ACL_MEMCPY_HOST_TO_DEVICE);
            APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "Mem operator error %d", ret);

            dstData = dstDataOffSet + tmpLenAlignFilter * sizeof(uint8_t);
            tmpData = dataPtr + tmpLen * sizeof(uint8_t);
            ret = aclrtMemcpy(dstData, tmpLenAlignFilter * sizeof(uint32_t),
                tmpData, tmpLen * sizeof(uint32_t), ACL_MEMCPY_HOST_TO_DEVICE);
            APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "Mem operator error %d", ret);

            dstData = dstDataOffSet + tmpLenAlignFilter * (sizeof(uint8_t) + sizeof(uint32_t));
            tmpData = dataPtr + tmpLen * (sizeof(uint8_t) + sizeof(uint32_t));
            ret = aclrtMemcpy(dstData, tmpLenAlignFilter * sizeof(uint32_t),
                tmpData, tmpLen * sizeof(uint32_t), ACL_MEMCPY_HOST_TO_DEVICE);
            APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "Mem operator error %d", ret);
        }

        bucketSize[listId] = listLen;
        this->ntotal += listLen;
        if (isEmptyList[listId]) {
            this->ntotal -= static_cast<size_t>(CUBE_ALIGN);
        }
        maxListLength = std::max(maxListLength, static_cast<int>(listLen));
        maxListLength = utils::roundUp(maxListLength, CUBE_ALIGN);
        addFinishFlag[listId] = 1;
    }

    pListBase = deviceAllData[0]->data();
    for (int i = 1; i < numLists; ++i) {
        pListBase = std::min(pListBase, deviceAllData[i]->data());
    }

    std::vector<float16_t> tmpDataPtr(numLists * dim2 * this->dims);
    float16_t *tmpData = tmpDataPtr.data();
    CopyDataForLoad(tmpData, data, numLists * dim2 * dims * sizeof(float16_t), offset, dataLen);
    if (loadedIndex == nullptr) {
        std::initializer_list<int> tmpShapedCentroidsInit = {utils::divUp(dim2*numLists, CUBE_ALIGN),
                                                             utils::divUp(dims, CUBE_ALIGN), CUBE_ALIGN, CUBE_ALIGN };
        auto tmpShapedCentroids = std::make_shared<AscendTensor<float16_t, DIMS_4>>(tmpShapedCentroidsInit);
        auto ret = aclrtMemcpy(tmpShapedCentroids->data(), numLists * dim2 * this->dims * sizeof(float16_t),
            tmpData, numLists * dim2 * this->dims * sizeof(float16_t), ACL_MEMCPY_HOST_TO_DEVICE);
        APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "Mem operator error %d", ret);
        coarseCentroidsShaped = tmpShapedCentroids;
    }
    
    std::vector<float16_t> tmpDataPtr2(numLists * dim2);
    tmpData = tmpDataPtr2.data();
    CopyDataForLoad(tmpData, data, numLists * dim2 * sizeof(float16_t), offset, dataLen);
    if (loadedIndex == nullptr) {
        std::initializer_list<int> tmpNormTensorInit = { numLists * dim2 };
        auto tmpNormTensor = std::make_shared<AscendTensor<float16_t, DIMS_1>>(tmpNormTensorInit);
        auto ret = aclrtMemcpy(tmpNormTensor->data(), numLists * dim2 * sizeof(float16_t),
            tmpData, numLists * dim2 * sizeof(float16_t), ACL_MEMCPY_HOST_TO_DEVICE);
        APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "Mem operator error %d", ret);
        normCoarseCentroids = tmpNormTensor;
    }

    if (loadedIndex != nullptr) {
        // 此处使当前索引共享传入的索引的码本
        this->updateCoarseCentroidsData(loadedIndex);
    }

    AscendTensor<float16_t, DIMS_1> minTensor({ this->dim2 });
    AscendTensor<float16_t, DIMS_1> diffTensor({ this->dim2 });
    std::vector<float16_t> tmpDataPtr3(this->dim2);
    CopyDataForLoad(tmpDataPtr3.data(), data, this->dim2 * sizeof(float16_t), offset, dataLen);
    auto ret = aclrtMemcpy(minTensor.data(), this->dim2 * sizeof(float16_t),
        tmpDataPtr3.data(), this->dim2 * sizeof(float16_t), ACL_MEMCPY_HOST_TO_DEVICE);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "Mem operator error %d", ret);

    std::vector<float16_t> tmpDataPtr4(this->dim2);
    CopyDataForLoad(tmpDataPtr4.data(), data, this->dim2 * sizeof(float16_t), offset, dataLen);
    ret = aclrtMemcpy(diffTensor.data(), this->dim2 * sizeof(float16_t),
                      tmpDataPtr4.data(), this->dim2 * sizeof(float16_t), ACL_MEMCPY_HOST_TO_DEVICE);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "Mem operator error %d", ret);
    updateTrainedValue(minTensor, diffTensor);

    std::vector<float> tmpDataPtr5(numLists * dim2 * this->dims);
    CopyDataForLoad(tmpDataPtr5.data(), data, numLists * dim2 * this->dims * sizeof(float), offset, dataLen);
    std::vector<float> tmpDataPtr6(this->dim2 * 2); // 2 means size of vec
    CopyDataForLoad(tmpDataPtr6.data(), data, this->dim2 * 2 * sizeof(float), offset, dataLen);

    if (loadedIndex == nullptr) {
        auto retNum = memcpy_s(codebookPtr, numLists * dim2 * this->dims * sizeof(float),
            tmpDataPtr5.data(), numLists * dim2 * this->dims * sizeof(float));
        APPERR_RETURN_IF_NOT_FMT(retNum == 0, APP_ERR_INNER_ERROR, "Mem operator error %d", retNum);
        retNum = memcpy_s(spsqPtr, this->dim2 * 2 * sizeof(float), // 2 means size of vec
            tmpDataPtr6.data(), this->dim2 * 2 * sizeof(float)); // 2 means size of vec
        APPERR_RETURN_IF_NOT_FMT(retNum == 0, APP_ERR_INNER_ERROR, "Mem operator error %d", retNum);
    }

    if (filterable) {
        for (int listId = 0; listId < numLists; ++listId) {
            size_t listLen = bucketSize[listId];
            size_t tmpLen = utils::roundUp(listLen, static_cast<size_t>(CUBE_ALIGN));
            size_t tmpLenAlignFilter = utils::roundUp(listLen, static_cast<size_t>(FILTER_ALIGN));
            uint8_t *cidIdxData = reinterpret_cast<uint8_t *>(deviceAllData[listId]->data()) +
                                  tmpLen * (this->dim2 + sizeof(float) + sizeof(idx_t));
            idxOffset[listId] = cidIdxData - reinterpret_cast<uint8_t *>(pListBase);

            int32_t *cidValData = reinterpret_cast<int32_t *>(
                reinterpret_cast<uint8_t *>(deviceAllData[listId]->data()) +
                tmpLen * (this->dim2 + sizeof(float) + sizeof(idx_t)) + tmpLenAlignFilter*sizeof(uint8_t));
            valOffset[listId] = cidValData - reinterpret_cast<int32_t *>(pListBase);

            int32_t *timestampsData = reinterpret_cast<int32_t *>(
                reinterpret_cast<uint8_t *>(deviceAllData[listId]->data()) +
                tmpLen * (this->dim2 + sizeof(float) + sizeof(idx_t))
                + tmpLenAlignFilter*(sizeof(uint8_t) + sizeof(uint32_t)));
            tsOffset[listId] = timestampsData - reinterpret_cast<int32_t *>(pListBase);
        }
    }

#ifdef HOSTCPU
#else
    topkQueue.resize(searchBatchSizes[0]);
    int maxScanSeg = utils::divUp(maxListLength, searchListSize);
    for (int i = 0; i < searchBatchSizes[0]; i++) {
        topkQueue[i].resize(maxScanSeg * nprobe);
    }
#endif
    isTrained = true;
    return APP_ERR_OK;
}

APP_ERROR IndexIVFSPSQ::saveCodeBook(uint8_t*& data, size_t& dataLen, float* codebookPtr)
{
    size_t dataLenTmp = 0; // 申请临时变量存储数据长度

    char fourcc[MAGIC_NUMBER_LEN] = {'I', 'W', 'C', 'B'}; // 如果仅序列化码本，魔术字设定为IWCB

    dataLenTmp += sizeof(fourcc) + // 校验符
        6 * sizeof(int) + // 存储 dims, dim2, numLists, handleBatch, nprobe, searchListSize 6个参数
        sizeof(bool) + // filterable布尔值
        numLists * dim2 * dims * sizeof(float16_t) + // tmpShapedCentriods
        numLists * dim2 * sizeof(float16_t) + // tmpNormTensors
        numLists * dim2 * dims * sizeof(float); // codebookPtr
    
    size_t dataOffset = 0; // 记录每次拷贝进入data指针需要的偏移量
    uint8_t* dataTmp = new uint8_t[dataLenTmp]; // 申请一个临时指针并用智能指针管理
    std::unique_ptr<uint8_t[]> del(dataTmp);

    CopyDataForSave(dataTmp, fourcc, sizeof(fourcc), dataOffset, dataLenTmp);
    CopyDataForSave(dataTmp, &(this->dims), sizeof(int), dataOffset, dataLenTmp);
    CopyDataForSave(dataTmp, &(this->dim2), sizeof(int), dataOffset, dataLenTmp);
    CopyDataForSave(dataTmp, &(numLists), sizeof(int), dataOffset, dataLenTmp);
    CopyDataForSave(dataTmp, &(filterable), sizeof(filterable), dataOffset, dataLenTmp);
    CopyDataForSave(dataTmp, &(handleBatch), sizeof(handleBatch), dataOffset, dataLenTmp);
    CopyDataForSave(dataTmp, &(nprobe), sizeof(nprobe), dataOffset, dataLenTmp);
    CopyDataForSave(dataTmp, &(searchListSize), sizeof(searchListSize), dataOffset, dataLenTmp);

    std::vector<float16_t> tmpShapedCentroids(numLists * dim2 * this->dims);
    int ret = aclrtMemcpy(tmpShapedCentroids.data(), numLists * dim2 * this->dims * sizeof(float16_t),
        coarseCentroidsShaped->data(), numLists * dim2 * this->dims * sizeof(float16_t),
        ACL_MEMCPY_DEVICE_TO_HOST);
    ASCEND_THROW_IF_NOT_FMT(ret == ACL_SUCCESS, "memcpy (error %d)", ret);

    CopyDataForSave(dataTmp, tmpShapedCentroids.data(), numLists * dim2 * this->dims * sizeof(float16_t), dataOffset,
                    dataLenTmp);

    std::vector<float16_t> tmpNormTensor(numLists * dim2);
    ret = aclrtMemcpy(tmpNormTensor.data(), numLists * dim2 * sizeof(float16_t),
        normCoarseCentroids->data(), numLists * dim2 * sizeof(float16_t), ACL_MEMCPY_DEVICE_TO_HOST);
    ASCEND_THROW_IF_NOT_FMT(ret == ACL_SUCCESS, "memcpy (error %d)", ret);
        
    CopyDataForSave(dataTmp, tmpNormTensor.data(), numLists * dim2 * sizeof(float16_t), dataOffset, dataLenTmp);
    CopyDataForSave(dataTmp, codebookPtr, this->dim2 * this->dims * numLists * sizeof(float), dataOffset, dataLenTmp);

    del.release(); // del对被管理的临时指针放权，并将其赋值给用户输入data; 将dataLenTmp赋值给dataLen
    data = dataTmp;
    dataLen = dataLenTmp;
    return APP_ERR_OK;
}

APP_ERROR IndexIVFSPSQ::loadCodeBook(const uint8_t* data, size_t dataLen, float* codebookPtr)
{
    // 计算每个参数的偏移量
    size_t offset = 0;

    char fourcc[MAGIC_NUMBER_LEN] = {'I', 'W', 'C', 'B'};
    char foureccBuffer[MAGIC_NUMBER_LEN];
    CopyDataForLoad(foureccBuffer, data, sizeof(foureccBuffer), offset, dataLen);
    for (int i = 0; i < MAGIC_NUMBER_LEN; i++) {
        if (foureccBuffer[i] != fourcc[i]) {
            ASCEND_THROW_MSG("index format for a codebook-only index is not correct.\n");
        }
    }

    int dim_ = 0;
    int dim2_ = 0;
    int ncentroids_ = 0;
    bool filterable_ = false;
    int handleBatch_ = 0;
    int nprobe_ = 0;
    int searchListSize_ = 0;
    CopyDataForLoad(&dim_, data, sizeof(int), offset, dataLen);
    CopyDataForLoad(&dim2_, data, sizeof(int), offset, dataLen);
    CopyDataForLoad(&ncentroids_, data, sizeof(int), offset, dataLen);
    CopyDataForLoad(&filterable_, data, sizeof(bool), offset, dataLen);
    CopyDataForLoad(&handleBatch_, data, sizeof(int), offset, dataLen);
    CopyDataForLoad(&nprobe_, data, sizeof(int), offset, dataLen);
    CopyDataForLoad(&searchListSize_, data, sizeof(int), offset, dataLen);

    if ((dim_ != dims) || (dim2_ != dim2) || (ncentroids_ != numLists)
        || (filterable_ != filterable) || (handleBatch_ != handleBatch)
        || (nprobe_ != nprobe) || (searchListSize_ != searchListSize)) {
        ASCEND_THROW_MSG("index shape is not correct."); // raise error
    }

    std::vector<float16_t> tmpShapedCentroidsHost(numLists * dim2 * this->dims);
    CopyDataForLoad(tmpShapedCentroidsHost.data(), data, numLists * dim2 * dims * sizeof(float16_t), offset, dataLen);
    std::initializer_list<int> tmpShapedCentroidsInit = {utils::divUp(dim2*numLists, CUBE_ALIGN),
                                                         utils::divUp(dims, CUBE_ALIGN), CUBE_ALIGN, CUBE_ALIGN };
    auto tmpShapedCentroids = std::make_shared<AscendTensor<float16_t, DIMS_4>>(tmpShapedCentroidsInit);
    int ret = aclrtMemcpy(tmpShapedCentroids->data(), numLists * dim2 * this->dims * sizeof(float16_t),
        tmpShapedCentroidsHost.data(), numLists * dim2 * this->dims * sizeof(float16_t), ACL_MEMCPY_HOST_TO_DEVICE);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "Mem operator error %d", ret);
    coarseCentroidsShaped = tmpShapedCentroids;
    
    std::vector<float16_t> tmpNormTensorHost(numLists * dim2);
    CopyDataForLoad(tmpNormTensorHost.data(), data, numLists * dim2 * sizeof(float16_t), offset, dataLen);
    std::initializer_list<int> tmpNormTensorInit = { numLists * dim2 };
    auto tmpNormTensor = std::make_shared<AscendTensor<float16_t, DIMS_1>>(tmpNormTensorInit);
    ret = aclrtMemcpy(tmpNormTensor->data(), numLists * dim2 * sizeof(float16_t),
        tmpNormTensorHost.data(), numLists * dim2 * sizeof(float16_t), ACL_MEMCPY_HOST_TO_DEVICE);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "Mem operator error %d", ret);
    normCoarseCentroids = tmpNormTensor;

    // 加载host侧(Impl侧)原始码本
    CopyDataForLoad(codebookPtr, data, numLists * dim2 * this->dims * sizeof(float), offset, dataLen);

    isTrained = true;
    return APP_ERR_OK;
}

size_t IndexIVFSPSQ::removeIds(const ::ascendSearch::IDSelector& sel)
{
    if (!isAddFinish()) {
        addFinishMerge();
    }

    //
    // | id0 id1 id2 ... id98 id99 id100 |
    //            ^                  |
    //            |__________________|
    // if id2 is to be removed, copy the last id(id100) to the mem place of id2, and decrease the list size to size-1.
    size_t removeCnt = 0;

    // dims is alignd with CUBE_ALIGN, no padding data in horizontal direction
    int dimShaped = utils::divUp(this->dim2, CUBE_ALIGN);

#pragma omp parallel for reduction(+ : removeCnt)
    for (int id = 0; id < numLists; id++) {
        addDynamicUpdate(id);

        auto &indicesList = deviceListIndices[id];
        auto &codeList = deviceListData[id];
        auto &precompList = preComputeData[id];

        auto &deviceAllDataList = deviceAllData[id];
                
        if (indicesList->size() == 0) {
            continue;
        }

#ifdef HOSTCPU
        // host need copy indices from device to host to check is_member
        std::vector<idx_t> indicesVec(indicesList->size());
        auto ret = aclrtMemcpy(indicesVec.data(), indicesList->size() * sizeof(idx_t),
                               indicesList->data(), indicesList->size() * sizeof(idx_t), ACL_MEMCPY_DEVICE_TO_HOST);
        ASCEND_THROW_IF_NOT_FMT(ret == ACL_SUCCESS, "Memcpy error %d", ret);
        idx_t *indicesCheckerPtr = indicesVec.data();
#else
        DeviceScope device;
        idx_t *indicesCheckerPtr = indicesList->data();
#endif

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
#ifdef HOSTCPU
            indicesCheckerPtr[i] = indicesCheckerPtr[j];
#endif

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

            if (filterable) {
                size_t listLen = bucketSize[id];
                size_t tmpLen = utils::roundUp(listLen, static_cast<size_t>(CUBE_ALIGN));
                size_t tmpLenAlignFilter = utils::roundUp(listLen, static_cast<size_t>(FILTER_ALIGN));
                
                uint8_t *cidIdxData = reinterpret_cast<uint8_t *>(deviceAllDataList->data()) +
                    tmpLen * (this->dim2 + sizeof(float) + sizeof(idx_t));

                int32_t *cidValData =
                    reinterpret_cast<int32_t *>(reinterpret_cast<uint8_t *>(deviceAllDataList->data()) +
                    tmpLen * (this->dim2 + sizeof(float) + sizeof(idx_t)) + tmpLenAlignFilter*sizeof(uint8_t));
                
                int32_t *timestampsData =
                    reinterpret_cast<int32_t *>(reinterpret_cast<uint8_t *>(deviceAllDataList->data()) +
                    tmpLen * (this->dim2 + sizeof(float) + sizeof(idx_t))
                    + tmpLenAlignFilter*(sizeof(uint8_t) + sizeof(uint32_t)));
                        
                err = aclrtMemcpy(cidIdxData + i, sizeof(uint8_t),
                                  cidIdxData + j, sizeof(uint8_t), ACL_MEMCPY_DEVICE_TO_DEVICE);
                ASCEND_THROW_IF_NOT_FMT(err == ACL_SUCCESS, "Memcpy error %d", (int)err);

                err = aclrtMemcpy(cidValData + i, sizeof(int32_t),
                                  cidValData + j, sizeof(int32_t), ACL_MEMCPY_DEVICE_TO_DEVICE);
                ASCEND_THROW_IF_NOT_FMT(err == ACL_SUCCESS, "Memcpy error %d", (int)err);

                err = aclrtMemcpy(timestampsData + i, sizeof(int32_t),
                                  timestampsData + j, sizeof(int32_t), ACL_MEMCPY_DEVICE_TO_DEVICE);
                ASCEND_THROW_IF_NOT_FMT(err == ACL_SUCCESS, "Memcpy error %d", (int)err);
            }

            j--;
            removeCnt++;
            hasMoved = true;
        }

        // if some code has been removed, list need to be resize and reclaim memory
        if (hasMoved) {
            size_t tmpLen = utils::roundUp((j + 1), CUBE_ALIGN);
            indicesList->resize(j + 1);
            codeList->resize(tmpLen * this->dim2);
            precompList->resize(tmpLen);

            indicesList->reclaim(false);
            codeList->reclaim(false);
            precompList->reclaim(false);
            
            bucketSize[id] = j + 1;
        }
    }
    this->ntotal -= removeCnt;
    
    addFinishMerge();
    
    return removeCnt;
}

void IndexIVFSPSQ::addCoarseCentroidsAiCpu(AscendTensor<float16_t, DIMS_2> &src,
                                           AscendTensor<float16_t, DIMS_4> &dst)
{
    std::string opName = "TransdataShapedSp";
    auto &mem = resources.getMemoryManager();
    auto stream = resources.getDefaultStream();
    AscendTensor<int64_t, DIMS_1> attr(mem, {aicpu::TRANSDATA_SHAPED_SP_ATTR_IDX_COUNT}, stream);
    attr[aicpu::TRANSDATA_SHAPED_SP_ATTR_NTOTAL_IDX] = 0;
    LaunchOpTwoInOneOut<float16_t, DIMS_2, ACL_FLOAT16,
                        int64_t, DIMS_1, ACL_INT64,
                        float16_t, DIMS_4, ACL_FLOAT16>(opName, stream, src, attr, dst);
    auto ret = aclrtSynchronizeStream(stream);
    ASCEND_THROW_IF_NOT_FMT(ret == ACL_SUCCESS, "aclrtSynchronizeStream addCoarseCentroids stream failed: %i\n", ret);
}

void IndexIVFSPSQ::fvecNormsL2sqrAicpu(AscendTensor<float16_t, DIMS_1> &nr,
                                       AscendTensor<float16_t, DIMS_2> &x)
{
    AscendOpDesc desc("VecL2SqrSp");
    std::vector<int64_t> shape0 { x.getSize(0), x.getSize(1) };
    std::vector<int64_t> shape1 { nr.getSize(0) };
    desc.addInputTensorDesc(ACL_FLOAT16, shape0.size(), shape0.data(), ACL_FORMAT_ND);
    desc.addOutputTensorDesc(ACL_FLOAT16, shape1.size(), shape1.data(), ACL_FORMAT_ND);
    auto op = CREATE_UNIQUE_PTR(AscendOperator, desc);
    if (!op->init()) {
        APP_LOG_ERROR("vec l2sqr op init failed");
        return;
    }

    std::shared_ptr<std::vector<const aclDataBuffer *>> distOpInput(
        new std::vector<const aclDataBuffer *>, CommonUtils::AclInputBufferDelete);
    distOpInput->emplace_back(aclCreateDataBuffer(x.data(), x.getSizeInBytes()));
    std::shared_ptr<std::vector<aclDataBuffer *>> distOpOutput(
         new std::vector<aclDataBuffer *>, CommonUtils::AclOutputBufferDelete);
    distOpOutput->emplace_back(aclCreateDataBuffer(nr.data(), nr.getSizeInBytes()));
    auto stream = resources.getDefaultStream();
    op->exec(*distOpInput, *distOpOutput, stream);

    auto ret = aclrtSynchronizeStream(stream);
    if (ret != ACL_SUCCESS) {
        APP_LOG_ERROR("vec l2sqr op init run failed");
    }
}

bool IndexIVFSPSQ::isAddFinish()
{
    for (size_t i = 0; i < addFinishFlag.size(); ++i) {
        if (addFinishFlag[i] == 0) { // 如果任何桶被重新添加，则它们的addFinishFlag为0，此时返回false
            return false;
        }
    }
    return true;
}

} // namespace ascendSearch
