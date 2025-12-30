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


#include <ascendsearch/ascend/utils/fp16.h>
#include "ascend/AscendIndex.h"
#include "ascenddaemon/utils/Limits.h"
#include "ascenddaemon/utils/AscendTensor.h"
#include "common/utils/CommonUtils.h"
#include "ops/cpukernel/impl/utils/kernel_shared_def.h"
#include "ascenddaemon/impl_custom/IndexIVFSPSQL2.h"
#include "index_custom/IndexIVFSPSQL2Aicpu.h"

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
const int HELPER_MASK = 255;
}

IndexIVFSPSQL2Aicpu::IndexIVFSPSQL2Aicpu(int dim, int dim2, int k, int nlist,
                                         bool encodeResidual, int nprobes,
                                         int searchListSize, int handleBatch,
                                         bool filterable, int64_t resourceSize)
    : IndexIVFSPSQL2(dim, dim2, k, nlist, encodeResidual, nprobes,
                     searchListSize, handleBatch, filterable, resourceSize)
{}

IndexIVFSPSQL2Aicpu::~IndexIVFSPSQL2Aicpu() {}

APP_ERROR IndexIVFSPSQL2Aicpu::init()
{
    APPERR_RETURN_IF_NOT_OK(resetL1TopkOp());
    APPERR_RETURN_IF_NOT_OK(resetL2TopkOp());
    APPERR_RETURN_IF_NOT_OK(resetL2TopkMultiSearchOp());
    APPERR_RETURN_IF_NOT_OK(resetL2TopkMultiSearchOpV2());
    APPERR_RETURN_IF_NOT_OK(initL1TopkAttrs());
    APPERR_RETURN_IF_NOT_OK(initL2TopkAttrs());
    return IndexIVFSPSQL2::init();
}

APP_ERROR IndexIVFSPSQL2Aicpu::initL1TopkAttrs()
{
    AscendTensor<int64_t, DIMS_1> attrsInput({ aicpu::TOPK_IVFSP_L1_ATTR_IDX_COUNT });
    std::vector<int64_t> attrs(aicpu::TOPK_IVFSP_L1_ATTR_IDX_COUNT);
    attrs[aicpu::TOPK_IVFSP_L1_ATTR_ASC_IDX] = 0;
    attrs[aicpu::TOPK_IVFSP_L1_ATTR_K_IDX] = nprobe;
    attrs[aicpu::TOPK_IVFSP_L1_ATTR_QUICK_HEAP] = 0;
    auto ret = aclrtMemcpy(attrsInput.data(), attrs.size() * sizeof(int64_t),
                           attrs.data(), attrs.size() * sizeof(int64_t), ACL_MEMCPY_HOST_TO_DEVICE);
    APPERR_RETURN_IF_NOT_LOG(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "Failed to copy attr to device"); // 0.070068

    l1Attrs = std::move(attrsInput);
    return APP_ERR_OK;
}

APP_ERROR IndexIVFSPSQL2Aicpu::initL2TopkAttrs()
{
    int maxScanSeg = utils::divUp(maxListLength, searchListSize);
    int tiles = utils::divUp(nprobe, handleBatch);
    int topk = 100;
    AscendTensor<int64_t, DIMS_1> attrs({ aicpu::TOPK_IVF_SP_ATTR_IDX_COUNT });
    std::vector<int64_t> attrsVec(aicpu::TOPK_IVF_SP_ATTR_IDX_COUNT);
    attrsVec[aicpu::TOPK_IVF_SP_ATTR_ASC_IDX] = 1;
    attrsVec[aicpu::TOPK_IVF_SP_ATTR_K_IDX] = topk; // 加个参数
    attrsVec[aicpu::TOPK_IVF_SP_ATTR_BURST_LEN_IDX] = burstLen;
    attrsVec[aicpu::TOPK_IVF_SP_ATTR_BLOCK_NUM_IDX] = tiles * maxScanSeg;
    attrsVec[aicpu::TOPK_IVF_SP_ATTR_FLAG_NUM_IDX] = SP_CORE_NUM;
    auto ret = aclrtMemcpy(attrs.data(), attrs.getSizeInBytes(),
        attrsVec.data(), attrsVec.size() * sizeof(int64_t), ACL_MEMCPY_HOST_TO_DEVICE);
    APPERR_RETURN_IF_NOT_LOG(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "Failed to copy attr to device");

    l2Attrs = std::move(attrs);
    return APP_ERR_OK;
}

void IndexIVFSPSQL2Aicpu::setNumProbes(int nprobes)
{
    IndexIVF::setNumProbes(nprobes);
    initL1TopkAttrs();
    initL2TopkAttrs();
    return;
}

APP_ERROR IndexIVFSPSQL2Aicpu::addBatched(int n, float *feature,
    float16_t *codeWord, idx_t *labels)
{
    VALUE_UNUSED(codeWord);
    auto &mem = resources.getMemoryManager();
    auto stream = resources.getDefaultStream();
    // If remaining batches to be processed > 16, we use the big batchsize QUERY_NUM because
    // we know QUERY_NUM is enough to process all batches; otherwise tailored to 16 for small batch add efficiency
    int actualQueryNum = (n > 16) ? QUERY_NUM : 16;
    AscendTensor<float16_t, DIMS_2> query(mem, { actualQueryNum, dims }, stream);

    AscendTensor<float, DIMS_2> featureFp(mem, { actualQueryNum, dims }, stream);
    auto ret = aclrtMemcpy(featureFp.data(), actualQueryNum * dims * sizeof(float),
        feature, n * dims * sizeof(float), ACL_MEMCPY_DEVICE_TO_DEVICE);
    APPERR_RETURN_IF_NOT_LOG(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "Failed to copy feature data to device");

    AscendTensor<float16_t, DIMS_2> featureFp16(mem, { actualQueryNum, dims }, stream);
    AscendTensor<uint16_t, DIMS_2> flag(mem, {8, 16}, stream);
    runFpToFp16(featureFp, featureFp16, flag, stream);

    ret = aclrtSynchronizeStream(stream);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR,
        "aclrtSynchronizeStream default stream: %i\n", ret);

    ret = aclrtMemcpy(query.data(), actualQueryNum * dims * sizeof(float16_t),
        featureFp16.data(), n * dims * sizeof(float16_t), ACL_MEMCPY_DEVICE_TO_DEVICE);
    APPERR_RETURN_IF_NOT_LOG(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "Failed to copy featureFp16 data to device");

    AscendTensor<int64_t, DIMS_2> l1TopKIndices(mem, { actualQueryNum, 1 }, stream);
    addImplL1(query, l1TopKIndices);
    ret = aclrtMemcpy(labels, n * sizeof(int64_t), l1TopKIndices.data(),
                      n * sizeof(int64_t), ACL_MEMCPY_DEVICE_TO_HOST);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "Mem operator error %d", ret);
    return APP_ERR_OK;
}


APP_ERROR IndexIVFSPSQL2Aicpu::getCodeWord(int n, float *feature,
                                           float16_t *codeWord, idx_t *labels)
{
    APP_LOG_INFO("IndexIVFSPSQL2Aicpu::getCodeWord start\n");
    APP_ERROR ret = APP_ERR_OK;
    int searched = 0;
    size_t batches = utils::divUp(n, QUERY_NUM);
    for (size_t i = 0; i < batches; i++) {
        int batchSize = QUERY_NUM;
        if ((n - searched) >= batchSize) {
            int page = (n - searched) / batchSize;
            for (int j = 0; j < page; j++) {
                ret = addBatched(batchSize, feature + searched * this->dims, codeWord + searched * this->dim2,
                                 labels + searched);
                APPERR_RETURN_IF(ret, ret);
                searched += batchSize;
            }
        }
    }
    if (n - searched > 0) {
        ret = addBatched(n - searched, feature + searched * this->dims,
            codeWord + searched * this->dim2, labels + searched);
        APPERR_RETURN_IF(ret, ret);
    }

    APP_LOG_INFO("IndexIVFSPSQL2Aicpu::getCodeWord end\n");
    return APP_ERR_OK;
}


APP_ERROR IndexIVFSPSQL2Aicpu::addImplL1(AscendTensor<float16_t, DIMS_2> &queries,
    AscendTensor<int64_t, DIMS_2> &l1TopKIndices)
{
    auto &mem = resources.getMemoryManager();
    auto stream = resources.getDefaultStream();
    auto streamAicpu = resources.getAlternateStreams()[0];
    int n = queries.getSize(0);
    AscendTensor<float16_t, DIMS_2> dists(mem, { n, codebookNum }, stream);
    AscendTensor<uint32_t, DIMS_2> opSize(mem, { CORE_NUM, SIZE_ALIGN }, stream); // op size, no use
    opSize[0][0] = codebookNum;
    AscendTensor<uint16_t, DIMS_2> opFlag(mem, { CORE_NUM, FLAG_SIZE }, stream); // 0.01973
    opFlag.zero();

    AscendTensor<float16_t, DIMS_2> l1TopNprobeDists(mem, { n, 1 }, stream);
    AscendTensor<int64_t, DIMS_1> attrsInput({ aicpu::TOPK_IVFSP_L1_ATTR_IDX_COUNT });
    std::vector<int64_t> attrs(aicpu::TOPK_IVFSP_L1_ATTR_IDX_COUNT);
    attrs[aicpu::TOPK_IVFSP_L1_ATTR_ASC_IDX] = 0;
    attrs[aicpu::TOPK_IVFSP_L1_ATTR_K_IDX] = 1;
    attrs[aicpu::TOPK_IVFSP_L1_ATTR_QUICK_HEAP] = 0;
    auto ret = aclrtMemcpy(attrsInput.data(), attrs.size() * sizeof(int64_t),
                           attrs.data(), attrs.size() * sizeof(int64_t), ACL_MEMCPY_HOST_TO_DEVICE);
    APPERR_RETURN_IF_NOT_LOG(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "Failed to copy attr to device"); // 0.070068

    runL1TopkOp1(dists, opSize, opFlag, attrsInput, l1TopNprobeDists, l1TopKIndices, streamAicpu);
    runL1DistOp(queries, *coarseCentroidsShaped, *normCoarseCentroids, dists, opFlag, stream);

    ret = aclrtSynchronizeStream(stream);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR,
        "aclrtSynchronizeStream default stream: %i\n", ret);

    ret = aclrtSynchronizeStream(streamAicpu);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR,
        "aclrtSynchronizeStream aicpu stream failed: %i\n", ret);

    return APP_ERR_OK;
}


APP_ERROR IndexIVFSPSQL2Aicpu::searchImpl(int n, const float16_t *x, int k, float16_t *distances, idx_t *labels)
{
    APP_LOG_INFO("IndexIVFSPSQL2Aicpu::searchImpl start\n");
    APP_ERROR ret = APP_ERR_OK;
    auto stream = resources.getDefaultStream();
    auto &mem = resources.getMemoryManager();

    AscendTensor<float16_t, DIMS_2> queries(const_cast<float16_t *>(x), { n, dims });

    // L1 search, to find nprobe IVF list
    // L1 topk op output: top nprobe dist and top nprobe label
    std::vector<int64_t> l1TopNprobeIndicesVec(n * nprobe, 0);
    AscendTensor<int64_t, DIMS_2> l1TopNprobeIndicesHost(l1TopNprobeIndicesVec.data(), { n, nprobe });
    ret = searchImplL1(queries, l1TopNprobeIndicesHost);
    APPERR_RETURN_IF(ret, ret);
    auto retAcl = aclrtMemcpy(l1TopNprobeIndicesVec.data(),
        l1TopNprobeIndicesVec.size() * sizeof(int64_t),
        l1TopNprobeIndicesHost.data(),
        l1TopNprobeIndicesVec.size() * sizeof(int64_t),
        ACL_MEMCPY_DEVICE_TO_HOST);
    APPERR_RETURN_IF_NOT_LOG(retAcl == ACL_SUCCESS, APP_ERR_INNER_ERROR, "Failed to copy labels back to host");
    
    AscendTensor<float16_t, DIMS_2> outDists(mem, { n, k }, stream);
    AscendTensor<int64_t, DIMS_2> outIndices(mem, { n, k }, stream);
    ret = searchImplL2(queries, l1TopNprobeIndicesHost, outDists, outIndices);
    APPERR_RETURN_IF(ret, ret);
    int outDistSize = n * k * sizeof(float16_t);
    int outIndicesSize = n * k * sizeof(int64_t);
    retAcl = aclrtMemcpy(distances, outDistSize, outDists.data(), outDistSize,
        ACL_MEMCPY_DEVICE_TO_HOST);
    APPERR_RETURN_IF_NOT_LOG(retAcl == ACL_SUCCESS, APP_ERR_INNER_ERROR, "Failed to copy dists back to host");

    retAcl = aclrtMemcpy(labels, outIndicesSize, outIndices.data(), outIndicesSize,
        ACL_MEMCPY_DEVICE_TO_HOST);
    APPERR_RETURN_IF_NOT_LOG(retAcl == ACL_SUCCESS, APP_ERR_INNER_ERROR, "Failed to copy labels back to host");

    return APP_ERR_OK;
}


APP_ERROR IndexIVFSPSQL2Aicpu::searchImplL1(AscendTensor<float16_t, DIMS_2> &queries,
    AscendTensor<int64_t, DIMS_2> &l1TopNprobeIndicesHost)
{
    auto &mem = resources.getMemoryManager();
    auto stream = resources.getDefaultStream();
    auto streamAicpu = resources.getAlternateStreams()[0];
    int n = queries.getSize(0);

    AscendTensor<float16_t, DIMS_2> dists(mem, { n, numLists }, stream);
    AscendTensor<uint32_t, DIMS_2> opSize(mem, { CORE_NUM, SIZE_ALIGN }, stream); // op size, no use
    opSize[0][0] = numLists;
    AscendTensor<uint16_t, DIMS_2> opFlag(mem, { CORE_NUM, FLAG_SIZE }, stream); // 0.01973
    std::vector<uint16_t> opFlagVec(CORE_NUM * FLAG_SIZE, 0);
    auto ret = aclrtMemcpy(opFlag.data(), opFlagVec.size() * sizeof(uint16_t),
                           opFlagVec.data(), opFlagVec.size() * sizeof(uint16_t),
                           ACL_MEMCPY_HOST_TO_DEVICE);
    APPERR_RETURN_IF_NOT_LOG(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "Failed to copy attr to device");

    AscendTensor<float16_t, DIMS_2> l1TopNprobeDists(mem, { n, nprobe }, stream);
    AscendTensor<int64_t, DIMS_2> l1TopNprobeIndices(mem, { n, nprobe }, stream);
    runL1TopkOp1(dists, opSize, opFlag, l1Attrs, l1TopNprobeDists, l1TopNprobeIndices, streamAicpu);
    runL1DistOp(queries, *coarseCentroidsShaped, *normCoarseCentroids, dists, opFlag, stream);

    ret = aclrtSynchronizeStream(stream);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS,
        APP_ERR_INNER_ERROR, "aclrtSynchronizeStream default stream: %i\n",
        ret);

    ret = aclrtSynchronizeStream(streamAicpu);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR,
        "aclrtSynchronizeStream aicpu stream failed: %i\n", ret);

    int copySize = n * nprobe * sizeof(int64_t);
    ret = aclrtMemcpy(l1TopNprobeIndicesHost.data(),
        copySize, l1TopNprobeIndices.data(),
        copySize, ACL_MEMCPY_DEVICE_TO_HOST);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS,
        APP_ERR_INNER_ERROR, "Mem operator error %d", ret);

    return APP_ERR_OK;
}

APP_ERROR IndexIVFSPSQL2Aicpu::searchImplL2(AscendTensor<float16_t, DIMS_2> &queries,
    AscendTensor<int64_t, DIMS_2> &l1TopNprobeIndices,
    AscendTensor<float16_t, DIMS_2> &outDists,
    AscendTensor<int64_t, DIMS_2> &outIndices)
{
    auto stream = resources.getDefaultStream();
    auto streamAicpu = resources.getAlternateStreams()[0];
    auto &mem = resources.getMemoryManager();
    int n = queries.getSize(0);
    int maxScanSeg = utils::divUp(maxListLength, searchListSize);
    int tiles = utils::divUp(nprobe, handleBatch);

    // inputs of dist op
    int mallocSize = n * tiles * maxScanSeg * handleBatch;
    int segHandleBatch = maxScanSeg * handleBatch;
    std::vector<int> listIdVec(mallocSize);
    AscendTensor<int, DIMS_3> listIdHost(listIdVec.data(), { n, tiles, segHandleBatch }); // 0.002-0.003

    // outputs of dist op, also inputs of aicpu topk op
    AscendTensor<float16_t, DIMS_3> distResult(mem, { n, tiles, (segHandleBatch * distsLen) }, stream);
    AscendTensor<float16_t, DIMS_3> minDistResult(mem, { n, tiles, (segHandleBatch * maxesLen) }, stream);

    uint32_t opFlagSize = static_cast<uint32_t>(n * tiles * maxScanSeg * SP_CORE_NUM * FLAG_SIZE) * sizeof(uint16_t);
    uint32_t attrsSize = aicpu::TOPK_IVF_SP_ATTR_IDX_COUNT * sizeof(int64_t);
    uint32_t opSizeLen = static_cast<uint32_t>(mallocSize) * sizeof(uint32_t);
    uint32_t normOffsetSize = static_cast<uint32_t>(mallocSize) * sizeof(uint64_t);
    uint32_t codeBookOffsetSize = static_cast<uint32_t>(mallocSize) * sizeof(uint64_t);
    uint32_t listOffsetSize = static_cast<uint32_t>(mallocSize) * sizeof(uint64_t);
    uint32_t idsSize = static_cast<uint32_t>(mallocSize) * sizeof(idx_t);
    uint32_t continuousMemSize = opFlagSize + opSizeLen + attrsSize
        + normOffsetSize + codeBookOffsetSize + listOffsetSize + idsSize;

    AscendTensor<uint8_t, DIMS_1, uint32_t> continuousMem(mem, { continuousMemSize }, streamAicpu);
    std::vector<uint8_t> continuousValue(continuousMemSize, 0);
    uint8_t *data = continuousValue.data();
    int64_t *attrs = reinterpret_cast<int64_t *>(data + opFlagSize + opSizeLen);
    attrs[aicpu::TOPK_IVF_SP_ATTR_ASC_IDX] = 1;
    attrs[aicpu::TOPK_IVF_SP_ATTR_K_IDX] = outDists.getSize(1);
    attrs[aicpu::TOPK_IVF_SP_ATTR_BURST_LEN_IDX] = burstLen;
    attrs[aicpu::TOPK_IVF_SP_ATTR_BLOCK_NUM_IDX] = tiles * maxScanSeg;
    attrs[aicpu::TOPK_IVF_SP_ATTR_FLAG_NUM_IDX] = SP_CORE_NUM;

    uint32_t *opSizeData = reinterpret_cast<uint32_t *>(data + opFlagSize);
    uint64_t *normOffsetData = reinterpret_cast<uint64_t *>(data
        + opFlagSize + opSizeLen + attrsSize);
    uint64_t *codeBookOffsetData = reinterpret_cast<uint64_t *>(data
        + opFlagSize + opSizeLen + attrsSize + normOffsetSize);
    uint64_t *listOffsetData = reinterpret_cast<uint64_t *>(data
        + opFlagSize + opSizeLen + attrsSize
        + normOffsetSize + codeBookOffsetSize);
    idx_t *idsData = reinterpret_cast<idx_t *>(data + opFlagSize
        + opSizeLen + attrsSize + normOffsetSize
        + codeBookOffsetSize + listOffsetSize);

    auto nprobeAlign = utils::roundUp(nprobe, handleBatch);
    std::vector<int> segsVec(n * utils::divUp(nprobeAlign, handleBatch));
    AscendTensor<int, DIMS_2> segsHost(segsVec.data(), { n, utils::divUp(nprobeAlign, handleBatch) });

    for (int nIdx = 0; nIdx < n; ++nIdx) {
        for (int probeIdx = 0; probeIdx < nprobeAlign; probeIdx += handleBatch) {
            int tIdx = probeIdx / handleBatch;
            AscendTensor<int, DIMS_1> listId(listIdHost[nIdx][tIdx].data(), { handleBatch });

            int* listId_ptr = listId.data();
            int64_t* l1KIndices_ptr = l1TopNprobeIndices[nIdx].data();
            for (int i = 0; i < handleBatch; ++i) {
                *(listId_ptr+i) = *(l1KIndices_ptr+std::min(probeIdx + i, nprobe - 1));
            }
            auto maxSize = isEmptyList[listId[0].value()] ? 0 : bucketSize[listId[0].value()];
            for (int i = 1; i < handleBatch; ++i) {
                maxSize = std::max(maxSize, isEmptyList[listId[i].value()] ? 0 : bucketSize[listId[i].value()]);
            }
            int segs = (int)(utils::divUp(maxSize, (size_t)distsLen));
            segsHost[nIdx][tIdx].value(segs);
            for (int m = 0; m < segs; ++m) {
                int nOffset = nIdx * tiles * segHandleBatch + tIdx * segHandleBatch + m * handleBatch;
                AscendTensor<float, DIMS_1> preNorms(reinterpret_cast<float *>(pListBase), { searchListSize });

                int segOffset = m * distsLen;
                for (int i = 0; i < handleBatch; ++i) {
                    int bucketIdx = listId[i].value();
                    int listSize = isEmptyList[bucketIdx]?0:static_cast<int>(bucketSize[bucketIdx]);
                    int tmpLen = utils::roundUp(listSize, static_cast<size_t>(CUBE_ALIGN));
                    uint32_t size = static_cast<uint32_t>(std::max(std::min(distsLen, listSize - segOffset), 0));
                    *(opSizeData + nOffset + i) = (probeIdx + i) >= nprobe ? 0 : size;
                    *(listOffsetData + nOffset + i) = deviceAllData[bucketIdx]->data() - pListBase + segOffset * dim2;
                    *(normOffsetData + nOffset + i) = reinterpret_cast<float *>(
                         deviceAllData[bucketIdx]->data() + tmpLen * dim2)
                        - preNorms.data()  + segOffset;
                    *(codeBookOffsetData + nOffset + i) = bucketIdx;
                    *(idsData + nOffset + i) = reinterpret_cast<int64_t>(reinterpret_cast<idx_t *>(
                        deviceAllData[bucketIdx]->data()
                        + tmpLen * (dim2 + sizeof(float))) + segOffset);
                }
            }
        }
    }

    auto ret = aclrtMemcpy(continuousMem.data(), continuousMem.getSizeInBytes(), continuousValue.data(),
                           continuousValue.size() * sizeof(uint8_t), ACL_MEMCPY_HOST_TO_DEVICE);
    APPERR_RETURN_IF_NOT_FMT((ret == ACL_SUCCESS), APP_ERR_INNER_ERROR, "Mem operator error %d", static_cast<int>(ret));

    uint16_t *opFlagMem = reinterpret_cast<uint16_t *>(continuousMem.data());
    AscendTensor<uint16_t, DIMS_3> opFlag(opFlagMem, { n, tiles, (maxScanSeg * SP_CORE_NUM * FLAG_SIZE)  });
    uint32_t *opSizeMem = reinterpret_cast<uint32_t *>(continuousMem.data() + opFlagSize);
    AscendTensor<uint32_t, DIMS_3> opSize(opSizeMem, { n, tiles, segHandleBatch });
    int64_t *attrMem = reinterpret_cast<int64_t *>(continuousMem.data() + opFlagSize + opSizeLen);
    AscendTensor<int64_t, DIMS_1> attrsInput(attrMem, { aicpu::TOPK_IVF_SP_ATTR_IDX_COUNT });
    uint64_t *normOffsetMem = reinterpret_cast<uint64_t *>(continuousMem.data() + opFlagSize + opSizeLen + attrsSize);
    AscendTensor<uint64_t, DIMS_3> normOffset(normOffsetMem, { n, tiles, segHandleBatch });
    uint64_t *codeBookOffsetMem = reinterpret_cast<uint64_t *>(
        continuousMem.data() + opFlagSize
        + opSizeLen + attrsSize + normOffsetSize);
    AscendTensor<uint64_t, DIMS_3> codeBookOffset(codeBookOffsetMem, { n, tiles, segHandleBatch });
    uint64_t *listOffsetMem = reinterpret_cast<uint64_t *>(
        continuousMem.data() + opFlagSize + opSizeLen
        + attrsSize + normOffsetSize + codeBookOffsetSize);
    AscendTensor<uint64_t, DIMS_3> listOffset(listOffsetMem, { n, tiles, segHandleBatch });
    idx_t *idsMem = reinterpret_cast<idx_t *>(
        continuousMem.data() + opFlagSize + opSizeLen + attrsSize
        + normOffsetSize + codeBookOffsetSize + listOffsetSize);
    AscendTensor<idx_t, DIMS_3> ids(idsMem, { n, tiles, segHandleBatch });

    runL2TopkOp(distResult, minDistResult, ids, opSize,
        opFlag, attrsInput, outDists, outIndices, streamAicpu);

    for (int nIdx = 0; nIdx < n; ++nIdx) {
        for (int probeIdx = 0; probeIdx < nprobeAlign; probeIdx += handleBatch) {
            int tIdx = probeIdx / handleBatch;
            int segs = segsHost[nIdx][tIdx].value();
            for (int m = 0; m < segs; ++m) {
                int segHandleBatch = m * handleBatch;
                AscendTensor<float16_t, DIMS_2> query(queries[nIdx].data(), { 1, dims });
                AscendTensor<float16_t, DIMS_4> book(coarseCentroidsShaped->data(),
                    { utils::divUp(dim2 * codebookNum, CUBE_ALIGN),
                    utils::divUp(dims, CUBE_ALIGN), CUBE_ALIGN, CUBE_ALIGN });
                AscendTensor<uint8_t, DIMS_1> base(pListBase, { searchListSize * dim2});
                AscendTensor<float, DIMS_1> preNorms(reinterpret_cast<float *>(pListBase), { searchListSize });
                AscendTensor<float16_t, DIMS_1> diff(vDiff.data(), { dim2 });
                AscendTensor<float16_t, DIMS_1> min(vMin.data(), { dim2 });
                AscendTensor<uint64_t, DIMS_1> baseOffset(
                    listOffset[nIdx][tIdx][segHandleBatch].data(), { handleBatch });
                AscendTensor<uint64_t, DIMS_1> codebookOffset(
                    codeBookOffset[nIdx][tIdx][segHandleBatch].data(), { handleBatch });
                AscendTensor<uint64_t, DIMS_1> preNormsOffset(
                    normOffset[nIdx][tIdx][segHandleBatch].data(), { handleBatch });
                AscendTensor<uint32_t, DIMS_1> actualSize(
                    opSize[nIdx][tIdx][segHandleBatch].data(), { handleBatch });
                AscendTensor<float16_t, DIMS_2> result(
                    distResult[nIdx][tIdx][segHandleBatch * distsLen].data(),
                    { handleBatch, distsLen });
                AscendTensor<float16_t, DIMS_2> minResult(
                    minDistResult[nIdx][tIdx][segHandleBatch * maxesLen].data(),
                    { handleBatch, maxesLen });
                AscendTensor<uint16_t, DIMS_2> flag(opFlag[nIdx][tIdx][m * SP_CORE_NUM * FLAG_SIZE].data(),
                    { SP_CORE_NUM, FLAG_SIZE });
                runSqDistOperator(query, book, base, preNorms, diff, min, baseOffset, codebookOffset,
                                  preNormsOffset, actualSize, result, minResult, flag);
            }
        }
    }

    ret = aclrtSynchronizeStream(stream);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR,
                             "aclrtSynchronizeStream default stream: %i\n", ret);
    ret = aclrtSynchronizeStream(streamAicpu);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR,
                             "aclrtSynchronizeStream aicpu stream failed: %i\n", ret);
    return APP_ERR_OK;
}
APP_ERROR IndexIVFSPSQL2Aicpu::searchImpl(std::vector<Index *> indexes, int n, int batchSize, const float16_t *x, int k,
    float16_t *distances, idx_t *labels)
{
    APP_LOG_INFO("IndexIVFSPSQL2Aicpu::searchImpl start\n");
    APP_ERROR ret = APP_ERR_OK;
    auto stream = resources.getDefaultStream();
    auto &mem = resources.getMemoryManager();

    int indexSize = static_cast<int>(indexes.size());

    AscendTensor<float16_t, DIMS_2> queries(const_cast<float16_t *>(x), { batchSize, dims });

    // L1 search, to find nprobe IVF list
    // L1 topk op output: top nprobe dist and top nprobe label
    std::vector<int64_t> l1TopNprobeIndicesVec(batchSize * nprobe, 0);
    AscendTensor<int64_t, DIMS_2> l1TopNprobeIndicesHost(
        l1TopNprobeIndicesVec.data(), {batchSize, nprobe});
    ret = searchImplL1(queries, l1TopNprobeIndicesHost);
    APPERR_RETURN_IF(ret, ret);

    AscendTensor<float16_t, DIMS_3> outDists(mem, {indexSize, batchSize, k }, stream);
    AscendTensor<int64_t, DIMS_3> outIndices(mem, {indexSize, batchSize, k }, stream);

    ret = searchImplL2(indexes, queries, l1TopNprobeIndicesHost, outDists, outIndices);
    for (int indexId = 0; indexId < indexSize; ++indexId) {
        ret = aclrtMemcpy(distances + (size_t)indexId * (size_t)n * (size_t)k,
            (size_t)batchSize * (size_t)k * sizeof(float16_t), outDists[(size_t)indexId].data(),
            (size_t)batchSize * (size_t)k * sizeof(float16_t), ACL_MEMCPY_DEVICE_TO_HOST);
        APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "Mem operator error %d", ret);

        ret = aclrtMemcpy(labels + (size_t)indexId * (size_t)n * (size_t)k,
            (size_t)batchSize * (size_t)k * sizeof(idx_t), outIndices[(size_t)indexId].data(),
            (size_t)batchSize * (size_t)k * sizeof(idx_t), ACL_MEMCPY_DEVICE_TO_HOST);
        APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "Mem operator error %d", ret);
    }

    return APP_ERR_OK;
}

APP_ERROR IndexIVFSPSQL2Aicpu::searchImplL2(std::vector<Index *> indexes,
    AscendTensor<float16_t, DIMS_2> &queries,
    AscendTensor<int64_t, DIMS_2> &l1TopNprobeIndices,
    AscendTensor<float16_t, DIMS_3> &outDists,
    AscendTensor<int64_t, DIMS_3> &outIndices)
{
    int indexSize = static_cast<int>(indexes.size());
    auto stream = resources.getDefaultStream();
    auto streamAicpu = resources.getAlternateStreams()[0];
    auto &mem = resources.getMemoryManager();
    int n = queries.getSize(0);
    int allMaxListLength = maxListLength;

    for (int indexId = 0; indexId < indexSize; ++indexId) {
            auto index = dynamic_cast<IndexIVFSPSQL2 *>(indexes[indexId]);
            if (allMaxListLength < index->maxListLength) {
                allMaxListLength = index->maxListLength;
            }
    }

    int maxScanSeg = utils::divUp(allMaxListLength, searchListSize);
    int tiles = utils::divUp(nprobe, handleBatch);

    // inputs of dist op
    int mallocSize = n * indexSize * tiles * maxScanSeg * handleBatch;
    int segHandleBatch = maxScanSeg * handleBatch;
    std::vector<int> listIdVec(n * tiles * maxScanSeg * handleBatch);
    AscendTensor<int, DIMS_3> listIdHost(listIdVec.data(), { n, tiles, segHandleBatch }); // 0.002-0.003

    // outputs of dist op, also inputs of aicpu topk op
    int MAX_INDEX_NUM = std::min(32, indexSize);
    AscendTensor<float16_t, DIMS_3> distResult(mem, {
        std::min(MAX_INDEX_NUM, indexSize), tiles,
        (segHandleBatch * distsLen) }, stream);
    AscendTensor<float16_t, DIMS_3> minDistResult(mem, {
        std::min(MAX_INDEX_NUM, indexSize), tiles,
        (segHandleBatch * maxesLen) }, stream);

    uint32_t opFlagSize = static_cast<uint32_t>(n * indexSize * tiles
        * maxScanSeg * SP_CORE_NUM * FLAG_SIZE) * sizeof(uint16_t);
    uint32_t attrsSize = aicpu::TOPK_MULTISEARCH_IVF_V2_ATTR_IDX_COUNT * sizeof(int64_t) * n;
    uint32_t opSizeLen = static_cast<uint32_t>(mallocSize) * sizeof(uint32_t);
    uint32_t normOffsetSize = static_cast<uint32_t>(mallocSize) * sizeof(uint64_t);
    uint32_t codeBookOffsetSize = static_cast<uint32_t>(mallocSize) * sizeof(uint64_t);
    uint32_t listOffsetSize = static_cast<uint32_t>(mallocSize) * sizeof(uint64_t);
    uint32_t idsSize = static_cast<uint32_t>(mallocSize) * sizeof(idx_t);
    uint32_t continuousMemSize = opFlagSize + opSizeLen + attrsSize
        + normOffsetSize + codeBookOffsetSize + listOffsetSize + idsSize;
    int OutDistTopkDim = 2;
    AscendTensor<uint8_t, DIMS_1, uint32_t> continuousMem(mem, { continuousMemSize }, streamAicpu);
    std::vector<uint8_t> continuousValue(continuousMemSize, 0);
    uint8_t *data = continuousValue.data();
    int64_t *attrs = reinterpret_cast<int64_t *>(data + opFlagSize + opSizeLen);
    int attrOffset = 0;
    for (int nIdx = 0; nIdx < n; nIdx++) {
        attrs[attrOffset + aicpu::TOPK_MULTISEARCH_IVF_V2_ATTR_ASC_IDX] = 1;
        attrs[attrOffset + aicpu::TOPK_MULTISEARCH_IVF_V2_ATTR_K_IDX] = outDists.getSize(OutDistTopkDim);
        attrs[attrOffset + aicpu::TOPK_MULTISEARCH_IVF_V2_ATTR_BURST_LEN_IDX] = burstLen;
        attrs[attrOffset + aicpu::TOPK_MULTISEARCH_IVF_V2_ATTR_BLOCK_NUM_IDX] = tiles * maxScanSeg;
        attrs[attrOffset + aicpu::TOPK_MULTISEARCH_IVF_V2_ATTR_FLAG_NUM_IDX] = SP_CORE_NUM;
        attrs[attrOffset + aicpu::TOPK_MULTISEARCH_IVF_V2_ATTR_MAX_INDEX_NUM_IDX] = MAX_INDEX_NUM;
        attrs[attrOffset + aicpu::TOPK_MULTISEARCH_IVF_V2_ATTR_INDEX_NUM_IDX] = indexSize;
        attrs[attrOffset + aicpu::TOPK_MULTISEARCH_IVF_V2_ATTR_Q_IDX] = nIdx;
        attrOffset += aicpu::TOPK_MULTISEARCH_IVF_V2_ATTR_IDX_COUNT;
    }

    uint32_t *opSizeData = reinterpret_cast<uint32_t *>(data + opFlagSize);
    uint64_t *normOffsetData = reinterpret_cast<uint64_t *>(data + opFlagSize + opSizeLen + attrsSize);
    uint64_t *codeBookOffsetData = reinterpret_cast<uint64_t *>(data + opFlagSize
        + opSizeLen + attrsSize + normOffsetSize);
    uint64_t *listOffsetData = reinterpret_cast<uint64_t *>(data + opFlagSize + opSizeLen
        + attrsSize + normOffsetSize + codeBookOffsetSize);
    idx_t *idsData = reinterpret_cast<idx_t *>(data + opFlagSize + opSizeLen
        + attrsSize + normOffsetSize + codeBookOffsetSize + listOffsetSize);

    auto nprobeAlign = utils::roundUp(nprobe, handleBatch);
    std::vector<int> segsVec(n * indexSize * utils::divUp(nprobeAlign, handleBatch));
    AscendTensor<int, DIMS_3> segsHost(segsVec.data(), { n, indexSize, utils::divUp(nprobeAlign, handleBatch) });

    for (int nIdx = 0; nIdx < n; ++nIdx) {
        for (int indexId = 0; indexId < indexSize; ++indexId) {
            auto index = dynamic_cast<IndexIVFSPSQL2 *>(indexes[indexId]);

            AscendTensor<float, DIMS_1> preNorms(reinterpret_cast<float *>(index->pListBase), { searchListSize });
            for (int probeIdx = 0; probeIdx < nprobeAlign; probeIdx += handleBatch) {
                int tIdx = probeIdx / handleBatch;

                AscendTensor<int, DIMS_1> listId(listIdHost[nIdx][tIdx].data(), { handleBatch });
                int* listId_ptr = listId.data();
                int64_t* l1KIndices_ptr = l1TopNprobeIndices[nIdx].data();
                for (int i = 0; i < handleBatch; ++i) {
                    *(listId_ptr+i) = *(l1KIndices_ptr+std::min(probeIdx + i, nprobe - 1));
                }

                auto maxSize = index->isEmptyList[listId[0].value()] ? 0 : index->bucketSize[listId[0].value()];
                for (int i = 1; i < handleBatch; ++i) {
                    maxSize = std::max(maxSize,
                        index->isEmptyList[listId[i].value()] ? 0 : index->bucketSize[listId[i].value()]);
                }
                int segs = (int)(utils::divUp(maxSize, (size_t)distsLen));
                segsHost[nIdx][indexId][tIdx].value(segs);
                for (int m = 0; m < segs; ++m) {
                    int nOffset = nIdx * indexSize * tiles * segHandleBatch
                        + indexId * tiles * segHandleBatch
                        + tIdx * segHandleBatch + m * handleBatch;

                    int segOffset = m * distsLen;
                    for (int i = 0; i < handleBatch; ++i) {
                        int bucketIdx = listId[i].value();
                        int listSize = index->isEmptyList[bucketIdx]?0:static_cast<int>(index->bucketSize[bucketIdx]);
                        int tmpLen = utils::roundUp(listSize, static_cast<size_t>(CUBE_ALIGN));
                        uint32_t size = static_cast<uint32_t>(std::max(std::min(distsLen, listSize - segOffset), 0));
                        *(opSizeData + nOffset + i) = (probeIdx + i) >= nprobe ? 0 : size;
                        *(listOffsetData + nOffset + i) = index->deviceAllData[bucketIdx]->data()
                            - index->pListBase + segOffset * dim2;
                        *(normOffsetData + nOffset + i) = reinterpret_cast<float *>(
                            index->deviceAllData[bucketIdx]->data()
                            + tmpLen * dim2) - preNorms.data()  + segOffset;
                        *(codeBookOffsetData + nOffset + i) = bucketIdx;
                        *(idsData + nOffset + i) = reinterpret_cast<int64_t>(reinterpret_cast<idx_t *>(
                            index->deviceAllData[bucketIdx]->data() + tmpLen * (dim2 + sizeof(float))) + segOffset);
                    }
                }
            }
        }
    }

    auto ret = aclrtMemcpy(continuousMem.data(), continuousMem.getSizeInBytes(), continuousValue.data(),
                           continuousValue.size() * sizeof(uint8_t), ACL_MEMCPY_HOST_TO_DEVICE);
    APPERR_RETURN_IF_NOT_FMT((ret == ACL_SUCCESS), APP_ERR_INNER_ERROR, "Mem operator error %d", static_cast<int>(ret));

    uint16_t *opFlagMem = reinterpret_cast<uint16_t *>(continuousMem.data());
    AscendTensor<uint16_t, DIMS_4> opFlag(opFlagMem, { n, indexSize, tiles, (maxScanSeg * SP_CORE_NUM * FLAG_SIZE)  });
    uint32_t *opSizeMem = reinterpret_cast<uint32_t *>(continuousMem.data() + opFlagSize);
    AscendTensor<uint32_t, DIMS_4> opSize(opSizeMem, { n, indexSize, tiles, segHandleBatch });
    int64_t *attrMem = reinterpret_cast<int64_t *>(continuousMem.data() + opFlagSize + opSizeLen);
    AscendTensor<int64_t, DIMS_2> attrsInputAll(attrMem, {n, aicpu::TOPK_MULTISEARCH_IVF_V2_ATTR_IDX_COUNT });
    uint64_t *normOffsetMem = reinterpret_cast<uint64_t *>(continuousMem.data() + opFlagSize + opSizeLen + attrsSize);
    AscendTensor<uint64_t, DIMS_4> normOffset(normOffsetMem, {n, indexSize, tiles, segHandleBatch });
    uint64_t *codeBookOffsetMem = reinterpret_cast<uint64_t *>(continuousMem.data()
        + opFlagSize + opSizeLen + attrsSize + normOffsetSize);
    AscendTensor<uint64_t, DIMS_4> codeBookOffset(codeBookOffsetMem, {n, indexSize, tiles, segHandleBatch });
    uint64_t *listOffsetMem = reinterpret_cast<uint64_t *>(continuousMem.data()
        + opFlagSize + opSizeLen + attrsSize
        + normOffsetSize + codeBookOffsetSize);
    AscendTensor<uint64_t, DIMS_4> listOffset(listOffsetMem, {n, indexSize, tiles, segHandleBatch });
    idx_t *idsMem = reinterpret_cast<idx_t *>(continuousMem.data() + opFlagSize
        + opSizeLen + attrsSize + normOffsetSize
        + codeBookOffsetSize + listOffsetSize);
    AscendTensor<idx_t, DIMS_4> ids(idsMem, { n, indexSize, tiles, segHandleBatch });

    for (int nIdx = 0; nIdx < n; ++nIdx) {
        AscendTensor<int64_t, DIMS_1> attrsInput(attrsInputAll[nIdx].data(),
            {aicpu::TOPK_MULTISEARCH_IVF_V2_ATTR_IDX_COUNT });

        runL2TopkMultiSearchV2Op(distResult, minDistResult, ids, opSize, opFlag,
            attrsInput, outDists, outIndices, streamAicpu);

        for (int indexIdx = 0; indexIdx < indexSize; indexIdx++) {
            auto index = dynamic_cast<IndexIVFSPSQL2 *>(indexes[indexIdx]);
            for (int probeIdx = 0; probeIdx < nprobeAlign; probeIdx += handleBatch) {
                int tIdx = probeIdx / handleBatch;
                int segs = segsHost[nIdx][indexIdx][tIdx].value();
                for (int m = 0; m < segs; ++m) {
                    int segHandleBatch = m * handleBatch;
                    AscendTensor<float16_t, DIMS_2> query(queries[nIdx].data(), { 1, dims });
                    AscendTensor<float16_t, DIMS_4> book(
                        index->coarseCentroidsShaped->data(),
                        { utils::divUp(dim2 * codebookNum, CUBE_ALIGN),
                        utils::divUp(dims, CUBE_ALIGN), CUBE_ALIGN, CUBE_ALIGN });
                    AscendTensor<uint8_t, DIMS_1> base(index->pListBase, { searchListSize * dim2});
                    AscendTensor<float, DIMS_1> preNorms(
                        reinterpret_cast<float *>(index->pListBase), { searchListSize });
                    AscendTensor<float16_t, DIMS_1> diff(index->vDiff.data(), { dim2 });
                    AscendTensor<float16_t, DIMS_1> min(index->vMin.data(), { dim2 });
                    AscendTensor<uint64_t, DIMS_1> baseOffset(
                        listOffset[nIdx][indexIdx][tIdx][segHandleBatch].data(), { handleBatch });
                    AscendTensor<uint64_t, DIMS_1> codebookOffset(
                        codeBookOffset[nIdx][indexIdx][tIdx][segHandleBatch].data(), { handleBatch });
                    AscendTensor<uint64_t, DIMS_1> preNormsOffset(
                        normOffset[nIdx][indexIdx][tIdx][segHandleBatch].data(), { handleBatch });
                    AscendTensor<uint32_t, DIMS_1> actualSize(
                        opSize[nIdx][indexIdx][tIdx][segHandleBatch].data(), { handleBatch });
                    AscendTensor<float16_t, DIMS_2> result(
                        distResult[indexIdx%MAX_INDEX_NUM][tIdx][segHandleBatch * distsLen].data(),
                        { handleBatch, distsLen });
                    AscendTensor<float16_t, DIMS_2> minResult(
                        minDistResult[indexIdx%MAX_INDEX_NUM][tIdx][segHandleBatch * maxesLen].data(),
                        { handleBatch, maxesLen });
                    AscendTensor<uint16_t, DIMS_2> flag(
                        opFlag[nIdx][indexIdx][tIdx][m * SP_CORE_NUM * FLAG_SIZE].data(),
                        { SP_CORE_NUM, FLAG_SIZE });
                    runSqDistOperator(query, book, base, preNorms, diff, min, baseOffset, codebookOffset,
                                      preNormsOffset, actualSize, result, minResult, flag);
                }
            }
        }
        ret = aclrtSynchronizeStream(stream);
        APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR,
                                 "aclrtSynchronizeStream default stream: %i\n", ret);

        ret = aclrtSynchronizeStream(streamAicpu);
        APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR,
                                 "aclrtSynchronizeStream aicpu stream failed: %i\n", ret);
    }
    return APP_ERR_OK;
}

APP_ERROR IndexIVFSPSQL2Aicpu::searchFilterImpl(int n, const float16_t *x, int k, float16_t *distances,
    idx_t *labels, uint32_t filterSize, uint32_t* filters)
{
    APP_LOG_INFO("IndexIVFSPSQL2Aicpu::searchFilterImpl start\n");
    APP_ERROR ret = APP_ERR_OK;
    auto stream = resources.getDefaultStream();
    auto &mem = resources.getMemoryManager();

    AscendTensor<float16_t, DIMS_2> queries(const_cast<float16_t *>(x), { n, dims });

    std::vector<int64_t> l1TopNprobeIndicesVec(n * nprobe, 0);
    AscendTensor<int64_t, DIMS_2> l1TopNprobeIndicesHost(l1TopNprobeIndicesVec.data(), { n, nprobe });
    APP_LOG_INFO("IndexIVFSPSQL2Aicpu::searchFilterImpl start\n");
    ret = searchImplL1(queries, l1TopNprobeIndicesHost);
    APPERR_RETURN_IF(ret, ret);

    AscendTensor<float16_t, DIMS_2> outDists(mem, { n, k }, stream);
    AscendTensor<int64_t, DIMS_2> outIndices(mem, { n, k }, stream);
    APP_LOG_INFO("IndexIVFSPSQL2Aicpu::searchFilterImpl start\n");
    // L2 search, search codes in nprobe IVF list to find topk results

    ret = searchFilterImplL2(queries, filterSize, filters, l1TopNprobeIndicesHost, outDists, outIndices);
    APPERR_RETURN_IF(ret, ret);

    auto retAcl = aclrtMemcpy(distances, outDists.getSizeInBytes(), outDists.data(), outDists.getSizeInBytes(),
                              ACL_MEMCPY_DEVICE_TO_HOST);
    APPERR_RETURN_IF_NOT_LOG(retAcl == ACL_SUCCESS, APP_ERR_INNER_ERROR, "Failed to copy dists back to host");

    retAcl = aclrtMemcpy(labels, outIndices.getSizeInBytes(), outIndices.data(), outIndices.getSizeInBytes(),
                         ACL_MEMCPY_DEVICE_TO_HOST);
    APPERR_RETURN_IF_NOT_LOG(retAcl == ACL_SUCCESS, APP_ERR_INNER_ERROR, "Failed to copy labels back to host");

    return APP_ERR_OK;
}

APP_ERROR IndexIVFSPSQL2Aicpu::searchFilterImplL2(AscendTensor<float16_t, DIMS_2> &queries,
    uint32_t, uint32_t* filters,
    AscendTensor<int64_t, DIMS_2> &l1TopNprobeIndices,
    AscendTensor<float16_t, DIMS_2> &outDists,
    AscendTensor<int64_t, DIMS_2> &outIndices)
{
    auto stream = resources.getDefaultStream();
    auto streamAicpu = resources.getAlternateStreams()[0];
    auto &mem = resources.getMemoryManager();
    int n = queries.getSize(0);
    int maxScanSeg = utils::divUp(maxListLength, searchListSize);
    int tiles = utils::divUp(nprobe, handleBatch);

    // inputs of dist op
    int mallocSize = n * tiles * maxScanSeg * handleBatch;
    int segHandleBatch = maxScanSeg * handleBatch;
    std::vector<int> listIdVec(mallocSize);
    AscendTensor<int, DIMS_3> listIdHost(listIdVec.data(), { n, tiles, segHandleBatch }); // 0.002-0.003

    uint32_t opFlagSize = static_cast<uint32_t>(n * tiles * maxScanSeg * SP_CORE_NUM * FLAG_SIZE) * sizeof(uint16_t);
    uint32_t attrsSize = aicpu::TOPK_IVF_SP_ATTR_IDX_COUNT * sizeof(int64_t);
    uint32_t opSizeLen = static_cast<uint32_t>(mallocSize) * sizeof(uint32_t);
    uint32_t normOffsetSize = static_cast<uint32_t>(mallocSize) * sizeof(uint64_t);
    uint32_t codeBookOffsetSize = static_cast<uint32_t>(mallocSize) * sizeof(uint64_t);
    uint32_t listOffsetSize = static_cast<uint32_t>(mallocSize) * sizeof(uint64_t);
    uint32_t idsSize = static_cast<uint32_t>(mallocSize) * sizeof(idx_t);
    uint32_t continuousMemSize = opFlagSize + opSizeLen + attrsSize
        + normOffsetSize + codeBookOffsetSize + listOffsetSize + idsSize;

        // outputs of dist op, also inputs of aicpu topk op
    AscendTensor<float16_t, DIMS_3> distResult(mem, { n, tiles, (segHandleBatch * distsLen) }, stream);
    AscendTensor<float16_t, DIMS_3> minDistResult(mem, { n, tiles, (segHandleBatch * maxesLen) }, stream);

    AscendTensor<uint8_t, DIMS_1, uint32_t> continuousMem(mem, { continuousMemSize }, streamAicpu);
    std::vector<uint8_t> continuousValue(continuousMemSize, 0);
    uint8_t *data = continuousValue.data();
    int64_t *attrs = reinterpret_cast<int64_t *>(data + opFlagSize + opSizeLen);
    attrs[aicpu::TOPK_IVF_SP_ATTR_ASC_IDX] = 1;
    attrs[aicpu::TOPK_IVF_SP_ATTR_K_IDX] = outDists.getSize(1);
    attrs[aicpu::TOPK_IVF_SP_ATTR_BURST_LEN_IDX] = burstLen;
    attrs[aicpu::TOPK_IVF_SP_ATTR_BLOCK_NUM_IDX] = tiles * maxScanSeg;
    attrs[aicpu::TOPK_IVF_SP_ATTR_FLAG_NUM_IDX] = SP_CORE_NUM;

    uint32_t *opSizeData = reinterpret_cast<uint32_t *>(data + opFlagSize);
    uint64_t *normOffsetData = reinterpret_cast<uint64_t *>(data
        + opFlagSize + opSizeLen + attrsSize);
    uint64_t *codeBookOffsetData = reinterpret_cast<uint64_t *>(data
        + opFlagSize + opSizeLen + attrsSize + normOffsetSize);
    uint64_t *listOffsetData = reinterpret_cast<uint64_t *>(data
        + opFlagSize + opSizeLen + attrsSize
        + normOffsetSize + codeBookOffsetSize);
    idx_t *idsData = reinterpret_cast<idx_t *>(data + opFlagSize
        + opSizeLen + attrsSize + normOffsetSize
        + codeBookOffsetSize + listOffsetSize);

    auto nprobeAlign = utils::roundUp(nprobe, handleBatch);
    std::vector<int> segsVec(n * utils::divUp(nprobeAlign, handleBatch));
    AscendTensor<int, DIMS_2> segsHost(segsVec.data(), { n, utils::divUp(nprobeAlign, handleBatch) });
    AscendTensor<uint8_t, DIMS_4> maskData(mem,
        { n, tiles, handleBatch * maxScanSeg, searchListSize / BIT_OF_UINT8 }, stream);
    maskData.zero();
    for (int nIdx = 0; nIdx < n; ++nIdx) {
        for (int probeIdx = 0; probeIdx < nprobeAlign; probeIdx += handleBatch) {
            int tIdx = probeIdx / handleBatch;
            AscendTensor<int, DIMS_1> listId(listIdHost[nIdx][tIdx].data(), { handleBatch });
            int* listId_ptr = listId.data();
            int64_t* l1KIndices_ptr = l1TopNprobeIndices[nIdx].data();
            for (int i = 0; i < handleBatch; ++i) {
                *(listId_ptr+i) = *(l1KIndices_ptr+std::min(probeIdx + i, nprobe - 1));
            }
            auto maxSize = isEmptyList[listId[0].value()] ? 0 : bucketSize[listId[0].value()];
            for (int i = 1; i < handleBatch; ++i) {
                maxSize = std::max(maxSize,
                    isEmptyList[listId[i].value()] ? 0 : bucketSize[listId[i].value()]);
            }
            int segs = (int)(utils::divUp(maxSize, (size_t)distsLen));
            segsHost[nIdx][tIdx].value(segs);
            for (int m = 0; m < segs; ++m) {
                int nOffset = nIdx * tiles * segHandleBatch + tIdx * segHandleBatch + m * handleBatch;
                AscendTensor<float, DIMS_1> preNorms(reinterpret_cast<float *>(pListBase), { searchListSize });
                int segOffset = m * distsLen;
                for (int i = 0; i < handleBatch; ++i) {
                    int bucketIdx = listId[i].value();
                    int listSize = isEmptyList[bucketIdx]?0:static_cast<int>(bucketSize[bucketIdx]);
                    int tmpLen = utils::roundUp(listSize, static_cast<size_t>(CUBE_ALIGN));
                    uint32_t size = static_cast<uint32_t>(std::max(std::min(distsLen, listSize - segOffset), 0));
                    *(opSizeData + nOffset + i) = (probeIdx + i) >= nprobe ? 0 : size;
                    *(listOffsetData + nOffset + i) = deviceAllData[bucketIdx]->data() - pListBase + segOffset * dim2;
                    *(normOffsetData + nOffset + i) = reinterpret_cast<float *>(
                         deviceAllData[bucketIdx]->data() + tmpLen * dim2)
                        - preNorms.data()  + segOffset;
                    *(codeBookOffsetData + nOffset + i) = bucketIdx;
                    *(idsData + nOffset + i) = reinterpret_cast<int64_t>(reinterpret_cast<idx_t *>(
                        deviceAllData[bucketIdx]->data()
                        + tmpLen * (dim2 + sizeof(float))) + segOffset);
                }
            }
        }
    }

    auto ret = aclrtMemcpy(continuousMem.data(), continuousMem.getSizeInBytes(), continuousValue.data(),
                           continuousValue.size() * sizeof(uint8_t), ACL_MEMCPY_HOST_TO_DEVICE);
    APPERR_RETURN_IF_NOT_FMT((ret == ACL_SUCCESS), APP_ERR_INNER_ERROR, "Mem operator error %d", static_cast<int>(ret));

    uint16_t *opFlagMem = reinterpret_cast<uint16_t *>(continuousMem.data());
    AscendTensor<uint16_t, DIMS_3> opFlag(opFlagMem, { n, tiles, (maxScanSeg * SP_CORE_NUM * FLAG_SIZE)  });
    uint32_t *opSizeMem = reinterpret_cast<uint32_t *>(continuousMem.data() + opFlagSize);
    AscendTensor<uint32_t, DIMS_3> opSize(opSizeMem, { n, tiles, segHandleBatch });
    int64_t *attrMem = reinterpret_cast<int64_t *>(continuousMem.data() + opFlagSize + opSizeLen);
    AscendTensor<int64_t, DIMS_1> attrsInput(attrMem, { aicpu::TOPK_IVF_SP_ATTR_IDX_COUNT });
    uint64_t *normOffsetMem = reinterpret_cast<uint64_t *>(continuousMem.data() + opFlagSize + opSizeLen + attrsSize);
    AscendTensor<uint64_t, DIMS_3> normOffset(normOffsetMem, { n, tiles, segHandleBatch });
    uint64_t *codeBookOffsetMem = reinterpret_cast<uint64_t *>(
        continuousMem.data() + opFlagSize
        + opSizeLen + attrsSize + normOffsetSize);
    AscendTensor<uint64_t, DIMS_3> codeBookOffset(codeBookOffsetMem, { n, tiles, segHandleBatch });
    uint64_t *listOffsetMem = reinterpret_cast<uint64_t *>(
        continuousMem.data() + opFlagSize + opSizeLen
        + attrsSize + normOffsetSize + codeBookOffsetSize);
    AscendTensor<uint64_t, DIMS_3> listOffset(listOffsetMem, { n, tiles, segHandleBatch });
    idx_t *idsMem = reinterpret_cast<idx_t *>(
        continuousMem.data() + opFlagSize + opSizeLen + attrsSize
        + normOffsetSize + codeBookOffsetSize + listOffsetSize);
    AscendTensor<idx_t, DIMS_3> ids(idsMem, { n, tiles, segHandleBatch });
    runL2TopkOp(distResult, minDistResult, ids, opSize,
        opFlag, attrsInput, outDists, outIndices, streamAicpu);
    for (int nIdx = 0; nIdx < n; ++nIdx) {
        for (int probeIdx = 0; probeIdx < nprobeAlign; probeIdx += handleBatch) {
            int tIdx = probeIdx / handleBatch;
            int segs = segsHost[nIdx][tIdx].value();
            AscendTensor<int, DIMS_1> listId(listIdHost[nIdx][tIdx].data(), { handleBatch });
            for (int m = 0; m < segs; ++m) {
                AscendTensor<uint8_t, DIMS_1> mask1(
                    maskData[nIdx][tIdx][m * handleBatch].data(), { handleBatch * utils::divUp(searchListSize, 8)});
                computeMask(1, m, filters + nIdx * FILTER_SIZE, mask1, listId);
                int segHandleBatch = m * handleBatch;
                AscendTensor<float16_t, DIMS_2> query(queries[nIdx].data(), { 1, dims });
                AscendTensor<float16_t, DIMS_4> book(coarseCentroidsShaped->data(),
                    { utils::divUp(dim2 * codebookNum, CUBE_ALIGN),
                    utils::divUp(dims, CUBE_ALIGN), CUBE_ALIGN, CUBE_ALIGN });
                AscendTensor<uint8_t, DIMS_1> base(pListBase, { searchListSize * dim2});
                AscendTensor<float, DIMS_1> preNorms(reinterpret_cast<float *>(pListBase), { searchListSize });
                AscendTensor<float16_t, DIMS_1> diff(vDiff.data(), { dim2 });
                AscendTensor<float16_t, DIMS_1> min(vMin.data(), { dim2 });
                AscendTensor<uint64_t, DIMS_1> baseOffset(
                    listOffset[nIdx][tIdx][segHandleBatch].data(), { handleBatch });
                AscendTensor<uint64_t, DIMS_1> codebookOffset(
                    codeBookOffset[nIdx][tIdx][segHandleBatch].data(), { handleBatch });
                AscendTensor<uint64_t, DIMS_1> preNormsOffset(
                    normOffset[nIdx][tIdx][segHandleBatch].data(), { handleBatch });
                AscendTensor<uint32_t, DIMS_1> actualSize(
                    opSize[nIdx][tIdx][segHandleBatch].data(), { handleBatch });
                AscendTensor<float16_t, DIMS_2> result(
                    distResult[nIdx][tIdx][segHandleBatch * distsLen].data(),
                    { handleBatch, distsLen });
                AscendTensor<float16_t, DIMS_2> minResult(
                    minDistResult[nIdx][tIdx][segHandleBatch * maxesLen].data(),
                    { handleBatch, maxesLen });
                AscendTensor<uint8_t, DIMS_2> mask(
                    mask1.data(), { handleBatch, utils::divUp(searchListSize, 8)});
                AscendTensor<uint16_t, DIMS_2> flag(opFlag[nIdx][tIdx][m * SP_CORE_NUM * FLAG_SIZE].data(),
                    { SP_CORE_NUM, FLAG_SIZE });
                AscendTensor<float16_t, DIMS_2> dm(vDM.data(), { 2, dim2 });
                runSqDistOperator(query, book, base, mask, preNorms, dm, baseOffset, codebookOffset,
                                  preNormsOffset, actualSize, result, minResult, flag);
            }
        }
    }
    ret = aclrtSynchronizeStream(stream);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR,
                             "aclrtSynchronizeStream default stream: %i\n", ret);
    ret = aclrtSynchronizeStream(streamAicpu);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR,
                             "aclrtSynchronizeStream aicpu stream failed: %i\n", ret);
    return APP_ERR_OK;
}

APP_ERROR IndexIVFSPSQL2Aicpu::searchFilterImpl(std::vector<Index *> indexes,
    int n, int batchSize, const float16_t *x, int k,
    float16_t *distances, idx_t *labels, uint32_t filterSize, uint32_t* filters)
{
    APP_LOG_INFO("IndexIVFSPSQL2Aicpu::searchImpl start\n");
    APP_ERROR ret = APP_ERR_OK;
    auto stream = resources.getDefaultStream();
    auto &mem = resources.getMemoryManager();

    int indexSize = static_cast<int>(indexes.size());

    AscendTensor<float16_t, DIMS_2> queries(const_cast<float16_t *>(x), { batchSize, dims });

    // L1 search, to find nprobe IVF list
    // L1 topk op output: top nprobe dist and top nprobe label
    std::vector<int64_t> l1TopNprobeIndicesVec(batchSize * nprobe, 0);
    AscendTensor<int64_t, DIMS_2> l1TopNprobeIndicesHost(l1TopNprobeIndicesVec.data(), {batchSize, nprobe});
    ret = searchImplL1(queries, l1TopNprobeIndicesHost);
    APPERR_RETURN_IF(ret, ret);

    AscendTensor<float16_t, DIMS_3> outDists(mem, {indexSize, batchSize, k }, stream);
    AscendTensor<int64_t, DIMS_3> outIndices(mem, {indexSize, batchSize, k }, stream);
    ret = searchFilterImplL2(indexes, queries, l1TopNprobeIndicesHost, outDists, outIndices, filterSize, filters);
    for (int indexId = 0; indexId < indexSize; ++indexId) {
            ret = aclrtMemcpy(distances + (size_t)indexId * (size_t)(n) * (size_t)k,
                              (size_t)batchSize *(size_t)k * sizeof(float16_t),  outDists[(size_t)indexId].data(),
                              (size_t)batchSize *(size_t)k * sizeof(float16_t), ACL_MEMCPY_DEVICE_TO_HOST);
            APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "Mem operator error %d", ret);

            ret = aclrtMemcpy(labels + (size_t)indexId * (size_t)(n) * (size_t)k,
                              (size_t)batchSize *(size_t)k * sizeof(idx_t), outIndices[(size_t)indexId].data(),
                              (size_t)batchSize *(size_t)k * sizeof(idx_t), ACL_MEMCPY_DEVICE_TO_HOST);
            APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "Mem operator error %d", ret);
    }
    return APP_ERR_OK;
}

APP_ERROR IndexIVFSPSQL2Aicpu::searchFilterImpl(std::vector<Index *> indexes,
    int n, int batchSize, const float16_t *x, int k,
    float16_t *distances, idx_t *labels, uint32_t filterSize, uint32_t** filters)
{
    APP_LOG_INFO("IndexIVFSPSQL2Aicpu::searchImpl start\n");
    APP_ERROR ret = APP_ERR_OK;
    auto stream = resources.getDefaultStream();
    auto &mem = resources.getMemoryManager();

    int indexSize = static_cast<int>(indexes.size());

    AscendTensor<float16_t, DIMS_2> queries(const_cast<float16_t *>(x), { batchSize, dims });

    // L1 search, to find nprobe IVF list
    // L1 topk op output: top nprobe dist and top nprobe label
    std::vector<int64_t> l1TopNprobeIndicesVec(batchSize * nprobe, 0);
    AscendTensor<int64_t, DIMS_2> l1TopNprobeIndicesHost(l1TopNprobeIndicesVec.data(), {batchSize, nprobe});
    ret = searchImplL1(queries, l1TopNprobeIndicesHost);
    APPERR_RETURN_IF(ret, ret);

    AscendTensor<float16_t, DIMS_3> outDists(mem, {indexSize, batchSize, k }, stream);
    AscendTensor<int64_t, DIMS_3> outIndices(mem, {indexSize, batchSize, k }, stream);

    ret = searchFilterImplL2(indexes, queries, l1TopNprobeIndicesHost, outDists, outIndices, filterSize, filters);
    for (int indexId = 0; indexId < indexSize; ++indexId) {
            ret = aclrtMemcpy(distances + (size_t)indexId * (size_t)(n) * (size_t)k,
                              (size_t)batchSize * (size_t)k * sizeof(float16_t),  outDists[(size_t)indexId].data(),
                              (size_t)batchSize * (size_t)k * sizeof(float16_t), ACL_MEMCPY_DEVICE_TO_HOST);
            APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "Mem operator error %d", ret);

            ret = aclrtMemcpy(labels + (size_t)indexId * (size_t)(n) * (size_t)k,
                              (size_t)batchSize * (size_t)k * sizeof(idx_t), outIndices[(size_t)indexId].data(),
                              (size_t)batchSize * (size_t)k * sizeof(idx_t), ACL_MEMCPY_DEVICE_TO_HOST);
            APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "Mem operator error %d", ret);
    }
    return APP_ERR_OK;
}

APP_ERROR IndexIVFSPSQL2Aicpu::searchFilterImplL2(std::vector<Index *> indexes,
    AscendTensor<float16_t, DIMS_2> &queries,
    AscendTensor<int64_t, DIMS_2> &l1TopNprobeIndices,
    AscendTensor<float16_t, DIMS_3> &outDists,
    AscendTensor<int64_t, DIMS_3> &outIndices,
    uint32_t filterSize, uint32_t* filters)
{
    VALUE_UNUSED(filterSize);
    int indexSize = static_cast<int>(indexes.size());
    auto stream = resources.getDefaultStream();
    auto streamAicpu = resources.getAlternateStreams()[0];
    auto &mem = resources.getMemoryManager();
    int n = queries.getSize(0);
    int allMaxListLength = maxListLength;

    for (int indexId = 0; indexId < indexSize; ++indexId) {
            auto index = dynamic_cast<IndexIVFSPSQL2 *>(indexes[indexId]);
            if (allMaxListLength < index->maxListLength) {
                allMaxListLength = index->maxListLength;
            }
    }

    int maxScanSeg = utils::divUp(allMaxListLength, searchListSize);
    int tiles = utils::divUp(nprobe, handleBatch);

    // inputs of dist op
    int mallocSize = n * indexSize * tiles * maxScanSeg * handleBatch;
    int segHandleBatch = maxScanSeg * handleBatch;
    std::vector<int> listIdVec(n * tiles * segHandleBatch);
    AscendTensor<int, DIMS_3> listIdHost(listIdVec.data(), { n, tiles, segHandleBatch }); // 0.002-0.003

    // outputs of dist op, also inputs of aicpu topk op
    int MAX_INDEX_NUM = std::min(32, indexSize);
    AscendTensor<float16_t, DIMS_3> distResult(mem, {
        std::min(MAX_INDEX_NUM, indexSize), tiles, (segHandleBatch * distsLen) }, stream);
    AscendTensor<float16_t, DIMS_3> minDistResult(mem, {
        std::min(MAX_INDEX_NUM, indexSize), tiles, (segHandleBatch * maxesLen) }, stream);

    // mask
    AscendTensor<uint8_t, DIMS_4> maskData(mem, { indexSize, tiles, maxScanSeg * handleBatch,
        searchListSize / BIT_OF_UINT8 }, stream);

    uint32_t opFlagSize = static_cast<uint32_t>(n * indexSize * tiles
        * maxScanSeg * SP_CORE_NUM * FLAG_SIZE) * sizeof(uint16_t);
    uint32_t attrsSize = aicpu::TOPK_MULTISEARCH_IVF_V2_ATTR_IDX_COUNT * sizeof(int64_t) * n;
    uint32_t opSizeLen = static_cast<uint32_t>(mallocSize) * sizeof(uint32_t);
    uint32_t normOffsetSize = static_cast<uint32_t>(mallocSize) * sizeof(uint64_t);
    uint32_t codeBookOffsetSize = static_cast<uint32_t>(mallocSize) * sizeof(uint64_t);
    uint32_t listOffsetSize = static_cast<uint32_t>(mallocSize) * sizeof(uint64_t);
    uint32_t idsSize = static_cast<uint32_t>(mallocSize) * sizeof(idx_t);
    uint32_t continuousMemSize = opFlagSize + opSizeLen + attrsSize
        + normOffsetSize + codeBookOffsetSize + listOffsetSize + idsSize;
    int OutDistTopkDim = 2;
    AscendTensor<uint8_t, DIMS_1, uint32_t> continuousMem(mem, { continuousMemSize }, streamAicpu);
    std::vector<uint8_t> continuousValue(continuousMemSize, 0);
    uint8_t *data = continuousValue.data();
    int64_t *attrs = reinterpret_cast<int64_t *>(data + opFlagSize + opSizeLen);
    int attrOffset = 0;
    for (int nIdx = 0; nIdx < n; nIdx++) {
        attrs[attrOffset + aicpu::TOPK_MULTISEARCH_IVF_V2_ATTR_ASC_IDX] = 1;
        attrs[attrOffset + aicpu::TOPK_MULTISEARCH_IVF_V2_ATTR_K_IDX] = outDists.getSize(OutDistTopkDim);
        attrs[attrOffset + aicpu::TOPK_MULTISEARCH_IVF_V2_ATTR_BURST_LEN_IDX] = burstLen;
        attrs[attrOffset + aicpu::TOPK_MULTISEARCH_IVF_V2_ATTR_BLOCK_NUM_IDX] = tiles * maxScanSeg;
        attrs[attrOffset + aicpu::TOPK_MULTISEARCH_IVF_V2_ATTR_FLAG_NUM_IDX] = SP_CORE_NUM;
        attrs[attrOffset + aicpu::TOPK_MULTISEARCH_IVF_V2_ATTR_MAX_INDEX_NUM_IDX] = MAX_INDEX_NUM;
        attrs[attrOffset + aicpu::TOPK_MULTISEARCH_IVF_V2_ATTR_INDEX_NUM_IDX] = indexSize;
        attrs[attrOffset + aicpu::TOPK_MULTISEARCH_IVF_V2_ATTR_Q_IDX] = nIdx;
        attrOffset += aicpu::TOPK_MULTISEARCH_IVF_V2_ATTR_IDX_COUNT;
    }

    uint32_t *opSizeData = reinterpret_cast<uint32_t *>(data + opFlagSize);
    uint64_t *normOffsetData = reinterpret_cast<uint64_t *>(data + opFlagSize
        + opSizeLen + attrsSize);
    uint64_t *codeBookOffsetData = reinterpret_cast<uint64_t *>(data + opFlagSize
        + opSizeLen + attrsSize + normOffsetSize);
    uint64_t *listOffsetData = reinterpret_cast<uint64_t *>(data + opFlagSize
        + opSizeLen + attrsSize + normOffsetSize + codeBookOffsetSize);
    idx_t *idsData = reinterpret_cast<idx_t *>(data + opFlagSize + opSizeLen
        + attrsSize + normOffsetSize + codeBookOffsetSize + listOffsetSize);

    auto nprobeAlign = utils::roundUp(nprobe, handleBatch);
    std::vector<int> segsVec(n * indexSize * utils::divUp(nprobeAlign, handleBatch));
    AscendTensor<int, DIMS_3> segsHost(segsVec.data(), { n, indexSize, utils::divUp(nprobeAlign, handleBatch) });
    for (int nIdx = 0; nIdx < n; ++nIdx) {
        for (int indexId = 0; indexId < indexSize; ++indexId) {
            auto index = dynamic_cast<IndexIVFSPSQL2 *>(indexes[indexId]);
            AscendTensor<float, DIMS_1> preNorms(
                    reinterpret_cast<float *>(index->pListBase), { searchListSize });
            for (int probeIdx = 0; probeIdx < nprobeAlign; probeIdx += handleBatch) {
                int tIdx = probeIdx / handleBatch;
                AscendTensor<int, DIMS_1> listId(listIdHost[nIdx][tIdx].data(), { handleBatch });
                int* listId_ptr = listId.data();
                int64_t* l1KIndices_ptr = l1TopNprobeIndices[nIdx].data();
                for (int i = 0; i < handleBatch; ++i) {
                    *(listId_ptr+i) = *(l1KIndices_ptr+std::min(probeIdx + i, nprobe - 1));
                }
                auto maxSize = index->isEmptyList[listId[0].value()] ? 0 : index->bucketSize[listId[0].value()];
                for (int i = 1; i < handleBatch; ++i) {
                    maxSize = std::max(maxSize,
                        index->isEmptyList[listId[i].value()] ? 0 : index->bucketSize[listId[i].value()]);
                }
                int segs = (int)(utils::divUp(maxSize, (size_t)distsLen));
                segsHost[nIdx][indexId][tIdx].value(segs);
                for (int m = 0; m < segs; ++m) {
                    int nOffset = nIdx * indexSize * tiles * segHandleBatch
                        + indexId * tiles * segHandleBatch
                        + tIdx * segHandleBatch + m * handleBatch;
                    int segOffset = m * distsLen;
                    for (int i = 0; i < handleBatch; ++i) {
                        int bucketIdx = listId[i].value();
                        int listSize = index->isEmptyList[bucketIdx]?0:static_cast<int>(index->bucketSize[bucketIdx]);
                        int tmpLen = utils::roundUp(listSize, static_cast<size_t>(CUBE_ALIGN));
                        uint32_t size = static_cast<uint32_t>(std::max(std::min(distsLen, listSize - segOffset), 0));
                        *(opSizeData + nOffset + i) = (probeIdx + i) >= nprobe ? 0 : size;
                        *(listOffsetData + nOffset + i) = index->deviceAllData[bucketIdx]->data()
                            - index->pListBase + segOffset * dim2;
                        *(normOffsetData + nOffset + i) = reinterpret_cast<float *>(
                            index->deviceAllData[bucketIdx]->data() + tmpLen * dim2)
                            - preNorms.data()  + segOffset;
                        *(codeBookOffsetData + nOffset + i) = bucketIdx;
                        *(idsData + nOffset + i) = reinterpret_cast<int64_t>(reinterpret_cast<idx_t *>(
                            index->deviceAllData[bucketIdx]->data()
                            + tmpLen * (dim2 + sizeof(float))) + segOffset);
                    }
                }
            }
        }
    }

    auto ret = aclrtMemcpy(continuousMem.data(), continuousMem.getSizeInBytes(), continuousValue.data(),
                           continuousValue.size() * sizeof(uint8_t), ACL_MEMCPY_HOST_TO_DEVICE);
    APPERR_RETURN_IF_NOT_FMT((ret == ACL_SUCCESS), APP_ERR_INNER_ERROR, "Mem operator error %d", static_cast<int>(ret));

    uint16_t *opFlagMem = reinterpret_cast<uint16_t *>(continuousMem.data());
    AscendTensor<uint16_t, DIMS_4> opFlag(opFlagMem, { n, indexSize, tiles, (maxScanSeg * SP_CORE_NUM * FLAG_SIZE)  });
    uint32_t *opSizeMem = reinterpret_cast<uint32_t *>(continuousMem.data() + opFlagSize);
    AscendTensor<uint32_t, DIMS_4> opSize(opSizeMem, { n, indexSize, tiles, segHandleBatch });
    int64_t *attrMem = reinterpret_cast<int64_t *>(continuousMem.data() + opFlagSize + opSizeLen);
    AscendTensor<int64_t, DIMS_2> attrsInputAll(attrMem, {n, aicpu::TOPK_MULTISEARCH_IVF_V2_ATTR_IDX_COUNT });
    uint64_t *normOffsetMem = reinterpret_cast<uint64_t *>(continuousMem.data() + opFlagSize + opSizeLen + attrsSize);
    AscendTensor<uint64_t, DIMS_4> normOffset(normOffsetMem, {n, indexSize, tiles, segHandleBatch });
    uint64_t *codeBookOffsetMem = reinterpret_cast<uint64_t *>(continuousMem.data()
        + opFlagSize + opSizeLen + attrsSize + normOffsetSize);
    AscendTensor<uint64_t, DIMS_4> codeBookOffset(codeBookOffsetMem, {n, indexSize, tiles, segHandleBatch });
    uint64_t *listOffsetMem = reinterpret_cast<uint64_t *>(continuousMem.data()
        + opFlagSize + opSizeLen + attrsSize + normOffsetSize + codeBookOffsetSize);
    AscendTensor<uint64_t, DIMS_4> listOffset(listOffsetMem, {n, indexSize, tiles, segHandleBatch });
    idx_t *idsMem = reinterpret_cast<idx_t *>(continuousMem.data() + opFlagSize
        + opSizeLen + attrsSize + normOffsetSize + codeBookOffsetSize + listOffsetSize);
    AscendTensor<idx_t, DIMS_4> ids(idsMem, { n, indexSize, tiles, segHandleBatch });

    for (int nIdx = 0; nIdx < n; ++nIdx) {
        AscendTensor<int64_t, DIMS_1> attrsInput(attrsInputAll[nIdx].data(),
            {aicpu::TOPK_MULTISEARCH_IVF_V2_ATTR_IDX_COUNT });

        runL2TopkMultiSearchV2Op(distResult, minDistResult, ids,
            opSize, opFlag, attrsInput, outDists, outIndices, streamAicpu);

        for (int indexIdx = 0; indexIdx < indexSize; indexIdx++) {
            auto index = dynamic_cast<IndexIVFSPSQL2 *>(indexes[indexIdx]);
            for (int probeIdx = 0; probeIdx < nprobeAlign; probeIdx += handleBatch) {
                int tIdx = probeIdx / handleBatch;
                int segs = segsHost[nIdx][indexIdx][tIdx].value();
                AscendTensor<int, DIMS_1> listId(listIdHost[nIdx][tIdx].data(), { handleBatch });
                for (int m = 0; m < segs; ++m) {
                    AscendTensor<uint8_t, DIMS_1> mask1(maskData[indexIdx][tIdx][m * handleBatch].data(),
                        { handleBatch * searchListSize / BIT_OF_UINT8 });
                    computeMask(1, m, filters + nIdx * FILTER_SIZE, mask1, listId);
                    int segHandleBatch = m * handleBatch;
                    AscendTensor<float16_t, DIMS_2> query(queries[nIdx].data(), { 1, dims });
                    AscendTensor<float16_t, DIMS_4> book(index->coarseCentroidsShaped->data(),
                        { utils::divUp(dim2 * codebookNum, CUBE_ALIGN),
                        utils::divUp(dims, CUBE_ALIGN), CUBE_ALIGN, CUBE_ALIGN });
                    AscendTensor<uint8_t, DIMS_1> base(index->pListBase, { searchListSize * dim2});
                    AscendTensor<uint8_t, DIMS_2> mask(
                        mask1.data(),
                        { handleBatch, utils::divUp(searchListSize, 8)});
                    AscendTensor<float, DIMS_1> preNorms(
                        reinterpret_cast<float *>(index->pListBase), { searchListSize }); //
                    AscendTensor<float16_t, DIMS_2> dm(vDM.data(), { 2, dim2 });
                    AscendTensor<uint64_t, DIMS_1> baseOffset(
                        listOffset[nIdx][indexIdx][tIdx][segHandleBatch].data(), { handleBatch });
                    AscendTensor<uint64_t, DIMS_1> codebookOffset(
                        codeBookOffset[nIdx][indexIdx][tIdx][segHandleBatch].data(), { handleBatch });
                    AscendTensor<uint64_t, DIMS_1> preNormsOffset(
                        normOffset[nIdx][indexIdx][tIdx][segHandleBatch].data(), { handleBatch });
                    AscendTensor<uint32_t, DIMS_1> actualSize(
                        opSize[nIdx][indexIdx][tIdx][segHandleBatch].data(), { handleBatch });
                    AscendTensor<float16_t, DIMS_2> result(
                        distResult[indexIdx%MAX_INDEX_NUM][tIdx][segHandleBatch * distsLen].data(),
                        { handleBatch, distsLen });
                    AscendTensor<float16_t, DIMS_2> minResult(
                        minDistResult[indexIdx%MAX_INDEX_NUM][tIdx][segHandleBatch * maxesLen].data(),
                        { handleBatch, maxesLen });
                    AscendTensor<uint16_t, DIMS_2> flag(
                        opFlag[nIdx][indexIdx][tIdx][m * SP_CORE_NUM * FLAG_SIZE].data(),
                        { SP_CORE_NUM, FLAG_SIZE });
                    runSqDistOperator(query, book, base, mask, preNorms, dm, baseOffset, codebookOffset,
                                      preNormsOffset, actualSize, result, minResult, flag);
                }
            }
        }
        ret = aclrtSynchronizeStream(stream);
        APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR,
                                 "aclrtSynchronizeStream default stream: %i\n", ret);

        ret = aclrtSynchronizeStream(streamAicpu);
        APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR,
                                 "aclrtSynchronizeStream aicpu stream failed: %i\n", ret);
    }
    return APP_ERR_OK;
}

APP_ERROR IndexIVFSPSQL2Aicpu::searchFilterImplL2(std::vector<Index *> indexes,
    AscendTensor<float16_t, DIMS_2> &queries,
    AscendTensor<int64_t, DIMS_2> &l1TopNprobeIndices,
    AscendTensor<float16_t, DIMS_3> &outDists,
    AscendTensor<int64_t, DIMS_3> &outIndices,
    uint32_t, uint32_t** filters)
{
    int indexSize = static_cast<int>(indexes.size());
    auto stream = resources.getDefaultStream();
    auto streamAicpu = resources.getAlternateStreams()[0];
    auto &mem = resources.getMemoryManager();
    int n = queries.getSize(0);
    int allMaxListLength = maxListLength;
    for (int indexId = 0; indexId < indexSize; ++indexId) {
            auto index = dynamic_cast<IndexIVFSPSQL2 *>(indexes[indexId]);
            if (allMaxListLength < index->maxListLength) {
                allMaxListLength = index->maxListLength;
            }
    }

    int maxScanSeg = utils::divUp(allMaxListLength, searchListSize);
    int tiles = utils::divUp(nprobe, handleBatch);

    // inputs of dist op
    int mallocSize = n * indexSize * tiles * maxScanSeg * handleBatch;
    int segHandleBatch = maxScanSeg * handleBatch;
    std::vector<int> listIdVec(n * tiles* segHandleBatch);
    AscendTensor<int, DIMS_3> listIdHost(listIdVec.data(), { n, tiles, segHandleBatch }); // 0.002-0.003

    // outputs of dist op, also inputs of aicpu topk op
    int MAX_INDEX_NUM = std::min(32, indexSize);
    AscendTensor<float16_t, DIMS_3> distResult(mem, {
        std::min(MAX_INDEX_NUM, indexSize), tiles, (segHandleBatch * distsLen) }, stream);
    AscendTensor<float16_t, DIMS_3> minDistResult(mem, {
        std::min(MAX_INDEX_NUM, indexSize), tiles, (segHandleBatch * maxesLen) }, stream);

    // mask
    AscendTensor<uint8_t, DIMS_4> maskData(mem, { indexSize, tiles, maxScanSeg * handleBatch,
        searchListSize / BIT_OF_UINT8 }, stream);

    uint32_t opFlagSize = static_cast<uint32_t>(n * indexSize * tiles
        * maxScanSeg * SP_CORE_NUM * FLAG_SIZE) * sizeof(uint16_t);
    uint32_t attrsSize = aicpu::TOPK_MULTISEARCH_IVF_V2_ATTR_IDX_COUNT * sizeof(int64_t) * n;
    uint32_t opSizeLen = static_cast<uint32_t>(mallocSize) * sizeof(uint32_t);
    uint32_t normOffsetSize = static_cast<uint32_t>(mallocSize) * sizeof(uint64_t);
    uint32_t codeBookOffsetSize = static_cast<uint32_t>(mallocSize) * sizeof(uint64_t);
    uint32_t listOffsetSize = static_cast<uint32_t>(mallocSize) * sizeof(uint64_t);
    uint32_t idsSize = static_cast<uint32_t>(mallocSize) * sizeof(idx_t);
    uint32_t continuousMemSize = opFlagSize + opSizeLen + attrsSize
        + normOffsetSize + codeBookOffsetSize + listOffsetSize + idsSize;
    int OutDistTopkDim = 2;
    AscendTensor<uint8_t, DIMS_1, uint32_t> continuousMem(mem, { continuousMemSize }, streamAicpu);
    std::vector<uint8_t> continuousValue(continuousMemSize, 0);
    uint8_t *data = continuousValue.data();
    int64_t *attrs = reinterpret_cast<int64_t *>(data + opFlagSize + opSizeLen);
    int attrOffset = 0;
    for (int nIdx = 0; nIdx < n; nIdx++) {
        attrs[attrOffset + aicpu::TOPK_MULTISEARCH_IVF_V2_ATTR_ASC_IDX] = 1;
        attrs[attrOffset + aicpu::TOPK_MULTISEARCH_IVF_V2_ATTR_K_IDX] = outDists.getSize(OutDistTopkDim);
        attrs[attrOffset + aicpu::TOPK_MULTISEARCH_IVF_V2_ATTR_BURST_LEN_IDX] = burstLen;
        attrs[attrOffset + aicpu::TOPK_MULTISEARCH_IVF_V2_ATTR_BLOCK_NUM_IDX] = tiles * maxScanSeg;
        attrs[attrOffset + aicpu::TOPK_MULTISEARCH_IVF_V2_ATTR_FLAG_NUM_IDX] = SP_CORE_NUM;
        attrs[attrOffset + aicpu::TOPK_MULTISEARCH_IVF_V2_ATTR_MAX_INDEX_NUM_IDX] = MAX_INDEX_NUM;
        attrs[attrOffset + aicpu::TOPK_MULTISEARCH_IVF_V2_ATTR_INDEX_NUM_IDX] = indexSize;
        attrs[attrOffset + aicpu::TOPK_MULTISEARCH_IVF_V2_ATTR_Q_IDX] = nIdx;
        attrOffset += aicpu::TOPK_MULTISEARCH_IVF_V2_ATTR_IDX_COUNT;
    }

    uint32_t *opSizeData = reinterpret_cast<uint32_t *>(data + opFlagSize);
    uint64_t *normOffsetData = reinterpret_cast<uint64_t *>(data + opFlagSize
        + opSizeLen + attrsSize);
    uint64_t *codeBookOffsetData = reinterpret_cast<uint64_t *>(data + opFlagSize
        + opSizeLen + attrsSize + normOffsetSize);
    uint64_t *listOffsetData = reinterpret_cast<uint64_t *>(data + opFlagSize
        + opSizeLen + attrsSize + normOffsetSize + codeBookOffsetSize);
    idx_t *idsData = reinterpret_cast<idx_t *>(data + opFlagSize + opSizeLen
        + attrsSize + normOffsetSize + codeBookOffsetSize + listOffsetSize);

    auto nprobeAlign = utils::roundUp(nprobe, handleBatch);
    std::vector<int> segsVec(n * indexSize * utils::divUp(nprobeAlign, handleBatch));
    AscendTensor<int, DIMS_3> segsHost(segsVec.data(), { n, indexSize, utils::divUp(nprobeAlign, handleBatch) });
    for (int nIdx = 0; nIdx < n; ++nIdx) {
        for (int indexId = 0; indexId < indexSize; ++indexId) {
            auto index = dynamic_cast<IndexIVFSPSQL2 *>(indexes[indexId]);
            for (int probeIdx = 0; probeIdx < nprobeAlign; probeIdx += handleBatch) {
                int tIdx = probeIdx / handleBatch;
                AscendTensor<int, DIMS_1> listId(listIdHost[nIdx][tIdx].data(), { handleBatch });
                int* listId_ptr = listId.data();
                int64_t* l1KIndices_ptr = l1TopNprobeIndices[nIdx].data();
                for (int i = 0; i < handleBatch; ++i) {
                    *(listId_ptr+i) = *(l1KIndices_ptr+std::min(probeIdx + i, nprobe - 1));
                }

                auto maxSize = index->isEmptyList[listId[0].value()] ? 0 : index->bucketSize[listId[0].value()];
                for (int i = 1; i < handleBatch; ++i) {
                    maxSize = std::max(maxSize,
                        index->isEmptyList[listId[i].value()] ? 0 : index->bucketSize[listId[i].value()]);
                }
                int segs = (int)(utils::divUp(maxSize, (size_t)distsLen));
                segsHost[nIdx][indexId][tIdx].value(segs);
                for (int m = 0; m < segs; ++m) {
                    int nOffset = nIdx * indexSize * tiles * segHandleBatch
                        + indexId * tiles * segHandleBatch
                        + tIdx * segHandleBatch + m * handleBatch;
                    AscendTensor<float, DIMS_1> preNorms(
                        reinterpret_cast<float *>(index->pListBase), { searchListSize });
                    int segOffset = m * distsLen;
                    for (int i = 0; i < handleBatch; ++i) {
                        int bucketIdx = listId[i].value();
                        int listSize = index->isEmptyList[bucketIdx]?0:static_cast<int>(index->bucketSize[bucketIdx]);
                        int tmpLen = utils::roundUp(listSize, static_cast<size_t>(CUBE_ALIGN));
                        uint32_t size = static_cast<uint32_t>(std::max(std::min(distsLen, listSize - segOffset), 0));
                        *(opSizeData + nOffset + i) = (probeIdx + i) >= nprobe ? 0 : size;
                        *(listOffsetData + nOffset + i) = index->deviceAllData[bucketIdx]->data()
                            - index->pListBase + segOffset * dim2;
                        *(normOffsetData + nOffset + i) = reinterpret_cast<float *>(
                            index->deviceAllData[bucketIdx]->data() + tmpLen * dim2)
                            - preNorms.data()  + segOffset;
                        *(codeBookOffsetData + nOffset + i) = bucketIdx;
                        *(idsData + nOffset + i) = reinterpret_cast<int64_t>(reinterpret_cast<idx_t *>(
                            index->deviceAllData[bucketIdx]->data()
                            + tmpLen * (dim2 + sizeof(float))) + segOffset);
                    }
                }
            }
        }
    }

    auto ret = aclrtMemcpy(continuousMem.data(), continuousMem.getSizeInBytes(), continuousValue.data(),
                           continuousValue.size() * sizeof(uint8_t), ACL_MEMCPY_HOST_TO_DEVICE);
    APPERR_RETURN_IF_NOT_FMT((ret == ACL_SUCCESS), APP_ERR_INNER_ERROR, "Mem operator error %d", static_cast<int>(ret));

    uint16_t *opFlagMem = reinterpret_cast<uint16_t *>(continuousMem.data());
    AscendTensor<uint16_t, DIMS_4> opFlag(opFlagMem, { n, indexSize, tiles, (maxScanSeg * SP_CORE_NUM * FLAG_SIZE)  });
    uint32_t *opSizeMem = reinterpret_cast<uint32_t *>(continuousMem.data() + opFlagSize);
    AscendTensor<uint32_t, DIMS_4> opSize(opSizeMem, { n, indexSize, tiles, segHandleBatch });
    int64_t *attrMem = reinterpret_cast<int64_t *>(continuousMem.data() + opFlagSize + opSizeLen);
    AscendTensor<int64_t, DIMS_2> attrsInputAll(attrMem, {n, aicpu::TOPK_MULTISEARCH_IVF_V2_ATTR_IDX_COUNT });
    uint64_t *normOffsetMem = reinterpret_cast<uint64_t *>(continuousMem.data() + opFlagSize + opSizeLen + attrsSize);
    AscendTensor<uint64_t, DIMS_4> normOffset(normOffsetMem, {n, indexSize, tiles, segHandleBatch });
    uint64_t *codeBookOffsetMem = reinterpret_cast<uint64_t *>(continuousMem.data()
        + opFlagSize + opSizeLen + attrsSize + normOffsetSize);
    AscendTensor<uint64_t, DIMS_4> codeBookOffset(codeBookOffsetMem, {n, indexSize, tiles, segHandleBatch });
    uint64_t *listOffsetMem = reinterpret_cast<uint64_t *>(continuousMem.data()
        + opFlagSize + opSizeLen + attrsSize + normOffsetSize + codeBookOffsetSize);
    AscendTensor<uint64_t, DIMS_4> listOffset(listOffsetMem, {n, indexSize, tiles, segHandleBatch });
    idx_t *idsMem = reinterpret_cast<idx_t *>(continuousMem.data() + opFlagSize
        + opSizeLen + attrsSize + normOffsetSize + codeBookOffsetSize + listOffsetSize);
    AscendTensor<idx_t, DIMS_4> ids(idsMem, { n, indexSize, tiles, segHandleBatch });
    for (int nIdx = 0; nIdx < n; ++nIdx) {
        AscendTensor<int64_t, DIMS_1> attrsInput(attrsInputAll[nIdx].data(),
            {aicpu::TOPK_MULTISEARCH_IVF_V2_ATTR_IDX_COUNT });

        runL2TopkMultiSearchV2Op(distResult, minDistResult, ids, opSize,
            opFlag, attrsInput, outDists, outIndices, streamAicpu);

        for (int indexIdx = 0; indexIdx < indexSize; indexIdx++) {
            auto index = dynamic_cast<IndexIVFSPSQL2 *>(indexes[indexIdx]);
            for (int probeIdx = 0; probeIdx < nprobeAlign; probeIdx += handleBatch) {
                int tIdx = probeIdx / handleBatch;
                int segs = segsHost[nIdx][indexIdx][tIdx].value();
                AscendTensor<int, DIMS_1> listId(listIdHost[nIdx][tIdx].data(), { handleBatch });
                for (int m = 0; m < segs; ++m) {
                    AscendTensor<uint8_t, DIMS_1> mask1(maskData[indexIdx][tIdx][m * handleBatch].data(),
                        { handleBatch * searchListSize / BIT_OF_UINT8 });
                    computeMask(1, m, *(filters+nIdx) + indexIdx * FILTER_SIZE, mask1, listId);
                    int segHandleBatch = m * handleBatch;
                    AscendTensor<float16_t, DIMS_2> query(queries[nIdx].data(), { 1, dims });
                    AscendTensor<float16_t, DIMS_4> book(index->coarseCentroidsShaped->data(),
                        { utils::divUp(dim2 * codebookNum, CUBE_ALIGN),
                        utils::divUp(dims, CUBE_ALIGN), CUBE_ALIGN, CUBE_ALIGN });
                    AscendTensor<uint8_t, DIMS_1> base(index->pListBase, { searchListSize * dim2});
                    AscendTensor<uint8_t, DIMS_2> mask(
                        mask1.data(),
                        { handleBatch, utils::divUp(searchListSize, 8)});
                    AscendTensor<float, DIMS_1> preNorms(
                        reinterpret_cast<float *>(index->pListBase), { searchListSize }); //
                    AscendTensor<float16_t, DIMS_2> dm(vDM.data(), { 2, dim2 });
                    AscendTensor<uint64_t, DIMS_1> baseOffset(
                        listOffset[nIdx][indexIdx][tIdx][segHandleBatch].data(), { handleBatch });
                    AscendTensor<uint64_t, DIMS_1> codebookOffset(
                        codeBookOffset[nIdx][indexIdx][tIdx][segHandleBatch].data(), { handleBatch });
                    AscendTensor<uint64_t, DIMS_1> preNormsOffset(
                        normOffset[nIdx][indexIdx][tIdx][segHandleBatch].data(), { handleBatch });
                    AscendTensor<uint32_t, DIMS_1> actualSize(
                        opSize[nIdx][indexIdx][tIdx][segHandleBatch].data(), { handleBatch });
                    AscendTensor<float16_t, DIMS_2> result(
                        distResult[indexIdx%MAX_INDEX_NUM][tIdx][segHandleBatch * distsLen].data(),
                        { handleBatch, distsLen });
                    AscendTensor<float16_t, DIMS_2> minResult(
                        minDistResult[indexIdx%MAX_INDEX_NUM][tIdx][segHandleBatch * maxesLen].data(),
                        { handleBatch, maxesLen });
                    AscendTensor<uint16_t, DIMS_2> flag(
                        opFlag[nIdx][indexIdx][tIdx][m * SP_CORE_NUM * FLAG_SIZE].data(),
                        { SP_CORE_NUM, FLAG_SIZE });
                    runSqDistOperator(query, book, base, mask, preNorms, dm, baseOffset, codebookOffset,
                                      preNormsOffset, actualSize, result, minResult, flag);
                }
            }
        }
        ret = aclrtSynchronizeStream(stream);
        APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR,
                                 "aclrtSynchronizeStream default stream: %i\n", ret);

        ret = aclrtSynchronizeStream(streamAicpu);
        APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR,
                                 "aclrtSynchronizeStream aicpu stream failed: %i\n", ret);
    }
    return APP_ERR_OK;
}

APP_ERROR IndexIVFSPSQL2Aicpu::computeMask(int n, int seg, uint32_t* filters,
                                           AscendTensor<uint8_t, DIMS_1>& masks,
                                           AscendTensor<int, DIMS_1> &listId)
{
    APP_ERROR ret = APP_ERR_OK;
    APP_LOG_INFO("IndexIVFSPSQL2Aicpu computeMask start\n");
    auto stream = resources.getDefaultStream();
    auto &mem = resources.getMemoryManager();
    AscendTensor<uint32_t, DIMS_2> filterData(filters, { n, FILTER_SIZE });
    AscendTensor<uint16_t, DIMS_3> opResult(
        reinterpret_cast<uint16_t *>(masks.data()),
        { n, handleBatch, searchListSize / BIT_OF_UINT16 });

    AscendTensor<int32_t, DIMS_2> opTsFilter(mem, { n, TS_SIZE }, stream);
    std::vector<int32_t> vOpTsFilter(n * TS_SIZE);
    AscendTensor<int32_t, DIMS_2> opTsFilterHost(vOpTsFilter.data(), { n, TS_SIZE });

    AscendTensor<uint16_t, DIMS_4> opFlag(mem, { n, handleBatch, SP_CORE_NUM, FLAG_SIZE }, stream);

    AscendTensor<int32_t, DIMS_3> opMaskFilter(mem, { n, ID_BLOCKS, MASK_SIZE }, stream);
    std::vector<int32_t> vOpMaskFilter(n * ID_BLOCKS * MASK_SIZE);
    AscendTensor<int32_t, DIMS_3> opMaskFilterHost(vOpMaskFilter.data(), { n, ID_BLOCKS, MASK_SIZE });

    AscendTensor<uint8_t, DIMS_1> idx(reinterpret_cast<uint8_t *>(pListBase), { searchListSize });
    AscendTensor<int32_t, DIMS_1> val(reinterpret_cast<int32_t *>(pListBase), { searchListSize });
    AscendTensor<int32_t, DIMS_1> ts(reinterpret_cast<int32_t *>(pListBase), { searchListSize });
    int offsetDim0 = 3;
    AscendTensor<uint64_t, DIMS_2> offset(mem, { offsetDim0, handleBatch }, stream);
    AscendTensor<uint32_t, DIMS_1> bucketSizes(mem, { handleBatch }, stream);
    std::vector<uint64_t> offsetHost(offsetDim0 * handleBatch, 0);
    std::vector<uint32_t> bucketSizesHost(handleBatch, 0);

    for (int i = 0; i < handleBatch; ++i) {
        int bucketId = listId[i].value();
        uint8_t *idxData = reinterpret_cast<uint8_t *>(*(idxOffset.data() + bucketId) +
            reinterpret_cast<uint8_t *>(pListBase));

        *(offsetHost.data() + i) = (idxData + seg * searchListSize -
            reinterpret_cast<uint8_t *>(pListBase));

        int32_t *valData = reinterpret_cast<int32_t *>(*(valOffset.data() + bucketId) +
            reinterpret_cast<int32_t *>(pListBase));

        *(offsetHost.data() + handleBatch + i) = valData + seg * searchListSize -
            reinterpret_cast<int32_t *>(pListBase);

        int32_t *timestampsData = reinterpret_cast<int32_t *>(*(tsOffset.data() + bucketId) +
            reinterpret_cast<int32_t *>(pListBase));

        *(offsetHost.data() + handleBatch + handleBatch + i) = timestampsData +
            seg * searchListSize - reinterpret_cast<int32_t *>(pListBase);

        size_t listSize = *(bucketSize.data() + bucketId) >
            uint32_t(seg*searchListSize) ? *(bucketSize.data() + bucketId) - seg * searchListSize : 0;

        *(bucketSizesHost.data() + i) = (isEmptyList[bucketId]) ? 0 : utils::roundUp(listSize,
            static_cast<size_t>(FILTER_ALIGN));
    }

    ret = aclrtMemcpy(offset.data(), offset.getSizeInBytes(), offsetHost.data(),
        offsetHost.size() * sizeof(uint64_t), ACL_MEMCPY_HOST_TO_DEVICE);
    APPERR_RETURN_IF_NOT_LOG(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "Failed to copy op size to device");

    ret = aclrtMemcpy(bucketSizes.data(), bucketSizes.getSizeInBytes(),
        bucketSizesHost.data(), bucketSizesHost.size() * sizeof(uint32_t), ACL_MEMCPY_HOST_TO_DEVICE);
    APPERR_RETURN_IF_NOT_LOG(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "Failed to copy op size to device");
    for (int nIdx = 0; nIdx < n; ++nIdx) {
        AscendTensor<int32_t, DIMS_2> maskFilterHost(
            reinterpret_cast<int32_t *>(opMaskFilterHost[nIdx].data()), { ID_BLOCKS, MASK_SIZE });
        for (int bIdx = 0; bIdx < ID_BLOCKS; ++bIdx) {
            std::fill_n(maskFilterHost[bIdx].data(), MASK_SIZE, filterData[nIdx][bIdx].value());
        }

        AscendTensor<int32_t, DIMS_1> timeFilterHost(reinterpret_cast<int32_t *>(opTsFilterHost[nIdx].data()),
            { TS_SIZE });
        vOpTsFilter[nIdx * TS_SIZE + 0] = -filterData[nIdx][ID_BLOCKS].value();
        vOpTsFilter[nIdx * TS_SIZE + 1] = -filterData[nIdx][ID_BLOCKS + 1].value();
        AscendTensor<int32_t, DIMS_2> maskFilter(opMaskFilter[nIdx].data(), { ID_BLOCKS, MASK_SIZE });
        AscendTensor<int32_t, DIMS_1> timeFilter(opTsFilter[nIdx].data(), { TS_SIZE });
        std::vector<int32_t> tmptimeFilter(TS_SIZE, 0);
        ret = aclrtMemcpy(maskFilter.data(), maskFilter.getSizeInBytes(),
            maskFilterHost.data(),
            maskFilter.getSizeInBytes(), ACL_MEMCPY_HOST_TO_DEVICE);
        APPERR_RETURN_IF_NOT_LOG(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "Failed to copy op size to device");

        ret = aclrtMemcpy(timeFilter.data(), timeFilter.getSizeInBytes(),
            vOpTsFilter.data() + nIdx * TS_SIZE,
            timeFilter.getSizeInBytes(), ACL_MEMCPY_HOST_TO_DEVICE);

        APPERR_RETURN_IF_NOT_LOG(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "Failed to copy op size to device");

        AscendTensor<uint16_t, DIMS_2> result = opResult[nIdx].view();
        AscendTensor<uint16_t, DIMS_2> flag = opFlag[nIdx][0].view();
        runCidFilterOperator(idx, val, ts, offset, bucketSizes,
                             maskFilter, timeFilter, vand, vmul, result, flag, stream);
    }
    aclrtSynchronizeStream(stream);
    APP_LOG_INFO("IndexIVFSPSQ computeMask end\n");
    return APP_ERR_OK;
}

void IndexIVFSPSQL2Aicpu::runL2TopkOp(AscendTensor<float16_t, DIMS_3> &dists,
    AscendTensor<float16_t, DIMS_3> &vmdists,
    AscendTensor<idx_t, DIMS_3> &ids,
    AscendTensor<uint32_t, DIMS_3> &sizes,
    AscendTensor<uint16_t, DIMS_3> &flags,
    AscendTensor<int64_t, DIMS_1> &attrs,
    AscendTensor<float16_t, DIMS_2> &outdists,
    AscendTensor<int64_t, DIMS_2> &outlabel, aclrtStream stream)
{
    AscendOperator *op = nullptr;
    int batch = dists.getSize(0);
    if (l2TopkOps.find(batch) != l2TopkOps.end()) {
        op = l2TopkOps[batch].get();
    }
    ASCEND_THROW_IF_NOT(op);

    std::shared_ptr<std::vector<const aclDataBuffer *>> topkOpInput(
        new std::vector<const aclDataBuffer *>(), CommonUtils::AclInputBufferDelete);
    topkOpInput->emplace_back(aclCreateDataBuffer(dists.data(), dists.getSizeInBytes()));
    topkOpInput->emplace_back(aclCreateDataBuffer(vmdists.data(), vmdists.getSizeInBytes()));
    topkOpInput->emplace_back(aclCreateDataBuffer(ids.data(), ids.getSizeInBytes()));
    topkOpInput->emplace_back(aclCreateDataBuffer(sizes.data(), sizes.getSizeInBytes()));
    topkOpInput->emplace_back(aclCreateDataBuffer(flags.data(), flags.getSizeInBytes()));
    topkOpInput->emplace_back(aclCreateDataBuffer(attrs.data(), attrs.getSizeInBytes()));

    std::shared_ptr<std::vector<aclDataBuffer *>> topkOpOutput(
        new std::vector<aclDataBuffer *>(), CommonUtils::AclOutputBufferDelete);
    topkOpOutput->emplace_back(aclCreateDataBuffer(outdists.data(), outdists.getSizeInBytes()));
    topkOpOutput->emplace_back(aclCreateDataBuffer(outlabel.data(), outlabel.getSizeInBytes()));

    op->exec(*topkOpInput, *topkOpOutput, stream);
}


void IndexIVFSPSQL2Aicpu::runL2TopkMultiSearchOp(AscendTensor<float16_t, DIMS_3> &dists,
    AscendTensor<float16_t, DIMS_3> &vmdists,
    AscendTensor<idx_t, DIMS_4> &ids,
    AscendTensor<uint32_t, DIMS_4> &sizes,
    AscendTensor<uint16_t, DIMS_4> &flags,
    AscendTensor<int64_t, DIMS_1> &attrs,
    AscendTensor<float16_t, DIMS_3> &outdists,
    AscendTensor<int64_t, DIMS_3> &outlabel,
    aclrtStream stream)
{
    AscendOperator *op = nullptr;
    int batch = outdists.getSize(1);
    if (l2TopkMultiSearchOps.find(batch) != l2TopkMultiSearchOps.end()) {
        op = l2TopkMultiSearchOps[batch].get();
    }
    ASCEND_THROW_IF_NOT(op);
    std::shared_ptr<std::vector<const aclDataBuffer *>> topkOpInput(
        new std::vector<const aclDataBuffer *>(), CommonUtils::AclInputBufferDelete);
    topkOpInput->emplace_back(aclCreateDataBuffer(dists.data(), dists.getSizeInBytes()));
    topkOpInput->emplace_back(aclCreateDataBuffer(vmdists.data(), vmdists.getSizeInBytes()));
    topkOpInput->emplace_back(aclCreateDataBuffer(ids.data(), ids.getSizeInBytes()));
    topkOpInput->emplace_back(aclCreateDataBuffer(sizes.data(), sizes.getSizeInBytes()));
    topkOpInput->emplace_back(aclCreateDataBuffer(flags.data(), flags.getSizeInBytes()));
    topkOpInput->emplace_back(aclCreateDataBuffer(attrs.data(), attrs.getSizeInBytes()));

    std::shared_ptr<std::vector<aclDataBuffer *>> topkOpOutput(
        new std::vector<aclDataBuffer *>(), CommonUtils::AclOutputBufferDelete);
    topkOpOutput->emplace_back(aclCreateDataBuffer(outdists.data(), outdists.getSizeInBytes()));
    topkOpOutput->emplace_back(aclCreateDataBuffer(outlabel.data(), outlabel.getSizeInBytes()));
    op->exec(*topkOpInput, *topkOpOutput, stream);
}

void IndexIVFSPSQL2Aicpu::runL2TopkMultiSearchV2Op(AscendTensor<float16_t, DIMS_3> &dists,
    AscendTensor<float16_t, DIMS_3> &vmdists,
    AscendTensor<idx_t, DIMS_4> &ids,
    AscendTensor<uint32_t, DIMS_4> &sizes,
    AscendTensor<uint16_t, DIMS_4> &flags,
    AscendTensor<int64_t, DIMS_1> &attrs,
    AscendTensor<float16_t, DIMS_3> &outdists,
    AscendTensor<int64_t, DIMS_3> &outlabel,
    aclrtStream stream)
{
    AscendOperator *op = nullptr;
    int batch = outdists.getSize(1);
    if (l2TopkMultiSearchOpsV2.find(batch) != l2TopkMultiSearchOpsV2.end()) {
        op = l2TopkMultiSearchOpsV2[batch].get();
    }
    ASCEND_THROW_IF_NOT(op);
    std::shared_ptr<std::vector<const aclDataBuffer *>> topkOpInput(
        new std::vector<const aclDataBuffer *>(), CommonUtils::AclInputBufferDelete);
    topkOpInput->emplace_back(aclCreateDataBuffer(dists.data(), dists.getSizeInBytes()));
    topkOpInput->emplace_back(aclCreateDataBuffer(vmdists.data(), vmdists.getSizeInBytes()));
    topkOpInput->emplace_back(aclCreateDataBuffer(ids.data(), ids.getSizeInBytes()));
    topkOpInput->emplace_back(aclCreateDataBuffer(sizes.data(), sizes.getSizeInBytes()));
    topkOpInput->emplace_back(aclCreateDataBuffer(flags.data(), flags.getSizeInBytes()));
    topkOpInput->emplace_back(aclCreateDataBuffer(attrs.data(), attrs.getSizeInBytes()));

    std::shared_ptr<std::vector<aclDataBuffer *>> topkOpOutput(
        new std::vector<aclDataBuffer *>(), CommonUtils::AclOutputBufferDelete);
    topkOpOutput->emplace_back(aclCreateDataBuffer(outdists.data(), outdists.getSizeInBytes()));
    topkOpOutput->emplace_back(aclCreateDataBuffer(outlabel.data(), outlabel.getSizeInBytes()));
    op->exec(*topkOpInput, *topkOpOutput, stream);
}

APP_ERROR IndexIVFSPSQL2Aicpu::resetL2TopkOp()
{
    auto topkCompOpReset = [&](std::unique_ptr<AscendOperator> &op, int64_t batch) {
        AscendOpDesc desc("TopkIvfSp");
        std::vector<int64_t> shape0 { batch, 0, handleBatch, distsLen };
        std::vector<int64_t> shape1 { batch, 0, handleBatch, maxesLen };
        std::vector<int64_t> shape2 { batch, 0, handleBatch };
        std::vector<int64_t> shape3 { batch, 0, handleBatch };
        std::vector<int64_t> shape4 { batch, 0, SP_CORE_NUM, FLAG_SIZE };
        std::vector<int64_t> shape5 { aicpu::TOPK_IVF_SP_ATTR_IDX_COUNT };

        std::vector<int64_t> shape6 { batch, 0 };

        desc.addInputTensorDesc(ACL_FLOAT16, shape0.size(), shape0.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_FLOAT16, shape1.size(), shape1.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_INT64, shape2.size(), shape2.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_UINT32, shape3.size(), shape3.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_UINT16, shape4.size(), shape4.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_INT64, shape5.size(), shape5.data(), ACL_FORMAT_ND);

        desc.addOutputTensorDesc(ACL_FLOAT16, shape6.size(), shape6.data(), ACL_FORMAT_ND);
        desc.addOutputTensorDesc(ACL_INT64, shape6.size(), shape6.data(), ACL_FORMAT_ND);

        op.reset();
        op = CREATE_UNIQUE_PTR(AscendOperator, desc);
        return op->init();
    };

    for (auto batch : searchBatchSizes) {
        l2TopkOps[batch] = std::unique_ptr<AscendOperator>(nullptr);
        APPERR_RETURN_IF_NOT_LOG(topkCompOpReset(l2TopkOps[batch], batch), APP_ERR_ACL_OP_LOAD_MODEL_FAILED,
                                 "l2 topk op init failed");
    }

    return APP_ERR_OK;
}


APP_ERROR IndexIVFSPSQL2Aicpu::resetL2TopkMultiSearchOp()
{
    auto topkCompOpReset = [&](std::unique_ptr<AscendOperator> &op, int64_t batch) {
        AscendOpDesc desc("TopkMultisearchIvf");
        std::vector<int64_t> shape0 { 0, 0, handleBatch, distsLen };
        std::vector<int64_t> shape1 { 0, 0, handleBatch, maxesLen };
        std::vector<int64_t> shape2 { batch, 0, 0, handleBatch };
        std::vector<int64_t> shape3 { batch, 0, 0, handleBatch };
        std::vector<int64_t> shape4 { batch, 0, 0, SP_CORE_NUM, FLAG_SIZE };
        std::vector<int64_t> shape5 { aicpu::TOPK_MULTISEARCH_IVF_ATTR_IDX_COUNT };

        std::vector<int64_t> shape6 {0,  batch,  0};

        desc.addInputTensorDesc(ACL_FLOAT16, shape0.size(), shape0.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_FLOAT16, shape1.size(), shape1.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_INT64, shape2.size(), shape2.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_UINT32, shape3.size(), shape3.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_UINT16, shape4.size(), shape4.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_INT64, shape5.size(), shape5.data(), ACL_FORMAT_ND);

        desc.addOutputTensorDesc(ACL_FLOAT16, shape6.size(), shape6.data(), ACL_FORMAT_ND);
        desc.addOutputTensorDesc(ACL_INT64, shape6.size(), shape6.data(), ACL_FORMAT_ND);

        op.reset();
        op = CREATE_UNIQUE_PTR(AscendOperator, desc);
        return op->init();
    };

    for (auto batch : searchBatchSizes) {
        l2TopkMultiSearchOps[batch] = std::unique_ptr<AscendOperator>(nullptr);
        APPERR_RETURN_IF_NOT_LOG(topkCompOpReset(l2TopkMultiSearchOps[batch], batch), APP_ERR_ACL_OP_LOAD_MODEL_FAILED,
                                 "l2 topk op init failed");
    }

    return APP_ERR_OK;
}

APP_ERROR IndexIVFSPSQL2Aicpu::resetL2TopkMultiSearchOpV2()
{
    auto topkCompOpReset = [&](std::unique_ptr<AscendOperator> &op, int64_t batch) {
        AscendOpDesc desc("TopkMultisearchIvfV2");
        std::vector<int64_t> shape0 { 0, 0, handleBatch, distsLen };
        std::vector<int64_t> shape1 { 0, 0, handleBatch, maxesLen };
        std::vector<int64_t> shape2 { batch, 0, 0, handleBatch };
        std::vector<int64_t> shape3 { batch, 0, 0, handleBatch };
        std::vector<int64_t> shape4 { batch, 0, 0, SP_CORE_NUM, FLAG_SIZE };
        std::vector<int64_t> shape5 { aicpu::TOPK_MULTISEARCH_IVF_V2_ATTR_IDX_COUNT };

        std::vector<int64_t> shape6 {0,  batch,  0};

        desc.addInputTensorDesc(ACL_FLOAT16, shape0.size(), shape0.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_FLOAT16, shape1.size(), shape1.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_INT64, shape2.size(), shape2.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_UINT32, shape3.size(), shape3.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_UINT16, shape4.size(), shape4.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_INT64, shape5.size(), shape5.data(), ACL_FORMAT_ND);

        desc.addOutputTensorDesc(ACL_FLOAT16, shape6.size(), shape6.data(), ACL_FORMAT_ND);
        desc.addOutputTensorDesc(ACL_INT64, shape6.size(), shape6.data(), ACL_FORMAT_ND);

        op.reset();
        op = CREATE_UNIQUE_PTR(AscendOperator, desc);
        return op->init();
    };

    for (auto batch : searchBatchSizes) {
        l2TopkMultiSearchOpsV2[batch] = std::unique_ptr<AscendOperator>(nullptr);
        APPERR_RETURN_IF_NOT_LOG(topkCompOpReset(l2TopkMultiSearchOpsV2[batch], batch),
            APP_ERR_ACL_OP_LOAD_MODEL_FAILED, "l2 topk op init failed");
    }

    return APP_ERR_OK;
}


APP_ERROR IndexIVFSPSQL2Aicpu::resetL1TopkOp()
{
    auto topkCompOpReset = [&](std::unique_ptr<AscendOperator> &op, int64_t batch) {
        AscendOpDesc desc("TopkIvfSpL1");
        std::vector<int64_t> shape0 { batch, numLists };
        std::vector<int64_t> shape1 { CORE_NUM, SIZE_ALIGN };
        std::vector<int64_t> shape2 { CORE_NUM, FLAG_SIZE };
        std::vector<int64_t> shape3 { aicpu::TOPK_IVFSP_L1_ATTR_IDX_COUNT };
        std::vector<int64_t> shape4 { batch, 0 };

        desc.addInputTensorDesc(ACL_FLOAT16, shape0.size(), shape0.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_UINT32, shape1.size(), shape1.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_UINT16, shape2.size(), shape2.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_INT64, shape3.size(), shape3.data(), ACL_FORMAT_ND);

        desc.addOutputTensorDesc(ACL_FLOAT16, shape4.size(), shape4.data(), ACL_FORMAT_ND);
        desc.addOutputTensorDesc(ACL_INT64, shape4.size(), shape4.data(), ACL_FORMAT_ND);

        op.reset();
        op = CREATE_UNIQUE_PTR(AscendOperator, desc);
        return op->init();
    };
    for (auto batch : searchBatchSizes) {
        l1TopkOps1[batch] = std::unique_ptr<AscendOperator>(nullptr);
        APPERR_RETURN_IF_NOT_LOG(topkCompOpReset(l1TopkOps1[batch], batch),
                                 APP_ERR_ACL_OP_LOAD_MODEL_FAILED, "l1 topk op init failed");
    }
    for (auto batch : addBatchSizes) {
        l1TopkOps1[batch] = std::unique_ptr<AscendOperator>(nullptr);
        APPERR_RETURN_IF_NOT_LOG(topkCompOpReset(l1TopkOps1[batch], batch),
                                 APP_ERR_ACL_OP_LOAD_MODEL_FAILED, "l1 topk op init failed");
    }

    return APP_ERR_OK;
}

void IndexIVFSPSQL2Aicpu::runL1TopkOp1(AscendTensor<float16_t, DIMS_2> &dists,
    AscendTensor<uint32_t, DIMS_2> &sizes,
    AscendTensor<uint16_t, DIMS_2> &flags,
    AscendTensor<int64_t, DIMS_1> &attrs,
    AscendTensor<float16_t, DIMS_2> &outdists,
    AscendTensor<int64_t, DIMS_2> &outlabel,
    aclrtStream stream)
{
    AscendOperator *op = nullptr;
    int batch = dists.getSize(0);
    if (l1TopkOps1.find(batch) != l1TopkOps1.end()) {
        op = l1TopkOps1[batch].get();
    }
    ASCEND_THROW_IF_NOT(op);

    std::shared_ptr<std::vector<const aclDataBuffer *>> topkOpInput(
        new std::vector<const aclDataBuffer *>(), CommonUtils::AclInputBufferDelete);
    topkOpInput->emplace_back(aclCreateDataBuffer(dists.data(), dists.getSizeInBytes()));
    topkOpInput->emplace_back(aclCreateDataBuffer(sizes.data(), sizes.getSizeInBytes()));
    topkOpInput->emplace_back(aclCreateDataBuffer(flags.data(), flags.getSizeInBytes()));
    topkOpInput->emplace_back(aclCreateDataBuffer(attrs.data(), attrs.getSizeInBytes()));

    std::shared_ptr<std::vector<aclDataBuffer *>> topkOpOutput(
        new std::vector<aclDataBuffer *>(), CommonUtils::AclOutputBufferDelete);
    topkOpOutput->emplace_back(aclCreateDataBuffer(outdists.data(), outdists.getSizeInBytes()));
    topkOpOutput->emplace_back(aclCreateDataBuffer(outlabel.data(), outlabel.getSizeInBytes()));

    op->exec(*topkOpInput, *topkOpOutput, stream);
}

} // namespace ascendSearch
