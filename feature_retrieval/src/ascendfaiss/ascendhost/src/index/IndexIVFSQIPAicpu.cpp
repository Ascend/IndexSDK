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


#include "index/IndexIVFSQIPAicpu.h"

#include <algorithm>

#include "ascenddaemon/impl/AuxIndexStructures.h"
#include "ascenddaemon/utils/Limits.h"
#include "common/utils/CommonUtils.h"
#include "ops/cpukernel/impl/utils/kernel_shared_def.h"

namespace ascend {
namespace {
const int SEARCH_LIST_SIZE = faiss::ascend::SocUtils::GetInstance().GetSearchListSize(); // must be CUBE_ALIGN aligned
const int SEARCH_SHAPED_SIZE = SEARCH_LIST_SIZE / CUBE_ALIGN;
const int EXTREME_LIST_SIZE = faiss::ascend::SocUtils::GetInstance().GetExtremeListSize();
const int HANDLE_BATCH = faiss::ascend::SocUtils::GetInstance().GetHandleBatch();
const int MAX_HANDLE_BATCH = 8;
}

IndexIVFSQIPAicpu::IndexIVFSQIPAicpu(int numList, int dim, bool encodeResidual, int nprobes, int64_t resourceSize)
    : IndexIVFSQ<float>(numList, dim, encodeResidual, nprobes, resourceSize),
      accumNum(0), accumAlign(0), actualAccumNum(0)
{
    this->handleBatch = HANDLE_BATCH;
    this->burstLen = (dim > 256) ? 16 : 32; // if dim more 256, ops vector calculate burst 16 else 32
    this->distsLen = SEARCH_LIST_SIZE;
    if (faiss::ascend::SocUtils::GetInstance().IsAscend310()) {
        this->maxesLen = EXTREME_LIST_SIZE;
    } else {
        this->maxesLen = SEARCH_LIST_SIZE / this->burstLen * 2; // each maximum contains 2 values
    }
}

IndexIVFSQIPAicpu::~IndexIVFSQIPAicpu() {}

APP_ERROR IndexIVFSQIPAicpu::init()
{
    APPERR_RETURN_IF_NOT_OK(resetL1TopkOp());
    APPERR_RETURN_IF_NOT_OK(resetL2TopkOp());
    APPERR_RETURN_IF_NOT_OK(resetL1DistOp(numLists));
    APPERR_RETURN_IF_NOT_OK(resetSqDistOperator());
    if (faiss::ascend::SocUtils::GetInstance().IsAscend310P()) {
        APPERR_RETURN_IF_NOT_OK(resetIvfsqAccumDistOp310P());
    }
    return IndexIVFSQ::init();
}

uint32_t IndexIVFSQIPAicpu::calMaxBatch() const
{
    auto &mem = resources.getMemoryManager();
    size_t avalibleSize = mem.getSizeAvailable();
    size_t tiles = static_cast<size_t>(utils::divUp(nprobe, handleBatch));
    size_t maxSegs = static_cast<size_t>(utils::divUp(maxListLength, SEARCH_LIST_SIZE));
    size_t distanceSize = maxSegs * tiles * static_cast<size_t>(handleBatch * distsLen) * sizeof(float16_t);
    size_t maxesSize = maxSegs * tiles * static_cast<size_t>(handleBatch * maxesLen) * sizeof(float16_t);
    size_t listOffetSize = maxSegs * tiles * static_cast<size_t>(SIZE_ALIGN) * sizeof(uint64_t);
    size_t opSize = maxSegs * tiles * static_cast<size_t>(SIZE_ALIGN) * sizeof(uint32_t);
    size_t flagSize = maxSegs * tiles * static_cast<size_t>(CORE_NUM * FLAG_ALIGN) * sizeof(uint16_t);
    size_t idsSize = maxSegs * tiles * static_cast<size_t>(handleBatch) * sizeof(uint64_t);
    size_t totalUse = distanceSize + maxesSize + listOffetSize + opSize + flagSize + idsSize;
    uint32_t batchSize = static_cast<uint32_t>(avalibleSize / totalUse);
    uint32_t batch = 1;
    uint32_t limit = static_cast<uint32_t>(searchBatchSizes.front());
    while (batch <= limit && batchSize >= batch) {
        batch = batch << 1;
    }
    return batch >> 1;
}

APP_ERROR IndexIVFSQIPAicpu::searchImplL2For310pBatch(int maxBatch,
                                                      AscendTensor<float16_t, DIMS_2> &queries,
                                                      AscendTensor<int64_t, DIMS_2> &l1TopNprobeIndicesHost,
                                                      AscendTensor<float16_t, DIMS_2> &outDists,
                                                      AscendTensor<int64_t, DIMS_2> &outIndices)
{
    int k = outDists.getSize(1);
    int n = queries.getSize(0);
    int batchs = utils::divUp(n, maxBatch);
    int retain = n;
    for (int i = 0; i < batchs; i++) {
        int subBatch = retain > maxBatch ? maxBatch : retain;
        AscendTensor<float16_t, DIMS_2> subQueries(queries[i * maxBatch][0].data(), {subBatch, dims});
        AscendTensor<int64_t, DIMS_2> subNprobeIndices(
            l1TopNprobeIndicesHost[i * maxBatch][0].data(), {subBatch, nprobe});
        AscendTensor<float16_t, DIMS_2> subOutDist(outDists[i * maxBatch][0].data(), {subBatch, k});
        AscendTensor<int64_t, DIMS_2> subOutIndices(outIndices[i * maxBatch][0].data(), {subBatch, k});
        auto ret = searchImplL2For310P(subQueries, subNprobeIndices, subOutDist, subOutIndices);
        APPERR_RETURN_IF_NOT_LOG(ret == ACL_SUCCESS, ACL_ERROR_FAILURE, "Failed to search sub batch");
        retain -= maxBatch;
    }
    return ACL_SUCCESS;
}

APP_ERROR IndexIVFSQIPAicpu::searchImplL2(AscendTensor<float16_t, DIMS_2> &queries,
                                          AscendTensor<int64_t, DIMS_2> &l1TopNprobeIndicesHost,
                                          AscendTensor<float16_t, DIMS_2> &outDists,
                                          AscendTensor<int64_t, DIMS_2> &outIndices)
{
    if (faiss::ascend::SocUtils::GetInstance().IsAscend310()) {
        return searchImplL2For310(queries, l1TopNprobeIndicesHost, outDists, outIndices);
    }
    int maxBatch = static_cast<int>(calMaxBatch());
    int n = queries.getSize(0);
    if (n <= maxBatch) {
        return searchImplL2For310P(queries, l1TopNprobeIndicesHost, outDists, outIndices);
    }
    return searchImplL2For310pBatch(maxBatch, queries, l1TopNprobeIndicesHost, outDists, outIndices);
}

void IndexIVFSQIPAicpu::fillAccumNum(int actualAccumNum, int &dim0Cnt, int &dim1Cnt, AccumOpTensor &accumOpTen)
{
    if (dim0Cnt % actualAccumNum == 0) {
        accumOpTen.accumNumHost[dim1Cnt][0].value(actualAccumNum);
        dim0Cnt = 0;
        dim1Cnt++;
    }
}

APP_ERROR IndexIVFSQIPAicpu::fillAccumAddr(int actualAccumNum, const DistOpTensor &distOpTen, AccumOpTensor &accumOpTen)
{
    int dim0Cnt = 0;
    int dim1Cnt = 0;
    int n = distOpTen.queries.getSize(0);
    for (int nIdx = 0; nIdx < n; nIdx++) {
        for (int probeIdx = 0; probeIdx < nprobe; probeIdx += handleBatch) {
            int tIdx = probeIdx / handleBatch;
            int segs = distOpTen.segsHost[nIdx][tIdx].value();
            for (int m = 0; m < segs; ++m) {
                accumOpTen.offsetAddrsHost[dim1Cnt][dim0Cnt].value(
                    reinterpret_cast<int64_t>(distOpTen.listOffset[nIdx][tIdx][m * SIZE_ALIGN].data()));
                accumOpTen.opSizeAddrsHost[dim1Cnt][dim0Cnt].value(
                    reinterpret_cast<int64_t>(distOpTen.opSize[nIdx][tIdx][m * SIZE_ALIGN].data()));
                accumOpTen.distAddrsHost[dim1Cnt][dim0Cnt].value(
                    reinterpret_cast<int64_t>(distOpTen.distResult[nIdx][tIdx][m * handleBatch * distsLen].data()));
                accumOpTen.maxAddrsHost[dim1Cnt][dim0Cnt].value(
                    reinterpret_cast<int64_t>(distOpTen.maxDistResult[nIdx][tIdx][m * handleBatch * maxesLen].data()));
                accumOpTen.flagAddrsHost[dim1Cnt][dim0Cnt].value(
                    reinterpret_cast<int64_t>(distOpTen.opFlag[nIdx][tIdx][m * CORE_NUM * FLAG_ALIGN].data()));
                accumOpTen.queryOffsetHost[dim1Cnt][dim0Cnt].value(static_cast<uint64_t>(nIdx));
                dim0Cnt++;
                fillAccumNum(actualAccumNum, dim0Cnt, dim1Cnt, accumOpTen);
            }
        }
    }
    accumOpTen.accumNumHost[dim1Cnt][0].value(dim0Cnt);
    return ACL_SUCCESS;
}

APP_ERROR IndexIVFSQIPAicpu::runAccumOp(DistOpTensor &dot, AccumOpTensor &aot, int accumBatchs, int accumAlign)
{
    auto streamPtr = resources.getDefaultStream();
    auto stream = streamPtr->GetStream();
    auto ret = aclrtMemcpy(aot.offsetAddrs.data(), aot.offsetAddrs.getSizeInBytes(),
                           aot.offsetAddrsHost.data(), aot.offsetAddrsHost.getSizeInBytes(), ACL_MEMCPY_HOST_TO_DEVICE);
    APPERR_RETURN_IF_NOT_LOG(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "Failed to copy offsetAddrs to device");
    ret = aclrtMemcpy(aot.opSizeAddrs.data(), aot.opSizeAddrs.getSizeInBytes(),
                      aot.opSizeAddrsHost.data(), aot.opSizeAddrsHost.getSizeInBytes(), ACL_MEMCPY_HOST_TO_DEVICE);
    APPERR_RETURN_IF_NOT_LOG(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "Failed to copy opSizeAddrs to device");
    ret = aclrtMemcpy(aot.distAddrs.data(), aot.distAddrs.getSizeInBytes(),
                      aot.distAddrsHost.data(), aot.distAddrsHost.getSizeInBytes(), ACL_MEMCPY_HOST_TO_DEVICE);
    APPERR_RETURN_IF_NOT_LOG(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "Failed to copy distAddrs to device");
    ret = aclrtMemcpy(aot.maxAddrs.data(), aot.maxAddrs.getSizeInBytes(),
                      aot.maxAddrsHost.data(), aot.maxAddrsHost.getSizeInBytes(), ACL_MEMCPY_HOST_TO_DEVICE);
    APPERR_RETURN_IF_NOT_LOG(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "Failed to copy maxAddrs to device");
    ret = aclrtMemcpy(aot.flagAddrs.data(), aot.flagAddrs.getSizeInBytes(),
                      aot.flagAddrsHost.data(), aot.flagAddrsHost.getSizeInBytes(), ACL_MEMCPY_HOST_TO_DEVICE);
    APPERR_RETURN_IF_NOT_LOG(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "Failed to copy flagAddrs to device");
    ret = aclrtMemcpy(aot.accumNum.data(), aot.accumNum.getSizeInBytes(),
                      aot.accumNumHost.data(), aot.accumNumHost.getSizeInBytes(), ACL_MEMCPY_HOST_TO_DEVICE);
    APPERR_RETURN_IF_NOT_LOG(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "Failed to copy accumTensor to device");
    ret = aclrtMemcpy(aot.queryOffset.data(), aot.queryOffset.getSizeInBytes(),
                      aot.queryOffsetHost.data(), aot.queryOffsetHost.getSizeInBytes(), ACL_MEMCPY_HOST_TO_DEVICE);
    APPERR_RETURN_IF_NOT_LOG(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "Failed to copy queryOffset to device");
    AscendTensor<uint8_t, DIMS_1> base(pListBase, { SEARCH_LIST_SIZE * dims });
    for (int i = 0; i < accumBatchs; i++) {
        AscendTensor<int64_t, DIMS_1> offsetAddrList(aot.offsetAddrs[i][0].data(), {accumAlign});
        AscendTensor<int64_t, DIMS_1> sizeAddrList(aot.opSizeAddrs[i][0].data(), {accumAlign});
        AscendTensor<int64_t, DIMS_1> distAddrList(aot.distAddrs[i][0].data(), {accumAlign});
        AscendTensor<int64_t, DIMS_1> maxAddrList(aot.maxAddrs[i][0].data(), {accumAlign});
        AscendTensor<int64_t, DIMS_1> flagAddrList(aot.flagAddrs[i][0].data(), {accumAlign});
        AscendTensor<uint32_t, DIMS_1> accumList(aot.accumNum[i][0].data(), {SIZE_ALIGN});
        AscendTensor<uint64_t, DIMS_1> queryOffset(aot.queryOffset[i][0].data(), {accumAlign});
        AscendTensor<float16_t, DIMS_2> query(dot.queries.data(), {accumAlign, dims});
        std::vector<const AscendTensorBase *> input{&query, &base, &offsetAddrList,
                                                    &vDM, &sizeAddrList, &accumList, &queryOffset};
        std::vector<const AscendTensorBase *> output{&distAddrList, &maxAddrList, &flagAddrList};
        runIvfsqAccumDistOp(input, output);
    }
    ret = synchronizeStream(stream);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR,
                             "synchronizeStream default stream: %i\n", ret);
    return APP_ERR_OK;
}

APP_ERROR IndexIVFSQIPAicpu::callAccumulateDist(int tiles, int maxScanSeg, DistOpTensor &distOpTensor)
{
    int n = distOpTensor.queries.getSize(0);
    auto &mem = resources.getMemoryManager();
    auto streamPtr = resources.getDefaultStream();
    auto stream = streamPtr->GetStream();
    int accumBatchs = utils::divUp(n * tiles * maxScanSeg, static_cast<int>(actualAccumNum));
    AscendTensor<int64_t, DIMS_2> offsetAddrs(mem, {accumBatchs, accumAlign}, stream);
    std::vector<int64_t> offsetAddrsVec(accumBatchs * accumAlign);
    AscendTensor<int64_t, DIMS_2> offsetAddrsHost(offsetAddrsVec.data(), {accumBatchs, accumAlign});
    AscendTensor<int64_t, DIMS_2> opSizeAddrs(mem, {accumBatchs, accumAlign}, stream);
    std::vector<int64_t> opSizeAddrsVec(accumBatchs * accumAlign);
    AscendTensor<int64_t, DIMS_2> opSizeAddrsHost(opSizeAddrsVec.data(), {accumBatchs, accumAlign});
    AscendTensor<int64_t, DIMS_2> distAddrs(mem, {accumBatchs, accumAlign}, stream);
    std::vector<int64_t> distAddrsVec(accumBatchs * accumAlign);
    AscendTensor<int64_t, DIMS_2> distAddrsHost(distAddrsVec.data(), {accumBatchs, accumAlign});
    AscendTensor<int64_t, DIMS_2> maxAddrs(mem, {accumBatchs, accumAlign}, stream);
    std::vector<int64_t> maxAddrsVec(accumBatchs * accumAlign);
    AscendTensor<int64_t, DIMS_2> maxAddrsHost(maxAddrsVec.data(), {accumBatchs, accumAlign});
    AscendTensor<int64_t, DIMS_2> flagAddrs(mem, {accumBatchs, accumAlign}, stream);
    std::vector<int64_t> flagAddrsVec(accumBatchs * accumAlign);
    AscendTensor<int64_t, DIMS_2> flagAddrsHost(flagAddrsVec.data(), {accumBatchs, accumAlign});
    AscendTensor<uint32_t, DIMS_2> accumNum(mem, {accumBatchs, SIZE_ALIGN}, stream);
    std::vector<uint32_t> accumNumVec(accumBatchs * SIZE_ALIGN);
    AscendTensor<uint32_t, DIMS_2> accumNumHost(accumNumVec.data(), {accumBatchs, SIZE_ALIGN});
    AscendTensor<uint64_t, DIMS_2> queryOffset(mem, {accumBatchs, accumAlign}, stream);
    std::vector<uint64_t> queryOffsetVec(accumBatchs * accumAlign);
    AscendTensor<uint64_t, DIMS_2> queryOffsetHost(queryOffsetVec.data(), {accumBatchs, accumAlign});
    AccumOpTensor accumOpTensor(offsetAddrs, offsetAddrsHost, opSizeAddrs, opSizeAddrsHost, distAddrs, distAddrsHost,
                                maxAddrs, maxAddrsHost, flagAddrs, flagAddrsHost, accumNum, accumNumHost, queryOffset,
                                queryOffsetHost);

    auto ret = fillAccumAddr(actualAccumNum, distOpTensor, accumOpTensor);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "fillAccumAddr error: %i\n", ret);

    ret = runAccumOp(distOpTensor, accumOpTensor, accumBatchs, accumAlign);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "runAccumOp error: %i\n", ret);
    return APP_ERR_OK;
}

APP_ERROR IndexIVFSQIPAicpu::callSqDistanceOp(DistOpTensor &distOpTensor)
{
    auto streamPtr = resources.getDefaultStream();
    auto stream = streamPtr->GetStream();
    int n = distOpTensor.queries.getSize(0);
    auto nprobeAlign = utils::roundUp(nprobe, handleBatch);
    AscendTensor<uint8_t, DIMS_1> base(pListBase, { SEARCH_LIST_SIZE * dims });
    for (int nIdx = 0; nIdx < n; ++nIdx) {
        for (int probeIdx = 0; probeIdx < nprobeAlign; probeIdx += handleBatch) {
            int tIdx = probeIdx / handleBatch;
            int segs = distOpTensor.segsHost[nIdx][tIdx].value();
            for (int m = 0; m < segs; ++m) {
                // code is stored in Zz format, Zz format is 4 dims shaped. z's shape is
                // 16 X 16(AiCore cube's matrix operation's size). Z's shape is (distsLen / 16) X (dims / 16).
                // Zz's 4 dims shape is ((distsLen / 16), (dims / 16), 16, 16)
                AscendTensor<float16_t, DIMS_2> query(distOpTensor.queries[nIdx].data(), { 1, dims });
                
                AscendTensor<uint64_t, DIMS_1> offset(distOpTensor.listOffset[nIdx][tIdx][m * SIZE_ALIGN].data(),
                    { SIZE_ALIGN });
                AscendTensor<uint32_t, DIMS_1> actualSize(distOpTensor.opSize[nIdx][tIdx][m * SIZE_ALIGN].data(),
                    { SIZE_ALIGN });
                AscendTensor<float16_t, DIMS_2> result(
                    distOpTensor.distResult[nIdx][tIdx][m * handleBatch * distsLen].data(),
                    { handleBatch, distsLen });
                AscendTensor<float16_t, DIMS_2> maxResult(
                    distOpTensor.maxDistResult[nIdx][tIdx][m * handleBatch * maxesLen].data(),
                    { handleBatch, maxesLen });
                AscendTensor<uint16_t, DIMS_2> flag(distOpTensor.opFlag[nIdx][tIdx][m * CORE_NUM * FLAG_ALIGN].data(),
                    { CORE_NUM, FLAG_ALIGN });
                std::vector<const AscendTensorBase *> input {&query, &base, &offset, &vDM, &actualSize};
                std::vector<const AscendTensorBase *> output {&result, &maxResult, &flag};
                int batch = query.getSize(0);
                runSqDistOperator310P(batch, input, output, stream);
            }
        }
    }
    auto ret = synchronizeStream(stream);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS,
        APP_ERR_INNER_ERROR, "synchronizeStream default stream: %i\n", ret);
    return APP_ERR_OK;
}

bool IndexIVFSQIPAicpu::useAccumlateOp(int n, int maxScanSeg)
{
    // dim 在512以上时，ubffer大小限制了Aicore单次循环计算底库的数目低于burst大小，不走accumulate算子
    if (dims >= 512 || accumNum <= 0) {
        return false;
    }
    int tiles = utils::divUp(nprobe, handleBatch);
    // when dim <= 64, Aicore resource limit is 3 * SEARCH_LIST_SIZE
    int opSourceLimit = (dims <= 64) ? (3 * SEARCH_LIST_SIZE) : SEARCH_LIST_SIZE;
    accumNum = utils::divDown(opSourceLimit, maxListLength);
    int validMax = ivfsqAccumBatchs.front();
    for (auto item : ivfsqAccumBatchs) {
        if ((item < accumNum) && item > validMax) {
            validMax = item;
        }
    }
    accumAlign = validMax;
    actualAccumNum = std::min(accumNum, accumAlign);
    accumNum = accumAlign;
    int callOpTimes =  n * tiles * maxScanSeg / static_cast<int>(actualAccumNum);
    // nprobe 小于32时部分场景会运行出错，减少算子调用次数低于16倍时无明显收益，均不走accumulate算子方案
    return (nprobe >= 32) && (callOpTimes > 16); // nprobe over 32, over 16
}

APP_ERROR IndexIVFSQIPAicpu::setAicpuTopkAttr(AscendTensor<int64_t, DIMS_1> &attrs,
                                              int k, int tiles, int maxScanSeg) const
{
    // inputs of aicpu topk op
    std::vector<int64_t> attrsVec(aicpu::TOPK_IVF_ATTR_IDX_COUNT);
    attrsVec[aicpu::TOPK_IVF_ATTR_ASC_IDX] = 0;
    attrsVec[aicpu::TOPK_IVF_ATTR_K_IDX] = k;
    attrsVec[aicpu::TOPK_IVF_ATTR_BURST_LEN_IDX] = burstLen;
    attrsVec[aicpu::TOPK_IVF_ATTR_BLOCK_NUM_IDX] = tiles * maxScanSeg;
    attrsVec[aicpu::TOPK_IVF_ATTR_FLAG_NUM_IDX] = CORE_NUM;
    auto ret = aclrtMemcpy(attrs.data(), attrs.getSizeInBytes(),
                           attrsVec.data(), attrsVec.size() * sizeof(int64_t), ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != ACL_SUCCESS) {
        return APP_ERR_INNER_ERROR;
    }
    return APP_ERR_OK;
}

APP_ERROR IndexIVFSQIPAicpu::searchImplL2For310P(AscendTensor<float16_t, DIMS_2> &queries,
                                                 AscendTensor<int64_t, DIMS_2> &l1TopNprobeIndicesHost,
                                                 AscendTensor<float16_t, DIMS_2> &outDists,
                                                 AscendTensor<int64_t, DIMS_2> &outIndices)
{
    auto streamPtr = resources.getDefaultStream();
    auto stream = streamPtr->GetStream();
    auto streamAicpuPtr = resources.getAlternateStreams()[0];
    auto streamAicpu = streamAicpuPtr->GetStream();
    auto &mem = resources.getMemoryManager();
    int n = queries.getSize(0);
    int maxScanSeg = utils::divUp(maxListLength, SEARCH_LIST_SIZE);
    int tiles = utils::divUp(nprobe, handleBatch);

    // inputs of dist op
    std::vector<int> listIdVec(n * tiles * maxScanSeg * handleBatch);
    AscendTensor<int, DIMS_3> listIdHost(listIdVec.data(), { n, tiles, (maxScanSeg * handleBatch) });
    AscendTensor<uint64_t, DIMS_3> listOffset(mem, { n, tiles, (maxScanSeg * SIZE_ALIGN) }, stream);
    std::vector<uint64_t> listOffsetVec(listOffset.numElements());
    AscendTensor<uint64_t, DIMS_3> listOffsetHost(listOffsetVec.data(), { n, tiles, (maxScanSeg * SIZE_ALIGN) });

    // outputs of dist op, also inputs of aicpu topk op
    AscendTensor<float16_t, DIMS_3> distResult(mem, { n, tiles, (maxScanSeg * handleBatch * distsLen) }, stream);
    AscendTensor<float16_t, DIMS_3> maxDistResult(mem, { n, tiles, (maxScanSeg * handleBatch * maxesLen) }, stream);
    // tensor for telling operator how many code to calculate
    AscendTensor<uint32_t, DIMS_3> opSize(mem, { n, tiles, (maxScanSeg * SIZE_ALIGN) }, stream);
    std::vector<uint32_t> opSizeVec(opSize.numElements());
    AscendTensor<uint32_t, DIMS_3> opSizeHost(opSizeVec.data(), { n, tiles, (maxScanSeg * SIZE_ALIGN) });
    // tensor for operator flags
    AscendTensor<uint16_t, DIMS_3> opFlag(mem, { n, tiles, (maxScanSeg * CORE_NUM * FLAG_ALIGN) }, stream);
    (void)opFlag.zero();
    // pointer to id of each batch
    AscendTensor<int64_t, DIMS_3> ids(mem, { n, tiles, (maxScanSeg * handleBatch) }, stream);
    std::vector<int64_t> idsVec(ids.numElements());
    AscendTensor<int64_t, DIMS_3> idsHost(idsVec.data(), { n, tiles, (maxScanSeg * handleBatch) });
    AscendTensor<int64_t, DIMS_1> attrs(mem, { aicpu::TOPK_IVF_ATTR_IDX_COUNT }, stream);
    auto ret = setAicpuTopkAttr(attrs, outDists.getSize(1), tiles, maxScanSeg);
    APPERR_RETURN_IF_NOT_LOG(ret == APP_ERR_OK, APP_ERR_INNER_ERROR, "Failed to copy attr to device");
    auto nprobeAlign = utils::roundUp(nprobe, handleBatch);
    std::vector<int> segsVec(n * utils::divUp(nprobeAlign, handleBatch));
    AscendTensor<int, DIMS_2> segsHost(segsVec.data(), { n, utils::divUp(nprobeAlign, handleBatch) });
    for (int nIdx = 0; nIdx < n; ++nIdx) {
        for (int probeIdx = 0; probeIdx < nprobeAlign; probeIdx += handleBatch) {
            int tIdx = probeIdx / handleBatch;

            AscendTensor<int, DIMS_1> listId(listIdHost[nIdx][tIdx].data(), { handleBatch });
            for (int i = 0; i < handleBatch; ++i) {
                listId[i].value(l1TopNprobeIndicesHost[nIdx][std::min(probeIdx + i, nprobe - 1)].value());
            }

            // seperator list's code for several segs to run sq distance,
            // because of fixed-shape limitation of aicore's operator.
            auto maxSize = deviceListIndices[listId[0].value()]->size();
            for (int i = 0; i < handleBatch; ++i) {
                maxSize = std::max(maxSize, deviceListIndices[listId[i].value()]->size());
            }
            int segs = (int)(utils::divUp(maxSize, (size_t)distsLen));
            segsHost[nIdx][tIdx].value(segs);

            for (int m = 0; m < segs; ++m) {
                AscendTensor<uint32_t, DIMS_1> actualSize(opSizeHost[nIdx][tIdx][m * SIZE_ALIGN].data(),
                    { SIZE_ALIGN });
                AscendTensor<uint64_t, DIMS_1> offset(listOffsetHost[nIdx][tIdx][m * SIZE_ALIGN].data(),
                    { SIZE_ALIGN });
                AscendTensor<int64_t, DIMS_1> id(idsHost[nIdx][tIdx][m * handleBatch].data(), { handleBatch });
                for (int i = 0; i < handleBatch; ++i) {
                    int listSize = static_cast<int>(deviceListIndices[listId[i].value()]->size());
                    uint32_t size = static_cast<uint32_t>(std::max(std::min(distsLen, listSize - m * distsLen), 0));
                    actualSize[i].value((probeIdx + i) >= nprobe ? 0 : size);
                    offset[i].value(deviceListData[listId[i].value()]->data() - pListBase + m * distsLen * dims);
                    id[i].value(reinterpret_cast<int64_t>(deviceListIndices[listId[i].value()]->data() + m * distsLen));
                }
            }
        }
    }
    ret = aclrtMemcpy(opSize.data(), opSize.getSizeInBytes(),
                      opSizeHost.data(), opSizeHost.getSizeInBytes(), ACL_MEMCPY_HOST_TO_DEVICE);
    APPERR_RETURN_IF_NOT_LOG(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "Failed to copy op size to device");
    ret = aclrtMemcpy(listOffset.data(), listOffset.getSizeInBytes(),
                      listOffsetHost.data(), listOffsetHost.getSizeInBytes(), ACL_MEMCPY_HOST_TO_DEVICE);
    APPERR_RETURN_IF_NOT_LOG(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "Failed to copy list offset to device");
    ret = aclrtMemcpy(ids.data(), ids.getSizeInBytes(),
                      idsHost.data(), idsHost.getSizeInBytes(), ACL_MEMCPY_HOST_TO_DEVICE);
    APPERR_RETURN_IF_NOT_LOG(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "Failed to copy ids to device");
    runL2TopkOp(distResult, maxDistResult, ids, opSize, opFlag, attrs, outDists, outIndices, streamAicpu);
    DistOpTensor distOpTensor(queries, listOffset, opSize, distResult, maxDistResult, opFlag, segsHost);
    if (useAccumlateOp(n, maxScanSeg)) {
        ret = callAccumulateDist(tiles, maxScanSeg, distOpTensor);
        APPERR_RETURN_IF_NOT_FMT(ret == APP_ERR_OK, ret, "callAccumulateDist error: %i\n", ret);
    } else {
        ret = callSqDistanceOp(distOpTensor);
        APPERR_RETURN_IF_NOT_FMT(ret == APP_ERR_OK, ret, "callSqDistanceOp error: %i\n", ret);
    }
    ret = synchronizeStream(streamAicpu);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR,
                             "synchronizeStream aicpu stream failed: %i\n", ret);
    return APP_ERR_OK;
}

APP_ERROR IndexIVFSQIPAicpu::searchImplL2For310(AscendTensor<float16_t, DIMS_2> &queries,
                                                AscendTensor<int64_t, DIMS_2> &l1TopNprobeIndicesHost,
                                                AscendTensor<float16_t, DIMS_2> &outDists,
                                                AscendTensor<int64_t, DIMS_2> &outIndices)
{
    auto streamPtr = resources.getDefaultStream();
    auto stream = streamPtr->GetStream();
    auto streamAicpuPtr = resources.getAlternateStreams()[0];
    auto streamAicpu = streamAicpuPtr->GetStream();
    auto &mem = resources.getMemoryManager();
    int n = queries.getSize(0);
    int maxScanSeg = utils::divUp(maxListLength, SEARCH_LIST_SIZE);
    int tiles = utils::divUp(nprobe, handleBatch);

    // inputs of dist op
    std::vector<int> listIdVec(n * tiles * maxScanSeg * handleBatch);
    AscendTensor<int, DIMS_3> listIdHost(listIdVec.data(), { n, tiles, (maxScanSeg * handleBatch) });

    // outputs of dist op, also inputs of aicpu topk op
    AscendTensor<float16_t, DIMS_3> distResult(mem, { n, tiles, (maxScanSeg * handleBatch * distsLen) }, stream);
    AscendTensor<float16_t, DIMS_3> maxDistResult(mem, { n, tiles, (maxScanSeg * handleBatch * maxesLen) }, stream);
    // tensor for telling operator how many code to calculate
    AscendTensor<uint32_t, DIMS_3> opSize(mem, { n, tiles, (maxScanSeg * SIZE_ALIGN) }, stream);
    std::vector<uint32_t> opSizeVec(opSize.numElements());
    AscendTensor<uint32_t, DIMS_3> opSizeHost(opSizeVec.data(), { n, tiles, (maxScanSeg * SIZE_ALIGN) });
    // tensor for operator flags
    AscendTensor<uint16_t, DIMS_3> opFlag(mem, { n, tiles, (maxScanSeg * CORE_NUM * FLAG_ALIGN) }, stream);
    (void)opFlag.zero();
    // pointer to id of each batch
    AscendTensor<int64_t, DIMS_3> ids(mem, { n, tiles, (maxScanSeg * handleBatch) }, stream);
    std::vector<int64_t> idsVec(ids.numElements());
    AscendTensor<int64_t, DIMS_3> idsHost(idsVec.data(), { n, tiles, (maxScanSeg * handleBatch) });
    AscendTensor<int64_t, DIMS_1> attrs(mem, { aicpu::TOPK_IVF_ATTR_IDX_COUNT }, stream);
    auto ret = setAicpuTopkAttr(attrs, outDists.getSize(1), tiles, maxScanSeg);
    APPERR_RETURN_IF_NOT_LOG(ret == APP_ERR_OK, APP_ERR_INNER_ERROR, "Failed to copy attr to device");

    std::vector<int> segsVec(n * utils::divUp(nprobe, handleBatch));
    AscendTensor<int, DIMS_2> segsHost(segsVec.data(), { n, utils::divUp(nprobe, handleBatch) });

    for (int nIdx = 0; nIdx < n; ++nIdx) {
        for (int probeIdx = 0; probeIdx < nprobe; probeIdx += handleBatch) {
            int tIdx = probeIdx / handleBatch;

            AscendTensor<int, DIMS_1> listId(listIdHost[nIdx][tIdx].data(), { handleBatch });
            for (int i = 0; i < handleBatch; ++i) {
                listId[i].value(l1TopNprobeIndicesHost[nIdx][std::min(probeIdx + i, nprobe - 1)].value());
            }

            // seperator list's code for several segs to run sq distance,
            // because of fixed-shape limitation of aicore's operator.
            int maxLen = (int)(std::max({ deviceListIndices[listId[0].value()]->size(),             // code list 0
                                    deviceListIndices[listId[1].value()]->size(),             // code list 1
                                    deviceListIndices[listId[2].value()]->size(),             // code list 2
                                    deviceListIndices[listId[3].value()]->size() }));          // code list 3
            int segs = utils::divUp(maxLen, distsLen);
            segsHost[nIdx][tIdx].value(segs);

            for (int m = 0; m < segs; ++m) {
                int offset = m * distsLen;
                AscendTensor<uint32_t, DIMS_1> actualSize(opSizeHost[nIdx][tIdx][m * SIZE_ALIGN].data(),
                    { SIZE_ALIGN });
                AscendTensor<int64_t, DIMS_1> id(idsHost[nIdx][tIdx][m * handleBatch].data(),
                    { handleBatch });
                for (int i = 0; i < handleBatch; ++i) {
                    uint32_t size = static_cast<uint32_t>(
                        std::max(std::min(distsLen,
                                          static_cast<int>(deviceListIndices[listId[i].value()]->size()) - offset),
                                 0));
                    actualSize[i].value((probeIdx + i) >= nprobe ? 0 : size);
                    id[i].value(reinterpret_cast<int64_t>(deviceListIndices[listId[i].value()]->data() + offset));
                }
            }
        }
    }

    ret = aclrtMemcpy(opSize.data(), opSize.getSizeInBytes(),
                      opSizeHost.data(), opSizeHost.getSizeInBytes(), ACL_MEMCPY_HOST_TO_DEVICE);
    APPERR_RETURN_IF_NOT_LOG(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "Failed to copy op size to device");
    ret = aclrtMemcpy(ids.data(), ids.getSizeInBytes(),
                      idsHost.data(), idsHost.getSizeInBytes(), ACL_MEMCPY_HOST_TO_DEVICE);
    APPERR_RETURN_IF_NOT_LOG(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "Failed to copy ids to device");

    runL2TopkOp(distResult, maxDistResult, ids, opSize, opFlag, attrs, outDists, outIndices, streamAicpu);

    for (int nIdx = 0; nIdx < n; ++nIdx) {
        for (int probeIdx = 0; probeIdx < nprobe; probeIdx += handleBatch) {
            int tIdx = probeIdx / handleBatch;
            int segs = segsHost[nIdx][tIdx].value();

            AscendTensor<int, DIMS_1> listId(listIdHost[nIdx][tIdx].data(), { handleBatch });
            for (int m = 0; m < segs; ++m) {
                int offset = m * distsLen;
                int maxOffset = m * maxesLen;

                // code is stored in Zz format, Zz format is 4 dims shaped. z's shape is
                // 16 X 16(AiCore cube's matrix operation's size). Z's shape is (distsLen / 16) X (dims / 16).
                // Zz's 4 dims shape is ((distsLen / 16), (dims / 16), 16, 16)
                AscendTensor<float16_t, DIMS_2> query(queries[nIdx].data(), { 1, dims });
                AscendTensor<uint8_t, DIMS_4> code0(
                    static_cast<uint8_t *>(deviceListData[listId[0].value()]->data()) + offset * dims,  // code data 0
                    { SEARCH_SHAPED_SIZE, dims / CUBE_ALIGN, CUBE_ALIGN, CUBE_ALIGN });
                AscendTensor<uint8_t, DIMS_4> code1(
                    static_cast<uint8_t *>(deviceListData[listId[1].value()]->data()) + offset * dims,  // code data 1
                    { SEARCH_SHAPED_SIZE, dims / CUBE_ALIGN, CUBE_ALIGN, CUBE_ALIGN });
                AscendTensor<uint8_t, DIMS_4> code2(
                    static_cast<uint8_t *>(deviceListData[listId[2].value()]->data()) + offset * dims,  // code data 2
                    { SEARCH_SHAPED_SIZE, dims / CUBE_ALIGN, CUBE_ALIGN, CUBE_ALIGN });
                AscendTensor<uint8_t, DIMS_4> code3(
                    static_cast<uint8_t *>(deviceListData[listId[3].value()]->data()) + offset * dims,  // code data 3
                    { SEARCH_SHAPED_SIZE, dims / CUBE_ALIGN, CUBE_ALIGN, CUBE_ALIGN });
                AscendTensor<uint32_t, DIMS_1> actualSize(opSize[nIdx][tIdx][m * SIZE_ALIGN].data(),
                    { SIZE_ALIGN });
                AscendTensor<float16_t, DIMS_1> result(distResult[nIdx][tIdx][handleBatch * offset].data(),
                    { handleBatch * distsLen });
                AscendTensor<float16_t, DIMS_1> maxResult(maxDistResult[nIdx][tIdx][handleBatch * maxOffset].data(),
                    { handleBatch * maxesLen });
                AscendTensor<uint16_t, DIMS_2> flag(opFlag[nIdx][tIdx][m * CORE_NUM * FLAG_ALIGN].data(),
                    { CORE_NUM, FLAG_ALIGN });
                std::vector<const AscendTensorBase *> input {&query, &code0, &code1, &code2, &code3, &vDM, &actualSize};
                std::vector<const AscendTensorBase *> output {&result, &maxResult, &flag};
                int batch = query.getSize(0);
                runSqDistOperator310(batch, input, output, stream);
            }
        }
    }

    ret = synchronizeStream(stream);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR,
                             "synchronizeStream default stream: %i\n", ret);

    ret = synchronizeStream(streamAicpu);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR,
                             "synchronizeStream aicpu stream failed: %i\n", ret);

    return APP_ERR_OK;
}

APP_ERROR IndexIVFSQIPAicpu::searchImplL1(AscendTensor<float16_t, DIMS_2> &queries,
                                          AscendTensor<int64_t, DIMS_2> &l1TopNprobeIndicesHost)
{
    auto &mem = resources.getMemoryManager();
    auto streamPtr = resources.getDefaultStream();
    auto stream = streamPtr->GetStream();
    auto streamAicpuPtr = resources.getAlternateStreams()[0];
    auto streamAicpu = streamAicpuPtr->GetStream();
    // L1 dist op output: dists / vmdists / opFlag
    int n = queries.getSize(0);
    AscendTensor<float16_t, DIMS_2> dists(mem, {n, numLists}, stream);
    // the result constain min value and index, the multi 2
    int minDistSize = std::max(numLists / BURST_LEN * 2, MIN_EXTREME_SIZE);
    AscendTensor<float16_t, DIMS_2> vmdists(mem, {n, minDistSize}, stream);
    AscendTensor<uint32_t, DIMS_2> opSize(mem, {CORE_NUM, SIZE_ALIGN}, stream); // op size, no use
    opSize[0][0] = numLists;
    AscendTensor<uint16_t, DIMS_2> opFlag(mem, {CORE_NUM, FLAG_SIZE}, stream);
    opFlag.zero();

    AscendTensor<int64_t, DIMS_1> attrsInput(mem, { aicpu::TOPK_FLAT_ATTR_IDX_COUNT }, stream);
    std::vector<int64_t> attrs(aicpu::TOPK_FLAT_ATTR_IDX_COUNT);
    attrs[aicpu::TOPK_FLAT_ATTR_ASC_IDX] = 1;
    attrs[aicpu::TOPK_FLAT_ATTR_K_IDX] = nprobe;
    attrs[aicpu::TOPK_FLAT_ATTR_BURST_LEN_IDX] = BURST_LEN;
    attrs[aicpu::TOPK_FLAT_ATTR_BLOCK_NUM_IDX] = 1;
    attrs[aicpu::TOPK_FLAT_ATTR_PAGE_IDX] = 0;
    attrs[aicpu::TOPK_FLAT_ATTR_PAGE_NUM_IDX] = 1;
    attrs[aicpu::TOPK_FLAT_ATTR_PAGE_SIZE_IDX] = 0;
    attrs[aicpu::TOPK_FLAT_ATTR_QUICK_HEAP] = 0;
    attrs[aicpu::TOPK_FLAT_ATTR_BLOCK_SIZE] = numLists;

    auto ret = aclrtMemcpy(attrsInput.data(), attrsInput.getSizeInBytes(),
                           attrs.data(), attrs.size() * sizeof(int64_t), ACL_MEMCPY_HOST_TO_DEVICE);
    APPERR_RETURN_IF_NOT_LOG(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "Failed to copy attr to device");

    AscendTensor<float16_t, DIMS_2> l1TopNprobeDists(mem, {n, nprobe}, stream);
    AscendTensor<int64_t, DIMS_2> l1TopNprobeIndices(mem, {n, nprobe}, stream);

    runL1TopkOp(dists, vmdists, opSize, opFlag, attrsInput, l1TopNprobeDists, l1TopNprobeIndices, streamAicpu);
    // run l1 distance calculation
    runL1DistOp(queries, coarseCentroidsShaped, normCoarseCentroids, dists, vmdists, opFlag, stream);

    ret = synchronizeStream(stream);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR,
        "synchronizeStream default stream: %i\n", ret);

    ret = synchronizeStream(streamAicpu);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR,
        "synchronizeStream aicpu stream failed: %i\n", ret);

    ret = aclrtMemcpy(l1TopNprobeIndicesHost.data(), l1TopNprobeIndicesHost.getSizeInBytes(),
                      l1TopNprobeIndices.data(), l1TopNprobeIndices.getSizeInBytes(),
                      ACL_MEMCPY_DEVICE_TO_HOST);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "Mem operator error %d", (int)ret);

    return APP_ERR_OK;
}

APP_ERROR IndexIVFSQIPAicpu::searchImpl(int n, const float16_t *x, int k, float16_t *dists, idx_t *labels)
{
    ASCEND_THROW_IF_NOT_MSG(this->ntotal != 0, "feature vector's number is 0, please check if add before search");
    APP_ERROR ret = APP_ERR_OK;
    APP_ERROR retAcl = APP_ERR_OK;
    auto streamPtr = resources.getDefaultStream();
    auto stream = streamPtr->GetStream();
    auto &mem = resources.getMemoryManager();

    AscendTensor<float16_t, DIMS_2> queries(const_cast<float16_t *>(x), { n, dims });

    // L1 search, to find nprobe IVF list
    // L1 topk op output: top nprobe dist / top nprobe label
    std::vector<int64_t> l1TopNprobeIndicesVec(n * nprobe, 0);
    AscendTensor<int64_t, DIMS_2> l1TopNprobeIndicesHost(l1TopNprobeIndicesVec.data(), { n, nprobe });

    ret = searchImplL1(queries, l1TopNprobeIndicesHost);
    APPERR_RETURN_IF(ret, ret);

    AscendTensor<float16_t, DIMS_2> outDists(mem, { n, k }, stream);
    AscendTensor<int64_t, DIMS_2> outIndices(mem, { n, k }, stream);

    // L2 search, search codes in nprobe IVF list to find topk results
    ret = searchImplL2(queries, l1TopNprobeIndicesHost, outDists, outIndices);
    APPERR_RETURN_IF(ret, ret);

    retAcl = aclrtMemcpy(dists, outDists.getSizeInBytes(),
                         outDists.data(), outDists.getSizeInBytes(), ACL_MEMCPY_DEVICE_TO_HOST);
    APPERR_RETURN_IF_NOT_LOG(retAcl == ACL_SUCCESS, APP_ERR_INNER_ERROR, "Failed to copy dists back to host");

    retAcl = aclrtMemcpy(labels, outIndices.getSizeInBytes(),
                         outIndices.data(), outIndices.getSizeInBytes(), ACL_MEMCPY_DEVICE_TO_HOST);
    APPERR_RETURN_IF_NOT_LOG(retAcl == ACL_SUCCESS, APP_ERR_INNER_ERROR, "Failed to copy labels back to host");

    return ret;
}


APP_ERROR IndexIVFSQIPAicpu::resetL2TopkOp()
{
    auto topkCompOpReset = [&](std::unique_ptr<AscendOperator> &op, int64_t batch) {
        AscendOpDesc desc("TopkIvf");
        std::vector<int64_t> shape0 { batch, 0, handleBatch, distsLen };
        std::vector<int64_t> shape1 { batch, 0, handleBatch, maxesLen };
        std::vector<int64_t> shape2 { batch, 0, handleBatch };
        std::vector<int64_t> shape3 { batch, 0, SIZE_ALIGN };
        std::vector<int64_t> shape4 { batch, 0, CORE_NUM, FLAG_ALIGN };
        std::vector<int64_t> shape5 { aicpu::TOPK_IVF_ATTR_IDX_COUNT };

        std::vector<int64_t> shape6 { batch, 0 };

        desc.addInputTensorDesc(ACL_FLOAT16, shape0.size(), shape0.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_FLOAT16, shape1.size(), shape1.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_INT64, shape2.size(), shape2.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_UINT32, shape3.size(), shape3.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_UINT16, shape4.size(), shape4.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_INT64, shape5.size(), shape5.data(), ACL_FORMAT_ND);

        desc.addOutputTensorDesc(ACL_FLOAT16, shape6.size(), shape5.data(), ACL_FORMAT_ND);
        desc.addOutputTensorDesc(ACL_INT64, shape6.size(), shape5.data(), ACL_FORMAT_ND);

        op.reset();
        op = CREATE_UNIQUE_PTR(AscendOperator, desc);
        return op->init();
    };

    for (auto batch : searchBatchSizes) {
        l2TopkOps[batch] = std::unique_ptr<AscendOperator>(nullptr);
        APPERR_RETURN_IF_NOT_LOG(topkCompOpReset(l2TopkOps[batch], batch),
                                 APP_ERR_ACL_OP_LOAD_MODEL_FAILED, "l2 topk op init failed");
    }

    return APP_ERR_OK;
}

APP_ERROR IndexIVFSQIPAicpu::resetIvfsqAccumDistOp310P()
{
    ivfsqAccumBatchs = {4, 8, 16, 32};
    if (dims >= 512) { // accumlate op is not suport dim 512
        return APP_ERR_OK;
    }
    auto disOpsRest = [&](std::unique_ptr<AscendOperator> &op, int batch) {
        op.reset();
        AscendOpDesc desc("DistanceIVFSQ8IP8Accum");
        int batchAlign = utils::roundUp(batch, 4);
        std::vector<int64_t> queryShape({ 1, dims });
        std::vector<int64_t> baseShape({ SEARCH_LIST_SIZE * dims});
        std::vector<int64_t> offsetShape({ batchAlign });
        std::vector<int64_t> vdmShape({ 2, dims });
        std::vector<int64_t> sizeShape({ batchAlign });
        std::vector<int64_t> accumNum({ 8 });
        std::vector<int64_t> resultAddrShape({batchAlign});
        std::vector<int64_t> maxResultAddrShape({batchAlign});
        std::vector<int64_t> flagAddrShape({batchAlign});
        std::vector<int64_t> queryOffsetShape({batchAlign});
        desc.addInputTensorDesc(ACL_FLOAT16, queryShape.size(), queryShape.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_UINT8, baseShape.size(), baseShape.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_INT64, offsetShape.size(), offsetShape.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_FLOAT16, vdmShape.size(), vdmShape.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_INT64, sizeShape.size(), sizeShape.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_UINT32, accumNum.size(), accumNum.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_UINT64, queryOffsetShape.size(), queryOffsetShape.data(), ACL_FORMAT_ND);

        desc.addOutputTensorDesc(ACL_INT64, resultAddrShape.size(), resultAddrShape.data(), ACL_FORMAT_ND);
        desc.addOutputTensorDesc(ACL_INT64, maxResultAddrShape.size(), maxResultAddrShape.data(), ACL_FORMAT_ND);
        desc.addOutputTensorDesc(ACL_INT64, flagAddrShape.size(), flagAddrShape.data(), ACL_FORMAT_ND);
        op = CREATE_UNIQUE_PTR(AscendOperator, desc);
        return op->init();
    };

    for (auto batch : ivfsqAccumBatchs) {
        distIvfsqAccumOpsMap[batch] = std::unique_ptr<AscendOperator>(nullptr);
        APPERR_RETURN_IF_NOT_LOG(disOpsRest(distIvfsqAccumOpsMap[batch], batch),
                                 APP_ERR_ACL_OP_LOAD_MODEL_FAILED, "op init failed");
    }

    return APP_ERR_OK;
}

void IndexIVFSQIPAicpu::runIvfsqAccumDistOp(std::vector<const AscendTensorBase *> &input,
                                            std::vector<const AscendTensorBase *> &output)
{
    AscendOperator *op = nullptr;
    if (distIvfsqAccumOpsMap.find(accumNum) != distIvfsqAccumOpsMap.end()) {
        op = distIvfsqAccumOpsMap[accumNum].get();
    }
    ASCEND_THROW_IF_NOT(op);
    std::shared_ptr<std::vector<const aclDataBuffer *>> distIvfsqOpInput(
        new std::vector<const aclDataBuffer *>(), CommonUtils::AclInputBufferDelete);
    for (auto inputItem : input) {
        distIvfsqOpInput->emplace_back(aclCreateDataBuffer(inputItem->getVoidData(), inputItem->getSizeInBytes()));
    }

    std::shared_ptr<std::vector<aclDataBuffer *>> distIvfsqOpOutput(
        new std::vector<aclDataBuffer *>(), CommonUtils::AclOutputBufferDelete);
    for (auto outputItem : output) {
        distIvfsqOpOutput->emplace_back(aclCreateDataBuffer(outputItem->getVoidData(), outputItem->getSizeInBytes()));
    }
    auto streamPtr = resources.getDefaultStream();
    auto stream = streamPtr->GetStream();
    op->exec(*distIvfsqOpInput, *distIvfsqOpOutput, stream);
}

void IndexIVFSQIPAicpu::runL2TopkOp(AscendTensor<float16_t, DIMS_3> &dists,
                                    AscendTensor<float16_t, DIMS_3> &vmdists,
                                    AscendTensor<int64_t, DIMS_3> &ids,
                                    AscendTensor<uint32_t, DIMS_3> &sizes,
                                    AscendTensor<uint16_t, DIMS_3> &flags,
                                    AscendTensor<int64_t, DIMS_1> &attrs,
                                    AscendTensor<float16_t, DIMS_2> &outdists,
                                    AscendTensor<int64_t, DIMS_2> &outlabel,
                                    aclrtStream stream)
{
    int batch = dists.getSize(0);
    std::vector<const AscendTensorBase *> input{&dists, &vmdists, &ids, &sizes, &flags, &attrs};
    std::vector<const AscendTensorBase *> output{&outdists, &outlabel};
    IndexIVFSQ::runL2TopkOp(batch, input, output, stream);
}

APP_ERROR IndexIVFSQIPAicpu::resetSqDistOperator() const
{
    if (faiss::ascend::SocUtils::GetInstance().IsAscend310()) {
        return resetSqDistOperatorFor310();
    } else {
        return resetSqDistOperatorFor310P();
    }
}

APP_ERROR IndexIVFSQIPAicpu::resetSqDistOperatorFor310() const
{
    IndexTypeIdx indexType = IndexTypeIdx::ITI_IVFSQ8_IP4;
    std::string opTypeName = "DistanceIVFSQ8IP4";
    std::vector<int64_t> queryShape({ 1, dims });
    std::vector<int64_t> codeShape({ SEARCH_SHAPED_SIZE, dims / CUBE_ALIGN, CUBE_ALIGN, CUBE_ALIGN });
    std::vector<int64_t> vdmShape({ 2, dims });
    std::vector<int64_t> sizeShape({ SIZE_ALIGN });
    std::vector<int64_t> resultShape({ (handleBatch * SEARCH_LIST_SIZE) });
    std::vector<int64_t> maxResultShape({ (handleBatch * EXTREME_LIST_SIZE) });
    std::vector<int64_t> flagShape({ CORE_NUM, FLAG_ALIGN });

    int batch = queryShape[0];
    std::vector<std::pair<aclDataType, std::vector<int64_t>>> input {
        { ACL_FLOAT16, queryShape },
        { ACL_UINT8, codeShape },
        { ACL_UINT8, codeShape },
        { ACL_UINT8, codeShape },
        { ACL_UINT8, codeShape },
        { ACL_FLOAT16, vdmShape },
        { ACL_UINT32, sizeShape },
    };
    std::vector<std::pair<aclDataType, std::vector<int64_t>>> output {
        { ACL_FLOAT16, resultShape },
        { ACL_FLOAT16, maxResultShape },
        { ACL_UINT16, flagShape }
    };
    std::vector<int> keys({batch, dims});
    OpsMngKey opsKey(keys);
    auto ret = DistComputeOpsManager::getInstance().resetOp(opTypeName, indexType, opsKey, input, output);
    APPERR_RETURN_IF_NOT_FMT(ret == APP_ERR_OK, ret, "op init failed: %i", ret);

    return APP_ERR_OK;
}

APP_ERROR IndexIVFSQIPAicpu::resetSqDistOperatorFor310P() const
{
    IndexTypeIdx indexType = IndexTypeIdx::ITI_IVFSQ8_IP8;
    std::string opTypeName = "DistanceIVFSQ8IP8";
    std::vector<int64_t> queryShape({ 1, dims });
    std::vector<int64_t> baseShape({ SEARCH_LIST_SIZE * dims});
    std::vector<int64_t> offsetShape({ SIZE_ALIGN });
    std::vector<int64_t> vdmShape({ 2, dims });
    std::vector<int64_t> sizeShape({ SIZE_ALIGN });
    std::vector<int64_t> resultShape({ handleBatch, SEARCH_LIST_SIZE });
    std::vector<int64_t> maxResultShape(
        { handleBatch, SEARCH_LIST_SIZE / this->burstLen * 2 }); // each maxResult has 2 values
    std::vector<int64_t> flagShape({ CORE_NUM, FLAG_ALIGN });
    std::vector<std::pair<aclDataType, std::vector<int64_t>>> input {
        {ACL_FLOAT16, queryShape},
        {ACL_UINT8, baseShape},
        {ACL_UINT64, offsetShape},
        {ACL_FLOAT16, vdmShape},
        {ACL_UINT32, sizeShape}
    };
    std::vector<std::pair<aclDataType, std::vector<int64_t>>> output {
        {ACL_FLOAT16, resultShape},
        {ACL_FLOAT16, maxResultShape},
        {ACL_UINT16, flagShape}
    };
    int batch = queryShape[0];
    std::vector<int> keys({batch, dims});
    OpsMngKey opsKey(keys);
    auto ret = DistComputeOpsManager::getInstance().resetOp(opTypeName, indexType, opsKey, input, output);
    APPERR_RETURN_IF_NOT_FMT(ret == APP_ERR_OK, ret, "op init failed: %i", ret);
    return APP_ERR_OK;
}

void IndexIVFSQIPAicpu::runSqDistOperator310P(int batch,
                                              const std::vector<const AscendTensorBase *> &input,
                                              const std::vector<const AscendTensorBase *> &output,
                                              aclrtStream stream) const
{
    IndexTypeIdx indexType = IndexTypeIdx::ITI_IVFSQ8_IP8;
    std::string opTypeName = "DistanceIVFSQ8IP8";
    // async executing operator
    std::vector<int> keys({batch, dims});
    OpsMngKey opsKey(keys);
    auto ret = DistComputeOpsManager::getInstance().runOp(indexType, opsKey, input, output, stream);
    ASCEND_THROW_IF_NOT_FMT(ret == APP_ERR_OK, "run operator failed: %i\n", ret);
}

void IndexIVFSQIPAicpu::runSqDistOperator310(int batch,
                                             const std::vector<const AscendTensorBase *> &input,
                                             const std::vector<const AscendTensorBase *> &output,
                                             aclrtStream stream) const
{
    IndexTypeIdx indexType = IndexTypeIdx::ITI_IVFSQ8_IP4;
    std::string opTypeName = "DistanceIVFSQ8IP4";
    std::vector<int> keys({batch, dims});
    OpsMngKey opsKey(keys);
    auto ret = DistComputeOpsManager::getInstance().runOp(indexType, opsKey, input, output, stream);
    ASCEND_THROW_IF_NOT_FMT(ret == APP_ERR_OK, "run operator failed: %i\n", ret);
}
} // ascend
