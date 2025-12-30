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


#include "index/IndexIVFSQL2Aicpu.h"

#include <algorithm>
#include "ascenddaemon/impl/AuxIndexStructures.h"
#include "ascenddaemon/utils/Limits.h"
#include "common/utils/CommonUtils.h"
#include "ops/cpukernel/impl/utils/kernel_shared_def.h"

namespace ascend {
namespace {
const int SEARCH_LIST_SIZE = 65536; // must be CUBE_ALIGN aligned
const int SEARCH_SHAPED_SIZE = SEARCH_LIST_SIZE / CUBE_ALIGN;
}

IndexIVFSQL2Aicpu::IndexIVFSQL2Aicpu(int numList, int dim, bool encodeResidual, int nprobes, int64_t resourceSize)
    : IndexIVFSQ<float>(numList, dim, encodeResidual, nprobes, resourceSize),
      byResidual(encodeResidual)
{
    ASCEND_THROW_IF_NOT(dim % CUBE_ALIGN == 0);

    // the burstLen * dim must be less than BURST_LEN * DEFAULT_DIM
    this->burstLen = BURST_LEN;
    // if the dim bigger than 128
    if (dim > 128) {
        // the burstLen only support 16, because of the ub size of the aicore
        this->burstLen = 16;
    }
    // the output of vcmin operator contain 2 value, the min and index
    this->bursts = SEARCH_LIST_SIZE / this->burstLen * 2;
}

IndexIVFSQL2Aicpu::~IndexIVFSQL2Aicpu() {}

APP_ERROR IndexIVFSQL2Aicpu::init()
{
    APPERR_RETURN_IF_NOT_OK(resetL1DistOp(numLists));
    APPERR_RETURN_IF_NOT_OK(resetL1TopkOp());
    APPERR_RETURN_IF_NOT_OK(resetL2TopkOp());
    APPERR_RETURN_IF_NOT_OK(resetResidualOp());
    APPERR_RETURN_IF_NOT_OK(resetDistCompOperator(numLists));
    APPERR_RETURN_IF_NOT_OK(resetSqDistOperator());

    return APP_ERR_OK;
}

void IndexIVFSQL2Aicpu::setNumProbes(int nprobes)
{
    IndexIVF::setNumProbes(nprobes);
    ASCEND_THROW_IF_NOT(resetResidualOp() == APP_ERR_OK);
}

uint32_t IndexIVFSQL2Aicpu::calMaxBatch() const
{
    auto &mem = resources.getMemoryManager();
    size_t avalibleSize = mem.getSizeAvailable();
    size_t nprobeSize = static_cast<size_t>(nprobe);
    size_t maxScanSeg = static_cast<size_t>(utils::divUp(maxListLength, SEARCH_LIST_SIZE));
    size_t distSize = nprobeSize * maxScanSeg * static_cast<size_t>(SEARCH_LIST_SIZE) * sizeof(float16_t);
    size_t minsSize = nprobeSize * maxScanSeg * static_cast<size_t>(bursts) * sizeof(float16_t);
    size_t residualSize = nprobeSize * static_cast<size_t>(dims) * sizeof(float16_t);
    size_t l1TopnSize = nprobeSize * sizeof(uint64_t);
    size_t opSize = nprobeSize * maxScanSeg * static_cast<size_t>(CORE_NUM * SIZE_ALIGN) * sizeof(uint32_t);
    size_t flagSize = nprobeSize * maxScanSeg * static_cast<size_t>(FLAG_NUM * FLAG_SIZE) * sizeof(uint16_t);
    size_t idsSize = nprobeSize * maxScanSeg * sizeof(uint64_t);
    size_t totalUse = distSize + l1TopnSize + minsSize + residualSize + opSize + flagSize + idsSize;
    uint32_t batchSize = static_cast<uint32_t>(avalibleSize / totalUse);
    uint32_t batch = 1;
    uint32_t limit = static_cast<uint32_t>(searchBatchSizes.front());
    while (batch <= limit && batchSize >= batch) {
        batch = batch << 1;
    }
    return batch >> 1;
}

APP_ERROR IndexIVFSQL2Aicpu::searchImplL2(AscendTensor<float16_t, DIMS_2> &queries,
    AscendTensor<int64_t, DIMS_2> &l1TopNprobeIndices, AscendTensor<float16_t, DIMS_2> &outDists,
    AscendTensor<int64_t, DIMS_2> &outIndices)
{
    auto streamPtr = resources.getDefaultStream();
    auto stream = streamPtr->GetStream();
    auto streamAicpuPtr = resources.getAlternateStreams()[0];
    auto streamAicpu = streamAicpuPtr->GetStream();
    auto &mem = resources.getMemoryManager();
    int n = queries.getSize(0);
    int k = outDists.getSize(1);
    int maxScanSeg = utils::divUp(maxListLength, SEARCH_LIST_SIZE);

    // input of distance op
    AscendTensor<uint16_t, DIMS_4> opFlag(mem, { n, nprobe, maxScanSeg, FLAG_NUM * FLAG_SIZE }, stream);
    (void)opFlag.zero();

    // tensor for telling operator how many code to calculate

    AscendTensor<uint32_t, DIMS_4> opSize(mem, { n, nprobe, maxScanSeg, CORE_NUM * SIZE_ALIGN }, stream);
    std::vector<uint32_t> opSizeVec(opSize.numElements());
    AscendTensor<uint32_t, DIMS_4> opSizeHost(opSizeVec.data(), { n, nprobe, maxScanSeg, CORE_NUM * SIZE_ALIGN });
    // tensor for operator outputing sq distance
    AscendTensor<float16_t, DIMS_3> distResult(mem, { n, nprobe, (maxScanSeg * SEARCH_LIST_SIZE) }, stream);
    AscendTensor<float16_t, DIMS_3> minDistResult(mem, { n, nprobe, (maxScanSeg * bursts) }, stream);
    AscendTensor<float16_t, DIMS_3> residuals(mem, {n, nprobe, dims}, stream);
    AscendTensor<int64_t, DIMS_2> l1TopNprobeIndicesDevice(mem, {n, nprobe}, stream);
    if (byResidual) {
        l1TopNprobeIndicesDevice.copyFromSync(l1TopNprobeIndices, ACL_MEMCPY_HOST_TO_DEVICE);
        std::vector<const AscendTensorBase *> input {&queries, &coarseCentroids, &l1TopNprobeIndicesDevice};
        std::vector<const AscendTensorBase *> output {&residuals};
        runResidualOp(n, input, output, streamAicpu);
    }

    // pointer to id of each batch
    AscendTensor<int64_t, DIMS_3> ids(mem, { n, nprobe, maxScanSeg },
        stream);
    std::vector<int64_t> idsVec(ids.numElements());
    AscendTensor<int64_t, DIMS_3> idsHost(idsVec.data(), { n, nprobe, maxScanSeg });

    // inputs of aicpu topk op
    AscendTensor<int64_t, DIMS_1> attrs(mem, { aicpu::TOPK_IVF_ATTR_IDX_COUNT }, stream);
    std::vector<int64_t> attrsVec(aicpu::TOPK_IVF_ATTR_IDX_COUNT);
    attrsVec[aicpu::TOPK_IVF_ATTR_ASC_IDX] = 1;
    attrsVec[aicpu::TOPK_IVF_ATTR_K_IDX] = k;
    attrsVec[aicpu::TOPK_IVF_ATTR_BURST_LEN_IDX] = burstLen;
    attrsVec[aicpu::TOPK_IVF_ATTR_BLOCK_NUM_IDX] = nprobe * maxScanSeg;
    attrsVec[aicpu::TOPK_IVF_ATTR_FLAG_NUM_IDX] = CORE_NUM;
    auto ret = aclrtMemcpy(attrs.data(), attrs.getSizeInBytes(), attrsVec.data(), attrsVec.size() * sizeof(int64_t),
        ACL_MEMCPY_HOST_TO_DEVICE);
    APPERR_RETURN_IF_NOT_LOG(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "Failed to copy attr to device");
    // segs
    std::vector<int> segsVec(n * nprobe);
    AscendTensor<int, DIMS_2> segsHost(segsVec.data(), { n, nprobe });

    for (int nIdx = 0; nIdx < n; ++nIdx) {
        for (int probeIdx = 0; probeIdx < nprobe; ++probeIdx) {
            int list = l1TopNprobeIndices[nIdx][probeIdx].value();

            int segs = static_cast<int>(utils::divUp(deviceListIndices[list]->size(), SEARCH_LIST_SIZE)); // 对应桶的计算次数
            segsHost[nIdx][probeIdx].value(segs);
            for (int m = 0; m < segs; m++) {
                int offset = m * SEARCH_LIST_SIZE;

                uint32_t size = std::min(static_cast<uint32_t>(SEARCH_LIST_SIZE),
                    static_cast<uint32_t>((deviceListIndices[list]->size() - offset)));

                AscendTensor<uint32_t, DIMS_2> actualSize(opSizeHost[nIdx][probeIdx][m].data(),
                    { CORE_NUM, SIZE_ALIGN });
                actualSize[0][0].value(size);

                idsHost[nIdx][probeIdx][m].value(reinterpret_cast<int64_t>(deviceListIndices[list]->data() + offset));
            }
        }
    }

    ret = aclrtMemcpy(opSize.data(), opSize.getSizeInBytes(), opSizeHost.data(), opSizeHost.getSizeInBytes(),
        ACL_MEMCPY_HOST_TO_DEVICE);
    APPERR_RETURN_IF_NOT_LOG(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "Failed to copy op size to device");

    ret = aclrtMemcpy(ids.data(), ids.getSizeInBytes(), idsHost.data(), idsHost.getSizeInBytes(),
        ACL_MEMCPY_HOST_TO_DEVICE);
    APPERR_RETURN_IF_NOT_LOG(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "Failed to copy ids to device");

    if (byResidual) {
        // wait aicpu op - residual
        APPERR_RETURN_IF_NOT_FMT(synchronizeStream(streamAicpu) == ACL_SUCCESS, APP_ERR_INNER_ERROR,
            "synchronizeStream residual aicpu stream failed: %d", ret);
    }

    runL2TopkOp(distResult, minDistResult, ids, opSize, opFlag, attrs, outDists, outIndices, streamAicpu);

    for (int nIdx = 0; nIdx < n; ++nIdx) {
        AscendTensor<float16_t, DIMS_2> queryResidual(residuals[nIdx].data(), { nprobe, dims });
        for (int probeIdx = 0; probeIdx < nprobe; ++probeIdx) {
            int segs = segsHost[nIdx][probeIdx].value();
            int list = l1TopNprobeIndices[nIdx][probeIdx].value();
            for (int m = 0; m < segs; m++) {
                int offset = m * SEARCH_LIST_SIZE;
                int minOffset = m * bursts;

                // code is stored in Zz format, Zz format is 4 dims shaped. z's shape is
                // 16 X 16(AiCore cube's matrix operation's size). Z's shape is (SEARCH_LIST_SIZE / 16) X (dims / 16).
                // Zz's 4 dims shape is ((SEARCH_LIST_SIZE / 16), (dims / 16), 16, 16)
                AscendTensor<float16_t, DIMS_2> query(
                    byResidual ? queryResidual[probeIdx].data() : queries[nIdx].data(), { 1, dims });
                AscendTensor<uint8_t, DIMS_4> code(static_cast<uint8_t *>(deviceListData[list]->data()) + dims * offset,
                    { SEARCH_SHAPED_SIZE, dims / CUBE_ALIGN, CUBE_ALIGN, CUBE_ALIGN });
                AscendTensor<float, DIMS_1> precomp(preComputeData[list]->data() + offset, { SEARCH_LIST_SIZE });
                AscendTensor<uint16_t, DIMS_2> flag(opFlag[nIdx][probeIdx][m].data(), { FLAG_NUM, FLAG_SIZE });
                AscendTensor<uint32_t, DIMS_2> actualSize(opSize[nIdx][probeIdx][m].data(), { CORE_NUM, SIZE_ALIGN });
                AscendTensor<float16_t, DIMS_2> result(distResult[nIdx][probeIdx][offset].data(),
                    { 1, SEARCH_LIST_SIZE });
                AscendTensor<float16_t, DIMS_2> minResult(minDistResult[nIdx][probeIdx][minOffset].data(),
                    { 1, bursts });
                std::vector<const AscendTensorBase *> input {&query, &code, &precomp, &vDiff, &vMin, &actualSize};
                std::vector<const AscendTensorBase *> output {&result, &minResult, &flag};
                runSqDistOperator(1, input, output, stream);
            }
        }
    }
    ret = synchronizeStream(stream);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "synchronizeStream default stream: %i\n",
        ret);

    ret = synchronizeStream(streamAicpu);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR,
        "synchronizeStream aicpu stream failed: %i\n", ret);

    return APP_ERR_OK;
}

APP_ERROR IndexIVFSQL2Aicpu::searchImplL1(AscendTensor<float16_t, DIMS_2> &queries,
    AscendTensor<int64_t, DIMS_2> &l1TopNprobeIndicesHost)
{
    auto &mem = resources.getMemoryManager();
    auto streamPtr = resources.getDefaultStream();
    auto stream = streamPtr->GetStream();
    auto streamAicpuPtr = resources.getAlternateStreams()[0];
    auto streamAicpu = streamAicpuPtr->GetStream();
    // L1 dist op output: dists / vmdists / opFlag
    int n = queries.getSize(0);
    AscendTensor<float16_t, DIMS_2> dists(mem, { n, numLists }, stream);
    // the result constain min value and index, the multi 2, numLists can divide by BURST_LEN
    int minDistSize = std::max(numLists / BURST_LEN * 2, MIN_EXTREME_SIZE);
    AscendTensor<float16_t, DIMS_2> vmdists(mem, { n, minDistSize }, stream);
    AscendTensor<uint32_t, DIMS_2> opSize(mem, { CORE_NUM, SIZE_ALIGN }, stream); // op size, no use
    opSize[0][0] = numLists;
    AscendTensor<uint16_t, DIMS_2> opFlag(mem, { CORE_NUM, FLAG_SIZE }, stream);
    opFlag.zero();

    AscendTensor<int64_t, DIMS_1> attrsInput(mem, { aicpu::TOPK_FLAT_ATTR_IDX_COUNT }, stream);
    std::vector<int64_t> attrs(aicpu::TOPK_FLAT_ATTR_IDX_COUNT);
    attrs[aicpu::TOPK_FLAT_ATTR_ASC_IDX] = 1;
    attrs[aicpu::TOPK_FLAT_ATTR_K_IDX] = nprobe;
    attrs[aicpu::TOPK_FLAT_ATTR_BURST_LEN_IDX] = BURST_LEN;
    attrs[aicpu::TOPK_FLAT_ATTR_BLOCK_NUM_IDX] = 1;
    attrs[aicpu::TOPK_FLAT_ATTR_PAGE_IDX] = 0;
    attrs[aicpu::TOPK_FLAT_ATTR_PAGE_NUM_IDX] = 1;
    attrs[aicpu::TOPK_FLAT_ATTR_QUICK_HEAP] = 0;
    attrs[aicpu::TOPK_FLAT_ATTR_BLOCK_SIZE] = numLists;

    auto ret = aclrtMemcpy(attrsInput.data(), attrsInput.getSizeInBytes(), attrs.data(), attrs.size() * sizeof(int64_t),
        ACL_MEMCPY_HOST_TO_DEVICE);
    APPERR_RETURN_IF_NOT_LOG(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "Failed to copy attr to device");

    AscendTensor<float16_t, DIMS_2> l1TopNprobeDists(mem, { n, nprobe }, stream);
    AscendTensor<int64_t, DIMS_2> l1TopNprobeIndices(mem, { n, nprobe }, stream);

    runL1TopkOp(dists, vmdists, opSize, opFlag, attrsInput, l1TopNprobeDists, l1TopNprobeIndices, streamAicpu);
    // run l1 distance calculation
    runL1DistOp(queries, coarseCentroidsShaped, normCoarseCentroids, dists, vmdists, opFlag, stream);

    ret = synchronizeStream(stream);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "synchronizeStream default stream: %i\n",
        ret);

    ret = synchronizeStream(streamAicpu);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR,
        "synchronizeStream aicpu stream failed: %i\n", ret);

    ret = aclrtMemcpy(l1TopNprobeIndicesHost.data(), l1TopNprobeIndicesHost.getSizeInBytes(), l1TopNprobeIndices.data(),
        l1TopNprobeIndices.getSizeInBytes(), ACL_MEMCPY_DEVICE_TO_HOST);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "Mem operator error %d", (int)ret);

    return APP_ERR_OK;
}

APP_ERROR IndexIVFSQL2Aicpu::searchImplL2Batch(int batch, AscendTensor<float16_t, DIMS_2> &queries,
    AscendTensor<int64_t, DIMS_2> &l1TopNprobeIndices, AscendTensor<float16_t, DIMS_2> &outDists,
    AscendTensor<int64_t, DIMS_2> &outIndices)
{
    int n = queries.getSize(0);
    int k = outDists.getSize(1);
    int batchs = utils::divUp(n, batch);
    int retain = n;
    for (int i = 0; i < batchs; i++) {
        int subBatch = retain > batch ? batch : retain;
        AscendTensor<float16_t, DIMS_2> subQueries(queries[i * batch][0].data(), {subBatch, dims});
        AscendTensor<int64_t, DIMS_2> subNprobeIndices(
            l1TopNprobeIndices[i * batch][0].data(), {subBatch, nprobe});
        AscendTensor<float16_t, DIMS_2> subOutDist(outDists[i * batch][0].data(), {subBatch, k});
        AscendTensor<int64_t, DIMS_2> subOutIndices(outIndices[i * batch][0].data(), {subBatch, k});
        auto ret = searchImplL2(subQueries, subNprobeIndices, subOutDist, subOutIndices);
        APPERR_RETURN_IF_NOT_LOG(ret == ACL_SUCCESS, ACL_ERROR_FAILURE, "Failed to search sub batch");
        retain -= batch;
    }
    return APP_ERR_OK;
}

APP_ERROR IndexIVFSQL2Aicpu::searchImpl(int n, const float16_t *x, int k, float16_t *distances, idx_t *labels)
{
    ASCEND_THROW_IF_NOT_MSG(this->ntotal != 0, "feature vector's number is 0, please check if add before search");
    APP_ERROR ret = APP_ERR_OK;
    auto streamPtr = resources.getDefaultStream();
    auto stream = streamPtr->GetStream();
    auto &mem = resources.getMemoryManager();

    AscendTensor<float16_t, DIMS_2> queries(const_cast<float16_t *>(x), { n, dims });

    // L1 search, to find nprobe IVF list
    // L1 topk op output: top nprobe dist and top nprobe label
    std::vector<int64_t> l1TopNprobeIndicesVec(n * nprobe, 0);
    AscendTensor<int64_t, DIMS_2> l1TopNprobeIndicesHost(l1TopNprobeIndicesVec.data(), { n, nprobe });

    ret = searchImplL1(queries, l1TopNprobeIndicesHost);
    APPERR_RETURN_IF(ret, ret);

    AscendTensor<float16_t, DIMS_2> outDists(mem, { n, k }, stream);
    AscendTensor<int64_t, DIMS_2> outIndices(mem, { n, k }, stream);

    // L2 search, search codes in nprobe IVF list to find topk results
    int maxBatch = static_cast<int>(calMaxBatch());
    if (n > maxBatch) {
        ret = searchImplL2Batch(maxBatch, queries, l1TopNprobeIndicesHost, outDists, outIndices);
    } else {
        ret = searchImplL2(queries, l1TopNprobeIndicesHost, outDists, outIndices);
    }
    APPERR_RETURN_IF(ret, ret);

    auto retAcl = aclrtMemcpy(distances, outDists.getSizeInBytes(), outDists.data(), outDists.getSizeInBytes(),
        ACL_MEMCPY_DEVICE_TO_HOST);
    APPERR_RETURN_IF_NOT_LOG(retAcl == ACL_SUCCESS, APP_ERR_INNER_ERROR, "Failed to copy dists back to host");

    retAcl = aclrtMemcpy(labels, outIndices.getSizeInBytes(), outIndices.data(), outIndices.getSizeInBytes(),
        ACL_MEMCPY_DEVICE_TO_HOST);
    APPERR_RETURN_IF_NOT_LOG(retAcl == ACL_SUCCESS, APP_ERR_INNER_ERROR, "Failed to copy labels back to host");

    return ret;
}

APP_ERROR IndexIVFSQL2Aicpu::resetL2TopkOp()
{
    auto topkCompOpReset = [&](std::unique_ptr<AscendOperator> &op, int64_t batch) {
        AscendOpDesc desc("TopkIvf");
        std::vector<int64_t> shape0 { batch, 0, 1, SEARCH_LIST_SIZE };
        std::vector<int64_t> shape1 { batch, 0, 1, bursts };
        std::vector<int64_t> shape2 { batch, 0, 1 };
        std::vector<int64_t> shape3 { batch, 0, CORE_NUM * SIZE_ALIGN };
        std::vector<int64_t> shape4 { batch, 0, FLAG_NUM, FLAG_SIZE };
        std::vector<int64_t> shape5 { aicpu::TOPK_IVF_ATTR_IDX_COUNT };

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

APP_ERROR IndexIVFSQL2Aicpu::resetResidualOp()
{
    std::string opTypeName = "ResidualIvf";

    for (auto batch : searchBatchSizes) {
        std::vector<int64_t> shape0 { batch, dims };
        std::vector<int64_t> shape1 { numLists, dims };
        std::vector<int64_t> shape2 { batch, nprobe };

        std::vector<int64_t> shape3 { batch, nprobe, dims };

        std::vector<std::pair<aclDataType, std::vector<int64_t>>> input {
            { ACL_FLOAT16, shape0 },
            { ACL_FLOAT16, shape1 },
            { ACL_UINT64, shape2 }
        };
        std::vector<std::pair<aclDataType, std::vector<int64_t>>> output {
            { ACL_FLOAT16, shape3 }
        };
        residualOps[batch] = std::unique_ptr<AscendOperator>(nullptr);
        auto ret = resetOp(opTypeName, residualOps[batch], input, output);
        APPERR_RETURN_IF_NOT_FMT(ret == APP_ERR_OK, ret, "op init failed: %i", ret);
    }

    return APP_ERR_OK;
}

void IndexIVFSQL2Aicpu::runL2TopkOp(AscendTensor<float16_t, DIMS_3> &dists,
                                    AscendTensor<float16_t, DIMS_3> &vmdists,
                                    AscendTensor<int64_t, DIMS_3> &ids,
                                    AscendTensor<uint32_t, DIMS_4> &sizes,
                                    AscendTensor<uint16_t, DIMS_4> &flags,
                                    AscendTensor<int64_t, DIMS_1> &attrs,
                                    AscendTensor<float16_t, DIMS_2> &outdists,
                                    AscendTensor<int64_t, DIMS_2> &outlabel, aclrtStream stream)
{
    int batch = dists.getSize(0);
    std::vector<const AscendTensorBase *> input{&dists, &vmdists, &ids, &sizes, &flags, &attrs};
    std::vector<const AscendTensorBase *> output{&outdists, &outlabel};
    IndexIVFSQ::runL2TopkOp(batch, input, output, stream);
}

void IndexIVFSQL2Aicpu::runResidualOp(int batch,
                                      const std::vector<const AscendTensorBase *> &input,
                                      const std::vector<const AscendTensorBase *> &output,
                                      aclrtStream stream)
{
    AscendOperator *op = nullptr;
    if (residualOps.find(batch) != residualOps.end()) {
        op = residualOps[batch].get();
    }
    ASCEND_THROW_IF_NOT(op);
    auto ret = runOp(op, input, output, stream);
    ASCEND_THROW_IF_NOT_FMT(ret == APP_ERR_OK, "run operator failed: %i\n", ret);
}

APP_ERROR IndexIVFSQL2Aicpu::resetSqDistOperator()
{
    std::string opTypeName = "DistanceIVFSQ8L2";
    int batch = 1;

    std::vector<int64_t> queryShape({ 1, dims });
    std::vector<int64_t> codeShape({ SEARCH_SHAPED_SIZE, dims / CUBE_ALIGN, CUBE_ALIGN, CUBE_ALIGN });
    std::vector<int64_t> precompShape({ SEARCH_LIST_SIZE });
    std::vector<int64_t> vdiffShape({ dims });
    std::vector<int64_t> vminShape({ dims });
    std::vector<int64_t> sizeShape({ CORE_NUM, SIZE_ALIGN });
    std::vector<int64_t> resultShape({ 1, SEARCH_LIST_SIZE });
    std::vector<int64_t> minResultShape({ 1, bursts });
    std::vector<int64_t> flagShape({ FLAG_NUM, FLAG_SIZE });

    std::vector<std::pair<aclDataType, std::vector<int64_t>>> input {
        { ACL_FLOAT16, queryShape },
        { ACL_UINT8, codeShape },
        { ACL_FLOAT, precompShape },
        { ACL_FLOAT16, vdiffShape },
        { ACL_FLOAT16, vminShape },
        { ACL_UINT32, sizeShape }
    };
    std::vector<std::pair<aclDataType, std::vector<int64_t>>> output {
        { ACL_FLOAT16, resultShape },
        { ACL_FLOAT16, minResultShape },
        { ACL_UINT16, flagShape }
    };
    l2DistOps[batch] = std::unique_ptr<AscendOperator>(nullptr);
    auto ret = resetOp(opTypeName, l2DistOps[batch], input, output);
    APPERR_RETURN_IF_NOT_FMT(ret == APP_ERR_OK, ret, "op init failed: %i", ret);
    return APP_ERR_OK;
}

void IndexIVFSQL2Aicpu::runSqDistOperator(int batch,
                                          const std::vector<const AscendTensorBase *> &input,
                                          const std::vector<const AscendTensorBase *> &output,
                                          aclrtStream stream)
{
    AscendOperator *op = nullptr;
    if (l2DistOps.find(batch) != l2DistOps.end()) {
        op = l2DistOps[batch].get();
    }

    auto ret = runOp(op, input, output, stream);
    ASCEND_THROW_IF_NOT_FMT(ret == APP_ERR_OK, "run operator failed: %i\n", ret);
}
} // ascend
