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


#include <algorithm>
#include <ascenddaemon/impl_custom/IndexIVFSPSQL2.h>
#include <ascenddaemon/impl/AuxIndexStructures.h>
#include <ascenddaemon/utils/Limits.h>
#include <common/utils/CommonUtils.h>
#include "ascenddaemon/utils/BatchQueueItem.h"

namespace ascendSearch {
namespace {
const int SP_SQ_BLOCK_SIZE = 16384 * 16;
const int SP_SQ_COMPUTE_PAGE = SP_SQ_BLOCK_SIZE * 16;
const int THREADS_CNT = 6;
const int SP_SQ_BURST_LEN = 64;
const int SP_CORE_NUM = 8;
const int SP_SQ_SIZE_ALIGN = 16;
const int SEARCH_LIST_SIZE = faiss::ascendSearch::SocUtils::GetInstance()
                                .GetSearchListSize(); // must be CUBE_ALIGN aligned
}

IndexIVFSPSQL2::IndexIVFSPSQL2(int dim, int dim2, int k, int nlist, bool encodeResidual, int nprobes,
                               int searchListSize, int handleBatch, bool filterable, int64_t resourceSize)
    : IndexIVFSPSQ(dim, dim2, nlist, encodeResidual, nprobes, searchListSize, handleBatch, filterable, resourceSize),
    pageSize(SP_SQ_COMPUTE_PAGE), dim2(dim2), nCodeBook(0), byResidual(encodeResidual), codebookNum(k)
{
    searchBatchSizes = { 32, 16, 8, 4, 2, 1};
    addBatchSizes = {1024};
    ASCEND_THROW_IF_NOT(dims % CUBE_ALIGN == 0);

    this->distsLen = searchListSize;
    this->handleBatch = handleBatch; // issue 8 lists at a time for 310P, 4 for 310
    // Due to the limitation of Ub size, if dim < 256, burstlen = 32, otherwise burstlen = 16
    int burstLen = 64;
    int extremNum = 2;
    this->burstLen = burstLen;
    this->maxesLen = searchListSize / this->burstLen * extremNum;
}

IndexIVFSPSQL2::~IndexIVFSPSQL2() {}

APP_ERROR IndexIVFSPSQL2::init()
{
    APPERR_RETURN_IF_NOT_OK(resetL1DistOp(numLists));
    APPERR_RETURN_IF_NOT_OK(resetFilterSqDistOperator());
    APPERR_RETURN_IF_NOT_OK(resetSqDistOperatorFor310P());

    return APP_ERR_OK;
}

APP_ERROR IndexIVFSPSQL2::getCodeWord(int n, float16_t *feature, float16_t *codeWord, idx_t *labels)
{
    VALUE_UNUSED(n);
    APPERR_RETURN_IF_NOT_LOG(feature, APP_ERR_INVALID_PARAM, "feature can not be nullptr.");
    APPERR_RETURN_IF_NOT_LOG(codeWord, APP_ERR_INVALID_PARAM, "codeWord can not be nullptr.");
    APPERR_RETURN_IF_NOT_LOG(labels, APP_ERR_INVALID_PARAM, "labels can not be nullptr.");
    return APP_ERR_OK;
}

APP_ERROR IndexIVFSPSQL2::resetL1DistOp(int numLists)
{
    VALUE_UNUSED(numLists);
    auto l1DistOpReset = [&](std::unique_ptr<AscendOperator> &op, int64_t batch) {
        AscendOpDesc desc("DistanceComputeQC");
        std::vector<int64_t> queryShape({ batch, dims });
        std::vector<int64_t> coarseCentroidsShape({ utils::divUp(dim2*codebookNum, CUBE_ALIGN),
                                                    utils::divUp(dims, CUBE_ALIGN), CUBE_ALIGN, CUBE_ALIGN });
        std::vector<int64_t> preNormsShape({ codebookNum });

        std::vector<int64_t> distResultShape({ batch, codebookNum });
        // the result constain min value and index, the multi 2
        std::vector<int64_t> flagShape({ SP_CORE_NUM, FLAG_SIZE });

        desc.addInputTensorDesc(ACL_FLOAT16, queryShape.size(), queryShape.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_FLOAT16, coarseCentroidsShape.size(), coarseCentroidsShape.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_FLOAT, preNormsShape.size(), preNormsShape.data(), ACL_FORMAT_ND);
        desc.addOutputTensorDesc(ACL_FLOAT16, distResultShape.size(), distResultShape.data(), ACL_FORMAT_ND);
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
    for (auto batch : addBatchSizes) {
        l1DistOps[batch] = std::unique_ptr<AscendOperator>(nullptr);
        APPERR_RETURN_IF_NOT_LOG(l1DistOpReset(l1DistOps[batch], batch),
                                 APP_ERR_ACL_OP_LOAD_MODEL_FAILED, "op init failed");
    }

    return APP_ERR_OK;
}

APP_ERROR IndexIVFSPSQL2::reset()
{
    for (int i = 0; i < numLists; ++i) {
        deviceAllData[i]->clear();
        bucketSize[i] = 0;
        addFinishFlag[i] = 0;
    }
    isTrained = true;
    ntotal = 0;
    maxListLength = 0;

    return APP_ERR_OK;
}

int IndexIVFSPSQL2::getCodeBookSize()
{
    return nCodeBook;
}

void IndexIVFSPSQL2::moveVectorForward(idx_t srcIdx, idx_t dstIdx)
{
    ASCEND_THROW_IF_NOT(srcIdx >= dstIdx);
    // 1. move code
    IndexIVFSPSQ::moveVectorForward(srcIdx, dstIdx);

    // 2. move precompute
    int srcIdx1 = srcIdx / this->blockSize;
    int srcIdx2 = srcIdx % this->blockSize;
    int dstIdx1 = dstIdx / this->blockSize;
    int dstIdx2 = dstIdx % this->blockSize;

    (*normBase[dstIdx1])[dstIdx2] = (*normBase[srcIdx1])[srcIdx2];
}

void IndexIVFSPSQL2::releaseUnusageSpace(int oldTotal, int remove)
{
    IndexIVFSPSQ::releaseUnusageSpace(oldTotal, remove);

    int oldVecSize = utils::divUp(oldTotal, this->blockSize);
    int vecSize = utils::divUp(oldTotal - remove, this->blockSize);

    for (int i = oldVecSize - 1; i >= vecSize; --i) {
        this->normBase.at(i)->clear();
    }
}

size_t IndexIVFSPSQL2::calcNormBaseSize(idx_t totalNum)
{
    int numBatch = utils::divUp(totalNum, blockSize);
    return numBatch * blockSize;
}

APP_ERROR IndexIVFSPSQL2::searchFilterImpl(int n, const float16_t *x, int k,
    float16_t *distances, idx_t *labels, uint32_t filterSize, uint32_t* filters)
{
    VALUE_UNUSED(n);
    VALUE_UNUSED(x);
    VALUE_UNUSED(k);
    VALUE_UNUSED(distances);
    VALUE_UNUSED(labels);
    VALUE_UNUSED(filterSize);
    VALUE_UNUSED(filters);
    return APP_ERR_OK;
}

APP_ERROR IndexIVFSPSQL2::searchFilterImplL2(AscendTensor<float16_t, DIMS_2> &queries,
    uint32_t filterSize, uint32_t* filters,
    AscendTensor<float16_t, DIMS_2> &l1Dists,
    AscendTensor<float16_t, DIMS_2> &outDists,
    AscendTensor<idx_t, DIMS_2> &outIndices)
{
    VALUE_UNUSED(queries);
    VALUE_UNUSED(filterSize);
    VALUE_UNUSED(filters);
    VALUE_UNUSED(l1Dists);
    VALUE_UNUSED(outDists);
    VALUE_UNUSED(outIndices);
    return APP_ERR_OK;
}

APP_ERROR IndexIVFSPSQL2::searchFilterImpl(std::vector<Index*> indexes, int n, int batchSize, const float16_t *x, int k,
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
    return APP_ERR_OK;
}

APP_ERROR IndexIVFSPSQL2::searchImplL2(AscendTensor<float16_t, DIMS_2> &queries,
                                       AscendTensor<float16_t, DIMS_2> &l1Dists,
                                       AscendTensor<float16_t, DIMS_2> &outDists,
                                       AscendTensor<idx_t, DIMS_2> &outIndices)
{
    VALUE_UNUSED(queries);
    VALUE_UNUSED(l1Dists);
    VALUE_UNUSED(outDists);
    VALUE_UNUSED(outIndices);
    return APP_ERR_OK;
}

APP_ERROR IndexIVFSPSQL2::searchImplL1(AscendTensor<float16_t, DIMS_2> &queries,
                                       AscendTensor<float16_t, DIMS_2> &dists,
                                       aclrtStream stream)
{
    auto &mem = resources.getMemoryManager();
    AscendTensor<uint16_t, DIMS_2> opFlag(mem, { SP_CORE_NUM, FLAG_SIZE }, stream);
    opFlag.zero();

    // run l1 distance calculation
    runL1DistOp(queries, *coarseCentroidsShaped, *normCoarseCentroids, dists, opFlag, stream);
    for (int i = 0; i < SP_CORE_NUM; ++i) {
        uint16_t *volatile flagPtr = opFlag.data() + i*FLAG_SIZE;
        WAITING_FLAG_READY(*flagPtr, TIMEOUT_CHECK_TICK, TIMEOUT_MS);
    }

    return APP_ERR_OK;
}

void IndexIVFSPSQL2::runL1DistOp(AscendTensor<float16_t, DIMS_2>& queryVecs,
    AscendTensor<float16_t, DIMS_4>& shapedData,
    AscendTensor<float16_t, DIMS_1>& norms,
    AscendTensor<float16_t, DIMS_2>& outDists,
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
        new std::vector<const aclDataBuffer *>, CommonUtils::AclInputBufferDelete);
    distOpInput->emplace_back(aclCreateDataBuffer(queryVecs.data(), queryVecs.getSizeInBytes()));
    distOpInput->emplace_back(aclCreateDataBuffer(shapedData.data(), shapedData.getSizeInBytes()));
    distOpInput->emplace_back(aclCreateDataBuffer(norms.data(), norms.getSizeInBytes()));

    std::shared_ptr<std::vector<aclDataBuffer *>> distOpOutput(
         new std::vector<aclDataBuffer *>, CommonUtils::AclOutputBufferDelete);
    distOpOutput->emplace_back(aclCreateDataBuffer(outDists.data(), outDists.getSizeInBytes()));
    distOpOutput->emplace_back(aclCreateDataBuffer(flag.data(), flag.getSizeInBytes()));

    op->exec(*distOpInput, *distOpOutput, stream);
}

APP_ERROR IndexIVFSPSQL2::searchImpl(int n, const float16_t *x, int k, float16_t *dists, idx_t *labels)
{
    APP_ERROR ret = APP_ERR_OK;
    auto stream = resources.getDefaultStream();
    auto &mem = resources.getMemoryManager();
    AscendTensor<float16_t, DIMS_2> queries(const_cast<float16_t *>(x), { n, dims });
    AscendTensor<float16_t, DIMS_2> outDists(dists, { n, k });
    AscendTensor<idx_t, DIMS_2> outIndices(labels, { n, k });

    // init results to invalid data.
    outDists.initValue(Limits<float16_t>::getMax());
    outIndices.initValue(std::numeric_limits<idx_t>::max());

    // for performance improving, bind the main thread to cpu4-5,
    // and bind the threadpool to cpu0-cpu3. when n == 1, attach
    // main thread to one cpu(cpu5) is better than multicpus.
    if (n > 1) {
        AscendUtils::attachToCpus({ 4, 5 });
    } else {
        AscendUtils::attachToCpus({ 5 });
    }

    // L1 search, to find nprobe IVF list
    AscendTensor<float16_t, DIMS_2> l1Dists(mem, { n, numLists }, stream);
    ret = searchImplL1(queries, l1Dists, stream);
    APPERR_RETURN_IF(ret, ret);

    // L2 search, search codes in nprobe IVF list to find topk results
    ret = searchImplL2(queries, l1Dists, outDists, outIndices);
    APPERR_RETURN_IF(ret, ret);

    // reattach cpus to cpu set { 0, 1, 2, 3, 4, 5 }
    AscendUtils::attachToCpus({ 0, 1, 2, 3, 4, 5 });
    return ret;
}

APP_ERROR IndexIVFSPSQL2::searchImpl(std::vector<Index *> indexes, int n, int batchSize, const float16_t *x, int k,
    float16_t *distances, idx_t *labels)
{
    VALUE_UNUSED(n);
    VALUE_UNUSED(batchSize);
    VALUE_UNUSED(k);
    APPERR_RETURN_IF_NOT_LOG(indexes.data(), APP_ERR_INVALID_PARAM, "indexes can not be nullptr.");
    APPERR_RETURN_IF_NOT_LOG(x, APP_ERR_INVALID_PARAM, "x can not be nullptr.");
    APPERR_RETURN_IF_NOT_LOG(distances, APP_ERR_INVALID_PARAM, "distances can not be nullptr.");
    APPERR_RETURN_IF_NOT_LOG(labels, APP_ERR_INVALID_PARAM, "labels can not be nullptr.");
    return APP_ERR_OK;
}

APP_ERROR IndexIVFSPSQL2::searchImpl(AscendTensor<float16_t, DIMS_2> &queries, int k,
    AscendTensor<float16_t, DIMS_2> &outDistance,
    AscendTensor<idx_t, DIMS_2> &outIndices)
{
    VALUE_UNUSED(k);
    APPERR_RETURN_IF_NOT_LOG(queries.data(), APP_ERR_INVALID_PARAM, "queries can not be nullptr.");
    APPERR_RETURN_IF_NOT_LOG(outDistance.data(), APP_ERR_INVALID_PARAM, "outDistance can not be nullptr.");
    APPERR_RETURN_IF_NOT_LOG(outIndices.data(), APP_ERR_INVALID_PARAM, "outIndices can not be nullptr.");
    return APP_ERR_OK;
}

APP_ERROR IndexIVFSPSQL2::resetSqDistOperatorFor310P()
{
    distSqOp.reset();
    AscendOpDesc desc("DistanceIVFSpIntL2Mins");
    std::vector<int64_t> queryShape({ 1, dims });
    std::vector<int64_t> codeBookShape({ utils::divUp(dim2*codebookNum, CUBE_ALIGN),
        utils::divUp(dims, CUBE_ALIGN), CUBE_ALIGN, CUBE_ALIGN });
    std::vector<int64_t> baseShape({ searchListSize * dim2 }); //  [n//16, dim2//16,16,16]
    std::vector<int64_t> preNormsShape({ searchListSize });
    std::vector<int64_t> vDiffShape({ dim2 }); // [dim2]
    std::vector<int64_t> vMinShape({ dim2 }); // [dim2]
    std::vector<int64_t> codebookOffsetShape({ handleBatch });
    std::vector<int64_t> baseOffsetShape({ handleBatch });
    std::vector<int64_t> normOffsetShape({ handleBatch });
    std::vector<int64_t> sizeShape({ handleBatch });
    std::vector<int64_t> resultShape({ handleBatch, searchListSize });
    std::vector<int64_t> minResultShape(
        {handleBatch, searchListSize/this->burstLen*2}); // each minResult has 2 values
    std::vector<int64_t> flagShape({ SP_CORE_NUM, FLAG_SIZE });
    desc.addInputTensorDesc(ACL_FLOAT16, queryShape.size(), queryShape.data(), ACL_FORMAT_ND);
    desc.addInputTensorDesc(ACL_FLOAT16, codeBookShape.size(), codeBookShape.data(), ACL_FORMAT_ND);
    desc.addInputTensorDesc(ACL_UINT8, baseShape.size(), baseShape.data(), ACL_FORMAT_ND);
    desc.addInputTensorDesc(ACL_FLOAT, preNormsShape.size(), preNormsShape.data(), ACL_FORMAT_ND);
    desc.addInputTensorDesc(ACL_FLOAT16, vDiffShape.size(), vDiffShape.data(), ACL_FORMAT_ND);
    desc.addInputTensorDesc(ACL_FLOAT16, vMinShape.size(), vMinShape.data(), ACL_FORMAT_ND);
    desc.addInputTensorDesc(ACL_UINT32, sizeShape.size(), sizeShape.data(), ACL_FORMAT_ND);
    desc.addInputTensorDesc(ACL_UINT64, codebookOffsetShape.size(), codebookOffsetShape.data(), ACL_FORMAT_ND);
    desc.addInputTensorDesc(ACL_UINT64, baseOffsetShape.size(), baseOffsetShape.data(), ACL_FORMAT_ND);
    desc.addInputTensorDesc(ACL_UINT64, normOffsetShape.size(), normOffsetShape.data(), ACL_FORMAT_ND);

    desc.addOutputTensorDesc(ACL_FLOAT16, resultShape.size(), resultShape.data(), ACL_FORMAT_ND);
    desc.addOutputTensorDesc(ACL_FLOAT16, minResultShape.size(), minResultShape.data(), ACL_FORMAT_ND);
    desc.addOutputTensorDesc(ACL_UINT16, flagShape.size(), flagShape.data(), ACL_FORMAT_ND);

    distSqOp = CREATE_UNIQUE_PTR(AscendOperator, desc);
    APPERR_RETURN_IF_NOT_LOG(distSqOp->init(), APP_ERR_ACL_OP_LOAD_MODEL_FAILED, "op init failed");

    return APP_ERR_OK;
}

APP_ERROR IndexIVFSPSQL2::resetFilterSqDistOperator()
{
    l2FilterDistOps.reset();
    AscendOpDesc desc("DistanceMaskedIVFSpIntL2Mins");
    std::vector<int64_t> queryShape({ 1, dims });
    std::vector<int64_t> codeBookShape({ utils::divUp(dim2*codebookNum, CUBE_ALIGN),
                                         utils::divUp(dims, CUBE_ALIGN), CUBE_ALIGN, CUBE_ALIGN });
    std::vector<int64_t> maskShape({ handleBatch, utils::divUp(searchListSize, 8) });
    std::vector<int64_t> baseShape({ searchListSize * dim2 }); //  [n//16, dim2//16,16,16]
    std::vector<int64_t> preNormsShape({ searchListSize });
    std::vector<int64_t> vDMShape({ 2, dim2 }); // [dim2]
    std::vector<int64_t> codebookOffsetShape({ handleBatch });
    std::vector<int64_t> baseOffsetShape({ handleBatch });
    std::vector<int64_t> normOffsetShape({ handleBatch });
    std::vector<int64_t> sizeShape({ handleBatch });
    std::vector<int64_t> resultShape({ handleBatch, searchListSize });
    std::vector<int64_t> minResultShape(
        {handleBatch, searchListSize/this->burstLen*2}); // each minResult has 2 values
    std::vector<int64_t> flagShape({ SP_CORE_NUM, FLAG_SIZE });

    desc.addInputTensorDesc(ACL_FLOAT16, queryShape.size(), queryShape.data(), ACL_FORMAT_ND);
    desc.addInputTensorDesc(ACL_FLOAT16, codeBookShape.size(), codeBookShape.data(), ACL_FORMAT_ND);
    desc.addInputTensorDesc(ACL_UINT8, baseShape.size(), baseShape.data(), ACL_FORMAT_ND);
    desc.addInputTensorDesc(ACL_UINT8, maskShape.size(), maskShape.data(), ACL_FORMAT_ND);
    desc.addInputTensorDesc(ACL_FLOAT, preNormsShape.size(), preNormsShape.data(), ACL_FORMAT_ND);
    desc.addInputTensorDesc(ACL_FLOAT16, vDMShape.size(), vDMShape.data(), ACL_FORMAT_ND);
    desc.addInputTensorDesc(ACL_UINT32, sizeShape.size(), sizeShape.data(), ACL_FORMAT_ND);
    desc.addInputTensorDesc(ACL_UINT64, codebookOffsetShape.size(), codebookOffsetShape.data(), ACL_FORMAT_ND);
    desc.addInputTensorDesc(ACL_UINT64, baseOffsetShape.size(), baseOffsetShape.data(), ACL_FORMAT_ND);
    desc.addInputTensorDesc(ACL_UINT64, normOffsetShape.size(), normOffsetShape.data(), ACL_FORMAT_ND);

    desc.addOutputTensorDesc(ACL_FLOAT16, resultShape.size(), resultShape.data(), ACL_FORMAT_ND);
    desc.addOutputTensorDesc(ACL_FLOAT16, minResultShape.size(), minResultShape.data(), ACL_FORMAT_ND);
    desc.addOutputTensorDesc(ACL_UINT16, flagShape.size(), flagShape.data(), ACL_FORMAT_ND);

    l2FilterDistOps = CREATE_UNIQUE_PTR(AscendOperator, desc);
    APPERR_RETURN_IF_NOT_LOG(l2FilterDistOps->init(), APP_ERR_ACL_OP_LOAD_MODEL_FAILED, "op init failed");

    return APP_ERR_OK;
}

void IndexIVFSPSQL2::runSqDistOperator(AscendTensor<float16_t, DIMS_2> &queries,
                                       AscendTensor<float16_t, DIMS_4> &book,
                                       AscendTensor<uint8_t, DIMS_1> &base,
                                       AscendTensor<uint8_t, DIMS_2> &mask,
                                       AscendTensor<float, DIMS_1> &norm,
                                       AscendTensor<float16_t, DIMS_2> &dm,
                                       AscendTensor<uint64_t, DIMS_1> &baseOffset,
                                       AscendTensor<uint64_t, DIMS_1> &codebookOffset,
                                       AscendTensor<uint64_t, DIMS_1> &normOffset,
                                       AscendTensor<uint32_t, DIMS_1> &size,
                                       AscendTensor<float16_t, DIMS_2> &result,
                                       AscendTensor<float16_t, DIMS_2> &maxResult,
                                       AscendTensor<uint16_t, DIMS_2> &flag)
{
    ASCEND_THROW_IF_NOT(l2FilterDistOps.get());
    auto stream = resources.getDefaultStream();
    // prepare for input data's buffer
    std::vector<const aclDataBuffer *> distSqOpInput;
    int distSqOpInputNum = 10;
    int distSqOpInputIdx = 0;
    distSqOpInput.resize(distSqOpInputNum);
    distSqOpInput[distSqOpInputIdx++] = aclCreateDataBuffer(queries.data(),
        queries.getSizeInBytes());               // input 0
    distSqOpInput[distSqOpInputIdx++] = aclCreateDataBuffer(book.data(),
        book.getSizeInBytes());                     // input 1
    distSqOpInput[distSqOpInputIdx++] = aclCreateDataBuffer(base.data(),
        base.getSizeInBytes());                 // input 2
    distSqOpInput[distSqOpInputIdx++] = aclCreateDataBuffer(mask.data(),
        mask.getSizeInBytes());
    distSqOpInput[distSqOpInputIdx++] = aclCreateDataBuffer(norm.data(),
        norm.getSizeInBytes());                       // input 3
    distSqOpInput[distSqOpInputIdx++] = aclCreateDataBuffer(dm.data(),
        dm.getSizeInBytes());                     // input 4
    distSqOpInput[distSqOpInputIdx++] = aclCreateDataBuffer(size.data(),
        size.getSizeInBytes());
    distSqOpInput[distSqOpInputIdx++] = aclCreateDataBuffer(codebookOffset.data(),
        codebookOffset.getSizeInBytes());
    distSqOpInput[distSqOpInputIdx++] = aclCreateDataBuffer(baseOffset.data(),
        baseOffset.getSizeInBytes());
    distSqOpInput[distSqOpInputIdx++] = aclCreateDataBuffer(normOffset.data(),
        normOffset.getSizeInBytes());

    // prepare for output data's buffer
    std::vector<aclDataBuffer *> distSqOpOutput;
    int distSqOpOutputIdx = 0;
    int distSqOpOutputNum = 3;
    distSqOpOutput.resize(distSqOpOutputNum);
    distSqOpOutput[distSqOpOutputIdx++] = aclCreateDataBuffer(result.data(),
        result.getSizeInBytes());            // output 0
    distSqOpOutput[distSqOpOutputIdx++] = aclCreateDataBuffer(maxResult.data(),
        maxResult.getSizeInBytes());      // output 1
    distSqOpOutput[distSqOpOutputIdx++] = aclCreateDataBuffer(flag.data(),
        flag.getSizeInBytes());                // output 2

    l2FilterDistOps->exec(distSqOpInput, distSqOpOutput, stream);

    for (auto &item : distSqOpInput) {
        ACL_REQUIRE_OK(aclDestroyDataBuffer(item));
    }

    for (auto &item : distSqOpOutput) {
        ACL_REQUIRE_OK(aclDestroyDataBuffer(item));
    }
}

void IndexIVFSPSQL2::runSqDistOperator(AscendTensor<float16_t, DIMS_2> &queries,
                                       AscendTensor<float16_t, DIMS_4> &book,
                                       AscendTensor<uint8_t, DIMS_1> &base,
                                       AscendTensor<float, DIMS_1> &norm,
                                       AscendTensor<float16_t, DIMS_1> &diff,
                                       AscendTensor<float16_t, DIMS_1> &min,
                                       AscendTensor<uint64_t, DIMS_1> &baseOffset,
                                       AscendTensor<uint64_t, DIMS_1> &codebookOffset,
                                       AscendTensor<uint64_t, DIMS_1> &normOffset,
                                       AscendTensor<uint32_t, DIMS_1> &size,
                                       AscendTensor<float16_t, DIMS_2> &result,
                                       AscendTensor<float16_t, DIMS_2> &maxResult,
                                       AscendTensor<uint16_t, DIMS_2> &flag)
{
    ASCEND_THROW_IF_NOT(distSqOp.get());
    auto stream = resources.getDefaultStream();

    // prepare for input data's buffer
    std::shared_ptr<std::vector<const aclDataBuffer *>> distSqOpInput(
        new std::vector<const aclDataBuffer *>, CommonUtils::AclInputBufferDelete);
    distSqOpInput->emplace_back(aclCreateDataBuffer(queries.data(), queries.getSizeInBytes()));
    distSqOpInput->emplace_back(aclCreateDataBuffer(book.data(), book.getSizeInBytes()));
    distSqOpInput->emplace_back(aclCreateDataBuffer(base.data(), base.getSizeInBytes()));
    distSqOpInput->emplace_back(aclCreateDataBuffer(norm.data(), norm.getSizeInBytes()));
    distSqOpInput->emplace_back(aclCreateDataBuffer(diff.data(), diff.getSizeInBytes()));
    distSqOpInput->emplace_back(aclCreateDataBuffer(min.data(), min.getSizeInBytes()));
    distSqOpInput->emplace_back(aclCreateDataBuffer(size.data(), size.getSizeInBytes()));
    distSqOpInput->emplace_back(aclCreateDataBuffer(codebookOffset.data(), codebookOffset.getSizeInBytes()));
    distSqOpInput->emplace_back(aclCreateDataBuffer(baseOffset.data(), baseOffset.getSizeInBytes()));
    distSqOpInput->emplace_back(aclCreateDataBuffer(normOffset.data(), normOffset.getSizeInBytes()));

    // prepare for output data's buffer
    std::shared_ptr<std::vector<aclDataBuffer *>> distSqOpOutput(
        new std::vector<aclDataBuffer *>, CommonUtils::AclOutputBufferDelete);
    distSqOpOutput->emplace_back(aclCreateDataBuffer(result.data(), result.getSizeInBytes()));
    distSqOpOutput->emplace_back(aclCreateDataBuffer(maxResult.data(), maxResult.getSizeInBytes()));
    distSqOpOutput->emplace_back(aclCreateDataBuffer(flag.data(), flag.getSizeInBytes()));

    // async executing operator
    distSqOp->exec(*distSqOpInput, *distSqOpOutput, stream);
}

} // namespace ascendSearch
