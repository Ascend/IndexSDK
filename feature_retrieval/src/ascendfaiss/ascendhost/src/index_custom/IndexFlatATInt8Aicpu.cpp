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


#include "index_custom/IndexFlatATInt8Aicpu.h"

#include <cmath>
#include "common/utils/OpLauncher.h"
#include "ops/cpukernel/impl/utils/kernel_shared_def.h"
#include "common/utils/CommonUtils.h"
#include "ascend/utils/fp16.h"

namespace ascend {
namespace {
const int QUERY_ALIGN = 32;
const int BURST_LEN = 64;
const int TRANSFER_SIZE = 256;
const int SEARCH_PAGE = 32768;
const int QUERY_BATCH = 1024;
const int CODE_ALIGN = 512;
const int CUBE_INT8_ALIGN = 32;
const int INT8_LOWER_BOUND = -128;
const int INT8_UPPER_BOUND = 127;
const int UINT8_UPPER_BOUND = 255;
}

IndexFlatATInt8Aicpu::IndexFlatATInt8Aicpu(int dim, int baseSize, int64_t resourceSize)
    : IndexFlatAT(dim, baseSize, resourceSize) {}

IndexFlatATInt8Aicpu::~IndexFlatATInt8Aicpu() {}

APP_ERROR IndexFlatATInt8Aicpu::init()
{
    // reset operator
    APPERR_RETURN_IF_NOT_OK(IndexFlatAT::resetTopkCompOp());
    APPERR_RETURN_IF_NOT_OK(resetL2NormTypingInt8Op());
    APPERR_RETURN_IF_NOT_OK(resetDistL2MinsInt8AtOp());

    // init transfer data
    std::vector<int32_t> transferDataInt32Temp(TRANSFER_SIZE * CUBE_ALIGN, 0);
    for (int i = 0; i < TRANSFER_SIZE; ++i) {
        transferDataInt32Temp[i * CUBE_ALIGN + (i % CUBE_ALIGN)] = 1;
    }
    AscendTensor<int32_t, DIMS_2> transferDataInt32({ TRANSFER_SIZE, CUBE_ALIGN });
    auto ret = aclrtMemcpy(transferDataInt32.data(), transferDataInt32.getSizeInBytes(), transferDataInt32Temp.data(),
        transferDataInt32Temp.size() * sizeof(int32_t), ACL_MEMCPY_HOST_TO_DEVICE);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "aclrtMemcpy operator error %d", (int)ret);
    transferInt32 = std::move(transferDataInt32);

    return APP_ERR_OK;
}

APP_ERROR IndexFlatATInt8Aicpu::reset()
{
    codes.clear();
    preComputeInt.clear();
    this->ntotal = 0;

    return APP_ERR_OK;
}

APP_ERROR IndexFlatATInt8Aicpu::addVectors(const AscendTensor<float16_t, DIMS_2> &rawData)
{
    int num = rawData.getSize(0);
    APPERR_RETURN_IF(num == 0, APP_ERR_OK);

    APPERR_RETURN_IF((std::fabs(qMin) < std::numeric_limits<float>::epsilon())
        && (std::fabs(qMax) < std::numeric_limits<float>::epsilon()), APP_ERR_OK);
    auto ret = saveCodesInt8(rawData);
    APPERR_RETURN_IF_NOT_FMT(ret == APP_ERR_OK, ret, "saveCodesInt8 failed: %i\n", ret);
    this->ntotal += static_cast<uint32_t>(num);
    return APP_ERR_OK;
}

APP_ERROR IndexFlatATInt8Aicpu::saveCodesInt8(const AscendTensor<float16_t, DIMS_2> &rawData)
{
    int num = rawData.getSize(0);
    int dim = rawData.getSize(1);
    int total = static_cast<int64_t>(ntotal) + num;

    auto &mem = resources.getMemoryManager();
    auto streamPtr = resources.getDefaultStream();
    auto stream = streamPtr->GetStream();

    // 1. resize vector and quantizing basecodes
    codes.resize(utils::roundUp(total, CUBE_INT8_ALIGN) * utils::roundUp(dim, CUBE_INT8_ALIGN), true);

    // 2. use aicpu to quantify codes
    std::string opName = "CodesQuantify";
    AscendTensor<int8_t, DIMS_2> codesQ(mem, { num, dim }, stream);
    AscendTensor<float16_t, DIMS_1> codesQuantifyAttr(mem, { aicpu::CODES_QUANTIFY_ATTR_IDX_COUNT }, stream);
    codesQuantifyAttr[aicpu::CODES_QUANTIFY_ATTR_QMAX_IDX] = qMax;
    codesQuantifyAttr[aicpu::CODES_QUANTIFY_ATTR_QMIN_IDX] = qMin;
    LaunchOpTwoInOneOut<float16_t, DIMS_2, ACL_FLOAT16, float16_t, DIMS_1, ACL_FLOAT16, int8_t, DIMS_2, ACL_INT8>(
        opName, stream, rawData, codesQuantifyAttr, codesQ);

    // 3. use aicpu to save codes
    opName = "TransdataShaped";
    AscendTensor<int8_t, DIMS_4> dst((codes.data()),
        { utils::divUp(total, CODE_ALIGN), utils::divUp(dim, CUBE_INT8_ALIGN), CODE_ALIGN, CUBE_INT8_ALIGN });
    AscendTensor<int64_t, DIMS_1> attr(mem, { aicpu::TRANSDATA_SHAPED_ATTR_IDX_COUNT }, stream);
    attr[aicpu::TRANSDATA_SHAPED_ATTR_NTOTAL_IDX] = ntotal;

    LaunchOpTwoInOneOut<int8_t, DIMS_2, ACL_INT8, int64_t, DIMS_1, ACL_INT64, int8_t, DIMS_4, ACL_INT8>(opName, stream,
        codesQ, attr, dst);

    // 4. save norms
    auto ret = saveNormsInt8(codesQ);
    APPERR_RETURN_IF_NOT_FMT(ret == APP_ERR_OK, ret, "saveNormsInt8 failed: %i\n", ret);
    return APP_ERR_OK;
}

APP_ERROR IndexFlatATInt8Aicpu::saveNormsInt8(AscendTensor<int8_t, DIMS_2> &codesQ)
{
    int num = codesQ.getSize(0);

    // 1. resize vector
    preComputeInt.resize((ntotal + num) * CUBE_ALIGN, true);

    // 2. use aicpu to compute and save norm
    std::string opName = "VecL2Sqr";
    auto streamPtr = resources.getDefaultStream();
    auto stream = streamPtr->GetStream();
    AscendTensor<int, DIMS_2> dst(preComputeInt.data() + ntotal * CUBE_ALIGN, { num, CUBE_ALIGN });
    LaunchOpOneInOneOut<int8_t, DIMS_2, ACL_INT8, int, DIMS_2, ACL_INT32>(opName, stream, codesQ, dst);
    auto ret = synchronizeStream(stream);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "synchronizeStream failed: %i\n", ret);

    return APP_ERR_OK;
}

APP_ERROR IndexFlatATInt8Aicpu::searchBatched(int64_t n, const float16_t* x, int64_t k, float16_t* distance,
    idx_t* labels)
{
    ASCEND_THROW_MSG("searchBatched() not implemented for this type of index");
    VALUE_UNUSED(n);
    VALUE_UNUSED(x);
    VALUE_UNUSED(k);
    VALUE_UNUSED(distance);
    VALUE_UNUSED(labels);
    return APP_ERR_OK;
}

APP_ERROR IndexFlatATInt8Aicpu::searchImpl(int n, const float16_t* x, int k, float16_t* distances, idx_t* labels)
{
    ASCEND_THROW_MSG("searchImpl() not implemented for this type of index");
    VALUE_UNUSED(n);
    VALUE_UNUSED(x);
    VALUE_UNUSED(k);
    VALUE_UNUSED(distances);
    VALUE_UNUSED(labels);
    return APP_ERR_OK;
}

APP_ERROR IndexFlatATInt8Aicpu::searchInt8(idx_t n, const int8_t *x, idx_t k, float16_t *distances, idx_t *labels)
{
    APPERR_RETURN_IF_NOT_LOG(x, APP_ERR_INVALID_PARAM, "x can not be nullptr.");
    APPERR_RETURN_IF_NOT_LOG(distances, APP_ERR_INVALID_PARAM, "distance can not be nullptr.");
    APPERR_RETURN_IF_NOT_LOG(labels, APP_ERR_INVALID_PARAM, "labels can not be nullptr.");
    APPERR_RETURN_IF_NOT_LOG(this->isTrained, APP_ERR_INVALID_PARAM, "Index not trained.");
    APPERR_RETURN_IF_NOT_FMT(n <= static_cast<Index::idx_t>(std::numeric_limits<int>::max()),
        APP_ERR_INVALID_PARAM, "indices exceeds max(%d)", std::numeric_limits<int>::max());
    APPERR_RETURN_IF(n == 0 || k == 0, APP_ERR_OK);

    return searchBatchedInt8(n, x, k, distances, labels);
}

APP_ERROR IndexFlatATInt8Aicpu::searchBatchedInt8(int n, const int8_t *x, int k, float16_t *distance, idx_t *labels)
{
    APP_ERROR ret = APP_ERR_OK;

    int pages = n / searchPage;
    for (int i = 0; i < pages; i++) {
        ret = searchImplInt8(searchPage, x + i * searchPage * this->dims, k,
            distance + i * searchPage * k, labels + i * searchPage * k);
        APPERR_RETURN_IF(ret, ret);
    }

    int queryLast = n % searchPage;
    if (queryLast > 0) {
        ret = searchImplInt8(queryLast, x + pages * searchPage * this->dims, k,
            distance + pages * searchPage * k, labels + pages * searchPage * k);
        APPERR_RETURN_IF(ret, ret);
    }

    return APP_ERR_OK;
}

APP_ERROR IndexFlatATInt8Aicpu::searchImplInt8(int n, const int8_t *x, int k, float16_t *distances, idx_t *labels)
{
    auto streamPtr = resources.getDefaultStream();
    auto stream = streamPtr->GetStream();
    auto &mem = resources.getMemoryManager();

    // 1. compute l2 norm of query
    AscendTensor<int8_t, DIMS_2> queries(const_cast<int8_t *>(x), { searchPage, dims });
    AscendTensor<uint32_t, DIMS_2> querySize(mem, { CORE_NUM, SIZE_ALIGN }, stream);
    AscendTensor<int32_t, DIMS_1> queriesNorm(mem, { searchPage }, stream);
    AscendTensor<int8_t, DIMS_4> queriesTyping(
        mem, { searchPage / QUERY_ALIGN, dims / CUBE_INT8_ALIGN, QUERY_ALIGN, CUBE_INT8_ALIGN }, stream);
    querySize[0][0] = n;

    std::vector<const AscendTensorBase *> input {&queries, &transferInt32, &querySize};
    std::vector<const AscendTensorBase *> output {&queriesNorm, &queriesTyping};
    runL2NormTypingInt8Op(input, output, stream);
    auto ret = synchronizeStream(stream);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR,
        "synchronizeStream failed: %i\n", ret);
    // 2. compute L2 distance
    ret = computeL2Int8(n, k, distances, labels, queriesTyping, queriesNorm, distResult,
                        minDistResult, opFlag);
    APPERR_RETURN_IF_NOT_FMT(ret == APP_ERR_OK, ret, "computeL2Int8 failed: %i\n", ret);

    return APP_ERR_OK;
}

APP_ERROR IndexFlatATInt8Aicpu::computeL2Int8Init(int n, int k, AscendTensor<float16_t, DIMS_3> &distResult,
    AscendTensor<float16_t, DIMS_3> &minDistResult, AscendTensor<uint16_t, DIMS_3> &opFlag,
    AscendTensor<int64_t, DIMS_1> &attrsInput)
{
    int batches = utils::divUp(n, queryBatch);
    if (distResult.numElements() < (size_t)(batches * queryBatch * baseSize)) {
        AscendTensor<float16_t, DIMS_3> distResultTmp({ batches, queryBatch, baseSize });
        AscendTensor<float16_t, DIMS_3> minDistResultTmp({ batches, queryBatch, bursts });
        AscendTensor<uint16_t, DIMS_3> opFlagTmp({ batches, CORE_NUM, FLAG_SIZE });
        distResult = std::move(distResultTmp);
        minDistResult = std::move(minDistResultTmp);
        opFlag = std::move(opFlagTmp);
    }
    opFlag.zero();

    // attrs: [0]asc, [1]k, [2]burst_len, [3]block_num [4]special page: -1:first page, 0:mid page, 1:last page
    std::vector<int64_t> attrs(aicpu::TOPK_FLAT_ATTR_IDX_COUNT);
    attrs[aicpu::TOPK_FLAT_ATTR_ASC_IDX] = 1;
    attrs[aicpu::TOPK_FLAT_ATTR_K_IDX] = k;
    attrs[aicpu::TOPK_FLAT_ATTR_BURST_LEN_IDX] = BURST_LEN;
    attrs[aicpu::TOPK_FLAT_ATTR_BLOCK_NUM_IDX] = 1;
    attrs[aicpu::TOPK_FLAT_ATTR_PAGE_IDX] = 0; // set as last page to reorder
    attrs[aicpu::TOPK_FLAT_ATTR_PAGE_NUM_IDX] = 1;
    attrs[aicpu::TOPK_FLAT_ATTR_PAGE_SIZE_IDX] = 0;
    attrs[aicpu::TOPK_FLAT_ATTR_QUICK_HEAP] = 1;
    attrs[aicpu::TOPK_FLAT_ATTR_BLOCK_SIZE] = baseSize;
    
    auto ret = aclrtMemcpy(attrsInput.data(), attrsInput.getSizeInBytes(), attrs.data(), attrs.size() * sizeof(int64_t),
        ACL_MEMCPY_HOST_TO_DEVICE);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "Mem operator error %d", ret);

    return APP_ERR_OK;
}

APP_ERROR  IndexFlatATInt8Aicpu::CopyLabelsDeviceToHost(idx_t* hostData, int n, int k,
    AscendTensor<int64_t, DIMS_3> &deviceData)
{
    const auto kind = ACL_MEMCPY_DEVICE_TO_HOST;
    // memcpy data back from dev to host
    auto ret = aclrtMemcpy(hostData, n * k * sizeof(int64_t), deviceData.data(), n * k * sizeof(int64_t), kind);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "CopyDeviceToHost failed: %i\n", ret);
    return APP_ERR_OK;
}

APP_ERROR  IndexFlatATInt8Aicpu::CopyDisDeviceToHost(float16_t* hostData, int n, int k,
    AscendTensor<float16_t, DIMS_3> &deviceData)
{
    const auto kind = ACL_MEMCPY_DEVICE_TO_HOST;
    // memcpy data back from dev to host
    auto ret = aclrtMemcpy(hostData, n * k * sizeof(float16_t), deviceData.data(), n * k * sizeof(float16_t), kind);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "CopyDeviceToHost failed: %i\n", ret);
    return APP_ERR_OK;
}

APP_ERROR IndexFlatATInt8Aicpu::computeL2Int8(int n, int k, float16_t *distances, idx_t *labels,
    AscendTensor<int8_t, DIMS_4> &queries, AscendTensor<int32_t, DIMS_1> &queriesNorms,
    AscendTensor<float16_t, DIMS_3> &distResult, AscendTensor<float16_t, DIMS_3> &minDistResult,
    AscendTensor<uint16_t, DIMS_3> &opFlag)
{
    auto streamPtr = resources.getDefaultStream();
    auto stream = streamPtr->GetStream();
    auto streamAicpuPtr = resources.getAlternateStreams()[0];
    auto streamAicpu = streamAicpuPtr->GetStream();
    auto &mem = resources.getMemoryManager();

    int batches = utils::divUp(n, queryBatch);
    AscendTensor<float16_t, DIMS_3> outDistances(mem, { batches, queryBatch, k }, streamAicpu);
    AscendTensor<int64_t, DIMS_3> outIndices(mem, { batches, queryBatch, k }, streamAicpu);
    AscendTensor<int64_t, DIMS_1> attrsInput(mem, { aicpu::TOPK_FLAT_ATTR_IDX_COUNT }, streamAicpu);

    auto ret = computeL2Int8Init(n, k, distResult, minDistResult, opFlag, attrsInput);
    APPERR_RETURN_IF_NOT_FMT(ret == APP_ERR_OK, ret, "computeL2Int8Init failed %d", ret);

    const int dim1 = utils::divUp(this->baseSize, CODE_ALIGN);
    const int dim2 = utils::divUp(this->dims, CUBE_INT8_ALIGN);
    const int coreNum = std::min(CORE_NUM, dim1);
    AscendTensor<uint32_t, DIMS_3> opSizeTopk(mem, { 1, coreNum, SIZE_ALIGN }, streamAicpu);
    opSizeTopk[0][0][0] = baseSize;
    for (int i = 0; i < batches; ++i) {
        // 1. run the topk operator to wait for distance result and compute topk
        AscendTensor<uint16_t, DIMS_3> opFlagTopk(opFlag.data() + i * CORE_NUM * FLAG_SIZE, { 1, CORE_NUM, FLAG_SIZE });
        AscendTensor<float16_t, DIMS_3> distResultTopk(distResult.data() + i * queryBatch * baseSize,
            { 1, queryBatch, baseSize });
        AscendTensor<float16_t, DIMS_3> minDistResultTopk(minDistResult.data() + i * queryBatch * bursts,
            { 1, queryBatch, bursts });
        AscendTensor<float16_t, DIMS_2> outDistancesTopk(outDistances.data() + i * queryBatch * k, { queryBatch, k });
        AscendTensor<int64_t, DIMS_2> outIndicesTopk(outIndices.data() + i * queryBatch * k, { queryBatch, k });

        runTopkCompute(distResultTopk, minDistResultTopk, opSizeTopk, opFlagTopk, attrsInput, outDistancesTopk,
            outIndicesTopk, streamAicpu);

        // 2. run the disance operator to compute the distance
        AscendTensor<int8_t, DIMS_4> query(queries.data() + i * queryBatch * dims,
            { queryBatch / QUERY_ALIGN, dim2, QUERY_ALIGN, CUBE_INT8_ALIGN });
        AscendTensor<int8_t, DIMS_4> shaped(codes.data(), { dim1, dim2, CODE_ALIGN, CUBE_INT8_ALIGN });
        AscendTensor<int32_t, DIMS_1> queryNorms(queriesNorms.data() + i * queryBatch, { queryBatch });
        AscendTensor<int32_t, DIMS_4> codeNorms(preComputeInt.data(),
            { baseSize / CODE_ALIGN, 1, CODE_ALIGN, CUBE_ALIGN });
        auto dist = distResult[i].view();
        auto minDist = minDistResult[i].view();
        auto flag = opFlag[i].view();

        std::vector<const AscendTensorBase *> input {&query, &shaped, &queryNorms, &codeNorms};
        std::vector<const AscendTensorBase *> output {&dist, &minDist, &flag};
        runInt8DistCompute(input, output, stream);
    }

    ret = synchronizeStream(stream);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "synchronize stream failed: %i\n", ret);

    ret = synchronizeStream(streamAicpu);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "synchronize streamAicpu failed: %i\n", ret);

    ret = CopyDisDeviceToHost(distances, n, k, outDistances);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "copy outDistances to host failed: %i\n", ret);

    ret = CopyLabelsDeviceToHost(labels, n, k, outIndices);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "copy outIndices to host failed: %i\n", ret);

    return APP_ERR_OK;
}

void IndexFlatATInt8Aicpu::runTopkCompute(AscendTensor<float16_t, DIMS_3> &dists,
    AscendTensor<float16_t, DIMS_3> &maxdists, AscendTensor<uint32_t, DIMS_3> &sizes,
    AscendTensor<uint16_t, DIMS_3> &flags, AscendTensor<int64_t, DIMS_1> &attrs,
    AscendTensor<float16_t, DIMS_2> &outdists, AscendTensor<int64_t, DIMS_2> &outlabel, aclrtStream stream)
{
    ASCEND_THROW_IF_NOT(topkComputeOps);

    std::shared_ptr<std::vector<const aclDataBuffer *>> topkOpInput(
        new std::vector<const aclDataBuffer *>(), CommonUtils::AclInputBufferDelete);
    topkOpInput->emplace_back(aclCreateDataBuffer(dists.data(), dists.getSizeInBytes()));
    topkOpInput->emplace_back(aclCreateDataBuffer(maxdists.data(), maxdists.getSizeInBytes()));
    topkOpInput->emplace_back(aclCreateDataBuffer(sizes.data(), sizes.getSizeInBytes()));
    topkOpInput->emplace_back(aclCreateDataBuffer(flags.data(), flags.getSizeInBytes()));
    topkOpInput->emplace_back(aclCreateDataBuffer(attrs.data(), attrs.getSizeInBytes()));

    std::shared_ptr<std::vector<aclDataBuffer *>> topkOpOutput(
        new std::vector<aclDataBuffer *>(), CommonUtils::AclOutputBufferDelete);
    topkOpOutput->emplace_back(aclCreateDataBuffer(outdists.data(), outdists.getSizeInBytes()));
    topkOpOutput->emplace_back(aclCreateDataBuffer(outlabel.data(), outlabel.getSizeInBytes()));

    topkComputeOps->exec(*topkOpInput, *topkOpOutput, stream);
}

void IndexFlatATInt8Aicpu::runL2NormTypingInt8Op(const std::vector<const AscendTensorBase *> &input,
                                                 const std::vector<const AscendTensorBase *> &output,
                                                 aclrtStream stream) const
{
    IndexTypeIdx indexType = IndexTypeIdx::ITI_L2_NORM_TYPING_INT8;
    std::vector<int> keys({searchPage, dims, baseSize});
    OpsMngKey opsKey(keys);
    auto ret = DistComputeOpsManager::getInstance().runOp(indexType, opsKey, input, output, stream);
    ASCEND_THROW_IF_NOT_FMT(ret == APP_ERR_OK, "run operator failed: %i\n", ret);
}

void IndexFlatATInt8Aicpu::runInt8DistCompute(const std::vector<const AscendTensorBase *> &input,
                                              const std::vector<const AscendTensorBase *> &output,
                                              aclrtStream stream) const
{
    IndexTypeIdx indexType = IndexTypeIdx::ITI_FLAT_L2_MINS_INT8_AT;
    std::vector<int> keys({queryBatch, dims, baseSize});
    OpsMngKey opsKey(keys);
    auto ret = DistComputeOpsManager::getInstance().runOp(indexType, opsKey, input, output, stream);
    ASCEND_THROW_IF_NOT_FMT(ret == APP_ERR_OK, "run operator failed: %i\n", ret);
}

APP_ERROR IndexFlatATInt8Aicpu::resetL2NormTypingInt8Op() const
{
    std::string opTypeName = "L2NormTypingInt8";
    IndexTypeIdx indexType = IndexTypeIdx::ITI_L2_NORM_TYPING_INT8;
    std::vector<int64_t> queryShape({ searchPage, dims });
    std::vector<int64_t> transferShape({ TRANSFER_SIZE, CUBE_ALIGN });
    std::vector<int64_t> sizeShape({ CORE_NUM, SIZE_ALIGN });
    std::vector<int64_t> resultShape({ searchPage });
    std::vector<int64_t> queryAlignShape({searchPage / QUERY_ALIGN,
        utils::divUp(dims, CUBE_INT8_ALIGN), QUERY_ALIGN, CUBE_INT8_ALIGN});
 
    std::vector<std::pair<aclDataType, std::vector<int64_t>>> input {
        { ACL_INT8, queryShape },
        { ACL_INT32, transferShape },
        { ACL_UINT32, sizeShape }
    };
    std::vector<std::pair<aclDataType, std::vector<int64_t>>> output {
        { ACL_INT32, resultShape },
        { ACL_INT8, queryAlignShape }
    };
    std::vector<int> keys({searchPage, dims, baseSize});
    OpsMngKey opsKey(keys);
    auto ret = DistComputeOpsManager::getInstance().resetOp(opTypeName, indexType, opsKey, input, output);
    APPERR_RETURN_IF_NOT_FMT(ret == APP_ERR_OK, ret, "op init failed: %i", ret);
 
    return APP_ERR_OK;
}

APP_ERROR IndexFlatATInt8Aicpu::resetDistL2MinsInt8AtOp() const
{
    std::string opTypeName = "DistanceL2MinsInt8At";
    IndexTypeIdx indexType = IndexTypeIdx::ITI_FLAT_L2_MINS_INT8_AT;
    std::vector<int64_t> queryShape({ queryBatch / QUERY_ALIGN,
        utils::divUp(dims, CUBE_INT8_ALIGN), QUERY_ALIGN, CUBE_INT8_ALIGN });
    std::vector<int64_t> codesShape({ utils::divUp(baseSize, CODE_ALIGN),
        utils::divUp(dims, CUBE_INT8_ALIGN), CODE_ALIGN, CUBE_INT8_ALIGN });
    std::vector<int64_t> queryNormsShape({ queryBatch });
    std::vector<int64_t> codeNormsShape({ baseSize / CODE_ALIGN, 1, CODE_ALIGN, CUBE_ALIGN });
    std::vector<int64_t> distResultShape({ queryBatch, baseSize });
    std::vector<int64_t> minResultShape({ queryBatch, this->bursts });
    std::vector<int64_t> flagShape({ CORE_NUM, FLAG_SIZE });
 
    std::vector<std::pair<aclDataType, std::vector<int64_t>>> input {
        { ACL_INT8, queryShape },
        { ACL_INT8, codesShape },
        { ACL_INT32, queryNormsShape },
        { ACL_INT32, codeNormsShape }
    };
    std::vector<std::pair<aclDataType, std::vector<int64_t>>> output {
        { ACL_FLOAT16, distResultShape },
        { ACL_FLOAT16, minResultShape },
        { ACL_UINT16, flagShape }
    };
    std::vector<int> keys({queryBatch, dims, baseSize});
    OpsMngKey opsKey(keys);
    auto ret = DistComputeOpsManager::getInstance().resetOp(opTypeName, indexType, opsKey, input, output);
    APPERR_RETURN_IF_NOT_FMT(ret == APP_ERR_OK, ret, "op init failed: %i", ret);
 
    return APP_ERR_OK;
}

void IndexFlatATInt8Aicpu::clearTmpAscendTensor()
{
    distResult.resetDataPtr();
    minDistResult.resetDataPtr();
    opFlag.resetDataPtr();
    minDistances.resetDataPtr();
    minIndices.resetDataPtr();
}

void IndexFlatATInt8Aicpu::updateQMinMax(float16_t qMin, float16_t qMax)
{
    this->qMin = qMin;
    this->qMax = qMax;
}
} // namespace ascend
