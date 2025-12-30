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


#include "index_custom/IndexFlatATAicpu.h"

#include "common/utils/OpLauncher.h"
#include "ops/cpukernel/impl/utils/kernel_shared_def.h"
#include "common/utils/CommonUtils.h"
#include "ascend/utils/fp16.h"

namespace ascend {

IndexFlatATAicpu::IndexFlatATAicpu(int dim, int baseSize, int64_t resourceSize)
    : IndexFlatAT(dim, baseSize, resourceSize) {}

IndexFlatATAicpu::~IndexFlatATAicpu() {}

APP_ERROR IndexFlatATAicpu::init()
{
    // reset operator
    APPERR_RETURN_IF_NOT_OK(resetTopkCompOp());
    APPERR_RETURN_IF_NOT_OK(resetL2NormOp());
    APPERR_RETURN_IF_NOT_OK(resetDistL2MinsAtOp());

    // init transfer data
    std::vector<float> transferDataTemp(TRANSFER_SIZE * CUBE_ALIGN, 0);
    for (int i = 0; i < TRANSFER_SIZE; ++i) {
        transferDataTemp[i * CUBE_ALIGN + (i % CUBE_ALIGN)] = 1;
    }
    AscendTensor<float, DIMS_2> transferData({ TRANSFER_SIZE, CUBE_ALIGN });
    auto ret = aclrtMemcpy(transferData.data(), transferData.getSizeInBytes(), transferDataTemp.data(),
        transferDataTemp.size() * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "Mem operator error %d", (int)ret);
    transfer = std::move(transferData);

    return APP_ERR_OK;
}

void IndexFlatATAicpu::setResultCopyBack(bool value)
{
    resultCopyBack = value;
}

APP_ERROR IndexFlatATAicpu::reset()
{
    codes.clear();
    preCompute.clear();
    this->ntotal = 0;

    return APP_ERR_OK;
}

APP_ERROR IndexFlatATAicpu::addVectors(const AscendTensor<float16_t, DIMS_2> &rawData)
{
    int num = rawData.getSize(0);
    APPERR_RETURN_IF(num == 0, APP_ERR_OK);

    auto ret = saveCodes(rawData);
    APPERR_RETURN_IF_NOT_FMT(ret == APP_ERR_OK, ret, "saveCodes failed: %d", ret);
    ret = saveNorms(rawData);
    APPERR_RETURN_IF_NOT_FMT(ret == APP_ERR_OK, ret, "saveNorms failed: %d", ret);
    this->ntotal += static_cast<idx_t>(num);

    return APP_ERR_OK;
}

APP_ERROR IndexFlatATAicpu::saveCodes(const AscendTensor<float16_t, DIMS_2> &rawData)
{
    int num = rawData.getSize(0);
    int dim = rawData.getSize(1);
    int total = static_cast<int64_t>(ntotal) + num;

    // 1. resize vector
    codes.resize(utils::roundUp(total, CUBE_ALIGN) * utils::roundUp(dim, CUBE_ALIGN), true);

    // 2. use aicpu to save codes
    std::string opName = "TransdataShaped";
    auto &mem = resources.getMemoryManager();
    auto streamPtr = resources.getDefaultStream();
    auto stream = streamPtr->GetStream();

    AscendTensor<float16_t, DIMS_4> dst((codes.data()),
        { utils::divUp(total, CODE_ALIGN), utils::divUp(dim, CUBE_ALIGN), CODE_ALIGN, CUBE_ALIGN });
    AscendTensor<int64_t, DIMS_1> attr(mem, { aicpu::TRANSDATA_SHAPED_ATTR_IDX_COUNT }, stream);
    attr[aicpu::TRANSDATA_SHAPED_ATTR_NTOTAL_IDX] = ntotal;

    LaunchOpTwoInOneOut<float16_t, DIMS_2, ACL_FLOAT16, int64_t, DIMS_1, ACL_INT64, float16_t, DIMS_4, ACL_FLOAT16>(
        opName, stream, rawData, attr, dst);

    auto ret = synchronizeStream(stream);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR,
        "synchronizeStream saveCodes stream failed: %i\n", ret);

    return APP_ERR_OK;
}

APP_ERROR IndexFlatATAicpu::saveNorms(const AscendTensor<float16_t, DIMS_2> &rawData)
{
    int num = rawData.getSize(0);

    // 1. resize vector
    preCompute.resize((ntotal + num) * CUBE_ALIGN, true);

    // 2. use aicpu to compute and save norm
    std::string opName = "VecL2Sqr";
    auto streamPtr = resources.getDefaultStream();
    auto stream = streamPtr->GetStream();
    AscendTensor<float, DIMS_2> dst(preCompute.data() + ntotal * CUBE_ALIGN, { num, CUBE_ALIGN });
    LaunchOpOneInOneOut<float16_t, DIMS_2, ACL_FLOAT16, float, DIMS_2, ACL_FLOAT>(opName, stream, rawData, dst);
    auto ret = synchronizeStream(stream);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "synchronizeStream failed: %i\n", ret);

    return APP_ERR_OK;
}

APP_ERROR IndexFlatATAicpu::searchBatched(int64_t n, const float16_t *x, int64_t k, float16_t *distance, idx_t *labels)
{
    APP_ERROR ret = APP_ERR_OK;

    int64_t pages = n / searchPage;
    for (int64_t i = 0; i < pages; i++) {
        ret = searchImpl(searchPage, x + i * searchPage * this->dims, k,
            distance + i * searchPage * k, labels + i * searchPage * k);
        APPERR_RETURN_IF(ret, ret);
    }

    int64_t queryLast = n % searchPage;
    if (queryLast > 0) {
        ret = searchImpl(queryLast, x + pages * searchPage * this->dims, k,
            distance + pages * searchPage * k, labels + pages * searchPage * k);
        APPERR_RETURN_IF(ret, ret);
    }

    return APP_ERR_OK;
}

APP_ERROR IndexFlatATAicpu::searchImpl(int n, const float16_t *x, int k, float16_t *distances, idx_t *labels)
{
    auto streamPtr = resources.getDefaultStream();
    auto stream = streamPtr->GetStream();
    auto &mem = resources.getMemoryManager();

    // 1. compute l2 norm of query
    AscendTensor<float16_t, DIMS_2> queries(const_cast<float16_t *>(x), { searchPage, dims });
    AscendTensor<uint32_t, DIMS_2> querySize(mem, { CORE_NUM, SIZE_ALIGN }, stream);
    AscendTensor<float, DIMS_1> queriesNorm(mem, { searchPage }, stream);
    AscendTensor<float16_t, DIMS_4> queriesTyping(
        mem, { searchPage / QUERY_ALIGN, dims / CUBE_ALIGN, QUERY_ALIGN, CUBE_ALIGN }, stream);
    querySize[0][0] = n;

    std::vector<const AscendTensorBase *> input {&queries, &transfer, &querySize};
    std::vector<const AscendTensorBase *> output {&queriesNorm, &queriesTyping};
    runL2NormOp(input, output, stream);

    auto ret = synchronizeStream(stream);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR,
        "synchronizeStream aicore stream failed: %i\n", ret);

    // 2. compute L2 distance
    ret = computeL2(n, k, distances, labels, queriesTyping, queriesNorm, distResult,
              minDistResult, opFlag);
    APPERR_RETURN_IF_NOT_FMT(ret == APP_ERR_OK, APP_ERR_INNER_ERROR, "computeL2 failed: %i\n", ret);

    return APP_ERR_OK;
}

APP_ERROR IndexFlatATAicpu::computeL2Init(int n, int k, AscendTensor<float16_t, DIMS_3> &distResult,
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
    attrs[aicpu::TOPK_FLAT_ATTR_BLOCK_SIZE] = baseSize;

    auto ret = aclrtMemcpy(attrsInput.data(), attrsInput.getSizeInBytes(), attrs.data(), attrs.size() * sizeof(int64_t),
        ACL_MEMCPY_HOST_TO_DEVICE);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "Mem operator error %d", (int)ret);

    return APP_ERR_OK;
}

APP_ERROR IndexFlatATAicpu::computeL2(int n, int k, float16_t *distances, idx_t *labels,
    AscendTensor<float16_t, DIMS_4> &queries, AscendTensor<float, DIMS_1> &queriesNorms,
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

    auto ret = computeL2Init(n, k, distResult, minDistResult, opFlag, attrsInput);
    APPERR_RETURN_IF_NOT_FMT(ret == APP_ERR_OK, ret, "computeL2Init failed %d", ret);

    const int dim1 = utils::divUp(this->baseSize, CODE_ALIGN);
    const int dim2 = utils::divUp(this->dims, CUBE_ALIGN);
    const int coreNum = std::min(CORE_NUM, dim1);
    AscendTensor<uint32_t, DIMS_3> opSizeTopk(mem, { 1, coreNum, SIZE_ALIGN }, streamAicpu);
    opSizeTopk[0][0][0] = baseSize;
    for (int i = 0; i < batches; ++i) {
        // 1. run the topk operator to wait for distance result and compute topk
        AscendTensor<uint16_t, DIMS_3> opFlagTopk(opFlag[i].data(), { 1, CORE_NUM, FLAG_SIZE });
        AscendTensor<float16_t, DIMS_3> distResultTopk(distResult[i].data(), { 1, queryBatch, baseSize });
        AscendTensor<float16_t, DIMS_3> minDistResultTopk(minDistResult[i].data(), { 1, queryBatch, bursts });
        AscendTensor<float16_t, DIMS_2> outDistancesTopk(outDistances[i].data(), { queryBatch, k });
        AscendTensor<int64_t, DIMS_2> outIndicesTopk(outIndices[i].data(), { queryBatch, k });

        runTopkCompute(distResultTopk, minDistResultTopk, opSizeTopk, opFlagTopk, attrsInput, outDistancesTopk,
            outIndicesTopk, streamAicpu);

        // 2. run the disance operator to compute the distance
        AscendTensor<float16_t, DIMS_4> query(queries.data() + i * queryBatch * dims,
            { queryBatch / QUERY_ALIGN, dim2, QUERY_ALIGN, CUBE_ALIGN });
        AscendTensor<float16_t, DIMS_4> shaped(codes.data(), { dim1, dim2, CODE_ALIGN, CUBE_ALIGN });
        AscendTensor<float, DIMS_1> queryNorms(queriesNorms.data() + i * queryBatch, { queryBatch });
        AscendTensor<float, DIMS_4> codeNorms(preCompute.data(), { baseSize / CODE_ALIGN, 1, CODE_ALIGN, CUBE_ALIGN });
        auto dist = distResult[i].view();
        auto minDist = minDistResult[i].view();
        auto flag = opFlag[i].view();

        std::vector<const AscendTensorBase *> input {&query, &shaped, &queryNorms, &codeNorms};
        std::vector<const AscendTensorBase *> output {&dist, &minDist, &flag};
        runDistCompute(input, output, stream);
    }

    ret = synchronizeStream(stream);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR,
        "synchronizeStream aicore stream failed: %i\n", ret);

    ret = synchronizeStream(streamAicpu);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR,
        "synchronizeStream aicpu stream failed: %i\n", ret);

    auto copyKind = resultCopyBack ? ACL_MEMCPY_DEVICE_TO_HOST : ACL_MEMCPY_DEVICE_TO_DEVICE;
    // memcpy data back from dev to host
    ret = aclrtMemcpy(distances, n * k * sizeof(float16_t), outDistances.data(), n * k * sizeof(float16_t), copyKind);
    APPERR_RETURN_IF_NOT_LOG(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "Failed to copy outDistances back to host");

    ret = aclrtMemcpy(labels, n * k * sizeof(int64_t), outIndices.data(), n * k * sizeof(int64_t), copyKind);
    APPERR_RETURN_IF_NOT_LOG(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "Failed to copy outDistances back to host");

    return APP_ERR_OK;
}

void IndexFlatATAicpu::runTopkCompute(AscendTensor<float16_t, DIMS_3> &dists, AscendTensor<float16_t, DIMS_3> &maxdists,
    AscendTensor<uint32_t, DIMS_3> &sizes, AscendTensor<uint16_t, DIMS_3> &flags, AscendTensor<int64_t, DIMS_1> &attrs,
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

void IndexFlatATAicpu::runL2NormOp(const std::vector<const AscendTensorBase *> &input,
                                   const std::vector<const AscendTensorBase *> &output,
                                   aclrtStream stream) const
{
    IndexTypeIdx indexType = IndexTypeIdx::ITI_L2_NORM;
    std::vector<int> keys({searchPage, dims, baseSize});
    OpsMngKey opsKey(keys);
    auto ret = DistComputeOpsManager::getInstance().runOp(indexType, opsKey, input, output, stream);
    ASCEND_THROW_IF_NOT_FMT(ret == APP_ERR_OK, "run operator failed: %i\n", ret);
}

void IndexFlatATAicpu::runDistCompute(const std::vector<const AscendTensorBase *> &input,
                                      const std::vector<const AscendTensorBase *> &output,
                                      aclrtStream stream) const
{
    IndexTypeIdx indexType = IndexTypeIdx::ITI_FLAT_L2_MINS_AT;
    std::vector<int> keys({queryBatch, dims, baseSize});
    OpsMngKey opsKey(keys);
    auto ret = DistComputeOpsManager::getInstance().runOp(indexType, opsKey, input, output, stream);
    ASCEND_THROW_IF_NOT_FMT(ret == APP_ERR_OK, "run operator failed: %i\n", ret);
}

APP_ERROR IndexFlatATAicpu::resetL2NormOp() const
{
    std::string opTypeName = "L2Norm";
    IndexTypeIdx indexType = IndexTypeIdx::ITI_L2_NORM;
    std::vector<int64_t> queryShape({ searchPage, dims });
    std::vector<int64_t> transferShape({ TRANSFER_SIZE, CUBE_ALIGN });
    std::vector<int64_t> sizeShape({ CORE_NUM, SIZE_ALIGN });
    std::vector<int64_t> resultShape({ searchPage });
    std::vector<int64_t> queryAlignShape({
        searchPage / QUERY_ALIGN, utils::divUp(dims, CUBE_ALIGN), QUERY_ALIGN, CUBE_ALIGN });

    std::vector<std::pair<aclDataType, std::vector<int64_t>>> input {
        { ACL_FLOAT16, queryShape },
        { ACL_FLOAT, transferShape },
        { ACL_UINT32, sizeShape }
    };
    std::vector<std::pair<aclDataType, std::vector<int64_t>>> output {
        { ACL_FLOAT, resultShape },
        { ACL_FLOAT16, queryAlignShape }
    };
    std::vector<int> keys({searchPage, dims, baseSize});
    OpsMngKey opsKey(keys);
    auto ret = DistComputeOpsManager::getInstance().resetOp(opTypeName, indexType, opsKey, input, output);
    APPERR_RETURN_IF_NOT_FMT(ret == APP_ERR_OK, ret, "op init failed: %i", ret);

    return APP_ERR_OK;
}

APP_ERROR IndexFlatATAicpu::resetDistL2MinsAtOp() const
{
    std::string opTypeName = "DistanceFlatL2MinsAt";
    IndexTypeIdx indexType = IndexTypeIdx::ITI_FLAT_L2_MINS_AT;
    std::vector<int64_t> queryShape({ queryBatch / QUERY_ALIGN,
        utils::divUp(dims, CUBE_ALIGN), QUERY_ALIGN, CUBE_ALIGN });
    std::vector<int64_t> codesShape({ utils::divUp(baseSize, CODE_ALIGN),
        utils::divUp(dims, CUBE_ALIGN), CODE_ALIGN, CUBE_ALIGN });
    std::vector<int64_t> queryNormsShape({ queryBatch });
    std::vector<int64_t> codeNormsShape({ baseSize / CODE_ALIGN, 1, CODE_ALIGN, CUBE_ALIGN });
    std::vector<int64_t> distResultShape({ queryBatch, baseSize });
    std::vector<int64_t> minResultShape({ queryBatch, this->bursts });
    std::vector<int64_t> flagShape({ CORE_NUM, FLAG_SIZE });

    std::vector<std::pair<aclDataType, std::vector<int64_t>>> input {
        { ACL_FLOAT16, queryShape },
        { ACL_FLOAT16, codesShape },
        { ACL_FLOAT, queryNormsShape },
        { ACL_FLOAT, codeNormsShape }
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

void IndexFlatATAicpu::clearTmpAscendTensor()
{
    distResult.resetDataPtr();
    minDistResult.resetDataPtr();
    opFlag.resetDataPtr();
    minDistances.resetDataPtr();
    minIndices.resetDataPtr();
}
} // namespace ascend
