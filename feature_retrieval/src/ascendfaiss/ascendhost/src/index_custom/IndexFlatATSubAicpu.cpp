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


#include "index_custom/IndexFlatATSubAicpu.h"

#include "common/utils/CommonUtils.h"
#include "common/utils/OpLauncher.h"
#include "ascend/utils/fp16.h"
#include "ops/cpukernel/impl/utils/kernel_shared_def.h"

namespace ascend {
namespace {
const int QUERY_ALIGN = 32;
const int CODE_ALIGN = 64;
const int SUBCENTER_NUM = 64;
const int TYPICAL_DIM = 256;
const int SEARCH_PAGE = 32768;
const int CORE_NUM_SUB = 8;
}

IndexFlatATSubAicpu::IndexFlatATSubAicpu(int dim, int baseSize, int64_t resourceSize)
    : Index(dim, resourceSize),
      baseSize(baseSize)
{
    ASCEND_THROW_IF_NOT(dims % CUBE_ALIGN == 0);
    this->searchPage = SEARCH_PAGE;
    isTrained = true;
}

IndexFlatATSubAicpu::~IndexFlatATSubAicpu() {}

APP_ERROR IndexFlatATSubAicpu::init()
{
    // reset operator
    APPERR_RETURN_IF_NOT_OK(resetDistL2AtOp());
    APPERR_RETURN_IF_NOT_OK(resetL2NormFlatSubOp());
    int len = utils::roundUp(SUBCENTER_NUM, CUBE_ALIGN) * utils::roundUp(TYPICAL_DIM, CUBE_ALIGN);
    codes.resize(static_cast<size_t>(len), true);

    len = SUBCENTER_NUM * CUBE_ALIGN;
    preCompute.resize(static_cast<size_t>(len), true);

    return APP_ERR_OK;
}

APP_ERROR IndexFlatATSubAicpu::reset()
{
    this->ntotal = 0;
    return APP_ERR_OK;
}

APP_ERROR IndexFlatATSubAicpu::addVectors(int num, int dim, const AscendTensor<float16_t, DIMS_2> &deviceData)
{
    APPERR_RETURN_IF(num == 0, APP_ERR_OK);

    int length = num * CUBE_ALIGN;
    preCompute.resize(length, true);

    length = utils::roundUp(num, CUBE_ALIGN) * utils::roundUp(dim, CUBE_ALIGN);
    codes.resize(length, true);

    auto streamPtr = resources.getDefaultStream();
    auto stream = streamPtr->GetStream();

    AscendTensor<float, DIMS_2> outNorm(preCompute.data(), { num, CUBE_ALIGN });
    AscendTensor<float16_t, DIMS_1> outTyping(codes.data(), { length });

    std::vector<const AscendTensorBase *> input {&deviceData};
    std::vector<const AscendTensorBase *> output {&outNorm, &outTyping};
    runL2NormFlatSubOp(input, output, stream);
    auto ret = synchronizeStream(stream);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR,
        "synchronizeStream aicore stream failed: %i\n", ret);

    this->ntotal += static_cast<idx_t>(num);

    return APP_ERR_OK;
}

APP_ERROR IndexFlatATSubAicpu::searchBatched(int n, const uint8_t *x, int k, int ncentroids, uint16_t *labels,
    AscendTensor<float16_t, DIMS_4>& subcentroids, AscendTensor<int16_t, DIMS_3>& hassign,
    AscendTensor<float16_t, DIMS_2>& vDM)
{
    APP_ERROR ret = APP_ERR_OK;

    CommonUtils::attachToCpus({ 5 });
    int pages = n / searchPage;
    for (int i = 0; i < pages; i++) {
        auto subcentroidsTmp = subcentroids[i].view();
        auto hassignTmp = hassign[i].view();
        ret = searchImpl(searchPage, x + i * searchPage * this->dims, k, ncentroids,
            labels + i * searchPage * k, subcentroidsTmp, hassignTmp, vDM);
        APPERR_RETURN_IF(ret, ret);
    }

    int queryLast = n % searchPage;
    if (queryLast > 0) {
        auto subcentroidsTmp = subcentroids[pages].view();
        auto hassignTmp = hassign[pages].view();
        ret = searchImpl(queryLast, x + pages * searchPage * this->dims, k, ncentroids,
            labels + pages * searchPage * k, subcentroidsTmp, hassignTmp, vDM);
        APPERR_RETURN_IF(ret, ret);
    }
    CommonUtils::attachToCpus({ 0, 1, 2, 3, 4, 5 });
    return APP_ERR_OK;
}

APP_ERROR IndexFlatATSubAicpu::searchImpl(int n, const uint8_t *x, int, int,
    uint16_t *labels, AscendTensor<float16_t, DIMS_3>& subcentroids,
    AscendTensor<int16_t, DIMS_2>& hassign, AscendTensor<float16_t, DIMS_2>& vDM)
{
    auto streamPtr = resources.getDefaultStream();
    auto stream = streamPtr->GetStream();
    auto &mem = resources.getMemoryManager();

    AscendTensor<uint16_t, DIMS_2> opFlag(mem, {CORE_NUM, FLAG_SIZE }, stream);

    // 2. run the disance operator to compute the distance
    const int dim1 = utils::divUp(this->baseSize, CODE_ALIGN);
    const int dim2 = utils::divUp(this->dims, CUBE_ALIGN);

    AscendTensor<uint8_t, DIMS_2> query(const_cast<uint8_t*>(x), {1, this->dims});
    AscendTensor<uint32_t, DIMS_1> actualSize(mem, {CORE_NUM_SUB, }, stream);
    std::vector<uint32_t> actualSizeHost(CORE_NUM_SUB, 0);
    actualSizeHost[0] = static_cast<uint32_t>(n);
    actualSizeHost[1] = std::min(CORE_NUM_SUB, n / QUERY_ALIGN);
    int ret = aclrtMemcpy(actualSize.data(), actualSize.getSizeInBytes(),
        actualSizeHost.data(), actualSize.getSizeInBytes(), ACL_MEMCPY_HOST_TO_DEVICE);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR,
        "Failed to copy actualSize to device: %i\n", ret);

    AscendTensor<float16_t, DIMS_4> shaped(codes.data(), { dim1, dim2, CODE_ALIGN, CUBE_ALIGN });
    AscendTensor<float, DIMS_4> codeNorms(preCompute.data(), { baseSize / CODE_ALIGN, 1, CODE_ALIGN, CUBE_ALIGN });
    AscendTensor<uint16_t, DIMS_1> outIndices(labels, { 1 });

    std::vector<const AscendTensorBase *> input {&query, &actualSize, &shaped, &codeNorms, &vDM};
    std::vector<const AscendTensorBase *> output {&subcentroids, &hassign, &outIndices, &opFlag};
    runDistComputeInt8(input, output, stream);

    ret = synchronizeStream(stream);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR,
        "synchronizeStream aicore stream failed: %i\n", ret);

    return APP_ERR_OK;
}

APP_ERROR IndexFlatATSubAicpu::searchImpl(int n, const float16_t* x, int k, float16_t* distances, idx_t * labels)
{
    ASCEND_THROW_MSG("searchImpl() not implemented for this type of index");
    VALUE_UNUSED(n);
    VALUE_UNUSED(x);
    VALUE_UNUSED(k);
    VALUE_UNUSED(distances);
    VALUE_UNUSED(labels);
    return APP_ERR_OK;
}

APP_ERROR IndexFlatATSubAicpu::search(idx_t n, const uint8_t *x, idx_t k, int ncentroids,
    uint16_t *labels, AscendTensor<float16_t, DIMS_4>& subcentroids,
    AscendTensor<int16_t, DIMS_3>& hassign, AscendTensor<float16_t, DIMS_2>& vDM)
{
    APPERR_RETURN_IF_NOT_LOG(x, APP_ERR_INVALID_PARAM, "x can not be nullptr.");
    APPERR_RETURN_IF_NOT_LOG(this->isTrained, APP_ERR_INVALID_PARAM, "Index not trained.");
    APPERR_RETURN_IF_NOT_FMT(n <= static_cast<idx_t>(std::numeric_limits<int>::max()),
        APP_ERR_INVALID_PARAM, "indices exceeds max(%d)", std::numeric_limits<int>::max());
    APPERR_RETURN_IF(n == 0 || k == 0, APP_ERR_OK);

    return searchBatched(n, x, k, ncentroids, labels, subcentroids, hassign, vDM);
}

APP_ERROR IndexFlatATSubAicpu::resetL2NormFlatSubOp()
{
    AscendOpDesc desc("L2NormFlatSub");

    std::vector<int64_t> queryShape({ SUBCENTER_NUM, dims });
    desc.addInputTensorDesc(ACL_FLOAT16, queryShape.size(), queryShape.data(), ACL_FORMAT_ND);

    std::vector<int64_t> queryNormShape({ SUBCENTER_NUM, CUBE_ALIGN });
    desc.addOutputTensorDesc(ACL_FLOAT, queryNormShape.size(), queryNormShape.data(), ACL_FORMAT_ND);

    std::vector<int64_t> queryTypingShape({ SUBCENTER_NUM * dims });
    desc.addOutputTensorDesc(ACL_FLOAT16, queryTypingShape.size(), queryTypingShape.data(), ACL_FORMAT_ND);

    l2NormFlatSub.reset();
    l2NormFlatSub = CREATE_UNIQUE_PTR(AscendOperator, desc);
    APPERR_RETURN_IF_NOT_LOG(l2NormFlatSub->init(), APP_ERR_ACL_OP_LOAD_MODEL_FAILED, "l2NormFlatSub op init failed");
 
    return APP_ERR_OK;
}

void IndexFlatATSubAicpu::runL2NormFlatSubOp(const std::vector<const AscendTensorBase *> &input,
                                             const std::vector<const AscendTensorBase *> &output,
                                             aclrtStream stream)
{
    ASCEND_THROW_IF_NOT(l2NormFlatSub);

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

    l2NormFlatSub->exec(*distSqOpInput, *distSqOpOutput, stream);
}

APP_ERROR IndexFlatATSubAicpu::resetDistL2AtOp()
{
    AscendOpDesc desc("DistanceFlatL2At");
    std::vector<int64_t> queryShape({1, dims});
    std::vector<int64_t> actualSizeShape({8, });
    std::vector<int64_t> codesShape({ utils::divUp(baseSize, CODE_ALIGN),
        utils::divUp(dims, CUBE_ALIGN), CODE_ALIGN, CUBE_ALIGN });
    std::vector<int64_t> codeNormsShape({ baseSize / CODE_ALIGN, 1, CODE_ALIGN, CUBE_ALIGN });
    std::vector<int64_t> vdmShape({2, dims});

    std::vector<int64_t> subCentroidsCoresShape({CORE_NUM, baseSize, dims});
    std::vector<int64_t> hassignCoresShape({CORE_NUM, baseSize});
    std::vector<int64_t> hassignQueriesShape({1, });
    std::vector<int64_t> flagShape({ CORE_NUM, FLAG_SIZE });

    desc.addInputTensorDesc(ACL_UINT8, queryShape.size(), queryShape.data(), ACL_FORMAT_ND);
    desc.addInputTensorDesc(ACL_UINT32, actualSizeShape.size(), actualSizeShape.data(), ACL_FORMAT_ND);
    desc.addInputTensorDesc(ACL_FLOAT16, codesShape.size(), codesShape.data(), ACL_FORMAT_ND);
    desc.addInputTensorDesc(ACL_FLOAT, codeNormsShape.size(), codeNormsShape.data(), ACL_FORMAT_ND);
    desc.addInputTensorDesc(ACL_FLOAT16, vdmShape.size(), vdmShape.data(), ACL_FORMAT_ND);

    desc.addOutputTensorDesc(ACL_FLOAT16, subCentroidsCoresShape.size(), subCentroidsCoresShape.data(), ACL_FORMAT_ND);
    desc.addOutputTensorDesc(ACL_INT16, hassignCoresShape.size(), hassignCoresShape.data(), ACL_FORMAT_ND);
    desc.addOutputTensorDesc(ACL_UINT16, hassignQueriesShape.size(), hassignQueriesShape.data(), ACL_FORMAT_ND);
    desc.addOutputTensorDesc(ACL_UINT16, flagShape.size(), flagShape.data(), ACL_FORMAT_ND);

    flatL2Op.reset();
    flatL2Op = CREATE_UNIQUE_PTR(AscendOperator, desc);
    APPERR_RETURN_IF_NOT_LOG(flatL2Op->init(), APP_ERR_ACL_OP_LOAD_MODEL_FAILED, "op init failed");
 
    return APP_ERR_OK;
}

void IndexFlatATSubAicpu::runDistComputeInt8(const std::vector<const AscendTensorBase *> &input,
                                             const std::vector<const AscendTensorBase *> &output,
                                             aclrtStream stream)
{
    ASCEND_THROW_IF_NOT(flatL2Op);

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

    flatL2Op->exec(*distSqOpInput, *distSqOpOutput, stream);
}
} // namespace ascend