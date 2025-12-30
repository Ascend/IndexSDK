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


#include "ascenddaemon/impl/Int8L2Norm.h"

#include <algorithm>
#include "ascend/utils/fp16.h"
#include "ascenddaemon/AscendResourcesProxy.h"
#include "ascenddaemon/impl/Index.h"
#include "ascenddaemon/utils/Limits.h"
#include "ascenddaemon/utils/StaticUtils.h"
#include "common/utils/AscendAssert.h"
#include "common/utils/CommonUtils.h"

namespace ascend {
namespace {
const int TRANSFER_SIZE = 256;
const int COMPUTE_BATCH = 16384;
const int L2NORM_OP_INPUT_NUM = 3;
const int L2NORM_OP_OUTPUT_NUM = 1;
}

Int8L2Norm::Int8L2Norm(int d) : dims(d)
{
    AscendTensor<float16_t, DIMS_2> transDevice({TRANSFER_SIZE, CUBE_ALIGN});
#ifdef HOSTCPU
    std::vector<float> trans(TRANSFER_SIZE * CUBE_ALIGN, 0);
    for (int i = 0; i < TRANSFER_SIZE / CUBE_ALIGN; ++i) {
        for (int j = 0; j < CUBE_ALIGN; ++j) {
            trans[i * CUBE_ALIGN * CUBE_ALIGN + j * CUBE_ALIGN + j] = 1;
        }
    }
    std::vector<float16_t> trans16(TRANSFER_SIZE * CUBE_ALIGN);
    transform(trans.begin(), trans.end(), trans16.begin(), [] (float tmp) {
        return faiss::ascend::fp16(tmp).data;
    });
    auto ret = aclrtMemcpy(transDevice.data(), transDevice.getSizeInBytes(), trans16.data(),
        TRANSFER_SIZE * CUBE_ALIGN * sizeof(float16_t), ACL_MEMCPY_HOST_TO_DEVICE);
    ASCEND_THROW_IF_NOT_FMT(ret == ACL_SUCCESS, "Mem operator error %d", (int)ret);
#else
    std::vector<float16_t> trans(TRANSFER_SIZE * CUBE_ALIGN, 0);
    for (int i = 0; i < TRANSFER_SIZE / CUBE_ALIGN; ++i) {
        for (int j = 0; j < CUBE_ALIGN; ++j) {
            trans[i * CUBE_ALIGN * CUBE_ALIGN + j * CUBE_ALIGN + j] = 1;
        }
    }
    auto ret = memcpy_s(transDevice.data(), transDevice.getSizeInBytes(),
        trans.data(), TRANSFER_SIZE * CUBE_ALIGN * sizeof(float16_t));
    ASCEND_THROW_IF_NOT_FMT(ret == EOK, "memcpy_s error %d", ret);
#endif
    transfer = std::move(transDevice);
}

Int8L2Norm::~Int8L2Norm() {}

APP_ERROR Int8L2Norm::init()
{
    return resetL2NormOperator();
}

void Int8L2Norm::runL2NormOperator(AscendTensor<int8_t, DIMS_2> &vectors,
                                   AscendTensor<float16_t, DIMS_2> &transfer,
                                   AscendTensor<uint32_t, DIMS_1> &actualNum,
                                   AscendTensor<float16_t, DIMS_1> &result,
                                   aclrtStream stream)
{
    ASCEND_THROW_IF_NOT(l2NormOp);

    std::shared_ptr<std::vector<const aclDataBuffer *>> l2NormOpInput(
        new std::vector<const aclDataBuffer *>(), CommonUtils::AclInputBufferDelete);
    // prepare for input data's buffer
    l2NormOpInput->emplace_back(aclCreateDataBuffer(vectors.data(), vectors.getSizeInBytes())); // input 0
    l2NormOpInput->emplace_back(aclCreateDataBuffer(transfer.data(), transfer.getSizeInBytes()));     // input 1
    l2NormOpInput->emplace_back(aclCreateDataBuffer(actualNum.data(), actualNum.getSizeInBytes()));     // input 2

    std::shared_ptr<std::vector<aclDataBuffer *>> l2NormOpOutput(
        new std::vector<aclDataBuffer *>(), CommonUtils::AclOutputBufferDelete);
    // prepare for output data's buffer
    l2NormOpOutput->emplace_back(aclCreateDataBuffer(result.data(), result.getSizeInBytes()));  // output 0
    
    // async executing operator
    l2NormOp->exec(*l2NormOpInput, *l2NormOpOutput, stream);
}

APP_ERROR Int8L2Norm::resetL2NormOperator()
{
    auto l2NormOpReset = [&](std::unique_ptr<AscendOperator> &l2NormOp, int vectorNum) {
        std::string opName = faiss::ascend::SocUtils::GetInstance().IsAscend910B() ? "AscendcL2Norm" : "Int8L2Norm";
        AscendOpDesc desc(opName);
        std::vector<int64_t> vectorShape({ vectorNum, dims });
        std::vector<int64_t> transferShape({ TRANSFER_SIZE, CUBE_ALIGN });
        std::vector<int64_t> actualNumShape({ SIZE_ALIGN });
        std::vector<int64_t> resultShape({ vectorNum });

        desc.addInputTensorDesc(ACL_INT8, vectorShape.size(), vectorShape.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_FLOAT16, transferShape.size(), transferShape.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_UINT32, actualNumShape.size(), actualNumShape.data(), ACL_FORMAT_ND);
        desc.addOutputTensorDesc(ACL_FLOAT16, resultShape.size(), resultShape.data(), ACL_FORMAT_ND);

        l2NormOp.reset();
        l2NormOp = CREATE_UNIQUE_PTR(AscendOperator, desc);
        return l2NormOp->init();
    };

    l2NormOp = std::unique_ptr<AscendOperator>(nullptr);
    APPERR_RETURN_IF_NOT_LOG(l2NormOpReset(l2NormOp, COMPUTE_BATCH),
                             APP_ERR_ACL_OP_LOAD_MODEL_FAILED, "op init failed");

    return APP_ERR_OK;
}

void Int8L2Norm::dispatchL2NormTask(AscendTensor<int8_t, DIMS_2> &codesData,
                                    AscendTensor<float16_t, DIMS_1> &normData,
                                    AscendTensor<uint32_t, DIMS_2> &actualNum,
                                    aclrtStream stream)
{
    ASCEND_THROW_IF_NOT(normData.getSize(0) % CUBE_ALIGN == 0);

    // dispatch the l2 norm task
    int offset = 0;
    int codeSize = codesData.getSize(0);
    int times = utils::divUp(codeSize, COMPUTE_BATCH);
    for (int t = 0; t < times; ++t) {
        int size = std::min(codeSize - offset, COMPUTE_BATCH);
        int8_t* pCodes = codesData[offset].data();
        AscendTensor<int8_t, DIMS_2> vectorData(pCodes, { COMPUTE_BATCH, dims });
        auto actualSize = actualNum[t].view();
        actualSize[0] = static_cast<uint32_t>(size);
        AscendTensor<float16_t, DIMS_1> result(normData.data() + offset, { COMPUTE_BATCH });

        runL2NormOperator(vectorData, transfer, actualSize, result, stream);
        offset += COMPUTE_BATCH;
    }
}
} // namespace ascend