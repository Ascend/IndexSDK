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


#ifndef ASCEND_OP_LAUNCHER_H
#define ASCEND_OP_LAUNCHER_H

#include <string>
#include <vector>

#include "npu/common/AscendTensor.h"
#include "AscendOpDesc.h"
#include "AscendOperator.h"
#include "CommonUtils.h"
#include "LogUtils.h"

namespace ascendSearchacc {
template <typename TI1, int DI1, aclDataType TAI1, typename TO1, int DO1, aclDataType TAO1>
void LaunchOpOneInOneOut(const std::string &opName, aclrtStream stream, AscendTensor<TI1, DI1> &in1,
                         AscendTensor<TO1, DO1> &out1)
{
    AscendOpDesc desc(opName);
    std::vector<int64_t> shapeInput1;
    for (int i = 0; i < DI1; i++) {
        shapeInput1.push_back(in1.getSize(i));
    }
    desc.addInputTensorDesc(TAI1, shapeInput1.size(), shapeInput1.data(), ACL_FORMAT_ND);
    std::vector<int64_t> shapeOutput1;
    for (int i = 0; i < DO1; i++) {
        shapeOutput1.push_back(out1.getSize(i));
    }
    desc.addOutputTensorDesc(TAO1, shapeOutput1.size(), shapeOutput1.data(), ACL_FORMAT_ND);
    auto op = CREATE_UNIQUE_PTR(AscendOperator, desc);
    if (!op->init()) {
        APP_LOG_ERROR("op Init failed: %s", opName.c_str());
        return;
    }

    std::vector<const aclDataBuffer *> distOpInput;
    distOpInput.emplace_back(aclCreateDataBuffer(in1.data(), in1.getSizeInBytes()));
    std::vector<aclDataBuffer *> distOpOutput;
    distOpOutput.emplace_back(aclCreateDataBuffer(out1.data(), out1.getSizeInBytes()));
    op->exec(distOpInput, distOpOutput, stream);

    for (auto &item : distOpInput) {
        ACL_REQUIRE_OK(aclDestroyDataBuffer(item));
    }
    distOpInput.clear();
    for (auto &item : distOpOutput) {
        ACL_REQUIRE_OK(aclDestroyDataBuffer(item));
    }
    distOpOutput.clear();
}

template <typename TI1, int DI1, aclDataType TAI1, typename TI2, int DI2, aclDataType TAI2, typename TO1, int DO1,
          aclDataType TAO1>
void LaunchOpTwoInOneOut(const std::string &opName, aclrtStream stream, AscendTensor<TI1, DI1> &in1,
                         AscendTensor<TI2, DI2> &in2, AscendTensor<TO1, DO1> &out1)
{
    AscendOpDesc desc(opName);
    std::vector<int64_t> shapeInput1;
    for (int i = 0; i < DI1; i++) {
        shapeInput1.push_back(in1.getSize(i));
    }
    desc.addInputTensorDesc(TAI1, shapeInput1.size(), shapeInput1.data(), ACL_FORMAT_ND);
    std::vector<int64_t> shapeInput2;
    for (int i = 0; i < DI2; i++) {
        shapeInput2.push_back(in2.getSize(i));
    }
    desc.addInputTensorDesc(TAI2, shapeInput2.size(), shapeInput2.data(), ACL_FORMAT_ND);
    std::vector<int64_t> shapeOutput1;
    for (int i = 0; i < DO1; i++) {
        shapeOutput1.push_back(out1.getSize(i));
    }
    desc.addOutputTensorDesc(TAO1, shapeOutput1.size(), shapeOutput1.data(), ACL_FORMAT_ND);
    auto op = CREATE_UNIQUE_PTR(AscendOperator, desc);
    if (!op->init()) {
        APP_LOG_ERROR("op Init failed: %s", opName.c_str());
        return;
    }

    std::vector<const aclDataBuffer *> distOpInput;
    distOpInput.emplace_back(aclCreateDataBuffer(in1.data(), in1.getSizeInBytes()));
    distOpInput.emplace_back(aclCreateDataBuffer(in2.data(), in2.getSizeInBytes()));
    std::vector<aclDataBuffer *> distOpOutput;
    distOpOutput.emplace_back(aclCreateDataBuffer(out1.data(), out1.getSizeInBytes()));
    op->exec(distOpInput, distOpOutput, stream);

    for (auto &item : distOpInput) {
        ACL_REQUIRE_OK(aclDestroyDataBuffer(item));
    }
    distOpInput.clear();
    for (auto &item : distOpOutput) {
        ACL_REQUIRE_OK(aclDestroyDataBuffer(item));
    }
    distOpOutput.clear();
}
}  // namespace ascendSearchacc

#endif  // ASCEND_OP_LAUNCHER_H
