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
#include <memory>

#include "ascenddaemon/utils/AscendTensor.h"
#include "common/utils/CommonUtils.h"

namespace ascendSearch {

template<typename T, int D, aclDataType TA, bool IsInputDesc = true>
void AddDesc(AscendOpDesc &desc, AscendTensor<T, D> &tensor)
{
    std::vector<int64_t> shape;
    for (int i = 0; i < D; i++) {
        shape.push_back(tensor.getSize(i));
    }
    if (IsInputDesc) {
        desc.addInputTensorDesc(TA, shape.size(), shape.data(), ACL_FORMAT_ND);
    } else {
        desc.addOutputTensorDesc(TA, shape.size(), shape.data(), ACL_FORMAT_ND);
    }
}

template<typename TI1, int DI1, aclDataType TAI1,
         typename TO1, int DO1, aclDataType TAO1>
void LaunchOpOneInOneOut(const std::string &opName, aclrtStream stream,
                         AscendTensor<TI1, DI1> &in1,
                         AscendTensor<TO1, DO1> &out1)
{
    AscendOpDesc desc(opName);
    AddDesc<TI1, DI1, TAI1>(desc, in1);
    AddDesc<TO1, DO1, TAO1, false>(desc, out1);

    auto op = CREATE_UNIQUE_PTR(AscendOperator, desc);
    if (!op->init()) {
        APP_LOG_ERROR("op init failed: %s", opName.c_str());
        return;
    }

    std::shared_ptr<std::vector<const aclDataBuffer *>> distOpInput(
        new std::vector<const aclDataBuffer *>(), CommonUtils::AclInputBufferDelete);
    distOpInput->emplace_back(aclCreateDataBuffer(in1.data(), in1.getSizeInBytes()));
    std::shared_ptr<std::vector<aclDataBuffer *>> distOpOutput(
        new std::vector<aclDataBuffer *>(), CommonUtils::AclOutputBufferDelete);
    distOpOutput->emplace_back(aclCreateDataBuffer(out1.data(), out1.getSizeInBytes()));
    op->exec(*distOpInput, *distOpOutput, stream);
}

template<typename TI1, int DI1, aclDataType TAI1,
         typename TI2, int DI2, aclDataType TAI2,
         typename TO1, int DO1, aclDataType TAO1>
void LaunchOpTwoInOneOut(const std::string &opName, aclrtStream stream,
                         AscendTensor<TI1, DI1> &in1,
                         AscendTensor<TI2, DI2> &in2,
                         AscendTensor<TO1, DO1> &out1)
{
    AscendOpDesc desc(opName);
    AddDesc<TI1, DI1, TAI1>(desc, in1);
    AddDesc<TI2, DI2, TAI2>(desc, in2);
    AddDesc<TO1, DO1, TAO1, false>(desc, out1);

    auto op = CREATE_UNIQUE_PTR(AscendOperator, desc);
    if (!op->init()) {
        APP_LOG_ERROR("op init failed: %s", opName.c_str());
        return;
    }

    std::shared_ptr<std::vector<const aclDataBuffer *>> distOpInput(
        new std::vector<const aclDataBuffer *>(), CommonUtils::AclInputBufferDelete);
    distOpInput->emplace_back(aclCreateDataBuffer(in1.data(), in1.getSizeInBytes()));
    distOpInput->emplace_back(aclCreateDataBuffer(in2.data(), in2.getSizeInBytes()));
    std::shared_ptr<std::vector<aclDataBuffer *>> distOpOutput(
        new std::vector<aclDataBuffer *>(), CommonUtils::AclOutputBufferDelete);
    distOpOutput->emplace_back(aclCreateDataBuffer(out1.data(), out1.getSizeInBytes()));
    op->exec(*distOpInput, *distOpOutput, stream);
}
}

#endif // ASCEND_OP_LAUNCHER_H
