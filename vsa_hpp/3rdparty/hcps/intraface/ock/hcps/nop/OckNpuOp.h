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

#ifndef OCK_HCPS_NOP_NPU_OP_H
#define OCK_HCPS_NOP_NPU_OP_H

#include <string>
#include <vector>
#include <memory>
#include "acl/acl.h"
#include "ock/log/OckHcpsLogger.h"

namespace ock {
namespace hcps {
namespace nop {
struct NpuParamsInfo {
    std::vector<std::vector<int64_t>> inputParamShapes{};
    std::vector<std::vector<int64_t>> outputParamShapes{};
    std::vector<aclDataType> inputParamTypes{};
    std::vector<aclDataType> outputParamTypes{};
    std::vector<aclFormat> inputParamFormats{};
    std::vector<aclFormat> outputParamFormats{};
};

class OckNpuOp {
public:
    explicit OckNpuOp(std::string operatorType, const std::shared_ptr<NpuParamsInfo> &parameters)
        : opType(std::move(operatorType)), paramsInfo(parameters)
    {
        opAttr = aclopCreateAttr();
    }

    OckNpuOp(const OckNpuOp &other)
    {
        this->opType = other.opType;
        this->paramsInfo = other.paramsInfo;
        if (handle != nullptr) {
            aclopDestroyHandle(handle);
            handle = nullptr;
        }
        if (opAttr != nullptr) {
            aclopDestroyAttr(opAttr);
            opAttr = nullptr;
        }
        opAttr = aclopCreateAttr();
    }

    OckNpuOp &operator=(const OckNpuOp &other)
    {
        this->opType = other.opType;
        this->paramsInfo = other.paramsInfo;
        if (handle != nullptr) {
            aclopDestroyHandle(handle);
            handle = nullptr;
        }
        if (opAttr != nullptr) {
            aclopDestroyAttr(opAttr);
            opAttr = nullptr;
        }
        opAttr = aclopCreateAttr();
        return *this;
    }

    ~OckNpuOp()
    {
        if (handle != nullptr) {
            aclopDestroyHandle(handle);
            handle = nullptr;
        }
        DestroyTensorDesc(inputDesc);
        DestroyTensorDesc(outputDesc);
        if (opAttr != nullptr) {
            aclopDestroyAttr(opAttr);
            opAttr = nullptr;
        }
    }
    aclError Enable()
    {
        FillDesc(inputDesc, paramsInfo->inputParamShapes, paramsInfo->inputParamTypes, paramsInfo->inputParamFormats);
        FillDesc(outputDesc, paramsInfo->outputParamShapes, paramsInfo->outputParamTypes,
            paramsInfo->outputParamFormats);

        if (opAttr == nullptr) {
            OCK_HCPS_LOG_ERROR("opAttr is for OckNpuOp " << opType << " is nullptr!");
            return ACL_ERROR_GE_PARAM_INVALID;
        }
        auto ret = aclopCreateHandle(opType.c_str(), static_cast<int>(inputDesc.size()), inputDesc.data(),
            static_cast<int>(outputDesc.size()), outputDesc.data(), opAttr, &handle);
        if (ret != ACL_ERROR_NONE) {
            OCK_HCPS_LOG_ERROR("the result of aclopCreateHandle() is err, opType=" << opType.c_str() << ", ret=" <<
                ret);
        }
        return ret;
    }
    std::string opType{ "" };
    std::shared_ptr<NpuParamsInfo> paramsInfo{ nullptr };
    std::vector<const aclTensorDesc *> inputDesc{};
    std::vector<const aclTensorDesc *> outputDesc{};
    aclopHandle *handle{ nullptr };

private:
    void DestroyTensorDesc(std::vector<const aclTensorDesc *> descriptions)
    {
        for (auto desc : descriptions) {
            aclDestroyTensorDesc(desc);
        }
    }
    static void FillDesc(std::vector<const aclTensorDesc *> &desc, const std::vector<std::vector<int64_t>> &shapes,
        const std::vector<aclDataType> &types, const std::vector<aclFormat> &formats)
    {
        for (size_t i = 0; i < shapes.size(); ++i) {
            auto *aclDesc = aclCreateTensorDesc(types[i], static_cast<int>(shapes[i].size()), shapes[i].data(),
                formats[i]);
            if (aclDesc == nullptr) {
                OCK_HCPS_LOG_ERROR("aclCreateTensorDesc for OckNpuOp failed!");
                return;
            }
            desc.push_back(aclDesc);
        }
    }

    aclopAttr *opAttr{ nullptr };
};
} // namespace nop
} // namespace hcps
} // namespace ock
#endif // OCK_HCPS_NOP_NPU_OP_H