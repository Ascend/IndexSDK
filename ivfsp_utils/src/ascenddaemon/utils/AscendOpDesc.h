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


#ifndef ASCEND_OP_DESC_INCLUDED
#define ASCEND_OP_DESC_INCLUDED

#include <string>
#include <vector>

#include "acl/acl.h"

namespace ascendSearch {
class AscendOpDesc {
public:
    explicit AscendOpDesc(std::string opName);

    AscendOpDesc(AscendOpDesc &&desc);
    AscendOpDesc& operator=(AscendOpDesc &&desc)
    {
        opType = std::move(desc.opType);
        opAttr = desc.opAttr;
        desc.opAttr = nullptr;

        inputDesc = std::move(desc.inputDesc);
        outputDesc = std::move(desc.outputDesc);

        return *this;
    }

    ~AscendOpDesc();

    AscendOpDesc &addInputTensorDesc(aclDataType dataType, int numDims, const int64_t *dims, aclFormat format);
    AscendOpDesc &addOutputTensorDesc(aclDataType dataType, int numDims, const int64_t *dims, aclFormat format);

    std::string opType;
    std::vector<const aclTensorDesc *> inputDesc;
    std::vector<const aclTensorDesc *> outputDesc;
    aclopAttr *opAttr = nullptr;
};
}  // namespace ascendSearch

#endif  // ASCEND_OP_DESC_INCLUDED
