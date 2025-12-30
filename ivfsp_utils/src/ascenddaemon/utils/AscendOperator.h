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


#ifndef ASCEND_OPERATOR_INCLUDED
#define ASCEND_OPERATOR_INCLUDED

#include <ascenddaemon/utils/AscendOpDesc.h>
#include <vector>

namespace ascendSearch {
class AscendOperator {
public:
    explicit AscendOperator(AscendOpDesc &desc);

    ~AscendOperator();

    bool init();

    void exec(std::vector<const aclDataBuffer *>& inputBuffers,
              std::vector<aclDataBuffer *>& outputBuffers, aclrtStream stream) const;

    inline int getNumInputs() const
    {
        return numInputs;
    }

    inline int getNumOutputs() const
    {
        return numOutputs;
    }

    size_t getInputNumDims(int index);

    int64_t getInputDim(int index, int dimIndex);

    size_t getInputSize(int index);

    size_t getOutputNumDims(int index);

    int64_t getOutputDim(int index, int dimIndex);

    size_t getOutputSize(int index);

private:
    AscendOpDesc opDesc;
    aclopHandle *handle;
    int numInputs;
    int numOutputs;
};
}  // namespace ascendSearch

#endif  // ASCEND_OPERATOR_INCLUDED
