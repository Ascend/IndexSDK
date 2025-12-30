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


#ifndef ASCEND_INT8_L2_NORM_INCLUDED
#define ASCEND_INT8_L2_NORM_INCLUDED

#include <map>
#include <vector>
#include <memory>
#include <cstdint>
#include <cstddef>

#include "ascenddaemon/utils/AscendTensor.h"
#include "ascenddaemon/utils/AscendOperator.h"
#include "common/AscendFp16.h"
#include "common/ErrorCode.h"

namespace ascend {
struct IDSelector;

class Int8L2Norm {
public:
    explicit Int8L2Norm(int d = 0);

    ~Int8L2Norm();

    APP_ERROR init();

    void dispatchL2NormTask(AscendTensor<int8_t, DIMS_2> &codesData,
                            AscendTensor<float16_t, DIMS_1> &normData,
                            AscendTensor<uint32_t, DIMS_2> &actualNum,
                            aclrtStream stream);

protected:
    // vector dimension
    int dims;

    std::unique_ptr<AscendOperator> l2NormOp;

    AscendTensor<float16_t, DIMS_2> transfer;

private:
    APP_ERROR resetL2NormOperator();

    void runL2NormOperator(AscendTensor<int8_t, DIMS_2> &vectors,
                           AscendTensor<float16_t, DIMS_2> &transfer,
                           AscendTensor<uint32_t, DIMS_1> &actualNum,
                           AscendTensor<float16_t, DIMS_1> &result,
                           aclrtStream stream);
};
}  // namespace ascend

#endif  // ASCEND_INT8_L2_NORM_INCLUDED