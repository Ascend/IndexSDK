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


#include "l2_norm_flat_sub.h"

#include <vector>
#include <string>
#include <iostream>

namespace ge {
    IMPLEMT_VERIFIER(L2NormFlatSub, L2NormFlatSubVerify)
    {
        return GRAPH_SUCCESS;
    }


    // Obtains the processing function of the output tensor description.
    IMPLEMT_COMMON_INFERFUNC(L2NormFlatSubInferShape)
    {
        return GRAPH_SUCCESS;
    } // namespace ge

    // Registered inferfunction
    COMMON_INFER_FUNC_REG(L2NormFlatSub, L2NormFlatSubInferShape);

    // Registered verify function
    VERIFY_FUNC_REG(L2NormFlatSub, L2NormFlatSubVerify);
} // namespace ge