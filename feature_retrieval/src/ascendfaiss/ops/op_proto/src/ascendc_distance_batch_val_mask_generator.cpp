/*
 * -------------------------------------------------------------------------
 * This file is part of the IndexSDK project.
 * Copyright (c) 2026 Huawei Technologies Co.,Ltd.
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

#include "./ascendc_distance_batch_val_mask_generator.h"

namespace ge
{
IMPLEMT_VERIFIER(AscendcDistanceBatchValMaskGenerator, Verify) { return GRAPH_SUCCESS; }
IMPLEMT_COMMON_INFERFUNC(InferShape) { return GRAPH_SUCCESS; }
COMMON_INFER_FUNC_REG(AscendcDistanceBatchValMaskGenerator, InferShape);
VERIFY_FUNC_REG(AscendcDistanceBatchValMaskGenerator, Verify);
}  // namespace ge
