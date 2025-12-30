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


#ifndef IVFSP_DISTANCESIMD_H
#define IVFSP_DISTANCESIMD_H

#include <stdlib.h>

float fvec_L2sqr(const float *x, const float *y, size_t d);
float fvec_inner_product(const float *x, const float *y, size_t d);
float fvec_norm_L2sqr(const float *x, size_t d);
void MatMul(float *dst, const float *leftM, const float *rightM, size_t n, size_t dim, size_t outDim, bool transpose);

void MatMul_4xm(float *dst, const float *leftM, const float *rightM, size_t dim, size_t outDim);

#endif  // IVFSP_DISTANCESIMD_H
