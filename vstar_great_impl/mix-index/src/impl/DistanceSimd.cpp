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


#include "impl/DistanceSimd.h"
#include <iostream>

#ifdef __SSE3__
#include <immintrin.h>
#endif

#ifdef __aarch64__
#include <arm_neon.h>
#endif

#ifdef __aarch64__
float fvec_L2sqr(const float *x, const float *y, size_t d)
{
    float32x4_t accux4 = vdupq_n_f32(0);
    const size_t d_simd = d - (d & 3);
    size_t i;
    // 4 mean each float32x4_t have 4 float elements
    for (i = 0; i < d_simd; i += 4) {
        float32x4_t xi = vld1q_f32(x + i);
        float32x4_t yi = vld1q_f32(y + i);
        float32x4_t sq = vsubq_f32(xi, yi);
        accux4 = vfmaq_f32(accux4, sq, sq);
    }
    float32x4_t accux2 = vpaddq_f32(accux4, accux4);
    float32_t accux1 = vdups_laneq_f32(accux2, 0) + vdups_laneq_f32(accux2, 1);
    for (; i < d; ++i) {
        float32_t xi = x[i];
        float32_t yi = y[i];
        float32_t sq = xi - yi;
        accux1 += sq * sq;
    }
    return accux1;
}

float fvec_inner_product(const float *x, const float *y, size_t d)
{
    float32x4_t accux4 = vdupq_n_f32(0);
    const size_t d_simd = d - (d & 3);
    size_t i;
    // 4 mean each float32x4_t have 4 float elements
    for (i = 0; i < d_simd; i += 4) {
        float32x4_t xi = vld1q_f32(x + i);
        float32x4_t yi = vld1q_f32(y + i);
        accux4 = vfmaq_f32(accux4, xi, yi);
    }
    float32x4_t accux2 = vpaddq_f32(accux4, accux4);
    float32_t accux1 = vdups_laneq_f32(accux2, 0) + vdups_laneq_f32(accux2, 1);
    for (; i < d; ++i) {
        float32_t xi = x[i];
        float32_t yi = y[i];
        accux1 += xi * yi;
    }
    return accux1;
}
float fvec_norm_L2sqr(const float *x, size_t d)
{
    float32x4_t accux4 = vdupq_n_f32(0);
    const size_t d_simd = d - (d & 3);
    size_t i;
    // 4 mean each float32x4_t have 4 float elements
    for (i = 0; i < d_simd; i += 4) {
        float32x4_t xi = vld1q_f32(x + i);
        accux4 = vfmaq_f32(accux4, xi, xi);
    }
    float32x4_t accux2 = vpaddq_f32(accux4, accux4);
    float32_t accux1 = vdups_laneq_f32(accux2, 0) + vdups_laneq_f32(accux2, 1);
    for (; i < d; ++i) {
        float32_t xi = x[i];
        accux1 += xi * xi;
    }
    return accux1;
}

void MatMul_4xm(float *dst, const float *leftM, const float *rightM, size_t dim, size_t outDim)
{
    int b_idx;

    // these are the columns of a 4x4 sub matrix of A
    float32x4_t A0;
    float32x4_t A1;
    float32x4_t A2;
    float32x4_t A3;

    // these are the columns of a 4x4 sub matrix of B
    float32x4_t B0;
    float32x4_t B1;
    float32x4_t B2;
    float32x4_t B3;

    // these are the columns of a 4x4 sub matrix of C
    float32x4_t C0;
    float32x4_t C1;
    float32x4_t C2;
    float32x4_t C3;

    // 4 mean each float32x4_t have 4 float elements
    for (size_t j_idx = 0; j_idx < outDim; j_idx += 4) {
        // zero accumulators before matrix op
        C0 = vmovq_n_f32(0);
        C1 = vmovq_n_f32(0);
        C2 = vmovq_n_f32(0);
        C3 = vmovq_n_f32(0);
        // 4 mean each float32x4_t have 4 float elements
        for (size_t k_idx = 0; k_idx < dim; k_idx += 4) {
            // compute base index to 4x4 block
            b_idx = dim * j_idx + k_idx;

            B0 = vld1q_f32(rightM + b_idx);
            B1 = vld1q_f32(rightM + b_idx + dim);
            // 2 means index 2
            B2 = vld1q_f32(rightM + b_idx + 2 * dim);
            // 3 means index 3
            B3 = vld1q_f32(rightM + b_idx + 3 * dim);

            // multiply accumulate 4x1 blocks, that is each column C
            A0 = vld1q_f32(leftM + k_idx);
            C0 = vfmaq_laneq_f32(C0, A0, B0, 0);
            C0 = vfmaq_laneq_f32(C0, A0, B1, 1);
            // 2 means index 2
            C0 = vfmaq_laneq_f32(C0, A0, B2, 2);
            // 3 means index 3
            C0 = vfmaq_laneq_f32(C0, A0, B3, 3);

            A1 = vld1q_f32(leftM + dim + k_idx);
            C1 = vfmaq_laneq_f32(C1, A1, B0, 0);
            C1 = vfmaq_laneq_f32(C1, A1, B1, 1);
            // 2 means index 2
            C1 = vfmaq_laneq_f32(C1, A1, B2, 2);
            // 3 means index 3
            C1 = vfmaq_laneq_f32(C1, A1, B3, 3);

            // 2 means index 2
            A2 = vld1q_f32(leftM + 2 * dim + k_idx);

            C2 = vfmaq_laneq_f32(C2, A2, B0, 0);
            C2 = vfmaq_laneq_f32(C2, A2, B1, 1);
            // 2 means index 2
            C2 = vfmaq_laneq_f32(C2, A2, B2, 2);
            // 3 means index 3
            C2 = vfmaq_laneq_f32(C2, A2, B3, 3);

            // 3 means index 3
            A3 = vld1q_f32(leftM + 3 * dim + k_idx);
            C3 = vfmaq_laneq_f32(C3, A3, B0, 0);
            C3 = vfmaq_laneq_f32(C3, A3, B1, 1);
            // 2 means index 2
            C3 = vfmaq_laneq_f32(C3, A3, B2, 2);
            // 3 means index 3
            C3 = vfmaq_laneq_f32(C3, A3, B3, 3);
        }
        // compute base index for stores
        vst1q_f32(dst + j_idx, C0);
        vst1q_f32(dst + outDim + j_idx, C1);
        // 2 means index 2
        vst1q_f32(dst + 2 * outDim + j_idx, C2);
        // 3 means index 3
        vst1q_f32(dst + 3 * outDim + j_idx, C3);
    }
}

/**
 * case transpose == false: require leftM.shape = (n, dim) and rightM.shape = (dim, outDim) => dst.shape = (n, outDim),
 * row-first storage case transpose == true: require leftM.shape = (n, dim) and rightM.shape = (outDim, dim) =>
 * dst.shape = (n, outDim), row-first storage
 * @param dst
 * @param leftM
 * @param rightM
 * @param n
 * @param dim
 * @param outDim
 * @param transpose
 */
void MatMul(float *dst, const float *leftM, const float *rightM, size_t n, size_t dim, size_t outDim, bool transpose)
{
    if (transpose) {
        for (size_t i = 0; i < n; i++) {
            for (size_t j = 0; j < outDim; j++) {
                dst[j + i * outDim] = fvec_inner_product(leftM + i * dim, rightM + j * dim, dim);
            }
        }
    } else {
        for (size_t m = 0; m < n; m++) {
            for (size_t j = 0; j < outDim; j++) {
                const float *x = leftM + m * dim;
                const float *y = rightM;
                float32x4_t accux4 = vdupq_n_f32(0);
                const size_t d_simd = dim - (dim & 3);
                size_t i;
                // 4 mean each float32x4_t have 4 float elements
                for (i = 0; i < d_simd; i += 4) {
                    float32x4_t xi = vld1q_f32(x + i);
                    float32x4_t yi = vld1q_f32(y + i * outDim + j);
                    accux4 = vfmaq_f32(accux4, xi, yi);
                }
                float32x4_t accux2 = vpaddq_f32(accux4, accux4);
                float32_t accux1 = vdups_laneq_f32(accux2, 0) + vdups_laneq_f32(accux2, 1);
                for (; i < dim; ++i) {
                    float32_t xi = x[i];
                    float32_t yi = y[i * outDim + j];
                    accux1 += xi * yi;
                }
                dst[j + m * outDim] = accux1;
            }
        }
    }
}

#elif defined(__AVX__)

inline float horizontal_sum(const __m128 v)
{
    // say, v is [x0, x1, x2, x3]

    // v0 is [x2, x3, ..., ...]
    const __m128 v0 = _mm_shuffle_ps(v, v, _MM_SHUFFLE(0, 0, 3, 2));
    // v1 is [x0 + x2, x1 + x3, ..., ...]
    const __m128 v1 = _mm_add_ps(v, v0);
    // v2 is [x1 + x3, ..., .... ,...]
    __m128 v2 = _mm_shuffle_ps(v1, v1, _MM_SHUFFLE(0, 0, 0, 1));
    // v3 is [x0 + x1 + x2 + x3, ..., ..., ...]
    const __m128 v3 = _mm_add_ps(v1, v2);
    // return v3[0]
    return _mm_cvtss_f32(v3);
}

inline float horizontal_sum(const __m256 v)
{
    // add high and low parts
    const __m128 v0 = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    // perform horizontal sum on v0
    return horizontal_sum(v0);
}

float fvec_norm_L2sqr(const float *x, size_t d)
{
    auto dim = d;
    int i = 0;
    const int mask = 0b11110001;
    __m256 c0 = _mm256_setzero_ps();
    // __m256 have 8 float elements
    while (dim >= 8) {
        __m256 x0 = _mm256_loadu_ps(x + i);
        c0 = _mm256_add_ps(_mm256_dp_ps(x0, x0, mask), c0);
        // __m256 have 8 float elements
        dim -= 8;
        // __m256 have 8 float elements
        i += 8;
    }
    __m128 c1 = _mm_add_ps(_mm256_castps256_ps128(c0), _mm256_extractf128_ps(c0, 1));
    // __m128 have 4 float elements
    while (dim >= 4) {
        __m128 x1 = _mm_loadu_ps(x + i);
        c1 = _mm_add_ps(_mm_dp_ps(x1, x1, mask), c1);
        dim -= 4; // __m128 have 4 float elements
        i += 4; // __m128 have 4 float elements
    }
    float res = _mm_cvtss_f32(c1);
    for (int k = 0; k < dim; k++) {
        res += x[i] * x[i];
        i++;
    }
    return res;
}

float fvec_L2sqr(const float *x, const float *y, size_t d)
{
    auto dim = d;
    int i = 0;
    const int mask = 0b11110001;
    __m256 z0 = _mm256_setzero_ps();
    while (dim >= 8) { // __m256 have 8 float elements
        __m256 x0 = _mm256_loadu_ps(x + i);
        __m256 y0 = _mm256_loadu_ps(y + i);
        __m256 c0 = _mm256_sub_ps(x0, y0);
        z0 = _mm256_add_ps(_mm256_dp_ps(c0, c0, mask), z0);
        dim -= 8; // __m256 have 8 float elements
        i += 8; // __m256 have 8 float elements
    }
    __m128 z1 = _mm_add_ps(_mm256_castps256_ps128(z0), _mm256_extractf128_ps(z0, 1));
    while (dim >= 4) { // __m128 have 4 float elements
        __m128 x1 = _mm_loadu_ps(x + i);
        __m128 y1 = _mm_loadu_ps(y + i);
        __m128 c1 = _mm_sub_ps(x1, y1);
        z1 = _mm_add_ps(_mm_dp_ps(c1, c1, mask), z1);
        dim -= 4; // __m128 have 4 float elements
        i += 4; // __m128 have 4 float elements
    }
    float res = _mm_cvtss_f32(z1);
    for (int k = 0; k < dim; k++) {
        float m = x[i] - y[i];
        res += m * m;
        i++;
    }
    return res;
}

float fvec_inner_product(const float *x, const float *y, size_t d)
{
    auto dim = d;
    int i = 0;
    __m256 c0 = _mm256_setzero_ps();
    const int mask = 0b11110001;
    while (dim >= 8) { // __m256 have 8 float elements
        __m256 x0 = _mm256_loadu_ps(x + i);
        __m256 y0 = _mm256_loadu_ps(y + i);
        c0 = _mm256_add_ps(_mm256_dp_ps(x0, y0, mask), c0);
        dim -= 8; // __m256 have 8 float elements
        i += 8; // __m256 have 8 float elements
    }
    __m128 c1 = _mm_add_ps(_mm256_castps256_ps128(c0), _mm256_extractf128_ps(c0, 1));
    while (dim >= 4) { // __m128 have 4 float elements
        __m128 x1 = _mm_loadu_ps(x + i);
        __m128 y1 = _mm_loadu_ps(y + i);
        c1 = _mm_add_ps(_mm_dp_ps(x1, y1, mask), c1);
        dim -= 4; // __m128 have 4 float elements
        i += 4; // __m128 have 4 float elements
    }
    float res = _mm_cvtss_f32(c1);
    for (int k = 0; k < dim; k++) {
        res += x[i] * y[i];
        i++;
    }
    return res;
}

void MatMul(float *dst, const float *leftM, const float *rightM, size_t n, size_t dim, size_t outDim, bool transpose)
{
    if (transpose) {
        for (size_t i = 0; i < n; i++) {
            for (size_t j = 0; j < outDim; j++) {
                dst[j + i * outDim] = fvec_inner_product(leftM + i * dim, rightM + j * dim, dim);
            }
        }
    } else {
        for (size_t m = 0; m < n; m++) {
            for (size_t j = 0; j < outDim; j++) {
                const float *x = leftM + m * dim;
                const float *y = rightM;

                __m256 c0 = _mm256_setzero_ps();
                const int mask = 0b11110001;
                int i = 0;
                while (dim >= 8) { // __m256 have 8 float elements
                    __m256 x0 = _mm256_loadu_ps(x + i);
                    __m256 y0 = _mm256_loadu_ps(y + i * outDim + j);
                    c0 = _mm256_add_ps(_mm256_dp_ps(x0, y0, mask), c0);
                    dim -= 8; // __m256 have 8 float elements
                    i += 8; // __m256 have 8 float elements
                }
                __m128 c1 = _mm_add_ps(_mm256_castps256_ps128(c0), _mm256_extractf128_ps(c0, 1));
                while (dim >= 4) { // __m128 have 4 float elements
                    __m128 x1 = _mm_loadu_ps(x + i);
                    __m128 y1 = _mm_loadu_ps(y + i * outDim + j);
                    c1 = _mm_add_ps(_mm_dp_ps(x1, y1, mask), c1);
                    dim -= 4; // __m128 have 4 float elements
                    i += 4; // __m128 have 4 float elements
                }
                float res = _mm_cvtss_f32(c1);
                for (int k = 0; k < dim; k++) {
                    res += x[i] * y[i];
                    i++;
                }
                dst[j + m * outDim] = res;
            }
        }
    }
}

#else

float fvec_L2sqr(const float *x, const float *y, size_t d)
{
    float res = 0.F;
    for (size_t i = 0; i != d; ++i) {
        const float tmp = x[i] - y[i];
        res += tmp * tmp;
    }
    return res;
}

float fvec_inner_product(const float *x, const float *y, size_t d)
{
    float res = 0.F;
    for (size_t i = 0; i != d; ++i) {
        res += x[i] * y[i];
    }
    return res;
}
float fvec_norm_L2sqr(const float *x, size_t d)
{
    float res = 0.F;
    for (size_t i = 0; i < d; ++i) {
        res += x[i] * x[i];
    }
    return res;
}

/**
 * case transpose == false: require leftM.shape = (n, dim) and rightM.shape = (dim, outDim) => dst.shape = (n, outDim)
 * case transpose == true: require leftM.shape = (n, dim) and rightM.shape = (outDim, dim) => dst.shape = (n, outDim)
 * @param dst
 * @param leftM
 * @param rightM
 * @param n
 * @param dim
 * @param outDim
 * @param transpose
 */
void MatMul(float *dst, const float *leftM, const float *rightM, size_t n, size_t dim, size_t outDim, bool transpose)
{
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < outDim; j++) {
            double res = 0;
            for (size_t p = 0; p < dim; p++) {
                if (transpose) {
                    res += leftM[p + i * dim] * rightM[p + dim * j];
                } else {
                    res += leftM[p + i * dim] * rightM[p * outDim + j];
                }
            }
            dst[j + i * outDim] = res;
        }
    }
}
#endif