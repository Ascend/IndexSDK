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


#include "utils/GraphDistanceComputer.h"
#include <bitset>

#ifdef __aarch64__
#include <arm_neon.h>
#endif
#include <cassert>

#if defined(USE_AVX) || defined(USE_SSE)
#ifdef _MSC_VER
void cpu_x86::cpuid(int32_t out[G_FOUR], int32_t eax, int32_t ecx)
{
    __cpuidex(out, eax, ecx);
}

__int64 xgetbv(unsigned int x)
{
    return _xgetbv(x);
}
#else
void CpuId(int32_t cpuInfo[G_FOUR], int32_t eax, int32_t ecx)
{
    __cpuid_count(eax, ecx, cpuInfo[0], cpuInfo[1], cpuInfo[G_TWO], cpuInfo[G_THREE]);
}

uint64_t Xgetbv(unsigned int index)
{
    uint32_t eax;
    uint32_t edx;
    __asm__ __volatile__("Xgetbv" : "=a"(eax), "=d"(edx) : "c"(index));
    return ((uint64_t)edx << G_THIRTY_TWO) | eax;
}
#endif

bool AVXCapable()
{
    int cpuInfo[G_FOUR];

    // CPU support
    CpuId(cpuInfo, 0, 0);
    int nIds = cpuInfo[0];

    bool HW_AVX = false;
    if (nIds >= 0x00000001) {
        CpuId(cpuInfo, 0x00000001, 0);
        HW_AVX = (cpuInfo[G_TWO] & ((int)1 << G_TWENTY_EIGHT)) != 0;
    }

    // OS support
    CpuId(cpuInfo, 1, 0);

    bool osUsesXSAVE_XRSTORE = (cpuInfo[G_TWO] & (1 << G_TWENTY_SEVEN)) != 0;
    bool cpuAVXSuport = (cpuInfo[G_TWO] & (1 << G_TWENTY_EIGHT)) != 0;

    bool avxSupported = false;
    if (osUsesXSAVE_XRSTORE && cpuAVXSuport) {
        uint64_t xcrFeatureMask = Xgetbv(XCR_XFEATURE_ENABLED_MASK);
        avxSupported = (xcrFeatureMask & 0x6) == 0x6;
    }
    return HW_AVX && avxSupported;
}

bool AVX512Capable()
{
    if (!AVXCapable())
        return false;

    int cpuInfo[G_FOUR];

    // CPU support
    CpuId(cpuInfo, 0, 0);
    int nIds = cpuInfo[0];

    bool HW_AVX512F = false;
    if (nIds >= 0x00000007) {  //  AVX512 Foundation
        CpuId(cpuInfo, 0x00000007, 0);
        HW_AVX512F = (cpuInfo[1] & ((int)1 << G_SIXTEEN)) != 0;
    }

    // OS support
    CpuId(cpuInfo, 1, 0);

    bool osUsesXSAVE_XRSTORE = (cpuInfo[G_TWO] & (1 << G_TWENTY_SEVEN)) != 0;
    bool cpuAVXSuport = (cpuInfo[G_TWO] & (1 << G_TWENTY_EIGHT)) != 0;

    bool avx512Supported = false;
    if (osUsesXSAVE_XRSTORE && cpuAVXSuport) {
        uint64_t xcrFeatureMask = Xgetbv(XCR_XFEATURE_ENABLED_MASK);
        avx512Supported = (xcrFeatureMask & 0xe6) == 0xe6;
    }
    return HW_AVX512F && avx512Supported;
}
#endif

namespace ascendsearch {
#if defined(USE_SSE) || defined(USE_AVX) || defined(USE_AVX512)
DISTFUNC<float> InnerProductSIMD16Ext = IPDisHnsw::InnerProductSIMD16ExtSSE;
DISTFUNC<float> InnerProductSIMD4Ext = IPDisHnsw::InnerProductSIMD4ExtSSE;
DISTFUNC<float> InnerProductDistanceSIMD16Ext = IPDisHnsw::InnerProductDistanceSIMD16ExtSSE;
DISTFUNC<float> InnerProductDistanceSIMD4Ext = IPDisHnsw::InnerProductDistanceSIMD4ExtSSE;

DISTFUNC<float> L2SqrSIMD16Ext = L2DisHnsw::L2SqrSIMD16ExtSSE;
#endif

IPDisHnsw::IPDisHnsw()
{
}

IPDisHnsw::IPDisHnsw(const faiss::IndexFlat &storage, const float *q)
{
    d = storage.d;
    nb = storage.ntotal;
    q_ = q;
    b = storage.get_xb();
    ndis = 0;

    data_size_ = d * sizeof(float);
    fstdistfunc_ = InnerProductDistance;
#if defined(USE_AVX) || defined(USE_SSE) || defined(USE_AVX512)
#if defined(USE_AVX512)
    if (AVX512Capable()) {
        InnerProductSIMD16Ext = InnerProductSIMD16ExtAVX512;
        InnerProductDistanceSIMD16Ext = InnerProductDistanceSIMD16ExtAVX512;
    } else if (AVXCapable()) {
        InnerProductSIMD16Ext = InnerProductSIMD16ExtAVX;
        InnerProductDistanceSIMD16Ext = InnerProductDistanceSIMD16ExtAVX;
    }
#elif defined(USE_AVX)
    if (AVXCapable()) {
        InnerProductSIMD16Ext = InnerProductSIMD16ExtAVX;
        InnerProductDistanceSIMD16Ext = InnerProductDistanceSIMD16ExtAVX;
    }
#endif
#if defined(USE_AVX)
    if (AVXCapable()) {
        InnerProductSIMD4Ext = InnerProductSIMD4ExtAVX;
        InnerProductDistanceSIMD4Ext = InnerProductDistanceSIMD4ExtAVX;
    }
#endif

    if (d % G_SIXTEEN == 0)
        fstdistfunc_ = InnerProductDistanceSIMD16Ext;
    else if (d % G_FOUR == 0)
        fstdistfunc_ = InnerProductDistanceSIMD4Ext;
    else if (d > G_SIXTEEN)
        fstdistfunc_ = InnerProductDistanceSIMD16ExtResiduals;
    else if (d > G_FOUR)
        fstdistfunc_ = InnerProductDistanceSIMD4ExtResiduals;
#endif
}

IPDisHnsw::~IPDisHnsw() noexcept
{
}

void IPDisHnsw::set_query(const float *x)
{
    q_ = x;
}

// / compute distance of vector i to current query
float IPDisHnsw::operator()(int64_t i)
{
    return fstdistfunc_(q_, b + i * d, &d);
}

// / compute distance between two stored vectors
float IPDisHnsw::symmetric_dis(int64_t i, int64_t j)
{
    return fstdistfunc_(b + j * d, b + i * d, &d);
}

// / 自定义函数
size_t IPDisHnsw::get_data_size()
{
    return data_size_;
}

DISTFUNC<float> IPDisHnsw::get_dist_func()
{
    return fstdistfunc_;
}

void *IPDisHnsw::get_dist_func_param()
{
    return &d;
}

// / 距离计算函数
float IPDisHnsw::InnerProduct(const void *pVect1, const void *pVect2, const void *qty_ptr)
{
    size_t qty = *((size_t *)qty_ptr);
#ifdef __aarch64__
    const float *x = (const float *)(pVect1);
    const float *y = (const float *)(pVect2);
    float32x4_t accux4 = vdupq_n_f32(0);
    float32x4_t accux5 = vdupq_n_f32(0);
    const size_t d_simd = qty - (qty & 7);
    size_t i;
    for (i = 0; i < d_simd; i += 8) {
        float32x4_t xi = vld1q_f32(x + i);
        float32x4_t yi = vld1q_f32(y + i);
        accux4 = vfmaq_f32(accux4, xi, yi);
        float32x4_t xii = vld1q_f32(x + i + 4);
        float32x4_t yii = vld1q_f32(y + i + 4);
        accux5 = vfmaq_f32(accux5, xii, yii);
    }
    float32_t accux1 = vaddvq_f32(accux4);
    float32_t accux2 = vaddvq_f32(accux5);
    accux1 += accux2;
    for (; i < qty; ++i) {
        float32_t xi = x[i];
        float32_t yi = y[i];
        accux1 += xi * yi;
    }
    return accux1;
#elif __x86_64__
    const float *x = (const float *)(pVect1);
    const float *y = (const float *)(pVect2);
    __m128 accux4 = _mm_set1_ps(0);
    __m128 accux5 = _mm_set1_ps(0);
    const size_t d_simd = qty - (qty & 7);
    size_t i;
    for (i = 0; i < d_simd; i += 8) {
        __m128 xi = _mm_loadu_ps(x + i);
        __m128 yi = _mm_loadu_ps(y + i);
        accux4 = _mm_add_ps(_mm_mul_ps(xi, yi), accux4);
        __m128 xii = _mm_loadu_ps(x + i + 4);
        __m128 yii = _mm_loadu_ps(y + i + 4);
        accux5 = _mm_add_ps(_mm_mul_ps(xii, yii), accux5);
    }
    const auto vaddvq_f32_emulation = [](__m128 v) -> float {
        // 将向量中的元素按照特定顺序重排
        __m128 shuffled = _mm_shuffle_ps(v, v, _MM_SHUFFLE(2, 3, 0, 1));
        // 对重排后的向量与原向量进行逐元素相加
        __m128 sum1 = _mm_add_ps(v, shuffled);
        // 再次重排结果向量
        shuffled = _mm_shuffle_ps(sum1, sum1, _MM_SHUFFLE(1, 2, 3, 0));
        // 对再次重排后的向量进行逐元素相加
        __m128 sum2 = _mm_add_ps(sum1, shuffled);
        // Extract the sum from the vector
        return _mm_cvtss_f32(sum2);
    };
    float accux1 = vaddvq_f32_emulation(accux4);
    float accux2 = vaddvq_f32_emulation(accux5);
    accux1 += accux2;
    for (; i < qty; ++i) {
        float xi = x[i];
        float yi = y[i];
        accux1 += xi * yi;
    }
    return accux1;
#endif
}

float IPDisHnsw::InnerProductDistance(const void *pVect1, const void *pVect2, const void *qty_ptr)
{
    return -InnerProduct(pVect1, pVect2, qty_ptr);
}

#if defined(USE_AVX)
float IPDisHnsw::InnerProductSIMD4ExtAVX(const void *pVect1v, const void *pVect2v, const void *qty_ptr)
{
    float PORTABLE_ALIGN32 TmpRes[G_EIGHT];
    float *pVect1 = (float *)pVect1v;
    float *pVect2 = (float *)pVect2v;
    size_t qty = *((size_t *)qty_ptr);

    size_t qty16 = qty / G_SIXTEEN;
    size_t qty4 = qty / G_FOUR;

    const float *pEnd1 = pVect1 + G_SIXTEEN * qty16;
    const float *pEnd2 = pVect1 + G_FOUR * qty4;

    __m256 sum256 = _mm256_set1_ps(0);

    while (pVect1 < pEnd1) {
        __m256 v1 = _mm256_loadu_ps(pVect1);
        pVect1 += G_EIGHT;
        __m256 v2 = _mm256_loadu_ps(pVect2);
        pVect2 += G_EIGHT;
        sum256 = _mm256_add_ps(sum256, _mm256_mul_ps(v1, v2));

        v1 = _mm256_loadu_ps(pVect1);
        pVect1 += G_EIGHT;
        v2 = _mm256_loadu_ps(pVect2);
        pVect2 += G_EIGHT;
        sum256 = _mm256_add_ps(sum256, _mm256_mul_ps(v1, v2));
    }

    __m128 v1;
    __m128 v2;
    __m128 sum_prod = _mm_add_ps(_mm256_extractf128_ps(sum256, 0), _mm256_extractf128_ps(sum256, 1));

    while (pVect1 < pEnd2) {
        v1 = _mm_loadu_ps(pVect1);
        pVect1 += G_FOUR;
        v2 = _mm_loadu_ps(pVect2);
        pVect2 += G_FOUR;
        sum_prod = _mm_add_ps(sum_prod, _mm_mul_ps(v1, v2));
    }

    _mm_store_ps(TmpRes, sum_prod);
    float sum = TmpRes[0] + TmpRes[1] + TmpRes[G_TWO] + TmpRes[G_THREE];
    return sum;
}

float IPDisHnsw::InnerProductDistanceSIMD4ExtAVX(const void *pVect1v, const void *pVect2v, const void *qty_ptr)
{
    return -InnerProductSIMD4ExtAVX(pVect1v, pVect2v, qty_ptr);
}
#endif

#if defined(USE_SSE)
float IPDisHnsw::InnerProductSIMD4ExtSSE(const void *pVect1v, const void *pVect2v, const void *qty_ptr)
{
    float PORTABLE_ALIGN32 TmpRes[G_EIGHT];
    float *pVect1 = (float *)pVect1v;
    float *pVect2 = (float *)pVect2v;
    size_t qty = *((size_t *)qty_ptr);

    size_t qty16 = qty / G_SIXTEEN;
    size_t qty4 = qty / G_FOUR;

    const float *pEnd1 = pVect1 + G_SIXTEEN * qty16;
    const float *pEnd2 = pVect1 + G_FOUR * qty4;

    __m128 v1;
    __m128 v2;
    __m128 sum_prod = _mm_set1_ps(0);

    while (pVect1 < pEnd1) {
        v1 = _mm_loadu_ps(pVect1);
        pVect1 += G_FOUR;
        v2 = _mm_loadu_ps(pVect2);
        pVect2 += G_FOUR;
        sum_prod = _mm_add_ps(sum_prod, _mm_mul_ps(v1, v2));

        v1 = _mm_loadu_ps(pVect1);
        pVect1 += G_FOUR;
        v2 = _mm_loadu_ps(pVect2);
        pVect2 += G_FOUR;
        sum_prod = _mm_add_ps(sum_prod, _mm_mul_ps(v1, v2));

        v1 = _mm_loadu_ps(pVect1);
        pVect1 += G_FOUR;
        v2 = _mm_loadu_ps(pVect2);
        pVect2 += G_FOUR;
        sum_prod = _mm_add_ps(sum_prod, _mm_mul_ps(v1, v2));

        v1 = _mm_loadu_ps(pVect1);
        pVect1 += G_FOUR;
        v2 = _mm_loadu_ps(pVect2);
        pVect2 += G_FOUR;
        sum_prod = _mm_add_ps(sum_prod, _mm_mul_ps(v1, v2));
    }

    while (pVect1 < pEnd2) {
        v1 = _mm_loadu_ps(pVect1);
        pVect1 += G_FOUR;
        v2 = _mm_loadu_ps(pVect2);
        pVect2 += G_FOUR;
        sum_prod = _mm_add_ps(sum_prod, _mm_mul_ps(v1, v2));
    }

    _mm_store_ps(TmpRes, sum_prod);
    float sum = TmpRes[0] + TmpRes[1] + TmpRes[G_TWO] + TmpRes[G_THREE];

    return sum;
}

float IPDisHnsw::InnerProductDistanceSIMD4ExtSSE(const void *pVect1v, const void *pVect2v, const void *qty_ptr)
{
    return -InnerProductSIMD4ExtSSE(pVect1v, pVect2v, qty_ptr);
}
#endif

#if defined(USE_AVX512)
float IPDisHnsw::InnerProductSIMD16ExtAVX512(const void *pVect1v, const void *pVect2v, const void *qty_ptr)
{
    float PORTABLE_ALIGN64 TmpRes[G_SIXTEEN];
    float *pVect1 = (float *)pVect1v;
    float *pVect2 = (float *)pVect2v;
    size_t qty = *((size_t *)qty_ptr);

    size_t qty16 = qty / G_SIXTEEN;

    const float *pEnd1 = pVect1 + G_SIXTEEN * qty16;

    __m512 sum512 = _mm512_set1_ps(0);

    while (pVect1 < pEnd1) {
        __m512 v1 = _mm512_loadu_ps(pVect1);
        pVect1 += G_SIXTEEN;
        __m512 v2 = _mm512_loadu_ps(pVect2);
        pVect2 += G_SIXTEEN;
        sum512 = _mm512_add_ps(sum512, _mm512_mul_ps(v1, v2));
    }

    _mm512_store_ps(TmpRes, sum512);
    float sum = TmpRes[0] + TmpRes[1] + TmpRes[G_TWO] + TmpRes[G_THREE] + TmpRes[G_FOUR] + TmpRes[G_FIVE] +
                TmpRes[G_SIX] + TmpRes[G_SEVEN] + TmpRes[G_EIGHT] + TmpRes[G_NINE] + TmpRes[G_TEN] + TmpRes[G_ELEVEN] +
                TmpRes[G_TWELVE] + TmpRes[G_THIRTEEN] + TmpRes[G_FOURTEEN] + TmpRes[G_FIFTEEN];

    return sum;
}

float IPDisHnsw::InnerProductDistanceSIMD16ExtAVX512(const void *pVect1v, const void *pVect2v, const void *qty_ptr)
{
    return -InnerProductSIMD16ExtAVX512(pVect1v, pVect2v, qty_ptr);
}
#endif

#if defined(USE_AVX)
float IPDisHnsw::InnerProductSIMD16ExtAVX(const void *pVect1v, const void *pVect2v, const void *qty_ptr)
{
    float PORTABLE_ALIGN32 TmpRes[G_EIGHT];
    float *pVect1 = (float *)pVect1v;
    float *pVect2 = (float *)pVect2v;
    size_t qty = *((size_t *)qty_ptr);

    size_t qty16 = qty / G_SIXTEEN;

    const float *pEnd1 = pVect1 + G_SIXTEEN * qty16;

    __m256 sum256 = _mm256_set1_ps(0);

    while (pVect1 < pEnd1) {
        __m256 v1 = _mm256_loadu_ps(pVect1);
        pVect1 += G_EIGHT;
        __m256 v2 = _mm256_loadu_ps(pVect2);
        pVect2 += G_EIGHT;
        sum256 = _mm256_add_ps(sum256, _mm256_mul_ps(v1, v2));

        v1 = _mm256_loadu_ps(pVect1);
        pVect1 += G_EIGHT;
        v2 = _mm256_loadu_ps(pVect2);
        pVect2 += G_EIGHT;
        sum256 = _mm256_add_ps(sum256, _mm256_mul_ps(v1, v2));
    }

    _mm256_store_ps(TmpRes, sum256);
    float sum = TmpRes[0] + TmpRes[1] + TmpRes[G_TWO] + TmpRes[G_THREE] + TmpRes[G_FOUR] + TmpRes[G_FIVE] +
                TmpRes[G_SIX] + TmpRes[G_SEVEN];

    return sum;
}

float IPDisHnsw::InnerProductDistanceSIMD16ExtAVX(const void *pVect1v, const void *pVect2v, const void *qty_ptr)
{
    return -InnerProductSIMD16ExtAVX(pVect1v, pVect2v, qty_ptr);
}
#endif

#if defined(USE_SSE)
float IPDisHnsw::InnerProductSIMD16ExtSSE(const void *pVect1v, const void *pVect2v, const void *qty_ptr)
{
    float PORTABLE_ALIGN32 TmpRes[G_EIGHT];
    float *pVect1 = (float *)pVect1v;
    float *pVect2 = (float *)pVect2v;
    size_t qty = *((size_t *)qty_ptr);

    size_t qty16 = qty / G_SIXTEEN;

    const float *pEnd1 = pVect1 + G_SIXTEEN * qty16;

    __m128 v1;
    __m128 v2;
    __m128 sum_prod = _mm_set1_ps(0);

    while (pVect1 < pEnd1) {
        v1 = _mm_loadu_ps(pVect1);
        pVect1 += G_FOUR;
        v2 = _mm_loadu_ps(pVect2);
        pVect2 += G_FOUR;
        sum_prod = _mm_add_ps(sum_prod, _mm_mul_ps(v1, v2));

        v1 = _mm_loadu_ps(pVect1);
        pVect1 += G_FOUR;
        v2 = _mm_loadu_ps(pVect2);
        pVect2 += G_FOUR;
        sum_prod = _mm_add_ps(sum_prod, _mm_mul_ps(v1, v2));

        v1 = _mm_loadu_ps(pVect1);
        pVect1 += G_FOUR;
        v2 = _mm_loadu_ps(pVect2);
        pVect2 += G_FOUR;
        sum_prod = _mm_add_ps(sum_prod, _mm_mul_ps(v1, v2));

        v1 = _mm_loadu_ps(pVect1);
        pVect1 += G_FOUR;
        v2 = _mm_loadu_ps(pVect2);
        pVect2 += G_FOUR;
        sum_prod = _mm_add_ps(sum_prod, _mm_mul_ps(v1, v2));
    }
    _mm_store_ps(TmpRes, sum_prod);
    float sum = TmpRes[0] + TmpRes[1] + TmpRes[G_TWO] + TmpRes[G_THREE];

    return sum;
}

float IPDisHnsw::InnerProductDistanceSIMD16ExtSSE(const void *pVect1v, const void *pVect2v, const void *qty_ptr)
{
    return -InnerProductSIMD16ExtSSE(pVect1v, pVect2v, qty_ptr);
}
#endif

#if defined(USE_SSE) || defined(USE_AVX) || defined(USE_AVX512)
float IPDisHnsw::InnerProductDistanceSIMD16ExtResiduals(const void *pVect1v, const void *pVect2v, const void *qty_ptr)
{
    size_t qty = *((size_t *)qty_ptr);
    size_t qty16 = qty >> G_FOUR << G_FOUR;
    float res = InnerProductSIMD16Ext(pVect1v, pVect2v, &qty16);
    float *pVect1 = (float *)pVect1v + qty16;
    float *pVect2 = (float *)pVect2v + qty16;

    size_t qty_left = qty - qty16;
    float res_tail = InnerProduct(pVect1, pVect2, &qty_left);
    return -(res + res_tail);
}

float IPDisHnsw::InnerProductDistanceSIMD4ExtResiduals(const void *pVect1v, const void *pVect2v, const void *qty_ptr)
{
    size_t qty = *((size_t *)qty_ptr);
    size_t qty4 = qty >> G_TWO << G_TWO;

    float res = InnerProductSIMD4Ext(pVect1v, pVect2v, &qty4);
    size_t qty_left = qty - qty4;

    float *pVect1 = (float *)pVect1v + qty4;
    float *pVect2 = (float *)pVect2v + qty4;
    float res_tail = InnerProduct(pVect1, pVect2, &qty_left);

    return -(res + res_tail);
}
#endif
// / L2距离度量函数
L2DisHnsw::L2DisHnsw()
{
}

L2DisHnsw::L2DisHnsw(const faiss::IndexFlat &storage, const float *q)
{
    d = storage.d;
    nb = storage.ntotal;
    q_ = q;
    b = storage.get_xb();
    ndis = 0;

    data_size_ = d * sizeof(float);
    fstdistfunc_ = L2Sqr;
#if defined(USE_SSE) || defined(USE_AVX) || defined(USE_AVX512)
#if defined(USE_AVX512)
    if (AVX512Capable())
        L2SqrSIMD16Ext = L2SqrSIMD16ExtAVX512;
    else if (AVXCapable())
        L2SqrSIMD16Ext = L2SqrSIMD16ExtAVX;
#elif defined(USE_AVX)
    if (AVXCapable())
        L2SqrSIMD16Ext = L2SqrSIMD16ExtAVX;
#endif

    if (d % G_SIXTEEN == 0)
        fstdistfunc_ = L2SqrSIMD16Ext;
    else if (d % G_FOUR == 0)
        fstdistfunc_ = L2SqrSIMD4Ext;
    else if (d > G_SIXTEEN)
        fstdistfunc_ = L2SqrSIMD16ExtResiduals;
    else if (d > G_FOUR)
        fstdistfunc_ = L2SqrSIMD4ExtResiduals;
#endif
}

L2DisHnsw::~L2DisHnsw() noexcept
{
}

void L2DisHnsw::set_query(const float *x)
{
    q_ = x;
}

// / compute distance of vector i to current query
float L2DisHnsw::operator()(int64_t i)
{
    return fstdistfunc_(q_, b + i * d, &d);
}

// / compute distance between two stored vectors
float L2DisHnsw::symmetric_dis(int64_t i, int64_t j)
{
    return fstdistfunc_(b + j * d, b + i * d, &d);
}

// / 自定义函数
size_t L2DisHnsw::get_data_size()
{
    return data_size_;
}

DISTFUNC<float> L2DisHnsw::get_dist_func()
{
    return fstdistfunc_;
}

void *L2DisHnsw::get_dist_func_param()
{
    return &d;
}

// / 距离计算函数
float L2DisHnsw::L2Sqr(const void *pVect1v, const void *pVect2v, const void *qty_ptr)
{
    float *pVect1 = (float *)pVect1v;
    float *pVect2 = (float *)pVect2v;
    size_t qty = *((size_t *)qty_ptr);

    float res = 0;
    for (size_t i = 0; i < qty; i++) {
        float t = *pVect1 - *pVect2;
        pVect1++;
        pVect2++;
        res += t * t;
    }
    return (res);
}

#if defined(USE_AVX512)
float L2DisHnsw::L2SqrSIMD16ExtAVX512(const void *pVect1v, const void *pVect2v, const void *qty_ptr)
{
    float *pVect1 = (float *)pVect1v;
    float *pVect2 = (float *)pVect2v;
    size_t qty = *((size_t *)qty_ptr);
    float PORTABLE_ALIGN64 TmpRes[G_SIXTEEN];
    size_t qty16 = qty >> G_FOUR;

    const float *pEnd1 = pVect1 + (qty16 << G_FOUR);

    __m512 diff;
    __m512 v1;
    __m512 v2;
    __m512 sum = _mm512_set1_ps(0);

    while (pVect1 < pEnd1) {
        v1 = _mm512_loadu_ps(pVect1);
        pVect1 += G_SIXTEEN;
        v2 = _mm512_loadu_ps(pVect2);
        pVect2 += G_SIXTEEN;
        diff = _mm512_sub_ps(v1, v2);
        sum = _mm512_add_ps(sum, _mm512_mul_ps(diff, diff));
    }

    _mm512_store_ps(TmpRes, sum);
    float res = TmpRes[0] + TmpRes[1] + TmpRes[G_TWO] + TmpRes[G_THREE] + TmpRes[G_FOUR] + TmpRes[G_FIVE] +
                TmpRes[G_SIX] + TmpRes[G_SEVEN] + TmpRes[G_EIGHT] + TmpRes[G_NINE] + TmpRes[G_TEN] + TmpRes[G_ELEVEN] +
                TmpRes[G_TWELVE] + TmpRes[G_THIRTEEN] + TmpRes[G_FOURTEEN] + TmpRes[G_FIFTEEN];

    return (res);
}
#endif

#if defined(USE_AVX)
float L2DisHnsw::L2SqrSIMD16ExtAVX(const void *pVect1v, const void *pVect2v, const void *qty_ptr)
{
    float *pVect1 = (float *)pVect1v;
    float *pVect2 = (float *)pVect2v;
    size_t qty = *((size_t *)qty_ptr);
    float PORTABLE_ALIGN32 TmpRes[G_EIGHT];
    size_t qty16 = qty >> G_FOUR;

    const float *pEnd1 = pVect1 + (qty16 << G_FOUR);

    __m256 diff;
    __m256 v1;
    __m256 v2;
    __m256 sum = _mm256_set1_ps(0);

    while (pVect1 < pEnd1) {
        v1 = _mm256_loadu_ps(pVect1);
        pVect1 += G_EIGHT;
        v2 = _mm256_loadu_ps(pVect2);
        pVect2 += G_EIGHT;
        diff = _mm256_sub_ps(v1, v2);
        sum = _mm256_add_ps(sum, _mm256_mul_ps(diff, diff));

        v1 = _mm256_loadu_ps(pVect1);
        pVect1 += G_EIGHT;
        v2 = _mm256_loadu_ps(pVect2);
        pVect2 += G_EIGHT;
        diff = _mm256_sub_ps(v1, v2);
        sum = _mm256_add_ps(sum, _mm256_mul_ps(diff, diff));
    }

    _mm256_store_ps(TmpRes, sum);
    return TmpRes[0] + TmpRes[1] + TmpRes[G_TWO] + TmpRes[G_THREE] + TmpRes[G_FOUR] + TmpRes[G_FIVE] + TmpRes[G_SIX] +
           TmpRes[G_SEVEN];
}
#endif

#if defined(USE_SSE)
float L2DisHnsw::L2SqrSIMD16ExtSSE(const void *pVect1v, const void *pVect2v, const void *qty_ptr)
{
    float *pVect1 = (float *)pVect1v;
    float *pVect2 = (float *)pVect2v;
    size_t qty = *((size_t *)qty_ptr);
    float PORTABLE_ALIGN32 TmpRes[G_EIGHT];
    size_t qty16 = qty >> G_FOUR;

    const float *pEnd1 = pVect1 + (qty16 << G_FOUR);

    __m128 diff;
    __m128 v1;
    __m128 v2;
    __m128 sum = _mm_set1_ps(0);

    while (pVect1 < pEnd1) {
        v1 = _mm_loadu_ps(pVect1);
        pVect1 += G_FOUR;
        v2 = _mm_loadu_ps(pVect2);
        pVect2 += G_FOUR;
        diff = _mm_sub_ps(v1, v2);
        sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff));

        v1 = _mm_loadu_ps(pVect1);
        pVect1 += G_FOUR;
        v2 = _mm_loadu_ps(pVect2);
        pVect2 += G_FOUR;
        diff = _mm_sub_ps(v1, v2);
        sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff));

        v1 = _mm_loadu_ps(pVect1);
        pVect1 += G_FOUR;
        v2 = _mm_loadu_ps(pVect2);
        pVect2 += G_FOUR;
        diff = _mm_sub_ps(v1, v2);
        sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff));

        v1 = _mm_loadu_ps(pVect1);
        pVect1 += G_FOUR;
        v2 = _mm_loadu_ps(pVect2);
        pVect2 += G_FOUR;
        diff = _mm_sub_ps(v1, v2);
        sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff));
    }

    _mm_store_ps(TmpRes, sum);
    return TmpRes[0] + TmpRes[1] + TmpRes[G_TWO] + TmpRes[G_THREE];
}
#endif

#if defined(USE_SSE) || defined(USE_AVX) || defined(USE_AVX512)
float L2DisHnsw::L2SqrSIMD16ExtResiduals(const void *pVect1v, const void *pVect2v, const void *qty_ptr)
{
    size_t qty = *((size_t *)qty_ptr);
    size_t qty16 = qty >> G_FOUR << G_FOUR;
    float res = L2SqrSIMD16Ext(pVect1v, pVect2v, &qty16);
    float *pVect1 = (float *)pVect1v + qty16;
    float *pVect2 = (float *)pVect2v + qty16;

    size_t qty_left = qty - qty16;
    float res_tail = L2Sqr(pVect1, pVect2, &qty_left);
    return (res + res_tail);
}
#endif

#if defined(USE_SSE)
float L2DisHnsw::L2SqrSIMD4Ext(const void *pVect1v, const void *pVect2v, const void *qty_ptr)
{
    float PORTABLE_ALIGN32 TmpRes[G_EIGHT];
    float *pVect1 = (float *)pVect1v;
    float *pVect2 = (float *)pVect2v;
    size_t qty = *((size_t *)qty_ptr);

    size_t qty4 = qty >> G_TWO;

    const float *pEnd1 = pVect1 + (qty4 << 2);

    __m128 diff;
    __m128 v1;
    __m128 v2;
    __m128 sum = _mm_set1_ps(0);

    while (pVect1 < pEnd1) {
        v1 = _mm_loadu_ps(pVect1);
        pVect1 += G_FOUR;
        v2 = _mm_loadu_ps(pVect2);
        pVect2 += G_FOUR;
        diff = _mm_sub_ps(v1, v2);
        sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff));
    }
    _mm_store_ps(TmpRes, sum);
    return TmpRes[0] + TmpRes[1] + TmpRes[G_TWO] + TmpRes[G_THREE];
}

float L2DisHnsw::L2SqrSIMD4ExtResiduals(const void *pVect1v, const void *pVect2v, const void *qty_ptr)
{
    size_t qty = *((size_t *)qty_ptr);
    size_t qty4 = qty >> G_TWO << G_TWO;

    float res = L2SqrSIMD4Ext(pVect1v, pVect2v, &qty4);
    size_t qty_left = qty - qty4;

    float *pVect1 = (float *)pVect1v + qty4;
    float *pVect2 = (float *)pVect2v + qty4;
    float res_tail = L2Sqr(pVect1, pVect2, &qty_left);

    return (res + res_tail);
}
#endif

}  // namespace ascendsearch
