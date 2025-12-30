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


#ifndef GFEATURERETRIEVAL_DISTANCECOMPUTER_H
#define GFEATURERETRIEVAL_DISTANCECOMPUTER_H

#ifndef NO_MANUAL_VECTORIZATION
#ifdef __SSE__
#define USE_SSE
#ifdef __AVX__
#define USE_AVX
#ifdef __AVX512F__
#define USE_AVX512
#endif
#endif
#endif
#endif

#if defined(USE_AVX) || defined(USE_SSE)
#ifdef _MSC_VER
#include <intrin.h>
#include <stdexcept>
#include "cpu_x86.h"
void cpu_x86::cpuid(int32_t out[4], int32_t eax, int32_t ecx);
__int64 xgetbv(unsigned int x);
#else
#include <x86intrin.h>
#include <cpuid.h>
#include <stdint.h>
void CpuId(int32_t cpuInfo[4], int32_t eax, int32_t ecx);
uint64_t Xgetbv(unsigned int index);
#endif

#if defined(USE_AVX512)
#include <immintrin.h>
#endif

#if defined(__GNUC__)
#define PORTABLE_ALIGN32 __attribute__((aligned(32)))
#define PORTABLE_ALIGN64 __attribute__((aligned(64)))
#else
#define PORTABLE_ALIGN32 __declspec(align(32))
#define PORTABLE_ALIGN64 __declspec(align(64))
#endif

#define XCR_XFEATURE_ENABLED_MASK 0

bool AVXCapable();

bool AVX512Capable();
#endif

#include <algorithm>
#include <memory>
#include <mutex>
#include <stack>
#include <iostream>

#include <faiss/impl/AuxIndexStructures.h>
#include "faiss/IndexFlat.h"

const int G_TWO = 2;
const int G_THREE = 3;
const int G_FOUR = 4;
const int G_FIVE = 5;
const int G_SIX = 6;
const int G_SEVEN = 7;
const int G_EIGHT = 8;
const int G_NINE = 9;
const int G_TEN = 10;
const int G_ELEVEN = 11;
const int G_TWELVE = 12;
const int G_THIRTEEN = 13;
const int G_FOURTEEN = 14;
const int G_FIFTEEN = 15;
const int G_SIXTEEN = 16;
const int G_TWENTY_SEVEN = 27;
const int G_TWENTY_EIGHT = 28;
const int G_THIRTY_TWO = 32;

namespace ascendsearch {
template <typename MTYPE>
using DISTFUNC = MTYPE (*)(const void *, const void *, const void *);
using idx_t = int64_t;


struct IPDisHnsw : faiss::DistanceComputer {
    size_t d;
    int64_t nb;
    const float *q_;
    const float *b;
    size_t ndis;
    DISTFUNC<float> fstdistfunc_;  // 距离计算函数
    size_t data_size_;

    // / compute distance of vector i to current query
    float operator()(int64_t i) override;

    // / compute distance between two stored vectors
    float symmetric_dis(int64_t i, int64_t j) override;

    void set_query(const float *x) override;

    // / 以下为自定义函数
    size_t get_data_size();

    DISTFUNC<float> get_dist_func();

    void *get_dist_func_param();

    // / 以下为IP距离计算函数
    IPDisHnsw();
    IPDisHnsw(const faiss::IndexFlat &storage, const float *q = nullptr);
    ~IPDisHnsw();

    static float InnerProduct(const void *pVect1, const void *pVect2, const void *qty_ptr);
    static float InnerProductDistance(const void *pVect1, const void *pVect2, const void *qty_ptr);

#if defined(USE_AVX)

    static float InnerProductSIMD4ExtAVX(const void *pVect1v, const void *pVect2v, const void *qty_ptr);
    static float InnerProductDistanceSIMD4ExtAVX(const void *pVect1v, const void *pVect2v, const void *qty_ptr);
#endif

#if defined(USE_SSE)
    static float InnerProductSIMD4ExtSSE(const void *pVect1v, const void *pVect2v, const void *qty_ptr);
    static float InnerProductDistanceSIMD4ExtSSE(const void *pVect1v, const void *pVect2v, const void *qty_ptr);
#endif

#if defined(USE_AVX512)
    static float InnerProductSIMD16ExtAVX512(const void *pVect1v, const void *pVect2v, const void *qty_ptr);
    static float InnerProductDistanceSIMD16ExtAVX512(const void *pVect1v, const void *pVect2v, const void *qty_ptr);
#endif

#if defined(USE_AVX)
    static float InnerProductSIMD16ExtAVX(const void *pVect1v, const void *pVect2v, const void *qty_ptr);
    static float InnerProductDistanceSIMD16ExtAVX(const void *pVect1v, const void *pVect2v, const void *qty_ptr);
#endif

#if defined(USE_SSE)
    static float InnerProductSIMD16ExtSSE(const void *pVect1v, const void *pVect2v, const void *qty_ptr);
    static float InnerProductDistanceSIMD16ExtSSE(const void *pVect1v, const void *pVect2v, const void *qty_ptr);
#endif

#if defined(USE_SSE) || defined(USE_AVX) || defined(USE_AVX512)
    static float InnerProductDistanceSIMD16ExtResiduals(const void *pVect1v, const void *pVect2v, const void *qty_ptr);
    static float InnerProductDistanceSIMD4ExtResiduals(const void *pVect1v, const void *pVect2v, const void *qty_ptr);
#endif
};

struct L2DisHnsw : faiss::DistanceComputer {
    size_t d;
    int64_t nb;
    const float *q_;
    const float *b;
    size_t ndis;
    DISTFUNC<float> fstdistfunc_;  // 距离计算函数
    size_t data_size_;

    // / compute distance of vector i to current query
    float operator()(int64_t i) override;

    // / compute distance between two stored vectors
    float symmetric_dis(int64_t i, int64_t j) override;

    void set_query(const float *x) override;

    // / 以下为自定义函数
    size_t get_data_size();

    DISTFUNC<float> get_dist_func();

    void *get_dist_func_param();

    // / 以下为IP距离计算函数
    L2DisHnsw();
    L2DisHnsw(const faiss::IndexFlat &storage, const float *q = nullptr);
    ~L2DisHnsw();

    static float L2Sqr(const void *pVect1v, const void *pVect2v, const void *qty_ptr);

#if defined(USE_AVX512)
    static float L2SqrSIMD16ExtAVX512(const void *pVect1v, const void *pVect2v, const void *qty_ptr);
#endif

#if defined(USE_AVX)
    static float L2SqrSIMD16ExtAVX(const void *pVect1v, const void *pVect2v, const void *qty_ptr);
#endif

#if defined(USE_SSE)
    static float L2SqrSIMD16ExtSSE(const void *pVect1v, const void *pVect2v, const void *qty_ptr);
#endif

#if defined(USE_SSE) || defined(USE_AVX) || defined(USE_AVX512)
    static float L2SqrSIMD16ExtResiduals(const void *pVect1v, const void *pVect2v, const void *qty_ptr);
#endif

#if defined(USE_SSE)
    static float L2SqrSIMD4Ext(const void *pVect1v, const void *pVect2v, const void *qty_ptr);
    static float L2SqrSIMD4ExtResiduals(const void *pVect1v, const void *pVect2v, const void *qty_ptr);
#endif
};
}  // namespace ascendsearch

#endif  // GFEATURERETRIEVAL_DISTANCECOMPUTER_H
