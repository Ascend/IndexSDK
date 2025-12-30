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


#ifndef ASCEND_RANDOM_GEN_INCLUDED
#define ASCEND_RANDOM_GEN_INCLUDED

#include <random>
#include <stdint.h>

namespace ascend {
struct RandomGenerator {
    std::mt19937 mt;

    // random positive integer
    int RandInt();

    // random int64_t
    int64_t RandInt64();

    // generate random integer between 0 and max-1
    size_t RandUnsignedInt(size_t max);

    // between 0 and 1
    float RandFloat();

    double RandDouble();

    explicit RandomGenerator(int64_t seed = 1234);
};

/* random permutation */
void RandPerm(int *perm, size_t n, int64_t seed);
}
#endif