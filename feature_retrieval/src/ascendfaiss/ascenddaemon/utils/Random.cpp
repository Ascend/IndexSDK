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


#include "Random.h"


namespace ascend {
RandomGenerator::RandomGenerator(int64_t seed) : mt(static_cast<uint32_t>(seed)) {}

int RandomGenerator::RandInt()
{
    return mt() & 0x7fffffff;
}

int64_t RandomGenerator::RandInt64()
{
    // Use rand int32 to generate rand int64, move 31 bits left to get significant parts
    return static_cast<int64_t>(static_cast<uint64_t>(RandInt()) | (static_cast<uint64_t>(RandInt()) << 31));
}

size_t RandomGenerator::RandUnsignedInt(size_t max)
{
    if (max == 0) {
        return 0;
    }
    return mt() % max;
}

float RandomGenerator::RandFloat()
{
    return mt() / static_cast<float>(mt.max());
}

double RandomGenerator::RandDouble()
{
    return mt() / static_cast<double>(mt.max());
}

void RandPerm(int *perm, size_t n, int64_t seed)
{
    for (size_t i = 0; i < n; i++) {
        perm[i] = static_cast<int>(i);
    }

    RandomGenerator rng(seed);

    for (size_t i = 0; i + 1 < n; i++) {
        size_t i2 = i + rng.RandUnsignedInt(n - i);
        std::swap(perm[i], perm[i2]);
    }
}
}
