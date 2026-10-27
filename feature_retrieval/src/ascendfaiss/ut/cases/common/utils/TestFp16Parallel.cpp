/*
 * This file is part of the IndexSDK project.
 * Copyright (c) 2026 Huawei Technologies Co.,Ltd.
 * Licensed under Mulan PSL v2: http://license.coscl.org.cn/MulanPSL2
 */

#include <gtest/gtest.h>

#include <limits>
#include <vector>

#include "ascend/utils/fp16.h"

TEST(TestFp16Parallel, MatchesSerialConversion)
{
    constexpr int halfValueCount = 65536;
    std::vector<float> input(halfValueCount);
    for (int i = 0; i < halfValueCount; ++i)
    {
        input[i] = static_cast<float>(faiss::ascend::fp16(static_cast<uint16_t>(i)));
    }
    input.insert(input.end(), {0.0f, -0.0f, 1.00048828125f, -1.00048828125f, std::numeric_limits<float>::max(),
                               std::numeric_limits<float>::lowest(), std::numeric_limits<float>::denorm_min(),
                               std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity(),
                               std::numeric_limits<float>::quiet_NaN()});
    const int count = static_cast<int>(input.size());
    std::vector<uint16_t> serial(count);
    std::vector<uint16_t> parallel(count);
    for (int i = 0; i < count; ++i)
    {
        serial[i] = faiss::ascend::fp16(input[i]).data;
    }
    for (int threads : {1, 2, 4})
    {
#pragma omp parallel for num_threads(threads)
        for (int i = 0; i < count; ++i)
        {
            parallel[i] = faiss::ascend::fp16(input[i]).data;
        }
        EXPECT_EQ(serial, parallel) << "threads=" << threads;
    }
}
