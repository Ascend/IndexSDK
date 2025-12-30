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


#include <ascenddaemon/impl/VectorTransform.h>
#include <ascenddaemon/impl/AuxIndexStructures.h>
#include <gtest/gtest.h>
#include <iostream>
#include <omp.h>
#include <cmath>

namespace {
TEST(TestLinearTranform, UpdateAndApply)
{
    // create LinearTransform
    const int dimIn = 512;
    const int dimOut = 128;

    ascend::AscendResourcesProxy resources;
    resources.initialize();
    ascend::LinearTransform ltrans(dimIn, dimOut);

    unsigned int seed = time(nullptr);
    srand(seed);
    std::vector<float16_t> matrix(dimIn * dimOut);
#pragma omp parallel for
    for (size_t i = 0; i < matrix.size(); i++) {
        matrix[i] = (float16_t)i / 100;
    }
    std::vector<float> bias(dimOut);
#pragma omp parallel for
    for (size_t i = 0; i < bias.size(); i++) {
        bias[i] = (float)(512 - i);
    }
    ascend::AscendTensor<float16_t, ascend::DIMS_2> mm(matrix.data(), { dimOut, dimIn });
    ascend::AscendTensor<float, ascend::DIMS_1> bb(bias.data(), { dimOut });
    ltrans.updateTrainedValue(mm, bb);
    std::cout << "update trained value success" << std::endl;

    // apply
    const std::vector<int> nums { 1, 3, 7, 15, 31, 63, 127, 255 };
    for (auto num : nums) {
        ascend::DeviceScope device;
        auto stream = resources.getDefaultStream();
        auto &mem = resources.getMemoryManager();
        ascend::AscendTensor<float16_t, ascend::DIMS_2> x(mem, { num, dimIn }, stream);
        x.initValue(1.0);
        ascend::AscendTensor<float16_t, ascend::DIMS_2> xt(mem, { num, dimOut }, stream);

        ltrans.apply(num, x.data(), xt.data(), stream);
        std::cout << "----- start apply, num_vec is " << num << "-----" << std::endl;
        double cost = 0.0;
        double min = std::numeric_limits<double>::max();
        double max = 0.0;

        const int loopSearch = 100;
        for (int i = 0; i < loopSearch; i++) {
            double start = ascend::utils::getMillisecs();
            ltrans.apply(num, x.data(), xt.data(), stream);
            double end = ascend::utils::getMillisecs();
            double cur_cost = end - start;
            min = std::min(min, cur_cost);
            max = std::max(max, cur_cost);
            cost += cur_cost;
        }

        std::cout << "apply cost time: " << cost / (loopSearch * num) << "ms in average, "
                  << "max:" << max / num << "ms, min:" << min / num << "ms." << std::endl;
    }
}
} // namespace

int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);

    ascend::DeviceScope deviceScope;
    return RUN_ALL_TESTS();
}
