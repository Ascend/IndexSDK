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


#include <gtest/gtest.h>
#include <mockcpp/mockcpp.hpp>
#include "ascendfaiss/ascenddaemon/utils/Random.h"

using namespace testing;
using namespace std;

namespace ascend {

class TestRandom : public Test {
public:
    void TearDown() override
    {
        GlobalMockObject::verify();
    }
};

TEST_F(TestRandom, genearte)
{
    RandomGenerator generator;
    generator.RandInt();
    generator.RandInt64();
    generator.RandUnsignedInt(0);
    generator.RandUnsignedInt(1);
    generator.RandFloat();
    generator.RandDouble();

    int64_t seed = 2;
    std::vector<int> perm;
    RandPerm(perm.data(), perm.size(), seed);

    perm.resize(seed);
    RandPerm(perm.data(), perm.size(), seed);
}
} // namespace ascend