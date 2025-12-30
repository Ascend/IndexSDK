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


#include "gtest/gtest.h"
#include "ock/utils/StrUtils.h"
#include "ock/conf/OckHmmDeviceInfoConf.h"

namespace ock {
namespace conf {
namespace test {

class TestOckHmmDeviceInfoConf : public testing::Test {
public:
    template<typename T, typename FunT>
    void NearMinValueTestScene(const OckHmmDeviceInfoConf &confA, OckHmmDeviceInfoConf &confB,
                               const ParamRange<T> &range, FunT fun)
    {
        fun(confB, range.minValue + 1);
        EXPECT_NE(confA, confB);
        EXPECT_NE(utils::ToString(confA), utils::ToString(confB));

        fun(confB, range.minValue);
        EXPECT_EQ(confA, confB);
        EXPECT_EQ(utils::ToString(confA), utils::ToString(confA));
    }

    template<typename T, typename FunT>
    void NearMaxValueTestScene(const OckHmmDeviceInfoConf &confA, OckHmmDeviceInfoConf &confB,
                               const ParamRange<T> &range, FunT fun)
    {
        fun(confB, range.maxValue + 1);
        EXPECT_NE(confA, confB);
        EXPECT_NE(utils::ToString(confA), utils::ToString(confB));

        fun(confB, range.maxValue);
        EXPECT_EQ(confA, confB);
        EXPECT_EQ(utils::ToString(confA), utils::ToString(confA));
    }
};

TEST_F(TestOckHmmDeviceInfoConf, operator)
{
    OckHmmDeviceInfoConf confA;
    OckHmmDeviceInfoConf confB;

    EXPECT_EQ(confA, confB);
    EXPECT_EQ(utils::ToString(confA), utils::ToString(confA));

    NearMinValueTestScene(confA, confB, confA.deviceId, [](OckHmmDeviceInfoConf &conf, hmm::OckHmmDeviceId newId)
                          { conf.deviceId.minValue = newId; });
    NearMaxValueTestScene(confA, confB, confA.deviceId, [](OckHmmDeviceInfoConf &conf, hmm::OckHmmDeviceId newId)
                          { conf.deviceId.maxValue = newId; });

    NearMinValueTestScene(confA, confB, confA.transferThreadNum, [](OckHmmDeviceInfoConf &conf, uint32_t newNum)
                          { conf.transferThreadNum.minValue = newNum; });
    NearMaxValueTestScene(confA, confB, confA.transferThreadNum, [](OckHmmDeviceInfoConf &conf, uint32_t newNum)
                          { conf.transferThreadNum.maxValue = newNum; });

    NearMinValueTestScene(confA, confB, confA.deviceBaseCapacity, [](OckHmmDeviceInfoConf &conf, uint64_t newSize)
                          { conf.deviceBaseCapacity.minValue = newSize; });
    NearMaxValueTestScene(confA, confB, confA.deviceBaseCapacity, [](OckHmmDeviceInfoConf &conf, uint64_t newSize)
                          { conf.deviceBaseCapacity.maxValue = newSize; });

    NearMinValueTestScene(confA, confB, confA.deviceBufferCapacity, [](OckHmmDeviceInfoConf &conf, uint64_t newSize)
                          { conf.deviceBufferCapacity.minValue = newSize; });
    NearMaxValueTestScene(confA, confB, confA.deviceBufferCapacity, [](OckHmmDeviceInfoConf &conf, uint64_t newSize)
                          { conf.deviceBufferCapacity.maxValue = newSize; });

    NearMinValueTestScene(confA, confB, confA.hostBaseCapacity, [](OckHmmDeviceInfoConf &conf, uint64_t newSize)
                          { conf.hostBaseCapacity.minValue = newSize; });
    NearMaxValueTestScene(confA, confB, confA.hostBaseCapacity, [](OckHmmDeviceInfoConf &conf, uint64_t newSize)
                          { conf.hostBaseCapacity.maxValue = newSize; });

    NearMinValueTestScene(confA, confB, confA.hostBufferCapacity, [](OckHmmDeviceInfoConf &conf, uint64_t newSize)
                          { conf.hostBufferCapacity.minValue = newSize; });
    NearMaxValueTestScene(confA, confB, confA.hostBufferCapacity, [](OckHmmDeviceInfoConf &conf, uint64_t newSize)
                          { conf.hostBufferCapacity.maxValue = newSize; });
}
}
}
}