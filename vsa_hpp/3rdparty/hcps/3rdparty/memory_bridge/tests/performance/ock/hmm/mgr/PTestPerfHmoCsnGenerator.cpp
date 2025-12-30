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

#include <ostream>
#include <cstdint>
#include <gtest/gtest.h>
#include <chrono>
#include "ock/hmm/mgr/OckHmmHMObjectExt.h"
#include "ock/conf/OckSysConf.h"
#include "ptest/ptest.h"

namespace ock {
namespace utils {
class PTestPerfHmoCsnGenerator : public testing::Test {
public:
    void BuildHighLoadScene(void)
    {
        uint32_t highLoadHmoCount = conf::OckSysConf::HmmConf().maxHMOCountPerDevice;
        for (uint32_t i = 0; i < conf::MaxHMOIDNumberPerDevice; ++i) {
            auto ret = generator.NewId();
            if (i > highLoadHmoCount) {
                generator.DelId(ret.second);
            }
        }
    }
    void DoNewAndDelId(uint32_t testTimes = 10000)
    {
        for (uint32_t i = 0; i < testTimes; ++i) {
            auto ret = generator.NewId();
            generator.DelId(ret.second);
        }
    }
    hmm::HmoCsnGenerator generator;
};
TEST_F(PTestPerfHmoCsnGenerator, performance_test)
{
    BuildHighLoadScene();
    auto timeGuard = fast::hdt::TestTimeGuard();
    DoNewAndDelId();
    EXPECT_TRUE(FAST_PTEST().Test(
        "OCK.MemoryBridge.HMM.HmoCsnGen.HighLoadGenerator", "NewDelTime", timeGuard.ElapsedMicroSeconds()));
}
}  // namespace utils
}  // namespace ock