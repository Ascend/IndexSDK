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
#include "acl/acl.h"
#include "ock/log/OckHmmLogHandler.h"
#include "ock/hcps/modelpath/OckModelPath.h"
#include "ock/hcps/modelpath/OckSetModelPath.h"
#include "ock/vsa/neighbor/hpp/OckVsaAnnHppSetup.h"

int main(int argc, char **argv)
{
    ock::OckHmmSetLogLevel(2U);

    ock::vsa::neighbor::hpp::SetUpHPPTsFactory();
    const char *modelpath = std::getenv("MX_INDEX_MODELPATH");
    ock::hcps::OckModelPath::Instance().SetPath(modelpath);
    ock::hcps::OckSetModelPath::Instance().NotifyDevice();

    aclInit(nullptr);
    aclrtSetDevice(0);
    testing::InitGoogleTest(&argc, argv);
    auto ret = RUN_ALL_TESTS();

    aclrtResetDevice(0);
    aclFinalize();
    return ret;
}
