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

#include <iostream>
#include <memory>
#include <syslog.h>
#include "ock/tools/topo/TopoDetectParam.h"
#include "ock/tools/topo/TopoDetectParamVerify.h"
#include "ock/tools/topo/TopoDetectChecker.h"
#include "ock/tools/topo/TopoDetectResult.h"
#include "ock/hmm/mgr/OckHmmErrorCode.h"
#include "ock/log/OckHmmLogHandler.h"
#include "ock/log/OckLogger.h"

int main(int argc, char **argv)
{
    openlog("TopoDetect", LOG_CONS | LOG_PID, LOG_USER);
    ock::OckHmmSetLogHandler(std::make_shared<ock::NullOckHmmLogHandler>());
    std::string cmdStr = ock::tools::topo::TopoDetectParam::ArgToString(argc, argv);
    auto param = ock::tools::topo::TopoDetectParam::ParseArgs(argc, argv);
    if (param.get() == nullptr) {
        syslog(LOG_INFO, "Invalid input param. cmd=%s\n", cmdStr.c_str());
        closelog();
        return ock::hmm::HMM_ERROR_EXEC_INVALID_INPUT_PARAM;
    }
    auto checkResult = ock::tools::topo::TopoDetectParamVerify::CheckAll(*param);
    if (checkResult->retCode != ock::hmm::HMM_SUCCESS) {
        std::cerr << checkResult->msg << std::endl;
        syslog(LOG_INFO, "Invalid input param. cmd=%s\n", cmdStr.c_str());
        closelog();
        return ock::hmm::HMM_ERROR_EXEC_INVALID_INPUT_PARAM;
    }
    auto dectecter = ock::tools::topo::TopoDetectChecker::Create(param);
    auto resultVec = dectecter->Check();
    std::cout << resultVec << std::endl;
    syslog(LOG_INFO, "cmd=%s Succeed.\n", cmdStr.c_str());
    closelog();
    return 0;
}