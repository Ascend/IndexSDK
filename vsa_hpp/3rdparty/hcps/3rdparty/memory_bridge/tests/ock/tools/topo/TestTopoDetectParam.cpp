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
#include <string>
#include <gtest/gtest.h>
#include <getopt.h>
#include "securec.h"
#include "ock/utils/StrUtils.h"
#include "ock/tools/topo/TopoDetectParam.h"

namespace ock {
namespace tools {
namespace topo {
namespace tests {

class TestTopoDetectParam : public testing::Test {
public:
    class InputParamBuilder {
    public:
        ~InputParamBuilder(void) noexcept
        {
            if (argc == 0 || argv == nullptr) {
                return;
            }
            for (int i = 0; i < argc; ++i) {
                delete[] argv[i];
            }
            delete[] argv;
            argc = 0;
        }
        explicit InputParamBuilder(const std::string &buff) : argc(0), argv(nullptr)
        {
            std::vector<std::string> tokens;
            utils::Split(buff, " ", tokens);
            argc = tokens.size();
            argv = new char *[tokens.size()];
            for (size_t i = 0; i < tokens.size(); ++i) {
                argv[i] = new char[tokens[i].size() + 1];
                argv[i][tokens[i].size()] = '\0';
                memcpy_s(argv[i], tokens[i].size() + 1, tokens[i].c_str(), tokens[i].size());
            }
        }
        int argc;
        char **argv;
    };
    void CheckDeviceInfoExists(hmm::OckHmmDeviceId deviceId, uint32_t startId, uint32_t endId)
    {
        ASSERT_TRUE(param.get() != nullptr);
        auto &info = param->GetDeviceInfo();
        auto iter = info.find(DeviceCpuSet(deviceId));
        ASSERT_TRUE(iter != info.end());
        bool bFindCpuRange = false;
        for (auto &range : iter->cpuIds) {
            if (range.startId == startId && range.endId == endId) {
                bFindCpuRange = true;
            }
        }
        EXPECT_TRUE(bFindCpuRange);
    }
    void CheckDeviceInfoExists(hmm::OckHmmDeviceId deviceId, uint32_t startId)
    {
        CheckDeviceInfoExists(deviceId, startId, startId);
    }
    void ParseErrorCheck(const std::string &buff)
    {
        auto inputArgs = std::make_shared<InputParamBuilder>(buff);
        param = TopoDetectParam::ParseArgs(inputArgs->argc, inputArgs->argv);
        ASSERT_TRUE(param.get() == nullptr);
    }
    void ParseGoodCheck(
        const std::string &model, const std::string &inputArgBuf, hmm::OckHmmDeviceId deviceId = 0, uint32_t cpuId = 4)
    {
        auto inputArgs = std::make_shared<InputParamBuilder>(inputArgBuf);
        param = TopoDetectParam::ParseArgs(inputArgs->argc, inputArgs->argv);
        ASSERT_TRUE(param.get() != nullptr);
        CheckDeviceInfoExists(deviceId, cpuId);
        EXPECT_NE(param->GetModel(), DetectModel::UNKNOWN);
        EXPECT_EQ(utils::ToString(param->GetModel()), model);
    }
    std::string ToString(uint64_t data1, uint64_t data2, uint64_t data3)
    {
        std::ostringstream osStr;
        osStr << data1 << "," << data2 << "," << data3;
        return osStr.str();
    }
    void ParseGoodCheckPNS(const std::string &params, const std::string &inputArgBuf)
    {
        auto inputArgs = std::make_shared<InputParamBuilder>(inputArgBuf);
        param = TopoDetectParam::ParseArgs(inputArgs->argc, inputArgs->argv);
        ASSERT_TRUE(param.get() != nullptr);
        EXPECT_EQ(params, ToString(param->ThreadNumPerDevice(), param->TestTime(), param->PackageMBytesPerTransfer()));
    }
    void ParseGoodCheckCpuId(const std::string &cpuId, const std::string &inputArgBuf)
    {
        auto inputArgs = std::make_shared<InputParamBuilder>(inputArgBuf);
        param = TopoDetectParam::ParseArgs(inputArgs->argc, inputArgs->argv);
        ASSERT_TRUE(param.get() != nullptr);
        auto &info = param->GetDeviceInfo();
        auto iter = info.find(DeviceCpuSet(0));
        EXPECT_EQ(utils::ToString(*iter), cpuId);
    }
    std::shared_ptr<TopoDetectParam> param;
};

TEST_F(TestTopoDetectParam, parseArgs_correct_while_one_cpu_id)
{
    ParseGoodCheck("SERIAL", "./TopoDetect -m SERIAL -d 0:4 -t HOST_TO_HOST");
    ParseGoodCheck("SERIAL", "./TopoDetect -m serial -d 0:4 -t HOST_TO_DEVICE");
    ParseGoodCheck("SERIAL", "./TopoDetect  -d 0:4 -m SERIAL -t device_to_device");
    ParseGoodCheck("SERIAL", "./TopoDetect  -d  0:4 -m  SERIAL -t device_to_host");
    ParseGoodCheck("PARALLEL", "./TopoDetect  -d 0:4 -m PARALLEL");
    ParseGoodCheck("PARALLEL", "./TopoDetect  -d  0:4 -m  PARALLEL");
}
TEST_F(TestTopoDetectParam, parseArgs_correct_while_cpu_id_range)
{
    ParseGoodCheckCpuId("0:4-7", "./TopoDetect -m SERIAL -d 0:4-7 -t HOST_TO_HOST");
    ParseGoodCheckCpuId("0:1-5,3-7", "./TopoDetect -m SERIAL -d 0:1-5,3-7 -t HOST_TO_HOST");
    ParseGoodCheckCpuId("0:1-5,6-10", "./TopoDetect -m SERIAL -d 0:1-5,6-10 -t HOST_TO_HOST");
    ParseGoodCheckCpuId("0:1-5,6,10,15-19", "./TopoDetect -m SERIAL -d 0:1-5,6,10,15-19 -t HOST_TO_HOST");
    ParseGoodCheckCpuId("0:3,5-7", "./TopoDetect -m SERIAL -d 0:3,5-7 -t HOST_TO_HOST");
}
TEST_F(TestTopoDetectParam, parseArgs_correct_check_threadnum_testtime_packagesize)
{
    ParseGoodCheckPNS("1,2,64", "./TopoDetect -m SERIAL -d 0:4 -t HOST_TO_HOST -p 1");
    ParseGoodCheckPNS("3,2,64", "./TopoDetect -m SERIAL -d 0:4 -t HOST_TO_HOST -p 3");
    ParseGoodCheckPNS("5,2,64", "./TopoDetect -m SERIAL -d 0:4 -t HOST_TO_HOST -p 5");
    ParseGoodCheckPNS("2,1,64", "./TopoDetect -m SERIAL -d 0:4 -t HOST_TO_HOST -n 1");
    ParseGoodCheckPNS("2,5,64", "./TopoDetect -m SERIAL -d 0:4 -t HOST_TO_HOST -n 5");
    ParseGoodCheckPNS("2,2,2", "./TopoDetect -m SERIAL -d 0:4 -t HOST_TO_HOST -s 2");
    ParseGoodCheckPNS("2,2,2047", "./TopoDetect -m SERIAL -d 0:4 -t HOST_TO_HOST -s 2047");
}
TEST_F(TestTopoDetectParam, parseArgs_incorrect_while_error_input)
{
    ParseErrorCheck("./TopoDetect -m SERIAL -d 0:x"); // CPU ID不能为字母
    ParseErrorCheck("./TopoDetect -m XX -d 0:1"); // Detection mode输入不合法
    ParseErrorCheck("./TopoDetect -m UNKOWN -d 0:1"); // Detection mode不能为空
    ParseErrorCheck("./TopoDetect -m SERIAL -d 0,1"); // 输入格式错误，设备ID与CPU ID使用冒号间隔
    ParseErrorCheck("./TopoDetect -m SERIAL -d 0:1?2"); // CPU ID之间使用逗号间隔
    ParseErrorCheck("./TopoDetect -m SERIAL -d a:1"); // 设备ID不能为字母
    ParseErrorCheck("./TopoDetect -d"); // 参数不完整
    ParseErrorCheck("./TopoDetect  -d  0:4 -m  PARALLEL -t device"); // -t参数不合法
    ParseErrorCheck("./TopoDetect -m SERIAL -d 0:4 -t HOST_TO_HOST -p 6"); // 线程数越界，上限为5
    ParseErrorCheck("./TopoDetect -m SERIAL -d 0:4 -t HOST_TO_HOST -p k"); // 线程数不能为字母
    ParseErrorCheck("./TopoDetect -m SERIAL -d 0:4 -t HOST_TO_HOST -p 0"); // 线程数越界，下限为1
    ParseErrorCheck("./TopoDetect -m SERIAL -d 0:4 -t HOST_TO_HOST -n 0"); // 越界，下限为1
    ParseErrorCheck("./TopoDetect -m SERIAL -d 0:4 -t HOST_TO_HOST -n a"); // 重复次数不能为字母
    ParseErrorCheck("./TopoDetect -m SERIAL -d 0:4 -t HOST_TO_HOST -s 1"); // 包大小下限为1M
    ParseErrorCheck("./TopoDetect -m SERIAL -d 0:4 -t HOST_TO_HOST -s 2049"); // 包大小上限为2048M
    ParseErrorCheck("./TopoDetect -m SERIAL -d 0:4 -t HOST_TO_HOST -s a"); // 包大小不能为字母
}
TEST(TestDetectModel, print)
{
    EXPECT_EQ(utils::ToString(DetectModel::PARALLEL), "PARALLEL");
    EXPECT_EQ(utils::ToString(DetectModel::SERIAL), "SERIAL");
    EXPECT_EQ(utils::ToString(DetectModel::UNKNOWN), "UNKNOWN");
}
}  // namespace tests
}  // namespace topo
}  // namespace tools
}  // namespace ock