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
#include <map>
#include <thread>
#include "ascendfaiss/ascenddaemon/utils/Limits.h"
#include "ascendfaiss/common/utils/CommonUtils.h"
#include "ascendfaiss/common/utils/SocUtils.h"
using namespace testing;
using namespace std;

namespace ascend {

class TestUtils : public Test {
public:
    void TearDown() override
    {
        GlobalMockObject::verify();
    }
};

constexpr int32_t MAX_THREAD_NUM = 256;
const std::string OMP_NUM_THREADS = "OMP_NUM_THREADS";


TEST_F(TestUtils, get_omp_by_core_max)
{
    // 打桩core超过MAX_THREAD_NUM
    MOCKER_CPP(&thread::hardware_concurrency).stubs().will(returnValue(MAX_THREAD_NUM + 1));

    int32_t ompNumThreads = MAX_THREAD_NUM;
    CommonUtils::GetOmpNumThreads(ompNumThreads);
}

TEST_F(TestUtils, get_omp_by_core_min)
{
    // 打桩core为1
    int32_t mockCoreNum = 1;
    MOCKER_CPP(&thread::hardware_concurrency).stubs().will(returnValue(mockCoreNum));

    int32_t ompNumThreads = MAX_THREAD_NUM;
    CommonUtils::GetOmpNumThreads(ompNumThreads);
    EXPECT_EQ(ompNumThreads, mockCoreNum);
}

TEST_F(TestUtils, get_omp_by_env_invalid_size_empty)
{
    // 打桩core为56
    int32_t mockCoreNum = 56;
    MOCKER_CPP(&thread::hardware_concurrency).stubs().will(returnValue(mockCoreNum));
    
    // 空的字符串
    char *mockEnv = "";
    MOCKER_CPP(&getenv).stubs().will(returnValue(mockEnv));

    int32_t ompNumThreads = MAX_THREAD_NUM;
    CommonUtils::GetOmpNumThreads(ompNumThreads);
    EXPECT_EQ(ompNumThreads, mockCoreNum);
}

TEST_F(TestUtils, get_omp_by_env_invalid_size_too_long)
{
    // 打桩core为56
    int32_t mockCoreNum = 56;
    MOCKER_CPP(&thread::hardware_concurrency).stubs().will(returnValue(mockCoreNum));

    // 长度大了
    char *mockEnv = "1000";
    MOCKER_CPP(&getenv).stubs().will(returnValue(mockEnv));
    int32_t ompNumThreads = MAX_THREAD_NUM;
    CommonUtils::GetOmpNumThreads(ompNumThreads);
    EXPECT_EQ(ompNumThreads, mockCoreNum);
}

TEST_F(TestUtils, get_omp_by_env_no_number)
{
    int32_t mockCoreNum = 56;
    MOCKER_CPP(&thread::hardware_concurrency).stubs().will(returnValue(mockCoreNum));
    
    // 非数字
    char *mockEnv = "abc";
    MOCKER_CPP(&getenv).stubs().will(returnValue(mockEnv));
    int32_t ompNumThreads = MAX_THREAD_NUM;
    CommonUtils::GetOmpNumThreads(ompNumThreads);
    EXPECT_EQ(ompNumThreads, mockCoreNum);
}

TEST_F(TestUtils, get_omp_by_env_negative_number)
{
    int32_t mockCoreNum = 56;
    MOCKER_CPP(&thread::hardware_concurrency).stubs().will(returnValue(mockCoreNum));
    
    // 非数字
    char *mockEnv = "-1";
    MOCKER_CPP(&getenv).stubs().will(returnValue(mockEnv));
    int32_t ompNumThreads = MAX_THREAD_NUM;
    CommonUtils::GetOmpNumThreads(ompNumThreads);
    EXPECT_EQ(ompNumThreads, mockCoreNum);
}

TEST_F(TestUtils, get_omp_by_env_too_big)
{
    // 打桩core为56
    int32_t mockCoreNum = 56;
    MOCKER_CPP(&thread::hardware_concurrency).stubs().will(returnValue(mockCoreNum));
    
    // 非数字
    char *mockEnv = "257";
    MOCKER_CPP(&getenv).stubs().will(returnValue(mockEnv));
    int32_t ompNumThreads = MAX_THREAD_NUM;
    CommonUtils::GetOmpNumThreads(ompNumThreads);
    EXPECT_EQ(ompNumThreads, mockCoreNum);
}

TEST_F(TestUtils, get_omp_by_env_greater_than_max)
{
    // 打桩core为56
    int32_t mockCoreNum = 56;
    MOCKER_CPP(&thread::hardware_concurrency).stubs().will(returnValue(mockCoreNum));
    
    // 值大了
    char *mockEnv = "257";
    MOCKER_CPP(&getenv).stubs().will(returnValue(mockEnv));
    int32_t ompNumThreads = MAX_THREAD_NUM;
    CommonUtils::GetOmpNumThreads(ompNumThreads);
    EXPECT_EQ(ompNumThreads, mockCoreNum);
}

TEST_F(TestUtils, get_omp_by_env_greater_than_core)
{
    // 打桩core为56
    int32_t mockCoreNum = 56;
    MOCKER_CPP(&thread::hardware_concurrency).stubs().will(returnValue(mockCoreNum));
    
    // 值比core大
    char *mockEnv = "57";
    MOCKER_CPP(&getenv).stubs().will(returnValue(mockEnv));
    int32_t ompNumThreads = MAX_THREAD_NUM;
    CommonUtils::GetOmpNumThreads(ompNumThreads);
    EXPECT_EQ(ompNumThreads, mockCoreNum);
}

TEST_F(TestUtils, get_omp_by_env_less)
{
    // 打桩core为56
    int32_t mockCoreNum = 56;
    MOCKER_CPP(&thread::hardware_concurrency).stubs().will(returnValue(mockCoreNum));
    
    // 值比core小
    int32_t setOmpNum = 55;
    char *mockEnv = "55";
    MOCKER_CPP(&getenv).stubs().will(returnValue(mockEnv));
    int32_t ompNumThreads = MAX_THREAD_NUM;
    CommonUtils::GetOmpNumThreads(ompNumThreads);
    EXPECT_EQ(ompNumThreads, setOmpNum);
}

TEST_F(TestUtils, get_mt_mode_with_env_set_value_1)
{
    char *multiThreadEnv = "1";
    MOCKER_CPP(&getenv).stubs().will(returnValue(multiThreadEnv));

    bool multiThreadFlag = false;
    AscendMultiThreadManager::GetMultiThreadMode(multiThreadFlag);
    EXPECT_EQ(multiThreadFlag, true);
}

TEST_F(TestUtils, get_mt_mode_with_env_set_value_nullptr)
{
    char *multiThreadEnv = nullptr;
    MOCKER_CPP(&getenv).stubs().will(returnValue(multiThreadEnv));

    bool multiThreadFlag = false;
    AscendMultiThreadManager::GetMultiThreadMode(multiThreadFlag);
    EXPECT_EQ(multiThreadFlag, false);
}

TEST_F(TestUtils, get_mt_mode_with_env_set_value_empty)
{
    char *multiThreadEnv = "";
    MOCKER_CPP(&getenv).stubs().will(returnValue(multiThreadEnv));

    bool multiThreadFlag = false;
    AscendMultiThreadManager::GetMultiThreadMode(multiThreadFlag);
    EXPECT_EQ(multiThreadFlag, false);
}

TEST_F(TestUtils, get_mt_mode_with_env_set_value_abc)
{
    char *multiThreadEnv = "abc";
    MOCKER_CPP(&getenv).stubs().will(returnValue(multiThreadEnv));

    bool multiThreadFlag = false;
    AscendMultiThreadManager::GetMultiThreadMode(multiThreadFlag);
    EXPECT_EQ(multiThreadFlag, false);
}

TEST_F(TestUtils, get_base_mtx_with_not_mt_mode)
{
    MOCKER_CPP(&AscendMultiThreadManager::IsMultiThreadMode).stubs().will(returnValue(false));

    vector<int> deviceList = { 0, 1, 2 };
    std::unordered_map<int, std::mutex> getBaseMtx;
    AscendMultiThreadManager::InitGetBaseMtx(deviceList, getBaseMtx);
    EXPECT_EQ(getBaseMtx.size(), 0);

    auto lock = AscendMultiThreadManager::LockGetBaseMtx(0, getBaseMtx);
    EXPECT_EQ(lock, std::nullopt);
}

TEST_F(TestUtils, get_base_mtx_with_mt_mode)
{
    MOCKER_CPP(&AscendMultiThreadManager::IsMultiThreadMode).stubs().will(returnValue(true));

    const vector<int> deviceList = { 0, 1, 2 };
    std::unordered_map<int, std::mutex> getBaseMtx;
    AscendMultiThreadManager::InitGetBaseMtx(deviceList, getBaseMtx);
    EXPECT_EQ(getBaseMtx.size(), deviceList.size());

    string actualMsg;
    const int invalidDeviceId = 4;
    try {
        AscendMultiThreadManager::LockGetBaseMtx(invalidDeviceId, getBaseMtx);
    } catch (std::exception &e) {
        actualMsg = e.what();
    }
    const string expectMsg("Invalid deviceId[" + to_string(invalidDeviceId) + "]");
    EXPECT_TRUE(actualMsg.find(expectMsg) != string::npos);

    const size_t deviceIdx1 = 1;
    const size_t deviceIdx2 = 2;
    auto lock1 = AscendMultiThreadManager::LockGetBaseMtx(deviceList[deviceIdx1], getBaseMtx);
    auto lock2 = AscendMultiThreadManager::LockGetBaseMtx(deviceList[deviceIdx2], getBaseMtx);
    std::unique_lock<std::mutex> lock1Again(getBaseMtx[deviceIdx1], std::defer_lock);
    std::unique_lock<std::mutex> lock2Again(getBaseMtx[deviceIdx2], std::defer_lock);
    EXPECT_FALSE(lock1Again.try_lock());
    EXPECT_FALSE(lock2Again.try_lock());
}

TEST_F(TestUtils, get_read_lock_return_nullptr)
{
    MOCKER_CPP(&AscendMultiThreadManager::IsMultiThreadMode).stubs().will(returnValue(false));
    std::shared_mutex mtx;
    AscendMultiThreadManager::GetReadLock(mtx);
}

TEST_F(TestUtils, get_read_lock_return_not_nullptr)
{
    MOCKER_CPP(&AscendMultiThreadManager::IsMultiThreadMode).stubs().will(returnValue(true));
    std::shared_mutex mtx;
    AscendMultiThreadManager::GetReadLock(mtx);
}

TEST_F(TestUtils, get_write_lock_return_nullptr)
{
    MOCKER_CPP(&AscendMultiThreadManager::IsMultiThreadMode).stubs().will(returnValue(false));
    std::shared_mutex mtx;
    AscendMultiThreadManager::GetWriteLock(mtx);
}

TEST_F(TestUtils, get_write_lock_return_not_nullptr)
{
    MOCKER_CPP(&AscendMultiThreadManager::IsMultiThreadMode).stubs().will(returnValue(true));
    std::shared_mutex mtx;
    AscendMultiThreadManager::GetWriteLock(mtx);
}

static int g_aclRet = 0;
const int ACL_ERROR_REPEAT_INITIALIZE = 100002;
int StubAclInit(const char *)
{
    return g_aclRet;
}

// 未设置环境变量，通过repeat判断，我们调用aclInit
TEST_F(TestUtils, exec_finalize_unset_env_init)
{
    const char *env = "MX_INDEX_FINALIZE";
    unsetenv(env);
    MOCKER_CPP(aclInit).expects(once()).will(invoke(StubAclInit));
    MOCKER_CPP(aclFinalize).expects(exactly(1)).will(returnValue(0));

    faiss::ascend::SocUtils utils;
}

// 未设置环境变量，通过repeat判断，不是我们调用aclInit
TEST_F(TestUtils, exec_finalize_unset_env_not_init)
{
    g_aclRet = ACL_ERROR_REPEAT_INITIALIZE;
    const char *env = "MX_INDEX_FINALIZE";
    unsetenv(env);
    MOCKER_CPP(aclInit).expects(once()).will(invoke(StubAclInit));
    MOCKER_CPP(aclFinalize).expects(exactly(0)).will(returnValue(0));

    faiss::ascend::SocUtils utils;
    g_aclRet = 0;
}

// 设置无效环境变量，通过repeat判断，我们调用aclInit
TEST_F(TestUtils, exec_finalize_set_invalid_env_init)
{
    const char *env = "MX_INDEX_FINALIZE";
    setenv(env, "1000", 1);
    MOCKER_CPP(aclInit).expects(once()).will(invoke(StubAclInit));
    MOCKER_CPP(aclFinalize).expects(exactly(1)).will(returnValue(0));

    faiss::ascend::SocUtils utils;
}

// 设置无效环境变量，通过repeat判断，不是我们调用aclInit
TEST_F(TestUtils, exec_finalize_set_invalid_env_not_init)
{
    g_aclRet = ACL_ERROR_REPEAT_INITIALIZE;
    const char *env = "MX_INDEX_FINALIZE";
    setenv(env, "1000", 1);
    MOCKER_CPP(aclInit).expects(once()).will(invoke(StubAclInit));
    MOCKER_CPP(aclFinalize).expects(exactly(0)).will(returnValue(0));

    faiss::ascend::SocUtils utils;
    g_aclRet = 0;
}

// 设置有效环境变量0，通过repeat判断，是我们调用aclInit的
TEST_F(TestUtils, exec_finalize_set_valid_0_env_init)
{
    const char *env = "MX_INDEX_FINALIZE";
    setenv(env, "0", 1);
    MOCKER_CPP(aclInit).expects(once()).will(invoke(StubAclInit));
    MOCKER_CPP(aclFinalize).expects(exactly(0)).will(returnValue(0));

    faiss::ascend::SocUtils utils;
}

// 设置有效环境变量0，通过repeat判断，不是我们调用aclInit的
TEST_F(TestUtils, exec_finalize_set_valid_env_0_not_aclInit)
{
    g_aclRet = ACL_ERROR_REPEAT_INITIALIZE;
    const char *env = "MX_INDEX_FINALIZE";
    setenv(env, "0", 1);
    MOCKER_CPP(aclInit).expects(once()).will(invoke(StubAclInit));
    MOCKER_CPP(aclFinalize).expects(exactly(0)).will(returnValue(0));

    faiss::ascend::SocUtils utils;
    g_aclRet = 0;
}

// 设置有效环境变量1，通过repeat判断，不是我们调用aclInit的
TEST_F(TestUtils, exec_finalize_set_valid_env_1_not_aclInit)
{
    g_aclRet = ACL_ERROR_REPEAT_INITIALIZE;
    const char *env = "MX_INDEX_FINALIZE";
    setenv(env, "1", 1);
    MOCKER_CPP(aclInit).expects(once()).will(invoke(StubAclInit));
    MOCKER_CPP(aclFinalize).expects(exactly(1)).will(returnValue(0));

    faiss::ascend::SocUtils utils;
    g_aclRet = 0;
}

// 设置有效环境变量1，通过repeat判断，是我们调用aclInit的
TEST_F(TestUtils, exec_finalize_set_valid_env_1_aclInit)
{
    const char *env = "MX_INDEX_FINALIZE";
    setenv(env, "1", 1);
    MOCKER_CPP(aclInit).expects(once()).will(invoke(StubAclInit));
    MOCKER_CPP(aclFinalize).expects(exactly(1)).will(returnValue(0));

    faiss::ascend::SocUtils utils;
}

TEST_F(TestUtils, test_limits)
{
    auto ret = Limits<float16_t>::getMin();
    EXPECT_EQ(ret, 0xfbffU);

    ret = Limits<float16_t>::getMax();
    EXPECT_EQ(ret, 0x7bffU);
}

static std::vector<std::string> socNames {
    "Ascend310", "Ascend710", "Ascend310P",
    "Ascend910B1", "Ascend910B2", "Ascend910B3", "Ascend910B4",
    "Ascend910_9392", "Ascend910_9382", "Ascend910_9372", "Ascend910_9362",
    "Ascend910" };

static const char *SOC_NAME = "";
TEST_F(TestUtils, socutils_getsocname_nullptr)
{
    SOC_NAME = nullptr;

    MOCKER_CPP(aclInit).expects(exactly(1)).will(returnValue(0));
    MOCKER_CPP(aclFinalize).expects(exactly(1)).will(returnValue(0));
    MOCKER_CPP(&aclrtGetSocName).stubs().will(returnValue(SOC_NAME));
    faiss::ascend::SocUtils utils;

    EXPECT_EQ(utils.socAttr.socType, faiss::ascend::SocUtils::SocType::SOC_INVALID);
    EXPECT_EQ(utils.socAttr.coreNum, 0);
    EXPECT_EQ(utils.socAttr.codeFormType, faiss::ascend::SocUtils::CodeFormatType::FORMAT_TYPE_INVALID);
}

TEST_F(TestUtils, socutils_getsocname_310P)
{
    SOC_NAME = socNames[1].c_str(); // 1 for 310P

    MOCKER_CPP(aclInit).expects(exactly(1)).will(returnValue(0));
    MOCKER_CPP(aclFinalize).expects(exactly(1)).will(returnValue(0));
    MOCKER_CPP(&aclrtGetSocName).stubs().will(returnValue(SOC_NAME));
    faiss::ascend::SocUtils utils;

    EXPECT_EQ(utils.socAttr.socType, faiss::ascend::SocUtils::SocType::SOC_310P);
    EXPECT_EQ(utils.socAttr.coreNum, 8); // Ascend310P has 8 aicore
    EXPECT_EQ(utils.socAttr.codeFormType, faiss::ascend::SocUtils::CodeFormatType::FORMAT_TYPE_ZZ);
}

TEST_F(TestUtils, socutils_getsocname_310)
{
    SOC_NAME = socNames[0].c_str(); // 1 for 310

    MOCKER_CPP(aclInit).expects(exactly(1)).will(returnValue(0));
    MOCKER_CPP(aclFinalize).expects(exactly(1)).will(returnValue(0));
    MOCKER_CPP(&aclrtGetSocName).stubs().will(returnValue(SOC_NAME));
    faiss::ascend::SocUtils utils;

    EXPECT_EQ(utils.socAttr.socType, faiss::ascend::SocUtils::SocType::SOC_310);
    EXPECT_EQ(utils.socAttr.coreNum, 2); // Ascend310 has 2 aicore
    EXPECT_EQ(utils.socAttr.codeFormType, faiss::ascend::SocUtils::CodeFormatType::FORMAT_TYPE_ZZ);
}

TEST_F(TestUtils, socutils_getsocname_A2_4)
{
    SOC_NAME = socNames[6].c_str(); // 6 for A2 4

    MOCKER_CPP(aclInit).expects(exactly(1)).will(returnValue(0));
    MOCKER_CPP(aclFinalize).expects(exactly(1)).will(returnValue(0));
    MOCKER_CPP(&aclrtGetSocName).stubs().will(returnValue(SOC_NAME));
    faiss::ascend::SocUtils utils;

    EXPECT_EQ(utils.socAttr.socType, faiss::ascend::SocUtils::SocType::SOC_910B4);
    EXPECT_EQ(utils.socAttr.coreNum, 40); // B4 has 40 aicore
    EXPECT_EQ(utils.socAttr.codeFormType, faiss::ascend::SocUtils::CodeFormatType::FORMAT_TYPE_ND);
}

TEST_F(TestUtils, socutils_getsocname_A2_3)
{
    SOC_NAME = socNames[5].c_str(); // 5 for A2 B3

    MOCKER_CPP(aclInit).expects(exactly(1)).will(returnValue(0));
    MOCKER_CPP(aclFinalize).expects(exactly(1)).will(returnValue(0));
    MOCKER_CPP(&aclrtGetSocName).stubs().will(returnValue(SOC_NAME));
    faiss::ascend::SocUtils utils;

    EXPECT_EQ(utils.socAttr.socType, faiss::ascend::SocUtils::SocType::SOC_910B3);
    EXPECT_EQ(utils.socAttr.coreNum, 40); // B3 has 40 aicore
    EXPECT_EQ(utils.socAttr.codeFormType, faiss::ascend::SocUtils::CodeFormatType::FORMAT_TYPE_ND);
}

TEST_F(TestUtils, socutils_getsocname_A2_2)
{
    SOC_NAME = socNames[4].c_str(); // 4 for A2 B2

    MOCKER_CPP(aclInit).expects(exactly(1)).will(returnValue(0));
    MOCKER_CPP(aclFinalize).expects(exactly(1)).will(returnValue(0));
    MOCKER_CPP(&aclrtGetSocName).stubs().will(returnValue(SOC_NAME));
    faiss::ascend::SocUtils utils;

    EXPECT_EQ(utils.socAttr.socType, faiss::ascend::SocUtils::SocType::SOC_910B2);
    EXPECT_EQ(utils.socAttr.coreNum, 48); // B2 has 48 aicore
    EXPECT_EQ(utils.socAttr.codeFormType, faiss::ascend::SocUtils::CodeFormatType::FORMAT_TYPE_ND);
}

TEST_F(TestUtils, socutils_getsocname_A2_1)
{
    SOC_NAME = socNames[3].c_str(); // 3 for A2 B1

    MOCKER_CPP(aclInit).expects(exactly(1)).will(returnValue(0));
    MOCKER_CPP(aclFinalize).expects(exactly(1)).will(returnValue(0));
    MOCKER_CPP(&aclrtGetSocName).stubs().will(returnValue(SOC_NAME));
    faiss::ascend::SocUtils utils;

    EXPECT_EQ(utils.socAttr.socType, faiss::ascend::SocUtils::SocType::SOC_910B1);
    EXPECT_EQ(utils.socAttr.coreNum, 48); // B1 has 48 aicore
    EXPECT_EQ(utils.socAttr.codeFormType, faiss::ascend::SocUtils::CodeFormatType::FORMAT_TYPE_ND);
}

TEST_F(TestUtils, socutils_getsocname_A3_9392)
{
    SOC_NAME = socNames[7].c_str(); // 7 for A3_9392

    MOCKER_CPP(aclInit).expects(exactly(1)).will(returnValue(0));
    MOCKER_CPP(aclFinalize).expects(exactly(1)).will(returnValue(0));
    MOCKER_CPP(&aclrtGetSocName).stubs().will(returnValue(SOC_NAME));
    faiss::ascend::SocUtils utils;

    EXPECT_EQ(utils.socAttr.socType, faiss::ascend::SocUtils::SocType::SOC_910_9392);
    EXPECT_EQ(utils.socAttr.coreNum, 48); // A3_9392 has 48 aicore
    EXPECT_EQ(utils.socAttr.codeFormType, faiss::ascend::SocUtils::CodeFormatType::FORMAT_TYPE_ND);
}

TEST_F(TestUtils, socutils_getsocname_A3_9382)
{
    SOC_NAME = socNames[8].c_str(); // 8 for A3_9382

    MOCKER_CPP(aclInit).expects(exactly(1)).will(returnValue(0));
    MOCKER_CPP(aclFinalize).expects(exactly(1)).will(returnValue(0));
    MOCKER_CPP(&aclrtGetSocName).stubs().will(returnValue(SOC_NAME));
    faiss::ascend::SocUtils utils;

    EXPECT_EQ(utils.socAttr.socType, faiss::ascend::SocUtils::SocType::SOC_910_9382);
    EXPECT_EQ(utils.socAttr.coreNum, 48); // A3_9382 has 48 aicore
    EXPECT_EQ(utils.socAttr.codeFormType, faiss::ascend::SocUtils::CodeFormatType::FORMAT_TYPE_ND);
}

TEST_F(TestUtils, socutils_getsocname_A3_9372)
{
    SOC_NAME = socNames[9].c_str(); // 9 for A3_9372

    MOCKER_CPP(aclInit).expects(exactly(1)).will(returnValue(0));
    MOCKER_CPP(aclFinalize).expects(exactly(1)).will(returnValue(0));
    MOCKER_CPP(&aclrtGetSocName).stubs().will(returnValue(SOC_NAME));
    faiss::ascend::SocUtils utils;

    EXPECT_EQ(utils.socAttr.socType, faiss::ascend::SocUtils::SocType::SOC_910_9372);
    EXPECT_EQ(utils.socAttr.coreNum, 40); // 9372 has 40 aicore
    EXPECT_EQ(utils.socAttr.codeFormType, faiss::ascend::SocUtils::CodeFormatType::FORMAT_TYPE_ND);
}

TEST_F(TestUtils, socutils_getsocname_A3_9362)
{
    SOC_NAME = socNames[10].c_str(); // 10 for A3_9362

    MOCKER_CPP(aclInit).expects(exactly(1)).will(returnValue(0));
    MOCKER_CPP(aclFinalize).expects(exactly(1)).will(returnValue(0));
    MOCKER_CPP(&aclrtGetSocName).stubs().will(returnValue(SOC_NAME));
    faiss::ascend::SocUtils utils;

    EXPECT_EQ(utils.socAttr.socType, faiss::ascend::SocUtils::SocType::SOC_910_9362);
    EXPECT_EQ(utils.socAttr.coreNum, 40); // 9362 has 40 aicore
    EXPECT_EQ(utils.socAttr.codeFormType, faiss::ascend::SocUtils::CodeFormatType::FORMAT_TYPE_ND);
}
} // namespace ascend