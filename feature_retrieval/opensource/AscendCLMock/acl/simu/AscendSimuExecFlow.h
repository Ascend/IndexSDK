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


#ifndef LIBASCENDCL_ASCENDSIMUEXECFLOW_H
#define LIBASCENDCL_ASCENDSIMUEXECFLOW_H

#include <queue>
#include <thread>
#include <mutex>
#include <atomic>
#include <condition_variable>
#include <map>
#include <functional>
#include "../acl.h"

// 模拟执行流类对应一个stream 生产者消费者模型
class AscendSimuExecFlow {
public:
    struct callbackHandle {
        int execType = 0;
        void *opHandle;
    };
    friend class AscendSimuDevice; // 只有device作为仲裁者可以访问execflow
    using SimuExecOpFunc = std::function<void(aclopHandle &)>;

    enum { COND_WAIT_TIME_OUT = 100 };

    AscendSimuExecFlow() = default;
    virtual ~AscendSimuExecFlow();

    static void reg(const char *opName, SimuExecOpFunc func);
    static void unReg(const char *opName);
    static bool IsOpReg(const char *opName);
    static void ShowOp();
    void waitForEmpty();

private:
    bool start();
    bool stop();
    bool isRuning();
    bool put(aclopHandle *opHandle);
    bool put(aclrtCallback fn, void *userData);
    void run();
    static void exec(callbackHandle *opHandle);
    static bool TryIsOpReg(const char *opName);
    static SimuExecOpFunc TryGetOpFunc(const char *opName);

    std::thread *m_execFlow{ nullptr };
    std::queue<callbackHandle *> m_queue;
    mutable std::mutex m_mut;
    std::condition_variable m_cond;
    std::condition_variable m_emptyCond;
    std::atomic<bool> m_running{ false };

    static std::mutex m_OpMut;
    static std::map<const char *, SimuExecOpFunc> m_opExecMap;
};

#ifndef REG_OP
    #define REG_OP(opName, Opfunctor) AscendSimuExecFlow::reg(opName, Opfunctor)
#endif

#ifndef UNREG_OP
    #define UNREG_OP(opName) AscendSimuExecFlow::unReg(opName)
#endif

#ifndef IS_OP_REG
    #define IS_OP_REG(opName) AscendSimuExecFlow::IsOpReg(opName)
#endif

#endif // LIBASCENDCL_ASCENDSIMUEXECFLOW_H
