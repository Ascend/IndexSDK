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

#ifndef LIBASCENDCL_ASCENDSIMULOG_H
#define LIBASCENDCL_ASCENDSIMULOG_H

#include <iostream>
#include <atomic>
#include <mutex>
#include <fstream>
#include "../acl_base.h"

class AscendSimuLog {
public:
    static AscendSimuLog &getIns();
    void SetLogLevel(aclLogLevel level);
    bool SetLogFile(const char *logFile);
    const char *GetLogLevel();
    const char *GetLogFile();
    void Reset();
    void write(aclLogLevel level, const char *buffer, size_t size);

private:
    AscendSimuLog() = default;
    virtual ~AscendSimuLog();

    aclLogLevel m_level{ ACL_DEBUG };
    std::fstream m_fObj;  // 文件对象
    std::mutex m_mutFile; // 日志文件锁
    std::string m_fileName;
};

#define LOGGER() AscendSimuLog::getIns()

#endif // LIBASCENDCL_ASCENDSIMULOG_H
