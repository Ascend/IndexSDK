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

#include "AscendSimuLog.h"


AscendSimuLog::~AscendSimuLog()
{
    std::lock_guard<std::mutex> lk(m_mutFile);
    if (m_fObj.is_open()) {
        m_fObj.close();
    }
}

AscendSimuLog &AscendSimuLog::getIns()
{
    static AscendSimuLog logger;
    return logger;
}

void AscendSimuLog::SetLogLevel(aclLogLevel level)
{
    m_level = level;
}

void AscendSimuLog::Reset()
{
    std::lock_guard<std::mutex> lk(m_mutFile);
    if (m_fObj.is_open()) {
        m_fObj.close();
    }

    m_level = ACL_DEBUG;
}

bool AscendSimuLog::SetLogFile(const char *logFile)
{
    std::lock_guard<std::mutex> lk(m_mutFile);
    if (m_fObj.is_open()) {
        m_fObj.close();
    }

    m_fObj.open(logFile, std::ios::out);
    if (!m_fObj.is_open()) {
        return false;
    }

    m_fileName = logFile;
    return true;
}

const char *AscendSimuLog::GetLogLevel()
{
    static const char *Level[] = {
        "ACL_DEBUG",
        "ACL_INFO",
        "ACL_WARNING",
        "ACL_ERROR"
    };

    return Level[m_level];
}

const char *AscendSimuLog::GetLogFile()
{
    return m_fileName.c_str();
}

void AscendSimuLog::write(aclLogLevel level, const char *buffer, size_t size)
{
    std::lock_guard<std::mutex> lk(m_mutFile);
    if (level < m_level) {
        return;
    }

    if (!m_fObj.is_open()) {
        printf("%s\n", buffer);
        return;
    }

    m_fObj.write(buffer, size);
    m_fObj.write("\n", 1);
}
