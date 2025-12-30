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


#include <utils/AscendSearchException.h>
#include <sstream>
#include "securec.h"

#ifdef __GNUG__
#include <cxxabi.h>
#endif

namespace ascendsearch {

AscendSearchException::AscendSearchException(const std::string &m) : msg(m)
{
}

AscendSearchException::AscendSearchException(const std::string &m, const char *funcName, const char *file, int line)
{
    const int msgLen = 1024;
    const int scaleSize = 2;
    int size = msgLen;
    this->msg.resize(size, '\0');
    while (snprintf_s(&this->msg[0], this->msg.size(), this->msg.size(), "Error in %s at %s:%d: %s", funcName, file,
        line, m.c_str()) < 0) {
        size = size * scaleSize;
        this->msg.resize(size, '\0');
    }
}

const char *AscendSearchException::what() const noexcept
{
    return msg.c_str();
}

}  // namespace ascendsearch