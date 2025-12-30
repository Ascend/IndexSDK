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


#pragma once

#include <exception>
#include <string>
#include <sstream>

namespace diskann_pro {
class DiskException : public std::exception {
public:
    explicit DiskException(const std::string& msg): msg(msg) {}

    DiskException(const std::string& msg,
                    const char* funcName,
                    const char* file,
                    int line)
    {
        std::ostringstream oss;
        oss << "Error in " << funcName << " at " << file << ":" << line << ": " << msg;
        this->msg = oss.str();
    }

    ~DiskException() override = default;

    // from std::exception
    const char* what() const noexcept override
    {
        return msg.c_str();
    }

    std::string msg;
};
}  // namespace diskann_pro
