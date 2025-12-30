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


#ifndef ASCENDSEARCH_ASCENDSEARCHEXCEPTION_H
#define ASCENDSEARCH_ASCENDSEARCHEXCEPTION_H

#include <exception>
#include <string>
#include <utility>
#include <vector>

namespace ascendsearch {
class AscendSearchException : public std::exception {
public:
    explicit AscendSearchException(const std::string &msg);

    AscendSearchException(const std::string &msg, const char *funcName, const char *file, int line);

    // / from std::exception
    const char *what() const noexcept override;

    std::string msg;
};

}  // namespace ascendsearch

#endif  // ASCENDSEARCH_ASCENDSEARCHEXCEPTION_H
