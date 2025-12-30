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


#ifndef OCK_MEMORY_BRIDGE_READABLE_UTILS_H
#define OCK_MEMORY_BRIDGE_READABLE_UTILS_H
#include <iostream>
#include <iomanip>
#include "ock/utils/OckSafeUtils.h"

namespace ock {
namespace utils {

inline std::ostream &ByteToReadable(std::ostream &os, uint64_t byteCount)
{
    const double BYTE_SYSTEM = 1024.0;
    double byteNum = static_cast<double>(byteCount);
    const int KEEP_DIGITS = 2; // 输出结果保留两位小数

    if (byteNum <= BYTE_SYSTEM) {
        os << byteNum << "B";
    } else if (byteNum <= (BYTE_SYSTEM * BYTE_SYSTEM)) {
        os << std::setiosflags(std::ios::fixed) << std::setprecision(KEEP_DIGITS)
           << byteNum / BYTE_SYSTEM << "KB";
    } else if (byteNum <= (BYTE_SYSTEM * BYTE_SYSTEM * BYTE_SYSTEM)) {
        os << std::setiosflags(std::ios::fixed) << std::setprecision(KEEP_DIGITS)
           << byteNum / (BYTE_SYSTEM * BYTE_SYSTEM) << "MB";
    } else {
        os << std::setiosflags(std::ios::fixed) << std::setprecision(KEEP_DIGITS)
           << byteNum / (BYTE_SYSTEM * BYTE_SYSTEM * BYTE_SYSTEM) << "GB";
    }
    return os;
}

}
}
#endif