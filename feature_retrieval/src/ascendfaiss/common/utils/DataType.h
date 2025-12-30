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


#ifndef ASCEND_DATA_TYPE_H
#define ASCEND_DATA_TYPE_H

#include <unordered_map>

namespace faiss {
namespace ascend {
enum Type {
    UNDEFINED = -1,
    FLOAT,
    FLOAT16,
    INT8,
    INT32,
    UINT8,
    UINT16,
    UINT32,
    INT64,
    UINT64,
    DOUBLE,
    BOOL,
    INT16,
    STRING
};

static std::unordered_map<int, int> TypeSize {
    {Type::FLOAT, 4},
    {Type::FLOAT16, 2},
    {Type::INT8, 1},
    {Type::INT32, 4},
    {Type::UINT8, 1},
    {Type::UINT16, 2},
    {Type::UINT32, 4},
    {Type::INT64, 8},
    {Type::UINT64, 8},
    {Type::DOUBLE, 8},
    {Type::BOOL, 1},
    {Type::INT16, 2},
    };

inline int getTypeSize(int type)
{
    return TypeSize[type];
}
} // namespace ascend
} // namespace faiss

#endif // ASCEND_DATA_TYPE_H
