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

#define API_PUBLIC __attribute__((visibility("default"))) // 仅将OpenGauss需要的接口的可见性设置为可见

#ifdef __cplusplus
extern "C" { // 防止编译器对cpp代码做命名修饰导致符号解析失败
#endif

typedef struct DiskPQParams {
    int pqChunks = 512; // 默认PQ子空间数为512
    int funcType = 1; // 目前PQ距离表计算仅支持L2距离，之后可能需要新增宏去区分L2/IP/consine距离
    int dim = 4096;
    char *pqTable = nullptr;
    uint32_t *offsets = nullptr;
    char *tablesTransposed = nullptr;
    char *centroids = nullptr;
} DiskPQParams;

// 为了与OpenGauss侧定义完全一致，初始值缺失，之后可以判断是否可以赋予初始值
typedef struct VectorArrayData {
    int length;
    int maxlen;
    int dim;
    size_t itemsize;
    char *items;
} VectorArrayData;

API_PUBLIC int ComputePQTable(VectorArrayData *sample, DiskPQParams *params);
API_PUBLIC int ComputeVectorPQCode(VectorArrayData *baseData, const DiskPQParams *params, uint8_t *pqCode);
API_PUBLIC int GetPQDistanceTable(char *vec, const DiskPQParams *params, float *pqDistanceTable);
API_PUBLIC int GetPQDistance(const uint8_t *basecode, const DiskPQParams *params,
                             const float *pqDistanceTable, float &pqDistance);

#ifdef __cplusplus
}
#endif