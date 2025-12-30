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


#ifndef AICPU_KERNEL_UTILS_H
#define AICPU_KERNEL_UTILS_H

#include <vector>
#include <sstream>
#include <string>

#include <arm_fp16.h>
#include <sys/time.h>

#ifdef AICPU_UTEST
#define KERNEL_LOG_ERROR(fmt, ...) \
    printf("[ERROR]" fmt "\n", ##__VA_ARGS__)
#define KERNEL_LOG_INFO(fmt, ...) \
    printf("[INFO]" fmt "\n", ##__VA_ARGS__)
#else
#define KERNEL_LOG_ERROR(fmt, ...)
#define KERNEL_LOG_INFO(fmt, ...)
#endif

#define WAITING_FLAG_READY(flag, checkTicks, timeout)                                                  \
    do {                                                                                               \
        int waitTicks_ = 0;                                                                            \
        double startWait_ = GetMillisecs();                                                            \
        while (!(flag)) {                                                                              \
            waitTicks_++;                                                                              \
            if (!(waitTicks_ % (checkTicks)) && ((GetMillisecs() - startWait_) >= (timeout))) {        \
                break;                                                                                 \
            }                                                                                          \
        }                                                                                              \
    } while (false)

#define KERNEL_CHECK_NULLPTR_VOID(value, logText...) \
    if (value == nullptr) {                          \
        KERNEL_LOG_ERROR(logText);                   \
        return;                                      \
    }

#define KERNEL_CHECK_NULLPTR(value, errorCode, logText...) \
    if (value == nullptr) {                                \
        KERNEL_LOG_ERROR(logText);                         \
        return errorCode;                                  \
    }

#define KERNEL_CHECK_TRUE(expr, errorCode, logText...) \
    do {                                               \
        if (!(expr)) {                                 \
            KERNEL_LOG_ERROR(logText);                 \
            return errorCode;                          \
        }                                              \
    } while (false)

namespace aicpu {
const double TIMEOUT_MS = 50000;
const int TIMEOUT_CHECK_TICK = 5120;

const size_t INPUT_NUM0 = 0;
const size_t INPUT_NUM1 = 1;
const size_t INPUT_NUM2 = 2;
const size_t INPUT_NUM3 = 3;
const size_t INPUT_NUM4 = 4;
const size_t INPUT_NUM5 = 5;
const size_t INPUT_NUM6 = 6;
const size_t INPUT_NUM7 = 7;
const size_t INPUT_NUM8 = 8;
const size_t INPUT_NUM9 = 9;

enum KernelStatus : uint32_t {
    KERNEL_STATUS_OK = 0,
    KERNEL_STATUS_PARAM_INVALID,
    KERNEL_STATUS_INNER_ERROR,
    KERNEL_STATUS_PROTOBUF_ERROR
};

inline double GetMillisecs()
{
    struct timeval tv;
    gettimeofday(&tv, nullptr);
    const double sec2msec = 1e3;
    const double usec2msec = 1e-3;
    return tv.tv_sec * sec2msec + tv.tv_usec * usec2msec;
}

std::string ShapeToString(const std::vector<int64_t> &v);

template <typename T, typename C>
void UpdateHeap(float16_t *dists, T *label, int64_t len, int64_t index, C &&cmp)
{
    int64_t l = 0;
    int64_t r = 0;
    int64_t m = 0;
    while (true) {
        l = 2 * index + 1; // 2 * index + 1 to find left subnode
        r = 2 * index + 2; // 2 * index + 2 to find right subnode
        m = index;
        if (l < len && cmp(dists[l], dists[m])) {
            m = l;
        }
        if (r < len && cmp(dists[r], dists[m])) {
            m = r;
        }
        if (m != index) {
            std::swap(dists[m], dists[index]);
            std::swap(label[m], label[index]);
            index = m;
        } else {
            break;
        }
    }
}

template <typename T, typename C>
void UpdateHeap(float *dists, T *label, int64_t len, int64_t index, C &&cmp)
{
    int64_t l = 0;
    int64_t r = 0;
    int64_t m = 0;
    while (true) {
        l = 2 * index + 1; // 2 * index + 1 to find left subnode
        r = 2 * index + 2; // 2 * index + 2 to find right subnode
        m = index;
        if (l < len && cmp(dists[l], dists[m])) {
            m = l;
        }
        if (r < len && cmp(dists[r], dists[m])) {
            m = r;
        }
        if (m != index) {
            std::swap(dists[m], dists[index]);
            std::swap(label[m], label[index]);
            index = m;
        } else {
            break;
        }
    }
}
} // namespace aicpu

#endif