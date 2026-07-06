/*
 * -------------------------------------------------------------------------
 * This file is part of the IndexSDK project.
 * Copyright (c) 2026 Huawei Technologies Co.,Ltd.
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

#ifndef ASCEND_COARSE_CENTER_VERIFY_H
#define ASCEND_COARSE_CENTER_VERIFY_H

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "acl/acl.h"
#include "common/utils/LogUtils.h"

namespace ascend
{
namespace coarseCenterVerify
{
inline bool IsVerifyEnabled()
{
    const char *env = std::getenv("IVFRABITQ_VERIFY_COARSE_CENTER");
    return env != nullptr && env[0] != '\0' && env[0] != '0';
}

#define VERIFY_LOG_INFO(fmt, ...)                \
    do                                           \
    {                                            \
        APP_LOG_INFO(fmt, ##__VA_ARGS__);        \
        if (IsVerifyEnabled())                   \
        {                                        \
            fprintf(stderr, fmt, ##__VA_ARGS__); \
        }                                        \
    } while (0)

#define VERIFY_LOG_ERROR(fmt, ...)               \
    do                                           \
    {                                            \
        APP_LOG_ERROR(fmt, ##__VA_ARGS__);       \
        if (IsVerifyEnabled())                   \
        {                                        \
            fprintf(stderr, fmt, ##__VA_ARGS__); \
        }                                        \
    } while (0)

inline void LogMemcpyAudit(const char *stage, int numLists, int dims, int bytesPerVector, size_t hostBytes,
                           size_t deviceCapacityBytes, size_t copyBytes)
{
    if (!IsVerifyEnabled())
    {
        return;
    }
    VERIFY_LOG_INFO(
        "[CoarseCenterVerify] %s: numLists=%d dims=%d bytesPerVector=%d hostBytes=%zu deviceCapacityBytes=%zu "
        "copyBytes=%zu (copyBytes/4=%zu)\n",
        stage, numLists, dims, bytesPerVector, hostBytes, deviceCapacityBytes, copyBytes, copyBytes / 4);
}

inline float RowAbsSum(const std::vector<float> &data, int row, int dims)
{
    float sum = 0.0f;
    const size_t base = static_cast<size_t>(row) * static_cast<size_t>(dims);
    for (int j = 0; j < dims; ++j)
    {
        sum += std::fabs(data[base + static_cast<size_t>(j)]);
    }
    return sum;
}

inline bool CompareRowWithHost(const std::vector<float> &host, const std::vector<float> &device, int row, int dims,
                               float tol = 1e-5f)
{
    const size_t base = static_cast<size_t>(row) * static_cast<size_t>(dims);
    for (int j = 0; j < dims; ++j)
    {
        const float h = host[base + static_cast<size_t>(j)];
        const float d = device[base + static_cast<size_t>(j)];
        if (std::fabs(h - d) > tol * std::max(1.0f, std::fabs(h)))
        {
            return false;
        }
    }
    return true;
}

inline void LogHostSampleRows(const char *tag, const std::vector<float> &hostGolden, int numLists, int dims)
{
    if (!IsVerifyEnabled() || numLists <= 0 || dims <= 0)
    {
        return;
    }

    const int sampleIdx[] = {0, 2511, 2512, 2513, 7535, 7536, 9999, numLists - 1};
    for (int row : sampleIdx)
    {
        if (row < 0 || row >= numLists)
        {
            continue;
        }
        const float hostNorm = RowAbsSum(hostGolden, row, dims);
        VERIFY_LOG_INFO("[CoarseCenterVerify] %s host golden: row %d absSum=%.6f\n", tag, row, hostNorm);
    }
}

inline void VerifySamplesD2H(const char *tag, const std::vector<float> &hostGolden, const float *devicePtr,
                             int numLists, int dims, bool compareToHost = true)
{
    if (!IsVerifyEnabled() || devicePtr == nullptr || numLists <= 0 || dims <= 0)
    {
        return;
    }

    const size_t totalFloats = static_cast<size_t>(numLists) * static_cast<size_t>(dims);
    std::vector<float> deviceHost(totalFloats, 0.0f);
    const size_t bytes = totalFloats * sizeof(float);
    auto ret = aclrtMemcpy(deviceHost.data(), bytes, devicePtr, bytes, ACL_MEMCPY_DEVICE_TO_HOST);
    if (ret != ACL_SUCCESS)
    {
        VERIFY_LOG_ERROR("[CoarseCenterVerify] %s: D2H failed ret=%d\n", tag, static_cast<int>(ret));
        return;
    }

    const int sampleIdx[] = {0, 2511, 2512, 2513, 7535, 7536, 9999, numLists - 1};
    for (int row : sampleIdx)
    {
        if (row < 0 || row >= numLists)
        {
            continue;
        }
        const float hostNorm = RowAbsSum(hostGolden, row, dims);
        const float devNorm = RowAbsSum(deviceHost, row, dims);
        if (compareToHost)
        {
            if (!CompareRowWithHost(hostGolden, deviceHost, row, dims))
            {
                VERIFY_LOG_ERROR("[CoarseCenterVerify] %s: row %d mismatch vs host (hostNorm=%.6f devNorm=%.6f)\n", tag,
                                 row, hostNorm, devNorm);
            }
            else
            {
                VERIFY_LOG_INFO("[CoarseCenterVerify] %s: row %d OK vs host (devNorm=%.6f)\n", tag, row, devNorm);
            }
        }
        else if (devNorm < 1e-12f)
        {
            VERIFY_LOG_ERROR("[CoarseCenterVerify] %s: row %d is all-zero on device (devNorm=%.6f)\n", tag, row,
                             devNorm);
        }
        else
        {
            VERIFY_LOG_INFO("[CoarseCenterVerify] %s: row %d OK (devNorm=%.6f)\n", tag, row, devNorm);
        }
    }
}

inline void VerifyFullD2H(const char *tag, const std::vector<float> &hostGolden, const float *devicePtr, int numLists,
                          int dims, bool compareToHost = true)
{
    if (!IsVerifyEnabled() || devicePtr == nullptr || numLists <= 0 || dims <= 0)
    {
        return;
    }

    const size_t totalFloats = static_cast<size_t>(numLists) * static_cast<size_t>(dims);
    std::vector<float> deviceHost(totalFloats, 0.0f);
    const size_t bytes = totalFloats * sizeof(float);
    auto ret = aclrtMemcpy(deviceHost.data(), bytes, devicePtr, bytes, ACL_MEMCPY_DEVICE_TO_HOST);
    if (ret != ACL_SUCCESS)
    {
        VERIFY_LOG_ERROR("[CoarseCenterVerify] %s full: D2H failed ret=%d\n", tag, static_cast<int>(ret));
        return;
    }

    int mismatchRows = 0;
    int zeroRowsBefore2512 = 0;
    int zeroRowsAfter2512 = 0;
    for (int row = 0; row < numLists; ++row)
    {
        const float devNorm = RowAbsSum(deviceHost, row, dims);
        if (devNorm < 1e-12f)
        {
            if (row < 2512)
            {
                ++zeroRowsBefore2512;
            }
            else
            {
                ++zeroRowsAfter2512;
            }
        }
        if (compareToHost && !CompareRowWithHost(hostGolden, deviceHost, row, dims))
        {
            ++mismatchRows;
        }
    }
    VERIFY_LOG_INFO("[CoarseCenterVerify] %s full: mismatchRows=%d zeroRowsBefore2512=%d zeroRowsAfter2512=%d / %d\n",
                    tag, mismatchRows, zeroRowsBefore2512, zeroRowsAfter2512, numLists - 2512);
}

inline void VerifyCentroidL2SamplesD2H(const char *tag, const float *devicePtr, int numLists)
{
    if (!IsVerifyEnabled() || devicePtr == nullptr || numLists <= 0)
    {
        return;
    }

    std::vector<float> l2Host(static_cast<size_t>(numLists), 0.0f);
    const size_t bytes = static_cast<size_t>(numLists) * sizeof(float);
    auto ret = aclrtMemcpy(l2Host.data(), bytes, devicePtr, bytes, ACL_MEMCPY_DEVICE_TO_HOST);
    if (ret != ACL_SUCCESS)
    {
        VERIFY_LOG_ERROR("[CoarseCenterVerify] %s: CentroidL2 D2H failed ret=%d\n", tag, static_cast<int>(ret));
        return;
    }

    const int sampleIdx[] = {0, 2511, 2512, 2513, 7535, 7536, 9999, numLists - 1};
    for (int row : sampleIdx)
    {
        if (row < 0 || row >= numLists)
        {
            continue;
        }
        const float l2 = l2Host[static_cast<size_t>(row)];
        if (l2 < 1e-12f)
        {
            VERIFY_LOG_ERROR("[CoarseCenterVerify] %s: CentroidL2 row %d is zero (l2=%.6f)\n", tag, row, l2);
        }
        else
        {
            VERIFY_LOG_INFO("[CoarseCenterVerify] %s: CentroidL2 row %d l2=%.6f\n", tag, row, l2);
        }
    }
}

inline void LogStageDone(const char *stage)
{
    if (!IsVerifyEnabled())
    {
        return;
    }
    VERIFY_LOG_INFO("[CoarseCenterVerify] %s done\n", stage);
}

}  // namespace coarseCenterVerify
}  // namespace ascend

#endif  // ASCEND_COARSE_CENTER_VERIFY_H
