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

#ifndef ASCEND_BM25_ERROR_H
#define ASCEND_BM25_ERROR_H
#include <cstddef>
#include <cstdint>
#include <ios>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

#include "ascendhost/include/index/AscendIndexBm25.h"
#include "common/ErrorCode.h"

namespace ascend::bm25
{
class InvalidState : public std::runtime_error
{
   public:
    explicit InvalidState(const std::string& message) : std::runtime_error(message) {}
};
class InvalidArgument : public std::runtime_error
{
   public:
    explicit InvalidArgument(const std::string& message) : std::runtime_error(message) {}
};

class DeviceError : public std::runtime_error
{
   public:
    DeviceError(int64_t aclStatusCode, const std::string& message)
        : std::runtime_error(message), aclCode_(aclStatusCode)
    {
    }
    int64_t GetCode() const noexcept { return aclCode_; }

   private:
    int64_t aclCode_;
};

inline void CheckArgument(bool valid, const std::string& message)
{
    if (!valid)
    {
        throw InvalidArgument(message);
    }
}

inline void Check(bool condition, const std::string& message)
{
    if (!condition) throw std::runtime_error(message);
}
inline size_t CheckedByteSize(uint64_t elementCount, size_t elementSizeBytes)
{
    Check(elementSizeBytes > 0 && elementCount <= std::numeric_limits<size_t>::max() / elementSizeBytes &&
              elementCount <= static_cast<uint64_t>(std::numeric_limits<std::streamsize>::max()) / elementSizeBytes,
          "Array byte count exceeds host limits");
    return static_cast<size_t>(elementCount) * elementSizeBytes;
}

// SDK integration boundary. Uses IndexSDK's real error constants; no local numeric copies.
// DeviceError retains the raw ACL code for diagnostics inside the supplied operation.
template <class Operation>
APP_ERROR InvokeSdkOperation(Operation&& operation, ErrorInfo* error = nullptr) noexcept
{
    if (error)
    {
        error->Set(nullptr);
    }
    try
    {
        std::forward<Operation>(operation)();
        return APP_ERR_OK;
    }
    catch (const InvalidArgument& failure)
    {
        if (error)
        {
            error->Set(failure.what());
        }
        return APP_ERR_INVALID_PARAM;
    }
    catch (const InvalidState& failure)
    {
        if (error)
        {
            error->Set(failure.what());
        }
        return APP_ERR_ILLEGAL_OPERATION;
    }
    catch (const DeviceError& failure)
    {
        if (error)
        {
            error->Set(failure.what(), failure.GetCode());
        }
        return APP_ERR_INNER_ERROR;
    }
    catch (const std::exception& failure)
    {
        if (error)
        {
            error->Set(failure.what());
        }
        return APP_ERR_INNER_ERROR;
    }
    catch (...)
    {
        if (error)
        {
            error->Set("Unknown BM25 failure");
        }
        return APP_ERR_INNER_ERROR;
    }
}
}  // namespace ascend::bm25

#endif  // ASCEND_BM25_ERROR_H
