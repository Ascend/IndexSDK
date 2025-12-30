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


#ifndef ASCEND_COMMON_UTILS_H
#define ASCEND_COMMON_UTILS_H

#include <algorithm>
#include <cstring>
#include <limits.h>
#include <pwd.h>
#include <unistd.h>
#include <string>
#include <sys/types.h>
#include <sys/stat.h>
#include <thread>
#include <vector>
#include "ascenddaemon/utils/AscendTensor.h"
#include "ascenddaemon/utils/DeviceVector.h"
#include "ErrorCode.h"

#define CREATE_UNIQUE_PTR(type, args...) \
    std::unique_ptr<type>(new type(args))

namespace {
constexpr int32_t MAX_THREAD_NUM = 256;
std::once_flag g_getOmpNumThreadsFlag;
}
namespace ascend {
struct RemoveForwardParam {
    size_t srcBlockIdx;
    size_t srcBlockOffset;

    size_t dstBlockIdx;
    size_t dstBlockOffset;
};

template<typename T>
APP_ERROR AddCodeNDFormat(const AscendTensor<T, DIMS_2> &rawData,
                          size_t ntotal,
                          size_t blockSize,
                          std::vector<std::unique_ptr<DeviceVector<T>>> &baseShaped)
{
    size_t n = static_cast<size_t>(rawData.getSize(0));
    size_t dims = static_cast<size_t>(rawData.getSize(1));
    for (size_t i = 0; i < n;) {
        size_t total = ntotal + i;
        size_t offsetInBlock = total % blockSize;
        size_t leftInBlock = blockSize - offsetInBlock;
        size_t leftInData = static_cast<size_t>(n) - i;
        size_t copyCount = std::min(leftInBlock, leftInData);
        size_t blockIdx = total / blockSize;

        T *src = rawData[i].data();

        APPERR_RETURN_IF_NOT_FMT((baseShaped.size() > blockIdx), APP_ERR_INNER_ERROR,
            "baseShaped size(%zu) need greater than blockIdx(%zu)", baseShaped.size(), blockIdx);
        T *dst = baseShaped[blockIdx]->data() + offsetInBlock * dims;

        auto ret = aclrtMemcpy(dst, leftInBlock * dims * sizeof(T),
                               src, copyCount * dims * sizeof(T),
                               ACL_MEMCPY_HOST_TO_DEVICE);
        APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR,
                                 "Mem operator h2d error %d", ret);

        i += copyCount;
    }
    return APP_ERR_OK;
}

template<typename T>
APP_ERROR GetVectorsNDFormat(uint32_t offset,
                             uint32_t num,
                             uint32_t blockSize,
                             const std::vector<std::unique_ptr<DeviceVector<T>>> &baseShaped,
                             std::vector<T> &vectors)
{
    if (num == 0) {
        return APP_ERR_OK;
    }
    uint32_t dims = static_cast<uint32_t>(vectors.size()) / num;
    for (uint32_t i = 0; i < num;) {
        uint32_t total = offset + i;
        uint32_t offsetInBlock = total % blockSize;
        uint32_t leftInBlock = blockSize - offsetInBlock;
        uint32_t leftInData = num - i;
        uint32_t copyCount = std::min(leftInBlock, leftInData);
        uint32_t blockIdx = total / blockSize;

        T *dst = vectors.data() + i * dims;
        APPERR_RETURN_IF_NOT_FMT((baseShaped.size() > blockIdx), APP_ERR_INNER_ERROR,
            "baseShaped size(%zu) need greater than blockIdx(%zu)", baseShaped.size(), blockIdx);
        T *src = baseShaped[blockIdx]->data() + offsetInBlock * dims;

        auto ret = aclrtMemcpy(dst, leftInData * dims * sizeof(T),
                               src, copyCount * dims * sizeof(T),
                               ACL_MEMCPY_DEVICE_TO_HOST);
        APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR,
                                 "Mem operator d2h error %d", ret);

        i += copyCount;
    }
    return APP_ERR_OK;
}

// 底层接口不对参数进行合法性校验，调用者保证参数的合法性
template<typename T>
aclError RemoveForwardNDFormat(const RemoveForwardParam& param, uint32_t dim,
                               std::vector<std::unique_ptr<DeviceVector<T>>> &baseShaped)
{
    return aclrtMemcpy(baseShaped[param.dstBlockIdx]->data() + param.dstBlockOffset * static_cast<size_t>(dim),
        (baseShaped[param.dstBlockIdx]->size() - param.dstBlockOffset * static_cast<size_t>(dim)) * sizeof(T),
        baseShaped[param.srcBlockIdx]->data() + param.srcBlockOffset * static_cast<size_t>(dim),
        static_cast<size_t>(dim) * sizeof(T),
        ACL_MEMCPY_DEVICE_TO_DEVICE);
}

class CommonUtils {
public:
static std::string RealPath(const std::string &inPath)
{
    std::string result = inPath;

    // 1. replace ~/ with /home/user/
    if (inPath.size() >= 2 && inPath[0] == '~' && inPath[1] == '/') { // >=2 means inPath startswith '~/'
        struct passwd *pw = getpwuid(getuid());
        if (pw == nullptr || pw->pw_dir == nullptr) {
            return std::string();
        }
        std::string homedir(pw->pw_dir);
        (void)memset_s(pw->pw_passwd, strlen(pw->pw_passwd), 0, strlen(pw->pw_passwd));
        homedir.append(inPath.substr(1));
        result = std::move(homedir);
    }

    // 2. realpath
    if (result.size() > PATH_MAX) {
        return std::string();
    }
    char realPath[PATH_MAX] = {0};
    char *ptr = realpath(result.c_str(), realPath);
    if (ptr == nullptr) {
        return std::string();
    }
    result = std::string(realPath);

    return result;
}

static APP_ERROR CheckSymLink(struct stat *fileStat, const std::string &filename)
{
    if (filename  == "") {
        return APP_ERR_OK ;
    }
    int ret = lstat(filename.c_str(), fileStat);
    if (ret == 0 && ((fileStat->st_mode & S_IFMT) == S_IFLNK)) {
        return APP_ERR_INVALID_PARAM;
    }
    auto idx = filename.find_last_of("/");
    if (idx == std::string::npos) {
        return APP_ERR_OK;
    }
    auto res = CheckSymLink(fileStat, filename.substr(0, idx));
    return  res;
}

static bool CheckPathValid(const std::string &path)
{
    if (path.find("/home/") != 0 && path.find("/root/") != 0) {
        return false;
    }
    if (access(path.c_str(), R_OK) != EOK) {
        return false;
    }
    return true;
}

static void attachToCpu(int cpuId)
{
    size_t cpu = static_cast<size_t>(cpuId) % std::thread::hardware_concurrency();
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(cpu, &cpuset);
    (void)pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
}

static void attachToCpus(std::initializer_list<uint8_t> cpus)
{
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);

    for (auto cpuId : cpus) {
        size_t cpu = static_cast<size_t>(cpuId) % std::thread::hardware_concurrency();
        CPU_SET(cpu, &cpuset);
    }

    (void)pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
}

static void AclInputBufferDelete(std::vector<const aclDataBuffer *> *distOpInput)
{
    for (auto &item : *distOpInput) {
        aclDestroyDataBuffer(item);
    }
    delete distOpInput;
}

static void AclOutputBufferDelete(std::vector<aclDataBuffer *> *distOpOutput)
{
    for (auto &item : *distOpOutput) {
        aclDestroyDataBuffer(item);
    }
    delete distOpOutput;
}

static bool IsNumber(const std::string& s)
{
    return std::all_of(s.begin(), s.end(), isdigit);
}

static void GetOmpNumThreads(int32_t &ompNumThreads)
{
    int32_t coreNum = static_cast<int32_t>(std::thread::hardware_concurrency());
    ompNumThreads = (coreNum > MAX_THREAD_NUM) ? MAX_THREAD_NUM : coreNum;
    char *ompNumThreadsData = std::getenv("OMP_NUM_THREADS");
    if (ompNumThreadsData == nullptr) {
        APP_LOG_INFO("not set OMP_NUM_THREADS, use default %d", ompNumThreads);
        return;
    }
    const size_t minOmpNumThreadsStrSize = 1;
    const size_t maxOmpNumThreadsStrSize = 3;
    std::string ompNumThreadsStr = std::string(ompNumThreadsData);
    if (ompNumThreadsStr.size() < minOmpNumThreadsStrSize || ompNumThreadsStr.size() > maxOmpNumThreadsStrSize) {
        APP_LOG_ERROR("set invalid OMP_NUM_THREADS, it's size is invalid, use default %d", ompNumThreads);
        return;
    }
    if (!IsNumber(ompNumThreadsStr)) {
        APP_LOG_ERROR("set invalid OMP_NUM_THREADS, not a numnber, use default %d", ompNumThreads);
        return;
    }
    try {
        int32_t ompNumSet = std::stoi(ompNumThreadsStr);
        const int32_t minOmpNumThreads = 1;
        const int32_t maxOmpNumThreads = MAX_THREAD_NUM;
        if (ompNumSet >= minOmpNumThreads && ompNumSet <= maxOmpNumThreads) {
            ompNumThreads = (ompNumSet > ompNumThreads) ? ompNumThreads : ompNumSet;
            APP_LOG_INFO("set OMP_NUM_THREADS: %d", ompNumThreads);
        } else {
            APP_LOG_ERROR("set invalid range OMP_NUM_THREADS, use default %d", ompNumThreads);
        }
    } catch (const std::exception &e) {
        APP_LOG_ERROR("transform OMP_NUM_THREADS failed, use default %d", ompNumThreads);
    }
}

static int32_t GetThreadMaxNums()
{
    static int32_t ompNumThreads = MAX_THREAD_NUM;
    std::call_once(g_getOmpNumThreadsFlag, [] { GetOmpNumThreads(ompNumThreads); });
    // 并发线程数最大为core的数量或者256
    return ompNumThreads;
}
};
}

#endif // ASCEND_COMMON_UTILS_H
