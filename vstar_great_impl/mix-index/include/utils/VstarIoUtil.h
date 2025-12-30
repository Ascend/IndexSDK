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


#ifndef IO_UTIL_INCLUDED
#define IO_UTIL_INCLUDED
#include <sys/types.h>
#include <sys/stat.h>
#include <errno.h>
#include <string>
#include <fstream>
#include <fcntl.h>
#include <unistd.h>
#include <limits.h>
#include "npu/common/utils/AscendAssert.h"

namespace ascendSearchacc {

constexpr size_t CSTYLE_IO_LIMIT = 2147418112; // 2147418112 bytes (0x7fff0000)

inline bool isValidCode(char c)
{
    return ('a' <= c && c <= 'z') || ('A' <= c && c <= 'Z') || ('0' <= c && c <= '9');
}

inline bool isInWhiteList(char c)
{
    std::string defaultPathWhiteList = "-_./~";
    return defaultPathWhiteList.find(c) != std::string::npos;
}

inline void checkSoftLink(const std::string &path)
{
    struct stat statbuf;
    if (lstat(path.c_str(), &statbuf) == 0) {
        ASCEND_THROW_IF_NOT_MSG(!S_ISLNK(statbuf.st_mode), "Accessing a softlink.\n");
    }
}

/**
 * @brief 关闭文件柄；在构造函数中，在文件已经打开的某些场合，如果存在问题，需要手动关闭文件柄（因为息构函数不会触发）
 *
 * @param fd 文件描述（文件柄）
 * @param name 文件名
 */
inline void closeFileDescription(int &fd)
{
    if (fd != -1) { // 仅在文件已经打开的情况下尝试关闭，否则行为可能undefined行为
        int ret = close(fd);
        fd = -1; // 不论结果，设置fd为-1, 后续不继续关闭
        if (ret != 0) {
            // we cannot raise and exception in the destructor
            fprintf(stderr, "file close error: %s", strerror(errno));
        }
    }
}

/**
 * @brief Check whether the filestream has enough data to be read into the input pointer
 *
 * @param readInputPtr
 * @param inputSize
 * @param fin
 */
inline void CheckInputSize(size_t inputSize, std::ifstream &fin)
{
    // Save the current position
    std::streampos currentPos = fin.tellg();

    // Seek to the end of the file to find out the total length
    fin.seekg(0, std::ios::end);
    std::streampos endPos = fin.tellg();

    // Calculate the number of bytes remaining from the current position
    std::streamsize bytesLeft = endPos - currentPos;

    // Restore the original position
    fin.seekg(currentPos, std::ios::beg);

    // Check if there are at least inputSize bytes left
    ASCEND_THROW_IF_NOT_MSG((size_t)bytesLeft >= inputSize,
        "Error when reading index: not enough data left in the index file to be loaded.\n");
}

/**
 * @brief 读取过程中，用于申请内存的值，进行一个范围校验
 *
 * @param value
 */
inline void loadedValueSanityCheck(int value, int upperBound)
{
    ASCEND_THROW_IF_NOT_FMT((value >= 0) && (value <= upperBound),
        "invalid loaded value, should be between 0 and %d inclusively", upperBound);
}

bool checkLinkRec(const std::string &realPathFunc);

/**
 * @brief 对输入路径的父目录做1）存在校验；2）软连接校验；在路径不存在的场合，移除通过句柄创造的路径
 *
 * @param realPath 全局路径
 * @param fd 文件句柄
 * @param removeDir 布尔值，决定是否对打开的路径进行移除，仅用于文件路径不存在的场合
 */
void parentDirCheckAndRemove(const std::string &realPath, int &fd, bool removeDir);

/**
 * @brief 对输入的文件路径进行校验，并对一个未初始化的文件柄进行赋值
 *
 * @param fd 待初始化的文件柄
 * @param fname 文件名
 */
void initializeFileDescription(int &fd, const std::string &fname);

/**
 * @brief 分片写入，避免一次写入字节数过大导致错误
 *
 * @tparam T
 * @param fd 文件句柄
 * @param ptr 数据来源指针（类型T）
 * @param size 数据类型T的字节数大小
 * @param expectedWrite 期待写的字节数
 */
template <typename T>
size_t WriteBatchImpl(const int &fd, const T *ptr, size_t typeSize, size_t expectedWrite)
{
    ASCEND_THROW_IF_NOT_FMT(typeSize > 0, "size[%lu] should be > 0", typeSize);
    ASCEND_THROW_IF_NOT_FMT(CSTYLE_IO_LIMIT % typeSize == 0,
                            "CSTYLE_IO_LIMIT[%lu] should be a multiple of size[%lu].",
                            CSTYLE_IO_LIMIT, typeSize);

    size_t writeCounts = expectedWrite / CSTYLE_IO_LIMIT;   // 写次数
    const size_t writeItemsMax = CSTYLE_IO_LIMIT / typeSize; // 每次写的元素数
    size_t totalBytesWritten = 0; // 总字节数

    // 分批写
    for (size_t i = 0; i < writeCounts; ++i) {
        ssize_t bytesWritePerCount = write(fd, ptr + i * writeItemsMax, CSTYLE_IO_LIMIT);
        if (bytesWritePerCount == -1) {
            // 出错，返回已写入的完整元素数
            return totalBytesWritten / typeSize;
        }
        totalBytesWritten += static_cast<size_t>(bytesWritePerCount);
    }

    // 写剩余不足一个批次的数据
    size_t remainBytes = expectedWrite - writeCounts * CSTYLE_IO_LIMIT;
    ssize_t remainBytesWrite = write(fd, ptr + writeCounts * writeItemsMax, remainBytes);
    if (remainBytesWrite == -1) {
        return totalBytesWritten / typeSize;
    }
    totalBytesWritten += static_cast<size_t>(remainBytesWrite);

    return totalBytesWritten / typeSize;
}


template <typename T>
size_t ReadBatchImpl(const int &fd, T *ptr, size_t typeSize, size_t expectedRead)
{
    ASCEND_THROW_IF_NOT_FMT(typeSize > 0, "size[%lu] should be > 0", typeSize);
    ASCEND_THROW_IF_NOT_FMT(CSTYLE_IO_LIMIT % typeSize == 0,
                            "CSTYLE_IO_LIMIT[%lu] should be a multiple of size[%lu].",
                            CSTYLE_IO_LIMIT, typeSize);

    size_t readCounts = expectedRead / CSTYLE_IO_LIMIT;   // 读取次数
    const size_t readItemsMax = CSTYLE_IO_LIMIT / typeSize; // 每次读取的元素数
    size_t totalBytesRead = 0; // 总字节数

    for (size_t i = 0; i < readCounts; ++i) {
        ssize_t bytesReadPerCount = read(fd, ptr + i * readItemsMax, CSTYLE_IO_LIMIT);
        if (bytesReadPerCount == -1) {
            // 出错，返回已读取的完整项数
            return totalBytesRead / typeSize;
        }
        totalBytesRead += static_cast<size_t>(bytesReadPerCount);
    }

    size_t remainBytes = expectedRead - readCounts * CSTYLE_IO_LIMIT;
    ssize_t remainBytesRead = read(fd, ptr + readCounts * readItemsMax, remainBytes);
    if (remainBytesRead == -1) {
        return totalBytesRead / typeSize;
    }
    totalBytesRead += static_cast<size_t>(remainBytesRead);

    return totalBytesRead / typeSize;
}


/**
 * @brief 包装文件描述（文件柄）的结构体，以遵循"资源获取即初始化(RAII)"原则
 *
 */
struct VstarIOWriter {
    explicit VstarIOWriter(const std::string &fname);

    ~VstarIOWriter();

    template <typename T>
    void WriteAndCheck(const T *dataPtr, size_t dataSize) const
    {
        size_t itemsWrite = WriteBatchImpl(fd, dataPtr, sizeof(T), dataSize);
        size_t bytesWrite = itemsWrite * sizeof(T);
        ASCEND_THROW_IF_NOT_FMT(bytesWrite == dataSize,
                                "Write index failed; Expect to write %lu bytes data, actually write %lu bytes data.\n",
                                dataSize, bytesWrite);
    }

    int fd = -1;
    std::string name;
};

struct VstarIOReader {
    explicit VstarIOReader(const std::string &fname);

    ~VstarIOReader();

    template <typename T>
    void ReadAndCheck(T *dataPtr, size_t dataSize) const
    {
        size_t itemsRead = ReadBatchImpl(fd, dataPtr, sizeof(T), dataSize);
        size_t bytesRead = itemsRead * sizeof(T);
        ASCEND_THROW_IF_NOT_FMT(bytesRead == dataSize,
                                "Read index failed; Expect to read %lu bytes data, actually read %lu bytes data.\n",
                                dataSize, bytesRead);
    }

    // 新版本索引落盘时增加部分变量，为保证老版本索引读取时不被变量校验报错，新增接口应对此类情况，如果读取失败不报错
    template <typename T>
    void ReadWithoutCheck(T *dataPtr, size_t dataSize) const
    {
        // 不对读取函数返回值进行校验
        (void) ReadBatchImpl(fd, dataPtr, sizeof(T), dataSize);
    }

    int fd = -1;
    std::string name;
};

}  // namespace ascendSearchacc

#endif