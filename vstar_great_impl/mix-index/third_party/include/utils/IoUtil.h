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


#ifndef RETRIEVAL_ALGO_ACC_LIB_IOUTIL_H
#define RETRIEVAL_ALGO_ACC_LIB_IOUTIL_H

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>

#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <sys/time.h>

#include "faiss/Index.h"
#include "faiss/IndexPQ.h"
#include "faiss/IndexFlat.h"
#include "impl/IndexHNSWRobustGraph.h"
#include "utils/AscendSearchAssert.h"
#include "utils/VstarIoUtil.h"

namespace ascendsearch {
    namespace graph {
#ifdef DEBUG
#define ASSERT(f) assert(f)
#else
#define ASSERT(f) ((void)0)
#endif
#define WRITEANDCHECK(ptr, n)                                                                                 \
    do {                                                                                                         \
        size_t ret = (*f)(ptr, sizeof(*(ptr)), n);                                                       \
        ASCENDSEARCH_THROW_IF_NOT_FMT(ret == (n), "write error in %s: %zu != %zu (%s)", f->name.c_str(), ret, \
                                      size_t(n), strerror(errno));                                            \
    } while (0)
#define WRITE(x) WRITEANDCHECK(&(x), 1)
#define WRITEVECTOR(vec)                   \
    do {                                      \
        size_t size = (vec).size();        \
        WRITEANDCHECK(&size, 1);           \
        WRITEANDCHECK((vec).data(), size); \
    } while (0)
#define WRITEVECTORCODES(vec, n)                   \
    do {                                      \
        size_t size = static_cast<size_t>((vec).size()) / static_cast<size_t>(n)  ;        \
        WRITEANDCHECK(&size, 1);           \
        WRITEANDCHECK((vec).data(), (vec).size()); \
    } while (0)

#define READANDCHECK(ptr, n)                                                                                 \
    do {                                                                                                        \
        size_t ret = (*f)(ptr, sizeof(*(ptr)), n);                                                           \
        ASCENDSEARCH_THROW_IF_NOT_FMT(ret == (n), "read error in %s: %zu != %zu (%s)", f->name.c_str(), ret, \
                                      size_t(n), strerror(errno));                                           \
    } while (0)
#define READ(x) READANDCHECK(&(x), 1)
#define READVECTOR(vec)                                                     \
    do {                                                                       \
        size_t size;                                                        \
        READANDCHECK(&size, 1);                                             \
        ASCENDSEARCH_THROW_IF_NOT(size < (uint64_t{1} << 40)); \
        (vec).resize(size);                                                 \
        READANDCHECK((vec).data(), size);                                   \
    } while (0)
#define READVECTORCODES(vec, n)                                                     \
    do {                                                                       \
        size_t size;                                                        \
        READANDCHECK(&size, 1);                                                      \
        size = size * static_cast<size_t>(n) ;                                       \
        ASCENDSEARCH_THROW_IF_NOT(size < (uint64_t{1} << 40)); \
        (vec).resize(size);                                                 \
        READANDCHECK((vec).data(), size);                                   \
    } while (0)

const int G_MAX_DIM = 1000000;
const int G_SEP_NUM = 1000;
const int G_SIZE_INT = 4;
const int G_SIZE_LONG = 8;

struct IOWriter {
    std::string name;

    virtual ~IOWriter() noexcept
    {
    }

    virtual int fileno();
};

struct FileIOWriter : IOWriter {
    FILE *f = nullptr;
    bool need_close = false;
    int fd = -1;

    explicit FileIOWriter(FILE *wf);

    explicit FileIOWriter(std::string fname);

    ~FileIOWriter() override;

    template <class T>
    size_t operator()(const T *ptr, size_t size, size_t nitems)
    {
        return ascendSearchacc::WriteBatchImpl(fd, ptr, size, size * nitems);
    }

    int fileno() override;
};

struct IOReader {
    std::string name;

    virtual ~IOReader()
    {
    }

    virtual int fileno();
};

struct FileIOReader : IOReader {
    FILE *f = nullptr;
    bool need_close = false;
    int fd = -1;

    explicit FileIOReader(FILE *rf);

    explicit FileIOReader(std::string fname);

    ~FileIOReader() override;

    template <class T>
    size_t operator()(T *ptr, size_t size, size_t nitems)
    {
        return ascendSearchacc::ReadBatchImpl(fd, ptr, size, size * nitems);
    }

    int fileno() override;
};

// / cast a 4-character string to a uint32_t that can be written and read easily
uint32_t fourcc(const char sx[4]);
uint32_t fourcc(const std::string &sx);

void WriteIndex(const faiss::Index *idx, FileIOWriter *f);
std::unique_ptr<IndexHNSWGraphPQHybrid> ReadIndexHNSWGraphPQHybrid(FileIOReader *f, int ioFlags);
}  // namespace graph
}  // namespace ascendsearch

#endif  // RETRIEVAL_ALGO_ACC_LIB_IOUTIL_H
