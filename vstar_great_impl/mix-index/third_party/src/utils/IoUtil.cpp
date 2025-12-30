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

#include "utils/IoUtil.h"

#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cstring>
#include <securec.h>

#include "impl/IndexHNSWRobustGraph.h"

namespace ascendsearch {
namespace graph {

int IOWriter::fileno()
{
    ASCENDSEARCH_THROW_MSG("IOWriter does not support memory mapping");
}

int IOReader::fileno()
{
    ASCENDSEARCH_THROW_MSG("IOReader does not support memory mapping");
}

FileIOWriter::FileIOWriter(FILE *wf) : f(wf)
{
}

FileIOWriter::FileIOWriter(std::string fname)
{
    name = fname;
    need_close = true;
    ascendSearchacc::initializeFileDescription(fd, name);
}

FileIOWriter::~FileIOWriter()
{
    if (need_close) {
        ascendSearchacc::closeFileDescription(fd);
    }
}

int FileIOWriter::fileno()
{
    return ::fileno(f);
}

FileIOReader::FileIOReader(FILE *rf) : f(rf)
{
}

FileIOReader::FileIOReader(std::string fname)
{
    name = fname;
    fd = open(name.c_str(), O_NOFOLLOW | O_RDONLY);
    ASCENDSEARCH_THROW_IF_NOT_MSG(fd >= 0, // 文件未打开成功，无需关闭
        "fname or one of its parent directory is a softlink, or error encoutered during opening fname.\n");
    need_close = true;
}

FileIOReader::~FileIOReader()
{
    if (need_close) {
        ascendSearchacc::closeFileDescription(fd);
    }
}

int FileIOReader::fileno()
{
    return ::fileno(f);
}

uint32_t fourcc(const char sx[4])
{
    ASCENDSEARCH_THROW_IF_NOT(strlen(sx) == 4); // 索引类型用4个char来区分
    return (static_cast<unsigned char>(sx[0]) |
            (static_cast<unsigned char>(sx[1]) << 8) |
            (static_cast<unsigned char>(sx[2]) << 16) |
            (static_cast<unsigned char>(sx[3]) << 24)); // 4个char分别左移0、8、16、24位组成一个uint32
}

uint32_t fourcc(const std::string &sx)
{
    ASCENDSEARCH_THROW_IF_NOT(sx.length() == 4); // 索引类型用4个char来区分
    return (static_cast<unsigned char>(sx[0]) |
            (static_cast<unsigned char>(sx[1]) << 8) |
            (static_cast<unsigned char>(sx[2]) << 16) |
            (static_cast<unsigned char>(sx[3]) << 24)); // 4个char分别左移0、8、16、24位组成一个uint32
}

template <class T>
void WriteAndCheck(FileIOWriter *f, T *ptr, size_t n)
{
    size_t ret = (*f)(ptr, sizeof(*(ptr)), n);
    ASCENDSEARCH_THROW_IF_NOT_FMT(ret == (n), "write error in %s: %zu != %zu (%s)", f->name.c_str(), ret, size_t(n),
                                  strerror(errno));
}

template <class T>
void ReadAndCheck(FileIOReader *f, T *ptr, size_t n)
{
    size_t ret = (*f)(ptr, sizeof(*(ptr)), n);
    ASCENDSEARCH_THROW_IF_NOT_FMT(ret == (n), "read error in %s: %zu != %zu (%s)", f->name.c_str(), ret, size_t(n),
                                  strerror(errno));
}

template <class T>
void WriteValue(FileIOWriter *f, T &x)
{
    WriteAndCheck(f, &(x), 1);
}

template <class T>
void ReadValue(FileIOReader *f, T &x)
{
    ReadAndCheck(f, &(x), 1);
}

template <class T>
void WriteVector(FileIOWriter *f, const std::vector<T> &vec)
{
    size_t size = (vec).size();
    WriteAndCheck(f, &size, 1);
    WriteAndCheck(f, vec.data(), size);
}

template <class T>
void ReadVector(FileIOReader *f, std::vector<T> &vec)
{
    const int shift = 40;
    size_t size;
    ReadAndCheck(f, &size, 1);
    ASCENDSEARCH_THROW_IF_NOT(size < (uint64_t{1} << shift));
    vec.resize(size);
    ReadAndCheck(f, vec.data(), size);
}

template <class T>
void CheckVector(std::vector<T> &vec, T begin, T end)
{
    for (size_t i = 0; i < vec.size(); ++i) {
        ASCENDSEARCH_THROW_IF_NOT(vec[i] >= begin && vec[i] < end);
    }
}

static void WriteIndexHeader(const faiss::Index *idx, FileIOWriter *f)
{
    WriteValue(f, idx->d);
    WriteValue(f, idx->ntotal);
    idx_t dummy = 1 << 20;
    WriteValue(f, dummy);
    WriteValue(f, dummy);
    WriteValue(f, idx->is_trained);
    WriteValue(f, idx->metric_type);
}

static void ReadIndexHeader(faiss::Index *idx, FileIOReader *f)
{
    ReadValue(f, idx->d);
    ASCENDSEARCH_THROW_IF_NOT(std::find(validDim.begin(), validDim.end(), idx->d) != validDim.end());
    ReadValue(f, idx->ntotal);
    ASCENDSEARCH_THROW_IF_NOT(idx->ntotal > 0 && idx->ntotal <= N_TOTAL_MAX);
    idx_t dummy;
    ReadValue(f, dummy);
    ReadValue(f, dummy);
    ReadValue(f, idx->is_trained);
    ASCENDSEARCH_THROW_IF_NOT(idx->is_trained); // only true
    ReadValue(f, idx->metric_type);
    // only support IP and L2
    ASCENDSEARCH_THROW_IF_NOT(idx->metric_type == faiss::METRIC_L2 || idx->metric_type == faiss::METRIC_INNER_PRODUCT);
    idx->verbose = false;
}

static void write_index_header(const faiss::Index *idx, FileIOWriter *f)
{
    WRITE(idx->d);
    WRITE(idx->ntotal);
    idx_t dummy = 1 << 20;
    WRITE(dummy);
    WRITE(dummy);
    WRITE(idx->is_trained);
    WRITE(idx->metric_type);
}
static void read_index_header(faiss::Index *idx, FileIOReader *f)
{
    READ(idx->d);
    ASCENDSEARCH_THROW_IF_NOT(std::find(validDim.begin(), validDim.end(), idx->d) != validDim.end());
    READ(idx->ntotal);
    ASCENDSEARCH_THROW_IF_NOT(idx->ntotal > 0 && idx->ntotal <= N_TOTAL_MAX);
    idx_t dummy;
    READ(dummy);
    READ(dummy);
    READ(idx->is_trained);
    ASCENDSEARCH_THROW_IF_NOT(idx->is_trained); // only true
    READ(idx->metric_type);
    // only support IP and L2
    ASCENDSEARCH_THROW_IF_NOT(idx->metric_type == faiss::METRIC_L2 || idx->metric_type == faiss::METRIC_INNER_PRODUCT);
    idx->verbose = false;
}
void write_ProductQuantizer(const faiss::ProductQuantizer *pq, FileIOWriter *f)
{
    WRITE(pq->d);
    WRITE(pq->M);
    WRITE(pq->nbits);
    WRITEVECTOR(pq->centroids);
}
static void read_index_ProductQuantizer(faiss::ProductQuantizer *pq, FileIOReader *f)
{
    READ(pq->d);
    ASCENDSEARCH_THROW_IF_NOT(std::find(validDim.begin(), validDim.end(), pq->d) != validDim.end());
    READ(pq->M);
    ASCENDSEARCH_THROW_IF_NOT(pq->M > 0 && pq->M <= pq->d);
    READ(pq->nbits);
    ASCENDSEARCH_THROW_IF_NOT(pq->nbits == DEFAULT_INDEX_PQ_BITS); // only support 8 bit
    pq->set_derived_values();
    READVECTOR(pq->centroids);
}

static void WriteHNSWGraph(const HNSWGraph *graph, FileIOWriter *f)
{
    WriteVector(f, graph->assignProbas);
    WriteVector(f, graph->cumNeighborCntPerLevel);
    WriteVector(f, graph->levels);
    WriteVector(f, graph->offsets);
    WriteVector(f, graph->neighbors);

    WriteValue(f, graph->entryPoint);
    WriteValue(f, graph->maxLevel);
    WriteValue(f, graph->efConstruction);
    WriteValue(f, graph->efSearch);
    WriteValue(f, graph->upperBeam);
}

static void ReadHNSW(HNSWGraph *hnsw, FileIOReader *f, idx_t nTotal)
{
    ReadVector(f, hnsw->assignProbas);
    ReadVector(f, hnsw->cumNeighborCntPerLevel);
    ReadVector(f, hnsw->levels);
    ReadVector(f, hnsw->offsets);
    ReadVector(f, hnsw->neighbors);

    ReadValue(f, hnsw->entryPoint);
    ASCENDSEARCH_THROW_IF_NOT(hnsw->entryPoint >= 0 && hnsw->entryPoint <= nTotal);
    ReadValue(f, hnsw->maxLevel);
    ASCENDSEARCH_THROW_IF_NOT(hnsw->maxLevel > 0 && hnsw->maxLevel < HNSW_MAX_LEVEL);
    ReadValue(f, hnsw->efConstruction);
    ASCENDSEARCH_THROW_IF_NOT(hnsw->efConstruction >= EF_CONSTRUCTION_MIN
        && hnsw->efConstruction <= EF_CONSTRUCTION_MAX && hnsw->efConstruction % 10 == 0); // divisible by 10
    ReadValue(f, hnsw->efSearch);
    ASCENDSEARCH_THROW_IF_NOT(hnsw->efSearch> 0 && hnsw->efSearch <= EF_SEARCH_MAX); // 0 ~ 200
    ReadValue(f, hnsw->upperBeam);
    ASCENDSEARCH_THROW_IF_NOT(hnsw->upperBeam == 1); // 目前取值只能为1

    // check vector value
    CheckVector(hnsw->assignProbas, 1e-9, 1.0);  // assignProbas的取值只能是1e-9到1.0范围内
    CheckVector(hnsw->levels, 1, hnsw->maxLevel + 2); // 1 到 hnsw->maxLevel + 1 取值都行
    CheckVector(hnsw->offsets, 0LU, static_cast<size_t>(hnsw->neighbors.size() + 1));
    CheckVector(hnsw->neighbors, -1, static_cast<int>(nTotal)); // -1 mean no neighbor
}

void WriteIndexInner(const IndexHNSWGraph *cpuIndex, FileIOWriter *f)
{
    uint32_t h = dynamic_cast<const IndexHNSWGraphPQHybrid *>(cpuIndex)
                        ? fourcc("IHNh")
                        : dynamic_cast<const IndexHNSWGraphFlat *>(cpuIndex) ? fourcc("IHNf") : 0;
    ASCENDSEARCH_THROW_IF_NOT_MSG(h != 0, "UnSupportIndex(Supper Class is IndexHNSWGraph)");
    WriteValue(f, h);
    WriteIndexHeader(cpuIndex, f);
    WriteValue(f, cpuIndex->orderType);
    WriteHNSWGraph(&cpuIndex->innerGraph, f);
    WriteIndex(cpuIndex->storage.get(), f);
    if (h == fourcc("IHNh")) {
        const IndexHNSWGraphPQHybrid *idx_g = dynamic_cast<const IndexHNSWGraphPQHybrid *>(cpuIndex);
        WriteIndex(idx_g->p_storage.get(), f);
    }
}

void write_index(const faiss::Index *idx, FileIOWriter *f)
{
    if (const faiss::IndexFlat *idxf = dynamic_cast<const faiss::IndexFlat *>(idx)) {
        uint32_t h = fourcc(idxf->metric_type == faiss::METRIC_INNER_PRODUCT ? "IxFI" :
            idxf->metric_type == faiss::METRIC_L2 ? "IxF2" : "IxFl");
        WRITE(h);
        write_index_header(idx, f);
        WRITEVECTORCODES(idxf->codes, idxf->code_size / idxf->d);
    } else if (const faiss::IndexPQ *idxpq = dynamic_cast<const faiss::IndexPQ *>(idx)) {
        uint32_t header = fourcc("IxPq");
        WRITE(header);
        write_index_header(idx, f);
        write_ProductQuantizer(&idxpq->pq, f);
        WRITEVECTOR(idxpq->codes);
        // search params -- maybe not useful to store?
        WRITE(idxpq->search_type);
        WRITE(idxpq->encode_signs);
        WRITE(idxpq->polysemous_ht);
    } else {
        ASCENDSEARCH_THROW_MSG("don't know how to serialize this type of index");
    }
}

std::unique_ptr<faiss::Index> read_index(FileIOReader *f, int)
{
    uint32_t h;
    READ(h);
    if (h == fourcc("IxFI") || h == fourcc("IxF2") || h == fourcc("IxFl")) {
        std::unique_ptr<faiss::IndexFlat> idxf = nullptr;
        if (h == fourcc("IxFI")) {
            idxf = std::make_unique<faiss::IndexFlatIP>();
        } else if (h == fourcc("IxF2")) {
            idxf = std::make_unique<faiss::IndexFlatL2>();
        } else {
            idxf = std::make_unique<faiss::IndexFlat>();
        }
        read_index_header(idxf.get(), f);
        idxf->code_size = sizeof(float) * idxf->d;
        READVECTORCODES(idxf->codes, sizeof(float));
        ASCENDSEARCH_THROW_IF_NOT(idxf->codes.size() == static_cast<size_t>(idxf->ntotal) * idxf->code_size);
        // leak!
        return idxf;
    } else if (h == fourcc("IxPQ") || h == fourcc("IxPo") || h == fourcc("IxPq")) {
        // IxPQ and IxPo were merged into the same faiss::IndexPQ object
        std::unique_ptr<faiss::IndexPQ> idxpq = std::make_unique<faiss::IndexPQ>();
        read_index_header(idxpq.get(), f);
        read_index_ProductQuantizer(&idxpq->pq, f);
        idxpq->code_size = idxpq->pq.code_size;
        READVECTOR(idxpq->codes);
        if (h == fourcc("IxPo") || h == fourcc("IxPq")) {
            READ(idxpq->search_type);
            READ(idxpq->encode_signs);
            READ(idxpq->polysemous_ht);
        }
        if (h == fourcc("IxPQ") || h == fourcc("IxPo")) {
            idxpq->metric_type = faiss::METRIC_L2;
        }
        return idxpq;
    } else {
        ASCENDSEARCH_THROW_MSG("Index type not recognized");
    }
    return nullptr;
}

std::unique_ptr<faiss::IndexFlat> read_index2(FileIOReader *f, int)
{
    std::unique_ptr<faiss::IndexFlat> idx = nullptr;
    uint32_t h;
    READ(h);
    if (h == fourcc("IxFI") || h == fourcc("IxF2") || h == fourcc("IxFl")) {
        if (h == fourcc("IxFI")) {
            idx = std::make_unique<faiss::IndexFlatIP>();
        } else if (h == fourcc("IxF2")) {
            idx = std::make_unique<faiss::IndexFlatL2>();
        } else {
            idx = std::make_unique<faiss::IndexFlat>();
        }
        read_index_header(idx.get(), f);
        idx->code_size = sizeof(float) * idx->d;
        READVECTORCODES(idx->codes, sizeof(float));
        ASCENDSEARCH_THROW_IF_NOT(idx->codes.size() == static_cast<size_t>(idx->ntotal) * idx->code_size);
    } else {
        ASCENDSEARCH_THROW_MSG("Read_index2 only accept IxF Index");
    }
    return idx;
}

void WriteIndex(const faiss::Index *idx, FileIOWriter *f)
{
    if (const IndexHNSWGraph *cpuIndex = dynamic_cast<const IndexHNSWGraph *>(idx)) {
        WriteIndexInner(cpuIndex, f);
    } else {
        write_index(idx, f);
    }
}

void ReadIndexHNSWGraph(IndexHNSWGraph &index, FileIOReader *f, int ioFlags)
{
    ReadIndexHeader(&index, f);
    ReadValue(f, index.orderType);
    ReadHNSW(&index.innerGraph, f, index.ntotal);
    index.storage = std::move(read_index2(f, ioFlags));
    index.ownFields = true;
}

std::unique_ptr<IndexHNSWGraphPQHybrid> ReadIndexHNSWGraphPQHybrid(FileIOReader *f, int ioFlags)
{
    uint32_t h { 0 };
    ReadValue(f, h);
    if (h != fourcc("IHNh")) {
        return nullptr;
    }

    auto index = std::make_unique<IndexHNSWGraphPQHybrid>();
    ReadIndexHNSWGraph(*index, f, ioFlags);
    index->p_storage = read_index(f, ioFlags);
    return index;
}

}  // namespace graph
}  // namespace ascendsearch
