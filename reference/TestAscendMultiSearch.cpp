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

// 需要生成aicpu算子+int8flat(-d 64)+sq8算子(-d 64)

#include <faiss/ascend/AscendIndexInt8Flat.h>
#include <faiss/ascend/AscendIndexSQ.h>
#include <faiss/ascend/AscendMultiIndexSearch.h>
#include <faiss/index_io.h>

#include <algorithm>
#include <cfloat>
#include <cstring>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

namespace
{

const auto METRIC_TYPE = faiss::METRIC_INNER_PRODUCT;

void Norm(float *data, int n, int dim)
{
#pragma omp parallel for if (n > 100)
    for (int i = 0; i < n; ++i)
    {
        float l2norm = 0;
        for (int j = 0; j < dim; ++j)
        {
            l2norm += data[i * dim + j] * data[i * dim + j];
        }
        l2norm = std::sqrt(l2norm);
        if (fabs(l2norm) < FLT_EPSILON)
        {
            std::cerr << "Error: Invalid l2norm value." << std::endl;
        }
        for (int j = 0; j < dim; ++j)
        {
            data[i * dim + j] = data[i * dim + j] / l2norm;
        }
    }
}

void GenerateCodes(int8_t *codes, int total, int dim, int seed = -1)
{
    int maxValue = 255;
    int offset = 128;
    auto timeSeed = time(nullptr);
    if (timeSeed < 0)
    {
        std::cerr << "Error: Invalid seed value." << std::endl;
        return;
    }
    std::default_random_engine e((seed > 0) ? seed : timeSeed);
    std::uniform_real_distribution<float> rCode(0.0f, 1.0f);
    for (int i = 0; i < total; i++)
    {
        for (int j = 0; j < dim; j++)
        {
            // uint8's max value is 255
            codes[i * dim + j] = static_cast<int8_t>(maxValue * rCode(e) - offset);
        }
    }
}

// camera id 简写为 cid， 7位数， 取值范围0~127

const int32_t K_MAX_CAMERA_NUM = 128;
const int MASK_LEN = 8;

struct IDFilter
{
    IDFilter()
    {
        std::fill_n(cameraIdMask, K_MAX_CAMERA_NUM / MASK_LEN, 0xFF);

        timeRange[0] = 0;
        timeRange[1] = -1;
    }

    uint8_t cameraIdMask[16] = {0xFF};  //  cid
    uint32_t timeRange[2] = {0};        // 时间戳
};

void DoSearchFilter(int ntotal, int dim, std::vector<float> &data)
{
    int searchNum = 2;
    int indexNum = 10;
    std::vector<faiss::ascend::AscendIndex *> indexes(indexNum);
    faiss::ascend::AscendIndexSQConfig conf({0});
    conf.filterable = true;
    std::vector<int64_t> ids(ntotal, 0);
    int seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine e1(seed);
    std::uniform_int_distribution<int32_t> id(0, std::numeric_limits<int32_t>::max());
    std::uniform_int_distribution<uint8_t> searchCid(0, K_MAX_CAMERA_NUM - 1);

    for (int i = 0; i < ntotal; i++)
    {
        // 默认拼接方式时间戳在第10位bit开始，cid在第42位bit开始
        ids[i] = (static_cast<int64_t>(searchCid(e1)) << 42) + (static_cast<int64_t>(id(e1)) << 10);
    }

    std::cout << "AscendIndex SQ MultiSearch with different filter start" << std::endl;
    for (int i = 0; i < indexNum; i++)
    {
        indexes[i] =
            new faiss::ascend::AscendIndexSQ(dim, faiss::ScalarQuantizer::QuantizerType::QT_8bit, METRIC_TYPE, conf);

        indexes[i]->train(ntotal, data.data());
        indexes[i]->add_with_ids(ntotal, data.data(), ids.data());
    }

    void *multiFilters[searchNum];
    IDFilter idFilters[indexNum * searchNum];
    for (int query = 0; query < searchNum; query++)
    {
        for (int indexId = 0; indexId < indexNum; indexId++)
        {
            IDFilter idFilter;
            idFilter.timeRange[0] = 0;
            idFilter.timeRange[1] = 0x7fffffff;
            // 16个uint8_t表示128位的cid
            for (int i = 0; i < 16; i++)
            {
                idFilter.cameraIdMask[i] = searchCid(e1);
            }
            idFilters[query * indexNum + indexId] = idFilter;
        }
        multiFilters[query] = &idFilters[query * indexNum];
    }
    int k = 10;
    std::vector<float> dist(indexNum * k * searchNum, 0);
    std::vector<faiss::idx_t> label(indexNum * k * searchNum, 0);
    SearchWithFilter(indexes, searchNum, data.data(), k, dist.data(), label.data(), multiFilters, false);

    for (int i = 0; i < indexNum; i++)
    {
        delete indexes[i];
    }
}

void AscendIndexSQMultiSearchFilter()
{
    int dim = 64;
    try
    {
        int ntotal = 128;
        std::vector<float> data(dim * ntotal);
        for (int i = 0; i < dim * ntotal; i++)
        {
            data[i] = drand48();
        }
        Norm(data.data(), ntotal, dim);
        DoSearchFilter(ntotal, dim, data);
        std::cout << "AscendIndex SQ MultiSearch with different filter end" << std::endl;
    }
    catch (std::exception &e)
    {
        printf("%s\n", e.what());
    }
}

void DoIndexSearchFilter(int ntotal, int dim, std::vector<float> &data)
{
    int searchNum = 2;
    int indexNum = 10;
    std::vector<faiss::Index *> indexes(indexNum);
    std::vector<int64_t> ids(ntotal, 0);
    faiss::ascend::AscendIndexSQConfig conf({0});
    conf.filterable = true;
    int seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine e1(seed);
    std::uniform_int_distribution<int32_t> id(0, std::numeric_limits<int32_t>::max());
    std::uniform_int_distribution<uint8_t> searchCid(0, K_MAX_CAMERA_NUM - 1);

    for (int i = 0; i < ntotal; i++)
    {
        // 默认拼接方式时间戳在第10位bit开始，cid在第42位bit开始
        ids[i] = (static_cast<int64_t>(searchCid(e1)) << 42) + (static_cast<int64_t>(id(e1)) << 10);
    }

    std::cout << "Index SQ MultiSearch with different filter start" << std::endl;
    for (int i = 0; i < indexNum; i++)
    {
        indexes[i] =
            new faiss::ascend::AscendIndexSQ(dim, faiss::ScalarQuantizer::QuantizerType::QT_8bit, METRIC_TYPE, conf);

        indexes[i]->train(ntotal, data.data());
        indexes[i]->add_with_ids(ntotal, data.data(), ids.data());
    }

    void *multiFilters[searchNum];
    IDFilter idFilters[indexNum * searchNum];
    for (int query = 0; query < searchNum; query++)
    {
        for (int indexId = 0; indexId < indexNum; indexId++)
        {
            IDFilter idFilter;
            idFilter.timeRange[0] = 0;
            idFilter.timeRange[1] = 0x7fffffff;
            // 16个uint8_t表示128位的cid
            for (int i = 0; i < 16; i++)
            {
                idFilter.cameraIdMask[i] = searchCid(e1);
            }
            idFilters[query * indexNum + indexId] = idFilter;
        }
        multiFilters[query] = &idFilters[query * indexNum];
    }
    int k = 10;
    std::vector<float> dist(indexNum * k * searchNum, 0);
    std::vector<faiss::idx_t> label(indexNum * k * searchNum, 0);
    faiss::ascend::SearchWithFilter(indexes, searchNum, data.data(), k, dist.data(), label.data(), multiFilters, false);
    for (int i = 0; i < indexNum; i++)
    {
        delete indexes[i];
    }
}

void IndexSQMultiSearchFilter()
{
    int dim = 64;
    try
    {
        int ntotal = 128;
        std::vector<float> data(dim * ntotal);
        for (int i = 0; i < dim * ntotal; i++)
        {
            data[i] = drand48();
        }
        Norm(data.data(), ntotal, dim);
        DoIndexSearchFilter(ntotal, dim, data);
        std::cout << "Index SQ MultiSearch with different filter end" << std::endl;
    }
    catch (std::exception &e)
    {
        printf("%s\n", e.what());
    }
}

void AscendIndexSQMultiSearch()
{
    size_t ntotal = 1000000;
    int k = 5;
    int dim = 64;
    int indexNum = 10;
    std::vector<int> searchNum = {1, 2, 4, 8};
    size_t maxSize = ntotal * dim;
    std::vector<float> data(maxSize);
    std::vector<faiss::ascend::AscendIndex *> indexes(indexNum);
    try
    {
        for (size_t i = 0; i < dim * ntotal; i++)
        {
            data[i] = drand48();
        }
        Norm(data.data(), ntotal, dim);

        std::cout << "AscendIndex SQ MultiSearch start" << std::endl;
        faiss::ascend::AscendIndexSQConfig conf({0});
        for (int i = 0; i < indexNum; i++)
        {
            indexes[i] = new faiss::ascend::AscendIndexSQ(dim, faiss::ScalarQuantizer::QuantizerType::QT_8bit,
                                                          METRIC_TYPE, conf);

            indexes[i]->train(ntotal, data.data());
            indexes[i]->add(ntotal, data.data());
        }

        for (size_t i = 0; i < searchNum.size(); i++)
        {
            std::vector<float> dist(indexNum * searchNum[i] * k, 0);
            std::vector<faiss::idx_t> label(indexNum * searchNum[i] * k, 0);

            Search(indexes, searchNum[i], data.data(), k, dist.data(), label.data(), false);
        }
        for (int i = 0; i < indexNum; i++)
        {
            delete indexes[i];
        }
        std::cout << "AscendIndex SQ MultiSearch end" << std::endl;
    }
    catch (std::exception &e)
    {
        for (int i = 0; i < indexNum; i++)
        {
            delete indexes[i];
        }
        printf("%s\n", e.what());
    }
}

void AscendIndexInt8MultiSearch()
{
    size_t ntotal = 100000;
    int k = 5;
    int dim = 64;
    int indexNum = 10;
    std::vector<int> searchNum = {1, 2, 4, 8};

    std::vector<std::vector<int8_t>> data(indexNum, std::vector<int8_t>(dim * ntotal));
    std::vector<faiss::ascend::AscendIndexInt8 *> indexes(indexNum);
    try
    {
        for (int i = 0; i < indexNum; i++)
        {
            GenerateCodes(data[i].data(), ntotal, dim);
        }

        std::cout << "AscendIndex Int8 MultiSearch start" << std::endl;
        faiss::ascend::AscendIndexInt8FlatConfig conf({0});

        for (int i = 0; i < indexNum; i++)
        {
            indexes[i] = new faiss::ascend::AscendIndexInt8Flat(dim, METRIC_TYPE, conf);

            indexes[i]->add(ntotal, data[i].data());
        }

        for (size_t i = 0; i < searchNum.size(); i++)
        {
            std::vector<float> dist(indexNum * searchNum[i] * k, 0);
            std::vector<faiss::idx_t> label(indexNum * searchNum[i] * k, 0);
            std::vector<int8_t> query(dim * searchNum[i]);
            GenerateCodes(query.data(), searchNum[i], dim);

            Search(indexes, searchNum[i], query.data(), k, dist.data(), label.data(), false);
        }
        for (int i = 0; i < indexNum; i++)
        {
            delete indexes[i];
        }
        std::cout << "AscendIndex Int8 MultiSearch end" << std::endl;
    }
    catch (std::exception &e)
    {
        for (int i = 0; i < indexNum; i++)
        {
            delete indexes[i];
        }
        printf("%s\n", e.what());
    }
}
}  // namespace

int main(int argc, char **argv)
{
    AscendIndexSQMultiSearchFilter();  // SQ算法多index批量检索带属性过滤 （AscendIndex）
    IndexSQMultiSearchFilter();        // SQ算法多index批量检索带属性过滤（Index）
    AscendIndexSQMultiSearch();        // SQ算法多index批量检索
    AscendIndexInt8MultiSearch();      // Int8算法多index批量检索
    return 0;
}
