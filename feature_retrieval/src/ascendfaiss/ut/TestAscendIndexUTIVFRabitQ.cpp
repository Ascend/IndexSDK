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

#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFRaBitQ.h>
#include <faiss/impl/IDSelector.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "Common.h"
#include "acl.h"
#include "common/utils/SocUtils.h"
#include "faiss/ascend/AscendIndexIVFRaBitQ.h"
#include "mockcpp/mockcpp.hpp"

namespace ascend
{

void Norm(float* data, int ntotal, int dim)
{
#pragma omp parallel for if (ntotal > 1)
    for (int i = 0; i < ntotal; ++i)
    {
        float l2norm = 0.0;
        for (int j = 0; j < dim; ++j)
        {
            l2norm += data[i * dim + j] * data[i * dim + j];
        }
        l2norm = std::sqrt(l2norm);
        if (fabs(l2norm) < 1e-6)
        {
            // reset data when l2norm = 0
            for (int j = 0; j < dim; ++j)
            {
                data[i * dim + j] = 1.0f / std::sqrt(dim);
            }
        }
        else
        {
            for (int j = 0; j < dim; ++j)
            {
                data[i * dim + j] = data[i * dim + j] / l2norm;
            }
        }
    }
}

void generateData(float* data, int ntotal, int dim)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);
    for (long long int i = 0; i < ntotal * dim; i++)
    {
        data[i] = dis(gen);
    }
    Norm(data, ntotal, dim);
}

TEST(TestAscendIndexIVFRaBitQ, ConstructValidDim)
{
    int dim = 128;  // 128为合法维度
    int nlist = 1024;

    std::string msg = "";
    faiss::MetricType type = faiss::METRIC_L2;
    faiss::ascend::AscendIndexIVFRaBitQConfig conf({0});
    try
    {
        faiss::ascend::AscendIndexIVFRaBitQ index(dim, type, nlist, conf);
        msg = "";
    }
    catch (std::exception& e)
    {
        msg = e.what();
        std::cout << "Exception in ConstructValidDim test: " << msg << std::endl;
    }
    EXPECT_EQ(msg, "");
}

TEST(TestAscendIndexIVFRaBitQ, ConstructInvalidDim)
{
    int dim = 130;  // 130不为合法维度
    int nlist = 1024;

    std::string msg = "";
    faiss::MetricType type = faiss::METRIC_L2;
    faiss::ascend::AscendIndexIVFRaBitQConfig conf({0});
    try
    {
        faiss::ascend::AscendIndexIVFRaBitQ index(dim, type, nlist, conf);
        msg = "";
    }
    catch (std::exception& e)
    {
        msg = e.what();
        std::cout << "Expected exception in ConstructInvalidDim test: " << msg << std::endl;
    }
    EXPECT_TRUE(msg.length() > 0);
}

TEST(TestAscendIndexIVFRaBitQ, ALL)
{
    int dim = 128;
    int nlist = 1024;
    int ntotal = 1000000;
    int nprobe = 64;
    int nq = 10;
    int k = 10;

    std::string msg = "";
    faiss::MetricType type = faiss::METRIC_L2;
    faiss::ascend::AscendIndexIVFRaBitQ* index = nullptr;
    faiss::ascend::AscendIndexIVFRaBitQConfig conf({0});
    std::vector<float> data(ntotal * dim);
    std::vector<int> ids(ntotal);
    std::vector<float> dist(nq * k, 0.0);
    std::vector<faiss::idx_t> label(nq * k, 0);
    generateData(data.data(), ntotal, dim);
    for (int i = 0; i < ntotal; i++)
    {
        ids[i] = i;
    }
    conf.useKmeansPP = false;
    int trainNum = ntotal > nlist * 40 ? nlist * 40 : ntotal;
    try
    {
        index = new faiss::ascend::AscendIndexIVFRaBitQ(dim, type, nlist, conf);
        index->verbose = true;
        index->setNumProbes(nprobe);
        index->train(trainNum, data.data());
        index->add_with_ids(ntotal, data.data(), ids.data());
        index->search(nq, data.data(), k, dist.data(), label.data());

        bool flag = true;
        for (int i = 0; i < nq * k; i++)
        {
            if (label[i] == -1)
            {
                flag = false;
                break;
            }
        }
        EXPECT_TRUE(flag);

        delete index;
    }
    catch (std::exception& e)
    {
        msg = e.what();
        std::cout << "Exception in ALL test: " << msg << std::endl;
    }
    EXPECT_EQ(msg, "");
}

TEST(TestAscendIndexIVFRaBitQ, copyTo)
{
    int dim = 128;
    int nlist = 1024;
    int ntotal = 1000000;
    int nprobe = 64;
    int nq = 10;
    int k = 10;

    std::string msg = "";
    faiss::MetricType type = faiss::METRIC_L2;
    faiss::ascend::AscendIndexIVFRaBitQ* index = nullptr;
    faiss::ascend::AscendIndexIVFRaBitQConfig conf({0});
    std::vector<float> data(ntotal * dim);
    std::vector<int> ids(ntotal);
    std::vector<float> dist(nq * k, 0.0);
    std::vector<faiss::idx_t> label(nq * k, 0);
    generateData(data.data(), ntotal, dim);
    for (int i = 0; i < ntotal; i++)
    {
        ids[i] = i;
    }
    conf.useKmeansPP = false;
    int trainNum = ntotal > nlist * 40 ? nlist * 40 : ntotal;
    try
    {
        index = new faiss::ascend::AscendIndexIVFRaBitQ(dim, type, nlist, conf);
        index->verbose = true;
        index->setNumProbes(nprobe);
        index->train(trainNum, data.data());
        index->add_with_ids(ntotal, data.data(), ids.data());

        faiss::IndexFlatL2 quantizer(dim);
        faiss::IndexIVFRaBitQ* index_cpu = new faiss::IndexIVFRaBitQ(&quantizer, dim, nlist);
        index->copyTo(index_cpu);

        EXPECT_EQ(index_cpu->d, dim);
        EXPECT_EQ(index_cpu->metric_type, type);
        EXPECT_EQ(index_cpu->is_trained, true);
        EXPECT_EQ(index_cpu->ntotal, ntotal);
        EXPECT_EQ(index_cpu->nlist, nlist);
        EXPECT_EQ(index_cpu->nprobe, nprobe);

        delete index_cpu;
        delete index;
    }
    catch (std::exception& e)
    {
        msg = e.what();
        std::cout << "Exception in copyTo test: " << msg << std::endl;
    }
    EXPECT_EQ(msg, "");
}

TEST(TestAscendIndexIVFRaBitQ, copyFrom)
{
    int dim = 128;
    int nlist = 1024;
    int ntotal = 1000000;
    int nprobe = 64;
    int nq = 10;
    int k = 10;

    std::string msg = "";
    faiss::MetricType type = faiss::METRIC_L2;
    faiss::ascend::AscendIndexIVFRaBitQ* index = nullptr;
    faiss::ascend::AscendIndexIVFRaBitQConfig conf({0});
    std::vector<float> data(ntotal * dim);
    std::vector<int> ids(ntotal);
    std::vector<float> dist(nq * k, 0.0);
    std::vector<faiss::idx_t> label(nq * k, 0);
    generateData(data.data(), ntotal, dim);
    for (int i = 0; i < ntotal; i++)
    {
        ids[i] = i;
    }
    conf.useKmeansPP = false;
    int trainNum = ntotal > nlist * 40 ? nlist * 40 : ntotal;
    try
    {
        index = new faiss::ascend::AscendIndexIVFRaBitQ(dim, type, nlist, conf);

        faiss::IndexFlatL2 quantizer(dim);
        faiss::IndexIVFRaBitQ* index_cpu = new faiss::IndexIVFRaBitQ(&quantizer, dim, nlist);
        index_cpu->train(trainNum, data.data());
        index_cpu->add(ntotal, data.data());

        index->copyFrom(index_cpu);

        EXPECT_EQ(index->ntotal, ntotal);
        EXPECT_EQ(index->is_trained, true);

        index->setNumProbes(nprobe);
        index->search(nq, data.data(), k, dist.data(), label.data());

        bool flag = true;
        for (int i = 0; i < nq * k; i++)
        {
            if (label[i] == -1)
            {
                flag = false;
                break;
            }
        }
        EXPECT_TRUE(flag);

        delete index_cpu;
        delete index;
    }
    catch (std::exception& e)
    {
        msg = e.what();
        std::cout << "Exception in copyFrom test: " << msg << std::endl;
    }
    EXPECT_EQ(msg, "");
}

TEST(TestAscendIndexIVFRaBitQ, SearchWithIdSelector)
{
    const int dim = 128;
    const int nlist = 1024;
    const int ntotal = 2000;
    const int nprobe = 64;
    const int nq = 5;
    const int k = 10;

    std::string msg = "";
    faiss::MetricType type = faiss::METRIC_L2;
    faiss::ascend::AscendIndexIVFRaBitQConfig conf({0});
    conf.useKmeansPP = false;
    std::vector<float> data(ntotal * dim);
    std::vector<faiss::idx_t> ids(ntotal);
    generateData(data.data(), ntotal, dim);
    for (int i = 0; i < ntotal; ++i)
    {
        ids[i] = i;
    }
    const int trainNum = ntotal > nlist * 40 ? nlist * 40 : ntotal;

    try
    {
        faiss::ascend::AscendIndexIVFRaBitQ index(dim, type, nlist, conf);
        index.setNumProbes(nprobe);
        index.train(trainNum, data.data());
        index.add_with_ids(ntotal, data.data(), ids.data());

        std::vector<float> dist(nq * k, 0.0f);
        std::vector<faiss::idx_t> label(nq * k, -1);

        // Range: keep ids in [0, ntotal/2)
        {
            faiss::IDSelectorRange rangeSel(0, ntotal / 2);
            faiss::SearchParameters params;
            params.sel = &rangeSel;
            index.search(nq, data.data(), k, dist.data(), label.data(), &params);
            int valid = 0;
            for (int i = 0; i < nq * k; ++i)
            {
                if (label[i] < 0)
                {
                    continue;
                }
                ++valid;
                EXPECT_LT(label[i], ntotal / 2);
                EXPECT_GE(label[i], 0);
            }
            EXPECT_GT(valid, 0);
        }

        // Batch: only even ids in [0, 100)
        {
            std::vector<faiss::idx_t> allow;
            for (int i = 0; i < 100; i += 2)
            {
                allow.push_back(i);
            }
            faiss::IDSelectorBatch batchSel(allow.size(), allow.data());
            faiss::SearchParameters params;
            params.sel = &batchSel;
            index.search(nq, data.data(), k, dist.data(), label.data(), &params);
            int valid = 0;
            for (int i = 0; i < nq * k; ++i)
            {
                if (label[i] < 0)
                {
                    continue;
                }
                ++valid;
                EXPECT_TRUE(batchSel.is_member(label[i]));
            }
            EXPECT_GT(valid, 0);
        }

        // Bitmap: allow first 64 ids
        {
            std::vector<uint8_t> bitmap(ntotal / 8 + 1, 0);
            for (int i = 0; i < 64; ++i)
            {
                bitmap[i / 8] |= static_cast<uint8_t>(1u << (i % 8));
            }
            faiss::IDSelectorBitmap bitmapSel(bitmap.size(), bitmap.data());
            faiss::SearchParameters params;
            params.sel = &bitmapSel;
            index.search(nq, data.data(), k, dist.data(), label.data(), &params);
            int valid = 0;
            for (int i = 0; i < nq * k; ++i)
            {
                if (label[i] < 0)
                {
                    continue;
                }
                ++valid;
                EXPECT_TRUE(bitmapSel.is_member(label[i]));
            }
            EXPECT_GT(valid, 0);
        }

        // Not(Range): exclude [0, ntotal/2)
        {
            faiss::IDSelectorRange rangeSel(0, ntotal / 2);
            faiss::IDSelectorNot notSel(&rangeSel);
            faiss::SearchParameters params;
            params.sel = &notSel;
            index.search(nq, data.data(), k, dist.data(), label.data(), &params);
            int valid = 0;
            for (int i = 0; i < nq * k; ++i)
            {
                if (label[i] < 0)
                {
                    continue;
                }
                ++valid;
                EXPECT_GE(label[i], ntotal / 2);
            }
            EXPECT_GT(valid, 0);
        }
    }
    catch (std::exception& e)
    {
        msg = e.what();
        std::cout << "Exception in SearchWithIdSelector test: " << msg << std::endl;
    }
    EXPECT_EQ(msg, "");
}

TEST(TestAscendIndexIVFRaBitQ, SearchWithIdSelectorMultiDevice)
{
    uint32_t deviceCount = 0;
    auto ret = aclrtGetDeviceCount(&deviceCount);
    if (ret != ACL_SUCCESS || deviceCount < 2)
    {
        GTEST_SKIP() << "Need >=2 devices for multi-card IDSelector test";
    }

    const int dim = 128;
    const int nlist = 1024;
    const int ntotal = 4000;
    const int nprobe = 64;
    const int nq = 4;
    const int k = 10;

    std::string msg = "";
    faiss::MetricType type = faiss::METRIC_L2;
    faiss::ascend::AscendIndexIVFRaBitQConfig conf({0, 1});
    conf.useKmeansPP = false;
    std::vector<float> data(ntotal * dim);
    std::vector<faiss::idx_t> ids(ntotal);
    generateData(data.data(), ntotal, dim);
    for (int i = 0; i < ntotal; ++i)
    {
        ids[i] = i;
    }
    const int trainNum = ntotal > nlist * 40 ? nlist * 40 : ntotal;

    try
    {
        faiss::ascend::AscendIndexIVFRaBitQ index(dim, type, nlist, conf);
        index.setNumProbes(nprobe);
        index.train(trainNum, data.data());
        index.add_with_ids(ntotal, data.data(), ids.data());

        faiss::IDSelectorRange rangeSel(0, ntotal / 2);
        faiss::SearchParameters params;
        params.sel = &rangeSel;

        std::vector<float> dist(nq * k, 0.0f);
        std::vector<faiss::idx_t> label(nq * k, -1);
        index.search(nq, data.data(), k, dist.data(), label.data(), &params);

        int valid = 0;
        for (int i = 0; i < nq * k; ++i)
        {
            if (label[i] < 0)
            {
                continue;
            }
            ++valid;
            EXPECT_LT(label[i], ntotal / 2);
        }
        EXPECT_GT(valid, 0);
    }
    catch (std::exception& e)
    {
        msg = e.what();
        std::cout << "Exception in SearchWithIdSelectorMultiDevice test: " << msg << std::endl;
    }
    EXPECT_EQ(msg, "");
}

}  // namespace ascend
