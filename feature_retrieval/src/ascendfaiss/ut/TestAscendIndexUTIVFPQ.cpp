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

#include <string>
#include <random>
#include <iostream>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFPQ.h>
#include "Common.h"
#include "common/utils/SocUtils.h"
#include "mockcpp/mockcpp.hpp"
#include "faiss/ascend/AscendIndexIVFPQ.h"

namespace ascend {

void Norm(float* data, int ntotal, int dim)
{
#pragma omp parallel for if (ntotal > 1)
    for (int i = 0; i < ntotal; ++i) {
        float l2norm = 0.0;
        for (int j = 0; j < dim; ++j) {
            l2norm += data[i * dim + j] * data[i * dim + j];
        }
        l2norm = std::sqrt(l2norm);
        if (fabs(l2norm) < 1e-6) {
            # reset data when l2norm = 0
            for (int j = 0; j < dim; ++j) {
                data[i * dim + j] = 1.0f / std::sqrt(dim);
            }
        } else {
            for (int j = 0; j < dim; ++j) {
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
    for (long long int i = 0; i < ntotal * dim; i++) {
        data[i] = dis(gen);
    }
    Norm(data, ntotal, dim);
}

TEST(TestAscendIndexIVFPQ, Construct)
{
    int dim = 64;
    int nlist = 1024;
    int msub = 4;
    int nbit = 8;

    std::string msg = "";
    faiss::MetricType type = faiss::METRIC_L2;
    faiss::ascend::AscendIndexIVFPQConfig conf({ 0 });
    try {
        faiss::ascend::AscendIndexIVFPQ index(dim, type, nlist, msub, nbit, conf);
    } catch(std::exception &e) {
        msg = e.what();
    }
    std::string str = "Unsupported dims";
    EXPECT_TRUE(msg.find(str) != std::string::npos);
}

TEST(TestAscendIndexIVFPQ, ALL)
{
    int dim = 128;
    int nlist = 1024;
    int msub = 4;
    int nbit = 8;
    int ntotal = 10000000;
    int nprobe = 64;
    int nq = 10;
    int k = 10;

    std::string msg = "";
    faiss::MetricType type = faiss::METRIC_L2;
    faiss::ascend::AscendIndexIVFPQ* index = nullptr;
    faiss::ascend::AscendIndexIVFPQConfig conf({ 0 });
    std::vector<float> data(ntotal * dim);
    std::vector<int> ids(ntotal);
    std::vector<float> dist(nq * k, 0.0);
    std::vector<faiss::idx_t> label(nq * k, 0);
    generateData(data.data(), ntotal, dim);
    for (int i = 0; i < ntotal; i++) {
        ids[i] = i;
    }
    conf.useKmeansPP = false;
    int trainNum = ntotal > nlist * 40 ? nlist * 40 : ntotal;
    try {
        index = new faiss::ascend::AscendIndexIVFPQ(dim, type, nlist, msub, nbit, conf);
        index->verbose = true;
        index->setNumProbes(nprobe);
        index->train(trainNum, data.data());
        index->add_with_ids(ntotal, data.data(), ids.data());
        index->search(nq, data.data(), k, dist.data(), label.data());
        delete index;
    } catch(std::exception &e) {
        msg = e.what();
    }
    EXPECT_EQ(msg, "");
}

TEST(TestAscendIndexIVFPQ, copyTo)
{
    int dim = 128;
    int nlist = 1024;
    int msub = 4;
    int nbit = 8;
    int ntotal = 10000000;
    int nprobe = 64;
    int nq = 10;
    int k = 10;

    std::string msg = "";
    faiss::MetricType type = faiss::METRIC_L2;
    faiss::ascend::AscendIndexIVFPQ* index = nullptr;
    faiss::ascend::AscendIndexIVFPQConfig conf({ 0 });
    std::vector<float> data(ntotal * dim);
    std::vector<int> ids(ntotal);
    std::vector<float> dist(nq * k, 0.0);
    std::vector<faiss::idx_t> label(nq * k, 0);
    generateData(data.data(), ntotal, dim);
    for (int i = 0; i < ntotal; i++) {
        ids[i] = i;
    }
    conf.useKmeansPP = false;
    int trainNum = ntotal > nlist * 40 ? nlist * 40 : ntotal;
    try {
        index = new faiss::ascend::AscendIndexIVFPQ(dim, type, nlist, msub, nbit, conf);
        index->verbose = true;
        index->setNumProbes(nprobe);
        index->train(trainNum, data.data());
        index->add_with_ids(ntotal, data.data(), ids.data());
        faiss::IndexFlatL2 quantizer(dim);
        faiss::IndexIVFPQ* index_cpu = new faiss::IndexIVFPQ(&quantizer, dim, nlist, msub, nbit);
        index->copyTo(index_cpu);
        EXPECT_EQ(index_cpu->d, dim);
        EXPECT_EQ(index_cpu->metric_type, type);
        EXPECT_EQ(index_cpu->is_trained, true);
        EXPECT_EQ(index_cpu->pq.M, msub);
        EXPECT_EQ(index_cpu->pq.nbits, nbit);
        delete index_cpu;
        delete index;
    } catch(std::exception &e) {
        msg = e.what();
    }
}

TEST(TestAscendIndexIVFPQ, copyFrom)
{
    int dim = 128;
    int nlist = 1024;
    int msub = 4;
    int nbit = 8;
    int ntotal = 10000000;
    int nprobe = 64;
    int nq = 10;
    int k = 10;

    std::string msg = "";
    faiss::MetricType type = faiss::METRIC_L2;
    faiss::ascend::AscendIndexIVFPQ* index = nullptr;
    faiss::ascend::AscendIndexIVFPQConfig conf({ 0 });
    std::vector<float> data(ntotal * dim);
    std::vector<int> ids(ntotal);
    std::vector<float> dist(nq * k, 0.0);
    std::vector<faiss::idx_t> label(nq * k, 0);
    generateData(data.data(), ntotal, dim);
    for (int i = 0; i < ntotal; i++) {
        ids[i] = i;
    }
    conf.useKmeansPP = false;
    int trainNum = ntotal > nlist * 40 ? nlist * 40 : ntotal;
    try {
        index = new faiss::ascend::AscendIndexIVFPQ(dim, type, nlist, msub, nbit, conf);
        faiss::IndexFlatL2 quantizer(dim);
        faiss::IndexIVFPQ* index_cpu = new faiss::IndexIVFPQ(&quantizer, dim, nlist, msub, nbit);
        index_cpu->train(trainNum, data.data());
        index_cpu->add(ntotal, data.data());
        index->copyFrom(index_cpu);
        EXPECT_EQ(index->ntotal, ntotal);
        delete index_cpu;
        delete index;
    } catch(std::exception &e) {
        msg = e.what();
    }
}

TEST(TestAscendIndexIVFPQ, remove)
{
    int dim = 128;
    int nlist = 1024;
    int msub = 4;
    int nbit = 8;
    int ntotal = 10000000;
    int nprobe = 64;
    int nq = 1;
    int k = 10;

    std::string msg = "";
    faiss::MetricType type = faiss::METRIC_L2;
    faiss::ascend::AscendIndexIVFPQ* index = nullptr;
    faiss::ascend::AscendIndexIVFPQConfig conf({ 0 });
    std::vector<float> data(ntotal * dim);
    std::vector<int> ids(ntotal);
    std::vector<float> dist(nq * k, 0.0);
    std::vector<faiss::idx_t> label(nq * k, 0);
    std::vector<float> dist_new(nq * k, 0.0);
    std::vector<faiss::idx_t> label_new(nq * k, 0);
    generateData(data.data(), ntotal, dim);
    for (int i = 0; i < ntotal; i++) {
        ids[i] = i;
    }
    conf.useKmeansPP = false;
    int trainNum = ntotal > nlist * 40 ? nlist * 40 : ntotal;
    try {
        index = new faiss::ascend::AscendIndexIVFPQ(dim, type, nlist, msub, nbit, conf);
        index->verbose = true;
        index->setNumProbes(nprobe);
        index->train(trainNum, data.data());
        index->add_with_ids(ntotal, data.data(), ids.data());
        index->search(1, data.data(), k, dist.data(), label.data());
        std::vector<faiss::idx_t> deleteIds(1);
        deleteIds[0] = 0;
        index->remove_ids(1, deleteIds.data());
        index->search(1, data.data(), k, dist_new.data(), label_new.data());
        EXPECT_NE(label[0], label_new[0]);
        delete index;
    } catch(std::exception &e) {
        msg = e.what();
    }
}

TEST(TestAscendIndexIVFPQ, update)
{
    int dim = 128;
    int nlist = 1024;
    int msub = 4;
    int nbit = 8;
    int ntotal = 10000000;
    int nprobe = 64;
    int nq = 1;
    int k = 10;

    std::string msg = "";
    faiss::MetricType type = faiss::METRIC_L2;
    faiss::ascend::AscendIndexIVFPQ* index = nullptr;
    faiss::ascend::AscendIndexIVFPQConfig conf({ 0 });
    std::vector<float> data(ntotal * dim);
    std::vector<int> ids(ntotal);
    std::vector<float> dist(nq * k, 0.0);
    std::vector<faiss::idx_t> label(nq * k, 0);
    std::vector<float> dist_new(nq * k, 0.0);
    std::vector<faiss::idx_t> label_new(nq * k, 0);
    generateData(data.data(), ntotal, dim);
    for (int i = 0; i < ntotal; i++) {
        ids[i] = 0;
    }
    conf.useKmeansPP = false;
    int trainNum = ntotal > nlist * 40 ? nlist * 40 : ntotal;
    try {
        index = new faiss::ascend::AscendIndexIVFPQ(dim, type, nlist, msub, nbit, conf);
        index->verbose = true;
        index->setNumProbes(nprobe);
        index->train(trainNum, data.data());
        index->add_with_ids(ntotal, data.data(), ids.data());
        std::vector<faiss::idx_t> updateIds(1);
        updateIds[0] = 1;
        std::vector<uint64_t> noExistIds(1);
        noExistIds[0] = 0;
        noExistIds = index->update(1, data.data(), updateIds.data());
        EXPECT_NE(noExistIds[0], 1);
        updateIds[0] = 10000001;
        noExistIds[0] = 0;
        noExistIds = index->update(1, data.data(), updateIds.data());
        EXPECT_EQ(noExistIds[0], 10000001);
        delete index;
    } catch(std::exception &e) {
        msg = e.what();
    }
}

}
