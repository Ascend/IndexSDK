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

#include <iostream>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <faiss/IndexFlat.h>
#include "Common.h"
#include "common/utils/SocUtils.h"
#include "mockcpp/mockcpp.hpp"
#include "faiss/IndexScalarQuantizer.h"
#include "ascendhost/include/index/IndexIVFSQL2Aicpu.h"
#include "ascend/impl/AscendIndexIVFSQImpl.h"
#include "ascend/impl/AscendIndexIVFImpl.h"
#include "index/IndexIVFSQIPAicpu.h"
#include "faiss/ascend/AscendIndexIVFSQ.h"
#include "faiss/ascend/AscendIndexQuantizerImpl.h"

namespace ascend {

#define TRY_EXCEPTION(code) try {code;  ASSERT_NE(0, 0);} catch (std::exception &e) { }

void StubIvfsqTrainQuantizer(faiss::ascend::AscendIndexIVFSQImpl *index, idx_t n, const float *x)
{
    index->pQuantizerImpl->cpuQuantizer->is_trained = true;
    index->pQuantizerImpl->cpuQuantizer->add(index->nlist, x);
    index->pQuantizerImpl->cpuQuantizer->ntotal == index->nlist;
    index->ivfsqConfig.useKmeansPP = false;
    index->updateDeviceCoarseCenter();
    index->trainResidualQuantizer(n, x);
    index->intf_->is_trained = true;
}

TEST(TestAscendIndexIVFSQ, All)
{
    MOCKER_CPP(&faiss::ascend::AscendIndexIVFSQImpl::train).stubs().will(invoke(StubIvfsqTrainQuantizer));
    auto utIvfsqForChips = [](faiss::MetricType type) {
        int dim = 64;
        int ntotal = 40000;
        int ncentroids = 1024;
        int nprobe = 32;
        std::vector<float> data(dim * ntotal);
        FeatureGenerator(data);
        faiss::ascend::SocUtils::SocType socBak = faiss::ascend::SocUtils::GetInstance().socAttr.socType;
        for (int chip = 0; chip < 2; chip++) {
            faiss::ascend::SocUtils::GetInstance().socAttr.socType = chip ?
                faiss::ascend::SocUtils::SocType::SOC_310 : faiss::ascend::SocUtils::SocType::SOC_310P;
            faiss::ascend::AscendIndexIVFSQConfig conf({ 0 });
            faiss::ascend::AscendIndexIVFSQ index(dim, ncentroids, faiss::ScalarQuantizer::QuantizerType::QT_8bit,
                type, true, conf);
            index.verbose = false;
            index.setNumProbes(nprobe);
            index.train(ntotal, data.data());
            for (int i = 0; i < ncentroids; i++) {
                int len = index.getListLength(i);
                ASSERT_EQ(len, 0);
            }
            index.add(ntotal, data.data());
            faiss::IndexIVFScalarQuantizer faissindex;
            std::vector<std::vector<uint8_t>> codes(ncentroids, std::vector<uint8_t>());
            std::vector<std::vector<faiss::ascend::ascend_idx_t>> indices(ncentroids,
                std::vector<faiss::ascend::ascend_idx_t>());
            for (int i = 0; i < ncentroids; i++) {
                index.getListCodesAndIds(i, codes[i], indices[i]);
                ASSERT_EQ(codes[i].size(), indices[i].size() * index.d);
            }
            int n = 1;
            int k = 10;
            for (int i = 3; i < 10; i++) {
                std::vector<float> dist(k, 0);
                std::vector<faiss::idx_t> label(k, 0);
                index.search(n, data.data() + i * dim, k, dist.data(), label.data());
                faiss::idx_t assign;
                index.assign(1, data.data() + i * dim, &assign);
            }
        }
        faiss::ascend::SocUtils::GetInstance().socAttr.socType = socBak;
    };
    utIvfsqForChips(faiss::METRIC_INNER_PRODUCT);
    utIvfsqForChips(faiss::METRIC_L2);
    GlobalMockObject::verify();
}

TEST(TestAscendIndexIVFSQ, reset)
{
    int dim = 64;
    int ntotal = 40000;
    int ncentroids = 1024;
    int nprobe = 32;
    std::vector<float> data(dim * ntotal);
    FeatureGenerator(data);
    MOCKER_CPP(&faiss::ascend::AscendIndexIVFSQImpl::train).stubs()
                                        .will(invoke(StubIvfsqTrainQuantizer));
    faiss::ascend::AscendIndexIVFSQConfig conf({ 0 });
    faiss::ascend::AscendIndexIVFSQ index(dim, ncentroids, faiss::ScalarQuantizer::QuantizerType::QT_8bit,
        faiss::METRIC_L2, true, conf);
    index.verbose = false;
    index.setNumProbes(nprobe);
    EXPECT_EQ(index.getNumProbes(), nprobe);
    index.train(ntotal, data.data());
    for (int i = 0; i < ncentroids; i++) {
        int len = index.getListLength(i);
        ASSERT_EQ(len, 0);
    }
    index.add(ntotal, data.data());
    {
        int tmpTotal = 0;
        for (int i = 0; i < ncentroids; i++) {
            tmpTotal += index.getListLength(i);
        }
        ASSERT_EQ(tmpTotal, ntotal);
    }
    index.reset();
    for (int i = 0; i < ncentroids; i++) {
        int len = index.getListLength(i);
        ASSERT_EQ(len, 0);
    }
    GlobalMockObject::verify();
}

TEST(TestAscendIndexIVFSQ, remove)
{
    int dim = 64;
    int64_t ntotal = 40000;
    int ncentroids = 1024;
    std::vector<float> data(dim * ntotal);
    FeatureGenerator(data);
    MOCKER_CPP(&faiss::ascend::AscendIndexIVFSQImpl::train).stubs()
                                        .will(invoke(StubIvfsqTrainQuantizer));
    faiss::ascend::AscendIndexIVFSQConfig conf({ 0 });
    faiss::ascend::AscendIndexIVFSQ index(dim, ncentroids, faiss::ScalarQuantizer::QuantizerType::QT_8bit,
        faiss::METRIC_L2, true, conf);
    index.verbose = false;
    index.train(ntotal, data.data());
    index.add(ntotal, data.data());
    faiss::IDSelectorRange del(0, 2);
    int64_t rmCnt = 0;
    for (int i = 0; i < ncentroids; i++) {
        std::vector<uint8_t> code;
        std::vector<faiss::ascend::ascend_idx_t> ids;

        index.getListCodesAndIds(i, code, ids);
        for (size_t k = 0; k < ids.size(); k++) {
            rmCnt += del.is_member((int64_t)ids[k]) ? 1 : 0;
        }
    }
    int64_t rmedCnt = index.remove_ids(del);
    ASSERT_EQ(rmedCnt, rmCnt);
    ASSERT_EQ(index.ntotal, (ntotal - rmedCnt));
    int64_t tmpTotal = 0;
    for (int i = 0; i < ncentroids; i++) {
        std::vector<uint8_t> code;
        std::vector<faiss::ascend::ascend_idx_t> ids;
        index.getListCodesAndIds(i, code, ids);
        ASSERT_EQ(code.size(), ids.size() * index.d);
        for (size_t k = 0; k < ids.size(); k++) {
            ASSERT_FALSE(del.is_member((int64_t)ids[k])) << "failure index:" << i;
        }
        tmpTotal += index.getListLength(i);
    }
    EXPECT_EQ(tmpTotal, (ntotal - rmedCnt));
    GlobalMockObject::verify();
}

void ivfsqSearch(faiss::MetricType type)
{
    int dim = 64;
    int ntotal = 40000;
    int ncentroids = 1024;
    int nprobe = 32;
    std::vector<float> data(dim * ntotal);
    FeatureGenerator(data);
    faiss::ascend::AscendIndexIVFSQConfig conf({ 0 });
    conf.useKmeansPP = true;
    std::string msg = "";
    try {
        faiss::ascend::AscendIndexIVFSQ index(1, ncentroids, faiss::ScalarQuantizer::QuantizerType::QT_8bit,
            type, true, conf);
    } catch(std::exception &e) {
        msg = e.what();
    }
    EXPECT_TRUE(msg.find("Invalid number of dimensions") != std::string::npos);
    try {
        faiss::ascend::AscendIndexIVFSQ index(dim, ncentroids, faiss::ScalarQuantizer::QuantizerType::QT_8bit,
            faiss::METRIC_L1, true, conf);
    } catch(std::exception &e) {
        msg = e.what();
    }
    EXPECT_TRUE(msg.find("Unsupported metric type") != std::string::npos);
    faiss::ascend::AscendIndexIVFSQ index(dim, ncentroids, faiss::ScalarQuantizer::QuantizerType::QT_8bit,
        type, true, conf);
    index.verbose = false;
    index.setNumProbes(nprobe);
    EXPECT_EQ(index.getNumProbes(), nprobe);
    index.train(ntotal, data.data());
    for (int i = 0; i < ncentroids; i++) {
        int len = index.getListLength(i);
        ASSERT_EQ(len, 0);
    }
    index.add(ntotal, data.data());
    std::vector<int> searchNum = { 2, 8 };
    // Since the qps might have some tricky issue
    for (size_t n = 0; n < searchNum.size(); n++) {
        int k = 100;
        int loopTimes = 2;
        std::vector<float> dist(searchNum[n] * k, 0);
        std::vector<faiss::idx_t> label(searchNum[n] * k, 0);
        for (int i = 0; i < loopTimes; i++) {
            index.search(searchNum[n], data.data(), k, dist.data(), label.data());
        }
    }
}

TEST(TestAscendIndexIVFSQ, search)
{
    ivfsqSearch(faiss::METRIC_L2);
    MOCKER_CPP(&ascend::IndexIVFSQIPAicpu::calMaxBatch).stubs().will(returnValue(2));
    ivfsqSearch(faiss::METRIC_L2);
    ivfsqSearch(faiss::METRIC_INNER_PRODUCT);
    GlobalMockObject::verify();
}

TEST(TestAscendIndexIVFSQ, copyFrom)
{
    size_t dim = 64;
    int ntotal = 40000;
    size_t ncentroids = 1024;
    int nprobe = 32;
    std::vector<float> data(dim * ntotal);
    FeatureGenerator(data);
    MOCKER_CPP(&faiss::ascend::AscendIndexIVFSQImpl::train).stubs()
                                        .will(invoke(StubIvfsqTrainQuantizer));
    faiss::IndexFlatL2 cpuQuantizer(dim);
    faiss::IndexIVFScalarQuantizer sq(&cpuQuantizer, dim, ncentroids, faiss::ScalarQuantizer::QuantizerType::QT_8bit);
    faiss::ascend::AscendIndexIVFSQConfig conf({ 0 });
    faiss::ascend::AscendIndexIVFSQ index(&sq, conf);
    index.setNumProbes(nprobe);
    std::shared_ptr<faiss::ascend::AscendIndexIVFSQImpl> tmp = std::move(index.impl_);
    index.impl_ = nullptr;
    index.impl_ = std::move(tmp);
    index.train(ntotal, data.data());
    TRY_EXCEPTION(faiss::ascend::AscendIndexIVFSQ index1(nullptr, conf));
    index.add(ntotal, data.data());
    faiss::IndexIVFScalarQuantizer faissindex;
    tmp = std::move(index.impl_);
    index.impl_ =  nullptr;
    TRY_EXCEPTION(index.copyTo(&faissindex));
    index.impl_ = std::move(tmp);
    index.train(ntotal, data.data());
    index.copyTo(&faissindex);
    tmp = std::move(index.impl_);
    index.impl_ =  nullptr;
    TRY_EXCEPTION(index.copyFrom(&faissindex));
    index.impl_ = std::move(tmp);
    auto ivf = dynamic_cast<faiss::ArrayInvertedLists*>(faissindex.invlists);
    faissindex.invlists = nullptr;
    TRY_EXCEPTION(index.copyFrom(&faissindex));
    faissindex.invlists = ivf;
    auto nlistTmp = ivf->nlist;
    ivf->nlist = 0;
    TRY_EXCEPTION(index.copyFrom(&faissindex));
    ivf->nlist = nlistTmp;
    std::vector<std::vector<uint8_t>> tmpCodes(0);
    tmpCodes.swap(ivf->codes);
    TRY_EXCEPTION(index.copyFrom(&faissindex));
    tmpCodes.swap(ivf->codes);
    std::vector<std::vector<faiss::idx_t>> tmpIds(0);
    tmpIds.swap(ivf->ids);
    TRY_EXCEPTION(index.copyFrom(&faissindex));
    GlobalMockObject::verify();
}

TEST(TestIndexIVFSQ, reserve)
{
    int dim = 64;
    int ntotal = 40000;
    int ncentroids = 1024;
    int nprobe = 32;
    std::vector<float> data(dim * ntotal);
    size_t resource = 4 * static_cast<size_t>(1024 * 1024 * 1024);
    FeatureGenerator(data);
    ::ascend::IndexIVFSQL2Aicpu index(ncentroids, dim, true, nprobe, resource);
    ASSERT_EQ(APP_ERR_OK, index.reserveMemory(0));
    index.reclaimMemory();
    ASSERT_EQ(APP_ERR_OK, index.reserveMemory(ntotal * dim * sizeof(float)));
    ASSERT_EQ(APP_ERR_OK, index.reserveMemory(0, 0));
    index.reclaimMemory(0);
    ASSERT_EQ(APP_ERR_OK, index.reserveMemory(0, ntotal * dim * sizeof(float) / ncentroids));
}

}
