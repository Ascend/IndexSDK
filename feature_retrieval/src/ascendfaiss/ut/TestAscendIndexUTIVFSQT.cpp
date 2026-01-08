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
#include <functional>
#include <cstdlib>
#include <memory>
#include <thread>
#include <mutex>
#include <gtest/gtest.h>
#include <mockcpp/mockcpp.hpp>
#include "Common.h"
#include "faiss/ascend/AscendCloner.h"
#include "faiss/ascend/AscendIndexQuantizerImpl.h"
#include "faiss/ascend/custom/AscendIndexFlatATInt8.h"
#include "faiss/ascend/custom/AscendIndexIVFSQT.h"
#include "faiss/ascend/custom/impl/AscendIndexIVFSQCImpl.h"
#include "faiss/ascend/custom/impl/AscendIndexIVFSQFuzzyImpl.h"
#include "faiss/ascend/custom/impl/AscendIndexIVFSQTImpl.h"
#include "faiss/ascend/impl/AscendNNInferenceImpl.h"
#include "faiss/ascendhost/include/impl/AscendClusteringImpl.h"
#include "faiss/impl/ScalarQuantizer.h"
#include "faiss/index_io.h"

namespace ascend {

void StubIvfsqcTrain(faiss::ascend::AscendIndexIVFSQCImpl *index, faiss::idx_t, const float *)
{
    index->intf_->is_trained = true;
}

using namespace std;

const size_t DIM_IN = 256;
const size_t DIM_OUT = 64;
const size_t NLIST = 1024;
const size_t NTOTAL = NLIST;
const int MAX_CLUS_POINTS = 16384;
const int MAX_TRAIN_NUM = 7000000;

void TestSearch(std::vector<int> &deviceList)
{
    std::vector<float> randData(DIM_IN * NTOTAL);
    faiss::ascend::AscendIndexIVFSQTConfig conf(deviceList);
    size_t nlist = 1024;
    faiss::ascend::AscendIndexIVFSQT index(DIM_IN, DIM_OUT, nlist,
        faiss::ScalarQuantizer::QuantizerType::QT_8bit, faiss::MetricType::METRIC_INNER_PRODUCT, conf);
    index.verbose = true;
    int fuzzyK = 3;
    float threshold = 1.5;
    index.setFuzzyK(fuzzyK);
    index.setThreshold(threshold);
    index.impl_->deviceNpuType = 0;

    index.train(NTOTAL, randData.data());
    index.add(NTOTAL, randData.data());
    index.update();

    size_t k = 2;
    size_t queryNum = 1;
    std::vector<float> dist(k * queryNum);
    std::vector<long> indices(k * queryNum);
    index.search(queryNum, randData.data(), k, dist.data(), indices.data());
    const char *filename = "IVFSQT.faiss";
    faiss::Index *cpuIndex = faiss::ascend::index_ascend_to_cpu(&index);
    ASSERT_FALSE(cpuIndex == nullptr);
    faiss::ascend::AscendIndexIVFSQT *realIndex
        = dynamic_cast<faiss::ascend::AscendIndexIVFSQT *>(faiss::ascend::index_cpu_to_ascend(deviceList, cpuIndex));
    delete realIndex;
    delete cpuIndex;
}

TEST(TestAscendIndexUTIVFSQT, train_invalid_input)
{
    faiss::ascend::AscendIndexIVFSQTConfig conf({ 0, 1, 2, 3 });
    faiss::ascend::AscendIndexIVFSQT index(DIM_IN, DIM_OUT, NLIST,
        faiss::ScalarQuantizer::QuantizerType::QT_8bit, faiss::MetricType::METRIC_INNER_PRODUCT, conf);
    std::string msg;

    // 输入x为空
    try {
        index.train(NTOTAL, nullptr);
    } catch(std::exception &e) {
        msg = e.what();
    }
    EXPECT_TRUE(msg.find("Invalid parameter, x can not be nullptr") != std::string::npos);

    std::vector<float> randData(DIM_IN * NTOTAL);
    // 输入n为NLIST - 1
    try {
        index.train(NLIST - 1, randData.data());
    } catch(std::exception &e) {
        msg = e.what();
    }
    EXPECT_TRUE(msg.find("Invalid n") != std::string::npos);

    // 输入x大于MAX_TRAIN_NUM
    try {
        index.train(MAX_TRAIN_NUM + 1, randData.data());
    } catch(std::exception &e) {
        msg = e.what();
    }
    EXPECT_TRUE(msg.find("Invalid n") != std::string::npos);

    // 已经traind了，再train
    index.impl_->intf_->is_trained = true;
    index.impl_->pQuantizerImpl->cpuQuantizer->is_trained = false;
    try {
        index.train(NTOTAL, randData.data());
    } catch(std::exception &e) {
        msg = e.what();
    }
    EXPECT_TRUE(msg.find("cpuQuantizer must be trained") != std::string::npos);

    index.impl_->pQuantizerImpl->cpuQuantizer->is_trained = true;
    try {
        index.train(NTOTAL, randData.data());
    } catch(std::exception &e) {
        msg = e.what();
    }
    EXPECT_TRUE(msg.find("cpuQuantizer.size must be nlist") != std::string::npos);

    // 已经update了，再train
    index.impl_->intf_->is_trained = false;
    index.impl_->isUpdated = true;
    try {
        index.train(NTOTAL, randData.data());
    } catch(std::exception &e) {
        msg = e.what();
    }
    EXPECT_TRUE(msg.find("train after Update is not supported") != std::string::npos);
}

// preciseMemControl为true
TEST(TestAscendIndexUTIVFSQT, train_preciseMemControl_is_true)
{
    MOCKER_CPP(&faiss::ascend::AscendIndexIVFSQCImpl::train).stubs()
                    .will(invoke(StubIvfsqcTrain));
    faiss::ScalarQuantizer ScalarQuantizerObj;
    MOCKER_CPP_VIRTUAL(ScalarQuantizerObj, &faiss::ScalarQuantizer::train).stubs();

    faiss::ascend::AscendIndexIVFSQTConfig conf({ 0, 1, 2, 3 });
    faiss::ascend::AscendIndexIVFSQT index(DIM_IN, DIM_OUT, NLIST,
        faiss::ScalarQuantizer::QuantizerType::QT_8bit, faiss::MetricType::METRIC_INNER_PRODUCT, conf);

    std::vector<float> randData(DIM_IN * NTOTAL);
    index.impl_->preciseMemControl = true;
    index.train(NTOTAL, randData.data());
    const size_t defaultThreshold = 1000000000;
    EXPECT_EQ(index.impl_->numHostThreshold, defaultThreshold);

    // mockcpp 需要显示调用该函数来恢复打桩
    GlobalMockObject::verify();
}

// useKmeansPP设置为true，deviceNpuType为1
TEST(TestAscendIndexUTIVFSQT, DISABLED_all_310P)
{
    std::vector<float> randData(DIM_IN * NTOTAL);
    FeatureGenerator(randData);

    faiss::ascend::AscendIndexIVFSQTConfig conf({ 0, 1, 2, 3 });
    conf.useKmeansPP = true;
    faiss::ascend::AscendIndexIVFSQT index(DIM_IN, DIM_OUT, NLIST,
        faiss::ScalarQuantizer::QuantizerType::QT_8bit, faiss::MetricType::METRIC_INNER_PRODUCT, conf);

    // stub deviceNpuType is 1
    index.impl_->deviceNpuType = 1;
    index.verbose = true;

    MOCKER_CPP(&faiss::ascend::AscendClusteringImpl::distributedTrain).stubs();
    faiss::ScalarQuantizer ScalarQuantizerObj;
    MOCKER_CPP_VIRTUAL(ScalarQuantizerObj, &faiss::ScalarQuantizer::train).stubs();
    MOCKER_CPP_VIRTUAL(ScalarQuantizerObj, &faiss::ScalarQuantizer::compute_codes).stubs();
    MOCKER_CPP_VIRTUAL(ScalarQuantizerObj, &faiss::ScalarQuantizer::decode).stubs();
    MOCKER_CPP(&faiss::ascend::AscendIndexIVFSQTImpl::trainHostBaseSubCluster).stubs();

    index.train(NTOTAL, randData.data());
    index.add(NTOTAL, randData.data());
    index.update();

    size_t k = 100;
    size_t queryNum = 1000;
    std::vector<float> dist(k * queryNum);
    std::vector<long> indices(k * queryNum);
    index.search(queryNum, randData.data(), k, dist.data(), indices.data());

    const char *filename = "IVFSQT.faiss";
    faiss::Index *cpuIndex = faiss::ascend::index_ascend_to_cpu(&index);
    ASSERT_FALSE(cpuIndex == nullptr);
    write_index(cpuIndex, filename);
    delete cpuIndex;

    faiss::Index *initIndex = faiss::read_index(filename);
    ASSERT_FALSE(initIndex == nullptr);

    auto isq = dynamic_cast<const faiss::IndexIVFScalarQuantizer *>(initIndex);
    index.copyFrom(isq);

    EXPECT_EQ(remove(filename), 0);

    delete initIndex;
    // mockcpp 需要显示调用该函数来恢复打桩
    GlobalMockObject::verify();
}

// useKmeansPP设置为false，deviceNpuType为0
TEST(TestAscendIndexUTIVFSQT, train_deviceNpuType_is_0)
{
    MOCKER_CPP(&faiss::ascend::AscendIndexIVFSQCImpl::train).stubs()
                .will(invoke(StubIvfsqcTrain));
    faiss::ScalarQuantizer ScalarQuantizerObj;
    MOCKER_CPP_VIRTUAL(ScalarQuantizerObj, &faiss::ScalarQuantizer::train).stubs();
    std::vector<float> randData(DIM_IN * NTOTAL);
    faiss::ascend::AscendIndexIVFSQTConfig conf({ 0, 1, 2, 3 });
    faiss::ascend::AscendIndexIVFSQT index(DIM_IN, DIM_OUT, NLIST,
        faiss::ScalarQuantizer::QuantizerType::QT_8bit, faiss::MetricType::METRIC_INNER_PRODUCT, conf);
    index.train(NTOTAL, randData.data());


    // mockcpp 需要显示调用该函数来恢复打桩
    GlobalMockObject::verify();
}

// 测试310的调用流程
TEST(TestAscendIndexUTIVFSQT, all_310)
{
    std::vector<int> deviceList {0, 1, 2, 3};
    TestSearch(deviceList);
}

TEST(TestAscendIndexUTIVFSQT, test_setLowerBound_setMergeThres)
{
    faiss::ascend::AscendIndexIVFSQTConfig conf({ 0, 1, 2, 3 });
    faiss::ascend::AscendIndexIVFSQT index(DIM_IN, DIM_OUT, NLIST,
        faiss::ScalarQuantizer::QuantizerType::QT_8bit, faiss::MetricType::METRIC_INNER_PRODUCT, conf);

    int defaultLowerBound = 32;
    index.setLowerBound(MAX_CLUS_POINTS);
    EXPECT_EQ(index.getLowerBound(), defaultLowerBound);

    index.setMergeThres(1);
    int defaultMergeThres = 5;
    EXPECT_EQ(index.getMergeThres(), defaultMergeThres);
}

TEST(TestAscendIndexUTIVFSQT, test_setMemoryLimit_setAddTotal_setPreciseMemControl)
{
    faiss::ascend::AscendIndexIVFSQTConfig conf({ 0, 1, 2, 3 });
    faiss::ascend::AscendIndexIVFSQT index(DIM_IN, DIM_OUT, NLIST,
        faiss::ScalarQuantizer::QuantizerType::QT_8bit, faiss::MetricType::METRIC_INNER_PRODUCT, conf);
    std::string msg;
    try {
        index.setMemoryLimit(0.0);
        index.setPreciseMemControl(false);
        size_t addTotalIn = 100000;
        index.setAddTotal(addTotalIn);
    } catch(std::exception &e) {
        msg = e.what();
    }
    EXPECT_TRUE(msg.empty());
}

TEST(TestAscendIndexUTIVFSQT, test_updateTParams)
{
    faiss::ascend::AscendIndexIVFSQTConfig conf({ 0, 1, 2, 3 });
    faiss::ascend::AscendIndexIVFSQT index(DIM_IN, DIM_OUT, NLIST,
        faiss::ScalarQuantizer::QuantizerType::QT_8bit, faiss::MetricType::METRIC_INNER_PRODUCT, conf);
    std::string msg;
    int l2Probe = 16;
    int l3Segment = 36;
    try {
        index.impl_->nprobe = 0;
        index.updateTParams(l2Probe, l3Segment);
    } catch(std::exception &e) {
        msg = e.what();
    }
    EXPECT_TRUE(msg.find("Check nprobe[0] is legal") != std::string::npos) << "msg: " << msg;

    try {
        index.impl_->nprobe = 64;
        index.updateTParams(l2Probe, l3Segment);
    } catch(std::exception &e) {
        msg = e.what();
    }
    EXPECT_TRUE(msg.find("Please Check l2Probe[16] is larger than or equal to nprobe[64]") != std::string::npos) <<
        "msg: " << msg;


    try {
        l2Probe = 64;
        index.updateTParams(l2Probe, l3Segment);
    } catch(std::exception &e) {
        msg = e.what();
    }
    EXPECT_TRUE(msg.find("Please Check l2Probe[64] is less than or equal to l3SegmentNum[36]") != std::string::npos) <<
        "msg: " << msg;

    try {
        l3Segment = 1021;
        index.updateTParams(l2Probe, l3Segment);
    } catch(std::exception &e) {
        msg = e.what();
    }
    EXPECT_TRUE(msg.find("Please Check l3SegmentNum[1021] is legal") != std::string::npos) << "msg: " << msg;

    l3Segment = 72;
    index.updateTParams(l2Probe, l3Segment);
    EXPECT_EQ(index.impl_->l2NProbe, l2Probe);
    EXPECT_EQ(index.impl_->l3SegmentNum, l3Segment);
}

TEST(TestAscendIndexUTIVFSQT, test_construct_and_remove_ids)
{
    faiss::ascend::AscendIndexIVFSQTConfig conf({ 0, 1, 2, 3 });
    bool dummy = false;
    faiss::ascend::AscendIndexIVFSQT index(DIM_IN, DIM_OUT, NLIST,
        faiss::ScalarQuantizer::QuantizerType::QT_8bit, faiss::MetricType::METRIC_INNER_PRODUCT, conf);
    std::string msg;
    try {
        faiss::IDSelectorRange del(0, 1);
        index.remove_ids(del);
    } catch(std::exception &e) {
        msg = e.what();
    }
    EXPECT_TRUE(msg.find("remove_ids() not implemented for this type of index") != std::string::npos);
}

TEST(TestAscendIndexUTIVFSQT, test_reset)
{
    faiss::ascend::AscendIndexIVFSQTConfig conf({ 0, 1, 2, 3 });
    faiss::ascend::AscendIndexIVFSQT index(DIM_IN, DIM_OUT, NLIST,
        faiss::ScalarQuantizer::QuantizerType::QT_8bit, faiss::MetricType::METRIC_INNER_PRODUCT, conf);
    index.reset();
}

TEST(TestAscendIndexUTIVFSQT, test_getMin_getMax)
{
    std::vector<float> randData(DIM_IN * NTOTAL);

    faiss::ascend::AscendIndexIVFSQTConfig conf({ 0 });
    faiss::ascend::AscendIndexIVFSQT index(DIM_IN, DIM_OUT, NLIST,
        faiss::ScalarQuantizer::QuantizerType::QT_8bit, faiss::MetricType::METRIC_INNER_PRODUCT, conf);

    faiss::ScalarQuantizer ScalarQuantizerObj;
    MOCKER_CPP_VIRTUAL(ScalarQuantizerObj, &faiss::ScalarQuantizer::train).stubs();
    MOCKER_CPP_VIRTUAL(ScalarQuantizerObj, &faiss::ScalarQuantizer::compute_codes).stubs();
    MOCKER_CPP_VIRTUAL(ScalarQuantizerObj, &faiss::ScalarQuantizer::decode).stubs();

    float max = 1.0;
    float min = 0.0;
    randData[0] = max;
    randData[1] = min;
    index.train(NTOTAL, randData.data());
    EXPECT_FLOAT_EQ(index.getQMin(), min);
    EXPECT_FLOAT_EQ(index.getQMax(), max);

    // mockcpp 需要显示调用该函数来恢复打桩
    GlobalMockObject::verify();
}

TEST(TestAscendIndexUTIVFSQT, test_updateTParams_resetop)
{
    MOCKER_CPP(&ascend::IndexIVFSQTIPAicpu::resetSubcentersDistOp).expects(once()).will(returnValue(0));
    MOCKER_CPP(&ascend::IndexIVFSQTIPAicpu::resetSqXDistOp).expects(exactly(2)).will(returnValue(0));
    MOCKER_CPP(&ascend::IndexIVFSQTIPAicpu::resetTopkIvfsqtL1Op).expects(once()).will(returnValue(0));
    MOCKER_CPP(&ascend::IndexIVFSQTIPAicpu::resetTopkIvfsqtL2Op).expects(once()).will(returnValue(0));
    MOCKER_CPP(&ascend::IndexIVFSQTIPAicpu::resetTopkIvfFuzzyOp).expects(once()).will(returnValue(0));
    MOCKER_CPP(&ascend::IndexIVFSQTIPAicpu::initL1TopkAttrs).expects(once()).will(returnValue(0));
    MOCKER_CPP(&ascend::IndexIVFSQTIPAicpu::initL2TopkAttrs).expects(exactly(2)).will(returnValue(0));

    faiss::ascend::AscendIndexIVFSQTConfig conf({ 0 });
    faiss::ascend::AscendIndexIVFSQT index(DIM_IN, DIM_OUT, NLIST,
        faiss::ScalarQuantizer::QuantizerType::QT_8bit, faiss::MetricType::METRIC_INNER_PRODUCT, conf);

    index.updateTParams(200, 960);

    GlobalMockObject::verify();
}

TEST(TestAscendIndexUTIVFSQT, test_setNumProbes_resetop)
{
    MOCKER_CPP(&ascend::IndexIVFSQTIPAicpu::resetSubcentersDistOp).expects(exactly(2)).will(returnValue(0));
    MOCKER_CPP(&ascend::IndexIVFSQTIPAicpu::resetSqXDistOp).expects(once()).will(returnValue(0));
    MOCKER_CPP(&ascend::IndexIVFSQTIPAicpu::resetTopkIvfsqtL1Op).expects(once()).will(returnValue(0));
    MOCKER_CPP(&ascend::IndexIVFSQTIPAicpu::resetTopkIvfsqtL2Op).expects(exactly(2)).will(returnValue(0));
    MOCKER_CPP(&ascend::IndexIVFSQTIPAicpu::resetTopkIvfFuzzyOp).expects(once()).will(returnValue(0));
    MOCKER_CPP(&ascend::IndexIVFSQTIPAicpu::initL1TopkAttrs).expects(exactly(2)).will(returnValue(0));
    MOCKER_CPP(&ascend::IndexIVFSQTIPAicpu::initL2TopkAttrs).expects(exactly(2)).will(returnValue(0));

    faiss::ascend::AscendIndexIVFSQTConfig conf({ 0 });
    faiss::ascend::AscendIndexIVFSQT index(DIM_IN, DIM_OUT, NLIST,
        faiss::ScalarQuantizer::QuantizerType::QT_8bit, faiss::MetricType::METRIC_INNER_PRODUCT, conf);

    index.setNumProbes(48);

    GlobalMockObject::verify();
}

TEST(TestAscendIndexUTIVFSQT, test_setSearchParams_resetop)
{
    MOCKER_CPP(&ascend::IndexIVFSQTIPAicpu::resetSubcentersDistOp).expects(exactly(2)).will(returnValue(0));
    MOCKER_CPP(&ascend::IndexIVFSQTIPAicpu::resetSqXDistOp).expects(exactly(2)).will(returnValue(0));
    MOCKER_CPP(&ascend::IndexIVFSQTIPAicpu::resetTopkIvfsqtL1Op).expects(once()).will(returnValue(0));
    MOCKER_CPP(&ascend::IndexIVFSQTIPAicpu::resetTopkIvfsqtL2Op).expects(exactly(2)).will(returnValue(0));
    MOCKER_CPP(&ascend::IndexIVFSQTIPAicpu::resetTopkIvfFuzzyOp).expects(once()).will(returnValue(0));
    MOCKER_CPP(&ascend::IndexIVFSQTIPAicpu::initL1TopkAttrs).expects(exactly(2)).will(returnValue(0));
    MOCKER_CPP(&ascend::IndexIVFSQTIPAicpu::initL2TopkAttrs).expects(exactly(2)).will(returnValue(0));

    faiss::ascend::AscendIndexIVFSQTConfig conf({ 0 });
    faiss::ascend::AscendIndexIVFSQT index(DIM_IN, DIM_OUT, NLIST,
        faiss::ScalarQuantizer::QuantizerType::QT_8bit, faiss::MetricType::METRIC_INNER_PRODUCT, conf);

    index.setSearchParams(48, 200, 960);

    GlobalMockObject::verify();
}

TEST(TestAscendIndexUTIVFSQT, test_setNumProbes_range)
{
    faiss::ascend::AscendIndexIVFSQTConfig conf({ 0 });
    faiss::ascend::AscendIndexIVFSQT index(DIM_IN, DIM_OUT, NLIST,
        faiss::ScalarQuantizer::QuantizerType::QT_8bit, faiss::MetricType::METRIC_INNER_PRODUCT, conf);

    index.setNumProbes(16);
    EXPECT_EQ(index.impl_->nprobe, 16);
    EXPECT_EQ(index.impl_->getActualIndex(0)->nprobe, 16);

    string msg;
    try {
        index.setNumProbes(1);
    } catch (std::exception &e) {
        msg = e.what();
    }
    EXPECT_TRUE(msg.find("Please Check nprobe[1] is legal") != string::npos) << "msg: " << msg;

    index.setSearchParams(8, 60, 1020);
    try {
        index.setNumProbes(64);
    } catch (std::exception &e) {
        msg = e.what();
    }
    EXPECT_TRUE(msg.find("Please Check l2Probe[60] is larger than or equal to nprobe[64]") != string::npos) <<
        "msg: " << msg;

    index.setSearchParams(60, 520, 1020);
    try {
        index.setNumProbes(8);
    } catch (std::exception &e) {
        msg = e.what();
    }
    EXPECT_TRUE(msg.find("Please Check l2Probe[520] is less than or equal to nprobe * subcenterNum [512]") !=
        string::npos) << "msg: " << msg;
}

TEST(TestAscendIndexUTIVFSQT, test_setSearchParams_range)
{
    faiss::ascend::AscendIndexIVFSQTConfig conf({ 0 });
    faiss::ascend::AscendIndexIVFSQT index(DIM_IN, DIM_OUT, NLIST,
        faiss::ScalarQuantizer::QuantizerType::QT_8bit, faiss::MetricType::METRIC_INNER_PRODUCT, conf);

    index.setSearchParams(8, 60, 1020);
    EXPECT_EQ(index.impl_->nprobe, 8);
    EXPECT_EQ(index.impl_->l2NProbe, 60);
    EXPECT_EQ(index.impl_->l3SegmentNum, 1020);
    EXPECT_EQ(index.impl_->getActualIndex(0)->nprobe, 8);
    EXPECT_EQ(index.impl_->getActualIndex(0)->l2NProbe, 60);
    EXPECT_EQ(index.impl_->getActualIndex(0)->l3SegmentNum, 1020);

    string msg;
    try {
        index.setSearchParams(2, 60, 1020);
    } catch (std::exception &e) {
        msg = e.what();
    }
    EXPECT_TRUE(msg.find("Please Check nprobe[2] is legal") != string::npos) << "msg: " << msg;

    try {
        index.setSearchParams(8, 60, 777);
    } catch (std::exception &e) {
        msg = e.what();
    }
    EXPECT_TRUE(msg.find("Please Check l3SegmentNum[777] is legal") != string::npos) << "msg: " << msg;

    try {
        index.setSearchParams(8, 7, 960);
    } catch (std::exception &e) {
        msg = e.what();
    }
    EXPECT_TRUE(msg.find("Please Check l2Probe[7] is larger than or equal to nprobe[8]") != string::npos) <<
        "msg: " << msg;

    try {
        index.setSearchParams(8, 961, 960);
    } catch (std::exception &e) {
        msg = e.what();
    }
    EXPECT_TRUE(msg.find("Please Check l2Probe[961] is less than or equal to l3SegmentNum[960]") !=
        string::npos) << "msg: " << msg;

    try {
        index.setSearchParams(8, 530, 840);
    } catch (std::exception &e) {
        msg = e.what();
    }
    EXPECT_TRUE(msg.find("Please Check l2Probe[530] is less than or equal to nprobe * subcenterNum [512]") !=
        string::npos) << "msg: " << msg;
}

TEST(TestAscendIndexUTIVFSQT, test_deviceList_range)
{
    std::vector<int> deviceList;
    for (int i = 0; i < 21; i++) {
        deviceList.emplace_back(i);
    }

    string msg;
    try {
        TestSearch(deviceList);
    } catch (std::exception &e) {
        msg = e.what();
    }
    string expectStr = "deviceCnt " + to_string(deviceList.size()) + " must <= 20, please check it";
    EXPECT_TRUE(msg.find(expectStr) != string::npos) << "msg: " << msg;
}

}