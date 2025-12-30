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

#include <numeric>
#include <cmath>
#include <random>
#include <gtest/gtest.h>
#include <cstring>
#include <sys/time.h>
#include <faiss/index_io.h>
#include <cstdlib>
#include <mockcpp/mockcpp.hpp>

#include "Common.h"
#include "faiss/ascend/custom/AscendClustering.h"
#include "faiss/ascendhost/include/impl/AscendClusteringImpl.h"
#include "acl.h"

namespace ascend {
const size_t DIM = 256;
const size_t NLIST = 1024;
const size_t NTOTAL = 50000;
const size_t TRAIN_ITER = 16;
const size_t CLUSTERING_MAX_N = 7000000;
const size_t SUB_NLIST = 64;
const int NUM_BUCK_PER_BATCH = 64;
const int SUBCENTER_NUM = 64;
const int LOWER_BOUND = 32;
const int MAX_CLUS_POINTS = 16384;

static int32_t StubAclrtSetDevice(int32_t)
{
    return 1;
}

int32_t StubResetCorrComputeOp(faiss::ascend::AscendClusteringImpl *)
{
    return 1;
}

bool StubAscendOperatorInit(faiss::ascend::AscendClusteringImpl *)
{
    return false;
}

int StubMemcpyS(void *, size_t, const void *, size_t)
{
    return 1;
}

aclError StubAclrtMemcpy(void *, size_t, const void *, size_t, aclrtMemcpyKind)
{
    return ACL_ERROR_INTERNAL_ERROR;
}

TEST(TestAscendIndexUTClustering, Invalid_Construct_Part1)
{
    std::vector<int> deviceList {0};
    faiss::ascend::AscendClusteringConfig conf(deviceList);
    std::string msg;
    try {
        faiss::ascend::AscendClustering cluster(DIM - 1, NLIST, faiss::MetricType::METRIC_INNER_PRODUCT, conf);
    } catch(std::exception &e) {
        msg = e.what();
    }
    EXPECT_TRUE(msg.find("Invalid number of dimensions") != std::string::npos);

    try {
        faiss::ascend::AscendClustering cluster(DIM, 0, faiss::MetricType::METRIC_INNER_PRODUCT, conf);
    } catch(std::exception &e) {
        msg = e.what();
    }
    EXPECT_TRUE(msg.find("number of centroids") != std::string::npos);

    try {
        faiss::ascend::AscendClustering cluster(DIM, NLIST, faiss::MetricType::METRIC_L1, conf);
    } catch(std::exception &e) {
        msg = e.what();
    }
    EXPECT_TRUE(msg.find("Unsupported metric type") != std::string::npos);

    try {
        std::vector<int> deviceList;
        faiss::ascend::AscendClusteringConfig conf(deviceList);
        faiss::ascend::AscendClustering cluster(DIM, NLIST, faiss::MetricType::METRIC_INNER_PRODUCT, conf);
    } catch(std::exception &e) {
        msg = e.what();
    }
    EXPECT_TRUE(msg.find("Only 1 chip supported for Ascend Clustering") != std::string::npos);

    try {
        faiss::ascend::AscendClusteringConfig conf(deviceList, -2); // -2 invalid resource size
        faiss::ascend::AscendClustering cluster(DIM, NLIST, faiss::MetricType::METRIC_INNER_PRODUCT, conf);
    } catch(std::exception &e) {
        msg = e.what();
    }
    EXPECT_TRUE(msg.find("resources should be -1 or in range [0, 4096MB]") != std::string::npos);
}

TEST(TestAscendIndexUTClustering, Invalid_Construct_Part2)
{
    std::vector<int> deviceList {0};
    faiss::ascend::AscendClusteringConfig conf(deviceList);
    std::string msg;
    try {
        std::vector<int> deviceList(1, -1);  // -1 invalid device id
        faiss::ascend::AscendClusteringConfig conf(deviceList);
        faiss::ascend::AscendClustering cluster(DIM, NLIST, faiss::MetricType::METRIC_INNER_PRODUCT, conf);
    } catch(std::exception &e) {
        msg = e.what();
    }
    EXPECT_TRUE(msg.find("Error") != std::string::npos);

    try {
        std::vector<int> deviceList(1, 100000);  // 100000 invalid device id
        faiss::ascend::AscendClusteringConfig conf(deviceList);
        faiss::ascend::AscendClustering cluster(DIM, NLIST, faiss::MetricType::METRIC_INNER_PRODUCT, conf);
    } catch(std::exception &e) {
        msg = e.what();
    }
    EXPECT_TRUE(msg.find("total device") != std::string::npos);

    try {
        MOCKER(aclrtSetDevice).stubs().will(invoke(StubAclrtSetDevice));
        faiss::ascend::AscendClustering cluster(DIM, NLIST, faiss::MetricType::METRIC_INNER_PRODUCT, conf);
    } catch(std::exception &e) {
        msg = e.what();
    }
    EXPECT_TRUE(msg.find("ACL error") != std::string::npos);
    GlobalMockObject::verify();
    
    try {
        MOCKER_CPP(&faiss::ascend::AscendClusteringImpl::resetCorrComputeOp).stubs()
            .will(invoke(StubResetCorrComputeOp));
        faiss::ascend::AscendClustering cluster(DIM, NLIST, faiss::MetricType::METRIC_INNER_PRODUCT, conf);
    } catch(std::exception &e) {
        msg = e.what();
    }
    EXPECT_TRUE(msg.find("Failed to reset corrcompute op") != std::string::npos);
    GlobalMockObject::verify();
}

TEST(TestAscendIndexUTClustering, Add)
{
    std::vector<int> deviceList {0};
    faiss::ascend::AscendClusteringConfig conf(deviceList);
    faiss::ascend::AscendClustering cluster(DIM, NLIST, faiss::MetricType::METRIC_INNER_PRODUCT, conf);
    std::vector<float> randData(DIM * NTOTAL);
    FeatureGenerator(randData);
    std::shared_ptr<float> maxAddData(new float[CLUSTERING_MAX_N * DIM], std::default_delete<float[]>());
    std::string msg;
    try {
        cluster.Add(NTOTAL, nullptr);
    } catch(std::exception &e) {
        msg = e.what();
    }
    EXPECT_TRUE(msg.find("x can not be nullptr") != std::string::npos);

    try {
        cluster.Add(0, randData.data());
    } catch(std::exception &e) {
        msg = e.what();
    }
    EXPECT_TRUE(msg.find("Error") != std::string::npos);

    try {
        cluster.Add(NTOTAL, maxAddData.get());
        cluster.Add(1, randData.data());
    } catch(std::exception &e) {
        msg = e.what();
    }
    EXPECT_TRUE(msg.find("Error") != std::string::npos);

    try {
         // 2048 max dim
        faiss::ascend::AscendClustering cluster(2048, NLIST, faiss::MetricType::METRIC_INNER_PRODUCT, conf);
        cluster.Add(CLUSTERING_MAX_N, maxAddData.get());
    } catch(std::exception &e) {
        msg = e.what();
    }
    EXPECT_TRUE(msg.find("Error") != std::string::npos);
}

TEST(TestAscendIndexUTClustering, DistributedTrain_Part1)
{
    std::vector<int> deviceList {0};
    faiss::ascend::AscendClusteringConfig conf(deviceList);
    faiss::ascend::AscendClustering cluster(DIM, NLIST, faiss::MetricType::METRIC_INNER_PRODUCT, conf);

    std::string msg;
    try {
        cluster.DistributedTrain(TRAIN_ITER, nullptr, deviceList, false);
    } catch(std::exception &e) {
        msg = e.what();
    }
    EXPECT_TRUE(msg.find("distances can not be nullptr") != std::string::npos);

    std::vector<float> centroids(DIM * NLIST);
    try {
        cluster.DistributedTrain(0, centroids.data(), deviceList, false);
    } catch(std::exception &e) {
        msg = e.what();
    }
    EXPECT_TRUE(msg.find("Niter must be in range") != std::string::npos);

    try {
        cluster.DistributedTrain(TRAIN_ITER, centroids.data(), deviceList, false);
    } catch(std::exception &e) {
        msg = e.what();
    }
    EXPECT_TRUE(msg.find("ntotal 0 must be > 0") != std::string::npos);

    std::vector<float> randData(DIM * NTOTAL);
    FeatureGenerator(randData);
    cluster.Add(NLIST - 1, randData.data());
    try {
        cluster.DistributedTrain(TRAIN_ITER, centroids.data(), deviceList, false);
    } catch(std::exception &e) {
        msg = e.what();
    }
    EXPECT_TRUE(msg.find("Number of training points") != std::string::npos);
}

TEST(TestAscendIndexUTClustering, DistributedTrain_Part2)
{
    std::vector<int> deviceList {0};
    faiss::ascend::AscendClusteringConfig conf(deviceList);
    faiss::ascend::AscendClustering cluster(DIM, NLIST, faiss::MetricType::METRIC_INNER_PRODUCT, conf);
    std::string msg;
    std::vector<float> randData(DIM * NTOTAL);
    std::vector<float> centroids(DIM * NLIST);
    cluster.Add(NTOTAL, randData.data());
    MOCKER(aclrtSetDevice).stubs().will(invoke(StubAclrtSetDevice));
    try {
        cluster.DistributedTrain(TRAIN_ITER, centroids.data(), deviceList, false);
    } catch(std::exception &e) {
        msg = e.what();
    }
    EXPECT_TRUE(msg.find("Error") != std::string::npos);
    GlobalMockObject::verify();

    MOCKER(&faiss::ascend::AscendOperator::init).stubs().will(invoke(StubAscendOperatorInit));
    try {
        cluster.DistributedTrain(TRAIN_ITER, centroids.data(), deviceList, false);
    } catch(std::exception &e) {
        msg = e.what();
    }
    GlobalMockObject::verify();

    MOCKER(&memcpy_s).stubs().will(invoke(StubMemcpyS));
    try {
        cluster.DistributedTrain(TRAIN_ITER, centroids.data(), deviceList, false);
    } catch(std::exception &e) {
        msg = e.what();
    }
    EXPECT_TRUE(msg.find("Failed to copy to centrodata") != std::string::npos);
    GlobalMockObject::verify();

    MOCKER(&aclrtMemcpy).stubs().will(invoke(StubAclrtMemcpy));
    try {
        cluster.DistributedTrain(TRAIN_ITER, centroids.data(), deviceList, false);
    } catch(std::exception &e) {
        msg = e.what();
    }
    EXPECT_TRUE(msg.find("Failed to copy codes to device") != std::string::npos);
    GlobalMockObject::verify();

    cluster.DistributedTrain(TRAIN_ITER, centroids.data(), deviceList, false);
    cluster.verbose = true;
    cluster.DistributedTrain(TRAIN_ITER, centroids.data(), deviceList, true);
}

TEST(TestAscendIndexUTClustering, Train)
{
    std::vector<int> deviceList {0};
    faiss::ascend::AscendClusteringConfig conf(deviceList);
    faiss::ascend::AscendClustering cluster(DIM, NLIST, faiss::MetricType::METRIC_INNER_PRODUCT, conf);
    std::vector<float> centroids(DIM * NLIST);
    std::string msg;
    try {
        cluster.Train(TRAIN_ITER, nullptr, false);
    } catch(std::exception &e) {
        msg = e.what();
    }
    EXPECT_TRUE(msg.find("distances can not be nullptr") != std::string::npos);

    try {
        cluster.Train(0, centroids.data(), false);
    } catch(std::exception &e) {
        msg = e.what();
    }
    EXPECT_TRUE(msg.find("Niter must be in range") != std::string::npos);

    try {
        cluster.Train(TRAIN_ITER, centroids.data(), false);
    } catch(std::exception &e) {
        msg = e.what();
    }
    EXPECT_TRUE(msg.find("ntotal 0 must be") != std::string::npos);

    std::vector<float> randData(DIM * NTOTAL);
    FeatureGenerator(randData);
    cluster.Add(NTOTAL, randData.data());
    MOCKER(aclrtSetDevice).stubs().will(invoke(StubAclrtSetDevice));
    try {
        cluster.Train(TRAIN_ITER, centroids.data(), false);
    } catch(std::exception &e) {
        msg = e.what();
    }
    EXPECT_TRUE(msg.find("Error") != std::string::npos);
    GlobalMockObject::verify();

    cluster.Train(TRAIN_ITER, centroids.data(), false);
    cluster.verbose = true;
    cluster.Train(TRAIN_ITER, centroids.data());
}

TEST(TestAscendIndexUTClustering, Get_Interface)
{
    std::vector<int> deviceList {0};
    faiss::ascend::AscendClusteringConfig conf(deviceList);
    faiss::ascend::AscendClustering cluster(DIM, NLIST, faiss::MetricType::METRIC_INNER_PRODUCT, conf);
    EXPECT_EQ(0u, cluster.GetNTotal());

    std::string msg;
    try {
        cluster.GetSubBucketNum(-1);
    } catch(std::exception &e) {
        msg = e.what();
    }
    EXPECT_TRUE(msg.find("Error: 'seq >= 0") != std::string::npos);

    std::vector<uint8_t> randData(DIM * NTOTAL);
    FeatureGenerator(randData);
    cluster.SubClusAddInt8(NTOTAL, randData.data(), 0, SUB_NLIST);
    cluster.GetSubBucketNum(0);
}

TEST(TestAscendIndexUTClustering, ComputeCorr)
{
    std::vector<int> deviceList {0};
    faiss::ascend::AscendClusteringConfig conf(deviceList);
    faiss::ascend::AscendClustering cluster(DIM, NLIST, faiss::MetricType::METRIC_INNER_PRODUCT, conf);
    std::string msg;
    try {
        cluster.ComputeCorr(nullptr, false);
    } catch(std::exception &e) {
        msg = e.what();
    }
    EXPECT_TRUE(msg.find("correlation result ptr can not be nullptr") != std::string::npos);

    std::vector<float> corr(DIM * DIM);
    try {
        cluster.ComputeCorr(corr.data(), false);
    } catch(std::exception &e) {
        msg = e.what();
    }
    EXPECT_TRUE(msg.find("ntotal must be >=") != std::string::npos);

    std::vector<float> randData(DIM * NTOTAL);
    FeatureGenerator(randData);
    cluster.Add(NTOTAL, randData.data());
    MOCKER(aclrtSetDevice).stubs().will(invoke(StubAclrtSetDevice));
    try {
        cluster.ComputeCorr(corr.data(), false);
    } catch(std::exception &e) {
        msg = e.what();
    }
    EXPECT_TRUE(msg.find("ACL error") != std::string::npos);
    GlobalMockObject::verify();

    cluster.ComputeCorr(corr.data(), false);
    cluster.ComputeCorr(corr.data(), true);
}

TEST(TestAscendIndexUTClustering, SubClusAddInt8)
{
    std::vector<int> deviceList {0};
    faiss::ascend::AscendClusteringConfig conf(deviceList);
    faiss::ascend::AscendClustering cluster(DIM, NLIST, faiss::MetricType::METRIC_INNER_PRODUCT, conf);
    std::string msg;
    try {
        cluster.SubClusAddInt8(NTOTAL, nullptr, 0, SUB_NLIST);
    } catch(std::exception &e) {
        msg = e.what();
    }
    EXPECT_TRUE(msg.find("x can not be nullptr") != std::string::npos);

    std::vector<uint8_t> randData(DIM * NTOTAL);
    FeatureGenerator(randData);
    try {
        cluster.SubClusAddInt8(0, randData.data(), 0, SUB_NLIST);
    } catch(std::exception &e) {
        msg = e.what();
    }
    EXPECT_TRUE(msg.find("n must be > 0") != std::string::npos);
    cluster.SubClusAddInt8(NTOTAL, randData.data(), 0, SUB_NLIST);
}

TEST(TestAscendIndexUTClustering, UpdateVdm)
{
    std::vector<int> deviceList {0};
    faiss::ascend::AscendClusteringConfig conf(deviceList);
    faiss::ascend::AscendClustering cluster(DIM, NLIST, faiss::MetricType::METRIC_INNER_PRODUCT, conf);
    std::string msg;
    try {
        cluster.UpdateVdm(nullptr, nullptr);
    } catch(std::exception &e) {
        msg = e.what();
    }
    EXPECT_TRUE(msg.find("vmin ptr can not be nullptr") != std::string::npos);

    std::vector<uint16_t> vmin(DIM, 0);
    std::vector<uint16_t> vdiff(DIM, 0);
    try {
        cluster.UpdateVdm(vmin.data(), nullptr);
    } catch(std::exception &e) {
        msg = e.what();
    }
    EXPECT_TRUE(msg.find("vdiff ptr can not be nullptr") != std::string::npos);

    MOCKER(&aclrtMemcpy).stubs().will(invoke(StubAclrtMemcpy));
    try {
        cluster.UpdateVdm(vmin.data(), vdiff.data());
    } catch(std::exception &e) {
        msg = e.what();
    }
    EXPECT_TRUE(msg.find("Failed to copy") != std::string::npos);
    GlobalMockObject::verify();

    MOCKER(&faiss::ascend::AscendOperator::init).stubs().will(invoke(StubAscendOperatorInit));
    try {
        cluster.UpdateVdm(vmin.data(), vdiff.data());
    } catch(std::exception &e) {
        msg = e.what();
    }
    GlobalMockObject::verify();

    cluster.UpdateVdm(vmin.data(), vdiff.data());
}

TEST(TestAscendIndexUTClustering, SubClusExecInt8_Part1)
{
    std::vector<int> deviceList {0};
    faiss::ascend::AscendClusteringConfig conf(deviceList);
    faiss::ascend::AscendClustering cluster(DIM, NLIST, faiss::MetricType::METRIC_INNER_PRODUCT, conf);
    std::string msg;
    try {
        cluster.SubClusExecInt8(TRAIN_ITER, nullptr, nullptr, SUB_NLIST, NTOTAL);
    } catch(std::exception &e) {
        msg = e.what();
    }
    EXPECT_TRUE(msg.find("labels ptr can not be nullptr") != std::string::npos);

    std::vector<float> centroids(SUB_NLIST * DIM, 0);
    std::vector<uint16_t> deviceLabel(NTOTAL, 0);
    try {
        cluster.SubClusExecInt8(TRAIN_ITER, deviceLabel.data(), nullptr, SUB_NLIST, NTOTAL);
    } catch(std::exception &e) {
        msg = e.what();
    }
    EXPECT_TRUE(msg.find("centroids result ptr can not be nullptr") != std::string::npos);

    try {
        cluster.SubClusExecInt8(TRAIN_ITER, deviceLabel.data(), centroids.data(), SUB_NLIST, NTOTAL);
    } catch(std::exception &e) {
        msg = e.what();
    }
    EXPECT_TRUE(msg.find("VDM not Update") != std::string::npos);

    try {
        std::vector<uint16_t> vmin(DIM, 0);
        std::vector<uint16_t> vdiff(DIM, 0);
        std::vector<uint8_t> randData(DIM * NTOTAL);
        FeatureGenerator(randData);
        cluster.UpdateVdm(vmin.data(), vdiff.data());
        cluster.SubClusAddInt8(NTOTAL, randData.data(), 0, SUB_NLIST);
        cluster.SubClusExecInt8(TRAIN_ITER, deviceLabel.data(), centroids.data(), SUB_NLIST, NTOTAL);
    } catch(std::exception &e) {
        msg = e.what();
    }
    EXPECT_TRUE(msg.find("numCodes >= LOWER_BOUND && numCodes <= MAX_CLUS_POINTS") != std::string::npos);
}

TEST(TestAscendIndexUTClustering, SubClusExecInt8_Part2)
{
    std::vector<int> deviceList {0};
    faiss::ascend::AscendClusteringConfig conf(deviceList);
    faiss::ascend::AscendClustering cluster(DIM, NLIST, faiss::MetricType::METRIC_INNER_PRODUCT, conf);
    std::vector<uint16_t> vmin(DIM, 0);
    std::vector<uint16_t> vdiff(DIM, 0);
    cluster.UpdateVdm(vmin.data(), vdiff.data());
    for (size_t i = 0; i < NUM_BUCK_PER_BATCH; i++) {
        std::vector<uint8_t> data(DIM * SUB_NLIST);
        FeatureGenerator(data);
        cluster.SubClusAddInt8(SUB_NLIST, data.data(), i, SUB_NLIST);
    }
    std::string msg;
    try {
        std::vector<float> centroids(NUM_BUCK_PER_BATCH * SUB_NLIST * DIM, 0);
        std::vector<uint16_t> deviceLabel(NUM_BUCK_PER_BATCH * SUB_NLIST, 0);
        cluster.SubClusExecInt8(TRAIN_ITER, deviceLabel.data(), centroids.data(),
                                   NUM_BUCK_PER_BATCH * SUB_NLIST, NUM_BUCK_PER_BATCH * SUB_NLIST);
    } catch(std::exception &e) {
        msg = e.what();
    }
    EXPECT_TRUE(msg.empty());
}
}
