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


#include <algorithm>
#include <numeric>
#include <thread>
#include <vector>
#include <memory>
#include <random>
#include <fstream>
#include <gtest/gtest.h>
#include <faiss/ascend/AscendIndexSQ.h>
#include <faiss/ascend/AscendCloner.h>
#include <faiss/index_io.h>
#include <sys/time.h>
#include <faiss/ascend/custom/IReduction.h>
#include <faiss/ascend/AscendNNInference.h>
inline double GetMillisecs()
{
    struct timeval tv = { 0, 0 };
    gettimeofday(&tv, nullptr);
    return tv.tv_sec * 1e3 + tv.tv_usec * 1e-3;
}

class TestData {
public:
    void SetSeed(int seed)
    {
        this->seed = seed;
    }

    void SetNTotal(size_t ntotal)
    {
        this->ntotal = ntotal;
    }

    size_t GetNTotal() const
    {
        return ntotal;
    }

    void Generate(size_t ntotal, size_t dim)
    {
        double ts = GetMillisecs();
        size_t maxSize = dim * ntotal;
        printf("start generate %zu test data\n", maxSize);
        features.resize(maxSize);
        for (size_t i = 0; i < maxSize; ++i) {
            features[i] = 1.0 * FastRand() / fastRandMax;
        }
        double te = GetMillisecs();
        printf("generate %zu %dD features duration: %.2fs\n", ntotal, dim, (te - ts) / 1000);
    }

    void Norm(std::vector<float> &data, size_t total, int dim)
    {
        printf("start norm test data\n");
        double t0 = GetMillisecs();
	
        for (size_t i = 0; i < total; ++i) {
            float mod = 0;
            for (int j = 0; j < dim; ++j) {
                mod += data[i * dim + j] * data[i * dim + j];
            }

            mod = sqrt(mod);
            for (int j = 0; j < dim; ++j) {
                data[i * dim + j] = data[i * dim + j] / mod;
            }
        }
        double t1 = GetMillisecs();
        printf("start norm test data, cost=%f\n", (t1 - t0));
    }

    void Generate(size_t ntotal, std::vector<float> &data, int seed = 5678)
    {
        this->seed = seed;
        data.resize(ntotal);
        for (size_t i = 0; i < ntotal; ++i) {
            data[i] = 1.0 * FastRand() / fastRandMax;
        }
    }

    std::vector<float> &GetFeatures()
    {
        return features;
    }

    void Release()
    {
        std::vector<float>().swap(features);
    }

private:
    // Compute a pseudorandom integer.
    // Output value in range [0, 32767]
    inline int FastRand(void)
    {
        const int mutipliyNum = 214013;
        const int addNum = 2531011;
        const int rshiftNum = 16;
        seed = (mutipliyNum * seed + addNum);
        return (seed >> rshiftNum) & fastRandMax;
    }

private:
    int seed = 1234;
    size_t ntotal = 0;
    const int fastRandMax = 0x7FFF;
    std::vector<float> features;
};

namespace {
int64_t resourceSize = (int64_t)1 * 1024 * 1024 * 1024;
int dimIn = 256;
int dimOut = 64;
std::string nnom;
std::string MetricTypeName = "INNER_PRODUCT"; // select MetricType, INNER_PRODUCT or L2
faiss::MetricType MetricType = MetricTypeName == "L2" ? faiss::METRIC_L2 : faiss::METRIC_INNER_PRODUCT;

std::unique_ptr<TestData> testData(new TestData());
}

class TestFilterTest : public testing::Test {
public:
};

void ReadModel()
{
    std::string modelPath = "./retrieval_20210910_v1_ckpt80_batch32.om";
    std::ifstream istrm(modelPath.c_str(), std::ios::binary);
    std::stringstream buffer;
    buffer << istrm.rdbuf();
    nnom = buffer.str();
    istrm.close();
}
void TestSampleInfer(const std::vector<int>& deviceList)
{
    int ntotal = 1000000;
    int maxSize = ntotal * dimIn;
    std::vector<float> data(maxSize);
    std::vector<float> outputData(ntotal * dimOut);
    for (int i = 0; i < maxSize; i++) {
        data[i] = drand48();
    }

    printf("TestSampleInfer start \n");
    ReadModel();
    // NN Infer
    faiss::ascend::AscendNNInference dimInference(deviceList, nnom.data(), nnom.size());
    dimInference.infer(ntotal, (char *)data.data(), (char *)outputData.data());

    printf("TestSampleInfer end \n");
}

void TestSampleReduce(const std::vector<int>& deviceList)
{
    int ntotal = 1000000;
    int maxSize = ntotal * dimIn;
    std::vector<float> data(maxSize);
    std::vector<float> outputData(ntotal * dimOut);
    for (int i = 0; i < maxSize; i++) {
        data[i] = drand48();
    }

    printf("TestSampleReduce start \n");
    ReadModel();
	// NN IReduction
    faiss::ascend::ReductionConfig reductionConfig(deviceList, nnom.data(), nnom.size());

    std::string method = "NN";
    faiss::ascend::IReduction* reduction = CreateReduction(method, reductionConfig);
    reduction->train(ntotal, data.data());

    size_t comSize = ntotal * dimOut;
    std::vector<float> compressLearns;
    compressLearns.resize(comSize);
    reduction->reduce(ntotal, data.data(), compressLearns.data());
    delete reduction;
    printf("TestSampleReduce end \n");
}

void CloneAscend2CPU(const std::vector<int>& deviceList)
{
    int ntotal = 1000000;
   
    int maxSize = ntotal * dimIn;
    std::vector<float> data(maxSize);
    std::vector<float> outputData(ntotal * dimOut);
    // 取训练数据
    for (int i = 0; i < maxSize; i++) {
        data[i] = drand48();
    }

    faiss::ascend::AscendIndexSQConfig conf(deviceList, resourceSize, 64 * 16384);
    faiss::ascend::AscendIndexSQ ascendIndex(dimOut, faiss::ScalarQuantizer::QuantizerType::QT_8bit, MetricType, conf);
    // ascend index
    ascendIndex.verbose = true;

    ReadModel();
    // 降维训练数据
    faiss::ascend::AscendNNInference dimInference(deviceList, nnom.data(), nnom.size());
    dimInference.infer(ntotal, (char *)data.data(), (char *)outputData.data());
    // 训练数据
    ascendIndex.train(ntotal, outputData.data());

    // 取add数据
    std::vector<float> &features = testData->GetFeatures();
    // 降维add数据
    std::vector<float> outData(ntotal * dimOut);
    dimInference.infer(ntotal, (char *)features.data(), (char *)outData.data());
    // add数据
    ascendIndex.add(ntotal, outData.data());

    // 重复多次 1次100w加5次
    for (int i = 1; i < 5; i++) {
        // 降维add数据
        std::vector<float> outData(ntotal * dimOut);
        dimInference.infer(ntotal, (char *)(features.data() + i * ntotal), (char *)outData.data());
        // add数据
        ascendIndex.add(ntotal, outData.data());
    }

    faiss::Index *cpuIndex = faiss::ascend::index_ascend_to_cpu(&ascendIndex);
    printf("index_ascend_to_cpu ok!\n");
    ASSERT_FALSE(cpuIndex == nullptr);
    const char *outfilename = "./SQTESTNN0.faiss";

    write_index(cpuIndex, outfilename);
    printf("write ascendIndex to file ok!\n");
    delete cpuIndex;
}

faiss::ascend::AscendIndex* CreateFromStream(const std::vector<int>& deviceList)
{
    faiss::ascend::AscendClonerOptions option;
    option.reserveVecs = 0;
    option.verbose = false;
    option.resourceSize = resourceSize;
    option.blockSize = 64 * 16384;
    const char *indexfilename = "./SQTESTNN0.faiss";
                                  
    faiss::Index *index = faiss::read_index(indexfilename);
    faiss::ascend::AscendIndex* sq_index(
        dynamic_cast<faiss::ascend::AscendIndexSQ *>(faiss::ascend::index_cpu_to_ascend(deviceList, index, &option)));
    delete index;
    return sq_index;
}

void TestPallPlate(const std::vector<int>& device)
{
    printf("\n-----------------------device ");
    for (auto x : device) {
        printf("%d ", x);
    }
    printf("-----------------------\n");
    
    std::vector<int> searchNums = { 1, 2, 4, 8, 16 };
    
    // 读盘之前落盘的数据
    faiss::ascend::AscendIndex* index = CreateFromStream(device);
    // read model
    ReadModel();
    printf("create NN start \n");
    faiss::ascend::AscendNNInference dimInference(device, nnom.data(), nnom.size());
    printf("create NN end \n");

    const int k = 100;
    std::vector<float> data;
 
    const int warmUpBatch = 1;
    int maxSearchNum = warmUpBatch > searchNums.back() ? warmUpBatch : searchNums.back();
    testData->Generate(maxSearchNum * dimIn, data, 5678);
    testData->Norm(data, maxSearchNum, dimIn);
    // 降维search数据
    std::vector<float> outputData(maxSearchNum * dimOut);
    dimInference.infer(maxSearchNum, (char *)data.data(), (char *)outputData.data());

    // warmup search
    const int warmUpTimes = 10;
    std::vector<float> distw(warmUpBatch * k, 0);
    std::vector<faiss::idx_t> labelw(warmUpBatch * k, 0);

    for (int j = 0; j < warmUpTimes; j++) {
        index->search(warmUpBatch, outputData.data(), k, distw.data(), labelw.data());
    }

    for (auto searchnum : searchNums) {
        std::vector<float> dist(k * searchnum, 0);
        std::vector<faiss::idx_t> label(k * searchnum, 0);

        // QPS
        int loopTime = 10;
        double tStart = GetMillisecs();
        for (int z = 0; z < loopTime; ++z) {
            index->search(searchnum, outputData.data(), k, dist.data(), label.data());
        }
        double tEnd = GetMillisecs();
        float qps = 1000 * searchnum * loopTime / (tEnd - tStart);
        printf("dim: %d, topk: %d,loopTime: %d, batch: %d,  QPS: %f\n", 
            dimOut, k, loopTime, searchnum, qps);
    }
}

TEST(TestAscendNN, Infer)
{   
    std::vector<int> dev0 = { 0 };
    TestSampleInfer(dev0);
}

TEST(TestAscendNN, Reduce)
{   
    std::vector<int> dev0 = { 0 };
    TestSampleReduce(dev0);
}

TEST(TestAscendNN, PallPlate)
{   
    std::vector<int> dev0 = { 0 };
    CloneAscend2CPU(dev0);
}

TEST(TestAscendNN, MultiThreadPall)
{   
    // 先落盘PallPlate再跑多线程多卡
    std::vector<std::vector<int>> dev = { { 0 }, { 1 }, { 2 }, { 3 } };
    std::vector<std::thread> threadPools;
    auto pall = [](std::vector<int> x) {
        TestPallPlate(x);
    };
    for (int i = 0; i < dev.size(); i++) {
        threadPools.emplace_back(std::thread(pall, dev[i]));
    }

    for (auto& devThread : threadPools) {
        devThread.join();
    }
}

int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);
    size_t ntotal = 5000000; 
    size_t dim1 = 256;       // The dimension before the reduction

    testData->SetNTotal(ntotal);
    testData->Generate(ntotal, dim1);
    testData->Norm(testData->GetFeatures(), ntotal, dim1);

    return RUN_ALL_TESTS();
}