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

#include <faiss/ascend/AscendIndexSQ.h>
#include <faiss/ascend/AscendNNInference.h>
#include <faiss/ascend/custom/IReduction.h>
#include <faiss/index_io.h>

#include <fstream>
#include <random>
#include <vector>

// 请在第36行导入对应的NN降维模型

namespace
{
int g_dimin = 256;
int g_dimout = 64;
std::string g_nnom;
std::string g_metricTypeName = "INNER_PRODUCT";
faiss::MetricType MetricType = g_metricTypeName == "L2" ? faiss::METRIC_L2 : faiss::METRIC_INNER_PRODUCT;

void ReadModel()
{
    std::string modelPath = "./";  // 导入对应的NN降维模型
    std::ifstream istrm(modelPath.c_str(), std::ios::binary);
    std::stringstream buffer;
    buffer << istrm.rdbuf();
    g_nnom = buffer.str();
    istrm.close();
}

void TestSampleNNInfer()
{
    std::vector<int> deviceList = {0};
    int ntotal = 100000;
    int maxSize = ntotal * g_dimin;
    std::vector<float> data(maxSize);
    std::vector<float> outputData(ntotal * g_dimout);
    try
    {
        for (int i = 0; i < maxSize; i++)
        {
            data[i] = drand48();
        }

        std::cout << "TestSampleNNInfer start " << std::endl;
        ReadModel();

        faiss::ascend::AscendNNInference dimInfer(deviceList, g_nnom.data(), g_nnom.size());
        dimInfer.infer(ntotal, reinterpret_cast<char *>(data.data()), reinterpret_cast<char *>(outputData.data()));

        std::cout << "TestSampleNNInfer end " << std::endl;
    }
    catch (std::exception &e)
    {
        printf("%s\n", e.what());
    }
}

void TestSampleNNReduce()
{
    std::vector<int> deviceList = {0};
    int ntotal = 100000;
    int maxSize = ntotal * g_dimin;
    std::vector<float> data(maxSize);
    std::vector<float> outputData(ntotal * g_dimout);
    try
    {
        for (int i = 0; i < maxSize; i++)
        {
            data[i] = drand48();
        }

        std::cout << "TestSampleNNReduce start " << std::endl;
        ReadModel();

        faiss::ascend::ReductionConfig reductionConfig(deviceList, g_nnom.data(), g_nnom.size());
        std::string method = "NN";
        faiss::ascend::IReduction *reduction = CreateReduction(method, reductionConfig);
        reduction->train(ntotal, data.data());
        reduction->reduce(ntotal, data.data(), outputData.data());

        std::cout << "TestSampleNNReduce end " << std::endl;
    }
    catch (std::exception &e)
    {
        printf("%s\n", e.what());
    }
}

void TestSamplePcarReduce()
{
    std::vector<int> deviceList = {0};
    int ntotal = 100000;
    int maxSize = ntotal * g_dimin;
    std::vector<float> data(maxSize);
    std::vector<float> outputData(ntotal * g_dimout);
    try
    {
        for (int i = 0; i < maxSize; i++)
        {
            data[i] = drand48();
        }

        std::cout << "TestSamplePcarReduce start " << std::endl;
        // Pcar IReduction
        faiss::ascend::ReductionConfig reductionConfig(g_dimin, g_dimout, 0, false);
        std::string method = "PCAR";
        faiss::ascend::IReduction *reduction = CreateReduction(method, reductionConfig);
        reduction->train(ntotal, data.data());
        reduction->reduce(ntotal, data.data(), outputData.data());

        std::cout << "TestSamplePcarReduce end " << std::endl;
    }
    catch (std::exception &e)
    {
        printf("%s\n", e.what());
    }
}
}  // namespace

int main(int argc, char **argv)
{
    TestSampleNNInfer();
    TestSampleNNReduce();
    TestSamplePcarReduce();
    return 0;
}
