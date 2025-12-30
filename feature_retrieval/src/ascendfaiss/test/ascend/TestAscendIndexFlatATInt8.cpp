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


#include <random>
#include <sys/time.h>
#include <gtest/gtest.h>
#include <faiss/IndexScalarQuantizer.h>
#include <faiss/ascend/custom/AscendIndexFlatATInt8.h>

namespace {
const int DIM = 128;
const int K = 1;
const size_t BASE_SIZE = 16384;
const std::vector<int> DEVICES = { 0 };
const int INT8_LOWER_BOUND = -128;
const int INT8_UPPER_BOUND = 127;
const int UINT8_UPPER_BOUND = 255;

inline double GetMillisecs()
{
    struct timeval tv = { 0, 0 };
    gettimeofday(&tv, nullptr);
    return tv.tv_sec * 1e3 + tv.tv_usec * 1e-3;
}

inline int8_t CodesQuantify(float value, float qMax, float qMin)
{
    int8_t result;
    if (value < qMin) {
        result = static_cast<int8_t>(INT8_LOWER_BOUND);
    } else if (value > qMax) {
        result = static_cast<int8_t>(INT8_UPPER_BOUND);
    } else {
        // 1e-6 is a minimum value to prevent division by 0
        result = static_cast<int8_t>(((value - qMin) / (qMax - qMin + 1e-6) * UINT8_UPPER_BOUND) + INT8_LOWER_BOUND);
    }
    return result;
}

TEST(TestAscendIndexFlatATInt8, All)
{
    std::vector<float> data(DIM * BASE_SIZE);
    for (size_t i = 0; i < DIM * BASE_SIZE; i++) {
        data[i] = drand48();
    }

    float qMin = std::numeric_limits<float>::max();
    float qMax = std::numeric_limits<float>::min();

    for (size_t i = 0; i < size_t(BASE_SIZE * DIM); i++) {
        if (data[i] < qMin) {
            qMin = data[i];
        }
        if (data[i] > qMax) {
            qMax = data[i];
        }
    }

    std::vector<int8_t> dataQ(DIM * BASE_SIZE);
    for (size_t i = 0; i < DIM * BASE_SIZE; i++) {
        dataQ[i] = CodesQuantify(data[i], qMax, qMin);
    }

    faiss::ascend::AscendIndexFlatATInt8Config conf(DEVICES);
    faiss::ascend::AscendIndexFlatATInt8 index(DIM, BASE_SIZE, conf);

    index.sendMinMax(qMin, qMax);

    index.add(BASE_SIZE, data.data());
    EXPECT_EQ(index.ntotal, BASE_SIZE);

    {
        int batch = 128;
        for (size_t i = BASE_SIZE - batch; i < BASE_SIZE; i += batch) {
            std::vector<float> dist(K * batch, 0);
            std::vector<faiss::idx_t> label(K * batch, 0);
            index.searchInt8(batch, dataQ.data() + i * DIM, K, dist.data(), label.data());
            ASSERT_EQ(label[0], i);
            ASSERT_EQ(label[K], i + 1);
            ASSERT_EQ(label[K * 2], i + 2);
            ASSERT_EQ(label[K * 3], i + 3);
        }
    }
    index.clearAscendTensor();
}

TEST(TestAscendIndexFlatATInt8, PRECISION)
{
    size_t num = 250 * 10000;
    size_t maxSize = num * DIM;

    std::vector<float> data(maxSize);
    for (size_t i = 0; i < maxSize; i++) {
        data[i] = drand48();
    }

    float qMin = std::numeric_limits<float>::max();
    float qMax = std::numeric_limits<float>::min();

    for (size_t i = 0; i < maxSize; i++) {
        if (data[i] < qMin) {
            qMin = data[i];
        }
        if (data[i] > qMax) {
            qMax = data[i];
        }
    }

    std::vector<int8_t> dataQ(maxSize);
    for (size_t i = 0; i < maxSize; i++) {
        dataQ[i] = CodesQuantify(data[i], qMax, qMin);
    }

    faiss::ascend::AscendIndexFlatATInt8Config conf(DEVICES, 768 * 1024 * 1024);
    faiss::ascend::AscendIndexFlatATInt8 index(DIM, BASE_SIZE, conf);

    faiss::IndexScalarQuantizer cpuIndex(DIM, faiss::ScalarQuantizer::QuantizerType::QT_8bit, faiss::METRIC_L2);
    cpuIndex.train(BASE_SIZE, data.data());
    cpuIndex.add(BASE_SIZE, data.data());
    EXPECT_EQ(cpuIndex.ntotal, BASE_SIZE);

    index.sendMinMax(qMin, qMax);
    index.add(BASE_SIZE, data.data());
    index.reset();
    index.sendMinMax(qMin, qMax);
    index.add(BASE_SIZE, data.data());
    EXPECT_EQ(index.ntotal, BASE_SIZE);

    std::vector<size_t> searchNum = { num / 10 };
    for (size_t n = 0; n < searchNum.size(); n++) {
        printf("Batch size %ld\n", searchNum[n]);
        std::vector<float> dist(searchNum[n] * K, 0);
        std::vector<faiss::idx_t> label(searchNum[n] * K, 0);

        std::vector<float> dist__(searchNum[n] * K, 0);
        std::vector<faiss::idx_t> label__(searchNum[n] * K, 0);
        index.searchInt8(searchNum[n], dataQ.data(), K, dist.data(), label.data());
        cpuIndex.search(searchNum[n], data.data(), K, dist__.data(), label__.data());

        int wrong = 0;
        float errNpu = 0;
        float errCpu = 0;
        for (size_t i = 0; i < searchNum[n]; i++) {
            for (int j = 0; j < K; ++j) {
                if (label[i * K + j] != label__[i * K + j]) {
                    wrong++;
                }
                errNpu += dist[i * K + j];
                errCpu += dist__[i * K + j];
            }
        }
        printf("WRONG / TOTAL = %d / %ld\n", wrong, num / 10);
        printf("error rate= %f\n", 1.0 * wrong * 10 / num);
        printf("errNpu: %f errCpu: %f\n", errNpu, errCpu);
    }
}

TEST(TestAscendIndexFlatATInt8, ASCEND_QPS)
{
    size_t num = 2500 * 10000;
    size_t maxSize = num * DIM;

    std::vector<float> data(maxSize);
    for (size_t i = 0; i < maxSize; i++) {
        data[i] = drand48();
    }

    float qMin = std::numeric_limits<float>::max();
    float qMax = std::numeric_limits<float>::min();

    for (size_t i = 0; i < maxSize; i++) {
        if (data[i] < qMin) {
            qMin = data[i];
        }
        if (data[i] > qMax) {
            qMax = data[i];
        }
    }

    std::vector<int8_t> dataQ(maxSize);
    for (size_t i = 0; i < DIM * BASE_SIZE; i++) {
        dataQ[i] = CodesQuantify(data[i], qMax, qMin);
    }

    faiss::ascend::AscendIndexFlatATInt8Config conf(DEVICES, 768 * 1024 * 1024);
    faiss::ascend::AscendIndexFlatATInt8 index(DIM, BASE_SIZE, conf);

    index.sendMinMax(qMin, qMax);
    index.add(BASE_SIZE, data.data());
    EXPECT_EQ(index.ntotal, BASE_SIZE);

    std::vector<size_t> searchNum = { num };
    for (size_t n = 0; n < searchNum.size(); n++) {
        std::vector<float> dist(searchNum[n] * K, 0);
        std::vector<faiss::idx_t> label(searchNum[n] * K, 0);

        double ts = GetMillisecs();
        index.searchInt8(searchNum[n], dataQ.data(), K, dist.data(), label.data());
        double te = GetMillisecs();

        double qps = searchNum[n] / (te - ts) * 1000;
        printf("AscendIndex case[%zu]: base:%zu, dim:%d, search num:%ld, QPS:%.4f\n", n, BASE_SIZE, DIM, searchNum[n],
            qps);
    }
}
} // namespace

int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();
}
