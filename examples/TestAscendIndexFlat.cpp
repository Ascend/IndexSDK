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

// python3 aicpu_generate_model.py -t npu_type
// python3 flat_generate_model.py -d 256 -t npu_type

#include <faiss/IndexFlat.h>
#include <faiss/ascend/AscendCloner.h>
#include <faiss/ascend/AscendIndexFlat.h>
#include <faiss/impl/IDSelector.h>
#include <faiss/index_io.h>
#include <gtest/gtest.h>
#include <sys/time.h>

#include <algorithm>
#include <atomic>
#include <cfloat>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>
#include <thread>

#include "acl/acl.h"

namespace
{
unsigned int g_seed;
const int FAST_RAND_MAX = 0x7FFF;
const int RECMAP_KEY_1 = 1;
const int RECMAP_KEY_10 = 10;
const int RECMAP_KEY_100 = 100;
const int MILLI_SECOND = 1000;
using recallMap = std::unordered_map<int, float>;
inline double GetMillisecs()
{
    struct timeval tv = {0, 0};
    gettimeofday(&tv, nullptr);
    return tv.tv_sec * 1e3 + tv.tv_usec * 1e-3;
}

inline void AssertEqual(const std::vector<float> &gt, const std::vector<float> &data)
{
    const float epson = 1e-3;
    ASSERT_EQ(gt.size(), data.size());
    for (size_t i = 0; i < gt.size(); i++)
    {
        ASSERT_TRUE(fabs(gt[i] - data[i]) <= epson)
            << "i: " << i << " gt: " << gt[i] << " data: " << data[i] << std::endl;
    }
}

void computeRecall(recallMap &recMap, int j)
{
    recMap[RECMAP_KEY_100]++;
    switch (j)
    {
        case 0:
            recMap[RECMAP_KEY_1]++;
            recMap[RECMAP_KEY_10]++;
            break;
        case 1 ... 9:  // case 1到9
            recMap[RECMAP_KEY_10]++;
            break;
        default:
            break;
    }
}

recallMap calRecall(std::vector<faiss::idx_t> label, int64_t *gt, int queryNum)
{
    recallMap map;
    map[RECMAP_KEY_1] = 0;
    map[RECMAP_KEY_10] = 0;
    map[RECMAP_KEY_100] = 0;
    if (queryNum == 0)
    {
        std::cerr << "Error: Invalid queryNum value." << std::endl;
        return map;
    }
    int k = label.size() / queryNum;

    for (int i = 0; i < queryNum; i++)
    {
        std::set<int> labelSet(label.begin() + i * k, label.begin() + i * k + k);
        if (labelSet.size() != static_cast<size_t>(k))
        {
            printf("current query have duplicated labels!!! \n");
        }
        for (int j = 0; j < k; j++)
        {
            if (gt[i * k] == label[i * k + j])
            {
                computeRecall(map, j);
                break;
            }
        }
    }
    map[RECMAP_KEY_1] = map[RECMAP_KEY_1] / queryNum * 100;  // recMap[1]的百分比 这里的100代表的是百分比的计算因子
    map[RECMAP_KEY_10] = map[RECMAP_KEY_10] / queryNum * 100;  // recMap[10]的百分比 这里的100代表的是百分比的计算因子
    map[RECMAP_KEY_100] =
        map[RECMAP_KEY_100] / queryNum * 100;  // recMap[100]的百分比 这里的100代表的是百分比的计算因子
    return map;
}

// Compute a pseudorandom integer.
// Output value in range [0, 32767]
inline int FastRand(void)
{
    const int mutipliyNum = 214013;
    const int addNum = 2531011;
    const int rshiftNum = 16;
    g_seed = (mutipliyNum * g_seed + addNum);
    return (g_seed >> rshiftNum) & FAST_RAND_MAX;
}

void Norm(float *data, size_t n, int dim)
{
#pragma omp parallel for if (n > 100)
    for (size_t i = 0; i < n; ++i)
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

TEST(TestAscendIndexFlat, QPS)
{
    int dim = 256;
    size_t ntotal = 1000000;
    size_t maxSize = ntotal * dim;
    try
    {
        std::vector<float> data(maxSize);
        for (size_t i = 0; i < maxSize; i++)
        {
            data[i] = 1.0 * FastRand() / FAST_RAND_MAX;
        }

        faiss::ascend::AscendIndexFlatConfig conf({0}, 1024 * 1024 * 1500);
        faiss::ascend::AscendIndexFlat index(dim, faiss::METRIC_L2, conf);
        index.verbose = true;

        // 标准化
        Norm(data.data(), ntotal, dim);

        index.add(ntotal, data.data());
        {
            size_t getTotal = 0;
            for (size_t i = 0; i < conf.deviceList.size(); i++)
            {
                size_t tmpTotal = index.getBaseSize(conf.deviceList[i]);
                getTotal += tmpTotal;
            }
            EXPECT_EQ(getTotal, ntotal);
        }

        int warmUpTimes = 10;
        std::vector<float> distw(127 * 10, 0);
        std::vector<faiss::idx_t> labelw(127 * 10, 0);
        for (int i = 0; i < warmUpTimes; i++)
        {
            index.search(127, data.data(), 10, distw.data(), labelw.data());
        }

        std::vector<int> searchNum = {8, 16, 32, 64, 128, 256};
        for (size_t n = 0; n < searchNum.size(); n++)
        {
            int k = 10;
            int loopTimes = 100;
            std::vector<float> dist(searchNum[n] * k, 0);
            std::vector<faiss::idx_t> label(searchNum[n] * k, 0);
            double ts = GetMillisecs();
            for (int i = 0; i < loopTimes; i++)
            {
                index.search(searchNum[n], data.data(), k, dist.data(), label.data());
            }
            double te = GetMillisecs();
            printf("case[%zu]: base:%zu, dim:%d, search num:%d, QPS:%.4f\n", n, ntotal, dim, searchNum[n],
                   MILLI_SECOND * searchNum[n] * loopTimes / (te - ts));
        }
    }
    catch (std::exception &e)
    {
        printf("%s\n", e.what());
    }
}

// A single-die baseline for the shared Bitmap IDSelector search API. Run only this case with
// --gtest_filter=TestAscendIndexFlat.FilterQPS to avoid the other large examples.
TEST(TestAscendIndexFlat, FilterQPS)
{
    constexpr int dim = 256;
    constexpr int ntotal = 100000;
    constexpr int k = 30;
    constexpr int repeat = 3;
    constexpr double targetPerDie = 100000.0 / 16;
    const int batches[] = {32, 48};
    const int passPercents[] = {100, 50, 1};
    using Clock = std::chrono::steady_clock;
    auto millis = [](Clock::duration duration) { return std::chrono::duration<double, std::milli>(duration).count(); };

    std::vector<float> base(static_cast<size_t>(ntotal) * dim);
    uint32_t state = 1;
    for (float &value : base)
    {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        value = static_cast<float>(state & 0xffff) / 65535.0f - 0.5f;
    }
    faiss::ascend::AscendIndexFlatConfig config({0});
    faiss::ascend::AscendIndexFlat index(dim, faiss::METRIC_INNER_PRODUCT, config);
    index.add(ntotal, base.data());
    std::vector<faiss::idx_t> ids;
    index.getIdxMap(0, ids);
    ASSERT_EQ(ids.size(), static_cast<size_t>(ntotal));
    const size_t maskBytes = (ntotal + 7) / 8;

    auto makeMasks = [&](int batch, const faiss::IDSelectorBitmap &selector, std::vector<uint8_t> &masks)
    {
        std::fill(masks.begin(), masks.end(), 0);
        for (int pos = 0; pos < ntotal; ++pos)
        {
            if (selector.is_member(ids[pos]))
            {
                masks[pos / 8] |= uint8_t(1u << (pos % 8));
            }
        }
        for (int q = 1; q < batch; ++q)
        {
            std::copy_n(masks.begin(), maskBytes, masks.begin() + static_cast<size_t>(q) * maskBytes);
        }
    };

    struct Result
    {
        double qps;
        double callMs;
    };
    auto measure = [&](int batch, int mode, const faiss::IDSelectorBitmap &selector, const float *queries,
                       std::vector<float> &distances, std::vector<faiss::idx_t> &labels, std::vector<uint8_t> &masks)
    {
        std::vector<Result> results;
        if (mode == 1) makeMasks(batch, selector, masks);
        for (int run = 0; run < repeat; ++run)
        {
            int calls = 0;
            const auto begin = Clock::now();
            do
            {
                if (mode == 0)
                {
                    index.search(batch, queries, k, distances.data(), labels.data());
                }
                else if (mode == 1)
                {
                    index.search_with_masks(batch, queries, k, distances.data(), labels.data(), masks.data());
                }
                else
                {
                    index.search_with_selector(batch, queries, k, distances.data(), labels.data(), &selector);
                }
                ++calls;
            } while (Clock::now() - begin < std::chrono::seconds(2));
            const double elapsedMs = millis(Clock::now() - begin);
            results.push_back({1000.0 * calls * batch / elapsedMs, elapsedMs / calls});
        }
        std::sort(results.begin(), results.end(), [](const Result &a, const Result &b) { return a.qps < b.qps; });
        return results[repeat / 2];
    };

    printf("mode,selector_type,pass_percent,batch,qps,call_ms,reference_qps\n");
    int qualifiedBatch[sizeof(passPercents) / sizeof(passPercents[0])] = {};
    for (int batch : batches)
    {
        std::vector<float> queries(base.begin(), base.begin() + static_cast<size_t>(batch) * dim);
        std::vector<float> distances(batch * k), plainDistances(batch * k);
        std::vector<faiss::idx_t> labels(batch * k), plainLabels(batch * k);
        std::vector<uint8_t> masks(static_cast<size_t>(batch) * maskBytes);
        index.search(batch, queries.data(), k, plainDistances.data(), plainLabels.data());
        std::vector<uint8_t> allBitmap(maskBytes, 0xff);
        faiss::IDSelectorBitmap all(allBitmap.size(), allBitmap.data());
        makeMasks(batch, all, masks);
        index.search_with_masks(batch, queries.data(), k, distances.data(), labels.data(), masks.data());
        ASSERT_EQ(distances, plainDistances) << "all-one distances differ at batch=" << batch;
        ASSERT_EQ(labels, plainLabels) << "all-one labels differ at batch=" << batch;

        const Result plain = measure(batch, 0, all, queries.data(), distances, labels, masks);
        printf("plain,none,100,%d,%.1f,%.6f,%.1f\n", batch, plain.qps, plain.callMs, targetPerDie);
        for (size_t p = 0; p < sizeof(passPercents) / sizeof(passPercents[0]); ++p)
        {
            const int passPercent = passPercents[p];
            const int allowed = ntotal * passPercent / 100;
            std::vector<uint8_t> bitmap(maskBytes, 0);
            for (int i = 0; i < allowed; ++i)
            {
                const faiss::idx_t id = ids[(static_cast<size_t>(i) * 7919) % ntotal];
                bitmap[id / 8] |= uint8_t(1u << (id % 8));
            }
            faiss::IDSelectorBitmap selectedBitmap(bitmap.size(), bitmap.data());
            makeMasks(batch, selectedBitmap, masks);
            std::vector<float> expectedDist(batch * k);
            std::vector<faiss::idx_t> expectedIds(batch * k);
            index.search_with_masks(batch, queries.data(), k, expectedDist.data(), expectedIds.data(), masks.data());
            index.search_with_selector(batch, queries.data(), k, distances.data(), labels.data(), &selectedBitmap);
            ASSERT_EQ(distances, expectedDist) << "bitmap pass=" << passPercent;
            ASSERT_EQ(labels, expectedIds) << "bitmap pass=" << passPercent;
            for (auto id : labels) ASSERT_TRUE(selectedBitmap.is_member(id));

            const Result masked = measure(batch, 1, selectedBitmap, queries.data(), distances, labels, masks);
            const Result endToEnd = measure(batch, 2, selectedBitmap, queries.data(), distances, labels, masks);
            printf("mask,bitmap,%d,%d,%.1f,%.6f,%.1f\n", passPercent, batch, masked.qps, masked.callMs, targetPerDie);
            printf("idselector,bitmap,%d,%d,%.1f,%.6f,%.1f\n", passPercent, batch, endToEnd.qps, endToEnd.callMs,
                   targetPerDie);
            if (endToEnd.qps > targetPerDie) qualifiedBatch[p] = batch;
        }
    }
    for (size_t p = 0; p < sizeof(passPercents) / sizeof(passPercents[0]); ++p)
    {
        printf("bitmap pass=%d%% single-die reference %s, batch=%d\n", passPercents[p],
               qualifiedBatch[p] ? "met" : "not met", qualifiedBatch[p]);
    }
}

// Non-contiguous custom-ID baseline: add_with_ids flips identityIds to false, so the
// Bitmap selector goes through the per-position remap path instead of the copy fast path.
// Run only this case with --gtest_filter=TestAscendIndexFlat.SlowPathFilterQPS.
TEST(TestAscendIndexFlat, SlowPathFilterQPS)
{
    constexpr int dim = 256;
    constexpr int ntotal = 100000;
    constexpr int k = 30;
    constexpr int repeat = 3;
    constexpr double targetPerDie = 100000.0 / 16;
    const int batches[] = {32, 48};
    const int passPercents[] = {100, 50, 1};
    using Clock = std::chrono::steady_clock;
    auto millis = [](Clock::duration duration) { return std::chrono::duration<double, std::milli>(duration).count(); };

    std::vector<float> base(static_cast<size_t>(ntotal) * dim);
    uint32_t state = 1;
    for (float &value : base)
    {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        value = static_cast<float>(state & 0xffff) / 65535.0f - 0.5f;
    }
    // Non-contiguous IDs force the remap path: idxDeviceMap[p] = ID_BASE + p * ID_STRIDE.
    constexpr faiss::idx_t idBase = 1000, idStride = 7;
    std::vector<faiss::idx_t> ids(ntotal);
    for (int i = 0; i < ntotal; ++i) ids[i] = idBase + static_cast<faiss::idx_t>(i) * idStride;
    faiss::ascend::AscendIndexFlatConfig config({0});
    faiss::ascend::AscendIndexFlat index(dim, faiss::METRIC_INNER_PRODUCT, config);
    index.add_with_ids(ntotal, base.data(), ids.data());
    std::vector<faiss::idx_t> idxMap;
    index.getIdxMap(0, idxMap);
    ASSERT_EQ(idxMap.size(), static_cast<size_t>(ntotal));
    ASSERT_NE(idxMap.front(), 0) << "expected non-contiguous custom IDs for the slow path";
    // Bitmap covers the largest custom ID, as a caller would size it by external ID.
    const size_t bitmapBytes = (ids.back() + 7) / 8;

    struct Result
    {
        double qps;
        double callMs;
    };
    auto measure = [&](int batch, const faiss::IDSelectorBitmap &selector, const float *queries,
                       std::vector<float> &distances, std::vector<faiss::idx_t> &labels)
    {
        std::vector<Result> results;
        for (int run = 0; run < repeat; ++run)
        {
            int calls = 0;
            const auto begin = Clock::now();
            do
            {
                index.search_with_selector(batch, queries, k, distances.data(), labels.data(), &selector);
                ++calls;
            } while (Clock::now() - begin < std::chrono::seconds(2));
            const double elapsedMs = millis(Clock::now() - begin);
            results.push_back({1000.0 * calls * batch / elapsedMs, elapsedMs / calls});
        }
        std::sort(results.begin(), results.end(), [](const Result &a, const Result &b) { return a.qps < b.qps; });
        return results[repeat / 2];
    };

    printf("mode,selector_type,pass_percent,batch,qps,call_ms,reference_qps\n");
    int qualifiedBatch[sizeof(passPercents) / sizeof(passPercents[0])] = {};
    for (int batch : batches)
    {
        std::vector<float> queries(base.begin(), base.begin() + static_cast<size_t>(batch) * dim);
        std::vector<float> distances(batch * k);
        std::vector<faiss::idx_t> labels(batch * k);
        for (size_t p = 0; p < sizeof(passPercents) / sizeof(passPercents[0]); ++p)
        {
            const int passPercent = passPercents[p];
            const int allowed = ntotal * passPercent / 100;
            std::vector<uint8_t> bitmap(bitmapBytes, 0);
            for (int i = 0; i < allowed; ++i)
            {
                const faiss::idx_t id = ids[(static_cast<size_t>(i) * 7919) % ntotal];
                bitmap[id / 8] |= uint8_t(1u << (id % 8));
            }
            faiss::IDSelectorBitmap selectedBitmap(bitmap.size(), bitmap.data());

            index.search_with_selector(batch, queries.data(), k, distances.data(), labels.data(), &selectedBitmap);
            for (auto id : labels) ASSERT_TRUE(id == -1 || selectedBitmap.is_member(id));

            const Result endToEnd = measure(batch, selectedBitmap, queries.data(), distances, labels);
            printf("idselector,bitmap,%d,%d,%.1f,%.6f,%.1f\n", passPercent, batch, endToEnd.qps, endToEnd.callMs,
                   targetPerDie);
            if (endToEnd.qps > targetPerDie) qualifiedBatch[p] = batch;
        }
    }
    for (size_t p = 0; p < sizeof(passPercents) / sizeof(passPercents[0]); ++p)
    {
        printf("bitmap pass=%d%% single-die reference %s, batch=%d\n", passPercents[p],
               qualifiedBatch[p] ? "met" : "not met", qualifiedBatch[p]);
    }
}

TEST(TestAscendIndexFlat, ReplicaFilterQPS)
{
    constexpr int dim = 256;
    constexpr int ntotal = 100000;
    constexpr int k = 30;
    const int batches[] = {32, 48};
    const int passPercents[] = {100, 50, 1};
    std::vector<int> devices;
    if (const char *env = std::getenv("FLAT_REPLICA_DEVICES"))
    {
        std::stringstream stream(env);
        std::string token;
        while (std::getline(stream, token, ',')) devices.push_back(std::stoi(token));
    }
    else
    {
        uint32_t deviceCount = 0;
        if (aclrtGetDeviceCount(&deviceCount) != ACL_SUCCESS || deviceCount == 0)
        {
            printf("skip replica test: no accessible Ascend devices\n");
            return;
        }
        for (uint32_t device = 0; device < deviceCount; ++device) devices.push_back(static_cast<int>(device));
    }
    ASSERT_FALSE(devices.empty());

    if (std::getenv("MX_INDEX_MULTITHREAD") == nullptr)
    {
        printf("skip replica test: set MX_INDEX_MULTITHREAD=1 before startup\n");
        return;
    }

    std::vector<float> base(static_cast<size_t>(ntotal) * dim);
    uint32_t state = 1;
    for (float &value : base)
    {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        value = static_cast<float>(state & 0xffff) / 65535.0f - 0.5f;
    }
    faiss::ascend::AscendIndexFlatConfig config(devices, faiss::ascend::FLAT_DEFAULT_MEM, false);
    faiss::ascend::AscendIndexFlat index(dim, faiss::METRIC_INNER_PRODUCT, config);
    index.add(ntotal, base.data());
    std::vector<faiss::idx_t> ids;
    index.getIdxMap(devices.front(), ids);
    ASSERT_EQ(ids.size(), static_cast<size_t>(ntotal));
    for (int device : devices)
    {
        ASSERT_EQ(index.getBaseSize(device), static_cast<size_t>(ntotal));
        std::vector<faiss::idx_t> replicaIds;
        index.getIdxMap(device, replicaIds);
        ASSERT_EQ(replicaIds, ids);
    }

    using Clock = std::chrono::steady_clock;
    printf("selector_type,pass_percent,batch,run,total_qps,target_qps\n");
    for (int batch : batches)
    {
        std::vector<float> queries(base.begin(), base.begin() + static_cast<size_t>(batch) * dim);
        for (int passPercent : passPercents)
        {
            const int allowed = ntotal * passPercent / 100;
            std::vector<uint8_t> bitmap((ntotal + 7) / 8, 0);
            for (int i = 0; i < allowed; ++i)
            {
                const faiss::idx_t id = ids[(static_cast<size_t>(i) * 7919) % ntotal];
                bitmap[id / 8] |= uint8_t(1u << (id % 8));
            }
            faiss::IDSelectorBitmap selectedBitmap(bitmap.size(), bitmap.data());
            std::vector<float> distances(batch * k);
            std::vector<faiss::idx_t> labels(batch * k);
            for (size_t i = 0; i < devices.size(); ++i)
            {
                index.search_with_selector(batch, queries.data(), k, distances.data(), labels.data(), &selectedBitmap);
            }
            for (auto id : labels) ASSERT_TRUE(selectedBitmap.is_member(id));

            std::vector<double> qpsRuns;
            for (int run = 0; run < 3; ++run)
            {
                std::atomic<size_t> ready{0};
                std::atomic<bool> start{false};
                std::atomic<size_t> calls{0};
                std::vector<std::thread> workers;
                Clock::time_point deadline;
                for (size_t worker = 0; worker < devices.size(); ++worker)
                {
                    workers.emplace_back(
                        [&]
                        {
                            std::vector<float> dist(batch * k);
                            std::vector<faiss::idx_t> resultIds(batch * k);
                            ready.fetch_add(1);
                            while (!start.load()) std::this_thread::yield();
                            size_t localCalls = 0;
                            do
                            {
                                index.search_with_selector(batch, queries.data(), k, dist.data(), resultIds.data(),
                                                           &selectedBitmap);
                                ++localCalls;
                            } while (Clock::now() < deadline);
                            calls.fetch_add(localCalls);
                        });
                }
                while (ready.load() != devices.size()) std::this_thread::yield();
                const auto begin = Clock::now();
                deadline = begin + std::chrono::seconds(2);
                start.store(true);
                for (auto &worker : workers) worker.join();
                const auto end = Clock::now();
                const double seconds = std::chrono::duration<double>(end - begin).count();
                const double qps = calls.load() * batch / seconds;
                qpsRuns.push_back(qps);
                printf("bitmap,%d,%d,%d,%.1f,100000.0\n", passPercent, batch, run + 1, qps);
            }
            std::sort(qpsRuns.begin(), qpsRuns.end());
            printf("bitmap,%d,%d,median,%.1f,100000.0\n", passPercent, batch, qpsRuns[1]);
            printf("target bitmap pass=%d%% batch=%d %s\n", passPercent, batch,
                   qpsRuns[1] > 100000.0 ? "met" : "not met");
        }
    }
}

// Multi-device replica with non-contiguous custom IDs: add_with_ids flips identityIds to
// false, so every replica converts the Bitmap through the per-position remap path.
// Run with MX_INDEX_MULTITHREAD=1 and
// --gtest_filter=TestAscendIndexFlat.ReplicaSlowPathFilterQPS.
TEST(TestAscendIndexFlat, ReplicaSlowPathFilterQPS)
{
    constexpr int dim = 256;
    constexpr int ntotal = 100000;
    constexpr int k = 30;
    const int batches[] = {32, 48};
    const int passPercents[] = {100, 50, 1};
    std::vector<int> devices;
    if (const char *env = std::getenv("FLAT_REPLICA_DEVICES"))
    {
        std::stringstream stream(env);
        std::string token;
        while (std::getline(stream, token, ',')) devices.push_back(std::stoi(token));
    }
    else
    {
        uint32_t deviceCount = 0;
        if (aclrtGetDeviceCount(&deviceCount) != ACL_SUCCESS || deviceCount == 0)
        {
            printf("skip replica test: no accessible Ascend devices\n");
            return;
        }
        for (uint32_t device = 0; device < deviceCount; ++device) devices.push_back(static_cast<int>(device));
    }
    ASSERT_FALSE(devices.empty());

    if (std::getenv("MX_INDEX_MULTITHREAD") == nullptr)
    {
        printf("skip replica test: set MX_INDEX_MULTITHREAD=1 before startup\n");
        return;
    }

    std::vector<float> base(static_cast<size_t>(ntotal) * dim);
    uint32_t state = 1;
    for (float &value : base)
    {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        value = static_cast<float>(state & 0xffff) / 65535.0f - 0.5f;
    }
    // Non-contiguous IDs force the remap path on every replica: ID_BASE + p * ID_STRIDE.
    constexpr faiss::idx_t idBase = 1000, idStride = 7;
    std::vector<faiss::idx_t> ids(ntotal);
    for (int i = 0; i < ntotal; ++i) ids[i] = idBase + static_cast<faiss::idx_t>(i) * idStride;
    faiss::ascend::AscendIndexFlatConfig config(devices, faiss::ascend::FLAT_DEFAULT_MEM, false);
    faiss::ascend::AscendIndexFlat index(dim, faiss::METRIC_INNER_PRODUCT, config);
    index.add_with_ids(ntotal, base.data(), ids.data());
    std::vector<faiss::idx_t> idxMap;
    index.getIdxMap(devices.front(), idxMap);
    ASSERT_EQ(idxMap.size(), static_cast<size_t>(ntotal));
    ASSERT_NE(idxMap.front(), 0) << "expected non-contiguous custom IDs for the slow path";
    for (int device : devices)
    {
        ASSERT_EQ(index.getBaseSize(device), static_cast<size_t>(ntotal));
        std::vector<faiss::idx_t> replicaIds;
        index.getIdxMap(device, replicaIds);
        ASSERT_EQ(replicaIds, ids);
    }
    // Bitmap sized by the largest custom ID, as a caller would do with external IDs.
    const size_t bitmapBytes = (ids.back() + 7) / 8;

    using Clock = std::chrono::steady_clock;
    printf("selector_type,pass_percent,batch,run,total_qps,target_qps\n");
    for (int batch : batches)
    {
        std::vector<float> queries(base.begin(), base.begin() + static_cast<size_t>(batch) * dim);
        for (int passPercent : passPercents)
        {
            const int allowed = ntotal * passPercent / 100;
            std::vector<uint8_t> bitmap(bitmapBytes, 0);
            for (int i = 0; i < allowed; ++i)
            {
                const faiss::idx_t id = ids[(static_cast<size_t>(i) * 7919) % ntotal];
                bitmap[id / 8] |= uint8_t(1u << (id % 8));
            }
            faiss::IDSelectorBitmap selectedBitmap(bitmap.size(), bitmap.data());

            std::vector<float> distances(batch * k);
            std::vector<faiss::idx_t> labels(batch * k);
            for (size_t i = 0; i < devices.size(); ++i)
            {
                index.search_with_selector(batch, queries.data(), k, distances.data(), labels.data(), &selectedBitmap);
            }
            for (auto id : labels) ASSERT_TRUE(id == -1 || selectedBitmap.is_member(id));

            std::vector<double> qpsRuns;
            for (int run = 0; run < 3; ++run)
            {
                std::atomic<size_t> ready{0};
                std::atomic<bool> start{false};
                std::atomic<size_t> calls{0};
                std::vector<std::thread> workers;
                Clock::time_point deadline;
                for (size_t worker = 0; worker < devices.size(); ++worker)
                {
                    workers.emplace_back(
                        [&]
                        {
                            std::vector<float> dist(batch * k);
                            std::vector<faiss::idx_t> resultIds(batch * k);
                            ready.fetch_add(1);
                            while (!start.load()) std::this_thread::yield();
                            size_t localCalls = 0;
                            do
                            {
                                index.search_with_selector(batch, queries.data(), k, dist.data(), resultIds.data(),
                                                           &selectedBitmap);
                                ++localCalls;
                            } while (Clock::now() < deadline);
                            calls.fetch_add(localCalls);
                        });
                }
                while (ready.load() != devices.size()) std::this_thread::yield();
                const auto begin = Clock::now();
                deadline = begin + std::chrono::seconds(2);
                start.store(true);
                for (auto &worker : workers) worker.join();
                const auto end = Clock::now();
                const double seconds = std::chrono::duration<double>(end - begin).count();
                const double qps = calls.load() * batch / seconds;
                qpsRuns.push_back(qps);
                printf("bitmap,%d,%d,%d,%.1f,100000.0\n", passPercent, batch, run + 1, qps);
            }
            std::sort(qpsRuns.begin(), qpsRuns.end());
            printf("bitmap,%d,%d,median,%.1f,100000.0\n", passPercent, batch, qpsRuns[1]);
            printf("target bitmap pass=%d%% batch=%d %s\n", passPercent, batch,
                   qpsRuns[1] > 100000.0 ? "met" : "not met");
        }
    }
}

TEST(TestAscendIndexFlat, Acc)
{
    int dim = 256;
    size_t ntotal = 1000000;
    size_t maxSize = ntotal * dim;
    faiss::MetricType type = faiss::METRIC_L2;
    int topk = 100;
    int queryNum = 8;
    printf("generate data\n");
    std::vector<float> data(maxSize);
    try
    {
        for (size_t i = 0; i < maxSize; i++)
        {
            data[i] = 1.0 * FastRand() / FAST_RAND_MAX;
        }
        faiss::ascend::AscendIndexFlatConfig conf({0}, 1024 * 1024 * 1500);
        faiss::ascend::AscendIndexFlat index(dim, faiss::METRIC_L2, conf);
        index.verbose = true;

        // 标准化
        Norm(data.data(), ntotal, dim);

        index.add(ntotal, data.data());
        printf("start search by npu\n");
        std::vector<float> dist(queryNum * topk, 0);
        std::vector<faiss::idx_t> label(queryNum * topk, 0);
        index.search(queryNum, data.data(), topk, dist.data(), label.data());

        printf("start add by cpu\n");
        faiss::IndexFlat faissIndex(dim, type);
        faissIndex.add(ntotal, data.data());
        std::vector<float> cpuDist(queryNum * topk, 0);
        std::vector<faiss::idx_t> cpuLabel(queryNum * topk, 0);
        printf("start search by cpu\n");
        faissIndex.search(queryNum, data.data(), topk, cpuDist.data(), cpuLabel.data());
        recallMap top = calRecall(label, cpuLabel.data(), queryNum);
        printf("Recall %d: @1 = %.2f, @10 = %.2f, @100 = %.2f \n", topk, top[RECMAP_KEY_1], top[RECMAP_KEY_10],
               top[RECMAP_KEY_100]);
    }
    catch (std::exception &e)
    {
        printf("%s\n", e.what());
    }
}

}  // namespace

int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();
}
