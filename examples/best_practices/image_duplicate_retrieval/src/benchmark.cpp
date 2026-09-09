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

#include <acl/acl_rt.h>
#include <faiss/ascend/AscendIndexFlat.h>
#include <faiss/ascend/AscendIndexIVFFlat.h>
#include <faiss/ascend/AscendIndexIVFRaBitQ.h>
#include <faiss/ascend/AscendIndexInt8Flat.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace
{
using Clock = std::chrono::steady_clock;

constexpr int kDimension = 128;
constexpr int kTopK = 100;
constexpr int kBatchSize = 64;
constexpr int kNlist = 1024;
constexpr int kIvfFlatNprobe = 48;
constexpr int kRaBitQNprobe = 48;
constexpr float kRaBitQRefineAlpha = 8.0F;
constexpr float kInt8Scale = 256.0F;
constexpr int kInt8BlockSize = 16384;
constexpr size_t kBaseMaximum = 100000000;
constexpr int64_t kResourceSize = 2LL * 1024 * 1024 * 1024;
constexpr std::chrono::milliseconds kFlatHbmSamplePeriod(5);
constexpr std::chrono::milliseconds kInt8FlatHbmSamplePeriod(5);
constexpr std::chrono::milliseconds kIvfFlatHbmSamplePeriod(100);
constexpr std::chrono::milliseconds kRaBitQHbmSamplePeriod(20);

const std::string kDataDir = "tmp/sift100m-prepared";
const std::string kResultDir = "tmp/bench-results";
const std::vector<size_t> kAllBaseSizes = {1000000, 10000000, 30000000, 40000000, 70000000, 100000000};
const std::vector<size_t> kIvfFlatSingleBaseSizes = {1000000, 10000000, 30000000, 40000000, 70000000};

enum class Algorithm
{
    FLAT,
    INT8_FLAT,
    IVF_FLAT,
    IVF_RABITQ,
};

struct BenchmarkPlan
{
    Algorithm algorithm;
    std::string algorithmName;
    std::string resultStem;
    std::vector<int> devices;
    std::vector<size_t> baseSizes;
};

BenchmarkPlan GetBenchmarkPlan(const std::string &name)
{
    if (name == "flat")
    {
        return {Algorithm::FLAT, "flat", "flat", {0}, kAllBaseSizes};
    }
    if (name == "int8flat")
    {
        return {Algorithm::INT8_FLAT, "int8flat", "int8flat", {0}, kAllBaseSizes};
    }
    if (name == "ivfflat_single")
    {
        return {Algorithm::IVF_FLAT, "ivfflat", "ivfflat_single", {0}, kIvfFlatSingleBaseSizes};
    }
    if (name == "ivfflat_dual")
    {
        return {Algorithm::IVF_FLAT, "ivfflat", "ivfflat_dual", {0, 1}, {kBaseMaximum}};
    }
    if (name == "ivfrabitq")
    {
        return {Algorithm::IVF_RABITQ, "ivfrabitq", "ivfrabitq", {0}, kAllBaseSizes};
    }
    throw std::runtime_error("unknown benchmark: " + name);
}

struct HbmStats
{
    size_t p70Bytes = 0;
    size_t p80Bytes = 0;
    size_t p90Bytes = 0;
    size_t p99Bytes = 0;
    size_t peakBytes = 0;
    size_t maxDevicePeakBytes = 0;
};

struct HbmSample
{
    size_t totalBytes = 0;
    std::vector<size_t> deviceBytes;
};

class HbmSampler
{
   public:
    explicit HbmSampler(const std::vector<int> &devices) : worker_(&HbmSampler::Run, this, devices)
    {
        std::unique_lock<std::mutex> lock(mutex_);
        condition_.wait(lock, [&] { return ready_; });
        if (error_ != ACL_SUCCESS)
        {
            lock.unlock();
            Stop();
            throw std::runtime_error("failed to start HBM sampler");
        }
    }

    ~HbmSampler() { Stop(); }

    HbmSampler(const HbmSampler &) = delete;
    HbmSampler &operator=(const HbmSampler &) = delete;
    HbmSampler(HbmSampler &&) = delete;
    HbmSampler &operator=(HbmSampler &&) = delete;

    void SetSamplePeriod(std::chrono::milliseconds period) { samplePeriodMs_.store(period.count()); }

    size_t Mark()
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return samples_.size();
    }

    HbmSample ReadBaseline()
    {
        std::unique_lock<std::mutex> lock(mutex_);
        const size_t previousSize = samples_.size();
        condition_.wait(lock, [&] { return samples_.size() > previousSize || error_ != ACL_SUCCESS; });
        CheckError();
        return samples_.back();
    }

    HbmStats StatsSince(size_t start, const HbmSample &baseline)
    {
        std::unique_lock<std::mutex> lock(mutex_);
        condition_.wait(lock, [&] { return samples_.size() > start || error_ != ACL_SUCCESS; });
        CheckError();

        std::vector<size_t> totalDeltas;
        totalDeltas.reserve(samples_.size() - start);
        size_t maxDevicePeak = 0;
        for (size_t i = start; i < samples_.size(); ++i)
        {
            totalDeltas.push_back(
                samples_[i].totalBytes > baseline.totalBytes ? samples_[i].totalBytes - baseline.totalBytes : 0);
            for (size_t device = 0; device < samples_[i].deviceBytes.size(); ++device)
            {
                const size_t used = samples_[i].deviceBytes[device];
                const size_t baselineUsed = baseline.deviceBytes[device];
                maxDevicePeak = std::max(maxDevicePeak, used > baselineUsed ? used - baselineUsed : 0);
            }
        }
        lock.unlock();

        std::sort(totalDeltas.begin(), totalDeltas.end());
        const auto percentile = [&totalDeltas](size_t numerator, size_t denominator)
        {
            const size_t index = (numerator * totalDeltas.size() + denominator - 1) / denominator - 1;
            return totalDeltas[index];
        };
        return {percentile(7, 10),   percentile(4, 5),   percentile(9, 10),
                percentile(99, 100), totalDeltas.back(), maxDevicePeak};
    }

   private:
    void CheckError() const
    {
        if (error_ != ACL_SUCCESS)
        {
            throw std::runtime_error("failed to sample HBM usage");
        }
    }

    void Run(const std::vector<int> &devices)
    {
        aclError result = ACL_SUCCESS;
        for (const int device : devices)
        {
            result = aclrtSetDevice(device);
            if (result != ACL_SUCCESS)
            {
                break;
            }
        }
        {
            std::lock_guard<std::mutex> lock(mutex_);
            error_ = result;
            ready_ = true;
        }
        condition_.notify_all();
        if (result != ACL_SUCCESS)
        {
            return;
        }

        while (running_.load())
        {
            HbmSample sample;
            sample.deviceBytes.reserve(devices.size());
            for (const int device : devices)
            {
                result = aclrtSetDevice(device);
                size_t freeBytes = 0;
                size_t totalBytes = 0;
                if (result == ACL_SUCCESS)
                {
                    result = aclrtGetMemInfo(ACL_HBM_MEM, &freeBytes, &totalBytes);
                }
                if (result != ACL_SUCCESS)
                {
                    break;
                }
                const size_t usedBytes = totalBytes - freeBytes;
                sample.totalBytes += usedBytes;
                sample.deviceBytes.push_back(usedBytes);
            }
            {
                std::lock_guard<std::mutex> lock(mutex_);
                error_ = result;
                if (result == ACL_SUCCESS)
                {
                    samples_.push_back(std::move(sample));
                }
            }
            condition_.notify_all();
            if (result != ACL_SUCCESS)
            {
                return;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(samplePeriodMs_.load()));
        }
    }

    void Stop()
    {
        running_.store(false);
        if (worker_.joinable())
        {
            worker_.join();
        }
    }

    std::atomic<bool> running_{true};
    std::atomic<int64_t> samplePeriodMs_{kFlatHbmSamplePeriod.count()};
    std::mutex mutex_;
    std::condition_variable condition_;
    std::vector<HbmSample> samples_;
    aclError error_ = ACL_SUCCESS;
    bool ready_ = false;
    std::thread worker_;
};

template <typename T>
struct Matrix
{
    size_t rows = 0;
    size_t cols = 0;
    std::vector<T> values;
};

Matrix<float> ReadFvecs(const std::string &path)
{
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input)
    {
        throw std::runtime_error("failed to open " + path);
    }
    const size_t bytes = static_cast<size_t>(input.tellg());
    input.seekg(0);
    int32_t dimension = 0;
    input.read(reinterpret_cast<char *>(&dimension), sizeof(dimension));
    if (dimension <= 0 || bytes % ((static_cast<size_t>(dimension) + 1) * sizeof(int32_t)) != 0)
    {
        throw std::runtime_error("invalid fvecs file " + path);
    }

    Matrix<float> result;
    result.cols = static_cast<size_t>(dimension);
    result.rows = bytes / ((result.cols + 1) * sizeof(int32_t));
    result.values.resize(result.rows * result.cols);
    constexpr size_t kReadBlock = 65536;
    std::vector<int32_t> raw(kReadBlock * (result.cols + 1));
    input.seekg(0);
    for (size_t start = 0; start < result.rows; start += kReadBlock)
    {
        const size_t count = std::min(kReadBlock, result.rows - start);
        input.read(reinterpret_cast<char *>(raw.data()),
                   static_cast<std::streamsize>(count * (result.cols + 1) * sizeof(int32_t)));
        for (size_t row = 0; row < count; ++row)
        {
            std::memcpy(result.values.data() + (start + row) * result.cols, raw.data() + row * (result.cols + 1) + 1,
                        result.cols * sizeof(float));
        }
    }
    return result;
}

Matrix<int8_t> ReadI8bin(const std::string &path)
{
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input)
    {
        throw std::runtime_error("failed to open " + path);
    }
    const size_t bytes = static_cast<size_t>(input.tellg());
    input.seekg(0);
    uint32_t rows = 0;
    uint32_t cols = 0;
    input.read(reinterpret_cast<char *>(&rows), sizeof(rows));
    input.read(reinterpret_cast<char *>(&cols), sizeof(cols));
    const size_t valueCount = static_cast<size_t>(rows) * static_cast<size_t>(cols);
    if (rows == 0 || cols == 0 || bytes != sizeof(rows) + sizeof(cols) + valueCount)
    {
        throw std::runtime_error("invalid i8bin file " + path);
    }
    Matrix<int8_t> result{rows, cols, std::vector<int8_t>(valueCount)};
    input.read(reinterpret_cast<char *>(result.values.data()), static_cast<std::streamsize>(result.values.size()));
    return result;
}

Matrix<float> ReadFloatBase()
{
    Matrix<float> base = ReadFvecs(kDataDir + "/sift_base.fvecs");
    if (base.rows != kBaseMaximum || base.cols != kDimension)
    {
        throw std::runtime_error("unexpected SIFT100M FP32 base shape");
    }
    return base;
}

Matrix<int8_t> ReadInt8Base()
{
    Matrix<int8_t> base = ReadI8bin(kDataDir + "/sift_base.i8bin");
    if (base.rows != kBaseMaximum || base.cols != kDimension)
    {
        throw std::runtime_error("unexpected SIFT100M Int8 base shape");
    }
    return base;
}

Matrix<float> ReadLearn()
{
    Matrix<float> learn = ReadFvecs(kDataDir + "/sift_learn.fvecs");
    if (learn.rows != 100000 || learn.cols != kDimension)
    {
        throw std::runtime_error("unexpected SIFT100M learn shape");
    }
    return learn;
}

void Quantize(const float *source, size_t count, std::vector<int8_t> &target)
{
    target.resize(count);
    for (size_t i = 0; i < count; ++i)
    {
        const long value = std::lround(source[i] * kInt8Scale);
        target[i] = static_cast<int8_t>(std::clamp<long>(value, -127, 127));
    }
}

void WriteIvecs(const std::string &path, const std::vector<faiss::idx_t> &labels, size_t rows)
{
    std::ofstream output(path, std::ios::binary);
    if (!output)
    {
        throw std::runtime_error("failed to open " + path);
    }
    std::vector<int32_t> record(kTopK + 1);
    record[0] = kTopK;
    for (size_t row = 0; row < rows; ++row)
    {
        for (int rank = 0; rank < kTopK; ++rank)
        {
            record[rank + 1] = static_cast<int32_t>(labels[row * kTopK + rank]);
        }
        output.write(reinterpret_cast<const char *>(record.data()),
                     static_cast<std::streamsize>(record.size() * sizeof(int32_t)));
    }
    output.close();
    if (!output)
    {
        throw std::runtime_error("failed to write " + path);
    }
}

std::string OutputPath(const BenchmarkPlan &plan, size_t baseCount)
{
    return kResultDir + "/" + plan.algorithmName + "_" + std::to_string(plan.devices.size()) + "card_" +
           std::to_string(baseCount / 1000000) + "m.ivecs";
}

template <typename Index, typename T>
void SearchAll(Index &index, const T *queries, size_t queryCount, size_t dimension, std::vector<float> &distances,
               std::vector<faiss::idx_t> &labels)
{
    for (size_t offset = 0; offset < queryCount; offset += kBatchSize)
    {
        const size_t count = std::min<size_t>(kBatchSize, queryCount - offset);
        index.search(static_cast<faiss::idx_t>(count), queries + offset * dimension, kTopK,
                     distances.data() + offset * kTopK, labels.data() + offset * kTopK);
    }
}

template <typename Search>
double Measure(Search search)
{
    search();
    const Clock::time_point start = Clock::now();
    search();
    return std::chrono::duration<double>(Clock::now() - start).count();
}

void SaveResult(std::ofstream &performance, const BenchmarkPlan &plan, size_t baseCount, size_t queryCount,
                double seconds, const std::vector<faiss::idx_t> &labels)
{
    performance << plan.algorithmName << ',' << baseCount << ',' << plan.devices.size() << ',' << std::setprecision(10)
                << static_cast<double>(queryCount) / seconds << '\n';
    WriteIvecs(OutputPath(plan, baseCount), labels, queryCount);
}

void SaveMemoryResult(std::ofstream &memory, const BenchmarkPlan &plan, size_t baseCount, const HbmStats &stats)
{
    memory << plan.algorithmName << ',' << baseCount << ',' << plan.devices.size() << ',' << stats.p70Bytes << ','
           << stats.p80Bytes << ',' << stats.p90Bytes << ',' << stats.p99Bytes << ',' << stats.peakBytes << ','
           << stats.maxDevicePeakBytes << '\n';
}

template <typename Config>
void ConfigureClustering(Config &config, bool spherical)
{
    config.useKmeansPP = true;
    config.cp.niter = 20;
    config.cp.seed = 1234;
    config.cp.min_points_per_centroid = 39;
    config.cp.max_points_per_centroid = 256;
    config.cp.spherical = spherical;
}

void ConfigureRaBitQ(faiss::ascend::AscendIndexIVFRaBitQConfig &config)
{
    ConfigureClustering(config, false);
    config.useRandomOrthogonalMatrix = true;
    config.needRefine = true;
    config.refineAlpha = kRaBitQRefineAlpha;
}

void RunFlat(const Matrix<float> &base, const Matrix<float> &queries, const BenchmarkPlan &plan, HbmSampler *sampler,
             std::ofstream &output)
{
    HbmSample baseline;
    size_t scaleStart = 0;
    if (sampler != nullptr)
    {
        sampler->SetSamplePeriod(kFlatHbmSamplePeriod);
        baseline = sampler->ReadBaseline();
        scaleStart = sampler->Mark();
    }
    faiss::ascend::AscendIndexFlatConfig config(plan.devices, kResourceSize);
    faiss::ascend::AscendIndexFlat index(kDimension, faiss::METRIC_INNER_PRODUCT, config);
    std::vector<float> distances(queries.rows * kTopK);
    std::vector<faiss::idx_t> labels(queries.rows * kTopK);
    size_t previousBaseCount = 0;
    for (const size_t baseCount : plan.baseSizes)
    {
        const size_t addCount = baseCount - previousBaseCount;
        index.add(static_cast<faiss::idx_t>(addCount), base.values.data() + previousBaseCount * kDimension);
        const double seconds =
            Measure([&] { SearchAll(index, queries.values.data(), queries.rows, queries.cols, distances, labels); });
        if (sampler != nullptr)
        {
            SaveMemoryResult(output, plan, baseCount, sampler->StatsSince(scaleStart, baseline));
            scaleStart = sampler->Mark();
        }
        else
        {
            SaveResult(output, plan, baseCount, queries.rows, seconds, labels);
        }
        previousBaseCount = baseCount;
    }
}

void RunInt8Flat(const Matrix<int8_t> &base, const Matrix<float> &queries, const BenchmarkPlan &plan,
                 HbmSampler *sampler, std::ofstream &output)
{
    HbmSample baseline;
    size_t scaleStart = 0;
    if (sampler != nullptr)
    {
        sampler->SetSamplePeriod(kInt8FlatHbmSamplePeriod);
        baseline = sampler->ReadBaseline();
        scaleStart = sampler->Mark();
    }
    faiss::ascend::AscendIndexInt8FlatConfig config(plan.devices, kResourceSize, kInt8BlockSize);
    faiss::ascend::AscendIndexInt8Flat index(kDimension, faiss::METRIC_INNER_PRODUCT, config);
    std::vector<int8_t> quantizedQueries;
    std::vector<float> distances(queries.rows * kTopK);
    std::vector<faiss::idx_t> labels(queries.rows * kTopK);
    size_t previousBaseCount = 0;
    for (const size_t baseCount : plan.baseSizes)
    {
        const size_t addCount = baseCount - previousBaseCount;
        index.add(static_cast<faiss::idx_t>(addCount), base.values.data() + previousBaseCount * kDimension);
        const double seconds = Measure(
            [&]
            {
                Quantize(queries.values.data(), queries.values.size(), quantizedQueries);
                SearchAll(index, quantizedQueries.data(), queries.rows, queries.cols, distances, labels);
            });
        if (sampler != nullptr)
        {
            SaveMemoryResult(output, plan, baseCount, sampler->StatsSince(scaleStart, baseline));
            scaleStart = sampler->Mark();
        }
        else
        {
            SaveResult(output, plan, baseCount, queries.rows, seconds, labels);
        }
        previousBaseCount = baseCount;
    }
}

void RunIVFFlat(const Matrix<float> &base, const Matrix<float> &queries, const Matrix<float> &learn,
                const BenchmarkPlan &plan, HbmSampler *sampler, std::ofstream &output)
{
    HbmSample baseline;
    size_t scaleStart = 0;
    if (sampler != nullptr)
    {
        sampler->SetSamplePeriod(kIvfFlatHbmSamplePeriod);
        baseline = sampler->ReadBaseline();
        scaleStart = sampler->Mark();
    }
    faiss::ascend::AscendIndexIVFFlatConfig config(plan.devices, kResourceSize);
    ConfigureClustering(config, true);
    faiss::ascend::AscendIndexIVFFlat index(kDimension, faiss::METRIC_INNER_PRODUCT, kNlist, config);
    index.setNumProbes(kIvfFlatNprobe);
    index.train(static_cast<faiss::idx_t>(learn.rows), learn.values.data());

    std::vector<float> distances(queries.rows * kTopK);
    std::vector<faiss::idx_t> labels(queries.rows * kTopK);
    size_t previousBaseCount = 0;
    for (const size_t baseCount : plan.baseSizes)
    {
        const size_t addCount = baseCount - previousBaseCount;
        index.add(static_cast<faiss::idx_t>(addCount), base.values.data() + previousBaseCount * kDimension);
        const double seconds =
            Measure([&] { SearchAll(index, queries.values.data(), queries.rows, queries.cols, distances, labels); });
        if (sampler != nullptr)
        {
            SaveMemoryResult(output, plan, baseCount, sampler->StatsSince(scaleStart, baseline));
            scaleStart = sampler->Mark();
        }
        else
        {
            SaveResult(output, plan, baseCount, queries.rows, seconds, labels);
        }
        previousBaseCount = baseCount;
    }
}

void RunIVFRaBitQ(const Matrix<float> &base, const Matrix<float> &queries, const Matrix<float> &learn,
                  const BenchmarkPlan &plan, HbmSampler *sampler, std::ofstream &output)
{
    HbmSample baseline;
    size_t scaleStart = 0;
    if (sampler != nullptr)
    {
        sampler->SetSamplePeriod(kRaBitQHbmSamplePeriod);
        baseline = sampler->ReadBaseline();
        scaleStart = sampler->Mark();
    }
    faiss::ascend::AscendIndexIVFRaBitQConfig config(plan.devices, kResourceSize);
    ConfigureRaBitQ(config);
    faiss::ascend::AscendIndexIVFRaBitQ index(kDimension, faiss::METRIC_L2, kNlist, config);
    index.setNumProbes(kRaBitQNprobe);
    index.train(static_cast<faiss::idx_t>(learn.rows), learn.values.data());

    std::vector<float> distances(queries.rows * kTopK);
    std::vector<faiss::idx_t> labels(queries.rows * kTopK);
    size_t previousBaseCount = 0;
    for (const size_t baseCount : plan.baseSizes)
    {
        const size_t addCount = baseCount - previousBaseCount;
        index.add(static_cast<faiss::idx_t>(addCount), base.values.data() + previousBaseCount * kDimension);
        const double seconds =
            Measure([&] { SearchAll(index, queries.values.data(), queries.rows, queries.cols, distances, labels); });
        if (sampler != nullptr)
        {
            SaveMemoryResult(output, plan, baseCount, sampler->StatsSince(scaleStart, baseline));
            scaleStart = sampler->Mark();
        }
        else
        {
            SaveResult(output, plan, baseCount, queries.rows, seconds, labels);
        }
        previousBaseCount = baseCount;
    }
}

void ExecutePlan(const BenchmarkPlan &plan, const Matrix<float> &queries, HbmSampler *sampler, std::ofstream &output)
{
    switch (plan.algorithm)
    {
        case Algorithm::FLAT:
            RunFlat(ReadFloatBase(), queries, plan, sampler, output);
            return;
        case Algorithm::INT8_FLAT:
            RunInt8Flat(ReadInt8Base(), queries, plan, sampler, output);
            return;
        case Algorithm::IVF_FLAT:
            RunIVFFlat(ReadFloatBase(), queries, ReadLearn(), plan, sampler, output);
            return;
        case Algorithm::IVF_RABITQ:
            RunIVFRaBitQ(ReadFloatBase(), queries, ReadLearn(), plan, sampler, output);
            return;
    }
    throw std::runtime_error("unsupported benchmark algorithm");
}
}  // namespace

int RunBenchmark(const std::string &planName, bool profileHbm)
{
    try
    {
        const BenchmarkPlan plan = GetBenchmarkPlan(planName);
        const Matrix<float> queries = ReadFvecs(kDataDir + "/sift_query.fvecs");
        if (queries.rows != 10000 || queries.cols != kDimension)
        {
            throw std::runtime_error("unexpected SIFT100M shape");
        }

        const std::string csvPath = kResultDir + (profileHbm ? "/memory_" : "/performance_") + plan.resultStem + ".csv";
        std::ofstream output(csvPath);
        if (!output)
        {
            throw std::runtime_error("failed to open " + csvPath);
        }
        HbmSampler *samplerPtr = nullptr;
        std::unique_ptr<HbmSampler> sampler;
        if (profileHbm)
        {
            output << "algorithm,n,device_count,hbm_total_p70_bytes,hbm_total_p80_bytes,hbm_total_p90_bytes,"
                      "hbm_total_p99_bytes,hbm_total_peak_bytes,hbm_max_device_peak_bytes\n";
            sampler = std::make_unique<HbmSampler>(plan.devices);
            samplerPtr = sampler.get();
        }
        else
        {
            output << "algorithm,n,device_count,qps\n";
        }

        ExecutePlan(plan, queries, samplerPtr, output);
        output.close();
        if (!output)
        {
            throw std::runtime_error("failed to write " + csvPath);
        }
    }
    catch (const std::exception &error)
    {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}

int main(int argc, char *argv[])
{
    if ((argc != 2 && argc != 3) || (argc == 3 && std::string(argv[2]) != "--hbm"))
    {
        std::cerr << "Usage: " << argv[0] << " <flat|int8flat|ivfflat_single|ivfflat_dual|ivfrabitq> [--hbm]\n";
        return 1;
    }
    return RunBenchmark(argv[1], argc == 3);
}
