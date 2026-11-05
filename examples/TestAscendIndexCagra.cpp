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

// Reuses the core I/O, build lifecycle and search timing from TestAscendCagraScale.
// Run this one SIFT1M end-to-end test; no standalone random-graph QPS test.
#if __has_include(<faiss/ascend/AscendCagraGraphBuilder.h>)
#include <faiss/ascend/AscendCagraGraphBuilder.h>
#include <faiss/ascend/AscendIndexCagra.h>
#else
#include "index/AscendCagraGraphBuilder.h"
#include "index/AscendIndexCagra.h"
#endif
#include <faiss/IndexFlat.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

namespace
{
using Clock = std::chrono::steady_clock;
struct Config
{
    int dim = 128;
    int dataNum = 1000000;
    int intermediateDegree = 128;
    int graphDegree = 64;
    int topK = 32;
    int queryNum = 100;
    int graphAccuracySamples = 0;
    int deviceId = 0;
    int maxIterations = 20;
    int searchWarmups = 3;
    int searchLoops = 20;
    float terminationThreshold = 0.0001F;
    bool guaranteeConnectivity = true;
    bool verbose = true;
    bool runCpuGroundTruth = true;
    std::string siftDir;
};
int GetEnvInt(const char *name, int defaultValue, int minimum, int maximum)
{
    const char *value = std::getenv(name);
    if (value == nullptr || value[0] == '\0')
    {
        return defaultValue;
    }
    const std::string text(value);
    size_t parsedLength = 0;
    const long long parsed = std::stoll(text, &parsedLength);
    if (parsedLength != text.size() || parsed < minimum || parsed > maximum)
    {
        throw std::runtime_error(std::string(name) + " must be in [" + std::to_string(minimum) + ", " +
                                 std::to_string(maximum) + "], got: " + text);
    }
    return static_cast<int>(parsed);
}

bool GetEnvBool(const char *name, bool defaultValue)
{
    const char *value = std::getenv(name);
    if (value == nullptr || value[0] == '\0')
    {
        return defaultValue;
    }
    const std::string text(value);
    if (text == "1" || text == "true" || text == "TRUE" || text == "on")
    {
        return true;
    }
    if (text == "0" || text == "false" || text == "FALSE" || text == "off")
    {
        return false;
    }
    throw std::runtime_error(std::string(name) + " must be 0/1, true/false or on/off, got: " + text);
}

std::string GetEnvString(const char *name, const std::string &defaultValue)
{
    const char *value = std::getenv(name);
    return value == nullptr || value[0] == '\0' ? defaultValue : value;
}

int GetCompatibleEnvInt(const char *legacyName, const char *alias, int fallback, int minimum, int maximum)
{
    const char *value = std::getenv(legacyName);
    return GetEnvInt(value != nullptr && value[0] != '\0' ? legacyName : alias, fallback, minimum, maximum);
}

Config ReadConfig()
{
    Config config;
    config.deviceId = GetEnvInt("CAGRA_DEVICE_ID", 0, 0, 1024);
    config.dataNum = GetEnvInt("CAGRA_DATA_NUM", 1000000, 129, std::numeric_limits<int>::max());
    config.topK = GetEnvInt("CAGRA_TOPK", 32, 32, std::min(256, config.dataNum - 1));
    config.queryNum = GetCompatibleEnvInt("CAGRA_QUERY_NUM", "CAGRA_SIFT_NQ", 100, 1, 4096);
    config.graphAccuracySamples =
        GetCompatibleEnvInt("CAGRA_GRAPH_ACCURACY_SAMPLES", "CAGRA_GRAPH_SAMPLES", 0, 0, config.dataNum);
    config.maxIterations = GetEnvInt("CAGRA_MAX_ITERATIONS", 20, 1, 1000);
    config.searchWarmups = GetCompatibleEnvInt("CAGRA_SEARCH_WARMUPS", "CAGRA_WARMUP", 3, 0, 1000);
    config.searchLoops = GetCompatibleEnvInt("CAGRA_SEARCH_LOOPS", "CAGRA_REPEAT", 20, 1, 10000);
    config.verbose = GetEnvBool("CAGRA_VERBOSE", true);
    config.runCpuGroundTruth = GetEnvBool("CAGRA_RUN_CPU_GT", true);
    config.siftDir = GetEnvString("CAGRA_SIFT_DIR", "");
    return config;
}

double ElapsedMs(Clock::time_point start, Clock::time_point end)
{
    return std::chrono::duration<double, std::milli>(end - start).count();
}

std::string JoinPath(const std::string &directory, const std::string &fileName)
{
    if (directory.empty() || directory == ".")
    {
        return directory.empty() ? fileName : directory + "/" + fileName;
    }
    const char last = directory.back();
    return last == '/' || last == '\\' ? directory + fileName : directory + "/" + fileName;
}

std::vector<float> ReadFvecs(const std::string &path, int expectedDim, int requestedRows)
{
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input)
    {
        throw std::runtime_error("failed to open fvecs file: " + path);
    }
    const std::streamoff fileBytes = input.tellg();
    const uint64_t recordBytes = static_cast<uint64_t>(expectedDim + 1) * sizeof(float);
    if (fileBytes <= 0 || static_cast<uint64_t>(fileBytes) % recordBytes != 0)
    {
        throw std::runtime_error("invalid fvecs file size for D=" + std::to_string(expectedDim) + ": " + path);
    }
    const uint64_t availableRows = static_cast<uint64_t>(fileBytes) / recordBytes;
    if (availableRows < static_cast<uint64_t>(requestedRows))
    {
        throw std::runtime_error("fvecs file has only " + std::to_string(availableRows) + " rows, but " +
                                 std::to_string(requestedRows) + " are required: " + path);
    }

    input.seekg(0, std::ios::beg);
    std::vector<float> result(static_cast<size_t>(requestedRows) * expectedDim);
    constexpr int batchRows = 4096;
    std::vector<char> buffer(static_cast<size_t>(batchRows) * recordBytes);
    for (int firstRow = 0; firstRow < requestedRows; firstRow += batchRows)
    {
        const int currentRows = std::min(batchRows, requestedRows - firstRow);
        const size_t currentBytes = static_cast<size_t>(currentRows) * recordBytes;
        if (!input.read(buffer.data(), static_cast<std::streamsize>(currentBytes)))
        {
            throw std::runtime_error("failed to read fvecs records from: " + path);
        }
        for (int offset = 0; offset < currentRows; ++offset)
        {
            const char *record = buffer.data() + static_cast<size_t>(offset) * recordBytes;
            int32_t rowDim = 0;
            std::memcpy(&rowDim, record, sizeof(rowDim));
            if (rowDim != expectedDim)
            {
                throw std::runtime_error("fvecs row " + std::to_string(firstRow + offset) +
                                         " has D=" + std::to_string(rowDim) +
                                         ", expected D=" + std::to_string(expectedDim) + ": " + path);
            }
            std::memcpy(result.data() + static_cast<size_t>(firstRow + offset) * expectedDim, record + sizeof(rowDim),
                        static_cast<size_t>(expectedDim) * sizeof(float));
        }
    }
    std::cout << "input_file path=" << path << " available_rows=" << availableRows << " loaded_rows=" << requestedRows
              << " dim=" << expectedDim << std::endl;
    return result;
}

class DisjointSet
{
   public:
    explicit DisjointSet(size_t size) : parents(size), ranks(size, 0) { std::iota(parents.begin(), parents.end(), 0U); }

    uint32_t Find(uint32_t node)
    {
        uint32_t root = node;
        while (parents[root] != root)
        {
            root = parents[root];
        }
        while (parents[node] != node)
        {
            const uint32_t parent = parents[node];
            parents[node] = root;
            node = parent;
        }
        return root;
    }

    void Unite(uint32_t left, uint32_t right)
    {
        left = Find(left);
        right = Find(right);
        if (left == right)
        {
            return;
        }
        if (ranks[left] < ranks[right])
        {
            std::swap(left, right);
        }
        parents[right] = left;
        if (ranks[left] == ranks[right])
        {
            ++ranks[left];
        }
    }

   private:
    std::vector<uint32_t> parents;
    std::vector<uint8_t> ranks;
};

struct GraphDiagnostics
{
    uint32_t weakComponents = 0;
    uint32_t zeroIndegreeNodes = 0;
    double largestWeakComponentRatio = 0.0;
    double validationMs = 0.0;
};

GraphDiagnostics ValidateAndAnalyzeGraph(const Config &config, const std::vector<uint32_t> &graph)
{
    const auto start = Clock::now();
    DisjointSet components(config.dataNum);
    std::vector<uint32_t> indegrees(config.dataNum, 0);
    std::vector<uint32_t> seenInRow(config.dataNum, std::numeric_limits<uint32_t>::max());
    for (uint32_t row = 0; row < static_cast<uint32_t>(config.dataNum); ++row)
    {
        const size_t rowOffset = static_cast<size_t>(row) * config.graphDegree;
        for (int rank = 0; rank < config.graphDegree; ++rank)
        {
            const uint32_t neighbor = graph[rowOffset + rank];
            if (neighbor >= static_cast<uint32_t>(config.dataNum))
            {
                throw std::runtime_error("graph contains an out-of-range node at row " + std::to_string(row));
            }
            if (neighbor == row)
            {
                throw std::runtime_error("graph contains a self edge at row " + std::to_string(row));
            }
            if (seenInRow[neighbor] == row)
            {
                throw std::runtime_error("graph contains a duplicate edge at row " + std::to_string(row));
            }
            seenInRow[neighbor] = row;
            ++indegrees[neighbor];
            components.Unite(row, neighbor);
        }
    }

    std::vector<uint32_t> componentSizes(config.dataNum, 0);
    for (uint32_t row = 0; row < static_cast<uint32_t>(config.dataNum); ++row)
    {
        ++componentSizes[components.Find(row)];
    }
    GraphDiagnostics result;
    uint32_t largestComponent = 0;
    for (uint32_t size : componentSizes)
    {
        if (size != 0)
        {
            ++result.weakComponents;
            largestComponent = std::max(largestComponent, size);
        }
    }
    result.zeroIndegreeNodes = static_cast<uint32_t>(std::count(indegrees.begin(), indegrees.end(), 0U));
    result.largestWeakComponentRatio = static_cast<double>(largestComponent) / config.dataNum;
    result.validationMs = ElapsedMs(start, Clock::now());
    return result;
}

struct ExactReferences
{
    std::vector<uint32_t> queryGroundTruth;
    std::vector<uint32_t> graphGroundTruth;
    std::vector<uint32_t> graphSampleRows;
    double indexAddMs = 0.0;
    double queryGroundTruthMs = 0.0;
    double graphGroundTruthMs = 0.0;
};

ExactReferences BuildExactReferences(const Config &config, const std::vector<float> &base,
                                     const std::vector<float> &queries)
{
    ExactReferences result;
    faiss::IndexFlatL2 exactIndex(config.dim);
    auto start = Clock::now();
    exactIndex.add(config.dataNum, base.data());
    result.indexAddMs = ElapsedMs(start, Clock::now());

    std::vector<float> queryDistances(static_cast<size_t>(config.queryNum) * config.topK);
    std::vector<faiss::idx_t> queryLabels(static_cast<size_t>(config.queryNum) * config.topK);
    start = Clock::now();
    exactIndex.search(config.queryNum, queries.data(), config.topK, queryDistances.data(), queryLabels.data());
    result.queryGroundTruthMs = ElapsedMs(start, Clock::now());
    result.queryGroundTruth.resize(queryLabels.size());
    std::transform(queryLabels.begin(), queryLabels.end(), result.queryGroundTruth.begin(),
                   [](faiss::idx_t label) { return static_cast<uint32_t>(label); });

    if (config.graphAccuracySamples == 0)
    {
        return result;
    }

    result.graphSampleRows.resize(config.graphAccuracySamples);
    std::vector<float> sampleVectors(static_cast<size_t>(config.graphAccuracySamples) * config.dim);
    for (int sample = 0; sample < config.graphAccuracySamples; ++sample)
    {
        const uint32_t row =
            static_cast<uint32_t>((static_cast<uint64_t>(sample) * config.dataNum) / config.graphAccuracySamples);
        result.graphSampleRows[sample] = row;
        std::copy_n(base.data() + static_cast<size_t>(row) * config.dim, config.dim,
                    sampleVectors.data() + static_cast<size_t>(sample) * config.dim);
    }

    const int candidates = config.topK + 1;
    std::vector<float> sampleDistances(static_cast<size_t>(config.graphAccuracySamples) * candidates);
    std::vector<faiss::idx_t> sampleLabels(static_cast<size_t>(config.graphAccuracySamples) * candidates);
    start = Clock::now();
    exactIndex.search(config.graphAccuracySamples, sampleVectors.data(), candidates, sampleDistances.data(),
                      sampleLabels.data());
    result.graphGroundTruthMs = ElapsedMs(start, Clock::now());
    result.graphGroundTruth.resize(static_cast<size_t>(config.graphAccuracySamples) * config.topK);
    for (int sample = 0; sample < config.graphAccuracySamples; ++sample)
    {
        int outputRank = 0;
        for (int rank = 0; rank < candidates && outputRank < config.topK; ++rank)
        {
            const faiss::idx_t candidate = sampleLabels[static_cast<size_t>(sample) * candidates + rank];
            if (candidate < 0 || candidate >= config.dataNum ||
                candidate == static_cast<faiss::idx_t>(result.graphSampleRows[sample]))
            {
                continue;
            }
            result.graphGroundTruth[static_cast<size_t>(sample) * config.topK + outputRank] =
                static_cast<uint32_t>(candidate);
            ++outputRank;
        }
        if (outputRank != config.topK)
        {
            throw std::runtime_error("exact graph ground truth returned an incomplete row");
        }
    }
    return result;
}

double RecallAtK(const std::vector<uint32_t> &expected, const std::vector<uint32_t> &actual, int rows, int resultWidth,
                 int k)
{
    uint64_t hits = 0;
    for (int row = 0; row < rows; ++row)
    {
        std::unordered_set<uint32_t> truth(expected.begin() + static_cast<size_t>(row) * resultWidth,
                                           expected.begin() + static_cast<size_t>(row) * resultWidth + k);
        for (int rank = 0; rank < k; ++rank)
        {
            hits += truth.count(actual[static_cast<size_t>(row) * resultWidth + rank]);
        }
    }
    return static_cast<double>(hits) / (static_cast<uint64_t>(rows) * k);
}

struct SearchDiagnostics
{
    double validRatio = 0.0;
    double uniqueRatio = 0.0;
};

SearchDiagnostics AnalyzeSearch(const Config &config, const std::vector<uint32_t> &labels)
{
    uint64_t valid = 0;
    uint64_t unique = 0;
    for (int query = 0; query < config.queryNum; ++query)
    {
        std::unordered_set<uint32_t> seen;
        for (int rank = 0; rank < config.topK; ++rank)
        {
            const uint32_t label = labels[static_cast<size_t>(query) * config.topK + rank];
            if (label >= static_cast<uint32_t>(config.dataNum))
            {
                continue;
            }
            ++valid;
            unique += seen.insert(label).second;
        }
    }
    const double total = static_cast<double>(static_cast<uint64_t>(config.queryNum) * config.topK);
    SearchDiagnostics result;
    result.validRatio = valid / total;
    result.uniqueRatio = unique / total;
    return result;
}

double GraphNeighborRecallAtK(const Config &config, const ExactReferences &reference,
                              const std::vector<uint32_t> &graph, int k, bool scanFullDegree)
{
    uint64_t hits = 0;
    const int graphRanks = scanFullDegree ? config.graphDegree : k;
    for (int sample = 0; sample < config.graphAccuracySamples; ++sample)
    {
        std::unordered_set<uint32_t> truth(
            reference.graphGroundTruth.begin() + static_cast<size_t>(sample) * config.topK,
            reference.graphGroundTruth.begin() + static_cast<size_t>(sample) * config.topK + k);
        const size_t graphOffset = static_cast<size_t>(reference.graphSampleRows[sample]) * config.graphDegree;
        for (int rank = 0; rank < graphRanks; ++rank)
        {
            hits += truth.count(graph[graphOffset + rank]);
        }
    }
    return static_cast<double>(hits) / (static_cast<uint64_t>(config.graphAccuracySamples) * k);
}

struct SearchResult
{
    std::vector<uint32_t> labels;
    double initMs = 0.0;
    double addMs = 0.0;
    double quantizeMs = 0.0;
    double warmupMs = 0.0;
    double averageSearchMs = 0.0;
    double qps = 0.0;
    double averageQueryPrepareMs = 0.0;
    double averageEndToEndMs = 0.0;
    double endToEndQps = 0.0;
};

SearchResult RunNpuSearch(const Config &config, const std::vector<float> &base, const std::vector<float> &queries,
                          const std::vector<uint32_t> &graph)
{
    SearchResult result;
    faiss::ascend::AscendIndexCagra index;
    auto start = Clock::now();
    int ret = index.Init(config.dim, config.graphDegree, config.dataNum, config.topK, {config.deviceId});
    result.initMs = ElapsedMs(start, Clock::now());
    if (ret != 0)
    {
        throw std::runtime_error("index.Init failed: " + std::to_string(ret));
    }

    // The public master API requires a non-null N*2 hash input.
    std::vector<uint32_t> hash(static_cast<size_t>(config.dataNum) * 2, 0);
    start = Clock::now();
    ret = index.Add(graph.data(), hash.data(), base.data());
    result.addMs = ElapsedMs(start, Clock::now());
    if (ret != 0)
    {
        throw std::runtime_error("index.Add failed: " + std::to_string(ret));
    }

    start = Clock::now();
    ret = index.QuantizeData(config.queryNum, queries.data(), config.dataNum, base.data());
    result.quantizeMs = ElapsedMs(start, Clock::now());
    if (ret != 0)
    {
        throw std::runtime_error("index.QuantizeData failed: " + std::to_string(ret));
    }

    std::vector<float> distances(static_cast<size_t>(config.queryNum) * config.topK);
    result.labels.resize(static_cast<size_t>(config.queryNum) * config.topK);
    start = Clock::now();
    for (int loop = 0; loop < config.searchWarmups; ++loop)
    {
        ret = index.Search(config.queryNum, queries.data(), config.topK, distances.data(), result.labels.data());
        if (ret != 0)
        {
            throw std::runtime_error("index.Search warmup failed: " + std::to_string(ret));
        }
    }
    result.warmupMs = ElapsedMs(start, Clock::now());

    start = Clock::now();
    for (int loop = 0; loop < config.searchLoops; ++loop)
    {
        ret = index.Search(config.queryNum, queries.data(), config.topK, distances.data(), result.labels.data());
        if (ret != 0)
        {
            throw std::runtime_error("index.Search failed: " + std::to_string(ret));
        }
    }
    const double searchMs = ElapsedMs(start, Clock::now());
    result.averageSearchMs = searchMs / config.searchLoops;
    result.qps = 1000.0 * config.queryNum * config.searchLoops / searchMs;

    for (int loop = 0; loop < config.searchWarmups; ++loop)
    {
        ret = index.QuantizeQueries(config.queryNum, queries.data());
        if (ret != 0)
        {
            throw std::runtime_error("index.QuantizeQueries warmup failed: " + std::to_string(ret));
        }
        ret = index.Search(config.queryNum, queries.data(), config.topK, distances.data(), result.labels.data());
        if (ret != 0)
        {
            throw std::runtime_error("index.Search end-to-end warmup failed: " + std::to_string(ret));
        }
    }
    double queryPrepareMs = 0.0;
    start = Clock::now();
    for (int loop = 0; loop < config.searchLoops; ++loop)
    {
        const auto prepareStart = Clock::now();
        ret = index.QuantizeQueries(config.queryNum, queries.data());
        queryPrepareMs += ElapsedMs(prepareStart, Clock::now());
        if (ret != 0)
        {
            throw std::runtime_error("index.QuantizeQueries failed: " + std::to_string(ret));
        }
        ret = index.Search(config.queryNum, queries.data(), config.topK, distances.data(), result.labels.data());
        if (ret != 0)
        {
            throw std::runtime_error("index.Search end-to-end failed: " + std::to_string(ret));
        }
    }
    const double endToEndMs = ElapsedMs(start, Clock::now());
    result.averageQueryPrepareMs = queryPrepareMs / config.searchLoops;
    result.averageEndToEndMs = endToEndMs / config.searchLoops;
    result.endToEndQps = 1000.0 * config.queryNum * config.searchLoops / endToEndMs;
    if (searchMs <= 0 || endToEndMs <= 0)
    {
        throw std::runtime_error("non-positive measured search interval");
    }
    const SearchDiagnostics diagnostics = AnalyzeSearch(config, result.labels);
    if (diagnostics.validRatio != 1.0 || diagnostics.uniqueRatio != 1.0)
    {
        throw std::runtime_error("NPU search returned invalid or duplicate IDs");
    }
    for (int query = 0; query < config.queryNum; ++query)
    {
        for (int rank = 0; rank < config.topK; ++rank)
        {
            const size_t offset = static_cast<size_t>(query) * config.topK + rank;
            if (!std::isfinite(distances[offset]) || (rank > 0 && distances[offset] < distances[offset - 1]))
            {
                throw std::runtime_error("NPU search returned non-finite or unsorted distances");
            }
        }
    }
    return result;
}

void RunSift1M(const Config &config)
{
    std::cout << std::fixed << std::setprecision(4) << "CAGRA scale: N=" << config.dataNum << " D=" << config.dim
              << " M=" << config.intermediateDegree << " G=" << config.graphDegree << " Q=" << config.queryNum
              << " topK=" << config.topK << " device=" << config.deviceId
              << " graph_samples=" << config.graphAccuracySamples << " max_iterations=" << config.maxIterations
              << " warmup=" << config.searchWarmups << " repeat=" << config.searchLoops
              << " run_npu_search=1 gt=" << (config.runCpuGroundTruth ? "CPU_IndexFlatL2" : "skipped") << std::endl;
    const char *modelPath = std::getenv("MX_INDEX_MODELPATH");
    std::cout << "MX_INDEX_MODELPATH=" << (modelPath == nullptr ? "<unset>" : modelPath) << std::endl;
    std::cout << "scale_phase data_loading start" << std::endl;
    auto start = Clock::now();
    const auto base = ReadFvecs(JoinPath(config.siftDir, "sift_base.fvecs"), config.dim, config.dataNum);
    const auto queries = ReadFvecs(JoinPath(config.siftDir, "sift_query.fvecs"), config.dim, config.queryNum);
    if (!std::all_of(base.begin(), base.end(), [](float x) { return std::isfinite(x); }) ||
        !std::all_of(queries.begin(), queries.end(), [](float x) { return std::isfinite(x); }))
    {
        throw std::runtime_error("SIFT input contains non-finite values");
    }
    std::cout << "scale_phase data_generation_ms=" << ElapsedMs(start, Clock::now()) << std::endl;
    std::vector<uint32_t> graph(static_cast<size_t>(config.dataNum) * config.graphDegree);
    double buildMs = 0.0;
    std::cout << "scale_phase npu_build start" << std::endl;
    // As in the legacy Scale benchmark, release build resources before search.
    {
        faiss::ascend::AscendCagraGraphBuilder builder;
        int ret = builder.Init(config.dim, config.graphDegree, config.dataNum, {config.deviceId});
        if (ret != 0)
        {
            throw std::runtime_error("builder.Init failed: " + std::to_string(ret));
        }
        faiss::ascend::AscendCagraGraphBuildConfig buildConfig;
        buildConfig.intermediateDegree = config.intermediateDegree;
        buildConfig.maxIterations = config.maxIterations;
        buildConfig.terminationThreshold = config.terminationThreshold;
        buildConfig.guaranteeConnectivity = config.guaranteeConnectivity;
        buildConfig.verbose = config.verbose;
        start = Clock::now();
        ret = builder.Build(base.data(), graph.data(), buildConfig);
        buildMs = ElapsedMs(start, Clock::now());
        if (ret != 0)
        {
            throw std::runtime_error("builder.Build failed: " + std::to_string(ret));
        }
    }
    std::cout << "build_performance build_ms=" << buildMs << " build_s=" << buildMs / 1000.0
              << " vectors_per_s=" << 1000.0 * config.dataNum / buildMs << std::endl;
    const auto graphDiagnostics = ValidateAndAnalyzeGraph(config, graph);
    std::cout << "graph_validation edges=" << graph.size() << " invalid=0 self_loops=0 duplicate_rows=0"
              << " weak_components=" << graphDiagnostics.weakComponents
              << " largest_weak_component_ratio=" << graphDiagnostics.largestWeakComponentRatio
              << " zero_indegree_nodes=" << graphDiagnostics.zeroIndegreeNodes << std::endl;
    ExactReferences exact;
    if (config.runCpuGroundTruth)
    {
        std::cout << "scale_phase cpu_exact_ground_truth start" << std::endl;
        exact = BuildExactReferences(config, base, queries);
        std::cout << "ground_truth exact_index_add_ms=" << exact.indexAddMs
                  << " query_gt_ms=" << exact.queryGroundTruthMs << " graph_sample_gt_ms=" << exact.graphGroundTruthMs
                  << std::endl;
    }
    else
    {
        std::cout << "scale_phase cpu_exact_ground_truth skipped reason=CAGRA_RUN_CPU_GT=0" << std::endl;
    }
    if (config.runCpuGroundTruth && config.graphAccuracySamples > 0)
    {
        std::cout << "graph_accuracy samples=" << config.graphAccuracySamples << " sampling=even";
        for (const int k : {1, 10, 32})
        {
            std::cout << " prefix_recall@" << k << '=' << GraphNeighborRecallAtK(config, exact, graph, k, false);
        }
        for (const int k : {1, 10, 32})
        {
            std::cout << " degree_coverage@" << k << '=' << GraphNeighborRecallAtK(config, exact, graph, k, true);
        }
        std::cout << std::endl;
    }
    std::cout << "scale_phase npu_search start" << std::endl;
    const auto search = RunNpuSearch(config, base, queries, graph);
    const auto diagnostics = AnalyzeSearch(config, search.labels);
    std::cout << "search_prepare graph=npu_builder init_ms=" << search.initMs << " add_ms=" << search.addMs
              << " quantize_ms=" << search.quantizeMs << " warmup_total_ms=" << search.warmupMs << std::endl;
    std::cout << "search_compare graph=npu_builder searcher=npu_rabitq average_batch_ms=" << search.averageSearchMs
              << " qps=" << search.qps;
    if (config.runCpuGroundTruth)
    {
        for (const int k : {1, 10, 32})
        {
            std::cout << " recall@" << k << '='
                      << RecallAtK(exact.queryGroundTruth, search.labels, config.queryNum, config.topK, k);
        }
    }
    else
    {
        std::cout << " recall=not_evaluated";
    }
    std::cout << " valid_ratio=" << diagnostics.validRatio << " unique_ratio=" << diagnostics.uniqueRatio << std::endl;
    std::cout << "search_timing graph=npu_builder scope=search_only average_batch_ms=" << search.averageSearchMs
              << " qps=" << search.qps << std::endl;
    std::cout << "search_timing graph=npu_builder scope=query_prepare_plus_search query_prepare_ms="
              << search.averageQueryPrepareMs << " average_batch_ms=" << search.averageEndToEndMs
              << " qps=" << search.endToEndQps << std::endl;
    std::cout << "PASS: SIFT1M NPU build, quantization and search completed"
              << (config.runCpuGroundTruth ? "" : " (Recall not evaluated)") << std::endl;
}

TEST(TestAscendIndexCagra, Sift1M)
{
    if (GetEnvString("CAGRA_SIFT_DIR", "").empty())
    {
        GTEST_SKIP() << "Set CAGRA_SIFT_DIR to run the SIFT1M end-to-end test";
    }
    try
    {
        RunSift1M(ReadConfig());
    }
    catch (const std::exception &error)
    {
        FAIL() << error.what();
    }
}
}  // namespace

int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
